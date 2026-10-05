# =====================================================================
# FILE: blocks/decide_block.py (GENERIC CONDITIONS VERSION)
# =====================================================================
from desk.core.base_block import BaseBlock
from desk.core.entity import Entity, EventLogger
from typing import Optional, Callable, Dict, Any
import simpy
import random


class DecideBlock(BaseBlock):
    """
    DECIDE block - route entities based on conditions, probabilities, or time.
    
    Supports multiple decision types:
    1. "probability" - Route based on probability distribution
    2. "condition" - Route based on entity attributes
    3. "condition_generic" - Route based on generic expressions (entity, model, resources)
    4. "time_condition" - Route based on simulation time
    
    NEW: Generic condition evaluation with access to:
    - Entity attributes
    - Model state
    - Resource states (queue length, utilization, etc.)
    - Simulation time
    - Custom model variables
    """
    
    def __init__(self, name: str, env: simpy.Environment, 
                 decision_type: str = "probability",
                 track_decisions: bool = True,
                 event_logger: EventLogger = None):
        super().__init__(name, env, event_logger)
        self.decision_type = decision_type
        self.routes = {}
        self.decision_counts = {}
        self.track_decisions = track_decisions
        
    def add_route(self, route_name: str, 
                  next_block: 'BaseBlock',
                  probability: Optional[float] = None,
                  condition: Optional[Callable[[Entity], bool]] = None,
                  condition_generic: Optional[Callable[[Entity, Any], bool]] = None,
                  time_condition: Optional[Callable[[float], bool]] = None):
        """
        Add a routing option.
        
        Args:
            route_name: Name of the route
            next_block: Target block for this route
            probability: Probability for this route (for "probability" type)
            condition: Function(entity) -> bool (for "condition" type)
            condition_generic: Function(entity, context) -> bool (for "condition_generic" type)
            time_condition: Function(time) -> bool (for "time_condition" type)
        
        Examples:
            # 1. Probability-based routing
            decide.add_route("high_priority", block1, probability=0.3)
            
            # 2. Entity-only condition
            decide.add_route("vip", block2, 
                           condition=lambda e: e.priority == 0)
            
            # 3. Generic condition with entity attributes
            decide.add_route("thirsty", block3,
                           condition_generic=lambda e, ctx: e.get_attribute('sede', 0) > 2)
            
            # 4. Generic condition with resource state
            decide.add_route("short_queue", block4,
                           condition_generic=lambda e, ctx: len(ctx['resources']['nurses'].queue) < 5)
            
            # 5. Generic condition with model variables
            decide.add_route("low_failure_rate", block5,
                           condition_generic=lambda e, ctx: ctx['model'].variable_tracker.get_current('percentual_falhas') < 10)
            
            # 6. Complex generic condition
            decide.add_route("priority_and_available", block6,
                           condition_generic=lambda e, ctx: (
                               e.priority == 0 and 
                               ctx['resources']['doctors'].count < ctx['resources']['doctors'].capacity and
                               ctx['time'] < 480  # Before 8 hours
                           ))
            
            # 7. Time-based routing
            decide.add_route("day_shift", block7, 
                           time_condition=lambda t: (t % 1440) < 720)
        """
        self.routes[route_name] = {
            'block': next_block,
            'probability': probability,
            'condition': condition,
            'condition_generic': condition_generic,
            'time_condition': time_condition
        }
        self.decision_counts[route_name] = 0
        
    def process_entity(self, entity: Entity):
        """Route entity based on decision type."""
        entity.route_history.append(self.name)
        
        chosen_route = None
        
        if self.decision_type == "probability":
            chosen_route = self._choose_by_probability()
        elif self.decision_type == "condition":
            chosen_route = self._choose_by_condition(entity)
        elif self.decision_type == "condition_generic":
            chosen_route = self._choose_by_condition_generic(entity)
        elif self.decision_type == "time_condition":
            chosen_route = self._choose_by_time_condition()
        else:
            raise ValueError(f"Invalid decision type: {self.decision_type}")
            
        if chosen_route and chosen_route in self.routes:
            self.decision_counts[chosen_route] += 1
            next_block = self.routes[chosen_route]['block']
            entity.add_attribute(f"{self.name}_decision", chosen_route)

            # Trace decision
            self._trace('decide', entity, details=f"route={chosen_route}")

            # Log decision as an event
            if self.event_logger:
                self.event_logger.log_event(
                    case_id=entity.id,
                    activity=f"{self.name}_{chosen_route}",
                    timestamp=self.env.now,
                    lifecycle='complete',
                    decision=chosen_route,
                    decision_time=self.env.now
                )
            
            # Update model variables if tracking enabled
            if self.track_decisions and hasattr(self.env, 'model'):
                self._update_decision_variables(route_name=chosen_route, entity=entity)

            self.env.process(next_block.process_entity(entity))
            yield self.env.timeout(0)
        else:
            # No valid route found - entity exits
            yield self.env.timeout(0)

    def _choose_by_probability(self) -> Optional[str]:
        """Choose route based on probabilities."""
        rand = random.random()
        cumulative = 0.0
        
        for route_name, route_info in self.routes.items():
            prob = route_info.get('probability', 0)
            cumulative += prob
            if rand <= cumulative:
                return route_name
                
        return None
        
    def _choose_by_condition(self, entity: Entity) -> Optional[str]:
        """
        Choose route based on entity-only conditions.
        
        Routes are evaluated in the order they were added.
        Returns the first route whose condition evaluates to True.
        """
        for route_name, route_info in self.routes.items():
            condition = route_info.get('condition')
            if condition and condition(entity):
                return route_name
                
        return None
    
    def _choose_by_condition_generic(self, entity: Entity) -> Optional[str]:
        """
        Choose route based on generic conditions with full context.
        
        Provides access to:
        - entity: The entity being routed
        - model: The simulation model
        - resources: Dictionary of all resources
        - time: Current simulation time
        - variables: Model variable tracker (if available)
        
        Routes are evaluated in the order they were added.
        Returns the first route whose condition evaluates to True.
        """
        # Build context dictionary with all available information
        context = self._build_decision_context(entity)
        
        for route_name, route_info in self.routes.items():
            condition_generic = route_info.get('condition_generic')
            if condition_generic:
                try:
                    if condition_generic(entity, context):
                        return route_name
                except Exception as e:
                    print(f"WARNING: Error evaluating condition for route '{route_name}': {e}")
                    continue
                
        return None
    
    def _build_decision_context(self, entity: Entity) -> Dict[str, Any]:
        """
        Build context dictionary for generic condition evaluation.
        
        Returns:
            Dictionary with:
            - 'model': Reference to simulation model
            - 'resources': Dictionary of all resources
            - 'time': Current simulation time
            - 'variables': Variable tracker (if available)
            - 'entity': The entity being evaluated
        """
        context = {
            'time': self.env.now,
            'entity': entity
        }
        
        # Add model reference if available
        if hasattr(self.env, 'model'):
            model = self.env.model
            context['model'] = model
            
            # Add resources dictionary
            context['resources'] = model.resources
            
            # Add variable tracker if available
            if hasattr(model, 'variable_tracker'):
                context['variables'] = model.variable_tracker
            
            # Add useful derived information
            context['entity_count'] = model.entity_count
            context['warm_up_period'] = model.warm_up_period
            
            # Add resource utilization info
            context['resource_utilization'] = {}
            for res_name, resource in model.resources.items():
                context['resource_utilization'][res_name] = {
                    'queue_length': len(resource.queue),
                    'in_use': resource.count,
                    'capacity': resource.capacity,
                    'available': resource.capacity - resource.count,
                    'utilization': resource.count / resource.capacity if resource.capacity > 0 else 0
                }
        
        return context
    
    def _choose_by_time_condition(self) -> Optional[str]:
        """
        Choose route based on simulation time conditions.
        
        Routes are evaluated in order until one matches.
        Returns the first route whose time_condition evaluates to True.
        """
        current_time = self.env.now
        
        for route_name, route_info in self.routes.items():
            time_condition = route_info.get('time_condition')
            if time_condition and time_condition(current_time):
                return route_name
                
        return None
    
    def _update_decision_variables(self, route_name: str, entity: Entity):
        """
        Update model variables based on decision route taken.
        
        Automatically tracks decision counts in model variables if they exist.
        """
        if not hasattr(self.env, 'model'):
            return
        
        model = self.env.model
        
        if hasattr(model, 'variable_tracker'):
            tracker = model.variable_tracker
            
            # Try to update route-specific counter
            var_name = f'{self.name}_{route_name}_count'
            if var_name in tracker.variables:
                current = tracker.get_current(var_name)
                tracker.update(var_name, self.env.now, current + 1)