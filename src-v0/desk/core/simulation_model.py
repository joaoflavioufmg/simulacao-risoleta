# =====================================================================
# FILE: core/simulation_model.py
# =====================================================================
from typing import Dict, Any, List, Optional, Union, Callable, Set
import simpy
import sys

from desk.core.base_block import BaseBlock
from desk.core.event_tracer import EventTracer
from desk.blocks.create_block import CreateBlock
from desk.blocks.dispose_block import DisposeBlock
from desk.core.model_variables import ModelVariableTracker


class SimulationModel:
    """
    Core simulation model orchestration.
    
    Responsibilities:
    - Manage simulation environment
    - Manage blocks and resources
    - Run simulation with warm-up handling
    - Provide basic results access
    
    Does NOT handle:
    - Metrics calculation (see analytics.metrics)
    - Plotting (see analytics.plotting)
    - Stability analysis (see validation.stability)
    - Warm-up analysis (see validation.warmup)
    """
    
    def __init__(self, verbose: bool = False,
                 entity_filter: Optional[Set[str]] = None,
                 resource_filter: Optional[Set[str]] = None,
                 event_type_filter: Optional[Set[str]] = None,
                 time_range: Optional[tuple] = None):
        """
        Initialize simulation model.
        
        Args:
            verbose: Enable event tracing
            entity_filter: Set of entity IDs to trace
            resource_filter: Set of resource names to trace
            event_type_filter: Set of event types to trace
            time_range: Tuple of (start_time, end_time) for tracing
        """
        self.env = simpy.Environment()
        self.env.model = self  # For safe_delay_time access
        self.blocks: Dict[str, 'BaseBlock'] = {}
        self.resources: Dict[str, Union[
            simpy.Resource, 
            simpy.PriorityResource, 
            simpy.PreemptiveResource]] = {}
        self.create_blocks: List['CreateBlock'] = []
        self.dispose_blocks: List['DisposeBlock'] = []
        self.stability_result: Optional[float] = None
        self.warm_up_period: float = 0.0
        self.is_warm_up_complete: bool = False
        self.variable_tracker = ModelVariableTracker(self)
        self.verbose = verbose  
        if verbose:
            self.event_tracer = EventTracer(
                self.env,
                entity_filter=entity_filter,
                resource_filter=resource_filter,
                event_type_filter=event_type_filter,
                time_range=time_range
            )
        else:
            self.event_tracer = None


    def validate_resources(self, raise_on_error: bool = True) -> bool:
        """
        Validate resource configuration before running simulation.
        
        Checks for:
        - Resource units exceeding capacity (CRITICAL)
        - Unregistered resources
        - Resource type mismatches
        - Potential deadlocks
        
        Args:
            raise_on_error: If True, raise exception on errors; 
                          if False, return False
            
        Returns:
            True if validation passes, False otherwise
            
        Raises:
            ResourceValidationError: If critical errors found and raise_on_error=True
        """
        from desk.validation.resource_validator import ResourceValidator
        
        validator = ResourceValidator(self)
        return validator.validate_all(raise_on_error=raise_on_error)

    def add_resource(self, name: str, capacity: int, 
                    resource_type: str = "regular") -> Union[simpy.Resource, 
                                                            simpy.PriorityResource]:
        """
        Add a resource to the model.
        
        Args:
            name: Resource name
            capacity: Resource capacity
            resource_type: "regular" or "priority"
            
        Returns:
            The created resource object
        """
        if resource_type == "preemptive":
            resource = simpy.PreemptiveResource(self.env, capacity=capacity)
        elif resource_type == "priority":
            resource = simpy.PriorityResource(self.env, capacity)
        else:
            resource = simpy.Resource(self.env, capacity)
        
        self.resources[name] = resource
        return resource
    
    def add_block(self, block: 'BaseBlock'):
        """Add a block to the model."""
        
        self.blocks[block.name] = block
        
        # Track special block types
        if isinstance(block, CreateBlock):
            self.create_blocks.append(block)
        elif isinstance(block, DisposeBlock):
            self.dispose_blocks.append(block)
    
    def connect_blocks(self, from_block_name: str, to_block_name: str):
        """Connect two blocks in sequence."""
        if from_block_name not in self.blocks or to_block_name not in self.blocks:
            raise ValueError(f"Block not found: {from_block_name} or {to_block_name}")
        
        self.blocks[from_block_name].connect_to(self.blocks[to_block_name])
    
    def set_warm_up_period(self, warm_up_time: float):
        """Set the warm-up period for the simulation."""
        self.warm_up_period = warm_up_time
        self.env.warm_up_period = warm_up_time
    
    def safe_delay_time(self, delay_function: Callable[[], float]) -> float:
        """
        Ensure delay times are non-negative.
        
        Wraps delay functions to replace negative values with 0,
        preventing simulation errors from statistical distributions
        that may generate negative values.
        
        Args:
            delay_function: Function returning delay time
            
        Returns:
            Non-negative delay time
        """
        delay = delay_function()
        return max(0.0, delay)
    
    def run_simulation(self, validate_resources: bool = True,  
                      until: Optional[float] = None, 
                      seed: Optional[int] = None,
                      warm_up_period: float = 0.0,
                      check_stability: bool = False):
        """
        Run the simulation.
        
        Args:
            until: Simulation end time (None = run until no events)
            seed: Random seed for reproducibility
            warm_up_period: Warm-up period duration
            check_stability: Whether to check system stability before running
        """
        if validate_resources:
            self.validate_resources(raise_on_error=True)

        # Validate stopping condition
        self._validate_stopping_condition(until)
        
        if seed:
            import random
            random.seed(seed)
        
        # Set warm-up period
        if warm_up_period > 0:
            self.set_warm_up_period(warm_up_period)
            self.env.process(self._warm_up_monitor())
        
        # Check stability if requested
        if check_stability:
            from validation.stability import StabilityAnalyzer
            analyzer = StabilityAnalyzer(self)
            self.stability_result = analyzer.check_system_stability()
            
            if self.stability_result >= 1.0:
                print("✅ Stable system detected, running full simulation...")
            else:
                print("🚨 Unstable system detected! Running anyway...")

        # Print trace header
        if self.verbose and self.event_tracer:
            self.event_tracer.print_header()
        
        # Start all CREATE blocks
        for create_block in self.create_blocks:
            create_block.start_generation()
        
        # Run simulation
        self.env.run(until=until)

        # NEW: Print trace footer
        if self.verbose and self.event_tracer:
            self.event_tracer.print_footer()
    
    def _validate_stopping_condition(self, until: Optional[float]):
        """Validate that simulation has a stopping condition."""
        has_time_limit = until is not None
        has_entity_limit = any(
            hasattr(cb, 'max_arrivals') and cb.max_arrivals is not None
            for cb in self.create_blocks
        )
        
        if not has_time_limit and not has_entity_limit:
            print("\n" + "=" * 70)
            print("CRITICAL ERROR: SIMULATION WITHOUT DEFINED STOP CONDITION!")
            print("=" * 70)
            print("The simulation has no termination criteria and would run indefinitely.")
            print("\nYou MUST specify at least ONE of the following conditions:")
            print("  1. Simulation time: run_simulation(until=<time>)")
            print("  2. Maximum number of arrivals: CreateBlock(..., max_arrivals=<n>)")
            print("\nValid examples:")
            print("  • model.run_simulation(until=1000)")
            print("  • CreateBlock(..., max_arrivals=500)")
            print("  • Ambos: until=1000 E max_arrivals=500")
            print("\nABORTED EXECUTION to prevent infinite loop.")
            print("=" * 70)
            sys.exit(1)
        
        if not has_time_limit and has_entity_limit:
            max_entities = max(
                cb.max_arrivals for cb in self.create_blocks
                if hasattr(cb, 'max_arrivals') and cb.max_arrivals is not None
            )
            print(f"\nWARNING: Simulation limited only by number of entities "
                  f"({max_entities}).")
            print("Execution time may be very long if the system is congested..")
            print("It is also recommended to set a time limit with until=<value>.\n")
    
    def _warm_up_monitor(self):
        """Monitor warm-up period completion."""
        if self.warm_up_period > 0:
            yield self.env.timeout(self.warm_up_period)
            self.is_warm_up_complete = True
            self._clear_warm_up_statistics()
    
    def _clear_warm_up_statistics(self):
        """Clear statistics collected during warm-up."""
        from desk.blocks.process_block import ProcessBlock, MultiProcessBlock
        
        # Reset DisposeBlock counters (keep data for plotting)
        for dispose_block in self.dispose_blocks:
            dispose_block.entities_disposed = 0
            dispose_block.total_system_time = 0.0
        
        # Reset ProcessBlock stats
        for block in self.blocks.values():
            if isinstance(block, (ProcessBlock, MultiProcessBlock)):
                block.entities_processed = 0
                block.total_delay_time = 0.0
                block.total_queue_time = 0.0
                
                if isinstance(block, ProcessBlock):
                    block.max_queue_length = 0
                    block.max_in_service = 0
                elif isinstance(block, MultiProcessBlock):
                    for metrics in block.max_metrics.values():
                        metrics['max_queue_length'] = 0
                        metrics['max_in_service'] = 0

    def add_model_variable(self, name: str, initial_value: Any = 0,
                          description: str = "", unit: str = "",
                          calculate_fn: Optional[Callable] = None):
        """Add a custom model variable to track."""
        self.variable_tracker.add_variable(
            name, initial_value, description, unit, calculate_fn
        )

    def update_model_variable(self, name: str, value: Any = None):
        """Update a model variable."""
        self.variable_tracker.update(name, value=value)


    @property
    def entity_count(self) -> int:
        """Total entities disposed (post warm-up)."""
        disposed_sum = sum(block.entities_disposed for block in self.dispose_blocks)
        if disposed_sum > 0:
            return disposed_sum
        return sum(block.entities_created for block in self.create_blocks)
    
    @property
    def overall_throughput(self) -> float:
        """Overall system throughput (entities per time unit)."""
        effective_time = self.env.now - self.warm_up_period
        if effective_time > 0:
            return self.entity_count / effective_time
        return 0
    
    def get_results(self) -> Dict[str, Any]:
        """
        Get basic simulation results.
        
        For detailed metrics, use:
        - analytics.metrics.MetricsCollector
        - analytics.reporting.SimulationReporter
        """
        results = {
            'simulation_time': self.env.now,
            'warm_up_period': self.warm_up_period,
            'entity_count': self.entity_count,
            'throughput': self.overall_throughput,
            'blocks': {}
        }
        
        for block_name, block in self.blocks.items():
            results['blocks'][block_name] = {
                'type': type(block).__name__,
                'statistics': block.statistics
            }
            
            if hasattr(block, 'entities_processed'):
                results['blocks'][block_name]['entities_processed'] = block.entities_processed
            if hasattr(block, 'entities_created'):
                results['blocks'][block_name]['entities_created'] = block.entities_created
            if hasattr(block, 'entities_disposed'):
                results['blocks'][block_name]['entities_disposed'] = block.entities_disposed
            if hasattr(block, 'decision_counts'):
                results['blocks'][block_name]['decision_counts'] = block.decision_counts
        
        return results  
    
    def trace_entity(self, entity_id: str):
        """
        Print complete journey of a specific entity.
        
        Args:
            entity_id: Entity ID to trace (e.g., 'Patient_5')
        """
        if self.event_tracer:
            self.event_tracer.print_entity_journey(entity_id)
        else:
            print("Verbose mode not enabled. Run simulation with verbose=True")
    
    def trace_entities(self, entity_ids: List[str]):
        """
        Print journeys of multiple entities.
        
        Args:
            entity_ids: List of entity IDs to trace
        """
        if self.event_tracer:
            for entity_id in entity_ids:
                self.event_tracer.print_entity_journey(entity_id)
                print()  # Blank line between journeys
        else:
            print("Verbose mode not enabled. Run simulation with verbose=True")
    
    def replay_trace(self, entity_filter: Optional[Set[str]] = None,
                    resource_filter: Optional[Set[str]] = None,
                    event_type_filter: Optional[Set[str]] = None,
                    time_range: Optional[tuple] = None,
                    entity_pattern: Optional[str] = None):
        """
        Replay simulation trace with filters.
        
        Args:
            entity_filter: Set of specific entity IDs (e.g., {'Patient_0', 'Patient_5'})
            resource_filter: Set of resources (e.g., {'doctors', 'nurses'})
            event_type_filter: Set of event types (e.g., {'queue', 'service_start'})
            time_range: Time window (e.g., (10, 50))
            entity_pattern: Regex pattern for entities (e.g., r'Patient_[0-5]')
        
        Examples:
            # Trace specific patient
            model.replay_trace(entity_filter={'Patient_1'})
            
            # Trace first 5 patients
            model.replay_trace(entity_pattern=r'Patient_[0-4]')
            
            # Trace only doctor interactions
            model.replay_trace(resource_filter={'doctors'})
            
            # Trace queue and service events
            model.replay_trace(event_type_filter={'queue', 'service_start', 'service_end'})
            
            # Trace specific time window
            model.replay_trace(time_range=(10, 50))
            
            # Combine filters
            model.replay_trace(entity_filter={'Patient_1'}, 
                             event_type_filter={'queue', 'service_start'})
        """
        if self.event_tracer:
            self.event_tracer.replay_trace(
                entity_filter=entity_filter,
                resource_filter=resource_filter,
                event_type_filter=event_type_filter,
                time_range=time_range,
                entity_pattern=entity_pattern
            )
        else:
            print("Verbose mode not enabled. Run simulation with verbose=True")
    
    def print_trace_statistics(self):
        """Print summary statistics of event trace."""
        if self.event_tracer:
            self.event_tracer.print_statistics()
        else:
            print("Verbose mode not enabled. Run simulation with verbose=True")