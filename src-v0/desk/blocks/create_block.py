# =====================================================================
# FILE: blocks/create_block.py
# =====================================================================
from desk.core.base_block import BaseBlock
from desk.core.entity import Entity, EventLogger
from typing import Optional, Callable
import simpy


class CreateBlock(BaseBlock):
    """CREATE block - generates entities into the system."""
    
    def __init__(self, name: str, env: simpy.Environment, 
             inter_arrival_time: Callable[[], float],
             entity_prefix: str = "Entity",
             max_arrivals: Optional[int] = None,
             first_creation: float = 0.0,
             priority_generator: Optional[Callable[[], int]] = None,
             event_logger: EventLogger = None):
        # Call parent class init FIRST with event_logger
        super().__init__(name, env, event_logger)
        # NOW we can safely set other attributes
        self.inter_arrival_time = inter_arrival_time
        self.entity_prefix = entity_prefix
        self.max_arrivals = max_arrivals
        self.first_creation = first_creation
        self.entities_created = 1
        self.priority_generator = priority_generator
        
    def start_generation(self):
        """Start the entity generation process."""
        return self.env.process(self._generation_process())
        
    def _generation_process(self):
        """Internal process for generating entities."""
        if self.first_creation > 0:
            yield self.env.timeout(self.first_creation)
            
        while True:
            if self.max_arrivals and self.entities_created > self.max_arrivals:
                break
                
            entity = Entity(
                id=f"{self.entity_prefix}_{self.entities_created}",
                creation_time=self.env.now,
                data={},
                route_history=[],
                priority=self.priority_generator() if self.priority_generator else 0
            )
            
            self.entities_created += 1
            entity.route_history.append(self.name)
            
            # Capture assigned attributes at creation
            assigned_attrs = self._apply_attributes(entity)

            # Include initial attributes in trace
            details = f"entity created, priority={entity.priority}"

            if assigned_attrs:
                attr_strs = []
                for name, value in assigned_attrs:
                    if isinstance(value, float):
                        attr_strs.append(f"{name}={value:.2f}")
                    else:
                        attr_strs.append(f"{name}={value}")
                details += f", Attrib: {', '.join(attr_strs)}"

            self._trace('generate', entity, details=details)
            
            # Log creation as an event
            if self.event_logger:
                self.event_logger.log_event(
                    case_id=entity.id,
                    activity="Arrival",
                    timestamp=self.env.now,
                    lifecycle='complete',
                    priority=entity.priority
                )
            
            if self.next_block:
                self.env.process(self.next_block.process_entity(entity))
                
            yield self.env.timeout(self.inter_arrival_time())
            
    def process_entity(self, entity: Entity):
        """CREATE blocks don't process incoming entities."""
        raise NotImplementedError("CREATE blocks generate entities, they don't process them")

