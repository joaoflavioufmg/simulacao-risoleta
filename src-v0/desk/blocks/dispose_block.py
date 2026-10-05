# =====================================================================
# FILE: blocks/dispose_block.py
# =====================================================================
from desk.core.base_block import BaseBlock
from desk.core.entity import Entity, EventLogger
import simpy


class DisposeBlock(BaseBlock):
    """DISPOSE block - removes entities from system and collects statistics."""
    
    def __init__(self, name: str, env: simpy.Environment, event_logger: EventLogger = None):
        super().__init__(name, env, event_logger)
        self.entities_disposed = 0
        self.total_system_time = 0.0
        self.disposed_entities = []
        
    def process_entity(self, entity: Entity):
        """Dispose of entity and collect final statistics."""
        entity.route_history.append(self.name)
        
        # Always collect entity data for plotting, but only count for statistics after warm-up
        system_time = self.env.now - entity.creation_time
        entity.add_attribute("system_time", system_time)
        entity.add_attribute("disposal_time", self.env.now)

        # Capture assigned attributes (e.g., revenue)
        assigned_attrs = self._apply_attributes(entity)

        self.disposed_entities.append(entity)  # Always keep for plotting
        
        # Only count for official statistics after warm-up period
        if self.env.now >= getattr(self.env, 'warm_up_period', 0):
            self.total_system_time += system_time
            self.entities_disposed += 1
        

        # Include attributes in departure trace
        details = f"total_time_in_system={system_time:.2f}"

        # Add attribute info if any were assigned
        if assigned_attrs:
            attr_strs = [f"{name}={value:.2f}" if isinstance(value, float) else f"{name}={value}" 
                        for name, value in assigned_attrs]
            details += f", Attrib: {', '.join(attr_strs)}"

        self._trace('departure', entity, details=details)

        # Log disposal
        if self.event_logger:
            self.event_logger.log_event(
                case_id=entity.id,
                activity="Discharge",
                timestamp=self.env.now,
                lifecycle='complete',
                system_time=system_time
            )
        
        # Entity is disposed - no further processing
        yield self.env.timeout(0)
        
    def get_average_system_time(self):
        """Get average system time for disposed entities."""
        if self.entities_disposed > 0:
            return self.total_system_time / self.entities_disposed
        return 0.0
