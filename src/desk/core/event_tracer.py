# =====================================================================
# FILE: core/event_tracer.py
# =====================================================================
from datetime import datetime
from typing import Optional, List, Set
import re

class EventTracer:
    """
    Traces and prints simulation events in a human-readable format.
    
    Provides verbose output for debugging and understanding simulation flow.
    """
    
    # Event icons
    ICONS = {
        'generate': '✨',
        'arrival': '🧍',
        'queue': '⏳',
        'service_start': '✅',
        'service_end': '🎯',
        'departure': '🚶',
        'decide': '🔀',
        'interrupt': '⚠️',
        'preempt': '🚨'
    }
    
    def __init__(self, env, 
                 entity_filter: Optional[Set[str]] = None,
                 resource_filter: Optional[Set[str]] = None,
                 event_type_filter: Optional[Set[str]] = None,
                 time_range: Optional[tuple] = None):
        """
        Initialize event tracer with optional filters.
        
        Args:
            env: SimPy environment
            entity_filter: Set of entity IDs to trace (e.g., {'Patient_0', 'Patient_5'})
            resource_filter: Set of resource names to trace (e.g., {'doctors', 'nurses'})
            event_type_filter: Set of event types to trace (e.g., {'queue', 'service_start'})
            time_range: Tuple of (start_time, end_time) to limit trace output
        """
        self.env = env
        self.event_count = 0
        self.start_time = datetime.now()

        # Filters
        self.entity_filter = entity_filter
        self.resource_filter = resource_filter
        self.event_type_filter = event_type_filter
        self.time_range = time_range
        
        # Storage for post-simulation filtering
        self.all_events: List[dict] = []
        self.store_all = True  # Always store for later filtering
    
    def set_filters(self, entity_filter: Optional[Set[str]] = None,
                   resource_filter: Optional[Set[str]] = None,
                   event_type_filter: Optional[Set[str]] = None,
                   time_range: Optional[tuple] = None):
        """Update filters dynamically."""
        if entity_filter is not None:
            self.entity_filter = entity_filter
        if resource_filter is not None:
            self.resource_filter = resource_filter
        if event_type_filter is not None:
            self.event_type_filter = event_type_filter
        if time_range is not None:
            self.time_range = time_range
    
    def clear_filters(self):
        """Remove all filters."""
        self.entity_filter = None
        self.resource_filter = None
        self.event_type_filter = None
        self.time_range = None
    
    def _should_trace(self, event_type: str, entity_id: str, resource_name: Optional[str], 
                     time: float) -> bool:
        """Check if event passes all active filters."""
        # Time range filter
        if self.time_range:
            start, end = self.time_range
            if time < start or time > end:
                return False
        
        # Entity filter
        if self.entity_filter and entity_id not in self.entity_filter:
            return False
        
        # Resource filter
        if self.resource_filter:
            if resource_name is None:
                return False

            # Handle multi-resource activities (comma-separated resources)
            # Split resource_name by comma and check if ANY match the filter
            resource_names_in_event = [r.strip() for r in resource_name.split(',')]
            
            # Check if any filtered resource is present in this event
            if not any(filter_resource in resource_names_in_event 
                    for filter_resource in self.resource_filter):
                return False
        
        # Event type filter
        if self.event_type_filter and event_type.lower() not in self.event_type_filter:
            return False
        
        return True
    
    def print_header(self):
        """Print trace header."""
        print("\n" + "=" * 120)
        print("=== SIMULATION EVENT TRACE ===")

        # Show active filters
        filters_active = []
        if self.entity_filter:
            filters_active.append(f"Entities: {', '.join(sorted(self.entity_filter))}")
        if self.resource_filter:
            filters_active.append(f"Resources: {', '.join(sorted(self.resource_filter))}")
        if self.event_type_filter:
            filters_active.append(f"Events: {', '.join(sorted(self.event_type_filter))}")
        if self.time_range:
            filters_active.append(f"Time: [{self.time_range[0]:.2f}, {self.time_range[1]:.2f}]")
        
        if filters_active:
            print("FILTERS ACTIVE: " + " | ".join(filters_active))

        print("=" * 120)        
        print(f"{'Time':<8} | {'Event':<22}  | {'Entity':<15} | {'Resource':<30} | {'Details':<50}")
        print("-" * 120)
    
    def print_footer(self):
        """Print trace footer."""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        print("-" * 120)
        print(f"End of trace — {end_time.strftime('%H:%M:%S')} | "
              f"Events shown: {self.event_count} | Total stored: {len(self.all_events)} | "
              f"Duration: {duration:.2f}s")
        print("=" * 120)
    
    def trace(self, event_type: str, entity_id: str, resource_name: Optional[str] = None, 
              details: str = "", time_override: Optional[float] = None):
        """
        Trace a single event.
        
        Args:
            event_type: Type of event (generate, arrival, queue, service_start, etc.)
            entity_id: ID of the entity
            resource_name: Name of resource involved (if any)
            details: Additional details to display
            time_override: Override current time (for retroactive logging)
        """
        time = time_override if time_override is not None else self.env.now

        # Store all events for later filtering
        event_data = {
            'time': time,
            'event_type': event_type,
            'entity_id': entity_id,
            'resource_name': resource_name,
            'details': details
        }
        self.all_events.append(event_data)
        
        # Check if should print now (based on filters)
        if not self._should_trace(event_type, entity_id, resource_name, time):
            return

        # Print event
        icon = self.ICONS.get(event_type.lower(), '•')        
        event_name = f"{icon} {event_type.upper()}"
        # Format resource string with usage/capacity
        resource_str = self._format_resource_string(resource_name) if resource_name else ""
        
        print(f"{time:>7.2f}  | {event_name:<22} | {entity_id:<15} | {resource_str:<30} | {details}")
        self.event_count += 1

    def _format_resource_string(self, resource_name: str) -> str:
        """
        Format resource string with current usage and capacity.
        
        Args:
            resource_name: Name of resource(s), possibly comma-separated
            
        Returns:
            Formatted string like "[3/30] Troncos" or "[2/4] doctors, [1/3] nurses"
        """
        if not resource_name:
            return ""
        
        # Handle multi-resource activities (comma-separated)
        resource_names = [r.strip() for r in resource_name.split(',')]
        formatted_parts = []
        
        for res_name in resource_names:
            # Try to find the resource object in the model
            resource_obj = self._find_resource_by_name(res_name)
            
            if resource_obj:
                # Get current usage and capacity
                current_usage = resource_obj.count
                capacity = resource_obj.capacity
                formatted_parts.append(f"[{current_usage}/{capacity}] {res_name}")
            else:
                # If resource not found, just use the name
                formatted_parts.append(res_name)
        
        return ", ".join(formatted_parts)


    def _find_resource_by_name(self, resource_name: str):
        """
        Find resource object by name from the model.
        
        Args:
            resource_name: Name of the resource
            
        Returns:
            Resource object or None if not found
        """
        # Access the model through env.model if it exists
        if not hasattr(self.env, 'model'):
            return None
        
        model = self.env.model
        
        if not hasattr(model, 'resources'):
            return None
        
        # Direct lookup
        if resource_name in model.resources:
            return model.resources[resource_name]
        
        # Fuzzy match (case-insensitive)
        for res_name, res_obj in model.resources.items():
            if res_name.lower() == resource_name.lower():
                return res_obj
        
        return None

    def replay_trace(self, entity_filter: Optional[Set[str]] = None,
                    resource_filter: Optional[Set[str]] = None,
                    event_type_filter: Optional[Set[str]] = None,
                    time_range: Optional[tuple] = None,
                    entity_pattern: Optional[str] = None):
        """
        Replay stored events with different filters.
        
        Args:
            entity_filter: Set of specific entity IDs to show
            resource_filter: Set of resource names to show
            event_type_filter: Set of event types to show
            time_range: Tuple of (start_time, end_time)
            entity_pattern: Regex pattern for entity ID matching (e.g., r'^Patient_[1-5]$')
        """
        # Temporarily save old filters
        old_entity_filter = self.entity_filter
        old_resource_filter = self.resource_filter
        old_event_type_filter = self.event_type_filter
        old_time_range = self.time_range
        
        # Apply new filters
        if entity_pattern:
            # Convert pattern to entity set
            pattern = re.compile(entity_pattern)
            matched_entities = {e['entity_id'] for e in self.all_events 
                              if pattern.match(e['entity_id'])}
            self.entity_filter = matched_entities
        else:
            self.entity_filter = entity_filter
        
        self.resource_filter = resource_filter
        self.event_type_filter = event_type_filter
        self.time_range = time_range
        
        # Reset counter
        self.event_count = 0
        
        # Print header
        self.print_header()
        
        # Replay events
        for event in self.all_events:
            if self._should_trace(event['event_type'], event['entity_id'], 
                                 event['resource_name'], event['time']):
                icon = self.ICONS.get(event['event_type'].lower(), '•')
                event_name = f"{icon} {event['event_type'].upper()}"
                resource_str = event['resource_name'] if event['resource_name'] else ""
                
                print(f"{event['time']:>7.2f}  | {event_name:<22} | "
                      f"{event['entity_id']:<15} | {resource_str:<30} | {event['details']}")
                self.event_count += 1
        
        # Print footer
        self.print_footer()
        
        # Restore old filters
        self.entity_filter = old_entity_filter
        self.resource_filter = old_resource_filter
        self.event_type_filter = old_event_type_filter
        self.time_range = old_time_range
    
    def get_entity_journey(self, entity_id: str) -> List[dict]:
        """
        Get complete journey of a specific entity.
        
        Args:
            entity_id: Entity ID to trace
            
        Returns:
            List of event dictionaries for this entity
        """
        return [e for e in self.all_events if e['entity_id'] == entity_id]
    
    def print_entity_journey(self, entity_id: str):
        """
        Print formatted journey of a specific entity.
        
        Args:
            entity_id: Entity ID to trace
        """
        journey = self.get_entity_journey(entity_id)
        
        if not journey:
            print(f"\nNo events found for entity: {entity_id}")
            return
        
        print("\n" + "=" * 80)
        print(f"=== ENTITY JOURNEY: {entity_id} ===")
        print("=" * 80)
        
        # Calculate statistics
        start_time = journey[0]['time']
        end_time = journey[-1]['time']
        total_time = end_time - start_time
        
        # Find queue and service times
        queue_times = []
        service_times = []
        resources_used = set()
        
        print(f"{'Time':<8} | {'Event':<22} | {'Resource':<30} | {'Details':<30}")
        print("-" * 80)
        
        for event in journey:
            icon = self.ICONS.get(event['event_type'].lower(), '•')
            event_name = f"{icon} {event['event_type'].upper()}"
            resource_str = event['resource_name'] if event['resource_name'] else ""            
            # resource_str = self._format_resource_string(event['resource_name']) if event['resource_name'] else ""
            
            print(f"{event['time']:>7.2f}  | {event_name:<21} | {resource_str:<30} | {event['details']}")
            
            # Extract statistics
            if event['resource_name']:
                resources_used.add(event['resource_name'])
            
            if 'queue_time=' in event['details']:
                try:
                    qt = float(event['details'].split('queue_time=')[1].split(',')[0])
                    queue_times.append(qt)
                except:
                    pass
            
            if 'service_time=' in event['details']:
                try:
                    st = float(event['details'].split('service_time=')[1].split(',')[0])
                    service_times.append(st)
                except:
                    pass
        
        print("-" * 80)
        print(f"\nJOURNEY SUMMARY:")
        # Check if journey is incomplete
        has_departure = any(e['event_type'] == 'departure' for e in journey)
        if total_time == 0 or not has_departure:
            print(f"  ⚠️  WARNING: Incomplete journey (entity still in system at simulation end)")
        print(f"  Total time in system: {total_time:.2f} minutes")
        print(f"  Number of events: {len(journey)}")
        print(f"  Resources used: {', '.join(sorted(resources_used)) if resources_used else 'None'}")
        
        # Prevent division by zero
        if queue_times:
            queue_total = sum(queue_times)
            queue_pct = (queue_total / total_time * 100) if total_time > 0 else 0.0
            print(f"  Total queue time: {queue_total:.2f} ({queue_pct:.1f}%)")
        
        if service_times:
            service_total = sum(service_times)
            service_pct = (service_total / total_time * 100) if total_time > 0 else 0.0
            print(f"  Total service time: {service_total:.2f} ({service_pct:.1f}%)")        
        print("=" * 80)
    
    def get_statistics(self) -> dict:
        """Get statistics about traced events."""
        entity_counts = {}
        resource_counts = {}
        event_type_counts = {}
        
        for event in self.all_events:
            # Count entities
            entity_id = event['entity_id']
            entity_counts[entity_id] = entity_counts.get(entity_id, 0) + 1
            
            # Count resources
            if event['resource_name']:
                resource_counts[event['resource_name']] = \
                    resource_counts.get(event['resource_name'], 0) + 1
            
            # Count event types
            event_type = event['event_type']
            event_type_counts[event_type] = event_type_counts.get(event_type, 0) + 1
        
        return {
            'total_events': len(self.all_events),
            'unique_entities': len(entity_counts),
            'entity_counts': entity_counts,
            'resource_counts': resource_counts,
            'event_type_counts': event_type_counts,
            'time_span': (self.all_events[0]['time'], self.all_events[-1]['time']) 
                        if self.all_events else (0, 0)
        }
    
    def print_statistics(self):
        """Print summary statistics of trace."""
        stats = self.get_statistics()
        
        print("\n" + "=" * 60)
        print("=== TRACE STATISTICS ===")
        print("=" * 60)
        print(f"Total events: {stats['total_events']}")
        print(f"Unique entities: {stats['unique_entities']}")
        print(f"Time span: {stats['time_span'][0]:.2f} - {stats['time_span'][1]:.2f}")
        
        print("\nEvents by type:")
        for event_type, count in sorted(stats['event_type_counts'].items(), 
                                       key=lambda x: x[1], reverse=True):
            print(f"  {event_type:.<20} {count:>6}")
        
        print("\nEvents by resource:")
        for resource, count in sorted(stats['resource_counts'].items(), 
                                     key=lambda x: x[1], reverse=True):
            print(f"  {resource:.<20} {count:>6}")
        
        print("\nTop 10 most active entities:")
        sorted_entities = sorted(stats['entity_counts'].items(), 
                               key=lambda x: x[1], reverse=True)[:10]
        for entity_id, count in sorted_entities:
            print(f"  {entity_id:.<20} {count:>6} events")
        
        print("=" * 60)