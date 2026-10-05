# =====================================================================
# FILE: core/entity.py
# =====================================================================
from dataclasses import dataclass, field
from typing import Dict, Any, List
import pandas as pd


@dataclass
class Entity:
    """Represents an entity flowing through the simulation."""
    id: str
    creation_time: float
    data: Dict[str, Any] = field(default_factory=dict)
    route_history: List[str] = field(default_factory=list)
    priority: int = 0  # Lower numbers = higher priority (0 = highest)
    
    def add_attribute(self, key: str, value: Any):
        self.data[key] = value
    
    def get_attribute(self, key: str, default=None):
        return self.data.get(key, default)


class EventLogger:
    """Logs events in BupaR format during simulation."""
    
    def __init__(self):
        self.events = []
    
    def log_event(self, case_id: str, activity: str, timestamp: float, 
                  lifecycle: str, resource: str = None, **attributes):
        """Log a single event."""
        event = {
            'case_id': case_id,
            'activity': activity,
            'timestamp': timestamp,
            'lifecycle': lifecycle,
            'resource': resource
        }
        event.update(attributes)
        self.events.append(event)
    
    def get_dataframe(self) -> pd.DataFrame:
        """Return events as a pandas DataFrame."""
        df = pd.DataFrame(self.events)
        df = df.sort_values(['case_id', 'timestamp']).reset_index(drop=True)
        return df
    
    def export_to_csv(self, filename: str = "event_log_bupar.csv"):
        """Export to CSV in BupaR format."""
        df = self.get_dataframe()
        df.to_csv(filename, index=False)
        print(f"Event log exported to {filename}")
        print(f"Total events: {len(df)}")
        print(f"Total cases: {df['case_id'].nunique()}")
        return df