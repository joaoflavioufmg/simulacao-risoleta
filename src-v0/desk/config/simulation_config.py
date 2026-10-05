# =====================================================================
# FILE: config/simulation_config.py
# =====================================================================
from dataclasses import dataclass
from typing import Optional


@dataclass
class SimulationConfig:
    """Configuration for simulation run."""
    duration: float
    warm_up_period: float = 0.0
    seed: Optional[int] = None
    check_stability: bool = False
    
    def validate(self):
        """Validate configuration."""
        if self.duration <= 0:
            raise ValueError("Duration must be positive")
        if self.warm_up_period < 0:
            raise ValueError("Warm-up period cannot be negative")
        if self.warm_up_period >= self.duration:
            raise ValueError("Warm-up period must be less than duration")