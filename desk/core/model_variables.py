# =====================================================================
# FILE: core/model_variables.py
# =====================================================================
from dataclasses import dataclass, field
from tabnanny import verbose
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class ModelVariable:
    """
    Represents a custom model state variable to monitor during simulation.
    
    Attributes:
        name: Variable name
        initial_value: Starting value
        description: Human-readable description
        unit: Unit of measurement (e.g., '%', 'units', 'R$')
        calculate_fn: Optional function to calculate value dynamically
    """
    name: str
    initial_value: Any = 0
    description: str = ""
    unit: str = ""
    calculate_fn: Optional[Callable] = None
    history: List[Tuple[float, Any]] = field(default_factory=list)
    
    def record(self, time: float, value: Any):
        """Record a value at a specific time."""
        self.history.append((time, value))
    
    def get_current_value(self) -> Any:
        """Get the most recent recorded value."""
        if self.history:
            return self.history[-1][1]
        return self.initial_value
    
    def get_average(self, start_time: float = 0) -> float:
        """Calculate time-weighted average after start_time."""
        if not self.history:
            return self.initial_value
        
        filtered = [(t, v) for t, v in self.history if t >= start_time]
        if not filtered:
            return self.initial_value
        
        # Time-weighted average
        total_area = 0
        prev_time = start_time
        prev_value = self.initial_value
        
        for time, value in filtered:
            total_area += prev_value * (time - prev_time)
            prev_time = time
            prev_value = value
        
        # Add final segment
        if filtered:
            final_time = filtered[-1][0]
            total_time = final_time - start_time
            return total_area / total_time if total_time > 0 else prev_value
        
        return self.initial_value
    
    def get_final_value(self) -> Any:
        """Get the final recorded value."""
        if self.history:
            return self.history[-1][1]
        return self.initial_value


class ModelVariableTracker:
    """
    Tracks and manages custom model state variables during simulation.
    
    Usage:
        tracker = ModelVariableTracker(model)
        
        # Define variables
        tracker.add_variable('percentual_falhas', 
                           initial_value=0, 
                           description='Percentual de falhas',
                           unit='%')
        
        # Update during simulation
        tracker.update('percentual_falhas', model.env.now, 15.5)
        
        # Analyze after simulation
        tracker.plot_variable('percentual_falhas')
        avg = tracker.get_average('percentual_falhas')
    """
    
    def __init__(self, model):
        """
        Initialize variable tracker.
        
        Args:
            model: SimulationModel instance
        """
        self.model = model
        self.variables: Dict[str, ModelVariable] = {}
    
    def add_variable(self, name: str, initial_value: Any = 0,
                    description: str = "", unit: str = "",
                    calculate_fn: Optional[Callable] = None):
        """
        Add a new variable to track.
        
        Args:
            name: Variable name
            initial_value: Starting value
            description: Human-readable description
            unit: Unit of measurement
            calculate_fn: Optional function to calculate value dynamically
                         Function signature: calculate_fn(model) -> value
        
        Example:
            tracker.add_variable(
                'percentual_falhas',
                initial_value=0,
                description='Percentual de entidades que falharam',
                unit='%',
                calculate_fn=lambda m: (m.num_falhas / m.num_total * 100) if m.num_total > 0 else 0
            )
        """
        var = ModelVariable(
            name=name,
            initial_value=initial_value,
            description=description,
            unit=unit,
            calculate_fn=calculate_fn
        )
        self.variables[name] = var
        
        # Record initial value
        var.record(self.model.env.now, initial_value)
        
        if verbose:
            print(f"Variable added: {name} = {initial_value} {unit}")
    
    def update(self, name: str, time: Optional[float] = None, value: Any = None):
        """
        Update a variable's value.
        
        Args:
            name: Variable name
            time: Timestamp (None = use current simulation time)
            value: New value (None = calculate using calculate_fn)
        
        Example:
            tracker.update('percentual_falhas', model.env.now, 12.5)
            tracker.update('percentual_falhas')  # Auto-calculate
        """
        if name not in self.variables:
            raise ValueError(f"Variable '{name}' not found. Add it first with add_variable()")
        
        var = self.variables[name]
        
        # Use current simulation time if not provided
        if time is None:
            time = self.model.env.now
        
        # Calculate value if function provided and value not given
        if value is None:
            if var.calculate_fn:
                value = var.calculate_fn(self.model)
            else:
                raise ValueError(f"No value provided and no calculate_fn defined for '{name}'")
        
        var.record(time, value)
    
    def get_current(self, name: str) -> Any:
        """Get current value of a variable."""
        if name not in self.variables:
            raise ValueError(f"Variable '{name}' not found")
        return self.variables[name].get_current_value()
    
    def get_average(self, name: str, start_time: Optional[float] = None) -> float:
        """
        Get time-weighted average of a variable.
        
        Args:
            name: Variable name
            start_time: Start time for average (None = use warm_up_period)
        """
        if name not in self.variables:
            raise ValueError(f"Variable '{name}' not found")
        
        if start_time is None:
            start_time = self.model.warm_up_period
        
        return self.variables[name].get_average(start_time)
    
    def get_final(self, name: str) -> Any:
        """Get final value of a variable."""
        if name not in self.variables:
            raise ValueError(f"Variable '{name}' not found")
        return self.variables[name].get_final_value()
    
    def plot_variable(self, name: str, show_warm_up: bool = True):
        """
        Plot variable evolution over time.
        
        Args:
            name: Variable name
            show_warm_up: Mark warm-up period on plot
        """
        if name not in self.variables:
            raise ValueError(f"Variable '{name}' not found")
        
        var = self.variables[name]
        
        if not var.history:
            print(f"No data recorded for variable '{name}'")
            return
        
        times = [t for t, _ in var.history]
        values = [v for _, v in var.history]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot as step function
        ax.step(times, values, where='post', linewidth=2, 
               color='steelblue', label=f'{name}')
        
        # Add average line (post warm-up)
        avg = self.get_average(name)
        ax.axhline(y=avg, color='red', linestyle='--', linewidth=2,
                  label=f'Average (post warm-up): {avg:.2f} {var.unit}')
        
        # Mark warm-up period
        if show_warm_up and self.model.warm_up_period > 0:
            ax.axvline(x=self.model.warm_up_period, color='orange',
                      linestyle='--', linewidth=2,
                      label=f'Warm-up end (t={self.model.warm_up_period})')
            ax.axvspan(0, self.model.warm_up_period, alpha=0.2, color='orange')
        
        # Formatting
        ax.set_xlabel('Simulation Time', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{name} ({var.unit})', fontsize=12, fontweight='bold')
        
        title = f'{name}'
        if var.description:
            title += f'\n{var.description}'
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_all_variables(self, show_warm_up: bool = True):
        """Plot all tracked variables in subplots."""
        if not self.variables:
            print("No variables to plot")
            return
        
        n_vars = len(self.variables)
        fig, axes = plt.subplots(n_vars, 1, figsize=(12, 4 * n_vars))
        
        if n_vars == 1:
            axes = [axes]
        
        fig.suptitle('Model State Variables Over Time',
                    fontsize=14, fontweight='bold')
        
        for idx, (name, var) in enumerate(self.variables.items()):
            ax = axes[idx]
            
            if not var.history:
                ax.text(0.5, 0.5, f'No data for {name}',
                       ha='center', va='center', transform=ax.transAxes)
                continue
            
            times = [t for t, _ in var.history]
            values = [v for _, v in var.history]
            
            ax.step(times, values, where='post', linewidth=2,
                   color='steelblue', label=name)
            
            avg = self.get_average(name)
            ax.axhline(y=avg, color='red', linestyle='--', linewidth=1.5,
                      label=f'Avg: {avg:.2f} {var.unit}')
            
            if show_warm_up and self.model.warm_up_period > 0:
                ax.axvline(x=self.model.warm_up_period, color='orange',
                          linestyle='--', linewidth=1.5)
                ax.axvspan(0, self.model.warm_up_period, alpha=0.2, color='orange')
            
            ax.set_ylabel(f'{name} ({var.unit})', fontsize=10, fontweight='bold')
            ax.legend(loc='best', framealpha=0.9, fontsize=9)
            ax.grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Simulation Time', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def print_summary(self):
        """Print summary of all tracked variables."""
        print("\n" + "=" * 70)
        print("MODEL STATE VARIABLES SUMMARY")
        print("=" * 70)
        
        if not self.variables:
            print("No variables tracked")
            return
        
        for name, var in self.variables.items():
            print(f"\n{name}:")
            if var.description:
                print(f"  Description: {var.description}")
            print(f"  Initial value: {var.initial_value} {var.unit}")
            print(f"  Final value: {self.get_final(name)} {var.unit}")
            print(f"  Average (post warm-up): {self.get_average(name):.2f} {var.unit}")
            print(f"  Data points recorded: {len(var.history)}")
        
        print("=" * 70)
    
    def get_dataframe(self) -> pd.DataFrame:
        """
        Export all variable histories as a pandas DataFrame.
        
        Returns:
            DataFrame with columns: time, variable_name, value
        """
        data = []
        for name, var in self.variables.items():
            for time, value in var.history:
                data.append({
                    'time': time,
                    'variable': name,
                    'value': value
                })
        
        return pd.DataFrame(data)
    
    def export_to_csv(self, filename: str = "model_variables.csv"):
        """Export variable histories to CSV."""
        df = self.get_dataframe()
        df.to_csv(filename, index=False)
        print(f"Model variables exported to {filename}")



# =====================================================================
# USAGE EXAMPLES
# =====================================================================
def example_usage():
    """Example showing how to use ModelVariableTracker."""
    
    # Assuming you have a model
    from desk.core.simulation_model import SimulationModel
    model = SimulationModel()
    
    # Create tracker
    tracker = ModelVariableTracker(model)
    
    # Example 1: Simple counter variable
    tracker.add_variable(
        'num_falhas',
        initial_value=0,
        description='Número total de falhas',
        unit='unidades'
    )
    
    # Example 2: Percentage with auto-calculation
    tracker.add_variable(
        'percentual_falhas',
        initial_value=0,
        description='Percentual de entidades que falharam',
        unit='%',
        calculate_fn=lambda m: (
            tracker.get_current('num_falhas') / m.entity_count * 100
            if m.entity_count > 0 else 0
        )
    )
    
    # Example 3: Financial metric
    tracker.add_variable(
        'lucro_acumulado',
        initial_value=0,
        description='Lucro acumulado total',
        unit='R$'
    )
    
    # During simulation:
    # Update manually
    tracker.update('num_falhas', model.env.now, 5)
    tracker.update('lucro_acumulado', model.env.now, 1250.50)
    
    # Auto-calculate percentual
    tracker.update('percentual_falhas')
    
    # After simulation:
    tracker.print_summary()
    tracker.plot_variable('percentual_falhas')
    tracker.plot_all_variables()
    tracker.export_to_csv()