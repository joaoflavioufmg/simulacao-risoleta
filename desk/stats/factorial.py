# =====================================================================
# FILE: statistics/factorial.py
# =====================================================================
"""
Factorial experimental design framework for simulation studies.

This module provides tools for:
- Designing full factorial experiments with multiple factors
- Running experiments with multiple replications per configuration
- Analyzing main effects and interaction effects
- Visualizing experimental results
"""

import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Callable, Optional, Tuple
from dataclasses import dataclass
import time


@dataclass
class FactorLevel:
    """
    Represents a factor and its levels for factorial analysis.
    
    Attributes:
        factor_name: Short name for the factor (e.g., 'arrival_rate')
        parameter_path: Path to parameter in model (for documentation)
        levels: List of values to test for this factor
        description: Human-readable description of the factor
    """
    factor_name: str
    parameter_path: str
    levels: List[Any]
    description: str = ""


class FactorialExperiment:
    """
    Framework for conducting factorial experiments on simulation models.
    
    This class implements a full factorial design where all combinations
    of factor levels are tested with multiple replications.
    """
    
    def __init__(self, simulation_function: Callable, base_seed: int = 12345):
        """
        Initialize factorial experiment framework.
        
        Args:
            simulation_function: Function that creates and runs simulation model.
                                Must accept factor parameters as kwargs and return model.
            base_seed: Base random seed for reproducibility
        """
        self.simulation_function = simulation_function
        self.base_seed = base_seed
        self.factors: List[FactorLevel] = []
        self.results: List[Dict[str, Any]] = []
        self.results_df: Optional[pd.DataFrame] = None
    
    def add_factor(self, factor_name: str, parameter_path: str,
                   levels: List[Any], description: str = ""):
        """
        Add a factor to the experimental design.
        
        Args:
            factor_name: Name of the factor (e.g., "arrival_rate")
            parameter_path: Path to parameter in model (for documentation)
            levels: List of values to test
            description: Human-readable description
        """
        factor = FactorLevel(
            factor_name=factor_name,
            parameter_path=parameter_path,
            levels=levels,
            description=description
        )
        self.factors.append(factor)
        print(f"✅ Factor added: {factor_name} ({len(levels)} levels)")
    
    def run_factorial_experiment(self, n_replications: int = 1,
                                 simulation_time: Optional[float] = None,
                                 warm_up_period: float = 0.0,
                                 verbose: bool = True):
        """
        Run full factorial experiment with all combinations of factor levels.
        
        Args:
            n_replications: Number of replications per combination
            simulation_time: Duration of each simulation run
            warm_up_period: Warm-up period for statistics collection
            verbose: Print progress messages
        """
        if not self.factors:
            print("❌ No factor defined! Use add_factor() first.")
            return
        
        # Generate all combinations
        factor_levels = [factor.levels for factor in self.factors]
        combinations = list(itertools.product(*factor_levels))
        total_runs = len(combinations) * n_replications
        
        self._print_experiment_header(combinations, n_replications, total_runs)
        
        self.results = []
        start_time = time.time()
        run_count = 0
        
        # Run all combinations
        for combo_idx, combination in enumerate(combinations):
            # Create factor configuration
            config = {
                self.factors[i].factor_name: combination[i]
                for i in range(len(self.factors))
            }
            
            if verbose:
                print(f"\n📊 Settings {combo_idx + 1}/{len(combinations)}: {config}")
            
            # Run replications for this combination
            for rep in range(n_replications):
                run_count += 1
                seed = self.base_seed + combo_idx * 1000 + rep
                
                if verbose and n_replications > 1:
                    print(f"  Replication {rep + 1}/{n_replications} (seed: {seed})")
                
                try:
                    # Run simulation with current configuration
                    model = self._run_simulation_with_config(
                        config, seed, simulation_time, warm_up_period
                    )
                    
                    # Extract results
                    result = self._extract_results(model, config, combo_idx, rep)
                    self.results.append(result)
                    
                    if verbose and run_count % 10 == 0:
                        self._print_progress(start_time, run_count, total_runs)
                
                except Exception as e:
                    print(f"  ❌ Runtime error: {e}")
                    continue
        
        self._print_completion_summary(start_time, total_runs)
        
        # Convert to DataFrame
        self.results_df = pd.DataFrame(self.results)
        print(f"📊 {len(self.results)} collected results")
    
    def _print_experiment_header(self, combinations: List, n_replications: int,
                                 total_runs: int):
        """Print experiment setup information."""
        print("\n🔬 FACTORIAL EXPERIMENT")
        print("=" * 60)
        print(f"Factors: {len(self.factors)}")
        for factor in self.factors:
            print(f"  - {factor.factor_name}: {len(factor.levels)} levels")
        print(f"Combinations: {len(combinations)}")
        print(f"Replications per combination: {n_replications}")
        print(f"Total runs: {total_runs}")
        print("=" * 60)
    
    def _print_progress(self, start_time: float, run_count: int, total_runs: int):
        """Print progress update."""
        elapsed = time.time() - start_time
        avg_time = elapsed / run_count
        remaining = (total_runs - run_count) * avg_time
        print(f"  Progress: {run_count}/{total_runs} | "
              f"Remaining time: {remaining/60:.1f} min")
    
    def _print_completion_summary(self, start_time: float, total_runs: int):
        """Print experiment completion summary."""
        total_time = time.time() - start_time
        print(f"\n✅ EXPERIMENT COMPLETED in {total_time/60:.1f} minutes")
        print(f"⏱️  Average time per run: {total_time/total_runs:.1f} seconds")
    
    def _run_simulation_with_config(self, config: Dict, seed: int,
                                     simulation_time: Optional[float],
                                     warm_up_period: float):
        """Run simulation with specific factor configuration."""
        # Build kwargs from configuration
        kwargs = {
            'seed': seed,
            'return_model': True
        }
        
        if simulation_time is not None:
            kwargs['until'] = simulation_time
        if warm_up_period > 0:
            kwargs['warm_up_period'] = warm_up_period
        
        # Add factor values to kwargs
        kwargs.update(config)
        
        # Run simulation
        model = self.simulation_function(**kwargs)
        return model
    
    def _extract_results(self, model, config: Dict, combo_idx: int,
                         rep: int) -> Dict[str, Any]:
        """
        Extract KPIs from simulation model.
        
        Args:
            model: Completed simulation model
            config: Factor configuration for this run
            combo_idx: Combination index
            rep: Replication number
            
        Returns:
            Dictionary of results including factors and KPIs
        """
        
        result = {
            'combination_id': combo_idx,
            'replication': rep,
            **config  # Include factor values
        }
        
        # System-level metrics
        result['simulation_time'] = model.env.now
        result['warm_up_period'] = model.warm_up_period
        result['entities_processed'] = model.entity_count
        result['throughput'] = model.overall_throughput
        
        ###############################################################
        # Import here to avoid circular dependencies
        from desk.analytics.metrics import MetricsCollector       
        ###############################################################
        # Entity metrics: Compute metrics using MetricsCollector
        metrics_collector = MetricsCollector(model)
        entity_summary = metrics_collector.get_entity_metrics_summary()
        result['system_time_avg'] = entity_summary.get('tempo_medio_sistema', 0)
        
        # Activity metrics
        for activity_name, metrics in entity_summary.get('atividades', {}).items():
            result[f'{activity_name}_queue_time'] = metrics.get('tempo_medio_fila', 0)
            result[f'{activity_name}_service_time'] = metrics.get('tempo_medio_atendimento', 0)
        
        # Resource metrics
        resource_summary = metrics_collector.get_resource_metrics_summary()
        for resource_name, metrics in resource_summary.items():
            result[f'{resource_name}_utilization'] = metrics['taxa_utilizacao']
            result[f'{resource_name}_avg_queue'] = metrics['numero_medio_fila']
            result[f'{resource_name}_max_queue'] = metrics['maximo_fila']
        
        return result
    
    def get_aggregated_results(self) -> Optional[pd.DataFrame]:
        """
        Aggregate results by factor combination (average over replications).
        
        Returns:
            DataFrame with mean and std for each metric by factor combination
        """
        if self.results_df is None:
            print("❌ Run the experiment first!")
            return None
        
        # Group by factor values
        factor_names = [f.factor_name for f in self.factors]
        
        # Aggregate numeric columns
        numeric_cols = self.results_df.select_dtypes(include=[np.number]).columns
        exclude_cols = ['combination_id', 'replication', 'simulation_time', 'warm_up_period']
        agg_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        aggregated = self.results_df.groupby(factor_names)[agg_cols].agg(
            ['mean', 'std']
        ).reset_index()
        
        return aggregated
    
    def plot_correlation_matrix(self):
        """Plot correlation matrix of key metrics with compact legend."""
        if self.results_df is None:
            print("❌ Run the experiment first!")
            return
        
        # Filter columns and create labels
        selected_cols, col_labels = self._prepare_correlation_data()
        
        if not selected_cols:
            print("❌ No relevant columns found!")
            return
        
        # Calculate correlation
        filtered_df = self.results_df[selected_cols]
        corr_matrix = filtered_df.corr()
        
        # Create short labels
        short_labels = [col_labels[col] for col in filtered_df.columns]
        
        # Plot
        fig, ax = plt.subplots(figsize=(16, 10))
        
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, linewidths=0.5,
                   xticklabels=short_labels, yticklabels=short_labels,
                   ax=ax, cbar_kws={'label': 'Correlation'})
        
        ax.set_title('Correlation Matrix (Key Metrics)',
                    fontsize=14, fontweight='bold', pad=15)
        
        # Create and position legend
        legend_text = self._create_correlation_legend(filtered_df, col_labels)
        fig.text(0.75, 0.5, legend_text,
                fontsize=9,
                verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9, pad=0.8),
                family='monospace')
        
        plt.subplots_adjust(left=0.1, right=0.75)
        plt.show()
        
        print(f"\nCorrelation matrix generated with {len(selected_cols)} variables")
        
        return corr_matrix
    
    def _prepare_correlation_data(self) -> Tuple[List[str], Dict[str, str]]:
        """Prepare data for correlation matrix."""
        selected_cols = []
        col_labels = {}
        label_counter = 1
        
        factor_names = [f.factor_name for f in self.factors]
        
        # Add factor columns
        for col in self.results_df.columns:
            if any(col.startswith(fname) for fname in factor_names):
                selected_cols.append(col)
                col_labels[col] = f"F{label_counter}"
                label_counter += 1
        
        # Add activity metrics
        metric_counter = 1
        for col in self.results_df.columns:
            if 'queue_time' in col or 'service_time' in col:
                selected_cols.append(col)
                label = f"Q{metric_counter}" if 'queue_time' in col else f"S{metric_counter}"
                col_labels[col] = label
                metric_counter += 1
        
        # Add resource utilization
        util_counter = 1
        for col in self.results_df.columns:
            if '_utilization' in col:
                selected_cols.append(col)
                col_labels[col] = f"U{util_counter}"
                util_counter += 1
        
        return selected_cols, col_labels
    
    def _create_correlation_legend(self, filtered_df: pd.DataFrame,
                                   col_labels: Dict[str, str]) -> str:
        """Create legend text for correlation plot."""
        factor_names = [f.factor_name for f in self.factors]
        legend_lines = ["CAPTION:", "", "Factors:"]
        
        for col in filtered_df.columns:
            if any(col.startswith(fname) for fname in factor_names):
                legend_lines.append(f"  {col_labels[col]}: {col}")
        
        legend_lines.append("")
        legend_lines.append("Activities:")
        for col in filtered_df.columns:
            if 'queue_time' in col or 'service_time' in col:
                short_name = col.replace('_queue_time', '').replace('_service_time', '')
                metric_type = 'Queue' if 'queue' in col else 'Service'
                legend_lines.append(f"  {col_labels[col]}: {short_name} ({metric_type})")
        
        legend_lines.append("")
        legend_lines.append("Resources:")
        for col in filtered_df.columns:
            if '_utilization' in col:
                resource_name = col.replace('_utilization', '')
                legend_lines.append(f"  {col_labels[col]}: {resource_name} (Util)")
        
        return "\n".join(legend_lines)
    
    def plot_main_effects(self, response_variable: str):
        """
        Plot main effects for each factor on a response variable.
        
        Args:
            response_variable: Name of the response variable to plot
        """
        if self.results_df is None:
            print("❌ Conduct the experiment first!")
            return
        
        if response_variable not in self.results_df.columns:
            print(f"❌ Variables '{response_variable}' not found!")
            return
        
        n_factors = len(self.factors)
        fig, axes = plt.subplots(1, n_factors, figsize=(5*n_factors, 4))
        if n_factors == 1:
            axes = [axes]
        
        for idx, factor in enumerate(self.factors):
            ax = axes[idx]
            
            # Group by factor level and calculate mean response
            grouped = self.results_df.groupby(factor.factor_name)[response_variable].agg(
                ['mean', 'std']
            )
            
            # Plot
            x_pos = range(len(grouped))
            ax.errorbar(x_pos, grouped['mean'], yerr=grouped['std'],
                       marker='o', markersize=8, capsize=5, linewidth=2)
            
            ax.set_xlabel(factor.factor_name, fontsize=11, fontweight='bold')
            ax.set_ylabel(response_variable, fontsize=11, fontweight='bold')
            ax.set_title(f'{factor.factor_name} effect', fontsize=12)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(grouped.index, rotation=45)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Main Effects in {response_variable}',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def plot_interaction_effects(self, response_variable: str,
                                 factor1_name: str, factor2_name: str):
        """
        Plot interaction effects between two factors.
        
        Args:
            response_variable: Response variable to analyze
            factor1_name: First factor name
            factor2_name: Second factor name
        """
        if self.results_df is None:
            print("❌ Conduct the experiment first!")
            return
        
        if response_variable not in self.results_df.columns:
            print(f"❌ Variables '{response_variable}' not found!")
            return
        
        # Group by both factors
        grouped = self.results_df.groupby(
            [factor1_name, factor2_name]
        )[response_variable].mean().reset_index()
        
        # Pivot for plotting
        pivot = grouped.pivot(index=factor1_name, columns=factor2_name,
                            values=response_variable)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for col in pivot.columns:
            ax.plot(pivot.index, pivot[col], marker='o', markersize=8,
                   linewidth=2, label=f'{factor2_name}={col}')
        
        ax.set_xlabel(factor1_name, fontsize=12, fontweight='bold')
        ax.set_ylabel(response_variable, fontsize=12, fontweight='bold')
        ax.set_title(f'Interaction between {factor1_name} and {factor2_name}',
                    fontsize=14, fontweight='bold')
        ax.legend(title=factor2_name, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def print_summary(self):
        """Print comprehensive summary of factorial analysis."""
        if self.results_df is None:
            print("❌ Conduct the experiment first!")
            return
        
        print("\n" + "=" * 70)
        print("📊 SUMMARY OF FACTORIAL ANALYSIS")
        print("=" * 70)
        
        self._print_factor_summary()
        self._print_best_worst_configurations()
        self._print_descriptive_statistics()
        self._print_general_analysis()
    
    def _print_factor_summary(self):
        """Print summary of factors tested."""
        print("\n🔬 TESTED FACTORS:")
        for factor in self.factors:
            print(f"  - {factor.factor_name}: {factor.levels}")
            if factor.description:
                print(f"    {factor.description}")
    
    def _print_best_worst_configurations(self):
        """Print best and worst configurations for key metrics."""
        factor_names = [f.factor_name for f in self.factors]
        
        # Find key metrics
        activity_metrics = [col for col in self.results_df.columns
                          if 'queue_time' in col or 'service_time' in col]
        utilization_metrics = [col for col in self.results_df.columns
                             if '_utilization' in col]
        
        sample_metrics = []
        if activity_metrics:
            sample_metrics.append(activity_metrics[0])
        if utilization_metrics:
            sample_metrics.append(utilization_metrics[0])
        
        for metric in sample_metrics[:3]:
            print(f"\n  {metric}:")
            
            # Find best and worst
            if 'time' in metric.lower():
                best_idx = self.results_df[metric].idxmin()
                worst_idx = self.results_df[metric].idxmax()
            elif 'utilization' in metric.lower():
                best_idx = (self.results_df[metric] - 0.75).abs().idxmin()
                worst_idx = self.results_df[metric].idxmax()
            else:
                best_idx = self.results_df[metric].idxmax()
                worst_idx = self.results_df[metric].idxmin()
            
            best_row = self.results_df.loc[best_idx]
            worst_row = self.results_df.loc[worst_idx]
            
            best_config = {fname: best_row[fname] for fname in factor_names}
            worst_config = {fname: worst_row[fname] for fname in factor_names}
            
            best_val = best_row[metric]
            worst_val = worst_row[metric]
            
            if 'utilization' in metric:
                print(f"    Best: {best_config} -> {best_val*100:.1f}%")
                print(f"    Worst: {worst_config} -> {worst_val*100:.1f}%")
            else:
                print(f"    Best: {best_config} -> {best_val:.2f}")
                print(f"    Worst: {worst_config} -> {worst_val:.2f}")
    
    def _print_descriptive_statistics(self):
        """Print descriptive statistics for key metrics."""
        print("\n📈 DESCRIPTIVE STATISTICS (Key Metrics):")
        print("-" * 70)
        
        # Activity times
        print("\n🕐 ACTIVITY TIMES:")
        activity_cols = [col for col in self.results_df.columns
                        if 'queue_time' in col or 'service_time' in col]
        if activity_cols:
            activity_df = self.results_df[activity_cols]
            print(activity_df.describe().T[['mean', 'std', 'min', 'max']].to_string())
        
        # Resource utilization
        print("\n🏭 USE OF RESOURCES:")
        util_cols = [col for col in self.results_df.columns if '_utilization' in col]
        if util_cols:
            util_df = self.results_df[util_cols]
            util_display = util_df.describe().T[['mean', 'std', 'min', 'max']] * 100
            print(util_display.to_string())
            print("(values in %)")
    
    def _print_general_analysis(self):
        """Print general analysis summary."""
        print("\n💡 GENERAL ANALYSIS:")
        n_combinations = len(self.results_df['combination_id'].unique())
        n_reps = len(self.results_df[self.results_df['combination_id']==0])
        print(f"   Total number of configurations tested: {n_combinations}")
        print(f"   Replications per configuration: {n_reps}")
        print(f"   Total executions: {len(self.results_df)}")
    
    def export_results(self, filename: str = "results/factorial_results.csv",
                      export_filtered: bool = False):
        """
        Export results to CSV.
        
        Args:
            filename: Output filename
            export_filtered: If True, export only key metrics; if False, export all
        """
        if self.results_df is None:
            print("❌ Conduct the experiment first!")
            return
        
        if export_filtered:
            export_df = self._get_filtered_results()
            print(f"📁 FILTERED results exported to {filename}")
            print(f"   Exported columns: {len(export_df.columns)}")
        else:
            export_df = self.results_df
            print(f"📁 FULL results exported to {filename}")
            print(f"   Exported columns: {len(export_df.columns)}")
        
        export_df.to_csv(filename, index=False)
        print(f"   Total entries: {len(export_df)}")
    
    def _get_filtered_results(self) -> pd.DataFrame:
        """Get filtered DataFrame with only key metrics."""
        factor_names = [f.factor_name for f in self.factors]
        key_cols = ['combination_id', 'replication']
        
        # Add factors
        for col in self.results_df.columns:
            if any(col.startswith(fname) for fname in factor_names):
                key_cols.append(col)
        
        # Add activity metrics
        for col in self.results_df.columns:
            if 'queue_time' in col or 'service_time' in col:
                key_cols.append(col)
        
        # Add resource utilization
        for col in self.results_df.columns:
            if '_utilization' in col:
                key_cols.append(col)
        
        return self.results_df[key_cols]