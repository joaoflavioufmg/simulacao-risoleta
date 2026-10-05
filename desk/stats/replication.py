# =====================================================================
# FILE: statistics/replication.py
# =====================================================================
"""
Replication framework for running multiple simulation runs with statistical analysis.

This module provides the ReplicationFramework class for:
- Running multiple independent simulation replications
- Collecting KPIs across replications
- Computing confidence intervals
- Generating statistical reports and visualizations
"""

from typing import Dict, Any, List, Callable, Optional
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
import time
import math


class ReplicationFramework:
    """
    Framework for running multiple simulation replications with statistical analysis.
    
    The framework follows the method of independent replications for steady-state
    simulation analysis, computing confidence intervals using the t-distribution.
    """
    
    def __init__(self, simulation_function: Callable, n_replications: int = 30):
        """
        Initialize replication framework.
        
        Args:
            simulation_function: Function that creates, runs, and returns a simulation model.
                                Should accept seed parameter and return a model instance.
            n_replications: Number of replications to run (default 30 for CLT applicability)
        """
        self.simulation_function = simulation_function
        self.n_replications = n_replications
        self.replication_results: List[Dict[str, Any]] = []
        self.summary_statistics: Dict[str, Dict[str, float]] = {}
    
    def run_replications(self, base_seed: int = 12345, **simulation_kwargs):
        """
        Run multiple simulation replications with different random seeds.
        
        Args:
            base_seed: Base seed for reproducibility
            **simulation_kwargs: Additional arguments to pass to simulation function
        """
        print(f"EXECUTING {self.n_replications} REPLICATIONS...")
        print("=" * 50)
        
        start_time = time.time()
        
        for replication in range(self.n_replications):
            # Set unique seed for each replication
            replication_seed = base_seed + replication * 1000
            
            print(f"Replication {replication + 1}/{self.n_replications} (seed: {replication_seed})")
            
            # Run simulation
            model = self.simulation_function(seed=replication_seed, **simulation_kwargs)
            
            # Extract KPIs from this replication
            kpis = self._extract_kpis(model, replication)
            self.replication_results.append(kpis)
            
            # Progress indicator
            if (replication + 1) % 5 == 0 or replication + 1 == self.n_replications:
                elapsed = time.time() - start_time
                avg_time = elapsed / (replication + 1)
                remaining = (self.n_replications - replication - 1) * avg_time
                print(f"  Progress: {replication + 1}/{self.n_replications} | "
                      f"Estimated time remaining: {remaining/60:.1f} min")
        
        total_time = time.time() - start_time
        print(f"\nREPLICATIONS COMPLETED in {total_time/60:.1f} minutes")
        print(f"Average time per replication: {total_time/self.n_replications:.1f} seconds")
        
        # Calculate summary statistics
        self._calculate_summary_statistics()
        
        # Print results
        self.print_statistical_summary()
        
        # Plot confidence intervals
        self.plot_confidence_intervals()
        
        # Export results
        self.export_results()
    
    def _extract_kpis(self, model, replication_id: int) -> Dict[str, Any]:
        """
        Extract key performance indicators from a simulation model.
        
        Args:
            model: Completed simulation model instance
            replication_id: Replication number (for tracking)
            
        Returns:
            Dictionary of KPIs for this replication
        """
        # Import here to avoid circular dependencies
        from desk.analytics.metrics import MetricsCollector
        from desk.analytics.financial import FinancialAnalyzer  # NEW
        from desk.analytics.wip_metrics import WIPTracker  # NEW
        
        metrics_collector = MetricsCollector(model)
        financial_analyzer = FinancialAnalyzer(model)  # NEW
        wip_tracker = WIPTracker(model)  # NEW
        
        kpis = {
            'replication_id': replication_id,
            'simulation_time': model.env.now,
            'warm_up_period': model.warm_up_period,
            'entities_processed': model.entity_count,
            'overall_throughput': model.overall_throughput
        }
        
        # Entity metrics
        entity_summary = metrics_collector.get_entity_metrics_summary()
        system_time = entity_summary.get('tempo_medio_sistema', 0)
        kpis['system_time_avg'] = 0 if (system_time is None or math.isnan(system_time)) else system_time
        
        
        # NEW: WIP metrics
        wip_summary = wip_tracker.get_wip_summary()
        kpis['average_wip'] = wip_summary['average_wip']
        kpis['max_wip'] = wip_summary['max_wip']
        kpis['final_wip'] = wip_summary['final_wip']
        
        # NEW: System time metrics
        system_time_summary = wip_tracker.get_system_time_summary()
        kpis['system_time_avg_detailed'] = system_time_summary['average_system_time']
        kpis['system_time_std'] = system_time_summary['std_system_time']
        kpis['system_time_min'] = system_time_summary['min_system_time']
        kpis['system_time_max'] = system_time_summary['max_system_time']

        
        # Activity metrics - Handle None and nan values
        activities = entity_summary.get('atividades', {})
        for activity_name, activity_metrics in activities.items():
            queue_time = activity_metrics.get('tempo_medio_fila', 0) or 0
            service_time = activity_metrics.get('tempo_medio_atendimento', 0) or 0
            activity_system_time = activity_metrics.get('tempo_medio_sistema', 0) or 0
            
            # Replace nan with 0
            kpis[f'{activity_name}_queue_time'] = 0 if math.isnan(queue_time) else queue_time
            kpis[f'{activity_name}_service_time'] = 0 if math.isnan(service_time) else service_time
            kpis[f'{activity_name}_system_time'] = 0 if math.isnan(activity_system_time) else activity_system_time
        
        # Resource metrics
        resource_summary = metrics_collector.get_resource_metrics_summary()
        for resource_name, resource_metrics in resource_summary.items():
            kpis[f'{resource_name}_utilization'] = resource_metrics['taxa_utilizacao']
            kpis[f'{resource_name}_avg_queue'] = resource_metrics['numero_medio_fila']
            kpis[f'{resource_name}_avg_in_service'] = resource_metrics['numero_medio_atendimento']
            kpis[f'{resource_name}_max_queue'] = resource_metrics['maximo_fila']
        
        # Decision routing metrics (if applicable)
        for block_name, block in model.blocks.items():
            if hasattr(block, 'decision_counts'):
                total_decisions = sum(block.decision_counts.values())
                for route_name, count in block.decision_counts.items():
                    percentage = (count / total_decisions * 100) if total_decisions > 0 else 0
                    kpis[f'{block_name}_{route_name}_percentage'] = percentage
        

        # NEW: Add financial metrics
        financial_summary = financial_analyzer.get_financial_summary()
        kpis['total_revenue'] = financial_summary['total_revenue']
        kpis['total_costs'] = financial_summary['total_costs']
        kpis['net_profit'] = financial_summary['net_profit']
        kpis['avg_revenue_per_entity'] = financial_summary['avg_revenue_per_entity']
        kpis['avg_cost_per_entity'] = financial_summary['avg_cost_per_entity']
        kpis['avg_profit_per_entity'] = financial_summary['avg_profit_per_entity']
        
        # Add costs by activity
        for activity, cost in financial_summary['costs_by_activity'].items():
            kpis[f'{activity}_total_cost'] = cost
        
        return kpis
    
    def _calculate_summary_statistics(self):
        """
        Calculate summary statistics with 95% confidence intervals.
        
        Uses t-distribution for confidence intervals to account for
        finite sample size and unknown population variance.
        """
        if not self.replication_results:
            print("No replication results available!")
            return
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(self.replication_results)
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        self.summary_statistics = {}
        
        for column in numeric_columns:
            if column in ['replication_id', 'simulation_time', 'warm_up_period']:
                continue
            
            values = df[column].values
            valid_values = values[~np.isnan(values)]
            n = len(valid_values)
            
            if n == 0:
                self.summary_statistics[column] = {
                    'mean': np.nan,
                    'std': np.nan,
                    'sem': np.nan,
                    'ci_lower': np.nan,
                    'ci_upper': np.nan,
                    'half_width': np.nan,
                    'relative_precision': np.nan,
                    'n_replications': 0,
                    'min': np.nan,
                    'max': np.nan
                }
                continue
            
            mean = np.mean(valid_values)
            min_val = np.min(valid_values)
            max_val = np.max(valid_values)
            
            if n > 1:
                std = np.std(valid_values, ddof=1)  # Sample standard deviation
                sem = std / np.sqrt(n)  # Standard error of the mean
                
                # 95% Confidence Interval using t-distribution
                confidence_level = 0.95
                alpha = 1 - confidence_level
                t_critical = stats.t.ppf(1 - alpha/2, df=n-1)
                
                half_width = t_critical * sem
                ci_lower = mean - half_width
                ci_upper = mean + half_width
                
                # Relative precision (half-width of CI as percentage of mean)
                relative_precision = (half_width / abs(mean) * 100) if mean != 0 else 0
            else:
                std = np.nan
                sem = np.nan
                ci_lower = np.nan
                ci_upper = np.nan
                half_width = np.nan
                relative_precision = np.nan
            
            self.summary_statistics[column] = {
                'mean': mean,
                'std': std,
                'sem': sem,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'half_width': half_width,
                'relative_precision': relative_precision,
                'n_replications': n,
                'min': min_val,
                'max': max_val
            }
    
    def print_statistical_summary(self):
        """Print comprehensive statistical summary with confidence intervals."""
        if not self.summary_statistics:
            print("Statistics not computed. Run run_replications() first.")
            return
        
        print("=" * 80)
        print(f"STATISTICAL RESULTS ({self.n_replications} REPLICATIONS)")
        print("=" * 80)
        
        # System-level metrics
        self._print_section_metrics(
            title="SYSTEM METRICS",
            metrics=[
                ('system_time_avg', 'Average time in the system'),
                ('entities_processed', 'Entities processed'),
                ('overall_throughput', 'Throughput (entities/min)')
            ]
        )
        
        # Activity metrics
        activity_metrics = [k for k in self.summary_statistics.keys()
                          if any(suffix in k for suffix in ['_queue_time', '_service_time', '_system_time'])]
        
        if activity_metrics:
            print("\nACTIVITY METRICS:")
            print("-" * 40)
            for metric_key in sorted(activity_metrics):
                stats_data = self.summary_statistics[metric_key]
                metric_name = metric_key.replace('_', ' ').title()
                self._print_metric_statistics(metric_name, stats_data)
        
        # Resource utilization metrics
        utilization_metrics = [k for k in self.summary_statistics.keys()
                             if '_utilization' in k]
        
        if utilization_metrics:
            print("\nUSE OF RESOURCES:")
            print("-" * 40)
            for metric_key in sorted(utilization_metrics):
                stats_data = self.summary_statistics[metric_key]
                resource_name = metric_key.replace('_utilization', '')
                metric_name = f"Utilization rate - {resource_name}"
                # Convert to percentage for display
                stats_pct = self._convert_to_percentage(stats_data)
                self._print_metric_statistics(metric_name, stats_pct, unit='%')
        
        # Precision analysis
        self._print_precision_analysis()
        
        print(f"\nNumber of replications: {self.n_replications}")
        print(f"Confidence level: 95%")
        print(f"Recommendation: {self._get_replication_recommendation()}")
    
    def _print_section_metrics(self, title: str, metrics: List[tuple]):
        """Print a section of metrics."""
        print(f"\n{title}:")
        print("-" * 40)
        for metric_key, metric_name in metrics:
            if metric_key in self.summary_statistics:
                stats_data = self.summary_statistics[metric_key]
                self._print_metric_statistics(metric_name, stats_data)
    
    def _convert_to_percentage(self, stats_data: Dict) -> Dict:
        """Convert statistics to percentage scale."""
        stats_pct = stats_data.copy()
        for key in ['mean', 'ci_lower', 'ci_upper', 'std', 'min', 'max', 'half_width']:
            if key in stats_pct and not np.isnan(stats_pct[key]):
                stats_pct[key] = stats_pct[key] * 100
        return stats_pct
    
    def _print_metric_statistics(self, metric_name: str, stats_data: Dict, unit: str = ""):
        """Print statistics for a single metric, handling NaN."""
        def format_val(val):
            return f"{val:.2f}" if not np.isnan(val) else "N/A"
        
        mean = stats_data['mean']
        half_width = stats_data['half_width']
        ci_lower = stats_data['ci_lower']
        ci_upper = stats_data['ci_upper']
        precision = stats_data['relative_precision']
        std = stats_data['std']
        min_val = stats_data['min']
        max_val = stats_data['max']
        
        print(f"{metric_name}:")
        print(f"  Average: {format_val(mean)}{unit} +/- {format_val(half_width)}")
        print(f"  CI 95%: [{format_val(ci_lower)}, {format_val(ci_upper)}]{unit}")
        print(f"  Precision: +/-{format_val(precision)}%")
        print(f"  Standard Deviation: {format_val(std)}")
        print(f"  Min-Max: [{format_val(min_val)}, {format_val(max_val)}]{unit}")
        print()
    
    def _print_precision_analysis(self):
        """Print precision analysis summary."""
        print("\nPRECISION ANALYSIS:")
        print("-" * 40)
        
        high_precision = []
        medium_precision = []
        low_precision = []
        
        for metric_key, stats_data in self.summary_statistics.items():
            if metric_key in ['replication_id']:
                continue
            precision = stats_data['relative_precision']
            if np.isnan(precision):
                continue
            
            if precision <= 5:
                high_precision.append((metric_key, precision))
            elif precision <= 10:
                medium_precision.append((metric_key, precision))
            else:
                low_precision.append((metric_key, precision))
        
        print(f"High precision (<=5%): {len(high_precision)} metrics")
        print(f"Medium precision (5-10%): {len(medium_precision)} metrics")
        print(f"Low precision (>10%): {len(low_precision)} metrics")
        
        if low_precision:
            print("\nMetrics with low precision (consider more replications):")
            for metric, precision in sorted(low_precision, key=lambda x: x[1], reverse=True):
                print(f"  {metric}: {precision:.1f}%")
    
    def _get_replication_recommendation(self) -> str:
        """Provide recommendation for number of replications."""
        if not self.summary_statistics:
            return "Perform statistical analysis first"
        
        valid_precisions = [stats_data['relative_precision']
                           for stats_data in self.summary_statistics.values()
                           if not np.isnan(stats_data['relative_precision'])]
        
        if not valid_precisions:
            return "Insufficient data for recommendation"
        
        avg_precision = np.mean(valid_precisions)
        
        if avg_precision <= 5:
            return "Excellent accuracy - adequate number of replications"
        elif avg_precision <= 10:
            return "Good accuracy - consider more replications for critical metrics"
        elif avg_precision <= 20:
            return "Moderate precision - it is recommended to double the number of replications"
        else:
            return "Low accuracy - significantly increase the number of replications"
    
    def plot_confidence_intervals(self, metrics_to_plot: Optional[List[str]] = None):
        """
        Plot confidence intervals for resource utilization metrics.
        
        Args:
            metrics_to_plot: List of specific metrics to plot (None = all utilization metrics)
        """
        if not self.summary_statistics:
            print("Statistics not computed.")
            return
        
        # Get only resource utilization metrics
        utilization_metrics = [k for k in self.summary_statistics.keys()
                            if '_utilization' in k]
        
        if not utilization_metrics:
            print("No resource usage metrics found.")
            return
        
        # Filter available metrics
        available_metrics = [m for m in utilization_metrics if m in self.summary_statistics]
        
        if not available_metrics:
            print("No usage metrics available for plotting.")
            return
        
        # Create figure
        _fig, ax = plt.subplots(figsize=(10, 6))
        
        y_pos = np.arange(len(available_metrics))
        means = []
        half_widths = []
        labels = []
        
        # Define color palette
        colors = plt.cm.Set3(np.linspace(0, 1, len(available_metrics)))
        
        for metric in available_metrics:
            stats_data = self.summary_statistics[metric]
            
            # Convert to percentages
            mean_pct = stats_data['mean'] * 100
            half_width_pct = stats_data['half_width'] * 100
            
            means.append(mean_pct)
            half_widths.append(half_width_pct)
            
            # Clean up label - extract resource name
            resource_name = metric.replace('_utilization', '').replace('_', ' ').title()
            labels.append(resource_name)
        
        # Create horizontal bar plot
        ax.barh(y_pos, means, xerr=half_widths, capsize=8,
               color=colors, edgecolor='black', linewidth=1, alpha=0.8)
        
        # Add value labels on bars
        for i, (mean, half_width) in enumerate(zip(means, half_widths)):
            label_text = f'{mean:.1f}% +/- {half_width:.1f}%'
            
            # Position label inside or outside bar based on bar width
            if mean > 15:
                ax.text(mean/2, i, label_text,
                       ha='center', va='center', fontweight='bold',
                       fontsize=9, color='black')
            else:
                ax.text(mean + half_width + 2, i, label_text,
                       ha='left', va='center', fontweight='bold',
                       fontsize=9, color='black')
        
        # Customize the plot
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=11)
        ax.set_xlabel('Utilization Rate (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'Resource Utilization - 95% Confidence Intervals\n'
                    f'({self.n_replications} replications)',
                    fontsize=14, fontweight='bold', pad=20)
        
        # Set x-axis from 0% to 100%
        ax.set_xlim(0, 100)
        
        # Add reference lines
        ax.axvline(x=85, color='red', linestyle='--', alpha=0.7, linewidth=1,
                  label='85% (Critical Limit)')
        ax.axvline(x=70, color='orange', linestyle='--', alpha=0.5, linewidth=1,
                  label='70% (High Utilization)')
        ax.axvline(x=50, color='green', linestyle='--', alpha=0.5, linewidth=1,
                  label='50% (Moderate Utilization)')
        ax.axvline(x=25, color='blue', linestyle='--', alpha=0.3, linewidth=1,
                  label='25% (Low Utilization)')
        
        # Add grid
        ax.grid(axis='x', alpha=0.3, linestyle='-')
        ax.set_axisbelow(True)
        
        # Add legend
        ax.legend(loc='lower right', framealpha=0.9, fontsize=9)
        
        plt.tight_layout()
        plt.show()
        
        # Print detailed resource analysis
        self._print_resource_analysis(available_metrics)
    
    def _print_resource_analysis(self, metrics: List[str]):
        """Print detailed resource utilization analysis."""
        print("\nDETAILED ANALYSIS OF RESOURCE UTILIZATION:")
        print("=" * 55)
        
        for metric in metrics:
            stats_data = self.summary_statistics[metric]
            mean_util = stats_data['mean'] * 100
            half_width = stats_data['half_width'] * 100
            ci_lower = stats_data['ci_lower'] * 100
            ci_upper = stats_data['ci_upper'] * 100
            precision = stats_data['relative_precision']
            
            resource_name = metric.replace('_utilization', '').replace('_', ' ').title()
            
            print(f"\n{resource_name}:")
            print(f"  Mean utilization: {mean_util:.1f}% +/- {half_width:.1f}%")
            print(f"  CI 95%: [{ci_lower:.1f}%, {ci_upper:.1f}%]")
            print(f"  Relative precision: +/-{precision:.1f}%")
            
            # Recommendations based on utilization level
            if mean_util >= 90:
                print("  RECOMMENDATION: Resource extremely overloaded!")
                print("     Urgently increase capacity")
            elif mean_util >= 85:
                print("  RECOMMENDATION: Resource overloaded")
                print("     Consider increasing capacity")
            elif mean_util >= 70:
                print("  RECOMMENDATION: High utilization")
                print("     Monitor and assess the need for additional resources")
            elif mean_util >= 50:
                print("  RECOMMENDATION: Moderate and efficient use")
                print("     Ideal level for most systems")
            elif mean_util >= 25:
                print("  RECOMMENDATION: Suboptimal usage")
                print("     Evaluate resizing or redistribution")
            else:
                print("  RECOMMENDATION: Underutilized resource")
                print("     Consider reducing capacity")
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """
        Return results as a pandas DataFrame for further analysis.
        
        Returns:
            DataFrame with one row per replication
        """
        df = pd.DataFrame(self.replication_results)
        # print(df.columns)
        df.drop(['replication_id', 'simulation_time',
        'warm_up_period', 'overall_throughput'], axis=1, inplace=True)
        return df
    
    def export_results(self, filename: str = "results/framework_results.csv"):
        """
        Export results to CSV file.
        
        Args:
            filename: Output CSV filename
        """
        df = self.get_results_dataframe()
        df.to_csv(filename, index=False)
        print(f"Results exported to {filename}")