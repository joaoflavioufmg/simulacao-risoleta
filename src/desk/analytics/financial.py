# =====================================================================
# FILE: analytics/financial.py
# =====================================================================
"""
Financial analysis tools for simulation models.

Provides methods for:
- Calculating revenue, costs, and profit from entity attributes
- Generating financial balance sheets
- Visualizing financial breakdowns
"""

from typing import Dict, Any
import matplotlib.pyplot as plt


class FinancialAnalyzer:
    """Analyzes financial metrics from simulation results."""
    
    def __init__(self, model):
        """
        Initialize financial analyzer.
        
        Args:
            model: SimulationModel instance with completed simulation
        """
        self.model = model
    
    def get_financial_summary(self) -> Dict[str, Any]:
        """
        Calculate financial summary from disposed entities.
        
        Returns:
            Dictionary with total revenue, costs by activity, and net profit
        """
        if not self.model.dispose_blocks:
            return self._empty_summary()
        
        post_warmup_entities = self._get_post_warmup_entities()
        
        if not post_warmup_entities:
            return self._empty_summary()
        
        total_revenue = 0
        costs_by_activity = {}
        
        for entity in post_warmup_entities:
            # print(f"[DEBUG ACTIVITY_NAME] {entity.data.items()}")
            for key, value in entity.data.items():
                if 'revenue' in key.lower() and isinstance(value, (int, float)):
                    total_revenue += value
                
                if '_cost' in key.lower() and isinstance(value, (int, float)):                    
                    # extract activity name from "BlockName_cost"
                    activity_name = key.replace('_cost', '').split('_')[0] if '_' in key else key
                    # print(f"[DEBUG ACTIVITY_NAME] {activity_name}")                    
                    if activity_name not in costs_by_activity:
                        costs_by_activity[activity_name] = 0
                    costs_by_activity[activity_name] += value
        
        
        total_costs = sum(costs_by_activity.values())
        net_profit = total_revenue - total_costs
        n_entities = len(post_warmup_entities)
        
        return {
            'total_revenue': total_revenue,
            'total_costs': total_costs,
            'net_profit': net_profit,
            'costs_by_activity': costs_by_activity,
            'num_entities': n_entities,
            'avg_revenue_per_entity': total_revenue / n_entities if n_entities else 0,
            'avg_cost_per_entity': total_costs / n_entities if n_entities else 0,
            'avg_profit_per_entity': net_profit / n_entities if n_entities else 0
        }
    
    def _empty_summary(self) -> Dict[str, Any]:
        """Return empty financial summary."""
        return {
            'total_revenue': 0,
            'total_costs': 0,
            'net_profit': 0,
            'costs_by_activity': {},
            'num_entities': 0,
            'avg_revenue_per_entity': 0,
            'avg_cost_per_entity': 0,
            'avg_profit_per_entity': 0
        }
    
    def _get_post_warmup_entities(self):
        """Get entities disposed after warm-up period."""
        return [
            e for dispose_block in self.model.dispose_blocks
            for e in dispose_block.disposed_entities
            if e.get_attribute('disposal_time', 0) >= self.model.warm_up_period
        ]
    
    def print_financial_summary(self):
        """Print formatted financial balance sheet."""
        financial_data = self.get_financial_summary()
        
        print("\n" + "=" * 60)
        print("FINANCIAL BALANCE SHEET")
        print("=" * 60)
        
        print(f"\nBased on {financial_data['num_entities']} entities (post warm-up)")
        
        self._print_revenue_section(financial_data)
        self._print_costs_section(financial_data)
        self._print_profit_section(financial_data)
        
        print("=" * 60)
    
    def _print_revenue_section(self, data: Dict):
        """Print revenue section."""
        print("\nREVENUE:")
        print(f"  Total Revenue: ${data['total_revenue']:,.2f}")
        print(f"  Average per Entity: ${data['avg_revenue_per_entity']:,.2f}")
    
    def _print_costs_section(self, data: Dict):
        """Print costs section."""
        print("\nCOSTS BY ACTIVITY:")
        if data['costs_by_activity']:
            for activity, cost in sorted(data['costs_by_activity'].items(),
                                        key=lambda x: x[1], reverse=True):
                percentage = (cost / data['total_costs'] * 100) if data['total_costs'] > 0 else 0
                print(f"  {activity}: ${cost:,.2f} ({percentage:.1f}%)")
        else:
            print("  No cost data available")
        
        print(f"\n  Total Costs: ${data['total_costs']:,.2f}")
        print(f"  Average per Entity: ${data['avg_cost_per_entity']:,.2f}")
    
    def _print_profit_section(self, data: Dict):
        """Print profit section with analysis."""
        print("\n" + "-" * 60)
        print(f"NET PROFIT: ${data['net_profit']:,.2f}")
        print(f"   Average per Entity: ${data['avg_profit_per_entity']:,.2f}")
        
        if data['total_revenue'] > 0:
            profit_margin = (data['net_profit'] / data['total_revenue']) * 100
            print(f"   Profit Margin: {profit_margin:.1f}%")
            
            if profit_margin > 20:
                print("   Excellent profit margin")
            elif profit_margin > 10:
                print("   Good profit margin")
            elif profit_margin > 0:
                print("   Low profit margin")
            else:
                print("   Operating at a loss!")
    
    def plot_financial_breakdown(self):
        """Create visualizations for financial data."""
        financial_data = self.get_financial_summary()
        
        if not financial_data['costs_by_activity']:
            print("No financial data available to plot.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Pie chart: Cost distribution
        activities = list(financial_data['costs_by_activity'].keys())
        costs = list(financial_data['costs_by_activity'].values())
        
        ax1.pie(costs, labels=activities, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Cost Distribution by Activity', fontsize=14, fontweight='bold')
        
        # Bar chart: Revenue vs Costs vs Profit
        categories = ['Revenue', 'Costs', 'Net Profit']
        values = [
            financial_data['total_revenue'],
            financial_data['total_costs'],
            financial_data['net_profit']
        ]
        colors = ['green', 'red', 'blue' if financial_data['net_profit'] >= 0 else 'darkred']
        
        bars = ax2.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_ylabel('Amount ($)', fontsize=12, fontweight='bold')
        ax2.set_title('Financial Overview', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'${value:,.0f}',
                    ha='center', va='bottom' if value >= 0 else 'top',
                    fontweight='bold', fontsize=10)
        
        plt.tight_layout()
        plt.show()