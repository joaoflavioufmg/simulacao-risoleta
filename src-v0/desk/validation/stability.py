# =====================================================================
# FILE: validation/stability.py
# =====================================================================
import statistics
from typing import Dict, List
import simpy


class StabilityAnalyzer:
    """Analyzes system stability and capacity."""
    
    def __init__(self, model):
        self.model = model
    
    def check_system_stability(self, sample_size: int = 1000) -> float:
        """
        Verify if system is mathematically stable.
        
        Args:
            sample_size: Number of samples for statistical estimation
            
        Returns:
            Stability index (>1.0 = stable, <1.0 = unstable)
        """
        print("\n🔍 SYSTEM STABILITY CHECK:")
        print("=" * 50)
        
        # Calculate arrival rate
        total_arrival_rate = self._calculate_arrival_rate(sample_size)
        print(f"📊 Estimated total arrival rate: "
              f"{total_arrival_rate * 60:.1f} entities/hour")
        
        # Find bottleneck resource
        bottleneck_rate, bottleneck_resource = self._find_bottleneck(sample_size)
        system_capacity = bottleneck_rate
        
        print(f"📊 SYSTEM CAPACITY (bottleneck at {bottleneck_resource}): "
              f"{system_capacity * 60:.1f} entities/hour")
        
        # Calculate stability index
        stability = (system_capacity / total_arrival_rate 
                    if total_arrival_rate > 0 else float('inf'))
        print(f"🎯 STABILITY INDEX: {stability:.2f}")
        
        self._print_stability_assessment(stability)
        print("=" * 50)
        
        return stability
    
    def _calculate_arrival_rate(self, sample_size: int) -> float:
        """Calculate total system arrival rate."""
        total_arrival_rate = 0
        
        for create_block in self.model.create_blocks:
            samples = [create_block.inter_arrival_time() 
                      for _ in range(sample_size)]
            avg_interarrival = statistics.mean(samples)
            arrival_rate = 1 / avg_interarrival if avg_interarrival > 0 else 0
            total_arrival_rate += arrival_rate
            print(f"Arrival rate ({create_block.name}): "
                  f"{arrival_rate:.2f} entities/min "
                  f"({arrival_rate*60:.1f}/h)")
        
        return total_arrival_rate
    
    def _find_bottleneck(self, sample_size: int) -> tuple:
        """
        Find bottleneck resource (lowest capacity).
        
        Returns:
            (bottleneck_rate, bottleneck_resource_name)
        """
        # from blocks.process_block import ProcessBlock, MultiProcessBlock
        
        bottleneck_rate = float('inf')
        bottleneck_resource = None
        
        # Group process blocks by resource
        resource_process_blocks = self._group_process_blocks_by_resource()
        
        for resource_name, process_blocks in resource_process_blocks.items():
            if resource_name in self.model.resources:
                resource = self.model.resources[resource_name]
                
                # Find slowest process block for this resource
                slowest_rate = self._calculate_resource_rate(
                    process_blocks, sample_size)
                
                # Resource capacity = capacity × service rate
                resource_capacity = resource.capacity * slowest_rate
                resource_type = ("Priority" if isinstance(resource, 
                                simpy.PriorityResource) else 
                                "Preemptive" if isinstance(resource, 
                                simpy.PreemptiveResource) 
                                else "Regular")
                
                print(f"  📋 {resource_name} ({resource_type}): "
                      f"{resource.capacity} × {slowest_rate:.3f}/min = "
                      f"{resource_capacity:.3f}/min ({resource_capacity * 60:.1f}/h)")
                
                if resource_capacity < bottleneck_rate:
                    bottleneck_rate = resource_capacity
                    bottleneck_resource = resource_name
        
        return bottleneck_rate, bottleneck_resource
    
    def _group_process_blocks_by_resource(self) -> Dict[str, List]:
        """Group process blocks by the resources they use."""
        from desk.blocks.process_block import ProcessBlock, MultiProcessBlock
        
        resource_process_blocks = {}
        
        for block in self.model.blocks.values():
            if isinstance(block, ProcessBlock):
                resource_name = self._find_resource_name(block.resource)
                if resource_name:
                    if resource_name not in resource_process_blocks:
                        resource_process_blocks[resource_name] = []
                    resource_process_blocks[resource_name].append(block)
                    
            elif isinstance(block, MultiProcessBlock):
                for resource, units_required in block.resource_requirements.items():
                    resource_name = self._find_resource_name(resource)
                    if resource_name:
                        if resource_name not in resource_process_blocks:
                            resource_process_blocks[resource_name] = []
                        resource_process_blocks[resource_name].append(
                            (block, units_required))
        
        return resource_process_blocks
    
    def _find_resource_name(self, resource_obj) -> str:
        """Find resource name from object."""
        for name, res in self.model.resources.items():
            if res == resource_obj:
                return name
        return None
    
    def _calculate_resource_rate(self, process_blocks: List, 
                                 sample_size: int) -> float:
        """Calculate effective service rate for a resource."""
        
        slowest_rate = float('inf')
        
        for item in process_blocks:
            if isinstance(item, tuple):  # MultiProcessBlock with units
                process_block, units_required = item
                samples = [process_block.delay_time() 
                          for _ in range(sample_size)]
                avg_service_time = statistics.mean(samples)
                service_rate = (1 / avg_service_time 
                              if avg_service_time > 0 else 0)
                effective_rate = service_rate / units_required
            else:  # Regular ProcessBlock
                process_block = item
                samples = [process_block.delay_time() 
                          for _ in range(sample_size)]
                avg_service_time = statistics.mean(samples)
                service_rate = (1 / avg_service_time 
                              if avg_service_time > 0 else 0)
                effective_rate = service_rate
            
            if effective_rate < slowest_rate:
                slowest_rate = effective_rate
        
        return slowest_rate
    
    def _print_stability_assessment(self, stability: float):
        """Print assessment of stability index."""
        if stability > 1.2:
            print("✅ Oversized system (capacity >> demand)")
        elif stability > 1.05:
            print("✅ Stable system (capacity > demand)")
        elif stability > 0.95:
            print("⚠️ System AT MAXIMUM CAPACITY (capacity ≈ demand) - caution!")
        elif stability > 0.8:
            print("🚨 UNSTABLE system (demand > capacity)")
        else:
            print("💥 IMMINENT COLLAPSE (demand >> capacity)")