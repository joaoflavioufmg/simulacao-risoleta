# =====================================================================
# FILE: analytics/metrics.py
# =====================================================================
import statistics
import math
from typing import Dict, Any, List, Tuple, Callable
import numpy as np


class MetricsCollector:
    """Collects and calculates metrics from a completed simulation."""
    
    def __init__(self, model):
        """
        Initialize metrics collector.
        
        Args:
            model: SimulationModel instance with completed simulation
        """
        self.model = model
    
    def get_entity_metrics_summary(self) -> Dict[str, Any]:
        """
        Calculate entity-level metrics (time in system, by activity).
        
        Returns:
            Dictionary containing system time and per-activity metrics
        """
        if not self.model.dispose_blocks:
            return {'tempo_medio_sistema': 0, 'atividades': {}}
        
        # Collect only post-warm-up disposed entities
        post_warmup_entities = [
            e for dispose_block in self.model.dispose_blocks
            for e in dispose_block.disposed_entities
            if e.get_attribute('disposal_time', 0) >= self.model.warm_up_period
        ]
        
        if not post_warmup_entities:
            return {'tempo_medio_sistema': 0, 'atividades': {}}
        
        # Calculate system time
        system_times = [entity.get_attribute('system_time', 0) 
                       for entity in post_warmup_entities]
        
        # Group metrics by activity
        activity_queue_times = {}
        activity_service_times = {}
        activity_system_times = {}
        
        for entity in post_warmup_entities:
            for key, value in entity.data.items():
                # Skip None or nan values
                if value is None or (isinstance(value, float) and math.isnan(value)):
                    continue
                    
                if key.endswith('_queue_time'):
                    activity_name = key.replace('_queue_time', '')
                    if activity_name not in activity_queue_times:
                        activity_queue_times[activity_name] = []
                    activity_queue_times[activity_name].append(value)
                    
                elif key.endswith('_service_time'):
                    activity_name = key.replace('_service_time', '')
                    if activity_name not in activity_service_times:
                        activity_service_times[activity_name] = []
                    activity_service_times[activity_name].append(value)
        
        # Calculate system time for each activity
        all_activities = set(list(activity_queue_times.keys()) + 
                           list(activity_service_times.keys()))
        
        for activity_name in all_activities:
            queue_times = activity_queue_times.get(activity_name, [])
            service_times = activity_service_times.get(activity_name, [])
            
            activity_system_times[activity_name] = []
            min_length = min(len(queue_times), len(service_times))
            
            for i in range(min_length):
                system_time = queue_times[i] + service_times[i]
                activity_system_times[activity_name].append(system_time)
        
        # Build summary
        summary = {
            'tempo_medio_sistema': statistics.mean(system_times) if system_times else 0,
            'atividades': {}
        }
        
        for activity_name in all_activities:
            qt = activity_queue_times.get(activity_name, [])
            st = activity_service_times.get(activity_name, [])
            sys_t = activity_system_times.get(activity_name, [])
            
            summary['atividades'][activity_name] = {
                'tempo_medio_fila': statistics.mean(qt) if len(qt) > 0 else 0,
                'tempo_medio_atendimento': statistics.mean(st) if len(st) > 0 else 0,
                'tempo_medio_sistema': statistics.mean(sys_t) if len(sys_t) > 0 else 0
            }
        
        return summary
    
    def get_resource_metrics_summary(self) -> Dict[str, Any]:
        """
        Calculate resource-level metrics (utilization, queue lengths).
        
        Returns:
            Dictionary mapping resource names to their metrics
        """
        from desk.blocks.process_block import ProcessBlock, MultiProcessBlock
        
        summary = {}
        
        # Group ProcessBlocks by resource
        resource_blocks = self._group_blocks_by_resource()
        
        for resource_name, blocks in resource_blocks.items():
            if resource_name in self.model.resources:
                resource = self.model.resources[resource_name]
                resource_obj = resource
                
                # Combine data from all blocks using this resource
                combined_data = []
                max_queue_length = 0
                max_in_service = 0
                
                for block in blocks:
                    if isinstance(block, ProcessBlock):
                        combined_data.extend(block.resource_data)
                        max_queue_length = max(max_queue_length, block.max_queue_length)
                        max_in_service = max(max_in_service, block.max_in_service)
                    elif isinstance(block, MultiProcessBlock):
                        if resource_obj in block.resource_data:
                            combined_data.extend(block.resource_data[resource_obj])
                            metrics = block.max_metrics[resource_obj]
                            max_queue_length = max(max_queue_length, 
                                                  metrics['max_queue_length'])
                            max_in_service = max(max_in_service, 
                                                metrics['max_in_service'])
                
                # Deduplicate data points by time
                combined_data = self._deduplicate_resource_data(combined_data)
                
                # Calculate metrics
                if combined_data:
                    avg_queue = self._calculate_time_weighted_avg(
                        combined_data, lambda x: x[2])
                    avg_in_service = self._calculate_time_weighted_avg(
                        combined_data, lambda x: x[1])
                    utilization = (avg_in_service / resource.capacity 
                                 if resource.capacity > 0 else 0)
                    
                    busy_time, idle_time = self._calculate_busy_idle_time(
                        combined_data, resource)
                else:
                    avg_queue = 0
                    avg_in_service = 0
                    utilization = 0
                    busy_time = 0
                    idle_time = self.model.env.now - self.model.warm_up_period
                
                effective_time = self.model.env.now - self.model.warm_up_period
                
                summary[resource_name] = {
                    'numero_medio_fila': avg_queue,
                    'numero_medio_atendimento': avg_in_service,
                    'numero_medio_sistema': avg_queue + avg_in_service,
                    'taxa_utilizacao': utilization,
                    'maximo_fila': max_queue_length,
                    'maximo_atendimento': max_in_service,
                    'maximo_sistema': max_queue_length + max_in_service,
                    'tempo_ocupado': busy_time,
                    'tempo_ocioso': idle_time,
                    'percentual_ocupacao': ((busy_time / effective_time * 100) 
                                           if effective_time > 0 else 0),
                    'percentual_ociosidade': ((idle_time / effective_time * 100) 
                                             if effective_time > 0 else 0)
                }
        
        return summary
    
    def _group_blocks_by_resource(self) -> Dict[str, List]:
        """Group ProcessBlocks by the resources they use."""
        from desk.blocks.process_block import ProcessBlock, MultiProcessBlock
        
        resource_blocks = {}
        
        for block in self.model.blocks.values():
            if isinstance(block, ProcessBlock):
                resource_name = self._find_resource_name(block.resource)
                if resource_name:
                    if resource_name not in resource_blocks:
                        resource_blocks[resource_name] = []
                    resource_blocks[resource_name].append(block)
                    
            elif isinstance(block, MultiProcessBlock):
                for resource in block.resource_requirements.keys():
                    resource_name = self._find_resource_name(resource)
                    if resource_name:
                        if resource_name not in resource_blocks:
                            resource_blocks[resource_name] = []
                        resource_blocks[resource_name].append(block)
        
        return resource_blocks
    
    def _find_resource_name(self, resource_obj) -> str:
        """Find resource name from resource object."""
        for res_name, res in self.model.resources.items():
            if res == resource_obj:
                return res_name
        return None
    
    def _deduplicate_resource_data(self, data: List[Tuple]) -> List[Tuple]:
        """Deduplicate resource data points by timestamp."""
        if not data:
            return []
        
        data.sort(key=lambda x: x[0])
        from itertools import groupby
        
        unique_data = []
        for timestamp, group in groupby(data, key=lambda x: x[0]):
            group_list = list(group)
            unique_data.append(group_list[-1])  # Keep last state at timestamp
        
        return unique_data
    
    def _calculate_time_weighted_avg(self, data: List[Tuple], 
                                     extractor: Callable) -> float:
        """Calculate time-weighted average from resource data."""
        if not data:
            return 0
        
        data.sort(key=lambda x: x[0])
        effective_time = self.model.env.now - self.model.warm_up_period
        
        if effective_time <= 0:
            return 0
        
        area = 0
        prev_time = self.model.warm_up_period
        
        # Find initial value at warm-up boundary
        pre_warmup_data = [point for point in data 
                          if point[0] <= self.model.warm_up_period]
        post_warmup_data = [point for point in data 
                           if point[0] > self.model.warm_up_period]
        
        if pre_warmup_data:
            prev_value = extractor(pre_warmup_data[-1])
        else:
            prev_value = 0
        
        # Process all post-warmup data points
        for point in post_warmup_data:
            time = point[0]
            area += prev_value * (time - prev_time)
            prev_time = time
            prev_value = extractor(point)
        
        # Add final interval
        area += prev_value * (self.model.env.now - prev_time)
        
        return area / effective_time if effective_time > 0 else 0
    
    def _calculate_busy_idle_time(self, data: List[Tuple], 
                                  resource) -> Tuple[float, float]:
        """Calculate busy and idle time for a resource."""
        busy_time = 0
        idle_time = 0
        prev_time = self.model.warm_up_period
        prev_count = 0
        
        post_warmup_data = [p for p in data if p[0] >= self.model.warm_up_period]
        
        if post_warmup_data:
            prev_count = post_warmup_data[0][1]
            prev_time = post_warmup_data[0][0]
            
            for time, count, qlen in post_warmup_data[1:]:
                time_interval = time - prev_time
                if prev_count > 0:
                    busy_time += time_interval
                else:
                    idle_time += time_interval
                prev_time = time
                prev_count = count
            
            final_interval = self.model.env.now - prev_time
            if prev_count > 0:
                busy_time += final_interval
            else:
                idle_time += final_interval
        else:
            idle_time = self.model.env.now - self.model.warm_up_period
        
        return busy_time, idle_time