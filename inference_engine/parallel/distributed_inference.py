"""
Distributed Inference Algorithms
================================

Advanced parallel algorithms for distributed symbolic inference across multiple GPUs.
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
import time
import threading
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import queue
import asyncio
from collections import defaultdict

from ...arbitrary_numbers.core.rational_list import RationalListNumber
from ...arbitrary_numbers.core.equation_nodes import EquationNode
from ...arbitrary_numbers.ml.pytorch_layers import ExplainableInferenceModel
from ..gpu.inference_kernels import GPUInferenceEngine, InferenceKernelConfig


@dataclass
class ParallelConfig:
    """Configuration for parallel inference algorithms."""
    num_gpus: int = 4
    num_cpu_workers: int = 8
    batch_size_per_gpu: int = 32
    pipeline_depth: int = 4
    memory_overlap: bool = True
    async_execution: bool = True
    load_balancing_strategy: str = "dynamic"
    communication_backend: str = "nccl"


class WorkloadPartitioner:
    """Intelligent workload partitioning for parallel inference."""
    
    def __init__(self, config: ParallelConfig):
        self.config = config
        self.gpu_capabilities = {}
        self.workload_history = defaultdict(list)
        self.performance_model = {}
    
    def partition_batch(self, batch_inputs: torch.Tensor, 
                       model_complexity: Dict[str, Any]) -> List[Tuple[torch.Tensor, int]]:
        """Partition batch across available GPUs based on capability."""
        batch_size = batch_inputs.size(0)
        
        if self.config.load_balancing_strategy == "static":
            return self._static_partition(batch_inputs)
        elif self.config.load_balancing_strategy == "dynamic":
            return self._dynamic_partition(batch_inputs, model_complexity)
        else:
            return self._adaptive_partition(batch_inputs, model_complexity)
    
    def partition_symbolic_expressions(self, expressions: List[EquationNode]) -> List[List[EquationNode]]:
        """Partition symbolic expressions for parallel evaluation."""
        num_partitions = self.config.num_cpu_workers
        
        complexity_scores = [expr.complexity() for expr in expressions]
        
        partitions = [[] for _ in range(num_partitions)]
        partition_loads = [0] * num_partitions
        
        sorted_indices = sorted(range(len(expressions)), 
                              key=lambda i: complexity_scores[i], reverse=True)
        
        for idx in sorted_indices:
            min_load_partition = min(range(num_partitions), key=lambda i: partition_loads[i])
            partitions[min_load_partition].append(expressions[idx])
            partition_loads[min_load_partition] += complexity_scores[idx]
        
        return partitions
    
    def update_performance_model(self, gpu_id: int, batch_size: int, 
                               inference_time: float, model_complexity: Dict[str, Any]) -> None:
        """Update performance model with new measurements."""
        key = (gpu_id, batch_size, model_complexity.get('total_parameters', 0))
        
        if key not in self.performance_model:
            self.performance_model[key] = []
        
        self.performance_model[key].append(inference_time)
        
        if len(self.performance_model[key]) > 100:
            self.performance_model[key] = self.performance_model[key][-50:]
    
    def _static_partition(self, batch_inputs: torch.Tensor) -> List[Tuple[torch.Tensor, int]]:
        """Static equal partitioning across GPUs."""
        batch_size = batch_inputs.size(0)
        chunk_size = batch_size // self.config.num_gpus
        
        partitions = []
        for gpu_id in range(self.config.num_gpus):
            start_idx = gpu_id * chunk_size
            if gpu_id == self.config.num_gpus - 1:
                end_idx = batch_size
            else:
                end_idx = start_idx + chunk_size
            
            chunk = batch_inputs[start_idx:end_idx]
            partitions.append((chunk, gpu_id))
        
        return partitions
    
    def _dynamic_partition(self, batch_inputs: torch.Tensor, 
                          model_complexity: Dict[str, Any]) -> List[Tuple[torch.Tensor, int]]:
        """Dynamic partitioning based on GPU capabilities."""
        batch_size = batch_inputs.size(0)
        
        gpu_weights = []
        for gpu_id in range(self.config.num_gpus):
            if gpu_id in self.gpu_capabilities:
                weight = self.gpu_capabilities[gpu_id]['compute_capability']
            else:
                weight = 1.0
            gpu_weights.append(weight)
        
        total_weight = sum(gpu_weights)
        normalized_weights = [w / total_weight for w in gpu_weights]
        
        partitions = []
        current_idx = 0
        
        for gpu_id, weight in enumerate(normalized_weights):
            chunk_size = int(batch_size * weight)
            if gpu_id == len(normalized_weights) - 1:
                chunk_size = batch_size - current_idx
            
            if chunk_size > 0:
                chunk = batch_inputs[current_idx:current_idx + chunk_size]
                partitions.append((chunk, gpu_id))
                current_idx += chunk_size
        
        return partitions
    
    def _adaptive_partition(self, batch_inputs: torch.Tensor,
                           model_complexity: Dict[str, Any]) -> List[Tuple[torch.Tensor, int]]:
        """Adaptive partitioning based on historical performance."""
        batch_size = batch_inputs.size(0)
        
        predicted_times = []
        for gpu_id in range(self.config.num_gpus):
            predicted_time = self._predict_inference_time(gpu_id, 
                                                        self.config.batch_size_per_gpu,
                                                        model_complexity)
            predicted_times.append(predicted_time)
        
        gpu_speeds = [1.0 / max(t, 0.001) for t in predicted_times]
        total_speed = sum(gpu_speeds)
        normalized_speeds = [s / total_speed for s in gpu_speeds]
        
        partitions = []
        current_idx = 0
        
        for gpu_id, speed_ratio in enumerate(normalized_speeds):
            chunk_size = int(batch_size * speed_ratio)
            if gpu_id == len(normalized_speeds) - 1:
                chunk_size = batch_size - current_idx
            
            if chunk_size > 0:
                chunk = batch_inputs[current_idx:current_idx + chunk_size]
                partitions.append((chunk, gpu_id))
                current_idx += chunk_size
        
        return partitions
    
    def _predict_inference_time(self, gpu_id: int, batch_size: int,
                               model_complexity: Dict[str, Any]) -> float:
        """Predict inference time based on historical data."""
        key = (gpu_id, batch_size, model_complexity.get('total_parameters', 0))
        
        if key in self.performance_model and self.performance_model[key]:
            return np.mean(self.performance_model[key])
        
        base_time = 0.01
        complexity_factor = model_complexity.get('total_parameters', 1000) / 1000
        batch_factor = batch_size / 32
        
        return base_time * complexity_factor * batch_factor


class PipelinedInferenceEngine:
    """Pipelined inference engine with overlapped computation and communication."""
    
    def __init__(self, config: ParallelConfig):
        self.config = config
        self.gpu_engines = {}
        self.pipeline_stages = queue.Queue(maxsize=config.pipeline_depth)
        self.result_queue = queue.Queue()
        self.partitioner = WorkloadPartitioner(config)
        
        self.pipeline_stats = {
            'stages_processed': 0,
            'total_pipeline_time': 0.0,
            'average_stage_time': 0.0,
            'pipeline_utilization': 0.0
        }
        
        self._initialize_gpu_engines()
        self._start_pipeline_workers()
    
    def infer_pipelined(self, model_name: str, batch_inputs: torch.Tensor,
                       require_symbolic: bool = False) -> Dict[str, Any]:
        """Execute pipelined inference across multiple GPUs."""
        start_time = time.time()
        
        model_complexity = self._get_model_complexity(model_name)
        partitions = self.partitioner.partition_batch(batch_inputs, model_complexity)
        
        stage_id = f"stage_{int(time.time() * 1000000)}"
        
        pipeline_stage = {
            'stage_id': stage_id,
            'model_name': model_name,
            'partitions': partitions,
            'require_symbolic': require_symbolic,
            'start_time': start_time
        }
        
        self.pipeline_stages.put(pipeline_stage)
        
        result = self._wait_for_result(stage_id)
        
        total_time = time.time() - start_time
        self.pipeline_stats['stages_processed'] += 1
        self.pipeline_stats['total_pipeline_time'] += total_time
        self.pipeline_stats['average_stage_time'] = (
            self.pipeline_stats['total_pipeline_time'] / 
            self.pipeline_stats['stages_processed']
        )
        
        return result
    
    def infer_async_batch(self, model_name: str, batch_inputs: List[torch.Tensor]) -> List[Dict[str, Any]]:
        """Asynchronous batch inference with maximum parallelism."""
        futures = []
        
        with ThreadPoolExecutor(max_workers=len(batch_inputs)) as executor:
            for batch in batch_inputs:
                future = executor.submit(self.infer_pipelined, model_name, batch)
                futures.append(future)
            
            results = []
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=30.0)
                    results.append(result)
                except Exception as e:
                    results.append({'error': str(e)})
        
        return results
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline performance statistics."""
        queue_utilization = (self.pipeline_stages.qsize() / 
                           max(1, self.config.pipeline_depth)) * 100
        
        return {
            **self.pipeline_stats,
            'queue_utilization': queue_utilization,
            'active_gpu_engines': len(self.gpu_engines),
            'pipeline_depth': self.config.pipeline_depth
        }
    
    def _initialize_gpu_engines(self) -> None:
        """Initialize GPU inference engines for each device."""
        for gpu_id in range(self.config.num_gpus):
            if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
                config = InferenceKernelConfig(
                    max_batch_size=self.config.batch_size_per_gpu,
                    memory_pool_size_gb=8.0
                )
                
                with torch.cuda.device(gpu_id):
                    engine = GPUInferenceEngine(config)
                    self.gpu_engines[gpu_id] = engine
    
    def _start_pipeline_workers(self) -> None:
        """Start pipeline worker threads."""
        for i in range(self.config.num_gpus):
            worker_thread = threading.Thread(
                target=self._pipeline_worker,
                args=(i,),
                daemon=True
            )
            worker_thread.start()
    
    def _pipeline_worker(self, worker_id: int) -> None:
        """Pipeline worker that processes stages."""
        while True:
            try:
                stage = self.pipeline_stages.get(timeout=1.0)
                
                result = self._process_pipeline_stage(stage, worker_id)
                
                self.result_queue.put((stage['stage_id'], result))
                self.pipeline_stages.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                error_result = {'error': str(e), 'worker_id': worker_id}
                self.result_queue.put((stage.get('stage_id', 'unknown'), error_result))
    
    def _process_pipeline_stage(self, stage: Dict[str, Any], worker_id: int) -> Dict[str, Any]:
        """Process a single pipeline stage."""
        model_name = stage['model_name']
        partitions = stage['partitions']
        require_symbolic = stage['require_symbolic']
        
        partition_results = []
        
        for partition_data, gpu_id in partitions:
            if gpu_id in self.gpu_engines:
                engine = self.gpu_engines[gpu_id]
                
                if model_name not in engine.models:
                    continue
                
                partition_result = engine.infer_batch(
                    model_name, partition_data, require_symbolic
                )
                partition_results.append(partition_result)
        
        combined_outputs = self._combine_partition_results(partition_results)
        
        return {
            'outputs': combined_outputs,
            'stage_id': stage['stage_id'],
            'processing_time': time.time() - stage['start_time'],
            'partitions_processed': len(partition_results),
            'worker_id': worker_id
        }
    
    def _combine_partition_results(self, partition_results: List[Dict[str, Any]]) -> torch.Tensor:
        """Combine results from multiple GPU partitions."""
        if not partition_results:
            return torch.tensor([])
        
        outputs = [result['outputs'] for result in partition_results if 'outputs' in result]
        
        if outputs:
            return torch.cat(outputs, dim=0)
        else:
            return torch.tensor([])
    
    def _wait_for_result(self, stage_id: str, timeout: float = 30.0) -> Dict[str, Any]:
        """Wait for pipeline stage result."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                result_stage_id, result = self.result_queue.get(timeout=1.0)
                
                if result_stage_id == stage_id:
                    return result
                else:
                    self.result_queue.put((result_stage_id, result))
                
            except queue.Empty:
                continue
        
        return {'error': 'Pipeline stage timeout', 'stage_id': stage_id}
    
    def _get_model_complexity(self, model_name: str) -> Dict[str, Any]:
        """Get model complexity from any available GPU engine."""
        for engine in self.gpu_engines.values():
            if model_name in engine.models:
                return engine.models[model_name]['complexity']
        
        return {'total_parameters': 1000}


class SymbolicExpressionParallelizer:
    """Parallel evaluation of symbolic expressions with work stealing."""
    
    def __init__(self, num_workers: int = 8):
        self.num_workers = num_workers
        self.work_queues = [queue.Queue() for _ in range(num_workers)]
        self.result_dict = {}
        self.result_lock = threading.RLock()
        
        self.workers = []
        self._start_workers()
    
    def evaluate_expressions_parallel(self, expressions: List[EquationNode],
                                    variables: Dict[str, RationalListNumber]) -> List[RationalListNumber]:
        """Evaluate multiple expressions in parallel with work stealing."""
        if not expressions:
            return []
        
        task_id = f"task_{int(time.time() * 1000000)}"
        
        for i, expr in enumerate(expressions):
            worker_id = i % self.num_workers
            task = {
                'task_id': task_id,
                'expr_id': i,
                'expression': expr,
                'variables': variables
            }
            self.work_queues[worker_id].put(task)
        
        results = self._collect_results(task_id, len(expressions))
        
        return [results[i] for i in range(len(expressions))]
    
    def optimize_expressions_parallel(self, expressions: List[EquationNode]) -> List[EquationNode]:
        """Optimize multiple expressions in parallel."""
        from ..optimization.symbolic_optimizer import SymbolicOptimizer
        
        optimizer = SymbolicOptimizer(enable_parallel_optimization=True)
        
        return optimizer.optimize_batch_expressions(expressions)
    
    def _start_workers(self) -> None:
        """Start worker threads with work stealing capability."""
        for worker_id in range(self.num_workers):
            worker = threading.Thread(
                target=self._worker_with_stealing,
                args=(worker_id,),
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
    
    def _worker_with_stealing(self, worker_id: int) -> None:
        """Worker thread that can steal work from other queues."""
        from ...arbitrary_numbers.core.evaluator import EquationEvaluator
        
        evaluator = EquationEvaluator()
        
        while True:
            task = self._get_task_with_stealing(worker_id)
            
            if task is None:
                time.sleep(0.001)
                continue
            
            try:
                result = evaluator.evaluate(task['expression'], task['variables'])
                
                with self.result_lock:
                    task_id = task['task_id']
                    if task_id not in self.result_dict:
                        self.result_dict[task_id] = {}
                    
                    self.result_dict[task_id][task['expr_id']] = result
                    
            except Exception as e:
                with self.result_lock:
                    task_id = task['task_id']
                    if task_id not in self.result_dict:
                        self.result_dict[task_id] = {}
                    
                    self.result_dict[task_id][task['expr_id']] = f"Error: {e}"
    
    def _get_task_with_stealing(self, worker_id: int) -> Optional[Dict[str, Any]]:
        """Get task from own queue or steal from others."""
        try:
            return self.work_queues[worker_id].get_nowait()
        except queue.Empty:
            pass
        
        for other_id in range(self.num_workers):
            if other_id != worker_id:
                try:
                    return self.work_queues[other_id].get_nowait()
                except queue.Empty:
                    continue
        
        return None
    
    def _collect_results(self, task_id: str, expected_count: int, 
                        timeout: float = 30.0) -> Dict[int, RationalListNumber]:
        """Collect results from parallel evaluation."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            with self.result_lock:
                if (task_id in self.result_dict and 
                    len(self.result_dict[task_id]) >= expected_count):
                    
                    results = self.result_dict[task_id].copy()
                    del self.result_dict[task_id]
                    return results
            
            time.sleep(0.001)
        
        with self.result_lock:
            if task_id in self.result_dict:
                results = self.result_dict[task_id].copy()
                del self.result_dict[task_id]
                return results
        
        return {}


class AdaptiveLoadBalancer:
    """Adaptive load balancer for dynamic workload distribution."""
    
    def __init__(self, config: ParallelConfig):
        self.config = config
        self.gpu_loads = [0.0] * config.num_gpus
        self.gpu_performance = [1.0] * config.num_gpus
        self.load_history = defaultdict(list)
        self.rebalance_threshold = 0.2
        
        self.monitoring_thread = threading.Thread(
            target=self._monitor_loads,
            daemon=True
        )
        self.monitoring_thread.start()
    
    def get_optimal_gpu(self, workload_size: int) -> int:
        """Get optimal GPU for given workload size."""
        adjusted_loads = []
        
        for i in range(self.config.num_gpus):
            adjusted_load = self.gpu_loads[i] / self.gpu_performance[i]
            adjusted_loads.append(adjusted_load)
        
        return min(range(len(adjusted_loads)), key=lambda i: adjusted_loads[i])
    
    def update_gpu_load(self, gpu_id: int, load_delta: float) -> None:
        """Update GPU load measurement."""
        if 0 <= gpu_id < len(self.gpu_loads):
            self.gpu_loads[gpu_id] = max(0.0, self.gpu_loads[gpu_id] + load_delta)
            
            current_time = time.time()
            self.load_history[gpu_id].append((current_time, self.gpu_loads[gpu_id]))
            
            if len(self.load_history[gpu_id]) > 100:
                self.load_history[gpu_id] = self.load_history[gpu_id][-50:]
    
    def should_rebalance(self) -> bool:
        """Determine if workload rebalancing is needed."""
        if not self.gpu_loads:
            return False
        
        max_load = max(self.gpu_loads)
        min_load = min(self.gpu_loads)
        
        if max_load == 0:
            return False
        
        load_imbalance = (max_load - min_load) / max_load
        return load_imbalance > self.rebalance_threshold
    
    def get_rebalancing_plan(self) -> List[Tuple[int, int, float]]:
        """Generate plan for rebalancing workloads."""
        if not self.should_rebalance():
            return []
        
        plan = []
        sorted_gpus = sorted(range(len(self.gpu_loads)), 
                           key=lambda i: self.gpu_loads[i])
        
        low_load_gpus = sorted_gpus[:len(sorted_gpus)//2]
        high_load_gpus = sorted_gpus[len(sorted_gpus)//2:]
        
        for high_gpu in reversed(high_load_gpus):
            for low_gpu in low_load_gpus:
                load_diff = self.gpu_loads[high_gpu] - self.gpu_loads[low_gpu]
                
                if load_diff > self.rebalance_threshold:
                    transfer_amount = load_diff * 0.3
                    plan.append((high_gpu, low_gpu, transfer_amount))
                    break
        
        return plan
    
    def _monitor_loads(self) -> None:
        """Background thread to monitor GPU loads."""
        while True:
            try:
                for gpu_id in range(self.config.num_gpus):
                    if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
                        with torch.cuda.device(gpu_id):
                            memory_used = torch.cuda.memory_allocated()
                            memory_total = torch.cuda.get_device_properties(gpu_id).total_memory
                            
                            current_load = memory_used / memory_total
                            
                            decay_factor = 0.9
                            self.gpu_loads[gpu_id] = (decay_factor * self.gpu_loads[gpu_id] + 
                                                    (1 - decay_factor) * current_load)
                
                time.sleep(0.1)
                
            except Exception:
                time.sleep(1.0)


class ParallelInferenceOrchestrator:
    """High-level orchestrator for all parallel inference algorithms."""
    
    def __init__(self, config: Optional[ParallelConfig] = None):
        self.config = config or ParallelConfig()
        
        self.pipelined_engine = PipelinedInferenceEngine(self.config)
        self.expression_parallelizer = SymbolicExpressionParallelizer(self.config.num_cpu_workers)
        self.load_balancer = AdaptiveLoadBalancer(self.config)
        
        self.orchestrator_stats = {
            'total_requests': 0,
            'parallel_requests': 0,
            'sequential_fallbacks': 0,
            'average_parallelization_factor': 0.0
        }
    
    def infer_with_optimal_parallelization(self, model_name: str, 
                                         batch_inputs: torch.Tensor,
                                         require_symbolic: bool = False) -> Dict[str, Any]:
        """Automatically choose optimal parallelization strategy."""
        batch_size = batch_inputs.size(0)
        
        self.orchestrator_stats['total_requests'] += 1
        
        if batch_size >= self.config.batch_size_per_gpu * 2:
            result = self.pipelined_engine.infer_pipelined(
                model_name, batch_inputs, require_symbolic
            )
            self.orchestrator_stats['parallel_requests'] += 1
            parallelization_factor = min(batch_size / self.config.batch_size_per_gpu, 
                                       self.config.num_gpus)
        else:
            optimal_gpu = self.load_balancer.get_optimal_gpu(batch_size)
            
            if optimal_gpu in self.pipelined_engine.gpu_engines:
                engine = self.pipelined_engine.gpu_engines[optimal_gpu]
                result = engine.infer_batch(model_name, batch_inputs, require_symbolic)
            else:
                result = {'error': 'No available GPU engines'}
            
            self.orchestrator_stats['sequential_fallbacks'] += 1
            parallelization_factor = 1.0
        
        self._update_parallelization_stats(parallelization_factor)
        
        return result
    
    def register_model_on_all_gpus(self, model_name: str, model: ExplainableInferenceModel) -> None:
        """Register model on all available GPU engines."""
        for gpu_id, engine in self.pipelined_engine.gpu_engines.items():
            try:
                engine.register_model(model_name, model)
            except Exception as e:
                print(f"Failed to register model on GPU {gpu_id}: {e}")
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics from all parallel components."""
        return {
            'orchestrator_stats': self.orchestrator_stats,
            'pipeline_stats': self.pipelined_engine.get_pipeline_stats(),
            'load_balancer_stats': {
                'gpu_loads': self.load_balancer.gpu_loads,
                'gpu_performance': self.load_balancer.gpu_performance,
                'should_rebalance': self.load_balancer.should_rebalance()
            },
            'config': {
                'num_gpus': self.config.num_gpus,
                'num_cpu_workers': self.config.num_cpu_workers,
                'batch_size_per_gpu': self.config.batch_size_per_gpu,
                'pipeline_depth': self.config.pipeline_depth
            }
        }
    
    def _update_parallelization_stats(self, parallelization_factor: float) -> None:
        """Update parallelization statistics."""
        current_avg = self.orchestrator_stats['average_parallelization_factor']
        total_requests = self.orchestrator_stats['total_requests']
        
        self.orchestrator_stats['average_parallelization_factor'] = (
            (current_avg * (total_requests - 1) + parallelization_factor) / total_requests
        )
