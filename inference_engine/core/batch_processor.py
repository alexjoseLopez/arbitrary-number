"""
Batch Processor
===============

High-performance batch processing system for symbolic inference with GPU optimization.
"""

import torch
import numpy as np
import time
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue, Empty
import asyncio

from .engine import InferenceRequest, InferenceResult
from ...arbitrary_numbers.gpu.cuda_kernels import GPUEvaluator
from ...arbitrary_numbers.ml.pytorch_layers import ExplainableInferenceModel


@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    max_batch_size: int = 64
    batch_timeout_ms: int = 100
    max_queue_size: int = 1000
    enable_dynamic_batching: bool = True
    gpu_memory_threshold: float = 0.8
    priority_levels: int = 3


class BatchMetrics:
    """Metrics tracking for batch processing."""
    
    def __init__(self):
        self.total_batches = 0
        self.total_requests = 0
        self.successful_batches = 0
        self.failed_batches = 0
        self.average_batch_size = 0.0
        self.average_processing_time = 0.0
        self.gpu_utilization = 0.0
        self.queue_utilization = 0.0
        self.throughput_rps = 0.0
        self.latency_p50 = 0.0
        self.latency_p95 = 0.0
        self.latency_p99 = 0.0
        self.processing_times = []
        self.batch_sizes = []
        self.start_time = time.time()
    
    def update_batch_metrics(self, batch_size: int, processing_time: float, success: bool):
        """Update metrics after batch processing."""
        self.total_batches += 1
        self.total_requests += batch_size
        
        if success:
            self.successful_batches += 1
        else:
            self.failed_batches += 1
        
        self.batch_sizes.append(batch_size)
        self.processing_times.append(processing_time)
        
        if len(self.batch_sizes) > 1000:
            self.batch_sizes = self.batch_sizes[-500:]
            self.processing_times = self.processing_times[-500:]
        
        self.average_batch_size = sum(self.batch_sizes) / len(self.batch_sizes)
        self.average_processing_time = sum(self.processing_times) / len(self.processing_times)
        
        elapsed_time = time.time() - self.start_time
        self.throughput_rps = self.total_requests / max(elapsed_time, 1.0)
        
        if len(self.processing_times) >= 10:
            sorted_times = sorted(self.processing_times)
            n = len(sorted_times)
            self.latency_p50 = sorted_times[int(n * 0.5)]
            self.latency_p95 = sorted_times[int(n * 0.95)]
            self.latency_p99 = sorted_times[int(n * 0.99)]
    
    def get_metrics_dict(self) -> Dict[str, Any]:
        """Get metrics as dictionary."""
        return {
            'total_batches': self.total_batches,
            'total_requests': self.total_requests,
            'successful_batches': self.successful_batches,
            'failed_batches': self.failed_batches,
            'success_rate': (self.successful_batches / max(1, self.total_batches)) * 100,
            'average_batch_size': self.average_batch_size,
            'average_processing_time_ms': self.average_processing_time * 1000,
            'throughput_rps': self.throughput_rps,
            'latency_p50_ms': self.latency_p50 * 1000,
            'latency_p95_ms': self.latency_p95 * 1000,
            'latency_p99_ms': self.latency_p99 * 1000,
            'gpu_utilization': self.gpu_utilization,
            'queue_utilization': self.queue_utilization
        }


class PriorityQueue:
    """Priority queue for inference requests."""
    
    def __init__(self, max_size: int = 1000, priority_levels: int = 3):
        self.max_size = max_size
        self.priority_levels = priority_levels
        self.queues = [Queue() for _ in range(priority_levels)]
        self.total_size = 0
        self.lock = threading.RLock()
    
    def put(self, item: InferenceRequest, priority: int = 1) -> bool:
        """Add item to priority queue."""
        with self.lock:
            if self.total_size >= self.max_size:
                return False
            
            priority = max(0, min(priority, self.priority_levels - 1))
            self.queues[priority].put(item)
            self.total_size += 1
            return True
    
    def get_batch(self, max_batch_size: int, timeout: float = 0.1) -> List[InferenceRequest]:
        """Get batch of items from priority queue."""
        batch = []
        start_time = time.time()
        
        with self.lock:
            for priority in range(self.priority_levels):
                while len(batch) < max_batch_size and not self.queues[priority].empty():
                    try:
                        item = self.queues[priority].get_nowait()
                        batch.append(item)
                        self.total_size -= 1
                    except Empty:
                        break
                
                if len(batch) >= max_batch_size:
                    break
            
            if not batch and timeout > 0:
                remaining_time = timeout - (time.time() - start_time)
                if remaining_time > 0:
                    try:
                        for priority in range(self.priority_levels):
                            if not self.queues[priority].empty():
                                item = self.queues[priority].get(timeout=remaining_time)
                                batch.append(item)
                                self.total_size -= 1
                                break
                    except Empty:
                        pass
        
        return batch
    
    def size(self) -> int:
        """Get total queue size."""
        return self.total_size
    
    def is_full(self) -> bool:
        """Check if queue is full."""
        return self.total_size >= self.max_size


class BatchProcessor:
    """
    High-performance batch processor for symbolic inference.
    Optimized for consumer 32GB Nvidia GPUs.
    """
    
    def __init__(self, 
                 config: BatchConfig = None,
                 gpu_evaluator: Optional[GPUEvaluator] = None):
        
        self.config = config or BatchConfig()
        self.gpu_evaluator = gpu_evaluator or GPUEvaluator()
        
        self.request_queue = PriorityQueue(
            max_size=self.config.max_queue_size,
            priority_levels=self.config.priority_levels
        )
        
        self.models = {}
        self.metrics = BatchMetrics()
        
        self.is_running = False
        self.processing_threads = []
        self.worker_count = 2
        
        self.batch_callbacks = []
        self.preprocessing_pipeline = []
        self.postprocessing_pipeline = []
        
        self.adaptive_batch_size = self.config.max_batch_size
        self.gpu_memory_monitor = GPUMemoryMonitor()
    
    def register_model(self, model_name: str, model: ExplainableInferenceModel) -> None:
        """Register a model for batch processing."""
        self.models[model_name] = {
            'model': model,
            'complexity': model.get_model_complexity(),
            'optimal_batch_size': self._calculate_optimal_batch_size(model),
            'registration_time': time.time()
        }
        print(f"Registered model '{model_name}' for batch processing")
    
    def add_preprocessing_step(self, step_fn: Callable) -> None:
        """Add preprocessing step to pipeline."""
        self.preprocessing_pipeline.append(step_fn)
    
    def add_postprocessing_step(self, step_fn: Callable) -> None:
        """Add postprocessing step to pipeline."""
        self.postprocessing_pipeline.append(step_fn)
    
    def add_batch_callback(self, callback_fn: Callable) -> None:
        """Add callback function for batch completion."""
        self.batch_callbacks.append(callback_fn)
    
    def start_processing(self) -> None:
        """Start batch processing threads."""
        if self.is_running:
            return
        
        self.is_running = True
        
        for i in range(self.worker_count):
            thread = threading.Thread(
                target=self._processing_worker,
                args=(i,),
                daemon=True
            )
            thread.start()
            self.processing_threads.append(thread)
        
        print(f"Started {self.worker_count} batch processing workers")
    
    def stop_processing(self) -> None:
        """Stop batch processing."""
        self.is_running = False
        
        for thread in self.processing_threads:
            thread.join(timeout=5.0)
        
        self.processing_threads.clear()
        print("Stopped batch processing")
    
    def submit_request(self, request: InferenceRequest, priority: int = 1) -> bool:
        """Submit inference request for batch processing."""
        if not self.is_running:
            return False
        
        success = self.request_queue.put(request, priority)
        
        if not success:
            print(f"Request queue full, dropping request {request.request_id}")
        
        return success
    
    def process_batch_sync(self, requests: List[InferenceRequest]) -> List[InferenceResult]:
        """Process batch synchronously."""
        if not requests:
            return []
        
        start_time = time.time()
        
        try:
            preprocessed_requests = self._apply_preprocessing(requests)
            
            grouped_requests = self._group_requests_by_model(preprocessed_requests)
            
            all_results = []
            
            for model_name, model_requests in grouped_requests.items():
                if model_name not in self.models:
                    for req in model_requests:
                        all_results.append(InferenceResult(
                            request_id=req.request_id,
                            output=torch.zeros(1),
                            exact_computation=False
                        ))
                    continue
                
                model_info = self.models[model_name]
                model = model_info['model']
                
                batch_results = self._process_model_batch(model, model_requests)
                all_results.extend(batch_results)
            
            final_results = self._apply_postprocessing(all_results)
            
            processing_time = time.time() - start_time
            self.metrics.update_batch_metrics(len(requests), processing_time, True)
            
            for callback in self.batch_callbacks:
                try:
                    callback(requests, final_results, processing_time)
                except Exception as e:
                    print(f"Batch callback error: {e}")
            
            return final_results
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.metrics.update_batch_metrics(len(requests), processing_time, False)
            
            error_results = []
            for req in requests:
                error_results.append(InferenceResult(
                    request_id=req.request_id,
                    output=torch.zeros(1),
                    exact_computation=False
                ))
            
            return error_results
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive batch processing metrics."""
        base_metrics = self.metrics.get_metrics_dict()
        
        queue_metrics = {
            'queue_size': self.request_queue.size(),
            'queue_utilization': (self.request_queue.size() / self.config.max_queue_size) * 100,
            'adaptive_batch_size': self.adaptive_batch_size
        }
        
        gpu_metrics = self.gpu_evaluator.get_performance_stats()
        memory_metrics = self.gpu_memory_monitor.get_memory_stats()
        
        return {
            **base_metrics,
            **queue_metrics,
            'gpu_metrics': gpu_metrics,
            'memory_metrics': memory_metrics,
            'registered_models': len(self.models),
            'processing_status': 'running' if self.is_running else 'stopped'
        }
    
    def _processing_worker(self, worker_id: int) -> None:
        """Main processing loop for worker thread."""
        print(f"Batch processing worker {worker_id} started")
        
        while self.is_running:
            try:
                batch = self.request_queue.get_batch(
                    max_batch_size=self.adaptive_batch_size,
                    timeout=self.config.batch_timeout_ms / 1000.0
                )
                
                if batch:
                    self.process_batch_sync(batch)
                else:
                    time.sleep(0.001)
                
                if self.config.enable_dynamic_batching:
                    self._adjust_batch_size()
                    
            except Exception as e:
                print(f"Worker {worker_id} error: {e}")
                time.sleep(0.1)
        
        print(f"Batch processing worker {worker_id} stopped")
    
    def _process_model_batch(self, model: ExplainableInferenceModel, requests: List[InferenceRequest]) -> List[InferenceResult]:
        """Process batch for specific model."""
        batch_inputs = []
        
        for req in requests:
            input_tensor = self._prepare_input(req.input_data)
            batch_inputs.append(input_tensor)
        
        batch_tensor = torch.cat(batch_inputs, dim=0)
        
        start_time = time.time()
        
        with torch.no_grad():
            if self.gpu_evaluator.gpu_available:
                batch_outputs = self._gpu_batch_inference(model, batch_tensor)
            else:
                batch_outputs = model(batch_tensor)
        
        inference_time = time.time() - start_time
        
        results = []
        for i, req in enumerate(requests):
            result = InferenceResult(
                request_id=req.request_id,
                output=batch_outputs[i:i+1],
                execution_time_ms=inference_time * 1000,
                exact_computation=True,
                gpu_utilization=self.gpu_memory_monitor.get_utilization()
            )
            
            if req.require_explanation:
                explanation = model.explain_prediction(batch_inputs[i].squeeze(0))
                result.symbolic_expressions = explanation['symbolic_expressions']
                result.computation_trace = explanation['computation_trace']
                result.precision_loss = explanation['precision_loss']
            
            results.append(result)
        
        return results
    
    def _gpu_batch_inference(self, model: ExplainableInferenceModel, batch_tensor: torch.Tensor) -> torch.Tensor:
        """Perform GPU-optimized batch inference."""
        try:
            return model(batch_tensor)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                smaller_batch_size = batch_tensor.size(0) // 2
                
                if smaller_batch_size > 0:
                    first_half = model(batch_tensor[:smaller_batch_size])
                    second_half = model(batch_tensor[smaller_batch_size:])
                    return torch.cat([first_half, second_half], dim=0)
                else:
                    raise e
            else:
                raise e
    
    def _prepare_input(self, input_data: Union[torch.Tensor, np.ndarray, List[float]]) -> torch.Tensor:
        """Prepare input data for batch processing."""
        if isinstance(input_data, torch.Tensor):
            tensor = input_data
        elif isinstance(input_data, np.ndarray):
            tensor = torch.from_numpy(input_data).float()
        else:
            tensor = torch.tensor(input_data, dtype=torch.float32)
        
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        
        return tensor
    
    def _group_requests_by_model(self, requests: List[InferenceRequest]) -> Dict[str, List[InferenceRequest]]:
        """Group requests by model name."""
        grouped = {}
        for request in requests:
            if request.model_name not in grouped:
                grouped[request.model_name] = []
            grouped[request.model_name].append(request)
        return grouped
    
    def _apply_preprocessing(self, requests: List[InferenceRequest]) -> List[InferenceRequest]:
        """Apply preprocessing pipeline to requests."""
        processed_requests = requests
        
        for step_fn in self.preprocessing_pipeline:
            try:
                processed_requests = step_fn(processed_requests)
            except Exception as e:
                print(f"Preprocessing step error: {e}")
        
        return processed_requests
    
    def _apply_postprocessing(self, results: List[InferenceResult]) -> List[InferenceResult]:
        """Apply postprocessing pipeline to results."""
        processed_results = results
        
        for step_fn in self.postprocessing_pipeline:
            try:
                processed_results = step_fn(processed_results)
            except Exception as e:
                print(f"Postprocessing step error: {e}")
        
        return processed_results
    
    def _calculate_optimal_batch_size(self, model: ExplainableInferenceModel) -> int:
        """Calculate optimal batch size for model."""
        complexity = model.get_model_complexity()
        
        base_batch_size = self.config.max_batch_size
        
        parameter_factor = min(1.0, 10000 / max(1, complexity['total_parameters']))
        
        optimal_size = int(base_batch_size * parameter_factor)
        
        return max(1, min(optimal_size, self.config.max_batch_size))
    
    def _adjust_batch_size(self) -> None:
        """Dynamically adjust batch size based on performance."""
        if len(self.metrics.processing_times) < 10:
            return
        
        recent_times = self.metrics.processing_times[-10:]
        avg_time = sum(recent_times) / len(recent_times)
        
        memory_usage = self.gpu_memory_monitor.get_utilization()
        
        if memory_usage > self.config.gpu_memory_threshold:
            self.adaptive_batch_size = max(1, int(self.adaptive_batch_size * 0.8))
        elif avg_time < 0.05 and memory_usage < 0.6:
            self.adaptive_batch_size = min(
                self.config.max_batch_size,
                int(self.adaptive_batch_size * 1.2)
            )


class GPUMemoryMonitor:
    """Monitor GPU memory usage for batch optimization."""
    
    def __init__(self):
        self.memory_history = []
        self.utilization_history = []
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current GPU memory statistics."""
        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / (1024**3)
                reserved = torch.cuda.memory_reserved() / (1024**3)
                total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                
                utilization = allocated / total
                
                self.memory_history.append(allocated)
                self.utilization_history.append(utilization)
                
                if len(self.memory_history) > 100:
                    self.memory_history = self.memory_history[-50:]
                    self.utilization_history = self.utilization_history[-50:]
                
                return {
                    'allocated_gb': allocated,
                    'reserved_gb': reserved,
                    'total_gb': total,
                    'utilization': utilization,
                    'average_utilization': sum(self.utilization_history) / len(self.utilization_history)
                }
            else:
                return {'gpu_available': False}
        except:
            return {'gpu_available': False}
    
    def get_utilization(self) -> float:
        """Get current GPU utilization."""
        stats = self.get_memory_stats()
        return stats.get('utilization', 0.0)
