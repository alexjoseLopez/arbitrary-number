"""
Core Inference Engine
====================

Main inference engine for exact symbolic computation with GPU acceleration.
"""

import torch
import numpy as np
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading

from ...arbitrary_numbers.core.rational_list import RationalListNumber, FractionTerm
from ...arbitrary_numbers.core.equation_nodes import EquationNode
from ...arbitrary_numbers.core.evaluator import EquationEvaluator
from ...arbitrary_numbers.gpu.cuda_kernels import GPUEvaluator
from ...arbitrary_numbers.ml.pytorch_layers import ExplainableInferenceModel


@dataclass
class InferenceRequest:
    """Single inference request with input data and metadata."""
    request_id: str
    input_data: Union[torch.Tensor, np.ndarray, List[float]]
    model_name: str
    require_explanation: bool = False
    precision_mode: str = "exact"
    timeout_ms: int = 5000


@dataclass
class InferenceResult:
    """Result of inference computation with symbolic traceability."""
    request_id: str
    output: torch.Tensor
    symbolic_expressions: Optional[Dict[str, List[str]]] = None
    computation_trace: Optional[List[Dict[str, Any]]] = None
    execution_time_ms: float = 0.0
    precision_loss: float = 0.0
    gpu_utilization: float = 0.0
    memory_usage_mb: float = 0.0
    exact_computation: bool = True


class InferenceEngine:
    """
    High-performance inference engine for exact symbolic computation.
    Optimized for consumer 32GB Nvidia GPUs.
    """
    
    def __init__(self, 
                 max_batch_size: int = 64,
                 gpu_memory_limit_gb: float = 24.0,
                 enable_symbolic_optimization: bool = True,
                 worker_threads: int = 4):
        
        self.max_batch_size = max_batch_size
        self.gpu_memory_limit_gb = gpu_memory_limit_gb
        self.enable_symbolic_optimization = enable_symbolic_optimization
        self.worker_threads = worker_threads
        
        self.models = {}
        self.evaluator = EquationEvaluator()
        self.gpu_evaluator = GPUEvaluator(max_memory_gb=gpu_memory_limit_gb)
        
        self.executor = ThreadPoolExecutor(max_workers=worker_threads)
        self.request_queue = []
        self.queue_lock = threading.RLock()
        
        self.performance_stats = {
            'total_requests': 0,
            'successful_inferences': 0,
            'failed_inferences': 0,
            'total_execution_time': 0.0,
            'gpu_time': 0.0,
            'cpu_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        self.is_running = False
        self.processing_thread = None
    
    def register_model(self, 
                      model_name: str, 
                      model: ExplainableInferenceModel,
                      preprocessing_fn: Optional[callable] = None,
                      postprocessing_fn: Optional[callable] = None) -> None:
        """Register a model for inference."""
        
        model_info = {
            'model': model,
            'preprocessing_fn': preprocessing_fn or self._default_preprocessing,
            'postprocessing_fn': postprocessing_fn or self._default_postprocessing,
            'input_shape': None,
            'output_shape': None,
            'symbolic_weights_count': 0,
            'registration_time': time.time()
        }
        
        complexity = model.get_model_complexity()
        model_info['symbolic_weights_count'] = complexity['total_symbolic_weights']
        
        self.models[model_name] = model_info
        print(f"Registered model '{model_name}' with {complexity['total_symbolic_weights']} symbolic weights")
    
    def start_engine(self) -> None:
        """Start the inference engine processing loop."""
        if self.is_running:
            return
        
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        print("Inference engine started")
    
    def stop_engine(self) -> None:
        """Stop the inference engine."""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)
        self.executor.shutdown(wait=True)
        print("Inference engine stopped")
    
    def infer(self, request: InferenceRequest) -> InferenceResult:
        """Perform single inference request."""
        start_time = time.time()
        
        try:
            if request.model_name not in self.models:
                raise ValueError(f"Model '{request.model_name}' not registered")
            
            model_info = self.models[request.model_name]
            model = model_info['model']
            
            input_tensor = self._prepare_input(request.input_data, model_info)
            
            with torch.no_grad():
                if request.precision_mode == "exact":
                    output = self._exact_inference(model, input_tensor)
                else:
                    output = model(input_tensor)
            
            result = InferenceResult(
                request_id=request.request_id,
                output=output,
                execution_time_ms=(time.time() - start_time) * 1000,
                exact_computation=(request.precision_mode == "exact")
            )
            
            if request.require_explanation:
                explanation = model.explain_prediction(input_tensor.squeeze(0))
                result.symbolic_expressions = explanation['symbolic_expressions']
                result.computation_trace = explanation['computation_trace']
                result.precision_loss = explanation['precision_loss']
            
            self.performance_stats['successful_inferences'] += 1
            return result
            
        except Exception as e:
            self.performance_stats['failed_inferences'] += 1
            return InferenceResult(
                request_id=request.request_id,
                output=torch.zeros(1),
                execution_time_ms=(time.time() - start_time) * 1000,
                exact_computation=False
            )
        finally:
            self.performance_stats['total_requests'] += 1
            self.performance_stats['total_execution_time'] += (time.time() - start_time) * 1000
    
    def infer_batch(self, requests: List[InferenceRequest]) -> List[InferenceResult]:
        """Perform batch inference with GPU optimization."""
        if len(requests) > self.max_batch_size:
            batches = [requests[i:i + self.max_batch_size] 
                      for i in range(0, len(requests), self.max_batch_size)]
            results = []
            for batch in batches:
                results.extend(self.infer_batch(batch))
            return results
        
        start_time = time.time()
        results = []
        
        try:
            grouped_requests = self._group_requests_by_model(requests)
            
            for model_name, model_requests in grouped_requests.items():
                if model_name not in self.models:
                    for req in model_requests:
                        results.append(InferenceResult(
                            request_id=req.request_id,
                            output=torch.zeros(1),
                            exact_computation=False
                        ))
                    continue
                
                model_info = self.models[model_name]
                model = model_info['model']
                
                batch_inputs = []
                for req in model_requests:
                    input_tensor = self._prepare_input(req.input_data, model_info)
                    batch_inputs.append(input_tensor)
                
                batch_tensor = torch.cat(batch_inputs, dim=0)
                
                with torch.no_grad():
                    batch_outputs = self._batch_exact_inference(model, batch_tensor)
                
                for i, req in enumerate(model_requests):
                    result = InferenceResult(
                        request_id=req.request_id,
                        output=batch_outputs[i:i+1],
                        execution_time_ms=(time.time() - start_time) * 1000,
                        exact_computation=True
                    )
                    
                    if req.require_explanation:
                        explanation = model.explain_prediction(batch_inputs[i].squeeze(0))
                        result.symbolic_expressions = explanation['symbolic_expressions']
                        result.computation_trace = explanation['computation_trace']
                        result.precision_loss = explanation['precision_loss']
                    
                    results.append(result)
            
            self.performance_stats['successful_inferences'] += len(requests)
            
        except Exception as e:
            self.performance_stats['failed_inferences'] += len(requests)
            for req in requests:
                results.append(InferenceResult(
                    request_id=req.request_id,
                    output=torch.zeros(1),
                    exact_computation=False
                ))
        finally:
            self.performance_stats['total_requests'] += len(requests)
            self.performance_stats['total_execution_time'] += (time.time() - start_time) * 1000
        
        return results
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get detailed information about a registered model."""
        if model_name not in self.models:
            return {}
        
        model_info = self.models[model_name]
        model = model_info['model']
        complexity = model.get_model_complexity()
        
        return {
            'model_name': model_name,
            'registration_time': model_info['registration_time'],
            'symbolic_weights_count': model_info['symbolic_weights_count'],
            'model_complexity': complexity,
            'input_shape': model_info['input_shape'],
            'output_shape': model_info['output_shape']
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        gpu_stats = self.gpu_evaluator.get_performance_stats()
        evaluator_stats = self.evaluator.get_stats()
        
        combined_stats = {
            **self.performance_stats,
            'gpu_stats': gpu_stats,
            'evaluator_stats': evaluator_stats,
            'average_execution_time_ms': (
                self.performance_stats['total_execution_time'] / 
                max(1, self.performance_stats['total_requests'])
            ),
            'success_rate': (
                self.performance_stats['successful_inferences'] / 
                max(1, self.performance_stats['total_requests']) * 100
            ),
            'registered_models': len(self.models),
            'engine_status': 'running' if self.is_running else 'stopped'
        }
        
        return combined_stats
    
    def _exact_inference(self, model: ExplainableInferenceModel, input_tensor: torch.Tensor) -> torch.Tensor:
        """Perform exact symbolic inference."""
        gpu_start = time.time()
        
        try:
            output = model(input_tensor)
            self.performance_stats['gpu_time'] += (time.time() - gpu_start) * 1000
            return output
        except Exception:
            self.performance_stats['cpu_time'] += (time.time() - gpu_start) * 1000
            return model(input_tensor)
    
    def _batch_exact_inference(self, model: ExplainableInferenceModel, batch_tensor: torch.Tensor) -> torch.Tensor:
        """Perform batch exact symbolic inference with GPU optimization."""
        return model(batch_tensor)
    
    def _prepare_input(self, input_data: Union[torch.Tensor, np.ndarray, List[float]], 
                      model_info: Dict[str, Any]) -> torch.Tensor:
        """Prepare input data for inference."""
        if isinstance(input_data, torch.Tensor):
            tensor = input_data
        elif isinstance(input_data, np.ndarray):
            tensor = torch.from_numpy(input_data).float()
        else:
            tensor = torch.tensor(input_data, dtype=torch.float32)
        
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        
        tensor = model_info['preprocessing_fn'](tensor)
        return tensor
    
    def _group_requests_by_model(self, requests: List[InferenceRequest]) -> Dict[str, List[InferenceRequest]]:
        """Group requests by model name for batch processing."""
        grouped = {}
        for request in requests:
            if request.model_name not in grouped:
                grouped[request.model_name] = []
            grouped[request.model_name].append(request)
        return grouped
    
    def _default_preprocessing(self, tensor: torch.Tensor) -> torch.Tensor:
        """Default preprocessing function."""
        return tensor
    
    def _default_postprocessing(self, tensor: torch.Tensor) -> torch.Tensor:
        """Default postprocessing function."""
        return tensor
    
    def _processing_loop(self) -> None:
        """Main processing loop for queued requests."""
        while self.is_running:
            try:
                with self.queue_lock:
                    if not self.request_queue:
                        time.sleep(0.001)
                        continue
                    
                    batch = self.request_queue[:self.max_batch_size]
                    self.request_queue = self.request_queue[self.max_batch_size:]
                
                if batch:
                    self.infer_batch(batch)
                    
            except Exception as e:
                print(f"Error in processing loop: {e}")
                time.sleep(0.1)
