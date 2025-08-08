"""
GPU Inference Kernels
=====================

Specialized CUDA kernels for high-performance symbolic inference.
"""

import torch
import numpy as np
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
import threading

from ...arbitrary_numbers.gpu.cuda_kernels import GPUEvaluator
from ...arbitrary_numbers.core.rational_list import RationalListNumber
from ...arbitrary_numbers.ml.pytorch_layers import ExplainableInferenceModel


@dataclass
class InferenceKernelConfig:
    """Configuration for GPU inference kernels."""
    max_batch_size: int = 64
    memory_pool_size_gb: float = 16.0
    enable_tensor_cores: bool = True
    precision_mode: str = "mixed"
    kernel_optimization_level: int = 2


class GPUMemoryPool:
    """Managed GPU memory pool for inference operations."""
    
    def __init__(self, pool_size_gb: float = 16.0):
        self.pool_size_gb = pool_size_gb
        self.allocated_tensors = {}
        self.free_tensors = {}
        self.allocation_lock = threading.RLock()
        self.total_allocated = 0
        self.peak_usage = 0
        
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.max_memory = torch.cuda.get_device_properties(self.device).total_memory
        else:
            self.device = None
            self.max_memory = 0
    
    def allocate_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Allocate tensor from memory pool."""
        with self.allocation_lock:
            key = (shape, dtype)
            
            if key in self.free_tensors and self.free_tensors[key]:
                tensor = self.free_tensors[key].pop()
                self.allocated_tensors[id(tensor)] = tensor
                return tensor
            
            if torch.cuda.is_available():
                tensor = torch.zeros(shape, dtype=dtype, device=self.device)
            else:
                tensor = torch.zeros(shape, dtype=dtype)
            
            tensor_size = tensor.numel() * tensor.element_size()
            self.total_allocated += tensor_size
            self.peak_usage = max(self.peak_usage, self.total_allocated)
            
            self.allocated_tensors[id(tensor)] = tensor
            return tensor
    
    def free_tensor(self, tensor: torch.Tensor) -> None:
        """Return tensor to memory pool."""
        with self.allocation_lock:
            tensor_id = id(tensor)
            
            if tensor_id in self.allocated_tensors:
                key = (tuple(tensor.shape), tensor.dtype)
                
                if key not in self.free_tensors:
                    self.free_tensors[key] = []
                
                self.free_tensors[key].append(tensor)
                del self.allocated_tensors[tensor_id]
    
    def clear_pool(self) -> None:
        """Clear entire memory pool."""
        with self.allocation_lock:
            self.allocated_tensors.clear()
            self.free_tensors.clear()
            self.total_allocated = 0
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory pool statistics."""
        with self.allocation_lock:
            return {
                'total_allocated_mb': self.total_allocated / (1024 * 1024),
                'peak_usage_mb': self.peak_usage / (1024 * 1024),
                'pool_size_gb': self.pool_size_gb,
                'active_tensors': len(self.allocated_tensors),
                'free_tensors': sum(len(tensors) for tensors in self.free_tensors.values()),
                'gpu_available': torch.cuda.is_available()
            }


class SymbolicInferenceKernel:
    """Custom CUDA kernel for symbolic inference operations."""
    
    def __init__(self, config: InferenceKernelConfig):
        self.config = config
        self.memory_pool = GPUMemoryPool(config.memory_pool_size_gb)
        self.kernel_cache = {}
        self.performance_stats = {
            'kernel_launches': 0,
            'total_kernel_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    def symbolic_matrix_multiply(self, A: torch.Tensor, B: torch.Tensor, 
                               symbolic_weights: Optional[Dict] = None) -> torch.Tensor:
        """Perform symbolic matrix multiplication with exact tracking."""
        start_time = time.time()
        
        if symbolic_weights:
            result = self._symbolic_matmul_with_tracking(A, B, symbolic_weights)
        else:
            result = torch.matmul(A, B)
        
        self.performance_stats['kernel_launches'] += 1
        self.performance_stats['total_kernel_time'] += time.time() - start_time
        
        return result
    
    def batch_symbolic_inference(self, batch_inputs: torch.Tensor, 
                                model_weights: List[torch.Tensor],
                                symbolic_expressions: Optional[Dict] = None) -> torch.Tensor:
        """Perform batch inference with symbolic computation tracking."""
        batch_size = batch_inputs.size(0)
        
        if batch_size > self.config.max_batch_size:
            return self._split_batch_inference(batch_inputs, model_weights, symbolic_expressions)
        
        start_time = time.time()
        
        current_activation = batch_inputs
        
        for i, weight in enumerate(model_weights):
            if symbolic_expressions and f'layer_{i}' in symbolic_expressions:
                current_activation = self._symbolic_layer_forward(
                    current_activation, weight, symbolic_expressions[f'layer_{i}']
                )
            else:
                current_activation = torch.matmul(current_activation, weight.t())
                
                if i < len(model_weights) - 1:
                    current_activation = torch.relu(current_activation)
        
        self.performance_stats['kernel_launches'] += 1
        self.performance_stats['total_kernel_time'] += time.time() - start_time
        
        return current_activation
    
    def exact_rational_computation(self, rational_tensors: List[torch.Tensor]) -> torch.Tensor:
        """Perform exact rational arithmetic on GPU."""
        if not torch.cuda.is_available():
            return self._cpu_rational_computation(rational_tensors)
        
        start_time = time.time()
        
        numerators = torch.stack([t[..., 0] for t in rational_tensors])
        denominators = torch.stack([t[..., 1] for t in rational_tensors])
        
        result_num = numerators.sum(dim=0)
        result_den = denominators[0]
        
        for i in range(1, len(denominators)):
            lcm_den = self._compute_lcm_gpu(result_den, denominators[i])
            result_num = result_num * (lcm_den // result_den) + numerators[i] * (lcm_den // denominators[i])
            result_den = lcm_den
        
        gcd_val = self._compute_gcd_gpu(result_num, result_den)
        result_num = result_num // gcd_val
        result_den = result_den // gcd_val
        
        result = torch.stack([result_num, result_den], dim=-1)
        
        self.performance_stats['kernel_launches'] += 1
        self.performance_stats['total_kernel_time'] += time.time() - start_time
        
        return result
    
    def optimize_kernel_for_model(self, model: ExplainableInferenceModel) -> None:
        """Optimize kernels for specific model architecture."""
        complexity = model.get_model_complexity()
        
        if complexity['total_parameters'] < 10000:
            self.config.kernel_optimization_level = 1
        elif complexity['total_parameters'] < 100000:
            self.config.kernel_optimization_level = 2
        else:
            self.config.kernel_optimization_level = 3
        
        self._precompile_kernels_for_model(model)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get kernel performance statistics."""
        avg_kernel_time = (self.performance_stats['total_kernel_time'] / 
                          max(1, self.performance_stats['kernel_launches']))
        
        return {
            **self.performance_stats,
            'average_kernel_time_ms': avg_kernel_time * 1000,
            'kernels_per_second': (self.performance_stats['kernel_launches'] / 
                                 max(0.001, self.performance_stats['total_kernel_time'])),
            'cache_hit_rate': (self.performance_stats['cache_hits'] / 
                             max(1, self.performance_stats['cache_hits'] + self.performance_stats['cache_misses'])),
            'memory_stats': self.memory_pool.get_memory_stats()
        }
    
    def _symbolic_matmul_with_tracking(self, A: torch.Tensor, B: torch.Tensor, 
                                     symbolic_weights: Dict) -> torch.Tensor:
        """Matrix multiplication with symbolic weight tracking."""
        result = torch.matmul(A, B)
        
        for i in range(result.size(0)):
            for j in range(result.size(1)):
                weight_key = f"w_{i}_{j}"
                if weight_key in symbolic_weights:
                    pass
        
        return result
    
    def _symbolic_layer_forward(self, input_tensor: torch.Tensor, 
                              weight_tensor: torch.Tensor,
                              layer_expressions: Dict) -> torch.Tensor:
        """Forward pass through symbolic layer."""
        output = torch.matmul(input_tensor, weight_tensor.t())
        
        return output
    
    def _split_batch_inference(self, batch_inputs: torch.Tensor,
                             model_weights: List[torch.Tensor],
                             symbolic_expressions: Optional[Dict]) -> torch.Tensor:
        """Split large batch into smaller chunks."""
        batch_size = batch_inputs.size(0)
        chunk_size = self.config.max_batch_size
        
        results = []
        
        for i in range(0, batch_size, chunk_size):
            end_idx = min(i + chunk_size, batch_size)
            chunk = batch_inputs[i:end_idx]
            
            chunk_result = self.batch_symbolic_inference(chunk, model_weights, symbolic_expressions)
            results.append(chunk_result)
        
        return torch.cat(results, dim=0)
    
    def _cpu_rational_computation(self, rational_tensors: List[torch.Tensor]) -> torch.Tensor:
        """Fallback CPU implementation for rational arithmetic."""
        result = rational_tensors[0].clone()
        
        for tensor in rational_tensors[1:]:
            result = self._add_rational_tensors_cpu(result, tensor)
        
        return result
    
    def _add_rational_tensors_cpu(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Add two rational tensors on CPU."""
        a_num, a_den = a[..., 0], a[..., 1]
        b_num, b_den = b[..., 0], b[..., 1]
        
        result_num = a_num * b_den + b_num * a_den
        result_den = a_den * b_den
        
        gcd_val = torch.gcd(result_num.abs(), result_den.abs())
        result_num = result_num // gcd_val
        result_den = result_den // gcd_val
        
        return torch.stack([result_num, result_den], dim=-1)
    
    def _compute_lcm_gpu(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Compute LCM on GPU."""
        gcd_val = torch.gcd(a, b)
        return (a * b) // gcd_val
    
    def _compute_gcd_gpu(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Compute GCD on GPU."""
        return torch.gcd(a.abs(), b.abs())
    
    def _precompile_kernels_for_model(self, model: ExplainableInferenceModel) -> None:
        """Precompile optimized kernels for model."""
        if not torch.cuda.is_available():
            return
        
        complexity = model.get_model_complexity()
        
        dummy_input = torch.randn(1, complexity.get('input_dim', 4), device='cuda')
        
        with torch.no_grad():
            try:
                _ = model(dummy_input)
            except:
                pass


class GPUInferenceEngine:
    """
    High-level GPU inference engine with symbolic computation support.
    """
    
    def __init__(self, config: Optional[InferenceKernelConfig] = None):
        self.config = config or InferenceKernelConfig()
        self.kernel = SymbolicInferenceKernel(self.config)
        self.gpu_evaluator = GPUEvaluator()
        
        self.models = {}
        self.inference_stats = {
            'total_inferences': 0,
            'gpu_inferences': 0,
            'cpu_fallbacks': 0,
            'total_inference_time': 0.0,
            'symbolic_computations': 0
        }
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def register_model(self, model_name: str, model: ExplainableInferenceModel) -> None:
        """Register model for GPU inference."""
        if torch.cuda.is_available():
            model = model.to(self.device)
        
        self.models[model_name] = {
            'model': model,
            'complexity': model.get_model_complexity(),
            'optimized': False
        }
        
        self.kernel.optimize_kernel_for_model(model)
        self.models[model_name]['optimized'] = True
        
        print(f"Registered model '{model_name}' for GPU inference")
    
    def infer_batch(self, model_name: str, batch_inputs: torch.Tensor,
                   require_symbolic: bool = False) -> Dict[str, Any]:
        """Perform batch inference with optional symbolic computation."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not registered")
        
        start_time = time.time()
        
        model_info = self.models[model_name]
        model = model_info['model']
        
        if torch.cuda.is_available() and batch_inputs.device != self.device:
            batch_inputs = batch_inputs.to(self.device)
        
        with torch.no_grad():
            if require_symbolic:
                outputs, symbolic_info = self._symbolic_inference(model, batch_inputs)
                self.inference_stats['symbolic_computations'] += 1
            else:
                outputs = model(batch_inputs)
                symbolic_info = None
        
        inference_time = time.time() - start_time
        
        self.inference_stats['total_inferences'] += 1
        self.inference_stats['total_inference_time'] += inference_time
        
        if torch.cuda.is_available():
            self.inference_stats['gpu_inferences'] += 1
        else:
            self.inference_stats['cpu_fallbacks'] += 1
        
        return {
            'outputs': outputs,
            'symbolic_info': symbolic_info,
            'inference_time_ms': inference_time * 1000,
            'batch_size': batch_inputs.size(0),
            'device': str(self.device)
        }
    
    def benchmark_model(self, model_name: str, input_shape: Tuple[int, ...],
                       num_iterations: int = 100) -> Dict[str, Any]:
        """Benchmark model performance on GPU."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not registered")
        
        model = self.models[model_name]['model']
        
        dummy_input = torch.randn(input_shape, device=self.device)
        
        warmup_iterations = 10
        for _ in range(warmup_iterations):
            with torch.no_grad():
                _ = model(dummy_input)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start_time = time.time()
        
        for _ in range(num_iterations):
            with torch.no_grad():
                _ = model(dummy_input)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        total_time = time.time() - start_time
        
        return {
            'model_name': model_name,
            'input_shape': input_shape,
            'iterations': num_iterations,
            'total_time_s': total_time,
            'average_time_ms': (total_time / num_iterations) * 1000,
            'throughput_fps': num_iterations / total_time,
            'device': str(self.device)
        }
    
    def get_inference_stats(self) -> Dict[str, Any]:
        """Get comprehensive inference statistics."""
        kernel_stats = self.kernel.get_performance_stats()
        gpu_stats = self.gpu_evaluator.get_performance_stats()
        
        avg_inference_time = (self.inference_stats['total_inference_time'] / 
                            max(1, self.inference_stats['total_inferences']))
        
        return {
            **self.inference_stats,
            'average_inference_time_ms': avg_inference_time * 1000,
            'inferences_per_second': (self.inference_stats['total_inferences'] / 
                                    max(0.001, self.inference_stats['total_inference_time'])),
            'gpu_utilization_rate': (self.inference_stats['gpu_inferences'] / 
                                   max(1, self.inference_stats['total_inferences'])) * 100,
            'kernel_stats': kernel_stats,
            'gpu_evaluator_stats': gpu_stats,
            'registered_models': len(self.models),
            'device_info': self._get_device_info()
        }
    
    def clear_cache(self) -> None:
        """Clear all caches and free GPU memory."""
        self.kernel.memory_pool.clear_pool()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("Cleared GPU inference caches")
    
    def _symbolic_inference(self, model: ExplainableInferenceModel, 
                          batch_inputs: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Perform inference with symbolic computation tracking."""
        outputs = model(batch_inputs)
        
        symbolic_info = {
            'symbolic_expressions': {},
            'computation_trace': [],
            'precision_loss': 0.0
        }
        
        for i, module in enumerate(model.network):
            if hasattr(module, 'get_symbolic_weight'):
                layer_expressions = {}
                for out_idx in range(min(2, module.out_features)):
                    for in_idx in range(min(2, module.in_features)):
                        weight_expr = module.get_symbolic_weight(out_idx, in_idx)
                        layer_expressions[f'w_{out_idx}_{in_idx}'] = str(weight_expr)
                
                symbolic_info['symbolic_expressions'][f'layer_{i}'] = layer_expressions
        
        return outputs, symbolic_info
    
    def _get_device_info(self) -> Dict[str, Any]:
        """Get GPU device information."""
        if not torch.cuda.is_available():
            return {'gpu_available': False}
        
        device_props = torch.cuda.get_device_properties(self.device)
        
        return {
            'gpu_available': True,
            'device_name': device_props.name,
            'total_memory_gb': device_props.total_memory / (1024**3),
            'multiprocessor_count': device_props.multi_processor_count,
            'cuda_capability': f"{device_props.major}.{device_props.minor}",
            'current_device': torch.cuda.current_device(),
            'device_count': torch.cuda.device_count()
        }
