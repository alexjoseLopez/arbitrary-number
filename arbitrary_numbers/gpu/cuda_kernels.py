"""
GPU Acceleration Layer
=====================

CUDA kernels and GPU-optimized evaluation for Arbitrary Numbers.
Designed for consumer-grade Nvidia 32GB GPUs (RTX 4090, RTX 6000 Ada).
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import threading
import time

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

from ..core.equation_nodes import EquationNode, ConstantNode, BinaryOpNode, UnaryOpNode
from ..core.rational_list import RationalListNumber, FractionTerm


class GPUMemoryManager:
    """
    Manages GPU memory allocation for 32GB consumer cards.
    Optimizes memory layout for coalesced access patterns.
    """
    
    def __init__(self, max_memory_gb: float = 24.0):
        self.max_memory_bytes = int(max_memory_gb * 1024**3)
        self.allocated_memory = 0
        self.memory_pools = {
            'equation_trees': [],
            'rational_terms': [],
            'result_cache': [],
            'temp_buffers': []
        }
        self.lock = threading.RLock()
    
    def allocate_buffer(self, size_bytes: int, pool_name: str = 'temp_buffers') -> Optional[cp.ndarray]:
        """Allocate GPU memory buffer with tracking."""
        if not CUPY_AVAILABLE:
            return None
        
        with self.lock:
            if self.allocated_memory + size_bytes > self.max_memory_bytes:
                self._cleanup_oldest_buffers()
            
            if self.allocated_memory + size_bytes > self.max_memory_bytes:
                return None
            
            try:
                buffer = cp.zeros(size_bytes // 4, dtype=cp.int32)  # 4 bytes per int32
                self.memory_pools[pool_name].append((buffer, time.time()))
                self.allocated_memory += size_bytes
                return buffer
            except cp.cuda.memory.OutOfMemoryError:
                return None
    
    def _cleanup_oldest_buffers(self) -> None:
        """Free oldest buffers to make room for new allocations."""
        for pool_name, pool in self.memory_pools.items():
            if len(pool) > 10:  # Keep at least 10 buffers per pool
                oldest_buffers = sorted(pool, key=lambda x: x[1])[:5]
                for buffer, _ in oldest_buffers:
                    self.allocated_memory -= buffer.nbytes
                    pool.remove((buffer, _))
                    del buffer
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory usage statistics."""
        with self.lock:
            return {
                'allocated_mb': self.allocated_memory / (1024**2),
                'max_memory_mb': self.max_memory_bytes / (1024**2),
                'utilization_percent': (self.allocated_memory / self.max_memory_bytes) * 100,
                'pool_sizes': {name: len(pool) for name, pool in self.memory_pools.items()}
            }


class GPUKernelManager:
    """
    Manages CUDA kernels for rational arithmetic and tree evaluation.
    """
    
    def __init__(self):
        self.kernels_compiled = False
        self.rational_sum_kernel = None
        self.rational_mul_kernel = None
        self.tree_eval_kernel = None
    
    def compile_kernels(self) -> bool:
        """Compile CUDA kernels for GPU execution."""
        if not CUPY_AVAILABLE:
            return False
        
        try:
            # Rational number sum kernel
            self.rational_sum_kernel = cp.RawKernel(r'''
            extern "C" __global__
            void rational_sum_kernel(
                const long long* numerators,
                const long long* denominators,
                long long* result_num,
                long long* result_den,
                int n_terms
            ) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx >= n_terms) return;
                
                // Each thread processes one term
                // Use shared memory for reduction
                __shared__ long long shared_num[256];
                __shared__ long long shared_den[256];
                
                int tid = threadIdx.x;
                shared_num[tid] = (idx < n_terms) ? numerators[idx] : 0;
                shared_den[tid] = (idx < n_terms) ? denominators[idx] : 1;
                
                __syncthreads();
                
                // Parallel reduction for sum of fractions
                for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                    if (tid < stride && idx + stride < n_terms) {
                        // Add fractions: a/b + c/d = (a*d + c*b) / (b*d)
                        long long a = shared_num[tid];
                        long long b = shared_den[tid];
                        long long c = shared_num[tid + stride];
                        long long d = shared_den[tid + stride];
                        
                        shared_num[tid] = a * d + c * b;
                        shared_den[tid] = b * d;
                    }
                    __syncthreads();
                }
                
                // Write result from thread 0
                if (tid == 0) {
                    result_num[blockIdx.x] = shared_num[0];
                    result_den[blockIdx.x] = shared_den[0];
                }
            }
            ''', 'rational_sum_kernel')
            
            # Rational number multiplication kernel
            self.rational_mul_kernel = cp.RawKernel(r'''
            extern "C" __global__
            void rational_mul_kernel(
                const long long* num_a, const long long* den_a,
                const long long* num_b, const long long* den_b,
                long long* result_num, long long* result_den,
                int n_terms
            ) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx >= n_terms) return;
                
                // Multiply fractions: (a/b) * (c/d) = (a*c) / (b*d)
                result_num[idx] = num_a[idx] * num_b[idx];
                result_den[idx] = den_a[idx] * den_b[idx];
            }
            ''', 'rational_mul_kernel')
            
            self.kernels_compiled = True
            return True
            
        except Exception as e:
            print(f"Failed to compile CUDA kernels: {e}")
            return False
    
    def sum_rationals_gpu(self, terms: List[FractionTerm]) -> FractionTerm:
        """Sum rational numbers on GPU using parallel reduction."""
        if not self.kernels_compiled or not terms:
            return FractionTerm(0, 1)
        
        n_terms = len(terms)
        
        # Prepare input arrays
        numerators = cp.array([term.numerator for term in terms], dtype=cp.int64)
        denominators = cp.array([term.denominator for term in terms], dtype=cp.int64)
        
        # Calculate grid and block dimensions
        block_size = 256
        grid_size = (n_terms + block_size - 1) // block_size
        
        # Allocate result arrays
        result_num = cp.zeros(grid_size, dtype=cp.int64)
        result_den = cp.zeros(grid_size, dtype=cp.int64)
        
        # Launch kernel
        self.rational_sum_kernel(
            (grid_size,), (block_size,),
            (numerators, denominators, result_num, result_den, n_terms)
        )
        
        # Reduce results from all blocks on CPU (for simplicity)
        result_num_cpu = cp.asnumpy(result_num)
        result_den_cpu = cp.asnumpy(result_den)
        
        # Final reduction on CPU
        final_num, final_den = 0, 1
        for i in range(grid_size):
            if result_den_cpu[i] != 0:
                # Add fractions
                final_num = final_num * result_den_cpu[i] + result_num_cpu[i] * final_den
                final_den = final_den * result_den_cpu[i]
        
        return FractionTerm(int(final_num), int(final_den))


class GPUEvaluator:
    """
    GPU-accelerated evaluator for Arbitrary Numbers.
    Optimized for consumer 32GB Nvidia cards.
    """
    
    def __init__(self, max_memory_gb: float = 24.0):
        self.memory_manager = GPUMemoryManager(max_memory_gb)
        self.kernel_manager = GPUKernelManager()
        self.gpu_available = CUPY_AVAILABLE
        self.performance_stats = {
            'gpu_evaluations': 0,
            'cpu_fallbacks': 0,
            'memory_errors': 0,
            'total_gpu_time': 0.0
        }
        
        if self.gpu_available:
            self.kernel_manager.compile_kernels()
    
    def evaluate_rational_list_gpu(self, rational_num: RationalListNumber) -> RationalListNumber:
        """
        Evaluate RationalListNumber on GPU with parallel reduction.
        Falls back to CPU if GPU is unavailable or memory is insufficient.
        """
        if not self.gpu_available or not rational_num.terms:
            self.performance_stats['cpu_fallbacks'] += 1
            return rational_num
        
        start_time = time.time()
        
        try:
            # Use GPU kernel for parallel sum
            result_term = self.kernel_manager.sum_rationals_gpu(rational_num.terms)
            simplified = RationalListNumber([result_term])
            
            self.performance_stats['gpu_evaluations'] += 1
            self.performance_stats['total_gpu_time'] += time.time() - start_time
            
            return simplified
            
        except Exception as e:
            print(f"GPU evaluation failed, falling back to CPU: {e}")
            self.performance_stats['cpu_fallbacks'] += 1
            self.performance_stats['memory_errors'] += 1
            return rational_num.simplify()
    
    def evaluate_batch_gpu(self, 
                          rational_numbers: List[RationalListNumber],
                          max_batch_size: int = 1000) -> List[RationalListNumber]:
        """
        Evaluate multiple RationalListNumbers in parallel on GPU.
        """
        if not self.gpu_available:
            return [num.simplify() for num in rational_numbers]
        
        results = []
        
        # Process in batches to manage memory
        for i in range(0, len(rational_numbers), max_batch_size):
            batch = rational_numbers[i:i + max_batch_size]
            batch_results = []
            
            for rational_num in batch:
                result = self.evaluate_rational_list_gpu(rational_num)
                batch_results.append(result)
            
            results.extend(batch_results)
        
        return results
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU device information and capabilities."""
        if not self.gpu_available:
            return {'gpu_available': False, 'reason': 'CuPy not installed'}
        
        try:
            device = cp.cuda.Device()
            meminfo = cp.cuda.MemoryInfo()
            
            return {
                'gpu_available': True,
                'device_name': device.name.decode('utf-8'),
                'compute_capability': f"{device.compute_capability[0]}.{device.compute_capability[1]}",
                'total_memory_gb': meminfo.total / (1024**3),
                'free_memory_gb': meminfo.free / (1024**3),
                'used_memory_gb': meminfo.used / (1024**3),
                'memory_utilization': (meminfo.used / meminfo.total) * 100,
                'multiprocessor_count': device.multiprocessor_count,
                'max_threads_per_block': device.max_threads_per_block,
                'kernels_compiled': self.kernel_manager.kernels_compiled
            }
        except Exception as e:
            return {'gpu_available': False, 'error': str(e)}
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get GPU evaluation performance statistics."""
        total_evaluations = self.performance_stats['gpu_evaluations'] + self.performance_stats['cpu_fallbacks']
        gpu_usage_percent = (self.performance_stats['gpu_evaluations'] / total_evaluations * 100) if total_evaluations > 0 else 0
        
        avg_gpu_time = (self.performance_stats['total_gpu_time'] / self.performance_stats['gpu_evaluations']) if self.performance_stats['gpu_evaluations'] > 0 else 0
        
        stats = {
            'total_evaluations': total_evaluations,
            'gpu_evaluations': self.performance_stats['gpu_evaluations'],
            'cpu_fallbacks': self.performance_stats['cpu_fallbacks'],
            'memory_errors': self.performance_stats['memory_errors'],
            'gpu_usage_percent': f"{gpu_usage_percent:.2f}%",
            'average_gpu_time_ms': f"{avg_gpu_time * 1000:.2f}",
            'total_gpu_time_seconds': f"{self.performance_stats['total_gpu_time']:.2f}"
        }
        
        stats.update(self.memory_manager.get_memory_stats())
        return stats
    
    def benchmark_gpu_vs_cpu(self, test_size: int = 10000) -> Dict[str, Any]:
        """
        Benchmark GPU vs CPU performance for rational arithmetic.
        """
        # Create test data
        test_terms = [FractionTerm(i, i + 1) for i in range(1, test_size + 1)]
        test_rational = RationalListNumber(test_terms)
        
        # CPU benchmark
        cpu_start = time.time()
        cpu_result = test_rational.simplify()
        cpu_time = time.time() - cpu_start
        
        # GPU benchmark
        gpu_start = time.time()
        gpu_result = self.evaluate_rational_list_gpu(test_rational)
        gpu_time = time.time() - gpu_start
        
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        
        return {
            'test_size': test_size,
            'cpu_time_ms': cpu_time * 1000,
            'gpu_time_ms': gpu_time * 1000,
            'speedup': f"{speedup:.2f}x",
            'results_match': cpu_result.evaluate_exact() == gpu_result.evaluate_exact(),
            'gpu_available': self.gpu_available
        }
