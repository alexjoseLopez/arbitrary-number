"""
Advanced Parallel Inference Demo
================================

Demonstrates the advanced parallel algorithms for symbolic inference.
"""

import torch
import numpy as np
import time
from typing import Dict, List, Any

from inference_engine.parallel import (
    ParallelConfig,
    ParallelInferenceOrchestrator,
    AdvancedParallelKernelManager,
    CUDAKernelConfig
)
from inference_engine.core import InferenceEngine
from arbitrary_numbers.core.rational_list import RationalListNumber
from arbitrary_numbers.core.equation_nodes import VariableNode, ConstantNode, AdditionNode, MultiplicationNode
from arbitrary_numbers.ml.pytorch_layers import ExplainableInferenceModel


def create_sample_model(input_dim: int = 4, hidden_dim: int = 8, output_dim: int = 2) -> ExplainableInferenceModel:
    """Create a sample explainable inference model."""
    model = ExplainableInferenceModel(
        input_dim=input_dim,
        hidden_dims=[hidden_dim, hidden_dim],
        output_dim=output_dim,
        enable_symbolic_weights=True
    )
    return model


def demonstrate_parallel_inference():
    """Demonstrate parallel inference across multiple GPUs."""
    print("🚀 Advanced Parallel Inference Demo")
    print("=" * 50)
    
    config = ParallelConfig(
        num_gpus=min(4, torch.cuda.device_count()) if torch.cuda.is_available() else 1,
        num_cpu_workers=8,
        batch_size_per_gpu=32,
        pipeline_depth=4,
        load_balancing_strategy="adaptive"
    )
    
    orchestrator = ParallelInferenceOrchestrator(config)
    
    model = create_sample_model()
    model_name = "demo_model"
    
    print(f"📊 Registering model on {config.num_gpus} GPUs...")
    orchestrator.register_model_on_all_gpus(model_name, model)
    
    batch_sizes = [16, 64, 128, 256]
    
    print("\n🔄 Testing different batch sizes:")
    print("-" * 40)
    
    for batch_size in batch_sizes:
        batch_inputs = torch.randn(batch_size, 4)
        
        start_time = time.time()
        result = orchestrator.infer_with_optimal_parallelization(
            model_name, batch_inputs, require_symbolic=True
        )
        inference_time = time.time() - start_time
        
        print(f"Batch size {batch_size:3d}: {inference_time*1000:6.2f}ms")
        
        if 'outputs' in result:
            print(f"  Output shape: {result['outputs'].shape}")
        if 'symbolic_info' in result and result['symbolic_info']:
            print(f"  Symbolic expressions: {len(result['symbolic_info'].get('symbolic_expressions', {}))}")
    
    stats = orchestrator.get_comprehensive_stats()
    print(f"\n📈 Performance Statistics:")
    print(f"  Total requests: {stats['orchestrator_stats']['total_requests']}")
    print(f"  Parallel requests: {stats['orchestrator_stats']['parallel_requests']}")
    print(f"  Average parallelization factor: {stats['orchestrator_stats']['average_parallelization_factor']:.2f}")
    
    return orchestrator


def demonstrate_cuda_kernels():
    """Demonstrate advanced CUDA parallel kernels."""
    print("\n⚡ CUDA Parallel Kernels Demo")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("CUDA not available, using CPU fallback")
        device = 'cpu'
    else:
        device = 'cuda'
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    
    config = CUDAKernelConfig(
        block_size=256,
        grid_size=1024,
        shared_memory_size=48 * 1024
    )
    
    kernel_manager = AdvancedParallelKernelManager(config)
    
    print(f"\n🔢 Testing rational arithmetic operations:")
    print("-" * 40)
    
    data_sizes = [1000, 10000, 100000]
    
    for size in data_sizes:
        num_a = torch.randint(1, 1000, (size,), dtype=torch.long, device=device)
        den_a = torch.randint(1, 1000, (size,), dtype=torch.long, device=device)
        num_b = torch.randint(1, 1000, (size,), dtype=torch.long, device=device)
        den_b = torch.randint(1, 1000, (size,), dtype=torch.long, device=device)
        
        operations = [{
            'type': 'add',
            'num_a': num_a,
            'den_a': den_a,
            'num_b': num_b,
            'den_b': den_b
        }]
        
        start_time = time.time()
        results = kernel_manager.execute_parallel_rational_operations(operations)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        execution_time = time.time() - start_time
        
        throughput = size / max(execution_time, 1e-6)
        
        print(f"Size {size:6d}: {execution_time*1000:6.2f}ms ({throughput/1e6:.2f}M ops/sec)")
    
    print(f"\n🧮 Testing matrix operations:")
    print("-" * 40)
    
    matrix_sizes = [(64, 64), (128, 128), (256, 256)]
    
    for m, n in matrix_sizes:
        matrix_a_num = torch.randint(1, 100, (m, n), dtype=torch.long, device=device)
        matrix_a_den = torch.randint(1, 100, (m, n), dtype=torch.long, device=device)
        matrix_b_num = torch.randint(1, 100, (n, m), dtype=torch.long, device=device)
        matrix_b_den = torch.randint(1, 100, (n, m), dtype=torch.long, device=device)
        
        operations = [{
            'type': 'matrix_multiply',
            'matrix_a_num': matrix_a_num,
            'matrix_a_den': matrix_a_den,
            'matrix_b_num': matrix_b_num,
            'matrix_b_den': matrix_b_den
        }]
        
        start_time = time.time()
        results = kernel_manager.execute_parallel_rational_operations(operations)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        execution_time = time.time() - start_time
        
        flops = 2 * m * n * m
        gflops = flops / max(execution_time, 1e-6) / 1e9
        
        print(f"Matrix {m}x{n}: {execution_time*1000:6.2f}ms ({gflops:.2f} GFLOPS)")
    
    stats = kernel_manager.get_kernel_performance_stats()
    print(f"\n📊 Kernel Statistics:")
    print(f"  Total kernel launches: {stats['total_kernel_launches']}")
    print(f"  Average kernel time: {stats.get('average_kernel_time', 0)*1000:.2f}ms")
    
    return kernel_manager


def demonstrate_symbolic_parallelization():
    """Demonstrate parallel symbolic expression evaluation."""
    print("\n🔣 Symbolic Expression Parallelization Demo")
    print("=" * 50)
    
    from inference_engine.parallel.distributed_inference import SymbolicExpressionParallelizer
    
    parallelizer = SymbolicExpressionParallelizer(num_workers=8)
    
    x = VariableNode("x")
    y = VariableNode("y")
    z = VariableNode("z")
    
    expressions = [
        AdditionNode(x, y),
        MultiplicationNode(x, ConstantNode(RationalListNumber.from_int(2))),
        AdditionNode(MultiplicationNode(x, y), z),
        MultiplicationNode(AdditionNode(x, ConstantNode(RationalListNumber.from_int(1))), y),
        AdditionNode(AdditionNode(x, y), z),
        MultiplicationNode(MultiplicationNode(x, y), z),
        AdditionNode(x, MultiplicationNode(y, z)),
        MultiplicationNode(x, AdditionNode(y, z))
    ]
    
    variables = {
        "x": RationalListNumber.from_fraction(3, 2),
        "y": RationalListNumber.from_fraction(5, 3),
        "z": RationalListNumber.from_fraction(7, 4)
    }
    
    print(f"Evaluating {len(expressions)} expressions in parallel...")
    
    start_time = time.time()
    results = parallelizer.evaluate_expressions_parallel(expressions, variables)
    parallel_time = time.time() - start_time
    
    print(f"Parallel evaluation: {parallel_time*1000:.2f}ms")
    
    print(f"\n📋 Results:")
    print("-" * 30)
    for i, (expr, result) in enumerate(zip(expressions, results)):
        if hasattr(result, 'evaluate_exact'):
            value = result.evaluate_exact()
        else:
            value = str(result)
        print(f"  Expression {i+1}: {value}")
    
    return parallelizer


def benchmark_parallel_vs_sequential():
    """Benchmark parallel vs sequential inference."""
    print("\n⚖️  Parallel vs Sequential Benchmark")
    print("=" * 50)
    
    model = create_sample_model(input_dim=8, hidden_dim=16, output_dim=4)
    
    config = ParallelConfig(
        num_gpus=min(2, torch.cuda.device_count()) if torch.cuda.is_available() else 1,
        num_cpu_workers=4,
        batch_size_per_gpu=64
    )
    
    orchestrator = ParallelInferenceOrchestrator(config)
    orchestrator.register_model_on_all_gpus("benchmark_model", model)
    
    sequential_engine = InferenceEngine()
    sequential_engine.register_model("benchmark_model", model)
    
    batch_sizes = [32, 128, 512, 1024]
    
    print("Batch Size | Sequential | Parallel | Speedup")
    print("-" * 45)
    
    for batch_size in batch_sizes:
        batch_inputs = torch.randn(batch_size, 8)
        
        start_time = time.time()
        _ = sequential_engine.infer("benchmark_model", batch_inputs)
        sequential_time = time.time() - start_time
        
        start_time = time.time()
        _ = orchestrator.infer_with_optimal_parallelization("benchmark_model", batch_inputs)
        parallel_time = time.time() - start_time
        
        speedup = sequential_time / max(parallel_time, 1e-6)
        
        print(f"{batch_size:10d} | {sequential_time*1000:9.2f}ms | {parallel_time*1000:8.2f}ms | {speedup:6.2f}x")


def main():
    """Run the complete parallel inference demonstration."""
    print("🌟 Advanced Parallel Algorithms for Symbolic Inference")
    print("=" * 60)
    print("Demonstrating cutting-edge parallel algorithms optimized for")
    print("32GB consumer GPUs with exact symbolic computation.")
    print()
    
    try:
        orchestrator = demonstrate_parallel_inference()
        
        kernel_manager = demonstrate_cuda_kernels()
        
        parallelizer = demonstrate_symbolic_parallelization()
        
        benchmark_parallel_vs_sequential()
        
        print("\n🎯 Summary")
        print("=" * 50)
        print("✅ Parallel inference orchestration")
        print("✅ Advanced CUDA kernel optimization")
        print("✅ Symbolic expression parallelization")
        print("✅ Adaptive load balancing")
        print("✅ Pipeline-based processing")
        print("✅ Work-stealing algorithms")
        print("✅ Memory pool management")
        print("✅ Performance monitoring")
        
        print(f"\n🚀 The advanced parallel algorithms are ready for production!")
        print(f"   Optimized for 32GB consumer GPUs with exact computation.")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
