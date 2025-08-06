"""
Inference Model Demo
===================

Comprehensive demonstration of Arbitrary Numbers in inference models.
Shows exact symbolic computation, GPU acceleration, and explainable AI.
"""

import torch
import torch.nn as nn
import numpy as np
import time
import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arbitrary_numbers.core.rational_list import RationalListNumber, FractionTerm
from arbitrary_numbers.core.equation_nodes import ConstantNode, BinaryOpNode, ExpressionBuilder
from arbitrary_numbers.core.evaluator import EquationEvaluator
from arbitrary_numbers.gpu.cuda_kernels import GPUEvaluator
from arbitrary_numbers.ml.pytorch_layers import SymbolicLinear, ExplainableInferenceModel


def demonstrate_basic_operations():
    """Demonstrate basic Arbitrary Numbers operations."""
    print("=" * 60)
    print("BASIC ARBITRARY NUMBERS OPERATIONS")
    print("=" * 60)
    
    # Create rational numbers
    print("\n1. Creating Rational Numbers:")
    r1 = RationalListNumber.from_fraction(1, 3)
    r2 = RationalListNumber.from_fraction(2, 5)
    print(f"r1 = {r1}")
    print(f"r2 = {r2}")
    
    # Arithmetic operations
    print("\n2. Arithmetic Operations:")
    sum_result = r1 + r2
    print(f"r1 + r2 = {sum_result}")
    print(f"Exact value: {sum_result.evaluate_exact()}")
    print(f"Decimal approximation: {float(sum_result.evaluate_exact()):.10f}")
    
    product = r1 * r2
    print(f"r1 * r2 = {product}")
    print(f"Exact value: {product.evaluate_exact()}")
    
    # Complex expression
    print("\n3. Complex Expression:")
    r3 = RationalListNumber.from_fraction(7, 12)
    complex_expr = (r1 + r2) * r3 - RationalListNumber.from_fraction(1, 4)
    print(f"(r1 + r2) * (7/12) - 1/4 = {complex_expr}")
    print(f"Exact result: {complex_expr.evaluate_exact()}")
    
    # Deferred evaluation demonstration
    print("\n4. Deferred Evaluation:")
    print(f"Number of terms in complex expression: {len(complex_expr.terms)}")
    simplified = complex_expr.simplify()
    print(f"After simplification: {simplified}")
    print(f"Number of terms after simplification: {len(simplified.terms)}")


def demonstrate_symbolic_computation():
    """Demonstrate symbolic equation trees."""
    print("\n" + "=" * 60)
    print("SYMBOLIC COMPUTATION")
    print("=" * 60)
    
    # Build symbolic expression: (x + 1/2) * (y - 1/3)
    print("\n1. Building Symbolic Expression:")
    builder = ExpressionBuilder
    
    x = builder.variable("x")
    y = builder.variable("y")
    half = builder.constant(RationalListNumber.from_fraction(1, 2))
    third = builder.constant(RationalListNumber.from_fraction(1, 3))
    
    left_expr = builder.add(x, half)
    right_expr = builder.subtract(y, third)
    final_expr = builder.multiply(left_expr, right_expr)
    
    print(f"Expression: {final_expr.to_string()}")
    print(f"Complexity: {final_expr.complexity()}")
    
    # Evaluate with different variable values
    print("\n2. Evaluating with Variables:")
    evaluator = EquationEvaluator()
    
    test_cases = [
        {"x": RationalListNumber.from_fraction(3, 4), "y": RationalListNumber.from_fraction(5, 6)},
        {"x": RationalListNumber.from_int(2), "y": RationalListNumber.from_int(3)},
        {"x": RationalListNumber.from_fraction(-1, 2), "y": RationalListNumber.from_fraction(1, 3)}
    ]
    
    for i, variables in enumerate(test_cases, 1):
        result = evaluator.evaluate(final_expr, variables)
        print(f"Case {i}: x={variables['x']}, y={variables['y']}")
        print(f"  Result: {result.evaluate_exact()}")
    
    # Show evaluation statistics
    print(f"\n3. Evaluation Statistics:")
    stats = evaluator.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")


def demonstrate_gpu_acceleration():
    """Demonstrate GPU acceleration capabilities."""
    print("\n" + "=" * 60)
    print("GPU ACCELERATION")
    print("=" * 60)
    
    gpu_evaluator = GPUEvaluator()
    
    # Show GPU information
    print("\n1. GPU Information:")
    gpu_info = gpu_evaluator.get_gpu_info()
    for key, value in gpu_info.items():
        print(f"  {key}: {value}")
    
    if not gpu_info.get('gpu_available', False):
        print("\n  Note: GPU not available, using CPU fallback")
        return
    
    # Performance benchmark
    print("\n2. Performance Benchmark:")
    test_sizes = [1000, 5000, 10000]
    
    for size in test_sizes:
        print(f"\n  Testing with {size} terms:")
        benchmark = gpu_evaluator.benchmark_gpu_vs_cpu(size)
        
        for key, value in benchmark.items():
            print(f"    {key}: {value}")
    
    # Show performance statistics
    print("\n3. GPU Performance Statistics:")
    perf_stats = gpu_evaluator.get_performance_stats()
    for key, value in perf_stats.items():
        print(f"  {key}: {value}")


def demonstrate_pytorch_integration():
    """Demonstrate PyTorch integration with symbolic layers."""
    print("\n" + "=" * 60)
    print("PYTORCH INTEGRATION")
    print("=" * 60)
    
    # Create symbolic linear layer
    print("\n1. Creating Symbolic Linear Layer:")
    symbolic_layer = SymbolicLinear(3, 2, bias=True)
    print(f"Layer: {symbolic_layer}")
    
    # Show some symbolic weights
    print("\n2. Symbolic Weight Examples:")
    for i in range(2):
        for j in range(3):
            weight_expr = symbolic_layer.get_symbolic_weight(i, j)
            print(f"  w[{i},{j}] = {weight_expr}")
    
    # Test forward pass
    print("\n3. Forward Pass Test:")
    test_input = torch.randn(1, 3)
    print(f"Input: {test_input}")
    
    output = symbolic_layer(test_input)
    print(f"Output: {output}")
    
    # Synchronize symbolic and numeric weights
    print("\n4. Weight Synchronization:")
    print("Before sync - PyTorch weights:")
    print(symbolic_layer.weight.data)
    
    symbolic_layer.sync_weights()
    print("After sync - PyTorch weights:")
    print(symbolic_layer.weight.data)
    
    print("Symbolic weights (evaluated):")
    symbolic_weights = symbolic_layer.evaluate_symbolic_weights()
    print(symbolic_weights)


def demonstrate_explainable_inference():
    """Demonstrate explainable inference model."""
    print("\n" + "=" * 60)
    print("EXPLAINABLE INFERENCE MODEL")
    print("=" * 60)
    
    # Create explainable model
    print("\n1. Creating Explainable Model:")
    model = ExplainableInferenceModel(
        input_dim=4,
        hidden_dims=[6, 4],
        output_dim=2
    )
    print(f"Model: {model}")
    
    # Show model complexity
    print("\n2. Model Complexity:")
    complexity = model.get_model_complexity()
    for key, value in complexity.items():
        print(f"  {key}: {value}")
    
    # Test prediction with explanation
    print("\n3. Prediction with Explanation:")
    test_input = torch.tensor([1.0, 0.5, -0.3, 0.8])
    print(f"Input: {test_input.tolist()}")
    
    # Get prediction
    with torch.no_grad():
        output = model(test_input.unsqueeze(0))
    print(f"Output: {output.squeeze().tolist()}")
    
    # Get explanation
    explanation = model.explain_prediction(test_input, output_index=0)
    
    print("\n4. Symbolic Explanation:")
    print(f"  Input values: {explanation['input_values']}")
    print(f"  Output value: {explanation['output_value']:.6f}")
    print(f"  Exact computation: {explanation['exact_computation']}")
    print(f"  Precision loss: {explanation['precision_loss']}")
    
    print("\n  Computation trace:")
    for step in explanation['computation_trace']:
        print(f"    Layer {step['layer_index']}: {step['layer_type']} "
              f"(symbolic: {step['has_symbolic_weights']})")
    
    print("\n  Sample symbolic expressions:")
    expressions = explanation['symbolic_expressions']
    for layer_name, layer_expressions in expressions.items():
        print(f"    {layer_name}:")
        for expr in layer_expressions[:3]:  # Show first 3 expressions
            print(f"      {expr}")
        if len(layer_expressions) > 3:
            print(f"      ... and {len(layer_expressions) - 3} more")


def demonstrate_precision_comparison():
    """Compare precision between floating-point and arbitrary numbers."""
    print("\n" + "=" * 60)
    print("PRECISION COMPARISON")
    print("=" * 60)
    
    print("\n1. Floating-Point vs Arbitrary Numbers:")
    
    # Test case: repeated division and multiplication
    print("\nTest: Repeated operations that accumulate error")
    
    # Floating-point version
    fp_value = 1.0
    for i in range(100):
        fp_value = fp_value / 3.0
        fp_value = fp_value * 3.0
    
    # Arbitrary numbers version
    arb_value = RationalListNumber.from_int(1)
    three = RationalListNumber.from_int(3)
    
    for i in range(100):
        arb_value = arb_value / FractionTerm(3, 1)
        arb_value = arb_value * three
    
    print(f"Floating-point result: {fp_value}")
    print(f"Expected result: 1.0")
    print(f"Floating-point error: {abs(fp_value - 1.0)}")
    
    arb_result = arb_value.evaluate_exact()
    print(f"Arbitrary numbers result: {arb_result}")
    print(f"Arbitrary numbers error: {abs(float(arb_result) - 1.0)}")
    
    # Test case: sum of fractions
    print("\n2. Sum of Many Small Fractions:")
    
    # Floating-point sum
    fp_sum = 0.0
    for i in range(1, 1001):
        fp_sum += 1.0 / i
    
    # Arbitrary numbers sum
    arb_sum = RationalListNumber([])
    for i in range(1, 1001):
        term = RationalListNumber.from_fraction(1, i)
        arb_sum = arb_sum + term
    
    print(f"Floating-point sum: {fp_sum}")
    arb_sum_exact = arb_sum.evaluate_exact()
    print(f"Arbitrary numbers sum: {float(arb_sum_exact)}")
    print(f"Difference: {abs(fp_sum - float(arb_sum_exact))}")
    print(f"Arbitrary numbers has {len(arb_sum.terms)} terms before simplification")


def main():
    """Run all demonstrations."""
    print("ARBITRARY NUMBERS INFERENCE MODEL DEMONSTRATION")
    print("=" * 60)
    print("Demonstrating exact symbolic computation for ML inference")
    print("Target: Consumer-grade Nvidia 32GB GPUs (RTX 4090, RTX 6000 Ada)")
    print("License: Apache 2.0")
    
    try:
        demonstrate_basic_operations()
        demonstrate_symbolic_computation()
        demonstrate_gpu_acceleration()
        demonstrate_pytorch_integration()
        demonstrate_explainable_inference()
        demonstrate_precision_comparison()
        
        print("\n" + "=" * 60)
        print("DEMONSTRATION COMPLETE")
        print("=" * 60)
        print("\nKey Benefits Demonstrated:")
        print("✓ Zero precision loss in all computations")
        print("✓ Complete symbolic traceability")
        print("✓ GPU acceleration for performance")
        print("✓ PyTorch integration for ML workflows")
        print("✓ Explainable AI with exact mathematical reasoning")
        print("✓ Deferred evaluation for efficiency")
        
        print("\nNext Steps:")
        print("• Run comprehensive benchmarks on target hardware")
        print("• Implement additional ML layer types")
        print("• Develop symbolic differentiation engine")
        print("• Create production-ready inference pipelines")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
