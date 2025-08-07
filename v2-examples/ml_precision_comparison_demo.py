"""
Machine Learning Precision Comparison Demo

This demonstration shows how ArbitraryNumber maintains exact precision
in machine learning calculations where floating-point arithmetic fails.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from v2.core.arbitrary_number import ArbitraryNumber
import time
import math


def floating_point_gradient_descent():
    """Traditional gradient descent with floating-point precision loss."""
    print("=== Floating-Point Gradient Descent ===")
    
    # Simple quadratic function: f(x) = (x - 0.1)^2
    # Minimum at x = 0.1
    learning_rate = 0.01
    x = 0.0
    
    print(f"Target minimum: x = 0.1")
    print(f"Learning rate: {learning_rate}")
    print("\nIteration | x value | f(x) | gradient")
    print("-" * 45)
    
    for i in range(50):
        # Calculate gradient: 2(x - 0.1)
        gradient = 2 * (x - 0.1)
        f_x = (x - 0.1) ** 2
        
        if i % 10 == 0:
            print(f"{i:9d} | {x:7.6f} | {f_x:.8f} | {gradient:.6f}")
        
        # Update x
        x = x - learning_rate * gradient
        
        # Check for convergence
        if abs(gradient) < 1e-10:
            print(f"Converged at iteration {i}")
            break
    
    print(f"\nFinal result: x = {x}")
    print(f"Error from true minimum: {abs(x - 0.1)}")
    print(f"Final function value: {(x - 0.1) ** 2}")
    
    return x


def arbitrary_number_gradient_descent():
    """Gradient descent with ArbitraryNumber exact precision."""
    print("\n=== ArbitraryNumber Gradient Descent ===")
    
    # Same function: f(x) = (x - 0.1)^2
    learning_rate = ArbitraryNumber("0.01")
    x = ArbitraryNumber("0")
    target = ArbitraryNumber("0.1")
    
    print(f"Target minimum: x = {target}")
    print(f"Learning rate: {learning_rate}")
    print("\nIteration | x value | f(x) | gradient")
    print("-" * 60)
    
    for i in range(50):
        # Calculate gradient: 2(x - 0.1)
        diff = x - target
        gradient = ArbitraryNumber("2") * diff
        f_x = diff * diff
        
        if i % 10 == 0:
            print(f"{i:9d} | {str(x):15s} | {str(f_x):15s} | {str(gradient):10s}")
        
        # Update x
        x = x - learning_rate * gradient
        
        # Check for convergence (using exact arithmetic)
        if gradient.is_zero():
            print(f"Exact convergence at iteration {i}")
            break
    
    print(f"\nFinal result: x = {x}")
    error = x - target
    print(f"Error from true minimum: {error}")
    print(f"Final function value: {error * error}")
    
    return x


def neural_network_weight_update_comparison():
    """Compare weight updates in neural networks."""
    print("\n" + "="*60)
    print("NEURAL NETWORK WEIGHT UPDATE COMPARISON")
    print("="*60)
    
    # Simulate weight updates with small learning rates
    print("\n--- Floating-Point Weight Updates ---")
    weight_fp = 0.1
    learning_rate_fp = 0.001
    
    for i in range(1000):
        gradient = 0.0001  # Very small gradient
        weight_fp = weight_fp - learning_rate_fp * gradient
    
    print(f"Initial weight: 0.1")
    print(f"Learning rate: 0.001")
    print(f"Gradient per step: 0.0001")
    print(f"Expected final weight: 0.1 - 1000 * 0.001 * 0.0001 = 0.0999")
    print(f"Actual final weight: {weight_fp}")
    print(f"Precision loss: {abs(weight_fp - 0.0999)}")
    
    print("\n--- ArbitraryNumber Weight Updates ---")
    weight_an = ArbitraryNumber("0.1")
    learning_rate_an = ArbitraryNumber("0.001")
    gradient_an = ArbitraryNumber("0.0001")
    
    for i in range(1000):
        weight_an = weight_an - learning_rate_an * gradient_an
    
    expected = ArbitraryNumber("0.0999")
    print(f"Initial weight: 0.1")
    print(f"Learning rate: 0.001")
    print(f"Gradient per step: 0.0001")
    print(f"Expected final weight: {expected}")
    print(f"Actual final weight: {weight_an}")
    print(f"Precision loss: {weight_an - expected}")


def matrix_multiplication_precision():
    """Demonstrate precision in matrix operations."""
    print("\n" + "="*60)
    print("MATRIX MULTIPLICATION PRECISION")
    print("="*60)
    
    # Small matrix multiplication that accumulates errors
    print("\n--- Floating-Point Matrix Operations ---")
    
    # 2x2 matrix multiplication repeated many times
    a_fp = [[0.1, 0.2], [0.3, 0.4]]
    b_fp = [[0.5, 0.6], [0.7, 0.8]]
    
    # Multiply matrices 100 times
    result_fp = [[1.0, 0.0], [0.0, 1.0]]  # Start with identity
    
    for _ in range(10):
        # Matrix multiplication
        new_result = [[0.0, 0.0], [0.0, 0.0]]
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    new_result[i][j] += result_fp[i][k] * a_fp[k][j]
        result_fp = new_result
    
    print("Result after 10 matrix multiplications (floating-point):")
    for row in result_fp:
        print([f"{x:.10f}" for x in row])
    
    print("\n--- ArbitraryNumber Matrix Operations ---")
    
    # Same operations with ArbitraryNumber
    a_an = [[ArbitraryNumber("0.1"), ArbitraryNumber("0.2")],
            [ArbitraryNumber("0.3"), ArbitraryNumber("0.4")]]
    
    result_an = [[ArbitraryNumber("1"), ArbitraryNumber("0")],
                 [ArbitraryNumber("0"), ArbitraryNumber("1")]]
    
    for _ in range(10):
        # Matrix multiplication
        new_result = [[ArbitraryNumber("0"), ArbitraryNumber("0")],
                      [ArbitraryNumber("0"), ArbitraryNumber("0")]]
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    new_result[i][j] = new_result[i][j] + result_an[i][k] * a_an[k][j]
        result_an = new_result
    
    print("Result after 10 matrix multiplications (ArbitraryNumber):")
    for row in result_an:
        print([str(x) for x in row])


def loss_function_precision():
    """Demonstrate precision in loss function calculations."""
    print("\n" + "="*60)
    print("LOSS FUNCTION PRECISION COMPARISON")
    print("="*60)
    
    # Mean Squared Error with very small differences
    print("\n--- Floating-Point MSE Calculation ---")
    
    predictions_fp = [0.1, 0.2, 0.3, 0.4, 0.5]
    targets_fp = [0.100001, 0.200001, 0.300001, 0.400001, 0.500001]
    
    mse_fp = 0.0
    for pred, target in zip(predictions_fp, targets_fp):
        diff = pred - target
        mse_fp += diff * diff
    mse_fp /= len(predictions_fp)
    
    print(f"Predictions: {predictions_fp}")
    print(f"Targets: {targets_fp}")
    print(f"MSE (floating-point): {mse_fp}")
    print(f"Expected MSE: {1e-12}")  # (0.000001)^2 = 1e-12
    
    print("\n--- ArbitraryNumber MSE Calculation ---")
    
    predictions_an = [ArbitraryNumber("0.1"), ArbitraryNumber("0.2"), 
                      ArbitraryNumber("0.3"), ArbitraryNumber("0.4"), ArbitraryNumber("0.5")]
    targets_an = [ArbitraryNumber("0.100001"), ArbitraryNumber("0.200001"),
                  ArbitraryNumber("0.300001"), ArbitraryNumber("0.400001"), ArbitraryNumber("0.500001")]
    
    mse_an = ArbitraryNumber("0")
    for pred, target in zip(predictions_an, targets_an):
        diff = pred - target
        mse_an = mse_an + diff * diff
    mse_an = mse_an / ArbitraryNumber(str(len(predictions_an)))
    
    print(f"Predictions: {[str(x) for x in predictions_an]}")
    print(f"Targets: {[str(x) for x in targets_an]}")
    print(f"MSE (ArbitraryNumber): {mse_an}")
    print(f"Expected MSE: {ArbitraryNumber('0.000001') ** ArbitraryNumber('2')}")


def main():
    """Run all precision comparison demonstrations."""
    print("ARBITRARYNUMBER ML PRECISION DEMONSTRATION")
    print("=" * 60)
    print("This demo shows how ArbitraryNumber maintains exact precision")
    print("in machine learning calculations where floating-point fails.")
    print("=" * 60)
    
    start_time = time.time()
    
    # Run demonstrations
    floating_point_gradient_descent()
    arbitrary_number_gradient_descent()
    
    neural_network_weight_update_comparison()
    matrix_multiplication_precision()
    loss_function_precision()
    
    end_time = time.time()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("ArbitraryNumber provides:")
    print("1. Exact precision in gradient descent convergence")
    print("2. No accumulation errors in weight updates")
    print("3. Perfect accuracy in matrix operations")
    print("4. Precise loss function calculations")
    print("5. Deterministic and reproducible results")
    print(f"\nTotal execution time: {end_time - start_time:.4f} seconds")


if __name__ == "__main__":
    main()
