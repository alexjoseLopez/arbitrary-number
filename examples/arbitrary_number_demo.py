"""
ArbitraryNumber Demonstration
============================

Comprehensive demonstration of the ArbitraryNumber class and its capabilities.
This example showcases the revolutionary new mathematical concept of ArbitraryNumbers
for exact symbolic computation with zero precision loss.
"""

import sys
import os
import time
from fractions import Fraction

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from arbitrary_numbers.core.arbitrary_number import ArbitraryNumber, FractionTerm


def demonstrate_basic_operations():
    """Demonstrate basic ArbitraryNumber operations."""
    print("=" * 60)
    print("BASIC ARBITRARY NUMBER OPERATIONS")
    print("=" * 60)
    
    # Creation methods
    print("1. Creating ArbitraryNumbers:")
    num1 = ArbitraryNumber.from_int(42)
    num2 = ArbitraryNumber.from_fraction(3, 4)
    num3 = ArbitraryNumber.from_decimal(0.125)
    
    print(f"   From integer: {num1} = {num1.evaluate_exact()}")
    print(f"   From fraction: {num2} = {num2.evaluate_exact()}")
    print(f"   From decimal: {num3} = {num3.evaluate_exact()}")
    print()
    
    # Basic arithmetic
    print("2. Basic Arithmetic:")
    a = ArbitraryNumber.from_fraction(1, 3)
    b = ArbitraryNumber.from_fraction(1, 6)
    
    addition = a + b
    subtraction = a - b
    multiplication = a * b
    division = a / b
    
    print(f"   {a} + {b} = {addition} = {addition.evaluate_exact()}")
    print(f"   {a} - {b} = {subtraction} = {subtraction.evaluate_exact()}")
    print(f"   {a} * {b} = {multiplication} = {multiplication.evaluate_exact()}")
    print(f"   {a} / {b} = {division} = {division.evaluate_exact()}")
    print()
    
    # Power operations
    print("3. Power Operations:")
    base = ArbitraryNumber.from_fraction(2, 3)
    
    print(f"   ({base})^0 = {base ** 0}")
    print(f"   ({base})^1 = {base ** 1}")
    print(f"   ({base})^2 = {base ** 2} = {(base ** 2).evaluate_exact()}")
    print(f"   ({base})^3 = {base ** 3} = {(base ** 3).evaluate_exact()}")
    print(f"   ({base})^-2 = {base ** -2} = {(base ** -2).evaluate_exact()}")
    print()


def demonstrate_precision_preservation():
    """Demonstrate zero precision loss property."""
    print("=" * 60)
    print("ZERO PRECISION LOSS DEMONSTRATION")
    print("=" * 60)
    
    print("Comparing floating-point vs ArbitraryNumber precision:")
    print()
    
    # Floating-point accumulation error
    print("1. Floating-point error accumulation:")
    fp_result = 1.0
    for i in range(50):
        fp_result = fp_result / 3.0 * 3.0
    
    fp_error = abs(fp_result - 1.0)
    print(f"   After 50 iterations of (x/3)*3:")
    print(f"   Float result: {fp_result}")
    print(f"   Error: {fp_error:.2e}")
    print()
    
    # ArbitraryNumber maintains exactness
    print("2. ArbitraryNumber exact computation:")
    arb_result = ArbitraryNumber.from_int(1)
    three = ArbitraryNumber.from_int(3)
    
    for i in range(50):
        arb_result = (arb_result / three) * three
    
    arb_exact = arb_result.evaluate_exact()
    arb_error = abs(float(arb_exact) - 1.0)
    
    print(f"   After 50 iterations of (x/3)*3:")
    print(f"   ArbitraryNumber result: {arb_exact}")
    print(f"   Error: {arb_error}")
    print(f"   Precision loss: {arb_result.get_precision_loss()}")
    print()
    
    # Complex fraction operations
    print("3. Complex fraction operations:")
    complex_calc = ArbitraryNumber.from_fraction(1, 7)
    complex_calc = complex_calc + ArbitraryNumber.from_fraction(1, 11)
    complex_calc = complex_calc * ArbitraryNumber.from_fraction(13, 17)
    complex_calc = complex_calc - ArbitraryNumber.from_fraction(1, 19)
    
    print(f"   (1/7 + 1/11) * 13/17 - 1/19 = {complex_calc.evaluate_exact()}")
    print(f"   As decimal: {float(complex_calc.evaluate_exact()):.15f}")
    print(f"   Precision loss: {complex_calc.get_precision_loss()}")
    print()


def demonstrate_deferred_evaluation():
    """Demonstrate deferred evaluation and symbolic representation."""
    print("=" * 60)
    print("DEFERRED EVALUATION DEMONSTRATION")
    print("=" * 60)
    
    print("Building complex expression with deferred evaluation:")
    print()
    
    # Build complex expression step by step
    expr = ArbitraryNumber.from_fraction(1, 2)
    print(f"1. Start with: {expr}")
    
    expr = expr + ArbitraryNumber.from_fraction(1, 3)
    print(f"2. Add 1/3: {expr}")
    
    expr = expr * ArbitraryNumber.from_fraction(6, 5)
    print(f"3. Multiply by 6/5: {expr}")
    
    expr = expr - ArbitraryNumber.from_fraction(1, 10)
    print(f"4. Subtract 1/10: {expr}")
    
    print(f"5. Number of terms: {expr.term_count()}")
    print(f"6. Memory usage: {expr.memory_usage()} bytes")
    print()
    
    # Evaluate when needed
    print("Evaluation:")
    exact_result = expr.evaluate_exact()
    decimal_result = expr.evaluate(precision=20)
    
    print(f"   Exact result: {exact_result}")
    print(f"   Decimal (20 digits): {decimal_result}")
    print()
    
    # Show simplification
    simplified = expr.simplify()
    print(f"Simplified form: {simplified}")
    print(f"Simplified terms: {simplified.term_count()}")
    print(f"Simplified memory: {simplified.memory_usage()} bytes")
    print()


def demonstrate_mathematical_applications():
    """Demonstrate mathematical applications and series."""
    print("=" * 60)
    print("MATHEMATICAL APPLICATIONS")
    print("=" * 60)
    
    # Harmonic series partial sum
    print("1. Harmonic Series (first 20 terms):")
    harmonic_sum = ArbitraryNumber.zero()
    
    for n in range(1, 21):
        term = ArbitraryNumber.from_fraction(1, n)
        harmonic_sum = harmonic_sum + term
    
    print(f"   H_20 = 1 + 1/2 + 1/3 + ... + 1/20")
    print(f"   Exact result: {harmonic_sum.evaluate_exact()}")
    print(f"   Decimal approximation: {float(harmonic_sum.evaluate_exact()):.10f}")
    print(f"   Number of terms in representation: {harmonic_sum.term_count()}")
    print()
    
    # Fibonacci ratios
    print("2. Fibonacci Ratios (approaching golden ratio):")
    fib_prev = ArbitraryNumber.one()
    fib_curr = ArbitraryNumber.one()
    
    print("   F(n+1)/F(n) ratios:")
    for i in range(2, 11):
        fib_next = fib_prev + fib_curr
        ratio = fib_next / fib_curr
        
        print(f"   F({i+1})/F({i}) = {ratio.evaluate_exact()} ≈ {float(ratio.evaluate_exact()):.8f}")
        
        fib_prev, fib_curr = fib_curr, fib_next
    
    golden_ratio = (1 + (5 ** 0.5)) / 2
    print(f"   Golden ratio φ ≈ {golden_ratio:.8f}")
    print()
    
    # Continued fraction approximation of π
    print("3. Continued Fraction Approximation of π:")
    print("   Using [3; 7, 15, 1] approximation:")
    
    # π ≈ 3 + 1/(7 + 1/(15 + 1/1))
    inner = ArbitraryNumber.one()  # 1
    middle = ArbitraryNumber.from_int(15) + inner  # 15 + 1
    middle_recip = ArbitraryNumber.one() / middle  # 1/(15 + 1)
    
    outer = ArbitraryNumber.from_int(7) + middle_recip  # 7 + 1/(15 + 1)
    outer_recip = ArbitraryNumber.one() / outer  # 1/(7 + 1/(15 + 1))
    
    pi_approx = ArbitraryNumber.from_int(3) + outer_recip
    
    print(f"   π ≈ {pi_approx.evaluate_exact()}")
    print(f"   π ≈ {float(pi_approx.evaluate_exact()):.10f}")
    print(f"   Actual π ≈ {3.141592653589793:.10f}")
    print(f"   Error: {abs(float(pi_approx.evaluate_exact()) - 3.141592653589793):.2e}")
    print()


def demonstrate_performance_characteristics():
    """Demonstrate performance characteristics and caching."""
    print("=" * 60)
    print("PERFORMANCE CHARACTERISTICS")
    print("=" * 60)
    
    # Create complex number for testing
    complex_num = ArbitraryNumber([
        FractionTerm(1, i) for i in range(2, 102)  # 1/2 + 1/3 + ... + 1/101
    ])
    
    print(f"Testing with {complex_num.term_count()} terms")
    print()
    
    # Time first evaluation (no cache)
    start_time = time.perf_counter()
    result1 = complex_num.evaluate_exact()
    first_time = time.perf_counter() - start_time
    
    print(f"1. First evaluation (no cache): {first_time*1000:.4f} ms")
    
    # Time second evaluation (with cache)
    start_time = time.perf_counter()
    result2 = complex_num.evaluate_exact()
    cached_time = time.perf_counter() - start_time
    
    print(f"2. Cached evaluation: {cached_time*1000:.4f} ms")
    print(f"3. Speedup: {first_time/cached_time:.1f}x")
    print(f"4. Results identical: {result1 == result2}")
    print()
    
    # Memory usage analysis
    print("Memory Usage Analysis:")
    sizes = [10, 50, 100, 500]
    
    for size in sizes:
        test_num = ArbitraryNumber([
            FractionTerm(i, i + 1) for i in range(1, size + 1)
        ])
        
        memory = test_num.memory_usage()
        memory_per_term = memory / size
        
        print(f"   {size:3d} terms: {memory:5d} bytes ({memory_per_term:.1f} bytes/term)")
    
    print()


def demonstrate_comparison_operations():
    """Demonstrate comparison and utility operations."""
    print("=" * 60)
    print("COMPARISON AND UTILITY OPERATIONS")
    print("=" * 60)
    
    # Create test numbers
    numbers = [
        ArbitraryNumber.from_fraction(1, 4),
        ArbitraryNumber.from_fraction(1, 3),
        ArbitraryNumber.from_fraction(3, 8),
        ArbitraryNumber.from_fraction(2, 5),
        ArbitraryNumber.from_fraction(1, 2)
    ]
    
    print("1. Sorting ArbitraryNumbers:")
    print("   Original order:", [str(num) for num in numbers])
    
    sorted_numbers = sorted(numbers)
    print("   Sorted order:  ", [str(num) for num in sorted_numbers])
    print("   As decimals:   ", [f"{float(num.evaluate_exact()):.6f}" for num in sorted_numbers])
    print()
    
    # Utility methods
    print("2. Utility Methods:")
    test_cases = [
        ArbitraryNumber.zero(),
        ArbitraryNumber.one(),
        ArbitraryNumber.from_int(42),
        ArbitraryNumber.from_fraction(8, 4),  # Equals 2
        ArbitraryNumber.from_fraction(22, 7),  # Pi approximation
    ]
    
    for num in test_cases:
        print(f"   {str(num):>8} -> zero: {num.is_zero()}, one: {num.is_one()}, integer: {num.is_integer()}")
    
    print()
    
    # Metadata and tracing
    print("3. Metadata and Computation Tracing:")
    
    # Create numbers with metadata
    a = ArbitraryNumber.from_fraction(3, 5, {'source': 'user_input', 'id': 'a'})
    b = ArbitraryNumber.from_fraction(2, 7, {'source': 'calculation', 'id': 'b'})
    
    result = a * b
    
    print(f"   Operation: {a} * {b} = {result}")
    print(f"   Result metadata: {result.metadata}")
    print(f"   Computation trace: {result.get_computation_trace()}")
    print()


def main():
    """Run all demonstrations."""
    print("ARBITRARY NUMBER COMPREHENSIVE DEMONSTRATION")
    print("A Revolutionary Mathematical Concept for Exact Computation")
    print()
    
    demonstrate_basic_operations()
    demonstrate_precision_preservation()
    demonstrate_deferred_evaluation()
    demonstrate_mathematical_applications()
    demonstrate_performance_characteristics()
    demonstrate_comparison_operations()
    
    print("=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print()
    print("Key Takeaways:")
    print("• ArbitraryNumbers maintain perfect precision (zero precision loss)")
    print("• Deferred evaluation optimizes performance until results are needed")
    print("• Complete symbolic traceability for explainable AI applications")
    print("• Efficient caching system for repeated evaluations")
    print("• Comprehensive mathematical operations with exact results")
    print("• Suitable for scientific computing, financial modeling, and ML inference")
    print()
    print("The ArbitraryNumber represents a new paradigm in numerical computation,")
    print("where every calculation is exact, every result is explainable, and")
    print("mathematical operations preserve their symbolic meaning until evaluation.")


if __name__ == "__main__":
    main()
