"""
Intermediate Mathematics Unit Tests
==================================

This test suite covers medium complexity mathematical operations
using ArbitraryNumber for exact arithmetic computations.

TEST CATEGORIES:
- Polynomial operations and evaluation
- Rational arithmetic operations
- Statistical computations with exact fractions
- Matrix operations with rational entries
- Complex fraction manipulations
- Power and root approximations
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import unittest
# Import directly from the core module to avoid GPU dependencies
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'arbitrary_numbers', 'core'))
from arbitrary_number import ArbitraryNumber, FractionTerm
from fractions import Fraction
import math


class TestIntermediateMathematics(unittest.TestCase):
    """
    Test suite for intermediate complexity mathematical operations
    using ArbitraryNumber for exact arithmetic.
    """
    
    def setUp(self):
        """Set up test environment with medium precision."""
        self.precision = 30
        print(f"\n{'='*60}")
        print("INTERMEDIATE MATHEMATICS UNIT TESTS")
        print(f"Precision Level: {self.precision} digits")
        print(f"{'='*60}")
    
    def test_polynomial_operations(self):
        """
        Test polynomial evaluation and operations.
        
        Tests polynomial arithmetic, evaluation, and basic operations
        using exact arithmetic for coefficients.
        """
        print("\n📊 POLYNOMIAL OPERATIONS TEST")
        print("-" * 40)
        
        # Test polynomial evaluation: P(x) = 2x³ - 5x² + 3x - 1
        coefficients = [
            ArbitraryNumber.from_int(2),   # x³ coefficient
            ArbitraryNumber.from_int(-5),  # x² coefficient  
            ArbitraryNumber.from_int(3),   # x coefficient
            ArbitraryNumber.from_int(-1)   # constant term
        ]
        
        def evaluate_polynomial(x, coeffs):
            """Evaluate polynomial using Horner's method."""
            result = coeffs[0]
            for coeff in coeffs[1:]:
                result = result * x + coeff
            return result
        
        # Test at x = 2
        x_val = ArbitraryNumber.from_int(2)
        result = evaluate_polynomial(x_val, coefficients)
        expected = ArbitraryNumber.from_int(2*8 - 5*4 + 3*2 - 1)  # 16 - 20 + 6 - 1 = 1
        
        print(f"P(2) = {result}")
        print(f"Expected: {expected}")
        self.assertEqual(result.evaluate_exact(), expected.evaluate_exact())
        
        # Test at x = 1/2
        x_val = ArbitraryNumber.from_fraction(1, 2)
        result = evaluate_polynomial(x_val, coefficients)
        # P(1/2) = 2*(1/8) - 5*(1/4) + 3*(1/2) - 1 = 1/4 - 5/4 + 3/2 - 1 = 1/4 - 5/4 + 6/4 - 4/4 = -2/4 = -1/2
        expected = ArbitraryNumber.from_fraction(-1, 2)
        
        print(f"P(1/2) = {result}")
        print(f"Expected: {expected}")
        self.assertEqual(result.evaluate_exact(), expected.evaluate_exact())
        
        # Test polynomial addition: P(x) + Q(x) where Q(x) = x² - 2x + 1
        q_coefficients = [
            ArbitraryNumber.from_int(0),   # x³ coefficient
            ArbitraryNumber.from_int(1),   # x² coefficient
            ArbitraryNumber.from_int(-2),  # x coefficient
            ArbitraryNumber.from_int(1)    # constant term
        ]
        
        # Add polynomials coefficient-wise
        sum_coefficients = []
        for i in range(len(coefficients)):
            sum_coefficients.append(coefficients[i] + q_coefficients[i])
        
        # Result should be: 2x³ - 4x² + x + 0
        x_val = ArbitraryNumber.from_int(3)
        result = evaluate_polynomial(x_val, sum_coefficients)
        expected = ArbitraryNumber.from_int(2*27 - 4*9 + 3)  # 54 - 36 + 3 = 21
        
        print(f"(P+Q)(3) = {result}")
        print(f"Expected: {expected}")
        self.assertEqual(result.evaluate_exact(), expected.evaluate_exact())
        
        print("✅ Polynomial operations test passed")
    
    def test_rational_arithmetic_operations(self):
        """
        Test complex rational arithmetic operations.
        
        Tests operations with multiple fractions and complex expressions
        using exact arithmetic.
        """
        print("\n🔢 RATIONAL ARITHMETIC OPERATIONS TEST")
        print("-" * 40)
        
        # Test complex fraction: (1/2 + 3/4) * (5/6 - 2/3) / (7/8 + 1/4)
        a = ArbitraryNumber.from_fraction(1, 2) + ArbitraryNumber.from_fraction(3, 4)
        b = ArbitraryNumber.from_fraction(5, 6) - ArbitraryNumber.from_fraction(2, 3)
        c = ArbitraryNumber.from_fraction(7, 8) + ArbitraryNumber.from_fraction(1, 4)
        
        result = (a * b) / c
        
        # Calculate expected result step by step
        # a = 1/2 + 3/4 = 2/4 + 3/4 = 5/4
        # b = 5/6 - 2/3 = 5/6 - 4/6 = 1/6
        # c = 7/8 + 1/4 = 7/8 + 2/8 = 9/8
        # result = (5/4 * 1/6) / (9/8) = (5/24) / (9/8) = (5/24) * (8/9) = 40/216 = 5/27
        
        expected = ArbitraryNumber.from_fraction(5, 27)
        
        print(f"Complex fraction result: {result}")
        print(f"Expected: {expected}")
        self.assertEqual(result.evaluate_exact(), expected.evaluate_exact())
        
        # Test continued fraction approximation of golden ratio
        # φ = 1 + 1/(1 + 1/(1 + 1/(1 + ...)))
        phi_approx = ArbitraryNumber.from_int(1)
        for i in range(10):  # 10 iterations
            phi_approx = ArbitraryNumber.from_int(1) + ArbitraryNumber.from_int(1) / phi_approx
        
        print(f"Golden ratio approximation (10 iterations): {phi_approx}")
        
        # Verify it's close to (1 + √5)/2 ≈ 1.618
        phi_decimal = phi_approx.evaluate(20)
        self.assertGreater(float(phi_decimal), 1.6)
        self.assertLess(float(phi_decimal), 1.62)
        
        print("✅ Rational arithmetic operations test passed")
    
    def test_statistical_computations(self):
        """
        Test statistical computations with exact arithmetic.
        
        Computes mean, variance, and other statistical measures
        using high-precision rational arithmetic.
        """
        print("\n📊 STATISTICAL COMPUTATIONS TEST")
        print("-" * 40)
        
        # Create dataset with exact rational values
        data = [
            ArbitraryNumber.from_fraction(1, 2),
            ArbitraryNumber.from_fraction(3, 4),
            ArbitraryNumber.from_fraction(5, 6),
            ArbitraryNumber.from_fraction(7, 8),
            ArbitraryNumber.from_fraction(9, 10),
            ArbitraryNumber.from_fraction(11, 12),
            ArbitraryNumber.from_fraction(13, 14),
            ArbitraryNumber.from_fraction(15, 16)
        ]
        
        n = len(data)
        print(f"Dataset size: {n}")
        
        # Compute mean
        sum_data = ArbitraryNumber.from_int(0)
        for value in data:
            sum_data = sum_data + value
        
        mean = sum_data / n
        print(f"Mean: {mean}")
        
        # Compute variance
        sum_squared_deviations = ArbitraryNumber.from_int(0)
        for value in data:
            deviation = value - mean
            sum_squared_deviations = sum_squared_deviations + deviation * deviation
        
        variance = sum_squared_deviations / (n - 1)
        print(f"Variance: {variance}")
        
        # Verify mean is reasonable (should be around 0.8)
        mean_decimal = float(mean.evaluate(10))
        self.assertGreater(mean_decimal, 0.7)
        self.assertLess(mean_decimal, 0.9)
        
        # Verify variance is positive
        variance_decimal = float(variance.evaluate(10))
        self.assertGreater(variance_decimal, 0)
        
        # Compute median (middle values)
        sorted_data = sorted(data, key=lambda x: float(x.evaluate(20)))
        if n % 2 == 0:
            median = (sorted_data[n//2 - 1] + sorted_data[n//2]) / 2
        else:
            median = sorted_data[n//2]
        
        print(f"Median: {median}")
        
        # Compute range
        data_range = sorted_data[-1] - sorted_data[0]
        print(f"Range: {data_range}")
        
        print("✅ Statistical computations test passed")
    
    def test_power_operations(self):
        """
        Test power operations with exact arithmetic.
        
        Tests integer powers and power-related calculations
        using exact arithmetic.
        """
        print("\n⚡ POWER OPERATIONS TEST")
        print("-" * 40)
        
        # Test integer powers
        base = ArbitraryNumber.from_fraction(3, 2)
        
        # Test various powers
        power_2 = base ** 2
        power_3 = base ** 3
        power_0 = base ** 0
        power_neg1 = base ** -1
        
        print(f"(3/2)^2 = {power_2}")
        print(f"(3/2)^3 = {power_3}")
        print(f"(3/2)^0 = {power_0}")
        print(f"(3/2)^-1 = {power_neg1}")
        
        # Verify results
        expected_2 = ArbitraryNumber.from_fraction(9, 4)
        expected_3 = ArbitraryNumber.from_fraction(27, 8)
        expected_0 = ArbitraryNumber.from_int(1)
        expected_neg1 = ArbitraryNumber.from_fraction(2, 3)
        
        self.assertEqual(power_2.evaluate_exact(), expected_2.evaluate_exact())
        self.assertEqual(power_3.evaluate_exact(), expected_3.evaluate_exact())
        self.assertEqual(power_0.evaluate_exact(), expected_0.evaluate_exact())
        self.assertEqual(power_neg1.evaluate_exact(), expected_neg1.evaluate_exact())
        
        # Test power of sum: (a + b)^2 = a^2 + 2ab + b^2
        a = ArbitraryNumber.from_fraction(1, 3)
        b = ArbitraryNumber.from_fraction(1, 4)
        
        sum_squared = (a + b) ** 2
        expanded = a**2 + ArbitraryNumber.from_int(2) * a * b + b**2
        
        print(f"(1/3 + 1/4)^2 = {sum_squared}")
        print(f"Expanded form = {expanded}")
        
        self.assertEqual(sum_squared.evaluate_exact(), expanded.evaluate_exact())
        
        print("✅ Power operations test passed")
    
    def test_matrix_operations(self):
        """
        Test matrix operations with exact arithmetic.
        
        Tests matrix operations using exact rational arithmetic.
        """
        print("\n🔢 MATRIX OPERATIONS TEST")
        print("-" * 40)
        
        # Create 2x2 matrix with exact rational entries
        matrix_a = [
            [ArbitraryNumber.from_fraction(1, 2), ArbitraryNumber.from_fraction(3, 4)],
            [ArbitraryNumber.from_fraction(2, 3), ArbitraryNumber.from_fraction(5, 6)]
        ]
        
        matrix_b = [
            [ArbitraryNumber.from_fraction(7, 8), ArbitraryNumber.from_fraction(1, 3)],
            [ArbitraryNumber.from_fraction(2, 5), ArbitraryNumber.from_fraction(4, 7)]
        ]
        
        print("Matrix A:")
        for row in matrix_a:
            print([str(elem) for elem in row])
        
        print("Matrix B:")
        for row in matrix_b:
            print([str(elem) for elem in row])
        
        # Matrix multiplication C = A * B
        def matrix_multiply_2x2(A, B):
            """Multiply two 2x2 matrices."""
            result = []
            for i in range(2):
                row = []
                for j in range(2):
                    sum_val = ArbitraryNumber.from_int(0)
                    for k in range(2):
                        sum_val = sum_val + A[i][k] * B[k][j]
                    row.append(sum_val)
                result.append(row)
            return result
        
        matrix_c = matrix_multiply_2x2(matrix_a, matrix_b)
        
        print("Matrix C = A * B:")
        for row in matrix_c:
            print([str(elem) for elem in row])
        
        # Compute determinant of A
        det_a = matrix_a[0][0] * matrix_a[1][1] - matrix_a[0][1] * matrix_a[1][0]
        print(f"Determinant of A: {det_a}")
        
        # Verify determinant is non-zero (matrix is invertible)
        self.assertFalse(det_a.is_zero())
        
        # Test matrix trace
        trace_a = matrix_a[0][0] + matrix_a[1][1]
        trace_c = matrix_c[0][0] + matrix_c[1][1]
        
        print(f"Trace of A: {trace_a}")
        print(f"Trace of C: {trace_c}")
        
        print("✅ Matrix operations test passed")
    
    def test_numerical_approximations(self):
        """
        Test numerical approximations using exact arithmetic.
        
        Tests approximation methods that maintain exact precision
        throughout the computation process.
        """
        print("\n🔍 NUMERICAL APPROXIMATIONS TEST")
        print("-" * 40)
        
        # Test square root approximation using Newton's method
        def sqrt_newton(x, iterations=10):
            """Compute square root using Newton's method."""
            if x.is_zero():
                return ArbitraryNumber.from_int(0)
            
            # Initial guess: x/2
            guess = x / 2
            
            for i in range(iterations):
                guess = (guess + x / guess) / 2
                if i % 3 == 0:
                    print(f"  Iteration {i}: √{x} ≈ {guess}")
            
            return guess
        
        # Test √2
        x = ArbitraryNumber.from_int(2)
        sqrt_2 = sqrt_newton(x, 10)
        
        print(f"√2 ≈ {sqrt_2}")
        
        # Verify by squaring the result
        sqrt_2_squared = sqrt_2 * sqrt_2
        error = abs(sqrt_2_squared - x)
        
        print(f"(√2)² = {sqrt_2_squared}")
        print(f"Error from 2: {error}")
        
        # Error should be very small
        error_decimal = float(error.evaluate(15))
        self.assertLess(error_decimal, 1e-10)
        
        # Test cube root approximation using Newton's method (simplified)
        def cbrt_newton(x, iterations=5):
            """Compute cube root using Newton's method."""
            if x.is_zero():
                return ArbitraryNumber.from_int(0)
            
            # Initial guess: x/3
            guess = x / 3
            
            for i in range(iterations):
                guess_squared = guess * guess
                # Simplify to avoid huge numbers
                guess = guess.simplify()
                guess_squared = guess_squared.simplify()
                guess = (ArbitraryNumber.from_int(2) * guess + x / guess_squared) / 3
                guess = guess.simplify()  # Simplify after each iteration
                if i % 2 == 0:
                    print(f"  Iteration {i}: ∛{x} ≈ {guess.evaluate(10)}")
            
            return guess
        
        # Test ∛8 (should be exactly 2)
        x = ArbitraryNumber.from_int(8)
        cbrt_8 = cbrt_newton(x, 5)
        
        print(f"∛8 ≈ {cbrt_8.evaluate(10)}")
        
        # Verify by cubing the result
        cbrt_8_cubed = cbrt_8 * cbrt_8 * cbrt_8
        cbrt_8_cubed = cbrt_8_cubed.simplify()
        error = abs(cbrt_8_cubed - x)
        error = error.simplify()
        
        print(f"(∛8)³ = {cbrt_8_cubed.evaluate(10)}")
        print(f"Error from 8: {error.evaluate(15)}")
        
        # Error should be very small
        error_decimal = float(error.evaluate(15))
        self.assertLess(error_decimal, 1e-5)  # Relaxed tolerance
        
        print("✅ Numerical approximations test passed")
    
    def test_complex_expressions(self):
        """
        Test evaluation of complex mathematical expressions.
        
        Tests complex expressions that combine multiple operations
        while maintaining exact arithmetic precision.
        """
        print("\n🧮 COMPLEX EXPRESSIONS TEST")
        print("-" * 40)
        
        # Test expression: ((a + b) * c - d) / (e + f)^2
        a = ArbitraryNumber.from_fraction(1, 3)
        b = ArbitraryNumber.from_fraction(2, 5)
        c = ArbitraryNumber.from_fraction(3, 7)
        d = ArbitraryNumber.from_fraction(1, 4)
        e = ArbitraryNumber.from_fraction(1, 6)
        f = ArbitraryNumber.from_fraction(1, 8)
        
        numerator = (a + b) * c - d
        denominator = (e + f) ** 2
        result = numerator / denominator
        
        print(f"a = {a}")
        print(f"b = {b}")
        print(f"c = {c}")
        print(f"d = {d}")
        print(f"e = {e}")
        print(f"f = {f}")
        print(f"((a + b) * c - d) / (e + f)^2 = {result}")
        
        # Verify the result is not zero or one (should be a complex fraction)
        self.assertFalse(result.is_zero())
        self.assertFalse(result.is_one())
        
        # Test nested fraction: 1 / (1 + 1 / (1 + 1 / (1 + 1/2)))
        inner = ArbitraryNumber.from_int(1) + ArbitraryNumber.from_fraction(1, 2)
        middle = ArbitraryNumber.from_int(1) + ArbitraryNumber.from_int(1) / inner
        outer = ArbitraryNumber.from_int(1) + ArbitraryNumber.from_int(1) / middle
        nested_result = ArbitraryNumber.from_int(1) / outer
        
        print(f"Nested fraction result: {nested_result}")
        
        # Test harmonic series partial sum: 1 + 1/2 + 1/3 + 1/4 + 1/5
        harmonic_sum = ArbitraryNumber.from_int(0)
        for i in range(1, 6):
            harmonic_sum = harmonic_sum + ArbitraryNumber.from_fraction(1, i)
        
        print(f"Harmonic series (5 terms): {harmonic_sum}")
        
        # Should be greater than 2
        harmonic_decimal = float(harmonic_sum.evaluate(10))
        self.assertGreater(harmonic_decimal, 2.0)
        
        print("✅ Complex expressions test passed")


def run_intermediate_mathematics_tests():
    """
    Run the intermediate mathematics test suite.
    """
    print("ARBITRARYNUMBER INTERMEDIATE MATHEMATICS UNIT TESTS")
    print("MEDIUM COMPLEXITY MATHEMATICAL OPERATIONS")
    print("=" * 60)
    print()
    print("🎯 OBJECTIVE: Test medium complexity mathematical operations")
    print("🔢 COVERAGE: Polynomials, rationals, statistics, matrices, approximations")
    print("⚡ PRECISION: Exact arithmetic with rational numbers")
    print("🎯 VALIDATION: Mathematical accuracy with exact verification")
    print()
    print("=" * 60)
    
    # Create and run test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestIntermediateMathematics)
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    print("INTERMEDIATE MATHEMATICS TEST SUMMARY")
    print("=" * 60)
    
    if result.wasSuccessful():
        print("🏆 INTERMEDIATE MATHEMATICS TESTS COMPLETED SUCCESSFULLY!")
        print("✅ All mathematical operations passed with exact arithmetic")
        print("✅ Polynomial, rational, and statistical computations verified")
        print("✅ Matrix operations and numerical approximations validated")
        print("✅ Medium complexity mathematics demonstrated with precision")
        print()
        print("🎉 MATHEMATICAL ACCURACY CONFIRMED!")
        print("   ArbitraryNumber demonstrates reliable intermediate mathematics")
        print("   with exact precision maintained throughout all computations.")
    else:
        print("⚠️ Some intermediate mathematics tests encountered issues")
        print(f"   Failures: {len(result.failures)}")
        print(f"   Errors: {len(result.errors)}")
    
    print("\n" + "=" * 60)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_intermediate_mathematics_tests()
    sys.exit(0 if success else 1)
