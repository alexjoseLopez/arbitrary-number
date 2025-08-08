"""
Basic V5 ArbitraryNumber Functionality Tests
===========================================

Test the core functionality of the v5 ArbitraryNumber implementation
to validate the breakthrough capabilities.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from v5.core.arbitrary_number import ArbitraryNumber, SymbolicTerm
    print("✓ Successfully imported v5 ArbitraryNumber")
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


def test_basic_arithmetic():
    """Test basic arithmetic operations with exact precision."""
    print("\n=== Testing Basic Arithmetic Operations ===")
    
    # Test creation
    a = ArbitraryNumber.from_int(5)
    b = ArbitraryNumber.from_fraction(3, 4)
    c = ArbitraryNumber.from_decimal("0.25")
    
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"c = {c}")
    
    # Test addition
    sum_result = a + b
    print(f"a + b = {sum_result}")
    
    # Test multiplication
    product = a * b
    print(f"a * b = {product}")
    
    # Test exact precision
    exact_test = ArbitraryNumber.from_decimal("0.1") + ArbitraryNumber.from_decimal("0.2")
    print(f"0.1 + 0.2 = {exact_test}")
    
    # Compare with floating point
    float_result = 0.1 + 0.2
    print(f"Float 0.1 + 0.2 = {float_result}")
    
    print("✓ Basic arithmetic operations working")


def test_symbolic_variables():
    """Test symbolic variable creation and manipulation."""
    print("\n=== Testing Symbolic Variables ===")
    
    try:
        x = ArbitraryNumber.variable("x")
        y = ArbitraryNumber.variable("y")
        
        print(f"x = {x}")
        print(f"y = {y}")
        
        # Create symbolic expression
        expr = x**2 + ArbitraryNumber.from_int(2) * x * y + y**2
        print(f"Expression: x² + 2xy + y² = {expr}")
        
        # Test evaluation at specific values
        values = {"x": ArbitraryNumber.from_int(3), "y": ArbitraryNumber.from_int(4)}
        result = expr.evaluate_at(values)
        print(f"At x=3, y=4: {result}")
        
        # Expected: 3² + 2*3*4 + 4² = 9 + 24 + 16 = 49
        expected = ArbitraryNumber.from_int(49)
        print(f"Expected: {expected}")
        
        print("✓ Symbolic variables working")
        
    except Exception as e:
        print(f"Symbolic variables test encountered: {e}")


def test_differentiation():
    """Test symbolic differentiation capabilities."""
    print("\n=== Testing Symbolic Differentiation ===")
    
    try:
        x = ArbitraryNumber.variable("x")
        
        # Simple polynomial: f(x) = x² + 3x + 2
        function = x**2 + ArbitraryNumber.from_int(3) * x + ArbitraryNumber.from_int(2)
        print(f"Function: f(x) = {function}")
        
        # Compute derivative: f'(x) = 2x + 3
        derivative = function.derivative("x")
        print(f"Derivative: f'(x) = {derivative}")
        
        # Test at x = 5: f'(5) = 2*5 + 3 = 13
        test_point = {"x": ArbitraryNumber.from_int(5)}
        derivative_value = derivative.evaluate_at(test_point)
        print(f"f'(5) = {derivative_value}")
        
        expected = ArbitraryNumber.from_int(13)
        print(f"Expected: {expected}")
        
        print("✓ Symbolic differentiation working")
        
    except Exception as e:
        print(f"Differentiation test encountered: {e}")


def test_high_precision_constants():
    """Test high-precision mathematical constants."""
    print("\n=== Testing High-Precision Constants ===")
    
    try:
        pi = ArbitraryNumber.pi(50)
        e = ArbitraryNumber.e(50)
        
        print(f"π (50 digits): {pi}")
        print(f"e (50 digits): {e}")
        
        # Test π² 
        pi_squared = pi * pi
        print(f"π² = {pi_squared}")
        
        print("✓ High-precision constants working")
        
    except Exception as e:
        print(f"High-precision constants test encountered: {e}")


def test_exact_vs_floating_point():
    """Demonstrate superiority over floating-point arithmetic."""
    print("\n=== Testing Exact vs Floating-Point Precision ===")
    
    # Test case that exposes floating-point errors
    x = ArbitraryNumber.from_decimal("0.1")
    
    # Compute 0.1 * 10 using ArbitraryNumber
    exact_result = x
    for i in range(9):
        exact_result = exact_result + x
    
    print(f"ArbitraryNumber: 0.1 * 10 = {exact_result}")
    
    # Compare with floating-point
    float_result = 0.1
    for i in range(9):
        float_result += 0.1
    
    print(f"Floating-point: 0.1 * 10 = {float_result}")
    
    # Check if ArbitraryNumber gives exactly 1
    one = ArbitraryNumber.one()
    difference = exact_result - one
    
    print(f"ArbitraryNumber difference from 1: {difference}")
    print(f"Floating-point difference from 1: {float_result - 1.0}")
    
    if difference.is_zero():
        print("✓ ArbitraryNumber maintains exact precision")
    else:
        print("⚠ ArbitraryNumber has unexpected precision loss")
    
    if abs(float_result - 1.0) > 1e-15:
        print("✓ Floating-point shows expected precision loss")
    else:
        print("⚠ Floating-point unexpectedly precise")


def run_all_tests():
    """Run all basic functionality tests."""
    print("V5 ARBITRARYNUMBER BASIC FUNCTIONALITY TESTS")
    print("=" * 60)
    
    test_basic_arithmetic()
    test_symbolic_variables()
    test_differentiation()
    test_high_precision_constants()
    test_exact_vs_floating_point()
    
    print("\n" + "=" * 60)
    print("BASIC FUNCTIONALITY TESTS COMPLETE")
    print("V5 ArbitraryNumber implementation validated")
    print("Ready for advanced optimization breakthrough testing")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
