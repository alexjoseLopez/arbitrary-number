"""
Edge Case Unit Tests for ArbitraryNumber
=======================================

Comprehensive test suite covering edge cases, error conditions, and boundary scenarios.
"""

import unittest
import sys
import os
from fractions import Fraction
from decimal import Decimal
import math

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from arbitrary_numbers.core.arbitrary_number import ArbitraryNumber, FractionTerm


class TestArbitraryNumberEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""
    
    def test_zero_operations(self):
        """Test operations with zero."""
        zero = ArbitraryNumber.zero()
        num = ArbitraryNumber.from_fraction(3, 4)
        
        # Addition with zero
        self.assertEqual(zero + num, num)
        self.assertEqual(num + zero, num)
        
        # Subtraction with zero
        self.assertEqual(num - zero, num)
        self.assertEqual(zero - num, -num)
        
        # Multiplication with zero
        self.assertTrue((zero * num).is_zero())
        self.assertTrue((num * 0).is_zero())
    
    def test_one_operations(self):
        """Test operations with one."""
        one = ArbitraryNumber.one()
        num = ArbitraryNumber.from_fraction(3, 4)
        
        # Multiplication with one
        self.assertEqual(one * num, num)
        self.assertEqual(num * one, num)
        
        # Division by one
        self.assertEqual(num / one, num)
    
    def test_division_by_zero_integer(self):
        """Test division by zero integer raises error."""
        num = ArbitraryNumber.from_fraction(3, 4)
        
        with self.assertRaises(ValueError):
            num / 0
    
    def test_division_by_zero_fraction_term(self):
        """Test division by zero FractionTerm raises error."""
        num = ArbitraryNumber.from_fraction(3, 4)
        zero_term = FractionTerm(0, 1)
        
        with self.assertRaises(ValueError):
            num / zero_term
    
    def test_division_by_zero_arbitrary_number(self):
        """Test division by zero ArbitraryNumber raises error."""
        num = ArbitraryNumber.from_fraction(3, 4)
        zero = ArbitraryNumber.zero()
        
        with self.assertRaises(ValueError):
            num / zero
    
    def test_power_with_non_integer(self):
        """Test power operation with non-integer exponent raises error."""
        num = ArbitraryNumber.from_fraction(2, 3)
        
        with self.assertRaises(TypeError):
            num ** 2.5
        
        with self.assertRaises(TypeError):
            num ** Fraction(3, 2)
    
    def test_very_large_numbers(self):
        """Test operations with very large numbers."""
        large_num = ArbitraryNumber.from_int(10**100)
        small_num = ArbitraryNumber.from_fraction(1, 10**100)
        
        # Multiplication should work
        result = large_num * small_num
        self.assertTrue(result.is_one())
        
        # Addition should work
        result = large_num + ArbitraryNumber.from_int(1)
        expected = Fraction(10**100 + 1)
        self.assertEqual(result.evaluate_exact(), expected)
    
    def test_very_small_fractions(self):
        """Test operations with very small fractions."""
        tiny = ArbitraryNumber.from_fraction(1, 10**50)
        
        # Should not be zero
        self.assertFalse(tiny.is_zero())
        
        # Should be very close to zero when converted to float
        self.assertAlmostEqual(tiny.to_float(), 0.0, places=50)
    
    def test_repeating_decimals(self):
        """Test handling of repeating decimals."""
        # 1/3 = 0.333...
        third = ArbitraryNumber.from_fraction(1, 3)
        
        # Should maintain exact representation
        self.assertEqual(third.evaluate_exact(), Fraction(1, 3))
        
        # Decimal evaluation should be accurate to specified precision
        decimal_result = third.evaluate(precision=20)
        expected_str = "0.33333333333333333333"
        self.assertTrue(str(decimal_result).startswith("0.3333"))
    
    def test_negative_zero_handling(self):
        """Test handling of negative zero scenarios."""
        pos_zero = ArbitraryNumber([FractionTerm(0, 1)])
        neg_zero = ArbitraryNumber([FractionTerm(0, -1)])  # This should normalize
        
        self.assertTrue(pos_zero.is_zero())
        self.assertTrue(neg_zero.is_zero())
        self.assertEqual(pos_zero, neg_zero)
    
    def test_empty_terms_list(self):
        """Test ArbitraryNumber with empty terms list."""
        empty = ArbitraryNumber([])
        
        self.assertTrue(empty.is_zero())
        self.assertEqual(len(empty), 0)
        self.assertEqual(empty.evaluate_exact(), Fraction(0))
        self.assertFalse(bool(empty))
    
    def test_mixed_positive_negative_terms(self):
        """Test ArbitraryNumber with mixed positive and negative terms."""
        mixed = ArbitraryNumber([
            FractionTerm(5, 6),
            FractionTerm(-1, 2),
            FractionTerm(1, 3)
        ])
        
        # 5/6 - 1/2 + 1/3 = 5/6 - 3/6 + 2/6 = 4/6 = 2/3
        expected = Fraction(2, 3)
        self.assertEqual(mixed.evaluate_exact(), expected)
    
    def test_simplification_with_many_terms(self):
        """Test simplification with many terms."""
        many_terms = ArbitraryNumber([
            FractionTerm(1, 10) for _ in range(100)
        ])
        
        # Should equal 100 * (1/10) = 10
        self.assertEqual(many_terms.evaluate_exact(), Fraction(10))
        
        # Simplification should reduce to single term
        simplified = many_terms.simplify()
        self.assertEqual(len(simplified.terms), 1)
        self.assertEqual(simplified.evaluate_exact(), Fraction(10))
    
    def test_precision_loss_always_zero(self):
        """Test that precision loss is always zero for ArbitraryNumbers."""
        numbers = [
            ArbitraryNumber.from_fraction(1, 3),
            ArbitraryNumber.from_fraction(22, 7),  # Approximation of pi
            ArbitraryNumber.from_decimal(0.1),
            ArbitraryNumber.from_int(42)
        ]
        
        for num in numbers:
            self.assertEqual(num.get_precision_loss(), 0.0)
    
    def test_metadata_preservation(self):
        """Test that metadata is preserved through operations."""
        metadata1 = {'source': 'test1', 'id': 1}
        metadata2 = {'source': 'test2', 'id': 2}
        
        num1 = ArbitraryNumber.from_fraction(1, 2, metadata1)
        num2 = ArbitraryNumber.from_fraction(1, 3, metadata2)
        
        result = num1 + num2
        
        # Should contain metadata from both operands plus operation info
        self.assertIn('source', result.metadata)
        self.assertIn('operation', result.metadata)
        self.assertEqual(result.metadata['operation'], 'addition')
    
    def test_computation_trace(self):
        """Test computation trace functionality."""
        num1 = ArbitraryNumber.from_fraction(1, 2)
        num2 = ArbitraryNumber.from_fraction(1, 3)
        
        result = num1 + num2
        trace = result.get_computation_trace()
        
        self.assertIsInstance(trace, list)
        self.assertTrue(any('addition' in str(item) for item in trace))
    
    def test_string_representation_edge_cases(self):
        """Test string representation for edge cases."""
        # Empty ArbitraryNumber
        empty = ArbitraryNumber([])
        self.assertEqual(str(empty), "0")
        
        # Single positive term
        single_pos = ArbitraryNumber([FractionTerm(3, 4)])
        self.assertEqual(str(single_pos), "3/4")
        
        # Single negative term
        single_neg = ArbitraryNumber([FractionTerm(-3, 4)])
        self.assertEqual(str(single_neg), "-3/4")
        
        # Multiple terms with mixed signs
        mixed = ArbitraryNumber([
            FractionTerm(1, 2),
            FractionTerm(-1, 3),
            FractionTerm(1, 4)
        ])
        expected = "1/2 - 1/3 + 1/4"
        self.assertEqual(str(mixed), expected)
    
    def test_boolean_conversion_edge_cases(self):
        """Test boolean conversion for edge cases."""
        # Zero should be False
        zero = ArbitraryNumber.zero()
        self.assertFalse(bool(zero))
        
        # Empty terms should be False
        empty = ArbitraryNumber([])
        self.assertFalse(bool(empty))
        
        # Terms that sum to zero should be False
        sum_to_zero = ArbitraryNumber([
            FractionTerm(1, 2),
            FractionTerm(-1, 2)
        ])
        self.assertFalse(bool(sum_to_zero))
        
        # Non-zero should be True
        non_zero = ArbitraryNumber.from_fraction(1, 1000000)
        self.assertTrue(bool(non_zero))
    
    def test_memory_usage_calculation(self):
        """Test memory usage calculation."""
        # Empty ArbitraryNumber
        empty = ArbitraryNumber([])
        self.assertGreaterEqual(empty.memory_usage(), 0)
        
        # ArbitraryNumber with terms
        with_terms = ArbitraryNumber([
            FractionTerm(1, 2),
            FractionTerm(1, 3),
            FractionTerm(1, 4)
        ])
        self.assertGreater(with_terms.memory_usage(), empty.memory_usage())
        
        # With metadata
        with_metadata = ArbitraryNumber.from_fraction(1, 2, {'test': 'data'})
        self.assertGreater(with_metadata.memory_usage(), 
                          ArbitraryNumber.from_fraction(1, 2).memory_usage())
    
    def test_term_count(self):
        """Test term count functionality."""
        # Empty
        empty = ArbitraryNumber([])
        self.assertEqual(empty.term_count(), 0)
        
        # Single term
        single = ArbitraryNumber.from_fraction(1, 2)
        self.assertEqual(single.term_count(), 1)
        
        # Multiple terms
        multiple = ArbitraryNumber([
            FractionTerm(1, 2),
            FractionTerm(1, 3),
            FractionTerm(1, 4)
        ])
        self.assertEqual(multiple.term_count(), 3)
    
    def test_is_integer_edge_cases(self):
        """Test is_integer method for edge cases."""
        # Zero is an integer
        zero = ArbitraryNumber.zero()
        self.assertTrue(zero.is_integer())
        
        # Positive integer
        pos_int = ArbitraryNumber.from_int(42)
        self.assertTrue(pos_int.is_integer())
        
        # Negative integer
        neg_int = ArbitraryNumber.from_int(-42)
        self.assertTrue(neg_int.is_integer())
        
        # Fraction that equals integer
        frac_int = ArbitraryNumber.from_fraction(8, 4)  # = 2
        self.assertTrue(frac_int.is_integer())
        
        # Proper fraction
        proper_frac = ArbitraryNumber.from_fraction(3, 4)
        self.assertFalse(proper_frac.is_integer())
        
        # Multiple terms that sum to integer
        sum_to_int = ArbitraryNumber([
            FractionTerm(1, 2),
            FractionTerm(1, 2)
        ])  # = 1
        self.assertTrue(sum_to_int.is_integer())


class TestArbitraryNumberErrorHandling(unittest.TestCase):
    """Test error handling and invalid operations."""
    
    def test_invalid_multiplication_types(self):
        """Test multiplication with invalid types raises errors."""
        num = ArbitraryNumber.from_fraction(1, 2)
        
        with self.assertRaises(TypeError):
            num * "string"
        
        with self.assertRaises(TypeError):
            num * [1, 2, 3]
        
        with self.assertRaises(TypeError):
            num * {'key': 'value'}
    
    def test_invalid_division_types(self):
        """Test division with invalid types raises errors."""
        num = ArbitraryNumber.from_fraction(1, 2)
        
        with self.assertRaises(TypeError):
            num / "string"
        
        with self.assertRaises(TypeError):
            num / [1, 2, 3]
        
        with self.assertRaises(TypeError):
            num / {'key': 'value'}
    
    def test_comparison_with_invalid_types(self):
        """Test comparisons with invalid types."""
        num = ArbitraryNumber.from_fraction(1, 2)
        
        # Equality should return False for different types
        self.assertNotEqual(num, "1/2")
        self.assertNotEqual(num, 0.5)
        self.assertNotEqual(num, [1, 2])
        
        # Other comparisons should return NotImplemented
        with self.assertRaises(TypeError):
            num < "string"
        
        with self.assertRaises(TypeError):
            num <= [1, 2]
        
        with self.assertRaises(TypeError):
            num > {'key': 'value'}
        
        with self.assertRaises(TypeError):
            num >= 0.5
    
    def test_from_decimal_with_invalid_types(self):
        """Test from_decimal with invalid types."""
        with self.assertRaises((TypeError, ValueError)):
            ArbitraryNumber.from_decimal("not a number")
        
        with self.assertRaises((TypeError, ValueError)):
            ArbitraryNumber.from_decimal([1, 2, 3])
    
    def test_evaluate_with_invalid_precision(self):
        """Test evaluate with invalid precision values."""
        num = ArbitraryNumber.from_fraction(1, 3)
        
        # Negative precision should still work (Python handles it)
        result = num.evaluate(precision=-1)
        self.assertIsInstance(result, Decimal)
        
        # Zero precision should work
        result = num.evaluate(precision=0)
        self.assertIsInstance(result, Decimal)


class TestArbitraryNumberComplexScenarios(unittest.TestCase):
    """Test complex mathematical scenarios."""
    
    def test_fibonacci_sequence(self):
        """Test ArbitraryNumber with Fibonacci sequence calculations."""
        # Calculate first 10 Fibonacci numbers using ArbitraryNumbers
        fib = [ArbitraryNumber.zero(), ArbitraryNumber.one()]
        
        for i in range(2, 10):
            next_fib = fib[i-1] + fib[i-2]
            fib.append(next_fib)
        
        # Check known Fibonacci values
        expected = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
        for i, expected_val in enumerate(expected):
            self.assertEqual(fib[i].evaluate_exact(), Fraction(expected_val))
    
    def test_harmonic_series_partial_sums(self):
        """Test partial sums of harmonic series."""
        # Calculate 1 + 1/2 + 1/3 + ... + 1/n for n=10
        harmonic_sum = ArbitraryNumber.zero()
        
        for i in range(1, 11):
            term = ArbitraryNumber.from_fraction(1, i)
            harmonic_sum = harmonic_sum + term
        
        # Should be exact rational number
        result = harmonic_sum.evaluate_exact()
        self.assertIsInstance(result, Fraction)
        
        # Should be approximately 2.928968...
        self.assertAlmostEqual(float(result), 2.928968, places=5)
    
    def test_continued_fraction_approximation(self):
        """Test continued fraction approximation of pi."""
        # Use continued fraction [3; 7, 15, 1, 292, ...] for pi
        # Approximation: 3 + 1/(7 + 1/(15 + 1/1))
        
        # Start from innermost fraction
        inner = ArbitraryNumber.one()  # 1
        middle = ArbitraryNumber.from_int(15) + inner  # 15 + 1 = 16
        middle_recip = ArbitraryNumber.one() / middle  # 1/16
        
        outer = ArbitraryNumber.from_int(7) + middle_recip  # 7 + 1/16
        outer_recip = ArbitraryNumber.one() / outer  # 1/(7 + 1/16)
        
        pi_approx = ArbitraryNumber.from_int(3) + outer_recip
        
        # Should be close to pi
        pi_value = float(pi_approx.evaluate_exact())
        self.assertAlmostEqual(pi_value, math.pi, places=2)
    
    def test_nested_operations(self):
        """Test deeply nested arithmetic operations."""
        # ((1/2 + 1/3) * (2/3 - 1/4)) / (3/4 + 1/6)
        
        term1 = ArbitraryNumber.from_fraction(1, 2) + ArbitraryNumber.from_fraction(1, 3)
        term2 = ArbitraryNumber.from_fraction(2, 3) - ArbitraryNumber.from_fraction(1, 4)
        numerator = term1 * term2
        
        term3 = ArbitraryNumber.from_fraction(3, 4) + ArbitraryNumber.from_fraction(1, 6)
        
        result = numerator / term3
        
        # Calculate expected result manually
        # term1 = 1/2 + 1/3 = 5/6
        # term2 = 2/3 - 1/4 = 8/12 - 3/12 = 5/12
        # numerator = 5/6 * 5/12 = 25/72
        # term3 = 3/4 + 1/6 = 9/12 + 2/12 = 11/12
        # result = (25/72) / (11/12) = (25/72) * (12/11) = 300/792 = 25/66
        
        expected = Fraction(25, 66)
        self.assertEqual(result.evaluate_exact(), expected)
    
    def test_power_series_expansion(self):
        """Test power series expansion (e.g., e^x approximation)."""
        # Approximate e^(1/2) using Taylor series: 1 + x + x^2/2! + x^3/3! + ...
        x = ArbitraryNumber.from_fraction(1, 2)
        
        # Calculate first 5 terms
        series_sum = ArbitraryNumber.one()  # 1
        
        # x term
        series_sum = series_sum + x
        
        # x^2/2! term
        x_squared = x * x
        factorial_2 = ArbitraryNumber.from_int(2)
        series_sum = series_sum + (x_squared / factorial_2)
        
        # x^3/3! term
        x_cubed = x_squared * x
        factorial_3 = ArbitraryNumber.from_int(6)
        series_sum = series_sum + (x_cubed / factorial_3)
        
        # x^4/4! term
        x_fourth = x_cubed * x
        factorial_4 = ArbitraryNumber.from_int(24)
        series_sum = series_sum + (x_fourth / factorial_4)
        
        # Should approximate e^(1/2) ≈ 1.6487
        result_float = float(series_sum.evaluate_exact())
        expected = math.exp(0.5)
        self.assertAlmostEqual(result_float, expected, places=3)


if __name__ == '__main__':
    unittest.main()
