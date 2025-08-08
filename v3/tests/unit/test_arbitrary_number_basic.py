"""
Basic Unit Tests for ArbitraryNumber
===================================

Comprehensive test suite covering basic functionality of ArbitraryNumber class.
"""

import unittest
import sys
import os
from fractions import Fraction
from decimal import Decimal

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from arbitrary_numbers.core.arbitrary_number import ArbitraryNumber, FractionTerm


class TestFractionTerm(unittest.TestCase):
    """Test cases for FractionTerm class."""
    
    def test_fraction_term_creation(self):
        """Test basic FractionTerm creation."""
        term = FractionTerm(3, 4)
        self.assertEqual(term.numerator, 3)
        self.assertEqual(term.denominator, 4)
    
    def test_fraction_term_default_denominator(self):
        """Test FractionTerm with default denominator."""
        term = FractionTerm(5)
        self.assertEqual(term.numerator, 5)
        self.assertEqual(term.denominator, 1)
    
    def test_fraction_term_zero_denominator(self):
        """Test FractionTerm with zero denominator raises error."""
        with self.assertRaises(ValueError):
            FractionTerm(1, 0)
    
    def test_fraction_term_negative_denominator(self):
        """Test FractionTerm normalizes negative denominators."""
        term = FractionTerm(3, -4)
        self.assertEqual(term.numerator, -3)
        self.assertEqual(term.denominator, 4)
    
    def test_fraction_term_value_property(self):
        """Test FractionTerm value property returns correct Fraction."""
        term = FractionTerm(3, 4)
        self.assertEqual(term.value, Fraction(3, 4))
    
    def test_fraction_term_string_representation(self):
        """Test FractionTerm string representations."""
        term1 = FractionTerm(3, 4)
        self.assertEqual(str(term1), "3/4")
        
        term2 = FractionTerm(5, 1)
        self.assertEqual(str(term2), "5")
    
    def test_fraction_term_equality(self):
        """Test FractionTerm equality comparison."""
        term1 = FractionTerm(3, 4)
        term2 = FractionTerm(6, 8)  # Equivalent fraction
        term3 = FractionTerm(1, 2)
        
        self.assertEqual(term1, term2)
        self.assertNotEqual(term1, term3)
    
    def test_fraction_term_hash(self):
        """Test FractionTerm hashing."""
        term1 = FractionTerm(3, 4)
        term2 = FractionTerm(6, 8)  # Equivalent fraction
        
        self.assertEqual(hash(term1), hash(term2))


class TestArbitraryNumberCreation(unittest.TestCase):
    """Test cases for ArbitraryNumber creation methods."""
    
    def test_empty_arbitrary_number(self):
        """Test creating empty ArbitraryNumber."""
        num = ArbitraryNumber()
        self.assertEqual(len(num.terms), 0)
        self.assertTrue(num.is_zero())
    
    def test_from_int(self):
        """Test creating ArbitraryNumber from integer."""
        num = ArbitraryNumber.from_int(42)
        self.assertEqual(len(num.terms), 1)
        self.assertEqual(num.terms[0].numerator, 42)
        self.assertEqual(num.terms[0].denominator, 1)
        self.assertEqual(num.evaluate_exact(), Fraction(42))
    
    def test_from_fraction(self):
        """Test creating ArbitraryNumber from fraction."""
        num = ArbitraryNumber.from_fraction(3, 4)
        self.assertEqual(len(num.terms), 1)
        self.assertEqual(num.terms[0].numerator, 3)
        self.assertEqual(num.terms[0].denominator, 4)
        self.assertEqual(num.evaluate_exact(), Fraction(3, 4))
    
    def test_from_decimal_float(self):
        """Test creating ArbitraryNumber from float."""
        num = ArbitraryNumber.from_decimal(0.5)
        self.assertEqual(num.evaluate_exact(), Fraction(1, 2))
    
    def test_from_decimal_decimal(self):
        """Test creating ArbitraryNumber from Decimal."""
        num = ArbitraryNumber.from_decimal(Decimal('0.25'))
        self.assertEqual(num.evaluate_exact(), Fraction(1, 4))
    
    def test_zero_constant(self):
        """Test ArbitraryNumber.zero() class method."""
        zero = ArbitraryNumber.zero()
        self.assertTrue(zero.is_zero())
        self.assertEqual(zero.evaluate_exact(), Fraction(0))
    
    def test_one_constant(self):
        """Test ArbitraryNumber.one() class method."""
        one = ArbitraryNumber.one()
        self.assertTrue(one.is_one())
        self.assertEqual(one.evaluate_exact(), Fraction(1))
    
    def test_with_metadata(self):
        """Test creating ArbitraryNumber with metadata."""
        metadata = {'source': 'test', 'precision': 'exact'}
        num = ArbitraryNumber.from_int(5, metadata)
        self.assertEqual(num.metadata, metadata)


class TestArbitraryNumberBasicOperations(unittest.TestCase):
    """Test cases for basic arithmetic operations."""
    
    def test_addition_simple(self):
        """Test simple addition of ArbitraryNumbers."""
        num1 = ArbitraryNumber.from_fraction(1, 2)
        num2 = ArbitraryNumber.from_fraction(1, 3)
        result = num1 + num2
        
        self.assertEqual(len(result.terms), 2)
        self.assertEqual(result.evaluate_exact(), Fraction(5, 6))
    
    def test_addition_multiple_terms(self):
        """Test addition with multiple terms."""
        num1 = ArbitraryNumber([FractionTerm(1, 2), FractionTerm(1, 4)])
        num2 = ArbitraryNumber([FractionTerm(1, 3)])
        result = num1 + num2
        
        self.assertEqual(len(result.terms), 3)
        expected = Fraction(1, 2) + Fraction(1, 4) + Fraction(1, 3)
        self.assertEqual(result.evaluate_exact(), expected)
    
    def test_subtraction_simple(self):
        """Test simple subtraction of ArbitraryNumbers."""
        num1 = ArbitraryNumber.from_fraction(3, 4)
        num2 = ArbitraryNumber.from_fraction(1, 4)
        result = num1 - num2
        
        self.assertEqual(result.evaluate_exact(), Fraction(1, 2))
    
    def test_multiplication_by_integer(self):
        """Test multiplication by integer."""
        num = ArbitraryNumber.from_fraction(2, 3)
        result = num * 3
        
        self.assertEqual(result.evaluate_exact(), Fraction(2))
    
    def test_multiplication_by_fraction_term(self):
        """Test multiplication by FractionTerm."""
        num = ArbitraryNumber.from_fraction(2, 3)
        term = FractionTerm(3, 4)
        result = num * term
        
        self.assertEqual(result.evaluate_exact(), Fraction(1, 2))
    
    def test_multiplication_by_arbitrary_number(self):
        """Test multiplication by another ArbitraryNumber."""
        num1 = ArbitraryNumber.from_fraction(2, 3)
        num2 = ArbitraryNumber.from_fraction(3, 4)
        result = num1 * num2
        
        self.assertEqual(result.evaluate_exact(), Fraction(1, 2))
    
    def test_division_by_integer(self):
        """Test division by integer."""
        num = ArbitraryNumber.from_fraction(3, 4)
        result = num / 2
        
        self.assertEqual(result.evaluate_exact(), Fraction(3, 8))
    
    def test_division_by_fraction_term(self):
        """Test division by FractionTerm."""
        num = ArbitraryNumber.from_fraction(3, 4)
        term = FractionTerm(2, 3)
        result = num / term
        
        self.assertEqual(result.evaluate_exact(), Fraction(9, 8))
    
    def test_division_by_arbitrary_number(self):
        """Test division by another ArbitraryNumber."""
        num1 = ArbitraryNumber.from_fraction(3, 4)
        num2 = ArbitraryNumber.from_fraction(2, 3)
        result = num1 / num2
        
        self.assertEqual(result.evaluate_exact(), Fraction(9, 8))
    
    def test_negation(self):
        """Test negation of ArbitraryNumber."""
        num = ArbitraryNumber.from_fraction(3, 4)
        result = -num
        
        self.assertEqual(result.evaluate_exact(), Fraction(-3, 4))
    
    def test_absolute_value_positive(self):
        """Test absolute value of positive ArbitraryNumber."""
        num = ArbitraryNumber.from_fraction(3, 4)
        result = abs(num)
        
        self.assertEqual(result.evaluate_exact(), Fraction(3, 4))
    
    def test_absolute_value_negative(self):
        """Test absolute value of negative ArbitraryNumber."""
        num = ArbitraryNumber.from_fraction(-3, 4)
        result = abs(num)
        
        self.assertEqual(result.evaluate_exact(), Fraction(3, 4))


class TestArbitraryNumberPowerOperations(unittest.TestCase):
    """Test cases for power operations."""
    
    def test_power_zero(self):
        """Test raising to power of zero."""
        num = ArbitraryNumber.from_fraction(3, 4)
        result = num ** 0
        
        self.assertTrue(result.is_one())
    
    def test_power_one(self):
        """Test raising to power of one."""
        num = ArbitraryNumber.from_fraction(3, 4)
        result = num ** 1
        
        self.assertEqual(result.evaluate_exact(), Fraction(3, 4))
    
    def test_power_positive(self):
        """Test raising to positive power."""
        num = ArbitraryNumber.from_fraction(2, 3)
        result = num ** 3
        
        self.assertEqual(result.evaluate_exact(), Fraction(8, 27))
    
    def test_power_negative(self):
        """Test raising to negative power."""
        num = ArbitraryNumber.from_fraction(2, 3)
        result = num ** -2
        
        self.assertEqual(result.evaluate_exact(), Fraction(9, 4))
    
    def test_power_large(self):
        """Test raising to large power."""
        num = ArbitraryNumber.from_fraction(1, 2)
        result = num ** 10
        
        self.assertEqual(result.evaluate_exact(), Fraction(1, 1024))


class TestArbitraryNumberComparisons(unittest.TestCase):
    """Test cases for comparison operations."""
    
    def test_equality_same_value(self):
        """Test equality with same value."""
        num1 = ArbitraryNumber.from_fraction(3, 4)
        num2 = ArbitraryNumber.from_fraction(6, 8)
        
        self.assertEqual(num1, num2)
    
    def test_equality_different_value(self):
        """Test inequality with different values."""
        num1 = ArbitraryNumber.from_fraction(3, 4)
        num2 = ArbitraryNumber.from_fraction(1, 2)
        
        self.assertNotEqual(num1, num2)
    
    def test_less_than(self):
        """Test less than comparison."""
        num1 = ArbitraryNumber.from_fraction(1, 3)
        num2 = ArbitraryNumber.from_fraction(1, 2)
        
        self.assertTrue(num1 < num2)
        self.assertFalse(num2 < num1)
    
    def test_less_than_or_equal(self):
        """Test less than or equal comparison."""
        num1 = ArbitraryNumber.from_fraction(1, 3)
        num2 = ArbitraryNumber.from_fraction(1, 2)
        num3 = ArbitraryNumber.from_fraction(1, 3)
        
        self.assertTrue(num1 <= num2)
        self.assertTrue(num1 <= num3)
        self.assertFalse(num2 <= num1)
    
    def test_greater_than(self):
        """Test greater than comparison."""
        num1 = ArbitraryNumber.from_fraction(1, 2)
        num2 = ArbitraryNumber.from_fraction(1, 3)
        
        self.assertTrue(num1 > num2)
        self.assertFalse(num2 > num1)
    
    def test_greater_than_or_equal(self):
        """Test greater than or equal comparison."""
        num1 = ArbitraryNumber.from_fraction(1, 2)
        num2 = ArbitraryNumber.from_fraction(1, 3)
        num3 = ArbitraryNumber.from_fraction(1, 2)
        
        self.assertTrue(num1 >= num2)
        self.assertTrue(num1 >= num3)
        self.assertFalse(num2 >= num1)


class TestArbitraryNumberEvaluation(unittest.TestCase):
    """Test cases for evaluation methods."""
    
    def test_evaluate_exact_simple(self):
        """Test exact evaluation of simple fraction."""
        num = ArbitraryNumber.from_fraction(3, 4)
        result = num.evaluate_exact()
        
        self.assertEqual(result, Fraction(3, 4))
        self.assertIsInstance(result, Fraction)
    
    def test_evaluate_exact_multiple_terms(self):
        """Test exact evaluation with multiple terms."""
        num = ArbitraryNumber([
            FractionTerm(1, 2),
            FractionTerm(1, 3),
            FractionTerm(1, 6)
        ])
        result = num.evaluate_exact()
        
        self.assertEqual(result, Fraction(1))
    
    def test_evaluate_decimal_default_precision(self):
        """Test decimal evaluation with default precision."""
        num = ArbitraryNumber.from_fraction(1, 3)
        result = num.evaluate(precision=10)
        
        self.assertIsInstance(result, Decimal)
        self.assertAlmostEqual(float(result), 1/3, places=9)
    
    def test_evaluate_decimal_high_precision(self):
        """Test decimal evaluation with high precision."""
        num = ArbitraryNumber.from_fraction(1, 7)
        result = num.evaluate(precision=50)
        
        self.assertIsInstance(result, Decimal)
        # 1/7 has a repeating decimal, check first few digits
        result_str = str(result)
        self.assertTrue(result_str.startswith('0.142857'))
    
    def test_evaluate_caching(self):
        """Test that evaluation results are cached."""
        num = ArbitraryNumber.from_fraction(1, 3)
        
        # First evaluation
        result1 = num.evaluate_exact()
        # Second evaluation should use cache
        result2 = num.evaluate_exact()
        
        self.assertIs(result1, result2)  # Same object reference
    
    def test_cache_invalidation(self):
        """Test that cache is invalidated when terms change."""
        num = ArbitraryNumber.from_fraction(1, 2)
        
        # First evaluation
        result1 = num.evaluate_exact()
        
        # Modify terms
        num.add_term(FractionTerm(1, 4))
        
        # Second evaluation should be different
        result2 = num.evaluate_exact()
        
        self.assertNotEqual(result1, result2)
        self.assertEqual(result2, Fraction(3, 4))


if __name__ == '__main__':
    unittest.main()
