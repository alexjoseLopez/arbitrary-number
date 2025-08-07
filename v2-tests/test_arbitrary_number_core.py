"""
Comprehensive unit tests for ArbitraryNumber core functionality.
Test-first approach implementation with extensive edge cases.
"""

import unittest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from v2.core.arbitrary_number import ArbitraryNumber


class TestArbitraryNumberCore(unittest.TestCase):
    """Core functionality tests for ArbitraryNumber."""
    
    def test_creation_from_integer(self):
        """Test creating ArbitraryNumber from integer."""
        num = ArbitraryNumber(42)
        self.assertEqual(str(num), "42")
        
    def test_creation_from_float(self):
        """Test creating ArbitraryNumber from float."""
        num = ArbitraryNumber(3.14159)
        self.assertTrue("3.14159" in str(num))
        
    def test_creation_from_string(self):
        """Test creating ArbitraryNumber from string."""
        num = ArbitraryNumber("123.456789")
        self.assertEqual(str(num), "123.456789")
        
    def test_creation_from_fraction(self):
        """Test creating ArbitraryNumber from fraction string."""
        num = ArbitraryNumber("22/7")
        self.assertTrue(num.is_rational())
        
    def test_zero_creation(self):
        """Test creating zero."""
        num = ArbitraryNumber(0)
        self.assertEqual(str(num), "0")
        self.assertTrue(num.is_zero())
        
    def test_negative_creation(self):
        """Test creating negative numbers."""
        num = ArbitraryNumber(-42)
        self.assertEqual(str(num), "-42")
        self.assertTrue(num.is_negative())
        
    def test_addition_integers(self):
        """Test addition of integers."""
        a = ArbitraryNumber(123)
        b = ArbitraryNumber(456)
        result = a + b
        self.assertEqual(str(result), "579")
        
    def test_addition_fractions(self):
        """Test addition of fractions."""
        a = ArbitraryNumber("1/3")
        b = ArbitraryNumber("1/6")
        result = a + b
        self.assertEqual(str(result), "1/2")
        
    def test_subtraction_integers(self):
        """Test subtraction of integers."""
        a = ArbitraryNumber(100)
        b = ArbitraryNumber(42)
        result = a - b
        self.assertEqual(str(result), "58")
        
    def test_multiplication_integers(self):
        """Test multiplication of integers."""
        a = ArbitraryNumber(12)
        b = ArbitraryNumber(13)
        result = a * b
        self.assertEqual(str(result), "156")
        
    def test_division_exact(self):
        """Test exact division."""
        a = ArbitraryNumber(22)
        b = ArbitraryNumber(7)
        result = a / b
        self.assertEqual(str(result), "22/7")
        
    def test_power_integer(self):
        """Test integer power."""
        a = ArbitraryNumber(2)
        result = a ** 10
        self.assertEqual(str(result), "1024")
        
    def test_comparison_equal(self):
        """Test equality comparison."""
        a = ArbitraryNumber(42)
        b = ArbitraryNumber(42)
        self.assertTrue(a == b)
        
    def test_comparison_less_than(self):
        """Test less than comparison."""
        a = ArbitraryNumber(41)
        b = ArbitraryNumber(42)
        self.assertTrue(a < b)
        
    def test_comparison_greater_than(self):
        """Test greater than comparison."""
        a = ArbitraryNumber(43)
        b = ArbitraryNumber(42)
        self.assertTrue(a > b)
        
    def test_precision_preservation(self):
        """Test that precision is preserved in calculations."""
        # This would fail with floating point
        a = ArbitraryNumber("0.1")
        b = ArbitraryNumber("0.2")
        result = a + b
        self.assertEqual(str(result), "0.3")
        
    def test_large_number_handling(self):
        """Test handling of very large numbers."""
        large_num = ArbitraryNumber("123456789012345678901234567890")
        result = large_num * ArbitraryNumber(2)
        self.assertEqual(str(result), "246913578024691357802469135780")
        
    def test_small_fraction_handling(self):
        """Test handling of very small fractions."""
        small_frac = ArbitraryNumber("1/1000000000000000000")
        result = small_frac * ArbitraryNumber(2)
        self.assertEqual(str(result), "1/500000000000000000")
        
    def test_irrational_approximation(self):
        """Test approximation of irrational numbers."""
        pi_approx = ArbitraryNumber.pi(precision=50)
        self.assertTrue("3.14159" in str(pi_approx))
        
    def test_square_root_exact(self):
        """Test exact square root where possible."""
        num = ArbitraryNumber(16)
        result = num.sqrt()
        self.assertEqual(str(result), "4")
        
    def test_factorial(self):
        """Test factorial calculation."""
        num = ArbitraryNumber(5)
        result = num.factorial()
        self.assertEqual(str(result), "120")
        
    def test_gcd_calculation(self):
        """Test greatest common divisor."""
        a = ArbitraryNumber(48)
        b = ArbitraryNumber(18)
        result = ArbitraryNumber.gcd(a, b)
        self.assertEqual(str(result), "6")
        
    def test_lcm_calculation(self):
        """Test least common multiple."""
        a = ArbitraryNumber(12)
        b = ArbitraryNumber(18)
        result = ArbitraryNumber.lcm(a, b)
        self.assertEqual(str(result), "36")
        
    def test_modular_arithmetic(self):
        """Test modular arithmetic operations."""
        a = ArbitraryNumber(17)
        b = ArbitraryNumber(5)
        result = a % b
        self.assertEqual(str(result), "2")
        
    def test_continued_fraction_representation(self):
        """Test continued fraction representation."""
        num = ArbitraryNumber("22/7")
        cf = num.to_continued_fraction()
        self.assertEqual(cf, [3, 7])
        
    def test_decimal_expansion(self):
        """Test decimal expansion with specified precision."""
        num = ArbitraryNumber("1/3")
        decimal = num.to_decimal(precision=10)
        self.assertEqual(decimal, "0.3333333333")
        
    def test_scientific_notation(self):
        """Test scientific notation representation."""
        num = ArbitraryNumber("123456789")
        sci_notation = num.to_scientific_notation()
        self.assertEqual(sci_notation, "1.23456789e+8")
        
    def test_hash_consistency(self):
        """Test that equal numbers have equal hashes."""
        a = ArbitraryNumber(42)
        b = ArbitraryNumber(42)
        self.assertEqual(hash(a), hash(b))
        
    def test_copy_and_deepcopy(self):
        """Test copying operations."""
        original = ArbitraryNumber("22/7")
        copied = original.copy()
        self.assertEqual(original, copied)
        self.assertIsNot(original, copied)
        
    def test_serialization(self):
        """Test serialization and deserialization."""
        num = ArbitraryNumber("355/113")
        serialized = num.serialize()
        deserialized = ArbitraryNumber.deserialize(serialized)
        self.assertEqual(num, deserialized)


class TestArbitraryNumberEdgeCases(unittest.TestCase):
    """Edge case tests for ArbitraryNumber."""
    
    def test_division_by_zero(self):
        """Test division by zero raises appropriate error."""
        a = ArbitraryNumber(42)
        b = ArbitraryNumber(0)
        with self.assertRaises(ZeroDivisionError):
            result = a / b
            
    def test_zero_power_zero(self):
        """Test 0^0 case."""
        a = ArbitraryNumber(0)
        with self.assertRaises(ValueError):
            result = a ** 0
            
    def test_negative_square_root(self):
        """Test square root of negative number."""
        a = ArbitraryNumber(-4)
        with self.assertRaises(ValueError):
            result = a.sqrt()
            
    def test_factorial_negative(self):
        """Test factorial of negative number."""
        a = ArbitraryNumber(-5)
        with self.assertRaises(ValueError):
            result = a.factorial()
            
    def test_factorial_non_integer(self):
        """Test factorial of non-integer."""
        a = ArbitraryNumber("3.5")
        with self.assertRaises(ValueError):
            result = a.factorial()
            
    def test_extremely_large_numbers(self):
        """Test operations with extremely large numbers."""
        large1 = ArbitraryNumber("9" * 1000)
        large2 = ArbitraryNumber("8" * 1000)
        result = large1 + large2
        self.assertTrue(len(str(result)) >= 1000)
        
    def test_extremely_small_fractions(self):
        """Test operations with extremely small fractions."""
        small = ArbitraryNumber("1/" + "9" * 1000)
        result = small * ArbitraryNumber(2)
        self.assertTrue("2/" in str(result))
        
    def test_mixed_operations_chain(self):
        """Test chained operations of different types."""
        result = (ArbitraryNumber(2) + ArbitraryNumber("1/3")) * ArbitraryNumber(3) - ArbitraryNumber("1/2")
        # Should be (2 + 1/3) * 3 - 1/2 = 7 - 1/2 = 13/2
        self.assertEqual(str(result), "13/2")
        
    def test_precision_stress_test(self):
        """Stress test precision with many operations."""
        result = ArbitraryNumber("0.1")
        for i in range(100):
            result = result + ArbitraryNumber("0.01")
        # Should be exactly 1.1, not 1.0999999999999999
        self.assertEqual(str(result), "1.1")


if __name__ == '__main__':
    unittest.main()
