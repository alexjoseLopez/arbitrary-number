"""
Basic Functionality Tests
========================

Test suite for core Arbitrary Numbers functionality.
"""

import unittest
import sys
import os

# Add the parent directory to the path so we can import arbitrary_numbers
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arbitrary_numbers.core.rational_list import RationalListNumber, FractionTerm
from arbitrary_numbers.core.equation_nodes import (
    ConstantNode, BinaryOpNode, UnaryOpNode, VariableNode, ExpressionBuilder
)
from arbitrary_numbers.core.evaluator import EquationEvaluator
from fractions import Fraction


class TestRationalListNumber(unittest.TestCase):
    """Test RationalListNumber functionality."""
    
    def test_fraction_term_creation(self):
        """Test FractionTerm creation and properties."""
        term = FractionTerm(3, 4)
        self.assertEqual(term.numerator, 3)
        self.assertEqual(term.denominator, 4)
        self.assertEqual(term.value, Fraction(3, 4))
        self.assertEqual(str(term), "3/4")
    
    def test_fraction_term_normalization(self):
        """Test FractionTerm handles negative denominators."""
        term = FractionTerm(3, -4)
        self.assertEqual(term.numerator, -3)
        self.assertEqual(term.denominator, 4)
    
    def test_rational_list_creation(self):
        """Test RationalListNumber creation."""
        terms = [FractionTerm(1, 2), FractionTerm(3, 4)]
        rational = RationalListNumber(terms)
        self.assertEqual(len(rational), 2)
        self.assertTrue(bool(rational))
    
    def test_rational_list_addition(self):
        """Test addition of RationalListNumbers."""
        r1 = RationalListNumber([FractionTerm(1, 2)])
        r2 = RationalListNumber([FractionTerm(1, 3)])
        result = r1 + r2
        
        self.assertEqual(len(result.terms), 2)
        # Should be 1/2 + 1/3 = 5/6
        exact_result = result.evaluate_exact()
        self.assertEqual(exact_result, Fraction(5, 6))
    
    def test_rational_list_subtraction(self):
        """Test subtraction of RationalListNumbers."""
        r1 = RationalListNumber([FractionTerm(3, 4)])
        r2 = RationalListNumber([FractionTerm(1, 4)])
        result = r1 - r2
        
        # Should be 3/4 - 1/4 = 1/2
        exact_result = result.evaluate_exact()
        self.assertEqual(exact_result, Fraction(1, 2))
    
    def test_rational_list_multiplication(self):
        """Test multiplication of RationalListNumbers."""
        r1 = RationalListNumber([FractionTerm(2, 3)])
        r2 = RationalListNumber([FractionTerm(3, 4)])
        result = r1 * r2
        
        # Should be (2/3) * (3/4) = 6/12 = 1/2
        exact_result = result.evaluate_exact()
        self.assertEqual(exact_result, Fraction(1, 2))
    
    def test_rational_list_division(self):
        """Test division of RationalListNumbers."""
        r1 = RationalListNumber([FractionTerm(3, 4)])
        result = r1 / FractionTerm(2, 3)
        
        # Should be (3/4) / (2/3) = (3/4) * (3/2) = 9/8
        exact_result = result.evaluate_exact()
        self.assertEqual(exact_result, Fraction(9, 8))
    
    def test_rational_list_simplification(self):
        """Test simplification of RationalListNumbers."""
        terms = [FractionTerm(1, 4), FractionTerm(1, 4), FractionTerm(1, 2)]
        rational = RationalListNumber(terms)
        simplified = rational.simplify()
        
        # Should be 1/4 + 1/4 + 1/2 = 1
        self.assertEqual(len(simplified.terms), 1)
        self.assertEqual(simplified.evaluate_exact(), Fraction(1, 1))
    
    def test_class_methods(self):
        """Test class method constructors."""
        r1 = RationalListNumber.from_int(5)
        self.assertEqual(r1.evaluate_exact(), Fraction(5, 1))
        
        r2 = RationalListNumber.from_fraction(3, 7)
        self.assertEqual(r2.evaluate_exact(), Fraction(3, 7))
        
        r3 = RationalListNumber.from_decimal(0.25)
        self.assertEqual(r3.evaluate_exact(), Fraction(1, 4))


class TestEquationNodes(unittest.TestCase):
    """Test equation node functionality."""
    
    def test_constant_node(self):
        """Test ConstantNode creation and evaluation."""
        rational = RationalListNumber([FractionTerm(3, 4)])
        node = ConstantNode(rational)
        
        result = node.evaluate()
        self.assertEqual(result.evaluate_exact(), Fraction(3, 4))
        self.assertEqual(node.to_string(), "3/4")
        self.assertEqual(node.complexity(), 1)
    
    def test_variable_node(self):
        """Test VariableNode creation and evaluation."""
        node = VariableNode("x")
        
        variables = {"x": RationalListNumber.from_int(5)}
        result = node.evaluate(variables)
        self.assertEqual(result.evaluate_exact(), Fraction(5, 1))
        
        # Test missing variable
        with self.assertRaises(ValueError):
            node.evaluate({})
    
    def test_binary_op_addition(self):
        """Test BinaryOpNode addition."""
        left = ConstantNode.from_fraction(1, 2)
        right = ConstantNode.from_fraction(1, 3)
        node = BinaryOpNode(left, right, 'add')
        
        result = node.evaluate()
        self.assertEqual(result.evaluate_exact(), Fraction(5, 6))
        self.assertEqual(node.to_string(), "1/2 + 1/3")
    
    def test_binary_op_multiplication(self):
        """Test BinaryOpNode multiplication."""
        left = ConstantNode.from_fraction(2, 3)
        right = ConstantNode.from_fraction(3, 4)
        node = BinaryOpNode(left, right, 'mul')
        
        result = node.evaluate()
        self.assertEqual(result.evaluate_exact(), Fraction(1, 2))
    
    def test_binary_op_power(self):
        """Test BinaryOpNode power operation."""
        base = ConstantNode.from_fraction(2, 1)
        exponent = ConstantNode.from_int(3)
        node = BinaryOpNode(base, exponent, 'pow')
        
        result = node.evaluate()
        self.assertEqual(result.evaluate_exact(), Fraction(8, 1))
    
    def test_unary_op_negation(self):
        """Test UnaryOpNode negation."""
        operand = ConstantNode.from_fraction(3, 4)
        node = UnaryOpNode(operand, 'neg')
        
        result = node.evaluate()
        self.assertEqual(result.evaluate_exact(), Fraction(-3, 4))
        self.assertEqual(node.to_string(), "-3/4")
    
    def test_unary_op_absolute(self):
        """Test UnaryOpNode absolute value."""
        operand = ConstantNode.from_fraction(-5, 2)
        node = UnaryOpNode(operand, 'abs')
        
        result = node.evaluate()
        self.assertEqual(result.evaluate_exact(), Fraction(5, 2))
    
    def test_expression_builder(self):
        """Test ExpressionBuilder helper methods."""
        # Build expression: (1/2 + 1/3) * 2
        half = ExpressionBuilder.constant(RationalListNumber.from_fraction(1, 2))
        third = ExpressionBuilder.constant(RationalListNumber.from_fraction(1, 3))
        two = ExpressionBuilder.constant(2)
        
        sum_node = ExpressionBuilder.add(half, third)
        result_node = ExpressionBuilder.multiply(sum_node, two)
        
        result = result_node.evaluate()
        # (1/2 + 1/3) * 2 = (5/6) * 2 = 10/6 = 5/3
        self.assertEqual(result.evaluate_exact(), Fraction(5, 3))


class TestEquationEvaluator(unittest.TestCase):
    """Test EquationEvaluator functionality."""
    
    def setUp(self):
        self.evaluator = EquationEvaluator()
    
    def test_basic_evaluation(self):
        """Test basic equation evaluation."""
        node = ConstantNode.from_fraction(3, 4)
        result = self.evaluator.evaluate(node)
        self.assertEqual(result.evaluate_exact(), Fraction(3, 4))
    
    def test_caching(self):
        """Test evaluation caching."""
        node = ConstantNode.from_fraction(7, 8)
        
        # First evaluation
        result1 = self.evaluator.evaluate(node)
        stats1 = self.evaluator.get_stats()
        
        # Second evaluation (should hit cache)
        result2 = self.evaluator.evaluate(node)
        stats2 = self.evaluator.get_stats()
        
        self.assertEqual(result1.evaluate_exact(), result2.evaluate_exact())
        self.assertGreater(stats2['cache_hits'], stats1['cache_hits'])
    
    def test_optimization(self):
        """Test equation optimization."""
        # Create expression: x + 0
        x = VariableNode("x")
        zero = ConstantNode.from_int(0)
        node = BinaryOpNode(x, zero, 'add')
        
        variables = {"x": RationalListNumber.from_int(5)}
        result = self.evaluator.evaluate(node, variables)
        
        # Should optimize to just x
        self.assertEqual(result.evaluate_exact(), Fraction(5, 1))
    
    def test_batch_evaluation(self):
        """Test batch evaluation."""
        nodes = [
            ConstantNode.from_int(1),
            ConstantNode.from_int(2),
            ConstantNode.from_int(3)
        ]
        
        results = self.evaluator.evaluate_batch(nodes)
        
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0].evaluate_exact(), Fraction(1, 1))
        self.assertEqual(results[1].evaluate_exact(), Fraction(2, 1))
        self.assertEqual(results[2].evaluate_exact(), Fraction(3, 1))
    
    def test_complex_expression(self):
        """Test evaluation of complex expression."""
        # Build expression: (1/2 + 1/3) * (2/3 - 1/4)
        builder = ExpressionBuilder
        
        left_sum = builder.add(
            builder.constant(RationalListNumber.from_fraction(1, 2)),
            builder.constant(RationalListNumber.from_fraction(1, 3))
        )
        
        right_diff = builder.subtract(
            builder.constant(RationalListNumber.from_fraction(2, 3)),
            builder.constant(RationalListNumber.from_fraction(1, 4))
        )
        
        product = builder.multiply(left_sum, right_diff)
        
        result = self.evaluator.evaluate(product)
        
        # (1/2 + 1/3) = 5/6
        # (2/3 - 1/4) = 8/12 - 3/12 = 5/12
        # (5/6) * (5/12) = 25/72
        expected = Fraction(25, 72)
        self.assertEqual(result.evaluate_exact(), expected)


if __name__ == '__main__':
    unittest.main()
