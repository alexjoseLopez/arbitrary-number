"""
Equation Node System
===================

Abstract syntax tree representation for symbolic mathematical expressions.
Supports deferred evaluation and symbolic manipulation.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
import math

from .rational_list import RationalListNumber, FractionTerm


class EquationNode(ABC):
    """
    Abstract base class for all equation nodes.
    Represents a node in the abstract syntax tree of a mathematical expression.
    """
    
    @abstractmethod
    def evaluate(self, variables: Optional[Dict[str, Any]] = None) -> RationalListNumber:
        """Evaluate this node to a RationalListNumber."""
        pass
    
    @abstractmethod
    def to_string(self) -> str:
        """Convert to human-readable string representation."""
        pass
    
    @abstractmethod
    def complexity(self) -> int:
        """Return complexity measure for optimization purposes."""
        pass
    
    def __str__(self) -> str:
        return self.to_string()


@dataclass
class ConstantNode(EquationNode):
    """
    Represents a constant value in the equation tree.
    """
    value: RationalListNumber
    
    def evaluate(self, variables: Optional[Dict[str, Any]] = None) -> RationalListNumber:
        return self.value
    
    def to_string(self) -> str:
        return str(self.value)
    
    def complexity(self) -> int:
        return 1
    
    @classmethod
    def from_int(cls, value: int) -> 'ConstantNode':
        """Create constant node from integer."""
        return cls(RationalListNumber.from_int(value))
    
    @classmethod
    def from_fraction(cls, numerator: int, denominator: int = 1) -> 'ConstantNode':
        """Create constant node from fraction."""
        return cls(RationalListNumber.from_fraction(numerator, denominator))


@dataclass
class VariableNode(EquationNode):
    """
    Represents a variable in the equation tree.
    """
    name: str
    
    def evaluate(self, variables: Optional[Dict[str, Any]] = None) -> RationalListNumber:
        if variables is None or self.name not in variables:
            raise ValueError(f"Variable '{self.name}' not found in evaluation context")
        
        value = variables[self.name]
        if isinstance(value, RationalListNumber):
            return value
        elif isinstance(value, (int, float)):
            return RationalListNumber.from_decimal(value)
        else:
            raise TypeError(f"Cannot convert {type(value)} to RationalListNumber")
    
    def to_string(self) -> str:
        return self.name
    
    def complexity(self) -> int:
        return 1


@dataclass
class BinaryOpNode(EquationNode):
    """
    Represents a binary operation (e.g., +, -, *, /, ^) in the equation tree.
    """
    left: EquationNode
    right: EquationNode
    operation: str
    
    SUPPORTED_OPERATIONS = {
        'add': '+',
        'sub': '-', 
        'mul': '*',
        'div': '/',
        'pow': '^'
    }
    
    def __post_init__(self):
        if self.operation not in self.SUPPORTED_OPERATIONS:
            raise ValueError(f"Unsupported operation: {self.operation}")
    
    def evaluate(self, variables: Optional[Dict[str, Any]] = None) -> RationalListNumber:
        left_val = self.left.evaluate(variables)
        right_val = self.right.evaluate(variables)
        
        if self.operation == 'add':
            return left_val + right_val
        
        elif self.operation == 'sub':
            return left_val - right_val
        
        elif self.operation == 'mul':
            return left_val * right_val
        
        elif self.operation == 'div':
            if len(right_val.terms) == 1:
                term = right_val.terms[0]
                return left_val / FractionTerm(term.numerator, term.denominator)
            else:
                simplified_right = right_val.simplify()
                term = simplified_right.terms[0]
                return left_val / FractionTerm(term.numerator, term.denominator)
        
        elif self.operation == 'pow':
            if len(right_val.terms) == 1 and right_val.terms[0].denominator == 1:
                exponent = right_val.terms[0].numerator
                if exponent >= 0:
                    return self._power_positive_integer(left_val, exponent)
                else:
                    raise NotImplementedError("Negative exponents not yet supported")
            else:
                raise NotImplementedError("Non-integer exponents not yet supported")
        
        else:
            raise ValueError(f"Unknown operation: {self.operation}")
    
    def _power_positive_integer(self, base: RationalListNumber, exponent: int) -> RationalListNumber:
        """Compute base^exponent for positive integer exponent."""
        if exponent == 0:
            return RationalListNumber.from_int(1)
        elif exponent == 1:
            return base
        else:
            result = base
            for _ in range(exponent - 1):
                result = result * base
            return result
    
    def to_string(self) -> str:
        op_symbol = self.SUPPORTED_OPERATIONS[self.operation]
        left_str = self.left.to_string()
        right_str = self.right.to_string()
        
        if isinstance(self.left, BinaryOpNode):
            left_str = f"({left_str})"
        if isinstance(self.right, BinaryOpNode):
            right_str = f"({right_str})"
        
        return f"{left_str} {op_symbol} {right_str}"
    
    def complexity(self) -> int:
        return 1 + self.left.complexity() + self.right.complexity()


@dataclass
class UnaryOpNode(EquationNode):
    """
    Represents a unary operation (e.g., -, sqrt, sin, cos) in the equation tree.
    """
    operand: EquationNode
    operation: str
    
    SUPPORTED_OPERATIONS = {
        'neg': '-',
        'sqrt': 'sqrt',
        'abs': 'abs'
    }
    
    def __post_init__(self):
        if self.operation not in self.SUPPORTED_OPERATIONS:
            raise ValueError(f"Unsupported unary operation: {self.operation}")
    
    def evaluate(self, variables: Optional[Dict[str, Any]] = None) -> RationalListNumber:
        operand_val = self.operand.evaluate(variables)
        
        if self.operation == 'neg':
            return -operand_val
        
        elif self.operation == 'abs':
            exact_val = operand_val.evaluate_exact()
            if exact_val >= 0:
                return operand_val
            else:
                return -operand_val
        
        elif self.operation == 'sqrt':
            exact_val = operand_val.evaluate_exact()
            if exact_val < 0:
                raise ValueError("Cannot take square root of negative number")
            
            sqrt_val = float(exact_val) ** 0.5
            return RationalListNumber.from_decimal(sqrt_val)
        
        else:
            raise ValueError(f"Unknown unary operation: {self.operation}")
    
    def to_string(self) -> str:
        op_symbol = self.SUPPORTED_OPERATIONS[self.operation]
        operand_str = self.operand.to_string()
        
        if self.operation == 'neg':
            if isinstance(self.operand, BinaryOpNode):
                return f"-({operand_str})"
            else:
                return f"-{operand_str}"
        else:
            return f"{op_symbol}({operand_str})"
    
    def complexity(self) -> int:
        return 1 + self.operand.complexity()


class ExpressionBuilder:
    """
    Helper class for building equation trees with operator overloading.
    """
    
    @staticmethod
    def constant(value: Union[int, float, RationalListNumber]) -> ConstantNode:
        """Create a constant node."""
        if isinstance(value, RationalListNumber):
            return ConstantNode(value)
        elif isinstance(value, int):
            return ConstantNode.from_int(value)
        else:
            return ConstantNode(RationalListNumber.from_decimal(value))
    
    @staticmethod
    def variable(name: str) -> VariableNode:
        """Create a variable node."""
        return VariableNode(name)
    
    @staticmethod
    def add(left: EquationNode, right: EquationNode) -> BinaryOpNode:
        """Create addition node."""
        return BinaryOpNode(left, right, 'add')
    
    @staticmethod
    def subtract(left: EquationNode, right: EquationNode) -> BinaryOpNode:
        """Create subtraction node."""
        return BinaryOpNode(left, right, 'sub')
    
    @staticmethod
    def multiply(left: EquationNode, right: EquationNode) -> BinaryOpNode:
        """Create multiplication node."""
        return BinaryOpNode(left, right, 'mul')
    
    @staticmethod
    def divide(left: EquationNode, right: EquationNode) -> BinaryOpNode:
        """Create division node."""
        return BinaryOpNode(left, right, 'div')
    
    @staticmethod
    def power(left: EquationNode, right: EquationNode) -> BinaryOpNode:
        """Create power node."""
        return BinaryOpNode(left, right, 'pow')
    
    @staticmethod
    def negate(operand: EquationNode) -> UnaryOpNode:
        """Create negation node."""
        return UnaryOpNode(operand, 'neg')
    
    @staticmethod
    def sqrt(operand: EquationNode) -> UnaryOpNode:
        """Create square root node."""
        return UnaryOpNode(operand, 'sqrt')
    
    @staticmethod
    def abs(operand: EquationNode) -> UnaryOpNode:
        """Create absolute value node."""
        return UnaryOpNode(operand, 'abs')


def parse_expression(expression: str) -> EquationNode:
    """
    Simple expression parser for basic mathematical expressions.
    This is a placeholder - a full parser would be more complex.
    """
    expression = expression.strip()
    
    if expression.isdigit():
        return ConstantNode.from_int(int(expression))
    
    if '/' in expression and expression.count('/') == 1:
        parts = expression.split('/')
        if len(parts) == 2 and parts[0].strip().isdigit() and parts[1].strip().isdigit():
            num = int(parts[0].strip())
            den = int(parts[1].strip())
            return ConstantNode.from_fraction(num, den)
    
    if expression.isalpha():
        return VariableNode(expression)
    
    raise NotImplementedError(f"Complex expression parsing not yet implemented: {expression}")
