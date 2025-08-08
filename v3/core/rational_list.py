"""
Rational List Number Implementation
==================================

Core implementation of RationalListNumber and FractionTerm classes
for exact fractional arithmetic with deferred evaluation.
"""

from dataclasses import dataclass
from typing import List, Union, Optional
from fractions import Fraction
from decimal import Decimal, getcontext
import math


@dataclass
class FractionTerm:
    """
    Represents a single fractional term as numerator/denominator.
    
    Uses Python's built-in Fraction for exact arithmetic.
    """
    numerator: int
    denominator: int = 1
    
    def __post_init__(self):
        if self.denominator == 0:
            raise ValueError("Denominator cannot be zero")
        
        if self.denominator < 0:
            self.numerator = -self.numerator
            self.denominator = -self.denominator
    
    @property
    def value(self) -> Fraction:
        """Get the exact fractional value."""
        return Fraction(self.numerator, self.denominator)
    
    def __str__(self) -> str:
        if self.denominator == 1:
            return str(self.numerator)
        return f"{self.numerator}/{self.denominator}"
    
    def __repr__(self) -> str:
        return f"FractionTerm({self.numerator}, {self.denominator})"


class RationalListNumber:
    """
    Represents a number as a list of fractional terms that can be
    added together. Evaluation is deferred until explicitly requested.
    
    Example: 1/2 + 3/4 - 2/3 is stored as [1/2, 3/4, -2/3]
    """
    
    def __init__(self, terms: Optional[List[FractionTerm]] = None):
        self.terms = terms or []
    
    @classmethod
    def from_int(cls, value: int) -> 'RationalListNumber':
        """Create from integer value."""
        return cls([FractionTerm(value, 1)])
    
    @classmethod
    def from_fraction(cls, numerator: int, denominator: int = 1) -> 'RationalListNumber':
        """Create from numerator/denominator."""
        return cls([FractionTerm(numerator, denominator)])
    
    @classmethod
    def from_decimal(cls, value: Union[float, Decimal]) -> 'RationalListNumber':
        """Create from decimal value (converts to exact fraction)."""
        frac = Fraction(value).limit_denominator()
        return cls([FractionTerm(frac.numerator, frac.denominator)])
    
    def add_term(self, term: FractionTerm) -> None:
        """Add a new fractional term to the list."""
        self.terms.append(term)
    
    def __add__(self, other: 'RationalListNumber') -> 'RationalListNumber':
        """Add two RationalListNumbers by combining their terms."""
        return RationalListNumber(self.terms + other.terms)
    
    def __sub__(self, other: 'RationalListNumber') -> 'RationalListNumber':
        """Subtract by negating other's terms and adding."""
        negated_terms = [FractionTerm(-term.numerator, term.denominator) 
                        for term in other.terms]
        return RationalListNumber(self.terms + negated_terms)
    
    def __mul__(self, other: Union['RationalListNumber', int, FractionTerm]) -> 'RationalListNumber':
        """Multiply all terms by a scalar or another RationalListNumber."""
        if isinstance(other, int):
            multiplied_terms = [FractionTerm(term.numerator * other, term.denominator) 
                              for term in self.terms]
            return RationalListNumber(multiplied_terms)
        
        elif isinstance(other, FractionTerm):
            multiplied_terms = [
                FractionTerm(
                    term.numerator * other.numerator,
                    term.denominator * other.denominator
                ) for term in self.terms
            ]
            return RationalListNumber(multiplied_terms)
        
        elif isinstance(other, RationalListNumber):
            result_terms = []
            for self_term in self.terms:
                for other_term in other.terms:
                    result_terms.append(FractionTerm(
                        self_term.numerator * other_term.numerator,
                        self_term.denominator * other_term.denominator
                    ))
            return RationalListNumber(result_terms)
        
        else:
            raise TypeError(f"Cannot multiply RationalListNumber by {type(other)}")
    
    def __truediv__(self, other: Union[int, FractionTerm]) -> 'RationalListNumber':
        """Divide all terms by a scalar."""
        if isinstance(other, int):
            if other == 0:
                raise ValueError("Division by zero")
            divided_terms = [FractionTerm(term.numerator, term.denominator * other) 
                           for term in self.terms]
            return RationalListNumber(divided_terms)
        
        elif isinstance(other, FractionTerm):
            if other.numerator == 0:
                raise ValueError("Division by zero")
            divided_terms = [
                FractionTerm(
                    term.numerator * other.denominator,
                    term.denominator * other.numerator
                ) for term in self.terms
            ]
            return RationalListNumber(divided_terms)
        
        else:
            raise TypeError(f"Cannot divide RationalListNumber by {type(other)}")
    
    def __neg__(self) -> 'RationalListNumber':
        """Negate all terms."""
        negated_terms = [FractionTerm(-term.numerator, term.denominator) 
                        for term in self.terms]
        return RationalListNumber(negated_terms)
    
    def evaluate(self, precision: int = 50) -> Decimal:
        """
        Evaluate the sum of all terms to a Decimal with specified precision.
        This is where deferred evaluation actually happens.
        """
        getcontext().prec = precision
        
        result = Decimal(0)
        for term in self.terms:
            term_value = Decimal(term.numerator) / Decimal(term.denominator)
            result += term_value
        
        return result
    
    def evaluate_exact(self) -> Fraction:
        """
        Evaluate to an exact Fraction by finding common denominator.
        This maintains perfect precision.
        """
        if not self.terms:
            return Fraction(0)
        
        result = Fraction(0)
        for term in self.terms:
            result += term.value
        
        return result
    
    def simplify(self) -> 'RationalListNumber':
        """
        Simplify by combining all terms into a single equivalent term.
        This loses the deferred evaluation benefit but reduces memory.
        """
        exact_value = self.evaluate_exact()
        return RationalListNumber([FractionTerm(exact_value.numerator, exact_value.denominator)])
    
    def __str__(self) -> str:
        if not self.terms:
            return "0"
        
        parts = []
        for i, term in enumerate(self.terms):
            if i == 0:
                parts.append(str(term))
            else:
                if term.numerator >= 0:
                    parts.append(f" + {term}")
                else:
                    parts.append(f" - {FractionTerm(-term.numerator, term.denominator)}")
        
        return "".join(parts)
    
    def __repr__(self) -> str:
        return f"RationalListNumber({self.terms})"
    
    def __len__(self) -> int:
        """Return number of terms."""
        return len(self.terms)
    
    def __bool__(self) -> bool:
        """Return True if non-zero."""
        return any(term.numerator != 0 for term in self.terms)
    
    def to_float(self) -> float:
        """Convert to float (may lose precision)."""
        return float(self.evaluate_exact())
    
    def memory_usage(self) -> int:
        """Estimate memory usage in bytes."""
        return len(self.terms) * (2 * 8)  # 2 integers per term, 8 bytes each
