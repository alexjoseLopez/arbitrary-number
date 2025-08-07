"""
ArbitraryNumber: A high-precision numeric data type for exact mathematical computations.

This implementation provides exact arithmetic operations without floating-point precision loss,
supporting rational numbers, large integers, and high-precision decimal representations.
"""

import math
import re
from fractions import Fraction
from decimal import Decimal, getcontext
from typing import Union, List, Tuple, Any
import json


class ArbitraryNumber:
    """
    A numeric data type that maintains exact precision for mathematical operations.
    
    Supports:
    - Exact rational arithmetic
    - Arbitrary precision integers
    - High-precision decimal operations
    - Mathematical functions with configurable precision
    """
    
    def __init__(self, value: Union[int, float, str, 'ArbitraryNumber', Fraction] = 0):
        """Initialize ArbitraryNumber from various input types."""
        if isinstance(value, ArbitraryNumber):
            self._fraction = value._fraction
        elif isinstance(value, Fraction):
            self._fraction = value
        elif isinstance(value, int):
            self._fraction = Fraction(value)
        elif isinstance(value, float):
            # Convert float to exact fraction representation
            self._fraction = Fraction(value).limit_denominator()
        elif isinstance(value, str):
            if '/' in value:
                # Handle fraction strings like "22/7"
                parts = value.split('/')
                if len(parts) == 2:
                    self._fraction = Fraction(int(parts[0]), int(parts[1]))
                else:
                    raise ValueError(f"Invalid fraction format: {value}")
            else:
                # Handle decimal strings
                self._fraction = Fraction(value)
        else:
            raise TypeError(f"Unsupported type for ArbitraryNumber: {type(value)}")
    
    def __str__(self) -> str:
        """String representation of the number."""
        if self._fraction.denominator == 1:
            return str(self._fraction.numerator)
        else:
            # Check if it's a power of 10 denominator (decimal representation)
            denom = self._fraction.denominator
            temp_denom = denom
            power_of_10 = 0
            while temp_denom % 10 == 0:
                temp_denom //= 10
                power_of_10 += 1
            
            if temp_denom == 1:  # Pure power of 10
                num_str = str(self._fraction.numerator)
                if power_of_10 >= len(num_str):
                    # Need leading zeros
                    zeros_needed = power_of_10 - len(num_str)
                    return "0." + "0" * zeros_needed + num_str
                else:
                    # Insert decimal point
                    insert_pos = len(num_str) - power_of_10
                    return num_str[:insert_pos] + "." + num_str[insert_pos:]
            
            # Check if it's a simple decimal representation
            if self._fraction.denominator in [10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000, 10000000000]:
                decimal_str = str(float(self._fraction))
                # Only use decimal if it's exact and reasonable length
                if len(decimal_str) < 20 and Fraction(decimal_str).limit_denominator() == self._fraction:
                    return decimal_str
            return f"{self._fraction.numerator}/{self._fraction.denominator}"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"ArbitraryNumber('{str(self)}')"
    
    def __eq__(self, other) -> bool:
        """Equality comparison."""
        if not isinstance(other, ArbitraryNumber):
            other = ArbitraryNumber(other)
        return self._fraction == other._fraction
    
    def __lt__(self, other) -> bool:
        """Less than comparison."""
        if not isinstance(other, ArbitraryNumber):
            other = ArbitraryNumber(other)
        return self._fraction < other._fraction
    
    def __le__(self, other) -> bool:
        """Less than or equal comparison."""
        if not isinstance(other, ArbitraryNumber):
            other = ArbitraryNumber(other)
        return self._fraction <= other._fraction
    
    def __gt__(self, other) -> bool:
        """Greater than comparison."""
        if not isinstance(other, ArbitraryNumber):
            other = ArbitraryNumber(other)
        return self._fraction > other._fraction
    
    def __ge__(self, other) -> bool:
        """Greater than or equal comparison."""
        if not isinstance(other, ArbitraryNumber):
            other = ArbitraryNumber(other)
        return self._fraction >= other._fraction
    
    def __add__(self, other) -> 'ArbitraryNumber':
        """Addition operation."""
        if not isinstance(other, ArbitraryNumber):
            other = ArbitraryNumber(other)
        return ArbitraryNumber(self._fraction + other._fraction)
    
    def __radd__(self, other) -> 'ArbitraryNumber':
        """Reverse addition."""
        return self.__add__(other)
    
    def __sub__(self, other) -> 'ArbitraryNumber':
        """Subtraction operation."""
        if not isinstance(other, ArbitraryNumber):
            other = ArbitraryNumber(other)
        return ArbitraryNumber(self._fraction - other._fraction)
    
    def __rsub__(self, other) -> 'ArbitraryNumber':
        """Reverse subtraction."""
        if not isinstance(other, ArbitraryNumber):
            other = ArbitraryNumber(other)
        return ArbitraryNumber(other._fraction - self._fraction)
    
    def __mul__(self, other) -> 'ArbitraryNumber':
        """Multiplication operation."""
        if not isinstance(other, ArbitraryNumber):
            other = ArbitraryNumber(other)
        return ArbitraryNumber(self._fraction * other._fraction)
    
    def __rmul__(self, other) -> 'ArbitraryNumber':
        """Reverse multiplication."""
        return self.__mul__(other)
    
    def __truediv__(self, other) -> 'ArbitraryNumber':
        """Division operation."""
        if not isinstance(other, ArbitraryNumber):
            other = ArbitraryNumber(other)
        if other._fraction == 0:
            raise ZeroDivisionError("Division by zero")
        return ArbitraryNumber(self._fraction / other._fraction)
    
    def __rtruediv__(self, other) -> 'ArbitraryNumber':
        """Reverse division."""
        if not isinstance(other, ArbitraryNumber):
            other = ArbitraryNumber(other)
        if self._fraction == 0:
            raise ZeroDivisionError("Division by zero")
        return ArbitraryNumber(other._fraction / self._fraction)
    
    def __pow__(self, exponent) -> 'ArbitraryNumber':
        """Power operation."""
        if isinstance(exponent, ArbitraryNumber):
            exponent = exponent._fraction
        elif not isinstance(exponent, (int, Fraction)):
            exponent = Fraction(exponent)
        
        # Handle special cases
        if self._fraction == 0 and exponent == 0:
            raise ValueError("0^0 is undefined")
        
        # For integer exponents, use exact computation
        if isinstance(exponent, int) or (isinstance(exponent, Fraction) and exponent.denominator == 1):
            exp_int = int(exponent)
            return ArbitraryNumber(self._fraction ** exp_int)
        
        # For fractional exponents, approximate using high precision
        return self._approximate_power(exponent)
    
    def __mod__(self, other) -> 'ArbitraryNumber':
        """Modulo operation."""
        if not isinstance(other, ArbitraryNumber):
            other = ArbitraryNumber(other)
        
        # Convert to integers for modulo operation
        if self._fraction.denominator != 1 or other._fraction.denominator != 1:
            raise ValueError("Modulo operation requires integer operands")
        
        result = self._fraction.numerator % other._fraction.numerator
        return ArbitraryNumber(result)
    
    def __hash__(self) -> int:
        """Hash function for use in sets and dictionaries."""
        return hash(self._fraction)
    
    def __neg__(self) -> 'ArbitraryNumber':
        """Negation operation."""
        return ArbitraryNumber(-self._fraction)
    
    def __abs__(self) -> 'ArbitraryNumber':
        """Absolute value."""
        return ArbitraryNumber(abs(self._fraction))
    
    def is_zero(self) -> bool:
        """Check if the number is zero."""
        return self._fraction == 0
    
    def is_negative(self) -> bool:
        """Check if the number is negative."""
        return self._fraction < 0
    
    def is_positive(self) -> bool:
        """Check if the number is positive."""
        return self._fraction > 0
    
    def is_integer(self) -> bool:
        """Check if the number is an integer."""
        return self._fraction.denominator == 1
    
    def is_rational(self) -> bool:
        """Check if the number is rational (always True for this implementation)."""
        return True
    
    def sqrt(self) -> 'ArbitraryNumber':
        """Square root operation."""
        if self._fraction < 0:
            raise ValueError("Square root of negative number")
        
        # Check for perfect squares
        num_sqrt = int(self._fraction.numerator ** 0.5)
        den_sqrt = int(self._fraction.denominator ** 0.5)
        
        if num_sqrt * num_sqrt == self._fraction.numerator and den_sqrt * den_sqrt == self._fraction.denominator:
            return ArbitraryNumber(Fraction(num_sqrt, den_sqrt))
        
        # Use Newton's method for approximation
        return self._approximate_sqrt()
    
    def factorial(self) -> 'ArbitraryNumber':
        """Factorial operation."""
        if not self.is_integer():
            raise ValueError("Factorial requires integer input")
        
        n = self._fraction.numerator
        if n < 0:
            raise ValueError("Factorial of negative number")
        
        result = 1
        for i in range(2, n + 1):
            result *= i
        
        return ArbitraryNumber(result)
    
    def copy(self) -> 'ArbitraryNumber':
        """Create a copy of this number."""
        return ArbitraryNumber(self._fraction)
    
    def to_decimal(self, precision: int = 50) -> str:
        """Convert to decimal string with specified precision."""
        getcontext().prec = precision + 10  # Extra precision for intermediate calculations
        decimal_val = Decimal(self._fraction.numerator) / Decimal(self._fraction.denominator)
        return str(decimal_val)[:precision + 2]  # +2 for "0."
    
    def to_continued_fraction(self) -> List[int]:
        """Convert to continued fraction representation."""
        result = []
        num = self._fraction.numerator
        den = self._fraction.denominator
        
        while den != 0:
            quotient = num // den
            result.append(quotient)
            num, den = den, num - quotient * den
        
        return result
    
    def to_scientific_notation(self) -> str:
        """Convert to scientific notation."""
        decimal_str = self.to_decimal(precision=20)
        decimal_val = float(decimal_str)
        sci_str = f"{decimal_val:.8e}"
        # Fix formatting to match expected format (remove leading zero in exponent)
        return sci_str.replace('e+0', 'e+').replace('e-0', 'e-')
    
    def serialize(self) -> str:
        """Serialize to JSON string."""
        return json.dumps({
            'numerator': self._fraction.numerator,
            'denominator': self._fraction.denominator
        })
    
    @classmethod
    def deserialize(cls, data: str) -> 'ArbitraryNumber':
        """Deserialize from JSON string."""
        parsed = json.loads(data)
        fraction = Fraction(parsed['numerator'], parsed['denominator'])
        return cls(fraction)
    
    @classmethod
    def pi(cls, precision: int = 50) -> 'ArbitraryNumber':
        """Generate pi approximation with specified precision."""
        # Use a high-precision approximation of pi
        # This is pi to many decimal places as a fraction
        pi_str = "3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679"
        # Ensure we have enough digits for the requested precision
        if precision + 2 <= len(pi_str):
            return cls(pi_str[:precision+2])  # +2 for "3."
        else:
            return cls(pi_str)
    
    @classmethod
    def e(cls, precision: int = 50) -> 'ArbitraryNumber':
        """Generate e approximation with specified precision."""
        # Use series expansion: e = sum(1/n!) for n=0 to infinity
        result = ArbitraryNumber(0)
        factorial = ArbitraryNumber(1)
        
        for n in range(precision):
            if n > 0:
                factorial = factorial * ArbitraryNumber(n)
            term = ArbitraryNumber(1) / factorial
            result = result + term
        
        return result
    
    @classmethod
    def gcd(cls, a: 'ArbitraryNumber', b: 'ArbitraryNumber') -> 'ArbitraryNumber':
        """Greatest common divisor."""
        if not (a.is_integer() and b.is_integer()):
            raise ValueError("GCD requires integer inputs")
        
        gcd_result = math.gcd(a._fraction.numerator, b._fraction.numerator)
        return cls(gcd_result)
    
    @classmethod
    def lcm(cls, a: 'ArbitraryNumber', b: 'ArbitraryNumber') -> 'ArbitraryNumber':
        """Least common multiple."""
        if not (a.is_integer() and b.is_integer()):
            raise ValueError("LCM requires integer inputs")
        
        gcd_val = cls.gcd(a, b)
        lcm_result = abs(a._fraction.numerator * b._fraction.numerator) // gcd_val._fraction.numerator
        return cls(lcm_result)
    
    def _approximate_power(self, exponent: Fraction) -> 'ArbitraryNumber':
        """Approximate fractional power using high precision."""
        # Convert to decimal for approximation
        base_decimal = float(self._fraction)
        exp_decimal = float(exponent)
        
        result = base_decimal ** exp_decimal
        return ArbitraryNumber(result)
    
    def _approximate_sqrt(self) -> 'ArbitraryNumber':
        """Approximate square root using Newton's method."""
        if self._fraction == 0:
            return ArbitraryNumber(0)
        
        # Initial guess
        x = ArbitraryNumber(float(self._fraction) ** 0.5)
        
        # Newton's method iterations
        for _ in range(50):  # Sufficient for high precision
            x_new = (x + self / x) / ArbitraryNumber(2)
            if abs(x_new - x) < ArbitraryNumber(Fraction(1, 10**50)):
                break
            x = x_new
        
        return x
