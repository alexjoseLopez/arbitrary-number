"""
Matrix Operations
=================

Comprehensive matrix mathematics for Arbitrary Numbers with exact computation.
"""

import torch
import numpy as np
from typing import List, Tuple, Union, Optional, Dict, Any
from dataclasses import dataclass

from .rational_list import RationalListNumber, FractionTerm
from .equation_nodes import EquationNode, ConstantNode, BinaryOpNode, VariableNode


@dataclass
class MatrixDimensions:
    """Matrix dimension information."""
    rows: int
    cols: int
    
    def __post_init__(self):
        if self.rows <= 0 or self.cols <= 0:
            raise ValueError("Matrix dimensions must be positive")
    
    def is_square(self) -> bool:
        return self.rows == self.cols
    
    def is_vector(self) -> bool:
        return self.rows == 1 or self.cols == 1
    
    def transpose(self) -> 'MatrixDimensions':
        return MatrixDimensions(self.cols, self.rows)


class ArbitraryMatrix:
    """
    Matrix of Arbitrary Numbers with exact computation.
    Supports all standard matrix operations with zero precision loss.
    """
    
    def __init__(self, data: Union[List[List[RationalListNumber]], 
                                 List[List[float]], 
                                 np.ndarray, 
                                 torch.Tensor]):
        
        if isinstance(data, (np.ndarray, torch.Tensor)):
            self.data = self._convert_from_tensor(data)
        elif isinstance(data, list):
            self.data = self._convert_from_list(data)
        else:
            raise TypeError("Unsupported data type for matrix initialization")
        
        self.dims = MatrixDimensions(len(self.data), len(self.data[0]))
        self._validate_matrix()
    
    @classmethod
    def zeros(cls, rows: int, cols: int) -> 'ArbitraryMatrix':
        """Create matrix of zeros."""
        zero = RationalListNumber.from_int(0)
        data = [[zero for _ in range(cols)] for _ in range(rows)]
        return cls(data)
    
    @classmethod
    def ones(cls, rows: int, cols: int) -> 'ArbitraryMatrix':
        """Create matrix of ones."""
        one = RationalListNumber.from_int(1)
        data = [[one for _ in range(cols)] for _ in range(rows)]
        return cls(data)
    
    @classmethod
    def identity(cls, size: int) -> 'ArbitraryMatrix':
        """Create identity matrix."""
        zero = RationalListNumber.from_int(0)
        one = RationalListNumber.from_int(1)
        
        data = []
        for i in range(size):
            row = []
            for j in range(size):
                row.append(one if i == j else zero)
            data.append(row)
        
        return cls(data)
    
    @classmethod
    def from_rational_fractions(cls, numerators: List[List[int]], 
                               denominators: List[List[int]]) -> 'ArbitraryMatrix':
        """Create matrix from separate numerator and denominator arrays."""
        if len(numerators) != len(denominators):
            raise ValueError("Numerator and denominator arrays must have same dimensions")
        
        data = []
        for i in range(len(numerators)):
            if len(numerators[i]) != len(denominators[i]):
                raise ValueError("Row dimensions must match")
            
            row = []
            for j in range(len(numerators[i])):
                rational = RationalListNumber.from_fraction(numerators[i][j], denominators[i][j])
                row.append(rational)
            data.append(row)
        
        return cls(data)
    
    def __getitem__(self, key: Tuple[int, int]) -> RationalListNumber:
        """Get matrix element."""
        row, col = key
        return self.data[row][col]
    
    def __setitem__(self, key: Tuple[int, int], value: RationalListNumber) -> None:
        """Set matrix element."""
        row, col = key
        self.data[row][col] = value
    
    def __add__(self, other: 'ArbitraryMatrix') -> 'ArbitraryMatrix':
        """Matrix addition."""
        if self.dims.rows != other.dims.rows or self.dims.cols != other.dims.cols:
            raise ValueError("Matrix dimensions must match for addition")
        
        result_data = []
        for i in range(self.dims.rows):
            row = []
            for j in range(self.dims.cols):
                sum_val = self.data[i][j] + other.data[i][j]
                row.append(sum_val)
            result_data.append(row)
        
        return ArbitraryMatrix(result_data)
    
    def __sub__(self, other: 'ArbitraryMatrix') -> 'ArbitraryMatrix':
        """Matrix subtraction."""
        if self.dims.rows != other.dims.rows or self.dims.cols != other.dims.cols:
            raise ValueError("Matrix dimensions must match for subtraction")
        
        result_data = []
        for i in range(self.dims.rows):
            row = []
            for j in range(self.dims.cols):
                diff_val = self.data[i][j] - other.data[i][j]
                row.append(diff_val)
            result_data.append(row)
        
        return ArbitraryMatrix(result_data)
    
    def __mul__(self, other: Union['ArbitraryMatrix', RationalListNumber, float, int]) -> 'ArbitraryMatrix':
        """Matrix multiplication or scalar multiplication."""
        if isinstance(other, ArbitraryMatrix):
            return self.matrix_multiply(other)
        else:
            return self.scalar_multiply(other)
    
    def __rmul__(self, other: Union[RationalListNumber, float, int]) -> 'ArbitraryMatrix':
        """Right scalar multiplication."""
        return self.scalar_multiply(other)
    
    def matrix_multiply(self, other: 'ArbitraryMatrix') -> 'ArbitraryMatrix':
        """Standard matrix multiplication."""
        if self.dims.cols != other.dims.rows:
            raise ValueError(f"Cannot multiply {self.dims.rows}x{self.dims.cols} by {other.dims.rows}x{other.dims.cols}")
        
        result_data = []
        for i in range(self.dims.rows):
            row = []
            for j in range(other.dims.cols):
                sum_val = RationalListNumber.from_int(0)
                
                for k in range(self.dims.cols):
                    product = self.data[i][k] * other.data[k][j]
                    sum_val = sum_val + product
                
                row.append(sum_val)
            result_data.append(row)
        
        return ArbitraryMatrix(result_data)
    
    def scalar_multiply(self, scalar: Union[RationalListNumber, float, int]) -> 'ArbitraryMatrix':
        """Scalar multiplication."""
        if isinstance(scalar, (float, int)):
            scalar = RationalListNumber.from_float(float(scalar))
        
        result_data = []
        for i in range(self.dims.rows):
            row = []
            for j in range(self.dims.cols):
                product = self.data[i][j] * scalar
                row.append(product)
            result_data.append(row)
        
        return ArbitraryMatrix(result_data)
    
    def transpose(self) -> 'ArbitraryMatrix':
        """Matrix transpose."""
        result_data = []
        for j in range(self.dims.cols):
            row = []
            for i in range(self.dims.rows):
                row.append(self.data[i][j])
            result_data.append(row)
        
        return ArbitraryMatrix(result_data)
    
    def determinant(self) -> RationalListNumber:
        """Calculate determinant using exact arithmetic."""
        if not self.dims.is_square():
            raise ValueError("Determinant only defined for square matrices")
        
        if self.dims.rows == 1:
            return self.data[0][0]
        
        if self.dims.rows == 2:
            return (self.data[0][0] * self.data[1][1] - 
                   self.data[0][1] * self.data[1][0])
        
        return self._determinant_recursive()
    
    def inverse(self) -> 'ArbitraryMatrix':
        """Calculate matrix inverse using exact arithmetic."""
        if not self.dims.is_square():
            raise ValueError("Inverse only defined for square matrices")
        
        det = self.determinant()
        if det.is_zero():
            raise ValueError("Matrix is singular (determinant is zero)")
        
        if self.dims.rows == 1:
            inv_val = RationalListNumber.from_int(1) / det
            return ArbitraryMatrix([[inv_val]])
        
        if self.dims.rows == 2:
            return self._inverse_2x2(det)
        
        return self._inverse_gauss_jordan()
    
    def trace(self) -> RationalListNumber:
        """Calculate matrix trace."""
        if not self.dims.is_square():
            raise ValueError("Trace only defined for square matrices")
        
        trace_val = RationalListNumber.from_int(0)
        for i in range(self.dims.rows):
            trace_val = trace_val + self.data[i][i]
        
        return trace_val
    
    def frobenius_norm(self) -> RationalListNumber:
        """Calculate Frobenius norm."""
        sum_squares = RationalListNumber.from_int(0)
        
        for i in range(self.dims.rows):
            for j in range(self.dims.cols):
                element_squared = self.data[i][j] * self.data[i][j]
                sum_squares = sum_squares + element_squared
        
        return sum_squares.sqrt()
    
    def rank(self) -> int:
        """Calculate matrix rank using exact arithmetic."""
        rref_matrix = self._row_reduce_to_rref()
        
        rank = 0
        for i in range(min(rref_matrix.dims.rows, rref_matrix.dims.cols)):
            row_has_pivot = False
            for j in range(rref_matrix.dims.cols):
                if not rref_matrix.data[i][j].is_zero():
                    row_has_pivot = True
                    break
            if row_has_pivot:
                rank += 1
        
        return rank
    
    def eigenvalues_2x2(self) -> Tuple[RationalListNumber, RationalListNumber]:
        """Calculate eigenvalues for 2x2 matrix using exact arithmetic."""
        if self.dims.rows != 2 or self.dims.cols != 2:
            raise ValueError("This method only works for 2x2 matrices")
        
        a = self.data[0][0]
        b = self.data[0][1]
        c = self.data[1][0]
        d = self.data[1][1]
        
        trace = a + d
        det = a * d - b * c
        
        discriminant = trace * trace - RationalListNumber.from_int(4) * det
        sqrt_discriminant = discriminant.sqrt()
        
        two = RationalListNumber.from_int(2)
        lambda1 = (trace + sqrt_discriminant) / two
        lambda2 = (trace - sqrt_discriminant) / two
        
        return lambda1, lambda2
    
    def lu_decomposition(self) -> Tuple['ArbitraryMatrix', 'ArbitraryMatrix']:
        """LU decomposition with exact arithmetic."""
        if not self.dims.is_square():
            raise ValueError("LU decomposition requires square matrix")
        
        n = self.dims.rows
        L = ArbitraryMatrix.identity(n)
        U = ArbitraryMatrix(self.data)
        
        for i in range(n):
            for j in range(i + 1, n):
                if U.data[i][i].is_zero():
                    raise ValueError("Matrix requires pivoting for LU decomposition")
                
                factor = U.data[j][i] / U.data[i][i]
                L.data[j][i] = factor
                
                for k in range(i, n):
                    U.data[j][k] = U.data[j][k] - factor * U.data[i][k]
        
        return L, U
    
    def qr_decomposition(self) -> Tuple['ArbitraryMatrix', 'ArbitraryMatrix']:
        """QR decomposition using Gram-Schmidt with exact arithmetic."""
        Q_data = []
        R_data = []
        
        for j in range(self.dims.cols):
            col_j = [self.data[i][j] for i in range(self.dims.rows)]
            
            for i in range(j):
                r_ij = RationalListNumber.from_int(0)
                for k in range(self.dims.rows):
                    r_ij = r_ij + Q_data[k][i] * self.data[k][j]
                
                R_data[i].append(r_ij)
                
                for k in range(self.dims.rows):
                    col_j[k] = col_j[k] - r_ij * Q_data[k][i]
            
            norm = RationalListNumber.from_int(0)
            for val in col_j:
                norm = norm + val * val
            norm = norm.sqrt()
            
            R_data.append([RationalListNumber.from_int(0)] * j + [norm])
            
            if len(Q_data) == 0:
                Q_data = [[val / norm] for val in col_j]
            else:
                for k in range(self.dims.rows):
                    Q_data[k].append(col_j[k] / norm)
        
        Q = ArbitraryMatrix(Q_data)
        R = ArbitraryMatrix(R_data)
        
        return Q, R
    
    def solve_linear_system(self, b: 'ArbitraryMatrix') -> 'ArbitraryMatrix':
        """Solve Ax = b using exact arithmetic."""
        if not self.dims.is_square():
            raise ValueError("Matrix must be square to solve linear system")
        
        if b.dims.rows != self.dims.rows:
            raise ValueError("Right-hand side must have same number of rows")
        
        augmented = self._create_augmented_matrix(b)
        rref = augmented._row_reduce_to_rref()
        
        solution_data = []
        for i in range(self.dims.rows):
            row = []
            for j in range(b.dims.cols):
                row.append(rref.data[i][self.dims.cols + j])
            solution_data.append(row)
        
        return ArbitraryMatrix(solution_data)
    
    def to_numpy(self) -> np.ndarray:
        """Convert to NumPy array with float approximation."""
        result = np.zeros((self.dims.rows, self.dims.cols))
        for i in range(self.dims.rows):
            for j in range(self.dims.cols):
                result[i, j] = float(self.data[i][j].evaluate_exact())
        return result
    
    def to_torch(self) -> torch.Tensor:
        """Convert to PyTorch tensor with float approximation."""
        return torch.from_numpy(self.to_numpy()).float()
    
    def to_string(self, precision: int = 6) -> str:
        """String representation with specified precision."""
        lines = []
        for i in range(self.dims.rows):
            row_strs = []
            for j in range(self.dims.cols):
                val = self.data[i][j]
                if hasattr(val, 'to_string'):
                    row_strs.append(val.to_string())
                else:
                    row_strs.append(f"{float(val.evaluate_exact()):.{precision}f}")
            lines.append("[" + ", ".join(row_strs) + "]")
        
        return "[\n  " + ",\n  ".join(lines) + "\n]"
    
    def __str__(self) -> str:
        return self.to_string()
    
    def __repr__(self) -> str:
        return f"ArbitraryMatrix({self.dims.rows}x{self.dims.cols})"
    
    def _convert_from_tensor(self, tensor: Union[np.ndarray, torch.Tensor]) -> List[List[RationalListNumber]]:
        """Convert tensor to internal representation."""
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.detach().cpu().numpy()
        
        data = []
        for i in range(tensor.shape[0]):
            row = []
            for j in range(tensor.shape[1]):
                rational = RationalListNumber.from_float(float(tensor[i, j]))
                row.append(rational)
            data.append(row)
        
        return data
    
    def _convert_from_list(self, data: List[List[Union[RationalListNumber, float, int]]]) -> List[List[RationalListNumber]]:
        """Convert list to internal representation."""
        result = []
        for row in data:
            converted_row = []
            for val in row:
                if isinstance(val, RationalListNumber):
                    converted_row.append(val)
                elif isinstance(val, (int, float)):
                    converted_row.append(RationalListNumber.from_float(float(val)))
                else:
                    raise TypeError(f"Unsupported element type: {type(val)}")
            result.append(converted_row)
        
        return result
    
    def _validate_matrix(self) -> None:
        """Validate matrix structure."""
        if not self.data:
            raise ValueError("Matrix cannot be empty")
        
        row_length = len(self.data[0])
        for i, row in enumerate(self.data):
            if len(row) != row_length:
                raise ValueError(f"Row {i} has different length than first row")
    
    def _determinant_recursive(self) -> RationalListNumber:
        """Calculate determinant recursively using cofactor expansion."""
        det = RationalListNumber.from_int(0)
        
        for j in range(self.dims.cols):
            minor = self._get_minor(0, j)
            cofactor = minor.determinant()
            
            if j % 2 == 1:
                cofactor = RationalListNumber.from_int(0) - cofactor
            
            det = det + self.data[0][j] * cofactor
        
        return det
    
    def _get_minor(self, exclude_row: int, exclude_col: int) -> 'ArbitraryMatrix':
        """Get minor matrix by excluding specified row and column."""
        minor_data = []
        
        for i in range(self.dims.rows):
            if i == exclude_row:
                continue
            
            row = []
            for j in range(self.dims.cols):
                if j == exclude_col:
                    continue
                row.append(self.data[i][j])
            
            minor_data.append(row)
        
        return ArbitraryMatrix(minor_data)
    
    def _inverse_2x2(self, det: RationalListNumber) -> 'ArbitraryMatrix':
        """Calculate 2x2 matrix inverse."""
        a = self.data[0][0]
        b = self.data[0][1]
        c = self.data[1][0]
        d = self.data[1][1]
        
        inv_data = [
            [d / det, (RationalListNumber.from_int(0) - b) / det],
            [(RationalListNumber.from_int(0) - c) / det, a / det]
        ]
        
        return ArbitraryMatrix(inv_data)
    
    def _inverse_gauss_jordan(self) -> 'ArbitraryMatrix':
        """Calculate inverse using Gauss-Jordan elimination."""
        identity = ArbitraryMatrix.identity(self.dims.rows)
        augmented = self._create_augmented_matrix(identity)
        rref = augmented._row_reduce_to_rref()
        
        inverse_data = []
        for i in range(self.dims.rows):
            row = []
            for j in range(self.dims.cols):
                row.append(rref.data[i][self.dims.cols + j])
            inverse_data.append(row)
        
        return ArbitraryMatrix(inverse_data)
    
    def _create_augmented_matrix(self, other: 'ArbitraryMatrix') -> 'ArbitraryMatrix':
        """Create augmented matrix [A|B]."""
        augmented_data = []
        
        for i in range(self.dims.rows):
            row = []
            for j in range(self.dims.cols):
                row.append(self.data[i][j])
            for j in range(other.dims.cols):
                row.append(other.data[i][j])
            augmented_data.append(row)
        
        return ArbitraryMatrix(augmented_data)
    
    def _row_reduce_to_rref(self) -> 'ArbitraryMatrix':
        """Row reduce to reduced row echelon form."""
        result = ArbitraryMatrix(self.data)
        
        current_row = 0
        for col in range(result.dims.cols):
            pivot_row = self._find_pivot_row(result, current_row, col)
            
            if pivot_row == -1:
                continue
            
            if pivot_row != current_row:
                result._swap_rows(current_row, pivot_row)
            
            pivot_val = result.data[current_row][col]
            if not pivot_val.is_zero():
                result._scale_row(current_row, RationalListNumber.from_int(1) / pivot_val)
                
                for i in range(result.dims.rows):
                    if i != current_row and not result.data[i][col].is_zero():
                        factor = result.data[i][col]
                        result._add_scaled_row(i, current_row, RationalListNumber.from_int(0) - factor)
                
                current_row += 1
        
        return result
    
    def _find_pivot_row(self, matrix: 'ArbitraryMatrix', start_row: int, col: int) -> int:
        """Find pivot row for given column."""
        for i in range(start_row, matrix.dims.rows):
            if not matrix.data[i][col].is_zero():
                return i
        return -1
    
    def _swap_rows(self, row1: int, row2: int) -> None:
        """Swap two rows."""
        self.data[row1], self.data[row2] = self.data[row2], self.data[row1]
    
    def _scale_row(self, row: int, scalar: RationalListNumber) -> None:
        """Scale row by scalar."""
        for j in range(self.dims.cols):
            self.data[row][j] = self.data[row][j] * scalar
    
    def _add_scaled_row(self, target_row: int, source_row: int, scalar: RationalListNumber) -> None:
        """Add scaled row to target row."""
        for j in range(self.dims.cols):
            self.data[target_row][j] = self.data[target_row][j] + scalar * self.data[source_row][j]


class MatrixOperations:
    """
    Static utility class for advanced matrix operations.
    """
    
    @staticmethod
    def kronecker_product(A: ArbitraryMatrix, B: ArbitraryMatrix) -> ArbitraryMatrix:
        """Compute Kronecker product A ⊗ B."""
        result_rows = A.dims.rows * B.dims.rows
        result_cols = A.dims.cols * B.dims.cols
        
        result_data = []
        for i in range(result_rows):
            row = []
            for j in range(result_cols):
                a_row = i // B.dims.rows
                a_col = j // B.dims.cols
                b_row = i % B.dims.rows
                b_col = j % B.dims.cols
                
                product = A.data[a_row][a_col] * B.data[b_row][b_col]
                row.append(product)
            result_data.append(row)
        
        return ArbitraryMatrix(result_data)
    
    @staticmethod
    def hadamard_product(A: ArbitraryMatrix, B: ArbitraryMatrix) -> ArbitraryMatrix:
        """Compute Hadamard (element-wise) product."""
        if A.dims.rows != B.dims.rows or A.dims.cols != B.dims.cols:
            raise ValueError("Matrices must have same dimensions for Hadamard product")
        
        result_data = []
        for i in range(A.dims.rows):
            row = []
            for j in range(A.dims.cols):
                product = A.data[i][j] * B.data[i][j]
                row.append(product)
            result_data.append(row)
        
        return ArbitraryMatrix(result_data)
    
    @staticmethod
    def matrix_power(A: ArbitraryMatrix, n: int) -> ArbitraryMatrix:
        """Compute matrix power A^n using exact arithmetic."""
        if not A.dims.is_square():
            raise ValueError("Matrix power requires square matrix")
        
        if n == 0:
            return ArbitraryMatrix.identity(A.dims.rows)
        
        if n == 1:
            return ArbitraryMatrix(A.data)
        
        if n < 0:
            return MatrixOperations.matrix_power(A.inverse(), -n)
        
        result = ArbitraryMatrix.identity(A.dims.rows)
        base = ArbitraryMatrix(A.data)
        
        while n > 0:
            if n % 2 == 1:
                result = result * base
            base = base * base
            n //= 2
        
        return result
    
    @staticmethod
    def matrix_exponential_series(A: ArbitraryMatrix, terms: int = 20) -> ArbitraryMatrix:
        """Compute matrix exponential using Taylor series with exact arithmetic."""
        if not A.dims.is_square():
            raise ValueError("Matrix exponential requires square matrix")
        
        result = ArbitraryMatrix.identity(A.dims.rows)
        term = ArbitraryMatrix.identity(A.dims.rows)
        
        for k in range(1, terms + 1):
            term = term * A
            factorial = RationalListNumber.from_int(1)
            for i in range(1, k + 1):
                factorial = factorial * RationalListNumber.from_int(i)
            
            scaled_term = term.scalar_multiply(RationalListNumber.from_int(1) / factorial)
            result = result + scaled_term
        
        return result
    
    @staticmethod
    def is_orthogonal(A: ArbitraryMatrix, tolerance: Optional[RationalListNumber] = None) -> bool:
        """Check if matrix is orthogonal (A^T * A = I)."""
        if not A.dims.is_square():
            return False
        
        if tolerance is None:
            tolerance = RationalListNumber.from_fraction(1, 1000000)
        
        AT = A.transpose()
        product = AT * A
        identity = ArbitraryMatrix.identity(A.dims.rows)
        
        difference = product - identity
        
        for i in range(difference.dims.rows):
            for j in range(difference.dims.cols):
                if difference.data[i][j].abs() > tolerance:
                    return False
        
        return True
    
    @staticmethod
    def is_symmetric(A: ArbitraryMatrix) -> bool:
        """Check if matrix is symmetric (A = A^T)."""
        if not A.dims.is_square():
            return False
        
        for i in range(A.dims.rows):
            for j in range(A.dims.cols):
                if A.data[i][j] != A.data[j][i]:
                    return False
        
        return True
    
    @staticmethod
    def condition_number(A: ArbitraryMatrix) -> RationalListNumber:
        """Compute condition number using Frobenius norm."""
        if not A.dims.is_square():
            raise ValueError("Condition number requires square matrix")
        
        A_inv = A.inverse()
        norm_A = A.frobenius_norm()
        norm_A_inv = A_inv.frobenius_norm()
        
        return norm_A * norm_A_inv
