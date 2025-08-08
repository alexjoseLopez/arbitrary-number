"""
Comprehensive tests for matrix operations with Arbitrary Numbers.
"""

import pytest
import numpy as np
import torch
from typing import List

from arbitrary_numbers.core.matrix_operations import ArbitraryMatrix, MatrixOperations, MatrixDimensions
from arbitrary_numbers.core.rational_list import RationalListNumber, FractionTerm


class TestMatrixDimensions:
    """Test matrix dimension utilities."""
    
    def test_valid_dimensions(self):
        dims = MatrixDimensions(3, 4)
        assert dims.rows == 3
        assert dims.cols == 4
        assert not dims.is_square()
        assert not dims.is_vector()
    
    def test_square_matrix(self):
        dims = MatrixDimensions(3, 3)
        assert dims.is_square()
        assert not dims.is_vector()
    
    def test_vector_dimensions(self):
        row_vector = MatrixDimensions(1, 5)
        col_vector = MatrixDimensions(5, 1)
        assert row_vector.is_vector()
        assert col_vector.is_vector()
    
    def test_transpose_dimensions(self):
        dims = MatrixDimensions(3, 4)
        transposed = dims.transpose()
        assert transposed.rows == 4
        assert transposed.cols == 3
    
    def test_invalid_dimensions(self):
        with pytest.raises(ValueError):
            MatrixDimensions(0, 5)
        with pytest.raises(ValueError):
            MatrixDimensions(5, -1)


class TestArbitraryMatrixCreation:
    """Test matrix creation methods."""
    
    def test_from_list_integers(self):
        data = [[1, 2], [3, 4]]
        matrix = ArbitraryMatrix(data)
        assert matrix.dims.rows == 2
        assert matrix.dims.cols == 2
        assert matrix[0, 0].evaluate_exact() == 1
        assert matrix[1, 1].evaluate_exact() == 4
    
    def test_from_list_floats(self):
        data = [[1.5, 2.25], [3.75, 4.125]]
        matrix = ArbitraryMatrix(data)
        assert matrix.dims.rows == 2
        assert matrix.dims.cols == 2
        assert abs(float(matrix[0, 0].evaluate_exact()) - 1.5) < 1e-10
    
    def test_from_numpy_array(self):
        np_array = np.array([[1, 2, 3], [4, 5, 6]])
        matrix = ArbitraryMatrix(np_array)
        assert matrix.dims.rows == 2
        assert matrix.dims.cols == 3
        assert matrix[1, 2].evaluate_exact() == 6
    
    def test_from_torch_tensor(self):
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        matrix = ArbitraryMatrix(tensor)
        assert matrix.dims.rows == 2
        assert matrix.dims.cols == 2
        assert matrix[0, 1].evaluate_exact() == 2
    
    def test_zeros_matrix(self):
        matrix = ArbitraryMatrix.zeros(3, 4)
        assert matrix.dims.rows == 3
        assert matrix.dims.cols == 4
        for i in range(3):
            for j in range(4):
                assert matrix[i, j].is_zero()
    
    def test_ones_matrix(self):
        matrix = ArbitraryMatrix.ones(2, 3)
        assert matrix.dims.rows == 2
        assert matrix.dims.cols == 3
        for i in range(2):
            for j in range(3):
                assert matrix[i, j].evaluate_exact() == 1
    
    def test_identity_matrix(self):
        matrix = ArbitraryMatrix.identity(3)
        assert matrix.dims.rows == 3
        assert matrix.dims.cols == 3
        for i in range(3):
            for j in range(3):
                if i == j:
                    assert matrix[i, j].evaluate_exact() == 1
                else:
                    assert matrix[i, j].is_zero()
    
    def test_from_rational_fractions(self):
        numerators = [[1, 2], [3, 4]]
        denominators = [[2, 3], [4, 5]]
        matrix = ArbitraryMatrix.from_rational_fractions(numerators, denominators)
        
        assert matrix.dims.rows == 2
        assert matrix.dims.cols == 2
        assert matrix[0, 0].evaluate_exact() == 0.5
        assert abs(float(matrix[0, 1].evaluate_exact()) - (2.0/3.0)) < 1e-10
        assert matrix[1, 0].evaluate_exact() == 0.75
        assert matrix[1, 1].evaluate_exact() == 0.8
    
    def test_invalid_matrix_data(self):
        with pytest.raises(ValueError):
            ArbitraryMatrix([])
        
        with pytest.raises(ValueError):
            ArbitraryMatrix([[1, 2], [3, 4, 5]])


class TestBasicMatrixOperations:
    """Test basic matrix arithmetic operations."""
    
    def test_matrix_addition(self):
        A = ArbitraryMatrix([[1, 2], [3, 4]])
        B = ArbitraryMatrix([[5, 6], [7, 8]])
        C = A + B
        
        assert C[0, 0].evaluate_exact() == 6
        assert C[0, 1].evaluate_exact() == 8
        assert C[1, 0].evaluate_exact() == 10
        assert C[1, 1].evaluate_exact() == 12
    
    def test_matrix_subtraction(self):
        A = ArbitraryMatrix([[5, 6], [7, 8]])
        B = ArbitraryMatrix([[1, 2], [3, 4]])
        C = A - B
        
        assert C[0, 0].evaluate_exact() == 4
        assert C[0, 1].evaluate_exact() == 4
        assert C[1, 0].evaluate_exact() == 4
        assert C[1, 1].evaluate_exact() == 4
    
    def test_scalar_multiplication(self):
        A = ArbitraryMatrix([[1, 2], [3, 4]])
        scalar = RationalListNumber.from_fraction(3, 2)
        B = A * scalar
        
        assert B[0, 0].evaluate_exact() == 1.5
        assert B[0, 1].evaluate_exact() == 3.0
        assert B[1, 0].evaluate_exact() == 4.5
        assert B[1, 1].evaluate_exact() == 6.0
    
    def test_scalar_multiplication_with_float(self):
        A = ArbitraryMatrix([[2, 4], [6, 8]])
        B = A * 0.5
        
        assert B[0, 0].evaluate_exact() == 1
        assert B[0, 1].evaluate_exact() == 2
        assert B[1, 0].evaluate_exact() == 3
        assert B[1, 1].evaluate_exact() == 4
    
    def test_matrix_multiplication(self):
        A = ArbitraryMatrix([[1, 2], [3, 4]])
        B = ArbitraryMatrix([[5, 6], [7, 8]])
        C = A * B
        
        assert C[0, 0].evaluate_exact() == 19  # 1*5 + 2*7
        assert C[0, 1].evaluate_exact() == 22  # 1*6 + 2*8
        assert C[1, 0].evaluate_exact() == 43  # 3*5 + 4*7
        assert C[1, 1].evaluate_exact() == 50  # 3*6 + 4*8
    
    def test_matrix_transpose(self):
        A = ArbitraryMatrix([[1, 2, 3], [4, 5, 6]])
        AT = A.transpose()
        
        assert AT.dims.rows == 3
        assert AT.dims.cols == 2
        assert AT[0, 0].evaluate_exact() == 1
        assert AT[1, 0].evaluate_exact() == 2
        assert AT[2, 1].evaluate_exact() == 6
    
    def test_dimension_mismatch_errors(self):
        A = ArbitraryMatrix([[1, 2]])
        B = ArbitraryMatrix([[1], [2]])
        
        with pytest.raises(ValueError):
            A + B
        
        with pytest.raises(ValueError):
            A - B


class TestAdvancedMatrixOperations:
    """Test advanced matrix operations."""
    
    def test_determinant_2x2(self):
        A = ArbitraryMatrix([[3, 2], [1, 4]])
        det = A.determinant()
        assert det.evaluate_exact() == 10  # 3*4 - 2*1
    
    def test_determinant_3x3(self):
        A = ArbitraryMatrix([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
        det = A.determinant()
        assert det.evaluate_exact() == -3
    
    def test_determinant_1x1(self):
        A = ArbitraryMatrix([[5]])
        det = A.determinant()
        assert det.evaluate_exact() == 5
    
    def test_matrix_inverse_2x2(self):
        A = ArbitraryMatrix([[4, 2], [1, 3]])
        A_inv = A.inverse()
        
        identity = A * A_inv
        
        tolerance = 1e-10
        assert abs(float(identity[0, 0].evaluate_exact()) - 1.0) < tolerance
        assert abs(float(identity[0, 1].evaluate_exact()) - 0.0) < tolerance
        assert abs(float(identity[1, 0].evaluate_exact()) - 0.0) < tolerance
        assert abs(float(identity[1, 1].evaluate_exact()) - 1.0) < tolerance
    
    def test_matrix_inverse_1x1(self):
        A = ArbitraryMatrix([[4]])
        A_inv = A.inverse()
        assert A_inv[0, 0].evaluate_exact() == 0.25
    
    def test_singular_matrix_inverse(self):
        A = ArbitraryMatrix([[1, 2], [2, 4]])  # Singular matrix
        with pytest.raises(ValueError):
            A.inverse()
    
    def test_matrix_trace(self):
        A = ArbitraryMatrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        trace = A.trace()
        assert trace.evaluate_exact() == 15  # 1 + 5 + 9
    
    def test_frobenius_norm(self):
        A = ArbitraryMatrix([[3, 4]])
        norm = A.frobenius_norm()
        assert norm.evaluate_exact() == 5  # sqrt(3^2 + 4^2)
    
    def test_matrix_rank(self):
        A = ArbitraryMatrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        rank = A.rank()
        assert rank == 2
        
        identity = ArbitraryMatrix.identity(3)
        assert identity.rank() == 3
    
    def test_eigenvalues_2x2(self):
        A = ArbitraryMatrix([[4, 1], [2, 3]])
        lambda1, lambda2 = A.eigenvalues_2x2()
        
        eigenvalues = [float(lambda1.evaluate_exact()), float(lambda2.evaluate_exact())]
        eigenvalues.sort()
        
        expected = [2.0, 5.0]
        assert abs(eigenvalues[0] - expected[0]) < 1e-10
        assert abs(eigenvalues[1] - expected[1]) < 1e-10


class TestMatrixDecompositions:
    """Test matrix decomposition algorithms."""
    
    def test_lu_decomposition(self):
        A = ArbitraryMatrix([[2, 1, 1], [4, 3, 3], [8, 7, 9]])
        L, U = A.lu_decomposition()
        
        reconstructed = L * U
        
        tolerance = 1e-10
        for i in range(A.dims.rows):
            for j in range(A.dims.cols):
                original = float(A[i, j].evaluate_exact())
                reconstructed_val = float(reconstructed[i, j].evaluate_exact())
                assert abs(original - reconstructed_val) < tolerance
    
    def test_qr_decomposition(self):
        A = ArbitraryMatrix([[1, 1], [1, 0], [0, 1]])
        Q, R = A.qr_decomposition()
        
        reconstructed = Q * R
        
        tolerance = 1e-8
        for i in range(A.dims.rows):
            for j in range(A.dims.cols):
                original = float(A[i, j].evaluate_exact())
                reconstructed_val = float(reconstructed[i, j].evaluate_exact())
                assert abs(original - reconstructed_val) < tolerance
    
    def test_solve_linear_system(self):
        A = ArbitraryMatrix([[2, 1], [1, 3]])
        b = ArbitraryMatrix([[5], [6]])
        x = A.solve_linear_system(b)
        
        result = A * x
        
        tolerance = 1e-10
        for i in range(b.dims.rows):
            original = float(b[i, 0].evaluate_exact())
            computed = float(result[i, 0].evaluate_exact())
            assert abs(original - computed) < tolerance


class TestMatrixUtilities:
    """Test matrix utility functions."""
    
    def test_to_numpy_conversion(self):
        A = ArbitraryMatrix([[1, 2], [3, 4]])
        np_array = A.to_numpy()
        
        assert isinstance(np_array, np.ndarray)
        assert np_array.shape == (2, 2)
        assert np_array[0, 0] == 1.0
        assert np_array[1, 1] == 4.0
    
    def test_to_torch_conversion(self):
        A = ArbitraryMatrix([[1, 2], [3, 4]])
        tensor = A.to_torch()
        
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (2, 2)
        assert tensor[0, 0].item() == 1.0
        assert tensor[1, 1].item() == 4.0
    
    def test_string_representation(self):
        A = ArbitraryMatrix([[1, 2], [3, 4]])
        string_repr = A.to_string()
        
        assert isinstance(string_repr, str)
        assert "1" in string_repr
        assert "4" in string_repr
    
    def test_matrix_indexing(self):
        A = ArbitraryMatrix([[1, 2], [3, 4]])
        
        assert A[0, 0].evaluate_exact() == 1
        assert A[1, 1].evaluate_exact() == 4
        
        new_val = RationalListNumber.from_int(10)
        A[0, 0] = new_val
        assert A[0, 0].evaluate_exact() == 10


class TestAdvancedMatrixOperations:
    """Test advanced matrix operations from MatrixOperations class."""
    
    def test_kronecker_product(self):
        A = ArbitraryMatrix([[1, 2], [3, 4]])
        B = ArbitraryMatrix([[0, 5], [6, 7]])
        
        kron = MatrixOperations.kronecker_product(A, B)
        
        assert kron.dims.rows == 4
        assert kron.dims.cols == 4
        assert kron[0, 0].evaluate_exact() == 0   # 1 * 0
        assert kron[0, 1].evaluate_exact() == 5   # 1 * 5
        assert kron[2, 2].evaluate_exact() == 18  # 3 * 6
    
    def test_hadamard_product(self):
        A = ArbitraryMatrix([[1, 2], [3, 4]])
        B = ArbitraryMatrix([[5, 6], [7, 8]])
        
        hadamard = MatrixOperations.hadamard_product(A, B)
        
        assert hadamard[0, 0].evaluate_exact() == 5   # 1 * 5
        assert hadamard[0, 1].evaluate_exact() == 12  # 2 * 6
        assert hadamard[1, 0].evaluate_exact() == 21  # 3 * 7
        assert hadamard[1, 1].evaluate_exact() == 32  # 4 * 8
    
    def test_matrix_power(self):
        A = ArbitraryMatrix([[2, 1], [0, 2]])
        
        A_squared = MatrixOperations.matrix_power(A, 2)
        expected = A * A
        
        for i in range(A.dims.rows):
            for j in range(A.dims.cols):
                assert A_squared[i, j].evaluate_exact() == expected[i, j].evaluate_exact()
        
        A_zero = MatrixOperations.matrix_power(A, 0)
        identity = ArbitraryMatrix.identity(2)
        
        for i in range(2):
            for j in range(2):
                assert A_zero[i, j].evaluate_exact() == identity[i, j].evaluate_exact()
    
    def test_matrix_exponential_series(self):
        A = ArbitraryMatrix([[0, 1], [0, 0]])  # Nilpotent matrix
        exp_A = MatrixOperations.matrix_exponential_series(A, terms=10)
        
        expected = ArbitraryMatrix([[1, 1], [0, 1]])
        
        tolerance = 1e-10
        for i in range(A.dims.rows):
            for j in range(A.dims.cols):
                computed = float(exp_A[i, j].evaluate_exact())
                expected_val = float(expected[i, j].evaluate_exact())
                assert abs(computed - expected_val) < tolerance
    
    def test_is_orthogonal(self):
        rotation_matrix = ArbitraryMatrix([
            [RationalListNumber.from_float(0.6), RationalListNumber.from_float(-0.8)],
            [RationalListNumber.from_float(0.8), RationalListNumber.from_float(0.6)]
        ])
        
        assert MatrixOperations.is_orthogonal(rotation_matrix, 
                                            RationalListNumber.from_fraction(1, 100))
        
        non_orthogonal = ArbitraryMatrix([[1, 2], [3, 4]])
        assert not MatrixOperations.is_orthogonal(non_orthogonal)
    
    def test_is_symmetric(self):
        symmetric = ArbitraryMatrix([[1, 2, 3], [2, 4, 5], [3, 5, 6]])
        assert MatrixOperations.is_symmetric(symmetric)
        
        non_symmetric = ArbitraryMatrix([[1, 2], [3, 4]])
        assert not MatrixOperations.is_symmetric(non_symmetric)
    
    def test_condition_number(self):
        well_conditioned = ArbitraryMatrix.identity(2)
        cond_num = MatrixOperations.condition_number(well_conditioned)
        
        assert abs(float(cond_num.evaluate_exact()) - 1.0) < 1e-10
        
        ill_conditioned = ArbitraryMatrix([[1, 1], [1, 1.0001]])
        cond_num_ill = MatrixOperations.condition_number(ill_conditioned)
        
        assert float(cond_num_ill.evaluate_exact()) > 1000


class TestMatrixErrorHandling:
    """Test error handling in matrix operations."""
    
    def test_non_square_determinant(self):
        A = ArbitraryMatrix([[1, 2, 3], [4, 5, 6]])
        with pytest.raises(ValueError):
            A.determinant()
    
    def test_non_square_inverse(self):
        A = ArbitraryMatrix([[1, 2, 3], [4, 5, 6]])
        with pytest.raises(ValueError):
            A.inverse()
    
    def test_non_square_trace(self):
        A = ArbitraryMatrix([[1, 2, 3], [4, 5, 6]])
        with pytest.raises(ValueError):
            A.trace()
    
    def test_incompatible_matrix_multiplication(self):
        A = ArbitraryMatrix([[1, 2]])  # 1x2
        B = ArbitraryMatrix([[1, 2, 3]])  # 1x3
        
        with pytest.raises(ValueError):
            A * B
    
    def test_incompatible_hadamard_product(self):
        A = ArbitraryMatrix([[1, 2]])
        B = ArbitraryMatrix([[1], [2]])
        
        with pytest.raises(ValueError):
            MatrixOperations.hadamard_product(A, B)
    
    def test_non_square_eigenvalues(self):
        A = ArbitraryMatrix([[1, 2, 3], [4, 5, 6]])
        with pytest.raises(ValueError):
            A.eigenvalues_2x2()
    
    def test_wrong_size_eigenvalues(self):
        A = ArbitraryMatrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        with pytest.raises(ValueError):
            A.eigenvalues_2x2()


class TestMatrixPrecisionPreservation:
    """Test that matrix operations preserve exact precision."""
    
    def test_exact_rational_arithmetic(self):
        A = ArbitraryMatrix.from_rational_fractions([[1, 2]], [[3, 5]])
        B = ArbitraryMatrix.from_rational_fractions([[4, 6]], [[7, 11]])
        
        C = A + B
        
        expected_00 = RationalListNumber.from_fraction(1, 3) + RationalListNumber.from_fraction(4, 7)
        expected_01 = RationalListNumber.from_fraction(2, 5) + RationalListNumber.from_fraction(6, 11)
        
        assert C[0, 0].evaluate_exact() == expected_00.evaluate_exact()
        assert C[0, 1].evaluate_exact() == expected_01.evaluate_exact()
    
    def test_no_precision_loss_in_operations(self):
        third = RationalListNumber.from_fraction(1, 3)
        A = ArbitraryMatrix([[third, third], [third, third]])
        
        B = A * 3
        
        for i in range(2):
            for j in range(2):
                assert B[i, j].evaluate_exact() == 1.0
    
    def test_determinant_exact_computation(self):
        A = ArbitraryMatrix.from_rational_fractions([[1, 2], [3, 4]], [[2, 3], [4, 5]])
        det = A.determinant()
        
        a = RationalListNumber.from_fraction(1, 2)
        b = RationalListNumber.from_fraction(2, 3)
        c = RationalListNumber.from_fraction(3, 4)
        d = RationalListNumber.from_fraction(4, 5)
        
        expected_det = a * d - b * c
        
        assert det.evaluate_exact() == expected_det.evaluate_exact()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
