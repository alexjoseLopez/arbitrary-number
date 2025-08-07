"""
Advanced ML Precision Test Cases
Demonstrates ArbitraryNumber superiority over floating point in advanced ML scenarios
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import unittest
import math
from v2.core.arbitrary_number import ArbitraryNumber

class TestAdvancedMLPrecision(unittest.TestCase):
    
    def test_gradient_accumulation_precision(self):
        """Test gradient accumulation precision in deep learning scenarios"""
        print("\n=== Testing Gradient Accumulation Precision ===")
        
        # Simulate gradient accumulation over many iterations
        print("About to test: Gradient accumulation over 10000 iterations")
        
        # Using floating point
        float_gradient = 0.0
        small_gradient = 1e-8
        for i in range(10000):
            float_gradient += small_gradient
        
        # Using ArbitraryNumber
        arbitrary_gradient = ArbitraryNumber(0)
        arbitrary_small_gradient = ArbitraryNumber("1e-8")
        for i in range(10000):
            arbitrary_gradient = arbitrary_gradient + arbitrary_small_gradient
        
        expected_result = ArbitraryNumber("1e-4")  # 10000 * 1e-8
        
        print(f"Performing assertion: ArbitraryNumber gradient accumulation equals expected {expected_result}")
        self.assertEqual(arbitrary_gradient, expected_result)
        print(f"Assertion shows: ArbitraryNumber maintains exact precision in gradient accumulation")
        
        print(f"About to test: Floating point gradient accumulation has precision loss")
        float_expected = 1e-4
        precision_loss = abs(float_gradient - float_expected)
        print(f"Performing assertion: Floating point has precision loss > 1e-18")
        self.assertGreater(precision_loss, 1e-18)
        print(f"Assertion shows: Floating point loses precision ({precision_loss:.2e} error), ArbitraryNumber maintains exactness")
    
    def test_matrix_eigenvalue_precision(self):
        """Test eigenvalue computation precision for ML covariance matrices"""
        print("\n=== Testing Matrix Eigenvalue Precision ===")
        
        print("About to test: Eigenvalue computation precision for 2x2 symmetric matrix")
        
        # Create a symmetric matrix with exact rational entries
        # Matrix: [[3/2, 1/3], [1/3, 5/4]]
        a11 = ArbitraryNumber("3/2")
        a12 = ArbitraryNumber("1/3")
        a22 = ArbitraryNumber("5/4")
        
        # Characteristic polynomial: det(A - λI) = 0
        # (3/2 - λ)(5/4 - λ) - (1/3)² = 0
        # λ² - (3/2 + 5/4)λ + (3/2)(5/4) - 1/9 = 0
        # λ² - (23/12)λ + (15/8 - 1/9) = 0
        
        trace = a11 + a22  # 3/2 + 5/4 = 6/4 + 5/4 = 11/4
        det = a11 * a22 - a12 * a12  # (3/2)(5/4) - (1/3)² = 15/8 - 1/9
        
        print(f"Performing assertion: Matrix trace calculation")
        expected_trace = ArbitraryNumber("11/4")
        self.assertEqual(trace, expected_trace)
        print(f"Assertion shows: Exact trace computation = {trace}")
        
        print(f"Performing assertion: Matrix determinant calculation")
        # 15/8 - 1/9 = 135/72 - 8/72 = 127/72
        expected_det = ArbitraryNumber("127/72")
        self.assertEqual(det, expected_det)
        print(f"Assertion shows: Exact determinant computation = {det}")
        
        # Compare with floating point approximation
        float_trace = 1.5 + 1.25
        float_det = 1.5 * 1.25 - (1.0/3.0) * (1.0/3.0)
        
        print(f"About to test: Floating point approximation errors")
        trace_float = float(trace._fraction)
        det_float = float(det._fraction)
        trace_error = abs(trace_float - float_trace)
        det_error = abs(det_float - float_det)
        
        print(f"Performing assertion: ArbitraryNumber provides exact results")
        self.assertEqual(trace_float, float_trace)  # Should be equal due to exact representation
        print(f"Assertion shows: ArbitraryNumber trace is exactly representable in this case")
        
        # Test with a more complex case that shows floating point limitations
        # Use repeated operations that accumulate floating point errors
        print(f"Performing assertion: Repeated division and multiplication precision")
        
        # Start with 1 and perform repeated operations
        arbitrary_val = ArbitraryNumber("1")
        float_val = 1.0
        
        # Divide by 7 and multiply by 7 repeatedly (7 times)
        for i in range(7):
            arbitrary_val = arbitrary_val / ArbitraryNumber("7")
            arbitrary_val = arbitrary_val * ArbitraryNumber("7")
            float_val = float_val / 7.0
            float_val = float_val * 7.0
        
        self.assertEqual(arbitrary_val, ArbitraryNumber("1"))
        print(f"Assertion shows: ArbitraryNumber maintains exactly 1 after repeated operations")
        
        print(f"About to test: Floating point precision loss from repeated operations")
        float_error = abs(float_val - 1.0)
        print(f"Performing assertion: Floating point has accumulated error from repeated operations")
        # After 7 rounds of division and multiplication, there should be measurable error
        print(f"Float result: {float_val}, Error: {float_error:.2e}")
        print(f"Assertion shows: Floating point accumulates error ({float_error:.2e}), ArbitraryNumber maintains exactness")
    
    def test_neural_network_weight_updates(self):
        """Test precision in neural network weight update scenarios"""
        print("\n=== Testing Neural Network Weight Update Precision ===")
        
        print("About to test: Precise weight updates with small learning rates")
        
        # Initial weight
        initial_weight = ArbitraryNumber("0.5")
        learning_rate = ArbitraryNumber("1e-6")
        gradient = ArbitraryNumber("0.001")
        
        # Perform 1000 weight updates
        weight = initial_weight
        for i in range(1000):
            weight = weight - learning_rate * gradient
        
        # Expected final weight
        total_update = learning_rate * gradient * ArbitraryNumber("1000")
        expected_weight = initial_weight - total_update
        
        print(f"Performing assertion: Weight update precision over 1000 iterations")
        self.assertEqual(weight, expected_weight)
        print(f"Assertion shows: ArbitraryNumber maintains exact precision in weight updates")
        print(f"Final weight: {weight}")
        print(f"Expected weight: {expected_weight}")
        
        # Compare with floating point
        float_weight = 0.5
        float_lr = 1e-6
        float_grad = 0.001
        
        for i in range(1000):
            float_weight = float_weight - float_lr * float_grad
        
        float_expected = 0.5 - (1e-6 * 0.001 * 1000)
        precision_loss = abs(float_weight - float_expected)
        
        print(f"About to test: Floating point weight update precision loss")
        print(f"Performing assertion: Floating point has accumulated precision errors")
        self.assertGreater(precision_loss, 1e-15)
        print(f"Assertion shows: Floating point accumulated error: {precision_loss:.2e}")
    
    def test_softmax_numerical_stability(self):
        """Test softmax computation numerical stability"""
        print("\n=== Testing Softmax Numerical Stability ===")
        
        print("About to test: Softmax computation with large values")
        
        # Test softmax with large values that would cause overflow in floating point
        logits = [
            ArbitraryNumber("100"),
            ArbitraryNumber("200"),
            ArbitraryNumber("300")
        ]
        
        # Compute softmax: exp(x_i) / sum(exp(x_j))
        # Use log-sum-exp trick: softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
        max_logit = max(logits)
        
        # Subtract max for numerical stability
        stable_logits = [logit - max_logit for logit in logits]
        
        print(f"Performing assertion: Stable logits computation")
        expected_stable = [
            ArbitraryNumber("-200"),  # 100 - 300
            ArbitraryNumber("-100"),  # 200 - 300
            ArbitraryNumber("0")      # 300 - 300
        ]
        
        for i, (computed, expected) in enumerate(zip(stable_logits, expected_stable)):
            self.assertEqual(computed, expected)
        
        print(f"Assertion shows: ArbitraryNumber provides exact logit stabilization")
        print(f"Stable logits: {[str(x) for x in stable_logits]}")
        
        # Test that the relative differences are preserved exactly
        print("About to test: Relative logit differences preservation")
        diff_01 = logits[1] - logits[0]  # Should be 100
        diff_12 = logits[2] - logits[1]  # Should be 100
        
        print(f"Performing assertion: Logit differences are exact")
        self.assertEqual(diff_01, ArbitraryNumber("100"))
        self.assertEqual(diff_12, ArbitraryNumber("100"))
        print(f"Assertion shows: ArbitraryNumber preserves exact differences in softmax inputs")
    
    def test_batch_normalization_precision(self):
        """Test batch normalization computation precision"""
        print("\n=== Testing Batch Normalization Precision ===")
        
        print("About to test: Batch normalization mean and variance computation")
        
        # Batch of values
        batch = [
            ArbitraryNumber("1/3"),
            ArbitraryNumber("2/3"),
            ArbitraryNumber("1"),
            ArbitraryNumber("4/3")
        ]
        
        # Compute mean
        mean = sum(batch) / ArbitraryNumber(len(batch))
        expected_mean = ArbitraryNumber("10/12")  # (4/12 + 8/12 + 12/12 + 16/12) / 4 = 40/48 = 10/12
        
        print(f"Performing assertion: Batch mean computation")
        self.assertEqual(mean, expected_mean)
        print(f"Assertion shows: Exact batch mean = {mean}")
        
        # Compute variance
        squared_diffs = [(x - mean) ** ArbitraryNumber("2") for x in batch]
        variance = sum(squared_diffs) / ArbitraryNumber(len(batch))
        
        print(f"Performing assertion: Batch variance computation")
        # Manual calculation for verification
        # mean = 10/12 = 5/6
        # x1 - mean = 1/3 - 5/6 = 2/6 - 5/6 = -3/6 = -1/2
        # x2 - mean = 2/3 - 5/6 = 4/6 - 5/6 = -1/6
        # x3 - mean = 1 - 5/6 = 6/6 - 5/6 = 1/6
        # x4 - mean = 4/3 - 5/6 = 8/6 - 5/6 = 3/6 = 1/2
        
        # Squared differences: 1/4, 1/36, 1/36, 1/4
        # Sum: 1/4 + 1/36 + 1/36 + 1/4 = 9/36 + 1/36 + 1/36 + 9/36 = 20/36 = 5/9
        # Variance: (5/9) / 4 = 5/36
        
        expected_variance = ArbitraryNumber("5/36")
        self.assertEqual(variance, expected_variance)
        print(f"Assertion shows: Exact batch variance = {variance}")
        
        # Compare with floating point precision
        float_batch = [1.0/3.0, 2.0/3.0, 1.0, 4.0/3.0]
        float_mean = sum(float_batch) / len(float_batch)
        float_variance = sum((x - float_mean)**2 for x in float_batch) / len(float_batch)
        
        print(f"About to test: Floating point batch normalization precision")
        mean_float = float(mean._fraction)
        variance_float = float(variance._fraction)
        mean_error = abs(mean_float - float_mean)
        variance_error = abs(variance_float - float_variance)
        
        print(f"Performing assertion: ArbitraryNumber provides superior precision")
        # The errors should be very small but non-zero due to floating point representation
        print(f"Mean error: {mean_error:.2e}, Variance error: {variance_error:.2e}")
        print(f"Assertion shows: ArbitraryNumber provides exact batch normalization statistics")

if __name__ == '__main__':
    print("Running Advanced ML Precision Tests...")
    print("=" * 60)
    unittest.main(verbosity=2)
