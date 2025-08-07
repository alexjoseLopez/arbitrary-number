#!/usr/bin/env python3
"""
Intermediate-level ML Algorithm Precision Tests for ArbitraryNumber V2
Tests specific ML algorithms with focus on precision verification
"""

import unittest
import sys
import os
import time
import math

# Add the v2 module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from v2.core.arbitrary_number import ArbitraryNumber


class TestMLAlgorithmsPrecision(unittest.TestCase):
    """Test ML algorithms precision with ArbitraryNumber vs floating-point"""

    def test_gradient_descent_step_precision(self):
        """Test single gradient descent step precision"""
        # Parameters
        x = ArbitraryNumber("0.5")
        learning_rate = ArbitraryNumber("0.01")
        gradient = ArbitraryNumber("0.1")
        
        # Single step: x_new = x - learning_rate * gradient
        x_new = x - learning_rate * gradient
        
        # Expected: 0.5 - 0.01 * 0.1 = 0.5 - 0.001 = 0.499
        expected = ArbitraryNumber("0.499")
        
        self.assertEqual(x_new, expected)
        self.assertEqual(str(x_new), "0.499")

    def test_momentum_calculation_precision(self):
        """Test momentum calculation in SGD"""
        # Previous momentum
        momentum = ArbitraryNumber("0.1")
        # Momentum coefficient
        beta = ArbitraryNumber("0.9")
        # Current gradient
        gradient = ArbitraryNumber("0.05")
        
        # New momentum = beta * momentum + gradient
        new_momentum = beta * momentum + gradient
        
        # Expected: 0.9 * 0.1 + 0.05 = 0.09 + 0.05 = 0.14
        expected = ArbitraryNumber("0.14")
        
        self.assertEqual(new_momentum, expected)

    def test_adam_first_moment_precision(self):
        """Test Adam optimizer first moment calculation"""
        # Previous first moment
        m_prev = ArbitraryNumber("0.01")
        # Beta1 parameter
        beta1 = ArbitraryNumber("0.9")
        # Current gradient
        gradient = ArbitraryNumber("0.1")
        
        # First moment: m = beta1 * m_prev + (1 - beta1) * gradient
        one_minus_beta1 = ArbitraryNumber("1") - beta1
        m = beta1 * m_prev + one_minus_beta1 * gradient
        
        # Expected: 0.9 * 0.01 + 0.1 * 0.1 = 0.009 + 0.01 = 0.019
        expected = ArbitraryNumber("0.019")
        
        self.assertEqual(m, expected)

    def test_adam_second_moment_precision(self):
        """Test Adam optimizer second moment calculation"""
        # Previous second moment
        v_prev = ArbitraryNumber("0.001")
        # Beta2 parameter
        beta2 = ArbitraryNumber("0.999")
        # Current gradient
        gradient = ArbitraryNumber("0.1")
        
        # Second moment: v = beta2 * v_prev + (1 - beta2) * gradient^2
        one_minus_beta2 = ArbitraryNumber("1") - beta2
        gradient_squared = gradient * gradient
        v = beta2 * v_prev + one_minus_beta2 * gradient_squared
        
        # Expected: 0.999 * 0.001 + 0.001 * 0.01 = 0.000999 + 0.00001 = 0.001009
        expected = ArbitraryNumber("0.001009")
        
        self.assertEqual(v, expected)

    def test_learning_rate_decay_precision(self):
        """Test exponential learning rate decay"""
        # Initial learning rate
        lr_initial = ArbitraryNumber("0.1")
        # Decay rate
        decay_rate = ArbitraryNumber("0.95")
        # Step
        step = 5
        
        # Exponential decay: lr = lr_initial * decay_rate^step
        lr_decayed = lr_initial * (decay_rate ** step)
        
        # Check that the result is reasonable (between 0.07 and 0.08)
        self.assertTrue(lr_decayed > ArbitraryNumber("0.07"))
        self.assertTrue(lr_decayed < ArbitraryNumber("0.08"))

    def test_batch_normalization_mean_precision(self):
        """Test batch normalization mean calculation"""
        # Batch values
        values = [
            ArbitraryNumber("0.1"),
            ArbitraryNumber("0.2"),
            ArbitraryNumber("0.3"),
            ArbitraryNumber("0.4"),
            ArbitraryNumber("0.5")
        ]
        
        # Calculate mean
        total = sum(values)
        mean = total / ArbitraryNumber("5")
        
        # Expected: (0.1 + 0.2 + 0.3 + 0.4 + 0.5) / 5 = 1.5 / 5 = 0.3
        expected = ArbitraryNumber("0.3")
        
        self.assertEqual(mean, expected)

    def test_batch_normalization_variance_precision(self):
        """Test batch normalization variance calculation"""
        # Batch values
        values = [
            ArbitraryNumber("0.1"),
            ArbitraryNumber("0.2"),
            ArbitraryNumber("0.3"),
            ArbitraryNumber("0.4"),
            ArbitraryNumber("0.5")
        ]
        
        # Calculate mean
        mean = ArbitraryNumber("0.3")
        
        # Calculate variance: sum((x - mean)^2) / n
        variance_sum = ArbitraryNumber("0")
        for value in values:
            diff = value - mean
            variance_sum += diff * diff
        
        variance = variance_sum / ArbitraryNumber("5")
        
        # Expected: ((0.1-0.3)^2 + (0.2-0.3)^2 + (0.3-0.3)^2 + (0.4-0.3)^2 + (0.5-0.3)^2) / 5
        # = (0.04 + 0.01 + 0 + 0.01 + 0.04) / 5 = 0.1 / 5 = 0.02
        expected = ArbitraryNumber("0.02")
        
        self.assertEqual(variance, expected)

    def test_softmax_numerator_precision(self):
        """Test softmax numerator calculation"""
        # Input value
        x = ArbitraryNumber("2.0")
        
        # Calculate exp(x) using Taylor series approximation
        # exp(x) ≈ 1 + x + x^2/2! + x^3/3! + x^4/4! + ...
        exp_x = ArbitraryNumber("1")  # First term
        term = ArbitraryNumber("1")
        
        for i in range(1, 10):  # Use first 10 terms
            term = term * x / ArbitraryNumber(str(i))
            exp_x += term
        
        # exp(2) ≈ 7.389 (we expect reasonable approximation)
        self.assertTrue(ArbitraryNumber("7") < exp_x < ArbitraryNumber("8"))

    def test_cross_entropy_loss_precision(self):
        """Test cross-entropy loss calculation"""
        # Predicted probability
        p = ArbitraryNumber("0.8")
        # True label (1 for positive class)
        y = ArbitraryNumber("1")
        
        # Cross-entropy: -y * log(p) - (1-y) * log(1-p)
        # For y=1: -log(p)
        # We'll use natural log approximation: ln(x) ≈ (x-1) - (x-1)^2/2 + (x-1)^3/3 - ...
        # For x close to 1, ln(x) ≈ x - 1
        
        # Simplified approximation for demonstration
        log_p_approx = p - ArbitraryNumber("1")  # ln(0.8) ≈ -0.2 (rough approximation)
        loss = -y * log_p_approx
        
        # This is a simplified test - in practice we'd use more accurate log calculation
        self.assertTrue(loss > ArbitraryNumber("0"))

    def test_weight_update_accumulation_precision(self):
        """Test weight update accumulation over multiple steps"""
        # Initial weight
        weight = ArbitraryNumber("1.0")
        learning_rate = ArbitraryNumber("0.01")
        
        # Apply 100 small updates
        for i in range(100):
            gradient = ArbitraryNumber("0.001")
            weight = weight - learning_rate * gradient
        
        # Expected: 1.0 - 100 * 0.01 * 0.001 = 1.0 - 0.001 = 0.999
        expected = ArbitraryNumber("0.999")
        
        self.assertEqual(weight, expected)

    def test_matrix_multiplication_precision(self):
        """Test matrix multiplication precision for neural networks"""
        # 2x2 matrix multiplication
        # A = [[0.1, 0.2], [0.3, 0.4]]
        # B = [[0.5, 0.6], [0.7, 0.8]]
        # C = A * B
        
        # Calculate C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
        c_00 = ArbitraryNumber("0.1") * ArbitraryNumber("0.5") + ArbitraryNumber("0.2") * ArbitraryNumber("0.7")
        
        # Expected: 0.1 * 0.5 + 0.2 * 0.7 = 0.05 + 0.14 = 0.19
        expected = ArbitraryNumber("0.19")
        
        self.assertEqual(c_00, expected)

    def test_activation_function_precision(self):
        """Test activation function calculations"""
        # Test ReLU
        x_positive = ArbitraryNumber("0.5")
        x_negative = ArbitraryNumber("-0.3")
        
        # ReLU(x) = max(0, x)
        relu_positive = max(ArbitraryNumber("0"), x_positive)
        relu_negative = max(ArbitraryNumber("0"), x_negative)
        
        self.assertEqual(relu_positive, ArbitraryNumber("0.5"))
        self.assertEqual(relu_negative, ArbitraryNumber("0"))

    def test_dropout_mask_precision(self):
        """Test dropout mask application"""
        # Original value
        x = ArbitraryNumber("0.8")
        # Dropout probability (keep probability = 0.7)
        keep_prob = ArbitraryNumber("0.7")
        
        # During training, scale by 1/keep_prob
        scaling_factor = ArbitraryNumber("1") / keep_prob
        x_scaled = x * scaling_factor
        
        # Expected: 0.8 / 0.7 = 8/7 ≈ 1.142857...
        # Check that the result is reasonable
        self.assertTrue(x_scaled > ArbitraryNumber("1.14"))
        self.assertTrue(x_scaled < ArbitraryNumber("1.15"))
        
        # Also verify the exact calculation
        expected = ArbitraryNumber("8") / ArbitraryNumber("7")
        self.assertEqual(x_scaled, expected)

    def test_l2_regularization_precision(self):
        """Test L2 regularization calculation"""
        # Weight
        w = ArbitraryNumber("0.5")
        # L2 regularization parameter
        lambda_reg = ArbitraryNumber("0.01")
        
        # L2 penalty: lambda * w^2
        l2_penalty = lambda_reg * w * w
        
        # Expected: 0.01 * 0.5^2 = 0.01 * 0.25 = 0.0025
        expected = ArbitraryNumber("0.0025")
        
        self.assertEqual(l2_penalty, expected)

    def test_precision_vs_floating_point(self):
        """Test precision advantage over floating-point"""
        # Perform calculation that loses precision in floating-point
        x = ArbitraryNumber("0.1")
        
        # Add 0.1 ten times
        for i in range(10):
            x += ArbitraryNumber("0.1")
        
        # Should equal exactly 1.1
        expected = ArbitraryNumber("1.1")
        self.assertEqual(x, expected)
        
        # Compare with floating-point
        x_float = 0.1
        for i in range(10):
            x_float += 0.1
        
        # Floating-point will have precision errors
        self.assertNotEqual(x_float, 1.1)  # This will likely fail due to floating-point errors


class TestMLPerformanceMetrics(unittest.TestCase):
    """Test ML performance metrics calculations"""

    def test_accuracy_calculation_precision(self):
        """Test accuracy metric calculation"""
        # Correct predictions
        correct = ArbitraryNumber("85")
        # Total predictions
        total = ArbitraryNumber("100")
        
        # Accuracy = correct / total
        accuracy = correct / total
        
        # Expected: 85/100 = 0.85
        expected = ArbitraryNumber("0.85")
        
        self.assertEqual(accuracy, expected)

    def test_precision_metric_calculation(self):
        """Test precision metric calculation"""
        # True positives
        tp = ArbitraryNumber("20")
        # False positives
        fp = ArbitraryNumber("5")
        
        # Precision = TP / (TP + FP)
        precision = tp / (tp + fp)
        
        # Expected: 20 / (20 + 5) = 20/25 = 0.8
        expected = ArbitraryNumber("0.8")
        
        self.assertEqual(precision, expected)

    def test_recall_metric_calculation(self):
        """Test recall metric calculation"""
        # True positives
        tp = ArbitraryNumber("20")
        # False negatives
        fn = ArbitraryNumber("10")
        
        # Recall = TP / (TP + FN)
        recall = tp / (tp + fn)
        
        # Expected: 20 / (20 + 10) = 20/30 = 2/3
        expected = ArbitraryNumber("2") / ArbitraryNumber("3")
        
        self.assertEqual(recall, expected)

    def test_f1_score_calculation(self):
        """Test F1 score calculation"""
        # Precision and recall
        precision = ArbitraryNumber("0.8")
        recall = ArbitraryNumber("2") / ArbitraryNumber("3")  # 2/3
        
        # F1 = 2 * (precision * recall) / (precision + recall)
        numerator = ArbitraryNumber("2") * precision * recall
        denominator = precision + recall
        f1_score = numerator / denominator
        
        # Expected calculation can be verified
        self.assertTrue(f1_score > ArbitraryNumber("0.7"))
        self.assertTrue(f1_score < ArbitraryNumber("0.8"))


if __name__ == '__main__':
    print("=" * 60)
    print("ARBITRARYNUMBER V2 ML ALGORITHMS PRECISION TESTS")
    print("=" * 60)
    print("Testing intermediate-level ML algorithm precision...")
    print()
    
    # Run the tests
    unittest.main(verbosity=2)
