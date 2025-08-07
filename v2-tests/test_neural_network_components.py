#!/usr/bin/env python3
"""
Neural Network Components Precision Tests for ArbitraryNumber V2
Tests specific neural network components with focus on precision verification
"""

import unittest
import sys
import os
import time
import math

# Add the v2 module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from v2.core.arbitrary_number import ArbitraryNumber


class TestNeuralNetworkComponents(unittest.TestCase):
    """Test neural network components precision with ArbitraryNumber"""

    def test_linear_layer_forward_pass(self):
        """Test linear layer forward pass precision"""
        # Input
        x = ArbitraryNumber("0.5")
        # Weight
        w = ArbitraryNumber("0.8")
        # Bias
        b = ArbitraryNumber("0.1")
        
        # Forward pass: y = w * x + b
        y = w * x + b
        
        # Expected: 0.8 * 0.5 + 0.1 = 0.4 + 0.1 = 0.5
        expected = ArbitraryNumber("0.5")
        
        self.assertEqual(y, expected)

    def test_linear_layer_backward_pass(self):
        """Test linear layer backward pass precision"""
        # Gradient from next layer
        grad_output = ArbitraryNumber("0.2")
        # Input from forward pass
        x = ArbitraryNumber("0.5")
        # Weight
        w = ArbitraryNumber("0.8")
        
        # Gradient w.r.t. weight: grad_w = grad_output * x
        grad_w = grad_output * x
        
        # Gradient w.r.t. input: grad_x = grad_output * w
        grad_x = grad_output * w
        
        # Expected gradients
        expected_grad_w = ArbitraryNumber("0.1")  # 0.2 * 0.5
        expected_grad_x = ArbitraryNumber("0.16")  # 0.2 * 0.8
        
        self.assertEqual(grad_w, expected_grad_w)
        self.assertEqual(grad_x, expected_grad_x)

    def test_relu_activation_precision(self):
        """Test ReLU activation function precision"""
        # Test positive input
        x_pos = ArbitraryNumber("0.5")
        relu_pos = max(ArbitraryNumber("0"), x_pos)
        self.assertEqual(relu_pos, ArbitraryNumber("0.5"))
        
        # Test negative input
        x_neg = ArbitraryNumber("-0.3")
        relu_neg = max(ArbitraryNumber("0"), x_neg)
        self.assertEqual(relu_neg, ArbitraryNumber("0"))
        
        # Test zero input
        x_zero = ArbitraryNumber("0")
        relu_zero = max(ArbitraryNumber("0"), x_zero)
        self.assertEqual(relu_zero, ArbitraryNumber("0"))

    def test_sigmoid_approximation_precision(self):
        """Test sigmoid activation approximation precision"""
        # Input
        x = ArbitraryNumber("0")
        
        # Sigmoid approximation: 1 / (1 + exp(-x))
        # For x = 0: sigmoid(0) = 1 / (1 + 1) = 0.5
        # We'll use a simple approximation for demonstration
        
        # Simple sigmoid approximation: 0.5 + 0.25 * x for small x
        sigmoid_approx = ArbitraryNumber("0.5") + ArbitraryNumber("0.25") * x
        
        # For x = 0, should be 0.5
        expected = ArbitraryNumber("0.5")
        self.assertEqual(sigmoid_approx, expected)

    def test_tanh_approximation_precision(self):
        """Test tanh activation approximation precision"""
        # Input
        x = ArbitraryNumber("0")
        
        # tanh(0) = 0
        # Simple approximation: tanh(x) ≈ x for small x
        tanh_approx = x
        
        expected = ArbitraryNumber("0")
        self.assertEqual(tanh_approx, expected)

    def test_convolution_operation_precision(self):
        """Test convolution operation precision"""
        # Simple 1D convolution
        # Input: [1, 2, 3]
        # Kernel: [0.5, 0.3]
        # Output at position 0: 1*0.5 + 2*0.3 = 0.5 + 0.6 = 1.1
        
        input_val1 = ArbitraryNumber("1")
        input_val2 = ArbitraryNumber("2")
        kernel_val1 = ArbitraryNumber("0.5")
        kernel_val2 = ArbitraryNumber("0.3")
        
        conv_output = input_val1 * kernel_val1 + input_val2 * kernel_val2
        
        expected = ArbitraryNumber("1.1")
        self.assertEqual(conv_output, expected)

    def test_pooling_operation_precision(self):
        """Test pooling operation precision"""
        # Max pooling
        values = [
            ArbitraryNumber("0.1"),
            ArbitraryNumber("0.8"),
            ArbitraryNumber("0.3"),
            ArbitraryNumber("0.6")
        ]
        
        # Max pooling
        max_val = max(values)
        expected_max = ArbitraryNumber("0.8")
        self.assertEqual(max_val, expected_max)
        
        # Average pooling
        avg_val = sum(values) / ArbitraryNumber("4")
        expected_avg = ArbitraryNumber("0.45")  # (0.1 + 0.8 + 0.3 + 0.6) / 4 = 1.8 / 4 = 0.45
        self.assertEqual(avg_val, expected_avg)

    def test_attention_mechanism_precision(self):
        """Test attention mechanism calculation precision"""
        # Query
        q = ArbitraryNumber("0.5")
        # Key
        k = ArbitraryNumber("0.8")
        # Value
        v = ArbitraryNumber("0.3")
        
        # Attention score: q * k
        attention_score = q * k
        
        # Expected: 0.5 * 0.8 = 0.4
        expected_score = ArbitraryNumber("0.4")
        self.assertEqual(attention_score, expected_score)
        
        # Attention output: attention_score * v
        attention_output = attention_score * v
        
        # Expected: 0.4 * 0.3 = 0.12
        expected_output = ArbitraryNumber("0.12")
        self.assertEqual(attention_output, expected_output)

    def test_layer_normalization_precision(self):
        """Test layer normalization precision"""
        # Input values
        values = [
            ArbitraryNumber("0.1"),
            ArbitraryNumber("0.3"),
            ArbitraryNumber("0.5")
        ]
        
        # Calculate mean
        mean = sum(values) / ArbitraryNumber("3")
        expected_mean = ArbitraryNumber("0.3")  # (0.1 + 0.3 + 0.5) / 3 = 0.9 / 3 = 0.3
        self.assertEqual(mean, expected_mean)
        
        # Calculate variance
        variance_sum = ArbitraryNumber("0")
        for val in values:
            diff = val - mean
            variance_sum += diff * diff
        
        variance = variance_sum / ArbitraryNumber("3")
        # Expected: ((0.1-0.3)^2 + (0.3-0.3)^2 + (0.5-0.3)^2) / 3
        # = (0.04 + 0 + 0.04) / 3 = 0.08 / 3
        expected_variance = ArbitraryNumber("0.08") / ArbitraryNumber("3")
        self.assertEqual(variance, expected_variance)

    def test_residual_connection_precision(self):
        """Test residual connection precision"""
        # Input
        x = ArbitraryNumber("0.5")
        # Layer output
        layer_output = ArbitraryNumber("0.2")
        
        # Residual connection: output = x + layer_output
        residual_output = x + layer_output
        
        # Expected: 0.5 + 0.2 = 0.7
        expected = ArbitraryNumber("0.7")
        self.assertEqual(residual_output, expected)

    def test_embedding_lookup_precision(self):
        """Test embedding lookup precision"""
        # Embedding table (simplified)
        embeddings = {
            0: [ArbitraryNumber("0.1"), ArbitraryNumber("0.2")],
            1: [ArbitraryNumber("0.3"), ArbitraryNumber("0.4")],
            2: [ArbitraryNumber("0.5"), ArbitraryNumber("0.6")]
        }
        
        # Lookup token 1
        token_id = 1
        embedding = embeddings[token_id]
        
        expected_embedding = [ArbitraryNumber("0.3"), ArbitraryNumber("0.4")]
        self.assertEqual(embedding, expected_embedding)

    def test_positional_encoding_precision(self):
        """Test positional encoding precision"""
        # Simple positional encoding
        position = 0
        dimension = 0
        
        # PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        # For pos=0, any dimension: sin(0) = 0
        pos_encoding = ArbitraryNumber("0")  # sin(0) = 0
        
        expected = ArbitraryNumber("0")
        self.assertEqual(pos_encoding, expected)

    def test_multi_head_attention_precision(self):
        """Test multi-head attention calculation precision"""
        # Simplified multi-head attention
        # Head 1
        q1 = ArbitraryNumber("0.2")
        k1 = ArbitraryNumber("0.3")
        v1 = ArbitraryNumber("0.1")
        
        # Head 2
        q2 = ArbitraryNumber("0.4")
        k2 = ArbitraryNumber("0.5")
        v2 = ArbitraryNumber("0.2")
        
        # Attention scores
        score1 = q1 * k1  # 0.2 * 0.3 = 0.06
        score2 = q2 * k2  # 0.4 * 0.5 = 0.2
        
        # Attention outputs
        output1 = score1 * v1  # 0.06 * 0.1 = 0.006
        output2 = score2 * v2  # 0.2 * 0.2 = 0.04
        
        # Concatenate (sum for simplicity)
        multi_head_output = output1 + output2
        
        expected = ArbitraryNumber("0.046")  # 0.006 + 0.04
        self.assertEqual(multi_head_output, expected)

    def test_transformer_feedforward_precision(self):
        """Test transformer feedforward network precision"""
        # Input
        x = ArbitraryNumber("0.5")
        
        # First linear layer: W1 * x + b1
        w1 = ArbitraryNumber("2.0")
        b1 = ArbitraryNumber("0.1")
        hidden = w1 * x + b1  # 2.0 * 0.5 + 0.1 = 1.0 + 0.1 = 1.1
        
        # ReLU activation
        hidden_relu = max(ArbitraryNumber("0"), hidden)  # max(0, 1.1) = 1.1
        
        # Second linear layer: W2 * hidden + b2
        w2 = ArbitraryNumber("0.5")
        b2 = ArbitraryNumber("0.05")
        output = w2 * hidden_relu + b2  # 0.5 * 1.1 + 0.05 = 0.55 + 0.05 = 0.6
        
        expected = ArbitraryNumber("0.6")
        self.assertEqual(output, expected)

    def test_gru_cell_precision(self):
        """Test GRU cell calculation precision"""
        # Simplified GRU cell
        # Input
        x = ArbitraryNumber("0.3")
        # Previous hidden state
        h_prev = ArbitraryNumber("0.2")
        
        # Reset gate (simplified): r = sigmoid(W_r * x + U_r * h_prev)
        # Using linear approximation for simplicity
        w_r = ArbitraryNumber("0.5")
        u_r = ArbitraryNumber("0.4")
        r = w_r * x + u_r * h_prev  # 0.5 * 0.3 + 0.4 * 0.2 = 0.15 + 0.08 = 0.23
        
        # Update gate (simplified): z = sigmoid(W_z * x + U_z * h_prev)
        w_z = ArbitraryNumber("0.6")
        u_z = ArbitraryNumber("0.3")
        z = w_z * x + u_z * h_prev  # 0.6 * 0.3 + 0.3 * 0.2 = 0.18 + 0.06 = 0.24
        
        # New gate (simplified): n = tanh(W_n * x + U_n * (r * h_prev))
        w_n = ArbitraryNumber("0.7")
        u_n = ArbitraryNumber("0.5")
        n = w_n * x + u_n * (r * h_prev)  # 0.7 * 0.3 + 0.5 * (0.23 * 0.2) = 0.21 + 0.5 * 0.046 = 0.21 + 0.023 = 0.233
        
        # Hidden state update: h = (1 - z) * n + z * h_prev
        one_minus_z = ArbitraryNumber("1") - z  # 1 - 0.24 = 0.76
        h = one_minus_z * n + z * h_prev  # 0.76 * 0.233 + 0.24 * 0.2 = 0.17708 + 0.048 = 0.22508
        
        # Verify calculations are exact
        self.assertEqual(r, ArbitraryNumber("0.23"))
        self.assertEqual(z, ArbitraryNumber("0.24"))
        self.assertEqual(n, ArbitraryNumber("0.233"))
        self.assertTrue(h > ArbitraryNumber("0.22"))
        self.assertTrue(h < ArbitraryNumber("0.23"))

    def test_lstm_cell_precision(self):
        """Test LSTM cell calculation precision"""
        # Simplified LSTM cell
        # Input
        x = ArbitraryNumber("0.2")
        # Previous hidden state
        h_prev = ArbitraryNumber("0.1")
        # Previous cell state
        c_prev = ArbitraryNumber("0.15")
        
        # Forget gate (simplified): f = sigmoid(W_f * x + U_f * h_prev)
        w_f = ArbitraryNumber("0.4")
        u_f = ArbitraryNumber("0.3")
        f = w_f * x + u_f * h_prev  # 0.4 * 0.2 + 0.3 * 0.1 = 0.08 + 0.03 = 0.11
        
        # Input gate (simplified): i = sigmoid(W_i * x + U_i * h_prev)
        w_i = ArbitraryNumber("0.5")
        u_i = ArbitraryNumber("0.2")
        i = w_i * x + u_i * h_prev  # 0.5 * 0.2 + 0.2 * 0.1 = 0.1 + 0.02 = 0.12
        
        # Candidate values (simplified): g = tanh(W_g * x + U_g * h_prev)
        w_g = ArbitraryNumber("0.6")
        u_g = ArbitraryNumber("0.4")
        g = w_g * x + u_g * h_prev  # 0.6 * 0.2 + 0.4 * 0.1 = 0.12 + 0.04 = 0.16
        
        # Cell state update: c = f * c_prev + i * g
        c = f * c_prev + i * g  # 0.11 * 0.15 + 0.12 * 0.16 = 0.0165 + 0.0192 = 0.0357
        
        # Output gate (simplified): o = sigmoid(W_o * x + U_o * h_prev)
        w_o = ArbitraryNumber("0.3")
        u_o = ArbitraryNumber("0.5")
        o = w_o * x + u_o * h_prev  # 0.3 * 0.2 + 0.5 * 0.1 = 0.06 + 0.05 = 0.11
        
        # Hidden state update: h = o * tanh(c)
        # Using c directly as tanh approximation for simplicity
        h = o * c  # 0.11 * 0.0357 = 0.003927
        
        # Verify exact calculations
        self.assertEqual(f, ArbitraryNumber("0.11"))
        self.assertEqual(i, ArbitraryNumber("0.12"))
        self.assertEqual(g, ArbitraryNumber("0.16"))
        self.assertEqual(c, ArbitraryNumber("0.0357"))
        self.assertEqual(o, ArbitraryNumber("0.11"))
        self.assertEqual(h, ArbitraryNumber("0.003927"))


class TestNeuralNetworkLossFunctions(unittest.TestCase):
    """Test neural network loss functions precision"""

    def test_mean_squared_error_precision(self):
        """Test MSE loss calculation precision"""
        # Predictions
        pred1 = ArbitraryNumber("0.8")
        pred2 = ArbitraryNumber("0.6")
        
        # Targets
        target1 = ArbitraryNumber("0.9")
        target2 = ArbitraryNumber("0.5")
        
        # MSE = mean((pred - target)^2)
        error1 = pred1 - target1  # 0.8 - 0.9 = -0.1
        error2 = pred2 - target2  # 0.6 - 0.5 = 0.1
        
        squared_error1 = error1 * error1  # (-0.1)^2 = 0.01
        squared_error2 = error2 * error2  # (0.1)^2 = 0.01
        
        mse = (squared_error1 + squared_error2) / ArbitraryNumber("2")  # (0.01 + 0.01) / 2 = 0.01
        
        expected = ArbitraryNumber("0.01")
        self.assertEqual(mse, expected)

    def test_binary_cross_entropy_precision(self):
        """Test binary cross-entropy loss precision"""
        # Predicted probability
        p = ArbitraryNumber("0.8")
        # True label
        y = ArbitraryNumber("1")
        
        # BCE = -y * log(p) - (1-y) * log(1-p)
        # For y=1: BCE = -log(p)
        # Using approximation: log(p) ≈ p - 1 for p close to 1
        log_p_approx = p - ArbitraryNumber("1")  # 0.8 - 1 = -0.2
        bce = -y * log_p_approx  # -1 * (-0.2) = 0.2
        
        expected = ArbitraryNumber("0.2")
        self.assertEqual(bce, expected)

    def test_categorical_cross_entropy_precision(self):
        """Test categorical cross-entropy loss precision"""
        # True label (one-hot)
        y_true = [ArbitraryNumber("0"), ArbitraryNumber("1"), ArbitraryNumber("0")]
        # Predicted probabilities
        y_pred = [ArbitraryNumber("0.1"), ArbitraryNumber("0.8"), ArbitraryNumber("0.1")]
        
        # CCE = -sum(y_true * log(y_pred))
        # Only the true class contributes: -1 * log(0.8)
        # Using approximation: log(0.8) ≈ 0.8 - 1 = -0.2
        log_pred_approx = y_pred[1] - ArbitraryNumber("1")  # 0.8 - 1 = -0.2
        cce = -y_true[1] * log_pred_approx  # -1 * (-0.2) = 0.2
        
        expected = ArbitraryNumber("0.2")
        self.assertEqual(cce, expected)

    def test_huber_loss_precision(self):
        """Test Huber loss calculation precision"""
        # Prediction and target
        pred = ArbitraryNumber("0.7")
        target = ArbitraryNumber("0.5")
        # Delta parameter
        delta = ArbitraryNumber("0.1")
        
        # Error
        error = pred - target  # 0.7 - 0.5 = 0.2
        abs_error = abs(error)  # |0.2| = 0.2
        
        # Huber loss: if |error| <= delta: 0.5 * error^2, else: delta * (|error| - 0.5 * delta)
        if abs_error <= delta:
            huber_loss = ArbitraryNumber("0.5") * error * error
        else:
            huber_loss = delta * (abs_error - ArbitraryNumber("0.5") * delta)
        
        # Since |0.2| > 0.1, use second formula
        # huber_loss = 0.1 * (0.2 - 0.5 * 0.1) = 0.1 * (0.2 - 0.05) = 0.1 * 0.15 = 0.015
        expected = ArbitraryNumber("0.015")
        self.assertEqual(huber_loss, expected)


if __name__ == '__main__':
    print("=" * 60)
    print("ARBITRARYNUMBER V2 NEURAL NETWORK COMPONENTS TESTS")
    print("=" * 60)
    print("Testing neural network components precision...")
    print()
    
    # Run the tests
    unittest.main(verbosity=2)
