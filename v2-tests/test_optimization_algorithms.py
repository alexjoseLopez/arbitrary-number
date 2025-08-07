#!/usr/bin/env python3
"""
Optimization Algorithms Precision Tests for ArbitraryNumber V2
Tests specific optimization algorithms with focus on precision verification
"""

import unittest
import sys
import os
import time
import math

# Add the v2 module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from v2.core.arbitrary_number import ArbitraryNumber


class TestOptimizationAlgorithms(unittest.TestCase):
    """Test optimization algorithms precision with ArbitraryNumber"""

    def test_gradient_descent_single_step(self):
        """Test single gradient descent step precision"""
        # Current parameter
        theta = ArbitraryNumber("1.5")
        # Learning rate
        alpha = ArbitraryNumber("0.01")
        # Gradient
        gradient = ArbitraryNumber("0.5")
        
        # Update: theta_new = theta - alpha * gradient
        theta_new = theta - alpha * gradient
        
        # Expected: 1.5 - 0.01 * 0.5 = 1.5 - 0.005 = 1.495
        expected = ArbitraryNumber("1.495")
        
        self.assertEqual(theta_new, expected)

    def test_gradient_descent_multiple_steps(self):
        """Test multiple gradient descent steps precision"""
        # Initial parameter
        theta = ArbitraryNumber("2.0")
        # Learning rate
        alpha = ArbitraryNumber("0.1")
        
        # Simulate 3 steps with different gradients
        gradients = [ArbitraryNumber("0.2"), ArbitraryNumber("0.15"), ArbitraryNumber("0.1")]
        
        for gradient in gradients:
            theta = theta - alpha * gradient
        
        # Step 1: 2.0 - 0.1 * 0.2 = 2.0 - 0.02 = 1.98
        # Step 2: 1.98 - 0.1 * 0.15 = 1.98 - 0.015 = 1.965
        # Step 3: 1.965 - 0.1 * 0.1 = 1.965 - 0.01 = 1.955
        expected = ArbitraryNumber("1.955")
        
        self.assertEqual(theta, expected)

    def test_sgd_with_momentum_precision(self):
        """Test SGD with momentum precision"""
        # Parameters
        theta = ArbitraryNumber("1.0")
        velocity = ArbitraryNumber("0.0")
        
        # Hyperparameters
        learning_rate = ArbitraryNumber("0.01")
        momentum = ArbitraryNumber("0.9")
        gradient = ArbitraryNumber("0.5")
        
        # Update velocity: v = momentum * v + gradient
        velocity = momentum * velocity + gradient
        
        # Update parameter: theta = theta - learning_rate * velocity
        theta = theta - learning_rate * velocity
        
        # Expected velocity: 0.9 * 0.0 + 0.5 = 0.5
        # Expected theta: 1.0 - 0.01 * 0.5 = 1.0 - 0.005 = 0.995
        expected_velocity = ArbitraryNumber("0.5")
        expected_theta = ArbitraryNumber("0.995")
        
        self.assertEqual(velocity, expected_velocity)
        self.assertEqual(theta, expected_theta)

    def test_adam_optimizer_precision(self):
        """Test Adam optimizer precision"""
        # Parameters
        theta = ArbitraryNumber("1.0")
        m = ArbitraryNumber("0.0")  # First moment
        v = ArbitraryNumber("0.0")  # Second moment
        
        # Hyperparameters
        learning_rate = ArbitraryNumber("0.001")
        beta1 = ArbitraryNumber("0.9")
        beta2 = ArbitraryNumber("0.999")
        epsilon = ArbitraryNumber("1e-8")
        gradient = ArbitraryNumber("0.1")
        t = 1  # Time step
        
        # Update first moment: m = beta1 * m + (1 - beta1) * gradient
        m = beta1 * m + (ArbitraryNumber("1") - beta1) * gradient
        
        # Update second moment: v = beta2 * v + (1 - beta2) * gradient^2
        v = beta2 * v + (ArbitraryNumber("1") - beta2) * gradient * gradient
        
        # Bias correction
        m_hat = m / (ArbitraryNumber("1") - beta1 ** t)
        v_hat = v / (ArbitraryNumber("1") - beta2 ** t)
        
        # Expected m: 0.9 * 0 + 0.1 * 0.1 = 0.01
        # Expected v: 0.999 * 0 + 0.001 * 0.01 = 0.00001
        expected_m = ArbitraryNumber("0.01")
        expected_v = ArbitraryNumber("0.00001")
        
        self.assertEqual(m, expected_m)
        self.assertEqual(v, expected_v)

    def test_rmsprop_optimizer_precision(self):
        """Test RMSprop optimizer precision"""
        # Parameters
        theta = ArbitraryNumber("1.0")
        s = ArbitraryNumber("0.0")  # Exponential average of squared gradients
        
        # Hyperparameters
        learning_rate = ArbitraryNumber("0.01")
        decay_rate = ArbitraryNumber("0.9")
        epsilon = ArbitraryNumber("1e-6")
        gradient = ArbitraryNumber("0.2")
        
        # Update exponential average: s = decay_rate * s + (1 - decay_rate) * gradient^2
        s = decay_rate * s + (ArbitraryNumber("1") - decay_rate) * gradient * gradient
        
        # Expected s: 0.9 * 0 + 0.1 * 0.04 = 0.004
        expected_s = ArbitraryNumber("0.004")
        
        self.assertEqual(s, expected_s)

    def test_adagrad_optimizer_precision(self):
        """Test Adagrad optimizer precision"""
        # Parameters
        theta = ArbitraryNumber("1.0")
        G = ArbitraryNumber("0.0")  # Sum of squared gradients
        
        # Hyperparameters
        learning_rate = ArbitraryNumber("0.01")
        epsilon = ArbitraryNumber("1e-8")
        gradient = ArbitraryNumber("0.3")
        
        # Update sum of squared gradients: G = G + gradient^2
        G = G + gradient * gradient
        
        # Expected G: 0 + 0.09 = 0.09
        expected_G = ArbitraryNumber("0.09")
        
        self.assertEqual(G, expected_G)

    def test_learning_rate_scheduling_precision(self):
        """Test learning rate scheduling precision"""
        # Initial learning rate
        lr_initial = ArbitraryNumber("0.1")
        
        # Step decay
        decay_factor = ArbitraryNumber("0.5")
        step = 2
        lr_step = lr_initial * (decay_factor ** step)
        
        # Expected: 0.1 * 0.5^2 = 0.1 * 0.25 = 0.025
        expected_step = ArbitraryNumber("0.025")
        self.assertEqual(lr_step, expected_step)
        
        # Exponential decay
        decay_rate = ArbitraryNumber("0.95")
        epoch = 3
        lr_exp = lr_initial * (decay_rate ** epoch)
        
        # Expected: 0.1 * 0.95^3 = 0.1 * 0.857375 = 0.0857375
        self.assertTrue(lr_exp > ArbitraryNumber("0.085"))
        self.assertTrue(lr_exp < ArbitraryNumber("0.086"))

    def test_line_search_precision(self):
        """Test line search algorithm precision"""
        # Current point
        x = ArbitraryNumber("1.0")
        # Search direction
        direction = ArbitraryNumber("-0.5")
        # Step sizes to test
        alphas = [ArbitraryNumber("0.1"), ArbitraryNumber("0.01"), ArbitraryNumber("0.001")]
        
        # Simple quadratic function: f(x) = x^2
        # Gradient: f'(x) = 2x
        
        best_alpha = None
        best_value = None
        
        for alpha in alphas:
            x_new = x + alpha * direction
            f_new = x_new * x_new  # f(x_new) = x_new^2
            
            if best_value is None or f_new < best_value:
                best_value = f_new
                best_alpha = alpha
        
        # For x=1, direction=-0.5:
        # alpha=0.1: x_new = 1 + 0.1*(-0.5) = 0.95, f = 0.9025
        # alpha=0.01: x_new = 1 + 0.01*(-0.5) = 0.995, f = 0.990025
        # alpha=0.001: x_new = 1 + 0.001*(-0.5) = 0.9995, f = 0.99900025
        
        # Best should be alpha=0.1 with smallest function value
        expected_best_alpha = ArbitraryNumber("0.1")
        self.assertEqual(best_alpha, expected_best_alpha)

    def test_newton_method_precision(self):
        """Test Newton's method precision"""
        # Current point
        x = ArbitraryNumber("2.0")
        
        # For function f(x) = x^2 - 4 (root at x=2)
        # f'(x) = 2x, f''(x) = 2
        
        # Newton update: x_new = x - f'(x) / f''(x)
        f_prime = ArbitraryNumber("2") * x  # 2 * 2 = 4
        f_double_prime = ArbitraryNumber("2")
        
        x_new = x - f_prime / f_double_prime
        
        # Expected: 2 - 4/2 = 2 - 2 = 0
        expected = ArbitraryNumber("0")
        self.assertEqual(x_new, expected)

    def test_conjugate_gradient_precision(self):
        """Test conjugate gradient method precision"""
        # Simplified conjugate gradient step
        # Current solution
        x = ArbitraryNumber("1.0")
        # Current residual
        r = ArbitraryNumber("0.5")
        # Search direction
        p = ArbitraryNumber("0.5")
        
        # Step size calculation (simplified)
        # alpha = r^T * r / (p^T * A * p)
        # For simplicity, assume A * p = p (identity-like)
        alpha = (r * r) / (p * p)
        
        # Update solution: x_new = x + alpha * p
        x_new = x + alpha * p
        
        # Expected alpha: 0.25 / 0.25 = 1
        # Expected x_new: 1 + 1 * 0.5 = 1.5
        expected_alpha = ArbitraryNumber("1")
        expected_x = ArbitraryNumber("1.5")
        
        self.assertEqual(alpha, expected_alpha)
        self.assertEqual(x_new, expected_x)

    def test_bfgs_approximation_precision(self):
        """Test BFGS Hessian approximation precision"""
        # Current Hessian approximation (scalar for simplicity)
        H = ArbitraryNumber("1.0")
        
        # Gradient difference
        y = ArbitraryNumber("0.2")
        # Step difference
        s = ArbitraryNumber("0.1")
        
        # BFGS update (simplified scalar version)
        # H_new = H + (y*y)/(y*s) - (H*s*s*H)/(s*H*s)
        
        # First term
        term1 = (y * y) / (y * s)  # 0.04 / 0.02 = 2
        
        # Second term (simplified)
        term2 = (H * s * s * H) / (s * H * s)  # (1*0.01*1) / (0.1*1*0.1) = 0.01 / 0.01 = 1
        
        H_new = H + term1 - term2  # 1 + 2 - 1 = 2
        
        expected = ArbitraryNumber("2")
        self.assertEqual(H_new, expected)

    def test_trust_region_precision(self):
        """Test trust region method precision"""
        # Current point
        x = ArbitraryNumber("1.0")
        # Trust region radius
        delta = ArbitraryNumber("0.5")
        # Proposed step
        p = ArbitraryNumber("0.3")
        
        # Check if step is within trust region
        step_norm = abs(p)  # |0.3| = 0.3
        
        # Accept step if within trust region
        if step_norm <= delta:
            x_new = x + p
            accepted = True
        else:
            # Scale step to trust region boundary
            p_scaled = p * (delta / step_norm)
            x_new = x + p_scaled
            accepted = False
        
        # Since 0.3 <= 0.5, step should be accepted
        expected_x = ArbitraryNumber("1.3")
        self.assertEqual(x_new, expected_x)
        self.assertTrue(accepted)

    def test_proximal_gradient_precision(self):
        """Test proximal gradient method precision"""
        # Current point
        x = ArbitraryNumber("1.0")
        # Gradient
        gradient = ArbitraryNumber("0.4")
        # Step size
        t = ArbitraryNumber("0.1")
        # L1 regularization parameter
        lambda_reg = ArbitraryNumber("0.01")
        
        # Gradient step
        y = x - t * gradient  # 1.0 - 0.1 * 0.4 = 1.0 - 0.04 = 0.96
        
        # Soft thresholding (proximal operator for L1)
        threshold = t * lambda_reg  # 0.1 * 0.01 = 0.001
        
        if y > threshold:
            x_new = y - threshold  # 0.96 - 0.001 = 0.959
        elif y < -threshold:
            x_new = y + threshold
        else:
            x_new = ArbitraryNumber("0")
        
        expected = ArbitraryNumber("0.959")
        self.assertEqual(x_new, expected)

    def test_coordinate_descent_precision(self):
        """Test coordinate descent precision"""
        # Current solution vector (2D for simplicity)
        x1 = ArbitraryNumber("1.0")
        x2 = ArbitraryNumber("0.5")
        
        # Update first coordinate
        # For quadratic: f(x1, x2) = x1^2 + x1*x2 + x2^2
        # Partial derivative w.r.t. x1: 2*x1 + x2
        grad_x1 = ArbitraryNumber("2") * x1 + x2  # 2*1 + 0.5 = 2.5
        
        # Learning rate
        lr = ArbitraryNumber("0.1")
        
        # Update x1
        x1_new = x1 - lr * grad_x1  # 1.0 - 0.1 * 2.5 = 1.0 - 0.25 = 0.75
        
        expected_x1 = ArbitraryNumber("0.75")
        self.assertEqual(x1_new, expected_x1)

    def test_nesterov_momentum_precision(self):
        """Test Nesterov accelerated gradient precision"""
        # Parameters
        theta = ArbitraryNumber("1.0")
        velocity = ArbitraryNumber("0.1")
        
        # Hyperparameters
        learning_rate = ArbitraryNumber("0.01")
        momentum = ArbitraryNumber("0.9")
        gradient = ArbitraryNumber("0.5")
        
        # Nesterov update
        # Look-ahead point
        theta_lookahead = theta - momentum * velocity  # 1.0 - 0.9 * 0.1 = 1.0 - 0.09 = 0.91
        
        # Update velocity with gradient at look-ahead point
        velocity_new = momentum * velocity + learning_rate * gradient  # 0.9 * 0.1 + 0.01 * 0.5 = 0.09 + 0.005 = 0.095
        
        # Update parameter
        theta_new = theta - velocity_new  # 1.0 - 0.095 = 0.905
        
        expected_velocity = ArbitraryNumber("0.095")
        expected_theta = ArbitraryNumber("0.905")
        
        self.assertEqual(velocity_new, expected_velocity)
        self.assertEqual(theta_new, expected_theta)


class TestConstrainedOptimization(unittest.TestCase):
    """Test constrained optimization algorithms precision"""

    def test_lagrange_multiplier_precision(self):
        """Test Lagrange multiplier calculation precision"""
        # Lagrangian: L(x, lambda) = f(x) + lambda * g(x)
        # For simplicity: f(x) = x^2, g(x) = x - 1 (constraint: x = 1)
        
        x = ArbitraryNumber("1.0")
        lambda_val = ArbitraryNumber("2.0")
        
        # Gradient of Lagrangian w.r.t. x: 2x + lambda
        grad_L_x = ArbitraryNumber("2") * x + lambda_val  # 2*1 + 2 = 4
        
        # Constraint: g(x) = x - 1
        constraint = x - ArbitraryNumber("1")  # 1 - 1 = 0
        
        expected_grad = ArbitraryNumber("4")
        expected_constraint = ArbitraryNumber("0")
        
        self.assertEqual(grad_L_x, expected_grad)
        self.assertEqual(constraint, expected_constraint)

    def test_penalty_method_precision(self):
        """Test penalty method precision"""
        # Objective function value
        f_x = ArbitraryNumber("0.5")
        # Constraint violation
        g_x = ArbitraryNumber("0.1")  # g(x) > 0 means violation
        # Penalty parameter
        mu = ArbitraryNumber("10.0")
        
        # Penalty function: P(x) = f(x) + mu * max(0, g(x))^2
        penalty = mu * max(ArbitraryNumber("0"), g_x) * max(ArbitraryNumber("0"), g_x)
        penalized_objective = f_x + penalty
        
        # Expected penalty: 10 * 0.1^2 = 10 * 0.01 = 0.1
        # Expected total: 0.5 + 0.1 = 0.6
        expected_penalty = ArbitraryNumber("0.1")
        expected_total = ArbitraryNumber("0.6")
        
        self.assertEqual(penalty, expected_penalty)
        self.assertEqual(penalized_objective, expected_total)

    def test_barrier_method_precision(self):
        """Test barrier method precision"""
        # Objective function value
        f_x = ArbitraryNumber("1.0")
        # Constraint value (must be negative for feasibility)
        g_x = ArbitraryNumber("-0.5")  # g(x) < 0 means feasible
        # Barrier parameter
        t = ArbitraryNumber("1.0")
        
        # Logarithmic barrier: B(x) = f(x) - (1/t) * log(-g(x))
        # For simplicity, approximate log(-g(x)) ≈ -g(x) - 1 for small values
        log_approx = -g_x - ArbitraryNumber("1")  # -(-0.5) - 1 = 0.5 - 1 = -0.5
        barrier_term = -(ArbitraryNumber("1") / t) * log_approx  # -(1/1) * (-0.5) = 0.5
        barrier_objective = f_x + barrier_term
        
        # Expected: 1.0 + 0.5 = 1.5
        expected = ArbitraryNumber("1.5")
        self.assertEqual(barrier_objective, expected)


if __name__ == '__main__':
    print("=" * 60)
    print("ARBITRARYNUMBER V2 OPTIMIZATION ALGORITHMS TESTS")
    print("=" * 60)
    print("Testing optimization algorithms precision...")
    print()
    
    # Run the tests
    unittest.main(verbosity=2)
