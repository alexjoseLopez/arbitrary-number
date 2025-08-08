"""
Integration Tests for ArbitraryNumber
====================================

Integration tests that verify ArbitraryNumber works correctly with other components
and in real-world scenarios.
"""

import unittest
import sys
import os
from fractions import Fraction
import math

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from arbitrary_numbers.core.arbitrary_number import ArbitraryNumber, FractionTerm


class TestArbitraryNumberIntegration(unittest.TestCase):
    """Integration tests for ArbitraryNumber with various scenarios."""
    
    def test_scientific_computation_workflow(self):
        """Test ArbitraryNumber in a scientific computation workflow."""
        # Simulate calculating the area under a curve using trapezoidal rule
        # f(x) = 1/x from x=1 to x=2
        
        def f(x):
            """Function to integrate: f(x) = 1/x"""
            return ArbitraryNumber.one() / x
        
        # Trapezoidal rule with n intervals
        n = 100
        a = ArbitraryNumber.one()  # Lower bound
        b = ArbitraryNumber.from_int(2)  # Upper bound
        
        h = (b - a) / ArbitraryNumber.from_int(n)  # Step size
        
        # Calculate sum
        integral_sum = f(a) + f(b)  # First and last terms
        
        for i in range(1, n):
            x_i = a + h * ArbitraryNumber.from_int(i)
            integral_sum = integral_sum + ArbitraryNumber.from_int(2) * f(x_i)
        
        # Final result
        integral = h * integral_sum / ArbitraryNumber.from_int(2)
        
        # The exact integral of 1/x from 1 to 2 is ln(2)
        result_float = float(integral.evaluate_exact())
        expected = math.log(2)
        
        # Should be close to ln(2) ≈ 0.693147
        self.assertAlmostEqual(result_float, expected, places=3)
        
        # Verify zero precision loss
        self.assertEqual(integral.get_precision_loss(), 0.0)
    
    def test_financial_calculation_workflow(self):
        """Test ArbitraryNumber in financial calculations."""
        # Calculate compound interest with exact precision
        # A = P(1 + r/n)^(nt)
        
        principal = ArbitraryNumber.from_int(10000)  # $10,000
        annual_rate = ArbitraryNumber.from_fraction(5, 100)  # 5% annual rate
        compounds_per_year = ArbitraryNumber.from_int(12)  # Monthly compounding
        years = ArbitraryNumber.from_int(10)  # 10 years
        
        # Calculate (1 + r/n)
        rate_per_period = annual_rate / compounds_per_year
        base = ArbitraryNumber.one() + rate_per_period
        
        # Calculate nt (total number of periods)
        total_periods = compounds_per_year * years
        total_periods_int = int(total_periods.evaluate_exact())
        
        # Calculate (1 + r/n)^(nt)
        compound_factor = base ** total_periods_int
        
        # Final amount
        final_amount = principal * compound_factor
        
        # Verify result is reasonable
        result_float = float(final_amount.evaluate_exact())
        
        # Should be around $16,470 for these parameters
        self.assertGreater(result_float, 16000)
        self.assertLess(result_float, 17000)
        
        # Verify exact precision
        self.assertEqual(final_amount.get_precision_loss(), 0.0)
        
        # Calculate interest earned
        interest = final_amount - principal
        interest_float = float(interest.evaluate_exact())
        
        self.assertGreater(interest_float, 6000)
        self.assertLess(interest_float, 7000)
    
    def test_machine_learning_weight_calculation(self):
        """Test ArbitraryNumber in ML weight calculations."""
        # Simulate exact weight updates in a simple neural network
        
        # Initial weights (exact fractions)
        weights = [
            ArbitraryNumber.from_fraction(1, 10),   # 0.1
            ArbitraryNumber.from_fraction(-3, 20),  # -0.15
            ArbitraryNumber.from_fraction(2, 5),    # 0.4
            ArbitraryNumber.from_fraction(7, 25)    # 0.28
        ]
        
        # Learning rate
        learning_rate = ArbitraryNumber.from_fraction(1, 100)  # 0.01
        
        # Gradients (simulated)
        gradients = [
            ArbitraryNumber.from_fraction(3, 10),   # 0.3
            ArbitraryNumber.from_fraction(-1, 5),   # -0.2
            ArbitraryNumber.from_fraction(1, 8),    # 0.125
            ArbitraryNumber.from_fraction(-2, 15)   # -0.133...
        ]
        
        # Update weights: w = w - lr * grad
        updated_weights = []
        for w, g in zip(weights, gradients):
            update = learning_rate * g
            new_weight = w - update
            updated_weights.append(new_weight)
        
        # Verify all weights are exact
        for weight in updated_weights:
            self.assertEqual(weight.get_precision_loss(), 0.0)
            self.assertIsInstance(weight.evaluate_exact(), Fraction)
        
        # Verify weight updates are reasonable
        weight_changes = [
            float((updated_weights[i] - weights[i]).evaluate_exact())
            for i in range(len(weights))
        ]
        
        # All changes should be small (learning rate * gradient)
        for change in weight_changes:
            self.assertLess(abs(change), 0.01)
    
    def test_statistical_calculation_workflow(self):
        """Test ArbitraryNumber in statistical calculations."""
        # Calculate exact mean, variance, and standard deviation
        
        # Sample data as exact fractions
        data = [
            ArbitraryNumber.from_fraction(1, 2),    # 0.5
            ArbitraryNumber.from_fraction(3, 4),    # 0.75
            ArbitraryNumber.from_fraction(2, 3),    # 0.666...
            ArbitraryNumber.from_fraction(5, 6),    # 0.833...
            ArbitraryNumber.from_fraction(7, 8),    # 0.875
            ArbitraryNumber.from_fraction(4, 5),    # 0.8
            ArbitraryNumber.from_fraction(9, 10),   # 0.9
            ArbitraryNumber.from_fraction(11, 12)   # 0.916...
        ]
        
        n = ArbitraryNumber.from_int(len(data))
        
        # Calculate exact mean
        total = ArbitraryNumber.zero()
        for value in data:
            total = total + value
        
        mean = total / n
        
        # Calculate exact variance
        variance_sum = ArbitraryNumber.zero()
        for value in data:
            diff = value - mean
            variance_sum = variance_sum + (diff * diff)
        
        variance = variance_sum / n
        
        # Verify calculations
        mean_float = float(mean.evaluate_exact())
        variance_float = float(variance.evaluate_exact())
        
        # Mean should be reasonable
        self.assertGreater(mean_float, 0.6)
        self.assertLess(mean_float, 0.9)
        
        # Variance should be positive and reasonable
        self.assertGreater(variance_float, 0.0)
        self.assertLess(variance_float, 0.1)
        
        # Verify exact precision maintained
        self.assertEqual(mean.get_precision_loss(), 0.0)
        self.assertEqual(variance.get_precision_loss(), 0.0)
    
    def test_physics_simulation_workflow(self):
        """Test ArbitraryNumber in physics calculations."""
        # Simulate projectile motion with exact calculations
        
        # Initial conditions
        initial_velocity = ArbitraryNumber.from_int(50)  # m/s
        angle_degrees = ArbitraryNumber.from_int(45)     # degrees
        gravity = ArbitraryNumber.from_fraction(981, 100)  # 9.81 m/s^2
        
        # Convert angle to radians (approximately)
        # For 45 degrees, sin(45°) = cos(45°) = 1/√2 ≈ √2/2
        # We'll use the exact value √2/2 represented as a fraction approximation
        sin_45 = ArbitraryNumber.from_fraction(7071, 10000)  # ≈ 0.7071
        cos_45 = sin_45  # cos(45°) = sin(45°)
        
        # Initial velocity components
        v_x = initial_velocity * cos_45
        v_y = initial_velocity * sin_45
        
        # Time of flight: t = 2 * v_y / g
        time_of_flight = (ArbitraryNumber.from_int(2) * v_y) / gravity
        
        # Maximum height: h = v_y^2 / (2 * g)
        max_height = (v_y * v_y) / (ArbitraryNumber.from_int(2) * gravity)
        
        # Range: R = v_x * t
        range_distance = v_x * time_of_flight
        
        # Verify results are reasonable
        time_float = float(time_of_flight.evaluate_exact())
        height_float = float(max_height.evaluate_exact())
        range_float = float(range_distance.evaluate_exact())
        
        # Time should be around 7.2 seconds
        self.assertGreater(time_float, 6.0)
        self.assertLess(time_float, 8.0)
        
        # Max height should be around 63.8 meters
        self.assertGreater(height_float, 60.0)
        self.assertLess(height_float, 70.0)
        
        # Range should be around 255 meters
        self.assertGreater(range_float, 240.0)
        self.assertLess(range_float, 270.0)
        
        # All calculations maintain exact precision
        self.assertEqual(time_of_flight.get_precision_loss(), 0.0)
        self.assertEqual(max_height.get_precision_loss(), 0.0)
        self.assertEqual(range_distance.get_precision_loss(), 0.0)
    
    def test_cryptographic_calculation_workflow(self):
        """Test ArbitraryNumber in cryptographic calculations."""
        # Simulate RSA key generation components with exact arithmetic
        
        # Small primes for demonstration (in real RSA, these would be much larger)
        p = ArbitraryNumber.from_int(17)
        q = ArbitraryNumber.from_int(19)
        
        # Calculate n = p * q
        n = p * q
        
        # Calculate φ(n) = (p-1)(q-1)
        phi_n = (p - ArbitraryNumber.one()) * (q - ArbitraryNumber.one())
        
        # Choose e (public exponent) - commonly 65537, but we'll use 3 for simplicity
        e = ArbitraryNumber.from_int(3)
        
        # Verify gcd(e, φ(n)) = 1 by checking if φ(n) is not divisible by e
        phi_n_int = int(phi_n.evaluate_exact())
        e_int = int(e.evaluate_exact())
        
        self.assertNotEqual(phi_n_int % e_int, 0)  # Should not be divisible
        
        # Calculate some powers for encryption/decryption simulation
        message = ArbitraryNumber.from_int(5)  # Simple message
        
        # "Encrypt": c = m^e mod n
        # For exact arithmetic, we'll just calculate m^e
        encrypted_power = message ** e_int
        
        # Verify calculations
        n_int = int(n.evaluate_exact())
        phi_n_int = int(phi_n.evaluate_exact())
        
        self.assertEqual(n_int, 17 * 19)  # 323
        self.assertEqual(phi_n_int, 16 * 18)  # 288
        
        # All calculations are exact
        self.assertEqual(n.get_precision_loss(), 0.0)
        self.assertEqual(phi_n.get_precision_loss(), 0.0)
        self.assertEqual(encrypted_power.get_precision_loss(), 0.0)
    
    def test_optimization_algorithm_workflow(self):
        """Test ArbitraryNumber in optimization algorithms."""
        # Simulate gradient descent with exact arithmetic
        
        # Objective function: f(x) = (x - 2)^2 + 1
        # Derivative: f'(x) = 2(x - 2)
        # Minimum at x = 2
        
        def gradient(x):
            """Calculate gradient of f(x) = (x - 2)^2 + 1"""
            return ArbitraryNumber.from_int(2) * (x - ArbitraryNumber.from_int(2))
        
        # Initial point
        x = ArbitraryNumber.from_fraction(1, 2)  # Start at x = 0.5
        
        # Learning rate
        learning_rate = ArbitraryNumber.from_fraction(1, 10)  # 0.1
        
        # Perform gradient descent steps
        positions = [x]
        
        for step in range(10):
            grad = gradient(x)
            x = x - learning_rate * grad
            positions.append(x)
        
        # Final position should be close to 2
        final_x = float(x.evaluate_exact())
        self.assertAlmostEqual(final_x, 2.0, places=2)
        
        # All positions maintain exact precision
        for pos in positions:
            self.assertEqual(pos.get_precision_loss(), 0.0)
        
        # Verify convergence (positions should get closer to 2)
        distances_to_optimum = [
            abs(float(pos.evaluate_exact()) - 2.0) for pos in positions
        ]
        
        # Distance should generally decrease
        self.assertLess(distances_to_optimum[-1], distances_to_optimum[0])
    
    def test_signal_processing_workflow(self):
        """Test ArbitraryNumber in signal processing calculations."""
        # Simulate discrete Fourier transform components with exact arithmetic
        
        # Simple signal: [1, 0, -1, 0] (square wave approximation)
        signal = [
            ArbitraryNumber.one(),
            ArbitraryNumber.zero(),
            -ArbitraryNumber.one(),
            ArbitraryNumber.zero()
        ]
        
        N = len(signal)
        
        # Calculate DC component (k=0)
        dc_component = ArbitraryNumber.zero()
        for x in signal:
            dc_component = dc_component + x
        
        dc_component = dc_component / ArbitraryNumber.from_int(N)
        
        # Calculate first harmonic component (k=1)
        # Real part: sum(x[n] * cos(2πkn/N))
        # Imaginary part: sum(x[n] * sin(2πkn/N))
        
        # For k=1, N=4: cos(2πn/4) = cos(πn/2)
        # n=0: cos(0) = 1, sin(0) = 0
        # n=1: cos(π/2) = 0, sin(π/2) = 1
        # n=2: cos(π) = -1, sin(π) = 0
        # n=3: cos(3π/2) = 0, sin(3π/2) = -1
        
        cos_values = [
            ArbitraryNumber.one(),      # cos(0)
            ArbitraryNumber.zero(),     # cos(π/2)
            -ArbitraryNumber.one(),     # cos(π)
            ArbitraryNumber.zero()      # cos(3π/2)
        ]
        
        sin_values = [
            ArbitraryNumber.zero(),     # sin(0)
            ArbitraryNumber.one(),      # sin(π/2)
            ArbitraryNumber.zero(),     # sin(π)
            -ArbitraryNumber.one()      # sin(3π/2)
        ]
        
        # Calculate real and imaginary parts
        real_part = ArbitraryNumber.zero()
        imag_part = ArbitraryNumber.zero()
        
        for n in range(N):
            real_part = real_part + signal[n] * cos_values[n]
            imag_part = imag_part - signal[n] * sin_values[n]  # Negative for DFT
        
        # Verify results
        dc_float = float(dc_component.evaluate_exact())
        real_float = float(real_part.evaluate_exact())
        imag_float = float(imag_part.evaluate_exact())
        
        # DC component should be 0 (sum is 0)
        self.assertEqual(dc_float, 0.0)
        
        # For this signal, first harmonic should have specific values
        self.assertEqual(real_float, 0.0)  # Real part should be 0
        self.assertEqual(imag_float, 2.0)  # Imaginary part should be 2
        
        # All calculations are exact
        self.assertEqual(dc_component.get_precision_loss(), 0.0)
        self.assertEqual(real_part.get_precision_loss(), 0.0)
        self.assertEqual(imag_part.get_precision_loss(), 0.0)


if __name__ == '__main__':
    unittest.main()
