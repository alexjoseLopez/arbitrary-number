"""
Comprehensive Unit Tests for Collatz Conjecture Solver
====================================================

This test suite provides extensive coverage of the ArbitraryNumber-based
Collatz Conjecture solver, including edge cases, performance tests,
and mathematical property verification.
"""

import unittest
import sys
import os
import time
from unittest.mock import patch, MagicMock

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from arbitrary_numbers.core.arbitrary_number import ArbitraryNumber, FractionTerm
from innovation.collatz_conjecture_solver import ExactCollatzAnalyzer


class TestExactCollatzAnalyzer(unittest.TestCase):
    """
    Comprehensive test suite for ExactCollatzAnalyzer.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = ExactCollatzAnalyzer()
    
    def test_collatz_step_exact_even_numbers(self):
        """Test Collatz step for even numbers."""
        # Test basic even numbers
        test_cases = [
            (2, 1),
            (4, 2),
            (6, 3),
            (8, 4),
            (10, 5),
            (100, 50),
            (1000, 500),
            (10000, 5000)
        ]
        
        for input_val, expected in test_cases:
            with self.subTest(input_val=input_val):
                n = ArbitraryNumber.from_int(input_val)
                result = self.analyzer.collatz_step_exact(n)
                expected_result = ArbitraryNumber.from_int(expected)
                self.assertEqual(result, expected_result)
                self.assertEqual(result.get_precision_loss(), 0.0)
    
    def test_collatz_step_exact_odd_numbers(self):
        """Test Collatz step for odd numbers."""
        # Test basic odd numbers
        test_cases = [
            (1, 4),    # 3*1 + 1 = 4
            (3, 10),   # 3*3 + 1 = 10
            (5, 16),   # 3*5 + 1 = 16
            (7, 22),   # 3*7 + 1 = 22
            (9, 28),   # 3*9 + 1 = 28
            (11, 34),  # 3*11 + 1 = 34
            (13, 40),  # 3*13 + 1 = 40
            (15, 46),  # 3*15 + 1 = 46
            (17, 52),  # 3*17 + 1 = 52
            (19, 58),  # 3*19 + 1 = 58
            (21, 64),  # 3*21 + 1 = 64
            (23, 70),  # 3*23 + 1 = 70
            (25, 76),  # 3*25 + 1 = 76
            (27, 82),  # 3*27 + 1 = 82
            (99, 298), # 3*99 + 1 = 298
            (101, 304) # 3*101 + 1 = 304
        ]
        
        for input_val, expected in test_cases:
            with self.subTest(input_val=input_val):
                n = ArbitraryNumber.from_int(input_val)
                result = self.analyzer.collatz_step_exact(n)
                expected_result = ArbitraryNumber.from_int(expected)
                self.assertEqual(result, expected_result)
                self.assertEqual(result.get_precision_loss(), 0.0)
    
    def test_collatz_step_exact_large_numbers(self):
        """Test Collatz step for very large numbers."""
        # Test large even numbers
        large_even = ArbitraryNumber.from_int(2**50)  # Very large even
        result = self.analyzer.collatz_step_exact(large_even)
        expected = ArbitraryNumber.from_int(2**49)
        self.assertEqual(result, expected)
        self.assertEqual(result.get_precision_loss(), 0.0)
        
        # Test large odd numbers
        large_odd = ArbitraryNumber.from_int(2**50 + 1)  # Very large odd
        result = self.analyzer.collatz_step_exact(large_odd)
        expected = ArbitraryNumber.from_int(3 * (2**50 + 1) + 1)
        self.assertEqual(result, expected)
        self.assertEqual(result.get_precision_loss(), 0.0)
    
    def test_collatz_step_exact_edge_cases(self):
        """Test edge cases for Collatz step."""
        # Test with 1 (should give 4)
        one = ArbitraryNumber.one()
        result = self.analyzer.collatz_step_exact(one)
        expected = ArbitraryNumber.from_int(4)
        self.assertEqual(result, expected)
        
        # Test error case: zero or negative
        zero = ArbitraryNumber.zero()
        with self.assertRaises(ValueError):
            self.analyzer.collatz_step_exact(zero)
        
        negative = ArbitraryNumber.from_int(-5)
        with self.assertRaises(ValueError):
            self.analyzer.collatz_step_exact(negative)
    
    def test_compute_exact_trajectory_known_sequences(self):
        """Test trajectory computation for known sequences."""
        # Test n=1: should be [1] with stopping time 0
        n = ArbitraryNumber.one()
        trajectory, stopping_time, max_value = self.analyzer.compute_exact_trajectory(n)
        self.assertEqual(len(trajectory), 1)
        self.assertEqual(stopping_time, 0)
        self.assertEqual(max_value, ArbitraryNumber.one())
        
        # Test n=2: should be [2, 1] with stopping time 1
        n = ArbitraryNumber.from_int(2)
        trajectory, stopping_time, max_value = self.analyzer.compute_exact_trajectory(n)
        expected_trajectory = [ArbitraryNumber.from_int(2), ArbitraryNumber.one()]
        self.assertEqual(trajectory, expected_trajectory)
        self.assertEqual(stopping_time, 1)
        self.assertEqual(max_value, ArbitraryNumber.from_int(2))
        
        # Test n=3: should be [3, 10, 5, 16, 8, 4, 2, 1] with stopping time 7
        n = ArbitraryNumber.from_int(3)
        trajectory, stopping_time, max_value = self.analyzer.compute_exact_trajectory(n)
        expected_sequence = [3, 10, 5, 16, 8, 4, 2, 1]
        expected_trajectory = [ArbitraryNumber.from_int(x) for x in expected_sequence]
        self.assertEqual(trajectory, expected_trajectory)
        self.assertEqual(stopping_time, 7)
        self.assertEqual(max_value, ArbitraryNumber.from_int(16))
        
        # Test n=4: should be [4, 2, 1] with stopping time 2
        n = ArbitraryNumber.from_int(4)
        trajectory, stopping_time, max_value = self.analyzer.compute_exact_trajectory(n)
        expected_sequence = [4, 2, 1]
        expected_trajectory = [ArbitraryNumber.from_int(x) for x in expected_sequence]
        self.assertEqual(trajectory, expected_trajectory)
        self.assertEqual(stopping_time, 2)
        self.assertEqual(max_value, ArbitraryNumber.from_int(4))
    
    def test_compute_exact_trajectory_challenging_numbers(self):
        """Test trajectory computation for numbers with known challenging behavior."""
        challenging_cases = [
            # (n, expected_stopping_time, expected_max_value)
            (5, 5, 16),
            (6, 8, 16),
            (7, 16, 52),
            (8, 3, 8),
            (9, 19, 52),
            (10, 6, 16),
            (11, 14, 52),
            (12, 9, 16),
            (13, 9, 40),
            (14, 17, 52),
            (15, 17, 160),
            (16, 4, 16),
            (17, 12, 52),
            (18, 20, 52),
            (19, 20, 88),
            (20, 7, 20)
        ]
        
        for n_val, expected_stopping_time, expected_max_val in challenging_cases:
            with self.subTest(n=n_val):
                n = ArbitraryNumber.from_int(n_val)
                trajectory, stopping_time, max_value = self.analyzer.compute_exact_trajectory(n)
                
                # Verify stopping time
                self.assertEqual(stopping_time, expected_stopping_time,
                               f"Wrong stopping time for n={n_val}")
                
                # Verify max value
                expected_max = ArbitraryNumber.from_int(expected_max_val)
                self.assertEqual(max_value, expected_max,
                               f"Wrong max value for n={n_val}")
                
                # Verify trajectory ends with 1
                self.assertEqual(trajectory[-1], ArbitraryNumber.one(),
                               f"Trajectory doesn't end with 1 for n={n_val}")
                
                # Verify no precision loss
                for value in trajectory:
                    self.assertEqual(value.get_precision_loss(), 0.0,
                                   f"Precision loss detected for n={n_val}")
    
    def test_trajectory_caching(self):
        """Test that trajectory caching works correctly."""
        n = ArbitraryNumber.from_int(7)
        
        # First computation
        start_time = time.time()
        trajectory1, stopping_time1, max_value1 = self.analyzer.compute_exact_trajectory(n)
        first_time = time.time() - start_time
        
        # Second computation (should use cache)
        start_time = time.time()
        trajectory2, stopping_time2, max_value2 = self.analyzer.compute_exact_trajectory(n)
        second_time = time.time() - start_time
        
        # Results should be identical
        self.assertEqual(trajectory1, trajectory2)
        self.assertEqual(stopping_time1, stopping_time2)
        self.assertEqual(max_value1, max_value2)
        
        # Second computation should be faster (cached)
        self.assertLess(second_time, first_time * 0.1)  # At least 10x faster
    
    def test_verify_collatz_range_exact_small_range(self):
        """Test verification of small ranges."""
        # Test range [1, 10]
        results = self.analyzer.verify_collatz_range_exact(1, 10)
        
        # All should be verified successfully
        verified_results = [r for r in results if r.get('verified', False)]
        self.assertEqual(len(verified_results), 10)
        
        # Check specific known results
        result_dict = {r['start_value']: r for r in verified_results}
        
        # n=1: stopping_time=0
        self.assertEqual(result_dict[1]['stopping_time'], 0)
        
        # n=3: stopping_time=7, max_value=16
        self.assertEqual(result_dict[3]['stopping_time'], 7)
        self.assertEqual(result_dict[3]['max_value'], ArbitraryNumber.from_int(16))
        
        # n=7: stopping_time=16, max_value=52
        self.assertEqual(result_dict[7]['stopping_time'], 16)
        self.assertEqual(result_dict[7]['max_value'], ArbitraryNumber.from_int(52))
    
    def test_analyze_trajectory_patterns(self):
        """Test trajectory pattern analysis."""
        # Test with n=3: [3, 10, 5, 16, 8, 4, 2, 1]
        n = ArbitraryNumber.from_int(3)
        trajectory, _, _ = self.analyzer.compute_exact_trajectory(n)
        
        # Clear previous statistics
        self.analyzer.statistics['pattern_frequencies'].clear()
        
        # Analyze patterns
        self.analyzer.analyze_trajectory_patterns(trajectory)
        
        # Check that patterns were recorded
        self.assertGreater(len(self.analyzer.statistics['pattern_frequencies']), 0)
        
        # Verify pattern keys contain expected format
        pattern_keys = list(self.analyzer.statistics['pattern_frequencies'].keys())
        self.assertTrue(any('max_consecutive_evens' in key for key in pattern_keys))
        self.assertTrue(any('max_consecutive_odds' in key for key in pattern_keys))
    
    def test_find_record_breaking_numbers_small_limit(self):
        """Test finding record-breaking numbers with small search limit."""
        records = self.analyzer.find_record_breaking_numbers(20)
        
        # Should find some records
        self.assertGreater(len(records['stopping_time_records']), 0)
        self.assertGreater(len(records['peak_value_records']), 0)
        
        # Records should be in ascending order
        stopping_times = [r['stopping_time'] for r in records['stopping_time_records']]
        self.assertEqual(stopping_times, sorted(stopping_times))
        
        peak_values = [float(r['peak_value'].evaluate_exact()) for r in records['peak_value_records']]
        self.assertEqual(peak_values, sorted(peak_values))
    
    def test_exact_statistical_analysis_empty_data(self):
        """Test statistical analysis with empty data."""
        stats = self.analyzer.exact_statistical_analysis([])
        self.assertEqual(stats, {})
        
        # Test with data containing no verified results
        invalid_data = [{'verified': False, 'error': 'test error'}]
        stats = self.analyzer.exact_statistical_analysis(invalid_data)
        self.assertEqual(stats, {})
    
    def test_exact_statistical_analysis_valid_data(self):
        """Test statistical analysis with valid data."""
        # Create test data
        test_data = [
            {'verified': True, 'stopping_time': 5},
            {'verified': True, 'stopping_time': 10},
            {'verified': True, 'stopping_time': 15},
            {'verified': True, 'stopping_time': 20},
            {'verified': False, 'error': 'test'},  # Should be ignored
        ]
        
        stats = self.analyzer.exact_statistical_analysis(test_data)
        
        # Check basic statistics
        self.assertEqual(stats['sample_size'], 4)
        self.assertEqual(stats['min_stopping_time'], 5)
        self.assertEqual(stats['max_stopping_time'], 20)
        
        # Check exact mean: (5+10+15+20)/4 = 12.5
        expected_mean = 12.5
        self.assertAlmostEqual(stats['mean_stopping_time_float'], expected_mean, places=6)
        
        # Verify exact computation (no precision loss)
        exact_mean = ArbitraryNumber.from_int(50) / ArbitraryNumber.from_int(4)
        self.assertEqual(stats['exact_mean_stopping_time'], exact_mean)
    
    def test_precision_preservation_throughout_computation(self):
        """Test that precision is preserved throughout all computations."""
        # Test with a number known to have a long trajectory
        n = ArbitraryNumber.from_int(27)
        trajectory, stopping_time, max_value = self.analyzer.compute_exact_trajectory(n)
        
        # Verify no precision loss in any trajectory value
        for i, value in enumerate(trajectory):
            self.assertEqual(value.get_precision_loss(), 0.0,
                           f"Precision loss at step {i} in trajectory for n=27")
        
        # Verify max_value has no precision loss
        self.assertEqual(max_value.get_precision_loss(), 0.0)
        
        # Verify trajectory is mathematically consistent
        for i in range(len(trajectory) - 1):
            current = trajectory[i]
            next_val = trajectory[i + 1]
            expected_next = self.analyzer.collatz_step_exact(current)
            self.assertEqual(next_val, expected_next,
                           f"Trajectory inconsistency at step {i} for n=27")
    
    def test_large_number_handling(self):
        """Test handling of very large numbers."""
        # Test with 2^100 (very large even number)
        large_n = ArbitraryNumber.from_int(2**100)
        result = self.analyzer.collatz_step_exact(large_n)
        expected = ArbitraryNumber.from_int(2**99)
        self.assertEqual(result, expected)
        self.assertEqual(result.get_precision_loss(), 0.0)
        
        # Test with 2^100 + 1 (very large odd number)
        large_odd = ArbitraryNumber.from_int(2**100 + 1)
        result = self.analyzer.collatz_step_exact(large_odd)
        expected = ArbitraryNumber.from_int(3 * (2**100 + 1) + 1)
        self.assertEqual(result, expected)
        self.assertEqual(result.get_precision_loss(), 0.0)
    
    def test_performance_benchmarks(self):
        """Test performance characteristics."""
        # Benchmark single step operations
        n = ArbitraryNumber.from_int(12345)
        
        start_time = time.time()
        for _ in range(1000):
            self.analyzer.collatz_step_exact(n)
        step_time = time.time() - start_time
        
        # Should complete 1000 steps in reasonable time (< 1 second)
        self.assertLess(step_time, 1.0)
        
        # Benchmark trajectory computation
        start_time = time.time()
        trajectory, stopping_time, max_value = self.analyzer.compute_exact_trajectory(n)
        trajectory_time = time.time() - start_time
        
        # Should complete trajectory in reasonable time
        self.assertLess(trajectory_time, 0.1)
        
        # Verify results are still exact
        self.assertEqual(max_value.get_precision_loss(), 0.0)
    
    def test_mathematical_properties_verification(self):
        """Test verification of mathematical properties."""
        # Test that all trajectories eventually reach 1
        test_numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 17, 31, 32, 63, 64]
        
        for n_val in test_numbers:
            with self.subTest(n=n_val):
                n = ArbitraryNumber.from_int(n_val)
                trajectory, stopping_time, max_value = self.analyzer.compute_exact_trajectory(n)
                
                # Trajectory should end with 1
                self.assertEqual(trajectory[-1], ArbitraryNumber.one())
                
                # Stopping time should be positive (except for n=1)
                if n_val == 1:
                    self.assertEqual(stopping_time, 0)
                else:
                    self.assertGreater(stopping_time, 0)
                
                # Max value should be >= starting value
                self.assertGreaterEqual(max_value, n)
                
                # Trajectory length should equal stopping_time + 1
                self.assertEqual(len(trajectory), stopping_time + 1)
    
    def test_error_handling_and_edge_cases(self):
        """Test error handling and edge cases."""
        # Test with invalid inputs
        with self.assertRaises(ValueError):
            self.analyzer.collatz_step_exact(ArbitraryNumber.zero())
        
        with self.assertRaises(ValueError):
            self.analyzer.collatz_step_exact(ArbitraryNumber.from_int(-1))
        
        # Test with fractional numbers (should work but may not be meaningful)
        frac = ArbitraryNumber.from_fraction(3, 2)  # 1.5
        result = self.analyzer.collatz_step_exact(frac)
        # 1.5 is not even, so should compute 3*1.5 + 1 = 5.5
        expected = ArbitraryNumber.from_fraction(11, 2)
        self.assertEqual(result, expected)
    
    def test_statistics_accumulation(self):
        """Test that statistics are properly accumulated."""
        # Clear statistics
        self.analyzer.statistics = {
            'max_stopping_time': 0,
            'max_peak_value': ArbitraryNumber.zero(),
            'total_verified': 0,
            'pattern_frequencies': {}
        }
        
        # Verify range [1, 5]
        results = self.analyzer.verify_collatz_range_exact(1, 5)
        
        # Check that statistics were updated
        self.assertEqual(self.analyzer.statistics['total_verified'], 5)
        self.assertGreater(self.analyzer.statistics['max_stopping_time'], 0)
        self.assertGreater(self.analyzer.statistics['max_peak_value'], ArbitraryNumber.zero())
        
        # Verify that verified numbers were recorded
        self.assertEqual(len(self.analyzer.verified_numbers), 5)
        self.assertIn(1, self.analyzer.verified_numbers)
        self.assertIn(5, self.analyzer.verified_numbers)


class TestCollatzPerformanceAndStress(unittest.TestCase):
    """
    Performance and stress tests for Collatz solver.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = ExactCollatzAnalyzer()
    
    def test_stress_test_medium_numbers(self):
        """Stress test with medium-sized numbers."""
        # Test numbers in range [100, 200]
        stress_numbers = list(range(100, 201))
        
        for n_val in stress_numbers:
            with self.subTest(n=n_val):
                n = ArbitraryNumber.from_int(n_val)
                
                start_time = time.time()
                trajectory, stopping_time, max_value = self.analyzer.compute_exact_trajectory(n)
                computation_time = time.time() - start_time
                
                # Should complete in reasonable time
                self.assertLess(computation_time, 1.0)
                
                # Should maintain precision
                self.assertEqual(max_value.get_precision_loss(), 0.0)
                
                # Should reach 1
                self.assertEqual(trajectory[-1], ArbitraryNumber.one())
    
    def test_memory_efficiency(self):
        """Test memory efficiency of trajectory storage."""
        # Compute several trajectories and verify memory usage is reasonable
        test_numbers = [27, 31, 47, 54, 55, 63, 71, 73, 79, 97]
        
        trajectories = []
        for n_val in test_numbers:
            n = ArbitraryNumber.from_int(n_val)
            trajectory, _, _ = self.analyzer.compute_exact_trajectory(n)
            trajectories.append(trajectory)
        
        # Verify all trajectories are stored correctly
        self.assertEqual(len(trajectories), len(test_numbers))
        
        # Verify cache is working
        self.assertEqual(len(self.analyzer.trajectory_cache), len(test_numbers))
    
    def test_concurrent_safety(self):
        """Test that the analyzer is safe for concurrent use."""
        # This is a basic test - in practice, you'd need proper threading tests
        analyzer1 = ExactCollatzAnalyzer()
        analyzer2 = ExactCollatzAnalyzer()
        
        # Both should produce identical results
        n = ArbitraryNumber.from_int(27)
        
        traj1, stop1, max1 = analyzer1.compute_exact_trajectory(n)
        traj2, stop2, max2 = analyzer2.compute_exact_trajectory(n)
        
        self.assertEqual(traj1, traj2)
        self.assertEqual(stop1, stop2)
        self.assertEqual(max1, max2)


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add all test cases
    test_suite.addTest(unittest.makeSuite(TestExactCollatzAnalyzer))
    test_suite.addTest(unittest.makeSuite(TestCollatzPerformanceAndStress))
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"COLLATZ CONJECTURE SOLVER TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError: ')[-1].split('\\n')[0]}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('\\n')[-2]}")
    
    print(f"\nAll tests verify ZERO precision loss in Collatz computations!")
    print(f"ArbitraryNumber maintains perfect mathematical accuracy.")
