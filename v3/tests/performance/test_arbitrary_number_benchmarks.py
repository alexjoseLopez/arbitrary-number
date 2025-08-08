"""
Performance Benchmark Tests for ArbitraryNumber
==============================================

Comprehensive benchmark suite for performance testing and optimization validation.
"""

import unittest
import sys
import os
import time
import statistics
from fractions import Fraction
import random
import gc

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from arbitrary_numbers.core.arbitrary_number import ArbitraryNumber, FractionTerm


class TestArbitraryNumberPerformanceBasics(unittest.TestCase):
    """Basic performance tests for ArbitraryNumber operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Disable garbage collection during timing for more consistent results
        gc.disable()
    
    def tearDown(self):
        """Clean up after tests."""
        gc.enable()
        gc.collect()
    
    def time_operation(self, operation, iterations=1000):
        """Time an operation over multiple iterations."""
        times = []
        
        for _ in range(iterations):
            start_time = time.perf_counter()
            operation()
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        return {
            'mean': statistics.mean(times),
            'median': statistics.median(times),
            'stdev': statistics.stdev(times) if len(times) > 1 else 0,
            'min': min(times),
            'max': max(times),
            'total': sum(times)
        }
    
    def test_creation_performance(self):
        """Benchmark ArbitraryNumber creation methods."""
        iterations = 10000
        
        # Test from_int
        def create_from_int():
            return ArbitraryNumber.from_int(42)
        
        int_stats = self.time_operation(create_from_int, iterations)
        
        # Test from_fraction
        def create_from_fraction():
            return ArbitraryNumber.from_fraction(3, 4)
        
        fraction_stats = self.time_operation(create_from_fraction, iterations)
        
        # Test from_decimal
        def create_from_decimal():
            return ArbitraryNumber.from_decimal(0.75)
        
        decimal_stats = self.time_operation(create_from_decimal, iterations)
        
        print(f"Creation benchmarks ({iterations} iterations):")
        print(f"  from_int: {int_stats['mean']*1000:.4f}ms avg")
        print(f"  from_fraction: {fraction_stats['mean']*1000:.4f}ms avg")
        print(f"  from_decimal: {decimal_stats['mean']*1000:.4f}ms avg")
        
        # All creation methods should be reasonably fast
        self.assertLess(int_stats['mean'], 0.001)  # Less than 1ms
        self.assertLess(fraction_stats['mean'], 0.001)
        self.assertLess(decimal_stats['mean'], 0.01)  # Decimal conversion is slower
    
    def test_arithmetic_performance(self):
        """Benchmark basic arithmetic operations."""
        iterations = 5000
        
        # Create test numbers
        num1 = ArbitraryNumber.from_fraction(3, 4)
        num2 = ArbitraryNumber.from_fraction(5, 6)
        
        # Test addition
        def test_addition():
            return num1 + num2
        
        add_stats = self.time_operation(test_addition, iterations)
        
        # Test multiplication
        def test_multiplication():
            return num1 * num2
        
        mul_stats = self.time_operation(test_multiplication, iterations)
        
        # Test division
        def test_division():
            return num1 / num2
        
        div_stats = self.time_operation(test_division, iterations)
        
        print(f"Arithmetic benchmarks ({iterations} iterations):")
        print(f"  Addition: {add_stats['mean']*1000:.4f}ms avg")
        print(f"  Multiplication: {mul_stats['mean']*1000:.4f}ms avg")
        print(f"  Division: {div_stats['mean']*1000:.4f}ms avg")
        
        # Operations should be reasonably fast
        self.assertLess(add_stats['mean'], 0.001)
        self.assertLess(mul_stats['mean'], 0.001)
        self.assertLess(div_stats['mean'], 0.001)
    
    def test_evaluation_performance(self):
        """Benchmark evaluation methods."""
        iterations = 1000
        
        # Create complex number with multiple terms
        complex_num = ArbitraryNumber([
            FractionTerm(1, 2),
            FractionTerm(1, 3),
            FractionTerm(1, 4),
            FractionTerm(1, 5),
            FractionTerm(1, 6)
        ])
        
        # Test exact evaluation
        def test_exact_eval():
            return complex_num.evaluate_exact()
        
        exact_stats = self.time_operation(test_exact_eval, iterations)
        
        # Test decimal evaluation
        def test_decimal_eval():
            return complex_num.evaluate(precision=50)
        
        decimal_stats = self.time_operation(test_decimal_eval, iterations)
        
        print(f"Evaluation benchmarks ({iterations} iterations):")
        print(f"  Exact evaluation: {exact_stats['mean']*1000:.4f}ms avg")
        print(f"  Decimal evaluation: {decimal_stats['mean']*1000:.4f}ms avg")
        
        # Evaluation should be reasonably fast
        self.assertLess(exact_stats['mean'], 0.01)
        self.assertLess(decimal_stats['mean'], 0.01)
    
    def test_caching_performance(self):
        """Test performance benefits of caching."""
        iterations = 1000
        
        # Create number for testing
        test_num = ArbitraryNumber([
            FractionTerm(1, 7),
            FractionTerm(2, 11),
            FractionTerm(3, 13)
        ])
        
        # First evaluation (no cache)
        def first_eval():
            # Clear cache first
            test_num._cached_exact_value = None
            return test_num.evaluate_exact()
        
        first_stats = self.time_operation(first_eval, iterations)
        
        # Second evaluation (with cache)
        # Ensure cache is populated
        test_num.evaluate_exact()
        
        def cached_eval():
            return test_num.evaluate_exact()
        
        cached_stats = self.time_operation(cached_eval, iterations)
        
        print(f"Caching benchmarks ({iterations} iterations):")
        print(f"  First evaluation: {first_stats['mean']*1000:.4f}ms avg")
        print(f"  Cached evaluation: {cached_stats['mean']*1000:.4f}ms avg")
        print(f"  Speedup: {first_stats['mean']/cached_stats['mean']:.2f}x")
        
        # Cached evaluation should be significantly faster
        self.assertLess(cached_stats['mean'], first_stats['mean'])


class TestArbitraryNumberScalabilityBenchmarks(unittest.TestCase):
    """Test scalability with increasing problem sizes."""
    
    def test_term_count_scaling(self):
        """Test performance scaling with number of terms."""
        term_counts = [1, 5, 10, 25, 50, 100]
        results = {}
        
        for count in term_counts:
            # Create ArbitraryNumber with specified number of terms
            terms = [FractionTerm(i + 1, i + 2) for i in range(count)]
            test_num = ArbitraryNumber(terms)
            
            # Time evaluation
            start_time = time.perf_counter()
            for _ in range(100):  # Multiple iterations for stability
                result = test_num.evaluate_exact()
            end_time = time.perf_counter()
            
            avg_time = (end_time - start_time) / 100
            results[count] = avg_time
            
            print(f"Terms: {count:3d}, Time: {avg_time*1000:.4f}ms")
        
        # Performance should scale reasonably (not exponentially)
        for i in range(1, len(term_counts)):
            prev_count = term_counts[i-1]
            curr_count = term_counts[i]
            prev_time = results[prev_count]
            curr_time = results[curr_count]
            
            # Time should not increase more than proportionally to term count
            ratio = curr_time / prev_time
            count_ratio = curr_count / prev_count
            
            # Allow some overhead, but should be roughly linear
            self.assertLess(ratio, count_ratio * 2)
    
    def test_precision_scaling(self):
        """Test performance scaling with decimal precision."""
        precisions = [10, 25, 50, 100, 200]
        test_num = ArbitraryNumber.from_fraction(1, 7)  # Repeating decimal
        results = {}
        
        for precision in precisions:
            start_time = time.perf_counter()
            for _ in range(100):
                result = test_num.evaluate(precision=precision)
            end_time = time.perf_counter()
            
            avg_time = (end_time - start_time) / 100
            results[precision] = avg_time
            
            print(f"Precision: {precision:3d}, Time: {avg_time*1000:.4f}ms")
        
        # Higher precision should take more time, but not exponentially
        for i in range(1, len(precisions)):
            prev_prec = precisions[i-1]
            curr_prec = precisions[i]
            prev_time = results[prev_prec]
            curr_time = results[curr_prec]
            
            # Time increase should be reasonable
            ratio = curr_time / prev_time
            prec_ratio = curr_prec / prev_prec
            
            # Allow some overhead for higher precision
            self.assertLess(ratio, prec_ratio * 3)
    
    def test_operation_chain_scaling(self):
        """Test performance with chains of operations."""
        chain_lengths = [1, 5, 10, 20, 50]
        results = {}
        
        for length in chain_lengths:
            # Create chain of operations
            start_time = time.perf_counter()
            
            result = ArbitraryNumber.from_fraction(1, 2)
            for i in range(length):
                term = ArbitraryNumber.from_fraction(1, i + 3)
                result = result + term
            
            final_value = result.evaluate_exact()
            end_time = time.perf_counter()
            
            total_time = end_time - start_time
            results[length] = total_time
            
            print(f"Chain length: {length:2d}, Time: {total_time*1000:.4f}ms, Terms: {len(result.terms)}")
        
        # Longer chains should take more time, but growth should be manageable
        for i in range(1, len(chain_lengths)):
            prev_length = chain_lengths[i-1]
            curr_length = chain_lengths[i]
            prev_time = results[prev_length]
            curr_time = results[curr_length]
            
            # Time should not grow exponentially
            ratio = curr_time / prev_time
            length_ratio = curr_length / prev_length
            
            # Allow reasonable growth
            self.assertLess(ratio, length_ratio * 2)


class TestArbitraryNumberMemoryBenchmarks(unittest.TestCase):
    """Test memory usage and efficiency."""
    
    def test_memory_usage_scaling(self):
        """Test memory usage scaling with problem size."""
        sizes = [100, 500, 1000, 2000, 5000]
        
        for size in sizes:
            # Create collection of ArbitraryNumbers
            numbers = []
            for i in range(size):
                if i % 3 == 0:
                    num = ArbitraryNumber.from_int(i)
                elif i % 3 == 1:
                    num = ArbitraryNumber.from_fraction(i, i + 1)
                else:
                    num = ArbitraryNumber([
                        FractionTerm(i, 10),
                        FractionTerm(1, i + 1)
                    ])
                numbers.append(num)
            
            # Calculate total memory usage
            total_memory = sum(num.memory_usage() for num in numbers)
            avg_memory = total_memory / size
            
            print(f"Size: {size:4d}, Total: {total_memory:8d} bytes, Avg: {avg_memory:.2f} bytes/number")
            
            # Memory usage should be reasonable
            self.assertLess(avg_memory, 1000)  # Less than 1KB per number on average
    
    def test_memory_efficiency_comparison(self):
        """Compare memory efficiency of different representations."""
        test_value = Fraction(355, 113)  # Pi approximation
        
        # ArbitraryNumber representation
        arb_num = ArbitraryNumber.from_fraction(355, 113)
        arb_memory = arb_num.memory_usage()
        
        # Python Fraction representation (estimate)
        py_fraction = Fraction(355, 113)
        py_memory = sys.getsizeof(py_fraction)
        
        # Float representation
        float_val = float(test_value)
        float_memory = sys.getsizeof(float_val)
        
        print(f"Memory comparison for {test_value}:")
        print(f"  ArbitraryNumber: {arb_memory} bytes")
        print(f"  Python Fraction: {py_memory} bytes")
        print(f"  Float: {float_memory} bytes")
        
        # ArbitraryNumber should be competitive with Fraction
        # (allowing some overhead for additional features)
        self.assertLess(arb_memory, py_memory * 3)
    
    def test_garbage_collection_impact(self):
        """Test impact of garbage collection on performance."""
        iterations = 1000
        
        # Test with garbage collection enabled
        gc.enable()
        start_time = time.perf_counter()
        
        for i in range(iterations):
            num1 = ArbitraryNumber.from_fraction(i, i + 1)
            num2 = ArbitraryNumber.from_fraction(i + 1, i + 2)
            result = num1 + num2
            exact = result.evaluate_exact()
        
        gc_enabled_time = time.perf_counter() - start_time
        
        # Test with garbage collection disabled
        gc.disable()
        start_time = time.perf_counter()
        
        for i in range(iterations):
            num1 = ArbitraryNumber.from_fraction(i, i + 1)
            num2 = ArbitraryNumber.from_fraction(i + 1, i + 2)
            result = num1 + num2
            exact = result.evaluate_exact()
        
        gc_disabled_time = time.perf_counter() - start_time
        gc.enable()  # Re-enable for cleanup
        
        print(f"Garbage collection impact ({iterations} iterations):")
        print(f"  GC enabled: {gc_enabled_time:.4f}s")
        print(f"  GC disabled: {gc_disabled_time:.4f}s")
        print(f"  Overhead: {((gc_enabled_time - gc_disabled_time) / gc_disabled_time * 100):.1f}%")
        
        # Both should complete successfully
        self.assertGreater(gc_enabled_time, 0)
        self.assertGreater(gc_disabled_time, 0)


class TestArbitraryNumberComparisonBenchmarks(unittest.TestCase):
    """Compare ArbitraryNumber performance with alternatives."""
    
    def test_vs_python_fraction(self):
        """Compare performance with Python's built-in Fraction."""
        iterations = 1000
        
        # Test data
        test_pairs = [
            (Fraction(1, 3), Fraction(1, 4)),
            (Fraction(22, 7), Fraction(355, 113)),
            (Fraction(1234, 5678), Fraction(9876, 5432))
        ]
        
        for frac1, frac2 in test_pairs:
            # Python Fraction operations
            start_time = time.perf_counter()
            for _ in range(iterations):
                result = frac1 + frac2
                result = result * Fraction(2, 3)
                final = float(result)
            py_time = time.perf_counter() - start_time
            
            # ArbitraryNumber operations
            arb1 = ArbitraryNumber.from_fraction(frac1.numerator, frac1.denominator)
            arb2 = ArbitraryNumber.from_fraction(frac2.numerator, frac2.denominator)
            arb_mult = ArbitraryNumber.from_fraction(2, 3)
            
            start_time = time.perf_counter()
            for _ in range(iterations):
                result = arb1 + arb2
                result = result * arb_mult
                final = result.to_float()
            arb_time = time.perf_counter() - start_time
            
            ratio = arb_time / py_time
            print(f"Fraction {frac1} + {frac2}:")
            print(f"  Python Fraction: {py_time*1000:.4f}ms")
            print(f"  ArbitraryNumber: {arb_time*1000:.4f}ms")
            print(f"  Ratio: {ratio:.2f}x")
            
            # ArbitraryNumber should be competitive (within 10x)
            self.assertLess(ratio, 10.0)
    
    def test_vs_float_operations(self):
        """Compare performance with float operations."""
        iterations = 10000
        
        # Float operations
        start_time = time.perf_counter()
        for i in range(iterations):
            a = (i + 1) / (i + 2)
            b = (i + 2) / (i + 3)
            result = a + b
            result = result * 0.75
        float_time = time.perf_counter() - start_time
        
        # ArbitraryNumber operations
        start_time = time.perf_counter()
        for i in range(iterations):
            a = ArbitraryNumber.from_fraction(i + 1, i + 2)
            b = ArbitraryNumber.from_fraction(i + 2, i + 3)
            result = a + b
            result = result * ArbitraryNumber.from_fraction(3, 4)
        arb_time = time.perf_counter() - start_time
        
        ratio = arb_time / float_time
        print(f"Float vs ArbitraryNumber ({iterations} iterations):")
        print(f"  Float operations: {float_time*1000:.4f}ms")
        print(f"  ArbitraryNumber operations: {arb_time*1000:.4f}ms")
        print(f"  Ratio: {ratio:.2f}x")
        
        # ArbitraryNumber will be slower than float, but should be reasonable
        self.assertLess(ratio, 1000.0)  # Less than 1000x slower
    
    def test_precision_vs_performance_tradeoff(self):
        """Test precision vs performance tradeoffs."""
        test_cases = [
            ("Simple fraction", ArbitraryNumber.from_fraction(1, 3)),
            ("Complex fraction", ArbitraryNumber.from_fraction(355, 113)),
            ("Multiple terms", ArbitraryNumber([
                FractionTerm(1, 3),
                FractionTerm(1, 7),
                FractionTerm(1, 11)
            ])),
            ("Large numbers", ArbitraryNumber.from_fraction(123456789, 987654321))
        ]
        
        for name, arb_num in test_cases:
            # Time exact evaluation
            start_time = time.perf_counter()
            for _ in range(1000):
                exact = arb_num.evaluate_exact()
            exact_time = time.perf_counter() - start_time
            
            # Time float conversion
            start_time = time.perf_counter()
            for _ in range(1000):
                approx = arb_num.to_float()
            float_time = time.perf_counter() - start_time
            
            # Calculate precision loss
            exact_val = arb_num.evaluate_exact()
            float_val = arb_num.to_float()
            precision_loss = abs(float(exact_val) - float_val)
            
            print(f"{name}:")
            print(f"  Exact evaluation: {exact_time*1000:.4f}ms")
            print(f"  Float conversion: {float_time*1000:.4f}ms")
            print(f"  Precision loss: {precision_loss}")
            print(f"  ArbitraryNumber precision loss: {arb_num.get_precision_loss()}")
            
            # ArbitraryNumber should have zero precision loss
            self.assertEqual(arb_num.get_precision_loss(), 0.0)


if __name__ == '__main__':
    print("Running ArbitraryNumber Performance Benchmarks")
    print("=" * 50)
    
    # Run with higher verbosity to see benchmark results
    unittest.main(verbosity=2)
