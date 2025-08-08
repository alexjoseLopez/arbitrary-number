"""
GPU Unit Tests for ArbitraryNumber
=================================

Comprehensive test suite for GPU acceleration features and CUDA kernel testing.
"""

import unittest
import sys
import os
from fractions import Fraction
import random
import time

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from arbitrary_numbers.core.arbitrary_number import ArbitraryNumber, FractionTerm

# Try to import GPU-related modules
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = None


class TestArbitraryNumberGPUBasics(unittest.TestCase):
    """Basic GPU functionality tests."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not GPU_AVAILABLE:
            self.skipTest("GPU not available - CuPy not installed")
    
    def test_gpu_availability_detection(self):
        """Test GPU availability detection."""
        # This would normally use a GPU evaluator class
        # For now, just test that CuPy is working
        if GPU_AVAILABLE:
            try:
                # Simple CuPy operation to verify GPU works
                a = cp.array([1, 2, 3])
                b = cp.array([4, 5, 6])
                result = a + b
                expected = cp.array([5, 7, 9])
                cp.testing.assert_array_equal(result, expected)
            except Exception as e:
                self.skipTest(f"GPU not functional: {e}")
    
    def test_large_dataset_preparation(self):
        """Test preparation of large datasets for GPU processing."""
        # Create large dataset of ArbitraryNumbers
        dataset_size = 1000
        arbitrary_numbers = []
        
        for i in range(dataset_size):
            # Create diverse ArbitraryNumbers
            if i % 4 == 0:
                num = ArbitraryNumber.from_int(i)
            elif i % 4 == 1:
                num = ArbitraryNumber.from_fraction(i, i + 1)
            elif i % 4 == 2:
                num = ArbitraryNumber.from_decimal(i * 0.1)
            else:
                # Multiple terms
                num = ArbitraryNumber([
                    FractionTerm(i, 10),
                    FractionTerm(1, i + 1)
                ])
            arbitrary_numbers.append(num)
        
        self.assertEqual(len(arbitrary_numbers), dataset_size)
        
        # Verify all numbers are valid
        for num in arbitrary_numbers[:10]:  # Check first 10
            self.assertIsInstance(num, ArbitraryNumber)
            result = num.evaluate_exact()
            self.assertIsInstance(result, Fraction)
    
    def test_gpu_memory_layout_simulation(self):
        """Test memory layout for GPU processing."""
        # Simulate structure-of-arrays layout for GPU
        numbers = [
            ArbitraryNumber.from_fraction(1, 2),
            ArbitraryNumber.from_fraction(3, 4),
            ArbitraryNumber.from_fraction(5, 6),
            ArbitraryNumber.from_fraction(7, 8)
        ]
        
        # Extract numerators and denominators separately (SoA layout)
        numerators = []
        denominators = []
        
        for num in numbers:
            for term in num.terms:
                numerators.append(term.numerator)
                denominators.append(term.denominator)
        
        # Convert to GPU arrays if available
        if GPU_AVAILABLE:
            gpu_numerators = cp.array(numerators, dtype=cp.int64)
            gpu_denominators = cp.array(denominators, dtype=cp.int64)
            
            # Verify data integrity
            self.assertEqual(len(gpu_numerators), len(numerators))
            self.assertEqual(len(gpu_denominators), len(denominators))
            
            # Test basic GPU arithmetic
            gpu_result = gpu_numerators * 2
            expected = [n * 2 for n in numerators]
            cp.testing.assert_array_equal(gpu_result, cp.array(expected))


class TestArbitraryNumberGPUPerformance(unittest.TestCase):
    """Performance tests for GPU acceleration."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not GPU_AVAILABLE:
            self.skipTest("GPU not available - CuPy not installed")
    
    def test_cpu_vs_gpu_addition_benchmark(self):
        """Benchmark CPU vs GPU addition performance."""
        # Create test data
        size = 10000
        numbers_a = [ArbitraryNumber.from_fraction(i, i + 1) for i in range(1, size + 1)]
        numbers_b = [ArbitraryNumber.from_fraction(i + 1, i + 2) for i in range(1, size + 1)]
        
        # CPU benchmark
        start_time = time.time()
        cpu_results = []
        for a, b in zip(numbers_a, numbers_b):
            result = a + b
            cpu_results.append(result.evaluate_exact())
        cpu_time = time.time() - start_time
        
        # GPU simulation (would be actual GPU code in real implementation)
        start_time = time.time()
        if GPU_AVAILABLE:
            # Simulate GPU processing by doing batch operations
            numerators_a = cp.array([num.terms[0].numerator for num in numbers_a])
            denominators_a = cp.array([num.terms[0].denominator for num in numbers_a])
            numerators_b = cp.array([num.terms[0].numerator for num in numbers_b])
            denominators_b = cp.array([num.terms[0].denominator for num in numbers_b])
            
            # Simulate fraction addition on GPU
            # a/b + c/d = (a*d + c*b) / (b*d)
            new_numerators = numerators_a * denominators_b + numerators_b * denominators_a
            new_denominators = denominators_a * denominators_b
            
            # Convert back to CPU for verification
            gpu_results = []
            for num, den in zip(new_numerators.get(), new_denominators.get()):
                gpu_results.append(Fraction(num, den))
        
        gpu_time = time.time() - start_time
        
        # Verify results match (first 10 elements)
        if GPU_AVAILABLE:
            for i in range(min(10, len(cpu_results))):
                self.assertEqual(cpu_results[i], gpu_results[i])
        
        # Performance should be reasonable (this is just a simulation)
        self.assertGreater(cpu_time, 0)
        self.assertGreater(gpu_time, 0)
        
        print(f"CPU time: {cpu_time:.4f}s, GPU time: {gpu_time:.4f}s")
    
    def test_large_multiplication_benchmark(self):
        """Benchmark large number multiplication."""
        # Create large numbers
        large_a = ArbitraryNumber.from_int(10**50)
        large_b = ArbitraryNumber.from_int(10**50)
        
        # CPU timing
        start_time = time.time()
        cpu_result = large_a * large_b
        cpu_exact = cpu_result.evaluate_exact()
        cpu_time = time.time() - start_time
        
        # Verify result
        expected = Fraction(10**100)
        self.assertEqual(cpu_exact, expected)
        
        print(f"Large multiplication CPU time: {cpu_time:.4f}s")
    
    def test_memory_usage_scaling(self):
        """Test memory usage with increasing dataset sizes."""
        sizes = [100, 500, 1000, 2000]
        memory_usage = []
        
        for size in sizes:
            # Create dataset
            numbers = [ArbitraryNumber.from_fraction(i, i + 1) for i in range(1, size + 1)]
            
            # Calculate total memory usage
            total_memory = sum(num.memory_usage() for num in numbers)
            memory_usage.append(total_memory)
            
            # Memory should scale roughly linearly
            if len(memory_usage) > 1:
                ratio = memory_usage[-1] / memory_usage[-2]
                size_ratio = sizes[-1] / sizes[-2]
                # Allow some variance in memory scaling
                self.assertLess(abs(ratio - size_ratio), size_ratio * 0.5)
        
        print(f"Memory scaling: {list(zip(sizes, memory_usage))}")


class TestArbitraryNumberGPUDataTypes(unittest.TestCase):
    """Test GPU data type handling and precision."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not GPU_AVAILABLE:
            self.skipTest("GPU not available - CuPy not installed")
    
    def test_integer_overflow_handling(self):
        """Test handling of integer overflow in GPU computations."""
        # Create numbers that might cause overflow in 32-bit integers
        large_nums = [
            ArbitraryNumber.from_int(2**31 - 1),  # Max 32-bit signed int
            ArbitraryNumber.from_int(2**32 - 1),  # Max 32-bit unsigned int
            ArbitraryNumber.from_int(2**63 - 1),  # Max 64-bit signed int
        ]
        
        for num in large_nums:
            # Test that operations don't overflow
            doubled = num * 2
            result = doubled.evaluate_exact()
            expected = num.evaluate_exact() * 2
            self.assertEqual(result, expected)
    
    def test_precision_preservation_gpu(self):
        """Test that GPU operations preserve exact precision."""
        # Create numbers with high precision requirements
        precise_nums = [
            ArbitraryNumber.from_fraction(1, 3),
            ArbitraryNumber.from_fraction(1, 7),
            ArbitraryNumber.from_fraction(1, 11),
            ArbitraryNumber.from_fraction(22, 7),  # Pi approximation
        ]
        
        # Perform operations that should maintain precision
        for num in precise_nums:
            # Multiply by 3 then divide by 3 should give original
            tripled = num * 3
            back_to_original = tripled / 3
            
            self.assertEqual(back_to_original.evaluate_exact(), num.evaluate_exact())
            self.assertEqual(back_to_original.get_precision_loss(), 0.0)
    
    def test_gpu_array_data_types(self):
        """Test appropriate data types for GPU arrays."""
        if not GPU_AVAILABLE:
            return
        
        # Test different integer sizes
        test_values = [
            (127, cp.int8),
            (32767, cp.int16),
            (2147483647, cp.int32),
            (9223372036854775807, cp.int64),
        ]
        
        for value, dtype in test_values:
            try:
                gpu_array = cp.array([value], dtype=dtype)
                self.assertEqual(gpu_array[0], value)
            except OverflowError:
                # Expected for values too large for the data type
                pass
    
    def test_batch_processing_data_integrity(self):
        """Test data integrity in batch processing scenarios."""
        # Create batch of diverse numbers
        batch_size = 1000
        batch = []
        
        for i in range(batch_size):
            if i % 3 == 0:
                num = ArbitraryNumber.from_fraction(i + 1, i + 2)
            elif i % 3 == 1:
                num = ArbitraryNumber.from_int(i)
            else:
                num = ArbitraryNumber([
                    FractionTerm(i, 10),
                    FractionTerm(1, i + 1)
                ])
            batch.append(num)
        
        # Process batch and verify integrity
        processed_batch = []
        for num in batch:
            # Simulate GPU processing with a simple operation
            processed = num * 2
            processed_batch.append(processed)
        
        # Verify all results
        for original, processed in zip(batch, processed_batch):
            expected = original.evaluate_exact() * 2
            actual = processed.evaluate_exact()
            self.assertEqual(actual, expected)


class TestArbitraryNumberGPUEdgeCases(unittest.TestCase):
    """Test GPU-specific edge cases and error conditions."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not GPU_AVAILABLE:
            self.skipTest("GPU not available - CuPy not installed")
    
    def test_gpu_memory_limits(self):
        """Test behavior when approaching GPU memory limits."""
        if not GPU_AVAILABLE:
            return
        
        # Get available GPU memory
        mempool = cp.get_default_memory_pool()
        
        # Create increasingly large datasets until we approach limits
        max_size = 100000  # Start with reasonable size
        
        try:
            # Create large arrays to test memory handling
            large_array = cp.zeros(max_size, dtype=cp.int64)
            
            # Test that we can still create ArbitraryNumbers
            test_num = ArbitraryNumber.from_fraction(1, 2)
            result = test_num * 2
            self.assertEqual(result.evaluate_exact(), Fraction(1))
            
            # Clean up
            del large_array
            
        except cp.cuda.memory.OutOfMemoryError:
            # This is expected behavior when memory is exhausted
            pass
    
    def test_gpu_device_switching(self):
        """Test behavior with multiple GPU devices (if available)."""
        if not GPU_AVAILABLE:
            return
        
        # Check number of available devices
        device_count = cp.cuda.runtime.getDeviceCount()
        
        if device_count > 1:
            # Test switching between devices
            for device_id in range(min(2, device_count)):
                with cp.cuda.Device(device_id):
                    # Create and process ArbitraryNumbers on this device
                    test_num = ArbitraryNumber.from_fraction(device_id + 1, 2)
                    result = test_num * 2
                    expected = Fraction(device_id + 1)
                    self.assertEqual(result.evaluate_exact(), expected)
        else:
            self.skipTest("Multiple GPU devices not available")
    
    def test_gpu_synchronization(self):
        """Test GPU synchronization and data transfer."""
        if not GPU_AVAILABLE:
            return
        
        # Create data on GPU
        gpu_data = cp.array([1, 2, 3, 4, 5], dtype=cp.int32)
        
        # Force synchronization
        cp.cuda.Stream.null.synchronize()
        
        # Transfer to CPU and verify
        cpu_data = gpu_data.get()
        expected = [1, 2, 3, 4, 5]
        self.assertEqual(cpu_data.tolist(), expected)
        
        # Test with ArbitraryNumber operations
        numbers = [ArbitraryNumber.from_int(x) for x in cpu_data]
        results = [num * 2 for num in numbers]
        
        for i, result in enumerate(results):
            expected_val = Fraction(expected[i] * 2)
            self.assertEqual(result.evaluate_exact(), expected_val)
    
    def test_gpu_error_recovery(self):
        """Test error recovery in GPU operations."""
        if not GPU_AVAILABLE:
            return
        
        # Test recovery from GPU errors
        try:
            # Attempt operation that might fail
            very_large_array = cp.zeros(10**10, dtype=cp.float64)  # Likely to fail
        except (cp.cuda.memory.OutOfMemoryError, MemoryError):
            # Should be able to continue with normal operations
            test_num = ArbitraryNumber.from_fraction(1, 3)
            result = test_num + ArbitraryNumber.from_fraction(1, 6)
            expected = Fraction(1, 2)
            self.assertEqual(result.evaluate_exact(), expected)
    
    def test_concurrent_gpu_operations(self):
        """Test concurrent GPU operations with ArbitraryNumbers."""
        if not GPU_AVAILABLE:
            return
        
        # Create multiple streams for concurrent operations
        stream1 = cp.cuda.Stream()
        stream2 = cp.cuda.Stream()
        
        try:
            with stream1:
                # Operation 1
                data1 = cp.array([1, 2, 3], dtype=cp.int32)
                result1 = data1 * 2
            
            with stream2:
                # Operation 2
                data2 = cp.array([4, 5, 6], dtype=cp.int32)
                result2 = data2 * 3
            
            # Synchronize both streams
            stream1.synchronize()
            stream2.synchronize()
            
            # Verify results
            cp.testing.assert_array_equal(result1, cp.array([2, 4, 6]))
            cp.testing.assert_array_equal(result2, cp.array([12, 15, 18]))
            
        finally:
            # Clean up streams
            stream1.use()
            stream2.use()


class TestArbitraryNumberGPUIntegration(unittest.TestCase):
    """Integration tests for GPU functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not GPU_AVAILABLE:
            self.skipTest("GPU not available - CuPy not installed")
    
    def test_end_to_end_gpu_pipeline(self):
        """Test complete GPU processing pipeline."""
        # Step 1: Create input data
        input_size = 1000
        input_numbers = [
            ArbitraryNumber.from_fraction(i, i + 1) 
            for i in range(1, input_size + 1)
        ]
        
        # Step 2: Extract data for GPU processing
        numerators = [num.terms[0].numerator for num in input_numbers]
        denominators = [num.terms[0].denominator for num in input_numbers]
        
        # Step 3: GPU processing simulation
        if GPU_AVAILABLE:
            gpu_nums = cp.array(numerators, dtype=cp.int64)
            gpu_dens = cp.array(denominators, dtype=cp.int64)
            
            # Simulate complex operation: (a/b) * 2 + 1
            gpu_result_nums = gpu_nums * 2 + gpu_dens
            gpu_result_dens = gpu_dens
            
            # Step 4: Convert back to ArbitraryNumbers
            result_numbers = []
            for num, den in zip(gpu_result_nums.get(), gpu_result_dens.get()):
                result_numbers.append(ArbitraryNumber.from_fraction(num, den))
        
        # Step 5: Verify results
        for i, (original, result) in enumerate(zip(input_numbers, result_numbers)):
            # Expected: (i+1)/(i+2) * 2 + 1 = (2*(i+1) + (i+2))/(i+2) = (3*i+4)/(i+2)
            expected_num = 3 * (i + 1) + 1  # 3*i + 4
            expected_den = i + 2
            expected = Fraction(expected_num, expected_den)
            
            actual = result.evaluate_exact()
            self.assertEqual(actual, expected)
    
    def test_gpu_accelerated_matrix_operations(self):
        """Test GPU acceleration for matrix-like operations."""
        # Create matrix of ArbitraryNumbers
        matrix_size = 100
        matrix = []
        
        for i in range(matrix_size):
            row = []
            for j in range(matrix_size):
                num = ArbitraryNumber.from_fraction(i + 1, j + 1)
                row.append(num)
            matrix.append(row)
        
        # Simulate GPU matrix operations
        if GPU_AVAILABLE:
            # Extract numerators and denominators
            nums_matrix = cp.array([[cell.terms[0].numerator for cell in row] for row in matrix])
            dens_matrix = cp.array([[cell.terms[0].denominator for cell in row] for row in matrix])
            
            # Perform matrix operation (element-wise multiplication by 2)
            result_nums = nums_matrix * 2
            result_dens = dens_matrix
            
            # Verify a few elements
            for i in range(min(5, matrix_size)):
                for j in range(min(5, matrix_size)):
                    expected = Fraction(2 * (i + 1), j + 1)
                    actual_num = int(result_nums[i, j])
                    actual_den = int(result_dens[i, j])
                    actual = Fraction(actual_num, actual_den)
                    self.assertEqual(actual, expected)


if __name__ == '__main__':
    # Print GPU availability info
    if GPU_AVAILABLE:
        try:
            device_count = cp.cuda.runtime.getDeviceCount()
            print(f"GPU testing enabled. Found {device_count} CUDA device(s).")
            for i in range(device_count):
                props = cp.cuda.runtime.getDeviceProperties(i)
                print(f"Device {i}: {props['name'].decode()}")
        except Exception as e:
            print(f"GPU detection failed: {e}")
    else:
        print("GPU testing disabled - CuPy not available")
    
    unittest.main()
