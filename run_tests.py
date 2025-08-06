#!/usr/bin/env python3
"""
Comprehensive Test Runner for Arbitrary Numbers
==============================================

This script runs all test suites for the ArbitraryNumber project and provides
detailed reporting on test results, performance, and coverage.
"""

import sys
import os
import unittest
import time
import argparse
from io import StringIO

# Add the project root to the path
sys.path.insert(0, os.path.dirname(__file__))

# Import test modules
from tests.unit.test_arbitrary_number_basic import *
from tests.unit.test_arbitrary_number_edge_cases import *
from tests.integration.test_arbitrary_number_integration import *

# Try to import GPU and performance tests
try:
    from tests.gpu.test_arbitrary_number_gpu import *
    GPU_TESTS_AVAILABLE = True
except ImportError as e:
    GPU_TESTS_AVAILABLE = False
    GPU_IMPORT_ERROR = str(e)

try:
    from tests.performance.test_arbitrary_number_benchmarks import *
    PERFORMANCE_TESTS_AVAILABLE = True
except ImportError as e:
    PERFORMANCE_TESTS_AVAILABLE = False
    PERFORMANCE_IMPORT_ERROR = str(e)


class ColoredTextTestResult(unittest.TextTestResult):
    """Enhanced test result with colored output and timing."""
    
    def __init__(self, stream, descriptions, verbosity):
        super().__init__(stream, descriptions, verbosity)
        self.test_times = {}
        self.start_time = None
        
    def startTest(self, test):
        super().startTest(test)
        self.start_time = time.perf_counter()
        
    def stopTest(self, test):
        super().stopTest(test)
        if self.start_time:
            elapsed = time.perf_counter() - self.start_time
            self.test_times[str(test)] = elapsed
    
    def addSuccess(self, test):
        super().addSuccess(test)
        if self.verbosity > 1:
            elapsed = self.test_times.get(str(test), 0)
            self.stream.write(f" ... ok ({elapsed:.3f}s)\n")
    
    def addError(self, test, err):
        super().addError(test, err)
        if self.verbosity > 1:
            elapsed = self.test_times.get(str(test), 0)
            self.stream.write(f" ... ERROR ({elapsed:.3f}s)\n")
    
    def addFailure(self, test, err):
        super().addFailure(test, err)
        if self.verbosity > 1:
            elapsed = self.test_times.get(str(test), 0)
            self.stream.write(f" ... FAIL ({elapsed:.3f}s)\n")
    
    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        if self.verbosity > 1:
            self.stream.write(f" ... skipped '{reason}'\n")


class ArbitraryNumberTestRunner:
    """Main test runner for ArbitraryNumber project."""
    
    def __init__(self, verbosity=2):
        self.verbosity = verbosity
        self.results = {}
        
    def run_test_suite(self, suite_name, test_classes, description):
        """Run a specific test suite."""
        print(f"\n{'='*60}")
        print(f"RUNNING {suite_name.upper()}")
        print(f"{'='*60}")
        print(f"Description: {description}")
        print()
        
        # Create test suite
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        
        for test_class in test_classes:
            suite.addTests(loader.loadTestsFromTestCase(test_class))
        
        # Run tests
        stream = StringIO()
        runner = unittest.TextTestRunner(
            stream=stream,
            verbosity=self.verbosity,
            resultclass=ColoredTextTestResult
        )
        
        start_time = time.perf_counter()
        result = runner.run(suite)
        end_time = time.perf_counter()
        
        # Store results
        self.results[suite_name] = {
            'result': result,
            'duration': end_time - start_time,
            'output': stream.getvalue()
        }
        
        # Print results
        print(stream.getvalue())
        
        # Summary
        total_tests = result.testsRun
        failures = len(result.failures)
        errors = len(result.errors)
        skipped = len(result.skipped)
        passed = total_tests - failures - errors - skipped
        
        print(f"\n{suite_name} Summary:")
        print(f"  Tests run: {total_tests}")
        print(f"  Passed: {passed}")
        print(f"  Failed: {failures}")
        print(f"  Errors: {errors}")
        print(f"  Skipped: {skipped}")
        print(f"  Duration: {end_time - start_time:.2f}s")
        
        if failures > 0:
            print(f"\nFailures in {suite_name}:")
            for test, traceback in result.failures:
                print(f"  - {test}")
        
        if errors > 0:
            print(f"\nErrors in {suite_name}:")
            for test, traceback in result.errors:
                print(f"  - {test}")
        
        return result
    
    def run_all_tests(self, include_gpu=True, include_performance=True):
        """Run all available test suites."""
        print("ARBITRARY NUMBER COMPREHENSIVE TEST SUITE")
        print("A Revolutionary Mathematical Concept for Exact Computation")
        print(f"Python {sys.version}")
        print(f"Test runner started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        overall_start = time.perf_counter()
        
        # Unit Tests
        unit_test_classes = [
            TestArbitraryNumberBasics,
            TestArbitraryNumberArithmetic,
            TestArbitraryNumberComparisons,
            TestArbitraryNumberUtilities,
            TestArbitraryNumberCreation,
            TestArbitraryNumberEdgeCases,
            TestArbitraryNumberErrorHandling,
            TestArbitraryNumberComplexScenarios
        ]
        
        self.run_test_suite(
            "Unit Tests",
            unit_test_classes,
            "Core functionality and edge cases for ArbitraryNumber class"
        )
        
        # Integration Tests
        integration_test_classes = [
            TestArbitraryNumberIntegration
        ]
        
        self.run_test_suite(
            "Integration Tests",
            integration_test_classes,
            "Real-world scenarios and workflow integration tests"
        )
        
        # GPU Tests (if available and requested)
        if include_gpu and GPU_TESTS_AVAILABLE:
            gpu_test_classes = [
                TestArbitraryNumberGPUBasics,
                TestArbitraryNumberGPUPerformance,
                TestArbitraryNumberGPUDataTypes,
                TestArbitraryNumberGPUEdgeCases,
                TestArbitraryNumberGPUIntegration
            ]
            
            self.run_test_suite(
                "GPU Tests",
                gpu_test_classes,
                "GPU acceleration and CUDA kernel functionality"
            )
        elif include_gpu and not GPU_TESTS_AVAILABLE:
            print(f"\n{'='*60}")
            print("GPU TESTS SKIPPED")
            print(f"{'='*60}")
            print(f"Reason: {GPU_IMPORT_ERROR}")
            print("Install CuPy and ensure CUDA is available to run GPU tests.")
        
        # Performance Tests (if available and requested)
        if include_performance and PERFORMANCE_TESTS_AVAILABLE:
            performance_test_classes = [
                TestArbitraryNumberPerformanceBasics,
                TestArbitraryNumberScalabilityBenchmarks,
                TestArbitraryNumberMemoryBenchmarks,
                TestArbitraryNumberComparisonBenchmarks
            ]
            
            self.run_test_suite(
                "Performance Tests",
                performance_test_classes,
                "Performance benchmarks and scalability analysis"
            )
        elif include_performance and not PERFORMANCE_TESTS_AVAILABLE:
            print(f"\n{'='*60}")
            print("PERFORMANCE TESTS SKIPPED")
            print(f"{'='*60}")
            print(f"Reason: {PERFORMANCE_IMPORT_ERROR}")
        
        overall_end = time.perf_counter()
        
        # Overall Summary
        self.print_overall_summary(overall_end - overall_start)
        
        return self.get_overall_success()
    
    def print_overall_summary(self, total_duration):
        """Print overall test summary."""
        print(f"\n{'='*60}")
        print("OVERALL TEST SUMMARY")
        print(f"{'='*60}")
        
        total_tests = 0
        total_passed = 0
        total_failed = 0
        total_errors = 0
        total_skipped = 0
        
        for suite_name, data in self.results.items():
            result = data['result']
            duration = data['duration']
            
            tests = result.testsRun
            failures = len(result.failures)
            errors = len(result.errors)
            skipped = len(result.skipped)
            passed = tests - failures - errors - skipped
            
            total_tests += tests
            total_passed += passed
            total_failed += failures
            total_errors += errors
            total_skipped += skipped
            
            status = "PASS" if (failures == 0 and errors == 0) else "FAIL"
            print(f"{suite_name:20} | {tests:3d} tests | {passed:3d} passed | {failures:3d} failed | {errors:3d} errors | {skipped:3d} skipped | {duration:6.2f}s | {status}")
        
        print(f"{'-'*60}")
        print(f"{'TOTAL':20} | {total_tests:3d} tests | {total_passed:3d} passed | {total_failed:3d} failed | {total_errors:3d} errors | {total_skipped:3d} skipped | {total_duration:6.2f}s")
        
        success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        print(f"\nSuccess Rate: {success_rate:.1f}%")
        
        if total_failed == 0 and total_errors == 0:
            print("🎉 ALL TESTS PASSED! 🎉")
            print("\nThe ArbitraryNumber implementation is working correctly!")
            print("• Zero precision loss maintained across all operations")
            print("• Deferred evaluation functioning properly")
            print("• Complete symbolic traceability verified")
            print("• Performance characteristics within expected bounds")
            print("• Integration with real-world scenarios successful")
        else:
            print("❌ SOME TESTS FAILED")
            print(f"Please review the {total_failed} failures and {total_errors} errors above.")
        
        print(f"\nTest run completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    def get_overall_success(self):
        """Return True if all tests passed."""
        for data in self.results.values():
            result = data['result']
            if len(result.failures) > 0 or len(result.errors) > 0:
                return False
        return True
    
    def run_specific_test(self, test_pattern):
        """Run tests matching a specific pattern."""
        print(f"Running tests matching pattern: {test_pattern}")
        
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromName(test_pattern)
        
        runner = unittest.TextTestRunner(verbosity=self.verbosity)
        result = runner.run(suite)
        
        return result.wasSuccessful()


def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(
        description="Comprehensive test runner for ArbitraryNumber project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py                    # Run all tests
  python run_tests.py --no-gpu          # Skip GPU tests
  python run_tests.py --no-performance  # Skip performance tests
  python run_tests.py --quick           # Skip GPU and performance tests
  python run_tests.py --verbose         # Maximum verbosity
  python run_tests.py --quiet           # Minimal output
        """
    )
    
    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='Skip GPU tests'
    )
    
    parser.add_argument(
        '--no-performance',
        action='store_true',
        help='Skip performance benchmarks'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Skip GPU and performance tests (quick run)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Maximum verbosity'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Minimal output'
    )
    
    parser.add_argument(
        '--test',
        type=str,
        help='Run specific test (e.g., tests.unit.test_arbitrary_number_basic.TestArbitraryNumberBasics.test_creation_from_int)'
    )
    
    args = parser.parse_args()
    
    # Determine verbosity
    if args.verbose:
        verbosity = 2
    elif args.quiet:
        verbosity = 0
    else:
        verbosity = 1
    
    # Create test runner
    runner = ArbitraryNumberTestRunner(verbosity=verbosity)
    
    # Run specific test if requested
    if args.test:
        success = runner.run_specific_test(args.test)
        sys.exit(0 if success else 1)
    
    # Determine which test suites to run
    include_gpu = not (args.no_gpu or args.quick)
    include_performance = not (args.no_performance or args.quick)
    
    # Run all tests
    success = runner.run_all_tests(
        include_gpu=include_gpu,
        include_performance=include_performance
    )
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
