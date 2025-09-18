#!/usr/bin/env python3
"""
TEST SUITE FOR MATHEMATICAL VALIDATION FRAMEWORK
===============================================

This module provides comprehensive tests for the mathematical validation and 
cross-implementation analysis framework, ensuring all validation components
meet Science journal publication standards.

Test Coverage:
- Cross-implementation validator functionality
- Numerical precision analyzer accuracy
- Statistical significance testing
- Error handling and edge cases
- Performance benchmarking

Authors: Bell Inequality Validation Team
Date: September 2025
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import tempfile
import os
import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from mathematical_validation_framework import (
    CrossImplementationValidator,
    NumericalPrecisionAnalyzer,
    ValidationResult,
    PrecisionReport,
    ComparisonReport,
    run_comprehensive_validation
)

class TestCrossImplementationValidator(unittest.TestCase):
    """Test cases for CrossImplementationValidator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = CrossImplementationValidator(tolerance=1e-12)
        
        # Create synthetic test data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        self.test_prices = pd.DataFrame({
            'ASSET_A': 100 * (1 + np.random.normal(0, 0.02, 50)).cumprod(),
            'ASSET_B': 100 * (1 + np.random.normal(0, 0.02, 50)).cumprod(),
            'ASSET_C': 100 * (1 + np.random.normal(0, 0.02, 50)).cumprod()
        }, index=dates)
        
        self.test_returns = self.test_prices.pct_change().dropna()
        self.test_pairs = [('ASSET_A', 'ASSET_B'), ('ASSET_A', 'ASSET_C')]
    
    def test_validator_initialization(self):
        """Test validator initialization."""
        self.assertEqual(self.validator.tolerance, 1e-12)
        self.assertEqual(self.validator.bootstrap_samples, 1000)
        self.assertIsInstance(self.validator.validation_results, list)
        self.assertIsInstance(self.validator.precision_reports, list)
        self.assertIsInstance(self.validator.comparison_reports, list)
    
    def test_validate_daily_returns_calculation(self):
        """Test daily returns calculation validation."""
        result = self.validator.validate_daily_returns_calculation(self.test_prices)
        
        self.assertIsInstance(result, ValidationResult)
        self.assertEqual(result.test_name, "Daily Returns Calculation")
        self.assertIsInstance(result.passed, bool)
        self.assertIsInstance(result.numerical_difference, float)
        self.assertGreaterEqual(result.numerical_difference, 0)
        self.assertIsInstance(result.execution_time, float)
        self.assertGreater(result.execution_time, 0)
    
    def test_validate_sign_calculations(self):
        """Test sign function calculation validation."""
        result = self.validator.validate_sign_calculations(self.test_returns)
        
        self.assertIsInstance(result, ValidationResult)
        self.assertEqual(result.test_name, "Sign Function Calculation")
        self.assertIsInstance(result.passed, bool)
        self.assertEqual(result.tolerance, 0.0)  # Sign function must be exact
        
        # Sign function should pass (identical implementations)
        self.assertTrue(result.passed)
        self.assertEqual(result.numerical_difference, 0.0)
    
    def test_validate_threshold_methods(self):
        """Test threshold calculation method validation."""
        quantiles = [0.5, 0.75, 0.9]
        result = self.validator.validate_threshold_methods(self.test_returns, quantiles)
        
        self.assertIsInstance(result, ValidationResult)
        self.assertEqual(result.test_name, "Threshold Calculation Methods")
        self.assertIsInstance(result.passed, bool)
        self.assertGreaterEqual(result.numerical_difference, 0)
    
    def test_validate_bell_violations(self):
        """Test Bell violation detection validation."""
        result = self.validator.validate_bell_violations(self.test_prices, self.test_pairs)
        
        self.assertIsInstance(result, ValidationResult)
        self.assertEqual(result.test_name, "Bell Violation Detection")
        self.assertIsInstance(result.passed, bool)
        self.assertEqual(result.tolerance, 10.0)  # 10% tolerance for violation rates
    
    def test_cross_validate_methods(self):
        """Test cross-validation between methods."""
        report = self.validator.cross_validate_methods(self.test_prices, self.test_pairs)
        
        self.assertIsInstance(report, ComparisonReport)
        self.assertEqual(report.implementation_a, "S1 Conditional")
        self.assertEqual(report.implementation_b, "CHSH Sliding Window")
        self.assertIsInstance(report.identical_results, bool)
        self.assertGreaterEqual(report.max_difference, 0)
        self.assertGreaterEqual(report.correlation, -1)
        self.assertLessEqual(report.correlation, 1)
    
    def test_generate_validation_report(self):
        """Test validation report generation."""
        # Run some validations first
        self.validator.validate_daily_returns_calculation(self.test_prices)
        self.validator.validate_sign_calculations(self.test_returns)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = os.path.join(temp_dir, "test_report.md")
            generated_path = self.validator.generate_validation_report(report_path)
            
            self.assertEqual(generated_path, report_path)
            self.assertTrue(os.path.exists(report_path))
            
            # Check report content
            with open(report_path, 'r') as f:
                content = f.read()
                self.assertIn("Mathematical Validation", content)
                self.assertIn("Executive Summary", content)
                self.assertIn("Validation Results Summary", content)
    
    def test_empty_data_handling(self):
        """Test handling of empty or invalid data."""
        empty_df = pd.DataFrame()
        
        # Should return a failed ValidationResult, not raise an exception
        result = self.validator.validate_daily_returns_calculation(empty_df)
        self.assertIsInstance(result, ValidationResult)
        self.assertFalse(result.passed)
        self.assertIn("Price data cannot be None or empty", result.notes)
    
    def test_single_asset_handling(self):
        """Test handling of single asset data."""
        single_asset = self.test_prices[['ASSET_A']]
        
        # Should handle gracefully but may not produce meaningful results
        result = self.validator.validate_daily_returns_calculation(single_asset)
        self.assertIsInstance(result, ValidationResult)

class TestNumericalPrecisionAnalyzer(unittest.TestCase):
    """Test cases for NumericalPrecisionAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = NumericalPrecisionAnalyzer(precision_target=1e-12)
        
        # Create test data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        self.test_data = pd.DataFrame({
            'ASSET_A': 100 * (1 + np.random.normal(0, 0.01, 30)).cumprod(),
            'ASSET_B': 100 * (1 + np.random.normal(0, 0.01, 30)).cumprod()
        }, index=dates)
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        self.assertEqual(self.analyzer.precision_target, 1e-12)
        self.assertIsInstance(self.analyzer.precision_reports, list)
    
    def test_analyze_floating_point_precision(self):
        """Test floating-point precision analysis."""
        test_calculations = [1.0, 1.000000000001, 0.999999999999, 1.0]
        report = self.analyzer.analyze_floating_point_precision(test_calculations)
        
        self.assertIsInstance(report, PrecisionReport)
        self.assertEqual(report.test_type, "Floating Point Precision")
        self.assertGreater(report.precision_achieved, 0)
        self.assertGreaterEqual(report.stability_score, 0)
        self.assertLessEqual(report.stability_score, 1)
        self.assertIsInstance(report.recommendations, list)
        self.assertGreater(len(report.recommendations), 0)
    
    def test_numerical_stability_testing(self):
        """Test numerical stability analysis."""
        perturbations = [1e-10, 1e-8, 1e-6]
        report = self.analyzer.test_numerical_stability(self.test_data, perturbations)
        
        self.assertIsInstance(report, PrecisionReport)
        self.assertEqual(report.test_type, "Numerical Stability")
        self.assertGreaterEqual(report.stability_score, 0)
        self.assertLessEqual(report.stability_score, 1)
        self.assertEqual(len(report.numerical_errors), len(perturbations))
    
    def test_convergence_validation(self):
        """Test convergence validation."""
        # Create test sequences with known convergence properties
        convergent_sequence = [1.0, 0.5, 0.25, 0.125, 0.0625]  # Geometric convergence
        divergent_sequence = [1.0, 2.0, 4.0, 8.0, 16.0]        # Divergent
        
        test_sequences = [convergent_sequence, divergent_sequence]
        report = self.analyzer.validate_convergence(test_sequences)
        
        self.assertIsInstance(report, PrecisionReport)
        self.assertEqual(report.test_type, "Convergence Validation")
        self.assertGreaterEqual(report.convergence_rate, 0)
        self.assertIsInstance(report.recommendations, list)
    
    def test_empty_calculations_handling(self):
        """Test handling of empty calculation lists."""
        empty_calculations = []
        report = self.analyzer.analyze_floating_point_precision(empty_calculations)
        
        self.assertIsInstance(report, PrecisionReport)
        self.assertEqual(report.stability_score, 1.0)  # Default for empty data
    
    def test_invalid_data_handling(self):
        """Test handling of invalid data (NaN, inf)."""
        invalid_calculations = [1.0, np.nan, np.inf, -np.inf, 2.0]
        report = self.analyzer.analyze_floating_point_precision(invalid_calculations)
        
        self.assertIsInstance(report, PrecisionReport)
        # Should handle invalid values gracefully
        self.assertIn("numerical overflow/underflow", " ".join(report.recommendations).lower())

class TestComprehensiveValidation(unittest.TestCase):
    """Test cases for comprehensive validation function."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=60, freq='D')
        self.test_data = pd.DataFrame({
            'ASSET_A': 100 * (1 + np.random.normal(0, 0.02, 60)).cumprod(),
            'ASSET_B': 100 * (1 + np.random.normal(0, 0.02, 60)).cumprod(),
            'ASSET_C': 100 * (1 + np.random.normal(0, 0.02, 60)).cumprod(),
            'ASSET_D': 100 * (1 + np.random.normal(0, 0.02, 60)).cumprod()
        }, index=dates)
        
        self.test_pairs = [('ASSET_A', 'ASSET_B'), ('ASSET_C', 'ASSET_D')]
    
    def test_comprehensive_validation_execution(self):
        """Test comprehensive validation execution."""
        with tempfile.TemporaryDirectory() as temp_dir:
            results = run_comprehensive_validation(
                self.test_data, 
                self.test_pairs, 
                output_dir=temp_dir
            )
            
            # Check result structure
            self.assertIn('validation_tests', results)
            self.assertIn('cross_validation', results)
            self.assertIn('precision_analysis', results)
            self.assertIn('reports', results)
            self.assertIn('summary', results)
            
            # Check validation tests
            validation_tests = results['validation_tests']
            self.assertIn('daily_returns', validation_tests)
            self.assertIn('sign_function', validation_tests)
            self.assertIn('threshold_methods', validation_tests)
            self.assertIn('bell_violations', validation_tests)
            
            # Check cross-validation
            cross_validation = results['cross_validation']
            self.assertIsInstance(cross_validation, ComparisonReport)
            
            # Check precision analysis
            precision_analysis = results['precision_analysis']
            self.assertIn('floating_point', precision_analysis)
            self.assertIn('stability', precision_analysis)
            self.assertIn('convergence', precision_analysis)
            
            # Check summary
            summary = results['summary']
            self.assertIn('all_tests_passed', summary)
            self.assertIn('precision_target_met', summary)
            self.assertIn('stability_acceptable', summary)
            self.assertIn('convergence_acceptable', summary)
            
            # Check that report files were created
            report_path = results['reports']['validation_report']
            self.assertTrue(os.path.exists(report_path))
    
    def test_comprehensive_validation_with_default_pairs(self):
        """Test comprehensive validation with default asset pairs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            results = run_comprehensive_validation(
                self.test_data, 
                asset_pairs=None,  # Use default pairs
                output_dir=temp_dir
            )
            
            self.assertIn('validation_tests', results)
            self.assertIn('summary', results)
    
    def test_comprehensive_validation_error_handling(self):
        """Test error handling in comprehensive validation."""
        # Test with insufficient data
        small_data = self.test_data.iloc[:5]  # Very small dataset
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Should handle gracefully without crashing
            results = run_comprehensive_validation(
                small_data, 
                self.test_pairs, 
                output_dir=temp_dir
            )
            
            self.assertIn('validation_tests', results)
            # Some tests may fail due to insufficient data, but should not crash

class TestValidationDataStructures(unittest.TestCase):
    """Test cases for validation data structures."""
    
    def test_validation_result_creation(self):
        """Test ValidationResult dataclass creation."""
        result = ValidationResult(
            test_name="Test",
            passed=True,
            numerical_difference=1e-15,
            tolerance=1e-12,
            confidence_interval=(0.0, 1e-15),
            p_value=0.05,
            effect_size=0.1,
            notes="Test note",
            execution_time=0.1
        )
        
        self.assertEqual(result.test_name, "Test")
        self.assertTrue(result.passed)
        self.assertEqual(result.numerical_difference, 1e-15)
        self.assertEqual(result.tolerance, 1e-12)
    
    def test_precision_report_creation(self):
        """Test PrecisionReport dataclass creation."""
        report = PrecisionReport(
            test_type="Test Type",
            precision_achieved=1e-16,
            stability_score=0.95,
            convergence_rate=0.8,
            numerical_errors=[1e-15, 1e-14],
            recommendations=["Good precision"]
        )
        
        self.assertEqual(report.test_type, "Test Type")
        self.assertEqual(report.precision_achieved, 1e-16)
        self.assertEqual(report.stability_score, 0.95)
        self.assertEqual(len(report.numerical_errors), 2)
    
    def test_comparison_report_creation(self):
        """Test ComparisonReport dataclass creation."""
        report = ComparisonReport(
            implementation_a="Method A",
            implementation_b="Method B",
            identical_results=True,
            max_difference=1e-15,
            mean_difference=1e-16,
            correlation=0.99,
            statistical_significance={'p_value': 0.01},
            performance_comparison={'time': 1.0}
        )
        
        self.assertEqual(report.implementation_a, "Method A")
        self.assertEqual(report.implementation_b, "Method B")
        self.assertTrue(report.identical_results)
        self.assertEqual(report.correlation, 0.99)

class TestPerformanceBenchmarking(unittest.TestCase):
    """Test cases for performance aspects of validation framework."""
    
    def setUp(self):
        """Set up performance test fixtures."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        self.large_dataset = pd.DataFrame({
            f'ASSET_{i}': 100 * (1 + np.random.normal(0, 0.02, 100)).cumprod()
            for i in range(10)  # 10 assets
        }, index=dates)
    
    def test_validation_performance(self):
        """Test validation framework performance."""
        validator = CrossImplementationValidator()
        
        start_time = datetime.now()
        result = validator.validate_daily_returns_calculation(self.large_dataset)
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Should complete within reasonable time (< 10 seconds for this dataset)
        self.assertLess(execution_time, 10.0)
        self.assertGreater(result.execution_time, 0)
    
    def test_precision_analysis_performance(self):
        """Test precision analysis performance."""
        analyzer = NumericalPrecisionAnalyzer()
        
        # Large calculation set
        large_calculations = list(np.random.normal(1.0, 0.01, 1000))
        
        start_time = datetime.now()
        report = analyzer.analyze_floating_point_precision(large_calculations)
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Should complete within reasonable time
        self.assertLess(execution_time, 5.0)
        self.assertIsInstance(report, PrecisionReport)

def run_validation_tests():
    """Run all validation framework tests."""
    print("🧪 Running Mathematical Validation Framework Tests")
    print("=" * 55)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestCrossImplementationValidator,
        TestNumericalPrecisionAnalyzer,
        TestComprehensiveValidation,
        TestValidationDataStructures,
        TestPerformanceBenchmarking
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n📊 TEST SUMMARY:")
    print(f"   Tests run: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")
    print(f"   Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"   {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print(f"\n💥 ERRORS:")
        for test, traceback in result.errors:
            print(f"   {test}: {traceback.split('Error:')[-1].strip()}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\n{'✅ ALL TESTS PASSED!' if success else '❌ SOME TESTS FAILED!'}")
    
    return success

if __name__ == "__main__":
    # Run all tests
    success = run_validation_tests()
    
    if success:
        print("\n🎯 Mathematical Validation Framework is ready for production use!")
    else:
        print("\n⚠️  Please fix test failures before proceeding.")
        sys.exit(1)