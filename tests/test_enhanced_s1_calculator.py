#!/usr/bin/env python3
"""
COMPREHENSIVE TESTS FOR ENHANCED S1 CALCULATOR
==============================================

This module contains comprehensive tests to verify that the Enhanced S1 Calculator
meets all mathematical accuracy requirements specified in the agricultural 
cross-sector analysis specification.

Test Coverage:
- Exact daily returns calculation: Ri,t = (Pi,t - Pi,t-1) / Pi,t-1
- Binary indicator functions: I{|RA,t| ≥ rA}
- Sign function: Sign(Ri,t) = +1 if Ri,t ≥ 0, -1 if Ri,t < 0
- Conditional expectations: ⟨ab⟩xy = Σ[sign(RA,t)sign(RB,t)I{conditions}] / Σ[I{conditions}]
- S1 formula: S1 = ⟨ab⟩00 + ⟨ab⟩01 + ⟨ab⟩10 - ⟨ab⟩11
- Missing data handling: ⟨ab⟩xy = 0 if no valid observations
- Bell violation detection: |S1| > 2

Authors: Agricultural Cross-Sector Analysis Team
Date: September 2025
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import unittest
import pandas as pd
import numpy as np
from src.enhanced_s1_calculator import EnhancedS1Calculator, quick_s1_analysis

class TestEnhancedS1Calculator(unittest.TestCase):
    """Comprehensive test suite for Enhanced S1 Calculator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.calculator = EnhancedS1Calculator(window_size=20, threshold_quantile=0.75)
        
        # Create test data
        np.random.seed(42)
        self.dates = pd.date_range('2023-01-01', periods=100, freq='D')
        self.test_prices = pd.DataFrame({
            'CORN': 100 * (1 + np.random.normal(0, 0.02, 100)).cumprod(),
            'ADM': 100 * (1 + np.random.normal(0, 0.015, 100)).cumprod(),
            'CF': 100 * (1 + np.random.normal(0, 0.025, 100)).cumprod()
        }, index=self.dates)
        
        self.test_returns = self.calculator.calculate_daily_returns(self.test_prices)
    
    def test_daily_returns_calculation(self):
        """Test exact daily returns calculation: Ri,t = (Pi,t - Pi,t-1) / Pi,t-1"""
        print("\n🧪 Testing daily returns calculation...")
        
        # Test basic calculation
        returns = self.calculator.calculate_daily_returns(self.test_prices)
        
        # Verify shape
        self.assertEqual(returns.shape[0], self.test_prices.shape[0] - 1)
        self.assertEqual(returns.shape[1], self.test_prices.shape[1])
        
        # Verify exact formula implementation
        for col in self.test_prices.columns:
            manual_returns = (self.test_prices[col] - self.test_prices[col].shift(1)) / self.test_prices[col].shift(1)
            manual_returns = manual_returns.dropna()
            
            # Compare with calculator results (allowing for floating point precision)
            np.testing.assert_array_almost_equal(
                returns[col].values, 
                manual_returns.values, 
                decimal=10,
                err_msg=f"Daily returns calculation failed for {col}"
            )
        
        # Test edge cases
        with self.assertRaises(ValueError):
            self.calculator.calculate_daily_returns(None)
        
        with self.assertRaises(ValueError):
            self.calculator.calculate_daily_returns(pd.DataFrame())
        
        print("✅ Daily returns calculation tests passed")
    
    def test_binary_indicators(self):
        """Test binary indicator functions: I{|RA,t| ≥ rA}"""
        print("\n🧪 Testing binary indicator functions...")
        
        # Calculate thresholds
        thresholds = self.test_returns.abs().quantile(0.75)
        
        # Compute binary indicators
        indicators_result = self.calculator.compute_binary_indicators(self.test_returns, thresholds)
        indicators = indicators_result['indicators']
        
        # Verify structure
        expected_columns = []
        for asset in self.test_returns.columns:
            expected_columns.extend([f"{asset}_strong", f"{asset}_weak"])
        
        self.assertEqual(set(indicators.columns), set(expected_columns))
        
        # Verify binary nature
        for col in indicators.columns:
            unique_values = set(indicators[col].unique())
            self.assertTrue(unique_values.issubset({True, False}), 
                          f"Binary indicator {col} contains non-boolean values")
        
        # Verify complementary nature (strong + weak = all observations)
        for asset in self.test_returns.columns:
            strong_col = f"{asset}_strong"
            weak_col = f"{asset}_weak"
            
            # Every observation should be either strong or weak, but not both
            combined = indicators[strong_col] | indicators[weak_col]
            self.assertTrue(combined.all(), f"Missing observations for {asset}")
            
            overlap = indicators[strong_col] & indicators[weak_col]
            self.assertFalse(overlap.any(), f"Overlapping regimes for {asset}")
        
        # Verify threshold logic
        for asset in self.test_returns.columns:
            threshold = thresholds[asset]
            abs_returns = self.test_returns[asset].abs()
            
            expected_strong = abs_returns >= threshold
            actual_strong = indicators[f"{asset}_strong"]
            
            pd.testing.assert_series_equal(expected_strong, actual_strong, 
                                         check_names=False)
        
        print("✅ Binary indicator tests passed")
    
    def test_sign_outcomes(self):
        """Test sign function: Sign(Ri,t) = +1 if Ri,t ≥ 0, -1 if Ri,t < 0"""
        print("\n🧪 Testing sign function...")
        
        signs = self.calculator.calculate_sign_outcomes(self.test_returns)
        
        # Verify shape
        self.assertEqual(signs.shape, self.test_returns.shape)
        
        # Verify sign logic
        for col in self.test_returns.columns:
            returns_col = self.test_returns[col]
            signs_col = signs[col]
            
            # Check positive returns
            positive_mask = returns_col >= 0
            self.assertTrue((signs_col[positive_mask] == 1).all(), 
                          f"Positive returns not mapped to +1 for {col}")
            
            # Check negative returns
            negative_mask = returns_col < 0
            self.assertTrue((signs_col[negative_mask] == -1).all(), 
                          f"Negative returns not mapped to -1 for {col}")
            
            # Verify only +1 and -1 values
            unique_signs = set(signs_col.unique())
            self.assertTrue(unique_signs.issubset({-1, 1}), 
                          f"Invalid sign values for {col}: {unique_signs}")
        
        # Test edge case: exactly zero returns
        zero_returns = pd.DataFrame({'TEST': [0.0, 0.0, 0.0]})
        zero_signs = self.calculator.calculate_sign_outcomes(zero_returns)
        self.assertTrue((zero_signs['TEST'] == 1).all(), 
                       "Zero returns should map to +1")
        
        print("✅ Sign function tests passed")
    
    def test_conditional_expectations(self):
        """Test conditional expectations formula"""
        print("\n🧪 Testing conditional expectations...")
        
        # Prepare test data
        thresholds = self.test_returns.abs().quantile(0.75)
        indicators_result = self.calculator.compute_binary_indicators(self.test_returns, thresholds)
        indicators = indicators_result['indicators']
        signs = self.calculator.calculate_sign_outcomes(self.test_returns)
        
        # Test calculation
        expectations = self.calculator.calculate_conditional_expectations(
            signs, indicators, 'CORN', 'ADM'
        )
        
        # Verify structure
        expected_keys = ['ab_00', 'ab_01', 'ab_10', 'ab_11']
        self.assertEqual(set(expectations.keys()), set(expected_keys))
        
        # Verify bounds: conditional expectations should be in [-1, 1]
        for key, value in expectations.items():
            self.assertGreaterEqual(value, -1, f"Expectation {key} below -1: {value}")
            self.assertLessEqual(value, 1, f"Expectation {key} above 1: {value}")
        
        # Test manual calculation for one regime
        sign_corn = signs['CORN']
        sign_adm = signs['ADM']
        mask_00 = indicators['CORN_strong'] & indicators['ADM_strong']
        
        if mask_00.sum() > 0:
            manual_ab_00 = (sign_corn[mask_00] * sign_adm[mask_00]).mean()
            self.assertAlmostEqual(expectations['ab_00'], manual_ab_00, places=10,
                                 msg="Manual calculation doesn't match")
        
        # Test missing data handling
        empty_indicators = pd.DataFrame({
            'CORN_strong': [False] * len(signs),
            'CORN_weak': [True] * len(signs),
            'ADM_strong': [False] * len(signs),
            'ADM_weak': [True] * len(signs)
        }, index=signs.index)
        
        empty_expectations = self.calculator.calculate_conditional_expectations(
            signs, empty_indicators, 'CORN', 'ADM'
        )
        
        # ab_00 should be 0 (no strong movements for either asset)
        self.assertEqual(empty_expectations['ab_00'], 0.0, 
                        "Missing data not handled correctly")
        
        print("✅ Conditional expectations tests passed")
    
    def test_s1_calculation(self):
        """Test S1 formula: S1 = ⟨ab⟩00 + ⟨ab⟩01 + ⟨ab⟩10 - ⟨ab⟩11"""
        print("\n🧪 Testing S1 calculation...")
        
        # Test with known values
        test_expectations = {
            'ab_00': 0.5,
            'ab_01': 0.3,
            'ab_10': 0.2,
            'ab_11': 0.1
        }
        
        s1 = self.calculator.compute_s1_value(test_expectations)
        expected_s1 = 0.5 + 0.3 + 0.2 - 0.1  # = 0.9
        
        self.assertAlmostEqual(s1, expected_s1, places=10,
                              msg="S1 calculation formula incorrect")
        
        # Test with extreme values
        extreme_expectations = {
            'ab_00': 1.0,
            'ab_01': 1.0,
            'ab_10': 1.0,
            'ab_11': -1.0
        }
        
        extreme_s1 = self.calculator.compute_s1_value(extreme_expectations)
        expected_extreme = 1.0 + 1.0 + 1.0 - (-1.0)  # = 4.0
        
        self.assertAlmostEqual(extreme_s1, expected_extreme, places=10,
                              msg="S1 calculation with extreme values failed")
        
        # Test error handling
        with self.assertRaises(ValueError):
            self.calculator.compute_s1_value({'ab_00': 0.5})  # Missing keys
        
        print("✅ S1 calculation tests passed")
    
    def test_violation_detection(self):
        """Test Bell violation detection: |S1| > 2"""
        print("\n🧪 Testing violation detection...")
        
        test_s1_values = [-3.0, -2.5, -2.0, -1.0, 0.0, 1.0, 2.0, 2.5, 3.0]
        violations = self.calculator.detect_violations(test_s1_values)
        
        # Expected violations: -3.0, -2.5, 2.5, 3.0 (4 violations)
        expected_violations = 4
        self.assertEqual(violations['violations'], expected_violations,
                        f"Expected {expected_violations} violations, got {violations['violations']}")
        
        # Test violation rate
        expected_rate = (4 / 9) * 100  # ≈ 44.44%
        self.assertAlmostEqual(violations['violation_rate'], expected_rate, places=2)
        
        # Test max violation
        expected_max = 3.0
        self.assertEqual(violations['max_violation'], expected_max)
        
        # Test bounds
        self.assertEqual(violations['classical_bound'], 2.0)
        self.assertAlmostEqual(violations['quantum_bound'], 2 * np.sqrt(2), places=10)
        
        # Test empty list
        empty_violations = self.calculator.detect_violations([])
        self.assertEqual(empty_violations['violations'], 0)
        self.assertEqual(empty_violations['violation_rate'], 0.0)
        
        print("✅ Violation detection tests passed")
    
    def test_asset_pair_analysis(self):
        """Test complete asset pair analysis workflow"""
        print("\n🧪 Testing complete asset pair analysis...")
        
        results = self.calculator.analyze_asset_pair(self.test_returns, 'CORN', 'ADM')
        
        # Verify structure
        required_keys = ['asset_pair', 's1_time_series', 'expectations_time_series', 
                        'timestamps', 'violation_results', 'analysis_parameters']
        self.assertEqual(set(results.keys()), set(required_keys))
        
        # Verify asset pair
        self.assertEqual(results['asset_pair'], ('CORN', 'ADM'))
        
        # Verify time series length
        expected_length = len(self.test_returns) - self.calculator.window_size
        self.assertEqual(len(results['s1_time_series']), expected_length)
        self.assertEqual(len(results['expectations_time_series']), expected_length)
        self.assertEqual(len(results['timestamps']), expected_length)
        
        # Verify S1 values are numbers
        for s1 in results['s1_time_series']:
            self.assertIsInstance(s1, (int, float))
        
        # Verify expectations structure
        for expectations in results['expectations_time_series']:
            self.assertEqual(set(expectations.keys()), {'ab_00', 'ab_01', 'ab_10', 'ab_11'})
        
        # Verify violation results
        violation_results = results['violation_results']
        self.assertIn('violations', violation_results)
        self.assertIn('violation_rate', violation_results)
        
        print("✅ Asset pair analysis tests passed")
    
    def test_batch_analysis(self):
        """Test batch analysis of multiple pairs"""
        print("\n🧪 Testing batch analysis...")
        
        asset_pairs = [('CORN', 'ADM'), ('CORN', 'CF'), ('ADM', 'CF')]
        batch_results = self.calculator.batch_analyze_pairs(self.test_returns, asset_pairs)
        
        # Verify structure
        self.assertIn('pair_results', batch_results)
        self.assertIn('summary', batch_results)
        
        # Verify all pairs analyzed
        pair_results = batch_results['pair_results']
        self.assertEqual(len(pair_results), len(asset_pairs))
        
        for pair in asset_pairs:
            self.assertIn(pair, pair_results)
        
        # Verify summary statistics
        summary = batch_results['summary']
        required_summary_keys = ['total_pairs', 'successful_pairs', 'total_calculations', 
                               'total_violations', 'overall_violation_rate']
        self.assertEqual(set(summary.keys()), set(required_summary_keys))
        
        self.assertEqual(summary['total_pairs'], len(asset_pairs))
        self.assertEqual(summary['successful_pairs'], len(asset_pairs))
        
        print("✅ Batch analysis tests passed")
    
    def test_crisis_period_parameters(self):
        """Test crisis period parameter settings"""
        print("\n🧪 Testing crisis period parameters...")
        
        # Test crisis period calculator (window_size=15, threshold_quantile=0.8)
        crisis_calculator = EnhancedS1Calculator(window_size=15, threshold_quantile=0.8)
        
        self.assertEqual(crisis_calculator.window_size, 15)
        self.assertEqual(crisis_calculator.threshold_quantile, 0.8)
        
        # Test that analysis works with crisis parameters
        if len(self.test_returns) >= 15:
            crisis_results = crisis_calculator.analyze_asset_pair(
                self.test_returns, 'CORN', 'ADM'
            )
            
            # Verify crisis parameters were used
            params = crisis_results['analysis_parameters']
            self.assertEqual(params['window_size'], 15)
            self.assertEqual(params['threshold_quantile'], 0.8)
        
        print("✅ Crisis period parameter tests passed")
    
    def test_mathematical_accuracy(self):
        """Test mathematical accuracy against known cases"""
        print("\n🧪 Testing mathematical accuracy...")
        
        # Create deterministic test case
        np.random.seed(123)
        
        # Create correlated returns for testing
        n_obs = 50
        base_returns = np.random.normal(0, 0.02, n_obs)
        
        test_data = pd.DataFrame({
            'A': base_returns + np.random.normal(0, 0.01, n_obs),
            'B': base_returns + np.random.normal(0, 0.01, n_obs)  # Correlated with A
        })
        
        # Analyze with small window for precise testing
        small_calculator = EnhancedS1Calculator(window_size=10, threshold_quantile=0.5)
        results = small_calculator.analyze_asset_pair(test_data, 'A', 'B')
        
        # Verify that S1 values are within reasonable bounds
        s1_values = results['s1_time_series']
        self.assertTrue(all(isinstance(s1, (int, float)) for s1 in s1_values))
        self.assertTrue(all(-10 <= s1 <= 10 for s1 in s1_values))  # Reasonable bounds
        
        # Verify expectations are within [-1, 1]
        for expectations in results['expectations_time_series']:
            for exp_value in expectations.values():
                self.assertGreaterEqual(exp_value, -1)
                self.assertLessEqual(exp_value, 1)
        
        print("✅ Mathematical accuracy tests passed")

def run_comprehensive_tests():
    """Run all comprehensive tests for Enhanced S1 Calculator."""
    print("🧪 RUNNING COMPREHENSIVE ENHANCED S1 CALCULATOR TESTS")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestEnhancedS1Calculator)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("✅ ALL TESTS PASSED - Enhanced S1 Calculator is mathematically accurate!")
        print(f"   Tests run: {result.testsRun}")
        print(f"   Failures: {len(result.failures)}")
        print(f"   Errors: {len(result.errors)}")
        return True
    else:
        print("❌ SOME TESTS FAILED")
        print(f"   Tests run: {result.testsRun}")
        print(f"   Failures: {len(result.failures)}")
        print(f"   Errors: {len(result.errors)}")
        return False

if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)