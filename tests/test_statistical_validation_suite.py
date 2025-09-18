#!/usr/bin/env python3
"""
TESTS FOR ADVANCED STATISTICAL VALIDATION SUITE
===============================================

Comprehensive tests for the statistical validation and innovative metrics
components of the agricultural cross-sector analysis system.

Authors: Agricultural Cross-Sector Analysis Team
Date: September 2025
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from statistical_validation_suite import (
    AdvancedStatisticalValidator,
    InnovativeMetricsCalculator,
    ComprehensiveStatisticalSuite,
    StatisticalResults,
    InnovativeMetrics
)

class TestAdvancedStatisticalValidator:
    """Test suite for AdvancedStatisticalValidator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = AdvancedStatisticalValidator(n_bootstrap=100, random_state=42)
        
        # Create test S1 values with known violations
        self.test_s1_values = np.array([-3.0, -1.0, 0.5, 1.5, 2.5, 3.2, -2.8, 1.8])
        self.no_violation_s1 = np.array([-1.5, -0.8, 0.3, 1.2, -1.9, 0.7])
    
    def test_initialization(self):
        """Test validator initialization."""
        assert self.validator.n_bootstrap == 100
        assert self.validator.alpha == 0.001
        assert self.validator.random_state == 42
        assert self.validator.classical_bound == 2.0
        assert abs(self.validator.quantum_bound - 2.828) < 0.01
    
    def test_bootstrap_violation_rate_with_violations(self):
        """Test bootstrap validation with S1 values containing violations."""
        results = self.validator.bootstrap_violation_rate(self.test_s1_values)
        
        # Check result structure
        assert isinstance(results, StatisticalResults)
        assert results.violation_rate > 0
        assert len(results.bootstrap_samples) == 100
        assert len(results.confidence_interval) == 2
        assert results.confidence_interval[0] <= results.confidence_interval[1]
        
        # Check violation detection
        expected_violations = np.sum(np.abs(self.test_s1_values) > 2.0)
        expected_rate = (expected_violations / len(self.test_s1_values)) * 100
        assert abs(results.violation_rate - expected_rate) < 0.1
        
        # Check statistical properties
        assert results.p_value >= 0
        assert results.p_value <= 1
        assert results.classical_bound_excess >= 0
    
    def test_bootstrap_violation_rate_no_violations(self):
        """Test bootstrap validation with S1 values containing no violations."""
        results = self.validator.bootstrap_violation_rate(self.no_violation_s1)
        
        assert results.violation_rate == 0.0
        assert results.classical_bound_excess >= 0
        assert not results.is_significant
    
    def test_bootstrap_violation_rate_empty_input(self):
        """Test bootstrap validation with empty input."""
        results = self.validator.bootstrap_violation_rate(np.array([]))
        
        assert results.violation_rate == 0.0
        assert results.confidence_interval == (0.0, 0.0)
        assert results.p_value == 1.0
        assert results.effect_size == 0.0
        assert len(results.bootstrap_samples) == 0
        assert not results.is_significant
    
    def test_multiple_testing_correction_bonferroni(self):
        """Test Bonferroni multiple testing correction."""
        p_values = [0.001, 0.01, 0.05, 0.0001]
        corrected_p, significance = self.validator.multiple_testing_correction(
            p_values, method='bonferroni'
        )
        
        assert len(corrected_p) == len(p_values)
        assert len(significance) == len(p_values)
        
        # Check Bonferroni correction: p_corrected = p * n_tests
        n_tests = len(p_values)
        for i, (orig_p, corr_p) in enumerate(zip(p_values, corrected_p)):
            expected_corr = min(orig_p * n_tests, 1.0)
            assert abs(corr_p - expected_corr) < 1e-10
        
        # Check significance flags
        for i, (corr_p, is_sig) in enumerate(zip(corrected_p, significance)):
            assert is_sig == (corr_p < self.validator.alpha)
    
    def test_multiple_testing_correction_holm(self):
        """Test Holm-Bonferroni multiple testing correction."""
        p_values = [0.001, 0.01, 0.05, 0.0001]
        corrected_p, significance = self.validator.multiple_testing_correction(
            p_values, method='holm'
        )
        
        assert len(corrected_p) == len(p_values)
        assert len(significance) == len(p_values)
        
        # Holm correction should be less conservative than Bonferroni
        bonferroni_p, _ = self.validator.multiple_testing_correction(
            p_values, method='bonferroni'
        )
        
        # At least some corrected p-values should be smaller with Holm
        assert any(h <= b for h, b in zip(corrected_p, bonferroni_p))
    
    def test_multiple_testing_correction_fdr(self):
        """Test Benjamini-Hochberg FDR correction."""
        p_values = [0.001, 0.01, 0.05, 0.0001]
        corrected_p, significance = self.validator.multiple_testing_correction(
            p_values, method='fdr_bh'
        )
        
        assert len(corrected_p) == len(p_values)
        assert len(significance) == len(p_values)
        
        # FDR should be less conservative than Bonferroni
        bonferroni_p, _ = self.validator.multiple_testing_correction(
            p_values, method='bonferroni'
        )
        
        assert any(f <= b for f, b in zip(corrected_p, bonferroni_p))
    
    def test_multiple_testing_correction_empty_input(self):
        """Test multiple testing correction with empty input."""
        corrected_p, significance = self.validator.multiple_testing_correction([])
        
        assert corrected_p == []
        assert significance == []
    
    def test_multiple_testing_correction_invalid_method(self):
        """Test multiple testing correction with invalid method."""
        with pytest.raises(ValueError, match="Unknown correction method"):
            self.validator.multiple_testing_correction([0.01, 0.05], method='invalid')
    
    def test_calculate_effect_sizes(self):
        """Test effect size calculations."""
        effect_sizes = self.validator.calculate_effect_sizes(self.test_s1_values)
        
        # Check all required effect size measures
        required_keys = ['cohens_d', 'classical_excess', 'quantum_approach', 
                        'mean_violation_strength', 'max_violation_strength']
        for key in required_keys:
            assert key in effect_sizes
        
        # Check Cohen's d calculation
        mean_s1 = np.mean(np.abs(self.test_s1_values))
        std_s1 = np.std(np.abs(self.test_s1_values))
        expected_cohens_d = (mean_s1 - 2.0) / std_s1
        assert abs(effect_sizes['cohens_d'] - expected_cohens_d) < 1e-10
        
        # Check classical excess
        max_violation = np.max(np.abs(self.test_s1_values))
        expected_excess = ((max_violation - 2.0) / 2.0) * 100
        assert abs(effect_sizes['classical_excess'] - expected_excess) < 1e-10
        
        # Check quantum approach
        expected_quantum = (max_violation / (2 * np.sqrt(2))) * 100
        assert abs(effect_sizes['quantum_approach'] - expected_quantum) < 1e-10
    
    def test_calculate_effect_sizes_empty_input(self):
        """Test effect size calculation with empty input."""
        effect_sizes = self.validator.calculate_effect_sizes(np.array([]))
        
        assert effect_sizes['cohens_d'] == 0.0
        assert effect_sizes['classical_excess'] == 0.0
        assert effect_sizes['quantum_approach'] == 0.0
    
    def test_validate_significance_threshold(self):
        """Test significance threshold validation."""
        # Create mock statistical results
        results = [
            StatisticalResults(50.0, (40.0, 60.0), 0.0001, 2.5, np.array([]), True, 25.0),
            StatisticalResults(30.0, (20.0, 40.0), 0.01, 1.8, np.array([]), False, 15.0),
            StatisticalResults(60.0, (50.0, 70.0), 0.0005, 3.2, np.array([]), True, 35.0)
        ]
        
        validation = self.validator.validate_significance_threshold(results)
        
        assert validation['total_tests'] == 3
        assert validation['significant_tests'] == 2
        assert validation['ultra_significant_tests'] == 2  # p < 0.001
        assert validation['significance_rate'] == (2/3) * 100
        assert validation['ultra_significance_rate'] == (2/3) * 100
        assert validation['meets_requirement'] == True
    
    def test_validate_significance_threshold_empty_input(self):
        """Test significance threshold validation with empty input."""
        validation = self.validator.validate_significance_threshold([])
        
        assert validation['total_tests'] == 0
        assert validation['significant_tests'] == 0
        assert validation['significance_rate'] == 0.0

class TestInnovativeMetricsCalculator:
    """Test suite for InnovativeMetricsCalculator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.calculator = InnovativeMetricsCalculator()
        
        # Test data
        self.normal_violations = np.array([0.1, 0.15, 0.12, 0.18, 0.14])
        self.crisis_violations = np.array([0.4, 0.5, 0.45, 0.6, 0.55])
        self.tier_rates = {'Tier1': 45.0, 'Tier2': 30.0, 'Tier3': 20.0}
        self.transmission_lags = [5, 10, 15, 8, 12]
        self.transmission_strengths = [0.8, 0.6, 0.7, 0.9, 0.5]
        self.s1_time_series = np.array([2.5, -3.1, 1.8, 2.9, -2.2, 3.5, 1.5, -2.8])
    
    def test_crisis_amplification_factor(self):
        """Test crisis amplification factor calculation."""
        caf = self.calculator.crisis_amplification_factor(
            self.normal_violations, self.crisis_violations
        )
        
        normal_rate = np.mean(self.normal_violations) * 100
        crisis_rate = np.mean(self.crisis_violations) * 100
        expected_caf = crisis_rate / normal_rate
        
        assert abs(caf - expected_caf) < 1e-10
        assert caf > 1.0  # Crisis should amplify violations
    
    def test_crisis_amplification_factor_edge_cases(self):
        """Test crisis amplification factor edge cases."""
        # Empty arrays
        caf = self.calculator.crisis_amplification_factor(np.array([]), np.array([]))
        assert caf == 0.0
        
        # Zero normal violations
        zero_normal = np.array([0.0, 0.0, 0.0])
        crisis = np.array([0.3, 0.4, 0.5])
        caf = self.calculator.crisis_amplification_factor(zero_normal, crisis)
        assert caf == float('inf')
        
        # Zero both
        zero_crisis = np.array([0.0, 0.0, 0.0])
        caf = self.calculator.crisis_amplification_factor(zero_normal, zero_crisis)
        assert caf == 1.0
    
    def test_tier_vulnerability_index(self):
        """Test tier vulnerability index calculation."""
        vuln_index = self.calculator.tier_vulnerability_index(self.tier_rates)
        
        assert len(vuln_index) == 3
        assert all(0 <= score <= 100 for score in vuln_index.values())
        
        # Highest rate should have highest vulnerability
        max_rate_tier = max(self.tier_rates, key=self.tier_rates.get)
        assert vuln_index[max_rate_tier] == 100.0
        
        # Lowest rate should have lowest vulnerability
        min_rate_tier = min(self.tier_rates, key=self.tier_rates.get)
        assert vuln_index[min_rate_tier] == 0.0
    
    def test_tier_vulnerability_index_edge_cases(self):
        """Test tier vulnerability index edge cases."""
        # Empty input
        vuln_index = self.calculator.tier_vulnerability_index({})
        assert vuln_index == {}
        
        # All same rates
        same_rates = {'Tier1': 30.0, 'Tier2': 30.0, 'Tier3': 30.0}
        vuln_index = self.calculator.tier_vulnerability_index(same_rates)
        assert all(score == 50.0 for score in vuln_index.values())
    
    def test_transmission_efficiency_score(self):
        """Test transmission efficiency score calculation."""
        tes = self.calculator.transmission_efficiency_score(
            self.transmission_lags, self.transmission_strengths
        )
        
        assert 0 <= tes <= 100
        
        # Manual calculation
        expected_efficiencies = []
        for lag, strength in zip(self.transmission_lags, self.transmission_strengths):
            efficiency = abs(strength) / lag * 100
            expected_efficiencies.append(efficiency)
        expected_tes = np.mean(expected_efficiencies)
        
        assert abs(tes - expected_tes) < 1e-10
    
    def test_transmission_efficiency_score_edge_cases(self):
        """Test transmission efficiency score edge cases."""
        # Empty inputs
        tes = self.calculator.transmission_efficiency_score([], [])
        assert tes == 0.0
        
        # Zero lags (should be filtered out)
        zero_lags = [0, 5, 10]
        strengths = [0.8, 0.6, 0.7]
        tes = self.calculator.transmission_efficiency_score(zero_lags, strengths)
        
        # Should only consider non-zero lags
        expected = np.mean([0.6/5*100, 0.7/10*100])
        assert abs(tes - expected) < 1e-10
    
    def test_quantum_correlation_stability(self):
        """Test quantum correlation stability calculation."""
        stability = self.calculator.quantum_correlation_stability(self.s1_time_series)
        
        assert 0 <= stability <= 100
        
        # Should detect violations correctly
        violations = np.abs(self.s1_time_series) > 2.0
        assert np.sum(violations) > 0  # Test data should have violations
    
    def test_quantum_correlation_stability_edge_cases(self):
        """Test quantum correlation stability edge cases."""
        # Insufficient data
        short_series = np.array([2.5, -1.8])
        stability = self.calculator.quantum_correlation_stability(short_series, window_size=20)
        assert stability == 0.0
        
        # No violations
        no_violations = np.array([1.5, -1.2, 0.8, -1.9, 1.1] * 10)
        stability = self.calculator.quantum_correlation_stability(no_violations)
        assert stability == 0.0
    
    def test_cross_crisis_consistency_score(self):
        """Test cross-crisis consistency score calculation."""
        crisis_results = {
            'COVID': np.array([0.4, 0.5, 0.3, 0.6, 0.45]),
            'Ukraine': np.array([0.35, 0.55, 0.25, 0.65, 0.4]),
            'Financial': np.array([0.3, 0.6, 0.2, 0.7, 0.35])
        }
        
        consistency = self.calculator.cross_crisis_consistency_score(crisis_results)
        
        assert 0 <= consistency <= 100
    
    def test_cross_crisis_consistency_score_edge_cases(self):
        """Test cross-crisis consistency score edge cases."""
        # Single crisis
        single_crisis = {'COVID': np.array([0.4, 0.5, 0.3])}
        consistency = self.calculator.cross_crisis_consistency_score(single_crisis)
        assert consistency == 0.0
        
        # Empty input
        consistency = self.calculator.cross_crisis_consistency_score({})
        assert consistency == 0.0
    
    def test_sector_coupling_strength(self):
        """Test sector coupling strength calculation."""
        correlations = np.array([0.8, -0.6, 0.7, 0.9, -0.5])
        coupling = self.calculator.sector_coupling_strength(correlations)
        
        expected = np.mean(np.abs(correlations)) * 100
        assert abs(coupling - expected) < 1e-10
        assert 0 <= coupling <= 100
    
    def test_sector_coupling_strength_empty_input(self):
        """Test sector coupling strength with empty input."""
        coupling = self.calculator.sector_coupling_strength(np.array([]))
        assert coupling == 0.0
    
    def test_crisis_prediction_indicator(self):
        """Test crisis prediction indicator calculation."""
        pre_crisis = np.array([0.2, 0.25, 0.18, 0.3, 0.22])
        crisis = np.array([0.5, 0.6, 0.45, 0.7, 0.55])
        
        indicator = self.calculator.crisis_prediction_indicator(pre_crisis, crisis)
        
        assert 0 <= indicator <= 100
        
        # Manual calculation
        pre_mean = np.mean(pre_crisis)
        crisis_mean = np.mean(crisis)
        expected = min((pre_mean / crisis_mean) * 100, 100)
        
        assert abs(indicator - expected) < 1e-10
    
    def test_crisis_prediction_indicator_edge_cases(self):
        """Test crisis prediction indicator edge cases."""
        # Empty inputs
        indicator = self.calculator.crisis_prediction_indicator(np.array([]), np.array([]))
        assert indicator == 0.0
        
        # Zero crisis mean
        pre_crisis = np.array([0.2, 0.3])
        zero_crisis = np.array([0.0, 0.0])
        indicator = self.calculator.crisis_prediction_indicator(pre_crisis, zero_crisis)
        assert indicator == 0.0
    
    def test_recovery_resilience_metric(self):
        """Test recovery resilience metric calculation."""
        crisis = np.array([0.6, 0.7, 0.5, 0.8, 0.65])
        post_crisis = np.array([0.2, 0.3, 0.15, 0.25, 0.18])
        
        resilience = self.calculator.recovery_resilience_metric(crisis, post_crisis)
        
        assert 0 <= resilience <= 100
        
        # Manual calculation
        crisis_peak = np.max(crisis)
        post_mean = np.mean(post_crisis)
        expected = max(0, (1 - post_mean / crisis_peak) * 100)
        
        assert abs(resilience - expected) < 1e-10
    
    def test_recovery_resilience_metric_edge_cases(self):
        """Test recovery resilience metric edge cases."""
        # Empty inputs
        resilience = self.calculator.recovery_resilience_metric(np.array([]), np.array([]))
        assert resilience == 0.0
        
        # Zero crisis peak
        zero_crisis = np.array([0.0, 0.0])
        post_crisis = np.array([0.1, 0.2])
        resilience = self.calculator.recovery_resilience_metric(zero_crisis, post_crisis)
        assert resilience == 100.0
    
    def test_calculate_all_metrics(self):
        """Test calculation of all innovative metrics."""
        analysis_data = {
            'normal_violations': self.normal_violations,
            'crisis_violations': self.crisis_violations,
            'tier_violation_rates': self.tier_rates,
            'transmission_lags': self.transmission_lags,
            'transmission_strengths': self.transmission_strengths,
            's1_time_series': self.s1_time_series,
            'crisis_results': {
                'COVID': np.array([0.4, 0.5, 0.3]),
                'Ukraine': np.array([0.35, 0.55, 0.25])
            },
            'cross_sector_correlations': np.array([0.8, -0.6, 0.7]),
            'pre_crisis_violations': np.array([0.2, 0.25, 0.18]),
            'post_crisis_violations': np.array([0.15, 0.2, 0.12])
        }
        
        metrics = self.calculator.calculate_all_metrics(analysis_data)
        
        assert isinstance(metrics, InnovativeMetrics)
        assert metrics.crisis_amplification_factor > 0
        assert 0 <= metrics.tier_vulnerability_index <= 100
        assert 0 <= metrics.transmission_efficiency_score <= 100
        assert 0 <= metrics.quantum_correlation_stability <= 100
        assert 0 <= metrics.cross_crisis_consistency_score <= 100
        assert 0 <= metrics.sector_coupling_strength <= 100
        assert 0 <= metrics.crisis_prediction_indicator <= 100
        assert 0 <= metrics.recovery_resilience_metric <= 100

class TestComprehensiveStatisticalSuite:
    """Test suite for ComprehensiveStatisticalSuite class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.suite = ComprehensiveStatisticalSuite(n_bootstrap=50, random_state=42)
        
        # Mock S1 results
        self.mock_s1_results = {
            ('ASSET_A', 'ASSET_B'): {
                's1_time_series': [-2.5, 1.8, 3.1, -1.2, 2.8, -3.2, 1.5]
            },
            ('ASSET_C', 'ASSET_D'): {
                's1_time_series': [0.5, -3.2, 1.9, 2.4, -0.8, 2.6, -2.9]
            }
        }
        
        # Mock crisis data
        self.mock_crisis_data = {
            'normal_violations': np.array([0.1, 0.15, 0.12]),
            'crisis_violations': np.array([0.4, 0.5, 0.45]),
            'tier_violation_rates': {'Tier1': 45.0, 'Tier2': 30.0},
            'transmission_lags': [5, 10],
            'transmission_strengths': [0.8, 0.6],
            's1_time_series': np.array([2.5, -3.1, 1.8]),
            'crisis_results': {
                'COVID': np.array([0.4, 0.5]),
                'Ukraine': np.array([0.35, 0.55])
            },
            'cross_sector_correlations': np.array([0.8, -0.6]),
            'pre_crisis_violations': np.array([0.2, 0.25]),
            'post_crisis_violations': np.array([0.15, 0.2])
        }
    
    def test_comprehensive_analysis(self):
        """Test comprehensive statistical analysis."""
        results = self.suite.comprehensive_analysis(
            self.mock_s1_results, self.mock_crisis_data
        )
        
        # Check main structure
        assert 'pair_results' in results
        assert 'overall_statistics' in results
        assert 'overall_effect_sizes' in results
        assert 'multiple_testing_correction' in results
        assert 'significance_validation' in results
        assert 'innovative_metrics' in results
        assert 'summary' in results
        
        # Check pair results
        assert len(results['pair_results']) == 2
        for pair, pair_result in results['pair_results'].items():
            assert 'statistical_results' in pair_result
            assert 'effect_sizes' in pair_result
            assert 'corrected_p_value' in pair_result
            assert 'significant_after_correction' in pair_result
        
        # Check overall statistics
        overall_stats = results['overall_statistics']
        assert isinstance(overall_stats, StatisticalResults)
        assert overall_stats.violation_rate >= 0
        
        # Check innovative metrics
        innovative_metrics = results['innovative_metrics']
        assert isinstance(innovative_metrics, InnovativeMetrics)
        
        # Check summary
        summary = results['summary']
        assert summary['total_pairs'] == 2
        assert 'significant_pairs' in summary
        assert 'overall_violation_rate' in summary
        assert 'meets_science_requirements' in summary
    
    def test_comprehensive_analysis_no_crisis_data(self):
        """Test comprehensive analysis without crisis data."""
        results = self.suite.comprehensive_analysis(self.mock_s1_results)
        
        assert results['innovative_metrics'] is None
        assert 'pair_results' in results
        assert 'overall_statistics' in results
    
    def test_generate_publication_summary(self):
        """Test publication summary generation."""
        analysis_results = self.suite.comprehensive_analysis(
            self.mock_s1_results, self.mock_crisis_data
        )
        
        summary = self.suite.generate_publication_summary(analysis_results)
        
        assert isinstance(summary, str)
        assert 'STATISTICAL SUMMARY' in summary
        assert 'OVERALL RESULTS' in summary
        assert 'SIGNIFICANCE VALIDATION' in summary
        assert 'MULTIPLE TESTING CORRECTION' in summary
        assert 'INNOVATIVE AGRICULTURAL METRICS' in summary
        assert 'CONCLUSION' in summary
        
        # Check for key statistical values
        overall_stats = analysis_results['overall_statistics']
        assert f"{overall_stats.violation_rate:.2f}%" in summary
        assert f"p = {overall_stats.p_value:.2e}" in summary

def test_statistical_suite_integration():
    """Integration test for the complete statistical suite."""
    # Create realistic test data
    np.random.seed(42)
    
    # Generate S1 time series with violations
    n_points = 100
    s1_series_1 = np.random.normal(0, 1.5, n_points)
    s1_series_1[::10] = np.random.choice([-3.5, 3.5], size=len(s1_series_1[::10]))
    
    s1_series_2 = np.random.normal(0, 1.2, n_points)
    s1_series_2[::15] = np.random.choice([-2.8, 2.8], size=len(s1_series_2[::15]))
    
    mock_results = {
        ('CORN', 'ADM'): {'s1_time_series': s1_series_1.tolist()},
        ('WEAT', 'BG'): {'s1_time_series': s1_series_2.tolist()}
    }
    
    # Create crisis data
    crisis_data = {
        'normal_violations': np.random.beta(2, 8, 50),  # Low violation rates
        'crisis_violations': np.random.beta(5, 5, 30),  # Higher violation rates
        'tier_violation_rates': {
            'Energy': 45.0,
            'Transportation': 35.0,
            'Chemicals': 40.0
        },
        'transmission_lags': [3, 7, 14, 21, 30],
        'transmission_strengths': [0.8, 0.7, 0.6, 0.5, 0.4],
        's1_time_series': s1_series_1,
        'crisis_results': {
            'COVID-19': np.random.beta(6, 4, 40),
            'Ukraine War': np.random.beta(7, 3, 35),
            '2008 Crisis': np.random.beta(5, 5, 45)
        },
        'cross_sector_correlations': np.random.uniform(-0.9, 0.9, 20),
        'pre_crisis_violations': np.random.beta(3, 7, 30),
        'post_crisis_violations': np.random.beta(2, 8, 30)
    }
    
    # Run comprehensive analysis
    suite = ComprehensiveStatisticalSuite(n_bootstrap=100, random_state=42)
    results = suite.comprehensive_analysis(mock_results, crisis_data)
    
    # Validate results structure and content
    assert len(results['pair_results']) == 2
    assert results['overall_statistics'].violation_rate >= 0
    assert results['innovative_metrics'] is not None
    
    # Generate and validate publication summary
    summary = suite.generate_publication_summary(results)
    assert len(summary) > 1000  # Should be comprehensive
    
    # Check that all innovative metrics are calculated
    metrics = results['innovative_metrics']
    assert metrics.crisis_amplification_factor > 0
    assert 0 <= metrics.tier_vulnerability_index <= 100
    
    print("✅ Statistical suite integration test passed!")

if __name__ == "__main__":
    # Run integration test
    test_statistical_suite_integration()
    print("🧪 All statistical validation suite tests completed!")