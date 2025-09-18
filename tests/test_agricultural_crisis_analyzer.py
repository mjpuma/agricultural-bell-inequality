#!/usr/bin/env python3
"""
Test suite for Agricultural Crisis Analyzer

Tests the crisis analysis functionality including:
- 2008 Financial Crisis analysis
- EU Debt Crisis analysis  
- COVID-19 Pandemic analysis
- Crisis comparison functionality
- Crisis amplification detection
- Statistical significance testing
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from agricultural_crisis_analyzer import (
    AgriculturalCrisisAnalyzer, CrisisPeriod, CrisisResults, ComparisonResults,
    quick_crisis_analysis, compare_all_crises
)
from agricultural_universe_manager import AgriculturalUniverseManager


class TestAgriculturalCrisisAnalyzer(unittest.TestCase):
    """Test cases for Agricultural Crisis Analyzer."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test data covering all crisis periods
        np.random.seed(42)  # For reproducible tests
        
        # Date range covering all three crisis periods
        self.dates = pd.date_range('2008-01-01', '2021-12-31', freq='D')
        
        # Create test returns data with crisis-specific volatility patterns
        self.test_returns = self._create_test_returns_data()
        
        # Initialize analyzer
        self.analyzer = AgriculturalCrisisAnalyzer()
    
    def _create_test_returns_data(self) -> pd.DataFrame:
        """Create realistic test returns data with crisis periods."""
        returns_data = pd.DataFrame(index=self.dates)
        
        # Test assets from different tiers
        test_assets = {
            # Agricultural companies (Tier 0)
            'ADM': 0.02,   # Archer Daniels Midland
            'CF': 0.025,   # CF Industries
            'BG': 0.022,   # Bunge
            'NTR': 0.024,  # Nutrien
            
            # Tier 1: Energy/Transport/Chemicals
            'XOM': 0.025,  # Exxon Mobil
            'CVX': 0.023,  # Chevron
            'UNP': 0.02,   # Union Pacific
            'DOW': 0.024,  # Dow Chemical
            
            # Tier 2: Finance/Equipment
            'JPM': 0.022,  # JPMorgan Chase
            'BAC': 0.025,  # Bank of America
            'CAT': 0.023,  # Caterpillar
            
            # Tier 3: Policy-linked
            'NEE': 0.018,  # NextEra Energy
            'AWK': 0.016,  # American Water Works
        }
        
        # Define crisis periods for volatility amplification
        crisis_periods = [
            ('2008-09-01', '2009-03-31', 2.5),  # 2008 Financial Crisis - high amplification
            ('2010-05-01', '2012-12-31', 1.8),  # EU Debt Crisis - moderate amplification
            ('2020-02-01', '2020-12-31', 3.0),  # COVID-19 - highest amplification
        ]
        
        for asset, base_vol in test_assets.items():
            # Generate base returns
            returns = np.random.normal(0, base_vol, len(self.dates))
            
            # Amplify volatility during crisis periods
            for start_date, end_date, amplification in crisis_periods:
                mask = (self.dates >= start_date) & (self.dates <= end_date)
                returns[mask] *= amplification
                
                # Add some correlation during crises (simulate contagion)
                if asset in ['JPM', 'BAC', 'XOM', 'CVX']:  # Financial and energy more affected
                    returns[mask] *= 1.2
            
            returns_data[asset] = returns
        
        return returns_data
    
    def test_crisis_analyzer_initialization(self):
        """Test crisis analyzer initialization."""
        analyzer = AgriculturalCrisisAnalyzer()
        
        # Check crisis periods are initialized
        self.assertEqual(len(analyzer.crisis_periods), 3)
        self.assertIn("2008_financial_crisis", analyzer.crisis_periods)
        self.assertIn("eu_debt_crisis", analyzer.crisis_periods)
        self.assertIn("covid19_pandemic", analyzer.crisis_periods)
        
        # Check crisis-specific parameters
        self.assertEqual(analyzer.crisis_calculator.window_size, 15)
        self.assertEqual(analyzer.crisis_calculator.threshold_quantile, 0.8)
        
        # Check normal period parameters
        self.assertEqual(analyzer.normal_calculator.window_size, 20)
        self.assertEqual(analyzer.normal_calculator.threshold_quantile, 0.75)
    
    def test_crisis_period_definitions(self):
        """Test crisis period definitions match requirements."""
        # 2008 Financial Crisis
        crisis_2008 = self.analyzer.crisis_periods["2008_financial_crisis"]
        self.assertEqual(crisis_2008.start_date, "2008-09-01")
        self.assertEqual(crisis_2008.end_date, "2009-03-31")
        self.assertEqual(crisis_2008.expected_violation_rate, 50.0)
        
        # EU Debt Crisis
        crisis_eu = self.analyzer.crisis_periods["eu_debt_crisis"]
        self.assertEqual(crisis_eu.start_date, "2010-05-01")
        self.assertEqual(crisis_eu.end_date, "2012-12-31")
        self.assertEqual(crisis_eu.expected_violation_rate, 45.0)
        
        # COVID-19 Pandemic
        crisis_covid = self.analyzer.crisis_periods["covid19_pandemic"]
        self.assertEqual(crisis_covid.start_date, "2020-02-01")
        self.assertEqual(crisis_covid.end_date, "2020-12-31")
        self.assertEqual(crisis_covid.expected_violation_rate, 55.0)
    
    def test_2008_financial_crisis_analysis(self):
        """Test 2008 financial crisis analysis implementation."""
        results = self.analyzer.analyze_2008_financial_crisis(self.test_returns)
        
        # Check result structure
        self.assertIsInstance(results, CrisisResults)
        self.assertEqual(results.crisis_period.name, "2008 Financial Crisis")
        
        # Check tier results exist
        self.assertIn(1, results.tier_results)  # Tier 1 should be analyzed
        
        # Check crisis amplification is calculated
        self.assertIn("overall", results.crisis_amplification)
        self.assertGreater(results.crisis_amplification["overall"], 0)
        
        # Check statistical significance
        self.assertIsInstance(results.statistical_significance, dict)
        
        # Check comparison with normal periods
        self.assertIsInstance(results.comparison_with_normal, dict)
    
    def test_eu_debt_crisis_analysis(self):
        """Test EU debt crisis analysis implementation."""
        results = self.analyzer.analyze_eu_debt_crisis(self.test_returns)
        
        # Check result structure
        self.assertIsInstance(results, CrisisResults)
        self.assertEqual(results.crisis_period.name, "EU Debt Crisis")
        
        # Check tier results
        self.assertTrue(len(results.tier_results) > 0)
        
        # Check crisis-specific metrics
        for tier_result in results.tier_results.values():
            if "crisis_specific_metrics" in tier_result:
                metrics = tier_result["crisis_specific_metrics"]
                self.assertIn("violation_rate", metrics)
                self.assertIn("amplification_factor", metrics)
                self.assertIn("severity", metrics)
    
    def test_covid19_pandemic_analysis(self):
        """Test COVID-19 pandemic analysis implementation."""
        results = self.analyzer.analyze_covid19_pandemic(self.test_returns)
        
        # Check result structure
        self.assertIsInstance(results, CrisisResults)
        self.assertEqual(results.crisis_period.name, "COVID-19 Pandemic")
        
        # COVID-19 should have highest expected violation rate
        self.assertEqual(results.crisis_period.expected_violation_rate, 55.0)
        
        # Check transmission analysis
        self.assertIsInstance(results.transmission_analysis, dict)
        self.assertTrue(len(results.transmission_analysis) > 0)
    
    def test_crisis_comparison_functionality(self):
        """Test crisis comparison across three historical periods."""
        comparison_results = self.analyzer.compare_crisis_periods(self.test_returns)
        
        # Check result structure
        self.assertIsInstance(comparison_results, ComparisonResults)
        self.assertEqual(len(comparison_results.crisis_periods), 3)
        
        # Check comparative violation rates
        self.assertIsInstance(comparison_results.comparative_violation_rates, dict)
        
        # Check crisis ranking by tier
        self.assertIsInstance(comparison_results.crisis_ranking, dict)
        for tier in [1, 2, 3]:
            if tier in comparison_results.crisis_ranking:
                rankings = comparison_results.crisis_ranking[tier]
                self.assertIsInstance(rankings, list)
                # Rankings should be sorted by violation rate (descending)
                if len(rankings) > 1:
                    self.assertGreaterEqual(rankings[0][1], rankings[1][1])
        
        # Check tier vulnerability index
        self.assertIsInstance(comparison_results.tier_vulnerability_index, dict)
        for tier, vulnerability in comparison_results.tier_vulnerability_index.items():
            self.assertGreaterEqual(vulnerability, 0.0)
            self.assertLessEqual(vulnerability, 100.0)
    
    def test_crisis_amplification_detection(self):
        """Test crisis amplification detection (40-60% violation rates expected)."""
        # Test with COVID-19 (highest expected amplification)
        results = self.analyzer.analyze_covid19_pandemic(self.test_returns)
        
        # Check amplification factors
        amplification = results.crisis_amplification
        self.assertIn("overall", amplification)
        
        # Overall amplification should be positive
        self.assertGreater(amplification["overall"], 0)
        
        # Check crisis-specific metrics meet thresholds
        for tier_result in results.tier_results.values():
            if "crisis_specific_metrics" in tier_result:
                metrics = tier_result["crisis_specific_metrics"]
                
                # Check if meets crisis threshold (40-60% range)
                self.assertIn("meets_crisis_threshold", metrics)
                self.assertIsInstance(metrics["meets_crisis_threshold"], bool)
                
                # Check severity classification
                self.assertIn("severity", metrics)
                self.assertIn(metrics["severity"], ["Low", "Moderate", "High", "Severe"])
    
    def test_crisis_specific_parameters(self):
        """Test crisis-specific parameters (window size 15, threshold quantile 0.8)."""
        # Check crisis calculator parameters
        self.assertEqual(self.analyzer.crisis_calculator.window_size, 15)
        self.assertEqual(self.analyzer.crisis_calculator.threshold_quantile, 0.8)
        self.assertEqual(self.analyzer.crisis_calculator.threshold_method, 'quantile')
        
        # Check normal calculator parameters (for comparison)
        self.assertEqual(self.analyzer.normal_calculator.window_size, 20)
        self.assertEqual(self.analyzer.normal_calculator.threshold_quantile, 0.75)
    
    def test_statistical_significance_testing(self):
        """Test statistical significance with p < 0.001 requirement."""
        results = self.analyzer.analyze_covid19_pandemic(self.test_returns)
        
        significance = results.statistical_significance
        
        # Check p-values are calculated
        p_value_keys = [key for key in significance.keys() if key.endswith('_p_value')]
        self.assertTrue(len(p_value_keys) > 0)
        
        # Check significance flags
        sig_keys = [key for key in significance.keys() if key.endswith('_significant')]
        self.assertTrue(len(sig_keys) > 0)
        
        # P-values should be between 0 and 1
        for key in p_value_keys:
            p_value = significance[key]
            self.assertGreaterEqual(p_value, 0.0)
            self.assertLessEqual(p_value, 1.0)
    
    def test_data_filtering_to_crisis_periods(self):
        """Test data filtering to specific crisis periods."""
        # Test 2008 crisis period filtering
        crisis_2008 = self.analyzer.crisis_periods["2008_financial_crisis"]
        filtered_data = self.analyzer._filter_to_crisis_period(self.test_returns, crisis_2008)
        
        # Check date range
        self.assertGreaterEqual(filtered_data.index.min(), pd.to_datetime("2008-09-01"))
        self.assertLessEqual(filtered_data.index.max(), pd.to_datetime("2009-03-31"))
        
        # Check data is not empty
        self.assertFalse(filtered_data.empty)
        
        # Test COVID-19 period filtering
        crisis_covid = self.analyzer.crisis_periods["covid19_pandemic"]
        filtered_covid = self.analyzer._filter_to_crisis_period(self.test_returns, crisis_covid)
        
        self.assertGreaterEqual(filtered_covid.index.min(), pd.to_datetime("2020-02-01"))
        self.assertLessEqual(filtered_covid.index.max(), pd.to_datetime("2020-12-31"))
    
    def test_convenience_functions(self):
        """Test convenience functions for quick analysis."""
        # Test quick crisis analysis
        covid_results = quick_crisis_analysis(self.test_returns, "covid19_pandemic")
        self.assertIsInstance(covid_results, CrisisResults)
        self.assertEqual(covid_results.crisis_period.name, "COVID-19 Pandemic")
        
        # Test compare all crises
        comparison = compare_all_crises(self.test_returns)
        self.assertIsInstance(comparison, ComparisonResults)
        self.assertEqual(len(comparison.crisis_periods), 3)
    
    def test_error_handling(self):
        """Test error handling for edge cases."""
        # Test with empty data
        empty_data = pd.DataFrame()
        
        with self.assertRaises(ValueError):
            self.analyzer.analyze_2008_financial_crisis(empty_data)
        
        # Test with invalid crisis period
        with self.assertRaises(ValueError):
            quick_crisis_analysis(self.test_returns, "invalid_crisis")
        
        # Test with insufficient date range
        short_data = self.test_returns.loc['2020-01-01':'2020-01-31']  # Too short for COVID analysis
        
        # Should handle gracefully (may return empty results but not crash)
        try:
            results = self.analyzer.analyze_covid19_pandemic(short_data)
            # If it doesn't raise an error, check it handles gracefully
            self.assertIsInstance(results, CrisisResults)
        except ValueError as e:
            # Expected behavior for insufficient data
            self.assertIn("No data available", str(e))
    
    def test_tier_analysis_integration(self):
        """Test integration with tier classification system."""
        # Check that analyzer uses universe manager correctly
        self.assertIsInstance(self.analyzer.universe_manager, AgriculturalUniverseManager)
        
        # Check tier classifications
        tier1_assets = self.analyzer.universe_manager.classify_by_tier(1)
        tier2_assets = self.analyzer.universe_manager.classify_by_tier(2)
        tier3_assets = self.analyzer.universe_manager.classify_by_tier(3)
        agricultural_assets = self.analyzer.universe_manager.classify_by_tier(0)
        
        # All tier lists should be non-empty
        self.assertTrue(len(tier1_assets) > 0)
        self.assertTrue(len(tier2_assets) > 0)
        self.assertTrue(len(tier3_assets) > 0)
        self.assertTrue(len(agricultural_assets) > 0)
    
    def test_crisis_transmission_mechanisms(self):
        """Test transmission mechanism analysis."""
        results = self.analyzer.analyze_covid19_pandemic(self.test_returns)
        
        transmission = results.transmission_analysis
        self.assertIsInstance(transmission, dict)
        
        # Should have transmission mechanisms for COVID-19
        covid_period = self.analyzer.crisis_periods["covid19_pandemic"]
        expected_mechanisms = len(covid_period.key_transmission_mechanisms)
        self.assertEqual(len(transmission), expected_mechanisms)
        
        # Each mechanism should have analysis results
        for mechanism_key, mechanism_result in transmission.items():
            self.assertIn("mechanism", mechanism_result)
            self.assertIn("crisis_period", mechanism_result)
            self.assertIn("analysis_status", mechanism_result)


class TestCrisisDataModels(unittest.TestCase):
    """Test crisis data models and structures."""
    
    def test_crisis_period_model(self):
        """Test CrisisPeriod data model."""
        crisis = CrisisPeriod(
            name="Test Crisis",
            start_date="2020-01-01",
            end_date="2020-12-31",
            description="Test crisis period",
            affected_sectors=["Finance", "Energy"],
            expected_violation_rate=45.0,
            key_transmission_mechanisms=["Test mechanism"],
            crisis_specific_params={"window_size": 15}
        )
        
        self.assertEqual(crisis.name, "Test Crisis")
        self.assertEqual(crisis.expected_violation_rate, 45.0)
        self.assertIn("Finance", crisis.affected_sectors)
        self.assertEqual(crisis.crisis_specific_params["window_size"], 15)
    
    def test_crisis_results_model(self):
        """Test CrisisResults data model structure."""
        # Create minimal crisis period
        crisis_period = CrisisPeriod(
            name="Test", start_date="2020-01-01", end_date="2020-12-31",
            description="Test", affected_sectors=[], expected_violation_rate=50.0,
            key_transmission_mechanisms=[], crisis_specific_params={}
        )
        
        results = CrisisResults(
            crisis_period=crisis_period,
            tier_results={1: {"test": "data"}},
            crisis_amplification={"overall": 1.5},
            transmission_analysis={"test": {"status": "detected"}},
            statistical_significance={"tier_1_p_value": 0.001},
            comparison_with_normal={"tier_1": {"amplification_ratio": 2.0}}
        )
        
        self.assertEqual(results.crisis_period.name, "Test")
        self.assertIn(1, results.tier_results)
        self.assertEqual(results.crisis_amplification["overall"], 1.5)
    
    def test_comparison_results_model(self):
        """Test ComparisonResults data model structure."""
        comparison = ComparisonResults(
            crisis_periods=["Crisis 1", "Crisis 2"],
            comparative_violation_rates={"Crisis 1": {1: 45.0}, "Crisis 2": {1: 55.0}},
            crisis_ranking={1: [("Crisis 2", 55.0), ("Crisis 1", 45.0)]},
            cross_crisis_consistency={1: 0.8},
            tier_vulnerability_index={1: 50.0}
        )
        
        self.assertEqual(len(comparison.crisis_periods), 2)
        self.assertEqual(comparison.comparative_violation_rates["Crisis 1"][1], 45.0)
        self.assertEqual(comparison.crisis_ranking[1][0][0], "Crisis 2")  # Highest rate first
        self.assertEqual(comparison.tier_vulnerability_index[1], 50.0)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)