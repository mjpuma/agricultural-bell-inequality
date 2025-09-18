#!/usr/bin/env python3
"""
TESTS FOR SEASONAL GEOGRAPHIC ANALYZER
======================================

This module contains comprehensive tests for the seasonal and geographic analysis
functionality in the agricultural cross-sector analysis system.

Test Coverage:
- Seasonal effect detection for agricultural cycles
- Geographic analysis of regional production patterns
- Seasonal modulation analysis
- Regional crisis impact analysis
- Integration with main analyzer

Authors: Agricultural Cross-Sector Analysis Team
Date: September 2025
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from seasonal_geographic_analyzer import (
    SeasonalGeographicAnalyzer, 
    SeasonalAnalysisResults,
    GeographicAnalysisResults,
    SeasonalGeographicResults,
    SeasonalPattern,
    GeographicRegion
)
from agricultural_universe_manager import AgriculturalUniverseManager


class TestSeasonalGeographicAnalyzer(unittest.TestCase):
    """Test cases for SeasonalGeographicAnalyzer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = SeasonalGeographicAnalyzer()
        
        # Create sample returns data
        self.dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
        self.tickers = ['ADM', 'BG', 'CF', 'MOS', 'DE', 'XOM', 'CVX']
        
        # Generate sample returns with seasonal patterns
        np.random.seed(42)
        self.returns_data = pd.DataFrame(index=self.dates, columns=self.tickers)
        
        for ticker in self.tickers:
            base_returns = np.random.normal(0, 0.02, len(self.dates))
            
            # Add seasonal modulation
            seasonal_factor = np.sin(2 * np.pi * self.dates.dayofyear / 365.25) * 0.005
            
            # Agricultural companies have stronger seasonal effects
            if ticker in ['ADM', 'BG', 'CF', 'MOS', 'DE']:
                seasonal_factor *= 2.0
            
            self.returns_data[ticker] = base_returns + seasonal_factor
        
        # Create sample asset pairs
        self.asset_pairs = [
            ('ADM', 'BG'),   # Agricultural-Agricultural
            ('CF', 'MOS'),   # Fertilizer-Fertilizer
            ('XOM', 'CF'),   # Energy-Fertilizer
            ('DE', 'ADM'),   # Equipment-Agricultural
            ('CVX', 'MOS')   # Energy-Fertilizer
        ]
    
    def test_initialization(self):
        """Test analyzer initialization."""
        self.assertIsInstance(self.analyzer, SeasonalGeographicAnalyzer)
        self.assertIsInstance(self.analyzer.universe_manager, AgriculturalUniverseManager)
        
        # Check agricultural seasons are defined
        self.assertEqual(len(self.analyzer.agricultural_seasons), 4)
        self.assertIn('Spring', self.analyzer.agricultural_seasons)
        self.assertIn('Summer', self.analyzer.agricultural_seasons)
        self.assertIn('Fall', self.analyzer.agricultural_seasons)
        self.assertIn('Winter', self.analyzer.agricultural_seasons)
        
        # Check geographic regions are defined
        self.assertGreaterEqual(len(self.analyzer.geographic_regions), 3)
        self.assertIn('North_America', self.analyzer.geographic_regions)
    
    def test_seasonal_patterns_structure(self):
        """Test seasonal patterns data structure."""
        for season_name, season_pattern in self.analyzer.agricultural_seasons.items():
            self.assertIsInstance(season_pattern, SeasonalPattern)
            self.assertEqual(season_pattern.season, season_name)
            self.assertIsInstance(season_pattern.months, list)
            self.assertEqual(len(season_pattern.months), 3)  # Each season has 3 months
            self.assertIsInstance(season_pattern.correlation_strength_modifier, float)
    
    def test_geographic_regions_structure(self):
        """Test geographic regions data structure."""
        for region_name, region in self.analyzer.geographic_regions.items():
            self.assertIsInstance(region, GeographicRegion)
            self.assertEqual(region.region_name, region_name)
            self.assertIsInstance(region.countries, list)
            self.assertIsInstance(region.primary_crops, list)
            self.assertIsInstance(region.harvest_seasons, list)
            self.assertIsInstance(region.risk_factors, list)
    
    def test_analyze_seasonal_effects(self):
        """Test seasonal effects analysis."""
        results = self.analyzer.analyze_seasonal_effects(self.returns_data, self.asset_pairs)
        
        # Check results structure
        self.assertIsInstance(results, SeasonalAnalysisResults)
        self.assertIsInstance(results.seasonal_patterns, dict)
        self.assertIsInstance(results.correlation_modulation, dict)
        self.assertIsInstance(results.seasonal_violation_rates, dict)
        
        # Check that we have results for each season
        expected_seasons = ['Winter', 'Spring', 'Summer', 'Fall']
        for season in expected_seasons:
            if season in results.seasonal_patterns:
                season_data = results.seasonal_patterns[season]
                self.assertIn('season_info', season_data)
                self.assertIn('data_points', season_data)
                self.assertGreater(season_data['data_points'], 0)
        
        # Check violation rates are reasonable
        for season, rate in results.seasonal_violation_rates.items():
            self.assertGreaterEqual(rate, 0.0)
            self.assertLessEqual(rate, 100.0)
    
    def test_analyze_geographic_effects(self):
        """Test geographic effects analysis."""
        results = self.analyzer.analyze_geographic_effects(self.returns_data, self.asset_pairs)
        
        # Check results structure
        self.assertIsInstance(results, GeographicAnalysisResults)
        self.assertIsInstance(results.regional_patterns, dict)
        self.assertIsInstance(results.cross_regional_correlations, dict)
        self.assertIsInstance(results.crisis_impact_by_region, dict)
        
        # Check regional patterns structure
        for region_name, region_data in results.regional_patterns.items():
            self.assertIn('region_info', region_data)
            self.assertIn('companies', region_data)
            self.assertIsInstance(region_data['companies'], list)
    
    def test_analyze_seasonal_modulation(self):
        """Test seasonal modulation analysis."""
        results = self.analyzer.analyze_seasonal_modulation(self.returns_data, self.asset_pairs)
        
        # Check results structure
        self.assertIsInstance(results, dict)
        
        # Check for expected keys
        if 'seasonal_modulation' in results:
            seasonal_modulation = results['seasonal_modulation']
            self.assertIsInstance(seasonal_modulation, dict)
            
            for season, data in seasonal_modulation.items():
                self.assertIn('modulation_factor', data)
                self.assertIn('strength_change', data)
                self.assertIsInstance(data['modulation_factor'], float)
                self.assertIsInstance(data['strength_change'], float)
    
    def test_analyze_regional_crisis_impact(self):
        """Test regional crisis impact analysis."""
        crisis_periods = {
            'Test_Crisis': ('2020-03-01', '2020-06-30')
        }
        
        results = self.analyzer.analyze_regional_crisis_impact(
            self.returns_data, crisis_periods, self.asset_pairs
        )
        
        # Check results structure
        self.assertIsInstance(results, dict)
        
        # Results should be organized by region
        for region_name, region_results in results.items():
            self.assertIsInstance(region_results, dict)
    
    def test_comprehensive_analysis(self):
        """Test comprehensive seasonal-geographic analysis."""
        crisis_periods = {
            'Test_Crisis': ('2020-03-01', '2020-06-30')
        }
        
        results = self.analyzer.run_comprehensive_seasonal_geographic_analysis(
            self.returns_data, self.asset_pairs, crisis_periods
        )
        
        # Check results structure
        self.assertIsInstance(results, SeasonalGeographicResults)
        self.assertIsInstance(results.seasonal_results, SeasonalAnalysisResults)
        self.assertIsInstance(results.geographic_results, GeographicAnalysisResults)
        self.assertIsInstance(results.seasonal_geographic_interactions, dict)
    
    def test_get_regional_companies(self):
        """Test regional company identification."""
        # Test North America
        na_companies = self.analyzer._get_regional_companies('North_America')
        self.assertIsInstance(na_companies, list)
        
        # Test with non-existent region
        unknown_companies = self.analyzer._get_regional_companies('Unknown_Region')
        self.assertIsInstance(unknown_companies, list)
    
    def test_adjust_for_southern_hemisphere(self):
        """Test Southern Hemisphere season adjustment."""
        northern_months = [12, 1, 2]  # Winter in Northern Hemisphere
        southern_months = self.analyzer._adjust_for_southern_hemisphere(northern_months)
        
        # Should be shifted by 6 months (Summer in Southern Hemisphere)
        expected_southern = [6, 7, 8]
        self.assertEqual(southern_months, expected_southern)
    
    def test_seasonal_significance_calculation(self):
        """Test seasonal significance calculation."""
        seasonal_rates = {
            'Winter': 10.0,
            'Spring': 25.0,
            'Summer': 15.0,
            'Fall': 30.0
        }
        
        significance = self.analyzer._calculate_seasonal_significance(seasonal_rates)
        
        self.assertIsInstance(significance, dict)
        self.assertEqual(len(significance), len(seasonal_rates))
        
        for season, p_value in significance.items():
            self.assertGreaterEqual(p_value, 0.0)
            self.assertLessEqual(p_value, 1.0)
    
    def test_crisis_amplification_calculation(self):
        """Test crisis amplification calculation."""
        # Create mock S1 results with clear amplification
        normal_s1 = {
            'pair_results': {
                ('ADM', 'BG'): {'s1_values': [1.5, 1.8, 1.2, 1.6]},  # Low violation rates
                ('CF', 'MOS'): {'s1_values': [1.3, 1.7, 1.4, 1.5]}
            }
        }
        
        crisis_s1 = {
            'pair_results': {
                ('ADM', 'BG'): {'s1_values': [2.5, 2.8, 2.3, 3.1]},  # High violation rates
                ('CF', 'MOS'): {'s1_values': [2.7, 2.9, 2.6, 2.8]}
            }
        }
        
        amplification = self.analyzer._calculate_crisis_amplification(normal_s1, crisis_s1)
        
        self.assertIsInstance(amplification, float)
        self.assertGreaterEqual(amplification, 1.0)  # Should show amplification or at least no decrease
    
    def test_empty_data_handling(self):
        """Test handling of empty or insufficient data."""
        # Test with empty DataFrame
        empty_data = pd.DataFrame()
        
        # Should handle gracefully without crashing
        try:
            results = self.analyzer.analyze_seasonal_effects(empty_data, [])
            self.assertIsInstance(results, SeasonalAnalysisResults)
        except Exception as e:
            # Should either work or fail gracefully
            self.assertIsInstance(e, (ValueError, IndexError, AttributeError))
    
    def test_insufficient_data_handling(self):
        """Test handling of insufficient data periods."""
        # Create very short time series
        short_dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='D')
        short_data = pd.DataFrame(
            np.random.randn(len(short_dates), 2),
            index=short_dates,
            columns=['A', 'B']
        )
        
        results = self.analyzer.analyze_seasonal_effects(short_data, [('A', 'B')])
        
        # Should handle gracefully
        self.assertIsInstance(results, SeasonalAnalysisResults)
    
    def test_monthly_variations_calculation(self):
        """Test monthly variations calculation in seasonal modulation."""
        # Create data with clear monthly pattern
        dates = pd.date_range(start='2022-01-01', end='2022-12-31', freq='D')
        
        # Create returns with monthly pattern
        returns = pd.DataFrame(index=dates, columns=['A', 'B'])
        
        for month in range(1, 13):
            month_mask = dates.month == month
            # Higher volatility in certain months
            volatility = 0.01 if month in [3, 4, 9, 10] else 0.005
            returns.loc[month_mask, 'A'] = np.random.normal(0, volatility, np.sum(month_mask))
            returns.loc[month_mask, 'B'] = np.random.normal(0, volatility, np.sum(month_mask))
        
        results = self.analyzer.analyze_seasonal_modulation(returns, [('A', 'B')])
        
        if 'monthly_variations' in results:
            monthly_vars = results['monthly_variations']
            self.assertIsInstance(monthly_vars, dict)
            
            # Should have entries for months with sufficient data
            for month, variation in monthly_vars.items():
                self.assertGreaterEqual(month, 1)
                self.assertLessEqual(month, 12)
                self.assertGreaterEqual(variation, 0.0)


class TestSeasonalVisualizationIntegration(unittest.TestCase):
    """Test integration with seasonal visualization suite."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = SeasonalGeographicAnalyzer()
        
        # Create minimal test data
        dates = pd.date_range(start='2022-01-01', end='2022-12-31', freq='D')
        self.returns_data = pd.DataFrame(
            np.random.randn(len(dates), 3) * 0.02,
            index=dates,
            columns=['A', 'B', 'C']
        )
        self.asset_pairs = [('A', 'B'), ('B', 'C')]
    
    def test_results_compatible_with_visualization(self):
        """Test that analysis results are compatible with visualization suite."""
        # Run analysis
        results = self.analyzer.run_comprehensive_seasonal_geographic_analysis(
            self.returns_data, self.asset_pairs
        )
        
        # Check that results have required structure for visualization
        self.assertIsInstance(results, SeasonalGeographicResults)
        self.assertIsInstance(results.seasonal_results, SeasonalAnalysisResults)
        self.assertIsInstance(results.geographic_results, GeographicAnalysisResults)
        
        # Check seasonal results have visualization-required fields
        seasonal_results = results.seasonal_results
        self.assertIsInstance(seasonal_results.seasonal_violation_rates, dict)
        self.assertIsInstance(seasonal_results.correlation_modulation, dict)
        
        # Check geographic results have visualization-required fields
        geographic_results = results.geographic_results
        self.assertIsInstance(geographic_results.regional_patterns, dict)
        self.assertIsInstance(geographic_results.cross_regional_correlations, dict)


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestSeasonalGeographicAnalyzer))
    test_suite.addTest(unittest.makeSuite(TestSeasonalVisualizationIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"SEASONAL GEOGRAPHIC ANALYZER TESTS COMPLETE")
    print(f"{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"\nSuccess rate: {success_rate:.1f}%")
    
    if success_rate == 100.0:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed. Check the output above.")