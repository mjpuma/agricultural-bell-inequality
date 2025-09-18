#!/usr/bin/env python3
"""
TESTS FOR AGRICULTURAL VISUALIZATION SUITE
==========================================

Comprehensive test suite for the agricultural cross-sector visualization
system including crisis period time series, innovative statistical 
visualizations, and three-crisis analysis framework.

Test Coverage:
- CrisisPeriodTimeSeriesVisualizer (Task 7.1)
- InnovativeStatisticalVisualizer (Task 7.2)
- ComprehensiveVisualizationSuite (Task 7)
- ThreeCrisisAnalysisFramework (Task 7.3)

Authors: Agricultural Cross-Sector Analysis Team
Date: September 2025
"""

import unittest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import tempfile
import shutil
import warnings

# Import visualization suite components
from src.agricultural_visualization_suite import (
    CrisisPeriodTimeSeriesVisualizer,
    InnovativeStatisticalVisualizer,
    ComprehensiveVisualizationSuite,
    ThreeCrisisAnalysisFramework,
    create_sample_data_for_testing
)

warnings.filterwarnings('ignore')

class TestCrisisPeriodTimeSeriesVisualizer(unittest.TestCase):
    """Test suite for Crisis Period Time Series Visualizations (Task 7.1)."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.visualizer = CrisisPeriodTimeSeriesVisualizer(figsize=(12, 8), dpi=100)
        self.temp_dir = tempfile.mkdtemp()
        
        # Create sample data
        dates = pd.date_range('2008-01-01', '2021-12-31', freq='D')
        np.random.seed(42)
        
        # Sample S1 time series with crisis amplification
        base_values = np.random.normal(1.5, 0.5, len(dates))
        
        # Add crisis amplification
        crisis_periods = [
            ('2008-09-01', '2009-03-31', 2.5),
            ('2010-05-01', '2012-12-31', 1.8),
            ('2020-02-01', '2020-12-31', 3.0)
        ]
        
        for start, end, factor in crisis_periods:
            mask = (dates >= start) & (dates <= end)
            base_values[mask] *= factor
        
        self.sample_s1_series = pd.Series(base_values, index=dates)
        
        # Sample crisis data
        self.sample_crisis_data = {
            '2008_financial': self.sample_s1_series['2008-09-01':'2009-03-31'],
            'eu_debt': self.sample_s1_series['2010-05-01':'2012-12-31'],
            'covid19': self.sample_s1_series['2020-02-01':'2020-12-31']
        }
        
        # Sample transmission data
        self.sample_transmission_data = {
            'energy_to_agriculture': pd.DataFrame({
                'correlation': np.random.uniform(-0.5, 0.8, len(dates)),
                'source': np.random.normal(1.0, 0.3, len(dates)),
                'target': np.random.normal(1.2, 0.4, len(dates))
            }, index=dates)
        }
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        plt.close('all')
    
    def test_crisis_s1_time_series_creation(self):
        """Test creation of crisis S1 time series plots."""
        
        # Test with valid data
        fig = self.visualizer.create_crisis_s1_time_series(
            self.sample_crisis_data, 'TEST_PAIR'
        )
        
        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.axes), 3)  # Should have 3 subplots for 3 crises
        
        # Test with save path
        save_path = os.path.join(self.temp_dir, 'test_crisis_s1.png')
        fig = self.visualizer.create_crisis_s1_time_series(
            self.sample_crisis_data, 'TEST_PAIR', save_path
        )
        
        self.assertTrue(os.path.exists(save_path))
        plt.close(fig)
    
    def test_rolling_violation_rate_series(self):
        """Test rolling violation rate time series creation."""
        
        fig = self.visualizer.create_rolling_violation_rate_series(
            self.sample_s1_series, window_size=20, pair_name='TEST_PAIR'
        )
        
        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.axes), 2)  # Should have 2 subplots
        
        # Test with save path
        save_path = os.path.join(self.temp_dir, 'test_rolling_violations.png')
        fig = self.visualizer.create_rolling_violation_rate_series(
            self.sample_s1_series, pair_name='TEST_PAIR', save_path=save_path
        )
        
        self.assertTrue(os.path.exists(save_path))
        plt.close(fig)
    
    def test_transmission_propagation_series(self):
        """Test transmission propagation time series creation."""
        
        fig = self.visualizer.create_transmission_propagation_series(
            self.sample_transmission_data
        )
        
        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.axes), 1)  # Should have 1 subplot for 1 transmission type
        
        # Test with save path
        save_path = os.path.join(self.temp_dir, 'test_transmission.png')
        fig = self.visualizer.create_transmission_propagation_series(
            self.sample_transmission_data, save_path
        )
        
        self.assertTrue(os.path.exists(save_path))
        plt.close(fig)
    
    def test_crisis_onset_detection(self):
        """Test crisis onset detection visualization."""
        
        fig = self.visualizer.create_crisis_onset_detection(
            self.sample_s1_series, detection_window=30
        )
        
        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.axes), 3)  # Should have 3 subplots
        
        # Test with save path
        save_path = os.path.join(self.temp_dir, 'test_crisis_onset.png')
        fig = self.visualizer.create_crisis_onset_detection(
            self.sample_s1_series, save_path=save_path
        )
        
        self.assertTrue(os.path.exists(save_path))
        plt.close(fig)
    
    def test_tier_specific_crisis_comparison(self):
        """Test tier-specific crisis comparison visualization."""
        
        # Create sample tier data
        tier_data = {
            'Food Processing': self.sample_crisis_data,
            'Fertilizer': self.sample_crisis_data
        }
        
        fig = self.visualizer.create_tier_specific_crisis_comparison(tier_data)
        
        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.axes), 2)  # Should have 2 subplots for 2 tiers
        
        # Test with save path
        save_path = os.path.join(self.temp_dir, 'test_tier_comparison.png')
        fig = self.visualizer.create_tier_specific_crisis_comparison(
            tier_data, save_path
        )
        
        self.assertTrue(os.path.exists(save_path))
        plt.close(fig)
    
    def test_seasonal_overlay_analysis(self):
        """Test seasonal overlay analysis visualization."""
        
        fig = self.visualizer.create_seasonal_overlay_analysis(self.sample_s1_series)
        
        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.axes), 4)  # Should have 4 subplots
        
        # Test with save path
        save_path = os.path.join(self.temp_dir, 'test_seasonal.png')
        fig = self.visualizer.create_seasonal_overlay_analysis(
            self.sample_s1_series, save_path
        )
        
        self.assertTrue(os.path.exists(save_path))
        plt.close(fig)


class TestInnovativeStatisticalVisualizer(unittest.TestCase):
    """Test suite for Innovative Statistical Analysis and Visualization Suite (Task 7.2)."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.visualizer = InnovativeStatisticalVisualizer(figsize=(12, 8), dpi=100)
        self.temp_dir = tempfile.mkdtemp()
        
        # Create sample correlation matrix
        assets = ['ADM_CORN', 'CAG_SOYB', 'CF_WEAT', 'DE_DBA', 'MOS_RICE']
        np.random.seed(42)
        corr_data = np.random.uniform(-0.8, 0.8, (len(assets), len(assets)))
        np.fill_diagonal(corr_data, 1.0)  # Perfect self-correlation
        
        self.sample_correlation_matrix = pd.DataFrame(
            corr_data, index=assets, columns=assets
        )
        
        # Create sample violation data
        dates = pd.date_range('2008-01-01', '2021-12-31', freq='D')
        self.sample_violation_data = {}
        
        for tier in ['Food Processing', 'Fertilizer', 'Equipment']:
            tier_data = pd.DataFrame({
                f'{tier}_Asset1': np.random.normal(1.5, 0.5, len(dates)),
                f'{tier}_Asset2': np.random.normal(1.3, 0.4, len(dates))
            }, index=dates)
            self.sample_violation_data[tier] = tier_data
        
        # Create sample transmission data
        self.sample_transmission_data = {
            'energy_to_agriculture': pd.DataFrame({
                'correlation': np.random.uniform(-0.5, 0.8, len(dates)),
            }, index=dates),
            'transport_to_agriculture': pd.DataFrame({
                'correlation': np.random.uniform(-0.3, 0.6, len(dates)),
            }, index=dates)
        }
        
        # Create sample S1 data
        self.sample_s1_data = {
            asset: pd.Series(np.random.normal(1.5, 0.5, len(dates)), index=dates)
            for asset in assets
        }
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        plt.close('all')
    
    def test_quantum_entanglement_network(self):
        """Test quantum entanglement network visualization."""
        
        fig = self.visualizer.create_quantum_entanglement_network(
            self.sample_correlation_matrix, threshold=0.3
        )
        
        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.axes), 2)  # Should have 2 subplots
        
        # Test with save path
        save_path = os.path.join(self.temp_dir, 'test_quantum_network.png')
        fig = self.visualizer.create_quantum_entanglement_network(
            self.sample_correlation_matrix, save_path=save_path
        )
        
        self.assertTrue(os.path.exists(save_path))
        plt.close(fig)
    
    def test_crisis_contagion_map(self):
        """Test crisis contagion map visualization."""
        
        fig = self.visualizer.create_crisis_contagion_map(self.sample_violation_data)
        
        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.axes), 4)  # Should have 4 subplots
        
        # Test with save path
        save_path = os.path.join(self.temp_dir, 'test_contagion_map.png')
        fig = self.visualizer.create_crisis_contagion_map(
            self.sample_violation_data, save_path
        )
        
        self.assertTrue(os.path.exists(save_path))
        plt.close(fig)
    
    def test_transmission_velocity_analysis(self):
        """Test transmission velocity analysis visualization."""
        
        fig = self.visualizer.create_transmission_velocity_analysis(
            self.sample_transmission_data
        )
        
        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.axes), 4)  # Should have 4 subplots
        
        # Test with save path
        save_path = os.path.join(self.temp_dir, 'test_velocity_analysis.png')
        fig = self.visualizer.create_transmission_velocity_analysis(
            self.sample_transmission_data, save_path
        )
        
        self.assertTrue(os.path.exists(save_path))
        plt.close(fig)
    
    def test_violation_intensity_heatmap(self):
        """Test violation intensity heatmap visualization."""
        
        fig = self.visualizer.create_violation_intensity_heatmap(self.sample_s1_data)
        
        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.axes), 2)  # Should have 2 subplots
        
        # Test with save path
        save_path = os.path.join(self.temp_dir, 'test_intensity_heatmap.png')
        fig = self.visualizer.create_violation_intensity_heatmap(
            self.sample_s1_data, save_path
        )
        
        self.assertTrue(os.path.exists(save_path))
        plt.close(fig)
    
    def test_tier_sensitivity_radar_chart(self):
        """Test tier sensitivity radar chart visualization."""
        
        # Create sample tier crisis data
        tier_crisis_data = {
            'Food Processing': {
                '2008 Financial': 35.5,
                'EU Debt': 28.3,
                'COVID-19': 42.1,
                'Overall Volatility': 35.3
            },
            'Fertilizer': {
                '2008 Financial': 28.7,
                'EU Debt': 31.2,
                'COVID-19': 25.8,
                'Overall Volatility': 28.6
            }
        }
        
        fig = self.visualizer.create_tier_sensitivity_radar_chart(tier_crisis_data)
        
        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.axes), 2)  # Should have 2 subplots for 2 tiers
        
        # Test with save path
        save_path = os.path.join(self.temp_dir, 'test_radar_chart.png')
        fig = self.visualizer.create_tier_sensitivity_radar_chart(
            tier_crisis_data, save_path
        )
        
        self.assertTrue(os.path.exists(save_path))
        plt.close(fig)


class TestThreeCrisisAnalysisFramework(unittest.TestCase):
    """Test suite for Three-Crisis Analysis Framework (Task 7.3)."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.framework = ThreeCrisisAnalysisFramework()
        self.temp_dir = tempfile.mkdtemp()
        
        # Create sample S1 data
        dates = pd.date_range('2007-01-01', '2022-12-31', freq='D')
        np.random.seed(42)
        
        self.sample_s1_data = {}
        self.sample_tier_mapping = {}
        
        asset_pairs = ['ADM_CORN', 'CAG_SOYB', 'CF_WEAT', 'DE_DBA', 'MOS_RICE']
        tiers = ['Food Processing', 'Food Processing', 'Fertilizer', 'Equipment', 'Fertilizer']
        
        for pair, tier in zip(asset_pairs, tiers):
            # Generate S1 data with crisis amplification
            base_values = np.random.normal(1.5, 0.5, len(dates))
            
            # Add crisis effects
            crisis_periods = [
                ('2008-09-01', '2009-03-31', 2.5),
                ('2010-05-01', '2012-12-31', 1.8),
                ('2020-02-01', '2020-12-31', 3.0)
            ]
            
            for start, end, factor in crisis_periods:
                mask = (dates >= start) & (dates <= end)
                base_values[mask] *= factor
            
            self.sample_s1_data[pair] = pd.Series(base_values, index=dates)
            self.sample_tier_mapping[pair] = tier
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_crisis_period_definitions(self):
        """Test crisis period definitions creation."""
        
        save_path = os.path.join(self.temp_dir, 'crisis_definitions.csv')
        crisis_df = self.framework.create_crisis_period_definitions(save_path)
        
        self.assertIsInstance(crisis_df, pd.DataFrame)
        self.assertEqual(len(crisis_df), 3)  # Should have 3 crisis periods
        self.assertTrue(os.path.exists(save_path))
        
        # Check required columns
        required_columns = ['name', 'start', 'end', 'duration_days', 'duration_months', 'description']
        for col in required_columns:
            self.assertIn(col, crisis_df.columns)
    
    def test_tier_specific_crisis_analysis(self):
        """Test tier-specific crisis analysis implementation."""
        
        save_path = os.path.join(self.temp_dir, 'tier_analysis.csv')
        tier_analysis = self.framework.implement_tier_specific_crisis_analysis(
            self.sample_s1_data, self.sample_tier_mapping, save_path
        )
        
        self.assertIsInstance(tier_analysis, dict)
        self.assertTrue(len(tier_analysis) > 0)
        self.assertTrue(os.path.exists(save_path))
        
        # Check structure
        for tier, crisis_data in tier_analysis.items():
            self.assertIsInstance(crisis_data, dict)
            for crisis_id in ['2008_financial', 'eu_debt', 'covid19']:
                self.assertIn(crisis_id, crisis_data)
    
    def test_crisis_amplification_metrics(self):
        """Test crisis amplification metrics creation."""
        
        save_path = os.path.join(self.temp_dir, 'amplification.csv')
        amplification_metrics = self.framework.create_crisis_amplification_metrics(
            self.sample_s1_data, save_path
        )
        
        self.assertIsInstance(amplification_metrics, dict)
        self.assertTrue(len(amplification_metrics) > 0)
        self.assertTrue(os.path.exists(save_path))
        
        # Check structure
        for pair, metrics in amplification_metrics.items():
            self.assertIsInstance(metrics, dict)
            for crisis_id in ['2008_financial', 'eu_debt', 'covid19']:
                self.assertIn(f'{crisis_id}_amplification', metrics)
                self.assertIn(f'{crisis_id}_crisis_rate', metrics)
                self.assertIn(f'{crisis_id}_normal_rate', metrics)
    
    def test_statistical_significance_testing(self):
        """Test statistical significance testing."""
        
        save_path = os.path.join(self.temp_dir, 'significance.csv')
        significance_results = self.framework.add_statistical_significance_testing(
            self.sample_s1_data, save_path
        )
        
        self.assertIsInstance(significance_results, dict)
        self.assertTrue(len(significance_results) > 0)
        self.assertTrue(os.path.exists(save_path))
        
        # Check structure
        for pair, results in significance_results.items():
            self.assertIsInstance(results, dict)
            for crisis_id in ['2008_financial', 'eu_debt', 'covid19']:
                self.assertIn(f'{crisis_id}_p_value', results)
                self.assertIn(f'{crisis_id}_significant', results)
    
    def test_cross_crisis_comparison_analysis(self):
        """Test cross-crisis comparison analysis implementation."""
        
        # First create amplification metrics
        amplification_metrics = self.framework.create_crisis_amplification_metrics(
            self.sample_s1_data
        )
        
        save_path = os.path.join(self.temp_dir, 'cross_crisis.csv')
        cross_crisis_comparison = self.framework.implement_cross_crisis_comparison_analysis(
            amplification_metrics, self.sample_tier_mapping, save_path
        )
        
        self.assertIsInstance(cross_crisis_comparison, dict)
        self.assertTrue(len(cross_crisis_comparison) > 0)
        self.assertTrue(os.path.exists(save_path))
        
        # Check structure
        for tier, data in cross_crisis_comparison.items():
            self.assertIsInstance(data, dict)
            for crisis_id in ['2008_financial', 'eu_debt', 'covid19']:
                self.assertIn(f'{crisis_id}_sensitivity', data)
                self.assertIn(f'{crisis_id}_rank', data)
    
    def test_crisis_recovery_analysis(self):
        """Test crisis recovery analysis."""
        
        save_path = os.path.join(self.temp_dir, 'recovery.csv')
        recovery_analysis = self.framework.add_crisis_recovery_analysis(
            self.sample_s1_data, recovery_window=90, save_path=save_path
        )
        
        self.assertIsInstance(recovery_analysis, dict)
        self.assertTrue(len(recovery_analysis) > 0)
        self.assertTrue(os.path.exists(save_path))
        
        # Check structure
        for pair, results in recovery_analysis.items():
            self.assertIsInstance(results, dict)
            for crisis_id in ['2008_financial', 'eu_debt', 'covid19']:
                self.assertIn(f'{crisis_id}_recovery_ratio', results)
                self.assertIn(f'{crisis_id}_recovery_speed', results)


class TestComprehensiveVisualizationSuite(unittest.TestCase):
    """Test suite for Comprehensive Visualization Suite integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.suite = ComprehensiveVisualizationSuite(figsize=(12, 8), dpi=100)
        self.temp_dir = tempfile.mkdtemp()
        
        # Create sample data
        self.sample_data = create_sample_data_for_testing()
        
        # Create sample sector/tier data
        self.sample_sector_data = {
            'Food Processing': {'Tier1': 35.5, 'Tier2': 28.3, 'Tier3': 22.1},
            'Fertilizer': {'Tier1': 28.7, 'Tier2': 31.2, 'Tier3': 25.8},
            'Equipment': {'Tier1': 22.3, 'Tier2': 19.5, 'Tier3': 18.2}
        }
        
        # Create sample three-crisis data
        self.sample_three_crisis_data = {
            '2008_financial': {'Food Processing': 42.1, 'Fertilizer': 35.8, 'Equipment': 28.5},
            'eu_debt': {'Food Processing': 38.7, 'Fertilizer': 41.2, 'Equipment': 32.1},
            'covid19': {'Food Processing': 55.3, 'Fertilizer': 28.9, 'Equipment': 25.7}
        }
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        plt.close('all')
    
    def test_sector_specific_heatmaps(self):
        """Test sector-specific heatmaps creation."""
        
        save_path = os.path.join(self.temp_dir, 'sector_heatmaps.png')
        fig = self.suite.create_sector_specific_heatmaps(
            self.sample_sector_data, save_path
        )
        
        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.axes), 2)  # Should have 2 subplots
        self.assertTrue(os.path.exists(save_path))
        plt.close(fig)
    
    def test_three_crisis_comparison_charts(self):
        """Test three-crisis comparison charts creation."""
        
        save_path = os.path.join(self.temp_dir, 'three_crisis.png')
        fig = self.suite.create_three_crisis_comparison_charts(
            self.sample_three_crisis_data, save_path
        )
        
        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.axes), 4)  # Should have 4 subplots
        self.assertTrue(os.path.exists(save_path))
        plt.close(fig)
    
    def test_publication_ready_summary(self):
        """Test publication-ready summary creation."""
        
        # Create sample analysis results
        analysis_results = {
            'violation_rates': {'ADM_CORN': 45.2, 'CAG_SOYB': 38.7, 'CF_WEAT': 32.1},
            'statistical_significance': {'ADM_CORN': 0.0001, 'CAG_SOYB': 0.0005, 'CF_WEAT': 0.002},
            'crisis_amplification': {'2008_financial': 2.5, 'eu_debt': 1.8, 'covid19': 3.2}
        }
        
        save_path = os.path.join(self.temp_dir, 'publication_summary.png')
        fig = self.suite.create_publication_ready_summary(analysis_results, save_path)
        
        self.assertIsNotNone(fig)
        self.assertTrue(os.path.exists(save_path))
        plt.close(fig)
    
    def test_visualization_summary(self):
        """Test visualization summary generation."""
        
        summary = self.suite.get_visualization_summary()
        
        self.assertIsInstance(summary, dict)
        self.assertIn('total_figures', summary)
        self.assertIn('visualization_types', summary)
        self.assertIn('publication_ready', summary)
        self.assertIn('crisis_periods_covered', summary)
        
        # Check expected visualization types
        expected_types = [
            'Crisis Period Time Series',
            'Rolling Violation Rates',
            'Transmission Propagation',
            'Crisis Onset Detection',
            'Tier-Specific Comparisons',
            'Seasonal Analysis',
            'Quantum Entanglement Networks',
            'Crisis Contagion Maps',
            'Transmission Velocity Analysis',
            'Violation Intensity Heatmaps',
            'Tier Sensitivity Radar Charts',
            'Sector-Specific Heatmaps',
            'Three-Crisis Comparisons',
            'Publication-Ready Summary'
        ]
        
        for viz_type in expected_types:
            self.assertIn(viz_type, summary['visualization_types'])


def run_comprehensive_tests():
    """Run all tests for the agricultural visualization suite."""
    
    print("🧪 RUNNING COMPREHENSIVE VISUALIZATION SUITE TESTS")
    print("=" * 60)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestCrisisPeriodTimeSeriesVisualizer,
        TestInnovativeStatisticalVisualizer,
        TestThreeCrisisAnalysisFramework,
        TestComprehensiveVisualizationSuite
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n📊 TEST SUMMARY")
    print("=" * 30)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print(f"\n⚠️  ERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    
    if success:
        print(f"\n✅ All tests passed! Visualization suite is ready for production.")
    else:
        print(f"\n❌ Some tests failed. Please review and fix issues.")
    
    return success


if __name__ == "__main__":
    success = run_comprehensive_tests()
    exit(0 if success else 1)