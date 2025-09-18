#!/usr/bin/env python3
"""
Tests for Agricultural Data Handling System

Comprehensive tests for data loading, validation, and processing capabilities.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import shutil
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from agricultural_data_handler import (
    AgriculturalDataHandler, 
    DataQualityMetrics, 
    RollingWindowConfig,
    load_agricultural_data,
    create_rolling_analysis_data
)

class TestAgriculturalDataHandler(unittest.TestCase):
    """Test cases for AgriculturalDataHandler."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.handler = AgriculturalDataHandler(
            min_observations=50,
            cache_dir=self.temp_dir
        )
        
        # Create sample data for testing
        self.sample_tickers = ['TEST1', 'TEST2', 'TEST3']
        self.sample_data = self._create_sample_data()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def _create_sample_data(self):
        """Create sample price data for testing."""
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        
        sample_data = {}
        for ticker in self.sample_tickers:
            # Create realistic price data with some volatility
            np.random.seed(hash(ticker) % 2**32)  # Consistent random data per ticker
            
            prices = 100 * np.exp(np.cumsum(np.random.normal(0, 0.02, len(dates))))
            volumes = np.random.randint(100000, 1000000, len(dates))
            
            df = pd.DataFrame({
                'Open': prices * (1 + np.random.normal(0, 0.01, len(dates))),
                'High': prices * (1 + np.abs(np.random.normal(0, 0.02, len(dates)))),
                'Low': prices * (1 - np.abs(np.random.normal(0, 0.02, len(dates)))),
                'Close': prices,
                'Adj Close': prices,  # Use space instead of underscore
                'Volume': volumes
            }, index=dates)
            
            sample_data[ticker] = df
        
        return sample_data
    
    def test_initialization(self):
        """Test handler initialization."""
        handler = AgriculturalDataHandler(
            min_observations=100,
            data_frequency='daily',
            validation_strict=True
        )
        
        self.assertEqual(handler.min_observations, 100)
        self.assertEqual(handler.data_frequency, 'daily')
        self.assertTrue(handler.validation_strict)
        self.assertEqual(len(handler.raw_data), 0)
        self.assertEqual(len(handler.processed_data), 0)
    
    def test_date_range_setting(self):
        """Test date range configuration."""
        # Test with explicit dates
        self.handler.set_date_range('2023-01-01', '2023-12-31')
        self.assertEqual(self.handler.start_date, '2023-01-01')
        self.assertEqual(self.handler.end_date, '2023-12-31')
        self.assertIsNotNone(self.handler.date_range_days)
        
        # Test with period
        self.handler.set_date_range(period='1y')
        self.assertIsNotNone(self.handler.start_date)
        self.assertIsNotNone(self.handler.end_date)
    
    def test_data_quality_metrics_calculation(self):
        """Test data quality metrics calculation."""
        # Test with good quality data
        good_data = self.sample_data['TEST1']
        metrics = self.handler._calculate_quality_metrics('TEST1', good_data)
        
        self.assertIsInstance(metrics, DataQualityMetrics)
        self.assertEqual(metrics.ticker, 'TEST1')
        self.assertTrue(metrics.total_observations > 0)
        self.assertTrue(metrics.validation_passed)
        self.assertGreater(metrics.data_quality_score, 0.8)
        
        # Test with poor quality data (lots of missing values)
        poor_data = good_data.copy()
        poor_data.loc[poor_data.index[::2]] = np.nan  # Remove every other row
        
        poor_metrics = self.handler._calculate_quality_metrics('POOR', poor_data)
        self.assertFalse(poor_metrics.validation_passed)
        self.assertGreater(poor_metrics.missing_values, 0)
        self.assertLess(poor_metrics.data_quality_score, 0.8)
    
    def test_data_validation(self):
        """Test comprehensive data validation."""
        # Set up handler with sample data
        self.handler.raw_data = self.sample_data
        
        quality_metrics = self.handler.validate_data_quality()
        
        self.assertEqual(len(quality_metrics), len(self.sample_tickers))
        
        for ticker in self.sample_tickers:
            self.assertIn(ticker, quality_metrics)
            metrics = quality_metrics[ticker]
            self.assertIsInstance(metrics, DataQualityMetrics)
            self.assertTrue(metrics.validation_passed)
            self.assertGreaterEqual(metrics.total_observations, self.handler.min_observations)
    
    def test_daily_returns_processing(self):
        """Test daily returns calculation."""
        # Set up handler with sample data
        self.handler.raw_data = self.sample_data
        
        processed_data = self.handler.process_daily_returns()
        
        self.assertEqual(len(processed_data), len(self.sample_tickers))
        
        for ticker in self.sample_tickers:
            self.assertIn(ticker, processed_data)
            returns_df = processed_data[ticker]
            
            # Check structure
            self.assertIn('returns', returns_df.columns)
            self.assertIn('price', returns_df.columns)
            
            # Check returns calculation
            returns = returns_df['returns']
            prices = returns_df['price']
            
            # Verify returns are calculated correctly: Ri,t = (Pi,t - Pi,t-1) / Pi,t-1
            expected_returns = prices.pct_change().dropna()
            
            # Check that returns match expected calculation
            common_index = returns.dropna().index.intersection(expected_returns.index)
            if len(common_index) > 0:
                pd.testing.assert_series_equal(
                    returns.loc[common_index], 
                    expected_returns.loc[common_index], 
                    check_names=False
                )
    
    def test_rolling_windows_creation(self):
        """Test rolling window creation."""
        # Set up processed data
        self.handler.processed_data = {
            'TEST1': pd.DataFrame({
                'returns': np.random.normal(0, 0.02, 100),
                'price': 100 + np.cumsum(np.random.normal(0, 1, 100))
            }, index=pd.date_range('2023-01-01', periods=100))
        }
        
        window_config = RollingWindowConfig(
            window_size=20,
            min_periods=15,
            step_size=1
        )
        
        rolling_windows = self.handler.create_rolling_windows(window_config)
        
        self.assertIn('TEST1', rolling_windows)
        windows = rolling_windows['TEST1']
        
        # Check that windows were created
        self.assertGreater(len(windows), 0)
        
        # Check window properties
        for window in windows:
            self.assertLessEqual(len(window), window_config.window_size)
            self.assertGreaterEqual(len(window), window_config.min_periods)
    
    def test_missing_data_handling(self):
        """Test missing data handling methods."""
        # Create data with missing values
        data_with_missing = {}
        for ticker in self.sample_tickers:
            df = self.sample_data[ticker].copy()
            # Introduce missing values
            df.iloc[10:15] = np.nan
            df.iloc[50:52] = np.nan
            data_with_missing[ticker] = df
        
        # Test forward fill
        cleaned_data = self.handler.handle_missing_data(
            data_with_missing, 
            method='forward_fill',
            max_consecutive_missing=10
        )
        
        for ticker in self.sample_tickers:
            original_missing = data_with_missing[ticker].isnull().sum().sum()
            cleaned_missing = cleaned_data[ticker].isnull().sum().sum()
            
            # Should have fewer missing values
            self.assertLessEqual(cleaned_missing, original_missing)
    
    def test_data_summary(self):
        """Test data summary generation."""
        # Set up handler with data
        self.handler.raw_data = self.sample_data
        self.handler.failed_tickers = ['FAILED1', 'FAILED2']
        self.handler.set_date_range('2023-01-01', '2023-12-31')
        
        # Validate and process data
        self.handler.validate_data_quality()
        self.handler.process_daily_returns()
        
        summary = self.handler.get_data_summary()
        
        # Check summary structure
        self.assertIn('total_tickers_requested', summary)
        self.assertIn('successful_loads', summary)
        self.assertIn('failed_loads', summary)
        self.assertIn('date_range', summary)
        self.assertIn('data_quality', summary)
        self.assertIn('processed_data', summary)
        
        # Check values
        self.assertEqual(summary['successful_loads'], len(self.sample_tickers))
        self.assertEqual(summary['failed_loads'], 2)
        self.assertEqual(summary['date_range']['start_date'], '2023-01-01')
        self.assertEqual(summary['date_range']['end_date'], '2023-12-31')
    
    def test_cache_functionality(self):
        """Test data caching functionality."""
        # Create sample data
        sample_df = self.sample_data['TEST1']
        
        # Test cache save
        self.handler.set_date_range('2023-01-01', '2023-12-31')
        self.handler._save_to_cache('TEST1', sample_df)
        
        # Test cache load
        loaded_data = self.handler._load_from_cache('TEST1')
        
        self.assertIsNotNone(loaded_data)
        # Compare data values, ignoring index frequency differences
        pd.testing.assert_frame_equal(sample_df, loaded_data, check_freq=False)
    
    def test_minimum_observations_requirement(self):
        """Test minimum observations requirement (Requirement 5.2)."""
        # Create handler with high minimum requirement
        handler = AgriculturalDataHandler(min_observations=200)
        
        # Create data with insufficient observations
        short_data = {
            'SHORT': pd.DataFrame({
                'Adj_Close': np.random.randn(50)
            }, index=pd.date_range('2023-01-01', periods=50))
        }
        
        quality_metrics = handler.validate_data_quality(short_data)
        
        # Should fail validation due to insufficient data
        self.assertFalse(quality_metrics['SHORT'].validation_passed)
        self.assertIn('Insufficient data', ' '.join(quality_metrics['SHORT'].issues))
    
    def test_daily_frequency_processing(self):
        """Test daily frequency data processing (Requirement 5.4)."""
        # Verify that returns are calculated using daily frequency
        self.handler.raw_data = self.sample_data
        processed_data = self.handler.process_daily_returns()
        
        for ticker, returns_df in processed_data.items():
            # Check that index is daily frequency
            self.assertTrue(returns_df.index.freq is None or 'D' in str(returns_df.index.freq))
            
            # Check that returns follow the formula: Ri,t = (Pi,t - Pi,t-1) / Pi,t-1
            returns = returns_df['returns'].dropna()
            prices = returns_df['price']
            
            # Manually calculate expected returns
            expected_returns = prices.pct_change().dropna()
            
            # Check that returns match expected calculation
            common_index = returns.index.intersection(expected_returns.index)
            if len(common_index) > 0:
                # Should be approximately equal (allowing for floating point precision)
                np.testing.assert_array_almost_equal(
                    returns.loc[common_index].values, 
                    expected_returns.loc[common_index].values,
                    decimal=10
                )
    
    def test_rolling_window_endpoints(self):
        """Test rolling window analysis with endpoints T ranging from w to N (Requirement 5.5)."""
        # Create test data with known length
        test_data = pd.DataFrame({
            'returns': np.random.normal(0, 0.02, 100),
            'price': 100 + np.cumsum(np.random.normal(0, 1, 100))
        }, index=pd.date_range('2023-01-01', periods=100))
        
        self.handler.processed_data = {'TEST': test_data}
        
        window_size = 20
        window_config = RollingWindowConfig(
            window_size=window_size,
            min_periods=window_size
        )
        
        rolling_windows = self.handler.create_rolling_windows(window_config)
        windows = rolling_windows['TEST']
        
        # Check that endpoints range from w to N
        # First window should end at position w (20)
        # Last window should end at position N (100)
        
        self.assertGreater(len(windows), 0)
        
        # Each window should have exactly window_size observations
        for window in windows:
            self.assertEqual(len(window), window_size)
        
        # Should have windows for T from w to N
        expected_num_windows = 100 - window_size + 1  # N - w + 1
        self.assertEqual(len(windows), expected_num_windows)


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions."""
    
    def test_load_agricultural_data(self):
        """Test convenience function for loading agricultural data."""
        # This would normally test with real tickers, but we'll test the structure
        test_tickers = ['AAPL']  # Use a reliable ticker for testing
        
        try:
            processed_data, quality_metrics = load_agricultural_data(
                test_tickers, 
                period='6m', 
                min_observations=50
            )
            
            # Check return types
            self.assertIsInstance(processed_data, dict)
            self.assertIsInstance(quality_metrics, dict)
            
            # If data was loaded successfully
            if processed_data:
                for ticker, data in processed_data.items():
                    self.assertIsInstance(data, pd.DataFrame)
                    self.assertIn('returns', data.columns)
                    
        except Exception as e:
            # Skip test if network/data issues
            self.skipTest(f"Data loading failed: {e}")
    
    def test_create_rolling_analysis_data(self):
        """Test convenience function for creating rolling analysis data."""
        test_tickers = ['AAPL']  # Use a reliable ticker
        
        try:
            rolling_data = create_rolling_analysis_data(
                test_tickers,
                window_size=10,
                period='6m'
            )
            
            self.assertIsInstance(rolling_data, dict)
            
            if rolling_data:
                for ticker, windows in rolling_data.items():
                    self.assertIsInstance(windows, list)
                    if windows:
                        for window in windows:
                            self.assertIsInstance(window, pd.DataFrame)
                            
        except Exception as e:
            # Skip test if network/data issues
            self.skipTest(f"Rolling data creation failed: {e}")


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.handler = AgriculturalDataHandler(cache_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_invalid_ticker_handling(self):
        """Test handling of invalid tickers."""
        invalid_tickers = ['INVALID123', 'NOTREAL456']
        
        self.handler.set_date_range(period='1y')
        data = self.handler.load_agricultural_universe_data(invalid_tickers)
        
        # Should handle gracefully
        self.assertIsInstance(data, dict)
        self.assertEqual(len(self.handler.failed_tickers), len(invalid_tickers))
    
    def test_empty_data_handling(self):
        """Test handling of empty datasets."""
        empty_data = {}
        
        quality_metrics = self.handler.validate_data_quality(empty_data)
        processed_data = self.handler.process_daily_returns(empty_data)
        
        self.assertEqual(len(quality_metrics), 0)
        self.assertEqual(len(processed_data), 0)
    
    def test_malformed_data_handling(self):
        """Test handling of malformed data."""
        # Create malformed data (missing required columns)
        dates = pd.date_range('2023-01-01', periods=5)
        malformed_data = {
            'MALFORMED': pd.DataFrame({
                'random_column': [1, 2, 3, 4, 5]
            }, index=dates)
        }
        
        quality_metrics = self.handler.validate_data_quality(malformed_data)
        
        # Should detect missing price columns
        self.assertFalse(quality_metrics['MALFORMED'].validation_passed)
        self.assertIn('No price data columns found', ' '.join(quality_metrics['MALFORMED'].issues))


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)