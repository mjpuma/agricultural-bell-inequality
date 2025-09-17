#!/usr/bin/env python3
"""
Agricultural Data Handling System

Comprehensive data handling system for 60+ agricultural companies with robust
validation, error handling, and rolling window analysis capabilities.

Following the design from agricultural-cross-sector-analysis spec.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import warnings
from dataclasses import dataclass
import logging
from pathlib import Path
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DataQualityMetrics:
    """Data quality metrics for validation."""
    ticker: str
    total_observations: int
    missing_values: int
    data_gaps: int
    start_date: str
    end_date: str
    data_quality_score: float
    validation_passed: bool
    issues: List[str]

@dataclass
class RollingWindowConfig:
    """Configuration for rolling window analysis."""
    window_size: int
    min_periods: int
    step_size: int = 1
    endpoints_range: Tuple[int, int] = None  # (start_T, end_T)

class AgriculturalDataHandler:
    """
    Comprehensive data handling system for agricultural cross-sector analysis.
    
    Provides robust data loading, validation, and processing for 60+ agricultural
    companies with proper error handling and data quality assurance.
    """
    
    def __init__(self, 
                 min_observations: int = 100,
                 data_frequency: str = 'daily',
                 cache_dir: str = 'data_cache',
                 validation_strict: bool = True):
        """
        Initialize agricultural data handler.
        
        Parameters:
        -----------
        min_observations : int
            Minimum required observations per asset (default: 100)
        data_frequency : str
            Data frequency ('daily', 'weekly', 'monthly')
        cache_dir : str
            Directory for caching downloaded data
        validation_strict : bool
            Whether to apply strict validation rules
        """
        self.min_observations = min_observations
        self.data_frequency = data_frequency
        self.cache_dir = Path(cache_dir)
        self.validation_strict = validation_strict
        
        # Create cache directory
        self.cache_dir.mkdir(exist_ok=True)
        
        # Data storage
        self.raw_data: Dict[str, pd.DataFrame] = {}
        self.processed_data: Dict[str, pd.DataFrame] = {}
        self.data_quality_metrics: Dict[str, DataQualityMetrics] = {}
        self.failed_tickers: List[str] = []
        
        # Date range handling
        self.start_date: Optional[str] = None
        self.end_date: Optional[str] = None
        self.date_range_days: Optional[int] = None
        
        logger.info(f"Agricultural Data Handler initialized")
        logger.info(f"  Min observations: {min_observations}")
        logger.info(f"  Data frequency: {data_frequency}")
        logger.info(f"  Cache directory: {cache_dir}")
    
    def set_date_range(self, 
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None,
                      period: Optional[str] = None) -> None:
        """
        Set date range for data loading.
        
        Parameters:
        -----------
        start_date : str, optional
            Start date in 'YYYY-MM-DD' format
        end_date : str, optional
            End date in 'YYYY-MM-DD' format
        period : str, optional
            Period string ('1y', '2y', '5y', etc.) - alternative to start/end dates
        """
        if period:
            # Convert period to date range
            end_date = datetime.now().strftime('%Y-%m-%d')
            
            period_days = {
                '1y': 365, '2y': 730, '3y': 1095, '5y': 1825, '10y': 3650,
                '6m': 180, '1m': 30, '3m': 90
            }
            
            if period in period_days:
                days = period_days[period]
                start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            else:
                logger.warning(f"Unknown period '{period}', using 2 years")
                start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
        
        self.start_date = start_date
        self.end_date = end_date
        
        if start_date and end_date:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            self.date_range_days = (end_dt - start_dt).days
        
        logger.info(f"Date range set: {start_date} to {end_date}")
    
    def load_agricultural_universe_data(self, 
                                      tickers: List[str],
                                      use_cache: bool = True,
                                      max_retries: int = 3) -> Dict[str, pd.DataFrame]:
        """
        Load data for the complete agricultural universe with robust error handling.
        
        Parameters:
        -----------
        tickers : List[str]
            List of ticker symbols to download
        use_cache : bool
            Whether to use cached data if available
        max_retries : int
            Maximum number of retry attempts for failed downloads
            
        Returns:
        --------
        Dict[str, pd.DataFrame] : Successfully loaded data by ticker
        """
        logger.info(f"Loading data for {len(tickers)} agricultural companies")
        logger.info(f"Date range: {self.start_date} to {self.end_date}")
        
        successful_loads = {}
        failed_loads = []
        
        for i, ticker in enumerate(tickers, 1):
            logger.info(f"Loading {ticker} ({i}/{len(tickers)})")
            
            try:
                # Check cache first
                if use_cache:
                    cached_data = self._load_from_cache(ticker)
                    if cached_data is not None:
                        successful_loads[ticker] = cached_data
                        logger.info(f"  ✅ Loaded from cache: {len(cached_data)} observations")
                        continue
                
                # Download fresh data
                data = self._download_ticker_data(ticker, max_retries)
                
                if data is not None and len(data) > 0:
                    # Cache the data
                    if use_cache:
                        self._save_to_cache(ticker, data)
                    
                    successful_loads[ticker] = data
                    logger.info(f"  ✅ Downloaded: {len(data)} observations")
                else:
                    failed_loads.append(ticker)
                    logger.warning(f"  ❌ Failed to load {ticker}")
                    
            except Exception as e:
                failed_loads.append(ticker)
                logger.error(f"  ❌ Error loading {ticker}: {str(e)}")
        
        # Store results
        self.raw_data = successful_loads
        self.failed_tickers = failed_loads
        
        logger.info(f"Data loading complete:")
        logger.info(f"  ✅ Successful: {len(successful_loads)} tickers")
        logger.info(f"  ❌ Failed: {len(failed_loads)} tickers")
        
        if failed_loads:
            logger.warning(f"Failed tickers: {failed_loads}")
        
        return successful_loads
    
    def validate_data_quality(self, 
                            data: Optional[Dict[str, pd.DataFrame]] = None) -> Dict[str, DataQualityMetrics]:
        """
        Comprehensive data quality validation with minimum observation requirements.
        
        Parameters:
        -----------
        data : Dict[str, pd.DataFrame], optional
            Data to validate. If None, uses self.raw_data
            
        Returns:
        --------
        Dict[str, DataQualityMetrics] : Quality metrics for each ticker
        """
        data_to_validate = data or self.raw_data
        
        logger.info(f"Validating data quality for {len(data_to_validate)} tickers")
        logger.info(f"Minimum observations required: {self.min_observations}")
        
        quality_metrics = {}
        passed_validation = []
        failed_validation = []
        
        for ticker, df in data_to_validate.items():
            metrics = self._calculate_quality_metrics(ticker, df)
            quality_metrics[ticker] = metrics
            
            if metrics.validation_passed:
                passed_validation.append(ticker)
            else:
                failed_validation.append(ticker)
                logger.warning(f"  ❌ {ticker}: {', '.join(metrics.issues)}")
        
        # Store quality metrics
        self.data_quality_metrics = quality_metrics
        
        logger.info(f"Data quality validation complete:")
        logger.info(f"  ✅ Passed: {len(passed_validation)} tickers")
        logger.info(f"  ❌ Failed: {len(failed_validation)} tickers")
        
        if failed_validation and self.validation_strict:
            logger.warning(f"Strict validation enabled - removing failed tickers")
            # Remove failed tickers from raw_data
            for ticker in failed_validation:
                if ticker in self.raw_data:
                    del self.raw_data[ticker]
        
        return quality_metrics
    
    def process_daily_returns(self, 
                            data: Optional[Dict[str, pd.DataFrame]] = None,
                            price_column: str = 'Adj Close') -> Dict[str, pd.DataFrame]:
        """
        Process daily frequency data with proper date range handling.
        
        Parameters:
        -----------
        data : Dict[str, pd.DataFrame], optional
            Raw price data. If None, uses self.raw_data
        price_column : str
            Column name for price data
            
        Returns:
        --------
        Dict[str, pd.DataFrame] : Processed returns data
        """
        data_to_process = data or self.raw_data
        
        logger.info(f"Processing daily returns for {len(data_to_process)} tickers")
        
        processed_returns = {}
        
        for ticker, df in data_to_process.items():
            try:
                # Find the correct price column
                price_columns_to_try = [price_column, 'Close', 'Adj Close', 'close', 'adj_close']
                price_col_found = None
                
                for col in price_columns_to_try:
                    if col in df.columns:
                        price_col_found = col
                        break
                
                if price_col_found:
                    # Calculate daily returns: Ri,t = (Pi,t - Pi,t-1) / Pi,t-1
                    prices = df[price_col_found].dropna()
                    returns = prices.pct_change().dropna()
                    
                    # Create DataFrame with proper index
                    returns_df = pd.DataFrame({
                        'returns': returns,
                        'price': prices[returns.index]
                    })
                    
                    processed_returns[ticker] = returns_df
                    logger.debug(f"  ✅ {ticker}: {len(returns_df)} return observations")
                else:
                    available_cols = list(df.columns)
                    logger.warning(f"  ❌ {ticker}: No price column found. Available: {available_cols}")
                    
            except Exception as e:
                logger.error(f"  ❌ Error processing {ticker}: {str(e)}")
        
        # Store processed data
        self.processed_data = processed_returns
        
        logger.info(f"Daily returns processing complete: {len(processed_returns)} tickers")
        return processed_returns
    
    def create_rolling_windows(self, 
                             window_config: RollingWindowConfig,
                             data: Optional[Dict[str, pd.DataFrame]] = None) -> Dict[str, List[pd.DataFrame]]:
        """
        Create rolling window analysis with endpoints T ranging from w to N.
        
        Parameters:
        -----------
        window_config : RollingWindowConfig
            Configuration for rolling windows
        data : Dict[str, pd.DataFrame], optional
            Data to create windows from. If None, uses self.processed_data
            
        Returns:
        --------
        Dict[str, List[pd.DataFrame]] : Rolling windows for each ticker
        """
        data_to_window = data or self.processed_data
        
        logger.info(f"Creating rolling windows for {len(data_to_window)} tickers")
        logger.info(f"Window size: {window_config.window_size}")
        logger.info(f"Min periods: {window_config.min_periods}")
        logger.info(f"Step size: {window_config.step_size}")
        
        rolling_windows = {}
        
        for ticker, df in data_to_window.items():
            try:
                windows = self._create_ticker_windows(ticker, df, window_config)
                rolling_windows[ticker] = windows
                logger.debug(f"  ✅ {ticker}: {len(windows)} windows created")
                
            except Exception as e:
                logger.error(f"  ❌ Error creating windows for {ticker}: {str(e)}")
        
        logger.info(f"Rolling windows created: {sum(len(windows) for windows in rolling_windows.values())} total windows")
        return rolling_windows
    
    def handle_missing_data(self, 
                          data: Dict[str, pd.DataFrame],
                          method: str = 'forward_fill',
                          max_consecutive_missing: int = 5) -> Dict[str, pd.DataFrame]:
        """
        Handle missing data and data quality issues.
        
        Parameters:
        -----------
        data : Dict[str, pd.DataFrame]
            Data with potential missing values
        method : str
            Method for handling missing data ('forward_fill', 'interpolate', 'drop')
        max_consecutive_missing : int
            Maximum consecutive missing values to fill
            
        Returns:
        --------
        Dict[str, pd.DataFrame] : Data with missing values handled
        """
        logger.info(f"Handling missing data for {len(data)} tickers")
        logger.info(f"Method: {method}")
        logger.info(f"Max consecutive missing: {max_consecutive_missing}")
        
        cleaned_data = {}
        
        for ticker, df in data.items():
            try:
                cleaned_df = self._clean_ticker_data(df, method, max_consecutive_missing)
                cleaned_data[ticker] = cleaned_df
                
                missing_before = df.isnull().sum().sum()
                missing_after = cleaned_df.isnull().sum().sum()
                logger.debug(f"  ✅ {ticker}: {missing_before} → {missing_after} missing values")
                
            except Exception as e:
                logger.error(f"  ❌ Error cleaning {ticker}: {str(e)}")
        
        logger.info(f"Missing data handling complete: {len(cleaned_data)} tickers")
        return cleaned_data
    
    def get_data_summary(self) -> Dict[str, any]:
        """
        Get comprehensive summary of loaded and processed data.
        
        Returns:
        --------
        Dict : Summary statistics and information
        """
        summary = {
            'total_tickers_requested': len(self.raw_data) + len(self.failed_tickers),
            'successful_loads': len(self.raw_data),
            'failed_loads': len(self.failed_tickers),
            'failed_tickers': self.failed_tickers,
            'date_range': {
                'start_date': self.start_date,
                'end_date': self.end_date,
                'days': self.date_range_days
            },
            'data_quality': {
                'passed_validation': len([m for m in self.data_quality_metrics.values() if m.validation_passed]),
                'failed_validation': len([m for m in self.data_quality_metrics.values() if not m.validation_passed]),
                'average_observations': np.mean([m.total_observations for m in self.data_quality_metrics.values()]) if self.data_quality_metrics else 0
            },
            'processed_data': {
                'tickers_processed': len(self.processed_data),
                'total_return_observations': sum(len(df) for df in self.processed_data.values())
            }
        }
        
        return summary
    
    def export_data_quality_report(self, filepath: str = 'data_quality_report.json') -> str:
        """
        Export comprehensive data quality report.
        
        Parameters:
        -----------
        filepath : str
            Path to save the report
            
        Returns:
        --------
        str : Path to saved report
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': self.get_data_summary(),
            'quality_metrics': {
                ticker: {
                    'total_observations': metrics.total_observations,
                    'missing_values': metrics.missing_values,
                    'data_gaps': metrics.data_gaps,
                    'start_date': metrics.start_date,
                    'end_date': metrics.end_date,
                    'data_quality_score': metrics.data_quality_score,
                    'validation_passed': metrics.validation_passed,
                    'issues': metrics.issues
                }
                for ticker, metrics in self.data_quality_metrics.items()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Data quality report exported to: {filepath}")
        return filepath
    
    # Private helper methods
    
    def _download_ticker_data(self, ticker: str, max_retries: int) -> Optional[pd.DataFrame]:
        """Download data for a single ticker with retry logic."""
        
        for attempt in range(max_retries):
            try:
                # Use yfinance to download data
                yf_ticker = yf.Ticker(ticker)
                
                # Download with specified date range
                data = yf_ticker.history(
                    start=self.start_date,
                    end=self.end_date,
                    interval='1d',
                    auto_adjust=True,
                    prepost=False
                )
                
                if data is not None and len(data) > 0:
                    # Clean column names
                    data.columns = [col.replace(' ', '_') for col in data.columns]
                    return data
                else:
                    logger.warning(f"    Attempt {attempt + 1}: No data returned for {ticker}")
                    
            except Exception as e:
                logger.warning(f"    Attempt {attempt + 1} failed for {ticker}: {str(e)}")
                
                if attempt < max_retries - 1:
                    # Wait before retry
                    import time
                    time.sleep(1)
        
        return None
    
    def _load_from_cache(self, ticker: str) -> Optional[pd.DataFrame]:
        """Load data from cache if available and valid."""
        
        cache_file = self.cache_dir / f"{ticker}_{self.start_date}_{self.end_date}.csv"
        
        if cache_file.exists():
            try:
                data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                logger.debug(f"    Loaded from cache: {cache_file}")
                return data
            except Exception as e:
                logger.warning(f"    Cache read failed for {ticker}: {str(e)}")
        
        return None
    
    def _save_to_cache(self, ticker: str, data: pd.DataFrame) -> None:
        """Save data to cache."""
        
        cache_file = self.cache_dir / f"{ticker}_{self.start_date}_{self.end_date}.csv"
        
        try:
            data.to_csv(cache_file)
            logger.debug(f"    Saved to cache: {cache_file}")
        except Exception as e:
            logger.warning(f"    Cache save failed for {ticker}: {str(e)}")
    
    def _calculate_quality_metrics(self, ticker: str, df: pd.DataFrame) -> DataQualityMetrics:
        """Calculate comprehensive data quality metrics for a ticker."""
        
        issues = []
        
        # Basic metrics
        total_obs = len(df)
        missing_values = df.isnull().sum().sum()
        
        # Date range
        if len(df) > 0 and hasattr(df.index, 'min') and hasattr(df.index.min(), 'strftime'):
            start_date = df.index.min().strftime('%Y-%m-%d')
            end_date = df.index.max().strftime('%Y-%m-%d')
        else:
            start_date = 'N/A'
            end_date = 'N/A'
        
        # Data gaps (missing dates)
        if len(df) > 1:
            expected_days = (df.index.max() - df.index.min()).days
            actual_days = len(df)
            data_gaps = max(0, expected_days - actual_days)
        else:
            data_gaps = 0
        
        # Quality checks
        if total_obs < self.min_observations:
            issues.append(f"Insufficient data: {total_obs} < {self.min_observations}")
        
        if missing_values > total_obs * 0.1:  # More than 10% missing
            issues.append(f"High missing values: {missing_values}/{total_obs}")
        
        if data_gaps > total_obs * 0.2:  # More than 20% gaps
            issues.append(f"High data gaps: {data_gaps}")
        
        # Check for price data
        price_columns = ['Adj Close', 'Close', 'Adj_Close', 'close', 'adj_close']
        has_price_data = any(col in df.columns for col in price_columns)
        if not has_price_data:
            issues.append("No price data columns found")
        
        # Calculate quality score (0-1)
        quality_score = 1.0
        quality_score -= min(0.5, missing_values / total_obs)  # Penalize missing values
        quality_score -= min(0.3, data_gaps / max(1, total_obs))  # Penalize gaps
        if total_obs < self.min_observations:
            quality_score -= 0.4  # Major penalty for insufficient data
        
        quality_score = max(0.0, quality_score)
        
        # Validation passed if no critical issues
        validation_passed = (
            total_obs >= self.min_observations and
            missing_values <= total_obs * 0.1 and
            has_price_data
        )
        
        return DataQualityMetrics(
            ticker=ticker,
            total_observations=total_obs,
            missing_values=missing_values,
            data_gaps=data_gaps,
            start_date=start_date,
            end_date=end_date,
            data_quality_score=quality_score,
            validation_passed=validation_passed,
            issues=issues
        )
    
    def _create_ticker_windows(self, 
                             ticker: str, 
                             df: pd.DataFrame, 
                             config: RollingWindowConfig) -> List[pd.DataFrame]:
        """Create rolling windows for a single ticker."""
        
        windows = []
        n_observations = len(df)
        
        # Determine endpoint range
        if config.endpoints_range:
            start_t, end_t = config.endpoints_range
            start_t = max(config.window_size, start_t)
            end_t = min(n_observations, end_t)
        else:
            start_t = config.window_size
            end_t = n_observations
        
        # Create windows with endpoints T ranging from w to N
        for t in range(start_t, end_t + 1, config.step_size):
            window_start = t - config.window_size
            window_end = t
            
            if window_start >= 0 and window_end <= n_observations:
                window_data = df.iloc[window_start:window_end].copy()
                
                # Only include windows with sufficient data
                if len(window_data) >= config.min_periods:
                    windows.append(window_data)
        
        return windows
    
    def _clean_ticker_data(self, 
                         df: pd.DataFrame, 
                         method: str, 
                         max_consecutive_missing: int) -> pd.DataFrame:
        """Clean data for a single ticker."""
        
        cleaned_df = df.copy()
        
        if method == 'forward_fill':
            # Forward fill with limit
            cleaned_df = cleaned_df.ffill(limit=max_consecutive_missing)
        elif method == 'interpolate':
            # Linear interpolation with limit
            cleaned_df = cleaned_df.interpolate(method='linear', limit=max_consecutive_missing)
        elif method == 'drop':
            # Drop rows with any missing values
            cleaned_df = cleaned_df.dropna()
        
        return cleaned_df


# Convenience functions for easy usage

def load_agricultural_data(tickers: List[str], 
                         period: str = '2y',
                         min_observations: int = 100) -> Tuple[Dict[str, pd.DataFrame], Dict[str, DataQualityMetrics]]:
    """
    Convenience function to load and validate agricultural data.
    
    Parameters:
    -----------
    tickers : List[str]
        List of ticker symbols
    period : str
        Time period ('1y', '2y', etc.)
    min_observations : int
        Minimum observations required
        
    Returns:
    --------
    Tuple[Dict, Dict] : (processed_data, quality_metrics)
    """
    handler = AgriculturalDataHandler(min_observations=min_observations)
    handler.set_date_range(period=period)
    
    # Load raw data
    raw_data = handler.load_agricultural_universe_data(tickers)
    
    # Validate quality
    quality_metrics = handler.validate_data_quality()
    
    # Process returns
    processed_data = handler.process_daily_returns()
    
    return processed_data, quality_metrics

def create_rolling_analysis_data(tickers: List[str],
                               window_size: int = 20,
                               period: str = '2y') -> Dict[str, List[pd.DataFrame]]:
    """
    Convenience function to create rolling window data for analysis.
    
    Parameters:
    -----------
    tickers : List[str]
        List of ticker symbols
    window_size : int
        Rolling window size
    period : str
        Time period
        
    Returns:
    --------
    Dict[str, List[pd.DataFrame]] : Rolling windows by ticker
    """
    handler = AgriculturalDataHandler()
    handler.set_date_range(period=period)
    
    # Load and process data
    raw_data = handler.load_agricultural_universe_data(tickers)
    handler.validate_data_quality()
    processed_data = handler.process_daily_returns()
    
    # Create rolling windows
    window_config = RollingWindowConfig(
        window_size=window_size,
        min_periods=window_size // 2
    )
    
    rolling_windows = handler.create_rolling_windows(window_config)
    
    return rolling_windows


if __name__ == "__main__":
    # Example usage
    print("🌾 Agricultural Data Handler Test")
    
    # Test with a few tickers
    test_tickers = ['ADM', 'CF', 'DE', 'CORN', 'WEAT']
    
    handler = AgriculturalDataHandler(min_observations=50)
    handler.set_date_range(period='1y')
    
    # Load data
    data = handler.load_agricultural_universe_data(test_tickers)
    
    # Validate quality
    quality = handler.validate_data_quality()
    
    # Process returns
    returns = handler.process_daily_returns()
    
    # Create rolling windows
    window_config = RollingWindowConfig(window_size=20, min_periods=15)
    windows = handler.create_rolling_windows(window_config)
    
    # Print summary
    summary = handler.get_data_summary()
    print(f"\n📊 Data Summary:")
    print(f"  Successful loads: {summary['successful_loads']}")
    print(f"  Failed loads: {summary['failed_loads']}")
    print(f"  Processed tickers: {summary['processed_data']['tickers_processed']}")
    print(f"  Total windows: {sum(len(w) for w in windows.values())}")
    
    print("\n✅ Agricultural data handler test complete!")