#!/usr/bin/env python3
"""
Agricultural Data Handling System Demo

Demonstrates the comprehensive data handling capabilities for agricultural
cross-sector analysis with 60+ companies.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from agricultural_data_handler import (
    AgriculturalDataHandler, 
    RollingWindowConfig,
    load_agricultural_data,
    create_rolling_analysis_data
)
from agricultural_universe_manager import AgriculturalUniverseManager

def demo_basic_data_loading():
    """Demonstrate basic data loading and validation."""
    print("🌾 AGRICULTURAL DATA HANDLING DEMO")
    print("=" * 50)
    
    # Initialize universe manager to get agricultural companies
    universe_manager = AgriculturalUniverseManager()
    
    # Get a sample of agricultural companies from different tiers
    agricultural_companies = universe_manager.classify_by_tier(0)[:5]  # Agricultural companies
    tier1_companies = universe_manager.classify_by_tier(1)[:3]  # Energy/Transport/Chemicals
    
    sample_tickers = agricultural_companies + tier1_companies
    
    print(f"📊 Sample tickers for demo: {sample_tickers}")
    
    # Initialize data handler
    handler = AgriculturalDataHandler(
        min_observations=100,
        data_frequency='daily',
        validation_strict=True
    )
    
    # Set date range
    handler.set_date_range(period='1y')
    
    print(f"\n📅 Date range: {handler.start_date} to {handler.end_date}")
    
    # Load data
    print(f"\n📥 Loading data for {len(sample_tickers)} companies...")
    raw_data = handler.load_agricultural_universe_data(sample_tickers)
    
    # Validate data quality
    print(f"\n🔍 Validating data quality...")
    quality_metrics = handler.validate_data_quality()
    
    # Process daily returns
    print(f"\n📈 Processing daily returns...")
    processed_data = handler.process_daily_returns()
    
    # Get summary
    summary = handler.get_data_summary()
    
    print(f"\n📋 DATA LOADING SUMMARY:")
    print(f"  ✅ Successful loads: {summary['successful_loads']}")
    print(f"  ❌ Failed loads: {summary['failed_loads']}")
    print(f"  📊 Processed tickers: {summary['processed_data']['tickers_processed']}")
    print(f"  📈 Total return observations: {summary['processed_data']['total_return_observations']:,}")
    
    if summary['failed_loads'] > 0:
        print(f"  ⚠️  Failed tickers: {summary['failed_tickers']}")
    
    return handler, processed_data, quality_metrics

def demo_rolling_window_analysis(handler, processed_data):
    """Demonstrate rolling window creation for Bell inequality analysis."""
    print(f"\n🔄 ROLLING WINDOW ANALYSIS DEMO")
    print("=" * 40)
    
    # Configure rolling windows for Bell inequality analysis
    window_config = RollingWindowConfig(
        window_size=20,  # Standard window size for Bell analysis
        min_periods=15,  # Minimum periods required
        step_size=1      # Daily step
    )
    
    print(f"📊 Window configuration:")
    print(f"  Window size: {window_config.window_size}")
    print(f"  Min periods: {window_config.min_periods}")
    print(f"  Step size: {window_config.step_size}")
    
    # Create rolling windows
    rolling_windows = handler.create_rolling_windows(window_config, processed_data)
    
    # Analyze windows
    total_windows = sum(len(windows) for windows in rolling_windows.values())
    
    print(f"\n📈 Rolling windows created:")
    print(f"  Total windows: {total_windows:,}")
    print(f"  Tickers with windows: {len(rolling_windows)}")
    
    # Show details for each ticker
    for ticker, windows in rolling_windows.items():
        if windows:
            avg_window_size = sum(len(w) for w in windows) / len(windows)
            print(f"  {ticker}: {len(windows)} windows, avg size: {avg_window_size:.1f}")
    
    return rolling_windows

def demo_data_quality_analysis(quality_metrics):
    """Demonstrate data quality analysis and reporting."""
    print(f"\n🔍 DATA QUALITY ANALYSIS DEMO")
    print("=" * 40)
    
    if not quality_metrics:
        print("No quality metrics available")
        return
    
    # Analyze quality metrics
    passed_validation = [m for m in quality_metrics.values() if m.validation_passed]
    failed_validation = [m for m in quality_metrics.values() if not m.validation_passed]
    
    print(f"📊 Quality Summary:")
    print(f"  ✅ Passed validation: {len(passed_validation)}")
    print(f"  ❌ Failed validation: {len(failed_validation)}")
    
    if passed_validation:
        avg_quality_score = sum(m.data_quality_score for m in passed_validation) / len(passed_validation)
        avg_observations = sum(m.total_observations for m in passed_validation) / len(passed_validation)
        
        print(f"  📈 Average quality score: {avg_quality_score:.3f}")
        print(f"  📊 Average observations: {avg_observations:.0f}")
    
    # Show detailed metrics for each ticker
    print(f"\n📋 Detailed Quality Metrics:")
    for ticker, metrics in quality_metrics.items():
        status = "✅" if metrics.validation_passed else "❌"
        print(f"  {status} {ticker}:")
        print(f"    Observations: {metrics.total_observations}")
        print(f"    Quality score: {metrics.data_quality_score:.3f}")
        print(f"    Date range: {metrics.start_date} to {metrics.end_date}")
        
        if metrics.issues:
            print(f"    Issues: {', '.join(metrics.issues)}")

def demo_missing_data_handling(handler):
    """Demonstrate missing data handling capabilities."""
    print(f"\n🔧 MISSING DATA HANDLING DEMO")
    print("=" * 40)
    
    # Create sample data with missing values for demonstration
    import pandas as pd
    import numpy as np
    
    dates = pd.date_range('2023-01-01', periods=100)
    sample_data = {
        'DEMO_TICKER': pd.DataFrame({
            'Adj Close': 100 + np.cumsum(np.random.normal(0, 1, 100)),
            'Volume': np.random.randint(100000, 1000000, 100)
        }, index=dates)
    }
    
    # Introduce missing values
    sample_data['DEMO_TICKER'].iloc[10:15] = np.nan  # 5 consecutive missing
    sample_data['DEMO_TICKER'].iloc[50] = np.nan     # Single missing
    sample_data['DEMO_TICKER'].iloc[75:77] = np.nan  # 2 consecutive missing
    
    original_missing = sample_data['DEMO_TICKER'].isnull().sum().sum()
    print(f"📊 Original missing values: {original_missing}")
    
    # Test different missing data handling methods
    methods = ['forward_fill', 'interpolate', 'drop']
    
    for method in methods:
        cleaned_data = handler.handle_missing_data(
            sample_data.copy(), 
            method=method,
            max_consecutive_missing=10
        )
        
        remaining_missing = cleaned_data['DEMO_TICKER'].isnull().sum().sum()
        remaining_obs = len(cleaned_data['DEMO_TICKER'].dropna())
        
        print(f"  {method}: {remaining_missing} missing, {remaining_obs} observations")

def demo_convenience_functions():
    """Demonstrate convenience functions for easy usage."""
    print(f"\n🚀 CONVENIENCE FUNCTIONS DEMO")
    print("=" * 40)
    
    # Sample agricultural tickers
    sample_tickers = ['ADM', 'CF', 'DE']  # Reliable agricultural companies
    
    print(f"📊 Testing with tickers: {sample_tickers}")
    
    try:
        # Test load_agricultural_data convenience function
        print(f"\n📥 Testing load_agricultural_data()...")
        processed_data, quality_metrics = load_agricultural_data(
            sample_tickers, 
            period='6m', 
            min_observations=50
        )
        
        print(f"  ✅ Loaded {len(processed_data)} tickers")
        print(f"  📊 Quality metrics for {len(quality_metrics)} tickers")
        
        # Test create_rolling_analysis_data convenience function
        print(f"\n🔄 Testing create_rolling_analysis_data()...")
        rolling_data = create_rolling_analysis_data(
            sample_tickers,
            window_size=15,
            period='6m'
        )
        
        total_windows = sum(len(windows) for windows in rolling_data.values())
        print(f"  ✅ Created {total_windows} rolling windows")
        print(f"  📊 Windows for {len(rolling_data)} tickers")
        
    except Exception as e:
        print(f"  ⚠️  Convenience function test failed: {e}")
        print(f"     This may be due to network issues or data availability")

def main():
    """Run the complete agricultural data handling demo."""
    print("🌾 AGRICULTURAL DATA HANDLING SYSTEM")
    print("Comprehensive demo of data loading, validation, and processing")
    print("=" * 60)
    
    try:
        # Demo 1: Basic data loading and validation
        handler, processed_data, quality_metrics = demo_basic_data_loading()
        
        # Demo 2: Rolling window analysis
        if processed_data:
            rolling_windows = demo_rolling_window_analysis(handler, processed_data)
        
        # Demo 3: Data quality analysis
        demo_data_quality_analysis(quality_metrics)
        
        # Demo 4: Missing data handling
        demo_missing_data_handling(handler)
        
        # Demo 5: Convenience functions
        demo_convenience_functions()
        
        print(f"\n✅ AGRICULTURAL DATA HANDLING DEMO COMPLETE!")
        print(f"📊 The system successfully demonstrates:")
        print(f"   • Robust data loading for 60+ agricultural companies")
        print(f"   • Data validation with minimum 100 observations requirement")
        print(f"   • Error handling for missing data and data quality issues")
        print(f"   • Daily frequency data processing with proper date range handling")
        print(f"   • Rolling window analysis with endpoints T ranging from w to N")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print(f"This may be due to network connectivity or data source issues")

if __name__ == "__main__":
    main()