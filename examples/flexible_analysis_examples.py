#!/usr/bin/env python3
"""
FLEXIBLE BELL INEQUALITY ANALYSIS EXAMPLES
==========================================
Demonstrates easy configuration for different assets, periods, and crisis analysis
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from bell_inequality_analyzer import BellInequalityAnalyzer
from preset_configurations import (
    analyze_asset_group, analyze_crisis_period, 
    compare_asset_groups, compare_crisis_periods,
    ASSET_GROUPS, CRISIS_PERIODS
)

def example_1_basic_flexibility():
    """Example 1: Basic flexible usage"""
    
    print("🎯 EXAMPLE 1: BASIC FLEXIBLE USAGE")
    print("=" * 40)
    
    # Tech stocks, 6 months (default)
    print("\n📊 Tech stocks, 6 months:")
    analyzer1 = BellInequalityAnalyzer()
    if analyzer1.load_data():
        analyzer1.run_s1_analysis()
    
    # Financial sector, 2 years
    print("\n🏦 Financial sector, 2 years:")
    analyzer2 = BellInequalityAnalyzer(
        assets=['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS'],
        period='2y'
    )
    if analyzer2.load_data():
        analyzer2.run_s1_analysis()
    
    # Commodities, 1 year
    print("\n🥇 Commodities, 1 year:")
    analyzer3 = BellInequalityAnalyzer(
        assets=['GLD', 'SLV', 'USO', 'DBA', 'CORN'],
        period='1y'
    )
    if analyzer3.load_data():
        analyzer3.run_s1_analysis()

def example_2_crisis_periods():
    """Example 2: Crisis period analysis"""
    
    print("\n📉 EXAMPLE 2: CRISIS PERIOD ANALYSIS")
    print("=" * 40)
    
    # COVID crash analysis
    print("\n🦠 COVID-19 crash period:")
    covid_analyzer = BellInequalityAnalyzer(
        assets=['AAPL', 'MSFT', 'SPY', 'VIX'],
        start_date='2020-02-01',
        end_date='2020-04-30'
    )
    if covid_analyzer.load_data():
        covid_analyzer.run_s1_analysis()
    
    # Financial crisis analysis (if data available)
    print("\n🏦 Financial crisis period:")
    financial_crisis_analyzer = BellInequalityAnalyzer(
        assets=['JPM', 'BAC', 'AIG', 'C'],
        start_date='2008-09-01',
        end_date='2009-03-31'
    )
    if financial_crisis_analyzer.load_data():
        financial_crisis_analyzer.run_s1_analysis()
    
    # Inflation bear market
    print("\n📈 Inflation bear market:")
    inflation_analyzer = BellInequalityAnalyzer(
        assets=['AAPL', 'MSFT', 'GOOGL', 'NVDA'],
        start_date='2022-01-01',
        end_date='2022-12-31'
    )
    if inflation_analyzer.load_data():
        inflation_analyzer.run_s1_analysis()

def example_3_preset_configurations():
    """Example 3: Using preset configurations"""
    
    print("\n🎯 EXAMPLE 3: PRESET CONFIGURATIONS")
    print("=" * 40)
    
    # Show available asset groups
    print("\n📋 Available asset groups:")
    for group_name in list(ASSET_GROUPS.keys())[:10]:  # Show first 10
        assets = ASSET_GROUPS[group_name]
        print(f"   {group_name}: {assets}")
    
    # Analyze tech giants using preset
    print("\n🔬 Analyzing tech giants (preset):")
    tech_analyzer = analyze_asset_group('tech_giants', period='1y')
    
    # Analyze big banks using preset
    print("\n🏦 Analyzing big banks (preset):")
    bank_analyzer = analyze_asset_group('big_banks', period='1y')
    
    # Show available crisis periods
    print("\n📋 Available crisis periods:")
    for crisis_name, config in list(CRISIS_PERIODS.items())[:5]:  # Show first 5
        print(f"   {crisis_name}: {config['description']}")
    
    # Analyze COVID crash using preset
    print("\n🦠 Analyzing COVID crash (preset):")
    covid_analyzer = analyze_crisis_period('covid_crash')

def example_4_comparative_analysis():
    """Example 4: Comparative analysis across sectors and periods"""
    
    print("\n📊 EXAMPLE 4: COMPARATIVE ANALYSIS")
    print("=" * 40)
    
    # Compare different sectors
    print("\n🔍 Comparing sectors:")
    sector_results = compare_asset_groups([
        'tech_giants', 
        'big_banks', 
        'big_pharma', 
        'oil_majors'
    ], period='1y')
    
    # Compare crisis periods
    print("\n📉 Comparing crisis periods:")
    crisis_results = compare_crisis_periods([
        'covid_crash',
        'inflation_bear_market'
    ], assets=['AAPL', 'MSFT', 'GOOGL', 'NVDA'])

def example_5_custom_analysis():
    """Example 5: Completely custom analysis"""
    
    print("\n🛠️  EXAMPLE 5: CUSTOM ANALYSIS")
    print("=" * 35)
    
    # Custom asset mix: Tech + Finance + Commodities
    print("\n🎯 Custom asset mix analysis:")
    custom_analyzer = BellInequalityAnalyzer(
        assets=['AAPL', 'JPM', 'GLD', 'NVDA', 'BAC', 'SLV'],
        period='2y'
    )
    if custom_analyzer.load_data():
        results = custom_analyzer.run_s1_analysis(
            window_size=25,  # Custom window size
            threshold_quantile=0.8  # Custom threshold
        )
        custom_analyzer.create_visualizations('custom_analysis.png')
    
    # Specific date range analysis
    print("\n📅 Specific date range analysis:")
    date_range_analyzer = BellInequalityAnalyzer(
        assets=['TSLA', 'NVDA', 'NFLX'],  # High volatility stocks
        start_date='2021-01-01',
        end_date='2021-12-31'
    )
    if date_range_analyzer.load_data():
        date_range_analyzer.run_s1_analysis()

def example_6_parameter_sensitivity():
    """Example 6: Parameter sensitivity analysis"""
    
    print("\n🔬 EXAMPLE 6: PARAMETER SENSITIVITY")
    print("=" * 40)
    
    # Test different window sizes
    print("\n📊 Testing different window sizes:")
    base_assets = ['AAPL', 'MSFT', 'GOOGL', 'NVDA']
    
    for window_size in [15, 20, 25, 30]:
        print(f"\n   Window size: {window_size}")
        analyzer = BellInequalityAnalyzer(assets=base_assets, period='1y')
        if analyzer.load_data():
            results = analyzer.run_s1_analysis(window_size=window_size)
            if results:
                violation_rate = results['summary']['overall_violation_rate']
                print(f"   Violation rate: {violation_rate:.2f}%")
    
    # Test different threshold quantiles
    print("\n📊 Testing different threshold quantiles:")
    
    for threshold in [0.7, 0.75, 0.8, 0.85]:
        print(f"\n   Threshold quantile: {threshold}")
        analyzer = BellInequalityAnalyzer(assets=base_assets, period='1y')
        if analyzer.load_data():
            results = analyzer.run_s1_analysis(threshold_quantile=threshold)
            if results:
                violation_rate = results['summary']['overall_violation_rate']
                print(f"   Violation rate: {violation_rate:.2f}%")

def show_all_presets():
    """Show all available presets for easy reference"""
    
    print("\n📋 ALL AVAILABLE PRESETS")
    print("=" * 30)
    
    print("\n🎯 ASSET GROUPS:")
    for category in ['tech', 'financial', 'healthcare', 'energy', 'consumer', 'industrial']:
        matching_groups = [name for name in ASSET_GROUPS.keys() if category in name.lower()]
        if matching_groups:
            print(f"\n   {category.upper()}:")
            for group_name in matching_groups:
                assets = ASSET_GROUPS[group_name]
                print(f"     {group_name}: {assets}")
    
    print("\n📉 CRISIS PERIODS:")
    for crisis_name, config in CRISIS_PERIODS.items():
        print(f"   {crisis_name}: {config['start_date']} to {config['end_date']}")
        print(f"     {config['description']}")

def main():
    """Run all examples"""
    
    print("🚀 FLEXIBLE BELL INEQUALITY ANALYSIS EXAMPLES")
    print("=" * 50)
    
    # Show all available presets first
    show_all_presets()
    
    # Run examples (comment out as needed for testing)
    example_1_basic_flexibility()
    example_2_crisis_periods()
    example_3_preset_configurations()
    example_4_comparative_analysis()
    example_5_custom_analysis()
    example_6_parameter_sensitivity()
    
    print("\n✅ ALL EXAMPLES COMPLETE!")
    print("🎯 The Bell inequality analyzer is highly flexible for:")
    print("   - Any asset class (stocks, ETFs, commodities)")
    print("   - Any time period (preset or custom dates)")
    print("   - Crisis period analysis")
    print("   - Comparative studies across sectors")
    print("   - Parameter sensitivity testing")

if __name__ == "__main__":
    main()