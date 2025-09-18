#!/usr/bin/env python3
"""
PRESET CONFIGURATIONS FOR BELL INEQUALITY ANALYSIS
==================================================
Pre-defined asset groups and crisis periods for easy analysis
"""

from datetime import datetime
from .bell_inequality_analyzer import BellInequalityAnalyzer

# =================== ASSET GROUP PRESETS ===================

ASSET_GROUPS = {
    # Technology Stocks
    'tech_giants': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'META'],
    'tech_extended': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'META', 'AMD', 'INTC', 'ORCL', 'CRM'],
    'semiconductor': ['NVDA', 'AMD', 'INTC', 'TSM', 'AVGO', 'QCOM'],
    'software': ['MSFT', 'GOOGL', 'ORCL', 'CRM', 'ADBE', 'NOW'],
    
    # Financial Sector
    'big_banks': ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS'],
    'regional_banks': ['PNC', 'USB', 'TFC', 'COF', 'BK', 'STT'],
    'insurance': ['BRK.B', 'AIG', 'PGR', 'TRV', 'ALL', 'MET'],
    'fintech': ['V', 'MA', 'PYPL', 'SQ', 'COIN', 'SOFI'],
    
    # Healthcare & Pharma
    'big_pharma': ['JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'LLY'],
    'biotech': ['GILD', 'BIIB', 'REGN', 'VRTX', 'AMGN', 'CELG'],
    'healthcare_services': ['UNH', 'ANTM', 'CVS', 'CI', 'HUM', 'CNC'],
    
    # Energy & Commodities
    'oil_majors': ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'OXY'],
    'energy_services': ['SLB', 'HAL', 'BKR', 'NOV', 'FTI', 'HP'],
    'commodities': ['GLD', 'SLV', 'USO', 'DBA', 'CORN', 'UNG'],
    'precious_metals': ['GLD', 'SLV', 'GOLD', 'NEM', 'ABX', 'KGC'],
    
    # Consumer & Retail
    'consumer_staples': ['PG', 'KO', 'PEP', 'WMT', 'COST', 'CL'],
    'consumer_discretionary': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX'],
    'retail': ['WMT', 'AMZN', 'COST', 'TGT', 'HD', 'LOW'],
    
    # Industrial & Manufacturing
    'industrials': ['BA', 'CAT', 'GE', 'MMM', 'HON', 'UPS'],
    'aerospace': ['BA', 'LMT', 'RTX', 'NOC', 'GD', 'TXT'],
    
    # Utilities & REITs
    'utilities': ['NEE', 'DUK', 'SO', 'D', 'EXC', 'XEL'],
    'reits': ['AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'EXR'],
    
    # Cross-Sector Combinations
    'mega_cap': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META'],
    'dividend_aristocrats': ['JNJ', 'PG', 'KO', 'MCD', 'WMT', 'XOM'],
    'high_beta': ['TSLA', 'NVDA', 'NFLX', 'ZOOM', 'PTON', 'ROKU'],
    'defensive': ['JNJ', 'PG', 'KO', 'WMT', 'VZ', 'T'],
    
    # Market Indices & ETFs
    'market_indices': ['SPY', 'QQQ', 'IWM', 'VTI', 'VEA', 'VWO'],
    'sector_etfs': ['XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLP'],
    'volatility': ['VIX', 'UVXY', 'SVXY', 'VXX', 'VIXY', 'TVIX'],
}

# =================== CRISIS PERIOD PRESETS ===================

CRISIS_PERIODS = {
    # COVID-19 Pandemic
    'covid_crash': {
        'start_date': '2020-02-01',
        'end_date': '2020-04-30',
        'description': 'COVID-19 market crash and initial recovery'
    },
    'covid_recovery': {
        'start_date': '2020-04-01',
        'end_date': '2020-12-31',
        'description': 'COVID-19 recovery period with massive stimulus'
    },
    'covid_full_cycle': {
        'start_date': '2020-01-01',
        'end_date': '2021-12-31',
        'description': 'Complete COVID-19 cycle: crash, recovery, and boom'
    },
    
    # Inflation & Rate Hikes
    'inflation_bear_market': {
        'start_date': '2022-01-01',
        'end_date': '2022-12-31',
        'description': 'Inflation surge and aggressive Fed rate hikes'
    },
    'rate_hike_cycle': {
        'start_date': '2022-03-01',
        'end_date': '2023-12-31',
        'description': 'Federal Reserve rate hiking cycle'
    },
    
    # Financial Crisis (if data available)
    'financial_crisis': {
        'start_date': '2008-09-01',
        'end_date': '2009-03-31',
        'description': 'Global financial crisis and bank failures'
    },
    'financial_crisis_extended': {
        'start_date': '2007-07-01',
        'end_date': '2009-12-31',
        'description': 'Extended financial crisis period'
    },
    
    # Dot-Com Bubble
    'dotcom_crash': {
        'start_date': '2000-03-01',
        'end_date': '2002-12-31',
        'description': 'Dot-com bubble burst and tech crash'
    },
    
    # European Debt Crisis
    'european_debt_crisis': {
        'start_date': '2010-05-01',
        'end_date': '2012-12-31',
        'description': 'European sovereign debt crisis'
    },
    
    # Flash Crash
    'flash_crash': {
        'start_date': '2010-05-01',
        'end_date': '2010-05-31',
        'description': 'May 2010 flash crash'
    },
    
    # Recent Volatility Events
    'china_devaluation': {
        'start_date': '2015-08-01',
        'end_date': '2016-02-29',
        'description': 'China currency devaluation and market turmoil'
    },
    'brexit_vote': {
        'start_date': '2016-06-01',
        'end_date': '2016-07-31',
        'description': 'Brexit referendum and aftermath'
    },
    'trade_war': {
        'start_date': '2018-01-01',
        'end_date': '2019-12-31',
        'description': 'US-China trade war escalation'
    },
    
    # Stable Periods (for comparison)
    'low_volatility_2017': {
        'start_date': '2017-01-01',
        'end_date': '2017-12-31',
        'description': 'Historically low volatility period'
    },
    'bull_market_2013_2015': {
        'start_date': '2013-01-01',
        'end_date': '2015-12-31',
        'description': 'Strong bull market with low volatility'
    },
}

# =================== CONVENIENCE FUNCTIONS ===================

def analyze_asset_group(group_name, period='1y', **kwargs):
    """
    Analyze a predefined asset group.
    
    Parameters:
    -----------
    group_name : str
        Name of asset group from ASSET_GROUPS
    period : str
        Time period for analysis
    **kwargs : dict
        Additional parameters for BellInequalityAnalyzer
        
    Returns:
    --------
    BellInequalityAnalyzer : Configured analyzer with results
    """
    if group_name not in ASSET_GROUPS:
        available_groups = list(ASSET_GROUPS.keys())
        raise ValueError(f"Unknown asset group '{group_name}'. Available: {available_groups}")
    
    assets = ASSET_GROUPS[group_name]
    
    print(f"🎯 Analyzing asset group: {group_name}")
    print(f"   Assets: {assets}")
    print(f"   Period: {period}")
    
    analyzer = BellInequalityAnalyzer(assets=assets, period=period, **kwargs)
    
    if analyzer.load_data():
        results = analyzer.run_s1_analysis()
        analyzer.create_visualizations(f'bell_analysis_{group_name}_{period}.png')
        analyzer.generate_summary_report()
        return analyzer
    else:
        print(f"❌ Failed to load data for {group_name}")
        return None

def analyze_crisis_period(crisis_name, assets=None, **kwargs):
    """
    Analyze a predefined crisis period.
    
    Parameters:
    -----------
    crisis_name : str
        Name of crisis period from CRISIS_PERIODS
    assets : list, optional
        Assets to analyze. Default: tech giants
    **kwargs : dict
        Additional parameters for BellInequalityAnalyzer
        
    Returns:
    --------
    BellInequalityAnalyzer : Configured analyzer with results
    """
    if crisis_name not in CRISIS_PERIODS:
        available_crises = list(CRISIS_PERIODS.keys())
        raise ValueError(f"Unknown crisis period '{crisis_name}'. Available: {available_crises}")
    
    crisis_config = CRISIS_PERIODS[crisis_name]
    assets = assets or ASSET_GROUPS['tech_giants']
    
    print(f"📉 Analyzing crisis period: {crisis_name}")
    print(f"   Description: {crisis_config['description']}")
    print(f"   Date range: {crisis_config['start_date']} to {crisis_config['end_date']}")
    print(f"   Assets: {assets}")
    
    analyzer = BellInequalityAnalyzer(
        assets=assets,
        start_date=crisis_config['start_date'],
        end_date=crisis_config['end_date'],
        **kwargs
    )
    
    if analyzer.load_data():
        results = analyzer.run_s1_analysis()
        analyzer.create_visualizations(f'bell_analysis_{crisis_name}.png')
        analyzer.generate_summary_report()
        return analyzer
    else:
        print(f"❌ Failed to load data for {crisis_name}")
        return None

def compare_asset_groups(group_names, period='1y'):
    """
    Compare Bell inequality violations across multiple asset groups.
    
    Parameters:
    -----------
    group_names : list
        List of asset group names to compare
    period : str
        Time period for analysis
        
    Returns:
    --------
    dict : Results for each asset group
    """
    results = {}
    
    print(f"📊 COMPARING ASSET GROUPS")
    print(f"=" * 30)
    
    for group_name in group_names:
        print(f"\n🔍 Analyzing {group_name}...")
        analyzer = analyze_asset_group(group_name, period=period)
        
        if analyzer and analyzer.s1_results:
            summary = analyzer.s1_results['summary']
            results[group_name] = {
                'violation_rate': summary['overall_violation_rate'],
                'max_violation': summary['max_violation_pct'],
                'total_violations': summary['total_violations'],
                'assets': analyzer.assets
            }
    
    # Print comparison summary
    print(f"\n📋 COMPARISON SUMMARY:")
    print(f"=" * 25)
    
    for group_name, result in results.items():
        print(f"{group_name:20}: {result['violation_rate']:6.2f}% violations "
              f"(max: {result['max_violation']:5.1f}%)")
    
    return results

def compare_crisis_periods(crisis_names, assets=None):
    """
    Compare Bell inequality violations across multiple crisis periods.
    
    Parameters:
    -----------
    crisis_names : list
        List of crisis period names to compare
    assets : list, optional
        Assets to analyze. Default: tech giants
        
    Returns:
    --------
    dict : Results for each crisis period
    """
    results = {}
    assets = assets or ASSET_GROUPS['tech_giants']
    
    print(f"📉 COMPARING CRISIS PERIODS")
    print(f"=" * 30)
    
    for crisis_name in crisis_names:
        print(f"\n🔍 Analyzing {crisis_name}...")
        analyzer = analyze_crisis_period(crisis_name, assets=assets)
        
        if analyzer and analyzer.s1_results:
            summary = analyzer.s1_results['summary']
            results[crisis_name] = {
                'violation_rate': summary['overall_violation_rate'],
                'max_violation': summary['max_violation_pct'],
                'total_violations': summary['total_violations'],
                'period': CRISIS_PERIODS[crisis_name]
            }
    
    # Print comparison summary
    print(f"\n📋 CRISIS COMPARISON SUMMARY:")
    print(f"=" * 30)
    
    for crisis_name, result in results.items():
        print(f"{crisis_name:20}: {result['violation_rate']:6.2f}% violations "
              f"(max: {result['max_violation']:5.1f}%)")
    
    return results

# =================== EXAMPLE USAGE ===================

if __name__ == "__main__":
    # Example 1: Analyze tech giants
    tech_analyzer = analyze_asset_group('tech_giants', period='1y')
    
    # Example 2: Analyze COVID crash
    covid_analyzer = analyze_crisis_period('covid_crash')
    
    # Example 3: Compare sectors
    sector_comparison = compare_asset_groups(['tech_giants', 'big_banks', 'big_pharma'])
    
    # Example 4: Compare crisis periods
    crisis_comparison = compare_crisis_periods(['covid_crash', 'inflation_bear_market'])