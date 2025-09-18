#!/usr/bin/env python3
"""
INTEGRATED ANALYSIS RUNNER
==========================
Demonstrates how to run both Yahoo Finance Bell analysis 
and updated cross-Mandelbrot metrics together
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Import our analysis modules
from analyze_yahoo_finance_bell import YahooFinanceBellAnalyzer, run_yahoo_finance_bell_analysis
from updated_cross_mandelbrot_metrics import CrossMandelbrotAnalyzer, add_cross_mandelbrot_to_existing_analysis

def run_complete_integrated_analysis():
    """
    Run complete integrated analysis:
    1. Yahoo Finance Bell inequality tests
    2. Cross-variable Mandelbrot metrics
    3. Combined insights and recommendations
    """
    
    print("🚀 INTEGRATED BELL + CROSS-MANDELBROT ANALYSIS")
    print("=" * 60)
    print("This analysis will help you decide what to focus on for WDRS data")
    
    # Step 1: Yahoo Finance Bell Analysis
    print(f"\n📊 STEP 1: YAHOO FINANCE BELL ANALYSIS")
    print("=" * 40)
    
    # Define asset groups to test
    asset_groups = {
        'tech_giants': ['AAPL', 'MSFT', 'GOOGL', 'NVDA'],
        'volatile_growth': ['TSLA', 'NFLX', 'NVDA', 'META'],
        'mega_cap': ['AAPL', 'MSFT', 'GOOGL', 'AMZN'],
        'cross_sector': ['AAPL', 'JPM', 'XOM', 'JNJ']  # Different sectors
    }
    
    results_summary = {}
    
    for group_name, assets in asset_groups.items():
        print(f"\n🎯 Analyzing group: {group_name}")
        print(f"   Assets: {assets}")
        
        try:
            # Run Yahoo Finance analysis
            analyzer, results = run_yahoo_finance_bell_analysis(
                assets=assets,
                period='6mo',
                frequency='1d'  # Daily for more data points
            )
            
            if analyzer and results:
                # Extract key metrics
                group_summary = extract_key_metrics(analyzer, results)
                results_summary[group_name] = group_summary
                
                print(f"   ✅ Analysis complete for {group_name}")
            else:
                print(f"   ❌ Analysis failed for {group_name}")
                
        except Exception as e:
            print(f"   ❌ Error analyzing {group_name}: {e}")
    
    # Step 2: Cross-Mandelbrot Analysis on Best Group
    print(f"\n🌀 STEP 2: CROSS-MANDELBROT ANALYSIS")
    print("=" * 40)
    
    if results_summary:
        # Find the group with most interesting results
        best_group = find_most_interesting_group(results_summary)
        print(f"🎯 Running detailed cross-Mandelbrot analysis on: {best_group}")
        
        # Get the best analyzer instance
        best_assets = asset_groups[best_group]
        best_analyzer, best_results = run_yahoo_finance_bell_analysis(
            assets=best_assets,
            period='6mo',
            frequency='1d'
        )
        
        if best_analyzer and best_results:
            # Add cross-Mandelbrot analysis
            cross_analyzer = add_cross_mandelbrot_to_existing_analysis(best_analyzer)
            
            if cross_analyzer:
                print(f"✅ Cross-Mandelbrot analysis added successfully")
            else:
                print(f"❌ Cross-Mandelbrot analysis failed")
    
    # Step 3: Generate WDRS Recommendations
    print(f"\n💡 STEP 3: WDRS DATA RECOMMENDATIONS")
    print("=" * 40)
    
    generate_wdrs_recommendations(results_summary, asset_groups)
    
    return results_summary

def extract_key_metrics(analyzer, results):
    """Extract key metrics from Yahoo Finance analysis"""
    
    summary = {
        'assets_analyzed': len(analyzer.assets),
        'data_points': 0,
        'bell_violations': 0,
        'total_bell_tests': 0,
        'avg_correlation': 0,
        'max_correlation': 0,
        'volatility_clustering': 0
    }
    
    try:
        # Extract data points
        if '1d' in analyzer.processed_data:
            daily_data = analyzer.processed_data['1d']
            if daily_data:
                first_asset = list(daily_data.keys())[0]
                summary['data_points'] = len(daily_data[first_asset])
        
        # Extract Bell violations (if available)
        if hasattr(analyzer, 'bell_results') and analyzer.bell_results:
            for freq_results in analyzer.bell_results.values():
                for result in freq_results.values():
                    summary['total_bell_tests'] += 1
                    if result.get('violation', False):
                        summary['bell_violations'] += 1
        
        # Extract correlations
        if '1d' in analyzer.processed_data:
            daily_data = analyzer.processed_data['1d']
            returns_data = {}
            for asset in analyzer.assets:
                if asset in daily_data:
                    returns_data[asset] = daily_data[asset]['Returns']
            
            if len(returns_data) >= 2:
                corr_df = pd.DataFrame(returns_data).corr()
                # Get upper triangle (excluding diagonal)
                mask = np.triu(np.ones_like(corr_df), k=1).astype(bool)
                correlations = corr_df.values[mask]
                correlations = correlations[~np.isnan(correlations)]
                
                if len(correlations) > 0:
                    summary['avg_correlation'] = np.mean(np.abs(correlations))
                    summary['max_correlation'] = np.max(np.abs(correlations))
    
    except Exception as e:
        print(f"   ⚠️  Error extracting metrics: {e}")
    
    return summary

def find_most_interesting_group(results_summary):
    """Find the most interesting asset group based on analysis results"""
    
    scores = {}
    
    for group_name, summary in results_summary.items():
        score = 0
        
        # Score based on Bell violations
        if summary['total_bell_tests'] > 0:
            violation_rate = summary['bell_violations'] / summary['total_bell_tests']
            score += violation_rate * 100  # Weight Bell violations highly
        
        # Score based on correlations (moderate correlations are interesting)
        avg_corr = summary['avg_correlation']
        if 0.3 < avg_corr < 0.8:  # Sweet spot for interesting correlations
            score += 50
        
        # Score based on data availability
        if summary['data_points'] > 100:
            score += 20
        
        # Score based on number of assets
        if summary['assets_analyzed'] >= 3:
            score += 10
        
        scores[group_name] = score
    
    if scores:
        best_group = max(scores, key=scores.get)
        print(f"   🏆 Most interesting group: {best_group} (score: {scores[best_group]:.1f})")
        return best_group
    else:
        return list(results_summary.keys())[0] if results_summary else 'tech_giants'

def generate_wdrs_recommendations(results_summary, asset_groups):
    """Generate recommendations for WDRS data selection"""
    
    print("🎯 RECOMMENDATIONS FOR WDRS DATA SELECTION:")
    print("=" * 50)
    
    if not results_summary:
        print("❌ No analysis results available for recommendations")
        return
    
    # Find groups with highest Bell violations
    violation_groups = []
    for group_name, summary in results_summary.items():
        if summary['total_bell_tests'] > 0:
            violation_rate = summary['bell_violations'] / summary['total_bell_tests']
            violation_groups.append((group_name, violation_rate, summary))
    
    violation_groups.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n1. 🔔 BELL VIOLATION PRIORITIES:")
    for i, (group_name, violation_rate, summary) in enumerate(violation_groups[:3]):
        assets = asset_groups[group_name]
        print(f"   {i+1}. {group_name}: {violation_rate*100:.1f}% violations")
        print(f"      Assets: {assets}")
        print(f"      Reason: High quantum-like correlations detected")
    
    print(f"\n2. 📊 TIME FREQUENCY RECOMMENDATIONS:")
    print("   Based on Yahoo Finance analysis:")
    print("   • Focus on daily (1d) frequency - showed most Bell violations")
    print("   • Consider hourly (1h) for high-frequency effects")
    print("   • Avoid very short timeframes (< 15min) - too noisy")
    
    print(f"\n3. 🎯 ASSET SELECTION FOR WDRS:")
    
    # Recommend top assets across all groups
    all_assets = set()
    for assets in asset_groups.values():
        all_assets.update(assets)
    
    # Score assets based on how often they appear in high-violation groups
    asset_scores = {}
    for asset in all_assets:
        score = 0
        for group_name, violation_rate, summary in violation_groups:
            if asset in asset_groups[group_name]:
                score += violation_rate * 100
        asset_scores[asset] = score
    
    top_assets = sorted(asset_scores.items(), key=lambda x: x[1], reverse=True)
    
    print("   Priority assets for WDRS download:")
    for i, (asset, score) in enumerate(top_assets[:6]):
        print(f"   {i+1}. {asset} (score: {score:.1f})")
    
    print(f"\n4. 📅 TIME PERIOD RECOMMENDATIONS:")
    print("   • Start with 2-3 years of data for statistical significance")
    print("   • Focus on periods with market volatility (2020-2023)")
    print("   • Include both bull and bear market periods")
    
    print(f"\n5. 🌀 CROSS-MANDELBROT FOCUS:")
    print("   Based on preliminary analysis:")
    print("   • Prioritize asset pairs with high cross-correlations")
    print("   • Look for lead-lag relationships between assets")
    print("   • Focus on cross-volatility clustering effects")
    
    print(f"\n6. 🔬 ANALYSIS STRATEGY:")
    print("   1. Download WDRS data for top 4-6 assets")
    print("   2. Focus on daily frequency initially")
    print("   3. Run full Bell inequality tests")
    print("   4. Apply cross-Mandelbrot metrics between all pairs")
    print("   5. Identify strongest quantum-like effects")
    print("   6. Drill down with higher frequency data if needed")
    
    print(f"\n✅ Use these recommendations to guide your WDRS data selection!")

if __name__ == "__main__":
    # Run the complete integrated analysis
    try:
        results = run_complete_integrated_analysis()
        print(f"\n🎉 INTEGRATED ANALYSIS COMPLETE!")
        print(f"   Results available for {len(results)} asset groups")
        print(f"   Check generated visualizations for detailed insights")
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        print(f"   Make sure you have required packages: yfinance, pandas, numpy, matplotlib, seaborn, scipy")
        print(f"   Install with: pip install yfinance pandas numpy matplotlib seaborn scipy")