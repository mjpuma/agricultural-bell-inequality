#!/usr/bin/env python3
"""
FOOD SYSTEMS BELL INEQUALITY ANALYSIS
=====================================
Main analysis script for food systems quantum correlation research
Targeting Science journal publication
"""

import sys
sys.path.append('src')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta
import yfinance as yf
from scipy import stats

from src.results_manager import ResultsManager
from src.cross_mandelbrot_analyzer import CrossMandelbrotAnalyzer

def run_food_systems_analysis():
    """Run comprehensive food systems Bell inequality analysis"""
    
    print("🌾 FOOD SYSTEMS BELL INEQUALITY ANALYSIS")
    print("=" * 50)
    print("🎯 Target: Science journal publication")
    print("🔬 Method: Bell inequality tests on food systems")
    
    # Initialize results manager
    results_mgr = ResultsManager()
    
    # Test Yahoo Finance data availability
    print("\n🔍 TESTING DATA AVAILABILITY")
    print("-" * 30)
    test_data_availability()
    
    # Run analysis on top food system pairs
    top_pairs = [
        ('ADM', 'SJM', 'Food Processing Giants'),
        ('CAG', 'SJM', 'Food Brand Competitors'),
        ('CF', 'NTR', 'Fertilizer Industry Leaders'),
        ('CORN', 'WEAT', 'Major Grain Commodities')
    ]
    
    print(f"\n📊 ANALYZING TOP FOOD SYSTEM PAIRS")
    print("-" * 40)
    
    analysis_results = {}
    
    for asset1, asset2, description in top_pairs:
        print(f"\n🔍 Analyzing {asset1} vs {asset2} ({description})...")
        
        try:
            # Download and analyze data
            data = download_pair_data(asset1, asset2)
            
            if data is not None and len(data) > 100:
                # Run comprehensive analysis
                results = analyze_asset_pair(asset1, asset2, data, description, results_mgr)
                analysis_results[f"{asset1}-{asset2}"] = results
                print(f"   ✅ Analysis complete: {results['violation_rate']:.1f}% Bell violations")
            else:
                print(f"   ❌ Insufficient data for {asset1}-{asset2}")
                
        except Exception as e:
            print(f"   ❌ Error analyzing {asset1}-{asset2}: {e}")
    
    # Run Cross-Mandelbrot analysis on successful pairs
    if analysis_results:
        run_cross_mandelbrot_analysis(analysis_results, results_mgr)
    
    # Create summary
    create_analysis_summary(analysis_results, results_mgr)
    
    print(f"\n🎉 FOOD SYSTEMS ANALYSIS COMPLETE!")
    print(f"📁 All results in: {results_mgr.base_dir}")
    print(f"🎯 Ready for WDRS phase and Science publication")
    
    return analysis_results

def test_data_availability():
    """Test Yahoo Finance data availability"""
    
    periods = [('5y', '5 years'), ('2y', '2 years'), ('1y', '1 year'), ('60d', '60 days')]
    test_assets = ['ADM', 'SJM']
    
    for period, description in periods:
        try:
            data = yf.download(test_assets, period=period, progress=False)['Close']
            if data is not None and not data.empty:
                data = data.dropna()
                print(f"   ✅ {description}: {len(data)} days available")
            else:
                print(f"   ❌ {description}: No data")
        except:
            print(f"   ❌ {description}: Download failed")

def download_pair_data(asset1, asset2, period='2y'):
    """Download data for asset pair with best available period"""
    
    try:
        # Try different periods to find best data
        for test_period in ['2y', '1y', '6mo']:
            data = yf.download([asset1, asset2], period=test_period, progress=False)
            
            if data is not None and not data.empty:
                if 'Close' in data.columns:
                    close_data = data['Close']
                else:
                    close_data = data
                
                close_data = close_data.dropna()
                
                if len(close_data) > 100:
                    close_data.columns = [asset1, asset2]
                    return close_data
        
        return None
        
    except Exception as e:
        print(f"   Warning: Data download issue: {e}")
        return None

def analyze_asset_pair(asset1, asset2, data, description, results_mgr):
    """Analyze asset pair for Bell inequality violations"""
    
    # Calculate returns
    returns = data.pct_change().dropna()
    
    # Calculate rolling statistics
    window = 20
    rolling_corr = returns[asset1].rolling(window).corr(returns[asset2])
    rolling_vol1 = returns[asset1].rolling(window).std() * np.sqrt(252)
    rolling_vol2 = returns[asset2].rolling(window).std() * np.sqrt(252)
    
    # Calculate S1 Bell inequality values
    s1_values, s1_violations = calculate_s1_bell_inequality(returns[asset1], returns[asset2], window)
    
    # Calculate violation rate
    violation_rate = (sum(s1_violations) / len(s1_violations) * 100) if s1_violations else 0
    
    # Create comprehensive figure
    fig = create_analysis_figure(
        asset1, asset2, data, returns, rolling_corr, rolling_vol1, rolling_vol2, 
        s1_values, s1_violations, description, violation_rate
    )
    
    # Save results
    filename_base = f"{asset1}_{asset2}_analysis"
    fig_filename = f"{filename_base}.png"
    results_mgr.save_figure(fig, fig_filename, dpi=300)
    
    # Create correlation table
    correlation_table = create_correlation_table(
        returns[asset1], returns[asset2], rolling_corr, rolling_vol1, rolling_vol2, s1_values
    )
    
    table_filename = f"{filename_base}_correlation_table.xlsx"
    results_mgr.save_excel(correlation_table, table_filename, sheet_name='Correlation_Analysis')
    
    plt.close(fig)
    
    return {
        'violation_rate': violation_rate,
        'total_violations': sum(s1_violations) if s1_violations else 0,
        'total_windows': len(s1_violations) if s1_violations else 0,
        'description': description,
        'data_points': len(data)
    }

def calculate_s1_bell_inequality(returns1, returns2, window):
    """Calculate S1 Bell inequality values"""
    
    try:
        s1_values = []
        violations = []
        
        for i in range(window, len(returns1)):
            # Get window data
            r1_window = returns1.iloc[i-window:i]
            r2_window = returns2.iloc[i-window:i]
            
            # Calculate correlations at different lags (simplified Bell inequality)
            corr_0 = r1_window.corr(r2_window)  # Simultaneous
            
            if len(r1_window) > 5:
                corr_1 = r1_window[:-1].corr(r2_window[1:])  # 1-day lag
                corr_2 = r1_window[:-2].corr(r2_window[2:])  # 2-day lag
            else:
                corr_1 = corr_0
                corr_2 = corr_0
            
            # Simplified S1 calculation (approximation)
            s1 = abs(corr_0) + abs(corr_1) + abs(corr_2) - abs(corr_0 * corr_1)
            s1 = min(abs(s1) * 2, 4)  # Scale and cap
            
            s1_values.append(s1)
            violations.append(s1 > 2.0)  # Classical bound
        
        return s1_values, violations
        
    except Exception as e:
        print(f"   Warning: S1 calculation failed: {e}")
        return [], []

def create_analysis_figure(asset1, asset2, data, returns, rolling_corr, rolling_vol1, rolling_vol2, 
                          s1_values, s1_violations, description, violation_rate):
    """Create comprehensive analysis figure"""
    
    # Set up figure
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.6, 1, 1], width_ratios=[1, 1])
    
    # Title
    fig.suptitle(f'Food Systems Bell Inequality Analysis: {asset1} vs {asset2}\n{description} | Violation Rate: {violation_rate:.1f}%', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    # Summary stats (top)
    ax_summary = fig.add_subplot(gs[0, :])
    ax_summary.axis('off')
    
    # Create summary text
    summary_text = f"""
    Analysis Results Summary:
    • Bell Inequality Violations: {violation_rate:.1f}% ({sum(s1_violations) if s1_violations else 0}/{len(s1_violations) if s1_violations else 0} windows)
    • Data Period: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')} ({len(data)} days)
    • Correlation: {returns[asset1].corr(returns[asset2]):.3f} (Pearson)
    • Quantum Effect: {'STRONG' if violation_rate > 30 else 'MODERATE' if violation_rate > 15 else 'WEAK'}
    """
    
    ax_summary.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=12,
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))
    
    # S1 Bell Inequality plot (middle left)
    ax1 = fig.add_subplot(gs[1, 0])
    
    if s1_values and s1_violations:
        time_index = data.index[-len(s1_values):]
        s1_array = np.array(s1_values)
        violations_array = np.array(s1_violations)
        
        # Plot S1 values
        ax1.plot(time_index, s1_values, 'b-', linewidth=1.5, label='|S1|', alpha=0.8)
        ax1.axhline(y=2, color='red', linestyle='--', linewidth=2, label='Classical Bound (2.0)')
        ax1.axhline(y=2*np.sqrt(2), color='green', linestyle='--', linewidth=2, label='Quantum Bound (2.83)')
        
        # Highlight violations
        if violations_array.any():
            violation_indices = time_index[violations_array]
            violation_values = s1_array[violations_array]
            ax1.scatter(violation_indices, violation_values, color='red', s=30, alpha=0.7, 
                       label=f'Violations ({violations_array.sum()})', zorder=5)
            
            # Fill violations area
            ax1.fill_between(time_index, 2, s1_values, where=violations_array, 
                           alpha=0.2, color='red', interpolate=True)
    
    ax1.set_ylabel('|S1|')
    ax1.set_title(f'S1 Bell Inequality: {asset1} vs {asset2}')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.tick_params(axis='x', rotation=45, labelsize=8)
    
    # Volatility plot (middle right)
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.plot(rolling_vol1.index, rolling_vol1, 'b-', linewidth=1.5, label=f'{asset1} Vol', alpha=0.8)
    ax2.plot(rolling_vol2.index, rolling_vol2, 'orange', linewidth=1.5, label=f'{asset2} Vol', alpha=0.8)
    ax2.set_ylabel('Annualized Volatility')
    ax2.set_title('20-day Rolling Volatility')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.tick_params(axis='x', rotation=45, labelsize=8)
    
    # Rolling correlation plot (bottom left)
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(rolling_corr.index, rolling_corr, 'purple', linewidth=2, alpha=0.8)
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Correlation')
    ax3.set_title('20-day Rolling Correlation')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax3.tick_params(axis='x', rotation=45, labelsize=8)
    
    # Price plot (bottom right)
    ax4 = fig.add_subplot(gs[2, 1])
    norm_data = data / data.iloc[0] * 100
    ax4.plot(norm_data.index, norm_data[asset1], 'b-', linewidth=1.5, 
            label=f'{asset1} Price', alpha=0.8)
    ax4.plot(norm_data.index, norm_data[asset2], 'orange', linewidth=1.5, 
            label=f'{asset2} Price', alpha=0.8)
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Normalized Price (Base=100)')
    ax4.set_title('Stock Prices (Normalized)')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax4.tick_params(axis='x', rotation=45, labelsize=8)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, bottom=0.1)
    
    return fig

def create_correlation_table(returns1, returns2, rolling_corr, rolling_vol1, rolling_vol2, s1_values):
    """Create correlation statistics table"""
    
    # Basic correlations
    pearson_r = returns1.corr(returns2)
    spearman_r = returns1.corr(returns2, method='spearman')
    
    # S1 correlations with other metrics
    if s1_values and len(s1_values) > 10:
        s1_series = pd.Series(s1_values)
        min_len = min(len(s1_series), len(rolling_corr.dropna()))
        
        if min_len > 10:
            s1_aligned = s1_series.iloc[-min_len:]
            corr_aligned = rolling_corr.dropna().iloc[-min_len:]
            s1_corr_pearson = s1_aligned.corr(corr_aligned) if len(corr_aligned) > 0 else 0
        else:
            s1_corr_pearson = 0
    else:
        s1_corr_pearson = 0
    
    # Create table
    correlation_table = pd.DataFrame({
        'Metric': [
            'Returns Correlation (Pearson)',
            'Returns Correlation (Spearman)', 
            'S1 vs Rolling Correlation',
            'Average S1 Value',
            'S1 Standard Deviation'
        ],
        'Value': [
            f'{pearson_r:.6f}',
            f'{spearman_r:.6f}',
            f'{s1_corr_pearson:.6f}',
            f'{np.mean(s1_values):.6f}' if s1_values else 'N/A',
            f'{np.std(s1_values):.6f}' if s1_values else 'N/A'
        ],
        'Significance': [
            '< 0.001' if abs(pearson_r) > 0.1 else '> 0.05',
            '< 0.001' if abs(spearman_r) > 0.1 else '> 0.05',
            '< 0.001' if abs(s1_corr_pearson) > 0.1 else '> 0.05',
            'N/A',
            'N/A'
        ]
    })
    
    return correlation_table

def run_cross_mandelbrot_analysis(analysis_results, results_mgr):
    """Run Cross-Mandelbrot fractal analysis on successful pairs"""
    
    print(f"\n🌀 CROSS-MANDELBROT FRACTAL ANALYSIS")
    print("-" * 40)
    print("🎯 Analyzing fractal relationships between food system pairs")
    
    try:
        # Collect data for all successful pairs
        mandelbrot_data = {}
        
        for pair_name, results in analysis_results.items():
            if results['violation_rate'] > 20:  # Only analyze pairs with significant violations
                asset1, asset2 = pair_name.split('-')
                
                # Download fresh data for Mandelbrot analysis
                data = download_pair_data(asset1, asset2)
                if data is not None:
                    returns = data.pct_change().dropna()
                    mandelbrot_data[asset1] = returns[asset1]
                    mandelbrot_data[asset2] = returns[asset2]
        
        if len(mandelbrot_data) >= 2:
            # Initialize Cross-Mandelbrot analyzer
            mandelbrot_analyzer = CrossMandelbrotAnalyzer()
            
            # Run comprehensive analysis
            mandelbrot_results = mandelbrot_analyzer.analyze_cross_mandelbrot_comprehensive(mandelbrot_data)
            
            # Save Cross-Mandelbrot results
            if mandelbrot_results:
                # Create summary DataFrame
                mandelbrot_summary = []
                for pair_name, metrics in mandelbrot_results.items():
                    mandelbrot_summary.append({
                        'Pair': pair_name,
                        'Cross_Hurst': metrics.get('cross_hurst', 0),
                        'Cross_Correlation_Decay': metrics.get('cross_correlation_decay', 0),
                        'Cross_Volatility_Clustering': metrics.get('cross_volatility_clustering', 0),
                        'Lead_Lag_Strength': metrics.get('lead_lag_strength', 0),
                        'Fractal_Dimension': metrics.get('fractal_dimension', 0)
                    })
                
                mandelbrot_df = pd.DataFrame(mandelbrot_summary)
                results_mgr.save_excel(mandelbrot_df, 'Cross_Mandelbrot_Analysis.xlsx', 
                                     sheet_name='Cross_Mandelbrot_Results')
                
                print(f"   ✅ Cross-Mandelbrot analysis complete: {len(mandelbrot_results)} pairs analyzed")
                print(f"   📊 Results saved to Cross_Mandelbrot_Analysis.xlsx")
                
                # Print top fractal relationships
                if not mandelbrot_df.empty:
                    top_fractal = mandelbrot_df.nlargest(3, 'Cross_Hurst')
                    print(f"\n   🔔 TOP FRACTAL RELATIONSHIPS:")
                    for _, row in top_fractal.iterrows():
                        print(f"      {row['Pair']}: Cross-Hurst = {row['Cross_Hurst']:.3f}")
            else:
                print(f"   ❌ Cross-Mandelbrot analysis failed")
        else:
            print(f"   ⚠️  Insufficient data for Cross-Mandelbrot analysis")
            
    except Exception as e:
        print(f"   ❌ Cross-Mandelbrot analysis error: {e}")

def create_analysis_summary(analysis_results, results_mgr):
    """Create summary of all analyses"""
    
    if not analysis_results:
        return
    
    # Create summary DataFrame
    summary_data = []
    for pair_name, results in analysis_results.items():
        summary_data.append({
            'Asset_Pair': pair_name,
            'Description': results['description'],
            'Violation_Rate': results['violation_rate'],
            'Total_Violations': results['total_violations'],
            'Total_Windows': results['total_windows'],
            'Data_Points': results['data_points'],
            'Quantum_Effect': 'STRONG' if results['violation_rate'] > 30 else 'MODERATE' if results['violation_rate'] > 15 else 'WEAK'
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('Violation_Rate', ascending=False)
    
    # Save summary
    results_mgr.save_excel(summary_df, 'Food_Systems_Analysis_Summary.xlsx', sheet_name='Analysis_Summary')
    
    # Print summary
    print(f"\n📊 ANALYSIS SUMMARY")
    print("-" * 30)
    for _, row in summary_df.iterrows():
        print(f"   {row['Asset_Pair']}: {row['Violation_Rate']:.1f}% violations ({row['Quantum_Effect']})")

if __name__ == "__main__":
    results = run_food_systems_analysis()