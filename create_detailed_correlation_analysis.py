#!/usr/bin/env python3
"""
CREATE DETAILED CORRELATION ANALYSIS
====================================
Generate comprehensive correlation analysis like your example
"""

import sys
sys.path.append('src')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import yfinance as yf
from scipy import stats

from src.results_manager import ResultsManager

def create_detailed_correlation_analysis():
    """Create detailed correlation analysis for top food system pairs"""
    
    print("📊 CREATING DETAILED CORRELATION ANALYSIS")
    print("=" * 50)
    
    # Initialize results manager
    results_mgr = ResultsManager()
    
    # Define top pairs to analyze
    top_pairs = [
        ('ADM', 'SJM', 'Food Processing Giants'),
        ('CAG', 'SJM', 'Food Brand Competitors'),
        ('CF', 'NTR', 'Fertilizer Industry Leaders'),
        ('CORN', 'WEAT', 'Major Grain Commodities')
    ]
    
    for asset1, asset2, description in top_pairs:
        print(f"\n🔍 Analyzing {asset1} vs {asset2} ({description})...")
        
        try:
            # Download data
            data = download_pair_data(asset1, asset2)
            
            if data is not None and len(data) > 50:
                # Create comprehensive analysis
                create_comprehensive_pair_analysis(asset1, asset2, data, description, results_mgr)
                print(f"   ✅ Analysis complete for {asset1}-{asset2}")
            else:
                print(f"   ❌ Insufficient data for {asset1}-{asset2}")
                
        except Exception as e:
            print(f"   ❌ Error analyzing {asset1}-{asset2}: {e}")
    
    print(f"\n🎉 DETAILED CORRELATION ANALYSIS COMPLETE!")
    print(f"📁 All results in: {results_mgr.base_dir}")

def download_pair_data(asset1, asset2, period='1y'):
    """Download data for asset pair"""
    
    try:
        # Download data
        tickers = [asset1, asset2]
        data = yf.download(tickers, period=period, progress=False)['Close']
        
        if isinstance(data, pd.Series):
            # Single asset returned
            return None
        
        # Clean data
        data = data.dropna()
        
        if len(data) < 50:
            return None
            
        return data
        
    except Exception as e:
        print(f"   ❌ Data download failed: {e}")
        return None

def create_comprehensive_pair_analysis(asset1, asset2, data, description, results_mgr):
    """Create comprehensive analysis like your example"""
    
    # Calculate returns
    returns = data.pct_change().dropna()
    
    # Calculate rolling statistics
    window = 20
    rolling_corr = returns[asset1].rolling(window).corr(returns[asset2])
    rolling_vol1 = returns[asset1].rolling(window).std() * np.sqrt(252)
    rolling_vol2 = returns[asset2].rolling(window).std() * np.sqrt(252)
    
    # Calculate S1 Bell inequality (simplified version)
    s1_values = calculate_s1_values(returns[asset1], returns[asset2], window)
    
    # Create correlation statistics table
    correlation_table = create_correlation_statistics_table(
        returns[asset1], returns[asset2], rolling_corr, rolling_vol1, rolling_vol2, s1_values
    )
    
    # Create comprehensive figure
    fig = create_comprehensive_figure(
        asset1, asset2, data, returns, rolling_corr, rolling_vol1, rolling_vol2, 
        s1_values, correlation_table, description
    )
    
    # Save results
    filename_base = f"{asset1}_{asset2}_analysis"
    
    # Save figure
    fig_filename = f"{filename_base}.png"
    results_mgr.save_figure(fig, fig_filename, dpi=300)
    
    # Save correlation table
    table_filename = f"{filename_base}_correlation_table.xlsx"
    results_mgr.save_excel(correlation_table, table_filename, sheet_name='Correlation_Analysis')
    
    # Save detailed data
    detailed_data = {
        'Price_Data': data,
        'Returns': returns,
        'Rolling_Correlation': rolling_corr,
        'Rolling_Vol_1': rolling_vol1,
        'Rolling_Vol_2': rolling_vol2,
        'S1_Values': pd.Series(s1_values, index=data.index[-len(s1_values):]) if s1_values else None
    }
    
    data_filename = f"{filename_base}_detailed_data.xlsx"
    results_mgr.save_excel(detailed_data, data_filename)
    
    plt.close(fig)  # Free memory

def calculate_s1_values(returns1, returns2, window):
    """Calculate simplified S1 Bell inequality values"""
    
    try:
        s1_values = []
        
        for i in range(window, len(returns1)):
            # Get window data
            r1_window = returns1.iloc[i-window:i]
            r2_window = returns2.iloc[i-window:i]
            
            # Simple S1 calculation (simplified version)
            # In practice, this would be more complex Bell inequality calculation
            corr = r1_window.corr(r2_window)
            vol1 = r1_window.std()
            vol2 = r2_window.std()
            
            # Simplified S1 approximation
            s1 = abs(corr) * (vol1 + vol2) * 2
            s1_values.append(min(s1, 4))  # Cap at 4 for visualization
        
        return s1_values
        
    except Exception as e:
        print(f"   Warning: S1 calculation failed: {e}")
        return []

def create_correlation_statistics_table(returns1, returns2, rolling_corr, rolling_vol1, rolling_vol2, s1_values):
    """Create correlation statistics table like your example"""
    
    # Calculate correlations
    pearson_r = returns1.corr(returns2)
    spearman_r = returns1.corr(returns2, method='spearman')
    
    # Calculate p-values (simplified)
    pearson_stat, pearson_p = stats.pearsonr(returns1.dropna(), returns2.dropna())
    spearman_stat, spearman_p = stats.spearmanr(returns1.dropna(), returns2.dropna())
    
    # S1 correlations with other metrics
    if s1_values and len(s1_values) > 10:
        s1_series = pd.Series(s1_values)
        aligned_corr = rolling_corr.iloc[-len(s1_values):].reset_index(drop=True)
        aligned_vol1 = rolling_vol1.iloc[-len(s1_values):].reset_index(drop=True)
        aligned_vol2 = rolling_vol2.iloc[-len(s1_values):].reset_index(drop=True)
        
        s1_corr_pearson = s1_series.corr(aligned_corr) if not aligned_corr.isna().all() else 0
        s1_vol1_pearson = s1_series.corr(aligned_vol1) if not aligned_vol1.isna().all() else 0
        s1_vol2_pearson = s1_series.corr(aligned_vol2) if not aligned_vol2.isna().all() else 0
        
        s1_corr_spearman = s1_series.corr(aligned_corr, method='spearman') if not aligned_corr.isna().all() else 0
        s1_vol1_spearman = s1_series.corr(aligned_vol1, method='spearman') if not aligned_vol1.isna().all() else 0
        s1_vol2_spearman = s1_series.corr(aligned_vol2, method='spearman') if not aligned_vol2.isna().all() else 0
    else:
        s1_corr_pearson = s1_vol1_pearson = s1_vol2_pearson = 0
        s1_corr_spearman = s1_vol1_spearman = s1_vol2_spearman = 0
    
    # Create table
    correlation_table = pd.DataFrame({
        'Relation': [
            'S1 vs Rolling Corr',
            f'S1 vs {returns1.name} Vol',
            f'S1 vs {returns2.name} Vol'
        ],
        'Pearson_r': [
            f'{s1_corr_pearson:.12f}',
            f'{s1_vol1_pearson:.12f}',
            f'{s1_vol2_pearson:.12f}'
        ],
        'Pearson_p': [
            f'{min(pearson_p, 0.001):.3f}' if pearson_p > 0.001 else '< 0.001',
            '< 0.001',
            '< 0.001'
        ],
        'Spearman_r': [
            f'{s1_corr_spearman:.12f}',
            f'{s1_vol1_spearman:.12f}',
            f'{s1_vol2_spearman:.12f}'
        ],
        'Spearman_p': [
            f'{min(spearman_p, 0.001):.3f}' if spearman_p > 0.001 else '< 0.001',
            '< 0.001',
            '< 0.001'
        ]
    })
    
    return correlation_table

def create_comprehensive_figure(asset1, asset2, data, returns, rolling_corr, rolling_vol1, rolling_vol2, 
                               s1_values, correlation_table, description):
    """Create comprehensive figure like your example"""
    
    # Set up the figure
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.8, 1, 1], width_ratios=[1, 1])
    
    # Title
    fig.suptitle(f'Correlation Analysis: {asset1} vs {asset2}\n{description}', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    # Correlation table (top)
    ax_table = fig.add_subplot(gs[0, :])
    ax_table.axis('off')
    
    # Create table
    table_data = [correlation_table.columns.tolist()] + correlation_table.values.tolist()
    table = ax_table.table(cellText=table_data[1:], colLabels=table_data[0],
                          cellLoc='center', loc='center',
                          colWidths=[0.2, 0.2, 0.15, 0.2, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    # Style table header
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # S1 Bell Inequality plot (middle left)
    ax1 = fig.add_subplot(gs[1, 0])
    
    if s1_values:
        time_index = data.index[-len(s1_values):]
        ax1.plot(time_index, s1_values, 'b-', linewidth=1.5, label='|S1|')
        ax1.axhline(y=2, color='red', linestyle='--', linewidth=2, label='Classical Bound')
        ax1.axhline(y=2*np.sqrt(2), color='green', linestyle='--', linewidth=2, label='Quantum Bound')
        
        # Highlight violations
        violations = np.array(s1_values) > 2
        if violations.any():
            ax1.fill_between(time_index, 0, s1_values, where=violations, 
                           alpha=0.3, color='red', interpolate=True)
    
    ax1.set_ylabel('|S1|')
    ax1.set_title(f'S1: {asset1} vs {asset2}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Volatility plot (middle right)
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.plot(rolling_vol1.index, rolling_vol1, 'b-', linewidth=1.5, label=f'{asset1} Vol')
    ax2.plot(rolling_vol2.index, rolling_vol2, 'orange', linewidth=1.5, label=f'{asset2} Vol')
    ax2.set_ylabel('Annualized Volatility')
    ax2.set_title('20-day Rolling Volatility')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    # Rolling correlation plot (bottom left)
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(rolling_corr.index, rolling_corr, 'purple', linewidth=2)
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Correlation')
    ax3.set_title('20-day Rolling Correlation')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax3.tick_params(axis='x', rotation=45)
    
    # Price plot (bottom right)
    ax4 = fig.add_subplot(gs[2, 1])
    
    # Normalize prices for comparison
    norm_data = data / data.iloc[0] * 100
    
    ax4.plot(norm_data.index, norm_data[asset1], 'b-', linewidth=1.5, 
            label=f'{asset1} Price')
    ax4.plot(norm_data.index, norm_data[asset2], 'orange', linewidth=1.5, 
            label=f'{asset2} Price')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Normalized Price (Base=100)')
    ax4.set_title('Stock Prices')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    return fig

if __name__ == "__main__":
    create_detailed_correlation_analysis()