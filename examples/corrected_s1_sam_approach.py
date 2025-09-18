#!/usr/bin/env python3
"""
CORRECTED S1 BELL INEQUALITY - SAM'S APPROACH
=============================================
Implements the exact same approach as sam.ipynb to reproduce violations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from itertools import combinations
from datetime import datetime, timedelta

def expectation_ab(x_mask, y_mask, a, b):
    """Calculate conditional expectation E[AB|x_mask & y_mask]"""
    mask = x_mask & y_mask
    if mask.sum() == 0:
        return 0.0
    return np.mean(a[mask] * b[mask])

def calculate_s1_sam_approach(returns_data, window_size=20, threshold_quantile=0.75):
    """
    Calculate S1 Bell inequality using Sam's exact approach
    
    Key differences from our previous approach:
    1. Uses cumulative returns (not regular returns)
    2. Uses np.sign() for binary outcomes (-1, 0, +1)
    3. Uses absolute return thresholds for regimes
    4. Calculates E[AB|regime] directly
    """
    
    print("🔬 CALCULATING S1 USING SAM'S APPROACH")
    print("=" * 45)
    
    # Convert to cumulative returns (KEY DIFFERENCE!)
    cumulative_returns = returns_data.cumsum()
    print(f"📊 Using cumulative returns (Sam's approach)")
    print(f"   Data shape: {cumulative_returns.shape}")
    
    tickers = cumulative_returns.columns.tolist()
    s1_violations = []
    s1_time_series = {pair: [] for pair in combinations(tickers, 2)}
    
    print(f"🎯 Analyzing {len(list(combinations(tickers, 2)))} asset pairs")
    print(f"   Window size: {window_size}")
    print(f"   Threshold quantile: {threshold_quantile}")
    
    violation_count = 0
    total_calculations = 0
    
    # Rolling window analysis
    for T in range(window_size, len(cumulative_returns)):
        window_returns = cumulative_returns.iloc[T - window_size:T]
        
        # Calculate thresholds based on absolute returns (Sam's approach)
        thresholds = window_returns.abs().quantile(threshold_quantile)
        
        s1_values = []
        
        for stock_A, stock_B in combinations(window_returns.columns, 2):
            RA = window_returns[stock_A]
            RB = window_returns[stock_B]
            
            # Binary outcomes using sign (Sam's approach: -1, 0, +1)
            a = np.sign(RA)
            b = np.sign(RB)
            
            # Regime definitions based on absolute return thresholds
            x0 = RA.abs() >= thresholds[stock_A]  # High absolute return regime
            x1 = ~x0                              # Low absolute return regime
            y0 = RB.abs() >= thresholds[stock_B]  # High absolute return regime
            y1 = ~y0                              # Low absolute return regime
            
            # Calculate conditional expectations E[AB|regime]
            ab_00 = expectation_ab(x0, y0, a, b)  # Both high absolute returns
            ab_01 = expectation_ab(x0, y1, a, b)  # A high, B low
            ab_10 = expectation_ab(x1, y0, a, b)  # A low, B high
            ab_11 = expectation_ab(x1, y1, a, b)  # Both low absolute returns
            
            # S1 Bell inequality (Zarifian et al. 2025 formula)
            S1 = ab_00 + ab_01 + ab_10 - ab_11
            
            s1_values.append((stock_A, stock_B, S1))
            s1_time_series[(stock_A, stock_B)].append(S1)
            
            total_calculations += 1
            if abs(S1) > 2:
                violation_count += 1
        
        # Calculate violation percentage for this time window
        violations = sum(abs(s1) > 2 for _, _, s1 in s1_values)
        violation_pct = 100 * violations / len(s1_values) if len(s1_values) > 0 else 0
        
        s1_violations.append({
            'timestamp': cumulative_returns.index[T], 
            'violation_%': violation_pct,
            'total_pairs': len(s1_values),
            'violations': violations
        })
    
    # Summary statistics
    violation_df = pd.DataFrame(s1_violations).set_index('timestamp')
    
    overall_violation_rate = (violation_count / total_calculations) * 100 if total_calculations > 0 else 0
    max_violation_pct = violation_df['violation_%'].max()
    mean_violation_pct = violation_df['violation_%'].mean()
    
    print(f"\n📊 S1 ANALYSIS RESULTS (SAM'S APPROACH):")
    print(f"   Total calculations: {total_calculations:,}")
    print(f"   Total violations: {violation_count:,}")
    print(f"   Overall violation rate: {overall_violation_rate:.2f}%")
    print(f"   Max violation % in any window: {max_violation_pct:.1f}%")
    print(f"   Mean violation % across windows: {mean_violation_pct:.1f}%")
    
    # Find pairs with highest violations
    pair_violation_counts = {}
    for pair, s1_values in s1_time_series.items():
        violations = sum(1 for s1 in s1_values if abs(s1) > 2)
        pair_violation_counts[pair] = violations
    
    top_violating_pairs = sorted(pair_violation_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    
    print(f"\n🔔 TOP VIOLATING PAIRS:")
    for pair, violation_count in top_violating_pairs:
        total_windows = len(s1_time_series[pair])
        violation_rate = (violation_count / total_windows) * 100 if total_windows > 0 else 0
        print(f"   {pair[0]}-{pair[1]}: {violation_count}/{total_windows} ({violation_rate:.1f}%)")
    
    return {
        'violation_df': violation_df,
        's1_time_series': s1_time_series,
        'summary': {
            'total_calculations': total_calculations,
            'total_violations': violation_count,
            'overall_violation_rate': overall_violation_rate,
            'max_violation_pct': max_violation_pct,
            'mean_violation_pct': mean_violation_pct,
            'top_violating_pairs': top_violating_pairs
        }
    }

def download_and_analyze_yahoo_finance_sam_approach(assets=None, period='6mo'):
    """Download Yahoo Finance data and analyze using Sam's approach"""
    
    if assets is None:
        assets = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'META']
    
    print(f"🚀 YAHOO FINANCE S1 ANALYSIS - SAM'S APPROACH")
    print(f"=" * 50)
    print(f"Assets: {assets}")
    print(f"Period: {period}")
    
    # Download data
    print(f"\n📥 Downloading Yahoo Finance data...")
    end_date = datetime.now()
    
    if period == '6mo':
        start_date = end_date - timedelta(days=180)
    elif period == '1y':
        start_date = end_date - timedelta(days=365)
    else:
        start_date = end_date - timedelta(days=180)  # Default to 6 months
    
    try:
        data = yf.download(assets, start=start_date, end=end_date)['Close']
        
        if isinstance(data, pd.Series):
            data = data.to_frame()
        
        print(f"✅ Downloaded data: {data.shape}")
        print(f"   Date range: {data.index[0]} to {data.index[-1]}")
        print(f"   Assets: {list(data.columns)}")
        
        # Calculate returns
        returns = data.pct_change().dropna()
        print(f"📊 Returns calculated: {returns.shape}")
        
        # Run S1 analysis using Sam's approach
        results = calculate_s1_sam_approach(returns, window_size=20, threshold_quantile=0.75)
        
        # Create visualization
        create_s1_visualization_sam_approach(results, assets)
        
        return results
        
    except Exception as e:
        print(f"❌ Error downloading or analyzing data: {e}")
        return None

def create_s1_visualization_sam_approach(results, assets):
    """Create visualization of S1 results using Sam's approach"""
    
    print(f"\n📊 Creating S1 visualizations...")
    
    violation_df = results['violation_df']
    s1_time_series = results['s1_time_series']
    
    # Create comprehensive plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Violation percentage over time
    ax1 = axes[0, 0]
    ax1.plot(violation_df.index, violation_df['violation_%'], label='S1 Violation %', linewidth=2)
    ax1.axhline(50, color='red', linestyle='--', label='50% Threshold', alpha=0.7)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('% of Pairs Violating |S1| > 2')
    ax1.set_title('Bell S1 Violations Over Time\n(Sam\'s Approach)', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. S1 values for top violating pairs
    ax2 = axes[0, 1]
    top_pairs = list(s1_time_series.keys())[:4]  # Show top 4 pairs
    
    for pair in top_pairs:
        s1_values = s1_time_series[pair]
        times = violation_df.index[-len(s1_values):]
        ax2.plot(times, s1_values, label=f'{pair[0]}-{pair[1]}', alpha=0.8)
    
    ax2.axhline(2, color='red', linestyle='--', label='Classical Limit (+2)', alpha=0.7)
    ax2.axhline(-2, color='red', linestyle='--', alpha=0.7)
    ax2.axhline(2*np.sqrt(2), color='blue', linestyle='-.', label='Quantum Limit (+2√2)', alpha=0.7)
    ax2.axhline(-2*np.sqrt(2), color='blue', linestyle='-.', alpha=0.7)
    
    ax2.set_xlabel('Date')
    ax2.set_ylabel('S1 Value')
    ax2.set_title('S1 Values for Top Pairs', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Violation distribution
    ax3 = axes[1, 0]
    ax3.hist(violation_df['violation_%'], bins=20, alpha=0.7, edgecolor='black')
    ax3.axvline(violation_df['violation_%'].mean(), color='red', linestyle='--', 
                label=f'Mean: {violation_df["violation_%"].mean():.1f}%')
    ax3.set_xlabel('Violation Percentage')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of Violation Percentages', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 4. Summary statistics
    ax4 = axes[1, 1]
    summary = results['summary']
    
    metrics = ['Total\nCalculations', 'Total\nViolations', 'Overall\nViolation %', 'Max\nViolation %']
    values = [summary['total_calculations'], summary['total_violations'], 
              summary['overall_violation_rate'], summary['max_violation_pct']]
    
    bars = ax4.bar(metrics, values, alpha=0.7, color=['blue', 'red', 'orange', 'green'])
    ax4.set_title('S1 Analysis Summary', fontweight='bold')
    ax4.set_ylabel('Value')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.1f}' if isinstance(value, float) else f'{value:,}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('s1_analysis_sam_approach.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Visualization saved as 's1_analysis_sam_approach.png'")

# Main execution
if __name__ == "__main__":
    # Run analysis with Sam's approach
    results = download_and_analyze_yahoo_finance_sam_approach(
        assets=['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'META'],
        period='6mo'
    )
    
    if results:
        print(f"\n🎉 Analysis complete using Sam's approach!")
        print(f"   This should show violations similar to sam.ipynb")
    else:
        print(f"\n❌ Analysis failed")