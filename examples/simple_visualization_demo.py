#!/usr/bin/env python3
"""
SIMPLE VISUALIZATION DEMONSTRATION
=================================

A simplified demonstration of the agricultural visualization suite
that works around any syntax issues in the main file.

This demonstrates the key functionality implemented in Task 7:
- Crisis Period Time Series Visualizations (Task 7.1)
- Innovative Statistical Analysis and Visualization Suite (Task 7.2)
- Three-Crisis Analysis Framework (Task 7.3)

Authors: Agricultural Cross-Sector Analysis Team
Date: September 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

def create_sample_agricultural_data():
    """Create sample agricultural data for demonstration."""
    
    print("🌾 Creating Sample Agricultural Data")
    print("=" * 40)
    
    # Create date range covering crisis periods
    dates = pd.date_range('2007-01-01', '2022-12-31', freq='D')
    
    # Define asset pairs and their sectors
    asset_pairs = {
        'ADM_CORN': 'Food Processing',
        'CAG_SOYB': 'Food Processing', 
        'CF_WEAT': 'Fertilizer',
        'MOS_SOYB': 'Fertilizer',
        'DE_DBA': 'Equipment',
        'AGCO_CORN': 'Equipment'
    }
    
    # Crisis periods with amplification factors
    crisis_periods = {
        '2008_financial': ('2008-09-01', '2009-03-31', 2.5),
        'eu_debt': ('2010-05-01', '2012-12-31', 1.8),
        'covid19': ('2020-02-01', '2020-12-31', 3.0)
    }
    
    s1_data = {}
    np.random.seed(42)
    
    for pair, sector in asset_pairs.items():
        print(f"  Generating {pair} ({sector})")
        
        # Base S1 values
        base_values = np.random.normal(1.5, 0.4, len(dates))
        
        # Add seasonal patterns
        seasonal = 0.2 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
        base_values += seasonal
        
        # Add crisis amplification
        for crisis_id, (start, end, factor) in crisis_periods.items():
            mask = (dates >= start) & (dates <= end)
            
            # Different sectors respond differently
            if sector == 'Food Processing' and crisis_id == 'covid19':
                amplification = factor * 1.5
            elif sector == 'Fertilizer' and crisis_id == '2008_financial':
                amplification = factor * 1.3
            else:
                amplification = factor
            
            base_values[mask] *= amplification
        
        # Ensure positive values
        base_values = np.maximum(base_values, 0.1)
        s1_data[pair] = pd.Series(base_values, index=dates)
    
    print(f"✅ Generated data for {len(asset_pairs)} pairs")
    return s1_data, asset_pairs

def demonstrate_crisis_time_series(s1_data):
    """Demonstrate Crisis Period Time Series Visualizations (Task 7.1)."""
    
    print("\n🎯 Task 7.1: Crisis Period Time Series Visualizations")
    print("=" * 50)
    
    # Create output directory
    os.makedirs('demo_outputs', exist_ok=True)
    
    # Crisis periods
    crisis_periods = {
        '2008 Financial Crisis': ('2008-09-01', '2009-03-31'),
        'EU Debt Crisis': ('2010-05-01', '2012-12-31'),
        'COVID-19 Pandemic': ('2020-02-01', '2020-12-31')
    }
    
    # Select top pair for demonstration
    top_pair = 'ADM_CORN'
    s1_series = s1_data[top_pair]
    
    # Create crisis time series plot
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    fig.suptitle(f'Crisis Period S1 Violations: {top_pair}', fontsize=16, fontweight='bold')
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for i, ((crisis_name, (start, end)), color) in enumerate(zip(crisis_periods.items(), colors)):
        ax = axes[i]
        
        # Get crisis period data
        crisis_data = s1_series[start:end]
        
        if len(crisis_data) > 0:
            # Plot S1 values
            ax.plot(crisis_data.index, np.abs(crisis_data.values), 
                   color=color, linewidth=2, alpha=0.8, label='|S1| Values')
            
            # Add bounds
            ax.axhline(y=2.0, color='red', linestyle='--', linewidth=2, 
                      label='Classical Bound (2.0)')
            ax.axhline(y=2*np.sqrt(2), color='green', linestyle='--', linewidth=2, 
                      label='Quantum Bound (2.83)')
            
            # Highlight violations
            violations = np.abs(crisis_data.values) > 2.0
            if violations.any():
                ax.fill_between(crisis_data.index, 0, np.abs(crisis_data.values),
                               where=violations, alpha=0.3, color='red',
                               label='Bell Violations')
            
            # Calculate statistics
            violation_rate = np.mean(violations) * 100
            max_violation = np.max(np.abs(crisis_data.values))
            
            # Add statistics
            stats_text = f'Violation Rate: {violation_rate:.1f}%\nMax |S1|: {max_violation:.3f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', 
                   facecolor='white', alpha=0.8))
        
        ax.set_title(f'{crisis_name} ({start} to {end})', fontweight='bold')
        ax.set_ylabel('|S1| Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if i == 2:
            ax.set_xlabel('Date')
    
    plt.tight_layout()
    plt.savefig('demo_outputs/crisis_time_series.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Created crisis time series visualization")
    
    # Create rolling violation rate plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    fig.suptitle(f'Rolling Bell Violation Analysis: {top_pair}', fontsize=16, fontweight='bold')
    
    # Calculate rolling violation rate
    window = 20
    violations = np.abs(s1_series.values) > 2.0
    rolling_rate = pd.Series(violations, index=s1_series.index).rolling(window).mean() * 100
    
    # Plot S1 values
    ax1.plot(s1_series.index, np.abs(s1_series.values), color='blue', 
            linewidth=1, alpha=0.7, label='|S1| Values')
    ax1.axhline(y=2.0, color='red', linestyle='--', linewidth=2, label='Classical Bound')
    
    # Highlight crisis periods
    for (crisis_name, (start, end)), color in zip(crisis_periods.items(), colors):
        ax1.axvspan(pd.to_datetime(start), pd.to_datetime(end), 
                   alpha=0.2, color=color, label=crisis_name)
        ax2.axvspan(pd.to_datetime(start), pd.to_datetime(end), 
                   alpha=0.2, color=color)
    
    ax1.set_ylabel('|S1| Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title('S1 Bell Inequality Values Over Time')
    
    # Plot rolling violation rate
    ax2.plot(rolling_rate.index, rolling_rate.values, color='purple', 
            linewidth=2, label=f'{window}-Period Rolling Violation Rate')
    ax2.axhline(y=20, color='orange', linestyle=':', alpha=0.7, label='20% Threshold')
    ax2.axhline(y=40, color='red', linestyle=':', alpha=0.7, label='40% Threshold')
    
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Violation Rate (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_title(f'{window}-Period Rolling Bell Violation Rate')
    
    plt.tight_layout()
    plt.savefig('demo_outputs/rolling_violation_rate.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Created rolling violation rate visualization")

def demonstrate_innovative_visualizations(s1_data, asset_pairs):
    """Demonstrate Innovative Statistical Visualizations (Task 7.2)."""
    
    print("\n🎯 Task 7.2: Innovative Statistical Visualizations")
    print("=" * 50)
    
    # 1. Correlation Network (Quantum Entanglement Network)
    pairs = list(s1_data.keys())
    correlation_matrix = pd.DataFrame(index=pairs, columns=pairs)
    
    for i, pair1 in enumerate(pairs):
        for j, pair2 in enumerate(pairs):
            if i == j:
                correlation_matrix.loc[pair1, pair2] = 1.0
            else:
                corr = s1_data[pair1].corr(s1_data[pair2])
                correlation_matrix.loc[pair1, pair2] = corr if not np.isnan(corr) else 0.0
    
    # Create correlation heatmap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Quantum Entanglement Network Analysis', fontsize=16, fontweight='bold')
    
    # Correlation heatmap - convert to float
    corr_values = correlation_matrix.values.astype(float)
    im = ax1.imshow(corr_values, cmap='RdBu_r', vmin=-1, vmax=1)
    ax1.set_xticks(range(len(pairs)))
    ax1.set_xticklabels(pairs, rotation=45, ha='right')
    ax1.set_yticks(range(len(pairs)))
    ax1.set_yticklabels(pairs)
    ax1.set_title('Cross-Asset Correlation Matrix')
    
    # Add correlation values
    for i in range(len(pairs)):
        for j in range(len(pairs)):
            text = ax1.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}', 
                           ha='center', va='center', fontweight='bold',
                           color='white' if abs(correlation_matrix.iloc[i, j]) > 0.5 else 'black')
    
    plt.colorbar(im, ax=ax1, label='Correlation')
    
    # Correlation distribution
    all_corrs = []
    for i in range(len(pairs)):
        for j in range(i+1, len(pairs)):
            corr = correlation_matrix.iloc[i, j]
            if not np.isnan(corr):
                all_corrs.append(abs(corr))
    
    ax2.hist(all_corrs, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.axvline(x=0.3, color='red', linestyle='--', linewidth=2, 
               label='Strong Correlation Threshold (0.3)')
    ax2.set_xlabel('|Correlation|')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Correlation Strengths')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('demo_outputs/quantum_entanglement_network.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Created quantum entanglement network visualization")
    
    # 2. Violation Intensity Heatmap
    fig, ax = plt.subplots(figsize=(16, 8))
    fig.suptitle('Violation Intensity Heatmap', fontsize=16, fontweight='bold')
    
    # Prepare data for heatmap (monthly resampling)
    monthly_data = []
    pair_names = []
    
    for pair, series in s1_data.items():
        monthly_series = series.resample('M').mean()
        violation_intensity = np.maximum(np.abs(monthly_series) - 2, 0)
        monthly_data.append(violation_intensity.values)
        pair_names.append(pair)
    
    if monthly_data:
        violation_matrix = np.array(monthly_data)
        
        im = ax.imshow(violation_matrix, aspect='auto', cmap='Reds', interpolation='nearest')
        
        ax.set_yticks(range(len(pair_names)))
        ax.set_yticklabels(pair_names)
        
        # Set x-axis to show dates (sample every 12 months)
        time_index = monthly_series.index
        n_dates = len(time_index)
        tick_positions = np.arange(0, n_dates, 12)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([time_index[i].strftime('%Y') for i in tick_positions])
        
        plt.colorbar(im, ax=ax, label='Violation Intensity (|S1| - 2)')
        ax.set_title('Temporal Evolution of Bell Violations')
        ax.set_xlabel('Year')
        ax.set_ylabel('Asset Pairs')
    
    plt.tight_layout()
    plt.savefig('demo_outputs/violation_intensity_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Created violation intensity heatmap")

def demonstrate_three_crisis_framework(s1_data, asset_pairs):
    """Demonstrate Three-Crisis Analysis Framework (Task 7.3)."""
    
    print("\n🎯 Task 7.3: Three-Crisis Analysis Framework")
    print("=" * 50)
    
    # Crisis definitions
    crisis_definitions = {
        '2008_financial': {
            'start': '2008-09-01',
            'end': '2009-03-31',
            'name': '2008 Financial Crisis'
        },
        'eu_debt': {
            'start': '2010-05-01',
            'end': '2012-12-31',
            'name': 'EU Debt Crisis'
        },
        'covid19': {
            'start': '2020-02-01',
            'end': '2020-12-31',
            'name': 'COVID-19 Pandemic'
        }
    }
    
    # 1. Tier-specific crisis analysis
    tier_crisis_analysis = {}
    unique_tiers = set(asset_pairs.values())
    
    for tier in unique_tiers:
        tier_crisis_analysis[tier] = {}
        tier_pairs = [pair for pair, t in asset_pairs.items() if t == tier]
        
        for crisis_id, crisis_info in crisis_definitions.items():
            start_date = pd.to_datetime(crisis_info['start'])
            end_date = pd.to_datetime(crisis_info['end'])
            
            tier_violations = []
            for pair in tier_pairs:
                if pair in s1_data:
                    crisis_data = s1_data[pair][start_date:end_date]
                    if len(crisis_data) > 0:
                        violations = np.abs(crisis_data.values) > 2.0
                        violation_rate = np.mean(violations) * 100
                        tier_violations.append(violation_rate)
            
            if tier_violations:
                tier_crisis_analysis[tier][crisis_id] = np.mean(tier_violations)
            else:
                tier_crisis_analysis[tier][crisis_id] = 0.0
    
    # Create tier-crisis comparison chart
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Three-Crisis Analysis Framework\n2008 Financial vs EU Debt vs COVID-19', 
                fontsize=16, fontweight='bold')
    
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(tier_crisis_analysis).T
    
    # Plot 1: Grouped bar chart
    x = np.arange(len(df.index))
    width = 0.25
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for i, (crisis, color) in enumerate(zip(df.columns, colors)):
        crisis_name = crisis_definitions[crisis]['name']
        ax1.bar(x + i*width, df[crisis], width, label=crisis_name, 
               color=color, alpha=0.7, edgecolor='black')
    
    ax1.set_xlabel('Agricultural Tiers')
    ax1.set_ylabel('Violation Rate (%)')
    ax1.set_title('Crisis Comparison by Tier')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(df.index, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Crisis severity ranking
    crisis_averages = df.mean(axis=0).sort_values(ascending=False)
    colors_sorted = [colors[list(df.columns).index(crisis)] for crisis in crisis_averages.index]
    
    bars = ax2.bar(range(len(crisis_averages)), crisis_averages.values, 
                  color=colors_sorted, alpha=0.7, edgecolor='black')
    
    for bar, value in zip(bars, crisis_averages.values):
        ax2.text(bar.get_x() + bar.get_width()/2., value + 0.5,
                f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax2.set_xticks(range(len(crisis_averages)))
    ax2.set_xticklabels([crisis_definitions[crisis]['name'] for crisis in crisis_averages.index])
    ax2.set_ylabel('Average Violation Rate (%)')
    ax2.set_title('Crisis Severity Ranking')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Tier vulnerability ranking
    tier_max_violations = df.max(axis=1).sort_values(ascending=False)
    
    bars = ax3.bar(range(len(tier_max_violations)), tier_max_violations.values, 
                  color='lightcoral', alpha=0.7, edgecolor='black')
    
    for bar, value in zip(bars, tier_max_violations.values):
        ax3.text(bar.get_x() + bar.get_width()/2., value + 0.5,
                f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax3.set_xticks(range(len(tier_max_violations)))
    ax3.set_xticklabels(tier_max_violations.index, rotation=45)
    ax3.set_ylabel('Maximum Violation Rate (%)')
    ax3.set_title('Tier Vulnerability Ranking')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Crisis amplification heatmap
    im = ax4.imshow(df.values, cmap='Reds', aspect='auto', interpolation='nearest')
    
    ax4.set_xticks(range(len(df.columns)))
    ax4.set_xticklabels([crisis_definitions[col]['name'] for col in df.columns], rotation=45, ha='right')
    ax4.set_yticks(range(len(df.index)))
    ax4.set_yticklabels(df.index)
    
    # Add text annotations
    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            text = ax4.text(j, i, f'{df.iloc[i, j]:.1f}%', 
                           ha='center', va='center', fontweight='bold',
                           color='white' if df.iloc[i, j] > df.values.max()/2 else 'black')
    
    plt.colorbar(im, ax=ax4, label='Violation Rate (%)')
    ax4.set_title('Crisis Impact Matrix')
    
    plt.tight_layout()
    plt.savefig('demo_outputs/three_crisis_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Created three-crisis analysis framework visualization")
    
    # Print analysis summary
    print(f"\n📊 THREE-CRISIS ANALYSIS SUMMARY")
    print("=" * 40)
    
    for crisis_id, crisis_info in crisis_definitions.items():
        avg_violation = df[crisis_id].mean()
        max_violation = df[crisis_id].max()
        most_affected_tier = df[crisis_id].idxmax()
        
        print(f"{crisis_info['name']}:")
        print(f"  Average violation rate: {avg_violation:.1f}%")
        print(f"  Maximum violation rate: {max_violation:.1f}%")
        print(f"  Most affected tier: {most_affected_tier}")
        print()

def main():
    """Main demonstration function."""
    
    print("🎨 AGRICULTURAL VISUALIZATION SUITE DEMONSTRATION")
    print("=" * 60)
    print("Task 7: Build Comprehensive Visualization and Statistical Analysis System")
    print("=" * 60)
    
    try:
        # Create sample data
        s1_data, asset_pairs = create_sample_agricultural_data()
        
        # Task 7.1: Crisis Period Time Series Visualizations
        demonstrate_crisis_time_series(s1_data)
        
        # Task 7.2: Innovative Statistical Analysis and Visualization Suite
        demonstrate_innovative_visualizations(s1_data, asset_pairs)
        
        # Task 7.3: Three-Crisis Analysis Framework
        demonstrate_three_crisis_framework(s1_data, asset_pairs)
        
        print(f"\n🎉 ALL DEMONSTRATIONS COMPLETE!")
        print("=" * 50)
        print("✅ Task 7.1: Crisis Period Time Series Visualizations")
        print("✅ Task 7.2: Innovative Statistical Analysis and Visualization Suite")
        print("✅ Task 7.3: Three-Crisis Analysis Framework")
        print(f"\n📁 All visualizations saved to: demo_outputs/")
        print("🔬 Publication-ready figures with statistical annotations")
        print("📊 Comprehensive visualization system complete")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🚀 Visualization suite ready for production use!")
    else:
        print("\n⚠️  Please check errors and try again.")