#!/usr/bin/env python3
"""
COMPREHENSIVE VISUALIZATION SUITE DEMONSTRATION
==============================================

This script demonstrates the complete agricultural cross-sector visualization
suite including crisis period time series, innovative statistical visualizations,
and three-crisis analysis framework.

Features Demonstrated:
- Crisis Period Time Series Visualizations (Task 7.1)
- Innovative Statistical Analysis and Visualization Suite (Task 7.2)  
- Three-Crisis Analysis Framework (Task 7.3)
- Publication-ready figures with statistical annotations
- Comprehensive visualization system integration

Authors: Agricultural Cross-Sector Analysis Team
Date: September 2025
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import warnings

# Import our visualization suite
from src.agricultural_visualization_suite import (
    CrisisPeriodTimeSeriesVisualizer,
    InnovativeStatisticalVisualizer, 
    ComprehensiveVisualizationSuite,
    ThreeCrisisAnalysisFramework
)

warnings.filterwarnings('ignore')

def create_realistic_agricultural_data():
    """Create realistic agricultural cross-sector data for demonstration."""
    
    print("🌾 CREATING REALISTIC AGRICULTURAL DATA")
    print("=" * 50)
    
    # Extended date range covering all three crisis periods
    dates = pd.date_range('2007-01-01', '2022-12-31', freq='D')
    
    # Define agricultural asset pairs with realistic relationships
    asset_pairs = {
        # Food Companies
        'ADM_CORN': 'Food Processing',
        'CAG_SOYB': 'Food Processing', 
        'GIS_WEAT': 'Food Processing',
        'KHC_RICE': 'Food Processing',
        
        # Fertilizer Companies  
        'CF_CORN': 'Fertilizer',
        'MOS_SOYB': 'Fertilizer',
        'NTR_WEAT': 'Fertilizer',
        
        # Farm Equipment
        'DE_DBA': 'Equipment',
        'AGCO_CORN': 'Equipment',
        
        # Food Retail
        'WMT_FOOD': 'Retail',
        'KR_FOOD': 'Retail'
    }
    
    # Crisis periods with different characteristics
    crisis_periods = {
        '2008_financial': {
            'start': '2008-09-01',
            'end': '2009-03-31',
            'intensity': 2.5,  # High intensity
            'volatility': 1.8
        },
        'eu_debt': {
            'start': '2010-05-01', 
            'end': '2012-12-31',
            'intensity': 1.8,  # Medium intensity
            'volatility': 1.4
        },
        'covid19': {
            'start': '2020-02-01',
            'end': '2020-12-31', 
            'intensity': 3.0,  # Very high intensity
            'volatility': 2.2
        }
    }
    
    s1_data = {}
    np.random.seed(42)  # For reproducible results
    
    for i, (pair, sector) in enumerate(asset_pairs.items()):
        print(f"  Generating data for {pair} ({sector})")
        
        # Base S1 values with sector-specific characteristics
        if sector == 'Food Processing':
            base_mean = 1.6
            base_std = 0.4
        elif sector == 'Fertilizer':
            base_mean = 1.4
            base_std = 0.5
        elif sector == 'Equipment':
            base_mean = 1.3
            base_std = 0.3
        else:  # Retail
            base_mean = 1.2
            base_std = 0.3
        
        # Generate base time series
        base_values = np.random.normal(base_mean, base_std, len(dates))
        
        # Add seasonal patterns (agricultural cycles)
        seasonal_component = 0.2 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
        base_values += seasonal_component
        
        # Add crisis amplification
        for crisis_id, crisis_info in crisis_periods.items():
            start_date = pd.to_datetime(crisis_info['start'])
            end_date = pd.to_datetime(crisis_info['end'])
            
            crisis_mask = (dates >= start_date) & (dates <= end_date)
            
            # Different sectors respond differently to crises
            if sector == 'Food Processing' and crisis_id == 'covid19':
                # Food processing highly affected by COVID-19
                amplification = crisis_info['intensity'] * 1.5
            elif sector == 'Fertilizer' and crisis_id == '2008_financial':
                # Fertilizer companies affected by financial crisis
                amplification = crisis_info['intensity'] * 1.3
            elif sector == 'Equipment' and crisis_id == 'eu_debt':
                # Equipment companies affected by EU debt crisis
                amplification = crisis_info['intensity'] * 1.2
            else:
                amplification = crisis_info['intensity']
            
            # Apply crisis amplification with gradual onset/recovery
            crisis_indices = np.where(crisis_mask)[0]
            if len(crisis_indices) > 0:
                # Gradual onset (first 20% of crisis period)
                onset_length = max(1, len(crisis_indices) // 5)
                for j in range(onset_length):
                    idx = crisis_indices[j]
                    factor = amplification * (j + 1) / onset_length
                    base_values[idx] *= factor
                
                # Peak crisis (middle 60% of crisis period)
                peak_start = onset_length
                peak_end = len(crisis_indices) - onset_length
                for j in range(peak_start, peak_end):
                    idx = crisis_indices[j]
                    # Add volatility during peak crisis
                    volatility_factor = 1 + np.random.normal(0, crisis_info['volatility'] * 0.1)
                    base_values[idx] *= amplification * volatility_factor
                
                # Gradual recovery (last 20% of crisis period)
                for j in range(peak_end, len(crisis_indices)):
                    idx = crisis_indices[j]
                    recovery_progress = (j - peak_end) / onset_length
                    factor = amplification * (1 - recovery_progress * 0.5)
                    base_values[idx] *= factor
        
        # Ensure no negative values
        base_values = np.maximum(base_values, 0.1)
        
        s1_data[pair] = pd.Series(base_values, index=dates)
    
    print(f"✅ Generated data for {len(asset_pairs)} asset pairs")
    print(f"📅 Date range: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
    
    return s1_data, asset_pairs

def create_transmission_data(s1_data):
    """Create transmission data showing cross-sector correlations."""
    
    print("\n🔄 CREATING TRANSMISSION DATA")
    print("=" * 30)
    
    transmission_data = {}
    
    # Define transmission pairs (source -> target)
    transmission_pairs = {
        'energy_to_agriculture': ['CF_CORN', 'MOS_SOYB'],
        'transport_to_agriculture': ['DE_DBA', 'AGCO_CORN'],
        'finance_to_food': ['WMT_FOOD', 'KR_FOOD'],
        'processing_to_retail': ['ADM_CORN', 'WMT_FOOD']
    }
    
    for transmission_type, pair_list in transmission_pairs.items():
        if len(pair_list) >= 2 and all(p in s1_data for p in pair_list):
            source_data = s1_data[pair_list[0]]
            target_data = s1_data[pair_list[1]]
            
            # Calculate rolling correlation
            window = 30
            correlation = source_data.rolling(window).corr(target_data)
            
            transmission_data[transmission_type] = pd.DataFrame({
                'correlation': correlation,
                'source': source_data,
                'target': target_data
            })
            
            print(f"  {transmission_type}: {pair_list[0]} → {pair_list[1]}")
    
    print(f"✅ Created {len(transmission_data)} transmission relationships")
    
    return transmission_data

def demonstrate_crisis_visualizations():
    """Demonstrate Crisis Period Time Series Visualizations (Task 7.1)."""
    
    print("\n" + "="*60)
    print("🎯 TASK 7.1: CRISIS PERIOD TIME SERIES VISUALIZATIONS")
    print("="*60)
    
    # Create data
    s1_data, asset_pairs = create_realistic_agricultural_data()
    transmission_data = create_transmission_data(s1_data)
    
    # Initialize visualizer
    crisis_viz = CrisisPeriodTimeSeriesVisualizer(figsize=(16, 12), dpi=150)
    
    # Create output directory
    os.makedirs('demo_visualizations/crisis_timeseries', exist_ok=True)
    
    # 1. Crisis S1 time series for top pair
    print("\n1️⃣ Creating Crisis S1 Time Series...")
    top_pair = 'ADM_CORN'
    
    # Extract crisis period data
    crisis_s1_data = {}
    for crisis_id in ['2008_financial', 'eu_debt', 'covid19']:
        if crisis_id == '2008_financial':
            crisis_s1_data[crisis_id] = s1_data[top_pair]['2008-09-01':'2009-03-31']
        elif crisis_id == 'eu_debt':
            crisis_s1_data[crisis_id] = s1_data[top_pair]['2010-05-01':'2012-12-31']
        elif crisis_id == 'covid19':
            crisis_s1_data[crisis_id] = s1_data[top_pair]['2020-02-01':'2020-12-31']
    
    fig1 = crisis_viz.create_crisis_s1_time_series(
        crisis_s1_data, top_pair, 
        'demo_visualizations/crisis_timeseries/crisis_s1_timeseries.png'
    )
    plt.close(fig1)
    
    # 2. Rolling violation rate series
    print("2️⃣ Creating Rolling Violation Rate Series...")
    fig2 = crisis_viz.create_rolling_violation_rate_series(
        s1_data[top_pair], pair_name=top_pair,
        save_path='demo_visualizations/crisis_timeseries/rolling_violations.png'
    )
    plt.close(fig2)
    
    # 3. Transmission propagation series
    print("3️⃣ Creating Transmission Propagation Series...")
    fig3 = crisis_viz.create_transmission_propagation_series(
        transmission_data,
        'demo_visualizations/crisis_timeseries/transmission_propagation.png'
    )
    plt.close(fig3)
    
    # 4. Crisis onset detection
    print("4️⃣ Creating Crisis Onset Detection...")
    fig4 = crisis_viz.create_crisis_onset_detection(
        s1_data[top_pair],
        save_path='demo_visualizations/crisis_timeseries/crisis_onset_detection.png'
    )
    plt.close(fig4)
    
    # 5. Tier-specific crisis comparison
    print("5️⃣ Creating Tier-Specific Crisis Comparison...")
    
    # Group data by tiers
    tier_crisis_data = {}
    for tier in set(asset_pairs.values()):
        tier_crisis_data[tier] = {}
        tier_pairs = [pair for pair, t in asset_pairs.items() if t == tier]
        
        if tier_pairs:
            # Use first pair from each tier for demonstration
            representative_pair = tier_pairs[0]
            
            tier_crisis_data[tier]['2008_financial'] = s1_data[representative_pair]['2008-09-01':'2009-03-31']
            tier_crisis_data[tier]['eu_debt'] = s1_data[representative_pair]['2010-05-01':'2012-12-31'] 
            tier_crisis_data[tier]['covid19'] = s1_data[representative_pair]['2020-02-01':'2020-12-31']
    
    fig5 = crisis_viz.create_tier_specific_crisis_comparison(
        tier_crisis_data,
        'demo_visualizations/crisis_timeseries/tier_crisis_comparison.png'
    )
    plt.close(fig5)
    
    # 6. Seasonal overlay analysis
    print("6️⃣ Creating Seasonal Overlay Analysis...")
    fig6 = crisis_viz.create_seasonal_overlay_analysis(
        s1_data[top_pair],
        'demo_visualizations/crisis_timeseries/seasonal_analysis.png'
    )
    plt.close(fig6)
    
    print("✅ Task 7.1 demonstrations complete!")
    return s1_data, asset_pairs, transmission_data

def demonstrate_innovative_visualizations(s1_data, asset_pairs, transmission_data):
    """Demonstrate Innovative Statistical Analysis and Visualization Suite (Task 7.2)."""
    
    print("\n" + "="*60)
    print("🎯 TASK 7.2: INNOVATIVE STATISTICAL VISUALIZATIONS")
    print("="*60)
    
    # Initialize visualizer
    innovative_viz = InnovativeStatisticalVisualizer(figsize=(16, 12), dpi=150)
    
    # Create output directory
    os.makedirs('demo_visualizations/innovative', exist_ok=True)
    
    # 1. Quantum entanglement network
    print("\n1️⃣ Creating Quantum Entanglement Network...")
    
    # Create correlation matrix
    pairs = list(s1_data.keys())
    correlation_matrix = pd.DataFrame(index=pairs, columns=pairs)
    
    for i, pair1 in enumerate(pairs):
        for j, pair2 in enumerate(pairs):
            if i == j:
                correlation_matrix.loc[pair1, pair2] = 1.0
            else:
                # Calculate correlation between S1 time series
                corr = s1_data[pair1].corr(s1_data[pair2])
                correlation_matrix.loc[pair1, pair2] = corr if not np.isnan(corr) else 0.0
    
    fig1 = innovative_viz.create_quantum_entanglement_network(
        correlation_matrix, threshold=0.3,
        save_path='demo_visualizations/innovative/quantum_entanglement_network.png'
    )
    plt.close(fig1)
    
    # 2. Crisis contagion map
    print("2️⃣ Creating Crisis Contagion Map...")
    
    # Prepare tier violation data
    tier_violation_data = {}
    for tier in set(asset_pairs.values()):
        tier_pairs = [pair for pair, t in asset_pairs.items() if t == tier]
        
        if tier_pairs:
            # Combine data from all pairs in tier
            tier_data = pd.concat([s1_data[pair] for pair in tier_pairs], axis=1)
            tier_data.columns = tier_pairs
            tier_violation_data[tier] = tier_data
    
    fig2 = innovative_viz.create_crisis_contagion_map(
        tier_violation_data,
        'demo_visualizations/innovative/crisis_contagion_map.png'
    )
    plt.close(fig2)
    
    # 3. Transmission velocity analysis
    print("3️⃣ Creating Transmission Velocity Analysis...")
    fig3 = innovative_viz.create_transmission_velocity_analysis(
        transmission_data,
        'demo_visualizations/innovative/transmission_velocity.png'
    )
    plt.close(fig3)
    
    # 4. Violation intensity heatmap
    print("4️⃣ Creating Violation Intensity Heatmap...")
    fig4 = innovative_viz.create_violation_intensity_heatmap(
        s1_data,
        'demo_visualizations/innovative/violation_intensity_heatmap.png'
    )
    plt.close(fig4)
    
    # 5. Tier sensitivity radar chart
    print("5️⃣ Creating Tier Sensitivity Radar Chart...")
    
    # Calculate tier crisis sensitivity
    tier_crisis_sensitivity = {}
    crisis_periods = {
        '2008 Financial': ('2008-09-01', '2009-03-31'),
        'EU Debt': ('2010-05-01', '2012-12-31'),
        'COVID-19': ('2020-02-01', '2020-12-31')
    }
    
    for tier in set(asset_pairs.values()):
        tier_crisis_sensitivity[tier] = {}
        tier_pairs = [pair for pair, t in asset_pairs.items() if t == tier]
        
        for crisis_name, (start, end) in crisis_periods.items():
            tier_violations = []
            
            for pair in tier_pairs:
                crisis_data = s1_data[pair][start:end]
                if len(crisis_data) > 0:
                    violations = np.abs(crisis_data.values) > 2.0
                    violation_rate = np.mean(violations) * 100
                    tier_violations.append(violation_rate)
            
            if tier_violations:
                tier_crisis_sensitivity[tier][crisis_name] = np.mean(tier_violations)
            else:
                tier_crisis_sensitivity[tier][crisis_name] = 0.0
    
    fig5 = innovative_viz.create_tier_sensitivity_radar_chart(
        tier_crisis_sensitivity,
        'demo_visualizations/innovative/tier_sensitivity_radar.png'
    )
    plt.close(fig5)
    
    print("✅ Task 7.2 demonstrations complete!")
    return tier_violation_data, tier_crisis_sensitivity

def demonstrate_three_crisis_framework(s1_data, asset_pairs):
    """Demonstrate Three-Crisis Analysis Framework (Task 7.3)."""
    
    print("\n" + "="*60)
    print("🎯 TASK 7.3: THREE-CRISIS ANALYSIS FRAMEWORK")
    print("="*60)
    
    # Initialize framework
    crisis_framework = ThreeCrisisAnalysisFramework()
    
    # Create output directory
    os.makedirs('demo_visualizations/three_crisis', exist_ok=True)
    
    # 1. Create crisis period definitions
    print("\n1️⃣ Creating Crisis Period Definitions...")
    crisis_df = crisis_framework.create_crisis_period_definitions(
        'demo_visualizations/three_crisis/crisis_definitions.csv'
    )
    
    # 2. Tier-specific crisis analysis
    print("2️⃣ Implementing Tier-Specific Crisis Analysis...")
    tier_crisis_analysis = crisis_framework.implement_tier_specific_crisis_analysis(
        s1_data, asset_pairs,
        'demo_visualizations/three_crisis/tier_crisis_analysis.csv'
    )
    
    # 3. Crisis amplification metrics
    print("3️⃣ Creating Crisis Amplification Metrics...")
    amplification_metrics = crisis_framework.create_crisis_amplification_metrics(
        s1_data,
        'demo_visualizations/three_crisis/amplification_metrics.csv'
    )
    
    # 4. Statistical significance testing
    print("4️⃣ Adding Statistical Significance Testing...")
    significance_results = crisis_framework.add_statistical_significance_testing(
        s1_data,
        'demo_visualizations/three_crisis/significance_results.csv'
    )
    
    # 5. Cross-crisis comparison analysis
    print("5️⃣ Implementing Cross-Crisis Comparison Analysis...")
    cross_crisis_comparison = crisis_framework.implement_cross_crisis_comparison_analysis(
        amplification_metrics, asset_pairs,
        'demo_visualizations/three_crisis/cross_crisis_comparison.csv'
    )
    
    # 6. Crisis recovery analysis
    print("6️⃣ Adding Crisis Recovery Analysis...")
    recovery_analysis = crisis_framework.add_crisis_recovery_analysis(
        s1_data, recovery_window=90,
        save_path='demo_visualizations/three_crisis/recovery_analysis.csv'
    )
    
    print("✅ Task 7.3 demonstrations complete!")
    
    return {
        'crisis_definitions': crisis_df,
        'tier_crisis_analysis': tier_crisis_analysis,
        'amplification_metrics': amplification_metrics,
        'significance_results': significance_results,
        'cross_crisis_comparison': cross_crisis_comparison,
        'recovery_analysis': recovery_analysis
    }

def demonstrate_comprehensive_suite():
    """Demonstrate the complete Comprehensive Visualization Suite."""
    
    print("\n" + "="*60)
    print("🎯 COMPREHENSIVE VISUALIZATION SUITE INTEGRATION")
    print("="*60)
    
    # Create comprehensive data
    s1_data, asset_pairs = create_realistic_agricultural_data()
    transmission_data = create_transmission_data(s1_data)
    
    # Initialize comprehensive suite
    comprehensive_suite = ComprehensiveVisualizationSuite(figsize=(20, 16), dpi=150)
    
    # Create output directory
    os.makedirs('demo_visualizations/comprehensive', exist_ok=True)
    
    # Prepare comprehensive analysis data
    analysis_data = {
        's1_data': s1_data,
        'crisis_s1_data': {
            pair: {
                '2008_financial': series['2008-09-01':'2009-03-31'],
                'eu_debt': series['2010-05-01':'2012-12-31'],
                'covid19': series['2020-02-01':'2020-12-31']
            } for pair, series in s1_data.items()
        },
        'transmission_data': transmission_data,
        'correlation_matrix': pd.DataFrame(
            np.random.rand(len(s1_data), len(s1_data)), 
            index=list(s1_data.keys()), 
            columns=list(s1_data.keys())
        ),
        'sector_tier_data': {
            sector: {
                'Tier1': np.random.uniform(15, 45),
                'Tier2': np.random.uniform(10, 35), 
                'Tier3': np.random.uniform(5, 25)
            } for sector in set(asset_pairs.values())
        },
        'three_crisis_data': {
            '2008_financial': {tier: np.random.uniform(20, 50) for tier in set(asset_pairs.values())},
            'eu_debt': {tier: np.random.uniform(15, 40) for tier in set(asset_pairs.values())},
            'covid19': {tier: np.random.uniform(25, 60) for tier in set(asset_pairs.values())}
        }
    }
    
    # 1. Sector-specific heatmaps
    print("\n1️⃣ Creating Sector-Specific Heatmaps...")
    fig1 = comprehensive_suite.create_sector_specific_heatmaps(
        analysis_data['sector_tier_data'],
        'demo_visualizations/comprehensive/sector_heatmaps.png'
    )
    plt.close(fig1)
    
    # 2. Three-crisis comparison charts
    print("2️⃣ Creating Three-Crisis Comparison Charts...")
    fig2 = comprehensive_suite.create_three_crisis_comparison_charts(
        analysis_data['three_crisis_data'],
        'demo_visualizations/comprehensive/three_crisis_comparison.png'
    )
    plt.close(fig2)
    
    # 3. Publication-ready summary
    print("3️⃣ Creating Publication-Ready Summary...")
    
    # Prepare analysis results for publication summary
    analysis_results = {
        'violation_rates': {pair: np.random.uniform(10, 50) for pair in list(s1_data.keys())[:10]},
        'statistical_significance': {pair: np.random.uniform(0.0001, 0.01) for pair in list(s1_data.keys())[:10]},
        'crisis_amplification': {crisis: np.random.uniform(1.5, 3.0) for crisis in ['2008_financial', 'eu_debt', 'covid19']}
    }
    
    analysis_data['analysis_results'] = analysis_results
    
    fig3 = comprehensive_suite.create_publication_ready_summary(
        analysis_results,
        'demo_visualizations/comprehensive/publication_summary.png'
    )
    plt.close(fig3)
    
    # 4. Generate all visualizations
    print("4️⃣ Generating All Visualizations...")
    generated_files = comprehensive_suite.generate_all_visualizations(
        analysis_data, 'demo_visualizations/comprehensive/all_visualizations'
    )
    
    # 5. Get summary
    summary = comprehensive_suite.get_visualization_summary()
    
    print(f"\n📊 COMPREHENSIVE SUITE SUMMARY")
    print("=" * 40)
    print(f"Total figures generated: {summary['total_figures']}")
    print(f"Visualization types: {len(summary['visualization_types'])}")
    print(f"Publication ready: {summary['publication_ready']}")
    print(f"Statistical annotations: {summary['statistical_annotations']}")
    print(f"Crisis periods covered: {len(summary['crisis_periods_covered'])}")
    
    return summary

def main():
    """Main demonstration function."""
    
    print("🎨 AGRICULTURAL CROSS-SECTOR VISUALIZATION SUITE")
    print("=" * 60)
    print("Complete demonstration of Task 7: Build Comprehensive")
    print("Visualization and Statistical Analysis System")
    print("=" * 60)
    
    try:
        # Task 7.1: Crisis Period Time Series Visualizations
        s1_data, asset_pairs, transmission_data = demonstrate_crisis_visualizations()
        
        # Task 7.2: Innovative Statistical Analysis and Visualization Suite
        tier_violation_data, tier_crisis_sensitivity = demonstrate_innovative_visualizations(
            s1_data, asset_pairs, transmission_data
        )
        
        # Task 7.3: Three-Crisis Analysis Framework
        crisis_analysis_results = demonstrate_three_crisis_framework(s1_data, asset_pairs)
        
        # Comprehensive Integration
        comprehensive_summary = demonstrate_comprehensive_suite()
        
        print(f"\n🎉 ALL DEMONSTRATIONS COMPLETE!")
        print("=" * 50)
        print("✅ Task 7.1: Crisis Period Time Series Visualizations")
        print("✅ Task 7.2: Innovative Statistical Analysis and Visualization Suite")
        print("✅ Task 7.3: Three-Crisis Analysis Framework")
        print("✅ Comprehensive Visualization Suite Integration")
        print(f"\n📁 All visualizations saved to: demo_visualizations/")
        print(f"📊 Total visualization types: {len(comprehensive_summary['visualization_types'])}")
        print(f"🔬 Publication ready: {comprehensive_summary['publication_ready']}")
        
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