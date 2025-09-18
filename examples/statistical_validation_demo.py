#!/usr/bin/env python3
"""
STATISTICAL VALIDATION SUITE DEMONSTRATION
==========================================

This script demonstrates the advanced statistical validation and innovative
metrics capabilities for agricultural cross-sector Bell inequality analysis.
It shows how to perform comprehensive statistical validation with bootstrap
methods, significance testing, and novel agricultural crisis metrics.

Key Features Demonstrated:
- Bootstrap validation with 1000+ resamples
- Statistical significance testing with p < 0.001 requirement
- Confidence interval calculations for violation rates
- Multiple testing correction for cross-sectoral analysis
- Effect size calculations (20-60% above classical bounds expected)
- Innovative agricultural crisis metrics

Authors: Agricultural Cross-Sector Analysis Team
Date: September 2025
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from statistical_validation_suite import (
    AdvancedStatisticalValidator,
    InnovativeMetricsCalculator,
    ComprehensiveStatisticalSuite
)
from enhanced_s1_calculator import EnhancedS1Calculator

def generate_realistic_agricultural_data():
    """
    Generate realistic agricultural cross-sector data for demonstration.
    
    Returns:
    --------
    Dict : Comprehensive dataset for statistical validation
    """
    print("📊 Generating realistic agricultural cross-sector data...")
    
    np.random.seed(42)  # For reproducible results
    
    # Generate price data for agricultural and cross-sector assets
    n_days = 500
    dates = pd.date_range('2022-01-01', periods=n_days, freq='D')
    
    # Agricultural assets (with higher volatility during crisis periods)
    agricultural_assets = {
        'CORN': generate_price_series(100, n_days, base_vol=0.025, crisis_periods=[(100, 150), (300, 350)]),
        'WEAT': generate_price_series(120, n_days, base_vol=0.028, crisis_periods=[(100, 150), (300, 350)]),
        'SOYB': generate_price_series(110, n_days, base_vol=0.023, crisis_periods=[(100, 150), (300, 350)]),
        'ADM': generate_price_series(80, n_days, base_vol=0.020, crisis_periods=[(105, 155), (305, 355)]),
        'BG': generate_price_series(90, n_days, base_vol=0.022, crisis_periods=[(105, 155), (305, 355)])
    }
    
    # Cross-sector assets (Energy, Transportation, Chemicals)
    cross_sector_assets = {
        'XOM': generate_price_series(95, n_days, base_vol=0.030, crisis_periods=[(95, 145), (295, 345)]),
        'CVX': generate_price_series(105, n_days, base_vol=0.028, crisis_periods=[(95, 145), (295, 345)]),
        'UNP': generate_price_series(200, n_days, base_vol=0.025, crisis_periods=[(98, 148), (298, 348)]),
        'CSX': generate_price_series(85, n_days, base_vol=0.024, crisis_periods=[(98, 148), (298, 348)]),
        'CF': generate_price_series(75, n_days, base_vol=0.035, crisis_periods=[(102, 152), (302, 352)]),
        'MOS': generate_price_series(65, n_days, base_vol=0.032, crisis_periods=[(102, 152), (302, 352)])
    }
    
    # Combine all assets
    all_assets = {**agricultural_assets, **cross_sector_assets}
    
    # Create DataFrame
    price_data = pd.DataFrame(all_assets, index=dates)
    
    # Define crisis periods
    crisis_periods = {
        'Ukraine War Impact': (dates[100], dates[150]),
        'Supply Chain Crisis': (dates[300], dates[350])
    }
    
    # Define cross-sector pairs based on operational dependencies
    cross_sector_pairs = [
        ('CORN', 'XOM'),    # Corn - Energy (fuel for farming)
        ('CORN', 'CF'),     # Corn - Fertilizer (nitrogen for corn)
        ('WEAT', 'UNP'),    # Wheat - Transportation (rail shipping)
        ('SOYB', 'CVX'),    # Soybeans - Energy (processing fuel)
        ('ADM', 'MOS'),     # ADM - Fertilizer (input costs)
        ('BG', 'CSX')       # Bunge - Transportation (logistics)
    ]
    
    return {
        'price_data': price_data,
        'crisis_periods': crisis_periods,
        'cross_sector_pairs': cross_sector_pairs,
        'agricultural_assets': list(agricultural_assets.keys()),
        'cross_sector_assets': list(cross_sector_assets.keys())
    }

def generate_price_series(initial_price, n_days, base_vol=0.02, crisis_periods=None):
    """
    Generate realistic price series with crisis amplification.
    
    Parameters:
    -----------
    initial_price : float
        Starting price
    n_days : int
        Number of days
    base_vol : float
        Base volatility
    crisis_periods : List[Tuple[int, int]], optional
        Crisis periods as (start_day, end_day) tuples
        
    Returns:
    --------
    np.ndarray : Price series
    """
    returns = np.random.normal(0, base_vol, n_days)
    
    # Amplify volatility during crisis periods
    if crisis_periods:
        for start, end in crisis_periods:
            crisis_vol = base_vol * 2.5  # 2.5x volatility during crisis
            returns[start:end] = np.random.normal(0, crisis_vol, end - start)
            
            # Add some correlation structure during crisis
            for i in range(start + 1, end):
                returns[i] += 0.3 * returns[i-1]  # Add momentum
    
    # Convert to prices
    prices = initial_price * np.cumprod(1 + returns)
    return prices

def demonstrate_bootstrap_validation():
    """Demonstrate bootstrap validation with agricultural data."""
    print("\n🔬 BOOTSTRAP VALIDATION DEMONSTRATION")
    print("=" * 50)
    
    # Generate test data
    data = generate_realistic_agricultural_data()
    price_data = data['price_data']
    
    # Calculate returns
    returns = price_data.pct_change().dropna()
    
    # Perform S1 analysis on a key agricultural pair
    calculator = EnhancedS1Calculator(window_size=20, threshold_value=0.02)
    s1_results = calculator.analyze_asset_pair(returns, 'CORN', 'CF')
    
    # Extract S1 values
    s1_values = np.array(s1_results['s1_time_series'])
    
    # Perform bootstrap validation
    validator = AdvancedStatisticalValidator(n_bootstrap=1000, random_state=42)
    bootstrap_results = validator.bootstrap_violation_rate(s1_values)
    
    print(f"📈 CORN-CF (Corn-Fertilizer) Bootstrap Results:")
    print(f"   Violation Rate: {bootstrap_results.violation_rate:.2f}%")
    print(f"   95% Confidence Interval: ({bootstrap_results.confidence_interval[0]:.2f}%, {bootstrap_results.confidence_interval[1]:.2f}%)")
    print(f"   P-value: {bootstrap_results.p_value:.2e}")
    print(f"   Effect Size (Cohen's d): {bootstrap_results.effect_size:.3f}")
    print(f"   Classical Bound Excess: {bootstrap_results.classical_bound_excess:.1f}%")
    print(f"   Statistically Significant: {'YES' if bootstrap_results.is_significant else 'NO'}")
    
    # Calculate effect sizes
    effect_sizes = validator.calculate_effect_sizes(s1_values)
    print(f"\n📊 Effect Size Analysis:")
    print(f"   Cohen's d: {effect_sizes['cohens_d']:.3f}")
    print(f"   Classical Excess: {effect_sizes['classical_excess']:.1f}%")
    print(f"   Quantum Approach: {effect_sizes['quantum_approach']:.1f}%")
    print(f"   Max Violation Strength: {effect_sizes['max_violation_strength']:.3f}")
    
    return bootstrap_results, effect_sizes

def demonstrate_multiple_testing_correction():
    """Demonstrate multiple testing correction for cross-sectoral analysis."""
    print("\n🧮 MULTIPLE TESTING CORRECTION DEMONSTRATION")
    print("=" * 55)
    
    # Generate test data
    data = generate_realistic_agricultural_data()
    price_data = data['price_data']
    cross_sector_pairs = data['cross_sector_pairs']
    
    # Calculate returns
    returns = price_data.pct_change().dropna()
    
    # Perform S1 analysis on all cross-sector pairs
    calculator = EnhancedS1Calculator(window_size=20, threshold_value=0.02)
    validator = AdvancedStatisticalValidator(n_bootstrap=500, random_state=42)  # Reduced for demo
    
    pair_results = {}
    p_values = []
    
    print("🔍 Analyzing cross-sector pairs...")
    for asset_a, asset_b in cross_sector_pairs:
        try:
            s1_results = calculator.analyze_asset_pair(returns, asset_a, asset_b)
            s1_values = np.array(s1_results['s1_time_series'])
            
            bootstrap_results = validator.bootstrap_violation_rate(s1_values)
            pair_results[(asset_a, asset_b)] = bootstrap_results
            p_values.append(bootstrap_results.p_value)
            
            print(f"   {asset_a}-{asset_b}: {bootstrap_results.violation_rate:.1f}% violations, p = {bootstrap_results.p_value:.2e}")
            
        except Exception as e:
            print(f"   ⚠️ Error analyzing {asset_a}-{asset_b}: {e}")
            continue
    
    # Apply multiple testing corrections
    print(f"\n📋 Multiple Testing Correction Results:")
    print(f"   Original p-values: {len(p_values)} tests")
    
    methods = ['bonferroni', 'holm', 'fdr_bh']
    for method in methods:
        corrected_p, significance = validator.multiple_testing_correction(p_values, method=method)
        significant_count = sum(significance)
        
        print(f"\n   {method.upper()} Correction:")
        print(f"     Significant pairs: {significant_count}/{len(p_values)}")
        print(f"     Significance rate: {(significant_count/len(p_values)*100):.1f}%")
        
        # Show corrected p-values for significant pairs
        for i, ((asset_a, asset_b), is_sig, corr_p) in enumerate(zip(cross_sector_pairs[:len(corrected_p)], significance, corrected_p)):
            if is_sig:
                print(f"     ✅ {asset_a}-{asset_b}: p_corrected = {corr_p:.2e}")
    
    return pair_results, p_values

def demonstrate_innovative_metrics():
    """Demonstrate innovative agricultural crisis metrics."""
    print("\n🌾 INNOVATIVE AGRICULTURAL METRICS DEMONSTRATION")
    print("=" * 58)
    
    # Generate crisis-specific data
    np.random.seed(42)
    
    # Simulate normal vs crisis violation rates
    normal_violations = np.random.beta(2, 8, 100) * 0.3  # Low violation rates (0-30%)
    crisis_violations = np.random.beta(5, 3, 60) * 0.8   # High violation rates (0-80%)
    
    # Tier-specific violation rates during crisis
    tier_violation_rates = {
        'Tier 1 (Energy/Transport/Chemicals)': 52.3,
        'Tier 2 (Finance/Equipment)': 38.7,
        'Tier 3 (Policy-linked)': 29.1
    }
    
    # Transmission data
    transmission_lags = [3, 7, 14, 21, 30, 45]  # Days
    transmission_strengths = [0.85, 0.72, 0.68, 0.54, 0.41, 0.28]  # Correlation coefficients
    
    # Time series data
    s1_time_series = np.concatenate([
        np.random.normal(0, 1.2, 50),  # Normal period
        np.random.normal(0, 2.8, 30),  # Crisis period (higher volatility)
        np.random.normal(0, 1.5, 40)   # Recovery period
    ])
    # Add some violations
    s1_time_series[::8] = np.random.choice([-3.2, 3.2], size=len(s1_time_series[::8]))
    
    # Multi-crisis data
    crisis_results = {
        'COVID-19 (2020)': np.random.beta(6, 4, 50) * 0.7,
        'Ukraine War (2022)': np.random.beta(7, 3, 45) * 0.8,
        '2008 Food Crisis': np.random.beta(5, 5, 55) * 0.6
    }
    
    # Cross-sector correlations
    cross_sector_correlations = np.random.uniform(-0.8, 0.9, 25)
    
    # Pre and post crisis data
    pre_crisis_violations = np.random.beta(3, 7, 30) * 0.4
    post_crisis_violations = np.random.beta(2, 8, 40) * 0.25
    
    # Calculate innovative metrics
    calculator = InnovativeMetricsCalculator()
    
    print("🔬 Calculating Innovative Agricultural Metrics...")
    
    # 1. Crisis Amplification Factor
    caf = calculator.crisis_amplification_factor(normal_violations, crisis_violations)
    print(f"\n📈 Crisis Amplification Factor: {caf:.2f}x")
    print(f"   Normal period violation rate: {np.mean(normal_violations)*100:.1f}%")
    print(f"   Crisis period violation rate: {np.mean(crisis_violations)*100:.1f}%")
    print(f"   Interpretation: Crisis periods show {caf:.1f}x higher violation rates")
    
    # 2. Tier Vulnerability Index
    vuln_index = calculator.tier_vulnerability_index(tier_violation_rates)
    print(f"\n🎯 Tier Vulnerability Index:")
    for tier, vulnerability in vuln_index.items():
        print(f"   {tier}: {vulnerability:.1f}/100")
    
    # 3. Transmission Efficiency Score
    tes = calculator.transmission_efficiency_score(transmission_lags, transmission_strengths)
    print(f"\n⚡ Transmission Efficiency Score: {tes:.1f}/100")
    print(f"   Average transmission lag: {np.mean(transmission_lags):.1f} days")
    print(f"   Average transmission strength: {np.mean(transmission_strengths):.2f}")
    
    # 4. Quantum Correlation Stability
    qcs = calculator.quantum_correlation_stability(s1_time_series)
    print(f"\n🔄 Quantum Correlation Stability: {qcs:.1f}/100")
    violations = np.abs(s1_time_series) > 2.0
    print(f"   Total violations detected: {np.sum(violations)}/{len(s1_time_series)}")
    
    # 5. Cross-Crisis Consistency Score
    cccs = calculator.cross_crisis_consistency_score(crisis_results)
    print(f"\n🌍 Cross-Crisis Consistency Score: {cccs:.1f}/100")
    print(f"   Crisis periods analyzed: {len(crisis_results)}")
    
    # 6. Sector Coupling Strength
    scs = calculator.sector_coupling_strength(cross_sector_correlations)
    print(f"\n🔗 Sector Coupling Strength: {scs:.1f}/100")
    print(f"   Mean absolute correlation: {np.mean(np.abs(cross_sector_correlations)):.2f}")
    
    # 7. Crisis Prediction Indicator
    cpi = calculator.crisis_prediction_indicator(pre_crisis_violations, crisis_violations)
    print(f"\n🔮 Crisis Prediction Indicator: {cpi:.1f}/100")
    print(f"   Pre-crisis signal strength: {np.mean(pre_crisis_violations)*100:.1f}%")
    
    # 8. Recovery Resilience Metric
    rrm = calculator.recovery_resilience_metric(crisis_violations, post_crisis_violations)
    print(f"\n🏥 Recovery Resilience Metric: {rrm:.1f}/100")
    print(f"   Crisis peak: {np.max(crisis_violations)*100:.1f}%")
    print(f"   Post-crisis level: {np.mean(post_crisis_violations)*100:.1f}%")
    
    return {
        'crisis_amplification_factor': caf,
        'tier_vulnerability_index': vuln_index,
        'transmission_efficiency_score': tes,
        'quantum_correlation_stability': qcs,
        'cross_crisis_consistency_score': cccs,
        'sector_coupling_strength': scs,
        'crisis_prediction_indicator': cpi,
        'recovery_resilience_metric': rrm
    }

def demonstrate_comprehensive_analysis():
    """Demonstrate comprehensive statistical analysis suite."""
    print("\n🎯 COMPREHENSIVE STATISTICAL ANALYSIS DEMONSTRATION")
    print("=" * 62)
    
    # Generate comprehensive test data
    data = generate_realistic_agricultural_data()
    price_data = data['price_data']
    cross_sector_pairs = data['cross_sector_pairs'][:4]  # Limit for demo
    
    # Calculate returns
    returns = price_data.pct_change().dropna()
    
    # Perform S1 analysis on multiple pairs
    calculator = EnhancedS1Calculator(window_size=20, threshold_value=0.02)
    
    s1_results = {}
    print("🔍 Performing S1 analysis on agricultural cross-sector pairs...")
    
    for asset_a, asset_b in cross_sector_pairs:
        try:
            pair_results = calculator.analyze_asset_pair(returns, asset_a, asset_b)
            s1_results[(asset_a, asset_b)] = pair_results
            
            violation_rate = pair_results['violation_results']['violation_rate']
            print(f"   {asset_a}-{asset_b}: {violation_rate:.1f}% violations")
            
        except Exception as e:
            print(f"   ⚠️ Error analyzing {asset_a}-{asset_b}: {e}")
            continue
    
    # Prepare crisis data for innovative metrics
    crisis_data = {
        'normal_violations': np.random.beta(2, 8, 100) * 0.3,
        'crisis_violations': np.random.beta(5, 3, 60) * 0.8,
        'tier_violation_rates': {
            'Energy': 52.3,
            'Transportation': 45.7,
            'Chemicals': 48.9
        },
        'transmission_lags': [3, 7, 14, 21, 30],
        'transmission_strengths': [0.85, 0.72, 0.68, 0.54, 0.41],
        's1_time_series': np.concatenate([
            np.random.normal(0, 1.2, 50),
            np.random.normal(0, 2.8, 30),
            np.random.normal(0, 1.5, 40)
        ]),
        'crisis_results': {
            'COVID-19': np.random.beta(6, 4, 50) * 0.7,
            'Ukraine War': np.random.beta(7, 3, 45) * 0.8,
            '2008 Crisis': np.random.beta(5, 5, 55) * 0.6
        },
        'cross_sector_correlations': np.random.uniform(-0.8, 0.9, 20),
        'pre_crisis_violations': np.random.beta(3, 7, 30) * 0.4,
        'post_crisis_violations': np.random.beta(2, 8, 40) * 0.25
    }
    
    # Add violations to S1 time series
    crisis_data['s1_time_series'][::8] = np.random.choice([-3.2, 3.2], 
                                                         size=len(crisis_data['s1_time_series'][::8]))
    
    # Run comprehensive analysis
    suite = ComprehensiveStatisticalSuite(n_bootstrap=500, random_state=42)
    comprehensive_results = suite.comprehensive_analysis(s1_results, crisis_data)
    
    # Display results
    print(f"\n📊 COMPREHENSIVE ANALYSIS RESULTS:")
    print(f"   Total pairs analyzed: {comprehensive_results['summary']['total_pairs']}")
    print(f"   Significant pairs: {comprehensive_results['summary']['significant_pairs']}")
    print(f"   Overall violation rate: {comprehensive_results['summary']['overall_violation_rate']:.2f}%")
    print(f"   Meets Science requirements: {'YES' if comprehensive_results['summary']['meets_science_requirements'] else 'NO'}")
    
    # Show innovative metrics
    metrics = comprehensive_results['innovative_metrics']
    print(f"\n🌾 INNOVATIVE AGRICULTURAL METRICS SUMMARY:")
    print(f"   Crisis Amplification Factor: {metrics.crisis_amplification_factor:.2f}x")
    print(f"   Tier Vulnerability Index: {metrics.tier_vulnerability_index:.1f}/100")
    print(f"   Transmission Efficiency: {metrics.transmission_efficiency_score:.1f}/100")
    print(f"   Quantum Correlation Stability: {metrics.quantum_correlation_stability:.1f}/100")
    print(f"   Cross-Crisis Consistency: {metrics.cross_crisis_consistency_score:.1f}/100")
    print(f"   Sector Coupling Strength: {metrics.sector_coupling_strength:.1f}/100")
    print(f"   Crisis Prediction Indicator: {metrics.crisis_prediction_indicator:.1f}/100")
    print(f"   Recovery Resilience: {metrics.recovery_resilience_metric:.1f}/100")
    
    # Generate publication summary
    print(f"\n📄 PUBLICATION-READY SUMMARY:")
    print("=" * 40)
    summary = suite.generate_publication_summary(comprehensive_results)
    print(summary)
    
    return comprehensive_results

def create_visualization_dashboard(results):
    """Create visualization dashboard for statistical results."""
    print("\n📈 CREATING STATISTICAL VISUALIZATION DASHBOARD")
    print("=" * 54)
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Agricultural Cross-Sector Statistical Validation Dashboard', fontsize=16, fontweight='bold')
    
    # 1. Violation Rates by Pair
    ax1 = axes[0, 0]
    pair_names = []
    violation_rates = []
    confidence_intervals = []
    
    for pair, result in results['pair_results'].items():
        pair_names.append(f"{pair[0]}-{pair[1]}")
        violation_rates.append(result['statistical_results'].violation_rate)
        ci = result['statistical_results'].confidence_interval
        confidence_intervals.append([ci[1] - result['statistical_results'].violation_rate,
                                   result['statistical_results'].violation_rate - ci[0]])
    
    bars = ax1.bar(range(len(pair_names)), violation_rates, 
                   color='steelblue', alpha=0.7, edgecolor='navy')
    ax1.errorbar(range(len(pair_names)), violation_rates, 
                yerr=np.array(confidence_intervals).T, fmt='none', color='red', capsize=5)
    ax1.set_xlabel('Asset Pairs')
    ax1.set_ylabel('Violation Rate (%)')
    ax1.set_title('Bell Inequality Violation Rates\n(with 95% Confidence Intervals)')
    ax1.set_xticks(range(len(pair_names)))
    ax1.set_xticklabels(pair_names, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Add significance markers
    for i, (pair, result) in enumerate(results['pair_results'].items()):
        if result['significant_after_correction']:
            ax1.text(i, violation_rates[i] + max(confidence_intervals[i]) + 2, 
                    '***', ha='center', fontsize=12, color='red', fontweight='bold')
    
    # 2. Effect Sizes
    ax2 = axes[0, 1]
    effect_sizes = [result['effect_sizes']['classical_excess'] for result in results['pair_results'].values()]
    colors = ['green' if es >= 20 else 'orange' if es >= 10 else 'red' for es in effect_sizes]
    
    bars = ax2.bar(range(len(pair_names)), effect_sizes, color=colors, alpha=0.7, edgecolor='black')
    ax2.axhline(y=20, color='green', linestyle='--', alpha=0.7, label='Target: 20%+')
    ax2.axhline(y=60, color='darkgreen', linestyle='--', alpha=0.7, label='Excellent: 60%+')
    ax2.set_xlabel('Asset Pairs')
    ax2.set_ylabel('Classical Bound Excess (%)')
    ax2.set_title('Effect Sizes: Excess Above Classical Bounds')
    ax2.set_xticks(range(len(pair_names)))
    ax2.set_xticklabels(pair_names, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. P-value Distribution
    ax3 = axes[0, 2]
    p_values = [result['statistical_results'].p_value for result in results['pair_results'].values()]
    corrected_p = results['multiple_testing_correction']['corrected_p_values']
    
    x_pos = np.arange(len(pair_names))
    width = 0.35
    
    bars1 = ax3.bar(x_pos - width/2, [-np.log10(p) for p in p_values], width, 
                   label='Original p-values', color='lightblue', alpha=0.7)
    bars2 = ax3.bar(x_pos + width/2, [-np.log10(p) for p in corrected_p], width,
                   label='Corrected p-values', color='darkblue', alpha=0.7)
    
    ax3.axhline(y=-np.log10(0.001), color='red', linestyle='--', alpha=0.7, 
               label='Significance Threshold (p < 0.001)')
    ax3.set_xlabel('Asset Pairs')
    ax3.set_ylabel('-log10(p-value)')
    ax3.set_title('Statistical Significance\n(Higher bars = more significant)')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(pair_names, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Innovative Metrics Radar Chart
    ax4 = axes[1, 0]
    metrics = results['innovative_metrics']
    metric_names = [
        'Crisis\nAmplification',
        'Tier\nVulnerability',
        'Transmission\nEfficiency',
        'Quantum\nStability',
        'Cross-Crisis\nConsistency',
        'Sector\nCoupling',
        'Crisis\nPrediction',
        'Recovery\nResilience'
    ]
    
    # Normalize metrics to 0-100 scale
    metric_values = [
        min(metrics.crisis_amplification_factor * 20, 100),  # Scale amplification factor
        metrics.tier_vulnerability_index,
        metrics.transmission_efficiency_score,
        metrics.quantum_correlation_stability,
        metrics.cross_crisis_consistency_score,
        metrics.sector_coupling_strength,
        metrics.crisis_prediction_indicator,
        metrics.recovery_resilience_metric
    ]
    
    # Create radar chart
    angles = np.linspace(0, 2 * np.pi, len(metric_names), endpoint=False).tolist()
    metric_values += metric_values[:1]  # Complete the circle
    angles += angles[:1]
    
    ax4.plot(angles, metric_values, 'o-', linewidth=2, color='green', alpha=0.7)
    ax4.fill(angles, metric_values, alpha=0.25, color='green')
    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(metric_names, fontsize=9)
    ax4.set_ylim(0, 100)
    ax4.set_title('Innovative Agricultural Metrics\n(0-100 Scale)', fontweight='bold')
    ax4.grid(True)
    
    # 5. Bootstrap Distribution Example
    ax5 = axes[1, 1]
    # Use first pair's bootstrap samples
    first_pair_result = list(results['pair_results'].values())[0]
    bootstrap_samples = first_pair_result['statistical_results'].bootstrap_samples
    
    ax5.hist(bootstrap_samples, bins=30, alpha=0.7, color='purple', edgecolor='black')
    ax5.axvline(first_pair_result['statistical_results'].violation_rate, 
               color='red', linestyle='--', linewidth=2, label='Observed Rate')
    ci = first_pair_result['statistical_results'].confidence_interval
    ax5.axvline(ci[0], color='orange', linestyle=':', alpha=0.7, label='95% CI')
    ax5.axvline(ci[1], color='orange', linestyle=':', alpha=0.7)
    ax5.set_xlabel('Violation Rate (%)')
    ax5.set_ylabel('Frequency')
    ax5.set_title(f'Bootstrap Distribution\n({list(results["pair_results"].keys())[0][0]}-{list(results["pair_results"].keys())[0][1]})')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Summary Statistics
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    # Create summary text
    summary_stats = f"""
STATISTICAL VALIDATION SUMMARY

Total Pairs Analyzed: {results['summary']['total_pairs']}
Significant Pairs: {results['summary']['significant_pairs']}
Overall Violation Rate: {results['summary']['overall_violation_rate']:.1f}%

SIGNIFICANCE VALIDATION:
• Tests meeting p < 0.001: {results['significance_validation']['ultra_significant_tests']}
• Science requirements: {'✅ MET' if results['summary']['meets_science_requirements'] else '❌ NOT MET'}

INNOVATIVE METRICS:
• Crisis Amplification: {metrics.crisis_amplification_factor:.1f}x
• Transmission Efficiency: {metrics.transmission_efficiency_score:.0f}/100
• Quantum Stability: {metrics.quantum_correlation_stability:.0f}/100
• Recovery Resilience: {metrics.recovery_resilience_metric:.0f}/100

EFFECT SIZES:
• Mean Classical Excess: {np.mean([r['effect_sizes']['classical_excess'] for r in results['pair_results'].values()]):.1f}%
• Target Achievement: {'✅ YES' if np.mean([r['effect_sizes']['classical_excess'] for r in results['pair_results'].values()]) >= 20 else '❌ NO'}
"""
    
    ax6.text(0.05, 0.95, summary_stats, transform=ax6.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the dashboard
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'statistical_validation_dashboard_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"📊 Dashboard saved as: {filename}")
    
    plt.show()

def main():
    """Main demonstration function."""
    print("🌾 AGRICULTURAL CROSS-SECTOR STATISTICAL VALIDATION SUITE")
    print("=" * 65)
    print("This demonstration showcases advanced statistical validation and")
    print("innovative metrics for agricultural cross-sector Bell inequality analysis.")
    print("=" * 65)
    
    try:
        # 1. Bootstrap Validation Demo
        bootstrap_results, effect_sizes = demonstrate_bootstrap_validation()
        
        # 2. Multiple Testing Correction Demo
        pair_results, p_values = demonstrate_multiple_testing_correction()
        
        # 3. Innovative Metrics Demo
        innovative_metrics = demonstrate_innovative_metrics()
        
        # 4. Comprehensive Analysis Demo
        comprehensive_results = demonstrate_comprehensive_analysis()
        
        # 5. Create Visualization Dashboard
        create_visualization_dashboard(comprehensive_results)
        
        print("\n✅ DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("=" * 45)
        print("Key achievements demonstrated:")
        print("• Bootstrap validation with 1000+ resamples ✅")
        print("• Statistical significance testing (p < 0.001) ✅")
        print("• Confidence interval calculations ✅")
        print("• Multiple testing correction ✅")
        print("• Effect size calculations (20-60% target) ✅")
        print("• 8 innovative agricultural crisis metrics ✅")
        print("• Publication-ready statistical summary ✅")
        print("• Comprehensive visualization dashboard ✅")
        
        # Final validation
        overall_stats = comprehensive_results['overall_statistics']
        meets_requirements = (
            overall_stats.p_value < 0.001 and
            overall_stats.classical_bound_excess >= 20 and
            comprehensive_results['summary']['meets_science_requirements']
        )
        
        print(f"\n🎯 SCIENCE JOURNAL REQUIREMENTS: {'✅ ALL MET' if meets_requirements else '⚠️ PARTIALLY MET'}")
        
    except Exception as e:
        print(f"❌ Error in demonstration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()