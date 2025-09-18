#!/usr/bin/env python3
"""
Agricultural Crisis Analysis Demo

This script demonstrates the Agricultural Crisis Analysis Module functionality
for analyzing Bell inequality violations during major historical crisis periods:
- 2008 Financial Crisis (September 2008 - March 2009)
- EU Debt Crisis (May 2010 - December 2012)
- COVID-19 Pandemic (February 2020 - December 2020)

The demo shows crisis-specific parameters, amplification detection, and
cross-crisis comparison functionality.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from agricultural_crisis_analyzer import (
    AgriculturalCrisisAnalyzer, 
    quick_crisis_analysis, 
    compare_all_crises
)
from agricultural_universe_manager import AgriculturalUniverseManager


def create_demo_data():
    """Create realistic demo data with crisis periods."""
    print("📊 Creating demo data with crisis periods...")
    
    # Set random seed for reproducible results
    np.random.seed(42)
    
    # Date range covering all crisis periods
    dates = pd.date_range('2008-01-01', '2021-12-31', freq='D')
    
    # Demo assets from different tiers
    demo_assets = {
        # Agricultural companies (Tier 0)
        'ADM': 0.020,   # Archer Daniels Midland
        'CF': 0.025,    # CF Industries
        'BG': 0.022,    # Bunge
        'NTR': 0.024,   # Nutrien
        'DE': 0.021,    # John Deere
        
        # Tier 1: Energy/Transport/Chemicals
        'XOM': 0.025,   # Exxon Mobil
        'CVX': 0.023,   # Chevron
        'UNP': 0.020,   # Union Pacific
        'DOW': 0.024,   # Dow Chemical
        'COP': 0.026,   # ConocoPhillips
        
        # Tier 2: Finance/Equipment
        'JPM': 0.022,   # JPMorgan Chase
        'BAC': 0.025,   # Bank of America
        'GS': 0.027,    # Goldman Sachs
        'CAT': 0.023,   # Caterpillar
        
        # Tier 3: Policy-linked
        'NEE': 0.018,   # NextEra Energy
        'AWK': 0.016,   # American Water Works
        'DUK': 0.019,   # Duke Energy
    }
    
    # Define crisis periods with amplification factors
    crisis_periods = [
        ('2008-09-01', '2009-03-31', 2.8, 'Financial Crisis'),      # 2008 Financial Crisis
        ('2010-05-01', '2012-12-31', 2.2, 'EU Debt Crisis'),       # EU Debt Crisis  
        ('2020-02-01', '2020-12-31', 3.5, 'COVID-19 Pandemic'),    # COVID-19 Pandemic
    ]
    
    returns_data = pd.DataFrame(index=dates)
    
    for asset, base_vol in demo_assets.items():
        # Generate base returns with some autocorrelation
        returns = np.random.normal(0, base_vol, len(dates))
        
        # Add some persistence (autocorrelation)
        for i in range(1, len(returns)):
            returns[i] += 0.1 * returns[i-1]
        
        # Amplify volatility and add correlation during crisis periods
        for start_date, end_date, amplification, crisis_name in crisis_periods:
            mask = (dates >= start_date) & (dates <= end_date)
            
            # Increase volatility during crisis
            returns[mask] *= amplification
            
            # Add sector-specific crisis effects
            if crisis_name == 'Financial Crisis' and asset in ['JPM', 'BAC', 'GS']:
                returns[mask] *= 1.3  # Financial sector hit harder
            elif crisis_name == 'COVID-19 Pandemic' and asset in ['ADM', 'BG', 'CF']:
                returns[mask] *= 1.2  # Food companies affected by supply chain
            elif crisis_name == 'EU Debt Crisis' and asset in ['XOM', 'CVX', 'COP']:
                returns[mask] *= 1.1  # Energy sector moderately affected
        
        returns_data[asset] = returns
    
    print(f"✅ Demo data created: {len(returns_data)} observations, {len(demo_assets)} assets")
    print(f"   Date range: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
    
    return returns_data


def demo_individual_crisis_analysis(returns_data):
    """Demonstrate individual crisis period analysis."""
    print("\n" + "="*60)
    print("🔍 INDIVIDUAL CRISIS ANALYSIS DEMO")
    print("="*60)
    
    analyzer = AgriculturalCrisisAnalyzer()
    
    # Analyze each crisis period
    crisis_results = {}
    
    print("\n1️⃣ 2008 Financial Crisis Analysis")
    print("-" * 40)
    crisis_results['2008'] = analyzer.analyze_2008_financial_crisis(returns_data)
    
    print("\n2️⃣ EU Debt Crisis Analysis")
    print("-" * 40)
    crisis_results['eu'] = analyzer.analyze_eu_debt_crisis(returns_data)
    
    print("\n3️⃣ COVID-19 Pandemic Analysis")
    print("-" * 40)
    crisis_results['covid'] = analyzer.analyze_covid19_pandemic(returns_data)
    
    return crisis_results


def demo_crisis_comparison(returns_data):
    """Demonstrate crisis comparison functionality."""
    print("\n" + "="*60)
    print("📊 CRISIS COMPARISON ANALYSIS DEMO")
    print("="*60)
    
    analyzer = AgriculturalCrisisAnalyzer()
    
    # Compare all three crisis periods
    comparison_results = analyzer.compare_crisis_periods(returns_data)
    
    print("\n🏆 Crisis Severity Ranking by Tier:")
    print("-" * 40)
    
    for tier in sorted(comparison_results.crisis_ranking.keys()):
        print(f"\nTier {tier} Rankings:")
        rankings = comparison_results.crisis_ranking[tier]
        for i, (crisis_name, violation_rate) in enumerate(rankings, 1):
            print(f"  {i}. {crisis_name}: {violation_rate:.1f}% violation rate")
    
    print("\n🎯 Tier Vulnerability Index:")
    print("-" * 40)
    for tier in sorted(comparison_results.tier_vulnerability_index.keys()):
        vulnerability = comparison_results.tier_vulnerability_index[tier]
        print(f"  Tier {tier}: {vulnerability:.1f}% average violation rate")
    
    print("\n🔄 Cross-Crisis Consistency:")
    print("-" * 40)
    for tier in sorted(comparison_results.cross_crisis_consistency.keys()):
        consistency = comparison_results.cross_crisis_consistency[tier]
        print(f"  Tier {tier}: {consistency:.2f} consistency score (0-1)")
    
    return comparison_results


def demo_convenience_functions(returns_data):
    """Demonstrate convenience functions for quick analysis."""
    print("\n" + "="*60)
    print("⚡ CONVENIENCE FUNCTIONS DEMO")
    print("="*60)
    
    print("\n🚀 Quick COVID-19 Analysis:")
    print("-" * 30)
    covid_results = quick_crisis_analysis(returns_data, "covid19_pandemic")
    print(f"   Crisis: {covid_results.crisis_period.name}")
    print(f"   Period: {covid_results.crisis_period.start_date} to {covid_results.crisis_period.end_date}")
    print(f"   Tiers analyzed: {len(covid_results.tier_results)}")
    
    print("\n🔄 Compare All Crises:")
    print("-" * 30)
    all_comparison = compare_all_crises(returns_data)
    print(f"   Crises compared: {len(all_comparison.crisis_periods)}")
    print(f"   Crisis periods: {', '.join(all_comparison.crisis_periods)}")


def demo_crisis_parameters():
    """Demonstrate crisis-specific parameters."""
    print("\n" + "="*60)
    print("⚙️  CRISIS-SPECIFIC PARAMETERS DEMO")
    print("="*60)
    
    analyzer = AgriculturalCrisisAnalyzer()
    
    print("\n🔬 Crisis Calculator Parameters:")
    print("-" * 35)
    print(f"   Window size: {analyzer.crisis_calculator.window_size}")
    print(f"   Threshold method: {analyzer.crisis_calculator.threshold_method}")
    print(f"   Threshold quantile: {analyzer.crisis_calculator.threshold_quantile}")
    
    print("\n📊 Normal Calculator Parameters (for comparison):")
    print("-" * 50)
    print(f"   Window size: {analyzer.normal_calculator.window_size}")
    print(f"   Threshold method: {analyzer.normal_calculator.threshold_method}")
    print(f"   Threshold quantile: {analyzer.normal_calculator.threshold_quantile}")
    
    print("\n📅 Crisis Period Definitions:")
    print("-" * 35)
    for crisis_key, crisis_period in analyzer.crisis_periods.items():
        print(f"\n   {crisis_period.name}:")
        print(f"     Period: {crisis_period.start_date} to {crisis_period.end_date}")
        print(f"     Expected violation rate: {crisis_period.expected_violation_rate}%")
        print(f"     Affected sectors: {', '.join(crisis_period.affected_sectors[:3])}...")


def demo_amplification_detection(crisis_results):
    """Demonstrate crisis amplification detection."""
    print("\n" + "="*60)
    print("📈 CRISIS AMPLIFICATION DETECTION DEMO")
    print("="*60)
    
    print("\n🔥 Amplification Factors by Crisis:")
    print("-" * 40)
    
    for crisis_name, results in crisis_results.items():
        print(f"\n{results.crisis_period.name}:")
        amplification = results.crisis_amplification
        
        for key, factor in amplification.items():
            if key == "overall":
                print(f"   Overall amplification: {factor:.2f}x")
            else:
                print(f"   {key.replace('_', ' ').title()}: {factor:.2f}x")
    
    print("\n🎯 Crisis-Specific Metrics:")
    print("-" * 30)
    
    for crisis_name, results in crisis_results.items():
        print(f"\n{results.crisis_period.name}:")
        
        for tier, tier_result in results.tier_results.items():
            if "crisis_specific_metrics" in tier_result:
                metrics = tier_result["crisis_specific_metrics"]
                print(f"   Tier {tier}:")
                print(f"     Violation rate: {metrics.get('violation_rate', 0):.1f}%")
                print(f"     Severity: {metrics.get('severity', 'Unknown')}")
                print(f"     Meets crisis threshold: {metrics.get('meets_crisis_threshold', False)}")


def create_summary_visualization(comparison_results):
    """Create a summary visualization of crisis comparison results."""
    print("\n" + "="*60)
    print("📊 CREATING SUMMARY VISUALIZATION")
    print("="*60)
    
    try:
        # Extract data for visualization
        tiers = sorted(comparison_results.tier_vulnerability_index.keys())
        vulnerabilities = [comparison_results.tier_vulnerability_index[tier] for tier in tiers]
        consistencies = [comparison_results.cross_crisis_consistency[tier] for tier in tiers]
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Tier Vulnerability Index
        bars1 = ax1.bar([f'Tier {t}' for t in tiers], vulnerabilities, 
                       color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax1.set_title('Tier Vulnerability Index\n(Average Violation Rate Across Crises)')
        ax1.set_ylabel('Violation Rate (%)')
        ax1.set_ylim(0, max(vulnerabilities) * 1.2)
        
        # Add value labels on bars
        for bar, value in zip(bars1, vulnerabilities):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{value:.1f}%', ha='center', va='bottom')
        
        # Cross-Crisis Consistency
        bars2 = ax2.bar([f'Tier {t}' for t in tiers], consistencies,
                       color=['#96CEB4', '#FFEAA7', '#DDA0DD'])
        ax2.set_title('Cross-Crisis Consistency\n(Response Consistency Across Crises)')
        ax2.set_ylabel('Consistency Score (0-1)')
        ax2.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars2, consistencies):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{value:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save the plot
        output_file = 'agricultural_crisis_analysis_summary.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✅ Summary visualization saved: {output_file}")
        
        plt.show()
        
    except Exception as e:
        print(f"⚠️  Could not create visualization: {e}")
        print("   (This is normal if matplotlib is not available)")


def main():
    """Main demo function."""
    print("🌾 AGRICULTURAL CRISIS ANALYSIS MODULE DEMO")
    print("=" * 60)
    print("This demo showcases the Agricultural Crisis Analysis Module")
    print("for detecting Bell inequality violations during major crises.")
    print("=" * 60)
    
    # Create demo data
    returns_data = create_demo_data()
    
    # Demo individual crisis analysis
    crisis_results = demo_individual_crisis_analysis(returns_data)
    
    # Demo crisis comparison
    comparison_results = demo_crisis_comparison(returns_data)
    
    # Demo convenience functions
    demo_convenience_functions(returns_data)
    
    # Demo crisis parameters
    demo_crisis_parameters()
    
    # Demo amplification detection
    demo_amplification_detection(crisis_results)
    
    # Create summary visualization
    create_summary_visualization(comparison_results)
    
    print("\n" + "="*60)
    print("✅ AGRICULTURAL CRISIS ANALYSIS DEMO COMPLETE")
    print("="*60)
    print("\nKey Features Demonstrated:")
    print("• Crisis-specific parameters (window=15, threshold=0.8)")
    print("• Three historical crisis periods (2008, EU Debt, COVID-19)")
    print("• Crisis amplification detection (40-60% violation rates)")
    print("• Cross-crisis comparison functionality")
    print("• Tier-based vulnerability analysis")
    print("• Statistical significance testing")
    print("• Convenience functions for quick analysis")
    print("\nThe module is ready for integration with real market data!")


if __name__ == "__main__":
    main()