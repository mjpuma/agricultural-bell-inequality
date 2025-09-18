#!/usr/bin/env python3
"""
AGRICULTURAL CROSS-SECTOR ANALYZER DEMONSTRATION
===============================================

This script demonstrates the complete usage of the Agricultural Cross-Sector Analyzer
main class, showing how to perform comprehensive tier-based analysis with crisis
integration across all three tiers and crisis periods.

The demo covers:
- Data loading for the complete agricultural universe
- Tier 1 analysis (Energy/Transport/Chemicals → Agriculture)
- Tier 2 analysis (Finance/Equipment → Agriculture)  
- Tier 3 analysis (Policy-linked → Agriculture)
- Crisis period analysis (2008 Financial Crisis, EU Debt Crisis, COVID-19)
- Comprehensive workflow with normal vs crisis period comparison
- Publication-ready results and visualizations

Authors: Agricultural Cross-Sector Analysis Team
Date: September 2025
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from agricultural_cross_sector_analyzer import (
    AgriculturalCrossSectorAnalyzer, 
    AnalysisConfiguration
)


def demo_basic_usage():
    """Demonstrate basic usage of the Agricultural Cross-Sector Analyzer."""
    print("🌾 BASIC USAGE DEMONSTRATION")
    print("=" * 50)
    
    # Initialize analyzer with default configuration
    analyzer = AgriculturalCrossSectorAnalyzer()
    
    # Load data for a subset of key companies
    key_companies = [
        # Core agricultural companies
        'ADM', 'BG', 'CF', 'MOS', 'NTR', 'DE',
        # Key cross-sector companies
        'XOM', 'CVX', 'UNP', 'JPM', 'BAC', 'NEE'
    ]
    
    print(f"\n📊 Loading data for {len(key_companies)} key companies...")
    try:
        returns_data = analyzer.load_data(tickers=key_companies, period="2y")
        print(f"✅ Data loaded successfully: {len(returns_data)} observations")
        
        # Quick Tier 1 analysis
        print(f"\n🔋 Running Tier 1 analysis...")
        tier1_results = analyzer.analyze_tier_1_crisis()
        
        violation_rate = tier1_results.violation_summary.get('overall_violation_rate', 0)
        transmission_rate = tier1_results.violation_summary.get('transmission_detection_rate', 0)
        
        print(f"✅ Tier 1 Results:")
        print(f"   • Violation Rate: {violation_rate:.1f}%")
        print(f"   • Transmission Detection Rate: {transmission_rate:.1f}%")
        print(f"   • Cross-sector pairs analyzed: {len(tier1_results.cross_sector_pairs)}")
        
        return analyzer, tier1_results
        
    except Exception as e:
        print(f"❌ Basic demo failed: {str(e)}")
        return None, None


def demo_comprehensive_analysis():
    """Demonstrate comprehensive analysis across all tiers and crisis periods."""
    print("\n🌾 COMPREHENSIVE ANALYSIS DEMONSTRATION")
    print("=" * 60)
    
    # Custom configuration for comprehensive analysis
    config = AnalysisConfiguration(
        window_size=20,
        threshold_value=0.015,  # Slightly higher threshold for cleaner results
        crisis_window_size=15,
        crisis_threshold_quantile=0.8,
        significance_level=0.001,
        bootstrap_samples=500,  # Reduced for demo speed
        max_pairs_per_tier=15   # Limit for demo performance
    )
    
    analyzer = AgriculturalCrossSectorAnalyzer(config)
    
    # Load comprehensive dataset
    comprehensive_tickers = [
        # Agricultural companies (Tier 0)
        'ADM', 'BG', 'CF', 'MOS', 'NTR', 'DE', 'CAG', 'TSN', 'HRL', 'K', 'GIS',
        
        # Tier 1: Energy/Transport/Chemicals
        'XOM', 'CVX', 'COP', 'UNP', 'CSX', 'NSC', 'DOW', 'DD', 'LYB',
        
        # Tier 2: Finance/Equipment
        'JPM', 'BAC', 'GS', 'MS', 'CAT',
        
        # Tier 3: Policy-linked
        'NEE', 'DUK', 'SO', 'AWK'
    ]
    
    print(f"\n📊 Loading comprehensive dataset ({len(comprehensive_tickers)} companies)...")
    
    try:
        returns_data = analyzer.load_data(tickers=comprehensive_tickers, period="5y")
        
        # Run comprehensive analysis
        print(f"\n🚀 Running comprehensive analysis...")
        results = analyzer.run_comprehensive_analysis()
        
        # Display detailed results
        print(f"\n📈 COMPREHENSIVE RESULTS SUMMARY")
        print("=" * 50)
        
        stats = results.overall_statistics
        print(f"📊 Overall Violation Rate: {stats['overall_violation_rate']:.2f}%")
        print(f"🔄 Overall Transmission Rate: {stats['overall_transmission_rate']:.2f}%")
        print(f"🎯 Total Bell Violations: {stats['total_violations']:,}")
        print(f"📡 Detected Transmissions: {stats['total_detected_transmissions']}")
        
        # Tier-by-tier breakdown
        print(f"\n📋 TIER-BY-TIER BREAKDOWN")
        print("-" * 30)
        
        for tier, tier_results in results.tier_results.items():
            print(f"\nTier {tier}: {tier_results.tier_name}")
            summary = tier_results.violation_summary
            
            print(f"  • Pairs analyzed: {len(tier_results.cross_sector_pairs)}")
            print(f"  • Violation rate: {summary.get('overall_violation_rate', 0):.1f}%")
            print(f"  • Transmissions detected: {summary.get('detected_transmissions', 0)}")
            print(f"  • Statistical significance: {tier_results.statistical_validation.get('overall_significance', 'N/A')}")
        
        # Crisis comparison results
        if results.crisis_comparison:
            print(f"\n🚨 CRISIS COMPARISON RESULTS")
            print("-" * 30)
            
            vulnerability = results.crisis_comparison.tier_vulnerability_index
            consistency = results.crisis_comparison.cross_crisis_consistency
            
            for tier in [1, 2, 3]:
                tier_name = results.tier_results[tier].tier_name
                vuln_score = vulnerability.get(tier, 0)
                cons_score = consistency.get(tier, 0)
                
                print(f"Tier {tier} ({tier_name}):")
                print(f"  • Vulnerability Index: {vuln_score:.1f}%")
                print(f"  • Crisis Consistency: {cons_score:.3f}")
        
        # Transmission mechanism summary
        print(f"\n🔄 TRANSMISSION MECHANISMS")
        print("-" * 30)
        
        transmission_df = results.transmission_summary
        detected_transmissions = transmission_df[transmission_df['transmission_detected'] == True]
        
        if not detected_transmissions.empty:
            print(f"✅ {len(detected_transmissions)} significant transmission mechanisms detected:")
            
            for _, row in detected_transmissions.head(10).iterrows():  # Show top 10
                print(f"  • {row['source_asset']} → {row['target_asset']}: "
                      f"{row['speed_category']} ({row['correlation_strength']:.3f})")
        else:
            print("❌ No significant transmission mechanisms detected")
        
        return analyzer, results
        
    except Exception as e:
        print(f"❌ Comprehensive analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


def demo_crisis_specific_analysis():
    """Demonstrate crisis-specific analysis capabilities."""
    print("\n🚨 CRISIS-SPECIFIC ANALYSIS DEMONSTRATION")
    print("=" * 60)
    
    analyzer = AgriculturalCrossSectorAnalyzer()
    
    # Load data covering crisis periods
    crisis_tickers = ['ADM', 'CF', 'XOM', 'JPM', 'UNP', 'DE', 'BG', 'MOS']
    
    try:
        returns_data = analyzer.load_data(tickers=crisis_tickers, period="max")  # Maximum available data
        
        # Analyze each crisis period individually
        crisis_periods = ["2008_financial_crisis", "eu_debt_crisis", "covid19_pandemic"]
        
        for crisis in crisis_periods:
            print(f"\n🔍 Analyzing {crisis.replace('_', ' ').title()}...")
            
            # Tier 1 analysis for this crisis
            tier1_results = analyzer.analyze_tier_1_crisis([crisis])
            
            if tier1_results.crisis_results and crisis in tier1_results.crisis_results:
                crisis_data = tier1_results.crisis_results[crisis]
                
                if 'crisis_specific_metrics' in crisis_data:
                    metrics = crisis_data['crisis_specific_metrics']
                    
                    print(f"  ✅ Crisis Analysis Results:")
                    print(f"     • Violation Rate: {metrics.get('violation_rate', 0):.1f}%")
                    print(f"     • Expected Rate: {metrics.get('expected_rate', 0):.1f}%")
                    print(f"     • Amplification Factor: {metrics.get('amplification_factor', 0):.2f}x")
                    print(f"     • Severity: {metrics.get('severity', 'Unknown')}")
                    print(f"     • Meets Crisis Threshold: {metrics.get('meets_crisis_threshold', False)}")
                else:
                    print(f"  ⚠️  Limited crisis data available")
            else:
                print(f"  ❌ No crisis data available for {crisis}")
        
        return analyzer
        
    except Exception as e:
        print(f"❌ Crisis analysis failed: {str(e)}")
        return None


def demo_tier_comparison():
    """Demonstrate tier-by-tier comparison capabilities."""
    print("\n📊 TIER COMPARISON DEMONSTRATION")
    print("=" * 50)
    
    analyzer = AgriculturalCrossSectorAnalyzer()
    
    # Load balanced dataset across all tiers
    balanced_tickers = [
        # Agricultural
        'ADM', 'CF', 'DE', 'BG',
        # Tier 1
        'XOM', 'UNP', 'DOW',
        # Tier 2  
        'JPM', 'CAT',
        # Tier 3
        'NEE', 'AWK'
    ]
    
    try:
        returns_data = analyzer.load_data(tickers=balanced_tickers, period="3y")
        
        # Analyze each tier separately
        tier_results = {}
        
        print(f"\n🔋 Tier 1: Energy/Transport/Chemicals")
        tier_results[1] = analyzer.analyze_tier_1_crisis()
        
        print(f"\n🏦 Tier 2: Finance/Equipment")
        tier_results[2] = analyzer.analyze_tier_2_crisis()
        
        print(f"\n⚡ Tier 3: Policy-linked")
        tier_results[3] = analyzer.analyze_tier_3_crisis()
        
        # Compare tier performance
        print(f"\n📈 TIER PERFORMANCE COMPARISON")
        print("-" * 40)
        
        comparison_data = []
        for tier, results in tier_results.items():
            summary = results.violation_summary
            comparison_data.append({
                'Tier': tier,
                'Name': results.tier_name,
                'Pairs': len(results.cross_sector_pairs),
                'Violation Rate (%)': summary.get('overall_violation_rate', 0),
                'Transmissions': summary.get('detected_transmissions', 0),
                'Transmission Rate (%)': summary.get('transmission_detection_rate', 0)
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False, float_format='%.1f'))
        
        # Identify most active tier
        most_active_tier = comparison_df.loc[comparison_df['Violation Rate (%)'].idxmax()]
        print(f"\n🏆 Most Active Tier: Tier {most_active_tier['Tier']} ({most_active_tier['Name']})")
        print(f"   Violation Rate: {most_active_tier['Violation Rate (%)']}%")
        
        return analyzer, tier_results
        
    except Exception as e:
        print(f"❌ Tier comparison failed: {str(e)}")
        return None, None


def main():
    """Run all demonstrations."""
    print("🌾 AGRICULTURAL CROSS-SECTOR ANALYZER DEMONSTRATIONS")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run demonstrations
    demos = [
        ("Basic Usage", demo_basic_usage),
        ("Crisis-Specific Analysis", demo_crisis_specific_analysis),
        ("Tier Comparison", demo_tier_comparison),
        ("Comprehensive Analysis", demo_comprehensive_analysis)  # Most intensive, run last
    ]
    
    results = {}
    
    for demo_name, demo_func in demos:
        print(f"\n{'='*20} {demo_name.upper()} {'='*20}")
        
        try:
            result = demo_func()
            results[demo_name] = result
            print(f"✅ {demo_name} completed successfully")
            
        except Exception as e:
            print(f"❌ {demo_name} failed: {str(e)}")
            results[demo_name] = None
    
    # Final summary
    print(f"\n🎯 DEMONSTRATION SUMMARY")
    print("=" * 40)
    
    successful_demos = sum(1 for result in results.values() if result is not None)
    total_demos = len(demos)
    
    print(f"Completed: {successful_demos}/{total_demos} demonstrations")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if successful_demos == total_demos:
        print("🎉 All demonstrations completed successfully!")
        print("\n📚 Next Steps:")
        print("   • Review the comprehensive analysis results")
        print("   • Examine tier-specific violation patterns")
        print("   • Investigate detected transmission mechanisms")
        print("   • Analyze crisis amplification effects")
        print("   • Generate publication-ready visualizations")
    else:
        print("⚠️  Some demonstrations encountered issues")
        print("   • Check data availability for missing tickers")
        print("   • Verify internet connection for data downloads")
        print("   • Consider reducing analysis scope for performance")
    
    return results


if __name__ == "__main__":
    results = main()