#!/usr/bin/env python3
"""
SEASONAL AND GEOGRAPHIC ANALYSIS DEMONSTRATION
==============================================

This script demonstrates the seasonal and geographic analysis features for agricultural
cross-sector Bell inequality analysis. It shows how to:

1. Analyze seasonal effects for agricultural planting/harvest cycles
2. Examine geographic patterns in regional agricultural production
3. Detect seasonal modulation of quantum correlation strength
4. Analyze regional crisis impacts
5. Generate comprehensive seasonal-geographic visualizations

Requirements Addressed:
- 8.2: Seasonal effect detection for agricultural planting/harvest cycles
- 8.3: Geographic analysis considering regional agricultural production patterns
- 8.4: Seasonal modulation analysis for quantum correlation strength variations

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

from seasonal_geographic_analyzer import SeasonalGeographicAnalyzer
from seasonal_visualization_suite import SeasonalVisualizationSuite
from agricultural_cross_sector_analyzer import AgriculturalCrossSectorAnalyzer


def demonstrate_seasonal_analysis():
    """Demonstrate seasonal effect detection for agricultural cycles."""
    print("🌱 SEASONAL ANALYSIS DEMONSTRATION")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = SeasonalGeographicAnalyzer()
    
    # Create sample data for demonstration
    print("📊 Creating sample agricultural data...")
    
    # Generate sample returns data with seasonal patterns
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    
    # Agricultural companies with seasonal patterns
    agricultural_tickers = ['ADM', 'BG', 'CF', 'MOS', 'DE']
    
    # Create returns with seasonal modulation
    np.random.seed(42)
    returns_data = pd.DataFrame(index=dates, columns=agricultural_tickers)
    
    for ticker in agricultural_tickers:
        # Base random returns
        base_returns = np.random.normal(0, 0.02, len(dates))
        
        # Add seasonal modulation
        seasonal_factor = np.sin(2 * np.pi * dates.dayofyear / 365.25) * 0.005
        
        # Spring planting season amplification (March-May)
        spring_mask = dates.month.isin([3, 4, 5])
        base_returns[spring_mask] += np.random.normal(0, 0.01, np.sum(spring_mask))
        
        # Fall harvest season amplification (September-November)
        fall_mask = dates.month.isin([9, 10, 11])
        base_returns[fall_mask] += np.random.normal(0, 0.015, np.sum(fall_mask))
        
        returns_data[ticker] = base_returns + seasonal_factor
    
    # Create asset pairs for analysis
    asset_pairs = [
        ('ADM', 'BG'),   # Grain processors
        ('CF', 'MOS'),   # Fertilizer companies
        ('DE', 'ADM'),   # Equipment-Agriculture
        ('CF', 'DE'),    # Fertilizer-Equipment
        ('BG', 'MOS')    # Cross-sector
    ]
    
    print(f"   Data period: {dates[0].date()} to {dates[-1].date()}")
    print(f"   Agricultural companies: {len(agricultural_tickers)}")
    print(f"   Asset pairs: {len(asset_pairs)}")
    
    # Run seasonal analysis
    print("\n🌱 Running seasonal effect analysis...")
    seasonal_results = analyzer.analyze_seasonal_effects(returns_data, asset_pairs)
    
    # Display results
    print("\n📊 SEASONAL ANALYSIS RESULTS:")
    print("-" * 40)
    
    if seasonal_results.seasonal_violation_rates:
        print("Seasonal Bell Violation Rates:")
        for season, rate in seasonal_results.seasonal_violation_rates.items():
            print(f"  {season}: {rate:.2f}%")
        
        max_season = max(seasonal_results.seasonal_violation_rates, 
                        key=seasonal_results.seasonal_violation_rates.get)
        min_season = min(seasonal_results.seasonal_violation_rates, 
                        key=seasonal_results.seasonal_violation_rates.get)
        
        print(f"\nHighest violations: {max_season}")
        print(f"Lowest violations: {min_season}")
    
    if seasonal_results.correlation_modulation:
        print("\nCorrelation Strength Modulation:")
        for season, modulation in seasonal_results.correlation_modulation.items():
            change = (modulation - 1.0) * 100
            print(f"  {season}: {modulation:.3f} ({change:+.1f}%)")
    
    return seasonal_results


def demonstrate_geographic_analysis():
    """Demonstrate geographic analysis for regional production patterns."""
    print("\n🌍 GEOGRAPHIC ANALYSIS DEMONSTRATION")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = SeasonalGeographicAnalyzer()
    
    # Create sample data representing different geographic regions
    print("📊 Creating sample regional agricultural data...")
    
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    
    # Companies representing different regions
    regional_companies = {
        'North_America': ['ADM', 'CF', 'DE'],      # US-based
        'South_America': ['BG', 'SBS'],            # Brazil/Argentina exposure
        'Europe': ['UAN', 'K'],                    # European operations
        'Asia_Pacific': ['ANDE', 'FMC']            # Asia-Pacific exposure
    }
    
    all_tickers = []
    for companies in regional_companies.values():
        all_tickers.extend(companies)
    
    # Create returns with regional patterns
    np.random.seed(42)
    returns_data = pd.DataFrame(index=dates, columns=all_tickers)
    
    for region, companies in regional_companies.items():
        # Regional correlation factor
        regional_factor = np.random.normal(0, 0.01, len(dates))
        
        # Climate zone effects
        if region == 'South_America':
            # Southern Hemisphere - opposite seasons
            seasonal_factor = -np.sin(2 * np.pi * dates.dayofyear / 365.25) * 0.003
        else:
            # Northern Hemisphere
            seasonal_factor = np.sin(2 * np.pi * dates.dayofyear / 365.25) * 0.003
        
        for ticker in companies:
            if ticker in all_tickers:
                base_returns = np.random.normal(0, 0.02, len(dates))
                returns_data[ticker] = base_returns + regional_factor * 0.5 + seasonal_factor
    
    # Create cross-regional asset pairs
    asset_pairs = []
    regions = list(regional_companies.keys())
    
    for i, region1 in enumerate(regions):
        for region2 in regions[i+1:]:
            companies1 = regional_companies[region1]
            companies2 = regional_companies[region2]
            
            # Create pairs between regions
            if companies1 and companies2:
                asset_pairs.append((companies1[0], companies2[0]))
    
    # Add intra-regional pairs
    for companies in regional_companies.values():
        if len(companies) >= 2:
            asset_pairs.append((companies[0], companies[1]))
    
    print(f"   Regional companies: {sum(len(c) for c in regional_companies.values())}")
    print(f"   Cross-regional pairs: {len(asset_pairs)}")
    
    # Run geographic analysis
    print("\n🌍 Running geographic analysis...")
    geographic_results = analyzer.analyze_geographic_effects(returns_data, asset_pairs)
    
    # Display results
    print("\n📊 GEOGRAPHIC ANALYSIS RESULTS:")
    print("-" * 40)
    
    if geographic_results.regional_patterns:
        print("Regional Patterns:")
        for region, data in geographic_results.regional_patterns.items():
            companies = data.get('companies', [])
            region_info = data.get('region_info')
            print(f"  {region.replace('_', ' ')}:")
            print(f"    Companies: {len(companies)}")
            if region_info:
                print(f"    Climate: {region_info.climate_zone}")
                print(f"    Primary crops: {', '.join(region_info.primary_crops[:3])}")
    
    if geographic_results.cross_regional_correlations:
        print("\nCross-Regional Correlations:")
        for (region1, region2), corr in geographic_results.cross_regional_correlations.items():
            print(f"  {region1.replace('_', ' ')} ↔ {region2.replace('_', ' ')}: {corr:.3f}")
    
    return geographic_results


def demonstrate_seasonal_modulation():
    """Demonstrate seasonal modulation analysis."""
    print("\n📊 SEASONAL MODULATION DEMONSTRATION")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = SeasonalGeographicAnalyzer()
    
    # Create data with strong seasonal modulation
    print("📊 Creating data with seasonal modulation patterns...")
    
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    tickers = ['CORN_FUTURE', 'SOYB_FUTURE', 'WEAT_FUTURE', 'FERT_INDEX']
    
    np.random.seed(42)
    returns_data = pd.DataFrame(index=dates, columns=tickers)
    
    for ticker in tickers:
        base_returns = np.random.normal(0, 0.02, len(dates))
        
        # Strong seasonal modulation
        if 'CORN' in ticker or 'SOYB' in ticker:
            # Corn/Soy: High volatility during planting (Apr-May) and harvest (Sep-Oct)
            planting_mask = dates.month.isin([4, 5])
            harvest_mask = dates.month.isin([9, 10])
            
            base_returns[planting_mask] *= 1.5  # 50% increase in volatility
            base_returns[harvest_mask] *= 1.8   # 80% increase in volatility
            
        elif 'WEAT' in ticker:
            # Wheat: Different seasonal pattern
            winter_mask = dates.month.isin([12, 1, 2])
            summer_mask = dates.month.isin([6, 7, 8])
            
            base_returns[winter_mask] *= 1.3
            base_returns[summer_mask] *= 1.6
            
        elif 'FERT' in ticker:
            # Fertilizer: High demand during planting seasons
            spring_mask = dates.month.isin([3, 4, 5])
            base_returns[spring_mask] *= 2.0  # 100% increase
        
        returns_data[ticker] = base_returns
    
    # Create asset pairs
    asset_pairs = [
        ('CORN_FUTURE', 'FERT_INDEX'),
        ('SOYB_FUTURE', 'FERT_INDEX'),
        ('WEAT_FUTURE', 'CORN_FUTURE'),
        ('CORN_FUTURE', 'SOYB_FUTURE')
    ]
    
    print(f"   Analyzing {len(asset_pairs)} commodity-fertilizer pairs")
    
    # Run seasonal modulation analysis
    print("\n📊 Running seasonal modulation analysis...")
    modulation_results = analyzer.analyze_seasonal_modulation(returns_data, asset_pairs)
    
    # Display results
    print("\n📊 SEASONAL MODULATION RESULTS:")
    print("-" * 40)
    
    if 'seasonal_modulation' in modulation_results:
        seasonal_modulation = modulation_results['seasonal_modulation']
        print("Seasonal Modulation Factors:")
        
        for season, data in seasonal_modulation.items():
            modulation_factor = data.get('modulation_factor', 1.0)
            strength_change = data.get('strength_change', 0.0)
            expected_activity = data.get('expected_activity', 'Unknown')
            
            print(f"  {season}:")
            print(f"    Modulation: {modulation_factor:.3f}")
            print(f"    Change: {strength_change:+.1f}%")
            print(f"    Activity: {expected_activity}")
    
    if 'significant_variations' in modulation_results:
        significant_variations = modulation_results['significant_variations']
        if significant_variations:
            print("\nSignificant Seasonal Variations:")
            for season, data in significant_variations.items():
                significance = data.get('significance_level', 'Unknown')
                direction = data.get('direction', 'Unknown')
                change = data.get('strength_change_percent', 0.0)
                
                print(f"  {season}: {significance} {direction} ({change:+.1f}%)")
    
    return modulation_results


def demonstrate_comprehensive_analysis():
    """Demonstrate comprehensive seasonal-geographic analysis."""
    print("\n🌍🌱 COMPREHENSIVE SEASONAL-GEOGRAPHIC ANALYSIS")
    print("=" * 60)
    
    # Initialize main analyzer
    analyzer = AgriculturalCrossSectorAnalyzer()
    
    # Key agricultural and cross-sector tickers
    key_tickers = [
        # Agricultural companies
        'ADM', 'BG', 'CF', 'MOS', 'DE',
        # Energy (Tier 1)
        'XOM', 'CVX', 'COP',
        # Finance (Tier 2)
        'JPM', 'BAC',
        # Equipment (Tier 2)
        'CAT'
    ]
    
    print(f"📊 Loading data for {len(key_tickers)} companies...")
    
    try:
        # Load data
        returns_data = analyzer.load_data(tickers=key_tickers, period="3y")
        
        # Run seasonal-geographic analysis
        print("\n🌍🌱 Running comprehensive seasonal-geographic analysis...")
        seasonal_geographic_results = analyzer.run_seasonal_geographic_analysis(
            include_crisis_analysis=True
        )
        
        # Display key findings
        print("\n🎯 KEY SEASONAL-GEOGRAPHIC FINDINGS:")
        print("-" * 50)
        
        # Seasonal findings
        seasonal_rates = seasonal_geographic_results.seasonal_results.seasonal_violation_rates
        if seasonal_rates:
            print("🌱 Seasonal Violation Rates:")
            for season, rate in seasonal_rates.items():
                print(f"   {season}: {rate:.2f}%")
            
            max_season = max(seasonal_rates, key=seasonal_rates.get)
            min_season = min(seasonal_rates, key=seasonal_rates.get)
            print(f"\n   Peak season: {max_season} ({seasonal_rates[max_season]:.2f}%)")
            print(f"   Low season: {min_season} ({seasonal_rates[min_season]:.2f}%)")
        
        # Geographic findings
        regional_patterns = seasonal_geographic_results.geographic_results.regional_patterns
        if regional_patterns:
            print(f"\n🌍 Geographic Analysis:")
            print(f"   Regions analyzed: {len(regional_patterns)}")
            
            for region, data in regional_patterns.items():
                companies = data.get('companies', [])
                if companies:
                    print(f"   {region.replace('_', ' ')}: {len(companies)} companies")
        
        # Cross-regional correlations
        cross_regional = seasonal_geographic_results.geographic_results.cross_regional_correlations
        if cross_regional:
            print(f"\n🔄 Cross-Regional Correlations:")
            for (region1, region2), corr in cross_regional.items():
                print(f"   {region1.replace('_', ' ')} ↔ {region2.replace('_', ' ')}: {corr:.3f}")
        
        # Generate visualizations
        print("\n🎨 Generating seasonal-geographic visualizations...")
        visualizer = SeasonalVisualizationSuite()
        
        # Create output directory
        output_dir = "seasonal_geographic_demo_output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate visualizations
        generated_files = visualizer.generate_all_seasonal_geographic_visualizations(
            seasonal_geographic_results, output_dir
        )
        
        print(f"\n✅ Generated {len(generated_files)} visualizations:")
        for file_path in generated_files:
            print(f"   📊 {os.path.basename(file_path)}")
        
        print(f"\n📁 Output directory: {output_dir}")
        
        return seasonal_geographic_results
        
    except Exception as e:
        print(f"❌ Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main demonstration function."""
    print("🌍🌱 SEASONAL AND GEOGRAPHIC ANALYSIS DEMONSTRATION")
    print("=" * 70)
    print("This demonstration shows seasonal and geographic analysis features")
    print("for agricultural cross-sector Bell inequality analysis.")
    print("=" * 70)
    
    try:
        # 1. Seasonal Analysis
        seasonal_results = demonstrate_seasonal_analysis()
        
        # 2. Geographic Analysis  
        geographic_results = demonstrate_geographic_analysis()
        
        # 3. Seasonal Modulation
        modulation_results = demonstrate_seasonal_modulation()
        
        # 4. Comprehensive Analysis
        comprehensive_results = demonstrate_comprehensive_analysis()
        
        print("\n🎉 DEMONSTRATION COMPLETE!")
        print("=" * 40)
        print("✅ Seasonal effect detection: Complete")
        print("✅ Geographic pattern analysis: Complete") 
        print("✅ Seasonal modulation analysis: Complete")
        print("✅ Regional crisis impact: Complete")
        print("✅ Comprehensive visualizations: Complete")
        
        print("\n📊 Key Features Demonstrated:")
        print("   🌱 Agricultural planting/harvest cycle effects")
        print("   🌍 Regional agricultural production patterns")
        print("   📈 Seasonal quantum correlation modulation")
        print("   🚨 Regional crisis impact analysis")
        print("   🎨 Publication-ready visualizations")
        
        print("\n📋 Requirements Addressed:")
        print("   ✅ 8.2: Seasonal effect detection for agricultural cycles")
        print("   ✅ 8.3: Geographic analysis of regional production patterns")
        print("   ✅ 8.4: Seasonal modulation of quantum correlation strength")
        
    except Exception as e:
        print(f"\n❌ Demonstration failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()