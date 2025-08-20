#!/usr/bin/env python3
"""
COMPLETE YAHOO FINANCE FOOD SYSTEMS TEST
========================================
Test all food system groups and time periods to see what works
"""

import sys
sys.path.append('src')

from src.food_systems_analyzer import (
    FoodSystemsBellAnalyzer, 
    FOOD_COMMODITIES, 
    FOOD_CRISIS_PERIODS,
    run_science_publication_analysis
)

def test_all_food_groups():
    """Test all food commodity groups"""
    print("🌾 TESTING ALL FOOD COMMODITY GROUPS")
    print("=" * 50)
    
    results = {}
    
    for group_name, assets in FOOD_COMMODITIES.items():
        print(f"\n🔍 Testing {group_name}: {assets}")
        
        try:
            analyzer = FoodSystemsBellAnalyzer(assets, period='1y')
            if analyzer.load_data():
                group_results = analyzer.run_s1_analysis()
                if group_results:
                    violation_rate = group_results['summary']['overall_violation_rate']
                    results[group_name] = {
                        'status': 'SUCCESS',
                        'violation_rate': violation_rate,
                        'assets': assets
                    }
                    print(f"   ✅ {group_name}: {violation_rate:.2f}% Bell violations")
                else:
                    results[group_name] = {'status': 'ANALYSIS_FAILED', 'assets': assets}
                    print(f"   ❌ {group_name}: Analysis failed")
            else:
                results[group_name] = {'status': 'DATA_FAILED', 'assets': assets}
                print(f"   ❌ {group_name}: Data loading failed")
                
        except Exception as e:
            results[group_name] = {'status': 'ERROR', 'error': str(e), 'assets': assets}
            print(f"   ❌ {group_name}: Error - {e}")
    
    return results

def test_all_crisis_periods():
    """Test all crisis periods"""
    print("\n\n📉 TESTING ALL CRISIS PERIODS")
    print("=" * 50)
    
    crisis_results = {}
    
    for crisis_name, config in FOOD_CRISIS_PERIODS.items():
        print(f"\n🔍 Testing {crisis_name}")
        print(f"   Period: {config['start_date']} to {config['end_date']}")
        
        try:
            # Use reliable assets
            assets = ['CORN', 'WEAT', 'SOYB']
            analyzer = FoodSystemsBellAnalyzer(assets)
            
            # Set crisis dates
            analyzer.start_date = config['start_date']
            analyzer.end_date = config['end_date']
            
            if analyzer.load_data():
                results = analyzer.run_s1_analysis()
                if results:
                    violation_rate = results['summary']['overall_violation_rate']
                    crisis_results[crisis_name] = {
                        'status': 'SUCCESS',
                        'violation_rate': violation_rate,
                        'period': f"{config['start_date']} to {config['end_date']}"
                    }
                    print(f"   ✅ {crisis_name}: {violation_rate:.2f}% Bell violations")
                else:
                    crisis_results[crisis_name] = {'status': 'ANALYSIS_FAILED'}
                    print(f"   ❌ {crisis_name}: Analysis failed")
            else:
                crisis_results[crisis_name] = {'status': 'DATA_FAILED'}
                print(f"   ❌ {crisis_name}: Data loading failed")
                
        except Exception as e:
            crisis_results[crisis_name] = {'status': 'ERROR', 'error': str(e)}
            print(f"   ❌ {crisis_name}: Error - {e}")
    
    return crisis_results

def summarize_yahoo_finance_capabilities(group_results, crisis_results):
    """Summarize what works with Yahoo Finance"""
    print("\n\n📊 YAHOO FINANCE CAPABILITIES SUMMARY")
    print("=" * 60)
    
    # Food groups summary
    print("\n🌾 FOOD COMMODITY GROUPS:")
    successful_groups = []
    for group, result in group_results.items():
        if result['status'] == 'SUCCESS':
            successful_groups.append(group)
            print(f"   ✅ {group}: {result['violation_rate']:.2f}% violations")
        else:
            print(f"   ❌ {group}: {result['status']}")
    
    # Crisis periods summary
    print("\n📉 CRISIS PERIODS:")
    successful_crises = []
    for crisis, result in crisis_results.items():
        if result['status'] == 'SUCCESS':
            successful_crises.append(crisis)
            print(f"   ✅ {crisis}: {result['violation_rate']:.2f}% violations")
        else:
            print(f"   ❌ {crisis}: {result['status']}")
    
    # Overall assessment
    print(f"\n🎯 YAHOO FINANCE READINESS:")
    print(f"   Working food groups: {len(successful_groups)}/{len(group_results)}")
    print(f"   Working crisis periods: {len(successful_crises)}/{len(crisis_results)}")
    
    if len(successful_groups) >= 3 and len(successful_crises) >= 3:
        print(f"   🌟 READY FOR SCIENCE PUBLICATION!")
        print(f"   📊 Sufficient data for comprehensive analysis")
    elif len(successful_groups) >= 2 and len(successful_crises) >= 2:
        print(f"   ⚡ GOOD FOUNDATION for analysis")
        print(f"   📝 May need WDRS for full validation")
    else:
        print(f"   ⚠️  LIMITED DATA AVAILABILITY")
        print(f"   🔄 Consider WDRS for better coverage")
    
    # Top findings
    if successful_groups:
        top_group = max([g for g in group_results.values() if g['status'] == 'SUCCESS'], 
                       key=lambda x: x['violation_rate'])
        print(f"\n🔔 TOP FOOD GROUP: {top_group['violation_rate']:.2f}% violations")
    
    if successful_crises:
        top_crisis = max([c for c in crisis_results.values() if c['status'] == 'SUCCESS'], 
                        key=lambda x: x['violation_rate'])
        print(f"🔔 TOP CRISIS PERIOD: {top_crisis['violation_rate']:.2f}% violations")

def main():
    print("🚀 COMPREHENSIVE YAHOO FINANCE FOOD SYSTEMS TEST")
    print("=" * 60)
    print("Testing all food groups and crisis periods...")
    
    # Test all food groups
    group_results = test_all_food_groups()
    
    # Test all crisis periods  
    crisis_results = test_all_crisis_periods()
    
    # Summarize capabilities
    summarize_yahoo_finance_capabilities(group_results, crisis_results)
    
    print(f"\n✅ COMPREHENSIVE TEST COMPLETE!")
    print(f"📋 Results show Yahoo Finance readiness for food systems analysis")

if __name__ == "__main__":
    main()