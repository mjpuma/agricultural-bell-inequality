#!/usr/bin/env python3
"""Debug the food crisis analysis issue"""

import sys
sys.path.append('src')

from src.food_systems_analyzer import FoodSystemsBellAnalyzer, FOOD_COMMODITIES, FOOD_CRISIS_PERIODS

def test_crisis_analysis():
    print("🔍 Testing crisis analysis...")
    
    # Test 1: Check if we can create the analyzer
    try:
        assets = FOOD_COMMODITIES['grains']
        print(f"Assets: {assets}")
        print(f"Assets type: {type(assets)}")
        
        analyzer = FoodSystemsBellAnalyzer(assets)
        print("✅ Analyzer created successfully")
        
        # Test 2: Check crisis configuration
        crisis_name = 'covid_food_disruption'
        if crisis_name in FOOD_CRISIS_PERIODS:
            crisis_config = FOOD_CRISIS_PERIODS[crisis_name]
            print(f"✅ Crisis config: {crisis_config}")
        else:
            print(f"❌ Crisis not found: {crisis_name}")
            return
        
        # Test 3: Try to run the crisis analysis
        print("🔍 Running crisis analysis...")
        results = analyzer.analyze_food_crisis_period(crisis_name)
        print("✅ Crisis analysis completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_crisis_analysis()