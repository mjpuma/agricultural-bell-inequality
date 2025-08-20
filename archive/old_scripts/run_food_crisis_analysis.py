#!/usr/bin/env python3
"""
FOOD CRISIS QUANTUM CORRELATION ANALYSIS
========================================
Quick script to analyze specific food crises
"""

import sys
sys.path.append('src')

from src.food_systems_analyzer import analyze_food_crisis, FOOD_CRISIS_PERIODS

def main():
    print("🌾 FOOD CRISIS QUANTUM ANALYSIS")
    print("=" * 40)
    
    print("\n📉 Available Food Crisis Periods:")
    for i, (crisis_name, config) in enumerate(FOOD_CRISIS_PERIODS.items(), 1):
        print(f"{i}. {crisis_name.replace('_', ' ').title()}")
        print(f"   {config['start_date']} to {config['end_date']}")
        print(f"   {config['description']}")
        print()
    
    try:
        choice = input("Select crisis to analyze (1-6): ").strip()
        crisis_names = list(FOOD_CRISIS_PERIODS.keys())
        
        if choice.isdigit() and 1 <= int(choice) <= len(crisis_names):
            crisis_name = crisis_names[int(choice) - 1]
            print(f"\n🔍 Analyzing {crisis_name.replace('_', ' ').title()}...")
            
            results = analyze_food_crisis(crisis_name)
            
            if results:
                print(f"\n✅ CRISIS ANALYSIS COMPLETE!")
                violation_rate = results['summary']['overall_violation_rate']
                print(f"🔔 Bell Violation Rate: {violation_rate:.2f}%")
                
                if violation_rate > 80:
                    print("🌟 EXTREME quantum entanglement during crisis!")
                elif violation_rate > 50:
                    print("⚡ STRONG quantum correlations detected!")
                else:
                    print("✅ Moderate quantum effects observed")
                    
                print(f"📊 Visualization saved for {crisis_name}")
            else:
                print("❌ Analysis failed")
        else:
            print("Invalid choice")
            
    except KeyboardInterrupt:
        print("\n⚠️ Analysis interrupted")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()