#!/usr/bin/env python3
"""
CREATE S1 MATRICES FOR FOOD SYSTEMS ANALYSIS
============================================
Generate S1 violation matrices while managing file size
"""

import sys
sys.path.append('src')

import pandas as pd
import numpy as np
from datetime import datetime
import os

from src.food_systems_analyzer import FoodSystemsBellAnalyzer

def create_lightweight_s1_matrix(assets, period='1y', max_pairs=10):
    """Create S1 matrix for specific assets with size control"""
    
    print(f"🔍 Creating S1 matrix for {assets}")
    
    try:
        analyzer = FoodSystemsBellAnalyzer(assets, period=period)
        if analyzer.load_data():
            results = analyzer.run_s1_analysis()
            
            if results and 'violation_details' in results:
                # Extract S1 violation matrix
                violation_details = results['violation_details']
                
                # Create summary matrix instead of full time series
                s1_summary = {}
                
                if 'pair_results' in results:
                    for pair_name, pair_data in results['pair_results'].items():
                        s1_summary[pair_name] = {
                            'violation_rate': pair_data['violation_rate'],
                            'total_violations': pair_data['violations'],
                            'total_windows': pair_data['total_windows'],
                            'max_violation_pct': pair_data.get('max_violation_pct', 0)
                        }
                
                return s1_summary, results['summary']
            
    except Exception as e:
        print(f"   ❌ Error creating S1 matrix: {e}")
        return None, None
    
    return None, None

def create_food_systems_s1_matrices():
    """Create S1 matrices for successful food systems"""
    
    print("📊 CREATING FOOD SYSTEMS S1 MATRICES")
    print("=" * 50)
    
    # Focus on successful groups only
    successful_analyses = {
        'food_companies': ['ADM', 'BG', 'CAG', 'CPB', 'GIS', 'K', 'KHC', 'MDLZ', 'MKC', 'SJM'],
        'fertilizer': ['CF', 'MOS', 'NTR', 'IPI'],
        'food_retail': ['WMT', 'COST', 'KR', 'SYY'],
        'farm_equipment': ['DE', 'CAT', 'AGCO']
    }
    
    all_s1_data = []
    
    for group_name, assets in successful_analyses.items():
        print(f"\n🔍 Processing {group_name}...")
        
        s1_matrix, summary = create_lightweight_s1_matrix(assets, period='1y')
        
        if s1_matrix:
            for pair_name, pair_stats in s1_matrix.items():
                all_s1_data.append({
                    'Group': group_name,
                    'Pair': pair_name,
                    'Asset_1': pair_name.split('-')[0],
                    'Asset_2': pair_name.split('-')[1],
                    'Violation_Rate': pair_stats['violation_rate'],
                    'Total_Violations': pair_stats['total_violations'],
                    'Total_Windows': pair_stats['total_windows'],
                    'Max_Violation_Pct': pair_stats.get('max_violation_pct', 0),
                    'Period': '1y',
                    'Analysis_Date': datetime.now().strftime("%Y-%m-%d")
                })
            print(f"   ✅ {group_name}: {len(s1_matrix)} pairs processed")
        else:
            print(f"   ❌ {group_name}: Failed to create S1 matrix")
    
    return all_s1_data

def create_crisis_s1_matrices():
    """Create S1 matrices for crisis periods"""
    
    print("\n📉 CREATING CRISIS PERIOD S1 MATRICES")
    print("=" * 40)
    
    successful_crises = {
        'covid_food_disruption': ('2020-03-01', '2020-12-31'),
        'ukraine_war_food_crisis': ('2022-02-24', '2023-12-31'),
        'drought_2012': ('2012-06-01', '2012-12-31'),
        'china_food_demand_surge': ('2020-06-01', '2021-12-31')
    }
    
    crisis_s1_data = []
    
    for crisis_name, (start_date, end_date) in successful_crises.items():
        print(f"\n🔍 Processing {crisis_name}...")
        
        try:
            assets = ['CORN', 'WEAT', 'SOYB']
            analyzer = FoodSystemsBellAnalyzer(assets)
            analyzer.start_date = start_date
            analyzer.end_date = end_date
            
            if analyzer.load_data():
                results = analyzer.run_s1_analysis()
                
                if results and 'pair_results' in results:
                    for pair_name, pair_data in results['pair_results'].items():
                        crisis_s1_data.append({
                            'Crisis': crisis_name,
                            'Pair': pair_name,
                            'Asset_1': pair_name.split('-')[0],
                            'Asset_2': pair_name.split('-')[1],
                            'Violation_Rate': pair_data['violation_rate'],
                            'Total_Violations': pair_data['violations'],
                            'Total_Windows': pair_data['total_windows'],
                            'Start_Date': start_date,
                            'End_Date': end_date,
                            'Analysis_Date': datetime.now().strftime("%Y-%m-%d")
                        })
                    print(f"   ✅ {crisis_name}: {len(results['pair_results'])} pairs processed")
                else:
                    print(f"   ❌ {crisis_name}: No pair results")
            else:
                print(f"   ❌ {crisis_name}: Data loading failed")
                
        except Exception as e:
            print(f"   ❌ {crisis_name}: Error - {e}")
    
    return crisis_s1_data

def export_s1_matrices_to_excel():
    """Export S1 matrices to Excel (lightweight version)"""
    
    print("🚀 EXPORTING S1 MATRICES TO EXCEL")
    print("=" * 40)
    
    # Get S1 data
    food_s1_data = create_food_systems_s1_matrices()
    crisis_s1_data = create_crisis_s1_matrices()
    
    # Create Excel file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"Food_Systems_S1_Matrices_{timestamp}.xlsx"
    
    print(f"\n📁 Creating S1 matrices file: {filename}")
    
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        
        # S1 matrices sheets
        if food_s1_data:
            pd.DataFrame(food_s1_data).to_excel(writer, sheet_name='Food_Groups_S1_Matrix', index=False)
        
        if crisis_s1_data:
            pd.DataFrame(crisis_s1_data).to_excel(writer, sheet_name='Crisis_Periods_S1_Matrix', index=False)
        
        # Combined top violating pairs
        all_s1_data = food_s1_data + crisis_s1_data
        if all_s1_data:
            # Sort by violation rate
            sorted_data = sorted(all_s1_data, key=lambda x: x.get('Violation_Rate', 0), reverse=True)
            pd.DataFrame(sorted_data[:20]).to_excel(writer, sheet_name='Top_20_S1_Violations', index=False)
        
        # S1 methodology info
        methodology = {
            'Method': ['S1 Conditional Bell Inequality'],
            'Reference': ['Zarifian et al. (2025)'],
            'Window_Size': [20],
            'Threshold_Quantile': [0.75],
            'Data_Type': ['Cumulative returns'],
            'File_Size': ['Lightweight (<5MB vs 120MB full matrices)'],
            'Content': ['Summary statistics per pair, not raw time series'],
            'Advantage': ['Fast analysis, easy comparison, manageable file size'],
            'Note': ['For full time-series matrices, run individual analyses']
        }
        pd.DataFrame(methodology).to_excel(writer, sheet_name='S1_Methodology', index=False)
    
    # File size check
    file_size = os.path.getsize(filename) / (1024 * 1024)  # MB
    
    print(f"✅ S1 matrices file created: {filename}")
    print(f"📊 File size: {file_size:.2f} MB (vs 120MB for full matrices)")
    
    # Summary
    print(f"\n📊 S1 MATRICES SUMMARY:")
    print(f"   Food groups S1 pairs: {len(food_s1_data)}")
    print(f"   Crisis periods S1 pairs: {len(crisis_s1_data)}")
    print(f"   Total S1 pairs: {len(all_s1_data)}")
    
    if all_s1_data:
        top_s1 = max(all_s1_data, key=lambda x: x.get('Violation_Rate', 0))
        print(f"   Top S1 violation: {top_s1.get('Pair', 'N/A')} ({top_s1.get('Violation_Rate', 0):.2f}%)")
    
    print(f"\n💡 LIGHTWEIGHT APPROACH BENEFITS:")
    print(f"   ✅ Manageable file size ({file_size:.2f} MB vs 120MB)")
    print(f"   ✅ Fast loading and analysis")
    print(f"   ✅ Easy comparison across pairs")
    print(f"   ✅ Contains all key S1 statistics")
    print(f"   ✅ Suitable for WDRS comparison")
    
    return filename

if __name__ == "__main__":
    s1_filename = export_s1_matrices_to_excel()
    print(f"\n🎉 S1 MATRICES COMPLETE!")
    print(f"📁 S1 matrices file: {s1_filename}")
    print(f"📊 Lightweight approach avoids 120MB file size issue")
    print(f"🔬 Contains all essential S1 Bell inequality statistics")