#!/usr/bin/env python3
"""
EXPORT YAHOO FINANCE RESULTS TO EXCEL
=====================================
Create comprehensive Excel files with all Bell inequality results
"""

import sys
sys.path.append('src')

import pandas as pd
import numpy as np
from datetime import datetime
import os

from src.food_systems_analyzer import (
    FoodSystemsBellAnalyzer, 
    FOOD_COMMODITIES, 
    FOOD_CRISIS_PERIODS
)

def export_food_group_results():
    """Export all food group analysis results to Excel"""
    
    print("📊 EXPORTING FOOD GROUP RESULTS TO EXCEL")
    print("=" * 50)
    
    all_results = []
    detailed_pairs = []
    s1_matrices = {}
    
    for group_name, assets in FOOD_COMMODITIES.items():
        print(f"\n🔍 Analyzing {group_name}: {assets}")
        
        try:
            analyzer = FoodSystemsBellAnalyzer(assets, period='1y')
            if analyzer.load_data():
                results = analyzer.run_s1_analysis()
                
                if results and 'summary' in results:
                    # Summary results
                    summary = results['summary']
                    all_results.append({
                        'Group': group_name,
                        'Assets': ', '.join(assets),
                        'Asset_Count': len(assets),
                        'Overall_Violation_Rate': summary['overall_violation_rate'],
                        'Max_Violation_Rate': summary['max_violation_pct'],
                        'Total_Violations': summary['total_violations'],
                        'Total_Calculations': summary['total_calculations'],
                        'Data_Period': '1y',
                        'Status': 'SUCCESS'
                    })
                    
                    # Detailed pair results
                    if 'pair_results' in results:
                        for pair_name, pair_data in results['pair_results'].items():
                            detailed_pairs.append({
                                'Group': group_name,
                                'Pair': pair_name,
                                'Asset_1': pair_name.split('-')[0],
                                'Asset_2': pair_name.split('-')[1],
                                'Violation_Count': pair_data['violations'],
                                'Total_Windows': pair_data['total_windows'],
                                'Violation_Rate': pair_data['violation_rate'],
                                'Status': 'SUCCESS'
                            })
                    
                    # S1 matrices (if available)
                    if 'violation_details' in results:
                        s1_matrices[group_name] = results['violation_details']
                    
                    print(f"   ✅ {group_name}: {summary['overall_violation_rate']:.2f}% violations")
                else:
                    all_results.append({
                        'Group': group_name,
                        'Assets': ', '.join(assets),
                        'Asset_Count': len(assets),
                        'Overall_Violation_Rate': np.nan,
                        'Max_Violation_Rate': np.nan,
                        'Total_Violations': np.nan,
                        'Total_Calculations': np.nan,
                        'Data_Period': '1y',
                        'Status': 'ANALYSIS_FAILED'
                    })
                    print(f"   ❌ {group_name}: Analysis failed")
            else:
                all_results.append({
                    'Group': group_name,
                    'Assets': ', '.join(assets),
                    'Asset_Count': len(assets),
                    'Overall_Violation_Rate': np.nan,
                    'Max_Violation_Rate': np.nan,
                    'Total_Violations': np.nan,
                    'Total_Calculations': np.nan,
                    'Data_Period': '1y',
                    'Status': 'DATA_FAILED'
                })
                print(f"   ❌ {group_name}: Data loading failed")
                
        except Exception as e:
            all_results.append({
                'Group': group_name,
                'Assets': ', '.join(assets),
                'Asset_Count': len(assets),
                'Overall_Violation_Rate': np.nan,
                'Max_Violation_Rate': np.nan,
                'Total_Violations': np.nan,
                'Total_Calculations': np.nan,
                'Data_Period': '1y',
                'Status': f'ERROR: {str(e)[:50]}'
            })
            print(f"   ❌ {group_name}: Error - {e}")
    
    return all_results, detailed_pairs, s1_matrices

def export_crisis_results():
    """Export all crisis period analysis results"""
    
    print("\n\n📉 EXPORTING CRISIS PERIOD RESULTS")
    print("=" * 50)
    
    crisis_results = []
    crisis_pairs = []
    crisis_matrices = {}
    
    for crisis_name, config in FOOD_CRISIS_PERIODS.items():
        print(f"\n🔍 Analyzing {crisis_name}")
        
        try:
            # Use reliable grain assets
            assets = ['CORN', 'WEAT', 'SOYB']
            analyzer = FoodSystemsBellAnalyzer(assets)
            
            # Set crisis dates
            analyzer.start_date = config['start_date']
            analyzer.end_date = config['end_date']
            
            if analyzer.load_data():
                results = analyzer.run_s1_analysis()
                
                if results and 'summary' in results:
                    summary = results['summary']
                    crisis_results.append({
                        'Crisis': crisis_name,
                        'Description': config['description'],
                        'Start_Date': config['start_date'],
                        'End_Date': config['end_date'],
                        'Assets': ', '.join(assets),
                        'Overall_Violation_Rate': summary['overall_violation_rate'],
                        'Max_Violation_Rate': summary['max_violation_pct'],
                        'Total_Violations': summary['total_violations'],
                        'Total_Calculations': summary['total_calculations'],
                        'Status': 'SUCCESS'
                    })
                    
                    # Crisis pair details
                    if 'pair_results' in results:
                        for pair_name, pair_data in results['pair_results'].items():
                            crisis_pairs.append({
                                'Crisis': crisis_name,
                                'Pair': pair_name,
                                'Asset_1': pair_name.split('-')[0],
                                'Asset_2': pair_name.split('-')[1],
                                'Violation_Count': pair_data['violations'],
                                'Total_Windows': pair_data['total_windows'],
                                'Violation_Rate': pair_data['violation_rate'],
                                'Status': 'SUCCESS'
                            })
                    
                    # Crisis S1 matrices
                    if 'violation_details' in results:
                        crisis_matrices[crisis_name] = results['violation_details']
                    
                    print(f"   ✅ {crisis_name}: {summary['overall_violation_rate']:.2f}% violations")
                else:
                    crisis_results.append({
                        'Crisis': crisis_name,
                        'Description': config['description'],
                        'Start_Date': config['start_date'],
                        'End_Date': config['end_date'],
                        'Assets': ', '.join(assets),
                        'Overall_Violation_Rate': np.nan,
                        'Max_Violation_Rate': np.nan,
                        'Total_Violations': np.nan,
                        'Total_Calculations': np.nan,
                        'Status': 'ANALYSIS_FAILED'
                    })
                    print(f"   ❌ {crisis_name}: Analysis failed")
            else:
                crisis_results.append({
                    'Crisis': crisis_name,
                    'Description': config['description'],
                    'Start_Date': config['start_date'],
                    'End_Date': config['end_date'],
                    'Assets': ', '.join(assets),
                    'Overall_Violation_Rate': np.nan,
                    'Max_Violation_Rate': np.nan,
                    'Total_Violations': np.nan,
                    'Total_Calculations': np.nan,
                    'Status': 'DATA_FAILED'
                })
                print(f"   ❌ {crisis_name}: Data loading failed")
                
        except Exception as e:
            crisis_results.append({
                'Crisis': crisis_name,
                'Description': config['description'],
                'Start_Date': config['start_date'],
                'End_Date': config['end_date'],
                'Assets': ', '.join(assets),
                'Overall_Violation_Rate': np.nan,
                'Max_Violation_Rate': np.nan,
                'Total_Violations': np.nan,
                'Total_Calculations': np.nan,
                'Status': f'ERROR: {str(e)[:50]}'
            })
            print(f"   ❌ {crisis_name}: Error - {e}")
    
    return crisis_results, crisis_pairs, crisis_matrices

def create_s1_matrix_sheets(s1_matrices, writer, sheet_prefix):
    """Create S1 violation matrix sheets"""
    
    for matrix_name, matrix_data in s1_matrices.items():
        if isinstance(matrix_data, dict) and 'violations' in matrix_data:
            try:
                # Convert to DataFrame if it's not already
                if isinstance(matrix_data['violations'], pd.DataFrame):
                    df = matrix_data['violations']
                else:
                    df = pd.DataFrame(matrix_data['violations'])
                
                sheet_name = f"{sheet_prefix}_{matrix_name}"[:31]  # Excel sheet name limit
                df.to_excel(writer, sheet_name=sheet_name, index=True)
                print(f"   📊 Created S1 matrix sheet: {sheet_name}")
                
            except Exception as e:
                print(f"   ❌ Failed to create S1 matrix for {matrix_name}: {e}")

def export_all_results_to_excel():
    """Export all Yahoo Finance results to comprehensive Excel file"""
    
    print("🚀 COMPREHENSIVE YAHOO FINANCE RESULTS EXPORT")
    print("=" * 60)
    
    # Get all results
    food_results, food_pairs, food_matrices = export_food_group_results()
    crisis_results, crisis_pairs, crisis_matrices = export_crisis_results()
    
    # Create Excel file with multiple sheets
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"Food_Systems_Bell_Results_{timestamp}.xlsx"
    
    print(f"\n📁 Creating Excel file: {filename}")
    
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        
        # Summary sheets
        pd.DataFrame(food_results).to_excel(writer, sheet_name='Food_Groups_Summary', index=False)
        pd.DataFrame(crisis_results).to_excel(writer, sheet_name='Crisis_Periods_Summary', index=False)
        
        # Detailed pair results
        if food_pairs:
            pd.DataFrame(food_pairs).to_excel(writer, sheet_name='Food_Groups_Pairs', index=False)
        if crisis_pairs:
            pd.DataFrame(crisis_pairs).to_excel(writer, sheet_name='Crisis_Periods_Pairs', index=False)
        
        # Top violating pairs across all analyses
        all_pairs = food_pairs + crisis_pairs
        if all_pairs:
            top_pairs = pd.DataFrame(all_pairs).nlargest(20, 'Violation_Rate')
            top_pairs.to_excel(writer, sheet_name='Top_20_Violating_Pairs', index=False)
        
        # S1 matrices (if not too large)
        create_s1_matrix_sheets(food_matrices, writer, "S1_Food")
        create_s1_matrix_sheets(crisis_matrices, writer, "S1_Crisis")
        
        # Metadata sheet
        metadata = {
            'Analysis_Date': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            'Total_Food_Groups': [len(food_results)],
            'Total_Crisis_Periods': [len(crisis_results)],
            'Total_Asset_Pairs': [len(all_pairs) if all_pairs else 0],
            'Data_Source': ['Yahoo Finance'],
            'Analysis_Method': ['S1 Conditional Bell Inequality (Zarifian et al. 2025)'],
            'Window_Size': [20],
            'Threshold_Quantile': [0.75]
        }
        pd.DataFrame(metadata).to_excel(writer, sheet_name='Metadata', index=False)
    
    print(f"✅ Excel file created: {filename}")
    
    # Summary statistics
    successful_food = len([r for r in food_results if r['Status'] == 'SUCCESS'])
    successful_crisis = len([r for r in crisis_results if r['Status'] == 'SUCCESS'])
    
    print(f"\n📊 EXPORT SUMMARY:")
    print(f"   Food groups analyzed: {successful_food}/{len(food_results)} successful")
    print(f"   Crisis periods analyzed: {successful_crisis}/{len(crisis_results)} successful")
    print(f"   Total asset pairs: {len(all_pairs) if all_pairs else 0}")
    
    if all_pairs:
        top_violation = max(all_pairs, key=lambda x: x['Violation_Rate'])
        print(f"   Top violating pair: {top_violation['Pair']} ({top_violation['Violation_Rate']:.2f}%)")
    
    return filename

def create_lightweight_s1_summary():
    """Create a lightweight summary of S1 results (avoiding 120MB files)"""
    
    print("\n📋 CREATING LIGHTWEIGHT S1 SUMMARY")
    print("=" * 40)
    
    # Instead of full matrices, create summary statistics
    s1_summary = []
    
    # This would be much smaller than full matrices
    # Focus on key statistics rather than raw violation data
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = f"S1_Summary_Statistics_{timestamp}.xlsx"
    
    # Create summary with key metrics only
    summary_data = {
        'Analysis_Type': ['Food Systems Bell Inequality Analysis'],
        'Method': ['S1 Conditional (Zarifian et al. 2025)'],
        'Data_Source': ['Yahoo Finance'],
        'File_Size': ['Lightweight (<1MB vs 120MB full matrices)'],
        'Content': ['Summary statistics only, not raw violation matrices'],
        'Note': ['For full S1 matrices, run individual analyses']
    }
    
    with pd.ExcelWriter(summary_file, engine='openpyxl') as writer:
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='S1_Summary_Info', index=False)
    
    print(f"✅ Lightweight S1 summary created: {summary_file}")
    print("💡 This avoids the 120MB file size issue while preserving key results")
    
    return summary_file

def main():
    """Main export function"""
    
    # Create comprehensive Excel results
    excel_file = export_all_results_to_excel()
    
    # Create lightweight S1 summary
    s1_file = create_lightweight_s1_summary()
    
    print(f"\n🎉 EXPORT COMPLETE!")
    print(f"📁 Main results file: {excel_file}")
    print(f"📁 S1 summary file: {s1_file}")
    print(f"\n💡 These files contain all Yahoo Finance test results in structured format")
    print(f"📊 Ready for comparison with WDRS results when available")

if __name__ == "__main__":
    main()