"""
Integrated Cross-Sector Transmission Analysis

This script demonstrates how the Cross-Sector Transmission Detection System
integrates with the existing agricultural analysis framework to provide
comprehensive cross-sector analysis capabilities.

Combines:
1. Agricultural Universe Management
2. Enhanced S1 Bell Inequality Calculator  
3. Cross-Sector Transmission Detection
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from agricultural_universe_manager import AgriculturalUniverseManager
from enhanced_s1_calculator import EnhancedS1Calculator
from cross_sector_transmission_detector import CrossSectorTransmissionDetector


def run_integrated_analysis():
    """Run integrated cross-sector transmission and Bell inequality analysis"""
    print("🌾 Integrated Cross-Sector Transmission Analysis")
    print("=" * 70)
    
    # Initialize components
    universe_manager = AgriculturalUniverseManager()
    s1_calculator = EnhancedS1Calculator(window_size=20, threshold_method='quantile')
    transmission_detector = CrossSectorTransmissionDetector(transmission_window=90)
    
    print("📊 System Components Initialized:")
    print(f"  • Agricultural Universe: {len(universe_manager.companies)} companies")
    print(f"  • S1 Calculator: {s1_calculator.window_size}-day window, {s1_calculator.threshold_method} thresholds")
    print(f"  • Transmission Detector: {transmission_detector.transmission_window}-day window")
    print()
    
    # Get agricultural companies by tier
    tier_1_companies = universe_manager.classify_by_tier(1)  # Energy/Transport/Chemicals
    tier_2_companies = universe_manager.classify_by_tier(2)  # Finance/Equipment
    agricultural_companies = [comp for comp in universe_manager.companies.keys() 
                            if universe_manager.companies[comp].exposure == 'Primary']
    
    print("🏢 Company Classification:")
    print(f"  • Tier 1 (Energy/Transport/Chemicals): {len(tier_1_companies)} companies")
    print(f"  • Tier 2 (Finance/Equipment): {len(tier_2_companies)} companies") 
    print(f"  • Primary Agricultural: {len(agricultural_companies)} companies")
    print()
    
    # Focus on key transmission pairs for demonstration
    key_transmission_pairs = [
        # Energy → Agriculture (fertilizer dependency)
        ('XOM', 'CF'),   # ExxonMobil → CF Industries
        ('CVX', 'MOS'),  # Chevron → Mosaic
        ('COP', 'NTR'),  # ConocoPhillips → Nutrien
        
        # Transportation → Agriculture (logistics)
        ('UNP', 'ADM'),  # Union Pacific → Archer Daniels Midland
        ('CSX', 'BG'),   # CSX Corp → Bunge
        
        # Chemicals → Agriculture (input costs)
        ('DOW', 'CF'),   # Dow Chemical → CF Industries
        ('DD', 'MOS'),   # DuPont → Mosaic
        
        # Equipment → Agriculture (direct operational)
        ('CAT', 'DE'),   # Caterpillar → John Deere
    ]
    
    print("🔍 Analyzing Key Cross-Sector Transmission Pairs:")
    print("-" * 50)
    
    transmission_results = []
    bell_violation_results = []
    
    for source_asset, target_asset in key_transmission_pairs:
        print(f"\n📈 Analyzing {source_asset} → {target_asset}:")
        
        try:
            # 1. Transmission Detection Analysis
            print("  🔄 Transmission Detection...")
            
            # Determine transmission type based on source asset
            if source_asset in ['XOM', 'CVX', 'COP']:
                mechanism = transmission_detector.transmission_mechanisms['energy_agriculture']
            elif source_asset in ['UNP', 'CSX', 'FDX']:
                mechanism = transmission_detector.transmission_mechanisms['transport_agriculture']
            elif source_asset in ['DOW', 'DD', 'LYB']:
                mechanism = transmission_detector.transmission_mechanisms['chemicals_agriculture']
            else:
                mechanism = transmission_detector.transmission_mechanisms['finance_agriculture']
            
            transmission_result = transmission_detector._analyze_transmission_pair(
                source_asset=source_asset,
                target_asset=target_asset,
                mechanism=mechanism
            )
            
            transmission_results.append(transmission_result)
            
            if transmission_result.transmission_detected:
                print(f"    ✅ Transmission detected (lag: {transmission_result.transmission_lag} days, "
                      f"r={transmission_result.correlation_strength:.3f})")
            else:
                print(f"    ❌ No significant transmission (r={transmission_result.correlation_strength:.3f})")
            
            # 2. Bell Inequality Analysis
            print("  🔔 Bell Inequality Analysis...")
            
            try:
                # Download data for both assets
                source_data = transmission_detector._download_asset_data(source_asset)
                target_data = transmission_detector._download_asset_data(target_asset)
                
                if source_data is not None and target_data is not None:
                    # Calculate returns
                    source_returns = source_data['Close'].pct_change().dropna()
                    target_returns = target_data['Close'].pct_change().dropna()
                    
                    # Align data
                    common_dates = source_returns.index.intersection(target_returns.index)
                    if len(common_dates) >= 100:
                        aligned_returns = pd.DataFrame({
                            source_asset: source_returns.loc[common_dates],
                            target_asset: target_returns.loc[common_dates]
                        })
                        
                        # Calculate S1 Bell inequality
                        s1_results = s1_calculator.calculate_s1_rolling_analysis(aligned_returns)
                        
                        if s1_results and len(s1_results['s1_values']) > 0:
                            violation_rate = s1_results['violation_rate']
                            avg_s1 = np.mean(np.abs(s1_results['s1_values']))
                            
                            bell_violation_results.append({
                                'pair': (source_asset, target_asset),
                                'violation_rate': violation_rate,
                                'avg_s1': avg_s1,
                                'transmission_detected': transmission_result.transmission_detected
                            })
                            
                            print(f"    🔔 Bell violations: {violation_rate:.1f}% (avg |S1|: {avg_s1:.3f})")
                        else:
                            print("    ⚠️  Insufficient data for Bell analysis")
                    else:
                        print("    ⚠️  Insufficient overlapping data")
                else:
                    print("    ⚠️  Could not download data")
                    
            except Exception as e:
                print(f"    ❌ Bell analysis error: {str(e)}")
            
        except Exception as e:
            print(f"  ❌ Analysis error: {str(e)}")
            continue
    
    # Summary Analysis
    print("\n📊 Integrated Analysis Summary:")
    print("=" * 70)
    
    # Transmission Summary
    detected_transmissions = [r for r in transmission_results if r.transmission_detected]
    print(f"🔄 Transmission Detection:")
    print(f"  • Total pairs analyzed: {len(transmission_results)}")
    print(f"  • Transmissions detected: {len(detected_transmissions)}")
    print(f"  • Detection rate: {len(detected_transmissions)/len(transmission_results)*100:.1f}%")
    
    if detected_transmissions:
        avg_lag = np.mean([r.transmission_lag for r in detected_transmissions])
        avg_correlation = np.mean([abs(r.correlation_strength) for r in detected_transmissions])
        print(f"  • Average lag: {avg_lag:.0f} days")
        print(f"  • Average correlation: {avg_correlation:.3f}")
    
    # Bell Violation Summary
    if bell_violation_results:
        print(f"\n🔔 Bell Inequality Analysis:")
        print(f"  • Pairs with Bell analysis: {len(bell_violation_results)}")
        
        avg_violation_rate = np.mean([r['violation_rate'] for r in bell_violation_results])
        avg_s1 = np.mean([r['avg_s1'] for r in bell_violation_results])
        
        print(f"  • Average violation rate: {avg_violation_rate:.1f}%")
        print(f"  • Average |S1|: {avg_s1:.3f}")
        
        # Correlation between transmission and Bell violations
        transmission_pairs = [r for r in bell_violation_results if r['transmission_detected']]
        no_transmission_pairs = [r for r in bell_violation_results if not r['transmission_detected']]
        
        if transmission_pairs and no_transmission_pairs:
            transmission_violation_rate = np.mean([r['violation_rate'] for r in transmission_pairs])
            no_transmission_violation_rate = np.mean([r['violation_rate'] for r in no_transmission_pairs])
            
            print(f"\n🔗 Transmission-Bell Violation Correlation:")
            print(f"  • Pairs with transmission: {transmission_violation_rate:.1f}% Bell violations")
            print(f"  • Pairs without transmission: {no_transmission_violation_rate:.1f}% Bell violations")
            
            if transmission_violation_rate > no_transmission_violation_rate:
                print("  ✅ Transmission correlates with higher Bell violation rates!")
            else:
                print("  ❌ No clear correlation between transmission and Bell violations")
    
    # Create integrated visualization
    create_integrated_visualization(transmission_results, bell_violation_results)
    
    return transmission_results, bell_violation_results


def create_integrated_visualization(transmission_results, bell_results):
    """Create integrated visualization of transmission and Bell violation results"""
    print("\n📊 Creating Integrated Visualization...")
    
    try:
        if not transmission_results:
            print("  No transmission results to visualize")
            return
        
        # Prepare data
        viz_data = []
        for i, t_result in enumerate(transmission_results):
            pair_key = f"{t_result.pair[0]}→{t_result.pair[1]}"
            
            # Find corresponding Bell result
            bell_result = None
            for b_result in bell_results:
                if b_result['pair'] == t_result.pair:
                    bell_result = b_result
                    break
            
            viz_data.append({
                'Pair': pair_key,
                'Transmission_Detected': t_result.transmission_detected,
                'Correlation': abs(t_result.correlation_strength),
                'Lag': t_result.transmission_lag if t_result.transmission_lag else 0,
                'Bell_Violation_Rate': bell_result['violation_rate'] if bell_result else 0,
                'Avg_S1': bell_result['avg_s1'] if bell_result else 0
            })
        
        viz_df = pd.DataFrame(viz_data)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Integrated Cross-Sector Transmission & Bell Inequality Analysis', 
                     fontsize=16, fontweight='bold')
        
        # 1. Transmission detection overview
        detected = viz_df['Transmission_Detected'].sum()
        total = len(viz_df)
        
        axes[0, 0].pie([detected, total-detected], 
                      labels=[f'Detected ({detected})', f'Not Detected ({total-detected})'],
                      autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
        axes[0, 0].set_title('Transmission Detection Results')
        
        # 2. Correlation vs Bell Violation Rate
        if viz_df['Bell_Violation_Rate'].sum() > 0:
            scatter = axes[0, 1].scatter(viz_df['Correlation'], viz_df['Bell_Violation_Rate'],
                                       c=viz_df['Transmission_Detected'].astype(int),
                                       cmap='RdYlGn', alpha=0.7, s=80)
            axes[0, 1].set_xlabel('Transmission Correlation')
            axes[0, 1].set_ylabel('Bell Violation Rate (%)')
            axes[0, 1].set_title('Transmission Correlation vs Bell Violations')
            plt.colorbar(scatter, ax=axes[0, 1], label='Transmission Detected')
        else:
            axes[0, 1].text(0.5, 0.5, 'No Bell violation data', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
        
        # 3. Transmission lag distribution
        detected_data = viz_df[viz_df['Transmission_Detected'] == True]
        if not detected_data.empty and detected_data['Lag'].sum() > 0:
            axes[1, 0].hist(detected_data['Lag'], bins=8, color='skyblue', alpha=0.7)
            axes[1, 0].set_xlabel('Transmission Lag (days)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Transmission Lag Distribution')
        else:
            axes[1, 0].text(0.5, 0.5, 'No lag data available', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
        
        # 4. Pair-wise comparison
        if len(viz_df) <= 10:  # Only show if not too many pairs
            x_pos = range(len(viz_df))
            width = 0.35
            
            axes[1, 1].bar([x - width/2 for x in x_pos], viz_df['Correlation'], 
                          width, label='Transmission Correlation', alpha=0.7)
            
            if viz_df['Bell_Violation_Rate'].sum() > 0:
                # Normalize Bell violation rates to 0-1 scale for comparison
                normalized_bell = viz_df['Bell_Violation_Rate'] / 100
                axes[1, 1].bar([x + width/2 for x in x_pos], normalized_bell, 
                              width, label='Bell Violation Rate (normalized)', alpha=0.7)
            
            axes[1, 1].set_xlabel('Asset Pairs')
            axes[1, 1].set_ylabel('Strength')
            axes[1, 1].set_title('Pair-wise Comparison')
            axes[1, 1].set_xticks(x_pos)
            axes[1, 1].set_xticklabels(viz_df['Pair'], rotation=45, ha='right')
            axes[1, 1].legend()
        else:
            axes[1, 1].text(0.5, 0.5, f'Too many pairs ({len(viz_df)}) to display', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"integrated_transmission_analysis_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"  📊 Integrated visualization saved as: {filename}")
        
        plt.show()
        
    except Exception as e:
        print(f"  ❌ Error creating visualization: {str(e)}")


def main():
    """Main function"""
    print("🚀 Starting Integrated Cross-Sector Transmission Analysis...")
    
    # Run integrated analysis
    transmission_results, bell_results = run_integrated_analysis()
    
    print("\n✅ Integrated Analysis Complete!")
    print("\nKey Insights:")
    print("• Cross-sector transmission detection identifies operational dependencies")
    print("• Bell inequality analysis reveals quantum-like correlations")
    print("• Integration shows relationship between transmission and quantum effects")
    print("• Crisis periods likely amplify both transmission and Bell violations")
    
    print("\nNext Steps:")
    print("• Extend to crisis period analysis for amplification detection")
    print("• Add more sector pairs for comprehensive coverage")
    print("• Implement real-time monitoring for early warning systems")
    print("• Integrate with food security risk assessment frameworks")


if __name__ == "__main__":
    main()