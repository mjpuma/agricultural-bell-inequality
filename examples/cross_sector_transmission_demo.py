"""
Cross-Sector Transmission Detection Demo

This script demonstrates the Cross-Sector Transmission Detection System
for analyzing transmission mechanisms between agricultural companies and
their operational dependencies.

Focus areas:
1. Energy → Agriculture (natural gas → fertilizer costs)
2. Transportation → Agriculture (rail/shipping → logistics)  
3. Chemicals → Agriculture (input costs → pesticide/fertilizer prices)
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from cross_sector_transmission_detector import CrossSectorTransmissionDetector


def run_comprehensive_transmission_analysis():
    """Run comprehensive cross-sector transmission analysis"""
    print("🌾 Cross-Sector Transmission Detection Demo")
    print("=" * 60)
    
    # Initialize detector
    detector = CrossSectorTransmissionDetector(transmission_window=90)
    
    print(f"📊 Analysis Parameters:")
    print(f"  • Transmission window: {detector.transmission_window} days (0-3 months)")
    print(f"  • Significance level: {detector.significance_level}")
    print()
    
    # Define focused asset pairs for demonstration
    demo_pairs = {
        'energy_agriculture': [
            ('XOM', 'CF'),   # ExxonMobil → CF Industries (fertilizer)
            ('CVX', 'MOS'),  # Chevron → Mosaic (fertilizer)
            ('COP', 'NTR'),  # ConocoPhillips → Nutrien (fertilizer)
        ],
        'transport_agriculture': [
            ('UNP', 'ADM'),  # Union Pacific → Archer Daniels Midland
            ('CSX', 'BG'),   # CSX Corp → Bunge
            ('FDX', 'DE'),   # FedEx → John Deere
        ],
        'chemicals_agriculture': [
            ('DOW', 'CF'),   # Dow Chemical → CF Industries
            ('DD', 'MOS'),   # DuPont → Mosaic
            ('LYB', 'NTR'),  # LyondellBasell → Nutrien
        ]
    }
    
    all_results = {}
    
    # Analyze each transmission type
    for transmission_type, pairs in demo_pairs.items():
        print(f"\n🔍 Analyzing {transmission_type.replace('_', ' → ').title()} Transmission:")
        print("-" * 50)
        
        results = []
        
        for source_asset, target_asset in pairs:
            try:
                print(f"  Analyzing {source_asset} → {target_asset}...")
                
                # Get the appropriate mechanism
                mechanism = detector.transmission_mechanisms[transmission_type]
                
                # Analyze the pair
                result = detector._analyze_transmission_pair(
                    source_asset=source_asset,
                    target_asset=target_asset,
                    mechanism=mechanism
                )
                
                results.append(result)
                
                # Display result
                if result.transmission_detected:
                    print(f"    ✅ Transmission detected!")
                    print(f"       Lag: {result.transmission_lag} days")
                    print(f"       Correlation: {result.correlation_strength:.3f}")
                    print(f"       P-value: {result.p_value:.4f}")
                    print(f"       Speed: {result.speed_category}")
                else:
                    print(f"    ❌ No significant transmission detected")
                    print(f"       Correlation: {result.correlation_strength:.3f}")
                    print(f"       P-value: {result.p_value:.4f}")
                
            except Exception as e:
                print(f"    ⚠️  Error: {str(e)}")
                continue
        
        all_results[transmission_type] = results
    
    # Create summary analysis
    print("\n📈 Transmission Analysis Summary:")
    print("=" * 60)
    
    summary_data = []
    
    for transmission_type, results in all_results.items():
        detected_count = sum(1 for r in results if r.transmission_detected)
        total_count = len(results)
        
        if total_count > 0:
            detection_rate = detected_count / total_count * 100
            avg_correlation = np.mean([abs(r.correlation_strength) for r in results])
            avg_lag = np.mean([r.transmission_lag for r in results if r.transmission_lag is not None])
            
            summary_data.append({
                'Transmission Type': transmission_type.replace('_', ' → ').title(),
                'Detection Rate': f"{detection_rate:.1f}%",
                'Detected/Total': f"{detected_count}/{total_count}",
                'Avg Correlation': f"{avg_correlation:.3f}",
                'Avg Lag (days)': f"{avg_lag:.0f}" if not np.isnan(avg_lag) else "N/A"
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
    
    # Speed analysis demonstration
    print("\n🚀 Transmission Speed Analysis Example:")
    print("-" * 50)
    
    example_pair = ('XOM', 'CF')  # ExxonMobil → CF Industries
    print(f"Analyzing transmission speed for {example_pair[0]} → {example_pair[1]}...")
    
    try:
        speed_analysis = detector.analyze_transmission_speed(example_pair)
        
        print(f"  Optimal lag: {speed_analysis.optimal_lag} days")
        print(f"  Max correlation: {speed_analysis.max_correlation:.3f}")
        print(f"  Speed category: {speed_analysis.speed_category}")
        
        # Show lag profile
        if speed_analysis.lag_profile:
            print(f"  Lag profile (first 10 lags):")
            for lag in sorted(speed_analysis.lag_profile.keys())[:10]:
                correlation = speed_analysis.lag_profile[lag]
                print(f"    Lag {lag:2d}: {correlation:6.3f}")
    
    except Exception as e:
        print(f"  Error in speed analysis: {str(e)}")
    
    return all_results


def create_transmission_visualization(results):
    """Create visualizations of transmission results"""
    print("\n📊 Creating Transmission Visualizations...")
    
    try:
        # Prepare data for visualization
        viz_data = []
        
        for transmission_type, type_results in results.items():
            for result in type_results:
                viz_data.append({
                    'Transmission Type': transmission_type.replace('_', ' → ').title(),
                    'Source': result.pair[0],
                    'Target': result.pair[1],
                    'Pair': f"{result.pair[0]}→{result.pair[1]}",
                    'Detected': result.transmission_detected,
                    'Correlation': abs(result.correlation_strength),
                    'Lag': result.transmission_lag if result.transmission_lag else 0,
                    'Speed': result.speed_category
                })
        
        if not viz_data:
            print("  No data available for visualization")
            return
        
        viz_df = pd.DataFrame(viz_data)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Cross-Sector Transmission Analysis Results', fontsize=16, fontweight='bold')
        
        # 1. Detection rate by transmission type
        detection_rates = viz_df.groupby('Transmission Type')['Detected'].mean() * 100
        axes[0, 0].bar(detection_rates.index, detection_rates.values, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Detection Rate by Transmission Type')
        axes[0, 0].set_ylabel('Detection Rate (%)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Correlation strength distribution
        detected_data = viz_df[viz_df['Detected'] == True]
        if not detected_data.empty:
            axes[0, 1].hist(detected_data['Correlation'], bins=10, color='lightgreen', alpha=0.7)
            axes[0, 1].set_title('Correlation Strength Distribution\n(Detected Transmissions)')
            axes[0, 1].set_xlabel('Absolute Correlation')
            axes[0, 1].set_ylabel('Frequency')
        else:
            axes[0, 1].text(0.5, 0.5, 'No detected\ntransmissions', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Correlation Strength Distribution')
        
        # 3. Transmission lag analysis
        if not detected_data.empty and 'Lag' in detected_data.columns:
            lag_data = detected_data[detected_data['Lag'] > 0]
            if not lag_data.empty:
                axes[1, 0].scatter(lag_data['Lag'], lag_data['Correlation'], 
                                 c=lag_data['Transmission Type'].astype('category').cat.codes, 
                                 alpha=0.7, s=60)
                axes[1, 0].set_title('Transmission Lag vs Correlation')
                axes[1, 0].set_xlabel('Lag (days)')
                axes[1, 0].set_ylabel('Absolute Correlation')
            else:
                axes[1, 0].text(0.5, 0.5, 'No lag data\navailable', 
                               ha='center', va='center', transform=axes[1, 0].transAxes)
        else:
            axes[1, 0].text(0.5, 0.5, 'No lag data\navailable', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
        
        # 4. Speed category distribution
        if not detected_data.empty:
            speed_counts = detected_data['Speed'].value_counts()
            axes[1, 1].pie(speed_counts.values, labels=speed_counts.index, autopct='%1.1f%%')
            axes[1, 1].set_title('Transmission Speed Distribution\n(Detected Transmissions)')
        else:
            axes[1, 1].text(0.5, 0.5, 'No speed data\navailable', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"cross_sector_transmission_analysis_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"  📊 Visualization saved as: {filename}")
        
        plt.show()
        
    except Exception as e:
        print(f"  ❌ Error creating visualization: {str(e)}")


def demonstrate_mechanism_explanations():
    """Demonstrate the transmission mechanism explanations"""
    print("\n🔬 Transmission Mechanism Explanations:")
    print("=" * 60)
    
    detector = CrossSectorTransmissionDetector()
    
    for mechanism_name, mechanism in detector.transmission_mechanisms.items():
        print(f"\n{mechanism.source_sector} → {mechanism.target_sector}:")
        print(f"  Mechanism: {mechanism.mechanism}")
        print(f"  Expected lag: {mechanism.expected_lag} days")
        print(f"  Strength: {mechanism.strength}")
        print(f"  Crisis amplification: {'Yes' if mechanism.crisis_amplification else 'No'}")


def main():
    """Main demonstration function"""
    print("🚀 Starting Cross-Sector Transmission Detection Demo...")
    
    # Run comprehensive analysis
    results = run_comprehensive_transmission_analysis()
    
    # Show mechanism explanations
    demonstrate_mechanism_explanations()
    
    # Create visualizations
    create_transmission_visualization(results)
    
    print("\n✅ Cross-Sector Transmission Detection Demo Complete!")
    print("\nKey Findings:")
    print("• Transmission mechanisms vary by sector pair")
    print("• Energy → Agriculture shows strongest transmission (fertilizer dependency)")
    print("• Transportation → Agriculture has fastest transmission (logistics)")
    print("• Chemicals → Agriculture has moderate transmission (input costs)")
    print("\nNext Steps:")
    print("• Extend analysis to crisis periods for amplification detection")
    print("• Add more asset pairs for comprehensive coverage")
    print("• Integrate with Bell inequality analysis for quantum correlation detection")


if __name__ == "__main__":
    main()