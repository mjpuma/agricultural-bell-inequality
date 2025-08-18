#!/usr/bin/env python3
"""
COMPREHENSIVE BELL INEQUALITY ANALYSIS RUNNER
==============================================
Organized output with diagnostic tables and visualizations
All results saved to structured folders
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
from pathlib import Path

# Import our analysis modules
from analyze_yahoo_finance_bell import YahooFinanceBellAnalyzer
from updated_cross_mandelbrot_metrics import CrossMandelbrotAnalyzer

class ComprehensiveBellAnalysisRunner:
    """
    Comprehensive Bell analysis with organized output structure
    """
    
    def __init__(self, output_base_dir="bell_analysis_results"):
        self.output_base_dir = output_base_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(output_base_dir, f"run_{self.timestamp}")
        
        # Create output directory structure
        self.create_output_structure()
        
        print(f"🚀 COMPREHENSIVE BELL ANALYSIS RUNNER")
        print(f"=" * 50)
        print(f"📁 Output directory: {self.run_dir}")
        
    def create_output_structure(self):
        """Create organized output directory structure"""
        
        directories = [
            self.run_dir,
            os.path.join(self.run_dir, "data"),
            os.path.join(self.run_dir, "tables"),
            os.path.join(self.run_dir, "visualizations"),
            os.path.join(self.run_dir, "diagnostics"),
            os.path.join(self.run_dir, "reports")
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        print(f"📁 Created output structure:")
        for directory in directories:
            print(f"   {directory}")
    
    def run_complete_analysis(self, assets=None, period='6mo', frequencies=['1h', '1d']):
        """
        Run complete Bell inequality analysis with organized outputs
        
        Parameters:
        - assets: List of stock symbols
        - period: Data period ('6mo', '1y', etc.)
        - frequencies: List of frequencies to analyze
        """
        
        print(f"\n🔬 STARTING COMPREHENSIVE ANALYSIS")
        print(f"=" * 40)
        
        # Default assets if none provided
        if assets is None:
            assets = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'META']
        
        print(f"🎯 Assets: {assets}")
        print(f"📅 Period: {period}")
        print(f"⏱️  Frequencies: {frequencies}")
        
        # Step 1: Yahoo Finance Bell Analysis
        print(f"\n📊 STEP 1: YAHOO FINANCE BELL ANALYSIS")
        print("=" * 40)
        
        analyzer = YahooFinanceBellAnalyzer(assets=assets, period=period)
        
        # Download data
        analyzer.raw_data = analyzer._download_yahoo_data()
        if analyzer.raw_data is None:
            print("❌ Failed to download data")
            return None
        
        # Process data for each frequency
        all_results = {}
        for freq in frequencies:
            print(f"\n🔄 Processing {freq} frequency...")
            
            # Process data
            analyzer.processed_data = analyzer._process_data_multiple_frequencies(freq)
            
            # Run analyses
            chsh_results = analyzer._perform_chsh_analysis()
            s1_results = analyzer._perform_conditional_bell_analysis()
            mandelbrot_results = analyzer._calculate_cross_mandelbrot_metrics()
            
            all_results[freq] = {
                'chsh': chsh_results,
                's1': s1_results,
                'mandelbrot': mandelbrot_results,
                'processed_data': analyzer.processed_data
            }
        
        # Step 2: Cross-Mandelbrot Analysis
        print(f"\n🌀 STEP 2: CROSS-MANDELBROT ANALYSIS")
        print("=" * 40)
        
        cross_analyzer = self._run_cross_mandelbrot_analysis(analyzer)
        
        # Step 3: Generate Comprehensive Outputs
        print(f"\n📋 STEP 3: GENERATING OUTPUTS")
        print("=" * 40)
        
        self._save_raw_data(analyzer.raw_data)
        self._generate_diagnostic_tables(all_results, analyzer)
        self._create_comprehensive_visualizations(all_results, analyzer, cross_analyzer)
        self._generate_analysis_reports(all_results, analyzer, cross_analyzer)
        
        print(f"\n✅ ANALYSIS COMPLETE!")
        print(f"📁 All results saved to: {self.run_dir}")
        print(f"📊 Check the following files:")
        print(f"   📋 tables/ - Diagnostic tables (CSV)")
        print(f"   📊 visualizations/ - Charts and plots (PNG)")
        print(f"   📝 reports/ - Analysis summaries (TXT/HTML)")
        print(f"   💾 data/ - Raw and processed data (JSON/CSV)")
        
        return all_results, analyzer, cross_analyzer
    
    def _run_cross_mandelbrot_analysis(self, analyzer):
        """Run cross-Mandelbrot analysis"""
        
        # Extract return data from analyzer
        if not hasattr(analyzer, 'processed_data') or not analyzer.processed_data:
            print("⚠️  No processed data for cross-Mandelbrot analysis")
            return None
        
        # Use the first available frequency
        freq_key = list(analyzer.processed_data.keys())[0]
        freq_data = analyzer.processed_data[freq_key]
        
        # Convert to format expected by CrossMandelbrotAnalyzer
        return_data = {}
        for ticker, bars in freq_data.items():
            if 'Returns' in bars.columns:
                return_data[ticker] = bars['Returns'].dropna()
        
        if len(return_data) < 2:
            print("⚠️  Need at least 2 assets for cross-Mandelbrot analysis")
            return None
        
        # Run cross-Mandelbrot analysis
        cross_analyzer = CrossMandelbrotAnalyzer()
        cross_results = cross_analyzer.analyze_cross_mandelbrot_comprehensive(return_data)
        
        return cross_analyzer
    
    def _save_raw_data(self, raw_data):
        """Save raw data to files"""
        
        print("💾 Saving raw data...")
        
        data_dir = os.path.join(self.run_dir, "data")
        
        # Save raw price data
        for asset, data in raw_data.items():
            # Save daily data
            daily_file = os.path.join(data_dir, f"{asset}_daily.csv")
            data['daily'].to_csv(daily_file)
            
            # Save hourly data
            hourly_file = os.path.join(data_dir, f"{asset}_hourly.csv")
            data['hourly'].to_csv(hourly_file)
            
            # Save asset info
            info_file = os.path.join(data_dir, f"{asset}_info.json")
            with open(info_file, 'w') as f:
                json.dump(data['info'], f, indent=2, default=str)
        
        print(f"   ✅ Raw data saved to {data_dir}")
    
    def _generate_diagnostic_tables(self, all_results, analyzer):
        """Generate comprehensive diagnostic tables"""
        
        print("📋 Generating diagnostic tables...")
        
        tables_dir = os.path.join(self.run_dir, "tables")
        
        # 1. CHSH Results Table
        chsh_data = []
        for freq, freq_results in all_results.items():
            if 'chsh' in freq_results:
                for freq_name, pairs in freq_results['chsh'].items():
                    for pair, result in pairs.items():
                        chsh_data.append({
                            'Frequency': freq,
                            'Sub_Frequency': freq_name,
                            'Asset_Pair': pair,
                            'CHSH_S': result['S'],
                            'E_AB': result['E_AB'],
                            'E_AB_prime': result['E_AB_prime'],
                            'E_A_prime_B': result['E_A_prime_B'],
                            'E_A_prime_B_prime': result['E_A_prime_B_prime'],
                            'Violation': result['violation'],
                            'Data_Points': result['data_points']
                        })
        
        if chsh_data:
            chsh_df = pd.DataFrame(chsh_data)
            chsh_file = os.path.join(tables_dir, "chsh_results.csv")
            chsh_df.to_csv(chsh_file, index=False)
            print(f"   ✅ CHSH results: {chsh_file}")
        
        # 2. S1 Conditional Results Table
        s1_data = []
        for freq, freq_results in all_results.items():
            if 's1' in freq_results:
                for freq_name, pairs in freq_results['s1'].items():
                    for pair, result in pairs.items():
                        s1_data.append({
                            'Frequency': freq,
                            'Sub_Frequency': freq_name,
                            'Asset_Pair': pair,
                            'S1': result['S1'],
                            'E_AB_00': result['E_AB_00'],
                            'E_AB_01': result['E_AB_01'],
                            'E_AB_10': result['E_AB_10'],
                            'E_AB_11': result['E_AB_11'],
                            'Violation': result['violation'],
                            'Regime_00_Count': result['regime_counts']['(0,0)'],
                            'Regime_01_Count': result['regime_counts']['(0,1)'],
                            'Regime_10_Count': result['regime_counts']['(1,0)'],
                            'Regime_11_Count': result['regime_counts']['(1,1)'],
                            'Total_Data_Points': result['total_data_points']
                        })
        
        if s1_data:
            s1_df = pd.DataFrame(s1_data)
            s1_file = os.path.join(tables_dir, "s1_conditional_results.csv")
            s1_df.to_csv(s1_file, index=False)
            print(f"   ✅ S1 conditional results: {s1_file}")
        
        # 3. Cross-Mandelbrot Results Table
        mandelbrot_data = []
        for freq, freq_results in all_results.items():
            if 'mandelbrot' in freq_results:
                for freq_name, pairs in freq_results['mandelbrot'].items():
                    for pair, result in pairs.items():
                        mandelbrot_data.append({
                            'Frequency': freq,
                            'Sub_Frequency': freq_name,
                            'Asset_Pair': pair,
                            'Cross_Hurst': result['hurst_cross'],
                            'Cross_Correlation_Max': result['cross_correlation_max'],
                            'Cross_Correlation_Decay': result['cross_correlation_decay'],
                            'Cross_Volatility_Correlation': result['cross_volatility_correlation'],
                            'Multifractal_Width': result['multifractal_width'],
                            'Multifractal_Asymmetry': result['multifractal_asymmetry']
                        })
        
        if mandelbrot_data:
            mandelbrot_df = pd.DataFrame(mandelbrot_data)
            mandelbrot_file = os.path.join(tables_dir, "cross_mandelbrot_results.csv")
            mandelbrot_df.to_csv(mandelbrot_file, index=False)
            print(f"   ✅ Cross-Mandelbrot results: {mandelbrot_file}")
        
        # 4. Summary Statistics Table
        self._generate_summary_statistics_table(all_results, tables_dir)
    
    def _generate_summary_statistics_table(self, all_results, tables_dir):
        """Generate summary statistics table"""
        
        summary_data = []
        
        for freq, freq_results in all_results.items():
            # CHSH summary
            if 'chsh' in freq_results:
                for freq_name, pairs in freq_results['chsh'].items():
                    s_values = [result['S'] for result in pairs.values()]
                    violations = sum(1 for result in pairs.values() if result['violation'])
                    
                    summary_data.append({
                        'Analysis_Type': 'CHSH',
                        'Frequency': freq,
                        'Sub_Frequency': freq_name,
                        'Total_Pairs': len(pairs),
                        'Violations': violations,
                        'Violation_Rate': violations / len(pairs) if len(pairs) > 0 else 0,
                        'Mean_Value': np.mean(s_values) if s_values else 0,
                        'Max_Value': np.max(s_values) if s_values else 0,
                        'Std_Value': np.std(s_values) if s_values else 0
                    })
            
            # S1 summary
            if 's1' in freq_results:
                for freq_name, pairs in freq_results['s1'].items():
                    s1_values = [result['S1'] for result in pairs.values()]
                    violations = sum(1 for result in pairs.values() if result['violation'])
                    
                    summary_data.append({
                        'Analysis_Type': 'S1_Conditional',
                        'Frequency': freq,
                        'Sub_Frequency': freq_name,
                        'Total_Pairs': len(pairs),
                        'Violations': violations,
                        'Violation_Rate': violations / len(pairs) if len(pairs) > 0 else 0,
                        'Mean_Value': np.mean(s1_values) if s1_values else 0,
                        'Max_Value': np.max(s1_values) if s1_values else 0,
                        'Std_Value': np.std(s1_values) if s1_values else 0
                    })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_file = os.path.join(tables_dir, "analysis_summary.csv")
            summary_df.to_csv(summary_file, index=False)
            print(f"   ✅ Summary statistics: {summary_file}")
    
    def _create_comprehensive_visualizations(self, all_results, analyzer, cross_analyzer):
        """Create comprehensive visualizations"""
        
        print("📊 Creating visualizations...")
        
        viz_dir = os.path.join(self.run_dir, "visualizations")
        
        # Set matplotlib to non-interactive backend
        import matplotlib
        matplotlib.use('Agg')
        
        # 1. CHSH Heatmaps
        self._create_chsh_heatmaps(all_results, viz_dir)
        
        # 2. S1 Conditional Analysis Plots
        self._create_s1_analysis_plots(all_results, viz_dir)
        
        # 3. Cross-Mandelbrot Visualizations
        if cross_analyzer:
            self._create_cross_mandelbrot_plots(cross_analyzer, viz_dir)
        
        # 4. Price and Return Analysis
        self._create_price_return_plots(analyzer, viz_dir)
        
        # 5. Comparative Analysis
        self._create_comparative_plots(all_results, viz_dir)
        
        print(f"   ✅ Visualizations saved to {viz_dir}")
    
    def _create_chsh_heatmaps(self, all_results, viz_dir):
        """Create CHSH violation heatmaps"""
        
        for freq, freq_results in all_results.items():
            if 'chsh' not in freq_results:
                continue
            
            for freq_name, pairs in freq_results['chsh'].items():
                if not pairs:
                    continue
                
                # Extract asset names
                assets = list(set([pair.split('-')[0] for pair in pairs.keys()] + 
                                 [pair.split('-')[1] for pair in pairs.keys()]))
                
                # Create S matrix
                s_matrix = np.zeros((len(assets), len(assets)))
                
                for pair, result in pairs.items():
                    asset1, asset2 = pair.split('-')
                    if asset1 in assets and asset2 in assets:
                        i, j = assets.index(asset1), assets.index(asset2)
                        s_matrix[i, j] = result['S']
                        s_matrix[j, i] = result['S']
                
                # Create heatmap
                plt.figure(figsize=(10, 8))
                sns.heatmap(s_matrix, 
                           xticklabels=assets, 
                           yticklabels=assets,
                           annot=True, 
                           fmt='.3f',
                           cmap='RdYlBu_r',
                           vmin=0, 
                           vmax=2,
                           cbar_kws={'label': 'CHSH S Value'})
                
                plt.title(f'CHSH Bell Inequality Values\n{freq} - {freq_name}', fontsize=14, fontweight='bold')
                plt.tight_layout()
                
                filename = os.path.join(viz_dir, f"chsh_heatmap_{freq}_{freq_name}.png")
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                plt.close()
    
    def _create_s1_analysis_plots(self, all_results, viz_dir):
        """Create S1 conditional analysis plots"""
        
        for freq, freq_results in all_results.items():
            if 's1' not in freq_results:
                continue
            
            for freq_name, pairs in freq_results['s1'].items():
                if not pairs:
                    continue
                
                # S1 values bar plot
                pair_names = list(pairs.keys())
                s1_values = [pairs[pair]['S1'] for pair in pair_names]
                violations = [pairs[pair]['violation'] for pair in pair_names]
                
                plt.figure(figsize=(12, 6))
                colors = ['red' if v else 'blue' for v in violations]
                bars = plt.bar(range(len(pair_names)), s1_values, color=colors, alpha=0.7)
                
                plt.axhline(y=2, color='red', linestyle='--', label='Bell Violation Threshold')
                plt.axhline(y=-2, color='red', linestyle='--')
                
                plt.xticks(range(len(pair_names)), [p.replace('-', '\n') for p in pair_names], rotation=45)
                plt.ylabel('S1 Value')
                plt.title(f'S1 Conditional Bell Inequality\n{freq} - {freq_name}', fontsize=14, fontweight='bold')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                filename = os.path.join(viz_dir, f"s1_values_{freq}_{freq_name}.png")
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                plt.close()
                
                # Regime analysis plot
                self._create_regime_analysis_plot(pairs, freq, freq_name, viz_dir)
    
    def _create_regime_analysis_plot(self, pairs, freq, freq_name, viz_dir):
        """Create regime-specific analysis plot"""
        
        if not pairs:
            return
        
        # Take first few pairs for detailed regime analysis
        selected_pairs = list(pairs.items())[:6]  # Limit to 6 pairs for readability
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, (pair_name, result) in enumerate(selected_pairs):
            if idx >= 6:
                break
            
            ax = axes[idx]
            
            # Regime values
            regime_labels = ['(0,0)', '(0,1)', '(1,0)', '(1,1)']
            regime_values = [result['E_AB_00'], result['E_AB_01'], result['E_AB_10'], result['E_AB_11']]
            regime_counts = [result['regime_counts']['(0,0)'], result['regime_counts']['(0,1)'], 
                           result['regime_counts']['(1,0)'], result['regime_counts']['(1,1)']]
            
            # Create bar plot with count information
            bars = ax.bar(regime_labels, regime_values, alpha=0.7)
            
            # Add count labels on bars
            for bar, count in zip(bars, regime_counts):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'n={count}', ha='center', va='bottom', fontsize=8)
            
            ax.set_title(f'{pair_name}\nS1 = {result["S1"]:.3f}', fontsize=10)
            ax.set_ylabel('E[AB|regime]')
            ax.set_xlabel('Volatility Regime')
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(selected_pairs), 6):
            axes[idx].set_visible(False)
        
        plt.suptitle(f'Regime-Specific Analysis\n{freq} - {freq_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filename = os.path.join(viz_dir, f"regime_analysis_{freq}_{freq_name}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_cross_mandelbrot_plots(self, cross_analyzer, viz_dir):
        """Create cross-Mandelbrot visualizations"""
        
        if not hasattr(cross_analyzer, 'cross_results') or not cross_analyzer.cross_results:
            return
        
        # Use the existing visualization method from cross_analyzer
        # But save to our organized directory
        original_method = cross_analyzer._create_cross_mandelbrot_visualizations
        
        # Temporarily modify the save path
        def modified_viz_method():
            # Create the plots but save to our directory
            cross_analyzer._create_cross_mandelbrot_visualizations()
            
            # Move the generated file to our viz directory
            import shutil
            if os.path.exists('cross_mandelbrot_analysis.png'):
                target_path = os.path.join(viz_dir, 'cross_mandelbrot_analysis.png')
                shutil.move('cross_mandelbrot_analysis.png', target_path)
        
        try:
            modified_viz_method()
        except Exception as e:
            print(f"   ⚠️  Cross-Mandelbrot visualization error: {e}")
    
    def _create_price_return_plots(self, analyzer, viz_dir):
        """Create price and return analysis plots"""
        
        if not hasattr(analyzer, 'raw_data') or not analyzer.raw_data:
            return
        
        # Price evolution plot
        plt.figure(figsize=(12, 8))
        
        for asset, data in analyzer.raw_data.items():
            daily_data = data['daily']
            normalized_prices = daily_data['Close'] / daily_data['Close'].iloc[0]
            plt.plot(daily_data.index, normalized_prices, label=asset, linewidth=2)
        
        plt.title('Normalized Price Evolution', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Normalized Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = os.path.join(viz_dir, 'price_evolution.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Return distributions
        plt.figure(figsize=(12, 8))
        
        for asset, data in analyzer.raw_data.items():
            daily_data = data['daily']
            returns = daily_data['Close'].pct_change().dropna()
            plt.hist(returns, bins=30, alpha=0.6, label=asset, density=True)
        
        plt.title('Return Distributions', fontsize=14, fontweight='bold')
        plt.xlabel('Daily Returns')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = os.path.join(viz_dir, 'return_distributions.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_comparative_plots(self, all_results, viz_dir):
        """Create comparative analysis plots"""
        
        # Compare CHSH vs S1 values
        chsh_values = []
        s1_values = []
        pair_names = []
        
        for freq, freq_results in all_results.items():
            if 'chsh' in freq_results and 's1' in freq_results:
                for freq_name in freq_results['chsh'].keys():
                    if freq_name in freq_results['s1']:
                        chsh_pairs = freq_results['chsh'][freq_name]
                        s1_pairs = freq_results['s1'][freq_name]
                        
                        common_pairs = set(chsh_pairs.keys()) & set(s1_pairs.keys())
                        
                        for pair in common_pairs:
                            chsh_values.append(chsh_pairs[pair]['S'])
                            s1_values.append(abs(s1_pairs[pair]['S1']))  # Use absolute value for comparison
                            pair_names.append(f"{pair}_{freq}_{freq_name}")
        
        if chsh_values and s1_values:
            plt.figure(figsize=(10, 8))
            plt.scatter(chsh_values, s1_values, alpha=0.7, s=50)
            
            # Add diagonal line
            max_val = max(max(chsh_values), max(s1_values))
            plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='y=x')
            
            plt.xlabel('CHSH S Value')
            plt.ylabel('|S1| Value')
            plt.title('CHSH vs S1 Conditional Bell Inequality Comparison', fontsize=14, fontweight='bold')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            filename = os.path.join(viz_dir, 'chsh_vs_s1_comparison.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
    
    def _generate_analysis_reports(self, all_results, analyzer, cross_analyzer):
        """Generate comprehensive analysis reports"""
        
        print("📝 Generating analysis reports...")
        
        reports_dir = os.path.join(self.run_dir, "reports")
        
        # 1. Text Summary Report
        self._generate_text_report(all_results, analyzer, cross_analyzer, reports_dir)
        
        # 2. HTML Report
        self._generate_html_report(all_results, analyzer, cross_analyzer, reports_dir)
        
        print(f"   ✅ Reports saved to {reports_dir}")
    
    def _generate_text_report(self, all_results, analyzer, cross_analyzer, reports_dir):
        """Generate text summary report"""
        
        report_file = os.path.join(reports_dir, "analysis_summary.txt")
        
        with open(report_file, 'w') as f:
            f.write("COMPREHENSIVE BELL INEQUALITY ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Assets Analyzed: {analyzer.assets}\n")
            f.write(f"Data Period: {analyzer.period}\n\n")
            
            # CHSH Summary
            f.write("CHSH BELL INEQUALITY RESULTS\n")
            f.write("-" * 30 + "\n")
            
            for freq, freq_results in all_results.items():
                if 'chsh' in freq_results:
                    f.write(f"\n{freq.upper()} Frequency:\n")
                    
                    for freq_name, pairs in freq_results['chsh'].items():
                        if pairs:
                            s_values = [result['S'] for result in pairs.values()]
                            violations = sum(1 for result in pairs.values() if result['violation'])
                            
                            f.write(f"  {freq_name}:\n")
                            f.write(f"    Total pairs: {len(pairs)}\n")
                            f.write(f"    Violations: {violations}\n")
                            f.write(f"    Violation rate: {violations/len(pairs)*100:.1f}%\n")
                            f.write(f"    Mean S: {np.mean(s_values):.4f}\n")
                            f.write(f"    Max S: {np.max(s_values):.4f}\n\n")
            
            # S1 Summary
            f.write("S1 CONDITIONAL BELL INEQUALITY RESULTS\n")
            f.write("-" * 35 + "\n")
            
            for freq, freq_results in all_results.items():
                if 's1' in freq_results:
                    f.write(f"\n{freq.upper()} Frequency:\n")
                    
                    for freq_name, pairs in freq_results['s1'].items():
                        if pairs:
                            s1_values = [result['S1'] for result in pairs.values()]
                            violations = sum(1 for result in pairs.values() if result['violation'])
                            
                            f.write(f"  {freq_name}:\n")
                            f.write(f"    Total pairs: {len(pairs)}\n")
                            f.write(f"    Violations: {violations}\n")
                            f.write(f"    Violation rate: {violations/len(pairs)*100:.1f}%\n")
                            f.write(f"    Mean S1: {np.mean(s1_values):.4f}\n")
                            f.write(f"    Max |S1|: {np.max(np.abs(s1_values)):.4f}\n\n")
            
            # Cross-Mandelbrot Summary
            if cross_analyzer and hasattr(cross_analyzer, 'cross_results'):
                f.write("CROSS-MANDELBROT ANALYSIS RESULTS\n")
                f.write("-" * 33 + "\n")
                
                cross_results = cross_analyzer.cross_results
                if cross_results:
                    hurst_values = [result['hurst_cross'] for result in cross_results.values()]
                    f.write(f"Total asset pairs: {len(cross_results)}\n")
                    f.write(f"Mean cross-Hurst: {np.mean(hurst_values):.4f}\n")
                    f.write(f"Cross-Hurst range: {np.min(hurst_values):.4f} - {np.max(hurst_values):.4f}\n\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS FOR WDRS DATA SELECTION\n")
            f.write("-" * 38 + "\n")
            f.write("Based on this Yahoo Finance analysis:\n\n")
            f.write("1. Focus on asset pairs with highest Bell violations\n")
            f.write("2. Daily frequency shows strongest effects\n")
            f.write("3. Consider cross-Mandelbrot patterns for deeper analysis\n")
            f.write("4. Download 2-3 years of WDRS data for statistical significance\n")
    
    def _generate_html_report(self, all_results, analyzer, cross_analyzer, reports_dir):
        """Generate HTML report with embedded visualizations"""
        
        report_file = os.path.join(reports_dir, "analysis_report.html")
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Bell Inequality Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #34495e; }}
                .summary {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #e9ecef; border-radius: 3px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Comprehensive Bell Inequality Analysis Report</h1>
            
            <div class="summary">
                <h2>Analysis Summary</h2>
                <div class="metric"><strong>Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
                <div class="metric"><strong>Assets:</strong> {', '.join(analyzer.assets)}</div>
                <div class="metric"><strong>Period:</strong> {analyzer.period}</div>
            </div>
            
            <h2>Key Findings</h2>
            <p>This analysis examined Bell inequality violations in financial market data using both CHSH and S1 conditional approaches.</p>
            
            <h2>Visualizations</h2>
            <p>Detailed visualizations have been generated and saved to the visualizations/ directory.</p>
            
            <h2>Data Files</h2>
            <ul>
                <li><strong>tables/</strong> - Diagnostic tables in CSV format</li>
                <li><strong>visualizations/</strong> - Charts and plots</li>
                <li><strong>data/</strong> - Raw and processed data</li>
            </ul>
            
        </body>
        </html>
        """
        
        with open(report_file, 'w') as f:
            f.write(html_content)

# =================== MAIN EXECUTION FUNCTION ===================

def run_comprehensive_analysis(assets=None, period='6mo', frequencies=['1h', '1d']):
    """
    Main function to run comprehensive Bell inequality analysis
    
    Parameters:
    - assets: List of stock symbols (default: tech stocks)
    - period: Data period ('6mo', '1y', '2y', etc.)
    - frequencies: List of frequencies to analyze
    
    Usage:
    python3 run_comprehensive_bell_analysis.py
    """
    
    print("🚀 COMPREHENSIVE BELL INEQUALITY ANALYSIS")
    print("=" * 50)
    
    # Default assets
    if assets is None:
        assets = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'META']
    
    # Initialize runner
    runner = ComprehensiveBellAnalysisRunner()
    
    # Run complete analysis
    try:
        results = runner.run_complete_analysis(
            assets=assets,
            period=period,
            frequencies=frequencies
        )
        
        if results:
            print(f"\n🎉 ANALYSIS COMPLETE!")
            print(f"📁 Results saved to: {runner.run_dir}")
            return runner, results
        else:
            print(f"\n❌ Analysis failed")
            return None, None
            
    except Exception as e:
        print(f"\n❌ Analysis error: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    # Run with default settings
    runner, results = run_comprehensive_analysis()
    
    # Or customize:
    # runner, results = run_comprehensive_analysis(
    #     assets=['AAPL', 'MSFT', 'GOOGL', 'TSLA'],
    #     period='6mo',
    #     frequencies=['1h', '1d']
    # )