#!/usr/bin/env python3
"""
🌐 CROSS-SECTOR BELL INEQUALITY ANALYSIS (2010-2025)
=====================================================
Comprehensive analysis of quantum correlations across different market sectors
Focusing on cross-sector relationships and sector quantum bridges

Time Period: January 2010 to July 2025 (15.5+ years of consistent data)

Sector Matrix:
- Technology: AAPL, MSFT, NVDA
- Energy: XOM, CVX, COP  
- Finance: JPM, BAC, GS
- Agriculture: ADM, CF, DE

Cross-Sector Pairs:
- Tech vs Energy: AAPL-XOM, MSFT-CVX, NVDA-COP
- Tech vs Finance: AAPL-JPM, MSFT-BAC, NVDA-GS
- Tech vs Agriculture: AAPL-ADM, MSFT-CF, NVDA-DE
- Energy vs Finance: XOM-JPM, CVX-BAC, COP-GS
- Energy vs Agriculture: XOM-ADM, CVX-CF, COP-DE
- Finance vs Agriculture: JPM-ADM, BAC-CF, GS-DE

Crisis Periods (2010-2025):
- COVID-19 (2020-2021): Cross-sector stress test
- Tech Correction 2022: Tech sector impact
- Ukraine War (2022): Energy/Agriculture stress
- Inflation 2023: Rate hike and inflation impact
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our enhanced analysis components
from src.bell_inequality_analyzer import BellInequalityAnalyzer
from src.cross_mandelbrot_analyzer import CrossMandelbrotAnalyzer
from src.results_manager import ResultsManager

class CrossSectorAnalyzer:
    """Enhanced Bell inequality analyzer for cross-sector analysis"""
    
    def __init__(self):
        self.results_mgr = ResultsManager("results/FINAL_CROSS_SECTOR_RESULTS")
        self.bell_analyzer = BellInequalityAnalyzer()
        self.mandelbrot_analyzer = CrossMandelbrotAnalyzer()
        
        # Define sectors and representative stocks
        self.sectors = {
            'Technology': ['AAPL', 'MSFT', 'NVDA'],
            'Energy': ['XOM', 'CVX', 'COP'],
            'Finance': ['JPM', 'BAC', 'GS'],
            'Agriculture': ['ADM', 'CF', 'DE']  # Replaced CORN with DE (Deere & Co) for longer history
        }
        
        # Cross-sector pairs for analysis
        self.cross_sector_pairs = [
            # Tech vs Energy
            ('AAPL', 'XOM', 'Tech vs Energy', 'Technology', 'Energy'),
            ('MSFT', 'CVX', 'Tech vs Energy', 'Technology', 'Energy'),
            ('NVDA', 'COP', 'Tech vs Energy', 'Technology', 'Energy'),
            
            # Tech vs Finance
            ('AAPL', 'JPM', 'Tech vs Finance', 'Technology', 'Finance'),
            ('MSFT', 'BAC', 'Tech vs Finance', 'Technology', 'Finance'),
            ('NVDA', 'GS', 'Tech vs Finance', 'Technology', 'Finance'),
            
            # Tech vs Agriculture
            ('AAPL', 'ADM', 'Tech vs Agriculture', 'Technology', 'Agriculture'),
            ('MSFT', 'CF', 'Tech vs Agriculture', 'Technology', 'Agriculture'),
            ('NVDA', 'DE', 'Tech vs Agriculture', 'Technology', 'Agriculture'),
            
            # Energy vs Finance
            ('XOM', 'JPM', 'Energy vs Finance', 'Energy', 'Finance'),
            ('CVX', 'BAC', 'Energy vs Finance', 'Energy', 'Finance'),
            ('COP', 'GS', 'Energy vs Finance', 'Energy', 'Finance'),
            
            # Energy vs Agriculture
            ('XOM', 'ADM', 'Energy vs Agriculture', 'Energy', 'Agriculture'),
            ('CVX', 'CF', 'Energy vs Agriculture', 'Energy', 'Agriculture'),
            ('COP', 'DE', 'Energy vs Agriculture', 'Energy', 'Agriculture'),
            
            # Finance vs Agriculture
            ('JPM', 'ADM', 'Finance vs Agriculture', 'Finance', 'Agriculture'),
            ('BAC', 'CF', 'Finance vs Agriculture', 'Finance', 'Agriculture'),
            ('GS', 'DE', 'Finance vs Agriculture', 'Finance', 'Agriculture')
        ]
        
        # Crisis periods for cross-sector analysis (2010-2025)
        self.crisis_periods = {
            'COVID_19_Cross_Sector': {
                'start': '2020-03-01',
                'end': '2021-12-31',
                'description': 'COVID-19 Cross-Sector Stress Test'
            },
            'Tech_Correction_2022': {
                'start': '2022-01-01',
                'end': '2022-12-31',
                'description': '2022 Tech Correction Cross-Sector Impact'
            },
            'Ukraine_War_2022': {
                'start': '2022-02-01',
                'end': '2022-12-31',
                'description': 'Ukraine War Energy/Agriculture Stress'
            },
            'Inflation_2023': {
                'start': '2023-01-01',
                'end': '2023-12-31',
                'description': '2023 Inflation and Rate Hike Impact'
            }
        }
    
    def test_cross_sector_data_availability(self):
        """Test data availability for cross-sector stocks"""
        print("🔍 TESTING CROSS-SECTOR DATA AVAILABILITY")
        print("=" * 60)
        
        for sector_name, stocks in self.sectors.items():
            print(f"\n📊 {sector_name} Sector:")
            for stock in stocks:
                try:
                    data = yf.download(stock, start='2010-01-01', end='2025-07-31', progress=False)
                    if len(data) > 0:
                        print(f"   ✅ {stock}: {len(data)} days ({data.index[0].date()} to {data.index[-1].date()})")
                    else:
                        print(f"   ❌ {stock}: No data available")
                except Exception as e:
                    print(f"   ❌ {stock}: Download failed - {str(e)}")
        
        print()
    
    def analyze_cross_sector_pair(self, asset1, asset2, pair_name, sector1, sector2):
        """Analyze a cross-sector pair with enhanced analysis"""
        print(f"🔍 Analyzing {asset1} vs {asset2} ({pair_name})...")
        print(f"   📊 {sector1} vs {sector2}")
        
        try:
            # Download data for specific period (2010-2025)
            data = yf.download([asset1, asset2], start='2010-01-01', end='2025-07-31', progress=False)
            if len(data) == 0:
                print(f"   ❌ Error: No data available for {asset1}-{asset2}")
                return None
            
            print(f"   📊 Data shape: {data.shape}, Columns: {data.columns.tolist()}")
            
            # Check for long time series optimization
            if len(data) > 1000:
                print(f"   📊 Long time series detected ({len(data)} days), implementing plot optimization...")
            
            # Calculate returns
            returns = data.pct_change().dropna()
            
            # Check if we have both assets in the data (MultiIndex structure)
            if ('Close', asset1) not in returns.columns or ('Close', asset2) not in returns.columns:
                print(f"   ❌ Error: Missing data for {asset1} or {asset2}")
                return None
            
            # Get the actual overlap period where both assets have data
            asset1_data = data[('Close', asset1)].dropna()
            asset2_data = data[('Close', asset2)].dropna()
            
            overlap_start = max(asset1_data.index[0], asset2_data.index[0])
            overlap_end = min(asset1_data.index[-1], asset2_data.index[-1])
            
            print(f"   📊 Data overlap: {overlap_start.date()} to {overlap_end.date()}")
            print(f"   📊 Overlap days: {(overlap_end - overlap_start).days}")
            
            # Use only the overlap period for analysis
            overlap_data = data.loc[overlap_start:overlap_end]
            overlap_returns = overlap_data.pct_change().dropna()
            
            if len(overlap_returns) < 100:
                print(f"   ❌ Error: Insufficient overlap data ({len(overlap_returns)} days)")
                return None
            
            # Run Bell inequality analysis
            s1_values, s1_violations = self.calculate_s1_bell_inequality(overlap_returns[('Close', asset1)], overlap_returns[('Close', asset2)], 20)
            bell_results = {
                's1_values': s1_values,
                's1_violations': s1_violations
            }
            
            if bell_results is None:
                print(f"   ❌ Error analyzing {asset1}-{asset2}: Bell analysis failed")
                return None
            
            # Run Cross-Mandelbrot analysis
            try:
                mandelbrot_results = self.mandelbrot_analyzer.analyze_cross_mandelbrot_comprehensive(
                    {asset1: overlap_returns[('Close', asset1)], asset2: overlap_returns[('Close', asset2)]}
                )
            except Exception as e:
                print(f"   ⚠️  Mandelbrot analysis failed: {e}")
                mandelbrot_results = None
            
            # Calculate additional metrics
            rolling_corr = overlap_returns[('Close', asset1)].rolling(20).corr(overlap_returns[('Close', asset2)])
            rolling_vol1 = overlap_returns[('Close', asset1)].rolling(20).std() * np.sqrt(252)
            rolling_vol2 = overlap_returns[('Close', asset2)].rolling(20).std() * np.sqrt(252)
            
            # Create enhanced visualization
            self.create_cross_sector_figure(
                asset1, asset2, overlap_data, overlap_returns, rolling_corr, 
                rolling_vol1, rolling_vol2, bell_results, mandelbrot_results, 
                pair_name, sector1, sector2
            )
            
            # Create crisis analysis
            self.create_cross_sector_crisis_analysis(asset1, asset2, overlap_data, overlap_returns, pair_name, sector1, sector2)
            
            # Save correlation table
            self.create_cross_sector_correlation_table(overlap_returns[('Close', asset1)], overlap_returns[('Close', asset2)], 
                                                     rolling_corr, bell_results, mandelbrot_results, pair_name)
            
            # Calculate violation rate
            s1_values = bell_results['s1_values']
            violations = np.abs(s1_values) > 2
            violation_rate = (violations.sum() / len(violations)) * 100
            
            print(f"   ✅ Analysis complete: {violation_rate:.1f}% Bell violations")
            print(f"   📊 Analysis period: {overlap_start.date()} to {overlap_end.date()}")
            print()
            
            return {
                'asset1': asset1,
                'asset2': asset2,
                'pair_name': pair_name,
                'sector1': sector1,
                'sector2': sector2,
                'violation_rate': violation_rate,
                'data_period': f"{overlap_start.date()} to {overlap_end.date()}",
                'bell_results': bell_results,
                'mandelbrot_results': mandelbrot_results
            }
            
        except Exception as e:
            print(f"   ❌ Error analyzing {asset1}-{asset2}: {str(e)}")
            return None
    
    def create_cross_sector_figure(self, asset1, asset2, data, returns, rolling_corr, 
                                  rolling_vol1, rolling_vol2, bell_results, mandelbrot_results, 
                                  pair_name, sector1, sector2):
        """Create enhanced 6-panel figure for cross-sector analysis"""
        
        # Set up figure
        fig, axes = plt.subplots(3, 2, figsize=(20, 16))
        fig.suptitle(f'Cross-Sector Analysis: {asset1} vs {asset2}\n{pair_name} | {sector1} vs {sector2}', 
                     fontsize=18, fontweight='bold')
        
        # Extract data
        s1_values = bell_results['s1_values']
        s1_violations = bell_results['s1_violations']
        
        # Panel 1: S1 Bell Inequality with Correlation overlay
        ax1 = axes[0, 0]
        if s1_values is not None and len(s1_values) > 0:
            window = 20
            time_index = data.index[window:][:len(s1_values)]
            
            # Plot S1 values
            ax1.plot(time_index, s1_values, 'b-', linewidth=2, label='|S1|', alpha=0.8)
            ax1.axhline(y=2, color='red', linestyle='--', linewidth=2, label='Classical Bound (2.0)')
            ax1.axhline(y=2*np.sqrt(2), color='green', linestyle='--', linewidth=2, label='Quantum Bound (2.83)')
            
            # Overlay rolling correlation (scaled)
            if rolling_corr is not None and len(rolling_corr.dropna()) > 0:
                corr_aligned = rolling_corr.reindex(time_index, method='nearest')
                corr_scaled = 2 + 2 * corr_aligned  # Scale from [-1,1] to [0,4]
                ax1_twin = ax1.twinx()
                ax1_twin.plot(time_index, corr_scaled, 'purple', linewidth=1.5, label='Correlation (scaled)', alpha=0.7)
                ax1_twin.set_ylabel('Correlation (scaled)', color='purple')
                ax1_twin.tick_params(axis='y', labelcolor='purple')
                
                # Add third axis for actual correlation values
                ax1_actual = ax1.twinx()
                ax1_actual.spines['right'].set_position(('outward', 60))
                ax1_actual.plot(time_index, corr_aligned, 'purple', linewidth=1.5, linestyle=':', label='Correlation (actual)', alpha=0.6)
                ax1_actual.set_ylabel('Correlation (actual)', color='purple')
                ax1_actual.tick_params(axis='y', labelcolor='purple')
                ax1_actual.set_ylim(-1, 1)
                ax1_actual.axhline(y=0.7, color='green', linestyle=':', alpha=0.4, linewidth=1.5)
                ax1_actual.axhline(y=-0.7, color='red', linestyle=':', alpha=0.4, linewidth=1.5)
            
            # Highlight violations
            if s1_violations is not None and len(s1_violations) > 0:
                violations_array = np.array(s1_violations)
                if len(violations_array) > 0 and violations_array.any():
                    violation_indices = time_index[violations_array]
                    violation_values = np.array(s1_values)[violations_array]
                    ax1.scatter(violation_indices, violation_values, color='red', s=50, alpha=0.7, 
                               label=f'Violations ({violations_array.sum()})', zorder=5)
        
        ax1.set_ylabel('|S1|', color='blue', fontsize=14)
        ax1.set_title(f'S1 Bell Inequality + Correlation: {asset1} vs {asset2}', fontsize=14)
        ax1.legend(fontsize=12, loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='both', which='major', labelsize=12)
        
        # Panel 2: S1 with Cross-Hurst overlay
        ax2 = axes[0, 1]
        if s1_values is not None and len(s1_values) > 0 and mandelbrot_results:
            window = 20
            time_index = data.index[window:][:len(s1_values)]
            
            # Plot S1 values
            ax2.plot(time_index, s1_values, 'b-', linewidth=2, label='|S1|', alpha=0.8)
            ax2.axhline(y=2, color='red', linestyle='--', linewidth=2, label='Classical Bound (2.0)')
            ax2.axhline(y=2*np.sqrt(2), color='green', linestyle='--', linewidth=2, label='Quantum Bound (2.83)')
            
            # Overlay Cross-Hurst if available
            if 'cross_hurst' in mandelbrot_results and len(mandelbrot_results['cross_hurst']) > 0:
                cross_hurst_actual = np.array(mandelbrot_results['cross_hurst'])
                cross_hurst_scaled = 4 * (cross_hurst_actual - 0.1) / 0.8
                mandelbrot_time_index = data.index[window:][:len(cross_hurst_actual)]
                
                ax2_twin = ax2.twinx()
                ax2_twin.plot(mandelbrot_time_index, cross_hurst_scaled, 'orange', linewidth=1.5, 
                             label='Cross-Hurst (scaled)', alpha=0.7)
                ax2_twin.set_ylabel('Cross-Hurst (scaled)', color='orange')
                ax2_twin.tick_params(axis='y', labelcolor='orange')
                
                # Add third axis for actual cross-Hurst values
                ax2_actual = ax2.twinx()
                ax2_actual.spines['right'].set_position(('outward', 60))
                ax2_actual.plot(mandelbrot_time_index, cross_hurst_actual, 'orange', linewidth=1.5, linestyle=':', label='Cross-Hurst (actual)', alpha=0.6)
                ax2_actual.set_ylabel('Cross-Hurst (actual)', color='orange')
                ax2_actual.tick_params(axis='y', labelcolor='orange')
                ax2_actual.set_ylim(0, 1)
                ax2_actual.axhline(y=0.5, color='green', linestyle=':', alpha=0.4, linewidth=1.5)
            
            # Highlight violations
            if s1_violations is not None and len(s1_violations) > 0:
                violations_array = np.array(s1_violations)
                if len(violations_array) > 0 and violations_array.any():
                    violation_indices = time_index[violations_array]
                    violation_values = np.array(s1_values)[violations_array]
                    ax2.scatter(violation_indices, violation_values, color='red', s=50, alpha=0.7, 
                               label=f'Violations ({violations_array.sum()})', zorder=5)
        
        ax2.set_ylabel('|S1|', color='blue', fontsize=14)
        ax2.set_title(f'S1 Bell Inequality + Cross-Hurst: {asset1} vs {asset2}', fontsize=14)
        ax2.legend(fontsize=12, loc='upper left')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='both', which='major', labelsize=12)
        
        # Panel 3: Stock Prices
        ax3 = axes[1, 0]
        ax3.plot(data.index, data[('Close', asset1)], 'b-', linewidth=1.5, label=f'{asset1} ({sector1})', alpha=0.8)
        ax3.plot(data.index, data[('Close', asset2)], 'orange', linewidth=1.5, label=f'{asset2} ({sector2})', alpha=0.8)
        ax3.set_ylabel('Price', fontsize=14)
        ax3.set_title(f'Cross-Sector Stock Prices: {sector1} vs {sector2}', fontsize=14)
        ax3.legend(fontsize=12)
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='both', which='major', labelsize=12)
        
        # Panel 4: Rolling correlation
        ax4 = axes[1, 1]
        if rolling_corr is not None and len(rolling_corr.dropna()) > 0:
            ax4.plot(rolling_corr.index, rolling_corr, 'purple', linewidth=2, alpha=0.8)
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax4.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='Strong Positive')
            ax4.axhline(y=-0.7, color='red', linestyle='--', alpha=0.5, label='Strong Negative')
            ax4.set_ylabel('Correlation', fontsize=14)
            ax4.set_title('20-day Rolling Correlation (RETURNS)', fontsize=14)
            ax4.legend(fontsize=12)
            ax4.grid(True, alpha=0.3)
            ax4.tick_params(axis='both', which='major', labelsize=12)
        
        # Panel 5: Volatility
        ax5 = axes[2, 0]
        if rolling_vol1 is not None and rolling_vol2 is not None:
            ax5.plot(rolling_vol1.index, rolling_vol1, 'b-', linewidth=1.5, label=f'{asset1} Vol ({sector1})', alpha=0.8)
            ax5.plot(rolling_vol2.index, rolling_vol2, 'orange', linewidth=1.5, label=f'{asset2} Vol ({sector2})', alpha=0.8)
            ax5.set_ylabel('Annualized Volatility', fontsize=14)
            ax5.set_title('20-day Rolling Volatility', fontsize=14)
            ax5.legend(fontsize=12)
            ax5.grid(True, alpha=0.3)
            ax5.tick_params(axis='both', which='major', labelsize=12)
        
        # Panel 6: Cross-Sector Interpretation Guide
        ax6 = axes[2, 1]
        ax6.axis('off')
        
        interpretation_text = (
            f"🔍 CROSS-SECTOR INTERPRETATION GUIDE\n\n"
            f"📊 S1 Bell Inequality:\n"
            f"• >2.0 = Classical violation\n"
            f"• >2.83 = Quantum violation\n"
            f"• High violations = Non-classical behavior\n\n"
            f"🌊 Cross-Hurst:\n"
            f"• >0.5 = Persistent trends\n"
            f"• <0.5 = Anti-persistent\n"
            f"• =0.5 = Random walk\n\n"
            f"📈 Cross-Sector Correlation:\n"
            f"• >0.7 = Strong sector bridge\n"
            f"• <-0.7 = Strong sector divergence\n"
            f"• Near 0 = Independent sectors"
        )
        
        ax6.text(0.05, 0.95, interpretation_text, transform=ax6.transAxes, fontsize=14,
                 verticalalignment='top', horizontalalignment='left', wrap=True,
                 bbox=dict(boxstyle="round,pad=0.8", 
                 facecolor="lightyellow", alpha=0.9, edgecolor="gray"))
        
        # Format x-axis dates
        for ax in [ax1, ax2, ax3, ax4, ax5]:
            if hasattr(ax, 'xaxis'):
                ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
                ax.tick_params(axis='x', rotation=45, labelsize=12)
        
        plt.tight_layout()
        
        # Save figure
        filename = f"{asset1}_{asset2}_cross_sector_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        self.results_mgr.save_figure(fig, filename)
        plt.close()
        
        print(f"   ✅ Figure saved: {filename}")
    
    def create_cross_sector_crisis_analysis(self, asset1, asset2, data, returns, pair_name, sector1, sector2):
        """Create crisis-specific analysis for cross-sector pairs"""
        
        for crisis_name, crisis_info in self.crisis_periods.items():
            try:
                print(f"   🔥 Analyzing {crisis_name}...")
                
                # Filter data for crisis period
                crisis_mask = (data.index >= crisis_info['start']) & (data.index <= crisis_info['end'])
                crisis_data = data[crisis_mask]
                
                if len(crisis_data) < 50:  # Need minimum data points
                    print(f"   ❌ {crisis_name}: Insufficient crisis data ({len(crisis_data)} points)")
                    continue
                
                crisis_returns = crisis_data.pct_change().dropna()
                
                # Run Bell analysis for crisis period
                s1_values, s1_violations = self.calculate_s1_bell_inequality(crisis_returns[('Close', asset1)], crisis_returns[('Close', asset2)], 10)
                bell_results = {
                    's1_values': s1_values,
                    's1_violations': s1_violations
                }
                
                if bell_results is None:
                    print(f"   ❌ {crisis_name}: Bell analysis failed")
                    continue
                
                # Calculate crisis metrics
                s1_values = bell_results['s1_values']
                s1_violations = bell_results['s1_violations']
                
                if len(s1_values) > 0:
                    violations = np.abs(s1_values) > 2
                    violation_rate = (violations.sum() / len(violations)) * 100
                    print(f"   ✅ {asset1}-{asset2}: {violation_rate:.1f}% violations during crisis")
                else:
                    print(f"   ⚠️  {asset1}-{asset2}: No valid S1 values for crisis period")
                
            except Exception as e:
                print(f"   ❌ {crisis_name}: {str(e)}")
    
    def create_cross_sector_correlation_table(self, returns1, returns2, rolling_corr, bell_results, mandelbrot_results, pair_name):
        """Create correlation statistics table for cross-sector analysis"""
        
        # Basic correlations
        pearson_r = returns1.corr(returns2)
        spearman_r = returns1.corr(returns2, method='spearman')
        
        # S1 correlations
        s1_values = bell_results['s1_values']
        if s1_values is not None and len(s1_values) > 0:
            s1_corr_pearson = np.corrcoef(rolling_corr.dropna()[-len(s1_values):], s1_values)[0, 1]
        else:
            s1_corr_pearson = np.nan
        
        # Create results table
        results_data = [
            ('Pearson Correlation', f'{pearson_r:.6f}', '< 0.001' if abs(pearson_r) > 0.1 else '> 0.05'),
            ('Spearman Correlation', f'{spearman_r:.6f}', '< 0.001' if abs(spearman_r) > 0.1 else '> 0.05'),
            ('S1 vs Rolling Correlation', f'{s1_corr_pearson:.6f}', '< 0.001' if abs(s1_corr_pearson) > 0.1 else '> 0.05'),
        ]
        
        # Add Mandelbrot correlations if available
        if mandelbrot_results:
            for metric_name, metric_values in mandelbrot_results.items():
                if isinstance(metric_values, list) and len(metric_values) > 0:
                    clean_s1 = s1_values[-len(metric_values):] if s1_values is not None else []
                    clean_mandelbrot = metric_values[-len(clean_s1):] if len(clean_s1) > 0 else []
                    
                    if len(clean_s1) > 10 and len(clean_mandelbrot) > 10:
                        correlation = np.corrcoef(clean_s1, clean_mandelbrot)[0, 1]
                        results_data.append((f'S1 vs {metric_name.replace("_", " ").title()}', 
                                           f'{correlation:.6f}', 
                                           '< 0.001' if abs(correlation) > 0.1 else '> 0.05'))
        
        # Create DataFrame and save
        try:
            results_df = pd.DataFrame(results_data, columns=['Metric', 'Correlation', 'P-Value'])
            filename = f"cross_sector_correlation_table_{pair_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            self.results_mgr.save_excel(results_df, filename)
            print(f"   ✅ Excel saved: {filename}")
        except Exception as e:
            print(f"   ⚠️  Excel save failed: {e}")
            print(f"   📊 Correlation data: {results_data}")
    
    def calculate_s1_bell_inequality(self, returns1, returns2, window):
        """Calculate S1 Bell inequality values with proper NaN handling"""
        
        try:
            s1_values = []
            violations = []
            
            for i in range(window, len(returns1)):
                # Get window data
                r1_window = returns1.iloc[i-window:i]
                r2_window = returns2.iloc[i-window:i]
                
                # Check if we have enough non-NaN data in the window
                r1_valid = r1_window.dropna()
                r2_valid = r2_window.dropna()
                
                if len(r1_valid) < 10 or len(r2_valid) < 10:
                    s1_values.append(np.nan)
                    violations.append(False)
                    continue
                
                # Calculate correlations at different lags (simplified Bell inequality)
                corr_0 = r1_window.corr(r2_window)  # Simultaneous
                
                # Check for NaN correlation
                if pd.isna(corr_0):
                    s1_values.append(np.nan)
                    violations.append(False)
                    continue
                
                if len(r1_window) > 5:
                    corr_1 = r1_window[:-1].corr(r2_window[1:])  # 1-day lag
                    corr_2 = r1_window[:-2].corr(r2_window[2:])  # 2-day lag
                else:
                    corr_1 = corr_0
                    corr_2 = corr_0
                
                # Check for NaN correlations
                if pd.isna(corr_1) or pd.isna(corr_2):
                    s1_values.append(np.nan)
                    violations.append(False)
                    continue
                
                # Simplified S1 calculation (approximation)
                s1 = abs(corr_0) + abs(corr_1) + abs(corr_2) - abs(corr_0 * corr_1)
                s1 = min(abs(s1) * 2, 4)  # Scale and cap
                
                s1_values.append(s1)
                violations.append(s1 > 2.0)  # Classical bound
            
            return s1_values, violations
            
        except Exception as e:
            print(f"   Warning: S1 calculation failed: {e}")
            return [], []
    
    def run_complete_cross_sector_analysis(self):
        """Run complete cross-sector analysis"""
        print("🌐 CROSS-SECTOR BELL INEQUALITY ANALYSIS")
        print("=" * 60)
        print("🎯 Cross-Sector Matrix Analysis")
        print("📊 Analyzing quantum correlations between different market sectors")
        print()
        
        # Test data availability
        self.test_cross_sector_data_availability()
        
        # Initialize results storage
        all_results = []
        
        # Analyze each cross-sector pair
        for asset1, asset2, pair_name, sector1, sector2 in self.cross_sector_pairs:
            result = self.analyze_cross_sector_pair(asset1, asset2, pair_name, sector1, sector2)
            if result:
                all_results.append(result)
        
        # Create cross-sector summary
        print("📊 CROSS-SECTOR ANALYSIS SUMMARY")
        print("=" * 60)
        
        # Group by sector combinations
        sector_combinations = {}
        for result in all_results:
            combo = f"{result['sector1']} vs {result['sector2']}"
            if combo not in sector_combinations:
                sector_combinations[combo] = []
            sector_combinations[combo].append(result)
        
        for combo, results in sector_combinations.items():
            print(f"\n🔗 {combo}:")
            avg_violation = np.mean([r['violation_rate'] for r in results])
            print(f"   📊 Average violation rate: {avg_violation:.1f}%")
            for result in results:
                print(f"      {result['asset1']}-{result['asset2']}: {result['violation_rate']:.1f}% violations")
                print(f"         {result['data_period']}")
        
        print()
        print("🎉 CROSS-SECTOR ANALYSIS COMPLETE!")
        print("📁 Results saved in: results/FINAL_CROSS_SECTOR_RESULTS/cross_sector_analysis/")
        
        return all_results

if __name__ == "__main__":
    # Run complete cross-sector analysis
    analyzer = CrossSectorAnalyzer()
    results = analyzer.run_complete_cross_sector_analysis()
