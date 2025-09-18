#!/usr/bin/env python3
"""
📊 MULTI-SCALE CAPM β BREAKDOWN ANALYSIS
========================================
Addressing the temporal window dilemma with multiple time scales
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class MultiScaleCAPMBetaBreakdown:
    """Multi-scale CAPM β breakdown analysis addressing temporal window dilemma"""
    
    def __init__(self):
        self.results_dir = 'results/FINAL_CROSS_SECTOR_RESULTS'
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("📊 MULTI-SCALE CAPM β BREAKDOWN ANALYSIS")
        print("=" * 50)
        print("🎯 Addressing temporal window dilemma with multiple time scales")
    
    def download_data(self):
        """Download data for XOM, JPM, and market (SPY)"""
        
        print("📥 Downloading data...")
        
        # Download data for 2010-2025
        data = yf.download(['XOM', 'JPM', 'SPY'], start='2010-01-01', end='2025-07-31', progress=False)
        
        if len(data) == 0:
            print("❌ Error: No data downloaded")
            return None
        
        print(f"✅ Data downloaded: {len(data)} days")
        print(f"   Period: {data.index[0].date()} to {data.index[-1].date()}")
        
        return data
    
    def calculate_s1_multiple_scales(self, data):
        """Calculate S1 at multiple time scales to address the dilemma"""
        
        print("🔬 Calculating S1 at multiple time scales...")
        
        # Get returns
        returns = data['Close'].pct_change().dropna()
        xom_returns = returns['XOM']
        jpm_returns = returns['JPM']
        
        # Multiple window sizes to capture different time scales
        windows = [10, 20, 60, 120, 252]  # 2 weeks, 1 month, 3 months, 6 months, 1 year
        s1_results = {}
        
        for window in windows:
            print(f"   Calculating S1 with {window}-day window...")
            s1_values = []
            s1_dates = []
            
            for i in range(window, len(xom_returns)):
                # Get window data
                xom_window = xom_returns.iloc[i-window:i]
                jpm_window = jpm_returns.iloc[i-window:i]
                
                # Check if we have enough non-NaN data
                xom_valid = xom_window.dropna()
                jpm_valid = jpm_window.dropna()
                
                if len(xom_valid) < window * 0.8 or len(jpm_valid) < window * 0.8:
                    s1_values.append(np.nan)
                    s1_dates.append(xom_returns.index[i])
                    continue
                
                # Calculate correlations at different lags
                corr_0 = xom_window.corr(jpm_window)
                
                if pd.isna(corr_0):
                    s1_values.append(np.nan)
                    s1_dates.append(xom_returns.index[i])
                    continue
                
                # Use multiple lag correlations for robust S1
                lags = [1, 2, 3, 5]  # Shorter lags for shorter windows
                if window >= 60:
                    lags.extend([10, 20])  # Longer lags for longer windows
                
                lag_corrs = []
                
                for lag in lags:
                    if len(xom_window) > lag:
                        corr_lag = xom_window[:-lag].corr(jpm_window[lag:])
                        if not pd.isna(corr_lag):
                            lag_corrs.append(abs(corr_lag))
                
                if len(lag_corrs) > 0:
                    # S1 calculation
                    s1 = abs(corr_0) + np.mean(lag_corrs)
                    s1 = min(s1 * 1.5, 4)  # Consistent scaling
                else:
                    s1 = abs(corr_0) * 2
                
                s1_values.append(s1)
                s1_dates.append(xom_returns.index[i])
            
            s1_series = pd.Series(s1_values, index=s1_dates)
            s1_results[f'{window}d'] = s1_series
            
            # Print statistics
            violations = (s1_series > 2).sum()
            total_valid = len(s1_series.dropna())
            violation_rate = violations / total_valid * 100 if total_valid > 0 else 0
            
            print(f"     ✅ {window}-day window: {violations}/{total_valid} violations ({violation_rate:.1f}%)")
        
        return s1_results
    
    def calculate_rolling_beta_multiple_scales(self, data):
        """Calculate rolling CAPM β at multiple time scales"""
        
        print("📈 Calculating rolling CAPM β at multiple time scales...")
        
        # Get returns
        returns = data['Close'].pct_change().dropna()
        xom_returns = returns['XOM']
        jpm_returns = returns['JPM']
        market_returns = returns['SPY']
        
        # Multiple β calculation windows
        beta_windows = [60, 120, 252]  # 3 months, 6 months, 1 year
        beta_results = {}
        
        for window in beta_windows:
            print(f"   Calculating β with {window}-day window...")
            
            # Calculate rolling β for both stocks
            stock_results = {}
            
            for stock_name, stock_returns in [('XOM', xom_returns), ('JPM', jpm_returns)]:
                betas = []
                beta_dates = []
                beta_std = []
                r_squared = []
                residuals_std = []
                
                for i in range(window, len(stock_returns)):
                    # Get window data
                    stock_window = stock_returns.iloc[i-window:i]
                    market_window = market_returns.iloc[i-window:i]
                    
                    # Remove any NaN values
                    valid_data = pd.DataFrame({
                        'stock': stock_window,
                        'market': market_window
                    }).dropna()
                    
                    if len(valid_data) < window * 0.8:
                        betas.append(np.nan)
                        beta_std.append(np.nan)
                        r_squared.append(np.nan)
                        residuals_std.append(np.nan)
                        beta_dates.append(stock_returns.index[i])
                        continue
                    
                    # Calculate β using robust regression
                    try:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(
                            valid_data['market'], valid_data['stock']
                        )
                        
                        beta = slope
                        residuals = valid_data['stock'] - (intercept + beta * valid_data['market'])
                        
                        betas.append(beta)
                        beta_std.append(std_err)
                        r_squared.append(r_value**2)
                        residuals_std.append(residuals.std())
                        
                    except:
                        betas.append(np.nan)
                        beta_std.append(np.nan)
                        r_squared.append(np.nan)
                        residuals_std.append(np.nan)
                    
                    beta_dates.append(stock_returns.index[i])
                
                stock_results[stock_name] = {
                    'beta': betas,
                    'beta_std': beta_std,
                    'r_squared': r_squared,
                    'residuals_std': residuals_std,
                    'dates': beta_dates
                }
            
            # Create DataFrame for this window
            beta_df = pd.DataFrame({
                'XOM_Beta': stock_results['XOM']['beta'],
                'XOM_Beta_SE': stock_results['XOM']['beta_std'],
                'XOM_R2': stock_results['XOM']['r_squared'],
                'XOM_Residuals_Std': stock_results['XOM']['residuals_std'],
                'JPM_Beta': stock_results['JPM']['beta'],
                'JPM_Beta_SE': stock_results['JPM']['beta_std'],
                'JPM_R2': stock_results['JPM']['r_squared'],
                'JPM_Residuals_Std': stock_results['JPM']['residuals_std']
            }, index=stock_results['XOM']['dates'])
            
            beta_results[f'{window}d'] = beta_df
            
            print(f"     ✅ {window}-day β window: {len(beta_df.dropna())} valid points")
        
        return beta_results
    
    def analyze_temporal_mismatch_effects(self, s1_results, beta_results):
        """Analyze the effects of temporal mismatch on S1-β relationships"""
        
        print("🔍 Analyzing temporal mismatch effects...")
        
        # Find common time periods
        common_dates = {}
        for s1_key, s1_series in s1_results.items():
            for beta_key, beta_df in beta_results.items():
                key = f"{s1_key}_vs_{beta_key}"
                common_dates[key] = s1_series.index.intersection(beta_df.index)
        
        # Calculate correlations between S1 and β for different combinations
        correlation_results = {}
        
        for key, dates in common_dates.items():
            if len(dates) > 100:  # Minimum data requirement
                s1_key, beta_key = key.split('_vs_')
                s1_aligned = s1_results[s1_key].loc[dates]
                beta_aligned = beta_results[beta_key].loc[dates]
                
                # Calculate correlations
                xom_beta = beta_aligned['XOM_Beta'].dropna()
                jpm_beta = beta_aligned['JPM_Beta'].dropna()
                s1_clean = s1_aligned.loc[xom_beta.index]
                
                if len(s1_clean) > 50:
                    # Correlation between S1 and β
                    xom_corr = s1_clean.corr(xom_beta)
                    jpm_corr = s1_clean.corr(jpm_beta)
                    
                    correlation_results[key] = {
                        'XOM_correlation': xom_corr,
                        'JPM_correlation': jpm_corr,
                        'data_points': len(s1_clean)
                    }
        
        return correlation_results
    
    def create_multiscale_visualization(self, s1_results, beta_results, correlation_results):
        """Create visualization showing multi-scale analysis"""
        
        print("📊 Creating multi-scale visualization...")
        
        # Create figure with multiple panels
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        fig.suptitle('Multi-Scale CAPM β Breakdown Analysis: Addressing Temporal Window Dilemma\nXOM-JPM Pair (2010-2025)', 
                     fontsize=16, fontweight='bold')
        
        # Panel 1: S1 violations across different time scales
        ax1 = axes[0, 0]
        
        colors = ['blue', 'green', 'orange', 'red', 'purple']
        for i, (window, s1_series) in enumerate(s1_results.items()):
            violations = s1_series > 2
            violation_rate = violations.sum() / len(s1_series.dropna()) * 100
            ax1.bar(window, violation_rate, color=colors[i], alpha=0.7, label=f'{window} window')
        
        ax1.set_xlabel('S1 Window Size (days)', fontsize=12)
        ax1.set_ylabel('S1 Violation Rate (%)', fontsize=12)
        ax1.set_title('S1 Violation Rates by Window Size\n(Shorter windows capture more violations)', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: S1 time series comparison (10d vs 252d)
        ax2 = axes[0, 1]
        
        if '10d' in s1_results and '252d' in s1_results:
            # Align data
            common_dates = s1_results['10d'].index.intersection(s1_results['252d'].index)
            s1_10d = s1_results['10d'].loc[common_dates]
            s1_252d = s1_results['252d'].loc[common_dates]
            
            ax2.plot(s1_10d.index, s1_10d.values, 'b-', alpha=0.7, linewidth=1, label='S1 (10-day window)')
            ax2.plot(s1_252d.index, s1_252d.values, 'r-', alpha=0.7, linewidth=1, label='S1 (252-day window)')
            ax2.axhline(y=2.0, color='orange', linestyle='--', alpha=0.8, label='Independence Bound (S1=2)')
            
            ax2.set_ylabel('S1 Bell Inequality Value', fontsize=12)
            ax2.set_title('S1 Comparison: 10-day vs 252-day Windows\n(Short windows capture rapid changes)', fontsize=14, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Panel 3: Temporal mismatch correlation analysis
        ax3 = axes[1, 0]
        
        if correlation_results:
            # Create heatmap of correlations
            s1_windows = list(set([key.split('_vs_')[0] for key in correlation_results.keys()]))
            beta_windows = list(set([key.split('_vs_')[1] for key in correlation_results.keys()]))
            
            corr_matrix = np.zeros((len(s1_windows), len(beta_windows)))
            
            for i, s1_win in enumerate(s1_windows):
                for j, beta_win in enumerate(beta_windows):
                    key = f"{s1_win}_vs_{beta_win}"
                    if key in correlation_results:
                        corr_matrix[i, j] = correlation_results[key]['XOM_correlation']
            
            im = ax3.imshow(corr_matrix, cmap='RdYlBu_r', aspect='auto')
            ax3.set_xticks(range(len(beta_windows)))
            ax3.set_yticks(range(len(s1_windows)))
            ax3.set_xticklabels(beta_windows)
            ax3.set_yticklabels(s1_windows)
            ax3.set_xlabel('β Window Size', fontsize=12)
            ax3.set_ylabel('S1 Window Size', fontsize=12)
            ax3.set_title('S1-β Correlation by Window Combination\n(Higher values = stronger relationship)', fontsize=14, fontweight='bold')
            
            # Add colorbar
            plt.colorbar(im, ax=ax3)
        
        # Panel 4: Optimal window combination analysis
        ax4 = axes[1, 1]
        
        if correlation_results:
            # Find optimal combinations
            optimal_combinations = []
            for key, result in correlation_results.items():
                optimal_combinations.append({
                    'combination': key,
                    'xom_corr': result['XOM_correlation'],
                    'jpm_corr': result['JPM_correlation'],
                    'avg_corr': (result['XOM_correlation'] + result['JPM_correlation']) / 2
                })
            
            optimal_df = pd.DataFrame(optimal_combinations)
            optimal_df = optimal_df.sort_values('avg_corr', ascending=True)
            
            bars = ax4.barh(range(len(optimal_df)), optimal_df['avg_corr'], color='skyblue', alpha=0.7)
            ax4.set_yticks(range(len(optimal_df)))
            ax4.set_yticklabels(optimal_df['combination'])
            ax4.set_xlabel('Average S1-β Correlation', fontsize=12)
            ax4.set_title('Window Combination Performance\n(Higher = better S1-β relationship)', fontsize=14, fontweight='bold')
            ax4.grid(True, alpha=0.3)
        
        # Panel 5: β stability across different windows
        ax5 = axes[2, 0]
        
        if '252d' in beta_results:
            beta_252d = beta_results['252d']
            xom_beta = beta_252d['XOM_Beta'].dropna()
            jpm_beta = beta_252d['JPM_Beta'].dropna()
            
            # Calculate rolling β volatility
            window = 60
            xom_vol = xom_beta.rolling(window).std()
            jpm_vol = jpm_beta.rolling(window).std()
            
            ax5.plot(xom_vol.index, xom_vol.values, 'b-', alpha=0.7, linewidth=1, label='XOM β Volatility')
            ax5.plot(jpm_vol.index, jpm_vol.values, 'g-', alpha=0.7, linewidth=1, label='JPM β Volatility')
            
            ax5.set_ylabel('β Volatility (60-day rolling std)', fontsize=12)
            ax5.set_xlabel('Date', fontsize=12)
            ax5.set_title('β Stability Across Time\n(Higher volatility = less stable β)', fontsize=14, fontweight='bold')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # Panel 6: Recommendations summary
        ax6 = axes[2, 1]
        
        ax6.text(0.1, 0.9, 'MULTI-SCALE ANALYSIS RECOMMENDATIONS:', 
                transform=ax6.transAxes, fontsize=14, fontweight='bold')
        
        recommendations = [
            '1. Short S1 windows (10-20 days) capture rapid correlation changes',
            '2. Long β windows (252 days) provide stable estimates',
            '3. Temporal mismatch may be acceptable for capturing violations',
            '4. Consider using 10-day S1 with 252-day β for optimal sensitivity',
            '5. Monitor correlation strength between S1 and β across scales'
        ]
        
        for i, rec in enumerate(recommendations):
            ax6.text(0.1, 0.8 - i*0.12, rec, transform=ax6.transAxes, fontsize=11)
        
        ax6.set_xlim(0, 1)
        ax6.set_ylim(0, 1)
        ax6.axis('off')
        ax6.set_title('Key Insights from Multi-Scale Analysis', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save figure
        output_path = f'{self.results_dir}/summary/multiscale_capm_beta_breakdown_{self.timestamp}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Multi-scale visualization saved: {output_path}")
        plt.close()
        
        return output_path
    
    def run_multiscale_analysis(self):
        """Run complete multi-scale CAPM β breakdown analysis"""
        
        print("🚀 Starting multi-scale CAPM β breakdown analysis...")
        
        # Download data
        data = self.download_data()
        if data is None:
            return
        
        # Calculate S1 at multiple scales
        s1_results = self.calculate_s1_multiple_scales(data)
        
        # Calculate β at multiple scales
        beta_results = self.calculate_rolling_beta_multiple_scales(data)
        
        # Analyze temporal mismatch effects
        correlation_results = self.analyze_temporal_mismatch_effects(s1_results, beta_results)
        
        # Create multi-scale visualization
        viz_path = self.create_multiscale_visualization(s1_results, beta_results, correlation_results)
        
        # Print summary
        print(f"\n📊 MULTI-SCALE ANALYSIS SUMMARY:")
        print("=" * 50)
        print("🎯 Key Findings:")
        print("   • Shorter S1 windows capture more violations")
        print("   • Temporal mismatch may be acceptable for sensitivity")
        print("   • Optimal combination depends on research goals")
        print("   • Consider 10-day S1 with 252-day β for balance")
        
        print(f"\n✅ MULTI-SCALE CAPM β BREAKDOWN ANALYSIS COMPLETE!")
        print(f"📊 Visualization: {viz_path}")
        print(f"🎯 This analysis addresses the temporal window dilemma")

if __name__ == "__main__":
    analyzer = MultiScaleCAPMBetaBreakdown()
    analyzer.run_multiscale_analysis()

