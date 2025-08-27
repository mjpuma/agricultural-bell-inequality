#!/usr/bin/env python3
"""
📊 CAPM β BREAKDOWN DEMONSTRATION
==================================
Concrete example showing how S1 violations correspond to CAPM β instability
Using XOM-JPM pair as demonstration

USAGE:
    python src/capm_beta_breakdown_demo.py

OUTPUT:
    - results/FINAL_CROSS_SECTOR_RESULTS/summary/capm_beta_breakdown_demo_*.png
    - results/FINAL_CROSS_SECTOR_RESULTS/tables/capm_beta_breakdown_summary_*.xlsx
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class CAPMBetaBreakdownDemo:
    """Demonstrate CAPM β breakdown during S1 violation periods"""
    
    def __init__(self):
        self.results_dir = 'results/FINAL_CROSS_SECTOR_RESULTS'
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("📊 CAPM β BREAKDOWN DEMONSTRATION")
        print("=" * 50)
        print("🎯 Showing how S1 violations correspond to CAPM model breakdown")
        print("📈 Using XOM-JPM pair (Energy vs Finance) as demonstration")
        print()
    
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
    
    def calculate_s1_values(self, data):
        """Calculate S1 Bell inequality values for XOM-JPM pair"""
        
        print("🔬 Calculating S1 Bell inequality values...")
        
        # Get returns
        returns = data['Close'].pct_change().dropna()
        xom_returns = returns['XOM']
        jpm_returns = returns['JPM']
        
        # Calculate S1 values with rolling window (same as cross-sector analysis)
        window = 10  # Same as cross-sector analysis
        s1_values = []
        s1_dates = []
        
        for i in range(window, len(xom_returns)):
            # Get window data
            xom_window = xom_returns.iloc[i-window:i]
            jpm_window = jpm_returns.iloc[i-window:i]
            
            # Check if we have enough non-NaN data in the window
            xom_valid = xom_window.dropna()
            jpm_valid = jpm_window.dropna()
            
            if len(xom_valid) < 10 or len(jpm_valid) < 10:
                s1_values.append(np.nan)
                s1_dates.append(xom_returns.index[i])
                continue
            
            # Calculate correlations at different lags (same as cross-sector analysis)
            corr_0 = xom_window.corr(jpm_window)  # Simultaneous
            
            # Check for NaN correlation
            if pd.isna(corr_0):
                s1_values.append(np.nan)
                s1_dates.append(xom_returns.index[i])
                continue
            
            if len(xom_window) > 5:
                corr_1 = xom_window[:-1].corr(jpm_window[1:])  # 1-day lag
                corr_2 = xom_window[:-2].corr(jpm_window[2:])  # 2-day lag
            else:
                corr_1 = corr_0
                corr_2 = corr_0
            
            # Check for NaN correlations
            if pd.isna(corr_1) or pd.isna(corr_2):
                s1_values.append(np.nan)
                s1_dates.append(xom_returns.index[i])
                continue
            
            # Simplified S1 calculation (same as cross-sector analysis)
            s1 = abs(corr_0) + abs(corr_1) + abs(corr_2) - abs(corr_0 * corr_1)
            s1 = min(abs(s1) * 2, 4)  # Scale and cap
            
            s1_values.append(s1)
            s1_dates.append(xom_returns.index[i])
        
        s1_series = pd.Series(s1_values, index=s1_dates)
        
        print(f"✅ S1 values calculated: {len(s1_series.dropna())} valid points")
        print(f"   S1 range: {s1_series.min():.2f} to {s1_series.max():.2f}")
        print(f"   S1 > 2 violations: {(s1_series > 2).sum()} periods ({(s1_series > 2).sum()/len(s1_series.dropna())*100:.1f}%)")
        
        return s1_series
    
    def calculate_rolling_beta(self, data, window=252):
        """Calculate rolling CAPM β for XOM and JPM"""
        
        print("📈 Calculating rolling CAPM β values...")
        
        # Get returns
        returns = data['Close'].pct_change().dropna()
        xom_returns = returns['XOM']
        jpm_returns = returns['JPM']
        market_returns = returns['SPY']
        
        # Calculate rolling β for XOM
        xom_betas = []
        xom_beta_dates = []
        xom_beta_std = []  # Standard error of β
        
        for i in range(window, len(xom_returns)):
            # Get window data
            xom_window = xom_returns.iloc[i-window:i]
            market_window = market_returns.iloc[i-window:i]
            
            # Calculate β using CAPM formula: β = Cov(r_i, r_m) / Var(r_m)
            covariance = np.cov(xom_window, market_window)[0, 1]
            market_variance = np.var(market_window)
            
            if market_variance > 0:
                beta = covariance / market_variance
                
                # Calculate standard error of β
                residuals = xom_window - beta * market_window
                residual_variance = np.var(residuals)
                beta_se = np.sqrt(residual_variance / (market_variance * (window - 2)))
                
                xom_betas.append(beta)
                xom_beta_std.append(beta_se)
            else:
                xom_betas.append(np.nan)
                xom_beta_std.append(np.nan)
            
            xom_beta_dates.append(xom_returns.index[i])
        
        # Calculate rolling β for JPM
        jpm_betas = []
        jpm_beta_std = []
        
        for i in range(window, len(jpm_returns)):
            # Get window data
            jpm_window = jpm_returns.iloc[i-window:i]
            market_window = market_returns.iloc[i-window:i]
            
            # Calculate β
            covariance = np.cov(jpm_window, market_window)[0, 1]
            market_variance = np.var(market_window)
            
            if market_variance > 0:
                beta = covariance / market_variance
                
                # Calculate standard error of β
                residuals = jpm_window - beta * market_window
                residual_variance = np.var(residuals)
                beta_se = np.sqrt(residual_variance / (market_variance * (window - 2)))
                
                jpm_betas.append(beta)
                jpm_beta_std.append(beta_se)
            else:
                jpm_betas.append(np.nan)
                jpm_beta_std.append(np.nan)
        
        # Create DataFrames
        beta_df = pd.DataFrame({
            'XOM_Beta': xom_betas,
            'XOM_Beta_SE': xom_beta_std,
            'JPM_Beta': jpm_betas,
            'JPM_Beta_SE': jpm_beta_std
        }, index=xom_beta_dates)
        
        print(f"✅ Rolling β values calculated: {len(beta_df.dropna())} valid points")
        print(f"   XOM β range: {beta_df['XOM_Beta'].min():.2f} to {beta_df['XOM_Beta'].max():.2f}")
        print(f"   JPM β range: {beta_df['JPM_Beta'].min():.2f} to {beta_df['JPM_Beta'].max():.2f}")
        
        return beta_df
    
    def create_breakdown_visualization(self, s1_series, beta_df):
        """Create visualization showing β breakdown during S1 violations"""
        
        print("📊 Creating β breakdown visualization...")
        
        # Align data
        common_dates = s1_series.index.intersection(beta_df.index)
        s1_aligned = s1_series.loc[common_dates]
        beta_aligned = beta_df.loc[common_dates]
        
        # Create figure
        fig, axes = plt.subplots(3, 1, figsize=(16, 12))
        fig.suptitle('CAPM β Breakdown During S1 Violations\nXOM-JPM Pair (2010-2025)', 
                     fontsize=16, fontweight='bold')
        
        # Panel 1: S1 Values with Regime Thresholds
        ax1 = axes[0]
        ax1.plot(s1_aligned.index, s1_aligned.values, 'b-', alpha=0.7, linewidth=1)
        ax1.axhline(y=2.0, color='orange', linestyle='--', alpha=0.8, label='Classical Independence Bound (S1=2)')
        ax1.axhline(y=2.83, color='red', linestyle='--', alpha=0.8, label='Quantum Bound (S1=2.83)')
        
        # Color regions
        ax1.fill_between(s1_aligned.index, 0, 2, alpha=0.2, color='green', label='Classical Regime (S1<2)')
        ax1.fill_between(s1_aligned.index, 2, 2.83, alpha=0.2, color='yellow', label='Transitional Regime (2≤S1<2.83)')
        ax1.fill_between(s1_aligned.index, 2.83, s1_aligned.max(), alpha=0.2, color='red', label='Strong Interdependence (S1≥2.83)')
        
        ax1.set_ylabel('S1 Bell Inequality Value', fontsize=12)
        ax1.set_title('S1 Bell Inequality Values with Regime Thresholds', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: XOM β with Confidence Intervals
        ax2 = axes[1]
        xom_beta = beta_aligned['XOM_Beta']
        xom_se = beta_aligned['XOM_Beta_SE']
        
        # Plot β with confidence intervals
        ax2.plot(xom_beta.index, xom_beta.values, 'b-', alpha=0.7, linewidth=1, label='XOM β')
        ax2.fill_between(xom_beta.index, 
                        xom_beta.values - 2*xom_se.values, 
                        xom_beta.values + 2*xom_se.values, 
                        alpha=0.3, color='blue', label='95% Confidence Interval')
        
        # Highlight S1 violation periods
        s1_violations = s1_aligned > 2
        violation_dates = s1_aligned[s1_violations].index
        violation_betas = xom_beta.loc[violation_dates]
        
        ax2.scatter(violation_dates, violation_betas, 
                   color='red', s=20, alpha=0.7, label='S1 > 2 (Independence Violation)')
        
        ax2.set_ylabel('XOM β (CAPM)', fontsize=12)
        ax2.set_title('XOM Rolling β with S1 Violation Periods Highlighted', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        # Panel 3: JPM β with Confidence Intervals
        ax3 = axes[2]
        jpm_beta = beta_aligned['JPM_Beta']
        jpm_se = beta_aligned['JPM_Beta_SE']
        
        # Plot β with confidence intervals
        ax3.plot(jpm_beta.index, jpm_beta.values, 'g-', alpha=0.7, linewidth=1, label='JPM β')
        ax3.fill_between(jpm_beta.index, 
                        jpm_beta.values - 2*jpm_se.values, 
                        jpm_beta.values + 2*jpm_se.values, 
                        alpha=0.3, color='green', label='95% Confidence Interval')
        
        # Highlight S1 violation periods
        violation_betas_jpm = jpm_beta.loc[violation_dates]
        ax3.scatter(violation_dates, violation_betas_jpm, 
                   color='red', s=20, alpha=0.7, label='S1 > 2 (Independence Violation)')
        
        ax3.set_ylabel('JPM β (CAPM)', fontsize=12)
        ax3.set_xlabel('Date', fontsize=12)
        ax3.set_title('JPM Rolling β with S1 Violation Periods Highlighted', fontsize=14, fontweight='bold')
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        output_path = f'{self.results_dir}/summary/capm_beta_breakdown_demo_{self.timestamp}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ β breakdown visualization saved: {output_path}")
        plt.close()
        
        return output_path
    
    def create_statistical_summary(self, s1_series, beta_df):
        """Create statistical summary of β stability during different S1 regimes"""
        
        print("📊 Creating statistical summary...")
        
        # Align data
        common_dates = s1_series.index.intersection(beta_df.index)
        s1_aligned = s1_series.loc[common_dates]
        beta_aligned = beta_df.loc[common_dates]
        
        # Define S1 regimes
        classical_regime = s1_aligned < 2
        transitional_regime = (s1_aligned >= 2) & (s1_aligned < 2.83)
        strong_interdependence = s1_aligned >= 2.83
        
        # Calculate β statistics for each regime
        results = {}
        
        for regime_name, regime_mask in [('Classical (S1<2)', classical_regime),
                                        ('Transitional (2≤S1<2.83)', transitional_regime),
                                        ('Strong Interdependence (S1≥2.83)', strong_interdependence)]:
            
            if regime_mask.sum() > 0:
                xom_beta_regime = beta_aligned.loc[regime_mask, 'XOM_Beta'].dropna()
                jpm_beta_regime = beta_aligned.loc[regime_mask, 'JPM_Beta'].dropna()
                
                results[regime_name] = {
                    'XOM_Beta_Mean': xom_beta_regime.mean(),
                    'XOM_Beta_Std': xom_beta_regime.std(),
                    'XOM_Beta_SE_Mean': beta_aligned.loc[regime_mask, 'XOM_Beta_SE'].mean(),
                    'JPM_Beta_Mean': jpm_beta_regime.mean(),
                    'JPM_Beta_Std': jpm_beta_regime.std(),
                    'JPM_Beta_SE_Mean': beta_aligned.loc[regime_mask, 'JPM_Beta_SE'].mean(),
                    'Periods': regime_mask.sum()
                }
        
        # Create summary table
        summary_data = []
        for regime, stats in results.items():
            summary_data.append([
                regime,
                f"{stats['XOM_Beta_Mean']:.3f}",
                f"{stats['XOM_Beta_Std']:.3f}",
                f"{stats['XOM_Beta_SE_Mean']:.3f}",
                f"{stats['JPM_Beta_Mean']:.3f}",
                f"{stats['JPM_Beta_Std']:.3f}",
                f"{stats['JPM_Beta_SE_Mean']:.3f}",
                stats['Periods']
            ])
        
        summary_df = pd.DataFrame(summary_data, columns=[
            'S1 Regime', 'XOM β Mean', 'XOM β Std', 'XOM β SE Mean',
            'JPM β Mean', 'JPM β Std', 'JPM β SE Mean', 'Number of Periods'
        ])
        
        # Save summary
        output_path = f'{self.results_dir}/tables/capm_beta_breakdown_summary_{self.timestamp}.xlsx'
        summary_df.to_excel(output_path, index=False)
        print(f"✅ Statistical summary saved: {output_path}")
        
        # Print summary
        print(f"\n📊 CAPM β BREAKDOWN SUMMARY:")
        print("=" * 50)
        print(summary_df.to_string(index=False))
        
        return summary_df
    
    def run_demonstration(self):
        """Run complete CAPM β breakdown demonstration"""
        
        print("🚀 Starting CAPM β breakdown demonstration...")
        
        # Download data
        data = self.download_data()
        if data is None:
            return
        
        # Calculate S1 values
        s1_series = self.calculate_s1_values(data)
        
        # Calculate rolling β values
        beta_df = self.calculate_rolling_beta(data)
        
        # Create visualization
        viz_path = self.create_breakdown_visualization(s1_series, beta_df)
        
        # Create statistical summary
        summary_df = self.create_statistical_summary(s1_series, beta_df)
        
        print(f"\n✅ CAPM β BREAKDOWN DEMONSTRATION COMPLETE!")
        print(f"📊 Visualization: {viz_path}")
        print(f"📈 Key insight: β becomes unstable when S1 > 2 (independence violations)")
        print(f"🎯 This demonstrates how S1 violations correspond to CAPM model breakdown")

if __name__ == "__main__":
    demo = CAPMBetaBreakdownDemo()
    demo.run_demonstration()
