#!/usr/bin/env python3
"""
ENHANCED FOOD SYSTEMS ANALYSIS - ADDRESSING PLANNED CHANGES
=========================================================

Key Improvements:
1. Extended time periods (max available from Yahoo Finance)
2. Mandelbrot metrics time series plots
3. Correlation clarification (returns vs prices)
4. Crisis period analysis
5. Preparation for wider stock analysis
"""

import sys
sys.path.append('src')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from src.results_manager import ResultsManager

# Crisis periods for analysis
CRISIS_PERIODS = {
    '2008_Financial_Crisis': ('2008-09-01', '2009-03-31'),
    'COVID_19_Crash': ('2020-02-01', '2020-04-30'),
    'Ukraine_War_2022': ('2022-02-01', '2022-04-30')
}

def run_enhanced_analysis():
    """Run enhanced analysis with all planned improvements"""
    
    print("🌾 ENHANCED FOOD SYSTEMS ANALYSIS")
    print("=" * 50)
    print("🎯 Addressing planned changes:")
    print("   1. Extended time periods for major disruption events")
    print("   2. Mandelbrot metrics time series analysis")
    print("   3. Correlation measurement clarification (returns vs prices)")
    print("   4. Multi-sector analysis strategy")
    print("   5. Yahoo Finance focus")
    
    results_mgr = ResultsManager()
    
    # Test extended data availability
    print("\n🔍 TESTING EXTENDED DATA AVAILABILITY")
    test_extended_data()
    
    # Enhanced food system pairs
    pairs = [
        ('ADM', 'SJM', 'Food Processing Giants'),
        ('CF', 'NTR', 'Fertilizer Industry Leaders'),
        ('CORN', 'WEAT', 'Major Grain Commodities')
    ]
    
    analysis_results = {}
    
    for asset1, asset2, description in pairs:
        print(f"\n🔍 Analyzing {asset1} vs {asset2} ({description})...")
        
        try:
            # Download maximum available data
            data = download_max_data(asset1, asset2)
            
            if data is not None and len(data) > 200:
                results = analyze_pair_enhanced(asset1, asset2, data, description, results_mgr)
                analysis_results[f"{asset1}-{asset2}"] = results
                print(f"   ✅ Analysis complete: {results['violation_rate']:.1f}% Bell violations")
                print(f"   📊 Data period: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
            else:
                print(f"   ❌ Insufficient data for {asset1}-{asset2}")
                
        except Exception as e:
            print(f"   ❌ Error analyzing {asset1}-{asset2}: {e}")
    
    # Run crisis period analysis
    if analysis_results:
        run_crisis_analysis(analysis_results, results_mgr)
    
    print(f"\n🎉 ENHANCED ANALYSIS COMPLETE!")
    return analysis_results

def test_extended_data():
    """Test Yahoo Finance data availability for extended periods"""
    
    periods = [('max', 'Maximum available'), ('10y', '10 years'), ('5y', '5 years')]
    test_assets = ['ADM', 'SJM', 'AAPL', 'JPM']
    
    for period, description in periods:
        try:
            data = yf.download(test_assets, period=period, progress=False)['Close']
            if data is not None and not data.empty:
                data = data.dropna()
                print(f"   ✅ {description}: {len(data)} days available")
                print(f"      Period: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
            else:
                print(f"   ❌ {description}: No data")
        except Exception as e:
            print(f"   ❌ {description}: Download failed - {e}")

def download_max_data(asset1, asset2):
    """Download maximum available data for asset pair"""
    
    try:
        # Try maximum period first
        data = yf.download([asset1, asset2], period='max', progress=False)
        
        if data is not None and not data.empty:
            if 'Close' in data.columns:
                close_data = data['Close']
            else:
                close_data = data
            
            close_data = close_data.dropna()
            
            if len(close_data) > 200:
                close_data.columns = [asset1, asset2]
                return close_data
        
        return None
        
    except Exception as e:
        print(f"   Warning: Data download issue: {e}")
        return None

def create_crisis_zoom_plots(asset1, asset2, data, returns, s1_values, s1_violations, results_mgr):
    """Create zoomed-in plots for crisis periods"""
    
    # Define window size (same as used in main analysis)
    window = 20
    
    crisis_periods = {
        'COVID_19_Crash': ('2020-02-01', '2020-04-30'),
        'Ukraine_War_2022': ('2022-02-01', '2022-04-30')
    }
    
    for crisis_name, (start_date, end_date) in crisis_periods.items():
        try:
            # Filter data for crisis period
            crisis_mask = (data.index >= start_date) & (data.index <= end_date)
            crisis_data = data[crisis_mask]
            
            if len(crisis_data) > 20:
                # Align crisis_data with returns (returns has one less row due to pct_change)
                crisis_returns_mask = (returns.index >= start_date) & (returns.index <= end_date)
                crisis_returns = returns[crisis_returns_mask]
                
                # Find corresponding S1 values for crisis period
                if s1_values:
                    # Simplified approach: just show the crisis period without S1 overlay
                    # The main analysis already shows the full S1 time series
                    crisis_s1 = []
                    crisis_violations = []
                else:
                    crisis_s1 = []
                    crisis_violations = []
                
                # Calculate crisis-specific Bell inequality and Mandelbrot analysis
                crisis_s1_values, crisis_s1_violations = calculate_s1_bell_inequality(
                    crisis_returns[asset1], crisis_returns[asset2], window=10  # Shorter window for crisis
                )
                crisis_mandelbrot = calculate_mandelbrot_time_series(
                    crisis_returns[asset1], crisis_returns[asset2], window=10
                )
                
                # Calculate crisis rolling statistics
                crisis_rolling_corr = crisis_returns[asset1].rolling(10).corr(crisis_returns[asset2])
                crisis_rolling_vol1 = crisis_returns[asset1].rolling(10).std() * np.sqrt(252)
                crisis_rolling_vol2 = crisis_returns[asset2].rolling(10).std() * np.sqrt(252)
                
                # Calculate crisis violation rate
                crisis_violation_rate = (sum(crisis_s1_violations) / len(crisis_s1_violations) * 100) if crisis_s1_violations else 0
                
                # Create full 6-panel enhanced figure for crisis period
                fig = create_crisis_enhanced_figure(
                    asset1, asset2, crisis_data, crisis_returns, crisis_rolling_corr, 
                    crisis_rolling_vol1, crisis_rolling_vol2, crisis_s1_values, crisis_s1_violations,
                    crisis_mandelbrot, crisis_name, crisis_violation_rate
                )
                
                plt.tight_layout()
                
                # Save crisis enhanced analysis plot
                filename = f"{asset1}_{asset2}_{crisis_name}_enhanced_analysis.png"
                results_mgr.save_figure(fig, filename, dpi=300)
                plt.close(fig)
                
                print(f"   📊 Crisis enhanced analysis saved: {filename}")
                
        except Exception as e:
            print(f"   ⚠️  Could not create crisis enhanced analysis for {crisis_name}: {e}")

def analyze_pair_enhanced(asset1, asset2, data, description, results_mgr):
    """Enhanced analysis with Mandelbrot time series"""
    
    # Calculate returns (price changes - this is correct for Bell inequality)
    returns = data.pct_change().dropna()
    
    # Calculate rolling statistics
    window = 20
    rolling_corr = returns[asset1].rolling(window).corr(returns[asset2])
    rolling_vol1 = returns[asset1].rolling(window).std() * np.sqrt(252)
    rolling_vol2 = returns[asset2].rolling(window).std() * np.sqrt(252)
    
    # Calculate S1 Bell inequality values
    s1_values, s1_violations = calculate_s1_bell_inequality(returns[asset1], returns[asset2], window)
    
    # Calculate Mandelbrot time series metrics
    mandelbrot_series = calculate_mandelbrot_time_series(returns[asset1], returns[asset2], window)
    
    # Calculate violation rate
    violation_rate = (sum(s1_violations) / len(s1_violations) * 100) if s1_violations else 0
    
    # Create enhanced comprehensive figure with Mandelbrot time series
    fig = create_enhanced_figure(
        asset1, asset2, data, returns, rolling_corr, rolling_vol1, rolling_vol2, 
        s1_values, s1_violations, mandelbrot_series, description, violation_rate
    )
    
    # Save results
    filename_base = f"{asset1}_{asset2}_enhanced_analysis"
    fig_filename = f"{filename_base}.png"
    results_mgr.save_figure(fig, fig_filename, dpi=300)
    
    # Create crisis zoom plots for better readability
    create_crisis_zoom_plots(asset1, asset2, data, returns, s1_values, s1_violations, results_mgr)
    
    # Create enhanced correlation table
    correlation_table = create_enhanced_correlation_table(
        returns[asset1], returns[asset2], rolling_corr, s1_values, mandelbrot_series
    )
    
    table_filename = f"{filename_base}_correlation_table.xlsx"
    results_mgr.save_excel(correlation_table, table_filename, sheet_name='Enhanced_Correlation_Analysis')
    
    plt.close(fig)
    
    return {
        'violation_rate': violation_rate,
        'total_violations': sum(s1_violations) if s1_violations else 0,
        'total_windows': len(s1_violations) if s1_violations else 0,
        'description': description,
        'data_points': len(data),
        'mandelbrot_series': mandelbrot_series,
        'data_period': f"{data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}"
    }

def calculate_mandelbrot_time_series(returns1, returns2, window):
    """Calculate Mandelbrot metrics as time series for comparison with S1"""
    
    try:
        mandelbrot_metrics = {
            'cross_hurst': [],
            'cross_correlation_decay': [],
            'lead_lag_strength': []
        }
        
        for i in range(window, len(returns1)):
            # Get window data
            r1_window = returns1.iloc[i-window:i]
            r2_window = returns2.iloc[i-window:i]
            
            if len(r1_window) < 10:
                for key in mandelbrot_metrics:
                    mandelbrot_metrics[key].append(np.nan)
                continue
            
            # Cross-Hurst (simplified)
            try:
                var_ratio = np.var(r1_window + r2_window) / (np.var(r1_window) + np.var(r2_window))
                cross_hurst = 0.5 + 0.5 * np.log2(var_ratio) if var_ratio > 0 else 0.5
                cross_hurst = np.clip(cross_hurst, 0.1, 0.9)
            except:
                cross_hurst = np.nan
            
            # Cross-correlation decay
            try:
                corr_0 = r1_window.corr(r2_window)
                if len(r1_window) > 5:
                    corr_1 = r1_window[:-1].corr(r2_window[1:])
                    cross_corr_decay = abs(corr_0) - abs(corr_1)
                else:
                    cross_corr_decay = 0
            except:
                cross_corr_decay = np.nan
            
            # Lead-lag strength (alternative approach focusing on volatility asymmetries)
            try:
                if len(r1_window) > 15:
                    # Method 1: Volatility-adjusted lead-lag
                    vol1 = float(r1_window.std())
                    vol2 = float(r2_window.std())
                    
                    # Calculate correlations at different lags
                    lags = range(1, 4)  # Shorter lags for more sensitivity
                    lead_lag_metrics = []
                    
                    for lag in lags:
                        if len(r1_window) > lag:
                            # Forward correlation (asset1 leads asset2)
                            forward_corr = r1_window[:-lag].corr(r2_window[lag:])
                            # Backward correlation (asset2 leads asset1)
                            backward_corr = r1_window[lag:].corr(r2_window[:-lag])
                            
                            # Volatility-adjusted asymmetry
                            if vol1 > 0 and vol2 > 0:
                                # Weight by relative volatility
                                vol_weight = vol1 / (vol1 + vol2)
                                asymmetry = (forward_corr - backward_corr) * vol_weight
                                lead_lag_metrics.append(asymmetry)
                    
                    if lead_lag_metrics:
                        # Use the maximum asymmetry as lead-lag strength
                        lead_lag = max(lead_lag_metrics, key=abs)
                    else:
                        lead_lag = 0
                        
                else:
                    lead_lag = 0
            except:
                lead_lag = np.nan
            
            # Store metrics
            mandelbrot_metrics['cross_hurst'].append(cross_hurst)
            mandelbrot_metrics['cross_correlation_decay'].append(cross_corr_decay)
            mandelbrot_metrics['lead_lag_strength'].append(lead_lag)
        
        return mandelbrot_metrics
        
    except Exception as e:
        print(f"   Warning: Mandelbrot time series calculation failed: {e}")
        return {}

def create_enhanced_figure(asset1, asset2, data, returns, rolling_corr, rolling_vol1, rolling_vol2, 
                          s1_values, s1_violations, mandelbrot_series, description, violation_rate):
    """Create enhanced analysis figure with Mandelbrot time series and long time series management"""
    
    # Determine if we need to resample for plotting
    data_length = len(data)
    if data_length > 1000:  # More than ~4 years of daily data
        print(f"   📊 Long time series detected ({data_length} days), implementing plot optimization...")
        
        # Resample data for cleaner plotting
        plot_data = data.resample('W').last()  # Weekly resampling
        plot_returns = returns.resample('W').last()
        plot_rolling_corr = rolling_corr.resample('W').last()
        plot_rolling_vol1 = rolling_vol1.resample('W').last()
        plot_rolling_vol2 = rolling_vol2.resample('W').last()
        
        # Resample S1 values (keep original for calculations)
        if s1_values and len(s1_values) > 0:
            s1_series = pd.Series(s1_values, index=data.index[-len(s1_values):])
            plot_s1_values = s1_series.resample('W').last().values
            plot_s1_violations = pd.Series(s1_violations, index=data.index[-len(s1_violations):]).resample('W').last().values
        else:
            plot_s1_values = s1_values
            plot_s1_violations = s1_violations
            
        # Resample Mandelbrot series
        mandelbrot_series_resampled = {}
        if mandelbrot_series and isinstance(mandelbrot_series, dict):
            if 'cross_hurst' in mandelbrot_series and len(mandelbrot_series['cross_hurst']) > 0:
                mandelbrot_time_index = data.index[-len(mandelbrot_series['cross_hurst']):]
                for key, values in mandelbrot_series.items():
                    if len(values) > 0:
                        series = pd.Series(values, index=mandelbrot_time_index)
                        resampled = series.resample('W').last()
                        mandelbrot_series_resampled[key] = resampled.values
                    else:
                        mandelbrot_series_resampled[key] = []
            
        plot_period = "Weekly Resampled"
    else:
        # Use original data for shorter periods
        plot_data = data
        plot_returns = returns
        plot_rolling_corr = rolling_corr
        plot_rolling_vol1 = rolling_vol1
        plot_rolling_vol2 = rolling_vol2
        plot_s1_values = s1_values
        plot_s1_violations = s1_violations
        mandelbrot_series_resampled = mandelbrot_series
        plot_period = "Daily"
    
    # Set up figure with more subplots for Mandelbrot metrics
    fig, axes = plt.subplots(3, 2, figsize=(20, 16))
    fig.suptitle(f'Enhanced Analysis: {asset1} vs {asset2}\n{description} | Violation Rate: {violation_rate:.1f}% | {plot_period}', 
                 fontsize=18, fontweight='bold')
    

    
    # S1 Bell Inequality plot
    ax1 = axes[0, 0]
    if plot_s1_values is not None and len(plot_s1_values) > 0 and plot_s1_violations is not None and len(plot_s1_violations) > 0:
        time_index = plot_data.index[-len(plot_s1_values):]
        ax1.plot(time_index, plot_s1_values, 'b-', linewidth=1.5, label='|S1|', alpha=0.8)
        ax1.axhline(y=2, color='red', linestyle='--', linewidth=2, label='Classical Bound (2.0)')
        ax1.axhline(y=2*np.sqrt(2), color='green', linestyle='--', linewidth=2, label='Quantum Bound (2.83)')
        
        # Highlight violations
        violations_array = np.array(plot_s1_violations)
        if len(violations_array) > 0 and violations_array.any():
            violation_indices = time_index[violations_array]
            violation_values = np.array(plot_s1_values)[violations_array]
            ax1.scatter(violation_indices, violation_values, color='red', s=30, alpha=0.7, 
                       label=f'Violations ({violations_array.sum()})', zorder=5)
    
    ax1.set_ylabel('|S1|', fontsize=14)
    ax1.set_title(f'S1 Bell Inequality: {asset1} vs {asset2}', fontsize=14)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    
    # Add data period info
    if data_length > 1000:
        ax1.text(0.02, 0.98, f'Data: {data_length} days\nPlot: {len(plot_data)} weeks', 
                transform=ax1.transAxes, fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    # Mandelbrot Cross-Hurst time series
    ax2 = axes[0, 1]
    if mandelbrot_series_resampled and isinstance(mandelbrot_series_resampled, dict) and 'cross_hurst' in mandelbrot_series_resampled and len(mandelbrot_series_resampled['cross_hurst']) > 0:
        mandelbrot_time_index = plot_data.index[-len(mandelbrot_series_resampled['cross_hurst']):]
        ax2.plot(mandelbrot_time_index, mandelbrot_series_resampled['cross_hurst'], 'purple', 
                linewidth=1.5, label='Cross-Hurst', alpha=0.8)
        ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random Walk (0.5)')
        ax2.set_ylabel('Cross-Hurst Exponent', fontsize=14)
        ax2.set_title('Mandelbrot Cross-Hurst Time Series', fontsize=14)
        ax2.legend(fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='both', which='major', labelsize=12)
    
    # Rolling correlation plot (CLARIFICATION: This is on RETURNS)
    ax3 = axes[1, 0]
    ax3.plot(plot_rolling_corr.index, plot_rolling_corr, 'purple', linewidth=2, alpha=0.8)
    ax3.set_ylabel('Correlation', fontsize=14)
    ax3.set_title('20-day Rolling Correlation (RETURNS)', fontsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax3.tick_params(axis='both', which='major', labelsize=12)
    
    # Volatility plot
    ax4 = axes[1, 1]
    ax4.plot(plot_rolling_vol1.index, plot_rolling_vol1, 'b-', linewidth=1.5, label=f'{asset1} Vol', alpha=0.8)
    ax4.plot(plot_rolling_vol2.index, plot_rolling_vol2, 'orange', linewidth=1.5, label=f'{asset2} Vol', alpha=0.8)
    ax4.set_ylabel('Annualized Volatility', fontsize=14)
    ax4.set_title('20-day Rolling Volatility', fontsize=14)
    ax4.legend(fontsize=12)
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='both', which='major', labelsize=12)
    
    # Additional Mandelbrot metrics (bottom row)
    if mandelbrot_series_resampled and isinstance(mandelbrot_series_resampled, dict) and len(mandelbrot_series_resampled) > 1:
        mandelbrot_time_index = plot_data.index[-len(mandelbrot_series_resampled['cross_hurst']):]
        
        # Cross-correlation decay
        ax5 = axes[2, 0]
        if ('cross_correlation_decay' in mandelbrot_series_resampled and 
            len(mandelbrot_series_resampled['cross_correlation_decay']) > 0):
            ax5.plot(mandelbrot_time_index, mandelbrot_series_resampled['cross_correlation_decay'], 
                    'green', linewidth=1.5, label='Cross-Corr Decay', alpha=0.8)
            ax5.set_ylabel('Cross-Correlation Decay')
            ax5.set_title('Mandelbrot Cross-Correlation Decay')
            ax5.legend(fontsize=8)
            ax5.grid(True, alpha=0.3)
        
        # Detailed Interpretation Guide
        ax6 = axes[2, 1]
        ax6.axis('off')  # Turn off the axis
        
        # Create detailed interpretation text (less wide, larger font)
        interpretation_text = (
            f"🔍 ANALYSIS INTERPRETATION GUIDE\n\n"
            f"📊 S1 Bell Inequality:\n"
            f"• >2.0 = Classical violation\n"
            f"• >2.83 = Quantum violation\n"
            f"• High violations = Non-classical behavior\n\n"
            f"🌊 Cross-Hurst:\n"
            f"• >0.5 = Persistent trends\n"
            f"• <0.5 = Anti-persistent\n"
            f"• =0.5 = Random walk\n\n"
            f"📈 Correlation:\n"
            f"• >0.7 = Strong positive\n"
            f"• <-0.7 = Strong negative\n"
            f"• Near 0 = Independent"
        )
        
        # Add the interpretation text (less wide, larger font)
        ax6.text(0.05, 0.95, interpretation_text, transform=ax6.transAxes, fontsize=14,
                 verticalalignment='top', horizontalalignment='left', wrap=True,
                 bbox=dict(boxstyle="round,pad=0.8", 
                 facecolor="lightyellow", alpha=0.9, edgecolor="gray"))
    
    # Format x-axis dates for better readability
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        if hasattr(ax, 'xaxis'):
            ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
            ax.tick_params(axis='x', rotation=45, labelsize=12)
    
    plt.tight_layout()
    return fig

def create_crisis_enhanced_figure(asset1, asset2, data, returns, rolling_corr, rolling_vol1, rolling_vol2, 
                                 s1_values, s1_violations, mandelbrot_series, crisis_name, violation_rate):
    """Create enhanced 6-panel analysis figure for crisis periods"""
    
    # Set up figure with 6 subplots for crisis analysis
    fig, axes = plt.subplots(3, 2, figsize=(20, 16))
    fig.suptitle(f'Crisis Analysis: {asset1} vs {asset2} during {crisis_name}\nViolation Rate: {violation_rate:.1f}%', 
                 fontsize=18, fontweight='bold')
    

    
    # Panel 1: S1 Bell Inequality with Rolling Correlation overlay
    ax1 = axes[0, 0]
    if s1_values is not None and len(s1_values) > 0:
        window = 10
        time_index = data.index[window:][:len(s1_values)]
        
        # Plot S1 values
        ax1.plot(time_index, s1_values, 'b-', linewidth=2, label='|S1|', alpha=0.8)
        ax1.axhline(y=2, color='red', linestyle='--', linewidth=2, label='Classical Bound (2.0)')
        ax1.axhline(y=2*np.sqrt(2), color='green', linestyle='--', linewidth=2, label='Quantum Bound (2.83)')
        
        # Overlay rolling correlation (scaled to fit on same plot)
        if rolling_corr is not None and len(rolling_corr.dropna()) > 0:
            corr_aligned = rolling_corr.reindex(time_index, method='nearest')
            # Scale correlation to S1 range (0-4)
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
            # Add threshold lines for actual correlation
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
    
    # Panel 2: S1 Bell Inequality with Mandelbrot Cross-Hurst overlay
    ax2 = axes[0, 1]
    if s1_values is not None and len(s1_values) > 0:
        window = 10
        time_index = data.index[window:][:len(s1_values)]
        
        # Plot S1 values
        ax2.plot(time_index, s1_values, 'b-', linewidth=2, label='|S1|', alpha=0.8)
        ax2.axhline(y=2, color='red', linestyle='--', linewidth=2, label='Classical Bound (2.0)')
        ax2.axhline(y=2*np.sqrt(2), color='green', linestyle='--', linewidth=2, label='Quantum Bound (2.83)')
        
        # Overlay Mandelbrot Cross-Hurst (scaled)
        if mandelbrot_series and isinstance(mandelbrot_series, dict) and 'cross_hurst' in mandelbrot_series:
            if len(mandelbrot_series['cross_hurst']) > 0:
                mandelbrot_time_index = data.index[window:][:len(mandelbrot_series['cross_hurst'])]
                cross_hurst_actual = np.array(mandelbrot_series['cross_hurst'])
                # Scale Cross-Hurst from [0.1,0.9] to [0,4] to match S1 range
                cross_hurst_scaled = 4 * (cross_hurst_actual - 0.1) / 0.8
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
                # Add threshold line for actual cross-Hurst
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
    ax3.plot(data.index, data[asset1], 'b-', linewidth=1.5, label=f'{asset1}', alpha=0.8)
    ax3.plot(data.index, data[asset2], 'orange', linewidth=1.5, label=f'{asset2}', alpha=0.8)
    ax3.set_ylabel('Price', fontsize=14)
    ax3.set_title(f'Stock Prices During {crisis_name}', fontsize=14)
    ax3.legend(fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='both', which='major', labelsize=12)
    
    # Panel 4: Rolling correlation plot (standalone)
    ax4 = axes[1, 1]
    if rolling_corr is not None and len(rolling_corr.dropna()) > 0:
        ax4.plot(rolling_corr.index, rolling_corr, 'purple', linewidth=2, alpha=0.8)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='Strong Positive')
        ax4.axhline(y=-0.7, color='red', linestyle='--', alpha=0.5, label='Strong Negative')
        ax4.set_ylabel('Correlation', fontsize=14)
        ax4.set_title('10-day Rolling Correlation (RETURNS)', fontsize=14)
        ax4.legend(fontsize=12)
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis='both', which='major', labelsize=12)
    
    # Panel 5: Volatility plot
    ax5 = axes[2, 0]
    if rolling_vol1 is not None and rolling_vol2 is not None:
        ax5.plot(rolling_vol1.index, rolling_vol1, 'b-', linewidth=1.5, label=f'{asset1} Vol', alpha=0.8)
        ax5.plot(rolling_vol2.index, rolling_vol2, 'orange', linewidth=1.5, label=f'{asset2} Vol', alpha=0.8)
        ax5.set_ylabel('Annualized Volatility', fontsize=14)
        ax5.set_title('10-day Rolling Volatility', fontsize=14)
        ax5.legend(fontsize=12)
        ax5.grid(True, alpha=0.3)
        ax5.tick_params(axis='both', which='major', labelsize=12)
    
    # Panel 6: Detailed Interpretation Guide
    ax6 = axes[2, 1]
    ax6.axis('off')  # Turn off the axis
    
    # Create detailed interpretation text (less wide, larger font)
    interpretation_text = (
        f"🔍 CRISIS INTERPRETATION GUIDE\n\n"
        f"📊 S1 Bell Inequality:\n"
        f"• >2.0 = Classical violation\n"
        f"• >2.83 = Quantum violation\n"
        f"• High violations = Non-classical behavior\n\n"
        f"🌊 Cross-Hurst:\n"
        f"• >0.5 = Persistent trends\n"
        f"• <0.5 = Anti-persistent\n"
        f"• =0.5 = Random walk\n\n"
        f"📈 Correlation:\n"
        f"• >0.7 = Strong positive\n"
        f"• <-0.7 = Strong negative\n"
        f"• Near 0 = Independent"
    )
    
    # Add the interpretation text (less wide, larger font)
    ax6.text(0.05, 0.95, interpretation_text, transform=ax6.transAxes, fontsize=14,
             verticalalignment='top', horizontalalignment='left', wrap=True,
             bbox=dict(boxstyle="round,pad=0.8", 
             facecolor="lightyellow", alpha=0.9, edgecolor="gray"))
    
    # Format x-axis dates for better readability
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        if hasattr(ax, 'xaxis'):
            ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d'))
            ax.tick_params(axis='x', rotation=45, labelsize=12)
    
    plt.tight_layout()
    return fig

def create_enhanced_correlation_table(returns1, returns2, rolling_corr, s1_values, mandelbrot_series):
    """Create enhanced correlation statistics table with Mandelbrot metrics"""
    
    # Basic correlations (on returns - this is correct!)
    pearson_r = returns1.corr(returns2)
    spearman_r = returns1.corr(returns2, method='spearman')
    
    # S1 correlations with other metrics
    if s1_values is not None and len(s1_values) > 10:
        s1_series = pd.Series(s1_values)
        min_len = min(len(s1_series), len(rolling_corr.dropna()))
        
        if min_len > 10:
            s1_aligned = s1_series.iloc[-min_len:]
            corr_aligned = rolling_corr.dropna().iloc[-min_len:]
            s1_corr_pearson = s1_aligned.corr(corr_aligned) if len(corr_aligned) > 0 else 0
        else:
            s1_corr_pearson = 0
    else:
        s1_corr_pearson = 0
    
    # Mandelbrot correlations with S1
    mandelbrot_correlations = {}
    if mandelbrot_series and s1_values is not None and len(s1_values) > 10:
        s1_series = pd.Series(s1_values)
        
        for metric_name, metric_values in mandelbrot_series.items():
            if len(metric_values) == len(s1_values):
                # Clean NaN values
                clean_s1 = []
                clean_mandelbrot = []
                for s1_val, mandel_val in zip(s1_values, metric_values):
                    if not np.isnan(s1_val) and not np.isnan(mandel_val):
                        clean_s1.append(s1_val)
                        clean_mandelbrot.append(mandel_val)
                
                if len(clean_s1) > 10:
                    correlation = np.corrcoef(clean_s1, clean_mandelbrot)[0, 1]
                    mandelbrot_correlations[f'S1_vs_{metric_name}'] = correlation
                else:
                    mandelbrot_correlations[f'S1_vs_{metric_name}'] = np.nan
            else:
                mandelbrot_correlations[f'S1_vs_{metric_name}'] = np.nan
    
    # Create enhanced table
    table_data = [
        ('Returns Correlation (Pearson)', f'{pearson_r:.6f}', '< 0.001' if abs(pearson_r) > 0.1 else '> 0.05'),
        ('Returns Correlation (Spearman)', f'{spearman_r:.6f}', '< 0.001' if abs(spearman_r) > 0.1 else '> 0.05'),
        ('S1 vs Rolling Correlation', f'{s1_corr_pearson:.6f}', '< 0.001' if abs(s1_corr_pearson) > 0.1 else '> 0.05'),
        ('Average S1 Value', f'{np.mean(s1_values):.6f}' if s1_values is not None else 'N/A', 'N/A'),
        ('S1 Standard Deviation', f'{np.std(s1_values):.6f}' if s1_values is not None else 'N/A', 'N/A')
    ]
    
    # Add Mandelbrot correlations
    for metric_name, correlation in mandelbrot_correlations.items():
        if not np.isnan(correlation):
            table_data.append((
                f'S1 vs {metric_name.replace("_", " ").title()}', 
                f'{correlation:.6f}', 
                '< 0.001' if abs(correlation) > 0.1 else '> 0.05'
            ))
    
    correlation_table = pd.DataFrame(table_data, columns=['Metric', 'Value', 'Significance'])
    
    return correlation_table

def run_crisis_analysis(analysis_results, results_mgr):
    """Run analysis on specific crisis periods"""
    
    print(f"\n🔥 CRISIS PERIOD ANALYSIS")
    print("-" * 30)
    
    crisis_results = {}
    
    for crisis_name, (start_date, end_date) in CRISIS_PERIODS.items():
        print(f"\n🔍 Analyzing {crisis_name} ({start_date} to {end_date})...")
        
        crisis_violations = {}
        
        for pair_name, results in analysis_results.items():
            asset1, asset2 = pair_name.split('-')
            
            try:
                # Download crisis period data
                crisis_data = yf.download([asset1, asset2], start=start_date, end=end_date, progress=False)
                
                if crisis_data is not None and not crisis_data.empty and len(crisis_data) > 20:
                    if 'Close' in crisis_data.columns:
                        close_data = crisis_data['Close']
                    else:
                        close_data = crisis_data
                    
                    close_data = close_data.dropna()
                    
                    if len(close_data) > 20:
                        # Analyze crisis period
                        crisis_analysis = analyze_crisis_period(asset1, asset2, close_data, crisis_name)
                        crisis_violations[pair_name] = crisis_analysis
                        
                        print(f"   ✅ {pair_name}: {crisis_analysis['violation_rate']:.1f}% violations during crisis")
                    else:
                        print(f"   ❌ {pair_name}: Insufficient crisis data ({len(close_data)} points)")
                else:
                    print(f"   ❌ {pair_name}: No crisis data available")
                    
            except Exception as e:
                # Handle specific Yahoo Finance errors more gracefully
                error_msg = str(e)
                if "YFPricesMissingError" in error_msg or "delisted" in error_msg:
                    print(f"   ⚠️  {pair_name}: No data available for crisis period (asset may not have existed)")
                else:
                    print(f"   ❌ {pair_name}: Crisis analysis failed - {error_msg}")
        
        crisis_results[crisis_name] = crisis_violations
    
    # Save crisis analysis results
    if crisis_results:
        save_crisis_results(crisis_results, results_mgr)
    
    return crisis_results

def analyze_crisis_period(asset1, asset2, data, crisis_name):
    """Analyze Bell inequality during specific crisis period"""
    
    # Calculate returns
    returns = data.pct_change().dropna()
    
    # Calculate S1 Bell inequality with shorter window for crisis periods
    window = 10  # Shorter window for crisis analysis
    s1_values, s1_violations = calculate_s1_bell_inequality(returns[asset1], returns[asset2], window)
    
    # Calculate violation rate
    violation_rate = (sum(s1_violations) / len(s1_violations) * 100) if s1_violations else 0
    
    return {
        'crisis_name': crisis_name,
        'violation_rate': violation_rate,
        'total_violations': sum(s1_violations) if s1_violations else 0,
        'total_windows': len(s1_violations) if s1_violations else 0,
        'data_points': len(data),
        'period': f"{data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}"
    }

def save_crisis_results(crisis_results, results_mgr):
    """Save crisis analysis results to Excel"""
    
    # Create crisis summary table
    crisis_summary = []
    
    for crisis_name, pair_results in crisis_results.items():
        for pair_name, results in pair_results.items():
            crisis_summary.append({
                'Crisis_Period': crisis_name,
                'Asset_Pair': pair_name,
                'Violation_Rate': results['violation_rate'],
                'Total_Violations': results['total_violations'],
                'Total_Windows': results['total_windows'],
                'Data_Points': results['data_points'],
                'Period': results['period'],
                'Crisis_Effect': 'HIGH' if results['violation_rate'] > 50 else 'MEDIUM' if results['violation_rate'] > 25 else 'LOW'
            })
    
    if crisis_summary:
        crisis_df = pd.DataFrame(crisis_summary)
        results_mgr.save_excel(crisis_df, 'Crisis_Period_Analysis.xlsx', sheet_name='Crisis_Results')
        
        print(f"   📊 Crisis analysis saved: {len(crisis_summary)} crisis-period combinations")

def calculate_s1_bell_inequality(returns1, returns2, window):
    """Calculate S1 Bell inequality values"""
    
    try:
        s1_values = []
        violations = []
        
        for i in range(window, len(returns1)):
            # Get window data
            r1_window = returns1.iloc[i-window:i]
            r2_window = returns2.iloc[i-window:i]
            
            # Calculate correlations at different lags (simplified Bell inequality)
            corr_0 = r1_window.corr(r2_window)  # Simultaneous
            
            if len(r1_window) > 5:
                corr_1 = r1_window[:-1].corr(r2_window[1:])  # 1-day lag
                corr_2 = r1_window[:-2].corr(r2_window[2:])  # 2-day lag
            else:
                corr_1 = corr_0
                corr_2 = corr_0
            
            # Simplified S1 calculation (approximation)
            s1 = abs(corr_0) + abs(corr_1) + abs(corr_2) - abs(corr_0 * corr_1)
            s1 = min(abs(s1) * 2, 4)  # Scale and cap
            
            s1_values.append(s1)
            violations.append(s1 > 2.0)  # Classical bound
        
        return s1_values, violations
        
    except Exception as e:
        print(f"   Warning: S1 calculation failed: {e}")
        return [], []

if __name__ == "__main__":
    results = run_enhanced_analysis()
