#!/usr/bin/env python3
"""
YAHOO FINANCE BELL INEQUALITY ANALYSIS
=====================================
Adapted from WDRS analysis for Yahoo Finance data
Includes CHSH and S1 conditional Bell tests on 6 months of free data
Focus on cross-variable Mandelbrot metrics between time series
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta
from scipy import stats
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# =================== ASSET CONFIGURATION ===================
DEFAULT_ASSETS = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'META', 'AMZN', 'NFLX']
SECTOR_GROUPS = {
    'tech': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META'],
    'growth': ['TSLA', 'NVDA', 'NFLX', 'META'],
    'mega_cap': ['AAPL', 'MSFT', 'GOOGL', 'AMZN'],
    'volatile': ['TSLA', 'NFLX', 'NVDA']
}

class YahooFinanceBellAnalyzer:
    
    def __init__(self, assets=None, period='6mo'):
        self.assets = assets or DEFAULT_ASSETS
        self.period = period
        self.raw_data = None
        self.processed_data = {}
        self.bell_results = {}
        self.mandelbrot_results = {}
        
        print(f"🔬 Yahoo Finance Bell Analyzer initialized")
        print(f"   Assets: {self.assets}")
        print(f"   Period: {period}")
        
    def run_complete_analysis(self, frequency='1h'):
        """Complete Bell inequality analysis on Yahoo Finance data"""
        
        print("\n🚀 YAHOO FINANCE BELL INEQUALITY ANALYSIS")
        print("=" * 60)
        
        # Step 1: Download Yahoo Finance data
        self.raw_data = self._download_yahoo_data()
        if self.raw_data is None:
            return None
        
        # Step 2: Process data into different frequencies
        self.processed_data = self._process_data_multiple_frequencies(frequency)
        
        # Step 3: CHSH Bell inequality tests
        chsh_results = self._perform_chsh_analysis()
        
        # Step 4: S1 Conditional Bell tests (like your WDRS version)
        conditional_results = self._perform_conditional_bell_analysis()
        
        # Step 5: Cross-variable Mandelbrot metrics
        mandelbrot_results = self._calculate_cross_mandelbrot_metrics()
        
        # Step 6: Visualizations
        self._create_comprehensive_visualizations()
        
        # Step 7: Summary and recommendations
        self._provide_analysis_summary()
        
        return {
            'raw_data': self.raw_data,
            'processed_data': self.processed_data,
            'chsh_results': chsh_results,
            'conditional_results': conditional_results,
            'mandelbrot_results': mandelbrot_results
        }
    
    def _download_yahoo_data(self):
        """Download data from Yahoo Finance with error handling"""
        
        print(f"\n📥 DOWNLOADING YAHOO FINANCE DATA")
        print("=" * 40)
        
        all_data = {}
        successful_downloads = []
        failed_downloads = []
        
        for asset in self.assets:
            try:
                print(f"   Downloading {asset}...")
                ticker = yf.Ticker(asset)
                
                # Get both daily and intraday data
                daily_data = ticker.history(period=self.period, interval='1d')
                hourly_data = ticker.history(period='60d', interval='1h')  # Max 60 days for hourly
                
                if len(daily_data) > 0 and len(hourly_data) > 0:
                    all_data[asset] = {
                        'daily': daily_data,
                        'hourly': hourly_data,
                        'info': ticker.info
                    }
                    successful_downloads.append(asset)
                    print(f"   ✅ {asset}: {len(daily_data)} daily, {len(hourly_data)} hourly bars")
                else:
                    failed_downloads.append(asset)
                    print(f"   ❌ {asset}: No data received")
                    
            except Exception as e:
                failed_downloads.append(asset)
                print(f"   ❌ {asset}: Error - {e}")
        
        if not successful_downloads:
            print("❌ No data downloaded successfully")
            return None
        
        if failed_downloads:
            print(f"\n⚠️  Failed downloads: {failed_downloads}")
            print(f"✅ Proceeding with: {successful_downloads}")
            self.assets = successful_downloads
        
        print(f"\n✅ Data download complete: {len(successful_downloads)} assets")
        return all_data
    
    def _process_data_multiple_frequencies(self, primary_freq='1h'):
        """Process data into multiple time frequencies for analysis"""
        
        print(f"\n🔄 PROCESSING DATA - PRIMARY FREQUENCY: {primary_freq}")
        print("=" * 50)
        
        processed = {}
        
        # Define frequency mappings
        freq_configs = {
            '1h': {'source': 'hourly', 'resample': None},
            '4h': {'source': 'hourly', 'resample': '4H'},
            '1d': {'source': 'daily', 'resample': None},
            '15min': {'source': 'hourly', 'resample': '15T'}  # If available
        }
        
        for freq_name, config in freq_configs.items():
            print(f"\n📊 Processing {freq_name} frequency...")
            freq_data = {}
            
            for asset in self.assets:
                try:
                    source_data = self.raw_data[asset][config['source']]
                    
                    if config['resample']:
                        # Resample to target frequency
                        resampled = source_data.resample(config['resample']).agg({
                            'Open': 'first',
                            'High': 'max',
                            'Low': 'min',
                            'Close': 'last',
                            'Volume': 'sum'
                        }).dropna()
                        processed_bars = resampled
                    else:
                        processed_bars = source_data
                    
                    # Calculate returns and additional metrics
                    processed_bars = processed_bars.copy()
                    processed_bars['Returns'] = processed_bars['Close'].pct_change()
                    processed_bars['LogReturns'] = np.log(processed_bars['Close'] / processed_bars['Close'].shift(1))
                    processed_bars['HL_Ratio'] = (processed_bars['High'] - processed_bars['Low']) / processed_bars['Close']
                    processed_bars['Volume_MA'] = processed_bars['Volume'].rolling(20).mean()
                    
                    # Remove NaN values
                    processed_bars = processed_bars.dropna()
                    
                    if len(processed_bars) >= 10:  # Minimum bars required
                        freq_data[asset] = processed_bars
                        print(f"   ✅ {asset}: {len(processed_bars)} bars")
                    else:
                        print(f"   ❌ {asset}: Insufficient data ({len(processed_bars)} bars)")
                        
                except Exception as e:
                    print(f"   ❌ {asset}: Processing error - {e}")
            
            if freq_data:
                processed[freq_name] = freq_data
                print(f"   📈 {freq_name} summary: {len(freq_data)} assets processed")
        
        return processed
    
    def _perform_chsh_analysis(self):
        """Perform CHSH Bell inequality tests on asset pairs"""
        
        print(f"\n🔔 CHSH BELL INEQUALITY ANALYSIS")
        print("=" * 40)
        
        chsh_results = {}
        
        for freq_name, freq_data in self.processed_data.items():
            print(f"\n📊 CHSH Analysis - {freq_name} frequency")
            
            freq_results = {}
            asset_pairs = list(combinations(freq_data.keys(), 2))
            
            for asset1, asset2 in asset_pairs:
                try:
                    data1 = freq_data[asset1]['Returns'].dropna()
                    data2 = freq_data[asset2]['Returns'].dropna()
                    
                    # Align data by index
                    aligned_data = pd.DataFrame({
                        asset1: data1,
                        asset2: data2
                    }).dropna()
                    
                    if len(aligned_data) < 20:
                        continue
                    
                    # Calculate CHSH inequality
                    chsh_result = self._calculate_chsh_inequality(
                        aligned_data[asset1].values,
                        aligned_data[asset2].values
                    )
                    
                    if chsh_result:
                        freq_results[f"{asset1}-{asset2}"] = chsh_result
                        
                        violation = "VIOLATION" if chsh_result['S'] > 2 else "classical"
                        print(f"   {asset1}-{asset2}: S = {chsh_result['S']:.4f} ({violation})")
                
                except Exception as e:
                    print(f"   ❌ {asset1}-{asset2}: Error - {e}")
            
            chsh_results[freq_name] = freq_results
            print(f"   📈 Completed {len(freq_results)} pair analyses")
        
        return chsh_results
    
    def _calculate_chsh_inequality(self, data1, data2, n_bins=3):
        """Calculate CHSH inequality parameter S"""
        
        # Convert to binary outcomes based on quantiles
        threshold1_low = np.quantile(data1, 0.33)
        threshold1_high = np.quantile(data1, 0.67)
        threshold2_low = np.quantile(data2, 0.33)
        threshold2_high = np.quantile(data2, 0.67)
        
        # Create binary measurements A, A', B, B'
        A = (data1 > threshold1_low).astype(int)
        A_prime = (data1 > threshold1_high).astype(int)
        B = (data2 > threshold2_low).astype(int)
        B_prime = (data2 > threshold2_high).astype(int)
        
        # Calculate correlations E(AB), E(AB'), E(A'B), E(A'B')
        E_AB = np.corrcoef(A, B)[0, 1] if len(set(A)) > 1 and len(set(B)) > 1 else 0
        E_AB_prime = np.corrcoef(A, B_prime)[0, 1] if len(set(A)) > 1 and len(set(B_prime)) > 1 else 0
        E_A_prime_B = np.corrcoef(A_prime, B)[0, 1] if len(set(A_prime)) > 1 and len(set(B)) > 1 else 0
        E_A_prime_B_prime = np.corrcoef(A_prime, B_prime)[0, 1] if len(set(A_prime)) > 1 and len(set(B_prime)) > 1 else 0
        
        # CHSH parameter S
        S = abs(E_AB + E_AB_prime) + abs(E_A_prime_B - E_A_prime_B_prime)
        
        return {
            'S': S,
            'E_AB': E_AB,
            'E_AB_prime': E_AB_prime,
            'E_A_prime_B': E_A_prime_B,
            'E_A_prime_B_prime': E_A_prime_B_prime,
            'violation': S > 2,
            'data_points': len(data1)
        }
    
    def _perform_conditional_bell_analysis(self, window_size=20, threshold_quantile=0.75):
        """Perform S1 conditional Bell inequality tests (like WDRS version)"""
        
        print(f"\n🎯 S1 CONDITIONAL BELL INEQUALITY ANALYSIS")
        print("=" * 45)
        print(f"Window size: {window_size}, Threshold quantile: {threshold_quantile}")
        
        conditional_results = {}
        
        for freq_name, freq_data in self.processed_data.items():
            print(f"\n📊 S1 Analysis - {freq_name} frequency")
            
            freq_results = {}
            asset_pairs = list(combinations(freq_data.keys(), 2))
            
            for asset1, asset2 in asset_pairs:
                try:
                    data1 = freq_data[asset1]
                    data2 = freq_data[asset2]
                    
                    # Align data
                    aligned_data = pd.DataFrame({
                        f'{asset1}_returns': data1['Returns'],
                        f'{asset2}_returns': data2['Returns'],
                        f'{asset1}_volume': data1['Volume'],
                        f'{asset2}_volume': data2['Volume']
                    }).dropna()
                    
                    if len(aligned_data) < window_size * 2:
                        continue
                    
                    # Calculate conditional Bell inequality
                    s1_result = self._calculate_s1_conditional_bell(
                        aligned_data, asset1, asset2, window_size, threshold_quantile
                    )
                    
                    if s1_result:
                        freq_results[f"{asset1}-{asset2}"] = s1_result
                        
                        violation = "VIOLATION" if s1_result['violation'] else "classical"
                        print(f"   {asset1}-{asset2}: S1 = {s1_result['S1']:.4f} ({violation})")
                        
                        # Show regime breakdown for first few pairs (detailed output)
                        if len(freq_results) <= 3:
                            print(f"      Regimes: (0,0)={s1_result['E_AB_00']:.3f}, (0,1)={s1_result['E_AB_01']:.3f}, (1,0)={s1_result['E_AB_10']:.3f}, (1,1)={s1_result['E_AB_11']:.3f}")
                            print(f"      Data points: {s1_result['regime_counts']}")
                
                except Exception as e:
                    print(f"   ❌ {asset1}-{asset2}: Error - {e}")
            
            conditional_results[freq_name] = freq_results
            print(f"   📈 Completed {len(freq_results)} conditional analyses")
        
        return conditional_results
    
    def _calculate_s1_conditional_bell(self, data, asset1, asset2, window_size, threshold_quantile):
        """Calculate S1 conditional Bell inequality following Zarifian et al. (2025)
        
        Formula: S1 = E[AB|x₀,y₀] + E[AB|x₀,y₁] + E[AB|x₁,y₀] - E[AB|x₁,y₁]
        Where x,y are market regime indicators (0=low, 1=high)
        """
        
        # Calculate rolling volatility for regime detection
        data[f'{asset1}_vol'] = data[f'{asset1}_returns'].rolling(window_size).std()
        data[f'{asset2}_vol'] = data[f'{asset2}_returns'].rolling(window_size).std()
        
        # Remove NaN values from rolling calculations
        clean_data = data.dropna()
        
        if len(clean_data) < 40:  # Need sufficient data for regime analysis
            return None
        
        # Define market regimes based on volatility thresholds
        vol_threshold1 = clean_data[f'{asset1}_vol'].quantile(0.5)  # Median split
        vol_threshold2 = clean_data[f'{asset2}_vol'].quantile(0.5)  # Median split
        
        # Market regime indicators (0=low vol, 1=high vol)
        x = (clean_data[f'{asset1}_vol'] > vol_threshold1).astype(int)  # Asset 1 regime
        y = (clean_data[f'{asset2}_vol'] > vol_threshold2).astype(int)  # Asset 2 regime
        
        # Binary measurement outcomes (same AB measurement in all regimes)
        A = (clean_data[f'{asset1}_returns'] > 0).astype(int)  # Asset 1 positive return
        B = (clean_data[f'{asset2}_returns'] > 0).astype(int)  # Asset 2 positive return
        
        # Calculate conditional expectations E[AB|x,y] for each regime combination
        try:
            # Regime (0,0): Both assets in low volatility
            mask_00 = (x == 0) & (y == 0)
            if mask_00.sum() > 5:
                E_AB_00 = np.mean(A[mask_00] * B[mask_00])
            else:
                E_AB_00 = 0
            
            # Regime (0,1): Asset 1 low vol, Asset 2 high vol
            mask_01 = (x == 0) & (y == 1)
            if mask_01.sum() > 5:
                E_AB_01 = np.mean(A[mask_01] * B[mask_01])
            else:
                E_AB_01 = 0
            
            # Regime (1,0): Asset 1 high vol, Asset 2 low vol
            mask_10 = (x == 1) & (y == 0)
            if mask_10.sum() > 5:
                E_AB_10 = np.mean(A[mask_10] * B[mask_10])
            else:
                E_AB_10 = 0
            
            # Regime (1,1): Both assets in high volatility
            mask_11 = (x == 1) & (y == 1)
            if mask_11.sum() > 5:
                E_AB_11 = np.mean(A[mask_11] * B[mask_11])
            else:
                E_AB_11 = 0
            
            # S1 conditional Bell inequality (Zarifian et al. 2025)
            S1 = E_AB_00 + E_AB_01 + E_AB_10 - E_AB_11
            
            # Count data points in each regime
            regime_counts = {
                '(0,0)': mask_00.sum(),
                '(0,1)': mask_01.sum(), 
                '(1,0)': mask_10.sum(),
                '(1,1)': mask_11.sum()
            }
            
            return {
                'S1': S1,
                'E_AB_00': E_AB_00,  # E[AB|x₀,y₀]
                'E_AB_01': E_AB_01,  # E[AB|x₀,y₁]
                'E_AB_10': E_AB_10,  # E[AB|x₁,y₀]
                'E_AB_11': E_AB_11,  # E[AB|x₁,y₁]
                'violation': abs(S1) > 2,  # Bell violation threshold
                'regime_counts': regime_counts,
                'total_data_points': len(clean_data),
                'vol_threshold1': vol_threshold1,
                'vol_threshold2': vol_threshold2
            }
            
        except Exception as e:
            print(f"      Error in S1 calculation: {e}")
            return None
    
    def _calculate_cross_mandelbrot_metrics(self):
        """Calculate Mandelbrot metrics BETWEEN variables/time series"""
        
        print(f"\n🌀 CROSS-VARIABLE MANDELBROT METRICS")
        print("=" * 40)
        
        mandelbrot_results = {}
        
        for freq_name, freq_data in self.processed_data.items():
            print(f"\n📊 Mandelbrot Analysis - {freq_name} frequency")
            
            freq_results = {}
            asset_pairs = list(combinations(freq_data.keys(), 2))
            
            for asset1, asset2 in asset_pairs:
                try:
                    returns1 = freq_data[asset1]['Returns'].dropna()
                    returns2 = freq_data[asset2]['Returns'].dropna()
                    
                    # Align data
                    aligned_data = pd.DataFrame({
                        asset1: returns1,
                        asset2: returns2
                    }).dropna()
                    
                    if len(aligned_data) < 50:
                        continue
                    
                    # Calculate cross-Mandelbrot metrics
                    cross_metrics = self._calculate_cross_mandelbrot_pair(
                        aligned_data[asset1].values,
                        aligned_data[asset2].values
                    )
                    
                    if cross_metrics:
                        freq_results[f"{asset1}-{asset2}"] = cross_metrics
                        print(f"   {asset1}-{asset2}: H_cross = {cross_metrics['hurst_cross']:.3f}")
                
                except Exception as e:
                    print(f"   ❌ {asset1}-{asset2}: Error - {e}")
            
            mandelbrot_results[freq_name] = freq_results
            print(f"   📈 Completed {len(freq_results)} cross-Mandelbrot analyses")
        
        return mandelbrot_results
    
    def _calculate_cross_mandelbrot_pair(self, series1, series2):
        """Calculate cross-variable Mandelbrot metrics between two time series"""
        
        # Cross-correlation at different lags
        max_lag = min(20, len(series1) // 4)
        cross_corrs = []
        
        for lag in range(max_lag):
            if lag == 0:
                corr = np.corrcoef(series1, series2)[0, 1]
            else:
                corr = np.corrcoef(series1[:-lag], series2[lag:])[0, 1]
            cross_corrs.append(abs(corr))
        
        # Cross-Hurst exponent using R/S analysis on cross-correlation
        cross_hurst = self._calculate_hurst_exponent(np.array(cross_corrs))
        
        # Cross-volatility clustering
        vol1 = pd.Series(series1).rolling(5).std().dropna()
        vol2 = pd.Series(series2).rolling(5).std().dropna()
        
        if len(vol1) > 10 and len(vol2) > 10:
            cross_vol_corr = np.corrcoef(vol1, vol2)[0, 1]
        else:
            cross_vol_corr = 0
        
        # Multifractal cross-correlation
        cross_mf_spectrum = self._calculate_cross_multifractal_spectrum(series1, series2)
        
        return {
            'hurst_cross': cross_hurst,
            'cross_correlation_max': max(cross_corrs),
            'cross_correlation_decay': cross_corrs[0] - cross_corrs[-1] if len(cross_corrs) > 1 else 0,
            'cross_volatility_correlation': cross_vol_corr,
            'multifractal_width': cross_mf_spectrum['width'],
            'multifractal_asymmetry': cross_mf_spectrum['asymmetry']
        }
    
    def _calculate_hurst_exponent(self, series):
        """Calculate Hurst exponent using R/S analysis"""
        
        if len(series) < 10:
            return 0.5
        
        # R/S analysis
        lags = range(2, min(len(series) // 2, 50))
        rs_values = []
        
        for lag in lags:
            # Divide series into non-overlapping windows
            n_windows = len(series) // lag
            if n_windows < 2:
                continue
                
            rs_window = []
            for i in range(n_windows):
                window = series[i*lag:(i+1)*lag]
                if len(window) < lag:
                    continue
                    
                # Calculate R/S for this window
                mean_window = np.mean(window)
                deviations = np.cumsum(window - mean_window)
                R = np.max(deviations) - np.min(deviations)
                S = np.std(window)
                
                if S > 0:
                    rs_window.append(R / S)
            
            if rs_window:
                rs_values.append(np.mean(rs_window))
        
        if len(rs_values) < 3:
            return 0.5
        
        # Fit log(R/S) vs log(lag)
        log_lags = np.log(lags[:len(rs_values)])
        log_rs = np.log(rs_values)
        
        try:
            hurst, _ = np.polyfit(log_lags, log_rs, 1)
            return max(0, min(1, hurst))  # Clamp to [0, 1]
        except:
            return 0.5
    
    def _calculate_cross_multifractal_spectrum(self, series1, series2):
        """Calculate cross-multifractal spectrum between two series"""
        
        # Simplified multifractal analysis
        q_values = np.linspace(-5, 5, 21)
        tau_q = []
        
        for q in q_values:
            # Cross-correlation based multifractal
            cross_series = np.abs(series1 * series2)  # Simple cross-product
            
            if len(cross_series) < 20:
                tau_q.append(0)
                continue
            
            # Partition function
            scales = [2, 4, 8, 16, min(32, len(cross_series)//4)]
            log_scales = []
            log_partitions = []
            
            for scale in scales:
                if scale >= len(cross_series):
                    continue
                    
                n_boxes = len(cross_series) // scale
                partition_sum = 0
                
                for i in range(n_boxes):
                    box_sum = np.sum(np.abs(cross_series[i*scale:(i+1)*scale]))
                    if box_sum > 0:
                        partition_sum += box_sum ** q
                
                if partition_sum > 0:
                    log_scales.append(np.log(scale))
                    log_partitions.append(np.log(partition_sum))
            
            if len(log_scales) > 1:
                try:
                    slope, _ = np.polyfit(log_scales, log_partitions, 1)
                    tau_q.append(slope)
                except:
                    tau_q.append(0)
            else:
                tau_q.append(0)
        
        # Calculate multifractal spectrum width and asymmetry
        tau_q = np.array(tau_q)
        valid_tau = tau_q[np.isfinite(tau_q)]
        
        if len(valid_tau) > 5:
            width = np.max(valid_tau) - np.min(valid_tau)
            asymmetry = np.mean(valid_tau[len(valid_tau)//2:]) - np.mean(valid_tau[:len(valid_tau)//2])
        else:
            width = 0
            asymmetry = 0
        
        return {
            'width': width,
            'asymmetry': asymmetry,
            'tau_q': tau_q.tolist()
        }
    
    def _create_comprehensive_visualizations(self):
        """Create comprehensive visualizations of all analyses"""
        
        print(f"\n📊 CREATING COMPREHENSIVE VISUALIZATIONS")
        print("=" * 45)
        
        try:
            # Set matplotlib to non-interactive backend to avoid hanging
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            
            # Set up the plotting style
            plt.style.use('default')
            sns.set_palette("husl")
            
            # Create main figure with subplots
            fig = plt.figure(figsize=(20, 16))
            
            # 1. Price evolution plot
            ax1 = plt.subplot(3, 3, 1)
            self._plot_price_evolution(ax1)
            
            # 2. Return distributions
            ax2 = plt.subplot(3, 3, 2)
            self._plot_return_distributions(ax2)
            
            # 3. CHSH violations heatmap
            ax3 = plt.subplot(3, 3, 3)
            self._plot_chsh_heatmap(ax3)
            
            # 4. S1 conditional violations
            ax4 = plt.subplot(3, 3, 4)
            self._plot_s1_violations(ax4)
            
            # 5. Cross-Hurst exponents
            ax5 = plt.subplot(3, 3, 5)
            self._plot_cross_hurst_heatmap(ax5)
            
            # 6. Correlation matrix
            ax6 = plt.subplot(3, 3, 6)
            self._plot_correlation_matrix(ax6)
            
            # 7. Volume analysis
            ax7 = plt.subplot(3, 3, 7)
            self._plot_volume_analysis(ax7)
            
            # 8. Bell violation timeline
            ax8 = plt.subplot(3, 3, 8)
            self._plot_bell_timeline(ax8)
            
            # 9. Multifractal spectrum
            ax9 = plt.subplot(3, 3, 9)
            self._plot_multifractal_spectrum(ax9)
            
            plt.tight_layout()
            plt.savefig('yahoo_finance_bell_analysis.png', dpi=300, bbox_inches='tight')
            plt.close(fig)  # Close figure to free memory
            
            print("✅ Comprehensive visualization saved as 'yahoo_finance_bell_analysis.png'")
            
        except Exception as e:
            print(f"⚠️  Visualization creation encountered an issue: {e}")
            print("✅ Analysis data is still available, continuing without plots...")
    
    def _plot_price_evolution(self, ax):
        """Plot price evolution for all assets"""
        
        if '1d' in self.processed_data:
            daily_data = self.processed_data['1d']
            
            for asset in self.assets:
                if asset in daily_data:
                    prices = daily_data[asset]['Close']
                    normalized_prices = prices / prices.iloc[0]  # Normalize to start at 1
                    ax.plot(normalized_prices.index, normalized_prices, label=asset, linewidth=2)
        
        ax.set_title('Normalized Price Evolution', fontsize=12, fontweight='bold')
        ax.set_ylabel('Normalized Price')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    
    def _plot_return_distributions(self, ax):
        """Plot return distributions"""
        
        if '1d' in self.processed_data:
            daily_data = self.processed_data['1d']
            
            for asset in self.assets[:4]:  # Limit to first 4 for readability
                if asset in daily_data:
                    returns = daily_data[asset]['Returns'].dropna()
                    ax.hist(returns, bins=30, alpha=0.6, label=asset, density=True)
        
        ax.set_title('Return Distributions', fontsize=12, fontweight='bold')
        ax.set_xlabel('Daily Returns')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_chsh_heatmap(self, ax):
        """Plot CHSH violation heatmap"""
        
        if hasattr(self, 'bell_results') and '1d' in self.bell_results:
            chsh_data = self.bell_results['1d']
            
            # Create matrix of CHSH values
            assets = list(set([pair.split('-')[0] for pair in chsh_data.keys()] + 
                             [pair.split('-')[1] for pair in chsh_data.keys()]))
            
            chsh_matrix = np.zeros((len(assets), len(assets)))
            
            for pair, result in chsh_data.items():
                asset1, asset2 = pair.split('-')
                if asset1 in assets and asset2 in assets:
                    i, j = assets.index(asset1), assets.index(asset2)
                    chsh_matrix[i, j] = result['S']
                    chsh_matrix[j, i] = result['S']
            
            im = ax.imshow(chsh_matrix, cmap='RdYlBu_r', vmin=0, vmax=3)
            ax.set_xticks(range(len(assets)))
            ax.set_yticks(range(len(assets)))
            ax.set_xticklabels(assets, rotation=45)
            ax.set_yticklabels(assets)
            
            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        ax.set_title('CHSH Values Heatmap', fontsize=12, fontweight='bold')
    
    def _plot_s1_violations(self, ax):
        """Plot S1 conditional violations"""
        
        # Placeholder for S1 violations plot
        ax.text(0.5, 0.5, 'S1 Conditional\nViolations\n(Implementation pending)', 
                ha='center', va='center', transform=ax.transAxes, fontsize=10)
        ax.set_title('S1 Conditional Bell Violations', fontsize=12, fontweight='bold')
    
    def _plot_cross_hurst_heatmap(self, ax):
        """Plot cross-Hurst exponent heatmap"""
        
        # Placeholder for cross-Hurst heatmap
        ax.text(0.5, 0.5, 'Cross-Hurst\nExponents\n(Implementation pending)', 
                ha='center', va='center', transform=ax.transAxes, fontsize=10)
        ax.set_title('Cross-Hurst Exponents', fontsize=12, fontweight='bold')
    
    def _plot_correlation_matrix(self, ax):
        """Plot correlation matrix"""
        
        if '1d' in self.processed_data:
            daily_data = self.processed_data['1d']
            
            # Create correlation matrix
            returns_data = {}
            for asset in self.assets:
                if asset in daily_data:
                    returns_data[asset] = daily_data[asset]['Returns']
            
            if returns_data:
                corr_df = pd.DataFrame(returns_data).corr()
                
                im = ax.imshow(corr_df.values, cmap='RdBu_r', vmin=-1, vmax=1)
                ax.set_xticks(range(len(corr_df.columns)))
                ax.set_yticks(range(len(corr_df.columns)))
                ax.set_xticklabels(corr_df.columns, rotation=45)
                ax.set_yticklabels(corr_df.columns)
                
                # Add correlation values
                for i in range(len(corr_df.columns)):
                    for j in range(len(corr_df.columns)):
                        ax.text(j, i, f'{corr_df.iloc[i, j]:.2f}', 
                               ha='center', va='center', fontsize=8)
                
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        ax.set_title('Return Correlations', fontsize=12, fontweight='bold')
    
    def _plot_volume_analysis(self, ax):
        """Plot volume analysis"""
        
        # Placeholder for volume analysis
        ax.text(0.5, 0.5, 'Volume Analysis\n(Implementation pending)', 
                ha='center', va='center', transform=ax.transAxes, fontsize=10)
        ax.set_title('Volume Analysis', fontsize=12, fontweight='bold')
    
    def _plot_bell_timeline(self, ax):
        """Plot Bell violation timeline"""
        
        # Placeholder for Bell timeline
        ax.text(0.5, 0.5, 'Bell Violation\nTimeline\n(Implementation pending)', 
                ha='center', va='center', transform=ax.transAxes, fontsize=10)
        ax.set_title('Bell Violation Timeline', fontsize=12, fontweight='bold')
    
    def _plot_multifractal_spectrum(self, ax):
        """Plot multifractal spectrum"""
        
        # Placeholder for multifractal spectrum
        ax.text(0.5, 0.5, 'Multifractal\nSpectrum\n(Implementation pending)', 
                ha='center', va='center', transform=ax.transAxes, fontsize=10)
        ax.set_title('Cross-Multifractal Spectrum', fontsize=12, fontweight='bold')
    
    def _provide_analysis_summary(self):
        """Provide comprehensive analysis summary and recommendations"""
        
        print(f"\n📋 YAHOO FINANCE BELL ANALYSIS SUMMARY")
        print("=" * 50)
        
        print(f"🎯 Assets analyzed: {len(self.assets)}")
        print(f"📊 Data period: {self.period}")
        print(f"⏱️  Frequencies: {list(self.processed_data.keys())}")
        
        # CHSH violations summary
        if hasattr(self, 'bell_results'):
            total_violations = 0
            total_tests = 0
            
            for freq_results in self.bell_results.values():
                for result in freq_results.values():
                    total_tests += 1
                    if result.get('violation', False):
                        total_violations += 1
            
            if total_tests > 0:
                violation_rate = (total_violations / total_tests) * 100
                print(f"🔔 CHSH violations: {total_violations}/{total_tests} ({violation_rate:.1f}%)")
        
        print(f"\n💡 RECOMMENDATIONS FOR WDRS DATA SELECTION:")
        print("   Based on this Yahoo Finance analysis, focus on:")
        
        # Identify most interesting pairs
        if hasattr(self, 'bell_results') and self.bell_results:
            print("   🎯 Asset pairs with highest Bell violations")
            print("   📊 Time frequencies showing strongest effects")
            print("   🌀 Cross-Mandelbrot patterns for deeper analysis")
        
        print(f"\n✅ Analysis complete! Check visualizations for detailed insights.")

# =================== MAIN EXECUTION FUNCTION ===================

def run_yahoo_finance_bell_analysis(assets=None, period='6mo', frequency='1h'):
    """
    Main function to run Yahoo Finance Bell inequality analysis
    
    Parameters:
    - assets: List of stock symbols (default: tech stocks)
    - period: Data period ('6mo', '1y', '2y', etc.)
    - frequency: Primary analysis frequency ('1h', '1d', '4h')
    """
    
    print("🚀 STARTING YAHOO FINANCE BELL INEQUALITY ANALYSIS")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = YahooFinanceBellAnalyzer(assets=assets, period=period)
    
    # Run complete analysis
    results = analyzer.run_complete_analysis(frequency=frequency)
    
    if results:
        print(f"\n✅ ANALYSIS COMPLETE!")
        print(f"   Check 'yahoo_finance_bell_analysis.png' for visualizations")
        return analyzer, results
    else:
        print(f"\n❌ Analysis failed")
        return None, None

# Example usage
if __name__ == "__main__":
    # Run with default tech stocks
    analyzer, results = run_yahoo_finance_bell_analysis()
    
    # Or run with custom assets
    # analyzer, results = run_yahoo_finance_bell_analysis(
    #     assets=['AAPL', 'MSFT', 'GOOGL', 'TSLA'],
    #     period='6mo',
    #     frequency='1h'
    # )