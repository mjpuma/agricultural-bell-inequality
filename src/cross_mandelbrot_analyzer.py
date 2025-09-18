#!/usr/bin/env python3
"""
CROSS-MANDELBROT FRACTAL ANALYZER
=================================

This module implements cross-variable Mandelbrot/fractal analysis for financial
time series. Unlike traditional Mandelbrot analysis that examines individual
time series, this focuses on fractal relationships BETWEEN multiple time series.

Key Features:
- Cross-Hurst exponent calculation
- Cross-correlation decay analysis
- Lead-lag relationship detection
- Cross-volatility clustering
- Cross-multifractal spectrum analysis
- Network-level fractal properties

Author: Enhanced from original cross-Mandelbrot implementation
Date: August 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class CrossMandelbrotAnalyzer:
    """
    Analyzer for cross-variable Mandelbrot/fractal properties between time series.
    
    This class focuses on fractal relationships between different time series
    rather than individual time series properties. It's particularly useful
    for understanding how different financial assets exhibit correlated
    fractal behavior.
    """
    
    def __init__(self):
        """Initialize the Cross-Mandelbrot analyzer."""
        self.cross_results = {}
        self.network_metrics = {}
        
        print("🌀 Cross-Mandelbrot Analyzer Initialized")
        print("   Focus: Fractal relationships BETWEEN time series")
    
    def analyze_cross_mandelbrot_comprehensive(self, data_dict, focus_pairs=None):
        """
        Comprehensive cross-Mandelbrot analysis between time series.
        
        Parameters:
        -----------
        data_dict : dict
            Dictionary of {asset_name: pandas.Series of returns}
        focus_pairs : list of tuples, optional
            Specific pairs to analyze. If None, analyzes all combinations.
            
        Returns:
        --------
        dict : Comprehensive cross-Mandelbrot results
        """
        print(f"\n🌀 COMPREHENSIVE CROSS-MANDELBROT ANALYSIS")
        print(f"=" * 50)
        
        # Generate pairs to analyze
        if focus_pairs:
            asset_pairs = focus_pairs
            print(f"📊 Analyzing {len(asset_pairs)} specified asset pairs")
        else:
            assets = list(data_dict.keys())
            asset_pairs = list(combinations(assets, 2))
            print(f"📊 Analyzing all {len(asset_pairs)} possible asset pairs")
        
        # Analyze each pair
        for asset1, asset2 in asset_pairs:
            if asset1 not in data_dict or asset2 not in data_dict:
                print(f"⚠️  Skipping {asset1}-{asset2}: Data not available")
                continue
            
            print(f"\n🔍 Analyzing {asset1} ↔ {asset2}")
            
            # Get and align data
            series1 = data_dict[asset1].dropna()
            series2 = data_dict[asset2].dropna()
            
            aligned_data = pd.DataFrame({
                asset1: series1,
                asset2: series2
            }).dropna()
            
            if len(aligned_data) < 50:
                print(f"   ❌ Insufficient aligned data: {len(aligned_data)} points")
                continue
            
            # Calculate comprehensive cross-Mandelbrot metrics
            cross_metrics = self._calculate_comprehensive_cross_metrics(
                aligned_data[asset1].values,
                aligned_data[asset2].values,
                asset1, asset2
            )
            
            pair_key = f"{asset1}-{asset2}"
            self.cross_results[pair_key] = cross_metrics
            
            # Print key results
            print(f"   ✅ Cross-Hurst: {cross_metrics['cross_hurst']:.3f}")
            print(f"   📊 Cross-correlation decay: {cross_metrics['cross_correlation_decay']:.3f}")
            print(f"   🌊 Cross-volatility clustering: {cross_metrics['cross_volatility_clustering']:.3f}")
            print(f"   🔄 Lead-lag strength: {cross_metrics['lead_lag_strength']:.3f}")
        
        # Calculate network-level metrics
        self._calculate_network_mandelbrot_metrics()
        
        # Create visualizations
        self._create_cross_mandelbrot_visualizations()
        
        return self.cross_results
    
    def _calculate_comprehensive_cross_metrics(self, series1, series2, name1, name2):
        """
        Calculate comprehensive cross-Mandelbrot metrics between two series.
        
        Parameters:
        -----------
        series1, series2 : np.array
            Time series data for the two assets
        name1, name2 : str
            Asset names for reference
            
        Returns:
        --------
        dict : Comprehensive cross-Mandelbrot metrics
        """
        metrics = {}
        
        # 1. Cross-Hurst Exponent
        metrics['cross_hurst'] = self._calculate_cross_hurst_exponent(series1, series2)
        
        # 2. Cross-correlation structure
        cross_corr_metrics = self._analyze_cross_correlation_structure(series1, series2)
        metrics.update(cross_corr_metrics)
        
        # 3. Cross-volatility clustering
        metrics['cross_volatility_clustering'] = self._calculate_cross_volatility_clustering(series1, series2)
        
        # 4. Cross-multifractal spectrum
        cross_mf_metrics = self._calculate_cross_multifractal_spectrum(series1, series2)
        metrics.update(cross_mf_metrics)
        
        # 5. Lead-lag relationships
        lead_lag_metrics = self._analyze_lead_lag_relationships(series1, series2)
        metrics.update(lead_lag_metrics)
        
        # 6. Cross-entropy and information flow
        info_metrics = self._calculate_cross_information_metrics(series1, series2)
        metrics.update(info_metrics)
        
        return metrics
    
    def _calculate_cross_hurst_exponent(self, series1, series2):
        """
        Calculate cross-Hurst exponent using cross-correlation R/S analysis.
        
        The cross-Hurst exponent measures the persistence of cross-correlations
        between two time series at different time scales.
        
        Parameters:
        -----------
        series1, series2 : np.array
            Time series data
            
        Returns:
        --------
        float : Cross-Hurst exponent (0 < H < 1)
        """
        # Calculate cross-correlations at different lags
        max_lag = min(50, len(series1) // 4)
        cross_corrs = []
        
        for lag in range(max_lag):
            if lag == 0:
                corr = np.corrcoef(series1, series2)[0, 1]
            else:
                if len(series1) > lag:
                    corr = np.corrcoef(series1[:-lag], series2[lag:])[0, 1]
                else:
                    break
            cross_corrs.append(abs(corr))
        
        if len(cross_corrs) < 5:
            return 0.5
        
        # Apply R/S analysis to cross-correlation decay
        return self._rs_analysis(np.array(cross_corrs))
    
    def _rs_analysis(self, series):
        """
        R/S (Rescaled Range) analysis for Hurst exponent calculation.
        
        Parameters:
        -----------
        series : np.array
            Time series for R/S analysis
            
        Returns:
        --------
        float : Hurst exponent
        """
        if len(series) < 10:
            return 0.5
        
        lags = range(2, min(len(series) // 2, 20))
        rs_values = []
        
        for lag in lags:
            n_windows = len(series) // lag
            if n_windows < 2:
                continue
            
            rs_window = []
            for i in range(n_windows):
                window = series[i*lag:(i+1)*lag]
                if len(window) < lag:
                    continue
                
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
    
    def _analyze_cross_correlation_structure(self, series1, series2):
        """
        Analyze the structure of cross-correlations at different lags.
        
        Parameters:
        -----------
        series1, series2 : np.array
            Time series data
            
        Returns:
        --------
        dict : Cross-correlation structure metrics
        """
        max_lag = min(20, len(series1) // 4)
        cross_corrs = []
        lags = []
        
        for lag in range(-max_lag, max_lag + 1):
            if lag == 0:
                corr = np.corrcoef(series1, series2)[0, 1]
            elif lag > 0:
                if len(series1) > lag:
                    corr = np.corrcoef(series1[:-lag], series2[lag:])[0, 1]
                else:
                    continue
            else:  # lag < 0
                if len(series2) > abs(lag):
                    corr = np.corrcoef(series1[abs(lag):], series2[:lag])[0, 1]
                else:
                    continue
            
            cross_corrs.append(corr)
            lags.append(lag)
        
        cross_corrs = np.array(cross_corrs)
        
        return {
            'cross_correlation_max': np.max(np.abs(cross_corrs)),
            'cross_correlation_at_zero': cross_corrs[len(cross_corrs)//2] if len(cross_corrs) > 0 else 0,
            'cross_correlation_decay': self._calculate_correlation_decay(cross_corrs),
            'cross_correlation_asymmetry': self._calculate_correlation_asymmetry(cross_corrs, lags)
        }
    
    def _calculate_correlation_decay(self, cross_corrs):
        """Calculate how quickly cross-correlations decay with lag."""
        if len(cross_corrs) < 5:
            return 0
        
        abs_corrs = np.abs(cross_corrs)
        center_idx = len(abs_corrs) // 2
        
        # Take correlations from center outward
        decay_corrs = []
        for i in range(min(10, center_idx)):
            if center_idx + i < len(abs_corrs):
                decay_corrs.append(abs_corrs[center_idx + i])
        
        if len(decay_corrs) < 3:
            return 0
        
        # Calculate decay rate
        x = np.arange(len(decay_corrs))
        try:
            log_corrs = np.log(np.maximum(decay_corrs, 1e-10))
            decay_rate, _ = np.polyfit(x, log_corrs, 1)
            return abs(decay_rate)
        except:
            return 0
    
    def _calculate_correlation_asymmetry(self, cross_corrs, lags):
        """Calculate asymmetry in cross-correlations (lead-lag effects)."""
        if len(cross_corrs) != len(lags):
            return 0
        
        center_idx = len(lags) // 2
        positive_lag_corrs = cross_corrs[center_idx+1:]
        negative_lag_corrs = cross_corrs[:center_idx]
        
        if len(positive_lag_corrs) == 0 or len(negative_lag_corrs) == 0:
            return 0
        
        pos_mean = np.mean(np.abs(positive_lag_corrs))
        neg_mean = np.mean(np.abs(negative_lag_corrs))
        
        if pos_mean + neg_mean > 0:
            return (pos_mean - neg_mean) / (pos_mean + neg_mean)
        else:
            return 0
    
    def _calculate_cross_volatility_clustering(self, series1, series2):
        """
        Calculate cross-volatility clustering between two series.
        
        This measures how volatility in one series correlates with
        volatility in another series.
        """
        window = min(10, len(series1) // 5)
        if window < 3:
            return 0
        
        vol1 = pd.Series(series1).rolling(window).std().dropna()
        vol2 = pd.Series(series2).rolling(window).std().dropna()
        
        if len(vol1) < 10 or len(vol2) < 10:
            return 0
        
        min_len = min(len(vol1), len(vol2))
        vol1_aligned = vol1.iloc[:min_len]
        vol2_aligned = vol2.iloc[:min_len]
        
        try:
            cross_vol_corr = np.corrcoef(vol1_aligned, vol2_aligned)[0, 1]
            return cross_vol_corr if not np.isnan(cross_vol_corr) else 0
        except:
            return 0
    
    def _calculate_cross_multifractal_spectrum(self, series1, series2):
        """Calculate cross-multifractal spectrum between two series."""
        # Create cross-series for multifractal analysis
        cross_series = series1 * series2
        abs_cross_series = np.abs(cross_series)
        
        q_values = np.linspace(-3, 3, 13)
        tau_q = []
        
        for q in q_values:
            scales = [2, 4, 8, 16, min(32, len(abs_cross_series)//4)]
            valid_scales = [s for s in scales if s < len(abs_cross_series)]
            
            if len(valid_scales) < 3:
                tau_q.append(0)
                continue
            
            log_scales = []
            log_partitions = []
            
            for scale in valid_scales:
                n_boxes = len(abs_cross_series) // scale
                if n_boxes < 2:
                    continue
                
                partition_sum = 0
                for i in range(n_boxes):
                    box_data = abs_cross_series[i*scale:(i+1)*scale]
                    box_sum = np.sum(box_data)
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
        
        tau_q = np.array(tau_q)
        valid_tau = tau_q[np.isfinite(tau_q)]
        
        if len(valid_tau) > 3:
            mf_width = np.max(valid_tau) - np.min(valid_tau)
            mf_asymmetry = np.mean(valid_tau[len(valid_tau)//2:]) - np.mean(valid_tau[:len(valid_tau)//2])
            mf_complexity = np.std(valid_tau)
        else:
            mf_width = 0
            mf_asymmetry = 0
            mf_complexity = 0
        
        return {
            'multifractal_width': mf_width,
            'multifractal_asymmetry': mf_asymmetry,
            'multifractal_complexity': mf_complexity
        }
    
    def _analyze_lead_lag_relationships(self, series1, series2):
        """Analyze lead-lag relationships between two series."""
        max_lag = min(10, len(series1) // 10)
        correlations = []
        lags = []
        
        for lag in range(1, max_lag + 1):
            # Series1 leads Series2
            if len(series1) > lag:
                corr_lead = np.corrcoef(series1[:-lag], series2[lag:])[0, 1]
                correlations.append(abs(corr_lead))
                lags.append(lag)
            
            # Series2 leads Series1
            if len(series2) > lag:
                corr_lag = np.corrcoef(series2[:-lag], series1[lag:])[0, 1]
                correlations.append(abs(corr_lag))
                lags.append(-lag)
        
        if not correlations:
            return {'lead_lag_strength': 0, 'optimal_lag': 0, 'lead_lag_direction': 0}
        
        max_corr_idx = np.argmax(correlations)
        optimal_lag = lags[max_corr_idx]
        max_correlation = correlations[max_corr_idx]
        
        return {
            'lead_lag_strength': max_correlation,
            'optimal_lag': optimal_lag,
            'lead_lag_direction': np.sign(optimal_lag)
        }
    
    def _calculate_cross_information_metrics(self, series1, series2):
        """Calculate cross-information and entropy metrics."""
        n_bins = min(10, int(np.sqrt(len(series1))))
        
        try:
            bins1 = np.histogram_bin_edges(series1, bins=n_bins)
            bins2 = np.histogram_bin_edges(series2, bins=n_bins)
            
            discrete1 = np.digitize(series1, bins1) - 1
            discrete2 = np.digitize(series2, bins2) - 1
            
            discrete1 = np.clip(discrete1, 0, n_bins - 1)
            discrete2 = np.clip(discrete2, 0, n_bins - 1)
            
            # Joint entropy
            joint_hist, _, _ = np.histogram2d(discrete1, discrete2, bins=n_bins)
            joint_prob = joint_hist / np.sum(joint_hist)
            joint_prob = joint_prob[joint_prob > 0]
            joint_entropy = -np.sum(joint_prob * np.log2(joint_prob))
            
            # Individual entropies
            hist1, _ = np.histogram(discrete1, bins=n_bins)
            prob1 = hist1 / np.sum(hist1)
            prob1 = prob1[prob1 > 0]
            entropy1 = -np.sum(prob1 * np.log2(prob1))
            
            hist2, _ = np.histogram(discrete2, bins=n_bins)
            prob2 = hist2 / np.sum(hist2)
            prob2 = prob2[prob2 > 0]
            entropy2 = -np.sum(prob2 * np.log2(prob2))
            
            # Mutual information
            mutual_info = entropy1 + entropy2 - joint_entropy
            normalized_mutual_info = mutual_info / joint_entropy if joint_entropy > 0 else 0
            
            return {
                'cross_mutual_information': mutual_info,
                'normalized_mutual_information': normalized_mutual_info,
                'cross_entropy_ratio': joint_entropy / (entropy1 + entropy2) if (entropy1 + entropy2) > 0 else 0
            }
        
        except:
            return {
                'cross_mutual_information': 0,
                'normalized_mutual_information': 0,
                'cross_entropy_ratio': 0
            }
    
    def _calculate_network_mandelbrot_metrics(self):
        """Calculate network-level Mandelbrot metrics across all pairs."""
        if not self.cross_results:
            return
        
        print(f"\n🕸️  NETWORK-LEVEL MANDELBROT METRICS")
        print("=" * 40)
        
        # Collect metrics
        cross_hurst_values = [result['cross_hurst'] for result in self.cross_results.values()]
        cross_corr_values = [abs(result['cross_correlation_at_zero']) for result in self.cross_results.values()]
        mf_width_values = [result['multifractal_width'] for result in self.cross_results.values()]
        
        self.network_metrics = {
            'mean_cross_hurst': np.mean(cross_hurst_values),
            'std_cross_hurst': np.std(cross_hurst_values),
            'network_correlation_density': np.mean(cross_corr_values),
            'network_multifractal_complexity': np.mean(mf_width_values)
        }
        
        print(f"   📊 Mean cross-Hurst: {self.network_metrics['mean_cross_hurst']:.3f}")
        print(f"   🌐 Network correlation density: {self.network_metrics['network_correlation_density']:.3f}")
        print(f"   🌀 Network multifractal complexity: {self.network_metrics['network_multifractal_complexity']:.3f}")
    
    def _create_cross_mandelbrot_visualizations(self):
        """Create comprehensive visualizations of cross-Mandelbrot analysis."""
        if not self.cross_results:
            print("⚠️  No results to visualize")
            return
        
        print(f"\n📊 CREATING CROSS-MANDELBROT VISUALIZATIONS")
        print("=" * 45)
        
        # Set matplotlib backend
        import matplotlib
        matplotlib.use('Agg')
        
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Cross-Hurst heatmap
        ax1 = plt.subplot(2, 4, 1)
        self._plot_cross_hurst_heatmap(ax1)
        
        # 2. Cross-correlation decay
        ax2 = plt.subplot(2, 4, 2)
        self._plot_cross_correlation_decay(ax2)
        
        # 3. Lead-lag relationships
        ax3 = plt.subplot(2, 4, 3)
        self._plot_lead_lag_relationships(ax3)
        
        # 4. Cross-volatility clustering
        ax4 = plt.subplot(2, 4, 4)
        self._plot_cross_volatility_clustering(ax4)
        
        # 5. Multifractal spectrum comparison
        ax5 = plt.subplot(2, 4, 5)
        self._plot_multifractal_spectrum_comparison(ax5)
        
        # 6. Information flow metrics
        ax6 = plt.subplot(2, 4, 6)
        self._plot_information_flow_metrics(ax6)
        
        # 7. Network metrics summary
        ax7 = plt.subplot(2, 4, 7)
        self._plot_network_metrics_summary(ax7)
        
        # 8. Cross-correlation structure
        ax8 = plt.subplot(2, 4, 8)
        self._plot_correlation_structure(ax8)
        
        plt.tight_layout()
        plt.savefig('cross_mandelbrot_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Cross-Mandelbrot visualization saved as 'cross_mandelbrot_analysis.png'")
    
    def _plot_cross_hurst_heatmap(self, ax):
        """Plot cross-Hurst exponent heatmap."""
        pairs = list(self.cross_results.keys())
        assets = list(set([pair.split('-')[0] for pair in pairs] + [pair.split('-')[1] for pair in pairs]))
        
        hurst_matrix = np.zeros((len(assets), len(assets)))
        
        for pair, result in self.cross_results.items():
            asset1, asset2 = pair.split('-')
            if asset1 in assets and asset2 in assets:
                i, j = assets.index(asset1), assets.index(asset2)
                hurst_matrix[i, j] = result['cross_hurst']
                hurst_matrix[j, i] = result['cross_hurst']
        
        im = ax.imshow(hurst_matrix, cmap='viridis', vmin=0, vmax=1)
        ax.set_xticks(range(len(assets)))
        ax.set_yticks(range(len(assets)))
        ax.set_xticklabels(assets, rotation=45)
        ax.set_yticklabels(assets)
        ax.set_title('Cross-Hurst Exponents', fontweight='bold')
        
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    def _plot_cross_correlation_decay(self, ax):
        """Plot cross-correlation decay rates."""
        pairs = list(self.cross_results.keys())
        decay_rates = [self.cross_results[pair]['cross_correlation_decay'] for pair in pairs]
        
        ax.bar(range(len(pairs)), decay_rates, alpha=0.7, color='skyblue')
        ax.set_xticks(range(len(pairs)))
        ax.set_xticklabels([pair.replace('-', '\n') for pair in pairs], rotation=45, fontsize=8)
        ax.set_ylabel('Decay Rate')
        ax.set_title('Cross-Correlation Decay', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def _plot_lead_lag_relationships(self, ax):
        """Plot lead-lag relationship strengths."""
        pairs = list(self.cross_results.keys())
        lead_lag_strengths = [self.cross_results[pair]['lead_lag_strength'] for pair in pairs]
        optimal_lags = [self.cross_results[pair]['optimal_lag'] for pair in pairs]
        
        colors = ['red' if lag > 0 else 'blue' for lag in optimal_lags]
        
        ax.bar(range(len(pairs)), lead_lag_strengths, color=colors, alpha=0.7)
        ax.set_xticks(range(len(pairs)))
        ax.set_xticklabels([pair.replace('-', '\n') for pair in pairs], rotation=45, fontsize=8)
        ax.set_ylabel('Lead-Lag Strength')
        ax.set_title('Lead-Lag Relationships', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def _plot_cross_volatility_clustering(self, ax):
        """Plot cross-volatility clustering."""
        pairs = list(self.cross_results.keys())
        vol_clustering = [self.cross_results[pair]['cross_volatility_clustering'] for pair in pairs]
        
        ax.bar(range(len(pairs)), vol_clustering, color='green', alpha=0.7)
        ax.set_xticks(range(len(pairs)))
        ax.set_xticklabels([pair.replace('-', '\n') for pair in pairs], rotation=45, fontsize=8)
        ax.set_ylabel('Cross-Vol Clustering')
        ax.set_title('Cross-Volatility Clustering', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def _plot_multifractal_spectrum_comparison(self, ax):
        """Plot multifractal spectrum width comparison."""
        pairs = list(self.cross_results.keys())
        mf_widths = [self.cross_results[pair]['multifractal_width'] for pair in pairs]
        
        ax.bar(range(len(pairs)), mf_widths, color='purple', alpha=0.7)
        ax.set_xticks(range(len(pairs)))
        ax.set_xticklabels([pair.replace('-', '\n') for pair in pairs], rotation=45, fontsize=8)
        ax.set_ylabel('Multifractal Width')
        ax.set_title('Cross-Multifractal Spectrum', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def _plot_information_flow_metrics(self, ax):
        """Plot information flow metrics."""
        pairs = list(self.cross_results.keys())
        mutual_info = [self.cross_results[pair]['cross_mutual_information'] for pair in pairs]
        
        ax.bar(range(len(pairs)), mutual_info, color='orange', alpha=0.7)
        ax.set_xticks(range(len(pairs)))
        ax.set_xticklabels([pair.replace('-', '\n') for pair in pairs], rotation=45, fontsize=8)
        ax.set_ylabel('Mutual Information')
        ax.set_title('Cross-Information Flow', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def _plot_network_metrics_summary(self, ax):
        """Plot network-level metrics summary."""
        if hasattr(self, 'network_metrics'):
            metrics = self.network_metrics
            
            metric_names = ['Mean\nCross-Hurst', 'Network\nCorr Density', 'Multifractal\nComplexity']
            metric_values = [
                metrics['mean_cross_hurst'],
                metrics['network_correlation_density'],
                metrics['network_multifractal_complexity']
            ]
            
            ax.bar(metric_names, metric_values, color=['blue', 'green', 'purple'], alpha=0.7)
            ax.set_ylabel('Value')
            ax.set_title('Network Metrics Summary', fontweight='bold')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Network Metrics\nNot Available', 
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Network Metrics Summary', fontweight='bold')
    
    def _plot_correlation_structure(self, ax):
        """Plot correlation structure summary."""
        pairs = list(self.cross_results.keys())
        max_corrs = [self.cross_results[pair]['cross_correlation_max'] for pair in pairs]
        
        ax.bar(range(len(pairs)), max_corrs, color='coral', alpha=0.7)
        ax.set_xticks(range(len(pairs)))
        ax.set_xticklabels([pair.replace('-', '\n') for pair in pairs], rotation=45, fontsize=8)
        ax.set_ylabel('Max Cross-Correlation')
        ax.set_title('Cross-Correlation Structure', fontweight='bold')
        ax.grid(True, alpha=0.3)

if __name__ == "__main__":
    # Example usage
    print("🌀 Cross-Mandelbrot Analyzer Example")
    
    # Create synthetic data for demonstration
    np.random.seed(42)
    n_points = 500
    assets = ['AAPL', 'MSFT', 'GOOGL', 'NVDA']
    
    synthetic_data = {}
    for i, asset in enumerate(assets):
        base_returns = np.random.normal(0, 0.02, n_points)
        if i > 0:
            correlation = 0.3 + 0.1 * i
            base_returns += correlation * synthetic_data[assets[i-1]]
        synthetic_data[asset] = pd.Series(base_returns)
    
    # Run analysis
    analyzer = CrossMandelbrotAnalyzer()
    results = analyzer.analyze_cross_mandelbrot_comprehensive(synthetic_data)
    
    print(f"\n✅ Example analysis complete with {len(results)} pair results")