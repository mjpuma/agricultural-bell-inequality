#!/usr/bin/env python3
"""
AGRICULTURAL CROSS-SECTOR VISUALIZATION SUITE
=============================================

This module implements comprehensive visualization and statistical analysis
for agricultural cross-sector Bell inequality analysis. It provides sector-specific
heatmaps, transmission timeline visualizations, crisis comparison charts, and
innovative statistical visualizations for Science journal publication.

Key Features:
- Sector-specific heatmaps showing Bell violation rates by tier
- Transmission timeline visualizations showing 0-3 month propagation
- Three-crisis comparison charts (2008 financial crisis, EU debt crisis, COVID-19)
- Publication-ready figures with proper statistical annotations
- Crisis period time series visualizations
- Innovative statistical analysis and visualization suite

Authors: Agricultural Cross-Sector Analysis Team
Date: September 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import warnings
from datetime import datetime, timedelta
import networkx as nx
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

warnings.filterwarnings('ignore')

class CrisisPeriodTimeSeriesVisualizer:
    """
    Implements Crisis Period Time Series Visualizations for task 7.1.
    
    This class creates detailed time series plots for S1 violations during each
    crisis period, rolling violation rate time series with crisis highlighting,
    and transmission propagation visualizations.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (16, 12), dpi: int = 300):
        """
        Initialize the crisis period time series visualizer.
        
        Parameters:
        -----------
        figsize : Tuple[int, int], optional
            Figure size for plots. Default: (16, 12)
        dpi : int, optional
            DPI for high-resolution figures. Default: 300
        """
        self.figsize = figsize
        self.dpi = dpi
        
        # Define crisis periods following Zarifian et al. (2025)
        self.crisis_periods = {
            '2008_financial': {
                'start': '2008-09-01',
                'end': '2009-03-31',
                'name': '2008 Financial Crisis',
                'color': '#FF6B6B'
            },
            'eu_debt': {
                'start': '2010-05-01', 
                'end': '2012-12-31',
                'name': 'EU Debt Crisis',
                'color': '#4ECDC4'
            },
            'covid19': {
                'start': '2020-02-01',
                'end': '2020-12-31', 
                'name': 'COVID-19 Pandemic',
                'color': '#45B7D1'
            }
        }
        
        # Set publication-ready style
        plt.style.use('seaborn-v0_8-whitegrid')
        self.setup_publication_style()
    
    def setup_publication_style(self):
        """Setup publication-ready matplotlib style."""
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 11,
            'figure.titlesize': 16,
            'font.family': 'serif',
            'font.serif': ['Times New Roman'],
            'mathtext.fontset': 'stix',
            'axes.linewidth': 1.2,
            'grid.alpha': 0.3
        })
    
    def create_crisis_s1_time_series(self, s1_data: Dict[str, pd.Series], 
                                   pair_name: str, save_path: Optional[str] = None) -> plt.Figure:
        """
        Create detailed time series plots for S1 violations during each crisis period.
        
        This implements: "Create detailed time series plots for S1 violations 
        during each crisis period (2008, EU debt, COVID-19)"
        
        Parameters:
        -----------
        s1_data : Dict[str, pd.Series]
            Dictionary with crisis periods as keys and S1 time series as values
        pair_name : str
            Name of the asset pair being analyzed
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        plt.Figure : The created figure
        """
        fig, axes = plt.subplots(3, 1, figsize=self.figsize, sharex=False)
        fig.suptitle(f'S1 Bell Inequality Violations During Crisis Periods\n{pair_name}', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        classical_bound = 2.0
        quantum_bound = 2 * np.sqrt(2)
        
        for i, (crisis_key, crisis_info) in enumerate(self.crisis_periods.items()):
            ax = axes[i]
            
            if crisis_key in s1_data and len(s1_data[crisis_key]) > 0:
                s1_series = s1_data[crisis_key]
                
                # Plot S1 values
                ax.plot(s1_series.index, np.abs(s1_series.values), 
                       color=crisis_info['color'], linewidth=2, alpha=0.8, 
                       label='|S1| Values')
                
                # Add classical and quantum bounds
                ax.axhline(y=classical_bound, color='red', linestyle='--', 
                          linewidth=2, alpha=0.7, label='Classical Bound (2.0)')
                ax.axhline(y=quantum_bound, color='green', linestyle='--', 
                          linewidth=2, alpha=0.7, label='Quantum Bound (2.83)')
                
                # Highlight violations
                violations = np.abs(s1_series.values) > classical_bound
                if violations.any():
                    ax.fill_between(s1_series.index, 0, np.abs(s1_series.values),
                                   where=violations, alpha=0.3, color='red',
                                   label='Bell Violations')
                
                # Calculate and display statistics
                violation_rate = np.mean(violations) * 100
                max_violation = np.max(np.abs(s1_series.values))
                
                # Add statistics text box
                stats_text = f'Violation Rate: {violation_rate:.1f}%\nMax |S1|: {max_violation:.3f}'
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                       verticalalignment='top', bbox=dict(boxstyle='round', 
                       facecolor='white', alpha=0.8))
                
            else:
                ax.text(0.5, 0.5, f'No data available for {crisis_info["name"]}', 
                       transform=ax.transAxes, ha='center', va='center',
                       fontsize=12, style='italic')
            
            ax.set_title(f'{crisis_info["name"]} ({crisis_info["start"]} to {crisis_info["end"]})',
                        fontweight='bold')
            ax.set_ylabel('|S1| Value')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            
            # Format x-axis
            if i == 2:  # Last subplot
                ax.set_xlabel('Date')
            
            # Set y-axis limits for better visualization
            if crisis_key in s1_data and len(s1_data[crisis_key]) > 0:
                max_val = max(quantum_bound * 1.1, np.max(np.abs(s1_data[crisis_key].values)) * 1.1)
                ax.set_ylim(0, max_val)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"✅ Crisis S1 time series saved: {save_path}")
        
        return fig
    
    def create_rolling_violation_rate_series(self, s1_data: pd.Series, 
                                           window_size: int = 20,
                                           pair_name: str = "",
                                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Implement rolling violation rate time series with crisis period highlighting.
        
        This implements: "Implement rolling violation rate time series with 
        crisis period highlighting"
        
        Parameters:
        -----------
        s1_data : pd.Series
            Complete S1 time series data
        window_size : int, optional
            Rolling window size. Default: 20
        pair_name : str, optional
            Name of the asset pair
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        plt.Figure : The created figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, sharex=True)
        fig.suptitle(f'Rolling Bell Violation Analysis with Crisis Highlighting\n{pair_name}', 
                    fontsize=16, fontweight='bold')
        
        # Calculate rolling violation rate
        violations = np.abs(s1_data.values) > 2.0
        rolling_violation_rate = pd.Series(violations, index=s1_data.index).rolling(
            window=window_size, min_periods=1).mean() * 100
        
        # Plot S1 values (top panel)
        ax1.plot(s1_data.index, np.abs(s1_data.values), color='blue', 
                linewidth=1, alpha=0.7, label='|S1| Values')
        ax1.axhline(y=2.0, color='red', linestyle='--', linewidth=2, 
                   label='Classical Bound')
        ax1.axhline(y=2*np.sqrt(2), color='green', linestyle='--', linewidth=2, 
                   label='Quantum Bound')
        
        # Highlight crisis periods
        for crisis_key, crisis_info in self.crisis_periods.items():
            start_date = pd.to_datetime(crisis_info['start'])
            end_date = pd.to_datetime(crisis_info['end'])
            
            # Check if crisis period overlaps with data
            if start_date <= s1_data.index.max() and end_date >= s1_data.index.min():
                ax1.axvspan(start_date, end_date, alpha=0.2, 
                           color=crisis_info['color'], label=crisis_info['name'])
                ax2.axvspan(start_date, end_date, alpha=0.2, 
                           color=crisis_info['color'])
        
        ax1.set_ylabel('|S1| Value')
        ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax1.grid(True, alpha=0.3)
        ax1.set_title('S1 Bell Inequality Values Over Time')
        
        # Plot rolling violation rate (bottom panel)
        ax2.plot(rolling_violation_rate.index, rolling_violation_rate.values, 
                color='purple', linewidth=2, label=f'{window_size}-Period Rolling Violation Rate')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.axhline(y=20, color='orange', linestyle=':', alpha=0.7, 
                   label='20% Threshold')
        ax2.axhline(y=40, color='red', linestyle=':', alpha=0.7, 
                   label='40% Threshold')
        
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Violation Rate (%)')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        ax2.set_title(f'{window_size}-Period Rolling Bell Violation Rate')
        
        # Add crisis statistics
        crisis_stats = []
        for crisis_key, crisis_info in self.crisis_periods.items():
            start_date = pd.to_datetime(crisis_info['start'])
            end_date = pd.to_datetime(crisis_info['end'])
            
            # Get crisis period data
            crisis_mask = (s1_data.index >= start_date) & (s1_data.index <= end_date)
            if crisis_mask.any():
                crisis_s1 = s1_data[crisis_mask]
                crisis_violations = np.abs(crisis_s1.values) > 2.0
                crisis_rate = np.mean(crisis_violations) * 100
                crisis_stats.append(f'{crisis_info["name"]}: {crisis_rate:.1f}%')
        
        if crisis_stats:
            stats_text = 'Crisis Violation Rates:\n' + '\n'.join(crisis_stats)
            ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', 
                    facecolor='white', alpha=0.9))
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"✅ Rolling violation rate series saved: {save_path}")
        
        return fig
    
    def create_transmission_propagation_series(self, transmission_data: Dict[str, pd.DataFrame],
                                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Add transmission propagation time series showing energy→agriculture, transport→agriculture flows.
        
        This implements: "Add transmission propagation time series showing 
        energy→agriculture, transport→agriculture flows"
        
        Parameters:
        -----------
        transmission_data : Dict[str, pd.DataFrame]
            Dictionary with transmission types as keys and correlation data as values
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        plt.Figure : The created figure
        """
        fig, axes = plt.subplots(len(transmission_data), 1, figsize=self.figsize, 
                                sharex=True)
        if len(transmission_data) == 1:
            axes = [axes]
        
        fig.suptitle('Cross-Sector Transmission Propagation Analysis', 
                    fontsize=16, fontweight='bold')
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        
        for i, (transmission_type, data) in enumerate(transmission_data.items()):
            ax = axes[i]
            color = colors[i % len(colors)]
            
            if isinstance(data, pd.DataFrame) and not data.empty:
                # Plot correlation strength over time
                if 'correlation' in data.columns:
                    ax.plot(data.index, data['correlation'], color=color, 
                           linewidth=2, label=f'{transmission_type} Correlation')
                    
                    # Add transmission threshold
                    ax.axhline(y=0.3, color='orange', linestyle='--', alpha=0.7,
                              label='Transmission Threshold (0.3)')
                    ax.axhline(y=-0.3, color='orange', linestyle='--', alpha=0.7)
                    
                    # Highlight strong transmission periods
                    strong_transmission = np.abs(data['correlation']) > 0.3
                    if strong_transmission.any():
                        ax.fill_between(data.index, -1, 1, where=strong_transmission,
                                       alpha=0.2, color=color, 
                                       label='Strong Transmission')
                
                # Add crisis period highlighting
                for crisis_key, crisis_info in self.crisis_periods.items():
                    start_date = pd.to_datetime(crisis_info['start'])
                    end_date = pd.to_datetime(crisis_info['end'])
                    
                    if start_date <= data.index.max() and end_date >= data.index.min():
                        ax.axvspan(start_date, end_date, alpha=0.15, 
                                  color=crisis_info['color'])
            
            ax.set_title(f'{transmission_type.replace("_", " → ").title()} Transmission')
            ax.set_ylabel('Correlation Strength')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-1, 1)
            
            if i == len(transmission_data) - 1:
                ax.set_xlabel('Date')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"✅ Transmission propagation series saved: {save_path}")
        
        return fig
    
    def create_crisis_onset_detection(self, s1_data: pd.Series, 
                                    detection_window: int = 30,
                                    spike_threshold: float = 2.0,
                                    save_path: Optional[str] = None) -> plt.Figure:
        """
        Create crisis onset detection visualizations showing violation rate spikes.
        
        This implements: "Create crisis onset detection visualizations showing 
        violation rate spikes"
        
        Parameters:
        -----------
        s1_data : pd.Series
            S1 time series data
        detection_window : int, optional
            Window for spike detection. Default: 30
        spike_threshold : float, optional
            Threshold for spike detection. Default: 2.0
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        plt.Figure : The created figure
        """
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=self.figsize, sharex=True)
        fig.suptitle('Crisis Onset Detection Analysis', fontsize=16, fontweight='bold')
        
        # Calculate violation indicators
        violations = np.abs(s1_data.values) > 2.0
        violation_series = pd.Series(violations.astype(int), index=s1_data.index)
        
        # Calculate rolling violation rate and its derivative
        rolling_rate = violation_series.rolling(window=detection_window).mean() * 100
        rate_change = rolling_rate.diff()
        
        # Detect spikes (rapid increases in violation rate)
        spike_threshold_rate = np.percentile(rate_change.dropna(), 95)
        spikes = rate_change > spike_threshold_rate
        
        # Plot 1: S1 values with crisis periods
        ax1.plot(s1_data.index, np.abs(s1_data.values), color='blue', 
                linewidth=1, alpha=0.7, label='|S1| Values')
        ax1.axhline(y=2.0, color='red', linestyle='--', linewidth=2, 
                   label='Classical Bound')
        
        # Highlight detected crisis onsets
        for idx in s1_data.index[spikes]:
            ax1.axvline(x=idx, color='red', alpha=0.5, linewidth=2)
        
        # Add crisis period backgrounds
        for crisis_key, crisis_info in self.crisis_periods.items():
            start_date = pd.to_datetime(crisis_info['start'])
            end_date = pd.to_datetime(crisis_info['end'])
            
            if start_date <= s1_data.index.max() and end_date >= s1_data.index.min():
                ax1.axvspan(start_date, end_date, alpha=0.2, 
                           color=crisis_info['color'], label=crisis_info['name'])
        
        ax1.set_ylabel('|S1| Value')
        ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax1.grid(True, alpha=0.3)
        ax1.set_title('S1 Values with Crisis Onset Detection')
        
        # Plot 2: Rolling violation rate
        ax2.plot(rolling_rate.index, rolling_rate.values, color='purple', 
                linewidth=2, label=f'{detection_window}-Day Rolling Rate')
        
        # Highlight spikes
        spike_points = rolling_rate[spikes]
        if not spike_points.empty:
            ax2.scatter(spike_points.index, spike_points.values, 
                       color='red', s=50, zorder=5, label='Detected Onsets')
        
        ax2.set_ylabel('Violation Rate (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_title('Rolling Violation Rate with Spike Detection')
        
        # Plot 3: Rate of change (derivative)
        ax3.plot(rate_change.index, rate_change.values, color='green', 
                linewidth=1, alpha=0.7, label='Rate Change')
        ax3.axhline(y=spike_threshold_rate, color='red', linestyle='--', 
                   label=f'Spike Threshold ({spike_threshold_rate:.2f})')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Highlight spike periods
        ax3.fill_between(rate_change.index, 0, rate_change.values, 
                        where=spikes, alpha=0.3, color='red', 
                        label='Crisis Onset Signals')
        
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Rate Change (%/day)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_title('Violation Rate Change (Crisis Onset Indicator)')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"✅ Crisis onset detection saved: {save_path}")
        
        return fig
    
    def create_tier_specific_crisis_comparison(self, tier_data: Dict[str, Dict[str, pd.Series]],
                                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Implement tier-specific time series comparisons across all three crisis periods.
        
        This implements: "Implement tier-specific time series comparisons across 
        all three crisis periods"
        
        Parameters:
        -----------
        tier_data : Dict[str, Dict[str, pd.Series]]
            Nested dict: {tier: {crisis: s1_series}}
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        plt.Figure : The created figure
        """
        n_tiers = len(tier_data)
        fig, axes = plt.subplots(n_tiers, 1, figsize=(self.figsize[0], self.figsize[1] * 1.2), 
                                sharex=False)
        if n_tiers == 1:
            axes = [axes]
        
        fig.suptitle('Tier-Specific Crisis Period Comparison\nBell Violation Rates Across Three Major Crises', 
                    fontsize=16, fontweight='bold')
        
        for i, (tier_name, crisis_data) in enumerate(tier_data.items()):
            ax = axes[i]
            
            # Calculate violation rates for each crisis
            crisis_stats = {}
            bar_positions = []
            bar_heights = []
            bar_colors = []
            bar_labels = []
            
            for j, (crisis_key, crisis_info) in enumerate(self.crisis_periods.items()):
                if crisis_key in crisis_data and len(crisis_data[crisis_key]) > 0:
                    s1_values = crisis_data[crisis_key]
                    violations = np.abs(s1_values.values) > 2.0
                    violation_rate = np.mean(violations) * 100
                    
                    crisis_stats[crisis_key] = violation_rate
                    bar_positions.append(j)
                    bar_heights.append(violation_rate)
                    bar_colors.append(crisis_info['color'])
                    bar_labels.append(crisis_info['name'])
            
            # Create bar chart for this tier
            if bar_positions:
                bars = ax.bar(bar_positions, bar_heights, color=bar_colors, 
                             alpha=0.7, edgecolor='black', linewidth=1)
                
                # Add value labels on bars
                for bar, height in zip(bars, bar_heights):
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                           f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
                
                # Add horizontal reference lines
                ax.axhline(y=20, color='orange', linestyle=':', alpha=0.7, 
                          label='20% Threshold')
                ax.axhline(y=40, color='red', linestyle=':', alpha=0.7, 
                          label='40% Crisis Threshold')
                
                ax.set_xticks(bar_positions)
                ax.set_xticklabels(bar_labels, rotation=45, ha='right')
            
            ax.set_title(f'{tier_name} - Crisis Vulnerability Analysis', fontweight='bold')
            ax.set_ylabel('Bell Violation Rate (%)')
            ax.grid(True, alpha=0.3, axis='y')
            ax.legend()
            
            # Set consistent y-axis limits
            ax.set_ylim(0, max(60, max(bar_heights) * 1.2) if bar_heights else 60)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"✅ Tier-specific crisis comparison saved: {save_path}")
        
        return fig
    
    def create_seasonal_overlay_analysis(self, s1_data: pd.Series, 
                                       save_path: Optional[str] = None) -> plt.Figure:
        """
        Add seasonal overlay analysis showing crisis effects vs normal seasonal patterns.
        
        This implements: "Add seasonal overlay analysis showing crisis effects 
        vs normal seasonal patterns"
        
        Parameters:
        -----------
        s1_data : pd.Series
            S1 time series data with datetime index
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        plt.Figure : The created figure
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle('Seasonal Analysis of Bell Violations\nCrisis Effects vs Normal Patterns', 
                    fontsize=16, fontweight='bold')
        
        # Extract seasonal components
        s1_data_clean = s1_data.dropna()
        violations = np.abs(s1_data_clean.values) > 2.0
        violation_series = pd.Series(violations.astype(int), index=s1_data_clean.index)
        
        # Monthly aggregation
        monthly_violations = violation_series.groupby(violation_series.index.month).mean() * 100
        
        # Separate crisis and normal periods
        crisis_mask = pd.Series(False, index=s1_data_clean.index)
        normal_mask = pd.Series(True, index=s1_data_clean.index)
        
        for crisis_key, crisis_info in self.crisis_periods.items():
            start_date = pd.to_datetime(crisis_info['start'])
            end_date = pd.to_datetime(crisis_info['end'])
            
            period_mask = (s1_data_clean.index >= start_date) & (s1_data_clean.index <= end_date)
            crisis_mask |= period_mask
            normal_mask &= ~period_mask
        
        # Plot 1: Overall monthly pattern
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax1.bar(range(1, 13), monthly_violations.values, color='skyblue', 
               alpha=0.7, edgecolor='black')
        ax1.set_xticks(range(1, 13))
        ax1.set_xticklabels(months, rotation=45)
        ax1.set_ylabel('Violation Rate (%)')
        ax1.set_title('Overall Monthly Bell Violation Pattern')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Crisis vs Normal periods monthly comparison
        if crisis_mask.any() and normal_mask.any():
            crisis_monthly = violation_series[crisis_mask].groupby(
                violation_series[crisis_mask].index.month).mean() * 100
            normal_monthly = violation_series[normal_mask].groupby(
                violation_series[normal_mask].index.month).mean() * 100
            
            # Align months (fill missing with 0)
            all_months = range(1, 13)
            crisis_values = [crisis_monthly.get(m, 0) for m in all_months]
            normal_values = [normal_monthly.get(m, 0) for m in all_months]
            
            x = np.arange(len(months))
            width = 0.35
            
            ax2.bar(x - width/2, normal_values, width, label='Normal Periods', 
                   color='lightgreen', alpha=0.7, edgecolor='black')
            ax2.bar(x + width/2, crisis_values, width, label='Crisis Periods', 
                   color='lightcoral', alpha=0.7, edgecolor='black')
            
            ax2.set_xticks(x)
            ax2.set_xticklabels(months, rotation=45)
            ax2.set_ylabel('Violation Rate (%)')
            ax2.set_title('Crisis vs Normal: Monthly Comparison')
            ax2.legend()
            ax2.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Quarterly analysis
        quarterly_violations = violation_series.groupby(violation_series.index.quarter).mean() * 100
        quarters = ['Q1', 'Q2', 'Q3', 'Q4']
        ax3.bar(range(1, 5), quarterly_violations.values, color='lightblue', 
               alpha=0.7, edgecolor='black')
        ax3.set_xticks(range(1, 5))
        ax3.set_xticklabels(quarters)
        ax3.set_ylabel('Violation Rate (%)')
        ax3.set_title('Quarterly Bell Violation Pattern')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Agricultural season overlay
        # Define agricultural seasons (Northern Hemisphere)
        season_colors = {
            'Winter': '#87CEEB',  # Light blue
            'Spring': '#98FB98',  # Pale green  
            'Summer': '#FFD700',  # Gold
            'Fall': '#DEB887'     # Burlywood
        }
        
        def get_season(month):
            if month in [12, 1, 2]:
                return 'Winter'
            elif month in [3, 4, 5]:
                return 'Spring'
            elif month in [6, 7, 8]:
                return 'Summer'
            else:
                return 'Fall'
        
        seasonal_violations = violation_series.groupby(
            violation_series.index.map(lambda x: get_season(x.month))).mean() * 100
        
        seasons = ['Winter', 'Spring', 'Summer', 'Fall']
        season_values = [seasonal_violations.get(s, 0) for s in seasons]
        colors = [season_colors[s] for s in seasons]
        
        ax4.bar(seasons, season_values, color=colors, alpha=0.7, edgecolor='black')
        ax4.set_ylabel('Violation Rate (%)')
        ax4.set_title('Agricultural Seasonal Pattern')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add seasonal interpretation text
        max_season = seasons[np.argmax(season_values)]
        min_season = seasons[np.argmin(season_values)]
        
        interpretation = f'Highest violations: {max_season}\nLowest violations: {min_season}'
        ax4.text(0.02, 0.98, interpretation, transform=ax4.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"✅ Seasonal overlay analysis saved: {save_path}")
        
        return fig


class InnovativeStatisticalVisualizer:
    """
    Implements Innovative Statistical Analysis and Visualization Suite for task 7.2.
    
    This class creates quantum entanglement networks, crisis contagion maps,
    transmission velocity analysis, and other innovative visualizations.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (16, 12), dpi: int = 300):
        """Initialize the innovative statistical visualizer."""
        self.figsize = figsize
        self.dpi = dpi
        
        # Set publication-ready style
        plt.style.use('seaborn-v0_8-whitegrid')
        self.setup_publication_style()
    
    def setup_publication_style(self):
        """Setup publication-ready matplotlib style."""
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 11,
            'figure.titlesize': 16,
            'font.family': 'serif',
            'font.serif': ['Times New Roman'],
            'mathtext.fontset': 'stix',
            'axes.linewidth': 1.2,
            'grid.alpha': 0.3
        })
    
    def create_quantum_entanglement_network(self, correlation_matrix: pd.DataFrame,
                                          threshold: float = 0.3,
                                          save_path: Optional[str] = None) -> plt.Figure:
        """
        Create "Quantum Entanglement Networks" showing strongest cross-sector correlations.
        
        This implements: "Create 'Quantum Entanglement Networks' showing strongest 
        cross-sector correlations as network graphs"
        
        Parameters:
        -----------
        correlation_matrix : pd.DataFrame
            Correlation matrix between assets
        threshold : float, optional
            Minimum correlation threshold for edges. Default: 0.3
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        plt.Figure : The created figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        fig.suptitle('Quantum Entanglement Networks\nCross-Sector Correlation Analysis', 
                    fontsize=16, fontweight='bold')
        
        # Create network graph
        G = nx.Graph()
        
        # Add nodes
        for asset in correlation_matrix.index:
            G.add_node(asset)
        
        # Add edges for correlations above threshold
        for i, asset1 in enumerate(correlation_matrix.index):
            for j, asset2 in enumerate(correlation_matrix.columns):
                if i < j:  # Avoid duplicate edges
                    corr = correlation_matrix.loc[asset1, asset2]
                    if abs(corr) >= threshold:
                        G.add_edge(asset1, asset2, weight=abs(corr), correlation=corr)
        
        # Network visualization (left panel)
        pos = nx.spring_layout(G, k=1, iterations=50, seed=42)
        
        # Draw nodes with different colors for different sectors
        node_colors = []
        sector_colors = {
            'Energy': '#FF6B6B',
            'Agriculture': '#4ECDC4', 
            'Finance': '#45B7D1',
            'Technology': '#96CEB4',
            'Transport': '#FFEAA7'
        }
        
        for node in G.nodes():
            # Simple sector classification based on ticker patterns
            if any(x in node.upper() for x in ['XOM', 'CVX', 'COP']):
                node_colors.append(sector_colors['Energy'])
            elif any(x in node.upper() for x in ['ADM', 'BG', 'CF', 'MOS', 'DE']):
                node_colors.append(sector_colors['Agriculture'])
            elif any(x in node.upper() for x in ['JPM', 'BAC', 'GS']):
                node_colors.append(sector_colors['Finance'])
            elif any(x in node.upper() for x in ['AAPL', 'MSFT', 'GOOGL']):
                node_colors.append(sector_colors['Technology'])
            else:
                node_colors.append('#CCCCCC')  # Default gray
        
        # Draw network
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, 
                              alpha=0.8, ax=ax1)
        
        # Draw edges with thickness proportional to correlation strength
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        correlations = [G[u][v]['correlation'] for u, v in edges]
        
        # Color edges by correlation sign
        edge_colors = ['red' if corr > 0 else 'blue' for corr in correlations]
        
        nx.draw_networkx_edges(G, pos, width=[w*5 for w in weights], 
                              edge_color=edge_colors, alpha=0.6, ax=ax1)
        
        # Add labels
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax1)
        
        ax1.set_title('Quantum Entanglement Network\n(Edge thickness ∝ |correlation|)')
        ax1.axis('off')
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                      markersize=10, label=sector) 
            for sector, color in sector_colors.items()
        ]
        legend_elements.extend([
            plt.Line2D([0], [0], color='red', linewidth=3, label='Positive Correlation'),
            plt.Line2D([0], [0], color='blue', linewidth=3, label='Negative Correlation')
        ])
        ax1.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1))
        
        # Correlation strength distribution (right panel)
        all_correlations = []
        for i, asset1 in enumerate(correlation_matrix.index):
            for j, asset2 in enumerate(correlation_matrix.columns):
                if i < j:
                    corr = correlation_matrix.loc[asset1, asset2]
                    if not np.isnan(corr):
                        all_correlations.append(abs(corr))
        
        ax2.hist(all_correlations, bins=20, alpha=0.7, color='skyblue', 
                edgecolor='black', density=True)
        ax2.axvline(x=threshold, color='red', linestyle='--', linewidth=2, 
                   label=f'Network Threshold ({threshold})')
        ax2.set_xlabel('|Correlation|')
        ax2.set_ylabel('Density')
        ax2.set_title('Distribution of Correlation Strengths')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add network statistics
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        density = nx.density(G)
        
        stats_text = f'Nodes: {n_nodes}\nEdges: {n_edges}\nDensity: {density:.3f}'
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"✅ Quantum entanglement network saved: {save_path}")
        
        return fig
    
    def create_crisis_contagion_map(self, violation_data: Dict[str, pd.DataFrame],
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        Implement "Crisis Contagion Maps" visualizing violation spread across tiers.
        
        This implements: "Implement 'Crisis Contagion Maps' visualizing how 
        violations spread across tiers over time"
        
        Parameters:
        -----------
        violation_data : Dict[str, pd.DataFrame]
            Dictionary with tiers as keys and violation time series as values
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        plt.Figure : The created figure
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle('Crisis Contagion Maps\nViolation Spread Across Agricultural Tiers', 
                    fontsize=16, fontweight='bold')
        
        # Define crisis periods
        crisis_periods = {
            '2008_financial': ('2008-09-01', '2009-03-31'),
            'eu_debt': ('2010-05-01', '2012-12-31'),
            'covid19': ('2020-02-01', '2020-12-31')
        }
        
        # Plot 1: Heatmap of violation intensity over time
        ax1 = axes[0, 0]
        
        if violation_data:
            # Create combined time series matrix
            all_data = []
            tier_names = []
            
            for tier, data in violation_data.items():
                if isinstance(data, pd.DataFrame) and not data.empty:
                    # Calculate rolling violation rate
                    violations = data.apply(lambda x: (np.abs(x) > 2.0).astype(int))
                    rolling_violations = violations.rolling(window=20).mean() * 100
                    
                    # Resample to monthly for visualization
                    monthly_violations = rolling_violations.resample('M').mean()
                    
                    all_data.append(monthly_violations.mean(axis=1))
                    tier_names.append(tier)
            
            if all_data:
                # Create heatmap matrix
                heatmap_data = pd.concat(all_data, axis=1)
                heatmap_data.columns = tier_names
                
                # Plot heatmap
                im = ax1.imshow(heatmap_data.T.values, aspect='auto', cmap='Reds', 
                               interpolation='nearest')
                
                # Set ticks and labels
                ax1.set_yticks(range(len(tier_names)))
                ax1.set_yticklabels(tier_names)
                
                # Set x-axis to show dates
                n_dates = len(heatmap_data.index)
                tick_positions = np.linspace(0, n_dates-1, min(10, n_dates)).astype(int)
                ax1.set_xticks(tick_positions)
                ax1.set_xticklabels([heatmap_data.index[i].strftime('%Y-%m') 
                                   for i in tick_positions], rotation=45)
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax1)
                cbar.set_label('Violation Rate (%)')
                
                # Highlight crisis periods
                for crisis_name, (start, end) in crisis_periods.items():
                    start_date = pd.to_datetime(start)
                    end_date = pd.to_datetime(end)
                    
                    if start_date <= heatmap_data.index.max() and end_date >= heatmap_data.index.min():
                        # Find positions in the heatmap
                        start_pos = np.searchsorted(heatmap_data.index, start_date)
                        end_pos = np.searchsorted(heatmap_data.index, end_date)
                        
                        # Add rectangle to highlight crisis period
                        rect = Rectangle((start_pos-0.5, -0.5), end_pos-start_pos, 
                                       len(tier_names), linewidth=2, edgecolor='blue', 
                                       facecolor='none', alpha=0.7)
                        ax1.add_patch(rect)
        
        ax1.set_title('Violation Intensity Heatmap\n(Blue boxes = Crisis periods)')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Agricultural Tiers')
        
        # Plot 2: Contagion flow diagram
        ax2 = axes[0, 1]
        
        # Create a simplified flow diagram showing tier relationships
        tier_positions = {
            'Tier 1 (Energy/Transport)': (0.2, 0.8),
            'Tier 2 (Finance/Equipment)': (0.5, 0.5), 
            'Tier 3 (Policy/Utilities)': (0.8, 0.2)
        }
        
        # Draw nodes
        for tier, (x, y) in tier_positions.items():
            circle = plt.Circle((x, y), 0.1, color='lightblue', alpha=0.7)
            ax2.add_patch(circle)
            ax2.text(x, y, tier.split('(')[0], ha='center', va='center', 
                    fontweight='bold', fontsize=10)
        
        # Draw arrows showing contagion flow
        arrow_props = dict(arrowstyle='->', lw=2, color='red', alpha=0.7)
        
        # Tier 1 -> Tier 2
        ax2.annotate('', xy=(0.4, 0.55), xytext=(0.3, 0.75), arrowprops=arrow_props)
        # Tier 1 -> Tier 3  
        ax2.annotate('', xy=(0.7, 0.3), xytext=(0.3, 0.7), arrowprops=arrow_props)
        # Tier 2 -> Tier 3
        ax2.annotate('', xy=(0.7, 0.25), xytext=(0.6, 0.45), arrowprops=arrow_props)
        
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.set_title('Contagion Flow Model\n(Red arrows = Transmission paths)')
        ax2.axis('off')
        
        # Plot 3: Time-lagged correlation analysis
        ax3 = axes[1, 0]
        
        if len(violation_data) >= 2:
            tier_list = list(violation_data.keys())
            
            # Calculate cross-correlations with different lags
            lags = range(-30, 31, 5)  # -30 to +30 days, every 5 days
            correlations = []
            
            for lag in lags:
                if len(tier_list) >= 2:
                    tier1_data = violation_data[tier_list[0]]
                    tier2_data = violation_data[tier_list[1]]
                    
                    if isinstance(tier1_data, pd.DataFrame) and isinstance(tier2_data, pd.DataFrame):
                        # Calculate violations
                        tier1_violations = (np.abs(tier1_data) > 2.0).mean(axis=1)
                        tier2_violations = (np.abs(tier2_data) > 2.0).mean(axis=1)
                        
                        # Align data and apply lag
                        common_index = tier1_violations.index.intersection(tier2_violations.index)
                        if len(common_index) > abs(lag) + 10:
                            if lag >= 0:
                                corr = tier1_violations[common_index[:-lag if lag > 0 else None]].corr(
                                    tier2_violations[common_index[lag:]])
                            else:
                                corr = tier1_violations[common_index[-lag:]].corr(
                                    tier2_violations[common_index[:lag]])
                            
                            correlations.append(corr if not np.isnan(corr) else 0)
                        else:
                            correlations.append(0)
                    else:
                        correlations.append(0)
                else:
                    correlations.append(0)
            
            ax3.plot(lags, correlations, 'b-', linewidth=2, marker='o', markersize=4)
            ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax3.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='No lag')
            ax3.set_xlabel('Lag (days)')
            ax3.set_ylabel('Cross-correlation')
            ax3.set_title('Cross-Tier Contagion Timing\n(Positive lag = Tier1 leads Tier2)')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
        
        # Plot 4: Crisis amplification comparison
        ax4 = axes[1, 1]
        
        crisis_amplifications = {}
        for crisis_name, (start, end) in crisis_periods.items():
            start_date = pd.to_datetime(start)
            end_date = pd.to_datetime(end)
            
            tier_amplifications = []
            for tier, data in violation_data.items():
                if isinstance(data, pd.DataFrame) and not data.empty:
                    # Calculate normal and crisis violation rates
                    violations = (np.abs(data) > 2.0).mean(axis=1)
                    
                    crisis_mask = (violations.index >= start_date) & (violations.index <= end_date)
                    normal_mask = ~crisis_mask
                    
                    if crisis_mask.any() and normal_mask.any():
                        crisis_rate = violations[crisis_mask].mean() * 100
                        normal_rate = violations[normal_mask].mean() * 100
                        
                        amplification = crisis_rate / normal_rate if normal_rate > 0 else 0
                        tier_amplifications.append(amplification)
            
            if tier_amplifications:
                crisis_amplifications[crisis_name] = np.mean(tier_amplifications)
        
        if crisis_amplifications:
            crisis_names = list(crisis_amplifications.keys())
            amplification_values = list(crisis_amplifications.values())
            
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1'][:len(crisis_names)]
            bars = ax4.bar(crisis_names, amplification_values, color=colors, 
                          alpha=0.7, edgecolor='black')
            
            # Add value labels
            for bar, value in zip(bars, amplification_values):
                ax4.text(bar.get_x() + bar.get_width()/2., value + 0.05,
                        f'{value:.1f}x', ha='center', va='bottom', fontweight='bold')
            
            ax4.axhline(y=1, color='red', linestyle='--', alpha=0.7, 
                       label='No amplification')
            ax4.set_ylabel('Amplification Factor')
            ax4.set_title('Crisis Amplification Comparison\n(Violation rate increase)')
            ax4.legend()
            ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"✅ Crisis contagion map saved: {save_path}")
        
        return fig
    
    def create_transmission_velocity_analysis(self, transmission_data: Dict[str, pd.DataFrame],
                                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Add "Transmission Velocity Analysis" showing speed of correlation propagation.
        
        This implements: "Add 'Transmission Velocity Analysis' showing speed of 
        correlation propagation between sectors"
        
        Parameters:
        -----------
        transmission_data : Dict[str, pd.DataFrame]
            Dictionary with transmission pairs and their correlation time series
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        plt.Figure : The created figure
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle('Transmission Velocity Analysis\nSpeed of Cross-Sector Correlation Propagation', 
                    fontsize=16, fontweight='bold')
        
        # Plot 1: Transmission speed heatmap
        if transmission_data:
            transmission_speeds = {}
            transmission_strengths = {}
            
            for pair_name, data in transmission_data.items():
                if isinstance(data, pd.DataFrame) and 'correlation' in data.columns:
                    correlations = data['correlation'].dropna()
                    
                    if len(correlations) > 10:
                        # Calculate transmission speed as rate of correlation change
                        correlation_changes = correlations.diff().abs()
                        avg_speed = correlation_changes.mean()
                        max_strength = correlations.abs().max()
                        
                        transmission_speeds[pair_name] = avg_speed
                        transmission_strengths[pair_name] = max_strength
            
            if transmission_speeds:
                # Create speed vs strength scatter plot
                pairs = list(transmission_speeds.keys())
                speeds = list(transmission_speeds.values())
                strengths = list(transmission_strengths.values())
                
                scatter = ax1.scatter(speeds, strengths, s=100, alpha=0.7, 
                                    c=range(len(pairs)), cmap='viridis')
                
                # Add labels for each point
                for i, pair in enumerate(pairs):
                    ax1.annotate(pair, (speeds[i], strengths[i]), 
                               xytext=(5, 5), textcoords='offset points', 
                               fontsize=8, alpha=0.8)
                
                ax1.set_xlabel('Transmission Speed (Δcorr/day)')
                ax1.set_ylabel('Maximum Correlation Strength')
                ax1.set_title('Speed vs Strength Analysis')
                ax1.grid(True, alpha=0.3)
                
                # Add quadrant lines
                median_speed = np.median(speeds)
                median_strength = np.median(strengths)
                ax1.axvline(x=median_speed, color='red', linestyle='--', alpha=0.5)
                ax1.axhline(y=median_strength, color='red', linestyle='--', alpha=0.5)
                
                # Add quadrant labels
                ax1.text(0.95, 0.95, 'Fast & Strong', transform=ax1.transAxes, 
                        ha='right', va='top', bbox=dict(boxstyle='round', facecolor='lightgreen'))
                ax1.text(0.05, 0.95, 'Slow & Strong', transform=ax1.transAxes, 
                        ha='left', va='top', bbox=dict(boxstyle='round', facecolor='lightyellow'))
                ax1.text(0.95, 0.05, 'Fast & Weak', transform=ax1.transAxes, 
                        ha='right', va='bottom', bbox=dict(boxstyle='round', facecolor='lightcoral'))
                ax1.text(0.05, 0.05, 'Slow & Weak', transform=ax1.transAxes, 
                        ha='left', va='bottom', bbox=dict(boxstyle='round', facecolor='lightgray'))
        
        # Plot 2: Transmission lag distribution
        transmission_lags = []
        for pair_name, data in transmission_data.items():
            if isinstance(data, pd.DataFrame) and 'correlation' in data.columns:
                correlations = data['correlation'].dropna()
                
                # Find first significant correlation (>0.3)
                significant_mask = correlations.abs() > 0.3
                if significant_mask.any():
                    first_significant = significant_mask.idxmax()
                    lag_days = (first_significant - correlations.index[0]).days
                    transmission_lags.append(min(lag_days, 90))  # Cap at 90 days
        
        if transmission_lags:
            ax2.hist(transmission_lags, bins=15, alpha=0.7, color='skyblue', 
                    edgecolor='black', density=True)
            ax2.axvline(x=np.mean(transmission_lags), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {np.mean(transmission_lags):.1f} days')
            ax2.set_xlabel('Transmission Lag (days)')
            ax2.set_ylabel('Density')
            ax2.set_title('Distribution of Transmission Lags')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Velocity time series for top pairs
        if transmission_data:
            # Select top 3 pairs by maximum correlation strength
            top_pairs = sorted(transmission_data.items(), 
                             key=lambda x: x[1]['correlation'].abs().max() if 'correlation' in x[1].columns else 0, 
                             reverse=True)[:3]
            
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
            
            for i, (pair_name, data) in enumerate(top_pairs):
                if 'correlation' in data.columns:
                    correlations = data['correlation'].dropna()
                    
                    # Calculate velocity (rate of change)
                    velocity = correlations.diff()
                    
                    ax3.plot(velocity.index, velocity.values, color=colors[i], 
                            linewidth=2, alpha=0.8, label=pair_name)
            
            ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax3.set_xlabel('Date')
            ax3.set_ylabel('Correlation Velocity (Δcorr/day)')
            ax3.set_title('Transmission Velocity Time Series')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Sector-wise transmission efficiency
        sector_efficiency = {}
        
        for pair_name, data in transmission_data.items():
            if isinstance(data, pd.DataFrame) and 'correlation' in data.columns:
                # Extract sector information from pair name
                sectors = pair_name.split('_')
                if len(sectors) >= 2:
                    source_sector = sectors[0]
                    
                    correlations = data['correlation'].dropna()
                    if len(correlations) > 10:
                        # Efficiency = max correlation / time to reach max
                        max_corr = correlations.abs().max()
                        max_idx = correlations.abs().idxmax()
                        time_to_max = (max_idx - correlations.index[0]).days
                        
                        efficiency = max_corr / max(time_to_max, 1) * 100  # Scale to percentage
                        
                        if source_sector not in sector_efficiency:
                            sector_efficiency[source_sector] = []
                        sector_efficiency[source_sector].append(efficiency)
        
        # Average efficiency by sector
        sector_avg_efficiency = {sector: np.mean(efficiencies) 
                               for sector, efficiencies in sector_efficiency.items()}
        
        if sector_avg_efficiency:
            sectors = list(sector_avg_efficiency.keys())
            efficiencies = list(sector_avg_efficiency.values())
            
            bars = ax4.bar(sectors, efficiencies, color='lightgreen', 
                          alpha=0.7, edgecolor='black')
            
            # Add value labels
            for bar, efficiency in zip(bars, efficiencies):
                ax4.text(bar.get_x() + bar.get_width()/2., efficiency + 0.5,
                        f'{efficiency:.1f}', ha='center', va='bottom', fontweight='bold')
            
            ax4.set_ylabel('Transmission Efficiency\n(Max Corr / Days × 100)')
            ax4.set_title('Sector Transmission Efficiency')
            ax4.grid(True, alpha=0.3, axis='y')
            
            # Rotate x-axis labels if needed
            if len(sectors) > 3:
                ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"✅ Transmission velocity analysis saved: {save_path}")
        
        return fig
    
    def create_violation_intensity_heatmap(self, s1_data: Dict[str, pd.Series],
                                          save_path: Optional[str] = None) -> plt.Figure:
        """
        Create "Violation Intensity Heatmaps" with time on x-axis, asset pairs on y-axis.
        
        This implements: "Create 'Violation Intensity Heatmaps' with time on x-axis, 
        asset pairs on y-axis, violation strength as color"
        
        Parameters:
         de
f generate_all_visualizations(self, analysis_data: Dict, output_dir: str = "visualizations") -> List[str]:
        """
        Generate all visualizations for comprehensive analysis.
        
        Parameters:
        -----------
        analysis_data : Dict
            Complete analysis data including S1 results, crisis data, etc.
        output_dir : str, optional
            Output directory for visualizations. Default: "visualizations"
            
        Returns:
        --------
        List[str] : List of generated figure paths
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        generated_files = []
        
        # 1. Crisis period time series visualizations
        if 'crisis_s1_data' in analysis_data:
            for pair_name, crisis_data in analysis_data['crisis_s1_data'].items():
                file_path = os.path.join(output_dir, f"crisis_timeseries_{pair_name}.png")
                self.crisis_visualizer.create_crisis_s1_time_series(
                    crisis_data, pair_name, file_path)
                generated_files.append(file_path)
        
        # 2. Rolling violation rate series
        if 's1_data' in analysis_data:
            for pair_name, s1_series in analysis_data['s1_data'].items():
                file_path = os.path.join(output_dir, f"rolling_violations_{pair_name}.png")
                self.crisis_visualizer.create_rolling_violation_rate_series(
                    s1_series, pair_name=pair_name, save_path=file_path)
                generated_files.append(file_path)
        
        # 3. Transmission propagation series
        if 'transmission_data' in analysis_data:
            file_path = os.path.join(output_dir, "transmission_propagation.png")
            self.crisis_visualizer.create_transmission_propagation_series(
                analysis_data['transmission_data'], file_path)
            generated_files.append(file_path)
        
        # 4. Crisis onset detection
        if 's1_data' in analysis_data:
            for pair_name, s1_series in list(analysis_data['s1_data'].items())[:3]:  # Top 3 pairs
                file_path = os.path.join(output_dir, f"crisis_onset_{pair_name}.png")
                self.crisis_visualizer.create_crisis_onset_detection(
                    s1_series, save_path=file_path)
                generated_files.append(file_path)
        
        # 5. Tier-specific crisis comparison
        if 'tier_crisis_data' in analysis_data:
            file_path = os.path.join(output_dir, "tier_crisis_comparison.png")
            self.crisis_visualizer.create_tier_specific_crisis_comparison(
                analysis_data['tier_crisis_data'], file_path)
            generated_files.append(file_path)
        
        # 6. Seasonal overlay analysis
        if 's1_data' in analysis_data:
            # Use first available S1 series for seasonal analysis
            first_series = next(iter(analysis_data['s1_data'].values()))
            file_path = os.path.join(output_dir, "seasonal_analysis.png")
            self.crisis_visualizer.create_seasonal_overlay_analysis(
                first_series, file_path)
            generated_files.append(file_path)
        
        # 7. Quantum entanglement network
        if 'correlation_matrix' in analysis_data:
            file_path = os.path.join(output_dir, "quantum_entanglement_network.png")
            self.innovative_visualizer.create_quantum_entanglement_network(
                analysis_data['correlation_matrix'], save_path=file_path)
            generated_files.append(file_path)
        
        # 8. Crisis contagion map
        if 'tier_violation_data' in analysis_data:
            file_path = os.path.join(output_dir, "crisis_contagion_map.png")
            self.innovative_visualizer.create_crisis_contagion_map(
                analysis_data['tier_violation_data'], file_path)
            generated_files.append(file_path)
        
        # 9. Transmission velocity analysis
        if 'transmission_data' in analysis_data:
            file_path = os.path.join(output_dir, "transmission_velocity.png")
            self.innovative_visualizer.create_transmission_velocity_analysis(
                analysis_data['transmission_data'], file_path)
            generated_files.append(file_path)
        
        # 10. Violation intensity heatmap
        if 's1_data' in analysis_data:
            file_path = os.path.join(output_dir, "violation_intensity_heatmap.png")
            self.innovative_visualizer.create_violation_intensity_heatmap(
                analysis_data['s1_data'], file_path)
            generated_files.append(file_path)
        
        # 11. Tier sensitivity radar chart
        if 'tier_crisis_sensitivity' in analysis_data:
            file_path = os.path.join(output_dir, "tier_sensitivity_radar.png")
            self.innovative_visualizer.create_tier_sensitivity_radar_chart(
                analysis_data['tier_crisis_sensitivity'], file_path)
            generated_files.append(file_path)
        
        # 12. Sector-specific heatmaps
        if 'sector_tier_data' in analysis_data:
            file_path = os.path.join(output_dir, "sector_heatmaps.png")
            self.create_sector_specific_heatmaps(
                analysis_data['sector_tier_data'], file_path)
            generated_files.append(file_path)
        
        # 13. Three-crisis comparison charts
        if 'three_crisis_data' in analysis_data:
            file_path = os.path.join(output_dir, "three_crisis_comparison.png")
            self.create_three_crisis_comparison_charts(
                analysis_data['three_crisis_data'], file_path)
            generated_files.append(file_path)
        
        # 14. Publication-ready summary
        if 'analysis_results' in analysis_data:
            file_path = os.path.join(output_dir, "publication_summary.png")
            self.create_publication_ready_summary(
                analysis_data['analysis_results'], file_path)
            generated_files.append(file_path)
        
        self.generated_figures.extend(generated_files)
        
        print(f"\n🎨 COMPREHENSIVE VISUALIZATION SUITE COMPLETE")
        print(f"=" * 50)
        print(f"Generated {len(generated_files)} visualizations:")
        for i, file_path in enumerate(generated_files, 1):
            print(f"  {i:2d}. {os.path.basename(file_path)}")
        print(f"📁 All files saved to: {output_dir}")
        
        return generated_files
    
    def get_visualization_summary(self) -> Dict[str, any]:
        """
        Get summary of all generated visualizations.
        
        Returns:
        --------
        Dict : Summary of visualization suite
        """
        return {
            'total_figures': len(self.generated_figures),
            'figure_paths': self.generated_figures,
            'visualization_types': [
                'Crisis Period Time Series',
                'Rolling Violation Rates', 
                'Transmission Propagation',
                'Crisis Onset Detection',
                'Tier-Specific Comparisons',
                'Seasonal Analysis',
                'Quantum Entanglement Networks',
                'Crisis Contagion Maps',
                'Transmission Velocity Analysis',
                'Violation Intensity Heatmaps',
                'Tier Sensitivity Radar Charts',
                'Sector-Specific Heatmaps',
                'Three-Crisis Comparisons',
                'Publication-Ready Summary'
            ],
            'publication_ready': True,
            'statistical_annotations': True,
            'crisis_periods_covered': ['2008 Financial Crisis', 'EU Debt Crisis', 'COVID-19'],
            'innovative_metrics_included': True
        }


class ThreeCrisisAnalysisFramework:
    """
    Implements Three-Crisis Analysis Framework for task 7.3.
    
    This class creates crisis period definitions, tier-specific crisis analysis,
    crisis amplification metrics, and cross-crisis comparison analysis.
    """
    
    def __init__(self):
        """Initialize the three-crisis analysis framework."""
        
        # Crisis period definitions matching Zarifian et al. (2025)
        self.crisis_definitions = {
            '2008_financial': {
                'start': '2008-09-01',
                'end': '2009-03-31', 
                'name': '2008 Financial Crisis',
                'description': 'Global financial crisis triggered by subprime mortgage collapse',
                'color': '#FF6B6B'
            },
            'eu_debt': {
                'start': '2010-05-01',
                'end': '2012-12-31',
                'name': 'EU Debt Crisis', 
                'description': 'European sovereign debt crisis affecting multiple EU countries',
                'color': '#4ECDC4'
            },
            'covid19': {
                'start': '2020-02-01',
                'end': '2020-12-31',
                'name': 'COVID-19 Pandemic',
                'description': 'Global pandemic causing widespread economic disruption',
                'color': '#45B7D1'
            }
        }
    
    def create_crisis_period_definitions(self, save_path: Optional[str] = None) -> pd.DataFrame:
        """
        Create crisis period definitions matching Zarifian et al. (2025).
        
        This implements: "Create crisis period definitions matching Zarifian et al. (2025): 
        2008 financial crisis (Sep 2008 - Mar 2009), EU debt crisis (May 2010 - Dec 2012), 
        COVID-19 (Feb 2020 - Dec 2020)"
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the definitions table
            
        Returns:
        --------
        pd.DataFrame : Crisis period definitions
        """
        crisis_df = pd.DataFrame.from_dict(self.crisis_definitions, orient='index')
        
        # Add duration calculations
        crisis_df['start_date'] = pd.to_datetime(crisis_df['start'])
        crisis_df['end_date'] = pd.to_datetime(crisis_df['end'])
        crisis_df['duration_days'] = (crisis_df['end_date'] - crisis_df['start_date']).dt.days
        crisis_df['duration_months'] = crisis_df['duration_days'] / 30.44  # Average days per month
        
        # Reorder columns
        crisis_df = crisis_df[['name', 'start', 'end', 'duration_days', 'duration_months', 'description']]
        
        if save_path:
            crisis_df.to_csv(save_path, index=True)
            print(f"✅ Crisis period definitions saved: {save_path}")
        
        print("\n📅 CRISIS PERIOD DEFINITIONS")
        print("=" * 40)
        for crisis_id, info in self.crisis_definitions.items():
            print(f"{info['name']}:")
            print(f"  Period: {info['start']} to {info['end']}")
            print(f"  Duration: {crisis_df.loc[crisis_id, 'duration_months']:.1f} months")
            print(f"  Description: {info['description']}")
            print()
        
        return crisis_df
    
    def implement_tier_specific_crisis_analysis(self, s1_data: Dict[str, pd.Series],
                                              tier_mapping: Dict[str, str],
                                              save_path: Optional[str] = None) -> Dict[str, Dict[str, float]]:
        """
        Implement tier-specific crisis analysis for each of the three periods.
        
        This implements: "Implement tier-specific crisis analysis for each of the three periods"
        
        Parameters:
        -----------
        s1_data : Dict[str, pd.Series]
            S1 time series data for asset pairs
        tier_mapping : Dict[str, str]
            Mapping of asset pairs to tiers
        save_path : str, optional
            Path to save the analysis results
            
        Returns:
        --------
        Dict[str, Dict[str, float]] : Nested dict {tier: {crisis: violation_rate}}
        """
        tier_crisis_analysis = {}
        
        # Initialize tier structure
        unique_tiers = set(tier_mapping.values())
        for tier in unique_tiers:
            tier_crisis_analysis[tier] = {}
        
        # Analyze each crisis period
        for crisis_id, crisis_info in self.crisis_definitions.items():
            start_date = pd.to_datetime(crisis_info['start'])
            end_date = pd.to_datetime(crisis_info['end'])
            
            print(f"\n🔍 Analyzing {crisis_info['name']} ({crisis_info['start']} to {crisis_info['end']})")
            
            # Analyze each tier during this crisis
            for tier in unique_tiers:
                tier_pairs = [pair for pair, t in tier_mapping.items() if t == tier]
                tier_violations = []
                
                for pair in tier_pairs:
                    if pair in s1_data:
                        s1_series = s1_data[pair]
                        
                        # Filter to crisis period
                        crisis_mask = (s1_series.index >= start_date) & (s1_series.index <= end_date)
                        crisis_s1 = s1_series[crisis_mask]
                        
                        if len(crisis_s1) > 0:
                            # Calculate violation rate
                            violations = np.abs(crisis_s1.values) > 2.0
                            violation_rate = np.mean(violations) * 100
                            tier_violations.append(violation_rate)
                
                # Calculate tier average for this crisis
                if tier_violations:
                    tier_avg_violation = np.mean(tier_violations)
                    tier_crisis_analysis[tier][crisis_id] = tier_avg_violation
                    print(f"  {tier}: {tier_avg_violation:.1f}% violation rate ({len(tier_violations)} pairs)")
                else:
                    tier_crisis_analysis[tier][crisis_id] = 0.0
                    print(f"  {tier}: No data available")
        
        if save_path:
            # Convert to DataFrame and save
            df = pd.DataFrame(tier_crisis_analysis).T
            df.to_csv(save_path, index=True)
            print(f"✅ Tier-specific crisis analysis saved: {save_path}")
        
        return tier_crisis_analysis
    
    def create_crisis_amplification_metrics(self, s1_data: Dict[str, pd.Series],
                                          save_path: Optional[str] = None) -> Dict[str, Dict[str, float]]:
        """
        Create crisis amplification metrics comparing violation rates during crisis vs normal periods.
        
        This implements: "Create crisis amplification metrics comparing violation rates 
        during crisis vs normal periods"
        
        Parameters:
        -----------
        s1_data : Dict[str, pd.Series]
            S1 time series data for asset pairs
        save_path : str, optional
            Path to save the amplification metrics
            
        Returns:
        --------
        Dict[str, Dict[str, float]] : Amplification metrics for each pair and crisis
        """
        amplification_metrics = {}
        
        for pair, s1_series in s1_data.items():
            amplification_metrics[pair] = {}
            
            # Calculate normal period violation rate (exclude all crisis periods)
            normal_mask = pd.Series(True, index=s1_series.index)
            
            for crisis_id, crisis_info in self.crisis_definitions.items():
                start_date = pd.to_datetime(crisis_info['start'])
                end_date = pd.to_datetime(crisis_info['end'])
                
                crisis_mask = (s1_series.index >= start_date) & (s1_series.index <= end_date)
                normal_mask &= ~crisis_mask
            
            # Calculate normal period violation rate
            if normal_mask.any():
                normal_s1 = s1_series[normal_mask]
                normal_violations = np.abs(normal_s1.values) > 2.0
                normal_rate = np.mean(normal_violations) * 100
            else:
                normal_rate = 0.0
            
            # Calculate amplification for each crisis
            for crisis_id, crisis_info in self.crisis_definitions.items():
                start_date = pd.to_datetime(crisis_info['start'])
                end_date = pd.to_datetime(crisis_info['end'])
                
                crisis_mask = (s1_series.index >= start_date) & (s1_series.index <= end_date)
                
                if crisis_mask.any():
                    crisis_s1 = s1_series[crisis_mask]
                    crisis_violations = np.abs(crisis_s1.values) > 2.0
                    crisis_rate = np.mean(crisis_violations) * 100
                    
                    # Calculate amplification factor
                    if normal_rate > 0:
                        amplification_factor = crisis_rate / normal_rate
                    else:
                        amplification_factor = float('inf') if crisis_rate > 0 else 1.0
                    
                    amplification_metrics[pair][f'{crisis_id}_amplification'] = amplification_factor
                    amplification_metrics[pair][f'{crisis_id}_crisis_rate'] = crisis_rate
                    amplification_metrics[pair][f'{crisis_id}_normal_rate'] = normal_rate
                else:
                    amplification_metrics[pair][f'{crisis_id}_amplification'] = 1.0
                    amplification_metrics[pair][f'{crisis_id}_crisis_rate'] = 0.0
                    amplification_metrics[pair][f'{crisis_id}_normal_rate'] = normal_rate
        
        if save_path:
            # Convert to DataFrame and save
            df = pd.DataFrame(amplification_metrics).T
            df.to_csv(save_path, index=True)
            print(f"✅ Crisis amplification metrics saved: {save_path}")
        
        # Print summary
        print(f"\n📈 CRISIS AMPLIFICATION SUMMARY")
        print("=" * 40)
        
        for crisis_id, crisis_info in self.crisis_definitions.items():
            amplifications = [metrics.get(f'{crisis_id}_amplification', 1.0) 
                            for metrics in amplification_metrics.values()]
            amplifications = [a for a in amplifications if a != float('inf')]
            
            if amplifications:
                avg_amplification = np.mean(amplifications)
                max_amplification = np.max(amplifications)
                print(f"{crisis_info['name']}:")
                print(f"  Average amplification: {avg_amplification:.2f}x")
                print(f"  Maximum amplification: {max_amplification:.2f}x")
                print(f"  Pairs with >2x amplification: {sum(1 for a in amplifications if a > 2)}")
                print()
        
        return amplification_metrics
    
    def add_statistical_significance_testing(self, s1_data: Dict[str, pd.Series],
                                           save_path: Optional[str] = None) -> Dict[str, Dict[str, float]]:
        """
        Add statistical significance testing for crisis vs normal period differences.
        
        This implements: "Add statistical significance testing for crisis vs normal period differences"
        
        Parameters:
        -----------
        s1_data : Dict[str, pd.Series]
            S1 time series data for asset pairs
        save_path : str, optional
            Path to save the significance test results
            
        Returns:
        --------
        Dict[str, Dict[str, float]] : Statistical significance test results
        """
        from scipy import stats
        
        significance_results = {}
        
        for pair, s1_series in s1_data.items():
            significance_results[pair] = {}
            
            # Calculate normal period data (exclude all crisis periods)
            normal_mask = pd.Series(True, index=s1_series.index)
            
            for crisis_id, crisis_info in self.crisis_definitions.items():
                start_date = pd.to_datetime(crisis_info['start'])
                end_date = pd.to_datetime(crisis_info['end'])
                
                crisis_mask = (s1_series.index >= start_date) & (s1_series.index <= end_date)
                normal_mask &= ~crisis_mask
            
            # Get normal period violations
            if normal_mask.any():
                normal_s1 = s1_series[normal_mask]
                normal_violations = (np.abs(normal_s1.values) > 2.0).astype(int)
            else:
                normal_violations = np.array([])
            
            # Test each crisis period
            for crisis_id, crisis_info in self.crisis_definitions.items():
                start_date = pd.to_datetime(crisis_info['start'])
                end_date = pd.to_datetime(crisis_info['end'])
                
                crisis_mask = (s1_series.index >= start_date) & (s1_series.index <= end_date)
                
                if crisis_mask.any() and len(normal_violations) > 0:
                    crisis_s1 = s1_series[crisis_mask]
                    crisis_violations = (np.abs(crisis_s1.values) > 2.0).astype(int)
                    
                    # Perform two-sample t-test
                    if len(crisis_violations) > 1 and len(normal_violations) > 1:
                        t_stat, p_value = stats.ttest_ind(crisis_violations, normal_violations)
                        
                        # Perform Mann-Whitney U test (non-parametric)
                        u_stat, u_p_value = stats.mannwhitneyu(crisis_violations, normal_violations, 
                                                              alternative='two-sided')
                        
                        # Effect size (Cohen's d)
                        pooled_std = np.sqrt(((len(crisis_violations) - 1) * np.var(crisis_violations, ddof=1) + 
                                            (len(normal_violations) - 1) * np.var(normal_violations, ddof=1)) / 
                                           (len(crisis_violations) + len(normal_violations) - 2))
                        
                        if pooled_std > 0:
                            cohens_d = (np.mean(crisis_violations) - np.mean(normal_violations)) / pooled_std
                        else:
                            cohens_d = 0.0
                        
                        significance_results[pair][f'{crisis_id}_t_stat'] = t_stat
                        significance_results[pair][f'{crisis_id}_p_value'] = p_value
                        significance_results[pair][f'{crisis_id}_u_p_value'] = u_p_value
                        significance_results[pair][f'{crisis_id}_cohens_d'] = cohens_d
                        significance_results[pair][f'{crisis_id}_significant'] = p_value < 0.05
                        significance_results[pair][f'{crisis_id}_highly_significant'] = p_value < 0.001
                    else:
                        # Insufficient data
                        significance_results[pair][f'{crisis_id}_t_stat'] = np.nan
                        significance_results[pair][f'{crisis_id}_p_value'] = 1.0
                        significance_results[pair][f'{crisis_id}_u_p_value'] = 1.0
                        significance_results[pair][f'{crisis_id}_cohens_d'] = 0.0
                        significance_results[pair][f'{crisis_id}_significant'] = False
                        significance_results[pair][f'{crisis_id}_highly_significant'] = False
        
        if save_path:
            # Convert to DataFrame and save
            df = pd.DataFrame(significance_results).T
            df.to_csv(save_path, index=True)
            print(f"✅ Statistical significance results saved: {save_path}")
        
        # Print summary
        print(f"\n📊 STATISTICAL SIGNIFICANCE SUMMARY")
        print("=" * 40)
        
        for crisis_id, crisis_info in self.crisis_definitions.items():
            significant_pairs = sum(1 for results in significance_results.values() 
                                  if results.get(f'{crisis_id}_significant', False))
            highly_significant_pairs = sum(1 for results in significance_results.values() 
                                         if results.get(f'{crisis_id}_highly_significant', False))
            total_pairs = len(significance_results)
            
            print(f"{crisis_info['name']}:")
            print(f"  Significant pairs (p<0.05): {significant_pairs}/{total_pairs} ({significant_pairs/total_pairs*100:.1f}%)")
            print(f"  Highly significant (p<0.001): {highly_significant_pairs}/{total_pairs} ({highly_significant_pairs/total_pairs*100:.1f}%)")
            print()
        
        return significance_results
    
    def implement_cross_crisis_comparison_analysis(self, amplification_metrics: Dict[str, Dict[str, float]],
                                                 tier_mapping: Dict[str, str],
                                                 save_path: Optional[str] = None) -> Dict[str, Dict[str, float]]:
        """
        Implement cross-crisis comparison analysis showing which tiers are most sensitive to each crisis type.
        
        This implements: "Implement cross-crisis comparison analysis showing which tiers 
        are most sensitive to each crisis type"
        
        Parameters:
        -----------
        amplification_metrics : Dict[str, Dict[str, float]]
            Crisis amplification metrics from previous analysis
        tier_mapping : Dict[str, str]
            Mapping of asset pairs to tiers
        save_path : str, optional
            Path to save the comparison analysis
            
        Returns:
        --------
        Dict[str, Dict[str, float]] : Cross-crisis comparison results
        """
        cross_crisis_comparison = {}
        
        # Group by tiers
        unique_tiers = set(tier_mapping.values())
        
        for tier in unique_tiers:
            cross_crisis_comparison[tier] = {}
            tier_pairs = [pair for pair, t in tier_mapping.items() if t == tier]
            
            # Calculate tier sensitivity for each crisis
            for crisis_id, crisis_info in self.crisis_definitions.items():
                amplifications = []
                
                for pair in tier_pairs:
                    if pair in amplification_metrics:
                        amp_key = f'{crisis_id}_amplification'
                        if amp_key in amplification_metrics[pair]:
                            amp_value = amplification_metrics[pair][amp_key]
                            if amp_value != float('inf'):
                                amplifications.append(amp_value)
                
                if amplifications:
                    tier_sensitivity = np.mean(amplifications)
                    cross_crisis_comparison[tier][f'{crisis_id}_sensitivity'] = tier_sensitivity
                    cross_crisis_comparison[tier][f'{crisis_id}_max_amplification'] = np.max(amplifications)
                    cross_crisis_comparison[tier][f'{crisis_id}_pairs_affected'] = len(amplifications)
                else:
                    cross_crisis_comparison[tier][f'{crisis_id}_sensitivity'] = 1.0
                    cross_crisis_comparison[tier][f'{crisis_id}_max_amplification'] = 1.0
                    cross_crisis_comparison[tier][f'{crisis_id}_pairs_affected'] = 0
        
        # Calculate relative sensitivity rankings
        for crisis_id, crisis_info in self.crisis_definitions.items():
            sensitivities = [(tier, data.get(f'{crisis_id}_sensitivity', 1.0)) 
                           for tier, data in cross_crisis_comparison.items()]
            sensitivities.sort(key=lambda x: x[1], reverse=True)
            
            for rank, (tier, sensitivity) in enumerate(sensitivities, 1):
                cross_crisis_comparison[tier][f'{crisis_id}_rank'] = rank
        
        if save_path:
            # Convert to DataFrame and save
            df = pd.DataFrame(cross_crisis_comparison).T
            df.to_csv(save_path, index=True)
            print(f"✅ Cross-crisis comparison analysis saved: {save_path}")
        
        # Print summary
        print(f"\n🔄 CROSS-CRISIS COMPARISON SUMMARY")
        print("=" * 40)
        
        for crisis_id, crisis_info in self.crisis_definitions.items():
            print(f"{crisis_info['name']} - Tier Sensitivity Ranking:")
            
            sensitivities = [(tier, data.get(f'{crisis_id}_sensitivity', 1.0)) 
                           for tier, data in cross_crisis_comparison.items()]
            sensitivities.sort(key=lambda x: x[1], reverse=True)
            
            for rank, (tier, sensitivity) in enumerate(sensitivities, 1):
                print(f"  {rank}. {tier}: {sensitivity:.2f}x amplification")
            print()
        
        return cross_crisis_comparison
    
    def add_crisis_recovery_analysis(self, s1_data: Dict[str, pd.Series],
                                   recovery_window: int = 90,
                                   save_path: Optional[str] = None) -> Dict[str, Dict[str, float]]:
        """
        Add crisis recovery analysis showing how violations decay post-crisis.
        
        This implements: "Add crisis recovery analysis showing how violations decay post-crisis"
        
        Parameters:
        -----------
        s1_data : Dict[str, pd.Series]
            S1 time series data for asset pairs
        recovery_window : int, optional
            Recovery window in days. Default: 90
        save_path : str, optional
            Path to save the recovery analysis
            
        Returns:
        --------
        Dict[str, Dict[str, float]] : Crisis recovery analysis results
        """
        recovery_analysis = {}
        
        for pair, s1_series in s1_data.items():
            recovery_analysis[pair] = {}
            
            for crisis_id, crisis_info in self.crisis_definitions.items():
                end_date = pd.to_datetime(crisis_info['end'])
                recovery_end = end_date + timedelta(days=recovery_window)
                
                # Get crisis period data
                crisis_start = pd.to_datetime(crisis_info['start'])
                crisis_mask = (s1_series.index >= crisis_start) & (s1_series.index <= end_date)
                
                # Get recovery period data
                recovery_mask = (s1_series.index > end_date) & (s1_series.index <= recovery_end)
                
                if crisis_mask.any() and recovery_mask.any():
                    crisis_s1 = s1_series[crisis_mask]
                    recovery_s1 = s1_series[recovery_mask]
                    
                    # Calculate violation rates
                    crisis_violations = np.abs(crisis_s1.values) > 2.0
                    recovery_violations = np.abs(recovery_s1.values) > 2.0
                    
                    crisis_rate = np.mean(crisis_violations) * 100
                    recovery_rate = np.mean(recovery_violations) * 100
                    
                    # Calculate recovery metrics
                    if crisis_rate > 0:
                        recovery_ratio = recovery_rate / crisis_rate
                        recovery_speed = (crisis_rate - recovery_rate) / recovery_window  # Rate per day
                    else:
                        recovery_ratio = 1.0
                        recovery_speed = 0.0
                    
                    recovery_analysis[pair][f'{crisis_id}_crisis_rate'] = crisis_rate
                    recovery_analysis[pair][f'{crisis_id}_recovery_rate'] = recovery_rate
                    recovery_analysis[pair][f'{crisis_id}_recovery_ratio'] = recovery_ratio
                    recovery_analysis[pair][f'{crisis_id}_recovery_speed'] = recovery_speed
                    recovery_analysis[pair][f'{crisis_id}_full_recovery'] = recovery_rate < crisis_rate * 0.5
                else:
                    # No data available
                    recovery_analysis[pair][f'{crisis_id}_crisis_rate'] = 0.0
                    recovery_analysis[pair][f'{crisis_id}_recovery_rate'] = 0.0
                    recovery_analysis[pair][f'{crisis_id}_recovery_ratio'] = 1.0
                    recovery_analysis[pair][f'{crisis_id}_recovery_speed'] = 0.0
                    recovery_analysis[pair][f'{crisis_id}_full_recovery'] = True
        
        if save_path:
            # Convert to DataFrame and save
            df = pd.DataFrame(recovery_analysis).T
            df.to_csv(save_path, index=True)
            print(f"✅ Crisis recovery analysis saved: {save_path}")
        
        # Print summary
        print(f"\n🔄 CRISIS RECOVERY SUMMARY ({recovery_window} days)")
        print("=" * 40)
        
        for crisis_id, crisis_info in self.crisis_definitions.items():
            recovery_ratios = [data.get(f'{crisis_id}_recovery_ratio', 1.0) 
                             for data in recovery_analysis.values()]
            full_recoveries = sum(1 for data in recovery_analysis.values() 
                                if data.get(f'{crisis_id}_full_recovery', False))
            
            avg_recovery_ratio = np.mean(recovery_ratios)
            total_pairs = len(recovery_analysis)
            
            print(f"{crisis_info['name']}:")
            print(f"  Average recovery ratio: {avg_recovery_ratio:.2f}")
            print(f"  Full recoveries: {full_recoveries}/{total_pairs} ({full_recoveries/total_pairs*100:.1f}%)")
            print()
        
        return recovery_analysis


# Example usage and testing functions
def create_sample_data_for_testing():
    """Create sample data for testing the visualization suite."""
    
    # Create sample S1 time series data
    dates = pd.date_range('2008-01-01', '2021-12-31', freq='D')
    
    sample_s1_data = {}
    asset_pairs = ['ADM_CORN', 'CAG_SOYB', 'CF_WEAT', 'DE_DBA', 'MOS_RICE']
    
    for pair in asset_pairs:
        # Generate realistic S1 values with crisis amplification
        np.random.seed(42)
        base_values = np.random.normal(1.5, 0.5, len(dates))
        
        # Add crisis amplification
        crisis_periods = [
            ('2008-09-01', '2009-03-31'),
            ('2010-05-01', '2012-12-31'), 
            ('2020-02-01', '2020-12-31')
        ]
        
        for start, end in crisis_periods:
            crisis_mask = (dates >= start) & (dates <= end)
            base_values[crisis_mask] *= np.random.uniform(1.5, 3.0)
        
        sample_s1_data[pair] = pd.Series(base_values, index=dates)
    
    return sample_s1_data


if __name__ == "__main__":
    # Example usage
    print("🎨 Agricultural Visualization Suite")
    print("=" * 40)
    
    # Create sample data
    sample_data = create_sample_data_for_testing()
    
    # Initialize visualization suite
    viz_suite = ComprehensiveVisualizationSuite()
    
    # Create sample analysis data structure
    analysis_data = {
        's1_data': sample_data,
        'crisis_s1_data': {pair: {
            '2008_financial': series['2008-09-01':'2009-03-31'],
            'eu_debt': series['2010-05-01':'2012-12-31'],
            'covid19': series['2020-02-01':'2020-12-31']
        } for pair, series in sample_data.items()},
        'correlation_matrix': pd.DataFrame(np.random.rand(5, 5), 
                                         index=list(sample_data.keys()), 
                                         columns=list(sample_data.keys()))
    }
    
    # Generate sample visualizations
    generated_files = viz_suite.generate_all_visualizations(analysis_data, "sample_visualizations")
    
    # Get summary
    summary = viz_suite.get_visualization_summary()
    print(f"\n📊 Generated {summary['total_figures']} visualizations")
    print("✅ Visualization suite ready for production use")