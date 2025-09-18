#!/usr/bin/env python3
"""
SEASONAL VISUALIZATION SUITE
============================

This module provides comprehensive visualization components for seasonal and geographic
analysis of agricultural cross-sector Bell inequality violations.

Key Features:
- Seasonal pattern visualizations
- Geographic distribution maps
- Seasonal modulation charts
- Regional crisis impact visualizations
- Interactive seasonal-geographic dashboards

Requirements Addressed:
- Create seasonal visualization components
- Seasonal effect detection visualization
- Geographic analysis visualization
- Regional crisis impact visualization

Authors: Agricultural Cross-Sector Analysis Team
Date: September 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
warnings.filterwarnings('ignore')

try:
    from .seasonal_geographic_analyzer import SeasonalGeographicResults, SeasonalAnalysisResults, GeographicAnalysisResults
except ImportError:
    from seasonal_geographic_analyzer import SeasonalGeographicResults, SeasonalAnalysisResults, GeographicAnalysisResults


class SeasonalVisualizationSuite:
    """
    Comprehensive visualization suite for seasonal and geographic analysis.
    
    This class provides publication-ready visualizations for seasonal effects,
    geographic patterns, and their interactions in agricultural cross-sector analysis.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 300):
        """
        Initialize the Seasonal Visualization Suite.
        
        Parameters:
        -----------
        figsize : Tuple[int, int], optional
            Default figure size. Default: (12, 8)
        dpi : int, optional
            Default DPI for saved figures. Default: 300
        """
        self.figsize = figsize
        self.dpi = dpi
        
        # Set style for publication-ready figures
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Define color schemes
        self.seasonal_colors = {
            'Winter': '#87CEEB',  # Light blue
            'Spring': '#98FB98',  # Pale green
            'Summer': '#FFD700',  # Gold
            'Fall': '#DEB887'     # Burlywood
        }
        
        self.regional_colors = {
            'North_America': '#FF6B6B',    # Red
            'South_America': '#4ECDC4',    # Teal
            'Europe': '#45B7D1',           # Blue
            'Asia_Pacific': '#96CEB4'      # Green
        }
        
        print("🎨 Seasonal Visualization Suite Initialized")
        print(f"   Figure size: {figsize}, DPI: {dpi}")
    
    def create_seasonal_violation_patterns(self, seasonal_results: SeasonalAnalysisResults,
                                         save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive seasonal violation pattern visualization.
        
        Parameters:
        -----------
        seasonal_results : SeasonalAnalysisResults
            Results from seasonal analysis
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        plt.Figure : The created figure
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Agricultural Seasonal Patterns in Bell Inequality Violations\n' +
                    'Quantum Correlation Strength Throughout Agricultural Cycles',
                    fontsize=16, fontweight='bold')
        
        # Plot 1: Seasonal violation rates
        seasonal_rates = seasonal_results.seasonal_violation_rates
        if seasonal_rates:
            seasons = list(seasonal_rates.keys())
            rates = list(seasonal_rates.values())
            colors = [self.seasonal_colors.get(season, '#888888') for season in seasons]
            
            bars = ax1.bar(seasons, rates, color=colors, alpha=0.7, edgecolor='black')
            ax1.set_ylabel('Bell Violation Rate (%)')
            ax1.set_title('Seasonal Bell Inequality Violation Rates')
            ax1.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, rate in zip(bars, rates):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Correlation modulation factors
        correlation_modulation = seasonal_results.correlation_modulation
        if correlation_modulation:
            seasons = list(correlation_modulation.keys())
            modulation = list(correlation_modulation.values())
            
            ax2.plot(seasons, modulation, 'o-', linewidth=3, markersize=8, color='darkblue')
            ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Baseline')
            ax2.set_ylabel('Correlation Strength Modulation')
            ax2.set_title('Seasonal Modulation of Quantum Correlations')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            # Highlight significant modulations
            for i, (season, mod) in enumerate(zip(seasons, modulation)):
                if abs(mod - 1.0) > 0.2:  # Significant modulation
                    ax2.annotate(f'{mod:.2f}', (i, mod), 
                               textcoords="offset points", xytext=(0,10), ha='center',
                               fontweight='bold', color='red')
        
        # Plot 3: Planting/Harvest cycle effects
        planting_harvest = seasonal_results.planting_harvest_effects
        if planting_harvest:
            periods = list(planting_harvest.keys())
            # Calculate average violation rates for each period
            period_rates = []
            for period_name, period_data in planting_harvest.items():
                s1_results = period_data.get('s1_results', {})
                if 'pair_results' in s1_results:
                    violations = []
                    for pair, results in s1_results['pair_results'].items():
                        if 's1_values' in results:
                            s1_values = np.array(results['s1_values'])
                            violation_rate = np.mean(np.abs(s1_values) > 2.0) * 100
                            violations.append(violation_rate)
                    period_rates.append(np.mean(violations) if violations else 0.0)
                else:
                    period_rates.append(0.0)
            
            # Create horizontal bar chart
            y_pos = np.arange(len(periods))
            bars = ax3.barh(y_pos, period_rates, color='lightcoral', alpha=0.7, edgecolor='black')
            ax3.set_yticks(y_pos)
            ax3.set_yticklabels([p.replace('_', ' ') for p in periods])
            ax3.set_xlabel('Bell Violation Rate (%)')
            ax3.set_title('Agricultural Cycle-Specific Effects')
            ax3.grid(True, alpha=0.3, axis='x')
            
            # Add value labels
            for i, (bar, rate) in enumerate(zip(bars, period_rates)):
                width = bar.get_width()
                ax3.text(width + 0.5, bar.get_y() + bar.get_height()/2.,
                        f'{rate:.1f}%', ha='left', va='center', fontweight='bold')
        
        # Plot 4: Statistical significance
        significance = seasonal_results.statistical_significance
        if significance:
            seasons = list(significance.keys())
            p_values = list(significance.values())
            
            # Convert p-values to -log10 for better visualization
            log_p_values = [-np.log10(max(p, 1e-10)) for p in p_values]
            
            bars = ax4.bar(seasons, log_p_values, color='mediumpurple', alpha=0.7, edgecolor='black')
            ax4.axhline(y=-np.log10(0.05), color='red', linestyle='--', alpha=0.7, 
                       label='p=0.05 threshold')
            ax4.set_ylabel('-log10(p-value)')
            ax4.set_title('Statistical Significance of Seasonal Effects')
            ax4.grid(True, alpha=0.3, axis='y')
            ax4.legend()
            
            # Add significance labels
            for bar, p_val in zip(bars, p_values):
                height = bar.get_height()
                significance_label = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        significance_label, ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"✅ Seasonal violation patterns saved: {save_path}")
        
        return fig
    
    def create_geographic_distribution_map(self, geographic_results: GeographicAnalysisResults,
                                         save_path: Optional[str] = None) -> plt.Figure:
        """
        Create geographic distribution visualization of regional patterns.
        
        Parameters:
        -----------
        geographic_results : GeographicAnalysisResults
            Results from geographic analysis
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        plt.Figure : The created figure
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Geographic Distribution of Agricultural Cross-Sector Correlations\n' +
                    'Regional Patterns in Quantum Entanglement',
                    fontsize=16, fontweight='bold')
        
        regional_patterns = geographic_results.regional_patterns
        
        # Plot 1: Regional company distribution
        if regional_patterns:
            regions = list(regional_patterns.keys())
            company_counts = [len(data.get('companies', [])) for data in regional_patterns.values()]
            colors = [self.regional_colors.get(region, '#888888') for region in regions]
            
            bars = ax1.bar(regions, company_counts, color=colors, alpha=0.7, edgecolor='black')
            ax1.set_ylabel('Number of Companies')
            ax1.set_title('Agricultural Companies by Geographic Region')
            ax1.grid(True, alpha=0.3, axis='y')
            
            # Rotate x-axis labels for better readability
            ax1.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, count in zip(bars, company_counts):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        str(count), ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Cross-regional correlations heatmap
        cross_regional = geographic_results.cross_regional_correlations
        if cross_regional:
            # Create correlation matrix
            regions = list(set([r for pair in cross_regional.keys() for r in pair]))
            corr_matrix = np.zeros((len(regions), len(regions)))
            
            for (r1, r2), corr in cross_regional.items():
                i1, i2 = regions.index(r1), regions.index(r2)
                corr_matrix[i1, i2] = corr
                corr_matrix[i2, i1] = corr  # Symmetric
            
            # Fill diagonal with 1.0
            np.fill_diagonal(corr_matrix, 1.0)
            
            im = ax2.imshow(corr_matrix, cmap='RdYlBu_r', vmin=0, vmax=1)
            ax2.set_xticks(range(len(regions)))
            ax2.set_yticks(range(len(regions)))
            ax2.set_xticklabels([r.replace('_', ' ') for r in regions], rotation=45)
            ax2.set_yticklabels([r.replace('_', ' ') for r in regions])
            ax2.set_title('Cross-Regional Correlation Matrix')
            
            # Add correlation values to heatmap
            for i in range(len(regions)):
                for j in range(len(regions)):
                    text = ax2.text(j, i, f'{corr_matrix[i, j]:.2f}',
                                   ha="center", va="center", color="black", fontweight='bold')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax2)
            cbar.set_label('Correlation Strength')
        
        # Plot 3: Regional crisis impact
        crisis_impact = geographic_results.crisis_impact_by_region
        if crisis_impact:
            regions = list(crisis_impact.keys())
            
            # Calculate average volatility increase across crises
            avg_volatility_increases = []
            for region_data in crisis_impact.values():
                volatility_increases = []
                for crisis_data in region_data.values():
                    if isinstance(crisis_data, dict) and 'volatility_increase' in crisis_data:
                        volatility_increases.append(crisis_data['volatility_increase'])
                avg_volatility_increases.append(np.mean(volatility_increases) if volatility_increases else 0)
            
            colors = [self.regional_colors.get(region, '#888888') for region in regions]
            bars = ax3.bar(regions, avg_volatility_increases, color=colors, alpha=0.7, edgecolor='black')
            ax3.set_ylabel('Average Volatility Increase (%)')
            ax3.set_title('Regional Crisis Impact (Volatility Increase)')
            ax3.grid(True, alpha=0.3, axis='y')
            ax3.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, increase in zip(bars, avg_volatility_increases):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{increase:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Plot 4: Production pattern effects
        production_effects = geographic_results.production_pattern_effects
        if production_effects:
            regions = list(production_effects.keys())
            
            # Create stacked bar chart for harvest season correlations
            harvest_data = {}
            all_seasons = set()
            
            for region, data in production_effects.items():
                harvest_correlations = data.get('harvest_correlations', {})
                harvest_data[region] = harvest_correlations
                all_seasons.update(harvest_correlations.keys())
            
            all_seasons = sorted(list(all_seasons))
            
            if all_seasons:
                bottom = np.zeros(len(regions))
                
                for season in all_seasons:
                    values = [harvest_data[region].get(season, 0) for region in regions]
                    ax4.bar(regions, values, bottom=bottom, 
                           label=season, alpha=0.7)
                    bottom += values
                
                ax4.set_ylabel('Harvest Season Correlation Strength')
                ax4.set_title('Production Pattern Effects by Region')
                ax4.legend()
                ax4.grid(True, alpha=0.3, axis='y')
                ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"✅ Geographic distribution map saved: {save_path}")
        
        return fig
    
    def create_seasonal_modulation_analysis(self, modulation_results: Dict[str, Dict],
                                          save_path: Optional[str] = None) -> plt.Figure:
        """
        Create seasonal modulation analysis visualization.
        
        Parameters:
        -----------
        modulation_results : Dict[str, Dict]
            Results from seasonal modulation analysis
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        plt.Figure : The created figure
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Seasonal Modulation of Quantum Correlation Strength\n' +
                    'Agricultural Cycle Effects on Bell Inequality Violations',
                    fontsize=16, fontweight='bold')
        
        # Plot 1: Monthly variations
        monthly_variations = modulation_results.get('monthly_variations', {})
        if monthly_variations:
            months = list(range(1, 13))
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            values = [monthly_variations.get(month, 0) for month in months]
            
            ax1.plot(month_names, values, 'o-', linewidth=3, markersize=8, color='darkgreen')
            ax1.set_ylabel('Average S1 Strength')
            ax1.set_title('Monthly Quantum Correlation Strength')
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis='x', rotation=45)
            
            # Highlight peak months
            max_idx = np.argmax(values)
            min_idx = np.argmin(values)
            ax1.annotate(f'Peak: {month_names[max_idx]}', 
                        xy=(max_idx, values[max_idx]), xytext=(max_idx, values[max_idx] + 0.1),
                        arrowprops=dict(arrowstyle='->', color='red'),
                        fontweight='bold', color='red')
            ax1.annotate(f'Low: {month_names[min_idx]}', 
                        xy=(min_idx, values[min_idx]), xytext=(min_idx, values[min_idx] - 0.1),
                        arrowprops=dict(arrowstyle='->', color='blue'),
                        fontweight='bold', color='blue')
        
        # Plot 2: Seasonal modulation factors
        seasonal_modulation = modulation_results.get('seasonal_modulation', {})
        if seasonal_modulation:
            seasons = list(seasonal_modulation.keys())
            modulation_factors = [data.get('modulation_factor', 1.0) for data in seasonal_modulation.values()]
            strength_changes = [data.get('strength_change', 0.0) for data in seasonal_modulation.values()]
            
            colors = [self.seasonal_colors.get(season, '#888888') for season in seasons]
            bars = ax2.bar(seasons, modulation_factors, color=colors, alpha=0.7, edgecolor='black')
            ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Baseline')
            ax2.set_ylabel('Modulation Factor')
            ax2.set_title('Seasonal Modulation Factors')
            ax2.grid(True, alpha=0.3, axis='y')
            ax2.legend()
            
            # Add percentage change labels
            for bar, change in zip(bars, strength_changes):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{change:+.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Plot 3: Yearly consistency
        yearly_consistency = modulation_results.get('yearly_consistency', {})
        if yearly_consistency:
            seasons = list(yearly_consistency.keys())
            consistency_scores = list(yearly_consistency.values())
            
            colors = [self.seasonal_colors.get(season, '#888888') for season in seasons]
            bars = ax3.bar(seasons, consistency_scores, color=colors, alpha=0.7, edgecolor='black')
            ax3.set_ylabel('Consistency Score (0-1)')
            ax3.set_title('Year-over-Year Consistency of Seasonal Patterns')
            ax3.grid(True, alpha=0.3, axis='y')
            ax3.set_ylim(0, 1)
            
            # Add consistency labels
            for bar, score in zip(bars, consistency_scores):
                height = bar.get_height()
                consistency_label = 'High' if score > 0.7 else 'Medium' if score > 0.4 else 'Low'
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        consistency_label, ha='center', va='bottom', fontweight='bold')
        
        # Plot 4: Significant variations
        significant_variations = modulation_results.get('significant_variations', {})
        if significant_variations:
            seasons = list(significant_variations.keys())
            strength_changes = [data.get('strength_change_percent', 0.0) for data in significant_variations.values()]
            significance_levels = [data.get('significance_level', 'Low') for data in significant_variations.values()]
            
            # Color by significance level
            colors = []
            for level in significance_levels:
                if level == 'High':
                    colors.append('red')
                elif level == 'Moderate':
                    colors.append('orange')
                else:
                    colors.append('yellow')
            
            bars = ax4.bar(seasons, strength_changes, color=colors, alpha=0.7, edgecolor='black')
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax4.set_ylabel('Strength Change (%)')
            ax4.set_title('Significant Seasonal Variations')
            ax4.grid(True, alpha=0.3, axis='y')
            
            # Add significance level labels
            for bar, level in zip(bars, significance_levels):
                height = bar.get_height()
                y_pos = height + 2 if height >= 0 else height - 5
                ax4.text(bar.get_x() + bar.get_width()/2., y_pos,
                        level, ha='center', va='bottom' if height >= 0 else 'top', 
                        fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"✅ Seasonal modulation analysis saved: {save_path}")
        
        return fig
    
    def create_regional_crisis_impact_visualization(self, crisis_results: Dict[str, Dict],
                                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        Create regional crisis impact visualization.
        
        Parameters:
        -----------
        crisis_results : Dict[str, Dict]
            Results from regional crisis impact analysis
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        plt.Figure : The created figure
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Regional Crisis Impact Analysis\n' +
                    'Geographic Variations in Crisis Response',
                    fontsize=16, fontweight='bold')
        
        if not crisis_results:
            # Create placeholder if no data
            for ax in [ax1, ax2, ax3, ax4]:
                ax.text(0.5, 0.5, 'No crisis data available', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=14, style='italic')
            plt.tight_layout()
            return fig
        
        regions = list(crisis_results.keys())
        
        # Plot 1: Crisis amplification by region
        amplification_data = {}
        for region, region_data in crisis_results.items():
            amplifications = []
            for crisis_name, crisis_data in region_data.items():
                if isinstance(crisis_data, dict) and 'amplification_factor' in crisis_data:
                    amplifications.append(crisis_data['amplification_factor'])
            amplification_data[region] = np.mean(amplifications) if amplifications else 1.0
        
        if amplification_data:
            regions_amp = list(amplification_data.keys())
            amplifications = list(amplification_data.values())
            colors = [self.regional_colors.get(region, '#888888') for region in regions_amp]
            
            bars = ax1.bar(regions_amp, amplifications, color=colors, alpha=0.7, edgecolor='black')
            ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='No amplification')
            ax1.set_ylabel('Crisis Amplification Factor')
            ax1.set_title('Average Crisis Amplification by Region')
            ax1.grid(True, alpha=0.3, axis='y')
            ax1.legend()
            ax1.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, amp in zip(bars, amplifications):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                        f'{amp:.2f}x', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Crisis-specific regional impacts
        # Create heatmap of crisis impacts by region
        all_crises = set()
        for region_data in crisis_results.values():
            all_crises.update(region_data.keys())
        all_crises = sorted(list(all_crises))
        
        if all_crises and regions:
            impact_matrix = np.zeros((len(regions), len(all_crises)))
            
            for i, region in enumerate(regions):
                for j, crisis in enumerate(all_crises):
                    crisis_data = crisis_results[region].get(crisis, {})
                    if isinstance(crisis_data, dict) and 'amplification_factor' in crisis_data:
                        impact_matrix[i, j] = crisis_data['amplification_factor']
                    else:
                        impact_matrix[i, j] = 1.0  # No amplification
            
            im = ax2.imshow(impact_matrix, cmap='Reds', vmin=1.0, vmax=np.max(impact_matrix))
            ax2.set_xticks(range(len(all_crises)))
            ax2.set_yticks(range(len(regions)))
            ax2.set_xticklabels([c.replace('_', ' ') for c in all_crises], rotation=45)
            ax2.set_yticklabels([r.replace('_', ' ') for r in regions])
            ax2.set_title('Crisis Impact Matrix (Amplification Factors)')
            
            # Add values to heatmap
            for i in range(len(regions)):
                for j in range(len(all_crises)):
                    text = ax2.text(j, i, f'{impact_matrix[i, j]:.2f}',
                                   ha="center", va="center", color="black", fontweight='bold')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax2)
            cbar.set_label('Amplification Factor')
        
        # Plot 3: Regional risk factors
        risk_factor_counts = {}
        all_risk_factors = set()
        
        for region, region_data in crisis_results.items():
            for crisis_name, crisis_data in region_data.items():
                if isinstance(crisis_data, dict) and 'regional_risk_factors' in crisis_data:
                    risk_factors = crisis_data['regional_risk_factors']
                    all_risk_factors.update(risk_factors)
                    for factor in risk_factors:
                        if factor not in risk_factor_counts:
                            risk_factor_counts[factor] = 0
                        risk_factor_counts[factor] += 1
        
        if risk_factor_counts:
            factors = list(risk_factor_counts.keys())
            counts = list(risk_factor_counts.values())
            
            bars = ax3.barh(factors, counts, color='lightcoral', alpha=0.7, edgecolor='black')
            ax3.set_xlabel('Frequency Across Regions/Crises')
            ax3.set_title('Regional Risk Factor Frequency')
            ax3.grid(True, alpha=0.3, axis='x')
            
            # Add count labels
            for bar, count in zip(bars, counts):
                width = bar.get_width()
                ax3.text(width + 0.1, bar.get_y() + bar.get_height()/2.,
                        str(count), ha='left', va='center', fontweight='bold')
        
        # Plot 4: Data availability by region and crisis
        data_availability = {}
        for region, region_data in crisis_results.items():
            region_availability = {}
            for crisis_name, crisis_data in region_data.items():
                if isinstance(crisis_data, dict) and 'data_points' in crisis_data:
                    region_availability[crisis_name] = crisis_data['data_points']
            data_availability[region] = region_availability
        
        if data_availability and all_crises:
            availability_matrix = np.zeros((len(regions), len(all_crises)))
            
            for i, region in enumerate(regions):
                for j, crisis in enumerate(all_crises):
                    availability_matrix[i, j] = data_availability.get(region, {}).get(crisis, 0)
            
            im = ax4.imshow(availability_matrix, cmap='Blues', vmin=0, vmax=np.max(availability_matrix))
            ax4.set_xticks(range(len(all_crises)))
            ax4.set_yticks(range(len(regions)))
            ax4.set_xticklabels([c.replace('_', ' ') for c in all_crises], rotation=45)
            ax4.set_yticklabels([r.replace('_', ' ') for r in regions])
            ax4.set_title('Data Availability (Number of Observations)')
            
            # Add values to heatmap
            for i in range(len(regions)):
                for j in range(len(all_crises)):
                    text = ax4.text(j, i, f'{int(availability_matrix[i, j])}',
                                   ha="center", va="center", color="white" if availability_matrix[i, j] > np.max(availability_matrix)/2 else "black", 
                                   fontweight='bold')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax4)
            cbar.set_label('Number of Observations')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"✅ Regional crisis impact visualization saved: {save_path}")
        
        return fig
    
    def create_comprehensive_seasonal_geographic_dashboard(self, results: SeasonalGeographicResults,
                                                         save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive dashboard combining seasonal and geographic analysis.
        
        Parameters:
        -----------
        results : SeasonalGeographicResults
            Complete seasonal and geographic analysis results
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        plt.Figure : The created figure
        """
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        fig.suptitle('Comprehensive Seasonal-Geographic Analysis Dashboard\n' +
                    'Agricultural Cross-Sector Quantum Correlations',
                    fontsize=18, fontweight='bold')
        
        # Top row: Seasonal analysis (spans 2 columns each)
        ax1 = fig.add_subplot(gs[0, :2])
        ax2 = fig.add_subplot(gs[0, 2:])
        
        # Middle row: Geographic analysis (spans 2 columns each)
        ax3 = fig.add_subplot(gs[1, :2])
        ax4 = fig.add_subplot(gs[1, 2:])
        
        # Bottom row: Interactions (4 separate plots)
        ax5 = fig.add_subplot(gs[2, 0])
        ax6 = fig.add_subplot(gs[2, 1])
        ax7 = fig.add_subplot(gs[2, 2])
        ax8 = fig.add_subplot(gs[2, 3])
        
        # Plot 1: Seasonal violation rates
        seasonal_rates = results.seasonal_results.seasonal_violation_rates
        if seasonal_rates:
            seasons = list(seasonal_rates.keys())
            rates = list(seasonal_rates.values())
            colors = [self.seasonal_colors.get(season, '#888888') for season in seasons]
            
            bars = ax1.bar(seasons, rates, color=colors, alpha=0.7, edgecolor='black')
            ax1.set_ylabel('Bell Violation Rate (%)')
            ax1.set_title('Seasonal Bell Violation Patterns')
            ax1.grid(True, alpha=0.3, axis='y')
            
            for bar, rate in zip(bars, rates):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Seasonal modulation
        correlation_modulation = results.seasonal_results.correlation_modulation
        if correlation_modulation:
            seasons = list(correlation_modulation.keys())
            modulation = list(correlation_modulation.values())
            
            ax2.plot(seasons, modulation, 'o-', linewidth=3, markersize=8, color='darkblue')
            ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.7)
            ax2.set_ylabel('Modulation Factor')
            ax2.set_title('Seasonal Correlation Modulation')
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Regional distribution
        regional_patterns = results.geographic_results.regional_patterns
        if regional_patterns:
            regions = list(regional_patterns.keys())
            company_counts = [len(data.get('companies', [])) for data in regional_patterns.values()]
            colors = [self.regional_colors.get(region, '#888888') for region in regions]
            
            bars = ax3.bar([r.replace('_', ' ') for r in regions], company_counts, 
                          color=colors, alpha=0.7, edgecolor='black')
            ax3.set_ylabel('Number of Companies')
            ax3.set_title('Regional Company Distribution')
            ax3.grid(True, alpha=0.3, axis='y')
            ax3.tick_params(axis='x', rotation=45)
        
        # Plot 4: Cross-regional correlations
        cross_regional = results.geographic_results.cross_regional_correlations
        if cross_regional:
            pairs = list(cross_regional.keys())
            correlations = list(cross_regional.values())
            
            pair_labels = [f"{p[0].replace('_', ' ')}\n↔\n{p[1].replace('_', ' ')}" for p in pairs]
            
            bars = ax4.bar(range(len(pairs)), correlations, color='lightblue', alpha=0.7, edgecolor='black')
            ax4.set_xticks(range(len(pairs)))
            ax4.set_xticklabels(pair_labels, fontsize=8)
            ax4.set_ylabel('Correlation Strength')
            ax4.set_title('Cross-Regional Correlations')
            ax4.grid(True, alpha=0.3, axis='y')
        
        # Plot 5: Seasonal-geographic interactions summary
        interactions = results.seasonal_geographic_interactions
        if interactions:
            interaction_strengths = []
            region_names = []
            
            for region, region_interactions in interactions.items():
                avg_strength = np.mean([data.get('interaction_strength', 1.0) 
                                      for data in region_interactions.values()])
                interaction_strengths.append(avg_strength)
                region_names.append(region.replace('_', ' '))
            
            bars = ax5.bar(region_names, interaction_strengths, 
                          color='mediumpurple', alpha=0.7, edgecolor='black')
            ax5.set_ylabel('Interaction\nStrength')
            ax5.set_title('Seasonal-Geographic\nInteractions')
            ax5.tick_params(axis='x', rotation=45, labelsize=8)
            ax5.grid(True, alpha=0.3, axis='y')
        
        # Plot 6: Crisis amplification summary
        crisis_amplification = results.crisis_seasonal_amplification
        if crisis_amplification:
            regions = list(crisis_amplification.keys())
            avg_amplifications = []
            
            for region_data in crisis_amplification.values():
                amplifications = []
                for crisis_data in region_data.values():
                    if isinstance(crisis_data, dict) and 'amplification_factor' in crisis_data:
                        amplifications.append(crisis_data['amplification_factor'])
                avg_amplifications.append(np.mean(amplifications) if amplifications else 1.0)
            
            bars = ax6.bar([r.replace('_', ' ') for r in regions], avg_amplifications,
                          color='lightcoral', alpha=0.7, edgecolor='black')
            ax6.axhline(y=1.0, color='red', linestyle='--', alpha=0.7)
            ax6.set_ylabel('Crisis\nAmplification')
            ax6.set_title('Regional Crisis\nAmplification')
            ax6.tick_params(axis='x', rotation=45, labelsize=8)
            ax6.grid(True, alpha=0.3, axis='y')
        
        # Plot 7: Regional seasonal variations
        regional_variations = results.regional_seasonal_variations
        if regional_variations:
            regions = list(regional_variations.keys())
            climate_factors = []
            
            for region_data in regional_variations.values():
                seasonal_effects = region_data.get('seasonal_effects', {})
                avg_factor = np.mean([effect.get('climate_factor', 1.0) 
                                    for effect in seasonal_effects.values()])
                climate_factors.append(avg_factor)
            
            bars = ax7.bar([r.replace('_', ' ') for r in regions], climate_factors,
                          color='lightgreen', alpha=0.7, edgecolor='black')
            ax7.set_ylabel('Climate\nFactor')
            ax7.set_title('Regional Climate\nEffects')
            ax7.tick_params(axis='x', rotation=45, labelsize=8)
            ax7.grid(True, alpha=0.3, axis='y')
        
        # Plot 8: Summary statistics
        # Create a text summary of key findings
        ax8.axis('off')
        
        summary_text = "KEY FINDINGS:\n\n"
        
        # Seasonal findings
        if seasonal_rates:
            max_season = max(seasonal_rates, key=seasonal_rates.get)
            min_season = min(seasonal_rates, key=seasonal_rates.get)
            summary_text += f"🌱 SEASONAL:\n"
            summary_text += f"• Peak: {max_season}\n  ({seasonal_rates[max_season]:.1f}%)\n"
            summary_text += f"• Low: {min_season}\n  ({seasonal_rates[min_season]:.1f}%)\n\n"
        
        # Geographic findings
        if regional_patterns:
            max_region = max(regional_patterns.keys(), 
                           key=lambda r: len(regional_patterns[r].get('companies', [])))
            summary_text += f"🌍 GEOGRAPHIC:\n"
            summary_text += f"• Most companies:\n  {max_region.replace('_', ' ')}\n"
            summary_text += f"• Regions analyzed: {len(regional_patterns)}\n\n"
        
        # Cross-regional findings
        if cross_regional:
            max_corr_pair = max(cross_regional, key=cross_regional.get)
            summary_text += f"🔄 CORRELATIONS:\n"
            summary_text += f"• Strongest:\n  {max_corr_pair[0].replace('_', ' ')}\n"
            summary_text += f"  ↔ {max_corr_pair[1].replace('_', ' ')}\n"
            summary_text += f"  ({cross_regional[max_corr_pair]:.3f})\n"
        
        ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"✅ Comprehensive seasonal-geographic dashboard saved: {save_path}")
        
        return fig
    
    def generate_all_seasonal_geographic_visualizations(self, results: SeasonalGeographicResults,
                                                      output_dir: str = "seasonal_geographic_analysis") -> List[str]:
        """
        Generate all seasonal and geographic visualizations.
        
        Parameters:
        -----------
        results : SeasonalGeographicResults
            Complete seasonal and geographic analysis results
        output_dir : str, optional
            Output directory for visualizations
            
        Returns:
        --------
        List[str] : List of generated file paths
        """
        import os
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        generated_files = []
        
        print("🎨 Generating comprehensive seasonal-geographic visualizations...")
        
        # 1. Seasonal violation patterns
        file_path = os.path.join(output_dir, "seasonal_violation_patterns.png")
        self.create_seasonal_violation_patterns(results.seasonal_results, file_path)
        generated_files.append(file_path)
        
        # 2. Geographic distribution map
        file_path = os.path.join(output_dir, "geographic_distribution_map.png")
        self.create_geographic_distribution_map(results.geographic_results, file_path)
        generated_files.append(file_path)
        
        # 3. Seasonal modulation analysis
        if hasattr(results, 'seasonal_modulation_results'):
            file_path = os.path.join(output_dir, "seasonal_modulation_analysis.png")
            # Note: This would need the modulation results to be passed separately
            # self.create_seasonal_modulation_analysis(results.seasonal_modulation_results, file_path)
            # generated_files.append(file_path)
        
        # 4. Regional crisis impact
        if results.crisis_seasonal_amplification:
            file_path = os.path.join(output_dir, "regional_crisis_impact.png")
            self.create_regional_crisis_impact_visualization(results.crisis_seasonal_amplification, file_path)
            generated_files.append(file_path)
        
        # 5. Comprehensive dashboard
        file_path = os.path.join(output_dir, "comprehensive_seasonal_geographic_dashboard.png")
        self.create_comprehensive_seasonal_geographic_dashboard(results, file_path)
        generated_files.append(file_path)
        
        print(f"✅ Generated {len(generated_files)} seasonal-geographic visualizations")
        print(f"   Output directory: {output_dir}")
        
        return generated_files


if __name__ == "__main__":
    # Example usage
    visualizer = SeasonalVisualizationSuite()
    print("Seasonal Visualization Suite initialized successfully!")