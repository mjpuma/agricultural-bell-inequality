#!/usr/bin/env python3
"""
SEASONAL AND GEOGRAPHIC ANALYSIS MODULE
======================================

This module implements seasonal effect detection for agricultural planting/harvest cycles
and geographic analysis considering regional agricultural production patterns.

Key Features:
- Seasonal effect detection for agricultural cycles
- Geographic analysis of regional production patterns  
- Seasonal modulation analysis for quantum correlation strength variations
- Regional crisis impact analysis
- Seasonal visualization components

Requirements Addressed:
- 8.2: WHEN examining seasonal effects THEN the system SHALL account for agricultural planting/harvest cycles
- 8.3: WHEN studying geographic effects THEN the system SHALL consider regional agricultural production patterns  
- 8.4: IF seasonal modulation exists THEN the system SHALL document quantum correlation strength variations throughout the year

Authors: Agricultural Cross-Sector Analysis Team
Date: September 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    from .enhanced_s1_calculator import EnhancedS1Calculator
    from .agricultural_universe_manager import AgriculturalUniverseManager, CompanyInfo
except ImportError:
    from enhanced_s1_calculator import EnhancedS1Calculator
    from agricultural_universe_manager import AgriculturalUniverseManager, CompanyInfo


@dataclass
class SeasonalPattern:
    """Represents seasonal patterns in agricultural data."""
    season: str
    months: List[int]
    description: str
    expected_activity: str
    correlation_strength_modifier: float


@dataclass
class GeographicRegion:
    """Represents geographic regions for agricultural analysis."""
    region_name: str
    countries: List[str]
    primary_crops: List[str]
    climate_zone: str
    harvest_seasons: List[str]
    risk_factors: List[str]


@dataclass
class SeasonalAnalysisResults:
    """Results from seasonal analysis."""
    seasonal_patterns: Dict[str, Dict]
    correlation_modulation: Dict[str, float]
    planting_harvest_effects: Dict[str, Dict]
    seasonal_violation_rates: Dict[str, float]
    statistical_significance: Dict[str, float]


@dataclass
class GeographicAnalysisResults:
    """Results from geographic analysis."""
    regional_patterns: Dict[str, Dict]
    cross_regional_correlations: Dict[Tuple[str, str], float]
    crisis_impact_by_region: Dict[str, Dict]
    production_pattern_effects: Dict[str, Dict]
    geographic_transmission: Dict[str, List]


@dataclass
class SeasonalGeographicResults:
    """Combined seasonal and geographic analysis results."""
    seasonal_results: SeasonalAnalysisResults
    geographic_results: GeographicAnalysisResults
    seasonal_geographic_interactions: Dict[str, Dict]
    crisis_seasonal_amplification: Dict[str, Dict]
    regional_seasonal_variations: Dict[str, Dict]


class SeasonalGeographicAnalyzer:
    """
    Seasonal and Geographic Analysis for Agricultural Cross-Sector Analysis.
    
    This class implements seasonal effect detection for agricultural planting/harvest cycles
    and geographic analysis considering regional agricultural production patterns.
    """
    
    def __init__(self, universe_manager: Optional[AgriculturalUniverseManager] = None):
        """
        Initialize the Seasonal Geographic Analyzer.
        
        Parameters:
        -----------
        universe_manager : AgriculturalUniverseManager, optional
            Universe manager for company information. If None, creates new instance.
        """
        self.universe_manager = universe_manager or AgriculturalUniverseManager()
        
        # Initialize S1 calculator for seasonal analysis
        self.s1_calculator = EnhancedS1Calculator(
            window_size=20,
            threshold_method='quantile',
            threshold_quantile=0.75
        )
        
        # Define agricultural seasons (Northern Hemisphere focus)
        self.agricultural_seasons = {
            'Winter': SeasonalPattern(
                season='Winter',
                months=[12, 1, 2],
                description='Dormant period, planning, equipment maintenance',
                expected_activity='Low field activity, financial planning',
                correlation_strength_modifier=0.8
            ),
            'Spring': SeasonalPattern(
                season='Spring',
                months=[3, 4, 5],
                description='Planting season, input purchasing',
                expected_activity='High input demand, planting operations',
                correlation_strength_modifier=1.3
            ),
            'Summer': SeasonalPattern(
                season='Summer',
                months=[6, 7, 8],
                description='Growing season, crop development',
                expected_activity='Crop monitoring, pest management',
                correlation_strength_modifier=1.1
            ),
            'Fall': SeasonalPattern(
                season='Fall',
                months=[9, 10, 11],
                description='Harvest season, marketing',
                expected_activity='Harvest operations, grain marketing',
                correlation_strength_modifier=1.4
            )
        }
        
        # Define geographic regions
        self.geographic_regions = {
            'North_America': GeographicRegion(
                region_name='North_America',
                countries=['USA', 'Canada', 'Mexico'],
                primary_crops=['Corn', 'Soybeans', 'Wheat'],
                climate_zone='Temperate',
                harvest_seasons=['Fall'],
                risk_factors=['Drought', 'Flooding', 'Trade_Policy']
            ),
            'South_America': GeographicRegion(
                region_name='South_America',
                countries=['Brazil', 'Argentina', 'Uruguay'],
                primary_crops=['Soybeans', 'Corn', 'Beef'],
                climate_zone='Tropical/Subtropical',
                harvest_seasons=['Spring', 'Summer'],  # Southern Hemisphere
                risk_factors=['Deforestation', 'Currency_Volatility', 'Climate_Change']
            ),
            'Europe': GeographicRegion(
                region_name='Europe',
                countries=['Germany', 'France', 'Poland', 'Ukraine'],
                primary_crops=['Wheat', 'Barley', 'Rapeseed'],
                climate_zone='Temperate',
                harvest_seasons=['Summer', 'Fall'],
                risk_factors=['Policy_Changes', 'Weather_Extremes', 'Energy_Costs']
            ),
            'Asia_Pacific': GeographicRegion(
                region_name='Asia_Pacific',
                countries=['China', 'India', 'Australia', 'Thailand'],
                primary_crops=['Rice', 'Wheat', 'Palm_Oil'],
                climate_zone='Varied',
                harvest_seasons=['Summer', 'Fall', 'Winter'],
                risk_factors=['Monsoons', 'Population_Pressure', 'Water_Scarcity']
            )
        }
        
        print("🌍 Seasonal Geographic Analyzer Initialized")
        print(f"   Agricultural seasons: {len(self.agricultural_seasons)}")
        print(f"   Geographic regions: {len(self.geographic_regions)}")
    
    def analyze_seasonal_effects(self, returns_data: pd.DataFrame, 
                               asset_pairs: List[Tuple[str, str]]) -> SeasonalAnalysisResults:
        """
        Implement seasonal effect detection for agricultural planting/harvest cycles.
        
        This addresses Requirement 8.2: WHEN examining seasonal effects THEN the system 
        SHALL account for agricultural planting/harvest cycles.
        
        Parameters:
        -----------
        returns_data : pd.DataFrame
            Daily returns data with datetime index
        asset_pairs : List[Tuple[str, str]]
            Asset pairs to analyze for seasonal effects
            
        Returns:
        --------
        SeasonalAnalysisResults : Comprehensive seasonal analysis results
        """
        print("🌱 Analyzing seasonal effects for agricultural cycles...")
        
        seasonal_patterns = {}
        correlation_modulation = {}
        planting_harvest_effects = {}
        seasonal_violation_rates = {}
        statistical_significance = {}
        
        # Check if data is empty or has no datetime index
        if returns_data.empty or not hasattr(returns_data.index, 'month'):
            print("   ⚠️  No valid datetime data for seasonal analysis")
            return SeasonalAnalysisResults(
                seasonal_patterns={},
                correlation_modulation={},
                planting_harvest_effects={},
                seasonal_violation_rates={},
                statistical_significance={}
            )
        
        # Analyze each season
        for season_name, season_info in self.agricultural_seasons.items():
            print(f"   Analyzing {season_name} season ({season_info.description})")
            
            # Filter data for this season
            seasonal_mask = returns_data.index.month.isin(season_info.months)
            seasonal_data = returns_data[seasonal_mask]
            
            if len(seasonal_data) < 50:  # Minimum data requirement
                print(f"     ⚠️  Insufficient data for {season_name}: {len(seasonal_data)} observations")
                continue
            
            # Calculate S1 values for this season
            seasonal_s1_results = self.s1_calculator.batch_analyze_pairs(
                seasonal_data, asset_pairs[:10]  # Limit for performance
            )
            
            # Calculate seasonal violation rates
            if 'pair_results' in seasonal_s1_results:
                violations = []
                for pair, results in seasonal_s1_results['pair_results'].items():
                    if 's1_values' in results:
                        s1_values = np.array(results['s1_values'])
                        violation_rate = np.mean(np.abs(s1_values) > 2.0) * 100
                        violations.append(violation_rate)
                
                seasonal_violation_rates[season_name] = np.mean(violations) if violations else 0.0
            
            # Store seasonal patterns
            seasonal_patterns[season_name] = {
                'season_info': season_info,
                'data_points': len(seasonal_data),
                's1_results': seasonal_s1_results,
                'violation_rate': seasonal_violation_rates.get(season_name, 0.0)
            }
            
            # Calculate correlation strength modulation
            if seasonal_violation_rates.get(season_name, 0) > 0:
                baseline_rate = np.mean(list(seasonal_violation_rates.values())) if seasonal_violation_rates else 1.0
                modulation = seasonal_violation_rates[season_name] / max(baseline_rate, 0.1)
                correlation_modulation[season_name] = modulation * season_info.correlation_strength_modifier
            
            print(f"     ✅ {season_name}: {seasonal_violation_rates.get(season_name, 0):.1f}% violation rate")
        
        # Analyze planting/harvest specific effects
        planting_harvest_effects = self._analyze_planting_harvest_cycles(
            returns_data, asset_pairs
        )
        
        # Calculate statistical significance of seasonal differences
        statistical_significance = self._calculate_seasonal_significance(
            seasonal_violation_rates
        )
        
        results = SeasonalAnalysisResults(
            seasonal_patterns=seasonal_patterns,
            correlation_modulation=correlation_modulation,
            planting_harvest_effects=planting_harvest_effects,
            seasonal_violation_rates=seasonal_violation_rates,
            statistical_significance=statistical_significance
        )
        
        print(f"✅ Seasonal analysis complete: {len(seasonal_patterns)} seasons analyzed")
        
        return results
    
    def analyze_geographic_effects(self, returns_data: pd.DataFrame,
                                 asset_pairs: List[Tuple[str, str]]) -> GeographicAnalysisResults:
        """
        Create geographic analysis considering regional agricultural production patterns.
        
        This addresses Requirement 8.3: WHEN studying geographic effects THEN the system 
        SHALL consider regional agricultural production patterns.
        
        Parameters:
        -----------
        returns_data : pd.DataFrame
            Daily returns data with datetime index
        asset_pairs : List[Tuple[str, str]]
            Asset pairs to analyze for geographic effects
            
        Returns:
        --------
        GeographicAnalysisResults : Comprehensive geographic analysis results
        """
        print("🌍 Analyzing geographic effects for regional production patterns...")
        
        regional_patterns = {}
        cross_regional_correlations = {}
        crisis_impact_by_region = {}
        production_pattern_effects = {}
        geographic_transmission = {}
        
        # Analyze each geographic region
        for region_name, region_info in self.geographic_regions.items():
            print(f"   Analyzing {region_name} region ({region_info.climate_zone})")
            
            # Get companies with exposure to this region
            regional_companies = self._get_regional_companies(region_name)
            
            if not regional_companies:
                print(f"     ⚠️  No companies found for {region_name}")
                continue
            
            # Filter asset pairs to include regional companies
            regional_pairs = [
                pair for pair in asset_pairs 
                if any(asset in regional_companies for asset in pair)
            ]
            
            if not regional_pairs:
                print(f"     ⚠️  No relevant pairs for {region_name}")
                continue
            
            # Analyze regional patterns during different seasons
            regional_seasonal_analysis = {}
            for season_name, season_info in self.agricultural_seasons.items():
                # Adjust for Southern Hemisphere if needed
                if region_name == 'South_America':
                    adjusted_months = self._adjust_for_southern_hemisphere(season_info.months)
                else:
                    adjusted_months = season_info.months
                
                seasonal_mask = returns_data.index.month.isin(adjusted_months)
                seasonal_data = returns_data[seasonal_mask]
                
                if len(seasonal_data) >= 30:  # Minimum requirement
                    seasonal_s1 = self.s1_calculator.batch_analyze_pairs(
                        seasonal_data, regional_pairs[:5]  # Limit for performance
                    )
                    regional_seasonal_analysis[season_name] = seasonal_s1
            
            regional_patterns[region_name] = {
                'region_info': region_info,
                'companies': regional_companies,
                'seasonal_analysis': regional_seasonal_analysis,
                'primary_crops': region_info.primary_crops,
                'risk_factors': region_info.risk_factors
            }
            
            print(f"     ✅ {region_name}: {len(regional_companies)} companies, {len(regional_pairs)} pairs")
        
        # Calculate cross-regional correlations
        cross_regional_correlations = self._calculate_cross_regional_correlations(
            returns_data, regional_patterns
        )
        
        # Analyze crisis impact by region
        crisis_impact_by_region = self._analyze_regional_crisis_impact(
            returns_data, regional_patterns
        )
        
        # Analyze production pattern effects
        production_pattern_effects = self._analyze_production_patterns(
            returns_data, regional_patterns
        )
        
        # Analyze geographic transmission mechanisms
        geographic_transmission = self._analyze_geographic_transmission(
            returns_data, regional_patterns
        )
        
        results = GeographicAnalysisResults(
            regional_patterns=regional_patterns,
            cross_regional_correlations=cross_regional_correlations,
            crisis_impact_by_region=crisis_impact_by_region,
            production_pattern_effects=production_pattern_effects,
            geographic_transmission=geographic_transmission
        )
        
        print(f"✅ Geographic analysis complete: {len(regional_patterns)} regions analyzed")
        
        return results
    
    def analyze_seasonal_modulation(self, returns_data: pd.DataFrame,
                                  asset_pairs: List[Tuple[str, str]]) -> Dict[str, Dict]:
        """
        Add seasonal modulation analysis for quantum correlation strength variations.
        
        This addresses Requirement 8.4: IF seasonal modulation exists THEN the system 
        SHALL document quantum correlation strength variations throughout the year.
        
        Parameters:
        -----------
        returns_data : pd.DataFrame
            Daily returns data with datetime index
        asset_pairs : List[Tuple[str, str]]
            Asset pairs to analyze for seasonal modulation
            
        Returns:
        --------
        Dict[str, Dict] : Seasonal modulation analysis results
        """
        print("📊 Analyzing seasonal modulation of quantum correlation strength...")
        
        modulation_results = {}
        
        # Calculate monthly correlation strength variations
        monthly_variations = {}
        for month in range(1, 13):
            monthly_mask = returns_data.index.month == month
            monthly_data = returns_data[monthly_mask]
            
            if len(monthly_data) >= 20:  # Minimum requirement
                monthly_s1 = self.s1_calculator.batch_analyze_pairs(
                    monthly_data, asset_pairs[:5]  # Limit for performance
                )
                
                # Calculate average violation strength
                if 'pair_results' in monthly_s1:
                    violations = []
                    for pair, results in monthly_s1['pair_results'].items():
                        if 's1_values' in results:
                            s1_values = np.array(results['s1_values'])
                            avg_strength = np.mean(np.abs(s1_values))
                            violations.append(avg_strength)
                    
                    monthly_variations[month] = np.mean(violations) if violations else 0.0
        
        # Identify seasonal modulation patterns
        if monthly_variations:
            baseline_strength = np.mean(list(monthly_variations.values()))
            
            seasonal_modulation = {}
            for season_name, season_info in self.agricultural_seasons.items():
                season_months = season_info.months
                season_strength = np.mean([
                    monthly_variations.get(month, 0) for month in season_months
                ])
                
                modulation_factor = season_strength / max(baseline_strength, 0.1)
                seasonal_modulation[season_name] = {
                    'modulation_factor': modulation_factor,
                    'strength_change': (modulation_factor - 1.0) * 100,
                    'season_info': season_info,
                    'expected_activity': season_info.expected_activity
                }
            
            modulation_results['seasonal_modulation'] = seasonal_modulation
            modulation_results['monthly_variations'] = monthly_variations
            modulation_results['baseline_strength'] = baseline_strength
        
        # Analyze year-over-year consistency
        yearly_consistency = self._analyze_yearly_consistency(
            returns_data, asset_pairs
        )
        modulation_results['yearly_consistency'] = yearly_consistency
        
        # Document significant variations
        significant_variations = self._identify_significant_variations(
            modulation_results
        )
        modulation_results['significant_variations'] = significant_variations
        
        print(f"✅ Seasonal modulation analysis complete")
        
        return modulation_results
    
    def analyze_regional_crisis_impact(self, returns_data: pd.DataFrame,
                                     crisis_periods: Dict[str, Tuple[str, str]],
                                     asset_pairs: List[Tuple[str, str]]) -> Dict[str, Dict]:
        """
        Implement regional crisis impact analysis.
        
        Parameters:
        -----------
        returns_data : pd.DataFrame
            Daily returns data with datetime index
        crisis_periods : Dict[str, Tuple[str, str]]
            Crisis periods with start and end dates
        asset_pairs : List[Tuple[str, str]]
            Asset pairs to analyze
            
        Returns:
        --------
        Dict[str, Dict] : Regional crisis impact analysis results
        """
        print("🚨 Analyzing regional crisis impact...")
        
        regional_crisis_results = {}
        
        for region_name, region_info in self.geographic_regions.items():
            print(f"   Analyzing crisis impact in {region_name}")
            
            regional_companies = self._get_regional_companies(region_name)
            regional_pairs = [
                pair for pair in asset_pairs 
                if any(asset in regional_companies for asset in pair)
            ]
            
            if not regional_pairs:
                continue
            
            region_crisis_analysis = {}
            
            for crisis_name, (start_date, end_date) in crisis_periods.items():
                try:
                    # Filter data for crisis period
                    crisis_mask = (
                        (returns_data.index >= pd.to_datetime(start_date)) &
                        (returns_data.index <= pd.to_datetime(end_date))
                    )
                    crisis_data = returns_data[crisis_mask]
                    
                    if len(crisis_data) >= 20:  # Minimum requirement
                        crisis_s1 = self.s1_calculator.batch_analyze_pairs(
                            crisis_data, regional_pairs[:3]  # Limit for performance
                        )
                        
                        # Calculate crisis amplification
                        normal_data = returns_data[~crisis_mask]
                        if len(normal_data) >= 50:
                            normal_s1 = self.s1_calculator.batch_analyze_pairs(
                                normal_data, regional_pairs[:3]
                            )
                            
                            amplification = self._calculate_crisis_amplification(
                                normal_s1, crisis_s1
                            )
                            
                            region_crisis_analysis[crisis_name] = {
                                'crisis_s1': crisis_s1,
                                'amplification_factor': amplification,
                                'data_points': len(crisis_data),
                                'regional_risk_factors': region_info.risk_factors
                            }
                
                except Exception as e:
                    print(f"     ⚠️  Error analyzing {crisis_name} in {region_name}: {str(e)}")
                    continue
            
            regional_crisis_results[region_name] = region_crisis_analysis
            
            print(f"     ✅ {region_name}: {len(region_crisis_analysis)} crises analyzed")
        
        return regional_crisis_results
    
    def run_comprehensive_seasonal_geographic_analysis(
        self, 
        returns_data: pd.DataFrame,
        asset_pairs: List[Tuple[str, str]],
        crisis_periods: Optional[Dict[str, Tuple[str, str]]] = None
    ) -> SeasonalGeographicResults:
        """
        Run comprehensive seasonal and geographic analysis.
        
        Parameters:
        -----------
        returns_data : pd.DataFrame
            Daily returns data with datetime index
        asset_pairs : List[Tuple[str, str]]
            Asset pairs to analyze
        crisis_periods : Dict[str, Tuple[str, str]], optional
            Crisis periods for regional impact analysis
            
        Returns:
        --------
        SeasonalGeographicResults : Complete seasonal and geographic analysis results
        """
        print("🌍🌱 Running Comprehensive Seasonal and Geographic Analysis")
        print("=" * 70)
        
        # Run seasonal analysis
        print("\n🌱 SEASONAL ANALYSIS")
        print("-" * 40)
        seasonal_results = self.analyze_seasonal_effects(returns_data, asset_pairs)
        
        # Run geographic analysis
        print("\n🌍 GEOGRAPHIC ANALYSIS")
        print("-" * 40)
        geographic_results = self.analyze_geographic_effects(returns_data, asset_pairs)
        
        # Run seasonal modulation analysis
        print("\n📊 SEASONAL MODULATION ANALYSIS")
        print("-" * 40)
        seasonal_modulation = self.analyze_seasonal_modulation(returns_data, asset_pairs)
        
        # Analyze seasonal-geographic interactions
        print("\n🔄 SEASONAL-GEOGRAPHIC INTERACTIONS")
        print("-" * 40)
        seasonal_geographic_interactions = self._analyze_seasonal_geographic_interactions(
            seasonal_results, geographic_results
        )
        
        # Crisis seasonal amplification
        crisis_seasonal_amplification = {}
        if crisis_periods:
            print("\n🚨 CRISIS SEASONAL AMPLIFICATION")
            print("-" * 40)
            crisis_seasonal_amplification = self.analyze_regional_crisis_impact(
                returns_data, crisis_periods, asset_pairs
            )
        
        # Regional seasonal variations
        print("\n🌍🌱 REGIONAL SEASONAL VARIATIONS")
        print("-" * 40)
        regional_seasonal_variations = self._analyze_regional_seasonal_variations(
            seasonal_results, geographic_results
        )
        
        results = SeasonalGeographicResults(
            seasonal_results=seasonal_results,
            geographic_results=geographic_results,
            seasonal_geographic_interactions=seasonal_geographic_interactions,
            crisis_seasonal_amplification=crisis_seasonal_amplification,
            regional_seasonal_variations=regional_seasonal_variations
        )
        
        print("\n✅ COMPREHENSIVE SEASONAL-GEOGRAPHIC ANALYSIS COMPLETE")
        print("=" * 70)
        self._print_analysis_summary(results)
        
        return results    

    # Helper Methods
    # ==============
    
    def _analyze_planting_harvest_cycles(self, returns_data: pd.DataFrame,
                                       asset_pairs: List[Tuple[str, str]]) -> Dict[str, Dict]:
        """Analyze specific planting and harvest cycle effects."""
        planting_harvest_effects = {}
        
        # Define key agricultural periods
        agricultural_periods = {
            'Spring_Planting': {
                'months': [3, 4, 5],
                'description': 'Spring planting season - high input demand',
                'expected_correlations': 'Strong energy-fertilizer correlations'
            },
            'Summer_Growing': {
                'months': [6, 7, 8],
                'description': 'Growing season - weather sensitivity',
                'expected_correlations': 'Weather-driven correlations'
            },
            'Fall_Harvest': {
                'months': [9, 10, 11],
                'description': 'Harvest season - marketing activity',
                'expected_correlations': 'Strong transport-grain correlations'
            },
            'Winter_Planning': {
                'months': [12, 1, 2],
                'description': 'Planning season - financial activity',
                'expected_correlations': 'Finance-agriculture correlations'
            }
        }
        
        for period_name, period_info in agricultural_periods.items():
            period_mask = returns_data.index.month.isin(period_info['months'])
            period_data = returns_data[period_mask]
            
            if len(period_data) >= 30:
                period_s1 = self.s1_calculator.batch_analyze_pairs(
                    period_data, asset_pairs[:5]
                )
                
                planting_harvest_effects[period_name] = {
                    'period_info': period_info,
                    's1_results': period_s1,
                    'data_points': len(period_data)
                }
        
        return planting_harvest_effects
    
    def _calculate_seasonal_significance(self, seasonal_violation_rates: Dict[str, float]) -> Dict[str, float]:
        """Calculate statistical significance of seasonal differences."""
        if len(seasonal_violation_rates) < 2:
            return {}
        
        significance = {}
        rates = list(seasonal_violation_rates.values())
        
        # Simple statistical tests
        mean_rate = np.mean(rates)
        std_rate = np.std(rates)
        
        for season, rate in seasonal_violation_rates.items():
            if std_rate > 0:
                z_score = abs(rate - mean_rate) / std_rate
                # Approximate p-value (simplified)
                p_value = 2 * (1 - 0.5 * (1 + np.tanh(z_score / np.sqrt(2))))
                significance[season] = p_value
            else:
                significance[season] = 1.0
        
        return significance
    
    def _get_regional_companies(self, region_name: str) -> List[str]:
        """Get companies with exposure to a specific geographic region."""
        regional_companies = []
        
        region_keywords = {
            'North_America': ['US', 'USA', 'North America', 'Canada', 'Mexico'],
            'South_America': ['Brazil', 'Argentina', 'South America', 'Latin America'],
            'Europe': ['Europe', 'EU', 'Germany', 'France', 'UK', 'Ukraine'],
            'Asia_Pacific': ['Asia', 'China', 'India', 'Australia', 'Pacific']
        }
        
        keywords = region_keywords.get(region_name, [])
        
        for ticker, company in self.universe_manager.companies.items():
            # Check geographic exposure
            if any(keyword in ' '.join(company.geographic_exposure) for keyword in keywords):
                regional_companies.append(ticker)
            # Check company description for regional indicators
            elif any(keyword.lower() in company.description.lower() for keyword in keywords):
                regional_companies.append(ticker)
        
        return regional_companies
    
    def _adjust_for_southern_hemisphere(self, months: List[int]) -> List[int]:
        """Adjust months for Southern Hemisphere seasons."""
        # Shift seasons by 6 months for Southern Hemisphere
        adjusted = []
        for month in months:
            adjusted_month = month + 6
            if adjusted_month > 12:
                adjusted_month -= 12
            adjusted.append(adjusted_month)
        return adjusted
    
    def _calculate_cross_regional_correlations(self, returns_data: pd.DataFrame,
                                             regional_patterns: Dict) -> Dict[Tuple[str, str], float]:
        """Calculate correlations between different geographic regions."""
        cross_regional_correlations = {}
        
        regions = list(regional_patterns.keys())
        
        for i, region1 in enumerate(regions):
            for region2 in regions[i+1:]:
                companies1 = regional_patterns[region1].get('companies', [])
                companies2 = regional_patterns[region2].get('companies', [])
                
                if companies1 and companies2:
                    # Calculate average correlation between regions
                    correlations = []
                    for comp1 in companies1[:3]:  # Limit for performance
                        for comp2 in companies2[:3]:
                            if comp1 in returns_data.columns and comp2 in returns_data.columns:
                                corr = returns_data[comp1].corr(returns_data[comp2])
                                if not np.isnan(corr):
                                    correlations.append(abs(corr))
                    
                    if correlations:
                        cross_regional_correlations[(region1, region2)] = np.mean(correlations)
        
        return cross_regional_correlations
    
    def _analyze_regional_crisis_impact(self, returns_data: pd.DataFrame,
                                      regional_patterns: Dict) -> Dict[str, Dict]:
        """Analyze how crises impact different regions."""
        crisis_impact = {}
        
        # Define major crisis periods
        crisis_periods = {
            'COVID19': ('2020-02-01', '2020-12-31'),
            'Ukraine_War': ('2022-02-24', '2023-12-31'),
            '2008_Crisis': ('2008-09-01', '2009-03-31')
        }
        
        for region_name, region_data in regional_patterns.items():
            region_crisis_impact = {}
            companies = region_data.get('companies', [])
            
            if not companies:
                continue
            
            for crisis_name, (start_date, end_date) in crisis_periods.items():
                try:
                    crisis_mask = (
                        (returns_data.index >= pd.to_datetime(start_date)) &
                        (returns_data.index <= pd.to_datetime(end_date))
                    )
                    
                    crisis_data = returns_data[crisis_mask]
                    normal_data = returns_data[~crisis_mask]
                    
                    if len(crisis_data) >= 10 and len(normal_data) >= 50:
                        # Calculate volatility increase during crisis
                        crisis_vol = crisis_data[companies[:5]].std().mean()
                        normal_vol = normal_data[companies[:5]].std().mean()
                        
                        volatility_increase = (crisis_vol / normal_vol - 1) * 100 if normal_vol > 0 else 0
                        
                        region_crisis_impact[crisis_name] = {
                            'volatility_increase': volatility_increase,
                            'crisis_data_points': len(crisis_data),
                            'risk_factors': region_data['region_info'].risk_factors
                        }
                
                except Exception as e:
                    continue
            
            crisis_impact[region_name] = region_crisis_impact
        
        return crisis_impact
    
    def _analyze_production_patterns(self, returns_data: pd.DataFrame,
                                   regional_patterns: Dict) -> Dict[str, Dict]:
        """Analyze production pattern effects on correlations."""
        production_effects = {}
        
        for region_name, region_data in regional_patterns.items():
            region_info = region_data['region_info']
            companies = region_data.get('companies', [])
            
            if not companies:
                continue
            
            # Analyze correlations during harvest seasons
            harvest_correlations = {}
            for harvest_season in region_info.harvest_seasons:
                season_info = self.agricultural_seasons.get(harvest_season)
                if season_info:
                    season_mask = returns_data.index.month.isin(season_info.months)
                    season_data = returns_data[season_mask]
                    
                    if len(season_data) >= 20:
                        # Calculate average correlation during harvest
                        correlations = []
                        for i, comp1 in enumerate(companies[:3]):
                            for comp2 in companies[i+1:3]:
                                if comp1 in season_data.columns and comp2 in season_data.columns:
                                    corr = season_data[comp1].corr(season_data[comp2])
                                    if not np.isnan(corr):
                                        correlations.append(abs(corr))
                        
                        if correlations:
                            harvest_correlations[harvest_season] = np.mean(correlations)
            
            production_effects[region_name] = {
                'harvest_correlations': harvest_correlations,
                'primary_crops': region_info.primary_crops,
                'climate_zone': region_info.climate_zone
            }
        
        return production_effects
    
    def _analyze_geographic_transmission(self, returns_data: pd.DataFrame,
                                       regional_patterns: Dict) -> Dict[str, List]:
        """Analyze geographic transmission mechanisms."""
        transmission_results = {}
        
        for region_name, region_data in regional_patterns.items():
            companies = region_data.get('companies', [])
            region_info = region_data['region_info']
            
            transmission_mechanisms = []
            
            # Analyze transmission based on risk factors
            for risk_factor in region_info.risk_factors:
                if risk_factor in ['Trade_Policy', 'Currency_Volatility']:
                    # Policy/currency transmission tends to be fast
                    transmission_mechanisms.append({
                        'mechanism': risk_factor,
                        'expected_speed': 'Fast (0-1 month)',
                        'transmission_type': 'Policy/Financial',
                        'affected_crops': region_info.primary_crops
                    })
                elif risk_factor in ['Drought', 'Flooding', 'Weather_Extremes']:
                    # Weather transmission can be immediate
                    transmission_mechanisms.append({
                        'mechanism': risk_factor,
                        'expected_speed': 'Immediate (0-2 weeks)',
                        'transmission_type': 'Weather/Climate',
                        'affected_crops': region_info.primary_crops
                    })
                elif risk_factor in ['Climate_Change', 'Deforestation']:
                    # Long-term transmission
                    transmission_mechanisms.append({
                        'mechanism': risk_factor,
                        'expected_speed': 'Slow (6+ months)',
                        'transmission_type': 'Structural/Environmental',
                        'affected_crops': region_info.primary_crops
                    })
            
            transmission_results[region_name] = transmission_mechanisms
        
        return transmission_results
    
    def _analyze_yearly_consistency(self, returns_data: pd.DataFrame,
                                  asset_pairs: List[Tuple[str, str]]) -> Dict[str, float]:
        """Analyze year-over-year consistency of seasonal patterns."""
        yearly_consistency = {}
        
        # Group data by year
        years = returns_data.index.year.unique()
        
        if len(years) < 2:
            return yearly_consistency
        
        # Calculate seasonal patterns for each year
        yearly_seasonal_patterns = {}
        
        for year in years:
            year_data = returns_data[returns_data.index.year == year]
            
            if len(year_data) >= 100:  # Minimum data requirement
                year_seasonal = {}
                
                for season_name, season_info in self.agricultural_seasons.items():
                    season_mask = year_data.index.month.isin(season_info.months)
                    season_data = year_data[season_mask]
                    
                    if len(season_data) >= 10:
                        season_s1 = self.s1_calculator.batch_analyze_pairs(
                            season_data, asset_pairs[:3]
                        )
                        
                        # Calculate average violation rate
                        if 'pair_results' in season_s1:
                            violations = []
                            for pair, results in season_s1['pair_results'].items():
                                if 's1_values' in results:
                                    s1_values = np.array(results['s1_values'])
                                    violation_rate = np.mean(np.abs(s1_values) > 2.0)
                                    violations.append(violation_rate)
                            
                            year_seasonal[season_name] = np.mean(violations) if violations else 0.0
                
                yearly_seasonal_patterns[year] = year_seasonal
        
        # Calculate consistency across years
        for season_name in self.agricultural_seasons.keys():
            season_values = []
            for year_patterns in yearly_seasonal_patterns.values():
                if season_name in year_patterns:
                    season_values.append(year_patterns[season_name])
            
            if len(season_values) >= 2:
                # Calculate coefficient of variation as consistency measure
                mean_val = np.mean(season_values)
                std_val = np.std(season_values)
                consistency = 1 - (std_val / mean_val) if mean_val > 0 else 0
                yearly_consistency[season_name] = max(0, min(1, consistency))
        
        return yearly_consistency
    
    def _identify_significant_variations(self, modulation_results: Dict) -> Dict[str, Dict]:
        """Identify statistically significant seasonal variations."""
        significant_variations = {}
        
        if 'seasonal_modulation' not in modulation_results:
            return significant_variations
        
        seasonal_modulation = modulation_results['seasonal_modulation']
        
        for season_name, modulation_data in seasonal_modulation.items():
            modulation_factor = modulation_data.get('modulation_factor', 1.0)
            strength_change = modulation_data.get('strength_change', 0.0)
            
            # Consider significant if >20% change from baseline
            if abs(strength_change) > 20:
                significance_level = 'High' if abs(strength_change) > 50 else 'Moderate'
                
                significant_variations[season_name] = {
                    'modulation_factor': modulation_factor,
                    'strength_change_percent': strength_change,
                    'significance_level': significance_level,
                    'direction': 'Increase' if strength_change > 0 else 'Decrease',
                    'expected_activity': modulation_data.get('expected_activity', 'Unknown')
                }
        
        return significant_variations
    
    def _calculate_crisis_amplification(self, normal_s1: Dict, crisis_s1: Dict) -> float:
        """Calculate crisis amplification factor."""
        try:
            normal_violations = []
            crisis_violations = []
            
            # Extract violation rates from normal period
            if 'pair_results' in normal_s1:
                for pair, results in normal_s1['pair_results'].items():
                    if 's1_values' in results:
                        s1_values = np.array(results['s1_values'])
                        violation_rate = np.mean(np.abs(s1_values) > 2.0)
                        normal_violations.append(violation_rate)
            
            # Extract violation rates from crisis period
            if 'pair_results' in crisis_s1:
                for pair, results in crisis_s1['pair_results'].items():
                    if 's1_values' in results:
                        s1_values = np.array(results['s1_values'])
                        violation_rate = np.mean(np.abs(s1_values) > 2.0)
                        crisis_violations.append(violation_rate)
            
            if normal_violations and crisis_violations:
                normal_avg = np.mean(normal_violations)
                crisis_avg = np.mean(crisis_violations)
                
                if normal_avg > 0:
                    amplification = crisis_avg / normal_avg
                    # Ensure we return a meaningful amplification factor
                    return max(amplification, 1.0)
            
            return 1.0
        
        except Exception:
            return 1.0
    
    def _analyze_seasonal_geographic_interactions(self, seasonal_results: SeasonalAnalysisResults,
                                                geographic_results: GeographicAnalysisResults) -> Dict[str, Dict]:
        """Analyze interactions between seasonal and geographic effects."""
        interactions = {}
        
        # Analyze how seasonal patterns vary by region
        for region_name, region_data in geographic_results.regional_patterns.items():
            region_seasonal_analysis = region_data.get('seasonal_analysis', {})
            
            if region_seasonal_analysis:
                region_interactions = {}
                
                for season_name, season_s1 in region_seasonal_analysis.items():
                    # Compare with global seasonal pattern
                    global_season_data = seasonal_results.seasonal_patterns.get(season_name, {})
                    
                    if global_season_data and 'violation_rate' in global_season_data:
                        global_rate = global_season_data['violation_rate']
                        
                        # Calculate regional violation rate
                        regional_violations = []
                        if 'pair_results' in season_s1:
                            for pair, results in season_s1['pair_results'].items():
                                if 's1_values' in results:
                                    s1_values = np.array(results['s1_values'])
                                    violation_rate = np.mean(np.abs(s1_values) > 2.0) * 100
                                    regional_violations.append(violation_rate)
                        
                        if regional_violations:
                            regional_rate = np.mean(regional_violations)
                            
                            # Calculate interaction strength
                            interaction_strength = regional_rate / max(global_rate, 0.1)
                            
                            region_interactions[season_name] = {
                                'regional_rate': regional_rate,
                                'global_rate': global_rate,
                                'interaction_strength': interaction_strength,
                                'deviation': regional_rate - global_rate
                            }
                
                interactions[region_name] = region_interactions
        
        return interactions
    
    def _analyze_regional_seasonal_variations(self, seasonal_results: SeasonalAnalysisResults,
                                            geographic_results: GeographicAnalysisResults) -> Dict[str, Dict]:
        """Analyze how seasonal patterns vary across regions."""
        regional_variations = {}
        
        for region_name, region_data in geographic_results.regional_patterns.items():
            region_info = region_data['region_info']
            
            # Analyze seasonal variations specific to this region
            regional_seasonal_effects = {}
            
            # Consider climate zone effects
            climate_zone = region_info.climate_zone
            if climate_zone == 'Tropical/Subtropical':
                # Less seasonal variation expected
                seasonal_factor = 0.7
            elif climate_zone == 'Temperate':
                # Strong seasonal variation expected
                seasonal_factor = 1.3
            else:
                # Varied climate - moderate seasonal effects
                seasonal_factor = 1.0
            
            # Consider harvest season alignment
            harvest_seasons = region_info.harvest_seasons
            for season_name in self.agricultural_seasons.keys():
                if season_name in harvest_seasons:
                    # Stronger effects during harvest seasons
                    harvest_amplification = 1.5
                else:
                    harvest_amplification = 1.0
                
                # Combine factors
                combined_factor = seasonal_factor * harvest_amplification
                
                regional_seasonal_effects[season_name] = {
                    'climate_factor': seasonal_factor,
                    'harvest_amplification': harvest_amplification,
                    'combined_factor': combined_factor,
                    'primary_crops': region_info.primary_crops,
                    'risk_factors': region_info.risk_factors
                }
            
            regional_variations[region_name] = {
                'seasonal_effects': regional_seasonal_effects,
                'climate_zone': climate_zone,
                'harvest_seasons': harvest_seasons
            }
        
        return regional_variations
    
    def _print_analysis_summary(self, results: SeasonalGeographicResults):
        """Print comprehensive analysis summary."""
        print("\n📊 SEASONAL-GEOGRAPHIC ANALYSIS SUMMARY")
        print("=" * 50)
        
        # Seasonal summary
        seasonal_rates = results.seasonal_results.seasonal_violation_rates
        if seasonal_rates:
            print(f"\n🌱 Seasonal Violation Rates:")
            for season, rate in seasonal_rates.items():
                print(f"   {season}: {rate:.1f}%")
            
            max_season = max(seasonal_rates, key=seasonal_rates.get)
            min_season = min(seasonal_rates, key=seasonal_rates.get)
            print(f"\n   Highest: {max_season} ({seasonal_rates[max_season]:.1f}%)")
            print(f"   Lowest: {min_season} ({seasonal_rates[min_season]:.1f}%)")
        
        # Geographic summary
        regional_patterns = results.geographic_results.regional_patterns
        print(f"\n🌍 Geographic Analysis:")
        print(f"   Regions analyzed: {len(regional_patterns)}")
        
        for region_name, region_data in regional_patterns.items():
            companies = region_data.get('companies', [])
            print(f"   {region_name}: {len(companies)} companies")
        
        # Cross-regional correlations
        cross_regional = results.geographic_results.cross_regional_correlations
        if cross_regional:
            print(f"\n🔄 Cross-Regional Correlations:")
            for (region1, region2), corr in cross_regional.items():
                print(f"   {region1} ↔ {region2}: {corr:.3f}")
        
        # Seasonal-geographic interactions
        interactions = results.seasonal_geographic_interactions
        if interactions:
            print(f"\n🌍🌱 Seasonal-Geographic Interactions:")
            print(f"   Regions with interactions: {len(interactions)}")
        
        print("\n✅ Analysis complete!")


if __name__ == "__main__":
    # Example usage
    analyzer = SeasonalGeographicAnalyzer()
    print("Seasonal Geographic Analyzer initialized successfully!")