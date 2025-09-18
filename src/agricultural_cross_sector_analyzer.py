#!/usr/bin/env python3
"""
AGRICULTURAL CROSS-SECTOR ANALYZER MAIN CLASS
=============================================

This is the main integration class that combines all components of the agricultural
cross-sector analysis system. It provides tier-based analysis methods with crisis
integration and comprehensive workflow for comparing normal vs crisis periods.

The class integrates:
- Agricultural Universe Manager (60+ companies with tier classifications)
- Enhanced S1 Calculator (mathematically accurate Bell inequality implementation)
- Cross-Sector Transmission Detector (fast transmission mechanisms 0-3 months)
- Agricultural Crisis Analyzer (specialized crisis period analysis)
- Statistical Validation Suite (bootstrap validation, significance testing)
- Agricultural Visualization Suite (publication-ready figures)

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
    from .agricultural_universe_manager import AgriculturalUniverseManager, Tier, CompanyInfo
    from .enhanced_s1_calculator import EnhancedS1Calculator
    from .cross_sector_transmission_detector import CrossSectorTransmissionDetector
    from .agricultural_crisis_analyzer import AgriculturalCrisisAnalyzer, CrisisResults, ComparisonResults
    from .agricultural_data_handler import AgriculturalDataHandler
    from .statistical_validation_suite import ComprehensiveStatisticalSuite
    from .seasonal_geographic_analyzer import SeasonalGeographicAnalyzer, SeasonalGeographicResults
    from .seasonal_visualization_suite import SeasonalVisualizationSuite
    # from .agricultural_visualization_suite import CrisisPeriodTimeSeriesVisualizer
except ImportError:
    # Handle direct execution
    from agricultural_universe_manager import AgriculturalUniverseManager, Tier, CompanyInfo
    from enhanced_s1_calculator import EnhancedS1Calculator
    from cross_sector_transmission_detector import CrossSectorTransmissionDetector
    from agricultural_crisis_analyzer import AgriculturalCrisisAnalyzer, CrisisResults, ComparisonResults
    from agricultural_data_handler import AgriculturalDataHandler
    from statistical_validation_suite import ComprehensiveStatisticalSuite
    from seasonal_geographic_analyzer import SeasonalGeographicAnalyzer, SeasonalGeographicResults
    from seasonal_visualization_suite import SeasonalVisualizationSuite
    # from agricultural_visualization_suite import CrisisPeriodTimeSeriesVisualizer


@dataclass
class AnalysisConfiguration:
    """Configuration for agricultural cross-sector analysis."""
    window_size: int = 20
    threshold_method: str = 'absolute'
    threshold_value: float = 0.01
    crisis_window_size: int = 15
    crisis_threshold_quantile: float = 0.8
    significance_level: float = 0.001
    bootstrap_samples: int = 1000
    max_pairs_per_tier: int = 25  # Limit for performance


@dataclass
class TierAnalysisResults:
    """Results from tier-based analysis."""
    tier: int
    tier_name: str
    cross_sector_pairs: List[Tuple[str, str]]
    s1_results: Dict
    transmission_results: List
    crisis_results: Optional[Dict]
    statistical_validation: Dict
    violation_summary: Dict


@dataclass
class ComprehensiveAnalysisResults:
    """Complete results from agricultural cross-sector analysis."""
    analysis_config: AnalysisConfiguration
    tier_results: Dict[int, TierAnalysisResults]
    crisis_comparison: Optional[ComparisonResults]
    transmission_summary: pd.DataFrame
    overall_statistics: Dict
    execution_metadata: Dict
    seasonal_geographic_results: Optional[SeasonalGeographicResults] = None


class AgriculturalCrossSectorAnalyzer:
    """
    Main Agricultural Cross-Sector Analyzer class integrating all components.
    
    This class provides the primary interface for conducting comprehensive
    agricultural cross-sector Bell inequality analysis with crisis integration.
    """
    
    def __init__(self, config: Optional[AnalysisConfiguration] = None):
        """
        Initialize the Agricultural Cross-Sector Analyzer.
        
        Parameters:
        -----------
        config : AnalysisConfiguration, optional
            Analysis configuration. If None, uses default configuration.
        """
        self.config = config or AnalysisConfiguration()
        
        # Initialize core components
        self.universe_manager = AgriculturalUniverseManager()
        self.data_handler = AgriculturalDataHandler()
        self.transmission_detector = CrossSectorTransmissionDetector(
            transmission_window=90,  # 0-3 months
            significance_level=self.config.significance_level
        )
        self.crisis_analyzer = AgriculturalCrisisAnalyzer(self.universe_manager)
        self.statistical_validator = ComprehensiveStatisticalSuite(
            n_bootstrap=self.config.bootstrap_samples,
            alpha=self.config.significance_level
        )
        self.seasonal_geographic_analyzer = SeasonalGeographicAnalyzer(self.universe_manager)
        self.seasonal_visualizer = SeasonalVisualizationSuite()
        # self.visualizer = CrisisPeriodTimeSeriesVisualizer()  # Disabled for testing
        
        # Initialize S1 calculators
        self.normal_calculator = EnhancedS1Calculator(
            window_size=self.config.window_size,
            threshold_method=self.config.threshold_method,
            threshold_value=self.config.threshold_value
        )
        
        self.crisis_calculator = EnhancedS1Calculator(
            window_size=self.config.crisis_window_size,
            threshold_method='quantile',
            threshold_quantile=self.config.crisis_threshold_quantile
        )
        
        # Analysis state
        self.returns_data: Optional[pd.DataFrame] = None
        self.analysis_results: Optional[ComprehensiveAnalysisResults] = None
        
        print("🌾 Agricultural Cross-Sector Analyzer Initialized")
        print(f"   Universe: {len(self.universe_manager.companies)} companies")
        print(f"   Configuration: window={self.config.window_size}, threshold={self.config.threshold_value}")
        print(f"   Crisis parameters: window={self.config.crisis_window_size}, quantile={self.config.crisis_threshold_quantile}")
    
    def load_data(self, tickers: Optional[List[str]] = None, 
                  period: str = "5y", **kwargs) -> pd.DataFrame:
        """
        Load data for analysis using the agricultural data handler.
        
        Parameters:
        -----------
        tickers : List[str], optional
            List of tickers to load. If None, loads all universe companies.
        period : str, optional
            Data period to load. Default: "5y"
        **kwargs : Additional arguments for data loading
            
        Returns:
        --------
        pd.DataFrame : Returns data ready for analysis
        """
        print("📊 Loading data for agricultural cross-sector analysis...")
        
        if tickers is None:
            # Load all companies from universe
            tickers = list(self.universe_manager.companies.keys())
            print(f"   Loading data for {len(tickers)} universe companies")
        
        # Load price data
        price_data = self.data_handler.download_batch_data(
            tickers=tickers,
            period=period,
            **kwargs
        )
        
        if price_data.empty:
            raise ValueError("No data loaded successfully")
        
        # Calculate returns
        self.returns_data = self.data_handler.calculate_returns(price_data)
        
        print(f"✅ Data loaded: {len(self.returns_data)} observations, {len(self.returns_data.columns)} assets")
        print(f"   Date range: {self.returns_data.index[0].date()} to {self.returns_data.index[-1].date()}")
        
        return self.returns_data
    
    def analyze_tier_1_crisis(self, crisis_periods: Optional[List[str]] = None) -> TierAnalysisResults:
        """
        Create tier-based analysis methods with crisis integration for Tier 1.
        
        Tier 1: Energy/Transport/Chemicals - Direct operational dependencies
        
        Parameters:
        -----------
        crisis_periods : List[str], optional
            Crisis periods to analyze. If None, analyzes all three major crises.
            
        Returns:
        --------
        TierAnalysisResults : Complete Tier 1 analysis results
        """
        print("🔋 Analyzing Tier 1: Energy/Transport/Chemicals → Agriculture")
        
        if self.returns_data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Get Tier 1 and agricultural assets
        tier1_assets = self.universe_manager.classify_by_tier(1)
        agricultural_assets = self.universe_manager.classify_by_tier(0)
        
        # Filter to available assets
        available_tier1 = [asset for asset in tier1_assets if asset in self.returns_data.columns]
        available_ag = [asset for asset in agricultural_assets if asset in self.returns_data.columns]
        
        print(f"   Available assets: {len(available_tier1)} Tier 1, {len(available_ag)} Agricultural")
        
        # Create cross-sector pairs based on operational dependencies
        cross_sector_pairs = self._create_tier1_pairs(available_tier1, available_ag)
        
        # Limit pairs for performance
        if len(cross_sector_pairs) > self.config.max_pairs_per_tier:
            cross_sector_pairs = cross_sector_pairs[:self.config.max_pairs_per_tier]
            print(f"   Limited to {len(cross_sector_pairs)} pairs for performance")
        
        # Normal period S1 analysis
        print("   Performing normal period S1 analysis...")
        s1_results = self.normal_calculator.batch_analyze_pairs(
            self.returns_data, cross_sector_pairs
        )
        
        # Transmission detection
        print("   Detecting transmission mechanisms...")
        transmission_results = []
        
        # Energy transmission
        energy_assets = [asset for asset in available_tier1 
                        if self.universe_manager.companies[asset].sector == "Energy"]
        if energy_assets:
            energy_transmission = self.transmission_detector.detect_energy_transmission(
                energy_assets, available_ag
            )
            transmission_results.extend(energy_transmission)
        
        # Transportation transmission
        transport_assets = [asset for asset in available_tier1 
                           if self.universe_manager.companies[asset].sector == "Transportation"]
        if transport_assets:
            transport_transmission = self.transmission_detector.detect_transport_transmission(
                transport_assets, available_ag
            )
            transmission_results.extend(transport_transmission)
        
        # Chemical transmission
        chemical_assets = [asset for asset in available_tier1 
                          if self.universe_manager.companies[asset].sector == "Chemicals"]
        if chemical_assets:
            chemical_transmission = self.transmission_detector.detect_chemical_transmission(
                chemical_assets, available_ag
            )
            transmission_results.extend(chemical_transmission)
        
        # Crisis analysis
        crisis_results = None
        if crisis_periods is not None:
            print("   Performing crisis period analysis...")
            crisis_results = self._analyze_tier_crisis_periods(
                tier=1, crisis_periods=crisis_periods
            )
        
        # Statistical validation
        print("   Performing statistical validation...")
        statistical_validation = self.statistical_validator.comprehensive_analysis(
            s1_results.get('pair_results', {})
        )
        
        # Create violation summary
        violation_summary = self._create_violation_summary(s1_results, transmission_results)
        
        results = TierAnalysisResults(
            tier=1,
            tier_name="Energy/Transport/Chemicals",
            cross_sector_pairs=cross_sector_pairs,
            s1_results=s1_results,
            transmission_results=transmission_results,
            crisis_results=crisis_results,
            statistical_validation=statistical_validation,
            violation_summary=violation_summary
        )
        
        print(f"✅ Tier 1 analysis complete: {violation_summary.get('total_violations', 0)} violations detected")
        
        return results
    
    def analyze_tier_2_crisis(self, crisis_periods: Optional[List[str]] = None) -> TierAnalysisResults:
        """
        Create tier-based analysis methods with crisis integration for Tier 2.
        
        Tier 2: Finance/Equipment - Major cost drivers
        
        Parameters:
        -----------
        crisis_periods : List[str], optional
            Crisis periods to analyze. If None, analyzes all three major crises.
            
        Returns:
        --------
        TierAnalysisResults : Complete Tier 2 analysis results
        """
        print("🏦 Analyzing Tier 2: Finance/Equipment → Agriculture")
        
        if self.returns_data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Get Tier 2 and agricultural assets
        tier2_assets = self.universe_manager.classify_by_tier(2)
        agricultural_assets = self.universe_manager.classify_by_tier(0)
        
        # Filter to available assets
        available_tier2 = [asset for asset in tier2_assets if asset in self.returns_data.columns]
        available_ag = [asset for asset in agricultural_assets if asset in self.returns_data.columns]
        
        print(f"   Available assets: {len(available_tier2)} Tier 2, {len(available_ag)} Agricultural")
        
        # Create cross-sector pairs based on operational dependencies
        cross_sector_pairs = self._create_tier2_pairs(available_tier2, available_ag)
        
        # Limit pairs for performance
        if len(cross_sector_pairs) > self.config.max_pairs_per_tier:
            cross_sector_pairs = cross_sector_pairs[:self.config.max_pairs_per_tier]
            print(f"   Limited to {len(cross_sector_pairs)} pairs for performance")
        
        # Normal period S1 analysis
        print("   Performing normal period S1 analysis...")
        s1_results = self.normal_calculator.batch_analyze_pairs(
            self.returns_data, cross_sector_pairs
        )
        
        # Transmission detection (Finance → Agriculture)
        print("   Detecting financial transmission mechanisms...")
        transmission_results = []
        
        finance_assets = [asset for asset in available_tier2 
                         if self.universe_manager.companies[asset].sector == "Finance"]
        
        # Simplified transmission analysis for finance sector
        for finance_asset in finance_assets:
            for ag_asset in available_ag[:5]:  # Limit for performance
                transmission_speed = self.transmission_detector.analyze_transmission_speed(
                    (finance_asset, ag_asset)
                )
                if transmission_speed.max_correlation > 0.1:  # Threshold for detection
                    transmission_results.append({
                        'pair': (finance_asset, ag_asset),
                        'transmission_detected': True,
                        'correlation_strength': transmission_speed.max_correlation,
                        'transmission_lag': transmission_speed.optimal_lag,
                        'speed_category': transmission_speed.speed_category,
                        'mechanism': 'Credit availability → agricultural financing → operations'
                    })
        
        # Crisis analysis
        crisis_results = None
        if crisis_periods is not None:
            print("   Performing crisis period analysis...")
            crisis_results = self._analyze_tier_crisis_periods(
                tier=2, crisis_periods=crisis_periods
            )
        
        # Statistical validation
        print("   Performing statistical validation...")
        statistical_validation = self.statistical_validator.comprehensive_analysis(
            s1_results.get('pair_results', {})
        )
        
        # Create violation summary
        violation_summary = self._create_violation_summary(s1_results, transmission_results)
        
        results = TierAnalysisResults(
            tier=2,
            tier_name="Finance/Equipment",
            cross_sector_pairs=cross_sector_pairs,
            s1_results=s1_results,
            transmission_results=transmission_results,
            crisis_results=crisis_results,
            statistical_validation=statistical_validation,
            violation_summary=violation_summary
        )
        
        print(f"✅ Tier 2 analysis complete: {violation_summary.get('total_violations', 0)} violations detected")
        
        return results
    
    def analyze_tier_3_crisis(self, crisis_periods: Optional[List[str]] = None) -> TierAnalysisResults:
        """
        Create tier-based analysis methods with crisis integration for Tier 3.
        
        Tier 3: Policy-linked - Renewable Energy, Water Utilities
        
        Parameters:
        -----------
        crisis_periods : List[str], optional
            Crisis periods to analyze. If None, analyzes all three major crises.
            
        Returns:
        --------
        TierAnalysisResults : Complete Tier 3 analysis results
        """
        print("⚡ Analyzing Tier 3: Policy-linked → Agriculture")
        
        if self.returns_data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Get Tier 3 and agricultural assets
        tier3_assets = self.universe_manager.classify_by_tier(3)
        agricultural_assets = self.universe_manager.classify_by_tier(0)
        
        # Filter to available assets
        available_tier3 = [asset for asset in tier3_assets if asset in self.returns_data.columns]
        available_ag = [asset for asset in agricultural_assets if asset in self.returns_data.columns]
        
        print(f"   Available assets: {len(available_tier3)} Tier 3, {len(available_ag)} Agricultural")
        
        # Create cross-sector pairs based on operational dependencies
        cross_sector_pairs = self._create_tier3_pairs(available_tier3, available_ag)
        
        # Limit pairs for performance
        if len(cross_sector_pairs) > self.config.max_pairs_per_tier:
            cross_sector_pairs = cross_sector_pairs[:self.config.max_pairs_per_tier]
            print(f"   Limited to {len(cross_sector_pairs)} pairs for performance")
        
        # Normal period S1 analysis
        print("   Performing normal period S1 analysis...")
        s1_results = self.normal_calculator.batch_analyze_pairs(
            self.returns_data, cross_sector_pairs
        )
        
        # Transmission detection (Policy-linked → Agriculture)
        print("   Detecting policy transmission mechanisms...")
        transmission_results = []
        
        # Simplified transmission analysis for policy-linked sectors
        for tier3_asset in available_tier3:
            for ag_asset in available_ag[:3]:  # Limit for performance
                transmission_speed = self.transmission_detector.analyze_transmission_speed(
                    (tier3_asset, ag_asset)
                )
                if transmission_speed.max_correlation > 0.05:  # Lower threshold for policy effects
                    transmission_results.append({
                        'pair': (tier3_asset, ag_asset),
                        'transmission_detected': True,
                        'correlation_strength': transmission_speed.max_correlation,
                        'transmission_lag': transmission_speed.optimal_lag,
                        'speed_category': transmission_speed.speed_category,
                        'mechanism': 'Policy changes → utility costs → agricultural operations'
                    })
        
        # Crisis analysis
        crisis_results = None
        if crisis_periods is not None:
            print("   Performing crisis period analysis...")
            crisis_results = self._analyze_tier_crisis_periods(
                tier=3, crisis_periods=crisis_periods
            )
        
        # Statistical validation
        print("   Performing statistical validation...")
        statistical_validation = self.statistical_validator.comprehensive_analysis(
            s1_results.get('pair_results', {})
        )
        
        # Create violation summary
        violation_summary = self._create_violation_summary(s1_results, transmission_results)
        
        results = TierAnalysisResults(
            tier=3,
            tier_name="Policy-linked",
            cross_sector_pairs=cross_sector_pairs,
            s1_results=s1_results,
            transmission_results=transmission_results,
            crisis_results=crisis_results,
            statistical_validation=statistical_validation,
            violation_summary=violation_summary
        )
        
        print(f"✅ Tier 3 analysis complete: {violation_summary.get('total_violations', 0)} violations detected")
        
        return results
    
    def run_seasonal_geographic_analysis(self, include_crisis_analysis: bool = True) -> SeasonalGeographicResults:
        """
        Run comprehensive seasonal and geographic analysis.
        
        This addresses Requirements 8.2, 8.3, and 8.4 for seasonal and geographic effects.
        
        Parameters:
        -----------
        include_crisis_analysis : bool, optional
            Whether to include crisis period analysis. Default: True
            
        Returns:
        --------
        SeasonalGeographicResults : Complete seasonal and geographic analysis results
        """
        print("🌍🌱 Running Seasonal and Geographic Analysis")
        print("=" * 60)
        
        if self.returns_data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Create asset pairs for analysis
        all_tickers = list(self.returns_data.columns)
        
        # Get cross-sector pairs from all tiers
        tier1_assets = [t for t in all_tickers if t in self.universe_manager.classify_by_tier(1)]
        tier2_assets = [t for t in all_tickers if t in self.universe_manager.classify_by_tier(2)]
        agricultural_assets = [t for t in all_tickers if t in self.universe_manager.classify_by_tier(0)]
        
        # Create representative asset pairs
        asset_pairs = []
        
        # Tier 1 cross-sector pairs
        for tier1_asset in tier1_assets[:5]:  # Limit for performance
            for ag_asset in agricultural_assets[:3]:
                asset_pairs.append((tier1_asset, ag_asset))
        
        # Tier 2 cross-sector pairs
        for tier2_asset in tier2_assets[:3]:
            for ag_asset in agricultural_assets[:2]:
                asset_pairs.append((tier2_asset, ag_asset))
        
        # Agricultural intra-sector pairs
        for i, ag1 in enumerate(agricultural_assets[:4]):
            for ag2 in agricultural_assets[i+1:4]:
                asset_pairs.append((ag1, ag2))
        
        print(f"   Analyzing {len(asset_pairs)} asset pairs")
        
        # Define crisis periods if including crisis analysis
        crisis_periods = None
        if include_crisis_analysis:
            crisis_periods = {
                'COVID19_Food_Crisis': ('2020-02-01', '2020-12-31'),
                'Ukraine_War_Food_Crisis': ('2022-02-24', '2023-12-31'),
                '2008_Food_Crisis': ('2008-09-01', '2009-03-31')
            }
        
        # Run comprehensive seasonal-geographic analysis
        seasonal_geographic_results = self.seasonal_geographic_analyzer.run_comprehensive_seasonal_geographic_analysis(
            returns_data=self.returns_data,
            asset_pairs=asset_pairs,
            crisis_periods=crisis_periods
        )
        
        print("✅ Seasonal and Geographic Analysis Complete")
        
        return seasonal_geographic_results
    
    def run_comprehensive_analysis(self, crisis_periods: Optional[List[str]] = None, 
                                 include_seasonal_geographic: bool = True) -> ComprehensiveAnalysisResults:
        """
        Create comprehensive analysis workflow comparing normal vs crisis periods for each tier.
        
        This method runs the complete agricultural cross-sector analysis including:
        - All three tier analyses with crisis integration
        - Cross-crisis comparison
        - Transmission mechanism summary
        - Statistical validation
        - Seasonal and geographic analysis (if enabled)
        - Publication-ready results
        
        Parameters:
        -----------
        crisis_periods : List[str], optional
            Crisis periods to analyze. Default: all three major crises.
        include_seasonal_geographic : bool, optional
            Whether to include seasonal and geographic analysis. Default: True
            
        Returns:
        --------
        ComprehensiveAnalysisResults : Complete analysis results
        """
        print("🌾 Running Comprehensive Agricultural Cross-Sector Analysis")
        print("=" * 70)
        
        start_time = datetime.now()
        
        if self.returns_data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        if crisis_periods is None:
            crisis_periods = ["2008_financial_crisis", "eu_debt_crisis", "covid19_pandemic"]
        
        # Analyze each tier
        tier_results = {}
        
        print("\n🔋 TIER 1 ANALYSIS: Energy/Transport/Chemicals")
        print("-" * 50)
        tier_results[1] = self.analyze_tier_1_crisis(crisis_periods)
        
        print("\n🏦 TIER 2 ANALYSIS: Finance/Equipment")
        print("-" * 50)
        tier_results[2] = self.analyze_tier_2_crisis(crisis_periods)
        
        print("\n⚡ TIER 3 ANALYSIS: Policy-linked")
        print("-" * 50)
        tier_results[3] = self.analyze_tier_3_crisis(crisis_periods)
        
        # Crisis comparison analysis
        print("\n📊 CRISIS COMPARISON ANALYSIS")
        print("-" * 50)
        crisis_comparison = self.crisis_analyzer.compare_crisis_periods(
            self.returns_data, crisis_periods
        )
        
        # Create transmission summary
        print("\n🔄 TRANSMISSION MECHANISM SUMMARY")
        print("-" * 50)
        transmission_summary = self._create_transmission_summary(tier_results)
        
        # Calculate overall statistics
        print("\n📈 OVERALL STATISTICS")
        print("-" * 50)
        overall_statistics = self._calculate_overall_statistics(tier_results, crisis_comparison)
        
        # Run seasonal and geographic analysis if enabled
        seasonal_geographic_results = None
        if include_seasonal_geographic:
            print("\n🌍🌱 SEASONAL AND GEOGRAPHIC ANALYSIS")
            print("-" * 50)
            seasonal_geographic_results = self.run_seasonal_geographic_analysis(
                include_crisis_analysis=True
            )
        
        # Execution metadata
        end_time = datetime.now()
        execution_metadata = {
            'start_time': start_time,
            'end_time': end_time,
            'execution_duration': str(end_time - start_time),
            'data_period': f"{self.returns_data.index[0].date()} to {self.returns_data.index[-1].date()}",
            'total_observations': len(self.returns_data),
            'total_assets': len(self.returns_data.columns),
            'crisis_periods_analyzed': crisis_periods,
            'configuration': self.config
        }
        
        # Compile comprehensive results
        self.analysis_results = ComprehensiveAnalysisResults(
            analysis_config=self.config,
            tier_results=tier_results,
            crisis_comparison=crisis_comparison,
            transmission_summary=transmission_summary,
            overall_statistics=overall_statistics,
            execution_metadata=execution_metadata,
            seasonal_geographic_results=seasonal_geographic_results
        )
        
        print("\n✅ COMPREHENSIVE ANALYSIS COMPLETE")
        print("=" * 70)
        self._print_comprehensive_summary()
        
        return self.analysis_results
    
    def _create_tier1_pairs(self, tier1_assets: List[str], ag_assets: List[str]) -> List[Tuple[str, str]]:
        """Create cross-sector pairing logic based on operational dependencies for Tier 1."""
        pairs = []
        
        # Energy → Fertilizer companies (strong operational dependency)
        energy_assets = [asset for asset in tier1_assets 
                        if self.universe_manager.companies[asset].sector == "Energy"]
        fertilizer_assets = [asset for asset in ag_assets 
                           if "fertilizer" in self.universe_manager.companies[asset].subsector.lower()]
        
        for energy in energy_assets:
            for fertilizer in fertilizer_assets:
                pairs.append((energy, fertilizer))
        
        # Transportation → Grain trading companies
        transport_assets = [asset for asset in tier1_assets 
                           if self.universe_manager.companies[asset].sector == "Transportation"]
        grain_assets = [asset for asset in ag_assets 
                       if any(grain in self.universe_manager.companies[asset].subsector.lower() 
                             for grain in ["grain", "trading", "processing"])]
        
        for transport in transport_assets:
            for grain in grain_assets:
                pairs.append((transport, grain))
        
        # Chemicals → All agricultural companies (broad dependency)
        chemical_assets = [asset for asset in tier1_assets 
                          if self.universe_manager.companies[asset].sector == "Chemicals"]
        
        for chemical in chemical_assets:
            for ag in ag_assets[:5]:  # Limit for performance
                pairs.append((chemical, ag))
        
        return pairs
    
    def _create_tier2_pairs(self, tier2_assets: List[str], ag_assets: List[str]) -> List[Tuple[str, str]]:
        """Create cross-sector pairing logic based on operational dependencies for Tier 2."""
        pairs = []
        
        # Finance → All agricultural companies (credit dependency)
        finance_assets = [asset for asset in tier2_assets 
                         if self.universe_manager.companies[asset].sector == "Finance"]
        
        for finance in finance_assets:
            for ag in ag_assets[:8]:  # Broader coverage for finance
                pairs.append((finance, ag))
        
        # Equipment → Equipment-dependent agricultural companies
        equipment_assets = [asset for asset in tier2_assets 
                           if self.universe_manager.companies[asset].sector == "Equipment"]
        equipment_dependent = [asset for asset in ag_assets 
                              if any(dep in self.universe_manager.companies[asset].operational_dependencies 
                                    for dep in ["Steel", "Technology", "Equipment"])]
        
        for equipment in equipment_assets:
            for ag in equipment_dependent:
                pairs.append((equipment, ag))
        
        return pairs
    
    def _create_tier3_pairs(self, tier3_assets: List[str], ag_assets: List[str]) -> List[Tuple[str, str]]:
        """Create cross-sector pairing logic based on operational dependencies for Tier 3."""
        pairs = []
        
        # Utilities → All agricultural companies (energy dependency)
        for tier3 in tier3_assets:
            for ag in ag_assets[:5]:  # Limited coverage for policy effects
                pairs.append((tier3, ag))
        
        return pairs
    
    def _analyze_tier_crisis_periods(self, tier: int, crisis_periods: List[str]) -> Dict:
        """Analyze tier during specified crisis periods."""
        crisis_results = {}
        
        for crisis_period in crisis_periods:
            try:
                if crisis_period == "2008_financial_crisis":
                    result = self.crisis_analyzer.analyze_2008_financial_crisis(self.returns_data)
                elif crisis_period == "eu_debt_crisis":
                    result = self.crisis_analyzer.analyze_eu_debt_crisis(self.returns_data)
                elif crisis_period == "covid19_pandemic":
                    result = self.crisis_analyzer.analyze_covid19_pandemic(self.returns_data)
                else:
                    continue
                
                # Extract tier-specific results
                if tier in result.tier_results:
                    crisis_results[crisis_period] = result.tier_results[tier]
                
            except Exception as e:
                print(f"   ⚠️  Error analyzing {crisis_period}: {str(e)}")
                continue
        
        return crisis_results
    
    def _create_violation_summary(self, s1_results: Dict, transmission_results: List) -> Dict:
        """Create violation summary from S1 and transmission results."""
        summary = s1_results.get('summary', {})
        
        # Add transmission information
        detected_transmissions = len([t for t in transmission_results 
                                    if (hasattr(t, 'transmission_detected') and t.transmission_detected) or
                                       (isinstance(t, dict) and t.get('transmission_detected', False))])
        
        summary.update({
            'detected_transmissions': detected_transmissions,
            'total_transmission_tests': len(transmission_results),
            'transmission_detection_rate': (detected_transmissions / len(transmission_results) * 100) 
                                         if transmission_results else 0.0
        })
        
        return summary
    
    def _create_transmission_summary(self, tier_results: Dict[int, TierAnalysisResults]) -> pd.DataFrame:
        """Create comprehensive transmission mechanism summary."""
        summary_data = []
        
        for tier, results in tier_results.items():
            for transmission in results.transmission_results:
                if isinstance(transmission, dict):
                    summary_data.append({
                        'tier': tier,
                        'tier_name': results.tier_name,
                        'source_asset': transmission.get('pair', ('', ''))[0],
                        'target_asset': transmission.get('pair', ('', ''))[1],
                        'transmission_detected': transmission.get('transmission_detected', False),
                        'correlation_strength': transmission.get('correlation_strength', 0.0),
                        'transmission_lag': transmission.get('transmission_lag', None),
                        'speed_category': transmission.get('speed_category', 'Unknown'),
                        'mechanism': transmission.get('mechanism', 'Unknown')
                    })
                else:
                    # Handle TransmissionResults objects
                    summary_data.append({
                        'tier': tier,
                        'tier_name': results.tier_name,
                        'source_asset': transmission.pair[0],
                        'target_asset': transmission.pair[1],
                        'transmission_detected': transmission.transmission_detected,
                        'correlation_strength': transmission.correlation_strength,
                        'transmission_lag': transmission.transmission_lag,
                        'speed_category': transmission.speed_category,
                        'mechanism': transmission.mechanism
                    })
        
        return pd.DataFrame(summary_data)
    
    def _calculate_overall_statistics(self, tier_results: Dict[int, TierAnalysisResults], 
                                    crisis_comparison: ComparisonResults) -> Dict:
        """Calculate overall statistics across all tiers and crises."""
        stats = {}
        
        # Aggregate violation statistics
        total_violations = sum(results.violation_summary.get('total_violations', 0) 
                             for results in tier_results.values())
        total_calculations = sum(results.violation_summary.get('total_calculations', 0) 
                               for results in tier_results.values())
        
        stats['total_violations'] = total_violations
        stats['total_calculations'] = total_calculations
        stats['overall_violation_rate'] = (total_violations / total_calculations * 100) if total_calculations > 0 else 0.0
        
        # Transmission statistics
        total_transmissions = sum(results.violation_summary.get('detected_transmissions', 0) 
                                for results in tier_results.values())
        total_transmission_tests = sum(results.violation_summary.get('total_transmission_tests', 0) 
                                     for results in tier_results.values())
        
        stats['total_detected_transmissions'] = total_transmissions
        stats['total_transmission_tests'] = total_transmission_tests
        stats['overall_transmission_rate'] = (total_transmissions / total_transmission_tests * 100) if total_transmission_tests > 0 else 0.0
        
        # Crisis statistics
        if crisis_comparison:
            stats['most_vulnerable_tier'] = max(crisis_comparison.tier_vulnerability_index.items(), 
                                              key=lambda x: x[1])[0] if crisis_comparison.tier_vulnerability_index else None
            stats['tier_vulnerability_scores'] = crisis_comparison.tier_vulnerability_index
            stats['cross_crisis_consistency'] = crisis_comparison.cross_crisis_consistency
        
        return stats
    
    def _print_comprehensive_summary(self):
        """Print comprehensive analysis summary."""
        if not self.analysis_results:
            return
        
        results = self.analysis_results
        stats = results.overall_statistics
        
        print(f"📊 Analysis Duration: {results.execution_metadata['execution_duration']}")
        print(f"📈 Overall Violation Rate: {stats['overall_violation_rate']:.2f}%")
        print(f"🔄 Transmission Detection Rate: {stats['overall_transmission_rate']:.2f}%")
        
        if 'most_vulnerable_tier' in stats and stats['most_vulnerable_tier']:
            tier_name = results.tier_results[stats['most_vulnerable_tier']].tier_name
            print(f"⚠️  Most Vulnerable Tier: Tier {stats['most_vulnerable_tier']} ({tier_name})")
        
        print(f"🎯 Total Bell Violations: {stats['total_violations']:,}/{stats['total_calculations']:,}")
        print(f"📡 Detected Transmissions: {stats['total_detected_transmissions']}/{stats['total_transmission_tests']}")


def main():
    """Example usage of the Agricultural Cross-Sector Analyzer"""
    print("🌾 Agricultural Cross-Sector Analysis System")
    print("=" * 60)
    
    # Initialize analyzer with custom configuration
    config = AnalysisConfiguration(
        window_size=20,
        threshold_value=0.01,
        crisis_window_size=15,
        crisis_threshold_quantile=0.8,
        significance_level=0.001,
        max_pairs_per_tier=20
    )
    
    analyzer = AgriculturalCrossSectorAnalyzer(config)
    
    # Load data for key agricultural and cross-sector companies
    key_tickers = [
        # Agricultural companies
        'ADM', 'BG', 'CF', 'MOS', 'NTR', 'DE', 'CAG', 'TSN',
        # Tier 1: Energy/Transport/Chemicals
        'XOM', 'CVX', 'COP', 'UNP', 'CSX', 'DOW', 'DD',
        # Tier 2: Finance/Equipment
        'JPM', 'BAC', 'GS', 'CAT',
        # Tier 3: Policy-linked
        'NEE', 'DUK', 'AWK'
    ]
    
    try:
        # Load data
        returns_data = analyzer.load_data(tickers=key_tickers, period="3y")
        
        # Run comprehensive analysis
        results = analyzer.run_comprehensive_analysis()
        
        print("\n🎯 KEY FINDINGS:")
        print("-" * 30)
        
        # Display key results
        for tier, tier_results in results.tier_results.items():
            violation_rate = tier_results.violation_summary.get('overall_violation_rate', 0)
            transmission_rate = tier_results.violation_summary.get('transmission_detection_rate', 0)
            
            print(f"Tier {tier} ({tier_results.tier_name}):")
            print(f"  • Violation Rate: {violation_rate:.1f}%")
            print(f"  • Transmission Rate: {transmission_rate:.1f}%")
        
        # Crisis comparison summary
        if results.crisis_comparison:
            print(f"\nMost Vulnerable Tier: {results.overall_statistics.get('most_vulnerable_tier', 'Unknown')}")
            
        print(f"\n✅ Analysis complete! Results saved to analyzer.analysis_results")
        
    except Exception as e:
        print(f"❌ Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()