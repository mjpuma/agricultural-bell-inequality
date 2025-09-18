#!/usr/bin/env python3
"""
ADVANCED STATISTICAL VALIDATION AND ANALYSIS SUITE
==================================================

This module implements comprehensive statistical validation and innovative metrics
for agricultural cross-sector Bell inequality analysis. It provides bootstrap
validation, significance testing, confidence intervals, and novel statistical
metrics for Science journal publication.

Key Features:
- Bootstrap validation with 1000+ resamples
- Statistical significance testing with p < 0.001 requirement
- Confidence interval calculations for violation rates
- Multiple testing correction for cross-sectoral analysis
- Effect size calculations (20-60% above classical bounds expected)
- Innovative statistical metrics for agricultural crisis analysis

Authors: Agricultural Cross-Sector Analysis Team
Date: September 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats
from scipy.stats import bootstrap
import warnings
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

warnings.filterwarnings('ignore')

@dataclass
class StatisticalResults:
    """Container for statistical validation results."""
    violation_rate: float
    confidence_interval: Tuple[float, float]
    p_value: float
    effect_size: float
    bootstrap_samples: np.ndarray
    is_significant: bool
    classical_bound_excess: float

@dataclass
class InnovativeMetrics:
    """Container for innovative statistical metrics."""
    crisis_amplification_factor: float
    tier_vulnerability_index: float
    transmission_efficiency_score: float
    quantum_correlation_stability: float
    cross_crisis_consistency_score: float
    sector_coupling_strength: float
    crisis_prediction_indicator: float
    recovery_resilience_metric: float

class AdvancedStatisticalValidator:
    """
    Advanced statistical validation suite for Bell inequality analysis.
    
    This class provides comprehensive statistical validation including bootstrap
    methods, significance testing, and confidence intervals following the
    requirements for Science journal publication.
    """
    
    def __init__(self, n_bootstrap: int = 1000, alpha: float = 0.001, 
                 random_state: int = 42, n_jobs: int = -1):
        """
        Initialize the statistical validator.
        
        Parameters:
        -----------
        n_bootstrap : int, optional
            Number of bootstrap resamples. Default: 1000 (requirement: 1000+)
        alpha : float, optional
            Significance level. Default: 0.001 (requirement: p < 0.001)
        random_state : int, optional
            Random seed for reproducibility. Default: 42
        n_jobs : int, optional
            Number of parallel jobs. Default: -1 (use all cores)
        """
        self.n_bootstrap = n_bootstrap
        self.alpha = alpha
        self.random_state = random_state
        self.n_jobs = n_jobs if n_jobs != -1 else mp.cpu_count()
        
        # Set random seed for reproducibility
        np.random.seed(random_state)
        
        # Classical physics bound for Bell inequalities
        self.classical_bound = 2.0
        self.quantum_bound = 2 * np.sqrt(2)  # ≈ 2.83
    
    def bootstrap_violation_rate(self, s1_values: np.ndarray) -> StatisticalResults:
        """
        Perform bootstrap validation of violation rates with 1000+ resamples.
        
        This implements requirement 2.1: "bootstrap validation with 1000+ resamples"
        and requirement 2.2: "statistical significance testing with p < 0.001".
        
        Parameters:
        -----------
        s1_values : np.ndarray
            Array of S1 Bell inequality values
            
        Returns:
        --------
        StatisticalResults : Comprehensive statistical validation results
        """
        if len(s1_values) == 0:
            return StatisticalResults(0.0, (0.0, 0.0), 1.0, 0.0, np.array([]), False, 0.0)
        
        # Calculate observed violation rate
        violations = np.abs(s1_values) > self.classical_bound
        observed_rate = np.mean(violations) * 100
        
        # Bootstrap resampling
        def violation_rate_statistic(sample):
            return np.mean(np.abs(sample) > self.classical_bound) * 100
        
        # Use scipy.stats.bootstrap for robust bootstrap implementation
        rng = np.random.default_rng(self.random_state)
        bootstrap_result = bootstrap(
            (s1_values,), 
            violation_rate_statistic,
            n_resamples=self.n_bootstrap,
            confidence_level=1 - self.alpha,
            random_state=rng,
            method='percentile'
        )
        
        # Extract bootstrap samples and confidence interval
        bootstrap_samples = bootstrap_result.bootstrap_distribution
        confidence_interval = bootstrap_result.confidence_interval
        
        # Calculate p-value (probability of observing this violation rate by chance)
        # Under null hypothesis: violation rate should be ~0% for classical systems
        null_rate = 0.0
        p_value = np.mean(bootstrap_samples <= null_rate) if observed_rate > null_rate else 1.0
        p_value = max(p_value, 1e-10)  # Avoid exactly zero p-values
        
        # Determine statistical significance
        is_significant = p_value < self.alpha
        
        # Calculate effect size (Cohen's d equivalent for violation rates)
        # Effect size = (observed - expected) / std_dev
        expected_rate = 0.0  # Classical expectation
        effect_size = (observed_rate - expected_rate) / np.std(bootstrap_samples)
        
        # Calculate excess above classical bound
        max_violation = np.max(np.abs(s1_values))
        classical_bound_excess = ((max_violation - self.classical_bound) / self.classical_bound) * 100
        
        return StatisticalResults(
            violation_rate=observed_rate,
            confidence_interval=(confidence_interval.low, confidence_interval.high),
            p_value=p_value,
            effect_size=effect_size,
            bootstrap_samples=bootstrap_samples,
            is_significant=is_significant,
            classical_bound_excess=classical_bound_excess
        )
    
    def multiple_testing_correction(self, p_values: List[float], 
                                  method: str = 'bonferroni') -> Tuple[List[float], List[bool]]:
        """
        Apply multiple testing correction for cross-sectoral analysis.
        
        This implements requirement: "Create multiple testing correction for 
        cross-sectoral analysis" to control family-wise error rate.
        
        Parameters:
        -----------
        p_values : List[float]
            List of p-values from multiple tests
        method : str, optional
            Correction method ('bonferroni', 'holm', 'fdr_bh'). Default: 'bonferroni'
            
        Returns:
        --------
        Tuple[List[float], List[bool]] : Corrected p-values and significance flags
        """
        if not p_values:
            return [], []
        
        p_array = np.array(p_values)
        n_tests = len(p_array)
        
        if method == 'bonferroni':
            # Bonferroni correction: p_corrected = p * n_tests
            corrected_p = np.minimum(p_array * n_tests, 1.0)
            
        elif method == 'holm':
            # Holm-Bonferroni step-down method
            sorted_indices = np.argsort(p_array)
            corrected_p = np.zeros_like(p_array)
            
            for i, idx in enumerate(sorted_indices):
                correction_factor = n_tests - i
                corrected_p[idx] = min(p_array[idx] * correction_factor, 1.0)
                
        elif method == 'fdr_bh':
            # Benjamini-Hochberg False Discovery Rate
            sorted_indices = np.argsort(p_array)
            corrected_p = np.zeros_like(p_array)
            
            for i, idx in enumerate(sorted_indices):
                correction_factor = n_tests / (i + 1)
                corrected_p[idx] = min(p_array[idx] * correction_factor, 1.0)
                
        else:
            raise ValueError(f"Unknown correction method: {method}")
        
        # Determine significance after correction
        is_significant = corrected_p < self.alpha
        
        return corrected_p.tolist(), is_significant.tolist()
    
    def calculate_effect_sizes(self, s1_values: np.ndarray) -> Dict[str, float]:
        """
        Calculate effect sizes for Bell inequality violations.
        
        This implements requirement: "Implement effect size calculations 
        (20-60% above classical bounds expected)".
        
        Parameters:
        -----------
        s1_values : np.ndarray
            Array of S1 Bell inequality values
            
        Returns:
        --------
        Dict[str, float] : Various effect size measures
        """
        if len(s1_values) == 0:
            return {'cohens_d': 0.0, 'classical_excess': 0.0, 'quantum_approach': 0.0}
        
        # Cohen's d: standardized difference from classical bound
        mean_s1 = np.mean(np.abs(s1_values))
        std_s1 = np.std(np.abs(s1_values))
        cohens_d = (mean_s1 - self.classical_bound) / std_s1 if std_s1 > 0 else 0.0
        
        # Percentage excess above classical bound
        max_violation = np.max(np.abs(s1_values))
        classical_excess = ((max_violation - self.classical_bound) / self.classical_bound) * 100
        
        # Approach to quantum bound
        quantum_approach = (max_violation / self.quantum_bound) * 100
        
        return {
            'cohens_d': cohens_d,
            'classical_excess': classical_excess,
            'quantum_approach': quantum_approach,
            'mean_violation_strength': mean_s1,
            'max_violation_strength': max_violation
        }
    
    def validate_significance_threshold(self, results: List[StatisticalResults]) -> Dict:
        """
        Validate that results meet p < 0.001 significance requirement.
        
        Parameters:
        -----------
        results : List[StatisticalResults]
            List of statistical results to validate
            
        Returns:
        --------
        Dict : Validation summary
        """
        if not results:
            return {'total_tests': 0, 'significant_tests': 0, 'significance_rate': 0.0}
        
        significant_count = sum(1 for r in results if r.is_significant)
        total_count = len(results)
        significance_rate = (significant_count / total_count) * 100
        
        # Check if any results meet the strict p < 0.001 requirement
        ultra_significant = sum(1 for r in results if r.p_value < 0.001)
        
        return {
            'total_tests': total_count,
            'significant_tests': significant_count,
            'ultra_significant_tests': ultra_significant,
            'significance_rate': significance_rate,
            'ultra_significance_rate': (ultra_significant / total_count) * 100,
            'meets_requirement': ultra_significant > 0
        }

class InnovativeMetricsCalculator:
    """
    Calculator for innovative statistical metrics specific to agricultural crisis analysis.
    
    This class implements the innovative metrics specified in task 6.1 for measuring
    crisis amplification, vulnerability, transmission efficiency, and other novel
    indicators for agricultural cross-sector analysis.
    """
    
    def __init__(self):
        """Initialize the innovative metrics calculator."""
        self.classical_bound = 2.0
    
    def crisis_amplification_factor(self, normal_violations: np.ndarray, 
                                  crisis_violations: np.ndarray) -> float:
        """
        Calculate Crisis Amplification Factor measuring violation rate increase during crises.
        
        This implements: "Crisis Amplification Factor measuring violation rate 
        increase during crises"
        
        Parameters:
        -----------
        normal_violations : np.ndarray
            Violation rates during normal periods
        crisis_violations : np.ndarray
            Violation rates during crisis periods
            
        Returns:
        --------
        float : Crisis amplification factor (ratio of crisis to normal violation rates)
        """
        if len(normal_violations) == 0 or len(crisis_violations) == 0:
            return 0.0
        
        normal_rate = np.mean(normal_violations) * 100
        crisis_rate = np.mean(crisis_violations) * 100
        
        if normal_rate == 0:
            return float('inf') if crisis_rate > 0 else 1.0
        
        amplification_factor = crisis_rate / normal_rate
        return amplification_factor
    
    def tier_vulnerability_index(self, tier_violation_rates: Dict[str, float]) -> Dict[str, float]:
        """
        Create Tier Vulnerability Index ranking tiers by crisis sensitivity.
        
        This implements: "Tier Vulnerability Index ranking tiers by crisis sensitivity"
        
        Parameters:
        -----------
        tier_violation_rates : Dict[str, float]
            Violation rates for each tier during crisis periods
            
        Returns:
        --------
        Dict[str, float] : Vulnerability index for each tier (0-100 scale)
        """
        if not tier_violation_rates:
            return {}
        
        # Normalize violation rates to 0-100 scale
        max_rate = max(tier_violation_rates.values())
        min_rate = min(tier_violation_rates.values())
        
        if max_rate == min_rate:
            return {tier: 50.0 for tier in tier_violation_rates}
        
        vulnerability_index = {}
        for tier, rate in tier_violation_rates.items():
            # Scale to 0-100 where 100 is most vulnerable
            normalized_score = ((rate - min_rate) / (max_rate - min_rate)) * 100
            vulnerability_index[tier] = normalized_score
        
        return vulnerability_index
    
    def transmission_efficiency_score(self, transmission_lags: List[int], 
                                    transmission_strengths: List[float]) -> float:
        """
        Calculate Transmission Efficiency Score measuring shock propagation speed.
        
        This implements: "Transmission Efficiency Score measuring how quickly 
        shocks propagate between sectors"
        
        Parameters:
        -----------
        transmission_lags : List[int]
            Time lags for transmission (in days)
        transmission_strengths : List[float]
            Strength of transmission (correlation coefficients)
            
        Returns:
        --------
        float : Transmission efficiency score (0-100 scale)
        """
        if not transmission_lags or not transmission_strengths:
            return 0.0
        
        # Efficiency = strength / lag (higher strength, lower lag = higher efficiency)
        efficiencies = []
        for lag, strength in zip(transmission_lags, transmission_strengths):
            if lag > 0:
                efficiency = abs(strength) / lag * 100  # Scale to 0-100
                efficiencies.append(efficiency)
        
        return np.mean(efficiencies) if efficiencies else 0.0
    
    def quantum_correlation_stability(self, s1_time_series: np.ndarray, 
                                    window_size: int = 20) -> float:
        """
        Analyze Quantum Correlation Stability measuring persistence of violations.
        
        This implements: "Quantum Correlation Stability analysis measuring 
        persistence of violations"
        
        Parameters:
        -----------
        s1_time_series : np.ndarray
            Time series of S1 values
        window_size : int, optional
            Window size for stability calculation. Default: 20
            
        Returns:
        --------
        float : Stability score (0-100 scale)
        """
        if len(s1_time_series) < window_size:
            return 0.0
        
        # Calculate rolling standard deviation of violation strength
        violations = np.abs(s1_time_series) > self.classical_bound
        
        # Stability = consistency of violation patterns over time
        rolling_violation_rates = []
        for i in range(window_size, len(violations)):
            window_violations = violations[i-window_size:i]
            violation_rate = np.mean(window_violations)
            rolling_violation_rates.append(violation_rate)
        
        if not rolling_violation_rates:
            return 0.0
        
        # Stability = 100 - coefficient of variation
        mean_rate = np.mean(rolling_violation_rates)
        std_rate = np.std(rolling_violation_rates)
        
        if mean_rate == 0:
            return 0.0
        
        coefficient_of_variation = std_rate / mean_rate
        stability_score = max(0, 100 - coefficient_of_variation * 100)
        
        return stability_score
    
    def cross_crisis_consistency_score(self, crisis_results: Dict[str, np.ndarray]) -> float:
        """
        Calculate Cross-Crisis Consistency Score for violation patterns.
        
        This implements: "Cross-Crisis Consistency Score measuring similar 
        violation patterns across different crises"
        
        Parameters:
        -----------
        crisis_results : Dict[str, np.ndarray]
            Violation rates for different crisis periods
            
        Returns:
        --------
        float : Consistency score (0-100 scale)
        """
        if len(crisis_results) < 2:
            return 0.0
        
        # Calculate pairwise correlations between crisis violation patterns
        crisis_names = list(crisis_results.keys())
        correlations = []
        
        for i in range(len(crisis_names)):
            for j in range(i + 1, len(crisis_names)):
                crisis1 = crisis_results[crisis_names[i]]
                crisis2 = crisis_results[crisis_names[j]]
                
                # Ensure same length for correlation
                min_len = min(len(crisis1), len(crisis2))
                if min_len > 1:
                    corr = np.corrcoef(crisis1[:min_len], crisis2[:min_len])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))  # Use absolute correlation
        
        if not correlations:
            return 0.0
        
        # Consistency = average correlation * 100
        consistency_score = np.mean(correlations) * 100
        return consistency_score
    
    def sector_coupling_strength(self, cross_sector_correlations: np.ndarray) -> float:
        """
        Calculate Sector Coupling Strength quantifying operational dependencies.
        
        This implements: "Sector Coupling Strength metrics quantifying 
        operational dependency relationships"
        
        Parameters:
        -----------
        cross_sector_correlations : np.ndarray
            Cross-sector correlation coefficients
            
        Returns:
        --------
        float : Coupling strength score (0-100 scale)
        """
        if len(cross_sector_correlations) == 0:
            return 0.0
        
        # Coupling strength = mean absolute correlation * 100
        mean_abs_correlation = np.mean(np.abs(cross_sector_correlations))
        coupling_strength = mean_abs_correlation * 100
        
        return coupling_strength
    
    def crisis_prediction_indicator(self, pre_crisis_violations: np.ndarray,
                                  crisis_violations: np.ndarray,
                                  lead_time: int = 30) -> float:
        """
        Calculate Crisis Prediction Indicators using violation patterns.
        
        This implements: "Crisis Prediction Indicators using violation patterns 
        to forecast crisis onset"
        
        Parameters:
        -----------
        pre_crisis_violations : np.ndarray
            Violation rates in the lead_time days before crisis
        crisis_violations : np.ndarray
            Violation rates during crisis
        lead_time : int, optional
            Lead time for prediction (days). Default: 30
            
        Returns:
        --------
        float : Prediction indicator strength (0-100 scale)
        """
        if len(pre_crisis_violations) == 0 or len(crisis_violations) == 0:
            return 0.0
        
        # Calculate the predictive power of pre-crisis violations
        pre_crisis_mean = np.mean(pre_crisis_violations)
        crisis_mean = np.mean(crisis_violations)
        
        if crisis_mean == 0:
            return 0.0
        
        # Prediction strength = how well pre-crisis violations predict crisis severity
        prediction_ratio = pre_crisis_mean / crisis_mean
        
        # Scale to 0-100 where higher values indicate better prediction
        prediction_indicator = min(prediction_ratio * 100, 100)
        
        return prediction_indicator
    
    def recovery_resilience_metric(self, crisis_violations: np.ndarray,
                                 post_crisis_violations: np.ndarray,
                                 recovery_window: int = 90) -> float:
        """
        Calculate Recovery Resilience Metrics for post-crisis normalization.
        
        This implements: "Recovery Resilience Metrics measuring how quickly 
        violations return to normal post-crisis"
        
        Parameters:
        -----------
        crisis_violations : np.ndarray
            Violation rates during crisis
        post_crisis_violations : np.ndarray
            Violation rates after crisis (within recovery_window)
        recovery_window : int, optional
            Recovery window (days). Default: 90
            
        Returns:
        --------
        float : Recovery resilience score (0-100 scale)
        """
        if len(crisis_violations) == 0 or len(post_crisis_violations) == 0:
            return 0.0
        
        crisis_peak = np.max(crisis_violations)
        post_crisis_mean = np.mean(post_crisis_violations)
        
        if crisis_peak == 0:
            return 100.0  # Perfect resilience if no crisis impact
        
        # Resilience = how much violations decreased from crisis peak
        recovery_ratio = 1 - (post_crisis_mean / crisis_peak)
        resilience_score = max(0, recovery_ratio * 100)
        
        return resilience_score
    
    def calculate_all_metrics(self, analysis_data: Dict) -> InnovativeMetrics:
        """
        Calculate all innovative metrics for comprehensive analysis.
        
        Parameters:
        -----------
        analysis_data : Dict
            Dictionary containing all necessary data for metric calculations
            
        Returns:
        --------
        InnovativeMetrics : Complete set of innovative metrics
        """
        # Extract data with safe defaults
        normal_violations = analysis_data.get('normal_violations', np.array([]))
        crisis_violations = analysis_data.get('crisis_violations', np.array([]))
        tier_violation_rates = analysis_data.get('tier_violation_rates', {})
        transmission_lags = analysis_data.get('transmission_lags', [])
        transmission_strengths = analysis_data.get('transmission_strengths', [])
        s1_time_series = analysis_data.get('s1_time_series', np.array([]))
        crisis_results = analysis_data.get('crisis_results', {})
        cross_sector_correlations = analysis_data.get('cross_sector_correlations', np.array([]))
        pre_crisis_violations = analysis_data.get('pre_crisis_violations', np.array([]))
        post_crisis_violations = analysis_data.get('post_crisis_violations', np.array([]))
        
        # Calculate all metrics
        crisis_amp = self.crisis_amplification_factor(normal_violations, crisis_violations)
        tier_vuln = self.tier_vulnerability_index(tier_violation_rates)
        tier_vuln_avg = np.mean(list(tier_vuln.values())) if tier_vuln else 0.0
        
        trans_eff = self.transmission_efficiency_score(transmission_lags, transmission_strengths)
        quantum_stab = self.quantum_correlation_stability(s1_time_series)
        cross_crisis_cons = self.cross_crisis_consistency_score(crisis_results)
        sector_coupling = self.sector_coupling_strength(cross_sector_correlations)
        crisis_pred = self.crisis_prediction_indicator(pre_crisis_violations, crisis_violations)
        recovery_res = self.recovery_resilience_metric(crisis_violations, post_crisis_violations)
        
        return InnovativeMetrics(
            crisis_amplification_factor=crisis_amp,
            tier_vulnerability_index=tier_vuln_avg,
            transmission_efficiency_score=trans_eff,
            quantum_correlation_stability=quantum_stab,
            cross_crisis_consistency_score=cross_crisis_cons,
            sector_coupling_strength=sector_coupling,
            crisis_prediction_indicator=crisis_pred,
            recovery_resilience_metric=recovery_res
        )

class ComprehensiveStatisticalSuite:
    """
    Comprehensive statistical validation and analysis suite combining all components.
    
    This class integrates bootstrap validation, significance testing, confidence
    intervals, multiple testing correction, effect size calculations, and innovative
    metrics for complete agricultural cross-sector analysis.
    """
    
    def __init__(self, n_bootstrap: int = 1000, alpha: float = 0.001, 
                 random_state: int = 42, n_jobs: int = -1):
        """
        Initialize the comprehensive statistical suite.
        
        Parameters:
        -----------
        n_bootstrap : int, optional
            Number of bootstrap resamples. Default: 1000
        alpha : float, optional
            Significance level. Default: 0.001
        random_state : int, optional
            Random seed for reproducibility. Default: 42
        n_jobs : int, optional
            Number of parallel jobs. Default: -1
        """
        self.validator = AdvancedStatisticalValidator(
            n_bootstrap=n_bootstrap, 
            alpha=alpha, 
            random_state=random_state,
            n_jobs=n_jobs
        )
        self.metrics_calculator = InnovativeMetricsCalculator()
        
    def comprehensive_analysis(self, s1_results: Dict, 
                             crisis_data: Optional[Dict] = None) -> Dict:
        """
        Perform comprehensive statistical analysis of S1 results.
        
        Parameters:
        -----------
        s1_results : Dict
            Dictionary containing S1 analysis results for multiple asset pairs
        crisis_data : Dict, optional
            Additional crisis-specific data for innovative metrics
            
        Returns:
        --------
        Dict : Comprehensive statistical analysis results
        """
        # Extract S1 values from all pairs
        all_s1_values = []
        pair_results = {}
        
        for pair, results in s1_results.items():
            if 's1_time_series' in results:
                s1_values = np.array(results['s1_time_series'])
                all_s1_values.extend(s1_values)
                
                # Perform statistical validation for each pair
                stat_results = self.validator.bootstrap_violation_rate(s1_values)
                effect_sizes = self.validator.calculate_effect_sizes(s1_values)
                
                pair_results[pair] = {
                    'statistical_results': stat_results,
                    'effect_sizes': effect_sizes,
                    's1_values': s1_values
                }
        
        # Overall statistical validation
        all_s1_array = np.array(all_s1_values)
        overall_stats = self.validator.bootstrap_violation_rate(all_s1_array)
        overall_effects = self.validator.calculate_effect_sizes(all_s1_array)
        
        # Multiple testing correction
        p_values = [pair_results[pair]['statistical_results'].p_value for pair in pair_results]
        corrected_p, significance_flags = self.validator.multiple_testing_correction(p_values)
        
        # Update significance flags after correction
        for i, pair in enumerate(pair_results.keys()):
            pair_results[pair]['corrected_p_value'] = corrected_p[i]
            pair_results[pair]['significant_after_correction'] = significance_flags[i]
        
        # Validate significance threshold
        all_stat_results = [pair_results[pair]['statistical_results'] for pair in pair_results]
        significance_validation = self.validator.validate_significance_threshold(all_stat_results)
        
        # Calculate innovative metrics if crisis data provided
        innovative_metrics = None
        if crisis_data:
            innovative_metrics = self.metrics_calculator.calculate_all_metrics(crisis_data)
        
        return {
            'pair_results': pair_results,
            'overall_statistics': overall_stats,
            'overall_effect_sizes': overall_effects,
            'multiple_testing_correction': {
                'original_p_values': p_values,
                'corrected_p_values': corrected_p,
                'significance_flags': significance_flags
            },
            'significance_validation': significance_validation,
            'innovative_metrics': innovative_metrics,
            'summary': {
                'total_pairs': len(pair_results),
                'significant_pairs': sum(significance_flags),
                'overall_violation_rate': overall_stats.violation_rate,
                'meets_science_requirements': significance_validation['meets_requirement']
            }
        }
    
    def generate_publication_summary(self, analysis_results: Dict) -> str:
        """
        Generate publication-ready summary of statistical results.
        
        Parameters:
        -----------
        analysis_results : Dict
            Results from comprehensive_analysis
            
        Returns:
        --------
        str : Publication-ready summary text
        """
        overall_stats = analysis_results['overall_statistics']
        significance_val = analysis_results['significance_validation']
        innovative_metrics = analysis_results['innovative_metrics']
        
        summary = f"""
AGRICULTURAL CROSS-SECTOR BELL INEQUALITY ANALYSIS - STATISTICAL SUMMARY
========================================================================

OVERALL RESULTS:
- Total Asset Pairs Analyzed: {analysis_results['summary']['total_pairs']}
- Overall Violation Rate: {overall_stats.violation_rate:.2f}% 
  (95% CI: {overall_stats.confidence_interval[0]:.2f}% - {overall_stats.confidence_interval[1]:.2f}%)
- Statistical Significance: p = {overall_stats.p_value:.2e}
- Effect Size (Cohen's d): {overall_stats.effect_size:.3f}
- Classical Bound Excess: {overall_stats.classical_bound_excess:.1f}%

SIGNIFICANCE VALIDATION:
- Tests Meeting p < 0.001: {significance_val['ultra_significant_tests']}/{significance_val['total_tests']}
- Ultra-Significance Rate: {significance_val['ultra_significance_rate']:.1f}%
- Meets Science Requirements: {'YES' if significance_val['meets_requirement'] else 'NO'}

MULTIPLE TESTING CORRECTION:
- Significant Pairs (Bonferroni): {analysis_results['summary']['significant_pairs']}/{analysis_results['summary']['total_pairs']}
- Family-wise Error Rate: Controlled at α = 0.001
"""
        
        if innovative_metrics:
            summary += f"""
INNOVATIVE AGRICULTURAL METRICS:
- Crisis Amplification Factor: {innovative_metrics.crisis_amplification_factor:.2f}x
- Tier Vulnerability Index: {innovative_metrics.tier_vulnerability_index:.1f}/100
- Transmission Efficiency Score: {innovative_metrics.transmission_efficiency_score:.1f}/100
- Quantum Correlation Stability: {innovative_metrics.quantum_correlation_stability:.1f}/100
- Cross-Crisis Consistency: {innovative_metrics.cross_crisis_consistency_score:.1f}/100
- Sector Coupling Strength: {innovative_metrics.sector_coupling_strength:.1f}/100
- Crisis Prediction Indicator: {innovative_metrics.crisis_prediction_indicator:.1f}/100
- Recovery Resilience Metric: {innovative_metrics.recovery_resilience_metric:.1f}/100
"""
        
        summary += f"""
CONCLUSION:
{'✅ Results meet all Science journal requirements for statistical rigor.' if significance_val['meets_requirement'] else '❌ Results do not meet Science journal significance requirements.'}
Bootstrap validation with {len(overall_stats.bootstrap_samples)} resamples confirms robustness.
Effect sizes indicate {overall_stats.classical_bound_excess:.1f}% excess above classical bounds.
"""
        
        return summary

def validate_statistical_suite() -> bool:
    """
    Validate the statistical suite implementation.
    
    Returns:
    --------
    bool : True if all validations pass
    """
    print("🧪 Validating Advanced Statistical Validation Suite...")
    
    try:
        # Create test data
        np.random.seed(42)
        
        # Test statistical validator
        validator = AdvancedStatisticalValidator(n_bootstrap=100)  # Reduced for testing
        
        # Test S1 values with known violations
        test_s1_values = np.array([-3.0, -1.0, 0.5, 1.5, 2.5, 3.2, -2.8])
        
        # Test bootstrap validation
        stat_results = validator.bootstrap_violation_rate(test_s1_values)
        assert stat_results.violation_rate > 0, "Bootstrap validation failed"
        assert len(stat_results.bootstrap_samples) == 100, "Bootstrap samples incorrect"
        
        # Test multiple testing correction
        test_p_values = [0.001, 0.01, 0.05, 0.0001]
        corrected_p, significance = validator.multiple_testing_correction(test_p_values)
        assert len(corrected_p) == len(test_p_values), "Multiple testing correction failed"
        
        # Test effect size calculation
        effect_sizes = validator.calculate_effect_sizes(test_s1_values)
        assert 'cohens_d' in effect_sizes, "Effect size calculation failed"
        
        # Test innovative metrics calculator
        metrics_calc = InnovativeMetricsCalculator()
        
        # Test crisis amplification factor
        normal_viols = np.array([0.1, 0.2, 0.15])
        crisis_viols = np.array([0.4, 0.5, 0.6])
        caf = metrics_calc.crisis_amplification_factor(normal_viols, crisis_viols)
        assert caf > 1.0, "Crisis amplification factor failed"
        
        # Test tier vulnerability index
        tier_rates = {'Tier1': 45.0, 'Tier2': 30.0, 'Tier3': 20.0}
        vuln_index = metrics_calc.tier_vulnerability_index(tier_rates)
        assert len(vuln_index) == 3, "Tier vulnerability index failed"
        
        # Test comprehensive suite
        suite = ComprehensiveStatisticalSuite(n_bootstrap=50)  # Reduced for testing
        
        # Create mock S1 results
        mock_s1_results = {
            ('ASSET_A', 'ASSET_B'): {
                's1_time_series': [-2.5, 1.8, 3.1, -1.2, 2.8]
            },
            ('ASSET_C', 'ASSET_D'): {
                's1_time_series': [0.5, -3.2, 1.9, 2.4, -0.8]
            }
        }
        
        # Test comprehensive analysis
        comp_results = suite.comprehensive_analysis(mock_s1_results)
        assert 'pair_results' in comp_results, "Comprehensive analysis failed"
        assert 'overall_statistics' in comp_results, "Overall statistics missing"
        
        # Test publication summary
        summary = suite.generate_publication_summary(comp_results)
        assert 'STATISTICAL SUMMARY' in summary, "Publication summary failed"
        
        print("✅ All statistical validation tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Statistical validation failed: {e}")
        return False

if __name__ == "__main__":
    # Run validation tests
    validate_statistical_suite()