#!/usr/bin/env python3
"""
ENHANCED S1 BELL INEQUALITY CALCULATOR
=====================================

This module implements the mathematically accurate S1 Bell inequality calculator
following the exact specifications from Zarifian et al. (2025) for agricultural
cross-sector analysis. The implementation ensures scientific validity and 
reproducibility for Science journal publication.

Key Mathematical Components:
- Exact daily returns calculation: Ri,t = (Pi,t - Pi,t-1) / Pi,t-1
- Binary indicator functions: I{|RA,t| ≥ rA} for regime classification
- Sign function: Sign(Ri,t) = +1 if Ri,t ≥ 0, -1 if Ri,t < 0
- Conditional expectations: ⟨ab⟩xy = Σ[sign(RA,t)sign(RB,t)I{conditions}] / Σ[I{conditions}]
- S1 formula: S1 = ⟨ab⟩00 + ⟨ab⟩01 + ⟨ab⟩10 - ⟨ab⟩11
- Missing data handling: set ⟨ab⟩xy = 0 if no valid observations

Authors: Agricultural Cross-Sector Analysis Team
Date: September 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional, Union
import warnings
warnings.filterwarnings('ignore')

class EnhancedS1Calculator:
    """
    Enhanced S1 Bell inequality calculator with mathematical accuracy.
    
    This class implements the exact S1 conditional Bell inequality methodology
    specified in Zarifian et al. (2025) with precise mathematical formulations
    for agricultural cross-sector analysis.
    """
    
    def __init__(self, window_size: int = 20, threshold_method: str = 'absolute', 
                 threshold_value: float = 0.01, threshold_quantile: float = 0.75):
        """
        Initialize the Enhanced S1 Calculator.
        
        Parameters:
        -----------
        window_size : int, optional
            Rolling window size for analysis. Default: 20
            For crisis periods, use 15 as specified in requirements.
        threshold_method : str, optional
            Method for determining thresholds:
            - 'absolute': Fixed absolute return threshold (Zarifian et al. 2025)
            - 'quantile': Quantile-based threshold (alternative method)
            Default: 'absolute' (following Zarifian et al. 2025)
        threshold_value : float, optional
            Fixed absolute return threshold for 'absolute' method.
            Zarifian et al. (2025) suggest choosing based on data volatility:
            - For typical daily returns (σ ≈ 0.02): use 0.01-0.02
            - For high volatility periods: use 0.02-0.05  
            - For crisis detection: use higher thresholds to filter noise
            Default: 0.01 (good coverage for typical volatility)
        threshold_quantile : float, optional
            Quantile for regime threshold when using 'quantile' method.
            Default: 0.75. For crisis periods, use 0.8 as specified in requirements.
        """
        self.window_size = window_size
        self.threshold_method = threshold_method
        self.threshold_value = threshold_value
        self.threshold_quantile = threshold_quantile
        
        # Reduced verbosity - only print if explicitly requested
        self.verbose = False
    
    def calculate_daily_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate exact daily returns using the formula: Ri,t = (Pi,t - Pi,t-1) / Pi,t-1
        
        This implements the exact daily returns calculation as specified in 
        Zarifian et al. (2025) and requirement 5.3.
        
        Parameters:
        -----------
        prices : pd.DataFrame
            Price data with assets as columns and dates as index
            
        Returns:
        --------
        pd.DataFrame : Daily returns with same structure as input
        """
        if prices is None or prices.empty:
            raise ValueError("Price data cannot be None or empty")
        
        # Exact daily returns formula: Ri,t = (Pi,t - Pi,t-1) / Pi,t-1
        returns = prices.pct_change()
        
        # Remove any infinite or NaN values
        returns = returns.replace([np.inf, -np.inf], np.nan)
        returns = returns.dropna()
        
        if self.verbose:
            print(f"✅ Daily returns calculated: {returns.shape[0]} observations, {returns.shape[1]} assets")
        
        return returns
    
    def compute_binary_indicators(self, returns: pd.DataFrame, 
                                thresholds: Union[Dict, pd.Series]) -> Dict[str, pd.DataFrame]:
        """
        Create binary indicator functions I{|RA,t| ≥ rA} for regime classification.
        
        This implements the exact binary indicator specification from requirement 7.1:
        I{|RA,t| ≥ rA} where rA is the threshold for strong price movements.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Daily returns data
        thresholds : Dict or pd.Series
            Threshold values for each asset
            
        Returns:
        --------
        Dict[str, pd.DataFrame] : Dictionary containing binary indicators for each regime
        """
        if returns is None or returns.empty:
            raise ValueError("Returns data cannot be None or empty")
        
        indicators = {}
        
        for asset in returns.columns:
            if asset not in thresholds:
                continue
                
            threshold = thresholds[asset]
            abs_returns = returns[asset].abs()
            
            # Binary indicator: I{|RA,t| ≥ rA}
            # True when absolute return exceeds threshold (strong movement)
            # False when absolute return is below threshold (weak movement)
            strong_movement = abs_returns >= threshold
            weak_movement = ~strong_movement
            
            indicators[f"{asset}_strong"] = strong_movement
            indicators[f"{asset}_weak"] = weak_movement
        
        # Convert to DataFrame for easier handling
        indicators_df = pd.DataFrame(indicators, index=returns.index)
        
        if self.verbose:
            print(f"✅ Binary indicators computed for {len(returns.columns)} assets")
        
        return {
            'indicators': indicators_df,
            'thresholds': thresholds
        }
    
    def calculate_sign_outcomes(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Implement sign function: Sign(Ri,t) = +1 if Ri,t ≥ 0, -1 if Ri,t < 0
        
        This implements the exact sign function specification from requirement 7.2.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Daily returns data
            
        Returns:
        --------
        pd.DataFrame : Sign outcomes (+1, -1) for each asset
        """
        if returns is None or returns.empty:
            raise ValueError("Returns data cannot be None or empty")
        
        # Sign function: +1 if Ri,t ≥ 0, -1 if Ri,t < 0
        # Note: np.sign returns 0 for exactly 0, but requirement specifies +1 for ≥ 0
        signs = returns.copy()
        signs[returns >= 0] = 1
        signs[returns < 0] = -1
        
        if self.verbose:
            print(f"✅ Sign outcomes calculated for {signs.shape[0]} observations, {signs.shape[1]} assets")
        
        return signs
    
    def calculate_conditional_expectations(self, signs: pd.DataFrame, 
                                         indicators: pd.DataFrame,
                                         asset_a: str, asset_b: str) -> Dict[str, float]:
        """
        Calculate conditional expectations using the exact formula:
        ⟨ab⟩xy = Σ[sign(RA,t)sign(RB,t)I{|RA,t|≥rA}I{|RB,t|≥rB}] / Σ[I{|RA,t|≥rA}I{|RB,t|≥rB}]
        
        This implements the exact conditional expectation formula from requirement 2.3.
        
        Parameters:
        -----------
        signs : pd.DataFrame
            Sign outcomes for all assets
        indicators : pd.DataFrame
            Binary indicators for all assets and regimes
        asset_a : str
            First asset symbol
        asset_b : str
            Second asset symbol
            
        Returns:
        --------
        Dict[str, float] : Conditional expectations for all four regimes
        """
        if signs is None or indicators is None:
            raise ValueError("Signs and indicators data cannot be None")
        
        if asset_a not in signs.columns or asset_b not in signs.columns:
            raise ValueError(f"Assets {asset_a} or {asset_b} not found in signs data")
        
        # Get sign outcomes for both assets
        sign_a = signs[asset_a]
        sign_b = signs[asset_b]
        
        # Get binary indicators for both assets
        a_strong = indicators[f"{asset_a}_strong"]
        a_weak = indicators[f"{asset_a}_weak"]
        b_strong = indicators[f"{asset_b}_strong"]
        b_weak = indicators[f"{asset_b}_weak"]
        
        # Calculate conditional expectations for all four regimes
        # Regime notation: xy where x is asset A regime, y is asset B regime
        # 0 = strong movement (high absolute return), 1 = weak movement (low absolute return)
        
        expectations = {}
        
        # ⟨ab⟩00: Both assets have strong movements
        mask_00 = a_strong & b_strong
        expectations['ab_00'] = self._compute_expectation(sign_a, sign_b, mask_00)
        
        # ⟨ab⟩01: Asset A strong, Asset B weak
        mask_01 = a_strong & b_weak
        expectations['ab_01'] = self._compute_expectation(sign_a, sign_b, mask_01)
        
        # ⟨ab⟩10: Asset A weak, Asset B strong
        mask_10 = a_weak & b_strong
        expectations['ab_10'] = self._compute_expectation(sign_a, sign_b, mask_10)
        
        # ⟨ab⟩11: Both assets have weak movements
        mask_11 = a_weak & b_weak
        expectations['ab_11'] = self._compute_expectation(sign_a, sign_b, mask_11)
        
        return expectations
    
    def _compute_expectation(self, sign_a: pd.Series, sign_b: pd.Series, 
                           mask: pd.Series) -> float:
        """
        Compute conditional expectation for a specific regime.
        
        Implements: ⟨ab⟩xy = Σ[sign(RA,t)sign(RB,t)I{conditions}] / Σ[I{conditions}]
        
        Parameters:
        -----------
        sign_a : pd.Series
            Sign outcomes for asset A
        sign_b : pd.Series
            Sign outcomes for asset B
        mask : pd.Series
            Boolean mask for the regime condition
            
        Returns:
        --------
        float : Conditional expectation value
        """
        # Handle missing data as specified in requirement 7.4:
        # "set ⟨ab⟩xy = 0 if no valid observations exist for that regime"
        valid_observations = mask.sum()
        
        if valid_observations == 0:
            return 0.0
        
        # Calculate conditional expectation
        numerator = (sign_a[mask] * sign_b[mask]).sum()
        denominator = valid_observations
        
        return numerator / denominator
    
    def compute_s1_value(self, expectations: Dict[str, float]) -> float:
        """
        Implement S1 formula: S1 = ⟨ab⟩00 + ⟨ab⟩01 + ⟨ab⟩10 - ⟨ab⟩11
        
        This implements the exact S1 formula from requirement 2.2.
        
        Parameters:
        -----------
        expectations : Dict[str, float]
            Conditional expectations for all four regimes
            
        Returns:
        --------
        float : S1 Bell inequality value
        """
        if not all(key in expectations for key in ['ab_00', 'ab_01', 'ab_10', 'ab_11']):
            raise ValueError("Missing conditional expectations for S1 calculation")
        
        # S1 = ⟨ab⟩00 + ⟨ab⟩01 + ⟨ab⟩10 - ⟨ab⟩11
        s1 = (expectations['ab_00'] + 
              expectations['ab_01'] + 
              expectations['ab_10'] - 
              expectations['ab_11'])
        
        return s1
    
    def detect_violations(self, s1_values: List[float]) -> Dict:
        """
        Detect Bell inequality violations using |S1| > 2 criterion.
        
        This implements requirement 7.5: "IF |S1| > 2 THEN the system SHALL count 
        this as a Bell inequality violation"
        
        Parameters:
        -----------
        s1_values : List[float]
            List of S1 values to check for violations
            
        Returns:
        --------
        Dict : Violation detection results
        """
        if not s1_values:
            return {
                'total_values': 0,
                'violations': 0,
                'violation_rate': 0.0,
                'violating_values': [],
                'max_violation': 0.0
            }
        
        # Classical physics bound: |S1| ≤ 2
        violations = [s1 for s1 in s1_values if abs(s1) > 2]
        violation_count = len(violations)
        total_count = len(s1_values)
        violation_rate = (violation_count / total_count) * 100 if total_count > 0 else 0.0
        
        max_violation = max([abs(s1) for s1 in s1_values]) if s1_values else 0.0
        
        return {
            'total_values': total_count,
            'violations': violation_count,
            'violation_rate': violation_rate,
            'violating_values': violations,
            'max_violation': max_violation,
            'classical_bound': 2.0,
            'quantum_bound': 2 * np.sqrt(2)  # ≈ 2.83
        }
    
    def analyze_asset_pair(self, returns: pd.DataFrame, asset_a: str, asset_b: str,
                          window_size: Optional[int] = None) -> Dict:
        """
        Perform complete S1 analysis for a single asset pair.
        
        This method combines all the mathematical components to perform a complete
        S1 Bell inequality analysis for two assets.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Daily returns data
        asset_a : str
            First asset symbol
        asset_b : str
            Second asset symbol
        window_size : int, optional
            Override default window size for this analysis
            
        Returns:
        --------
        Dict : Complete analysis results for the asset pair
        """
        if window_size is None:
            window_size = self.window_size
        
        if len(returns) < window_size:
            raise ValueError(f"Insufficient data: {len(returns)} < {window_size}")
        
        if self.verbose:
            print(f"🔍 Analyzing pair: {asset_a} - {asset_b}")
        
        # Storage for rolling window results
        s1_time_series = []
        expectations_time_series = []
        timestamps = []
        
        # Rolling window analysis
        for T in range(window_size, len(returns)):
            window_returns = returns.iloc[T - window_size:T]
            
            # Calculate thresholds for this window
            if self.threshold_method == 'absolute':
                # Fixed absolute return threshold (Zarifian et al. 2025 approach)
                # Same threshold for all assets in the pair
                thresholds = pd.Series(
                    [self.threshold_value] * len(window_returns.columns),
                    index=window_returns.columns
                )
            elif self.threshold_method == 'quantile':
                # Quantile-based threshold (alternative approach)
                thresholds = window_returns.abs().quantile(self.threshold_quantile)
            else:
                raise ValueError(f"Unknown threshold method: {self.threshold_method}")
            
            # Compute binary indicators
            indicators_result = self.compute_binary_indicators(window_returns, thresholds)
            indicators = indicators_result['indicators']
            
            # Calculate sign outcomes
            signs = self.calculate_sign_outcomes(window_returns)
            
            # Calculate conditional expectations
            expectations = self.calculate_conditional_expectations(
                signs, indicators, asset_a, asset_b
            )
            
            # Compute S1 value
            s1 = self.compute_s1_value(expectations)
            
            # Store results
            s1_time_series.append(s1)
            expectations_time_series.append(expectations)
            timestamps.append(returns.index[T])
        
        # Detect violations
        violation_results = self.detect_violations(s1_time_series)
        
        # Compile complete results
        results = {
            'asset_pair': (asset_a, asset_b),
            's1_time_series': s1_time_series,
            'expectations_time_series': expectations_time_series,
            'timestamps': timestamps,
            'violation_results': violation_results,
            'analysis_parameters': {
                'window_size': window_size,
                'threshold_method': self.threshold_method,
                'threshold_quantile': self.threshold_quantile,
                'total_windows': len(s1_time_series)
            }
        }
        
        if self.verbose:
            print(f"✅ Analysis complete: {violation_results['violations']}/{violation_results['total_values']} violations ({violation_results['violation_rate']:.1f}%)")
        
        return results
    
    def batch_analyze_pairs(self, returns: pd.DataFrame, 
                           asset_pairs: List[Tuple[str, str]]) -> Dict:
        """
        Perform S1 analysis for multiple asset pairs.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Daily returns data
        asset_pairs : List[Tuple[str, str]]
            List of asset pairs to analyze
            
        Returns:
        --------
        Dict : Results for all asset pairs
        """
        if self.verbose:
            print(f"🔍 Batch analyzing {len(asset_pairs)} asset pairs...")
        
        batch_results = {}
        total_violations = 0
        total_calculations = 0
        
        for asset_a, asset_b in asset_pairs:
            try:
                pair_results = self.analyze_asset_pair(returns, asset_a, asset_b)
                batch_results[(asset_a, asset_b)] = pair_results
                
                # Accumulate statistics
                total_violations += pair_results['violation_results']['violations']
                total_calculations += pair_results['violation_results']['total_values']
                
            except Exception as e:
                if self.verbose:
                    print(f"⚠️  Error analyzing {asset_a}-{asset_b}: {e}")
                continue
        
        # Calculate overall statistics
        overall_violation_rate = (total_violations / total_calculations * 100) if total_calculations > 0 else 0.0
        
        summary = {
            'total_pairs': len(asset_pairs),
            'successful_pairs': len(batch_results),
            'total_calculations': total_calculations,
            'total_violations': total_violations,
            'overall_violation_rate': overall_violation_rate
        }
        
        if self.verbose:
            print(f"✅ Batch analysis complete:")
            print(f"   Pairs analyzed: {summary['successful_pairs']}/{summary['total_pairs']}")
            print(f"   Total violations: {total_violations:,}/{total_calculations:,} ({overall_violation_rate:.2f}%)")
        
        return {
            'pair_results': batch_results,
            'summary': summary
        }
    
    def validate_implementation(self) -> bool:
        """
        Validate the mathematical implementation against known test cases.
        
        Returns:
        --------
        bool : True if all validations pass
        """
        print("🧪 Validating Enhanced S1 Calculator implementation...")
        
        try:
            # Create test data
            np.random.seed(42)  # For reproducible tests
            dates = pd.date_range('2023-01-01', periods=100, freq='D')
            test_prices = pd.DataFrame({
                'ASSET_A': 100 * (1 + np.random.normal(0, 0.02, 100)).cumprod(),
                'ASSET_B': 100 * (1 + np.random.normal(0, 0.02, 100)).cumprod()
            }, index=dates)
            
            # Test daily returns calculation
            returns = self.calculate_daily_returns(test_prices)
            assert returns.shape[0] == 99, "Returns calculation failed"
            assert not returns.isnull().any().any(), "Returns contain NaN values"
            
            # Test sign outcomes
            signs = self.calculate_sign_outcomes(returns)
            assert set(signs.values.flatten()) <= {-1, 1}, "Sign function failed"
            
            # Test binary indicators
            thresholds = returns.abs().quantile(0.75)
            indicators_result = self.compute_binary_indicators(returns, thresholds)
            indicators = indicators_result['indicators']
            assert indicators.shape[1] == 4, "Binary indicators failed"  # 2 assets × 2 regimes each
            
            # Test conditional expectations
            expectations = self.calculate_conditional_expectations(
                signs, indicators, 'ASSET_A', 'ASSET_B'
            )
            assert len(expectations) == 4, "Conditional expectations failed"
            assert all(-1 <= exp <= 1 for exp in expectations.values()), "Expectations out of bounds"
            
            # Test S1 calculation
            s1 = self.compute_s1_value(expectations)
            assert isinstance(s1, (int, float)), "S1 calculation failed"
            
            # Test violation detection
            test_s1_values = [-3, -1, 0, 1, 2.5, 3]
            violations = self.detect_violations(test_s1_values)
            expected_violations = 3  # -3, 2.5, 3
            assert violations['violations'] == expected_violations, "Violation detection failed"
            
            print("✅ All validation tests passed!")
            return True
            
        except Exception as e:
            print(f"❌ Validation failed: {e}")
            return False

# Convenience functions for S1 analysis
def quick_s1_analysis(returns: pd.DataFrame, asset_a: str, asset_b: str,
                     window_size: int = 20, threshold_value: float = 0.01) -> Dict:
    """
    Perform quick S1 analysis with Zarifian et al. (2025) settings.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Daily returns data
    asset_a : str
        First asset symbol
    asset_b : str
        Second asset symbol
    window_size : int, optional
        Rolling window size. Default: 20
    threshold_value : float, optional
        Fixed absolute return threshold. Default: 0.02
        Zarifian et al. (2025) suggest: 0.01 (sensitive), 0.02 (balanced), 0.05 (crisis)
        
    Returns:
    --------
    Dict : S1 analysis results
    """
    calculator = EnhancedS1Calculator(
        window_size=window_size,
        threshold_method='absolute',
        threshold_value=threshold_value
    )
    
    return calculator.analyze_asset_pair(returns, asset_a, asset_b)

def crisis_s1_analysis(returns: pd.DataFrame, asset_a: str, asset_b: str,
                      window_size: int = 15, threshold_value: float = 0.02) -> Dict:
    """
    Perform S1 analysis optimized for crisis detection.
    
    Following Zarifian et al. (2025): "higher threshold like 0.05 isolates 
    significant market disruptions and provides the clearest visualization 
    for example in the case of the COVID-19 crisis."
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Daily returns data
    asset_a : str
        First asset symbol
    asset_b : str
        Second asset symbol
    window_size : int, optional
        Rolling window size for crisis detection. Default: 15
    threshold_value : float, optional
        Higher threshold for crisis detection. Default: 0.02
        
    Returns:
    --------
    Dict : S1 analysis results optimized for crisis detection
    """
    calculator = EnhancedS1Calculator(
        window_size=window_size,
        threshold_method='absolute',
        threshold_value=threshold_value
    )
    
    return calculator.analyze_asset_pair(returns, asset_a, asset_b)

def sensitive_s1_analysis(returns: pd.DataFrame, asset_a: str, asset_b: str,
                         window_size: int = 20, threshold_value: float = 0.005) -> Dict:
    """
    Perform S1 analysis with high sensitivity to market fluctuations.
    
    Following Zarifian et al. (2025): "smaller thresholds (e.g., ri = rj = 0.01) 
    offer broader but less precise coverage of the distributions."
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Daily returns data
    asset_a : str
        First asset symbol
    asset_b : str
        Second asset symbol
    window_size : int, optional
        Rolling window size. Default: 20
    threshold_value : float, optional
        Low threshold for high sensitivity. Default: 0.005
        
    Returns:
    --------
    Dict : S1 analysis results with high sensitivity
    """
    calculator = EnhancedS1Calculator(
        window_size=window_size,
        threshold_method='absolute',
        threshold_value=threshold_value
    )
    
    return calculator.analyze_asset_pair(returns, asset_a, asset_b)

if __name__ == "__main__":
    # Run validation tests
    calculator = EnhancedS1Calculator()
    calculator.validate_implementation()