"""
Cross-Sector Transmission Detection System

This module implements transmission mechanism detection between agricultural companies
and their operational dependencies across Energy, Transportation, and Chemicals sectors.
Focuses on fast transmission detection within 0-3 month windows.

Author: Agricultural Cross-Sector Analysis System
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import yfinance as yf
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')


@dataclass
class TransmissionMechanism:
    """Represents a transmission mechanism between sectors"""
    source_sector: str
    target_sector: str
    mechanism: str
    expected_lag: int  # Days
    strength: str  # Strong, Moderate, Weak
    crisis_amplification: bool


@dataclass
class TransmissionResults:
    """Results from transmission detection analysis"""
    pair: Tuple[str, str]
    transmission_detected: bool
    transmission_lag: Optional[int]
    correlation_strength: float
    p_value: float
    mechanism: str
    speed_category: str  # Fast (0-30 days), Medium (30-60 days), Slow (60-90 days)
    crisis_amplification: Optional[float]


@dataclass
class TransmissionSpeed:
    """Analysis of transmission speed between asset pairs"""
    pair: Tuple[str, str]
    optimal_lag: int
    max_correlation: float
    speed_category: str
    lag_profile: Dict[int, float]  # lag -> correlation
    significance_threshold: float


class CrossSectorTransmissionDetector:
    """
    Detects and analyzes transmission mechanisms between agricultural and cross-sector companies.
    
    Focuses on three main transmission channels:
    1. Energy → Agriculture (natural gas → fertilizer costs)
    2. Transportation → Agriculture (rail/shipping → logistics)
    3. Chemicals → Agriculture (input costs → pesticide/fertilizer prices)
    """
    
    def __init__(self, transmission_window: int = 90, significance_level: float = 0.05):
        """
        Initialize the transmission detector.
        
        Args:
            transmission_window: Maximum lag to test (0-3 months = 90 days)
            significance_level: Statistical significance threshold
        """
        self.transmission_window = transmission_window
        self.significance_level = significance_level
        
        # Define transmission mechanisms
        self.transmission_mechanisms = {
            'energy_agriculture': TransmissionMechanism(
                source_sector='Energy',
                target_sector='Agriculture',
                mechanism='Natural gas prices → fertilizer costs → crop production costs',
                expected_lag=30,  # ~1 month
                strength='Strong',
                crisis_amplification=True
            ),
            'transport_agriculture': TransmissionMechanism(
                source_sector='Transportation',
                target_sector='Agriculture',
                mechanism='Rail/shipping bottlenecks → commodity logistics → price volatility',
                expected_lag=15,  # ~2 weeks
                strength='Strong',
                crisis_amplification=True
            ),
            'chemicals_agriculture': TransmissionMechanism(
                source_sector='Chemicals',
                target_sector='Agriculture',
                mechanism='Chemical input costs → pesticide/fertilizer prices → farming costs',
                expected_lag=45,  # ~1.5 months
                strength='Moderate',
                crisis_amplification=True
            ),
            'finance_agriculture': TransmissionMechanism(
                source_sector='Finance',
                target_sector='Agriculture',
                mechanism='Credit availability → farming operations → commodity prices',
                expected_lag=60,  # ~2 months
                strength='Moderate',
                crisis_amplification=False
            )
        }
        
        # Define sector asset mappings
        self.sector_assets = {
            'energy': ['XOM', 'CVX', 'COP', 'UNG', 'NGAS'],  # Energy companies + natural gas
            'transportation': ['UNP', 'CSX', 'NSC', 'FDX', 'UPS'],  # Rail and shipping
            'chemicals': ['DOW', 'DD', 'LYB', 'PPG', 'SHW'],  # Chemical companies
            'finance': ['JPM', 'BAC', 'GS', 'MS', 'C'],  # Major banks
            'agriculture': ['ADM', 'BG', 'CF', 'MOS', 'NTR', 'DE', 'CAT', 'AGCO']  # Ag companies
        }
    
    def detect_energy_transmission(self, energy_assets: List[str], ag_assets: List[str]) -> List[TransmissionResults]:
        """
        Detect transmission from energy sector to agriculture.
        
        Focus: Natural gas prices → fertilizer costs → crop production costs
        
        Args:
            energy_assets: List of energy sector tickers
            ag_assets: List of agricultural sector tickers
            
        Returns:
            List of transmission results for each pair
        """
        print("🔋 Detecting Energy → Agriculture transmission mechanisms...")
        
        results = []
        mechanism = self.transmission_mechanisms['energy_agriculture']
        
        for energy_asset in energy_assets:
            for ag_asset in ag_assets:
                try:
                    result = self._analyze_transmission_pair(
                        source_asset=energy_asset,
                        target_asset=ag_asset,
                        mechanism=mechanism
                    )
                    results.append(result)
                    
                    if result.transmission_detected:
                        print(f"  ✅ {energy_asset} → {ag_asset}: {result.speed_category} transmission "
                              f"(lag: {result.transmission_lag} days, r={result.correlation_strength:.3f})")
                    
                except Exception as e:
                    print(f"  ❌ Error analyzing {energy_asset} → {ag_asset}: {str(e)}")
                    continue
        
        return results
    
    def detect_transport_transmission(self, transport_assets: List[str], ag_assets: List[str]) -> List[TransmissionResults]:
        """
        Detect transmission from transportation sector to agriculture.
        
        Focus: Rail/shipping bottlenecks → commodity logistics → price volatility
        
        Args:
            transport_assets: List of transportation sector tickers
            ag_assets: List of agricultural sector tickers
            
        Returns:
            List of transmission results for each pair
        """
        print("🚛 Detecting Transportation → Agriculture transmission mechanisms...")
        
        results = []
        mechanism = self.transmission_mechanisms['transport_agriculture']
        
        for transport_asset in transport_assets:
            for ag_asset in ag_assets:
                try:
                    result = self._analyze_transmission_pair(
                        source_asset=transport_asset,
                        target_asset=ag_asset,
                        mechanism=mechanism
                    )
                    results.append(result)
                    
                    if result.transmission_detected:
                        print(f"  ✅ {transport_asset} → {ag_asset}: {result.speed_category} transmission "
                              f"(lag: {result.transmission_lag} days, r={result.correlation_strength:.3f})")
                    
                except Exception as e:
                    print(f"  ❌ Error analyzing {transport_asset} → {ag_asset}: {str(e)}")
                    continue
        
        return results
    
    def detect_chemical_transmission(self, chemical_assets: List[str], ag_assets: List[str]) -> List[TransmissionResults]:
        """
        Detect transmission from chemicals sector to agriculture.
        
        Focus: Chemical input costs → pesticide/fertilizer prices → farming costs
        
        Args:
            chemical_assets: List of chemical sector tickers
            ag_assets: List of agricultural sector tickers
            
        Returns:
            List of transmission results for each pair
        """
        print("🧪 Detecting Chemicals → Agriculture transmission mechanisms...")
        
        results = []
        mechanism = self.transmission_mechanisms['chemicals_agriculture']
        
        for chemical_asset in chemical_assets:
            for ag_asset in ag_assets:
                try:
                    result = self._analyze_transmission_pair(
                        source_asset=chemical_asset,
                        target_asset=ag_asset,
                        mechanism=mechanism
                    )
                    results.append(result)
                    
                    if result.transmission_detected:
                        print(f"  ✅ {chemical_asset} → {ag_asset}: {result.speed_category} transmission "
                              f"(lag: {result.transmission_lag} days, r={result.correlation_strength:.3f})")
                    
                except Exception as e:
                    print(f"  ❌ Error analyzing {chemical_asset} → {ag_asset}: {str(e)}")
                    continue
        
        return results
    
    def analyze_transmission_speed(self, pair: Tuple[str, str]) -> TransmissionSpeed:
        """
        Analyze transmission speed between a specific asset pair.
        
        Tests correlations at different lags to find optimal transmission speed.
        
        Args:
            pair: Tuple of (source_asset, target_asset)
            
        Returns:
            TransmissionSpeed analysis results
        """
        source_asset, target_asset = pair
        
        try:
            # Download data for both assets
            source_data = self._download_asset_data(source_asset)
            target_data = self._download_asset_data(target_asset)
            
            if source_data is None or target_data is None:
                raise ValueError(f"Could not download data for {pair}")
            
            # Calculate returns
            source_returns = source_data['Close'].pct_change().dropna()
            target_returns = target_data['Close'].pct_change().dropna()
            
            # Align data
            common_dates = source_returns.index.intersection(target_returns.index)
            source_returns = source_returns.loc[common_dates]
            target_returns = target_returns.loc[common_dates]
            
            if len(common_dates) < 100:
                raise ValueError(f"Insufficient overlapping data for {pair}")
            
            # Test correlations at different lags
            lag_profile = {}
            max_correlation = 0
            optimal_lag = 0
            
            for lag in range(0, min(self.transmission_window + 1, len(source_returns) - 10)):
                try:
                    # Lag the source returns
                    lagged_source = source_returns.shift(lag)
                    
                    # Align data after lagging
                    valid_idx = ~(lagged_source.isna() | target_returns.isna())
                    
                    if valid_idx.sum() < 30:  # Need minimum observations
                        continue
                    
                    # Calculate correlation
                    correlation, p_value = pearsonr(
                        lagged_source[valid_idx],
                        target_returns[valid_idx]
                    )
                    
                    lag_profile[lag] = correlation
                    
                    # Track maximum absolute correlation
                    if abs(correlation) > abs(max_correlation):
                        max_correlation = correlation
                        optimal_lag = lag
                        
                except Exception:
                    continue
            
            # Determine speed category
            if optimal_lag <= 30:
                speed_category = "Fast"
            elif optimal_lag <= 60:
                speed_category = "Medium"
            else:
                speed_category = "Slow"
            
            return TransmissionSpeed(
                pair=pair,
                optimal_lag=optimal_lag,
                max_correlation=max_correlation,
                speed_category=speed_category,
                lag_profile=lag_profile,
                significance_threshold=self.significance_level
            )
            
        except Exception as e:
            print(f"Error analyzing transmission speed for {pair}: {str(e)}")
            return TransmissionSpeed(
                pair=pair,
                optimal_lag=0,
                max_correlation=0.0,
                speed_category="Unknown",
                lag_profile={},
                significance_threshold=self.significance_level
            )
    
    def _analyze_transmission_pair(self, source_asset: str, target_asset: str, 
                                 mechanism: TransmissionMechanism) -> TransmissionResults:
        """
        Analyze transmission between a specific source-target pair.
        
        Args:
            source_asset: Source sector asset ticker
            target_asset: Target agricultural asset ticker
            mechanism: Transmission mechanism definition
            
        Returns:
            TransmissionResults for the pair
        """
        try:
            # Download data
            source_data = self._download_asset_data(source_asset)
            target_data = self._download_asset_data(target_asset)
            
            if source_data is None or target_data is None:
                raise ValueError(f"Could not download data for {source_asset} or {target_asset}")
            
            # Calculate returns
            source_returns = source_data['Close'].pct_change().dropna()
            target_returns = target_data['Close'].pct_change().dropna()
            
            # Align data
            common_dates = source_returns.index.intersection(target_returns.index)
            source_returns = source_returns.loc[common_dates]
            target_returns = target_returns.loc[common_dates]
            
            if len(common_dates) < 100:
                raise ValueError(f"Insufficient data for {source_asset}-{target_asset}")
            
            # Test transmission at expected lag
            expected_lag = mechanism.expected_lag
            lagged_source = source_returns.shift(expected_lag)
            
            # Remove NaN values
            valid_idx = ~(lagged_source.isna() | target_returns.isna())
            
            if valid_idx.sum() < 30:
                raise ValueError(f"Insufficient valid observations after lagging")
            
            # Calculate correlation
            correlation, p_value = pearsonr(
                lagged_source[valid_idx],
                target_returns[valid_idx]
            )
            
            # Determine if transmission is detected
            transmission_detected = bool(
                abs(correlation) > 0.1 and  # Minimum correlation threshold
                p_value < self.significance_level
            )
            
            # Determine speed category based on expected lag
            if expected_lag <= 30:
                speed_category = "Fast"
            elif expected_lag <= 60:
                speed_category = "Medium"
            else:
                speed_category = "Slow"
            
            return TransmissionResults(
                pair=(source_asset, target_asset),
                transmission_detected=transmission_detected,
                transmission_lag=expected_lag if transmission_detected else None,
                correlation_strength=correlation,
                p_value=p_value,
                mechanism=mechanism.mechanism,
                speed_category=speed_category,
                crisis_amplification=None  # Will be calculated separately during crisis analysis
            )
            
        except Exception as e:
            return TransmissionResults(
                pair=(source_asset, target_asset),
                transmission_detected=False,
                transmission_lag=None,
                correlation_strength=0.0,
                p_value=1.0,
                mechanism=mechanism.mechanism,
                speed_category="Unknown",
                crisis_amplification=None
            )
    
    def _download_asset_data(self, ticker: str, period: str = "2y") -> Optional[pd.DataFrame]:
        """
        Download asset data using yfinance.
        
        Args:
            ticker: Asset ticker symbol
            period: Data period (default: 2 years)
            
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        try:
            asset = yf.Ticker(ticker)
            data = asset.history(period=period)
            
            if data.empty:
                return None
                
            return data
            
        except Exception:
            return None
    
    def detect_all_transmissions(self) -> Dict[str, List[TransmissionResults]]:
        """
        Detect all transmission mechanisms across all sectors.
        
        Returns:
            Dictionary mapping transmission type to results
        """
        print("🔍 Detecting all cross-sector transmission mechanisms...")
        
        results = {}
        
        # Energy → Agriculture
        results['energy_agriculture'] = self.detect_energy_transmission(
            self.sector_assets['energy'],
            self.sector_assets['agriculture']
        )
        
        # Transportation → Agriculture
        results['transport_agriculture'] = self.detect_transport_transmission(
            self.sector_assets['transportation'],
            self.sector_assets['agriculture']
        )
        
        # Chemicals → Agriculture
        results['chemicals_agriculture'] = self.detect_chemical_transmission(
            self.sector_assets['chemicals'],
            self.sector_assets['agriculture']
        )
        
        return results
    
    def summarize_transmission_results(self, results: Dict[str, List[TransmissionResults]]) -> pd.DataFrame:
        """
        Create a summary DataFrame of all transmission results.
        
        Args:
            results: Dictionary of transmission results by type
            
        Returns:
            Summary DataFrame
        """
        summary_data = []
        
        for transmission_type, transmission_results in results.items():
            for result in transmission_results:
                summary_data.append({
                    'transmission_type': transmission_type,
                    'source_asset': result.pair[0],
                    'target_asset': result.pair[1],
                    'transmission_detected': result.transmission_detected,
                    'transmission_lag': result.transmission_lag,
                    'correlation_strength': result.correlation_strength,
                    'p_value': result.p_value,
                    'speed_category': result.speed_category,
                    'mechanism': result.mechanism
                })
        
        return pd.DataFrame(summary_data)


def main():
    """Example usage of the Cross-Sector Transmission Detection System"""
    print("🌾 Cross-Sector Transmission Detection System")
    print("=" * 60)
    
    # Initialize detector
    detector = CrossSectorTransmissionDetector(transmission_window=90)
    
    # Detect all transmissions
    results = detector.detect_all_transmissions()
    
    # Create summary
    summary_df = detector.summarize_transmission_results(results)
    
    # Display results
    print("\n📊 Transmission Detection Summary:")
    print("-" * 40)
    
    detected_transmissions = summary_df[summary_df['transmission_detected'] == True]
    
    if not detected_transmissions.empty:
        print(f"✅ Found {len(detected_transmissions)} significant transmission mechanisms:")
        
        for _, row in detected_transmissions.iterrows():
            print(f"  • {row['source_asset']} → {row['target_asset']}: "
                  f"{row['speed_category']} ({row['transmission_lag']} days, "
                  f"r={row['correlation_strength']:.3f}, p={row['p_value']:.4f})")
    else:
        print("❌ No significant transmission mechanisms detected")
    
    # Speed analysis example
    print("\n🚀 Transmission Speed Analysis Example:")
    print("-" * 40)
    
    example_pair = ('XOM', 'CF')  # Energy → Fertilizer
    speed_analysis = detector.analyze_transmission_speed(example_pair)
    
    print(f"Pair: {example_pair[0]} → {example_pair[1]}")
    print(f"Optimal lag: {speed_analysis.optimal_lag} days")
    print(f"Max correlation: {speed_analysis.max_correlation:.3f}")
    print(f"Speed category: {speed_analysis.speed_category}")


if __name__ == "__main__":
    main()