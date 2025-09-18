#!/usr/bin/env python3
"""
COLLEAGUE IMPLEMENTATION ADAPTER
================================

This module provides an adapter to make the colleague's CHSH implementation
compatible with our validation framework.

Authors: Agricultural Cross-Sector Analysis Team
Date: September 2025
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'colleague_implementation'))

import numpy as np
import pandas as pd
from CHSH_Violations import compute_s1_sliding_pair

class ColleagueS1Calculator:
    """Adapter class to make colleague's implementation compatible with our validation framework."""
    
    def __init__(self, window_size=20, threshold_quantile=0.95):
        """
        Initialize colleague's calculator with our interface.
        
        Args:
            window_size: Rolling window size
            threshold_quantile: Quantile for threshold calculation (colleague uses 0.95)
        """
        self.window_size = window_size
        self.threshold_quantile = threshold_quantile
    
    def calculate_daily_returns(self, price_data):
        """Calculate daily returns using pandas pct_change (same as colleague's approach)."""
        if price_data is None or price_data.empty:
            raise ValueError("Price data cannot be None or empty")
        
        returns = price_data.pct_change().dropna()
        return returns
    
    def analyze_asset_pair(self, returns_data, asset_a, asset_b):
        """
        Analyze asset pair using colleague's CHSH implementation.
        
        Args:
            returns_data: DataFrame with returns data
            asset_a: First asset name
            asset_b: Second asset name
            
        Returns:
            Dictionary with results in our expected format
        """
        if asset_a not in returns_data.columns or asset_b not in returns_data.columns:
            raise ValueError(f"Assets {asset_a} or {asset_b} not found in data")
        
        # Get clean data for the pair
        pair_data = returns_data[[asset_a, asset_b]].dropna()
        
        if len(pair_data) <= self.window_size:
            raise ValueError(f"Insufficient data: {len(pair_data)} observations, need > {self.window_size}")
        
        # Extract arrays
        x = pair_data[asset_a].values
        y = pair_data[asset_b].values
        
        # Use colleague's sliding window S1 computation
        s1_values = compute_s1_sliding_pair(x, y, window_size=self.window_size, q=self.threshold_quantile)
        
        if len(s1_values) == 0:
            raise ValueError("No S1 values computed")
        
        # Calculate violation statistics
        violations = np.abs(s1_values) > 2.0
        violation_count = np.sum(violations)
        total_calculations = len(s1_values)
        violation_rate = (violation_count / total_calculations) * 100 if total_calculations > 0 else 0
        max_violation = np.max(np.abs(s1_values)) if len(s1_values) > 0 else 0
        
        # Create timestamps for S1 values
        timestamps = pair_data.index[self.window_size:self.window_size + len(s1_values)]
        
        # Return results in our expected format
        return {
            'asset_pair': (asset_a, asset_b),
            's1_time_series': s1_values.tolist(),
            'timestamps': timestamps.tolist(),
            'violation_results': {
                'violations': violation_count,
                'violation_rate': violation_rate,
                'max_violation': max_violation,
                'classical_bound': 2.0,
                'quantum_bound': 2 * np.sqrt(2)
            },
            'analysis_parameters': {
                'window_size': self.window_size,
                'threshold_quantile': self.threshold_quantile,
                'method': 'CHSH_sliding_window'
            }
        }
    
    def batch_analyze_pairs(self, returns_data, asset_pairs):
        """
        Batch analyze multiple asset pairs.
        
        Args:
            returns_data: DataFrame with returns data
            asset_pairs: List of (asset_a, asset_b) tuples
            
        Returns:
            Dictionary with batch results
        """
        pair_results = {}
        total_violations = 0
        total_calculations = 0
        successful_pairs = 0
        
        for asset_a, asset_b in asset_pairs:
            try:
                result = self.analyze_asset_pair(returns_data, asset_a, asset_b)
                pair_results[(asset_a, asset_b)] = result
                
                total_violations += result['violation_results']['violations']
                total_calculations += len(result['s1_time_series'])
                successful_pairs += 1
                
            except Exception as e:
                print(f"Failed to analyze pair ({asset_a}, {asset_b}): {str(e)}")
                continue
        
        overall_violation_rate = (total_violations / total_calculations * 100) if total_calculations > 0 else 0
        
        return {
            'pair_results': pair_results,
            'summary': {
                'total_pairs': len(asset_pairs),
                'successful_pairs': successful_pairs,
                'total_calculations': total_calculations,
                'total_violations': total_violations,
                'overall_violation_rate': overall_violation_rate
            }
        }

# Test the adapter
if __name__ == "__main__":
    print("🧪 Testing Colleague Implementation Adapter...")
    
    # Create test data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    test_data = pd.DataFrame({
        'ASSET_A': np.random.normal(0, 0.02, 100),
        'ASSET_B': np.random.normal(0, 0.02, 100)
    }, index=dates)
    
    # Test the adapter
    try:
        calculator = ColleagueS1Calculator(window_size=20, threshold_quantile=0.95)
        result = calculator.analyze_asset_pair(test_data, 'ASSET_A', 'ASSET_B')
        
        print(f"✅ Adapter working successfully!")
        print(f"   S1 calculations: {len(result['s1_time_series'])}")
        print(f"   Violation rate: {result['violation_results']['violation_rate']:.1f}%")
        print(f"   Max violation: {result['violation_results']['max_violation']:.3f}")
        
    except Exception as e:
        print(f"❌ Adapter test failed: {str(e)}")
        sys.exit(1)