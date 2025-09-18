#!/usr/bin/env python3
"""
Test S1 implementations with identical parameters to verify mathematical equivalence.
"""

import sys
import os
sys.path.append('src')
sys.path.append('S')

import numpy as np
import pandas as pd
from enhanced_s1_calculator import EnhancedS1Calculator

# Import colleague's function
def colleague_compute_s1_sliding_pair(x, y, window_size=20, q=0.75):
    """Colleague's S1 computation with matching parameters."""
    n = x.shape[0]
    m = n - window_size
    if m <= 0:
        return np.array([])

    shape = (m, window_size)
    stride = x.strides[0]
    x_win = np.lib.stride_tricks.as_strided(x, shape=shape, strides=(stride, stride)).copy()
    y_win = np.lib.stride_tricks.as_strided(y, shape=shape, strides=(stride, stride)).copy()

    a_sgn, b_sgn = np.sign(x_win), np.sign(y_win)
    abs_x, abs_y = np.abs(x_win), np.abs(y_win)

    thr_x = np.quantile(abs_x, q, axis=1)
    thr_y = np.quantile(abs_y, q, axis=1)

    mask_x0 = abs_x >= thr_x[:, None]
    mask_y0 = abs_y >= thr_y[:, None]

    def E(mask):
        term = (a_sgn * b_sgn) * mask
        s = term.sum(axis=1)
        cnt = mask.sum(axis=1)
        e = np.zeros_like(s, dtype=float)
        nz = cnt > 0
        e[nz] = s[nz] / cnt[nz]
        return e

    return E(mask_x0 & ~mask_y0) + E(mask_x0 & mask_y0) + \
           E(~mask_x0 & mask_y0) - E(~mask_x0 & ~mask_y0)

def test_with_matching_parameters():
    """Test both implementations with identical parameters."""
    print("🧪 Testing S1 implementations with IDENTICAL parameters...")
    
    # Create test data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    returns_data = pd.DataFrame({
        'ASSET_A': np.random.normal(0, 0.02, 100),
        'ASSET_B': np.random.normal(0, 0.02, 100)
    }, index=dates)
    
    # MATCHING PARAMETERS
    window_size = 20
    threshold_quantile = 0.75  # Same for both!
    
    print(f"   Window size: {window_size}")
    print(f"   Threshold quantile: {threshold_quantile}")
    
    # Our implementation with matching parameters
    our_calc = EnhancedS1Calculator(
        window_size=window_size,
        threshold_method='quantile',  # Use quantile method to match
        threshold_quantile=threshold_quantile
    )
    
    our_result = our_calc.analyze_asset_pair(returns_data, 'ASSET_A', 'ASSET_B')
    our_s1_values = our_result['s1_time_series']
    
    # Colleague's implementation with matching parameters
    x = returns_data['ASSET_A'].values
    y = returns_data['ASSET_B'].values
    colleague_s1_values = colleague_compute_s1_sliding_pair(x, y, window_size, threshold_quantile)
    
    print(f"\n📊 Results with MATCHING parameters:")
    print(f"   Our S1 calculations: {len(our_s1_values)}")
    print(f"   Colleague S1 calculations: {len(colleague_s1_values)}")
    
    if len(our_s1_values) == len(colleague_s1_values) and len(our_s1_values) > 0:
        # Compare S1 values directly
        our_array = np.array(our_s1_values)
        colleague_array = np.array(colleague_s1_values)
        
        differences = np.abs(our_array - colleague_array)
        max_diff = np.max(differences)
        mean_diff = np.mean(differences)
        correlation = np.corrcoef(our_array, colleague_array)[0, 1]
        
        print(f"\n🔍 Mathematical Comparison:")
        print(f"   Max S1 difference: {max_diff:.6f}")
        print(f"   Mean S1 difference: {mean_diff:.6f}")
        print(f"   Correlation: {correlation:.6f}")
        
        # Violation rates
        our_violations = np.sum(np.abs(our_array) > 2) / len(our_array) * 100
        colleague_violations = np.sum(np.abs(colleague_array) > 2) / len(colleague_array) * 100
        
        print(f"\n📊 Violation Rates:")
        print(f"   Our violation rate: {our_violations:.1f}%")
        print(f"   Colleague violation rate: {colleague_violations:.1f}%")
        print(f"   Difference: {abs(our_violations - colleague_violations):.1f}%")
        
        # Interpretation
        if max_diff < 1e-6:
            print(f"\n✅ MATHEMATICALLY EQUIVALENT!")
            print(f"   Both implementations produce nearly identical S1 values")
        elif max_diff < 1e-3:
            print(f"\n⚠️ MINOR DIFFERENCES")
            print(f"   Likely due to numerical precision or implementation details")
        else:
            print(f"\n❌ SIGNIFICANT DIFFERENCES")
            print(f"   Different mathematical approaches or parameter handling")
        
        if correlation > 0.95:
            print(f"   🔗 Excellent correlation - methods are highly consistent")
        elif correlation > 0.8:
            print(f"   🔗 Good correlation - methods are reasonably consistent")
        else:
            print(f"   🔗 Low correlation - methods may have fundamental differences")
    
    else:
        print(f"❌ Different number of calculations - cannot compare directly")

if __name__ == "__main__":
    test_with_matching_parameters()