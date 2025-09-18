#!/usr/bin/env python3
"""
DIRECT IMPLEMENTATION COMPARISON
===============================

This script directly compares the two Bell inequality implementations
on identical datasets to validate mathematical accuracy.

Authors: Agricultural Cross-Sector Analysis Team
Date: September 2025
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
import time

# Import our implementation
from enhanced_s1_calculator import EnhancedS1Calculator

# Colleague's core S1 function (extracted to avoid file path issues)
def colleague_compute_s1_sliding_pair(x, y, window_size=20, q=0.95):
    """
    Colleague's S1 computation using sliding window approach.
    Extracted from CHSH_Violations.py to avoid dependency issues.
    """
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

class ColleagueCalculator:
    """Wrapper for colleague's implementation."""
    
    def __init__(self, window_size=20, threshold_quantile=0.95):
        self.window_size = window_size
        self.threshold_quantile = threshold_quantile
    
    def analyze_asset_pair(self, returns_data, asset_a, asset_b):
        """Analyze using colleague's method."""
        pair_data = returns_data[[asset_a, asset_b]].dropna()
        x = pair_data[asset_a].values
        y = pair_data[asset_b].values
        
        s1_values = colleague_compute_s1_sliding_pair(x, y, self.window_size, self.threshold_quantile)
        
        if len(s1_values) == 0:
            return {'s1_time_series': [], 'violation_results': {'violation_rate': 0, 'max_violation': 0}}
        
        violations = np.abs(s1_values) > 2.0
        violation_rate = (np.sum(violations) / len(s1_values)) * 100
        max_violation = np.max(np.abs(s1_values))
        
        return {
            's1_time_series': s1_values.tolist(),
            'violation_results': {
                'violation_rate': violation_rate,
                'max_violation': max_violation
            }
        }

def run_comparison():
    """Run direct comparison between implementations."""
    print("🔬 DIRECT IMPLEMENTATION COMPARISON")
    print("=" * 50)
    
    # Create test datasets
    np.random.seed(42)
    
    test_scenarios = {
        'uncorrelated': pd.DataFrame({
            'A': np.random.normal(0, 0.02, 150),
            'B': np.random.normal(0, 0.02, 150)
        }),
        'correlated': None,  # Will create below
        'high_correlation': None  # Will create below
    }
    
    # Create correlated data
    base_factor = np.random.normal(0, 0.015, 150)
    test_scenarios['correlated'] = pd.DataFrame({
        'A': 0.6 * base_factor + 0.4 * np.random.normal(0, 0.01, 150),
        'B': 0.6 * base_factor + 0.4 * np.random.normal(0, 0.01, 150)
    })
    
    test_scenarios['high_correlation'] = pd.DataFrame({
        'A': 0.8 * base_factor + 0.2 * np.random.normal(0, 0.005, 150),
        'B': 0.8 * base_factor + 0.2 * np.random.normal(0, 0.005, 150)
    })
    
    # Initialize calculators
    our_calc = EnhancedS1Calculator(window_size=20, threshold_quantile=0.75)
    colleague_calc = ColleagueCalculator(window_size=20, threshold_quantile=0.75)  # Match our threshold
    
    results = []
    
    for scenario_name, data in test_scenarios.items():
        print(f"\n🧪 Testing {scenario_name} scenario...")
        
        # Our implementation
        start_time = time.time()
        try:
            our_result = our_calc.analyze_asset_pair(data, 'A', 'B')
            our_time = time.time() - start_time
            our_s1_values = our_result['s1_time_series']
            our_violation_rate = our_result['violation_results']['violation_rate']
            our_max_violation = our_result['violation_results']['max_violation']
        except Exception as e:
            print(f"   ❌ Our implementation failed: {str(e)}")
            continue
        
        # Colleague's implementation
        start_time = time.time()
        try:
            colleague_result = colleague_calc.analyze_asset_pair(data, 'A', 'B')
            colleague_time = time.time() - start_time
            colleague_s1_values = colleague_result['s1_time_series']
            colleague_violation_rate = colleague_result['violation_results']['violation_rate']
            colleague_max_violation = colleague_result['violation_results']['max_violation']
        except Exception as e:
            print(f"   ❌ Colleague implementation failed: {str(e)}")
            continue
        
        # Compare results
        print(f"   📊 Results Comparison:")
        print(f"      Our implementation:")
        print(f"        S1 calculations: {len(our_s1_values)}")
        print(f"        Violation rate: {our_violation_rate:.1f}%")
        print(f"        Max violation: {our_max_violation:.3f}")
        print(f"        Execution time: {our_time:.3f}s")
        
        print(f"      Colleague implementation:")
        print(f"        S1 calculations: {len(colleague_s1_values)}")
        print(f"        Violation rate: {colleague_violation_rate:.1f}%")
        print(f"        Max violation: {colleague_max_violation:.3f}")
        print(f"        Execution time: {colleague_time:.3f}s")
        
        # Calculate differences
        if len(our_s1_values) == len(colleague_s1_values) and len(our_s1_values) > 0:
            s1_diff = np.max(np.abs(np.array(our_s1_values) - np.array(colleague_s1_values)))
            print(f"      📏 Differences:")
            print(f"        Max S1 difference: {s1_diff:.6f}")
            print(f"        Violation rate diff: {abs(our_violation_rate - colleague_violation_rate):.1f}%")
            print(f"        Speed ratio: {colleague_time/our_time:.2f}x")
            
            # Determine compatibility
            if s1_diff < 1e-6:  # Very small numerical differences acceptable
                print(f"        ✅ Mathematically compatible (diff < 1e-6)")
            elif s1_diff < 1e-3:
                print(f"        ⚠️ Minor differences (diff < 1e-3)")
            else:
                print(f"        ❌ Significant differences (diff > 1e-3)")
        else:
            print(f"      ❌ Different number of calculations: {len(our_s1_values)} vs {len(colleague_s1_values)}")
        
        results.append({
            'scenario': scenario_name,
            'our_violation_rate': our_violation_rate,
            'colleague_violation_rate': colleague_violation_rate,
            'our_time': our_time,
            'colleague_time': colleague_time,
            's1_difference': s1_diff if len(our_s1_values) == len(colleague_s1_values) else float('inf')
        })
    
    # Summary
    print(f"\n" + "=" * 50)
    print("📊 COMPARISON SUMMARY")
    print("=" * 50)
    
    if results:
        avg_violation_diff = np.mean([abs(r['our_violation_rate'] - r['colleague_violation_rate']) for r in results])
        avg_speed_ratio = np.mean([r['colleague_time'] / r['our_time'] for r in results])
        max_s1_diff = max([r['s1_difference'] for r in results if r['s1_difference'] != float('inf')])
        
        print(f"Average violation rate difference: {avg_violation_diff:.1f}%")
        print(f"Average speed ratio (colleague/ours): {avg_speed_ratio:.2f}x")
        print(f"Maximum S1 value difference: {max_s1_diff:.6f}")
        
        if max_s1_diff < 1e-6:
            print(f"\n✅ IMPLEMENTATIONS ARE MATHEMATICALLY EQUIVALENT!")
            print(f"   Both produce nearly identical S1 values")
        elif max_s1_diff < 1e-3:
            print(f"\n⚠️ IMPLEMENTATIONS HAVE MINOR DIFFERENCES")
            print(f"   Differences likely due to numerical precision or threshold calculation")
        else:
            print(f"\n❌ IMPLEMENTATIONS HAVE SIGNIFICANT DIFFERENCES")
            print(f"   May use different mathematical approaches")
        
        if avg_speed_ratio < 0.8:
            print(f"   🚀 Colleague's implementation is faster ({avg_speed_ratio:.2f}x)")
        elif avg_speed_ratio > 1.2:
            print(f"   🚀 Our implementation is faster ({1/avg_speed_ratio:.2f}x)")
        else:
            print(f"   ⚖️ Both implementations have similar performance")
    
    return results

if __name__ == "__main__":
    results = run_comparison()
    
    # Save results
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv('comparison/direct_comparison_results.csv', index=False)
        print(f"\n📄 Results saved to comparison/direct_comparison_results.csv")
    
    print(f"\n🎯 NEXT STEPS:")
    print(f"   1. Review mathematical differences (if any)")
    print(f"   2. Consider adopting performance optimizations")
    print(f"   3. Integrate best practices from both implementations")
    print(f"   4. Create unified analysis framework")