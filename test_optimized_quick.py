#!/usr/bin/env python3
"""
Quick test of the optimized S1 calculator
"""

import numpy as np
import pandas as pd
import time
from src.optimized_s1_calculator import OptimizedS1Calculator, create_normal_period_calculator

def main():
    print("Testing Optimized S1 Calculator...")
    
    # Create synthetic data
    np.random.seed(42)
    n_days = 200
    dates = pd.date_range('2023-01-01', periods=n_days, freq='D')
    
    # Create correlated returns
    corn_returns = np.random.normal(0, 0.02, n_days)
    adm_returns = 0.3 * corn_returns + np.random.normal(0, 0.015, n_days)
    
    data = pd.DataFrame({
        'CORN': corn_returns,
        'ADM': adm_returns
    }, index=dates)
    
    print(f"Created test data: {data.shape}")
    
    # Test the calculator
    print("\\nTesting optimized method...")
    start_time = time.time()
    
    calc = OptimizedS1Calculator(method='sliding_window', threshold_quantile=0.95)
    result = calc.analyze_asset_pair(data, 'CORN', 'ADM')
    
    elapsed = time.time() - start_time
    
    print(f"Analysis completed in {elapsed:.3f}s")
    print(f"S1 calculations: {len(result.s1_time_series)}")
    print(f"Violation rate: {result.violation_results['violation_rate']:.1f}%")
    print(f"Max |S1|: {result.violation_results['max_violation']:.3f}")
    
    # Test factory function
    print("\\nTesting factory function...")
    normal_calc = create_normal_period_calculator()
    result2 = normal_calc.analyze_asset_pair(data, 'CORN', 'ADM')
    print(f"Factory function works: {len(result2.s1_time_series)} S1 values")
    
    print("\\nAll tests passed!")

if __name__ == "__main__":
    main()