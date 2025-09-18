#!/usr/bin/env python3
"""
Fast Crisis Analysis Demo

Quick demonstration of the optimized crisis analyzer with minimal output
and maximum performance for MacBook usage.
"""

import sys
import os
import pandas as pd
import numpy as np
import time

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from optimized_crisis_analyzer import OptimizedCrisisAnalyzer, fast_crisis_analysis


def create_fast_demo_data():
    """Create minimal demo data for fast testing."""
    np.random.seed(42)
    
    # Smaller dataset for speed
    dates = pd.date_range('2020-01-01', '2020-06-30', freq='D')  # 6 months only
    
    # Key assets only
    assets = {
        'ADM': 0.020,   # Agricultural
        'CF': 0.025,    # Agricultural  
        'XOM': 0.025,   # Tier 1 Energy
        'JPM': 0.022,   # Tier 2 Finance
        'NEE': 0.018,   # Tier 3 Utilities
    }
    
    data = pd.DataFrame(index=dates)
    
    for asset, vol in assets.items():
        returns = np.random.normal(0, vol, len(dates))
        
        # Add COVID crisis effect (March-May 2020)
        covid_mask = (dates >= '2020-03-01') & (dates <= '2020-05-31')
        returns[covid_mask] *= 2.5  # Crisis amplification
        
        data[asset] = returns
    
    return data


def main():
    """Fast demo with minimal output."""
    print("🚀 Fast Crisis Analysis Demo")
    print("=" * 40)
    
    # Create demo data
    print("📊 Creating demo data...")
    data = create_fast_demo_data()
    print(f"   Data: {len(data)} days, {len(data.columns)} assets")
    
    # Test optimized analyzer
    print("\n⚡ Running optimized analysis...")
    start_time = time.time()
    
    # Use the fast convenience function
    results = fast_crisis_analysis(
        data, 
        crisis_period="covid19_pandemic",
        use_gpu=True,  # Will fallback to CPU if no GPU
        verbose=False  # Minimal output
    )
    
    end_time = time.time()
    
    # Print results
    print(f"✅ Analysis complete in {end_time - start_time:.2f} seconds")
    print(f"\n📈 Results:")
    print(f"   Crisis: {results.crisis_period.name}")
    print(f"   Tiers analyzed: {len(results.tier_results)}")
    
    for tier, result in results.tier_results.items():
        violation_rate = result['summary']['overall_violation_rate']
        pairs_count = result['summary']['pairs_analyzed']
        print(f"   Tier {tier}: {violation_rate:.1f}% violation rate ({pairs_count} pairs)")
    
    overall_amp = results.crisis_amplification.get('overall', 0)
    print(f"   Overall amplification: {overall_amp:.2f}x")
    
    # Performance comparison
    print(f"\n🔧 Performance Info:")
    analyzer = OptimizedCrisisAnalyzer(verbose=False)
    print(f"   GPU available: {'✅' if analyzer.use_gpu else '❌'}")
    print(f"   Multiprocessing: {'✅' if analyzer.use_multiprocessing else '❌'}")
    print(f"   Workers: {analyzer.max_workers}")
    
    print(f"\n🎉 Demo complete! Ready for larger datasets.")


if __name__ == "__main__":
    main()