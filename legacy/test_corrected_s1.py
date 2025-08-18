#!/usr/bin/env python3
"""
Quick test of corrected S1 Bell inequality implementation
"""

import sys
sys.path.append('.')

from analyze_yahoo_finance_bell import YahooFinanceBellAnalyzer

def test_corrected_s1():
    """Test the corrected S1 implementation with a small dataset"""
    
    print("🧪 TESTING CORRECTED S1 BELL INEQUALITY")
    print("=" * 45)
    print("Using Zarifian et al. (2025) formula:")
    print("S1 = E[AB|x₀,y₀] + E[AB|x₀,y₁] + E[AB|x₁,y₀] - E[AB|x₁,y₁]")
    
    # Test with just 2 assets for speed
    analyzer = YahooFinanceBellAnalyzer(
        assets=['AAPL', 'MSFT'], 
        period='3mo'  # Shorter period for faster testing
    )
    
    # Download data
    print("\n📥 Downloading test data...")
    analyzer.raw_data = analyzer._download_yahoo_data()
    
    if analyzer.raw_data is None:
        print("❌ Failed to download data")
        return
    
    # Process data
    print("\n🔄 Processing data...")
    analyzer.processed_data = analyzer._process_data_multiple_frequencies('1d')
    
    # Test S1 conditional Bell analysis
    print("\n🎯 Testing S1 conditional Bell analysis...")
    conditional_results = analyzer._perform_conditional_bell_analysis(
        window_size=10,  # Smaller window for testing
        threshold_quantile=0.5  # Median split
    )
    
    if conditional_results:
        print("\n✅ S1 TEST RESULTS:")
        for freq, freq_results in conditional_results.items():
            print(f"\n📊 {freq} frequency:")
            for pair, result in freq_results.items():
                print(f"   {pair}:")
                print(f"      S1 = {result['S1']:.4f}")
                print(f"      E[AB|0,0] = {result['E_AB_00']:.3f}")
                print(f"      E[AB|0,1] = {result['E_AB_01']:.3f}")
                print(f"      E[AB|1,0] = {result['E_AB_10']:.3f}")
                print(f"      E[AB|1,1] = {result['E_AB_11']:.3f}")
                print(f"      Violation: {result['violation']}")
                print(f"      Regime counts: {result['regime_counts']}")
    else:
        print("❌ No S1 results generated")
    
    print("\n✅ S1 test complete!")

if __name__ == "__main__":
    test_corrected_s1()