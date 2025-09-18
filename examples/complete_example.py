#!/usr/bin/env python3
"""
BELL INEQUALITY ANALYSIS - COMPLETE EXAMPLE
===========================================
Demonstrates the full Bell inequality analysis workflow
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from bell_inequality_analyzer import BellInequalityAnalyzer, quick_bell_analysis
from cross_mandelbrot_analyzer import CrossMandelbrotAnalyzer

def run_complete_example():
    """Run a complete Bell inequality analysis example"""
    
    print("🚀 BELL INEQUALITY ANALYSIS - COMPLETE EXAMPLE")
    print("=" * 50)
    
    # Method 1: Quick analysis (recommended for beginners)
    print("\n📊 METHOD 1: QUICK ANALYSIS")
    print("-" * 30)
    
    analyzer = quick_bell_analysis(
        assets=['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'META'],
        period='6mo',
        create_plots=True
    )
    
    if analyzer:
        print("✅ Quick analysis complete!")
        print("📊 Check 'bell_analysis_results.png' for visualizations")
    
    # Method 2: Advanced analysis (for researchers)
    print("\n🔬 METHOD 2: ADVANCED ANALYSIS")
    print("-" * 30)
    
    # Initialize analyzer
    advanced_analyzer = BellInequalityAnalyzer(
        assets=['AAPL', 'MSFT', 'GOOGL', 'NVDA'],
        data_source='yahoo',
        period='1y'
    )
    
    # Load data
    if advanced_analyzer.load_data():
        # Run S1 analysis with custom parameters
        s1_results = advanced_analyzer.run_s1_analysis(
            window_size=25,  # Larger window
            threshold_quantile=0.8  # Higher threshold
        )
        
        # Create visualizations
        advanced_analyzer.create_visualizations('advanced_bell_results.png')
        
        # Generate report
        advanced_analyzer.generate_summary_report()
        
        print("✅ Advanced analysis complete!")
    
    # Method 3: Cross-Mandelbrot analysis
    print("\n🌀 METHOD 3: CROSS-MANDELBROT ANALYSIS")
    print("-" * 35)
    
    if analyzer and analyzer.returns_data is not None:
        # Convert to format for cross-Mandelbrot
        return_data = {}
        for asset in analyzer.assets:
            if asset in analyzer.returns_data.columns:
                return_data[asset] = analyzer.returns_data[asset]
        
        # Run cross-Mandelbrot analysis
        cross_analyzer = CrossMandelbrotAnalyzer()
        cross_results = cross_analyzer.analyze_cross_mandelbrot_comprehensive(return_data)
        
        print(f"✅ Cross-Mandelbrot analysis complete!")
        print(f"📊 Analyzed {len(cross_results)} asset pairs")
    
    print("\n🎉 COMPLETE EXAMPLE FINISHED!")
    print("📁 Check generated files for detailed results")

if __name__ == "__main__":
    run_complete_example()
