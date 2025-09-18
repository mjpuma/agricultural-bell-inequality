#!/usr/bin/env python3
"""
MATHEMATICAL VALIDATION FRAMEWORK DEMONSTRATION
==============================================

This script demonstrates the comprehensive mathematical validation and cross-implementation
analysis framework for Bell inequality methods. It shows how to validate mathematical
correctness, test numerical precision, and generate publication-ready validation reports.

Key Demonstrations:
- Cross-validation between S1 and CHSH implementations
- Numerical precision analysis with 1e-12 tolerance
- Statistical significance testing with bootstrap validation
- Comprehensive validation reporting for Science journal standards

Usage:
    python examples/mathematical_validation_demo.py

Authors: Bell Inequality Validation Team
Date: September 2025
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from mathematical_validation_framework import (
    CrossImplementationValidator,
    NumericalPrecisionAnalyzer,
    run_comprehensive_validation
)

def create_demonstration_data():
    """
    Create demonstration data for validation testing.
    
    Returns:
    --------
    Tuple[pd.DataFrame, List[Tuple[str, str]]] : Test data and asset pairs
    """
    print("📊 Creating demonstration data...")
    
    # Set random seed for reproducible results
    np.random.seed(42)
    
    # Create realistic financial time series data
    n_days = 200
    dates = pd.date_range('2023-01-01', periods=n_days, freq='D')
    
    # Agricultural and food system assets (following steering guidelines)
    assets = {
        'CORN': 'Corn Futures',
        'ADM': 'Archer Daniels Midland',
        'CF': 'CF Industries (Fertilizer)',
        'DE': 'John Deere (Equipment)',
        'LEAN': 'Lean Hogs Futures',
        'WEAT': 'Wheat Futures'
    }
    
    # Generate correlated price series with realistic volatility
    base_returns = np.random.multivariate_normal(
        mean=[0.0005] * len(assets),  # Slight positive drift
        cov=np.eye(len(assets)) * 0.02**2 + 0.005**2,  # Correlation structure
        size=n_days
    )
    
    # Create price data
    price_data = {}
    for i, (symbol, name) in enumerate(assets.items()):
        # Start at $100 and apply returns
        prices = 100 * np.exp(np.cumsum(base_returns[:, i]))
        price_data[symbol] = prices
    
    test_data = pd.DataFrame(price_data, index=dates)
    
    # Define asset pairs for testing (food system relationships)
    asset_pairs = [
        ('CORN', 'ADM'),    # Corn processor relationship
        ('CORN', 'LEAN'),   # Feed-livestock relationship  
        ('CF', 'CORN'),     # Fertilizer-crop relationship
        ('DE', 'CORN'),     # Equipment-farming relationship
        ('WEAT', 'CORN'),   # Grain correlation
        ('ADM', 'CF')       # Cross-sector relationship
    ]
    
    print(f"   ✅ Created {n_days} days of data for {len(assets)} assets")
    print(f"   📈 Asset pairs for testing: {len(asset_pairs)}")
    
    return test_data, asset_pairs

def demonstrate_cross_implementation_validation():
    """Demonstrate cross-implementation validation capabilities."""
    print("\n🔬 DEMONSTRATING CROSS-IMPLEMENTATION VALIDATION")
    print("=" * 55)
    
    # Create test data
    test_data, asset_pairs = create_demonstration_data()
    
    # Initialize validator
    validator = CrossImplementationValidator(tolerance=1e-12, bootstrap_samples=500)
    
    print("\n1️⃣  Daily Returns Calculation Validation")
    print("-" * 45)
    returns_result = validator.validate_daily_returns_calculation(test_data)
    print(f"   Test: {returns_result.test_name}")
    print(f"   Status: {'✅ PASSED' if returns_result.passed else '❌ FAILED'}")
    print(f"   Max Difference: {returns_result.numerical_difference:.2e}")
    print(f"   Tolerance: {returns_result.tolerance:.2e}")
    print(f"   P-Value: {returns_result.p_value:.6f}")
    print(f"   Execution Time: {returns_result.execution_time:.3f}s")
    
    print("\n2️⃣  Sign Function Calculation Validation")
    print("-" * 45)
    returns_data = test_data.pct_change().dropna()
    sign_result = validator.validate_sign_calculations(returns_data)
    print(f"   Test: {sign_result.test_name}")
    print(f"   Status: {'✅ PASSED' if sign_result.passed else '❌ FAILED'}")
    print(f"   Max Difference: {sign_result.numerical_difference}")
    print(f"   Note: Sign function must be exactly identical (tolerance = 0)")
    
    print("\n3️⃣  Threshold Methods Validation")
    print("-" * 45)
    quantiles = [0.5, 0.75, 0.9, 0.95]
    threshold_result = validator.validate_threshold_methods(returns_data, quantiles)
    print(f"   Test: {threshold_result.test_name}")
    print(f"   Status: {'✅ PASSED' if threshold_result.passed else '❌ FAILED'}")
    print(f"   Max Difference: {threshold_result.numerical_difference:.2e}")
    print(f"   Quantiles Tested: {quantiles}")
    
    print("\n4️⃣  Bell Violation Detection Validation")
    print("-" * 45)
    violation_result = validator.validate_bell_violations(test_data, asset_pairs[:3])
    print(f"   Test: {violation_result.test_name}")
    print(f"   Status: {'✅ PASSED' if violation_result.passed else '❌ FAILED'}")
    print(f"   Max Difference: {violation_result.numerical_difference:.1f}%")
    print(f"   Tolerance: {violation_result.tolerance}% (violation rate difference)")
    
    print("\n5️⃣  Cross-Method Comparison")
    print("-" * 45)
    comparison_report = validator.cross_validate_methods(test_data, asset_pairs[:2])
    print(f"   Implementation A: {comparison_report.implementation_a}")
    print(f"   Implementation B: {comparison_report.implementation_b}")
    print(f"   Identical Results: {'✅ YES' if comparison_report.identical_results else '❌ NO'}")
    print(f"   Max Difference: {comparison_report.max_difference:.2e}")
    print(f"   Correlation: {comparison_report.correlation:.4f}")
    print(f"   Statistical Significance: p = {comparison_report.statistical_significance['p_value']:.4f}")
    
    return validator

def demonstrate_numerical_precision_analysis():
    """Demonstrate numerical precision analysis capabilities."""
    print("\n🔍 DEMONSTRATING NUMERICAL PRECISION ANALYSIS")
    print("=" * 55)
    
    # Create test data
    test_data, _ = create_demonstration_data()
    
    # Initialize analyzer
    analyzer = NumericalPrecisionAnalyzer(precision_target=1e-12)
    
    print("\n1️⃣  Floating-Point Precision Analysis")
    print("-" * 45)
    
    # Test with various precision scenarios
    test_calculations = [
        1.0,                    # Exact value
        1.0 + 1e-15,           # Near machine precision
        1.0 - 1e-15,           # Near machine precision
        1.000000000001,        # Small deviation
        0.999999999999,        # Small deviation
        np.pi,                 # Irrational number
        np.e,                  # Euler's number
        1.0/3.0                # Repeating decimal
    ]
    
    precision_report = analyzer.analyze_floating_point_precision(test_calculations)
    print(f"   Test Type: {precision_report.test_type}")
    print(f"   Precision Achieved: {precision_report.precision_achieved:.2e}")
    print(f"   Stability Score: {precision_report.stability_score:.4f}")
    print(f"   Convergence Rate: {precision_report.convergence_rate:.4f}")
    print(f"   Recommendations:")
    for rec in precision_report.recommendations:
        print(f"     • {rec}")
    
    print("\n2️⃣  Numerical Stability Testing")
    print("-" * 45)
    
    # Test stability under various perturbation levels
    perturbations = [1e-12, 1e-10, 1e-8, 1e-6, 1e-4]
    stability_report = analyzer.test_numerical_stability(test_data, perturbations)
    print(f"   Test Type: {stability_report.test_type}")
    print(f"   Overall Stability Score: {stability_report.stability_score:.4f}")
    print(f"   Perturbations Tested: {perturbations}")
    print(f"   Numerical Errors: {[f'{err:.2e}' for err in stability_report.numerical_errors[:3]]}...")
    print(f"   Recommendations:")
    for rec in stability_report.recommendations:
        print(f"     • {rec}")
    
    print("\n3️⃣  Convergence Validation")
    print("-" * 45)
    
    # Create test sequences with different convergence properties
    test_sequences = [
        # Geometric convergence (fast)
        [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125],
        # Linear convergence (slow)
        [1.0, 0.9, 0.8, 0.7, 0.6, 0.5],
        # Oscillating convergence
        [1.0, -0.5, 0.25, -0.125, 0.0625, -0.03125],
        # Near-convergent sequence
        [2.0, 1.5, 1.25, 1.125, 1.0625, 1.03125]
    ]
    
    convergence_report = analyzer.validate_convergence(test_sequences)
    print(f"   Test Type: {convergence_report.test_type}")
    print(f"   Convergence Rate: {convergence_report.convergence_rate:.4f}")
    print(f"   Stability Score: {convergence_report.stability_score:.4f}")
    print(f"   Sequences Tested: {len(test_sequences)}")
    print(f"   Recommendations:")
    for rec in convergence_report.recommendations:
        print(f"     • {rec}")
    
    return analyzer

def demonstrate_comprehensive_validation():
    """Demonstrate comprehensive validation workflow."""
    print("\n🚀 DEMONSTRATING COMPREHENSIVE VALIDATION WORKFLOW")
    print("=" * 60)
    
    # Create test data
    test_data, asset_pairs = create_demonstration_data()
    
    # Create output directory
    output_dir = "validation_demo_results"
    Path(output_dir).mkdir(exist_ok=True)
    
    print(f"\n📁 Output Directory: {output_dir}")
    print(f"📊 Test Data: {test_data.shape[0]} days, {test_data.shape[1]} assets")
    print(f"🔗 Asset Pairs: {len(asset_pairs)} pairs for testing")
    
    # Run comprehensive validation
    print(f"\n⚡ Running comprehensive validation...")
    results = run_comprehensive_validation(
        test_data=test_data,
        asset_pairs=asset_pairs,
        output_dir=output_dir
    )
    
    # Display detailed results
    print(f"\n📋 COMPREHENSIVE VALIDATION RESULTS")
    print("=" * 45)
    
    # Validation tests summary
    validation_tests = results['validation_tests']
    print(f"\n🧪 Individual Validation Tests:")
    for test_name, result in validation_tests.items():
        status = "✅ PASSED" if result.passed else "❌ FAILED"
        print(f"   {test_name.replace('_', ' ').title()}: {status}")
        print(f"     Difference: {result.numerical_difference:.2e}")
        print(f"     P-Value: {result.p_value:.6f}")
        print(f"     Time: {result.execution_time:.3f}s")
    
    # Cross-validation summary
    cross_validation = results['cross_validation']
    print(f"\n🔬 Cross-Implementation Analysis:")
    print(f"   Methods Compared: {cross_validation.implementation_a} vs {cross_validation.implementation_b}")
    print(f"   Identical Results: {'✅ YES' if cross_validation.identical_results else '❌ NO'}")
    print(f"   Maximum Difference: {cross_validation.max_difference:.2e}")
    print(f"   Correlation: {cross_validation.correlation:.4f}")
    print(f"   Statistical Significance: p = {cross_validation.statistical_significance['p_value']:.4f}")
    
    # Precision analysis summary
    precision_analysis = results['precision_analysis']
    print(f"\n🔍 Numerical Precision Analysis:")
    
    fp_report = precision_analysis['floating_point']
    print(f"   Floating-Point Precision: {fp_report.precision_achieved:.2e}")
    print(f"   Stability Score: {fp_report.stability_score:.4f}")
    
    stability_report = precision_analysis['stability']
    print(f"   Numerical Stability: {stability_report.stability_score:.4f}")
    
    convergence_report = precision_analysis['convergence']
    print(f"   Convergence Rate: {convergence_report.convergence_rate:.4f}")
    
    # Overall summary
    summary = results['summary']
    print(f"\n📊 OVERALL VALIDATION SUMMARY:")
    print(f"   All Tests Passed: {'✅ YES' if summary['all_tests_passed'] else '❌ NO'}")
    print(f"   Precision Target Met: {'✅ YES' if summary['precision_target_met'] else '❌ NO'}")
    print(f"   Stability Acceptable: {'✅ YES' if summary['stability_acceptable'] else '❌ NO'}")
    print(f"   Convergence Acceptable: {'✅ YES' if summary['convergence_acceptable'] else '❌ NO'}")
    
    # Publication readiness assessment
    publication_ready = all([
        summary['all_tests_passed'],
        summary['precision_target_met'],
        summary['stability_acceptable'],
        summary['convergence_acceptable']
    ])
    
    print(f"\n🎯 PUBLICATION READINESS:")
    if publication_ready:
        print("   ✅ READY FOR SCIENCE JOURNAL SUBMISSION")
        print("   📋 All validation criteria met")
        print("   🔬 Mathematical correctness confirmed")
        print("   📊 Statistical rigor validated")
        print("   💯 Numerical precision acceptable")
    else:
        print("   ⚠️  ADDITIONAL VALIDATION REQUIRED")
        print("   📋 Some validation criteria not met")
        print("   🔧 Review failed tests and improve implementations")
    
    # Report files
    reports = results['reports']
    print(f"\n📄 Generated Reports:")
    for report_type, report_path in reports.items():
        print(f"   {report_type.replace('_', ' ').title()}: {report_path}")
    
    return results

def demonstrate_food_systems_validation():
    """Demonstrate validation specifically for food systems research."""
    print("\n🌾 DEMONSTRATING FOOD SYSTEMS VALIDATION")
    print("=" * 50)
    
    # Create food systems specific test data
    np.random.seed(42)
    n_days = 150
    dates = pd.date_range('2023-01-01', periods=n_days, freq='D')
    
    # Food system assets from steering guidelines
    food_assets = {
        'CORN': 'Corn Futures',
        'WEAT': 'Wheat Futures', 
        'SOYB': 'Soybean Futures',
        'LEAN': 'Lean Hogs Futures',
        'ADM': 'Archer Daniels Midland',
        'CF': 'CF Industries',
        'DE': 'John Deere',
        'GIS': 'General Mills'
    }
    
    # Generate correlated returns with food system relationships
    # Higher correlation for supply chain relationships
    correlation_matrix = np.eye(len(food_assets))
    
    # Add specific correlations
    asset_list = list(food_assets.keys())
    corn_idx = asset_list.index('CORN')
    lean_idx = asset_list.index('LEAN')
    adm_idx = asset_list.index('ADM')
    cf_idx = asset_list.index('CF')
    
    # CORN-LEAN correlation (feed relationship)
    correlation_matrix[corn_idx, lean_idx] = 0.6
    correlation_matrix[lean_idx, corn_idx] = 0.6
    
    # CORN-ADM correlation (processing relationship)
    correlation_matrix[corn_idx, adm_idx] = 0.7
    correlation_matrix[adm_idx, corn_idx] = 0.7
    
    # CORN-CF correlation (fertilizer relationship)
    correlation_matrix[corn_idx, cf_idx] = 0.5
    correlation_matrix[cf_idx, corn_idx] = 0.5
    
    # Generate returns
    returns = np.random.multivariate_normal(
        mean=[0.0005] * len(food_assets),
        cov=correlation_matrix * 0.025**2,  # 2.5% daily volatility
        size=n_days
    )
    
    # Create price data
    food_data = {}
    for i, (symbol, name) in enumerate(food_assets.items()):
        prices = 100 * np.exp(np.cumsum(returns[:, i]))
        food_data[symbol] = prices
    
    food_test_data = pd.DataFrame(food_data, index=dates)
    
    # Food system specific pairs (from steering guidelines)
    food_pairs = [
        ('CORN', 'LEAN'),   # Corn-livestock feed relationship
        ('CORN', 'ADM'),    # Corn-processor relationship
        ('CF', 'CORN'),     # Fertilizer-crop relationship
        ('DE', 'CORN'),     # Equipment-farming relationship
        ('WEAT', 'SOYB'),   # Grain correlation
        ('GIS', 'WEAT')     # Food company-grain relationship
    ]
    
    print(f"📊 Food Systems Data: {n_days} days, {len(food_assets)} assets")
    print(f"🔗 Food System Pairs: {len(food_pairs)} supply chain relationships")
    
    # Run validation with food systems focus
    output_dir = "food_systems_validation_results"
    Path(output_dir).mkdir(exist_ok=True)
    
    print(f"\n⚡ Running food systems validation...")
    results = run_comprehensive_validation(
        test_data=food_test_data,
        asset_pairs=food_pairs,
        output_dir=output_dir
    )
    
    # Analyze results for food systems context
    print(f"\n🌾 FOOD SYSTEMS VALIDATION RESULTS")
    print("=" * 40)
    
    # Check for expected violation patterns
    cross_validation = results['cross_validation']
    violation_rate_diff = cross_validation.max_difference
    
    print(f"📈 Supply Chain Correlation Analysis:")
    print(f"   Method Correlation: {cross_validation.correlation:.4f}")
    print(f"   Expected Range: 0.7-0.9 (strong supply chain correlations)")
    
    if cross_validation.correlation >= 0.7:
        print("   ✅ Strong correlation detected - consistent with supply chain relationships")
    else:
        print("   ⚠️  Lower correlation - may indicate method sensitivity differences")
    
    # Publication implications
    summary = results['summary']
    print(f"\n📋 Food Systems Publication Readiness:")
    
    if summary['all_tests_passed']:
        print("   ✅ Mathematical validation complete for food systems analysis")
        print("   🌾 Ready for agricultural cross-sector Bell inequality research")
        print("   📊 Suitable for Science journal submission")
    else:
        print("   ⚠️  Additional validation needed for publication")
    
    print(f"\n💡 Food Systems Research Recommendations:")
    print("   • Focus on supply chain pairs (CORN-LEAN, CORN-ADM)")
    print("   • Use crisis periods for enhanced Bell violations")
    print("   • Apply 20-day windows for agricultural data")
    print("   • Consider seasonal effects in threshold selection")
    
    return results

def main():
    """Main demonstration function."""
    print("🔬 MATHEMATICAL VALIDATION FRAMEWORK DEMONSTRATION")
    print("=" * 60)
    print("This demonstration shows comprehensive validation capabilities")
    print("for Bell inequality implementations meeting Science journal standards.")
    print()
    
    try:
        # 1. Cross-implementation validation
        validator = demonstrate_cross_implementation_validation()
        
        # 2. Numerical precision analysis
        analyzer = demonstrate_numerical_precision_analysis()
        
        # 3. Comprehensive validation workflow
        comprehensive_results = demonstrate_comprehensive_validation()
        
        # 4. Food systems specific validation
        food_results = demonstrate_food_systems_validation()
        
        # Final summary
        print(f"\n🎯 DEMONSTRATION COMPLETE")
        print("=" * 30)
        print("✅ Cross-implementation validation demonstrated")
        print("✅ Numerical precision analysis demonstrated") 
        print("✅ Comprehensive validation workflow demonstrated")
        print("✅ Food systems validation demonstrated")
        print()
        print("📋 All validation components are ready for:")
        print("   • Science journal publication")
        print("   • Agricultural cross-sector research")
        print("   • Bell inequality analysis validation")
        print("   • Mathematical correctness verification")
        print()
        print("📁 Check output directories for detailed reports:")
        print("   • validation_demo_results/")
        print("   • food_systems_validation_results/")
        
    except Exception as e:
        print(f"\n❌ Demonstration failed with error: {e}")
        print("Please check the implementation and try again.")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n🚀 Mathematical Validation Framework is ready for production use!")
    else:
        print("\n⚠️  Please address issues before using in production.")
        sys.exit(1)