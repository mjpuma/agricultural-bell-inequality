#!/usr/bin/env python3
"""
CROSS-IMPLEMENTATION VALIDATION FRAMEWORK
========================================

This module provides a comprehensive framework for validating and comparing
different implementations of the Bell inequality analysis system.

Usage:
    python validation_framework.py --test-all
    python validation_framework.py --mathematical-only
    python validation_framework.py --performance-only

Authors: Agricultural Cross-Sector Analysis Team
Date: September 2025
"""

import sys
import os
import pandas as pd
import numpy as np
import time
import importlib.util
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Add paths for both implementations
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'colleague_implementation'))

@dataclass
class ValidationResult:
    """Container for validation test results."""
    test_name: str
    passed: bool
    our_result: Any
    colleague_result: Any
    difference: Optional[float]
    tolerance: float
    execution_time_ours: float
    execution_time_colleague: float
    notes: str

@dataclass
class ComparisonSummary:
    """Summary of all validation results."""
    total_tests: int
    passed_tests: int
    failed_tests: int
    mathematical_accuracy: bool
    performance_acceptable: bool
    overall_compatible: bool
    detailed_results: List[ValidationResult]

class CrossImplementationValidator:
    """Framework for validating multiple Bell inequality implementations."""
    
    def __init__(self, tolerance: float = 1e-10):
        """
        Initialize validator.
        
        Args:
            tolerance: Numerical tolerance for floating point comparisons
        """
        self.tolerance = tolerance
        self.test_results = []
        
        # Try to import both implementations
        self.our_implementation = self._import_our_implementation()
        self.colleague_implementation = self._import_colleague_implementation()
        
        # Create test datasets
        self.test_datasets = self._create_test_datasets()
    
    def _import_our_implementation(self):
        """Import our implementation."""
        try:
            from enhanced_s1_calculator import EnhancedS1Calculator
            from agricultural_cross_sector_analyzer import AgriculturalCrossSectorAnalyzer
            
            return {
                'calculator': EnhancedS1Calculator,
                'analyzer': AgriculturalCrossSectorAnalyzer,
                'available': True
            }
        except ImportError as e:
            print(f"⚠️ Could not import our implementation: {e}")
            return {'available': False}
    
    def _import_colleague_implementation(self):
        """Import colleague's implementation."""
        try:
            # This will need to be updated based on colleague's file structure
            # Example imports - adjust based on actual files
            
            # Try common naming patterns
            possible_modules = [
                'colleague_s1_calculator',
                's1_calculator', 
                'bell_inequality_calculator',
                'main_calculator'
            ]
            
            colleague_impl = {}
            
            for module_name in possible_modules:
                try:
                    spec = importlib.util.spec_from_file_location(
                        module_name, 
                        f"colleague_implementation/{module_name}.py"
                    )
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        
                        # Look for calculator class
                        for attr_name in dir(module):
                            attr = getattr(module, attr_name)
                            if (isinstance(attr, type) and 
                                'calculator' in attr_name.lower() or 
                                's1' in attr_name.lower()):
                                colleague_impl['calculator'] = attr
                                break
                        
                        if 'calculator' in colleague_impl:
                            break
                            
                except Exception:
                    continue
            
            if colleague_impl:
                colleague_impl['available'] = True
                return colleague_impl
            else:
                print("⚠️ Colleague implementation not found in colleague_implementation/")
                print("   Expected files: colleague_s1_calculator.py, s1_calculator.py, etc.")
                return {'available': False}
                
        except Exception as e:
            print(f"⚠️ Could not import colleague implementation: {e}")
            return {'available': False}
    
    def _create_test_datasets(self) -> Dict[str, pd.DataFrame]:
        """Create standardized test datasets for comparison."""
        np.random.seed(42)  # Ensure reproducible results
        
        datasets = {}
        
        # Dataset 1: Simple uncorrelated data
        datasets['uncorrelated'] = pd.DataFrame({
            'ASSET_A': np.random.normal(0, 0.02, 100),
            'ASSET_B': np.random.normal(0, 0.02, 100)
        }, index=pd.date_range('2023-01-01', periods=100))
        
        # Dataset 2: Correlated data
        base_factor = np.random.normal(0, 0.015, 100)
        datasets['correlated'] = pd.DataFrame({
            'ASSET_A': 0.7 * base_factor + 0.3 * np.random.normal(0, 0.01, 100),
            'ASSET_B': 0.7 * base_factor + 0.3 * np.random.normal(0, 0.01, 100)
        }, index=pd.date_range('2023-01-01', periods=100))
        
        # Dataset 3: Agricultural-style data
        datasets['agricultural'] = pd.DataFrame({
            'CORN': np.random.normal(0, 0.025, 150),
            'ADM': 0.5 * datasets['correlated']['ASSET_A'][:150] + 0.5 * np.random.normal(0, 0.02, 150),
            'XOM': np.random.normal(0, 0.03, 150)
        }, index=pd.date_range('2023-01-01', periods=150))
        
        return datasets
    
    def validate_daily_returns_calculation(self) -> ValidationResult:
        """Validate daily returns calculation between implementations."""
        print("🧮 Validating daily returns calculation...")
        
        test_data = self.test_datasets['uncorrelated']
        
        # Our implementation
        start_time = time.time()
        our_calculator = self.our_implementation['calculator'](window_size=20)
        our_returns = our_calculator.calculate_daily_returns(test_data)
        our_time = time.time() - start_time
        
        # Colleague's implementation
        start_time = time.time()
        try:
            # This will need adjustment based on colleague's API
            colleague_calculator = self.colleague_implementation['calculator']()
            
            # Try common method names
            if hasattr(colleague_calculator, 'calculate_daily_returns'):
                colleague_returns = colleague_calculator.calculate_daily_returns(test_data)
            elif hasattr(colleague_calculator, 'compute_returns'):
                colleague_returns = colleague_calculator.compute_returns(test_data)
            elif hasattr(colleague_calculator, 'get_returns'):
                colleague_returns = colleague_calculator.get_returns(test_data)
            else:
                raise AttributeError("No returns calculation method found")
                
            colleague_time = time.time() - start_time
            
            # Compare results
            if our_returns.shape == colleague_returns.shape:
                max_diff = np.max(np.abs(our_returns.values - colleague_returns.values))
                passed = max_diff < self.tolerance
            else:
                max_diff = float('inf')
                passed = False
                
        except Exception as e:
            colleague_returns = None
            colleague_time = 0
            max_diff = float('inf')
            passed = False
        
        return ValidationResult(
            test_name="Daily Returns Calculation",
            passed=passed,
            our_result=our_returns,
            colleague_result=colleague_returns,
            difference=max_diff,
            tolerance=self.tolerance,
            execution_time_ours=our_time,
            execution_time_colleague=colleague_time,
            notes=f"Max difference: {max_diff:.2e}" if max_diff != float('inf') else "Calculation failed"
        )
    
    def validate_s1_calculation(self) -> ValidationResult:
        """Validate S1 Bell inequality calculation."""
        print("🔢 Validating S1 calculation...")
        
        test_data = self.test_datasets['correlated']
        
        # Our implementation
        start_time = time.time()
        our_calculator = self.our_implementation['calculator'](window_size=20, threshold_quantile=0.75)
        our_results = our_calculator.analyze_asset_pair(test_data, 'ASSET_A', 'ASSET_B')
        our_s1_values = our_results['s1_time_series']
        our_time = time.time() - start_time
        
        # Colleague's implementation
        start_time = time.time()
        try:
            colleague_calculator = self.colleague_implementation['calculator']()
            
            # Try to run colleague's S1 analysis
            # This will need adjustment based on their API
            if hasattr(colleague_calculator, 'analyze_asset_pair'):
                colleague_results = colleague_calculator.analyze_asset_pair(test_data, 'ASSET_A', 'ASSET_B')
                colleague_s1_values = colleague_results.get('s1_time_series', colleague_results.get('s1_values', []))
            elif hasattr(colleague_calculator, 'calculate_s1'):
                colleague_s1_values = colleague_calculator.calculate_s1(test_data, 'ASSET_A', 'ASSET_B')
            else:
                raise AttributeError("No S1 calculation method found")
                
            colleague_time = time.time() - start_time
            
            # Compare S1 values
            if len(our_s1_values) == len(colleague_s1_values):
                max_diff = np.max(np.abs(np.array(our_s1_values) - np.array(colleague_s1_values)))
                passed = max_diff < self.tolerance
            else:
                max_diff = float('inf')
                passed = False
                
        except Exception as e:
            colleague_s1_values = None
            colleague_time = 0
            max_diff = float('inf')
            passed = False
        
        return ValidationResult(
            test_name="S1 Calculation",
            passed=passed,
            our_result=our_s1_values,
            colleague_result=colleague_s1_values,
            difference=max_diff,
            tolerance=self.tolerance,
            execution_time_ours=our_time,
            execution_time_colleague=colleague_time,
            notes=f"S1 values comparison, max diff: {max_diff:.2e}" if max_diff != float('inf') else "S1 calculation failed"
        )
    
    def validate_violation_detection(self) -> ValidationResult:
        """Validate Bell violation detection logic."""
        print("🚨 Validating violation detection...")
        
        test_data = self.test_datasets['agricultural']
        
        # Our implementation
        start_time = time.time()
        our_calculator = self.our_implementation['calculator'](window_size=20)
        our_results = our_calculator.analyze_asset_pair(test_data, 'CORN', 'ADM')
        our_violation_rate = our_results['violation_results']['violation_rate']
        our_time = time.time() - start_time
        
        # Colleague's implementation
        start_time = time.time()
        try:
            colleague_calculator = self.colleague_implementation['calculator']()
            
            # Try colleague's violation detection
            if hasattr(colleague_calculator, 'analyze_asset_pair'):
                colleague_results = colleague_calculator.analyze_asset_pair(test_data, 'CORN', 'ADM')
                colleague_violation_rate = colleague_results.get('violation_rate', 
                                                              colleague_results.get('violations', {}).get('rate', 0))
            else:
                raise AttributeError("No violation detection method found")
                
            colleague_time = time.time() - start_time
            
            # Compare violation rates
            diff = abs(our_violation_rate - colleague_violation_rate)
            passed = diff < 1.0  # Allow 1% difference in violation rates
            
        except Exception as e:
            colleague_violation_rate = None
            colleague_time = 0
            diff = float('inf')
            passed = False
        
        return ValidationResult(
            test_name="Violation Detection",
            passed=passed,
            our_result=our_violation_rate,
            colleague_result=colleague_violation_rate,
            difference=diff,
            tolerance=1.0,  # 1% tolerance for violation rates
            execution_time_ours=our_time,
            execution_time_colleague=colleague_time,
            notes=f"Violation rates: Ours={our_violation_rate:.1f}%, Colleague={colleague_violation_rate:.1f}%" if colleague_violation_rate is not None else "Violation detection failed"
        )
    
    def run_all_validations(self) -> ComparisonSummary:
        """Run all validation tests."""
        print("🧪 RUNNING CROSS-IMPLEMENTATION VALIDATION")
        print("=" * 50)
        
        if not self.our_implementation['available']:
            print("❌ Our implementation not available")
            return ComparisonSummary(0, 0, 0, False, False, False, [])
        
        if not self.colleague_implementation['available']:
            print("❌ Colleague implementation not available")
            print("   Place colleague's code in colleague_implementation/ directory")
            return ComparisonSummary(0, 0, 0, False, False, False, [])
        
        # Run validation tests
        validation_tests = [
            self.validate_daily_returns_calculation,
            self.validate_s1_calculation,
            self.validate_violation_detection
        ]
        
        results = []
        for test_func in validation_tests:
            try:
                result = test_func()
                results.append(result)
                
                status = "✅ PASS" if result.passed else "❌ FAIL"
                print(f"{status} {result.test_name}")
                print(f"   Our time: {result.execution_time_ours:.3f}s")
                print(f"   Colleague time: {result.execution_time_colleague:.3f}s")
                print(f"   Notes: {result.notes}")
                print()
                
            except Exception as e:
                print(f"❌ FAIL {test_func.__name__}: {str(e)}")
                results.append(ValidationResult(
                    test_name=test_func.__name__,
                    passed=False,
                    our_result=None,
                    colleague_result=None,
                    difference=float('inf'),
                    tolerance=self.tolerance,
                    execution_time_ours=0,
                    execution_time_colleague=0,
                    notes=f"Test failed: {str(e)}"
                ))
        
        # Generate summary
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.passed)
        failed_tests = total_tests - passed_tests
        
        mathematical_accuracy = all(r.passed for r in results if 'calculation' in r.test_name.lower())
        performance_acceptable = all(r.execution_time_colleague < r.execution_time_ours * 10 
                                   for r in results if r.execution_time_colleague > 0)
        overall_compatible = passed_tests >= total_tests * 0.8  # 80% pass rate
        
        summary = ComparisonSummary(
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            mathematical_accuracy=mathematical_accuracy,
            performance_acceptable=performance_acceptable,
            overall_compatible=overall_compatible,
            detailed_results=results
        )
        
        # Print summary
        print("=" * 50)
        print("📊 VALIDATION SUMMARY")
        print("=" * 50)
        print(f"Total tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success rate: {(passed_tests/total_tests)*100:.1f}%")
        print()
        print(f"Mathematical accuracy: {'✅' if mathematical_accuracy else '❌'}")
        print(f"Performance acceptable: {'✅' if performance_acceptable else '❌'}")
        print(f"Overall compatible: {'✅' if overall_compatible else '❌'}")
        
        if overall_compatible:
            print("\n🎉 IMPLEMENTATIONS ARE COMPATIBLE!")
            print("   Both implementations produce consistent results")
            print("   Ready for integration and optimization")
        else:
            print("\n⚠️ IMPLEMENTATIONS HAVE DIFFERENCES")
            print("   Review failed tests and resolve discrepancies")
            print("   Consider mathematical validation and debugging")
        
        return summary

def main():
    """Main entry point for validation framework."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Cross-implementation validation framework')
    parser.add_argument('--test-all', action='store_true', help='Run all validation tests')
    parser.add_argument('--mathematical-only', action='store_true', help='Run only mathematical validation')
    parser.add_argument('--tolerance', type=float, default=1e-10, help='Numerical tolerance')
    
    args = parser.parse_args()
    
    validator = CrossImplementationValidator(tolerance=args.tolerance)
    
    if args.test_all or args.mathematical_only:
        summary = validator.run_all_validations()
        
        # Save results
        import json
        results_dict = {
            'summary': {
                'total_tests': summary.total_tests,
                'passed_tests': summary.passed_tests,
                'failed_tests': summary.failed_tests,
                'mathematical_accuracy': summary.mathematical_accuracy,
                'performance_acceptable': summary.performance_acceptable,
                'overall_compatible': summary.overall_compatible
            },
            'detailed_results': [
                {
                    'test_name': r.test_name,
                    'passed': r.passed,
                    'difference': r.difference if r.difference != float('inf') else None,
                    'tolerance': r.tolerance,
                    'execution_time_ours': r.execution_time_ours,
                    'execution_time_colleague': r.execution_time_colleague,
                    'notes': r.notes
                }
                for r in summary.detailed_results
            ]
        }
        
        with open('comparison/validation_results.json', 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"\n📄 Results saved to comparison/validation_results.json")
        
        return summary.overall_compatible
    else:
        print("Use --test-all to run validation tests")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)