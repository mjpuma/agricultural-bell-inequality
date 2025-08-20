#!/usr/bin/env python3
"""
CLEAR OLD RESULTS AND RERUN FRESH ANALYSIS
==========================================
Clear all old analysis results and run fresh analysis with current optimized code
"""

import shutil
import os
from pathlib import Path
from datetime import datetime

def clear_old_results():
    """Clear all old analysis results"""
    
    print("🧹 CLEARING OLD ANALYSIS RESULTS")
    print("=" * 40)
    
    results_dir = Path('results')
    
    if results_dir.exists():
        # Count files before clearing
        total_files = 0
        for subdir in results_dir.iterdir():
            if subdir.is_dir():
                file_count = len(list(subdir.glob('*')))
                total_files += file_count
                print(f"   📁 {subdir.name}/: {file_count} files")
        
        print(f"\n🗑️  Clearing {total_files} old result files...")
        
        # Clear each subdirectory but keep the structure
        for subdir in results_dir.iterdir():
            if subdir.is_dir():
                for file in subdir.glob('*'):
                    if file.is_file():
                        try:
                            file.unlink()
                            print(f"   🗑️  Removed: {file.name}")
                        except Exception as e:
                            print(f"   ❌ Failed to remove {file.name}: {e}")
        
        print(f"✅ Cleared {total_files} old result files")
    else:
        print("📁 Results directory doesn't exist - will be created fresh")
    
    # Also clear any old plots in main directory
    old_plots = [
        'cross_mandelbrot_analysis.png',
        'food_systems_analysis.png',
        'bell_analysis_results.png'
    ]
    
    cleared_plots = 0
    for plot in old_plots:
        plot_path = Path(plot)
        if plot_path.exists():
            try:
                plot_path.unlink()
                print(f"   🗑️  Removed old plot: {plot}")
                cleared_plots += 1
            except Exception as e:
                print(f"   ❌ Failed to remove {plot}: {e}")
    
    if cleared_plots > 0:
        print(f"✅ Cleared {cleared_plots} old plots from main directory")
    
    return total_files + cleared_plots

def run_fresh_analysis():
    """Run fresh analysis with current optimized code"""
    
    print(f"\n🚀 RUNNING FRESH FOOD SYSTEMS ANALYSIS")
    print("=" * 50)
    print("🎯 Using current optimized code with all improvements")
    print("🔬 Bell inequality + Cross-Mandelbrot analysis")
    print("📊 Professional visualizations with fixed formatting")
    
    # Import and run the current analysis
    import subprocess
    import sys
    
    try:
        # Run the main analysis script
        result = subprocess.run([sys.executable, 'food_systems_analysis.py'], 
                              capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ FRESH ANALYSIS COMPLETED SUCCESSFULLY!")
            print("\n📊 Analysis Output:")
            print(result.stdout)
            
            if result.stderr:
                print("\n⚠️  Warnings:")
                print(result.stderr)
                
        else:
            print("❌ ANALYSIS FAILED!")
            print("Error output:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ ANALYSIS TIMED OUT (>5 minutes)")
        return False
    except Exception as e:
        print(f"❌ ANALYSIS ERROR: {e}")
        return False
    
    return True

def verify_fresh_results():
    """Verify that fresh results were generated"""
    
    print(f"\n🔍 VERIFYING FRESH RESULTS")
    print("=" * 30)
    
    results_dir = Path('results')
    
    if not results_dir.exists():
        print("❌ Results directory not created")
        return False
    
    # Check each subdirectory
    subdirs = ['figures', 'excel_files', 'reports']
    total_new_files = 0
    
    for subdir_name in subdirs:
        subdir = results_dir / subdir_name
        if subdir.exists():
            files = list(subdir.glob('*'))
            file_count = len(files)
            total_new_files += file_count
            print(f"   📁 {subdir_name}/: {file_count} new files")
            
            # Show newest files
            if files:
                newest_file = max(files, key=lambda f: f.stat().st_mtime)
                mod_time = datetime.fromtimestamp(newest_file.stat().st_mtime)
                print(f"      📄 Latest: {newest_file.name} ({mod_time.strftime('%H:%M:%S')})")
        else:
            print(f"   ❌ {subdir_name}/ directory missing")
    
    # Check for Cross-Mandelbrot visualization
    cross_mandelbrot_plot = Path('cross_mandelbrot_analysis.png')
    if cross_mandelbrot_plot.exists():
        mod_time = datetime.fromtimestamp(cross_mandelbrot_plot.stat().st_mtime)
        print(f"   📊 Cross-Mandelbrot plot: ✅ ({mod_time.strftime('%H:%M:%S')})")
        total_new_files += 1
    else:
        print(f"   📊 Cross-Mandelbrot plot: ❌")
    
    print(f"\n✅ VERIFICATION COMPLETE: {total_new_files} fresh files generated")
    
    if total_new_files > 10:
        print("🌟 EXCELLENT: Comprehensive fresh results generated!")
        return True
    elif total_new_files > 5:
        print("✅ GOOD: Basic fresh results generated")
        return True
    else:
        print("⚠️  LIMITED: Few results generated - may need investigation")
        return False

def main():
    """Main execution function"""
    
    print("🔄 FRESH ANALYSIS PIPELINE")
    print("=" * 30)
    print("This will:")
    print("1. Clear all old analysis results")
    print("2. Run fresh analysis with current optimized code")
    print("3. Generate clean, consistent outputs")
    print("4. Verify results were created successfully")
    
    # Ask for confirmation
    confirm = input("\nProceed with clearing old results and rerunning? (y/n): ").strip().lower()
    
    if confirm != 'y':
        print("Operation cancelled.")
        return
    
    # Step 1: Clear old results
    cleared_count = clear_old_results()
    
    # Step 2: Run fresh analysis
    success = run_fresh_analysis()
    
    if not success:
        print("\n❌ FRESH ANALYSIS FAILED - stopping here")
        return
    
    # Step 3: Verify results
    verification_success = verify_fresh_results()
    
    # Final summary
    print(f"\n🎉 FRESH ANALYSIS PIPELINE COMPLETE!")
    print("=" * 40)
    print(f"📊 Old files cleared: {cleared_count}")
    print(f"🔬 Fresh analysis: {'✅ SUCCESS' if success else '❌ FAILED'}")
    print(f"🔍 Results verification: {'✅ PASSED' if verification_success else '⚠️  PARTIAL'}")
    
    if success and verification_success:
        print(f"\n🌟 READY FOR SCIENCE PUBLICATION!")
        print("📁 All results in results/ folder are now fresh and consistent")
        print("🎯 Ready for WDRS phase with clean baseline")
    else:
        print(f"\n⚠️  Some issues detected - please review output above")

if __name__ == "__main__":
    main()