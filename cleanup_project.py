#!/usr/bin/env python3
"""
PROJECT CLEANUP SCRIPT
======================
Clean up old scripts, plots, and files while preserving important results
"""

import os
import shutil
from pathlib import Path
from datetime import datetime

def cleanup_project():
    """Clean up the project directory"""
    
    print("🧹 PROJECT CLEANUP STARTING")
    print("=" * 40)
    
    # Create cleanup directories
    cleanup_dirs = {
        'archive': Path('archive'),
        'archive/old_scripts': Path('archive/old_scripts'),
        'archive/old_docs': Path('archive/old_docs'),
        'archive/temp_files': Path('archive/temp_files')
    }
    
    for dir_path in cleanup_dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Files to archive (old/temporary scripts)
    scripts_to_archive = [
        'create_s1_matrices.py',
        'create_results_excel.py', 
        'export_results_to_excel.py',
        'complete_yahoo_finance_test.py',
        'verify_data_sources.py',
        'create_detailed_correlation_analysis.py',
        'create_organized_results.py',
        'github_setup_guide.py',
        'prepare_for_github.py',
        'organize_project.py',
        'run_food_crisis_analysis.py',
        'test_crisis_debug.py',
        'analyze_wharton_tick_file.py',
        'wdrs_recommendations.py'
    ]
    
    # Documentation to archive (superseded by organized results)
    docs_to_archive = [
        'advanced_bell_results.png',
        'bell_analysis_results.png',
        'cross_mandelbrot_analysis.png',
        'food_crisis_covid_food_disruption_analysis.png',
        'food_crisis_drought_2012_analysis.png', 
        'food_crisis_ukraine_war_food_crisis_analysis.png',
        's1_analysis_sam_approach.png',
        'yahoo_finance_bell_analysis.png'
    ]
    
    # Temporary files to remove
    temp_files = [
        '.DS_Store',
        '__pycache__',
        '*.pyc',
        '~$*.xlsx'  # Excel temp files
    ]
    
    moved_count = 0
    removed_count = 0
    
    # Archive old scripts
    print("\n📁 ARCHIVING OLD SCRIPTS")
    print("-" * 25)
    
    for script in scripts_to_archive:
        script_path = Path(script)
        if script_path.exists():
            archive_path = cleanup_dirs['archive/old_scripts'] / script
            try:
                shutil.move(str(script_path), str(archive_path))
                print(f"   📦 Archived: {script}")
                moved_count += 1
            except Exception as e:
                print(f"   ❌ Failed to archive {script}: {e}")
    
    # Archive old documentation/plots (now in results/)
    print("\n📊 ARCHIVING OLD PLOTS/DOCS")
    print("-" * 28)
    
    for doc in docs_to_archive:
        doc_path = Path(doc)
        if doc_path.exists():
            archive_path = cleanup_dirs['archive/old_docs'] / doc
            try:
                shutil.move(str(doc_path), str(archive_path))
                print(f"   📦 Archived: {doc}")
                moved_count += 1
            except Exception as e:
                print(f"   ❌ Failed to archive {doc}: {e}")
    
    # Remove temporary files
    print("\n🗑️  REMOVING TEMPORARY FILES")
    print("-" * 28)
    
    # Remove .DS_Store files
    for ds_store in Path('.').rglob('.DS_Store'):
        try:
            ds_store.unlink()
            print(f"   🗑️  Removed: {ds_store}")
            removed_count += 1
        except Exception as e:
            print(f"   ❌ Failed to remove {ds_store}: {e}")
    
    # Remove __pycache__ directories
    for pycache in Path('.').rglob('__pycache__'):
        if pycache.is_dir():
            try:
                shutil.rmtree(pycache)
                print(f"   🗑️  Removed: {pycache}")
                removed_count += 1
            except Exception as e:
                print(f"   ❌ Failed to remove {pycache}: {e}")
    
    # Remove Excel temp files
    for temp_excel in Path('.').rglob('~$*.xlsx'):
        try:
            temp_excel.unlink()
            print(f"   🗑️  Removed: {temp_excel}")
            removed_count += 1
        except Exception as e:
            print(f"   ❌ Failed to remove {temp_excel}: {e}")
    
    # Create clean project structure summary
    create_clean_structure_summary()
    
    print(f"\n✅ CLEANUP COMPLETE!")
    print(f"   📦 Files archived: {moved_count}")
    print(f"   🗑️  Files removed: {removed_count}")
    print(f"   📁 Archive location: archive/")
    
    return moved_count, removed_count

def create_clean_structure_summary():
    """Create a summary of the clean project structure"""
    
    structure_summary = """# 📁 CLEAN PROJECT STRUCTURE

## 🎯 **ACTIVE FILES (Keep These)**

### **Core Analysis Code**
- `src/bell_inequality_analyzer.py` - Main Bell inequality analysis engine
- `src/food_systems_analyzer.py` - Specialized food systems analysis
- `src/preset_configurations.py` - Asset groups and crisis period definitions
- `src/results_manager.py` - Results organization and management

### **Current Analysis Scripts**
- `create_improved_correlation_analysis.py` - Latest improved analysis (ACTIVE)
- `sam.ipynb` - Jupyter notebook for interactive analysis

### **Documentation & Guides**
- `ORGANIZED_RESULTS_SUMMARY.md` - Complete results summary
- `YAHOO_FINANCE_DATA_LIMITATIONS.md` - Data limitations and solutions
- `docs/bell_inequality_methodology.tex` - LaTeX methodology documentation

### **Project Configuration**
- `setup.py` - Package setup
- `LICENSE` - MIT license
- `.gitignore` - Git ignore rules

### **Organized Results (MOST IMPORTANT)**
- `results/` - All organized analysis outputs
  - `results/excel_files/` - Excel analyses and correlation tables
  - `results/figures/` - Professional publication-ready figures
  - `results/reports/` - Comprehensive analysis reports

## 🗂️ **ARCHIVED FILES (Moved to archive/)**

### **Old Scripts (archive/old_scripts/)**
- Various development and testing scripts
- Superseded by improved versions
- Kept for reference but not needed for active work

### **Old Plots (archive/old_docs/)**
- Original plots with formatting issues
- Superseded by improved figures in results/
- Historical reference only

## 🎯 **WHAT TO USE NOW**

### **For Analysis:**
1. **`create_improved_correlation_analysis.py`** - Latest analysis script
2. **`results/` folder** - All organized outputs
3. **`src/` modules** - Core analysis engines

### **For WDRS Phase:**
1. **`results/excel_files/`** - Priority asset lists and correlation tables
2. **`results/figures/`** - Professional figures for publication
3. **`YAHOO_FINANCE_DATA_LIMITATIONS.md`** - Data requirements documentation

### **For Science Publication:**
1. **`results/figures/`** - Publication-ready visualizations
2. **`docs/bell_inequality_methodology.tex`** - LaTeX methodology
3. **`results/reports/`** - Comprehensive analysis reports

## ✅ **CLEAN AND ORGANIZED**

The project is now clean and focused on the essential files needed for:
- WDRS data download phase
- Science journal publication
- Continued analysis and development

All historical work is preserved in `archive/` but the main directory is clean and professional.
"""
    
    with open('CLEAN_PROJECT_STRUCTURE.md', 'w') as f:
        f.write(structure_summary)
    
    print(f"   📋 Created: CLEAN_PROJECT_STRUCTURE.md")

def show_current_structure():
    """Show the current project structure"""
    
    print("\n📁 CURRENT PROJECT STRUCTURE")
    print("=" * 30)
    
    # Show main directory contents
    main_files = []
    for item in Path('.').iterdir():
        if item.name.startswith('.'):
            continue
        if item.is_file():
            main_files.append(f"📄 {item.name}")
        elif item.is_dir():
            main_files.append(f"📁 {item.name}/")
    
    for item in sorted(main_files):
        print(f"   {item}")
    
    # Show results structure
    results_path = Path('results')
    if results_path.exists():
        print(f"\n📊 RESULTS STRUCTURE")
        print("-" * 20)
        for subdir in results_path.iterdir():
            if subdir.is_dir():
                file_count = len(list(subdir.glob('*')))
                print(f"   📁 {subdir.name}/ ({file_count} files)")

if __name__ == "__main__":
    # Show current structure
    show_current_structure()
    
    # Ask for confirmation
    print(f"\n⚠️  This will archive old scripts and remove temporary files.")
    print(f"   All important results are preserved in results/ folder.")
    
    confirm = input("\nProceed with cleanup? (y/n): ").strip().lower()
    
    if confirm == 'y':
        moved, removed = cleanup_project()
        
        print(f"\n📁 FINAL PROJECT STRUCTURE")
        print("=" * 30)
        show_current_structure()
        
    else:
        print("Cleanup cancelled.")