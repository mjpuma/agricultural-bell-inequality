# Bell Inequality Analysis Project Structure

## 📁 Directory Structure

```
bell-inequality-analysis/
├── src/                          # Main source code
│   ├── bell_inequality_analyzer.py    # Main Bell inequality analysis
│   ├── cross_mandelbrot_analyzer.py   # Cross-variable fractal analysis
│   └── __init__.py                     # Package initialization
├── docs/                         # Documentation
│   ├── bell_inequality_methodology.tex # Comprehensive methodology
│   └── README.md                       # Project documentation
├── examples/                     # Example scripts
│   ├── complete_example.py            # Full workflow example
│   └── corrected_s1_sam_approach.py   # Standalone S1 implementation
├── legacy/                       # Original/comparison implementations
│   ├── analyze_yahoo_finance_bell.py  # Original Yahoo Finance code
│   ├── updated_cross_mandelbrot_metrics.py # Original cross-Mandelbrot
│   ├── run_comprehensive_bell_analysis.py # Original comprehensive runner
│   └── run_integrated_analysis.py     # Original integration script
├── results/                      # Analysis outputs (created during runs)
├── tests/                        # Test files (for future development)
├── requirements.txt              # Python dependencies
└── organize_project.py          # This organization script
```

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run basic analysis:**
   ```bash
   python examples/complete_example.py
   ```

3. **Advanced usage:**
   ```python
   from src.bell_inequality_analyzer import quick_bell_analysis
   analyzer = quick_bell_analysis()
   ```

## 📊 Key Files

- **`src/bell_inequality_analyzer.py`**: Main analysis class with S1 Bell inequality
- **`src/cross_mandelbrot_analyzer.py`**: Cross-variable fractal analysis
- **`docs/bell_inequality_methodology.tex`**: Complete methodology documentation
- **`examples/complete_example.py`**: Full workflow demonstration

## 🔬 Methodology

This implementation uses the **Zarifian et al. (2025) approach** with:
- Cumulative returns (critical for violations!)
- Sign-based binary outcomes
- Absolute return regime thresholds
- Direct expectation calculations

## 📈 Results

Typical findings on tech stocks:
- **14-30% Bell inequality violation rates**
- **Strongest effects in GOOGL-NVDA, GOOGL-TSLA pairs**
- **Quantum-like correlations in financial markets**

## 📚 Documentation

See `docs/bell_inequality_methodology.tex` for complete theoretical background,
implementation details, and empirical findings.
