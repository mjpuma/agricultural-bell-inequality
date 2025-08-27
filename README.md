# Bell Inequality Violations in Financial Markets

This repository contains comprehensive analysis of Bell inequality violations in financial markets, focusing on cross-sector correlations and their implications for traditional financial models.

## 📊 Key Analyses

### 1. Cross-Sector Bell Inequality Analysis
- **Script**: `cross_sector_analysis.py`
- **Purpose**: Analyze S1 Bell inequality violations across different market sectors
- **Results**: `results/FINAL_CROSS_SECTOR_RESULTS/`

### 2. Food Systems Analysis  
- **Script**: `food_systems_bell_analysis.py`
- **Purpose**: Detailed analysis of food system stocks and their quantum correlations
- **Results**: `results/FINAL_CROSS_SECTOR_RESULTS/`

### 3. CAPM β Breakdown Demonstration
- **Script**: `src/capm_beta_breakdown_demo.py`
- **Purpose**: Concrete example showing how S1 violations correspond to CAPM model breakdown
- **Demonstration**: XOM-JPM pair (Energy vs Finance) with 67.4% S1 violations
- **Key Finding**: β estimates become unreliable when S1 > 2 (independence violations)

## 🚀 Quick Start

### Run Cross-Sector Analysis
```bash
python cross_sector_analysis.py
```

### Run Food Systems Analysis
```bash
python food_systems_bell_analysis.py
```

### Run CAPM β Breakdown Demo
```bash
python src/capm_beta_breakdown_demo.py
```

## 📁 Output Locations

### Cross-Sector Results
- **Figures**: `results/FINAL_CROSS_SECTOR_RESULTS/figures/`
- **Summary**: `results/FINAL_CROSS_SECTOR_RESULTS/summary/`
- **Tables**: `results/FINAL_CROSS_SECTOR_RESULTS/tables/`

### CAPM Demo Results
- **Visualization**: `results/FINAL_CROSS_SECTOR_RESULTS/summary/capm_beta_breakdown_demo_*.png`
- **Statistics**: `results/FINAL_CROSS_SECTOR_RESULTS/tables/capm_beta_breakdown_summary_*.xlsx`

## 🎯 Key Findings

### Cross-Sector Analysis (2010-2025)
- **Highest violations**: Energy vs Finance (62.7%)
- **Top pair**: XOM-JPM (67.6% violations)
- **Crisis impact**: 37-90% violations during major crises

### CAPM Model Breakdown
- **When S1 > 2**: Independence assumptions violated
- **β becomes unstable**: Confidence intervals widen
- **Risk underestimated**: Traditional models fail during S1 violations

## 📈 S1 Interpretation

- **S1 < 2**: Classical independence regime
- **2 ≤ S1 < 2.83**: Transitional regime (enhanced correlations)
- **S1 ≥ 2.83**: Strong interdependence (breakdown of classical independence)

## 🔧 Dependencies

```bash
pip install pandas numpy matplotlib seaborn yfinance networkx openpyxl
```

## 📚 Documentation

- **Methodology**: `docs/bell_inequality_methodology.tex`
- **Results Summary**: `results/FINAL_CROSS_SECTOR_RESULTS/README.md`
- **Examples**: `examples/` directory

## 🤝 Contributing

This project demonstrates how quantum-like correlations in financial markets can indicate breakdowns in traditional financial models like CAPM, factor models, and VaR.