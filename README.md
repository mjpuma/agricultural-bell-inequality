# Agricultural Cross-Sector Bell Inequality Analysis

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-10%20passing-green.svg)](tests/)

**Detecting quantum-like correlations in global food systems using Bell inequality tests for Science journal publication.**

This repository implements the Enhanced S1 Bell inequality calculator following Zarifian et al. (2025) methodology to detect non-local correlations in agricultural cross-sector relationships. The analysis focuses on supply chain dependencies, crisis transmission mechanisms, and quantum-like entanglement effects in food systems.

## 🌾 Research Objective

Detect and analyze quantum-like correlations in global food systems using Bell inequality tests, with focus on:

- **Agricultural supply chains**: Commodity → Processor → Consumer relationships
- **Cross-sector dependencies**: Energy → Fertilizer → Agriculture transmission
- **Crisis amplification**: How market stress reveals hidden correlations
- **Food security implications**: Systemic risk assessment through quantum correlations

## 🔬 Methodology

### S1 Conditional Bell Inequality (Zarifian et al. 2025)

The implementation uses the mathematically accurate S1 Bell inequality:

```
S1 = ⟨ab⟩₀₀ + ⟨ab⟩₀₁ + ⟨ab⟩₁₀ - ⟨ab⟩₁₁
```

Where:
- **⟨ab⟩ₓᵧ**: Conditional expectations for regime combinations
- **Classical bound**: |S1| ≤ 2
- **Quantum bound**: |S1| ≤ 2√2 ≈ 2.83
- **Violation**: |S1| > 2 indicates quantum-like correlations

### Key Features

- ✅ **Exact daily returns**: `Rᵢ,ₜ = (Pᵢ,ₜ - Pᵢ,ₜ₋₁) / Pᵢ,ₜ₋₁`
- ✅ **Binary indicators**: `I{|Rₐ,ₜ| ≥ rₐ}` for regime classification
- ✅ **Sign function**: `Sign(Rᵢ,ₜ) = +1 if Rᵢ,ₜ ≥ 0, -1 if Rᵢ,ₜ < 0`
- ✅ **Conditional expectations**: Exact mathematical implementation
- ✅ **Missing data handling**: Robust error handling
- ✅ **Crisis detection**: Configurable parameters for crisis analysis

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/[username]/agricultural-bell-inequality.git
cd agricultural-bell-inequality

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

### Basic Usage

```python
from src.enhanced_s1_calculator import EnhancedS1Calculator
import yfinance as yf

# Download agricultural data
data = yf.download(['CORN', 'ADM'], period='1y')['Close']

# Initialize calculator
calculator = EnhancedS1Calculator(
    window_size=20,
    threshold_method='absolute',
    threshold_value=0.01  # Zarifian et al. (2025) methodology
)

# Calculate returns and analyze
returns = calculator.calculate_daily_returns(data)
results = calculator.analyze_asset_pair(returns, 'CORN', 'ADM')

# Check for Bell violations
violations = results['violation_results']
print(f"Violations: {violations['violations']}/{violations['total_values']}")
print(f"Violation rate: {violations['violation_rate']:.1f}%")
```

### Agricultural Cross-Sector Analysis

```python
from src.agricultural_universe_manager import AgriculturalUniverseManager

# Load agricultural universe (60+ companies)
universe = AgriculturalUniverseManager()
companies = universe.get_companies_by_tier('Tier 1')  # Energy/Transport/Chemicals

# Analyze cross-sector pairs
agricultural_pairs = [
    ('CORN', 'ADM'),   # Commodity → Processor
    ('CORN', 'CF'),    # Commodity → Fertilizer  
    ('CF', 'XOM'),     # Fertilizer → Energy
]

batch_results = calculator.batch_analyze_pairs(returns, agricultural_pairs)
print(f"Overall violation rate: {batch_results['summary']['overall_violation_rate']:.2f}%")
```

## 📊 Agricultural Universe

The analysis covers 60+ companies across three tiers:

### Tier 1: Energy, Transportation & Chemicals
- **Energy**: XOM, CVX, COP (natural gas for fertilizer)
- **Transportation**: UNP, CSX, NSC (rail/shipping logistics)
- **Chemicals**: CF, MOS, NTR (fertilizers, pesticides)

### Tier 2: Finance & Equipment  
- **Finance**: JPM, BAC, WFC (agricultural credit)
- **Equipment**: DE, CAT, AGCO (farming machinery)

### Tier 3: Policy-Linked
- **Government**: Policy-sensitive agricultural companies
- **Regulation**: Companies affected by agricultural policy

## 🔍 Crisis Period Analysis

### Supported Crisis Periods
- **COVID-19 Food Disruption** (2020): Supply chain disruptions
- **Ukraine War Food Crisis** (2022-2023): Global grain export disruptions  
- **2008 Global Food Price Crisis**: Food riots and export restrictions
- **2012 US Drought**: Severe drought in corn/soybean belt

### Crisis Detection Parameters
```python
# Crisis analysis with higher threshold
crisis_calculator = EnhancedS1Calculator(
    window_size=15,      # Shorter window for rapid detection
    threshold_value=0.02  # Higher threshold filters minor fluctuations
)
```

## 📈 Expected Results

Based on Zarifian et al. (2025) and food systems research:

- **Supply chain pairs**: 20-35% violation rates
- **Climate-sensitive pairs**: 25-40% violation rates
- **Crisis periods**: 40-60% violation rates (amplification effect)
- **Cross-sector pairs**: 10-20% violation rates

## 🧪 Testing

Comprehensive test suite with 10 test cases covering all mathematical components:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test
python tests/test_enhanced_s1_calculator.py
```

All tests validate:
- Mathematical accuracy (10+ decimal places)
- Zarifian et al. (2025) compliance
- Edge case handling
- Crisis parameter support

## 📚 Documentation

- **[Implementation Summary](docs/enhanced_s1_implementation_summary.md)**: Complete technical documentation
- **[Threshold Analysis](docs/threshold_correction_analysis.md)**: Zarifian et al. (2025) compliance analysis
- **[Methodology](docs/bell_inequality_methodology.tex)**: Mathematical foundations
- **[Specifications](.kiro/specs/agricultural-cross-sector-analysis/)**: Detailed requirements and design

## 🎯 Research Applications

### Food Security Analysis
- **Supply chain vulnerability assessment**
- **Crisis transmission mechanism detection**
- **Systemic risk quantification**

### Academic Research
- **First detection of quantum effects in food systems**
- **Non-local correlations in agricultural supply chains**
- **Crisis amplification of quantum correlations**

### Policy Applications
- **Early warning systems for food crises**
- **Supply chain resilience assessment**
- **Agricultural risk management**

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@software{agricultural_bell_inequality_2025,
  title={Agricultural Cross-Sector Bell Inequality Analysis},
  author={[Your Name]},
  year={2025},
  url={https://github.com/[username]/agricultural-bell-inequality},
  note={Implementation following Zarifian et al. (2025) methodology}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests for any improvements.

## 📧 Contact

For questions about the methodology or implementation, please open an issue or contact [your email].

---

**Keywords**: Bell inequality, quantum correlations, agricultural finance, food systems, crisis detection, supply chain analysis, non-local correlations, Zarifian methodology