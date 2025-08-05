# Bell Inequalities in Financial Markets: A Methodological Exploration

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Exploratory Research](https://img.shields.io/badge/Status-Exploratory-orange.svg)]()

## 🔬 **Exploring Bell Inequalities in Financial Data**

This repository provides a comprehensive framework for exploring **Bell inequalities** in financial market data, comparing traditional **CHSH** approaches with **conditional S1 Bell tests**. The code implements multiple methodologies and allows extensive parameter experimentation to investigate regime-dependent correlations in financial markets.

**⚠️ Note**: This is exploratory research. The methodology is designed for experimentation and parameter tuning rather than definitive conclusions.

## 📚 **Research Context**

### **Theoretical Foundation**

Bell inequalities, originally from quantum physics, provide a mathematical framework for detecting correlations that exceed classical physics bounds. Recent academic work by [Zarifian et al. (2025)](https://doi.org/10.1016/j.jfds.2025.100164) demonstrates their application to financial crisis detection, suggesting **conditional Bell tests** may be more sensitive than traditional approaches.

### **Two Methodological Approaches**

| Aspect | CHSH Inequality | S1 Conditional Bell Test |
|--------|-----------------|--------------------------|
| **Formula** | `\|E(AB) + E(AB') + E(A'B) - E(A'B')\|` | `E[AB\|x₀,y₀] + E[AB\|x₀,y₁] + E[AB\|x₁,y₀] - E[AB\|x₁,y₁]` |
| **Expectation Type** | Unconditional | Conditional on market regimes |
| **Market Interpretation** | Overall correlation | Regime-specific correlation |
| **Sensitivity** | General correlations | Context-dependent correlations |
| **Parameter Sensitivity** | Moderate | **High - Many tunable parameters** |

## 🛠 **Installation & Setup**

### **Requirements**
```bash
pip install pandas numpy matplotlib seaborn scipy
```

### **Basic Usage**
```python
from thorough_aggregation_analyzer import ThoroughAggregationAnalyzer

# Initialize with default parameters
analyzer = ThoroughAggregationAnalyzer()

# Run analysis with customizable parameters
results = analyzer.load_and_analyze_thorough(
    file_path='your_data.csv.gz',
    frequency='15min'  # Adjustable: '5min', '15min', '30min', '1hour'
)
```

## ⚙️ **Key Parameters to Experiment With**

### **1. Time Window Parameters**

```python
# Modify in _perform_conditional_bell_analysis()
conditional_results = analyzer._perform_conditional_bell_analysis(
    window_size=20,              # Try: 10, 20, 30, 50
    threshold_quantile=0.75      # Try: 0.6, 0.7, 0.75, 0.8, 0.9
)
```

**What to explore:**
- **`window_size`**: Shorter windows (10-15) capture rapid changes, longer windows (30-50) provide stability
- **`threshold_quantile`**: Higher values (0.8-0.9) focus on extreme events, lower values (0.6-0.7) capture more moderate regime changes

### **2. Aggregation Methods**

The framework tests 6 different price aggregation methods. Modify in `_create_bars_all_methods()`:

```python
methods = {
    'OHLC': 'First/Max/Min/Last prices',           # Traditional OHLC bars
    'Average': 'Simple average of all tick prices', # Smoothed pricing  
    'VWAP': 'Volume-weighted average price',       # Volume-adjusted
    'LastTick': 'Last tick price only',            # Most granular
    'Median': 'Median tick price (robust)',        # Outlier-resistant
    'TWAP': 'Time-weighted average price'          # Time-adjusted
}
```

**Experimentation suggestions:**
- Compare violation rates across methods
- Test which methods are most sensitive to regime changes
- Analyze correlation between aggregation method and violation patterns

### **3. Binary Observable Definitions**

The core of Bell tests lies in how you define binary measurements. Current implementation in `_perform_conditional_bell_analysis()`:

```python
# Current approach - modify these for experimentation
a = np.sign(RA)  # Price direction: +1 (up), -1 (down)
b = np.sign(RB)  # Price direction: +1 (up), -1 (down)

# Measurement settings (volatility regimes)
x0 = RA.abs() >= thresholds[ticker_A]  # High volatility
x1 = ~x0                               # Low volatility
y0 = RB.abs() >= thresholds[ticker_B]  # High volatility
y1 = ~y0                               # Low volatility
```

**Alternative observable definitions to try:**

```python
# Momentum-based observables
a = np.sign(RA - RA.rolling(5).mean())  # Above/below short MA
b = np.sign(RB - RB.rolling(5).mean())

# Magnitude-based observables  
a = np.where(RA.abs() > RA.abs().std(), 1, -1)  # Large/small moves
b = np.where(RB.abs() > RB.abs().std(), 1, -1)

# Z-score based observables
a = np.where((RA - RA.mean())/RA.std() > 0, 1, -1)  # Above/below mean
b = np.where((RB - RB.mean())/RB.std() > 0, 1, -1)
```

### **4. Regime Definition Variations**

Experiment with different ways to define market regimes:

```python
# Current: Percentile-based thresholds
thresholds = window_returns.abs().quantile(threshold_quantile)

# Alternative 1: Fixed percentage thresholds
thresholds = pd.Series({ticker_A: 0.02, ticker_B: 0.02})  # 2% moves

# Alternative 2: Standard deviation multiples
vol_A = window_returns[ticker_A].std()
vol_B = window_returns[ticker_B].std()
thresholds = pd.Series({ticker_A: 1.5*vol_A, ticker_B: 1.5*vol_B})

# Alternative 3: Rolling volatility regimes
rolling_vol_A = window_returns[ticker_A].abs().rolling(5).mean()
rolling_vol_B = window_returns[ticker_B].abs().rolling(5).mean()
x0 = RA.abs() >= rolling_vol_A.iloc[-1]
y0 = RB.abs() >= rolling_vol_B.iloc[-1]
```

## 🧪 **Experimental Framework**

### **Testing Different Market Conditions**

The framework allows you to focus analysis on specific market conditions:

```python
# Filter data by date ranges to test specific periods
crisis_periods = {
    'covid_crash': ('2020-02-15', '2020-04-15'),
    'post_covid': ('2020-05-01', '2021-12-31'),  
    'recent_volatility': ('2022-01-01', '2023-12-31')
}

# Modify data loading to focus on specific periods
def load_period_data(self, file_path, start_date, end_date):
    # Implementation to filter by date range
    pass
```

### **Asset Pair Experiments**

Experiment with different asset combinations:

```python
# High correlation pairs (tech stocks)
tech_pairs = ['AAPL', 'MSFT', 'GOOG', 'NVDA']

# Cross-sector pairs (low correlation expected)
cross_sector = ['AAPL', 'CORN', 'DBA']  

# Volatility pairs (similar volatility profiles)
high_vol = ['TSLA', 'NVDA', 'NFLX']
```

### **Sensitivity Analysis Framework**

Built-in functions for parameter sensitivity testing:

```python
# Test multiple window sizes
for window in [10, 15, 20, 25, 30]:
    results = analyzer._perform_conditional_bell_analysis(window_size=window)
    # Analyze violation rate vs window size

# Test multiple threshold quantiles  
for quantile in [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]:
    results = analyzer._perform_conditional_bell_analysis(threshold_quantile=quantile)
    # Analyze violation rate vs threshold
```

## 📊 **Output Analysis Tools**

### **Violation Pattern Analysis**

```python
# Extract violation patterns for analysis
def analyze_violation_patterns(results):
    for method_name, method_results in results['conditional_bell_analysis'].items():
        for pair_name, pair_result in method_results.items():
            s1_series = pair_result['s1_series']
            violations = s1_series[s1_series['S1_value'].abs() > 2.0]
            
            # Analyze timing, frequency, magnitude of violations
            print(f"{pair_name}: {len(violations)} violations")
            print(f"Max violation: {s1_series['S1_value'].abs().max():.4f}")
            print(f"Violation rate: {len(violations)/len(s1_series)*100:.2f}%")
```

### **Comparative Analysis**

```python
# Compare CHSH vs S1 results
def compare_methods(results):
    chsh_max = max([r['chsh_value'] for method in results['chsh_analysis'].values() 
                    for measurement in method.values() 
                    for r in measurement.values()])
    
    s1_violations = sum([r['total_violations'] for method in results['conditional_bell_analysis'].values()
                        for r in method.values()])
    
    print(f"CHSH max value: {chsh_max:.4f}")
    print(f"S1 total violations: {s1_violations}")
```

## 🔍 **Research Questions to Explore**

### **Methodological Questions**
1. **Optimal window size**: How does violation detection change with different rolling window sizes?
2. **Threshold sensitivity**: Which quantile thresholds provide the most stable results?
3. **Aggregation method impact**: Do different price aggregation methods reveal different correlation structures?
4. **Observable definitions**: How do different binary observable choices affect violation patterns?

### **Market Structure Questions**
1. **Asset pair selection**: Which types of asset pairs show the most violations?
2. **Temporal patterns**: Do violations cluster around specific market events?
3. **Regime dependency**: How do violations relate to traditional volatility measures?
4. **Cross-market analysis**: Do violations appear across different market sectors?

### **Validation Questions**
1. **Statistical significance**: Are violations statistically significant or due to random chance?
2. **Robustness**: Do results hold across different time periods and market conditions?
3. **Comparison with traditional measures**: How do Bell violations relate to correlation, VIX, etc.?

## 📝 **Customization Guide**

### **Adding New Aggregation Methods**

```python
# In _create_bars_single_method(), add new methods:
elif method == 'YourMethod':
    # Your custom aggregation logic
    custom_price = your_aggregation_function(ticker_data)
    bars = pd.DataFrame({
        'Close': custom_price,
        'Volume': ticker_data['size'].resample(pandas_freq).sum()
    })
    bars['Returns'] = bars['Close'].pct_change()
```

### **Modifying Bell Test Logic**

```python
# In _perform_conditional_bell_analysis(), experiment with:

# Different S-value calculations
S1 = ab_00 + ab_01 + ab_10 - ab_11  # Current
S2 = ab_00 + ab_01 - ab_10 + ab_11  # Alternative
S3 = ab_00 - ab_01 + ab_10 + ab_11  # Alternative  
S4 = -ab_00 + ab_01 + ab_10 + ab_11 # Alternative

# Different violation criteria
violation = abs(S1) > 2.0           # Classical bound
violation = abs(S1) > 2.828         # Quantum bound  
violation = abs(S1) > custom_bound  # Your threshold
```

### **Custom Visualization**

```python
# Add custom plotting functions
def plot_custom_analysis(self, results):
    # Your custom visualization logic
    pass

# Call in _create_conditional_bell_visualizations()
self.plot_custom_analysis(results)
```

## 🎯 **Next Steps for Exploration**

1. **Parameter sweeps**: Systematically test different parameter combinations
2. **Statistical validation**: Implement bootstrap tests for violation significance  
3. **Comparative studies**: Test against traditional correlation measures
4. **Real-time implementation**: Adapt for live market data analysis
5. **Multi-timeframe analysis**: Combine results across different frequencies
6. **Machine learning integration**: Use violations as features for market prediction

## 📖 **References**

- **Zarifian, A., et al. (2025)**. "Using Bell violations as an indicator for financial crisis." *The Journal of Finance and Data Science*, 100164.
- **Bell, J.S. (1964)**. "On the Einstein Podolsky Rosen paradox." *Physics Physique Fizika*, 1(3), 195-200.
- **Clauser, J.F., et al. (1969)**. "Proposed experiment to test local hidden-variable theories." *Physical Review Letters*, 23(15), 880-884.
