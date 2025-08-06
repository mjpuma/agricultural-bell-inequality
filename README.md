# Bell Inequalities in Financial Markets: A Methodological Exploration

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Exploratory Research](https://img.shields.io/badge/Status-Exploratory-orange.svg)]()

## 🔬 **Exploring Bell Inequalities in Financial Data**

This repository provides a comprehensive framework for exploring **Bell inequalities** in financial market data, comparing traditional **CHSH** approaches with **conditional S1 Bell tests**. The enhanced version includes **focus group functionality** for targeted analysis of specific asset combinations, allowing extensive parameter experimentation to investigate regime-dependent correlations in financial markets.

**⚠️ Note**: This is exploratory research. The methodology is designed for experimentation and parameter tuning rather than definitive conclusions.

## 🎯 **NEW: Focus Group Analysis**

The enhanced framework allows you to **focus analysis on specific asset groups** while maintaining comprehensive coverage:

- **🔍 Priority Processing**: Focus assets tested first in all analyses
- **📊 Enhanced Reporting**: Focus assets highlighted with 🎯 markers throughout  
- **🎨 Visual Highlighting**: Red colors, thicker lines, special styling for focus assets
- **📈 Separate Statistics**: Focus group stats reported separately from overall results
- **⚛️ Bell Test Priority**: Focus pairs prioritized in both CHSH and S1 conditional Bell tests

### **Predefined Focus Groups**
```python
tech_pairs   = ['AAPL', 'MSFT', 'GOOG', 'NVDA']     # Technology leaders
high_vol     = ['TSLA', 'NVDA', 'NFLX']             # High volatility stocks  
cross_sector = ['AAPL', 'CORN', 'DBA']              # Cross-sector mix
commodities  = ['CORN', 'DBA', 'GLD', 'SLV']        # Commodities ETFs
financials   = ['JPM', 'BAC', 'GS', 'WFC']          # Financial sector
```

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
from enhanced_bell_analyzer import ComprehensiveBellAnalyzer

# Initialize with default tech focus group
analyzer = ComprehensiveBellAnalyzer()

# Or initialize with custom focus group
analyzer = ComprehensiveBellAnalyzer(focus_assets=['AAPL', 'TSLA', 'NVDA'])

# Run comprehensive analysis
results = analyzer.load_and_analyze_comprehensive(
    file_path='your_data.csv.gz',
    frequency='15min'  # Adjustable: '5min', '15min', '30min'
)
```

### **🎯 Focus Group Usage**

```python
# Method 1: Change default focus group at top of file
DEFAULT_FOCUS_SET = tech_pairs  # or high_vol, cross_sector, etc.

# Method 2: Use convenience functions
analyzer, results = run_comprehensive_analysis(focus_assets='tech_pairs')
analyzer, results = run_comprehensive_analysis(focus_assets=['AAPL', 'TSLA'])

# Method 3: Interactive selection
focus_assets = interactive_focus_selection()
analyzer, results = run_comprehensive_analysis(focus_assets=focus_assets)

# Method 4: Compare different focus groups
(analyzer1, results1), (analyzer2, results2) = compare_focus_groups(
    focus_group1=['AAPL', 'MSFT'], 
    focus_group2=['TSLA', 'NVDA']
)
```

### **Quick Start Examples**
```python
# Example 1: Tech focus analysis
analyzer, results = example_tech_focus()

# Example 2: High volatility analysis  
analyzer, results = example_high_vol_focus()

# Example 3: Custom focus analysis
analyzer, results = example_custom_focus()

# Example 4: Run with specific focus group
analyzer, results = run_with_focus_group(['AAPL', 'TSLA', 'NVDA'])
```

## ⚙️ **Key Parameters to Experiment With**

### **1. Focus Group Selection**

**Most Important Parameter**: Choose your focus group strategically

```python
# Modify at top of file
DEFAULT_FOCUS_SET = tech_pairs      # Focus on tech stocks
DEFAULT_FOCUS_SET = high_vol        # Focus on volatile stocks
DEFAULT_FOCUS_SET = ['AAPL', 'BTC', 'GOLD']  # Custom focus

# Or pass to functions
analyzer, results = run_comprehensive_analysis(
    focus_assets=['AAPL', 'MSFT', 'GOOG'],  # Your target assets
    frequency='15min'
)
```

**Focus Group Strategy Tips:**
- **High correlation pairs** (tech stocks): More likely to show violations
- **Cross-sector pairs**: Test for unexpected correlations  
- **Volatility pairs**: Similar volatility profiles may show regime-dependent behavior
- **Crisis-sensitive pairs**: Assets that move together during market stress

### **2. Time Window Parameters**

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

**Focus Group Impact**: Focus pairs tested first with detailed parameter exploration

### **3. Aggregation Methods**

The framework tests 6 different price aggregation methods. Focus assets get priority testing:

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

**Focus Group Enhancement**: 
- Focus assets processed first with detailed diagnostics
- Focus pair correlations reported separately
- Focus assets highlighted in all method comparisons

### **4. Binary Observable Definitions**

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

### **5. Regime Definition Variations**

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

## 🧪 **Enhanced Experimental Framework**

### **🎯 Focus Group Experiments**

**Primary Research Strategy**: Use focus groups to systematically test different hypotheses

```python
# Test 1: Do tech stocks show quantum-like correlations?
analyzer, results = run_comprehensive_analysis(focus_assets='tech_pairs')

# Test 2: Do high volatility stocks violate Bell inequalities?
analyzer, results = run_comprehensive_analysis(focus_assets='high_vol')

# Test 3: Do cross-sector pairs show unexpected correlations?
analyzer, results = run_comprehensive_analysis(focus_assets='cross_sector')

# Test 4: Custom hypothesis testing
custom_hypothesis = ['AAPL', 'BITCOIN', 'GOLD']  # Your theory here
analyzer, results = run_comprehensive_analysis(focus_assets=custom_hypothesis)
```

### **Focus Group Validation**

```python
# Validate your focus group selection
analyzer = ComprehensiveBellAnalyzer(focus_assets=['AAPL', 'MSFT'])
validation_results = analyzer.validate_focus_assets(min_data_quality=0.8)

# Check focus asset availability and quality
print("Focus Asset Validation:")
for method_name, validation in validation_results.items():
    print(f"  {method_name}: {validation['coverage']*100:.1f}% coverage")
    for asset, quality in validation['focus_quality'].items():
        print(f"    {asset}: {quality:.1%} data quality")
```

### **Testing Different Market Conditions**

Focus on specific periods where your focus assets might show interesting behavior:

```python
# Filter data by date ranges to test specific periods
crisis_periods = {
    'covid_crash': ('2020-02-15', '2020-04-15'),
    'post_covid': ('2020-05-01', '2021-12-31'),  
    'recent_volatility': ('2022-01-01', '2023-12-31')
}

# Test focus groups during different market regimes
for period_name, (start, end) in crisis_periods.items():
    print(f"\n🔍 Testing {period_name} period:")
    # Filter your data and run analysis
    analyzer, results = run_comprehensive_analysis(
        focus_assets=['AAPL', 'TSLA'], 
        # Add date filtering logic here
    )
```

### **Sensitivity Analysis Framework**

Enhanced for focus groups:

```python
# Test multiple parameters with focus group emphasis
def focus_sensitivity_analysis(focus_assets, param_ranges):
    results = {}
    
    for window in param_ranges['window_sizes']:
        for quantile in param_ranges['quantiles']:
            print(f"🎯 Testing focus {focus_assets} with window={window}, quantile={quantile}")
            analyzer = ComprehensiveBellAnalyzer(focus_assets=focus_assets)
            
            # Run with specific parameters
            test_results = analyzer._perform_conditional_bell_analysis(
                window_size=window, 
                threshold_quantile=quantile
            )
            
            # Extract focus-specific results
            focus_violations = sum(
                r['total_violations'] for method_results in test_results.values() 
                for r in method_results.values() 
                if r.get('is_focus_pair', False)
            )
            
            results[(window, quantile)] = focus_violations
    
    return results

# Usage
param_ranges = {
    'window_sizes': [10, 15, 20, 25, 30],
    'quantiles': [0.6, 0.7, 0.75, 0.8, 0.9]
}

sensitivity_results = focus_sensitivity_analysis(['AAPL', 'MSFT'], param_ranges)
```

## 📊 **Enhanced Output Analysis Tools**

### **Focus Group Violation Analysis**

```python
# Analyze violations with focus group emphasis
def analyze_focus_violations(results, focus_assets):
    print(f"🎯 FOCUS GROUP VIOLATION ANALYSIS")
    print(f"Focus Assets: {focus_assets}")
    
    focus_violations = 0
    total_violations = 0
    
    for method_name, method_results in results['conditional_bell_analysis'].items():
        method_focus_violations = 0
        method_total_violations = 0
        
        for pair_name, pair_result in method_results.items():
            violations = pair_result['total_violations']
            method_total_violations += violations
            
            if pair_result.get('is_focus_pair', False):
                method_focus_violations += violations
                print(f"  🎯 {pair_name}: {violations} violations")
        
        focus_violations += method_focus_violations
        total_violations += method_total_violations
        
        print(f"\n📊 {method_name} Summary:")
        print(f"  Focus violations: {method_focus_violations}")
        print(f"  Total violations: {method_total_violations}")
        print(f"  Focus contribution: {method_focus_violations/method_total_violations*100:.1f}%" if method_total_violations > 0 else "  No violations")
    
    print(f"\n🎯 OVERALL FOCUS SUMMARY:")
    print(f"  Focus group violations: {focus_violations}")
    print(f"  Total violations: {total_violations}")
    print(f"  Focus success rate: {focus_violations/total_violations*100:.1f}%" if total_violations > 0 else "  No violations found")

# Usage
analyze_focus_violations(results, analyzer.focus_assets)
```

### **Focus Group Comparison Analysis**

```python
# Compare multiple focus groups
def compare_focus_performance(focus_groups, file_path):
    comparison_results = {}
    
    for group_name, focus_assets in focus_groups.items():
        print(f"\n🔍 Testing {group_name}: {focus_assets}")
        analyzer, results = run_comprehensive_analysis(
            focus_assets=focus_assets,
            file_path=file_path
        )
        
        # Extract key metrics
        total_violations = sum(
            r['total_violations'] 
            for method_results in results['conditional_bell_analysis'].values()
            for r in method_results.values()
        )
        
        focus_violations = sum(
            r['total_violations'] 
            for method_results in results['conditional_bell_analysis'].values()
            for r in method_results.values()
            if r.get('is_focus_pair', False)
        )
        
        comparison_results[group_name] = {
            'focus_assets': focus_assets,
            'total_violations': total_violations,
            'focus_violations': focus_violations,
            'focus_success_rate': focus_violations/total_violations*100 if total_violations > 0 else 0
        }
    
    # Display comparison
    print(f"\n🏆 FOCUS GROUP COMPARISON:")
    for group_name, metrics in comparison_results.items():
        print(f"  {group_name}:")
        print(f"    Assets: {metrics['focus_assets']}")
        print(f"    Violations: {metrics['focus_violations']}/{metrics['total_violations']}")
        print(f"    Success Rate: {metrics['focus_success_rate']:.1f}%")
    
    return comparison_results

# Usage
test_groups = {
    'tech_leaders': ['AAPL', 'MSFT', 'GOOG'],
    'high_volatility': ['TSLA', 'NVDA', 'NFLX'],
    'cross_sector': ['AAPL', 'CORN', 'DBA'],
    'custom_theory': ['AAPL', 'BITCOIN', 'GOLD']  # Your hypothesis
}

comparison = compare_focus_performance(test_groups, 'your_data.csv.gz')
```

## 🔍 **Enhanced Research Questions**

### **🎯 Focus Group Methodology Questions**
1. **Optimal focus group size**: How many assets should be in a focus group for reliable results?
2. **Focus group composition**: Do similar assets (tech stocks) or diverse assets (cross-sector) show more violations?
3. **Focus vs. random sampling**: Do focus groups find more violations than random asset pairs?
4. **Focus group stability**: Do the same focus groups show violations across different time periods?

### **Asset-Specific Questions**  
1. **Tech stock quantum behavior**: Do technology stocks show more quantum-like correlations?
2. **Volatility-violation relationship**: Do high-volatility assets violate Bell inequalities more often?
3. **Cross-sector surprises**: Do unexpected cross-sector correlations violate classical bounds?
4. **Crisis behavior**: How do focus group violations change during market stress?

### **Enhanced Methodological Questions**
1. **Focus group parameter optimization**: What parameters work best for different asset types?
2. **Focus group validation**: How do you validate that a focus group is appropriate?
3. **Multi-focus analysis**: Can you analyze multiple focus groups simultaneously?
4. **Focus group evolution**: How should focus groups change over time?

## 📝 **Enhanced Customization Guide**

### **Adding Custom Focus Groups**

```python
# Define custom focus groups at the top of the file
my_crypto_focus = ['BTC', 'ETH', 'SOL']  # If you have crypto data
my_energy_focus = ['XOM', 'CVX', 'COP'] # Energy sector
my_hypothesis = ['AAPL', 'TSLA', 'GOLD', 'VIX']  # Your market theory

# Add to predefined options
predefined_groups = {
    'my_crypto': my_crypto_focus,
    'my_energy': my_energy_focus,
    'my_hypothesis': my_hypothesis
}

# Use in analysis
analyzer, results = run_comprehensive_analysis(focus_assets='my_hypothesis')
```

### **Custom Focus Group Validation**

```python
# Add custom validation logic
def validate_custom_focus(focus_assets, tick_data):
    """Validate that focus assets are suitable for analysis"""
    
    validation = {}
    
    for asset in focus_assets:
        asset_data = tick_data[tick_data['ticker'] == asset]
        
        validation[asset] = {
            'available': len(asset_data) > 0,
            'tick_count': len(asset_data),
            'data_span_hours': (asset_data['datetime'].max() - asset_data['datetime'].min()).total_seconds() / 3600,
            'price_volatility': asset_data['price'].std() / asset_data['price'].mean()
        }
    
    # Check focus group quality
    available_assets = [a for a in focus_assets if validation[a]['available']]
    avg_volatility = np.mean([validation[a]['price_volatility'] for a in available_assets])
    
    print(f"🎯 FOCUS GROUP VALIDATION:")
    print(f"  Assets requested: {focus_assets}")
    print(f"  Assets available: {available_assets}")
    print(f"  Coverage: {len(available_assets)}/{len(focus_assets)} ({len(available_assets)/len(focus_assets)*100:.1f}%)")
    print(f"  Average volatility: {avg_volatility:.4f}")
    
    return validation
```

### **Enhanced Visualization for Focus Groups**

```python
# Add focus-specific plotting functions
def plot_focus_group_performance(self, results):
    """Plot focus group specific results"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Focus vs non-focus violation rates
    self._plot_focus_vs_others_violations(ax1, results)
    
    # Plot 2: Focus group correlation heatmap
    self._plot_focus_correlation_heatmap(ax2, results)
    
    # Plot 3: Focus group S1 time series (all pairs)
    self._plot_focus_s1_timeseries(ax3, results)
    
    # Plot 4: Focus group parameter sensitivity
    self._plot_focus_parameter_sensitivity(ax4, results)
    
    plt.suptitle(f'🎯 Focus Group Analysis: {", ".join(self.focus_assets)}', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

# Call in visualization section
self.plot_focus_group_performance(results)
```

## 🎯 **Focus Group Best Practices**

### **1. Strategic Focus Group Selection**
- **Start specific**: Begin with 2-3 highly correlated assets
- **Test hypotheses**: Choose assets based on market theories
- **Consider liquidity**: Ensure focus assets have sufficient tick data
- **Mixed approaches**: Test both similar and diverse asset combinations

### **2. Parameter Optimization for Focus Groups**
- **Shorter windows**: Focus groups may need more sensitive detection
- **Higher thresholds**: Well-known correlations may need extreme regime detection
- **Method comparison**: Different aggregation methods may suit different focus groups

### **3. Validation and Robustness**
- **Multiple time periods**: Test focus groups across different market conditions
- **Parameter sensitivity**: Ensure results aren't due to specific parameter choices  
- **Comparison testing**: Always compare focus group results with random pairs
- **Statistical significance**: Test whether focus group violations are statistically meaningful

### **4. Research Documentation**
- **Document hypotheses**: Clearly state why you chose specific focus groups
- **Record parameters**: Track which parameters work best for each focus group
- **Compare results**: Maintain comparison tables across different focus groups
- **Validate findings**: Cross-validate results with traditional correlation measures

## 🎯 **Next Steps for Focus Group Exploration**

1. **📊 Systematic focus group testing**: Test all predefined focus groups with your data
2. **⚙️ Parameter optimization**: Find optimal parameters for each focus group type  
3. **📈 Time series analysis**: Track focus group violation patterns over time
4. **🔬 Statistical validation**: Implement bootstrap tests for focus group significance
5. **🎯 Custom hypothesis testing**: Design focus groups based on your market theories
6. **🤖 Automated focus selection**: Develop algorithms to automatically identify optimal focus groups
7. **📱 Real-time monitoring**: Adapt focus group analysis for live market surveillance

## 📖 **References**

- **Zarifian, A., et al. (2025)**. "Using Bell violations as an indicator for financial crisis." *The Journal of Finance and Data Science*, 100164.
- **Bell, J.S. (1964)**. "On the Einstein Podolsky Rosen paradox." *Physics Physique Fizika*, 1(3), 195-200.
- **Clauser, J.F., et al. (1969)**. "Proposed experiment to test local hidden-variable theories." *Physical Review Letters*, 23(15), 880-884.
