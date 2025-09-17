# Design Document

## Overview

The Agricultural Cross-Sector Analysis system extends the existing Bell inequality framework to focus specifically on agricultural companies and their operational dependencies across three tiers. The system will implement a tiered analysis approach focusing on fast transmission mechanisms (0-3 months) between agricultural companies and their dependencies in Energy, Transportation, and Chemicals (Tier 1), with additional analysis of Finance/Equipment (Tier 2) and Policy-linked sectors (Tier 3).

The design leverages the proven S1 conditional Bell inequality methodology from Zarifian et al. (2025) while introducing agricultural-specific enhancements including sector-specific thresholds, crisis period analysis, and cross-sectoral transmission detection.

## Architecture

### System Components

```
Agricultural Cross-Sector Analysis System
├── Data Layer
│   ├── Agricultural Universe Manager
│   ├── Cross-Sector Data Handler
│   └── Crisis Period Data Manager
├── Analysis Engine
│   ├── S1 Bell Inequality Calculator (Enhanced)
│   ├── Cross-Sector Transmission Detector
│   └── Agricultural Crisis Analyzer
├── Visualization Layer
│   ├── Sector Heatmap Generator
│   ├── Transmission Timeline Visualizer
│   └── Crisis Comparison Charts
└── Results Management
    ├── Publication-Ready Report Generator
    ├── Statistical Validation Suite
    └── Export Manager (Excel/CSV/JSON)
```

### Integration with Existing System

The new system builds upon the existing `BellInequalityAnalyzer` and `FoodSystemsAnalyzer` classes, extending them with:

1. **Agricultural Universe Management**: Comprehensive 60+ company database with tier classifications
2. **Cross-Sector Pairing Logic**: Intelligent pairing based on operational dependencies
3. **Enhanced S1 Implementation**: Mathematically accurate implementation following Zarifian et al. (2025)
4. **Crisis Period Specialization**: Focused analysis on agricultural crisis periods

## Components and Interfaces

### 1. Agricultural Universe Manager

**Purpose**: Manage the comprehensive agricultural company database with tier classifications and operational dependencies.

**Interface**:
```python
class AgriculturalUniverseManager:
    def __init__(self):
        self.companies = {}  # Company metadata
        self.tiers = {}      # Tier classifications
        self.dependencies = {}  # Cross-sector relationships
    
    def load_agricultural_universe(self) -> Dict[str, CompanyInfo]
    def classify_by_tier(self, tier: int) -> List[str]
    def get_cross_sector_pairs(self, tier: int) -> List[Tuple[str, str]]
    def get_transmission_mechanisms(self, pair: Tuple[str, str]) -> TransmissionInfo
```

**Key Features**:
- **Company Classification**: Large-Cap (>$10B), Mid-Cap ($2B-$10B), Small-Cap ($250M-$2B)
- **Exposure Levels**: Primary (direct agricultural), Secondary (significant exposure), Tertiary (indirect)
- **Sector Mapping**: Equipment, Seeds/Crop Protection, Trading/Processing, Fertilizers, Food Processing, etc.
- **Dependency Tracking**: Operational relationships between agricultural and cross-sector companies

### 2. Enhanced S1 Bell Inequality Calculator

**Purpose**: Implement mathematically accurate S1 calculation following Zarifian et al. (2025) specification.

**Interface**:
```python
class EnhancedS1Calculator:
    def __init__(self, window_size: int = 20, threshold_method: str = 'quantile'):
        self.window_size = window_size
        self.threshold_method = threshold_method
    
    def calculate_daily_returns(self, prices: pd.DataFrame) -> pd.DataFrame
    def compute_binary_indicators(self, returns: pd.DataFrame, thresholds: Dict) -> pd.DataFrame
    def calculate_conditional_expectations(self, indicators: pd.DataFrame, signs: pd.DataFrame) -> Dict
    def compute_s1_value(self, expectations: Dict) -> float
    def detect_violations(self, s1_values: List[float]) -> ViolationResults
```

**Mathematical Implementation**:
- **Daily Returns**: `Ri,t = (Pi,t - Pi,t-1) / Pi,t-1`
- **Binary Indicators**: `I{|RA,t| ≥ rA}` for strong vs weak movements
- **Sign Outcomes**: `Sign(Ri,t) = +1 if Ri,t ≥ 0, -1 if Ri,t < 0`
- **Conditional Expectations**: `⟨ab⟩xy = Σ[sign(RA,t)sign(RB,t)I{conditions}] / Σ[I{conditions}]`
- **S1 Formula**: `S1 = ⟨ab⟩00 + ⟨ab⟩01 + ⟨ab⟩10 - ⟨ab⟩11`

### 3. Cross-Sector Transmission Detector

**Purpose**: Detect and analyze transmission mechanisms between agricultural and cross-sector companies.

**Interface**:
```python
class CrossSectorTransmissionDetector:
    def __init__(self, transmission_window: int = 90):  # 0-3 months
        self.transmission_window = transmission_window
    
    def detect_energy_transmission(self, energy_assets: List[str], ag_assets: List[str]) -> TransmissionResults
    def detect_transport_transmission(self, transport_assets: List[str], ag_assets: List[str]) -> TransmissionResults
    def detect_chemical_transmission(self, chemical_assets: List[str], ag_assets: List[str]) -> TransmissionResults
    def analyze_transmission_speed(self, pair: Tuple[str, str]) -> TransmissionSpeed
```

**Transmission Mechanisms**:
- **Energy → Agriculture**: Natural gas prices → fertilizer costs → crop production costs
- **Transportation → Agriculture**: Rail/shipping bottlenecks → commodity logistics → price volatility
- **Chemicals → Agriculture**: Chemical input costs → pesticide/fertilizer prices → farming costs
- **Finance → Agriculture**: Credit availability → farming operations → commodity prices

### 4. Agricultural Crisis Analyzer

**Purpose**: Specialized analysis for agricultural crisis periods with enhanced sensitivity.

**Interface**:
```python
class AgriculturalCrisisAnalyzer:
    def __init__(self, crisis_periods: Dict[str, CrisisPeriod]):
        self.crisis_periods = crisis_periods
    
    def analyze_covid_food_disruption(self) -> CrisisResults
    def analyze_ukraine_war_crisis(self) -> CrisisResults
    def analyze_2008_food_crisis(self) -> CrisisResults
    def analyze_2012_drought(self) -> CrisisResults
    def compare_crisis_periods(self, periods: List[str]) -> ComparisonResults
```

**Crisis-Specific Parameters**:
- **Window Size**: 15 periods (shorter for crisis analysis)
- **Threshold Quantile**: 0.8 (higher threshold for extreme events)
- **Focus Areas**: Supply chain disruptions, panic buying, export restrictions
- **Expected Violations**: 40-60% during crisis periods

## Data Models

### Company Information Model
```python
@dataclass
class CompanyInfo:
    ticker: str
    name: str
    sector: str  # Equipment, Fertilizers, Trading/Processing, etc.
    exposure: str  # Primary, Secondary, Tertiary
    market_cap_category: str  # Large-Cap, Mid-Cap, Small-Cap
    market_cap_approx: str
    operational_dependencies: List[str]
    geographic_exposure: List[str]
```

### Transmission Mechanism Model
```python
@dataclass
class TransmissionMechanism:
    source_sector: str
    target_sector: str
    mechanism: str  # e.g., "Natural gas prices → fertilizer costs"
    expected_lag: int  # Days
    strength: str  # Strong, Moderate, Weak
    crisis_amplification: bool
```

### Analysis Results Model
```python
@dataclass
class CrossSectorResults:
    pair: Tuple[str, str]
    s1_values: List[float]
    violation_rate: float
    transmission_detected: bool
    transmission_lag: Optional[int]
    crisis_amplification: Optional[float]
    statistical_significance: float
    bootstrap_confidence: Tuple[float, float]
```

### Crisis Period Model
```python
@dataclass
class CrisisPeriod:
    name: str
    start_date: str
    end_date: str
    description: str
    affected_sectors: List[str]
    expected_violation_rate: float
    key_transmission_mechanisms: List[str]
```

## Error Handling

### Data Quality Management
- **Missing Data Handling**: Set `⟨ab⟩xy = 0` if no valid observations for regime
- **Insufficient Data**: Minimum 100 observations per asset requirement
- **Data Validation**: Automatic detection of data gaps, outliers, and inconsistencies
- **Fallback Mechanisms**: Alternative data sources and interpolation methods

### Statistical Robustness
- **Bootstrap Validation**: 1000+ resamples for confidence intervals
- **Multiple Testing Correction**: Bonferroni correction for multiple pair analysis
- **Sensitivity Analysis**: Parameter robustness testing across window sizes and thresholds
- **Cross-Validation**: Time-series cross-validation for model stability

### System Reliability
- **Error Recovery**: Graceful handling of API failures and data source issues
- **Progress Tracking**: Detailed logging and progress indicators for long-running analyses
- **Memory Management**: Efficient handling of large datasets (60+ companies × multiple years)
- **Parallel Processing**: Multi-threading for independent pair analysis

## Testing Strategy

### Unit Testing
- **Mathematical Accuracy**: Verify S1 calculation against known test cases
- **Data Processing**: Test return calculations, threshold determination, regime classification
- **Statistical Functions**: Validate bootstrap procedures, confidence intervals, significance tests
- **Edge Cases**: Handle zero returns, missing data, single-observation regimes

### Integration Testing
- **End-to-End Workflows**: Complete analysis pipeline from data loading to report generation
- **Cross-Component Integration**: Data flow between universe manager, calculator, and visualizer
- **Crisis Period Analysis**: Verify crisis-specific parameter adjustments and results
- **Export Functionality**: Test all output formats (Excel, CSV, JSON, PNG)

### Performance Testing
- **Scalability**: Performance with full 60+ company universe
- **Memory Usage**: Monitor memory consumption during large-scale analysis
- **Processing Speed**: Benchmark analysis time for different dataset sizes
- **Concurrent Analysis**: Test parallel processing of multiple asset pairs

### Validation Testing
- **Historical Validation**: Reproduce known results from existing food systems analysis
- **Cross-Reference**: Compare results with existing `BellInequalityAnalyzer` on overlapping assets
- **Statistical Validation**: Verify violation rates match expected ranges for different sectors
- **Publication Standards**: Ensure results meet Science journal statistical requirements

### Agricultural Domain Testing
- **Sector Logic**: Verify agricultural sector classifications and dependencies
- **Transmission Mechanisms**: Test detection of known transmission patterns (e.g., energy → fertilizer)
- **Crisis Response**: Validate enhanced violation rates during known agricultural crises
- **Seasonal Effects**: Test for agricultural seasonal patterns in quantum correlations

## Implementation Considerations

### Performance Optimization
- **Vectorized Calculations**: Use NumPy/Pandas vectorized operations for S1 calculations
- **Caching Strategy**: Cache intermediate results for repeated analysis
- **Parallel Processing**: Multi-process analysis for independent asset pairs
- **Memory Efficiency**: Stream processing for large datasets

### Extensibility
- **Modular Design**: Easy addition of new sectors, crisis periods, or transmission mechanisms
- **Configuration-Driven**: External configuration files for asset universes and parameters
- **Plugin Architecture**: Support for custom analysis modules and visualizations
- **API Integration**: Ready for WDRS data source integration

### Scientific Rigor
- **Reproducibility**: Seed-based random number generation for consistent bootstrap results
- **Documentation**: Comprehensive mathematical documentation and methodology references
- **Version Control**: Track analysis versions and parameter changes
- **Audit Trail**: Complete logging of analysis parameters and data sources

### Publication Readiness
- **Statistical Standards**: All results meet p < 0.001 significance requirements
- **Professional Visualizations**: Publication-quality figures with proper labeling and legends
- **Comprehensive Reports**: Detailed methodology, results, and interpretation sections
- **Data Availability**: Export capabilities for supplementary materials and replication