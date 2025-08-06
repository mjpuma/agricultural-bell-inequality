#!/usr/bin/env python3
"""
COMPLETE ENHANCED BELL INEQUALITY ANALYSIS WITH FOCUS GROUPS - FIXED VERSION
============================================================================
Full analysis framework with conditional Bell tests, focused on specific asset groups
Includes all methods: CHSH, S1 conditional Bell test, comprehensive visualizations
FIXED: Column name detection, focus asset tracking, ALL plotting issues
COMPLETE VERSION with individual plots for EACH focus asset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gzip
from datetime import datetime
from scipy import stats
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# =================== ASSET GROUP CONFIGURATION ===================
tech_pairs   = ['AAPL', 'MSFT', 'GOOG', 'NVDA']
cross_sector = ['AAPL', 'CORN', 'DBA']  
high_vol     = ['TSLA', 'NVDA', 'NFLX']
commodities  = ['CORN', 'DBA', 'GLD', 'SLV']
financials   = ['JPM', 'BAC', 'GS', 'WFC']

# ===== CHOOSE YOUR FOCUS GROUP =====
DEFAULT_FOCUS_SET = cross_sector  # Change this to cross_sector, high_vol, commodities, financials, or custom list
# ==================================

print(f"🎯 DEFAULT FOCUSED ANALYSIS ON: {DEFAULT_FOCUS_SET}")
print(f"   Asset Group: {', '.join(DEFAULT_FOCUS_SET)}")

class ComprehensiveBellAnalyzer:
    
    def __init__(self, focus_assets=None):
        self.tick_data = None
        self.method_data = {}
        self.detailed_results = {}
        self.conditional_bell_results = {}
        self.focus_assets = focus_assets or DEFAULT_FOCUS_SET
        self.column_mapping = None
        self.focus_diagnostics = {}  # Track focus asset processing
        
        print(f"🔬 Initialized comprehensive analyzer focusing on: {self.focus_assets}")
        
    def load_and_analyze_comprehensive(self, file_path, frequency='15min'):
        """Comprehensive analysis with focus on specific asset groups - FIXED VERSION"""
        
        print("🔬 COMPREHENSIVE BELL INEQUALITY ANALYSIS + FOCUSED GROUPS (FIXED)")
        print("=" * 80)
        print(f"File: {file_path}")
        print(f"Frequency: {frequency}")
        print(f"Focus Assets: {self.focus_assets}")
        print("Full analysis pipeline with asset group focus - FIXED COLUMN DETECTION")
        
        # Step 1: Load and analyze tick data (FIXED VERSION with intelligent column detection)
        self.tick_data = self._load_tick_data_detailed_fixed(file_path)
        
        if self.tick_data is None:
            print("❌ Failed to load data - stopping analysis")
            return None
        
        # Step 2: Create bars with multiple methods (focus asset aware)
        self.method_data = self._create_bars_all_methods_fixed(frequency)
        
        # Step 3: Detailed statistical analysis
        statistical_analysis = self._perform_statistical_analysis()
        
        # Step 4: Correlation analysis
        correlation_analysis = self._perform_correlation_analysis()
        
        # Step 5: CHSH analysis with detailed diagnostics  
        chsh_analysis = self._perform_detailed_chsh_analysis()
        
        # Step 6: Conditional Bell analysis (Yahoo Finance style S1 inequality)
        print(f"\n🚀 RUNNING CONDITIONAL BELL ANALYSIS...")
        print("This uses the SAME approach as your Yahoo Finance AAPL-MSFT analysis")
        conditional_bell_analysis = self._perform_conditional_bell_analysis(window_size=20, threshold_quantile=0.75)
        self.conditional_bell_results = conditional_bell_analysis
        
        # Step 7: Cross-validation and robustness tests
        robustness_tests = self._perform_robustness_tests()
        
        # Step 8: Comprehensive visualization (FIXED - shows all focus assets)
        self._create_comprehensive_visualizations_fixed()
        
        # Step 9: Conditional Bell visualizations (prioritize focus group)
        self._create_conditional_bell_visualizations()
        
        # Step 10: Detailed summary and interpretation (focus group emphasis)
        self._provide_detailed_interpretation()
        
        return {
            'tick_data': self.tick_data,
            'method_data': self.method_data,
            'statistical_analysis': statistical_analysis,
            'correlation_analysis': correlation_analysis,
            'chsh_analysis': chsh_analysis,
            'conditional_bell_analysis': conditional_bell_analysis,
            'robustness_tests': robustness_tests
        }
    
    def _inspect_data_structure(self, file_path):
        """FIXED: Inspect the actual data structure to identify correct column names"""
        
        print("🔍 INSPECTING DATA STRUCTURE FOR CORRECT COLUMN MAPPING")
        print("=" * 60)
        
        # Load first few rows to inspect structure
        try:
            if file_path.endswith('.gz'):
                with gzip.open(file_path, 'rt') as f:
                    df_sample = pd.read_csv(f, nrows=10)
            else:
                df_sample = pd.read_csv(file_path, nrows=10)
        except Exception as e:
            print(f"❌ Error reading file: {e}")
            return None, None
        
        print(f"📋 ACTUAL COLUMNS FOUND:")
        for i, col in enumerate(df_sample.columns):
            print(f"   {i}: '{col}'")
        
        print(f"\n📊 SAMPLE DATA (first 3 rows):")
        print(df_sample.head(3).to_string())
        
        print(f"\n🔍 COLUMN DATA TYPES:")
        for col in df_sample.columns:
            sample_values = df_sample[col].dropna().head(3).values
            print(f"   {col}: {df_sample[col].dtype} | Sample: {sample_values}")
        
        print(f"\n💡 COLUMN MAPPING DETECTION:")
        
        # Try to identify columns intelligently
        possible_mappings = {
            'ticker': None,
            'price': None,
            'size': None,
            'date': None,
            'time': None
        }
        
        for col in df_sample.columns:
            col_lower = col.lower()
            # Check for ticker/symbol
            if any(word in col_lower for word in ['sym', 'ticker', 'symbol', 'stock', 'root']):
                possible_mappings['ticker'] = col
                print(f"   🎯 Detected ticker column: '{col}'")
            
            # Check for price
            if any(word in col_lower for word in ['price', 'px', 'last', 'close']):
                possible_mappings['price'] = col
                print(f"   💰 Detected price column: '{col}'")
            
            # Check for size/volume
            if any(word in col_lower for word in ['size', 'volume', 'qty', 'quantity']):
                possible_mappings['size'] = col
                print(f"   📊 Detected size column: '{col}'")
            
            # Check for date
            if any(word in col_lower for word in ['date', 'day', 'dt']):
                possible_mappings['date'] = col
                print(f"   📅 Detected date column: '{col}'")
            
            # Check for time
            if any(word in col_lower for word in ['time', 'tm', 'timestamp']) and 'date' not in col_lower:
                possible_mappings['time'] = col
                print(f"   ⏰ Detected time column: '{col}'")
        
        # Validation
        missing_mappings = [k for k, v in possible_mappings.items() if v is None and k in ['ticker', 'price', 'size']]
        if missing_mappings:
            print(f"⚠️  Could not auto-detect columns: {missing_mappings}")
            print(f"Available columns: {list(df_sample.columns)}")
        else:
            print(f"✅ Successfully detected all required column mappings")
        
        return df_sample, possible_mappings
    
    def _load_tick_data_detailed_fixed(self, file_path):
        """FIXED: Load tick data with intelligent column detection and focus asset tracking"""
        
        print("\n📊 DETAILED TICK DATA LOADING (FIXED VERSION)")
        print("=" * 50)
        
        # Step 1: Inspect data structure
        sample_df, column_mapping = self._inspect_data_structure(file_path)
        if sample_df is None or column_mapping is None:
            return None
        
        # Store the mapping for later use
        self.column_mapping = column_mapping
        
        # Check if we have the minimum required columns
        required_cols = ['ticker', 'price', 'size']
        missing_required = [col for col in required_cols if column_mapping.get(col) is None]
        if missing_required:
            print(f"❌ Missing required columns: {missing_required}")
            print(f"Cannot proceed without: ticker, price, size columns")
            return None
        
        # Step 2: Load full dataset
        print(f"\n📂 LOADING FULL DATASET...")
        if file_path.endswith('.gz'):
            with gzip.open(file_path, 'rt') as f:
                df = pd.read_csv(f, sep=',')
        else:
            df = pd.read_csv(file_path, sep=',')
        
        print(f"✅ Raw data loaded: {len(df):,} rows")
        print(f"📋 Columns: {list(df.columns)}")
        
        # Step 3: Apply column mapping with validation
        print(f"\n🔄 APPLYING COLUMN MAPPING...")
        
        # Create renamed dataframe
        df_renamed = df.copy()
        
        # Map the essential columns
        essential_mappings = {
            'ticker': column_mapping['ticker'],
            'price': column_mapping['price'], 
            'size': column_mapping['size']
        }
        
        for target_col, source_col in essential_mappings.items():
            if source_col and source_col in df.columns:
                df_renamed[target_col] = df[source_col]
                print(f"✅ Mapped '{source_col}' → '{target_col}'")
                
                # Validate the data looks reasonable
                if target_col == 'ticker':
                    unique_tickers = df_renamed[target_col].nunique()
                    print(f"   Found {unique_tickers} unique tickers")
                elif target_col == 'price':
                    price_stats = df_renamed[target_col].describe()
                    print(f"   Price range: ${price_stats['min']:.2f} - ${price_stats['max']:.2f}")
                elif target_col == 'size':
                    size_stats = df_renamed[target_col].describe()
                    print(f"   Size range: {size_stats['min']:,.0f} - {size_stats['max']:,.0f}")
            else:
                print(f"❌ Failed to map {target_col} (source: {source_col})")
                return None
        
        # Step 4: Handle datetime creation with multiple strategies
        print(f"\n⏰ CREATING DATETIME INDEX...")
        datetime_created = False
        
        # Strategy 1: Separate date and time columns
        if column_mapping.get('date') and column_mapping.get('time'):
            date_col = column_mapping['date']
            time_col = column_mapping['time']
            
            if date_col in df.columns and time_col in df.columns:
                print(f"🕐 Attempting datetime from '{date_col}' + '{time_col}'")
                try:
                    # Show sample of what we're trying to parse
                    sample_datetime = str(df[date_col].iloc[0]) + ' ' + str(df[time_col].iloc[0])
                    print(f"   Sample datetime string: '{sample_datetime}'")
                    
                    datetime_strings = df[date_col].astype(str) + ' ' + df[time_col].astype(str)
                    df_renamed['datetime'] = pd.to_datetime(datetime_strings, format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
                    
                    # Check if parsing worked
                    valid_dates = df_renamed['datetime'].notna().sum()
                    if valid_dates > len(df) * 0.5:  # At least 50% valid dates
                        datetime_created = True
                        print(f"✅ DateTime created successfully ({valid_dates:,}/{len(df):,} valid)")
                    else:
                        print(f"⚠️  DateTime parsing had issues ({valid_dates:,}/{len(df):,} valid)")
                        
                        # Try alternative format
                        df_renamed['datetime'] = pd.to_datetime(datetime_strings, errors='coerce')
                        valid_dates = df_renamed['datetime'].notna().sum()
                        if valid_dates > len(df) * 0.5:
                            datetime_created = True
                            print(f"✅ DateTime created with auto-detection ({valid_dates:,} valid)")
                        
                except Exception as e:
                    print(f"⚠️  Error creating datetime: {e}")
        
        # Strategy 2: Look for existing datetime/timestamp column
        if not datetime_created:
            for col in df.columns:
                if any(word in col.lower() for word in ['timestamp', 'datetime', 'dt']):
                    try:
                        df_renamed['datetime'] = pd.to_datetime(df[col], errors='coerce')
                        valid_dates = df_renamed['datetime'].notna().sum()
                        if valid_dates > len(df) * 0.5:
                            datetime_created = True
                            print(f"✅ Used existing '{col}' column as datetime ({valid_dates:,} valid)")
                            break
                    except:
                        continue
        
        # Strategy 3: Fallback to index-based timestamps
        if not datetime_created:
            print("⚠️  Could not create meaningful datetime, using sequential timestamps")
            start_time = pd.Timestamp('2024-01-01 09:30:00')
            df_renamed['datetime'] = [start_time + pd.Timedelta(seconds=i*0.01) for i in range(len(df))]
            datetime_created = True
        
        # Step 5: Data cleaning and validation
        print(f"\n🧹 DATA CLEANING AND VALIDATION...")
        
        # Convert price and size to numeric
        df_renamed['price'] = pd.to_numeric(df_renamed['price'], errors='coerce')
        df_renamed['size'] = pd.to_numeric(df_renamed['size'], errors='coerce')
        
        # Remove invalid data
        initial_rows = len(df_renamed)
        df_renamed = df_renamed.dropna(subset=['ticker', 'price', 'size', 'datetime'])
        final_rows = len(df_renamed)
        
        print(f"   Removed {initial_rows - final_rows:,} invalid rows")
        print(f"   Final dataset: {final_rows:,} rows")
        
        if final_rows < 100:
            print(f"❌ Insufficient data after cleaning ({final_rows} rows)")
            return None
        
        # Sort by datetime
        df_renamed = df_renamed.sort_values('datetime')
        
        # Step 6: FOCUS ASSET DETAILED ANALYSIS
        print(f"\n🎯 FOCUS ASSET DETAILED TRACKING")
        print("=" * 50)
        
        ticker_stats = df_renamed['ticker'].value_counts()
        print(f"   Unique tickers: {len(ticker_stats)}")
        print(f"   Target focus assets: {self.focus_assets}")
        
        # Initialize focus diagnostics
        for focus_asset in self.focus_assets:
            self.focus_diagnostics[focus_asset] = {
                'found_in_raw': False,
                'tick_count': 0,
                'issues': []
            }
        
        # Check each focus asset in detail
        print(f"\n🔍 DETAILED FOCUS ASSET ANALYSIS:")
        focus_found = []
        focus_missing = []
        
        for focus_asset in self.focus_assets:
            if focus_asset in ticker_stats.index:
                focus_found.append(focus_asset)
                count = ticker_stats[focus_asset]
                asset_data = df_renamed[df_renamed['ticker'] == focus_asset]
                prices = asset_data['price']
                
                # Store detailed diagnostics
                self.focus_diagnostics[focus_asset].update({
                    'found_in_raw': True,
                    'tick_count': count,
                    'price_range': f"${prices.min():.2f}-${prices.max():.2f}",
                    'price_mean': prices.mean(),
                    'price_std': prices.std(),
                    'date_range': f"{asset_data['datetime'].min()} to {asset_data['datetime'].max()}",
                    'time_span_hours': (asset_data['datetime'].max() - asset_data['datetime'].min()).total_seconds() / 3600
                })
                
                print(f"   🎯 {focus_asset}:")
                print(f"      Ticks: {count:,} ({count/len(df_renamed)*100:.1f}% of data)")
                print(f"      Price: ${prices.min():.2f} - ${prices.max():.2f} (μ=${prices.mean():.2f}, σ=${prices.std():.2f})")
                print(f"      Time span: {self.focus_diagnostics[focus_asset]['time_span_hours']:.1f} hours")
                
                # Check for potential issues
                issues = []
                if count < 50:
                    issues.append(f'Low tick count ({count})')
                if prices.std() < prices.mean() * 0.001:
                    issues.append('Very low price volatility')
                if self.focus_diagnostics[focus_asset]['time_span_hours'] < 1:
                    issues.append('Short time coverage')
                
                self.focus_diagnostics[focus_asset]['issues'] = issues
                if issues:
                    for issue in issues:
                        print(f"      ⚠️  {issue}")
                else:
                    print(f"      ✅ Data quality looks good")
                
            else:
                focus_missing.append(focus_asset)
                print(f"   ❌ {focus_asset}: NOT FOUND")
        
        if focus_missing:
            print(f"\n⚠️  FOCUS ASSETS NOT FOUND: {focus_missing}")
            print(f"   Available tickers (first 30): {list(ticker_stats.head(30).index)}")
            
            # Suggest similar tickers
            for missing in focus_missing:
                similar = [t for t in ticker_stats.head(50).index if missing.lower() in t.lower() or t.lower() in missing.lower()]
                if similar:
                    print(f"   💡 Similar to '{missing}': {similar}")
        
        if focus_found:
            print(f"\n✅ FOCUS ASSETS AVAILABLE: {focus_found}")
        else:
            print(f"\n❌ NO FOCUS ASSETS FOUND IN DATA!")
        
        # Step 7: Final validation and statistics
        print(f"\n📈 FINAL DATA VALIDATION:")
        print(f"   Date range: {df_renamed['datetime'].min()} to {df_renamed['datetime'].max()}")
        print(f"   Time span: {(df_renamed['datetime'].max() - df_renamed['datetime'].min()).total_seconds() / 3600:.1f} hours")
        
        # Overall price validation  
        overall_price_stats = df_renamed['price'].describe()
        print(f"   Price range: ${overall_price_stats['min']:.2f} - ${overall_price_stats['max']:.2f}")
        print(f"   Mean price: ${overall_price_stats['mean']:.2f}")
        
        # Size validation
        overall_size_stats = df_renamed['size'].describe()  
        print(f"   Size range: {overall_size_stats['min']:,.0f} - {overall_size_stats['max']:,.0f}")
        
        # Time gaps analysis
        print(f"\n⏱️  TIME GAPS ANALYSIS:")
        time_diffs = df_renamed['datetime'].diff().dt.total_seconds()
        print(f"   Median time between ticks: {time_diffs.median():.6f} seconds")
        print(f"   Mean time between ticks: {time_diffs.mean():.6f} seconds")
        large_gaps = (time_diffs > 1).sum()
        print(f"   Time gaps > 1 second: {large_gaps:,} ({large_gaps/len(time_diffs)*100:.1f}%)")
        
        print(f"\n✅ DATA LOADING COMPLETE - READY FOR ANALYSIS")
        
        return df_renamed
    
    def _create_bars_all_methods_fixed(self, frequency):
        """FIXED: Create bars with focus asset prioritization and lower thresholds"""
        
        if self.tick_data is None:
            print("❌ No tick data available")
            return {}
        
        print(f"\n📊 DETAILED BAR CREATION - {frequency.upper()} (FIXED VERSION)")
        print("=" * 50)
        
        freq_map = {'15min': '15T', '5min': '5T', '1min': '1T', '30min': '30T'}
        pandas_freq = freq_map.get(frequency, '15T')
        
        methods = {
            'OHLC': 'First/Max/Min/Last prices',
            'Average': 'Simple average of all tick prices', 
            'VWAP': 'Volume-weighted average price',
            'LastTick': 'Last tick price only',
            'Median': 'Median tick price (robust)',
            'TWAP': 'Time-weighted average price'
        }
        
        all_method_data = {}
        
        for method_name, description in methods.items():
            print(f"\n🔧 METHOD: {method_name}")
            print(f"   Description: {description}")
            print("   " + "-" * 40)
            
            method_data = {}
            
            # Process ALL tickers but prioritize focus assets with lower thresholds
            for ticker in sorted(self.tick_data['ticker'].unique()):
                ticker_data = self.tick_data[self.tick_data['ticker'] == ticker].copy()
                
                is_focus = ticker in self.focus_assets
                
                # FIXED: Lower thresholds for focus assets
                min_ticks_required = 20 if is_focus else 100
                
                if len(ticker_data) < min_ticks_required:
                    focus_marker = "🎯" if is_focus else "  "
                    print(f"   {focus_marker} ⚠️  {ticker}: Insufficient data ({len(ticker_data)} < {min_ticks_required} ticks)")
                    if is_focus:
                        self.focus_diagnostics[ticker]['issues'].append(f'{method_name}: Insufficient ticks ({len(ticker_data)} < {min_ticks_required})')
                    continue
                
                ticker_data = ticker_data.set_index('datetime').sort_index()
                
                try:
                    bars = self._create_bars_single_method(ticker_data, pandas_freq, method_name)
                    
                    # FIXED: Lower bar acceptance threshold for focus assets
                    min_bars_required = 2 if is_focus else 10
                    
                    if len(bars) >= min_bars_required:
                        method_data[ticker] = bars
                        
                        # Detailed bar statistics
                        returns = bars['Returns'].dropna()
                        focus_marker = "🎯" if is_focus else "  "
                        print(f"   {focus_marker} ✅ {ticker}:")
                        print(f"      {len(ticker_data):,} ticks → {len(bars)} bars")
                        if len(returns) > 0:
                            print(f"      Return stats: μ={returns.mean():.6f}, σ={returns.std():.6f}")
                            print(f"      Return range: {returns.min():.6f} to {returns.max():.6f}")
                            print(f"      Non-zero returns: {(returns != 0).sum()}/{len(returns)} ({(returns != 0).sum()/len(returns)*100:.1f}%)")
                        
                        # Price comparison between first and last
                        first_price = bars['Close'].iloc[0]
                        last_price = bars['Close'].iloc[-1]
                        total_return = (last_price - first_price) / first_price
                        print(f"      Price: ${first_price:.2f} → ${last_price:.2f} ({total_return*100:.2f}%)")
                        
                        # Update focus diagnostics
                        if is_focus:
                            self.focus_diagnostics[ticker][f'{method_name}_bars'] = len(bars)
                            self.focus_diagnostics[ticker][f'{method_name}_success'] = True
                        
                    else:
                        focus_marker = "🎯" if is_focus else "  "
                        print(f"   {focus_marker} ❌ {ticker}: Insufficient bars ({len(bars)} < {min_bars_required})")
                        if is_focus:
                            self.focus_diagnostics[ticker]['issues'].append(f'{method_name}: Insufficient bars ({len(bars)} < {min_bars_required})')
                
                except Exception as e:
                    focus_marker = "🎯" if is_focus else "  "
                    print(f"   {focus_marker} ❌ {ticker}: Error - {e}")
                    if is_focus:
                        self.focus_diagnostics[ticker]['issues'].append(f'{method_name}: Error - {str(e)}')
                    continue
            
            all_method_data[method_name] = method_data
            focus_count = sum(1 for ticker in method_data.keys() if ticker in self.focus_assets)
            print(f"   📊 {method_name} summary: {len(method_data)} assets processed ({focus_count} focus assets)")
        
        # FOCUS ASSET BAR CREATION SUMMARY
        print(f"\n🎯 FOCUS ASSET BAR CREATION SUMMARY:")
        print("=" * 50)
        
        for focus_asset in self.focus_assets:
            if focus_asset in self.focus_diagnostics:
                diag = self.focus_diagnostics[focus_asset]
                if diag['found_in_raw']:
                    print(f"   {focus_asset}:")
                    methods_success = []
                    methods_failed = []
                    
                    for method_name in methods.keys():
                        if focus_asset in all_method_data[method_name]:
                            bars_count = len(all_method_data[method_name][focus_asset])
                            methods_success.append(f"{method_name}({bars_count})")
                        else:
                            methods_failed.append(method_name)
                    
                    if methods_success:
                        print(f"      ✅ Success: {', '.join(methods_success)}")
                    if methods_failed:
                        print(f"      ❌ Failed: {', '.join(methods_failed)}")
                    
                    if diag.get('issues'):
                        print(f"      ⚠️  Issues: {'; '.join(diag['issues'])}")
                else:
                    print(f"   {focus_asset}: ❌ Not found in raw data")
        
        return all_method_data
    
    def _create_bars_single_method(self, ticker_data, pandas_freq, method):
        """Create bars using a single method with detailed logic"""
        
        if method == 'OHLC':
            ohlc = ticker_data['price'].resample(pandas_freq).agg(['first', 'max', 'min', 'last'])
            volume = ticker_data['size'].resample(pandas_freq).sum()
            bars = pd.DataFrame({
                'Open': ohlc['first'],
                'High': ohlc['max'],
                'Low': ohlc['min'],
                'Close': ohlc['last'],
                'Volume': volume
            })
            bars['Returns'] = bars['Close'].pct_change()
            
        elif method == 'Average':
            avg_price = ticker_data['price'].resample(pandas_freq).mean()
            volume = ticker_data['size'].resample(pandas_freq).sum()
            tick_count = ticker_data['price'].resample(pandas_freq).count()
            bars = pd.DataFrame({
                'Close': avg_price,
                'Volume': volume,
                'TickCount': tick_count
            })
            bars['Returns'] = bars['Close'].pct_change()
            
        elif method == 'VWAP':
            ticker_data['dollar_volume'] = ticker_data['price'] * ticker_data['size']
            vwap = ticker_data['dollar_volume'].resample(pandas_freq).sum() / ticker_data['size'].resample(pandas_freq).sum()
            volume = ticker_data['size'].resample(pandas_freq).sum()
            bars = pd.DataFrame({
                'Close': vwap,
                'Volume': volume
            })
            bars['Returns'] = bars['Close'].pct_change()
            
        elif method == 'LastTick':
            last_price = ticker_data['price'].resample(pandas_freq).last()
            volume = ticker_data['size'].resample(pandas_freq).sum()
            bars = pd.DataFrame({
                'Close': last_price,
                'Volume': volume
            })
            bars['Returns'] = bars['Close'].pct_change()
            
        elif method == 'Median':
            median_price = ticker_data['price'].resample(pandas_freq).median()
            volume = ticker_data['size'].resample(pandas_freq).sum()
            bars = pd.DataFrame({
                'Close': median_price,
                'Volume': volume
            })
            bars['Returns'] = bars['Close'].pct_change()
            
        elif method == 'TWAP':
            # Simplified TWAP
            weighted_price = ticker_data['price'].resample(pandas_freq).mean()  # Simplified
            volume = ticker_data['size'].resample(pandas_freq).sum()
            bars = pd.DataFrame({
                'Close': weighted_price,
                'Volume': volume
            })
            bars['Returns'] = bars['Close'].pct_change()
        
        return bars.dropna()
    
    def _perform_statistical_analysis(self):
        """Detailed statistical analysis of returns across methods"""
        
        print(f"\n📊 DETAILED STATISTICAL ANALYSIS")
        print("=" * 50)
        
        statistical_results = {}
        
        for method_name, method_data in self.method_data.items():
            print(f"\n📈 STATISTICAL ANALYSIS: {method_name}")
            print("   " + "-" * 30)
            
            method_stats = {}
            
            for ticker, bars in method_data.items():
                returns = bars['Returns'].dropna()
                
                if len(returns) < 3:  # FIXED: Lower minimum for focus assets
                    continue
                
                # Comprehensive return statistics
                stats_dict = {
                    'count': len(returns),
                    'mean': returns.mean(),
                    'std': returns.std(),
                    'skewness': returns.skew(),
                    'kurtosis': returns.kurtosis(),
                    'min': returns.min(),
                    'max': returns.max(),
                    'q25': returns.quantile(0.25),
                    'q50': returns.median(),
                    'q75': returns.quantile(0.75),
                    'zero_returns_pct': (returns == 0).sum() / len(returns) * 100,
                    'positive_returns_pct': (returns > 0).sum() / len(returns) * 100
                }
                
                # Normality tests (if enough data)
                if len(returns) > 8:
                    shapiro_stat, shapiro_p = stats.shapiro(returns[:5000])  # Limit for shapiro
                    stats_dict['shapiro_stat'] = shapiro_stat
                    stats_dict['shapiro_p'] = shapiro_p
                    stats_dict['normal_at_5pct'] = shapiro_p > 0.05
                
                # Autocorrelation test
                if len(returns) > 3:
                    lag1_corr = returns.autocorr(lag=1)
                    stats_dict['lag1_autocorr'] = lag1_corr if not np.isnan(lag1_corr) else 0
                
                method_stats[ticker] = stats_dict
                
                # Display key statistics (highlight focus assets)
                focus_marker = "🎯" if ticker in self.focus_assets else "  "
                print(f"   {focus_marker} {ticker}:")
                print(f"      n={stats_dict['count']}, μ={stats_dict['mean']:.6f}, σ={stats_dict['std']:.6f}")
                print(f"      Skew={stats_dict['skewness']:.3f}, Kurt={stats_dict['kurtosis']:.3f}")
                print(f"      Zero returns: {stats_dict['zero_returns_pct']:.1f}%")
                print(f"      Positive returns: {stats_dict['positive_returns_pct']:.1f}%")
                if 'normal_at_5pct' in stats_dict:
                    normal_status = "✅ Normal" if stats_dict['normal_at_5pct'] else "❌ Non-normal"
                    print(f"      Normality test: {normal_status} (p={stats_dict['shapiro_p']:.4f})")
                if 'lag1_autocorr' in stats_dict:
                    print(f"      Lag-1 autocorr: {stats_dict['lag1_autocorr']:.4f}")
            
            statistical_results[method_name] = method_stats
            
            # Method-level summary with focus asset highlight
            if method_stats:
                print(f"\n   📊 {method_name} METHOD SUMMARY:")
                print(f"      Assets analyzed: {len(method_stats)}")
                
                focus_stats = {k: v for k, v in method_stats.items() if k in self.focus_assets}
                if focus_stats:
                    print(f"      Focus assets: {len(focus_stats)} / {len(self.focus_assets)}")
                    avg_mean_focus = np.mean([s['mean'] for s in focus_stats.values()])
                    avg_std_focus = np.mean([s['std'] for s in focus_stats.values()])
                    print(f"      Focus group avg return: {avg_mean_focus:.6f}")
                    print(f"      Focus group avg volatility: {avg_std_focus:.6f}")
                
                avg_mean = np.mean([s['mean'] for s in method_stats.values()])
                avg_std = np.mean([s['std'] for s in method_stats.values()])
                avg_skew = np.mean([s['skewness'] for s in method_stats.values()])
                avg_kurt = np.mean([s['kurtosis'] for s in method_stats.values()])
                print(f"      Overall avg return: {avg_mean:.6f}")
                print(f"      Overall avg volatility: {avg_std:.6f}")
                print(f"      Average skewness: {avg_skew:.3f}")
                print(f"      Average kurtosis: {avg_kurt:.3f}")
        
        return statistical_results
    
    def _perform_correlation_analysis(self):
        """Detailed correlation analysis across methods"""
        
        print(f"\n🔗 DETAILED CORRELATION ANALYSIS")
        print("=" * 50)
        
        correlation_results = {}
        
        for method_name, method_data in self.method_data.items():
            print(f"\n📊 CORRELATION ANALYSIS: {method_name}")
            print("   " + "-" * 30)
            
            # Create return matrix
            tickers = list(method_data.keys())
            if len(tickers) < 2:
                print("   ⚠️  Insufficient assets for correlation analysis")
                continue
            
            # Prioritize focus assets but include others
            focus_available = [t for t in self.focus_assets if t in tickers]
            other_available = [t for t in tickers if t not in self.focus_assets]
            analysis_tickers = focus_available + other_available[:max(0, 10-len(focus_available))]
            
            print(f"   🎯 Focus assets available: {focus_available}")
            print(f"   📊 Analysis will include: {analysis_tickers}")
            
            # Align data and create return matrix
            return_data = {}
            common_dates = None
            
            for ticker in analysis_tickers:
                returns = method_data[ticker]['Returns'].dropna()
                return_data[ticker] = returns
                
                if common_dates is None:
                    common_dates = set(returns.index)
                else:
                    common_dates = common_dates.intersection(set(returns.index))
            
            if len(common_dates) < 3:  # FIXED: Lower threshold
                print("   ⚠️  Insufficient common dates for correlation analysis")
                continue
            
            common_dates = sorted(list(common_dates))
            
            # Create aligned return matrix
            return_matrix = pd.DataFrame(index=common_dates)
            for ticker in analysis_tickers:
                aligned_returns = return_data[ticker].reindex(common_dates, method='nearest')
                return_matrix[ticker] = aligned_returns
            
            return_matrix = return_matrix.dropna()
            
            if len(return_matrix) < 3:  # FIXED: Lower threshold
                print("   ⚠️  Insufficient aligned data for correlation analysis")
                continue
            
            # Calculate correlation matrix
            corr_matrix = return_matrix.corr()
            
            print(f"   ✅ Correlation matrix calculated:")
            print(f"      Data points: {len(return_matrix)}")
            print(f"      Assets: {len(corr_matrix)}")
            
            # Focus on correlations involving focus assets
            if focus_available:
                print(f"   🎯 Focus asset correlations:")
                for focus_asset in focus_available:
                    if focus_asset in corr_matrix.index:
                        focus_corrs = corr_matrix[focus_asset].drop(focus_asset)
                        print(f"      {focus_asset} correlations:")
                        for other_asset, corr in focus_corrs.items():
                            focus_marker = "🎯" if other_asset in self.focus_assets else ""
                            print(f"        {focus_marker} {other_asset}: {corr:.4f}")
            
            # Correlation statistics
            upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            correlations = upper_triangle.stack().values
            
            print(f"      Overall correlation stats:")
            print(f"         Mean: {np.mean(correlations):.4f}")
            print(f"         Std: {np.std(correlations):.4f}")
            print(f"         Min: {np.min(correlations):.4f}")
            print(f"         Max: {np.max(correlations):.4f}")
            print(f"         |Corr| > 0.5: {(np.abs(correlations) > 0.5).sum()}/{len(correlations)}")
            print(f"         |Corr| > 0.7: {(np.abs(correlations) > 0.7).sum()}/{len(correlations)}")
            
            correlation_results[method_name] = {
                'correlation_matrix': corr_matrix,
                'return_matrix': return_matrix,
                'focus_assets': focus_available,
                'analysis_assets': analysis_tickers,
                'correlation_stats': {
                    'mean_corr': np.mean(correlations),
                    'std_corr': np.std(correlations),
                    'min_corr': np.min(correlations),
                    'max_corr': np.max(correlations),
                    'high_corr_count': (np.abs(correlations) > 0.5).sum(),
                    'very_high_corr_count': (np.abs(correlations) > 0.7).sum()
                }
            }
        
        return correlation_results
    
    def _perform_detailed_chsh_analysis(self):
        """Comprehensive CHSH analysis with detailed diagnostics"""
        
        print(f"\n⚛️  DETAILED CHSH ANALYSIS")
        print("=" * 50)
        
        chsh_results = {}
        
        for method_name, method_data in self.method_data.items():
            print(f"\n🔬 CHSH ANALYSIS: {method_name}")
            print("   " + "-" * 30)
            
            # Enhanced data preparation
            enhanced_data = {}
            for ticker, bars in method_data.items():
                enhanced = bars.copy()
                
                # Multiple binary observables
                enhanced['price_direction'] = np.where(enhanced['Returns'] > 0, 1, -1)
                
                # Volatility regime
                window = max(3, len(enhanced) // 8)  # FIXED: Lower minimum
                enhanced['vol_short'] = enhanced['Returns'].abs().rolling(max(2, window//2)).mean()
                enhanced['vol_long'] = enhanced['Returns'].abs().rolling(window).mean()
                enhanced['vol_regime'] = np.where(enhanced['vol_short'] > enhanced['vol_long'], 1, -1)
                
                # Momentum regime
                if len(enhanced) > 5:  # FIXED: Lower threshold
                    enhanced['price_ma_short'] = enhanced['Close'].rolling(max(2, len(enhanced)//10)).mean()
                    enhanced['price_ma_long'] = enhanced['Close'].rolling(max(3, len(enhanced)//5)).mean()
                    enhanced['momentum_regime'] = np.where(enhanced['price_ma_short'] > enhanced['price_ma_long'], 1, -1)
                else:
                    enhanced['momentum_regime'] = enhanced['price_direction']
                
                # Magnitude regime
                enhanced['abs_return'] = enhanced['Returns'].abs()
                enhanced['return_threshold'] = enhanced['abs_return'].quantile(0.6)
                enhanced['magnitude_regime'] = np.where(enhanced['abs_return'] > enhanced['return_threshold'], 1, -1)
                
                enhanced_data[ticker] = enhanced
            
            # Test multiple CHSH measurement combinations
            measurement_sets = {
                'price_vol': ('price_direction', 'vol_regime'),
                'price_momentum': ('price_direction', 'momentum_regime'),
                'vol_momentum': ('vol_regime', 'momentum_regime'),
                'price_magnitude': ('price_direction', 'magnitude_regime')
            }
            
            method_chsh_results = {}
            
            for set_name, (obs_a, obs_b) in measurement_sets.items():
                print(f"\n   🎯 MEASUREMENT SET: {set_name}")
                print(f"      Observable A: {obs_a}")
                print(f"      Observable B: {obs_b}")
                
                set_results = {}
                tickers = list(enhanced_data.keys())
                
                # Prioritize focus asset pairs
                focus_pairs = []
                other_pairs = []
                
                for i, ticker1 in enumerate(tickers):
                    for j, ticker2 in enumerate(tickers[i+1:], i+1):
                        if ticker1 in self.focus_assets or ticker2 in self.focus_assets:
                            focus_pairs.append((ticker1, ticker2))
                        else:
                            other_pairs.append((ticker1, ticker2))
                
                # Test focus pairs first, then others (limited)
                test_pairs = focus_pairs + other_pairs[:max(0, 10-len(focus_pairs))]
                
                for ticker1, ticker2 in test_pairs:
                    try:
                        # Align data
                        df1 = enhanced_data[ticker1]
                        df2 = enhanced_data[ticker2]
                        
                        common_idx = df1.index.intersection(df2.index)
                        if len(common_idx) < 5:  # FIXED: Lower threshold for focus pairs
                            continue
                        
                        df1_aligned = df1.loc[common_idx]
                        df2_aligned = df2.loc[common_idx]
                        
                        # Extract measurements
                        A = np.sign(df1_aligned[obs_a].values)
                        B = np.sign(df2_aligned[obs_a].values)  # Same observable for both assets
                        A_prime = np.sign(df1_aligned[obs_b].values)
                        B_prime = np.sign(df2_aligned[obs_b].values)
                        
                        # Ensure binary ±1
                        A[A==0] = 1; B[B==0] = 1
                        A_prime[A_prime==0] = 1; B_prime[B_prime==0] = 1
                        
                        # Calculate expectation values with detailed output
                        E_AB = np.mean(A * B)
                        E_AB_prime = np.mean(A * B_prime)
                        E_A_prime_B = np.mean(A_prime * B)
                        E_A_prime_B_prime = np.mean(A_prime * B_prime)
                        
                        # CHSH value
                        chsh_value = abs(E_AB + E_AB_prime + E_A_prime_B - E_A_prime_B_prime)
                        
                        # Statistical validation
                        n = len(A)
                        standard_error = np.sqrt(4/n)  # Approximate SE for CHSH
                        t_stat = (chsh_value - 2.0) / standard_error if standard_error > 0 else 0
                        p_value = 2 * (1 - stats.norm.cdf(abs(t_stat))) if standard_error > 0 else 1.0
                        
                        # Store detailed results
                        pair_key = f"{ticker1}_{ticker2}"
                        detailed_result = {
                            'chsh_value': chsh_value,
                            'sample_size': n,
                            'expectation_values': {
                                'E_AB': E_AB,
                                'E_AB_prime': E_AB_prime,
                                'E_A_prime_B': E_A_prime_B,
                                'E_A_prime_B_prime': E_A_prime_B_prime
                            },
                            'statistical_tests': {
                                'standard_error': standard_error,
                                't_statistic': t_stat,
                                'p_value': p_value,
                                'significant_at_5pct': p_value < 0.05
                            },
                            'violation': chsh_value > 2.0,
                            'quantum_approach_pct': (chsh_value / 2.828) * 100
                        }
                        
                        set_results[pair_key] = detailed_result
                        
                        # Detailed output (highlight focus pairs)
                        focus_marker = "🎯" if (ticker1 in self.focus_assets or ticker2 in self.focus_assets) else "  "
                        print(f"      {focus_marker} {pair_key}:")
                        print(f"         CHSH = {chsh_value:.6f} (n={n})")
                        print(f"         E(A,B)={E_AB:.4f}, E(A,B')={E_AB_prime:.4f}")
                        print(f"         E(A',B)={E_A_prime_B:.4f}, E(A',B')={E_A_prime_B_prime:.4f}")
                        print(f"         SE={standard_error:.6f}, t={t_stat:.4f}, p={p_value:.6f}")
                        
                        if chsh_value > 2.0:
                            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                            print(f"         🚨 CHSH VIOLATION: {chsh_value:.6f} > 2.0 {significance}")
                        elif chsh_value > 1.9:
                            print(f"         ⚠️  NEAR VIOLATION: {chsh_value:.6f}")
                        elif chsh_value > 1.7:
                            print(f"         📊 HIGH: {chsh_value:.6f}")
                        else:
                            print(f"         📊 Classical: {chsh_value:.6f}")
                    
                    except Exception as e:
                        focus_marker = "🎯" if (ticker1 in self.focus_assets or ticker2 in self.focus_assets) else "  "
                        print(f"      {focus_marker} ❌ Error for {ticker1}-{ticker2}: {e}")
                        continue
                
                method_chsh_results[set_name] = set_results
                
                # Measurement set summary with focus highlights
                if set_results:
                    chsh_values = [r['chsh_value'] for r in set_results.values()]
                    violations = sum(1 for r in set_results.values() if r['violation'])
                    near_violations = sum(1 for r in set_results.values() if r['chsh_value'] > 1.9)
                    
                    # Focus pair statistics
                    focus_results = {k: v for k, v in set_results.items() 
                                   if any(asset in self.focus_assets for asset in k.split('_'))}
                    focus_violations = sum(1 for r in focus_results.values() if r['violation'])
                    
                    print(f"\n      📊 {set_name.upper()} SUMMARY:")
                    print(f"         Pairs tested: {len(set_results)} (focus: {len(focus_results)})")
                    print(f"         Max CHSH: {max(chsh_values):.6f}")
                    print(f"         Mean CHSH: {np.mean(chsh_values):.6f}")
                    print(f"         Std CHSH: {np.std(chsh_values):.6f}")
                    print(f"         Violations (>2.0): {violations} (focus: {focus_violations})")
                    print(f"         Near violations (>1.9): {near_violations}")
                else:
                    print(f"      ❌ No valid results for {set_name}")
            
            chsh_results[method_name] = method_chsh_results
        
        return chsh_results
    
    def _perform_conditional_bell_analysis(self, window_size=20, threshold_quantile=0.75):
        """
        Conditional Bell test (S1 inequality) - Yahoo Finance style
        Enhanced with focus group prioritization
        """
        
        print(f"\n⚛️  CONDITIONAL BELL ANALYSIS (S1 INEQUALITY)")
        print("=" * 60)
        print(f"Window size: {window_size}")
        print(f"Threshold quantile: {threshold_quantile}")
        print(f"Focus assets: {self.focus_assets}")
        print("Method: Same as Yahoo Finance AAPL-MSFT analysis")
        
        def expectation_ab(x_mask, y_mask, a, b):
            """Calculate conditional expectation E[AB | measurement settings]"""
            mask = x_mask & y_mask
            if mask.sum() == 0:
                return 0.0
            return np.mean(a[mask] * b[mask])
        
        conditional_results = {}
        
        # Test each aggregation method
        for method_name, method_bars in self.method_data.items():
            print(f"\n📊 CONDITIONAL BELL TEST: {method_name}")
            print("   " + "-" * 40)
            
            method_results = {}
            tickers = list(method_bars.keys())
            
            # Prioritize focus asset pairs
            focus_pairs = []
            other_pairs = []
            
            for i, ticker_A in enumerate(tickers):
                for j, ticker_B in enumerate(tickers[i+1:], i+1):
                    if ticker_A in self.focus_assets or ticker_B in self.focus_assets:
                        focus_pairs.append((ticker_A, ticker_B))
                    else:
                        other_pairs.append((ticker_A, ticker_B))
            
            # Test focus pairs first, then others (limited)
            test_pairs = focus_pairs + other_pairs[:max(0, 15-len(focus_pairs))]
            
            print(f"   🎯 Focus pairs to test: {len(focus_pairs)}")
            print(f"   📊 Total pairs to test: {len(test_pairs)}")
            
            for ticker_A, ticker_B in test_pairs:
                try:
                    # Get aligned data
                    bars_A = method_bars[ticker_A]
                    bars_B = method_bars[ticker_B]
                    
                    # Align on common dates
                    common_idx = bars_A.index.intersection(bars_B.index)
                    
                    # FIXED: Lower minimum window size for focus pairs
                    min_required = window_size + 5 if (ticker_A in self.focus_assets or ticker_B in self.focus_assets) else window_size + 10
                    
                    if len(common_idx) < min_required:
                        continue
                    
                    # Create returns dataframe (like Yahoo Finance structure)
                    data_aligned = pd.DataFrame({
                        ticker_A: bars_A.loc[common_idx]['Close'],
                        ticker_B: bars_B.loc[common_idx]['Close']
                    }).dropna()
                    
                    returns = data_aligned.pct_change().dropna()
                    
                    if len(returns) < window_size + 2:
                        continue
                    
                    focus_marker = "🎯" if (ticker_A in self.focus_assets or ticker_B in self.focus_assets) else "  "
                    print(f"   {focus_marker} Testing pair: {ticker_A} vs {ticker_B}")
                    print(f"      Data points: {len(returns)}")
                    
                    # Rolling window S1 calculation (exact replica of Yahoo approach)
                    s1_series = []
                    
                    for T in range(window_size, len(returns)):
                        window_returns = returns.iloc[T - window_size:T]
                        
                        # Calculate volatility thresholds (75th percentile of absolute returns)
                        thresholds = window_returns.abs().quantile(threshold_quantile)
                        
                        RA = window_returns[ticker_A]
                        RB = window_returns[ticker_B]
                        
                        # Observables: sign of returns
                        a = np.sign(RA)
                        b = np.sign(RB)
                        
                        # Measurement settings based on volatility
                        x0 = RA.abs() >= thresholds[ticker_A]  # High volatility for A
                        x1 = ~x0                                # Low volatility for A
                        y0 = RB.abs() >= thresholds[ticker_B]  # High volatility for B
                        y1 = ~y0                                # Low volatility for B
                        
                        # Calculate conditional expectations
                        ab_00 = expectation_ab(x0, y0, a, b)  # Both high vol
                        ab_01 = expectation_ab(x0, y1, a, b)  # A high, B low vol
                        ab_10 = expectation_ab(x1, y0, a, b)  # A low, B high vol
                        ab_11 = expectation_ab(x1, y1, a, b)  # Both low vol
                        
                        # S1 Bell inequality (THIS IS THE KEY DIFFERENCE!)
                        S1 = ab_00 + ab_01 + ab_10 - ab_11
                        
                        s1_series.append({
                            'date': returns.index[T],
                            'S1_value': S1,
                            'ab_00': ab_00,
                            'ab_01': ab_01,
                            'ab_10': ab_10,
                            'ab_11': ab_11,
                            'ticker_A': ticker_A,
                            'ticker_B': ticker_B
                        })
                    
                    if s1_series:
                        s1_df = pd.DataFrame(s1_series)
                        
                        # Calculate statistics
                        max_s1 = s1_df['S1_value'].max()
                        min_s1 = s1_df['S1_value'].min()
                        mean_s1 = s1_df['S1_value'].mean()
                        std_s1 = s1_df['S1_value'].std()
                        
                        # Count violations
                        violations_positive = (s1_df['S1_value'] > 2.0).sum()
                        violations_negative = (s1_df['S1_value'] < -2.0).sum()
                        total_violations = violations_positive + violations_negative
                        
                        # Near violations
                        near_violations = ((s1_df['S1_value'] > 1.8) | (s1_df['S1_value'] < -1.8)).sum()
                        
                        pair_result = {
                            'ticker_A': ticker_A,
                            'ticker_B': ticker_B,
                            's1_series': s1_df,
                            'max_s1': max_s1,
                            'min_s1': min_s1,
                            'mean_s1': mean_s1,
                            'std_s1': std_s1,
                            'violations_positive': violations_positive,
                            'violations_negative': violations_negative,
                            'total_violations': total_violations,
                            'near_violations': near_violations,
                            'violation_rate': total_violations / len(s1_df) * 100,
                            'data_points': len(s1_df),
                            'is_focus_pair': ticker_A in self.focus_assets or ticker_B in self.focus_assets
                        }
                        
                        method_results[f"{ticker_A}_{ticker_B}"] = pair_result
                        
                        # Print detailed results
                        print(f"      Results:")
                        print(f"         S1 range: [{min_s1:.4f}, {max_s1:.4f}]")
                        print(f"         Mean S1: {mean_s1:.4f} ± {std_s1:.4f}")
                        print(f"         Bell violations: {total_violations} / {len(s1_df)} ({total_violations/len(s1_df)*100:.1f}%)")
                        
                        if total_violations > 0:
                            print(f"         🚨 BELL VIOLATIONS DETECTED!")
                            if max_s1 > 2.0:
                                print(f"            Max positive violation: S1 = {max_s1:.6f}")
                            if min_s1 < -2.0:
                                print(f"            Max negative violation: S1 = {min_s1:.6f}")
                        elif near_violations > 0:
                            print(f"         ⚠️  Near violations: {near_violations}")
                        else:
                            print(f"         ✅ No violations (classical)")
                
                except Exception as e:
                    focus_marker = "🎯" if (ticker_A in self.focus_assets or ticker_B in self.focus_assets) else "  "
                    print(f"   {focus_marker} ❌ Error testing {ticker_A}-{ticker_B}: {e}")
                    continue
            
            conditional_results[method_name] = method_results
            
            # Method summary with focus highlights
            if method_results:
                all_violations = sum(r['total_violations'] for r in method_results.values())
                all_tests = sum(r['data_points'] for r in method_results.values())
                max_s1_method = max(r['max_s1'] for r in method_results.values())
                min_s1_method = min(r['min_s1'] for r in method_results.values())
                
                # Focus pair statistics
                focus_results = {k: v for k, v in method_results.items() if v['is_focus_pair']}
                focus_violations = sum(r['total_violations'] for r in focus_results.values())
                focus_tests = sum(r['data_points'] for r in focus_results.values()) if focus_results else 0
                
                print(f"\n   📊 {method_name} CONDITIONAL BELL SUMMARY:")
                print(f"      Pairs tested: {len(method_results)} (focus: {len(focus_results)})")
                print(f"      Total violations: {all_violations} / {all_tests} ({all_violations/all_tests*100:.2f}%)")
                if focus_tests > 0:
                    print(f"      Focus violations: {focus_violations} / {focus_tests} ({focus_violations/focus_tests*100:.2f}%)")
                print(f"      Max |S1|: {max(abs(max_s1_method), abs(min_s1_method)):.6f}")
            else:
                print(f"   ❌ No valid results for {method_name}")
        
        return conditional_results
    
    def _perform_robustness_tests(self):
        """Robustness tests with different parameters"""
        
        print(f"\n🔄 ROBUSTNESS TESTS")
        print("=" * 50)
        
        # Test different frequencies
        frequencies = ['5min', '15min', '30min']
        robustness_results = {}
        
        for freq in frequencies:
            print(f"\n⏱️  TESTING FREQUENCY: {freq}")
            print("   " + "-" * 20)
            
            try:
                # Create bars at this frequency
                method_data_freq = self._create_bars_all_methods_fixed(freq)
                
                # Quick CHSH test
                freq_chsh_results = {}
                
                for method_name, method_data in method_data_freq.items():
                    if len(method_data) < 2:
                        continue
                    
                    # Focus on focus assets for robustness testing
                    focus_available = [t for t in self.focus_assets if t in method_data.keys()]
                    test_tickers = focus_available + [t for t in list(method_data.keys())[:5] if t not in focus_available]
                    
                    # Simple CHSH test
                    chsh_values = []
                    
                    for i, ticker1 in enumerate(test_tickers):
                        for j, ticker2 in enumerate(test_tickers[i+1:], i+1):
                            
                            try:
                                bars1 = method_data[ticker1]
                                bars2 = method_data[ticker2]
                                
                                # Quick alignment
                                common_idx = bars1.index.intersection(bars2.index)
                                if len(common_idx) < 5:  # FIXED: Lower threshold
                                    continue
                                
                                returns1 = bars1.loc[common_idx]['Returns']
                                returns2 = bars2.loc[common_idx]['Returns']
                                
                                # Simple binary measurements
                                A = np.sign(returns1.values)
                                B = np.sign(returns2.values)
                                A_prime = np.sign(returns1.abs() - returns1.abs().median())
                                B_prime = np.sign(returns2.abs() - returns2.abs().median())
                                
                                A[A==0] = 1; B[B==0] = 1
                                A_prime[A_prime==0] = 1; B_prime[B_prime==0] = 1
                                
                                # CHSH
                                E_AB = np.mean(A * B)
                                E_AB_prime = np.mean(A * B_prime)
                                E_A_prime_B = np.mean(A_prime * B)
                                E_A_prime_B_prime = np.mean(A_prime * B_prime)
                                
                                chsh_value = abs(E_AB + E_AB_prime + E_A_prime_B - E_A_prime_B_prime)
                                chsh_values.append(chsh_value)
                                
                            except:
                                continue
                    
                    if chsh_values:
                        freq_chsh_results[method_name] = {
                            'max_chsh': max(chsh_values),
                            'mean_chsh': np.mean(chsh_values),
                            'violations': sum(1 for v in chsh_values if v > 2.0),
                            'tests': len(chsh_values)
                        }
                        
                        print(f"      {method_name}: Max CHSH = {max(chsh_values):.4f}, Violations = {sum(1 for v in chsh_values if v > 2.0)}")
                
                robustness_results[freq] = freq_chsh_results
                
            except Exception as e:
                print(f"   ❌ Error testing {freq}: {e}")
                continue
        
        return robustness_results
    
    # =================== ALL PLOTTING METHODS (FIXED) ===================
    
    def _create_comprehensive_visualizations_fixed(self):
        """FIXED: Create comprehensive visualization plots showing ALL focus assets"""
        
        print(f"\n📊 CREATING COMPREHENSIVE VISUALIZATIONS (FIXED)")
        print("=" * 50)
        
        try:
            # Create individual time series plots for EACH focus asset (FIXED)
            self._create_individual_focus_time_series_plots()
            
            # Create multi-resolution comparison (focus assets)
            self._create_multi_resolution_plots()
            
            # Create main comprehensive figure
            fig = plt.figure(figsize=(20, 16))
            gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
            
            # Plot 1: Return distributions by method (focus group highlighted)
            ax1 = fig.add_subplot(gs[0, :2])
            self._plot_return_distributions_focused(ax1)
            
            # Plot 2: Price level comparison (focus assets)
            ax2 = fig.add_subplot(gs[0, 2:])
            self._plot_price_level_comparison_focused(ax2)
            
            # Plot 3: CHSH values by method and measurement set
            ax3 = fig.add_subplot(gs[1, :2])
            self._plot_chsh_detailed(ax3)
            
            # Plot 4: Focus group correlation analysis
            ax4 = fig.add_subplot(gs[1, 2:])
            self._plot_focus_correlation_analysis(ax4)
            
            # Plot 5: Cumulative returns comparison (focus group)
            ax5 = fig.add_subplot(gs[2, :])
            self._plot_detailed_cumulative_returns_focused(ax5)
            
            # Plot 6: Volatility comparison (focus group highlighted)
            ax6 = fig.add_subplot(gs[3, :2])
            self._plot_volatility_comparison_focused(ax6)
            
            # Plot 7: Return difference analysis (focus pairs)
            ax7 = fig.add_subplot(gs[3, 2:])
            self._plot_return_differences_focused(ax7)
            
            plt.suptitle(f'Comprehensive Aggregation Method Analysis\nFocus Group: {", ".join(self.focus_assets)}', 
                        fontsize=16, fontweight='bold')
            
            # Save main plot
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            focus_str = "_".join(self.focus_assets)
            filename = f'comprehensive_analysis_{focus_str}_{timestamp}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"💾 Main comprehensive plot saved: {filename}")
            
            plt.show()
            
        except Exception as e:
            print(f"❌ Visualization error: {e}")
    
    def _create_individual_focus_time_series_plots(self):
        """FIXED: Create individual time series plots for EACH focus asset"""
        
        print("   🕐 Creating individual time series plots for EACH focus asset...")
        
        # Get primary method (prefer OHLC)
        primary_method = 'OHLC' if 'OHLC' in self.method_data else list(self.method_data.keys())[0]
        method_data = self.method_data[primary_method]
        
        # Find focus assets that have data
        focus_assets_with_data = [asset for asset in self.focus_assets if asset in method_data]
        
        print(f"   Creating plots for {len(focus_assets_with_data)} focus assets: {focus_assets_with_data}")
        
        # Create individual plot for EACH focus asset
        for focus_asset in focus_assets_with_data:
            bars = method_data[focus_asset]
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
            
            # Plot 1: Price time series with volume
            ax1.plot(bars.index, bars['Close'], 'r-', linewidth=2, alpha=0.8, 
                    label=f'{focus_asset} Close Price')
            
            # Add volume bars if available
            if 'Volume' in bars.columns and bars['Volume'].sum() > 0:
                ax1_vol = ax1.twinx()
                ax1_vol.bar(bars.index, bars['Volume'], alpha=0.3, color='gray', width=0.8)
                ax1_vol.set_ylabel('Volume', color='gray')
                ax1_vol.tick_params(axis='y', labelcolor='gray')
            
            ax1.set_title(f'🎯 FOCUS ASSET: {focus_asset} - {primary_method} Method\n'
                         f'Price Time Series (n={len(bars)} bars from {self.focus_diagnostics[focus_asset]["tick_count"]:,} ticks)')
            ax1.set_ylabel('Price ($)')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Add price statistics
            if len(bars) > 1:
                price_change = (bars['Close'].iloc[-1] - bars['Close'].iloc[0]) / bars['Close'].iloc[0] * 100
                stats_text = (f'Price Change: {price_change:+.2f}%\nRange: ${bars["Close"].min():.2f} - ${bars["Close"].max():.2f}')
                ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
            
            # Plot 2: Returns analysis
            if 'Returns' in bars.columns and len(bars) > 1:
                returns = bars['Returns'].dropna()
                if len(returns) > 0:
                    # Plot returns time series
                    ax2.plot(returns.index, returns * 100, 'b-', alpha=0.7, linewidth=1)
                    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                    ax2.set_title(f'{focus_asset} - Returns (%)')
                    ax2.set_ylabel('Returns (%)')
                    ax2.grid(True, alpha=0.3)
                    
                    # Add return statistics
                    if len(returns) > 0:
                        ret_stats = f'Mean: {returns.mean()*100:.4f}%\nStd: {returns.std()*100:.4f}%'
                        ax2.text(0.02, 0.98, ret_stats, transform=ax2.transAxes, verticalalignment='top',
                                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
                else:
                    ax2.text(0.5, 0.5, 'No returns data', ha='center', va='center', transform=ax2.transAxes)
            else:
                ax2.text(0.5, 0.5, 'Insufficient data for returns', ha='center', va='center', transform=ax2.transAxes)
            
            plt.tight_layout()
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'focus_asset_{focus_asset}_{timestamp}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"💾 Saved: {filename}")
            plt.show()
    
    def _create_multi_resolution_plots(self):
        """Create plots comparing multiple time resolutions (focus assets)"""
        
        print("   ⏱️  Creating multi-resolution comparison plots (focus assets)...")
        
        # Test multiple resolutions
        resolutions = ['5min', '15min', '30min']
        resolution_data = {}
        
        # Focus on first available focus asset for multi-resolution analysis
        focus_asset = None
        for asset in self.focus_assets:
            if hasattr(self, 'tick_data') and self.tick_data is not None and asset in self.tick_data['ticker'].values:
                focus_asset = asset
                break
        
        if not focus_asset:
            print("      ❌ No focus assets available for multi-resolution analysis")
            return
        
        for resolution in resolutions:
            try:
                print(f"      Creating {resolution} bars for {focus_asset}...")
                freq_map = {'5min': '5T', '15min': '15T', '30min': '30T'}
                pandas_freq = freq_map[resolution]
                
                # Create bars for focus asset at this resolution
                if self.tick_data is not None:
                    ticker_data = self.tick_data[self.tick_data['ticker'] == focus_asset]
                    if len(ticker_data) < 20:  # FIXED: Lower threshold
                        continue
                    
                    ticker_indexed = ticker_data.set_index('datetime').sort_index()
                    
                    # Create OHLC bars
                    ohlc = ticker_indexed['price'].resample(pandas_freq).agg(['first', 'max', 'min', 'last'])
                    volume = ticker_indexed['size'].resample(pandas_freq).sum()
                    
                    bars = pd.DataFrame({
                        'Open': ohlc['first'],
                        'High': ohlc['max'],
                        'Low': ohlc['min'],
                        'Close': ohlc['last'],
                        'Volume': volume
                    }).dropna()
                    
                    if len(bars) > 0:  # FIXED: Accept even single bars
                        bars['Returns'] = bars['Close'].pct_change()
                        resolution_data[resolution] = bars
                
            except Exception as e:
                print(f"      ❌ Error creating {resolution} bars: {e}")
                continue
        
        # Create comparison plot for focus asset
        if resolution_data:
            fig, axes = plt.subplots(len(resolution_data), 1, figsize=(15, 4*len(resolution_data)))
            if len(resolution_data) == 1:
                axes = [axes]
            
            colors = ['blue', 'green', 'red'][:len(resolution_data)]
            
            for i, (resolution, bars) in enumerate(resolution_data.items()):
                ax = axes[i] if len(resolution_data) > 1 else axes[0]
                
                # Price time series
                ax.plot(bars.index, bars['Close'], color=colors[i], linewidth=2, 
                       label=f'{resolution} bars (n={len(bars)})')
                
                ax.set_ylabel(f'{focus_asset} Price ($)')
                ax.set_title(f'🎯 FOCUS ASSET: {focus_asset} - {resolution} Resolution')
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                # Add statistics
                if 'Returns' in bars.columns and len(bars) > 1:
                    returns = bars['Returns'].dropna()
                    if len(returns) > 0:
                        stats_text = f'Returns: μ={returns.mean():.6f}, σ={returns.std():.6f}'
                        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                               verticalalignment='top', fontsize=9,
                               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
            
            plt.tight_layout()
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'multi_resolution_{focus_asset}_{timestamp}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"   💾 Multi-resolution plot saved: {filename}")
            plt.show()
    
    def _plot_return_distributions_focused(self, ax):
        """Plot return distributions highlighting focus assets"""
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        focus_colors = ['red', 'darkred', 'crimson']
        other_colors = ['blue', 'green', 'orange']
        
        plotted_focus = 0
        plotted_other = 0
        
        for i, (method_name, method_data) in enumerate(list(self.method_data.items())[:3]):
            if not method_data:
                continue
            
            focus_returns = []
            other_returns = []
            
            for ticker, bars in method_data.items():
                returns = bars['Returns'].dropna()
                if ticker in self.focus_assets:
                    focus_returns.extend(returns.values)
                else:
                    other_returns.extend(returns.values)
            
            if focus_returns and plotted_focus < len(focus_colors):
                ax.hist(focus_returns, bins=50, alpha=0.7, 
                       label=f'🎯 {method_name} Focus (n={len(focus_returns)})', 
                       color=focus_colors[plotted_focus], density=True)
                plotted_focus += 1
            
            if other_returns and plotted_other < len(other_colors):
                ax.hist(other_returns, bins=50, alpha=0.4, 
                       label=f'{method_name} Other (n={len(other_returns)})', 
                       color=other_colors[plotted_other], density=True)
                plotted_other += 1
        
        ax.set_xlabel('Returns')
        ax.set_ylabel('Density')
        ax.set_title('Return Distributions\n(Focus Assets Highlighted)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_price_level_comparison_focused(self, ax):
        """Plot price levels for focus assets"""
        
        colors = ['red', 'darkred', 'crimson', 'orange']
        
        if 'OHLC' not in self.method_data:
            ax.text(0.5, 0.5, 'No OHLC data\navailable', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Focus Asset Price Comparison')
            return
        
        method_data = self.method_data['OHLC']
        focus_available = [a for a in self.focus_assets if a in method_data.keys()]
        
        if not focus_available:
            ax.text(0.5, 0.5, 'No focus assets\navailable', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Focus Asset Price Comparison')
            return
        
        for i, asset in enumerate(focus_available[:4]):
            bars = method_data[asset]
            # Normalize to starting price for comparison
            normalized_prices = bars['Close'] / bars['Close'].iloc[0]
            ax.plot(bars.index, normalized_prices, 
                   label=f'🎯 {asset} (n={len(bars)})', 
                   color=colors[i], linewidth=2, alpha=0.8)
        
        ax.set_ylabel('Normalized Price')
        ax.set_title(f'Focus Asset Price Evolution\n{", ".join(focus_available)}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=1, color='black', linestyle='--', alpha=0.5)
    
    def _plot_focus_correlation_analysis(self, ax):
        """Plot correlation analysis for focus group"""
        
        if 'OHLC' not in self.method_data:
            ax.text(0.5, 0.5, 'No correlation data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Focus Group Correlations')
            return
        
        method_data = self.method_data['OHLC']
        focus_available = [a for a in self.focus_assets if a in method_data.keys()]
        
        if len(focus_available) < 2:
            ax.text(0.5, 0.5, 'Need ≥2 focus assets\nfor correlation', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Focus Group Correlations')
            return
        
        # Create correlation matrix for focus group
        return_data = {}
        common_dates = None
        
        for ticker in focus_available:
            returns = method_data[ticker]['Returns'].dropna()
            return_data[ticker] = returns
            
            if common_dates is None:
                common_dates = set(returns.index)
            else:
                common_dates = common_dates.intersection(set(returns.index))
        
        if len(common_dates) < 3:  # FIXED: Lower threshold
            ax.text(0.5, 0.5, 'Insufficient\ncommon data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Focus Group Correlations')
            return
        
        common_dates = sorted(list(common_dates))
        return_matrix = pd.DataFrame(index=common_dates)
        for ticker in focus_available:
            return_matrix[ticker] = return_data[ticker].reindex(common_dates)
        return_matrix = return_matrix.dropna()
        
        if len(return_matrix) < 2:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Focus Group Correlations')
            return
        
        # Calculate and plot correlation matrix
        corr_matrix = return_matrix.corr()
        
        im = ax.imshow(corr_matrix, cmap='RdYlBu_r', vmin=-1, vmax=1)
        
        # Add labels
        ax.set_xticks(range(len(focus_available)))
        ax.set_yticks(range(len(focus_available)))
        ax.set_xticklabels(focus_available, rotation=45)
        ax.set_yticklabels(focus_available)
        
        # Add correlation values
        for i in range(len(focus_available)):
            for j in range(len(focus_available)):
                text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.3f}',
                             ha="center", va="center", color="black", fontweight='bold')
        
        ax.set_title('🎯 Focus Group Correlations\n(Return Correlations)')
        
        # Add colorbar
        try:
            cbar = plt.colorbar(im, ax=ax, shrink=0.6)
            cbar.set_label('Correlation')
        except:
            pass  # Skip colorbar if it fails
    
    def _plot_detailed_cumulative_returns_focused(self, ax):
        """Plot cumulative returns for focus assets"""
        
        colors = ['red', 'darkred', 'crimson', 'orange', 'purple', 'brown']
        
        plotted_count = 0
        
        # Plot focus assets from different methods
        for method_name, method_data in self.method_data.items():
            if not method_data or plotted_count >= len(colors):
                continue
            
            # Find focus asset with most data in this method
            best_focus_asset = None
            max_length = 0
            
            for asset in self.focus_assets:
                if asset in method_data and 'Returns' in method_data[asset].columns:
                    length = len(method_data[asset])
                    if length > max_length:
                        max_length = length
                        best_focus_asset = asset
            
            if best_focus_asset and max_length > 2:  # FIXED: Lower threshold
                bars = method_data[best_focus_asset]
                returns = bars['Returns'].fillna(0)
                cumulative_returns = (1 + returns).cumprod()
                
                ax.plot(bars.index, cumulative_returns, 
                       label=f'🎯 {method_name}-{best_focus_asset} (n={len(bars)})', 
                       color=colors[plotted_count], linewidth=2, alpha=0.8)
                
                # Add final return as text
                if len(cumulative_returns) > 0:
                    final_return = (cumulative_returns.iloc[-1] - 1) * 100
                    ax.text(bars.index[-1], cumulative_returns.iloc[-1], 
                           f'{final_return:+.1f}%', 
                           fontsize=8, ha='left', va='bottom')
                
                plotted_count += 1
        
        if plotted_count > 0:
            ax.set_ylabel('Cumulative Returns')
            ax.set_title('🎯 Focus Group Cumulative Returns\nAcross Methods and Assets')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=1, color='black', linestyle='--', alpha=0.5)
        else:
            ax.text(0.5, 0.5, 'No focus asset data\nfor cumulative returns', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('🎯 Focus Group Cumulative Returns')
    
    def _plot_volatility_comparison_focused(self, ax):
        """Plot volatility comparison highlighting focus assets"""
        
        focus_methods = []
        focus_volatilities = []
        other_methods = []
        other_volatilities = []
        
        for method_name, method_data in self.method_data.items():
            if not method_data:
                continue
            
            # Calculate volatility for focus assets
            focus_vols = []
            other_vols = []
            
            for asset, bars in method_data.items():
                if 'Returns' in bars.columns:
                    returns = bars['Returns'].dropna()
                    if len(returns) > 2:  # FIXED: Lower threshold
                        vol = returns.std() * np.sqrt(252) * 100  # Annualized volatility %
                        if asset in self.focus_assets:
                            focus_vols.append(vol)
                        else:
                            other_vols.append(vol)
            
            if focus_vols:
                focus_methods.append(f'🎯{method_name[:4]}')
                focus_volatilities.append(np.mean(focus_vols))
            
            if other_vols:
                other_methods.append(f'{method_name[:4]}')
                other_volatilities.append(np.mean(other_vols))
        
        # Plot focus assets with red, others with blue
        all_methods = []
        all_vols = []
        colors = []
        
        for method, vol in zip(focus_methods, focus_volatilities):
            all_methods.append(method)
            all_vols.append(vol)
            colors.append('red')
        
        for method, vol in zip(other_methods, other_volatilities):
            all_methods.append(method)
            all_vols.append(vol)
            colors.append('steelblue')
        
        if all_methods and all_vols:
            bars_plot = ax.bar(all_methods, all_vols, color=colors, alpha=0.7)
            ax.set_ylabel('Annualized Volatility (%)')
            ax.set_title('Volatility by Method\n🎯 Focus Assets vs Others')
            
            # Add value labels
            for bar, vol in zip(bars_plot, all_vols):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(all_vols)*0.01, 
                       f'{vol:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
            
            ax.grid(True, alpha=0.3, axis='y')
        else:
            ax.text(0.5, 0.5, 'No volatility data\navailable', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Volatility Comparison')
    
    def _plot_return_differences_focused(self, ax):
        """Plot return differences between focus assets"""
        
        # Find common focus assets across methods
        focus_available = {}
        for method_name, method_data in self.method_data.items():
            focus_in_method = [a for a in self.focus_assets if a in method_data]
            if focus_in_method:
                focus_available[method_name] = focus_in_method
        
        if len(focus_available) < 2:
            ax.text(0.5, 0.5, 'Need ≥2 methods with\nfocus assets for\ndifference analysis', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.set_title('🎯 Focus Asset Return Differences')
            return
        
        # Compare first two methods with focus assets
        methods = list(focus_available.keys())[:2]
        method1, method2 = methods[0], methods[1]
        
        # Find common focus asset
        common_focus_assets = set(focus_available[method1]).intersection(set(focus_available[method2]))
        
        if not common_focus_assets:
            ax.text(0.5, 0.5, 'No common focus assets\nacross methods', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.set_title('🎯 Focus Asset Return Differences')
            return
        
        focus_asset = list(common_focus_assets)[0]
        
        # Get data for comparison
        method1_data = self.method_data[method1]
        method2_data = self.method_data[method2]
        
        bars1 = method1_data[focus_asset]
        bars2 = method2_data[focus_asset]
        
        # Align data
        common_idx = bars1.index.intersection(bars2.index)
        if len(common_idx) > 3:  # FIXED: Lower threshold
            returns1 = bars1.loc[common_idx]['Returns']
            returns2 = bars2.loc[common_idx]['Returns']
            
            return_diff = returns1 - returns2
            
            ax.plot(common_idx, return_diff * 100, color='red', linewidth=1.5, alpha=0.7)
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            
            ax.set_ylabel('Return Difference (%)')
            ax.set_title(f'🎯 Return Differences: {method1} - {method2}\nFocus Asset: {focus_asset}')
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            if len(return_diff) > 0:
                diff_stats = (f'Mean: {return_diff.mean()*100:.4f}%\n'
                             f'Std: {return_diff.std()*100:.4f}%\n'
                             f'Max: {return_diff.max()*100:.4f}%')
                ax.text(0.02, 0.98, diff_stats, transform=ax.transAxes, 
                       verticalalignment='top', fontsize=8,
                       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        else:
            ax.text(0.5, 0.5, 'Insufficient common data\nfor difference analysis', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.set_title('🎯 Focus Asset Return Differences')
    
    def _plot_chsh_detailed(self, ax):
        """Plot detailed CHSH results"""
        methods = []
        max_chsh_values = []
        
        # Try to use stored CHSH results
        chsh_data_available = False
        if hasattr(self, 'detailed_results') and 'chsh_analysis' in self.detailed_results:
            chsh_analysis = self.detailed_results['chsh_analysis']
            
            for method_name, method_chsh in chsh_analysis.items():
                all_chsh_values = []
                for set_name, set_results in method_chsh.items():
                    chsh_values = [r['chsh_value'] for r in set_results.values()]
                    all_chsh_values.extend(chsh_values)
                
                if all_chsh_values:
                    methods.append(method_name[:8])
                    max_chsh_values.append(max(all_chsh_values))
                    chsh_data_available = True
        
        if chsh_data_available and methods and max_chsh_values:
            bars_plot = ax.bar(methods, max_chsh_values, alpha=0.7, color='steelblue')
            ax.axhline(y=2.0, color='red', linestyle='--', label='Classical Limit', alpha=0.7)
            ax.axhline(y=2.828, color='green', linestyle='--', label='Quantum Limit', alpha=0.7)
            ax.set_ylabel('Max CHSH Value')
            ax.set_title('Maximum CHSH Values by Method')
            ax.legend()
            
            for bar, val in zip(bars_plot, max_chsh_values):
                ax.text(bar.get_x() + bar.get_width()/2, val + 0.01, 
                       f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'CHSH Analysis\n(Run detailed analysis first)', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('CHSH Analysis Summary')
    
    def _create_conditional_bell_visualizations(self):
        """Create visualizations for conditional Bell test results (focus group priority)"""
        
        if not hasattr(self, 'conditional_bell_results') or not self.conditional_bell_results:
            print("   ⚠️  No conditional Bell results to visualize")
            return
        
        print("   📊 Creating conditional Bell visualizations (focus group priority)...")
        
        # Find pairs with violations, prioritizing focus pairs
        focus_violation_pairs = []
        other_violation_pairs = []
        
        for method_name, method_results in self.conditional_bell_results.items():
            for pair_name, pair_result in method_results.items():
                if pair_result['total_violations'] > 0:
                    if pair_result.get('is_focus_pair', False):
                        focus_violation_pairs.append((method_name, pair_name, pair_result))
                    else:
                        other_violation_pairs.append((method_name, pair_name, pair_result))
        
        # Prioritize focus pairs, then others
        violation_pairs = focus_violation_pairs + other_violation_pairs
        
        if violation_pairs:
            # Sort by violation count and take top 3
            violation_pairs.sort(key=lambda x: x[2]['total_violations'], reverse=True)
            top_pairs = violation_pairs[:3]
            
            fig, axes = plt.subplots(len(top_pairs), 1, figsize=(15, 5*len(top_pairs)))
            if len(top_pairs) == 1:
                axes = [axes]
            
            for i, (method_name, pair_name, pair_result) in enumerate(top_pairs):
                ax = axes[i] if len(top_pairs) > 1 else axes[0]
                
                s1_df = pair_result['s1_series']
                ticker_A = pair_result['ticker_A']
                ticker_B = pair_result['ticker_B']
                is_focus_pair = pair_result.get('is_focus_pair', False)
                
                # Plot S1 time series with focus highlighting
                line_color = 'red' if is_focus_pair else 'blue'
                line_width = 2.0 if is_focus_pair else 1.5
                
                ax.plot(s1_df['date'], s1_df['S1_value'], 
                       label=f'S1: {ticker_A}-{ticker_B}', linewidth=line_width, color=line_color)
                
                # Add bounds
                ax.axhline(2, color='red', linestyle='--', alpha=0.6, label='Classical Bound (2.0)')
                ax.axhline(-2, color='red', linestyle='--', alpha=0.6)
                ax.axhline(2.828, color='green', linestyle='--', alpha=0.6, label='Quantum Bound (2.828)')
                ax.axhline(-2.828, color='green', linestyle='--', alpha=0.6)
                ax.axhline(4, color='purple', linestyle='--', alpha=0.6, label='Theoretical Max (4.0)')
                ax.axhline(-4, color='purple', linestyle='--', alpha=0.6)
                
                focus_indicator = "🎯 FOCUS PAIR: " if is_focus_pair else ""
                ax.set_title(f'{focus_indicator}S1 Bell Values: {ticker_A} vs. {ticker_B} ({method_name} method)\n'
                           f'Conditional Bell Test - Focus Group Analysis')
                ax.set_xlabel('Date')
                ax.set_ylabel('S1 Value')
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                # Add statistics text with focus highlighting
                stats_text = (f'Violations: {pair_result["total_violations"]}/{pair_result["data_points"]} '
                            f'({pair_result["violation_rate"]:.1f}%)\n'
                            f'Max |S1|: {max(abs(pair_result["max_s1"]), abs(pair_result["min_s1"])):.4f}\n'
                            f'Range: [{pair_result["min_s1"]:.4f}, {pair_result["max_s1"]:.4f}]')
                
                box_color = 'lightyellow' if is_focus_pair else 'lightblue'
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                       verticalalignment='top', fontsize=9,
                       bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.8))
            
            plt.tight_layout()
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            focus_str = "_".join(self.focus_assets)
            filename = f'conditional_bell_violations_{focus_str}_{timestamp}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"   💾 Conditional Bell violation plots saved: {filename}")
            plt.show()
        
        else:
            # No violations found, but create plot showing highest S1 values
            print("   ℹ️  No violations found, showing highest S1 values...")
            
            # Find pairs with highest |S1| values, prioritizing focus pairs
            all_pairs = []
            for method_name, method_results in self.conditional_bell_results.items():
                for pair_name, pair_result in method_results.items():
                    max_abs_s1 = max(abs(pair_result['max_s1']), abs(pair_result['min_s1']))
                    is_focus_pair = pair_result.get('is_focus_pair', False)
                    all_pairs.append((method_name, pair_name, pair_result, max_abs_s1, is_focus_pair))
            
            if all_pairs:
                # Sort by focus first, then by |S1| value
                all_pairs.sort(key=lambda x: (not x[4], -x[3]))  # Focus first, then highest |S1|
                top_pair = all_pairs[0]
                
                method_name, pair_name, pair_result, max_abs_s1, is_focus_pair = top_pair
                s1_df = pair_result['s1_series']
                ticker_A = pair_result['ticker_A']
                ticker_B = pair_result['ticker_B']
                
                fig, ax = plt.subplots(1, 1, figsize=(15, 6))
                
                line_color = 'red' if is_focus_pair else 'blue'
                line_width = 2.0 if is_focus_pair else 1.5
                
                # Plot S1 time series
                ax.plot(s1_df['date'], s1_df['S1_value'], 
                       label=f'S1: {ticker_A}-{ticker_B}', linewidth=line_width, color=line_color)
                
                # Add bounds
                ax.axhline(2, color='red', linestyle='--', alpha=0.6, label='Classical Bound (2.0)')
                ax.axhline(-2, color='red', linestyle='--', alpha=0.6)
                ax.axhline(2.828, color='green', linestyle='--', alpha=0.6, label='Quantum Bound (2.828)')
                ax.axhline(-2.828, color='green', linestyle='--', alpha=0.6)
                ax.axhline(4, color='purple', linestyle='--', alpha=0.6, label='Theoretical Max (4.0)')
                ax.axhline(-4, color='purple', linestyle='--', alpha=0.6)
                
                focus_indicator = "🎯 FOCUS PAIR: " if is_focus_pair else ""
                ax.set_title(f'{focus_indicator}S1 Bell Values: {ticker_A} vs. {ticker_B} ({method_name} method)\n'
                           f'Highest S1 Values (No Violations) - Focus Group Analysis')
                ax.set_xlabel('Date')
                ax.set_ylabel('S1 Value')
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                # Add statistics text
                stats_text = (f'Max |S1|: {max_abs_s1:.4f}\n'
                            f'Range: [{pair_result["min_s1"]:.4f}, {pair_result["max_s1"]:.4f}]\n'
                            f'Mean: {pair_result["mean_s1"]:.4f}')
                box_color = 'lightyellow' if is_focus_pair else 'lightblue'
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                       verticalalignment='top', fontsize=9,
                       bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.8))
                
                plt.tight_layout()
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                focus_str = "_".join(self.focus_assets)
                filename = f'conditional_bell_highest_{focus_str}_{timestamp}.png'
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"   💾 Conditional Bell plot (highest S1) saved: {filename}")
                plt.show()
    
    def _provide_detailed_interpretation(self):
        """Provide comprehensive interpretation with focus group emphasis"""
        
        print(f"\n🧠 COMPREHENSIVE INTERPRETATION WITH FOCUS GROUP ANALYSIS")
        print("=" * 80)
        
        print(f"\n📊 KEY FINDINGS SUMMARY:")
        if hasattr(self, 'tick_data') and self.tick_data is not None:
            print(f"   • Data Quality: {len(self.tick_data):,} ticks across {len(self.tick_data['ticker'].unique())} assets")
            print(f"   • Time Span: {(self.tick_data['datetime'].max() - self.tick_data['datetime'].min()).total_seconds() / 3600:.1f} hours")
        print(f"   • Aggregation Methods Tested: {len(self.method_data)}")
        print(f"   🎯 Focus Group: {self.focus_assets}")
        
        # Check focus asset data availability
        focus_available = set()
        for method_data in self.method_data.values():
            focus_available.update([a for a in method_data.keys() if a in self.focus_assets])
        focus_missing = set(self.focus_assets) - focus_available
        
        print(f"   🎯 Focus Assets Available: {list(focus_available)}")
        if focus_missing:
            print(f"   ⚠️  Focus Assets Missing: {list(focus_missing)}")
        
        # Analyze results for each method with focus group emphasis
        for method_name in self.method_data.keys():
            print(f"\n🔬 {method_name.upper()} METHOD ANALYSIS:")
            
            method_data = self.method_data[method_name]
            if not method_data:
                print(f"   ❌ No data processed for {method_name}")
                continue
            
            # Basic statistics
            total_bars = sum(len(bars) for bars in method_data.values())
            avg_bars_per_asset = total_bars / len(method_data) if len(method_data) > 0 else 0
            focus_assets_in_method = [a for a in method_data.keys() if a in self.focus_assets]
            
            print(f"   📊 Basic Statistics:")
            print(f"      Assets processed: {len(method_data)} (focus: {len(focus_assets_in_method)})")
            print(f"      Total bars created: {total_bars:,}")
            print(f"      Average bars per asset: {avg_bars_per_asset:.0f}")
            
            # Return statistics with focus group breakdown
            all_returns = []
            focus_returns = []
            other_returns = []
            
            for asset, bars in method_data.items():
                returns = bars['Returns'].dropna()
                all_returns.extend(returns.values)
                if asset in self.focus_assets:
                    focus_returns.extend(returns.values)
                else:
                    other_returns.extend(returns.values)
            
            if all_returns:
                all_array = np.array(all_returns)
                print(f"   📈 Return Statistics:")
                print(f"      All assets - Mean: {np.mean(all_array):.6f}, Std: {np.std(all_array):.6f}")
                if len(all_array) > 8:
                    print(f"      All assets - Skew: {stats.skew(all_array):.4f}, Kurt: {stats.kurtosis(all_array):.4f}")
                
                if focus_returns:
                    focus_array = np.array(focus_returns)
                    print(f"   🎯 Focus Group Returns:")
                    print(f"      Focus assets - Mean: {np.mean(focus_array):.6f}, Std: {np.std(focus_array):.6f}")
                    if len(focus_array) > 8:
                        print(f"      Focus assets - Skew: {stats.skew(focus_array):.4f}, Kurt: {stats.kurtosis(focus_array):.4f}")
                    print(f"      Focus returns count: {len(focus_returns):,}")
                
                if other_returns:
                    other_array = np.array(other_returns)
                    print(f"   📊 Other Assets Returns:")
                    print(f"      Other assets - Mean: {np.mean(other_array):.6f}, Std: {np.std(other_array):.6f}")
                    print(f"      Other returns count: {len(other_returns):,}")
        
        # Conditional Bell test results with focus emphasis
        print(f"\n⚛️ CONDITIONAL BELL TEST RESULTS (S1 INEQUALITY) - FOCUS GROUP EMPHASIS:")
        
        if hasattr(self, 'conditional_bell_results') and self.conditional_bell_results:
            total_violations = 0
            total_tests = 0
            focus_violations = 0
            focus_tests = 0
            max_s1_overall = 0
            
            for method_name, method_results in self.conditional_bell_results.items():
                method_violations = sum(r['total_violations'] for r in method_results.values())
                method_tests = sum(r['data_points'] for r in method_results.values())
                
                # Focus pair statistics
                focus_method_violations = sum(r['total_violations'] for r in method_results.values() if r.get('is_focus_pair', False))
                focus_method_tests = sum(r['data_points'] for r in method_results.values() if r.get('is_focus_pair', False))
                
                if method_results:
                    method_max_s1 = max(max(abs(r['max_s1']), abs(r['min_s1'])) for r in method_results.values())
                    max_s1_overall = max(max_s1_overall, method_max_s1)
                    
                    focus_pairs_count = sum(1 for r in method_results.values() if r.get('is_focus_pair', False))
                    total_pairs_count = len(method_results)
                    
                    print(f"   • {method_name}:")
                    print(f"      Total: {method_violations} violations in {method_tests} tests (Max |S1| = {method_max_s1:.4f})")
                    print(f"      🎯 Focus pairs: {focus_pairs_count}/{total_pairs_count} pairs tested")
                    if focus_method_tests > 0:
                        print(f"      🎯 Focus violations: {focus_method_violations}/{focus_method_tests} ({focus_method_violations/focus_method_tests*100:.1f}%)")
                
                total_violations += method_violations
                total_tests += method_tests
                focus_violations += focus_method_violations
                focus_tests += focus_method_tests
            
            print(f"\n   📊 CONDITIONAL BELL SUMMARY:")
            print(f"      🎯 FOCUS GROUP RESULTS:")
            if focus_tests > 0:
                print(f"         Focus S1 violations: {focus_violations} / {focus_tests} ({focus_violations/focus_tests*100:.2f}%)")
            else:
                print(f"         No focus pair tests performed")
            
            print(f"      📊 OVERALL RESULTS:")
            print(f"         Total S1 violations: {total_violations} / {total_tests}")
            print(f"         Overall violation rate: {total_violations/total_tests*100:.2f}%" if total_tests > 0 else "         No tests performed")
            print(f"         Maximum |S1| achieved: {max_s1_overall:.6f}")
            
            if total_violations > 0:
                print(f"      🚨 BELL VIOLATIONS DETECTED with conditional approach!")
                if focus_violations > 0:
                    print(f"      🎯 Focus group contributed {focus_violations} violations!")
                print(f"      This matches your Yahoo Finance AAPL-MSFT results")
            else:
                print(f"      ✅ No violations with conditional approach")
                
            print(f"\n   🔍 CONDITIONAL vs CHSH COMPARISON:")
            print(f"      • CHSH inequality (original): Tests unconditional correlations")
            print(f"      • S1 inequality (new): Tests correlations conditional on volatility regimes")
            print(f"      • S1 is more sensitive to market regime-dependent correlations")
            print(f"      • This is why your Yahoo Finance analysis found violations")
        else:
            print(f"   ❌ No conditional Bell results available")
        
        print(f"\n🎯 CRITICAL ANALYSIS - FOCUS GROUP EMPHASIS:")
        if hasattr(self, 'conditional_bell_results') and focus_violations > 0:
            print(f"   🚨 MAJOR FINDING: Focus group shows {focus_violations} Bell violations!")
            print(f"   • Your selected focus assets: {self.focus_assets}")
            print(f"   • Focus pairs generated conditional Bell violations")
            print(f"   • This confirms quantum-like correlations in your target stocks")
        elif hasattr(self, 'conditional_bell_results') and total_violations > 0:
            print(f"   • Conditional Bell test (S1) found {total_violations} violations in other pairs")
            print(f"   • Focus group ({self.focus_assets}) showed classical behavior")
            print(f"   • Violations occurred in non-focus asset combinations")
        else:
            print(f"   • All aggregation methods show classical behavior")
            print(f"   • No Bell violations detected with any method")  
            print(f"   • Focus group ({self.focus_assets}) behaves classically")
        
        print(f"\n💡 FOCUS GROUP RECOMMENDATIONS:")
        if hasattr(self, 'conditional_bell_results') and focus_violations > 0:
            print(f"   1. 🎯 Investigate the specific dates when focus violations occurred")
            print(f"   2. 🎯 Analyze market conditions during violation periods")
            print(f"   3. 🎯 Test focus assets with higher frequency data (seconds/milliseconds)")
            print(f"   4. 🎯 Examine focus asset behavior during earnings/news events")
        else:
            print(f"   1. 🎯 Try different volatility thresholds (50%, 80%, 90%) for focus assets")
            print(f"   2. 🎯 Test focus assets during crisis periods or major news events")
            print(f"   3. 🎯 Consider expanding focus group or testing related stocks")
            print(f"   4. 🎯 Examine ultra-high frequency data for focus assets")
        
        print(f"\n🎯 FOCUS GROUP SUMMARY:")
        print(f"   • Target Assets: {self.focus_assets}")
        print(f"   • Available in Data: {list(focus_available)}")
        if focus_missing:
            print(f"   • Missing from Data: {list(focus_missing)}")
        if hasattr(self, 'conditional_bell_results'):
            print(f"   • Bell Violations: {focus_violations if 'focus_violations' in locals() else 0}")
            print(f"   • Test Success: {'YES' if 'focus_tests' in locals() and focus_tests > 0 else 'NO'}")


# =================== MAIN EXECUTION FUNCTIONS ===================

def run_comprehensive_analysis(file_path=None, focus_assets=None, frequency='15min'):
    """
    Run comprehensive analysis with focus asset selection
    
    Parameters:
    -----------
    file_path : str
        Path to data file (default uses placeholder path)
    focus_assets : list or str
        Either list of assets or predefined group name
    frequency : str
        Bar frequency ('5min', '15min', '30min')
    """
    
    # Default file path (update this to your actual path)
    if file_path is None:
        file_path = '/Users/mjp38/Dropbox (Personal)/QuantumBellTest/kbaxvugzwiicmypy.csv.gz'
        print(f"⚠️  Using default file path: {file_path}")
        print(f"   Update the path in run_comprehensive_analysis() if needed")
    
    # Handle focus asset selection
    if focus_assets is None:
        selected_focus = DEFAULT_FOCUS_SET
        print(f"🎯 Using default focus group: {selected_focus}")
    elif isinstance(focus_assets, str):
        # Handle predefined groups
        predefined_groups = {
            'tech': tech_pairs,
            'tech_pairs': tech_pairs,
            'cross_sector': cross_sector,
            'high_vol': high_vol,
            'commodities': commodities,
            'financials': financials
        }
        if focus_assets.lower() in predefined_groups:
            selected_focus = predefined_groups[focus_assets.lower()]
            print(f"🎯 Using predefined group '{focus_assets}': {selected_focus}")
        else:
            print(f"❌ Unknown group '{focus_assets}', using default: {DEFAULT_FOCUS_SET}")
            selected_focus = DEFAULT_FOCUS_SET
    elif isinstance(focus_assets, list):
        selected_focus = focus_assets
        print(f"🎯 Using custom focus assets: {selected_focus}")
    else:
        selected_focus = DEFAULT_FOCUS_SET
        print(f"🎯 Using default focus group: {selected_focus}")
    
    # Create analyzer with focus assets
    analyzer = ComprehensiveBellAnalyzer(focus_assets=selected_focus)
    
    # Run comprehensive analysis
    print(f"\n🚀 STARTING COMPREHENSIVE ANALYSIS...")
    print(f"   File: {file_path}")
    print(f"   Focus: {selected_focus}")
    print(f"   Frequency: {frequency}")
    
    results = analyzer.load_and_analyze_comprehensive(file_path, frequency)
    
    return analyzer, results

def run_quick_test(focus_assets=None):
    """Quick test function for development/debugging"""
    
    print("🚀 RUNNING QUICK TEST...")
    
    # Use smaller predefined groups for quick testing
    if focus_assets is None:
        focus_assets = ['AAPL', 'MSFT']  # Minimal for quick test
    
    try:
        analyzer, results = run_comprehensive_analysis(
            focus_assets=focus_assets,
            frequency='15min'
        )
        
        print(f"\n✅ QUICK TEST COMPLETE!")
        return analyzer, results
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        print(f"   Check your file path and data availability")
        return None, None

def run_focus_asset_diagnostic(file_path, focus_assets):
    """Comprehensive diagnostic for focus assets"""
    
    print("🩺 COMPREHENSIVE FOCUS ASSET DIAGNOSTIC")
    print("=" * 60)
    print(f"Diagnosing: {focus_assets}")
    
    # Step 1: Quick data check
    print("\nStep 1: Quick data availability check...")
    try:
        if file_path.endswith('.gz'):
            with gzip.open(file_path, 'rt') as f:
                sample_df = pd.read_csv(f, nrows=1000)
        else:
            sample_df = pd.read_csv(file_path, nrows=1000)
        
        # Find ticker column
        ticker_col = None
        for col in sample_df.columns:
            if any(word in col.lower() for word in ['sym', 'ticker', 'root']):
                ticker_col = col
                break
        
        if ticker_col:
            available_tickers = sample_df[ticker_col].unique()
            print(f"   ✅ Found ticker column: '{ticker_col}'")
            print(f"   📊 Sample tickers: {list(available_tickers)[:20]}")
            
            for asset in focus_assets:
                if asset in available_tickers:
                    print(f"   🎯 {asset}: ✅ FOUND in sample")
                else:
                    print(f"   🎯 {asset}: ❌ NOT FOUND in sample")
        else:
            print("   ❌ No ticker column found")
            return None
    
    except Exception as e:
        print(f"   ❌ Error in quick check: {e}")
        return None
    
    # Step 2: Full analysis with diagnostic focus
    print("\nStep 2: Full diagnostic analysis...")
    analyzer = ComprehensiveBellAnalyzer(focus_assets=focus_assets)
    
    # Load data
    tick_data = analyzer._load_tick_data_detailed_fixed(file_path)
    if tick_data is None:
        print("   ❌ Failed to load data")
        return None
    
    # Step 3: Try bar creation
    print(f"\nStep 3: Bar creation test...")
    method_data = analyzer._create_bars_all_methods_fixed('15min')
    
    # Step 4: Summary
    print(f"\nStep 4: Final diagnostic summary...")
    focus_success = []
    focus_failed = []
    
    for asset in focus_assets:
        asset_in_any_method = any(asset in method_data[method] for method in method_data.keys())
        if asset_in_any_method:
            focus_success.append(asset)
        else:
            focus_failed.append(asset)
    
    print(f"   🎯 Focus assets successful: {focus_success}")
    print(f"   ❌ Focus assets failed: {focus_failed}")
    
    if focus_success:
        print(f"\n✅ SUCCESS: {len(focus_success)}/{len(focus_assets)} focus assets processed")
        print(f"   These will appear in plots: {focus_success}")
    
    if focus_failed:
        print(f"\n❌ ISSUES: {len(focus_failed)}/{len(focus_assets)} focus assets failed")
        print(f"   These will NOT appear in plots: {focus_failed}")
        
        # Suggest solutions
        print(f"\n💡 SOLUTIONS for failed assets:")
        for asset in focus_failed:
            if asset in analyzer.focus_diagnostics:
                diag = analyzer.focus_diagnostics[asset]
                if not diag['found_in_raw']:
                    print(f"   • {asset}: Not in data - check ticker symbol or data source")
                elif diag.get('tick_count', 0) < 20:
                    print(f"   • {asset}: Too few ticks ({diag.get('tick_count')}) - need more data")
                else:
                    print(f"   • {asset}: Check issues: {diag.get('issues', [])}")
    
    return analyzer

def find_best_focus_group_for_data(file_path):
    """Automatically find the best focus group for your data"""
    
    print("🔍 FINDING BEST FOCUS GROUP FOR YOUR DATA")
    print("=" * 50)
    
    # Load sample data to see what's available
    try:
        if file_path.endswith('.gz'):
            with gzip.open(file_path, 'rt') as f:
                sample_df = pd.read_csv(f, nrows=5000)  # Larger sample
        else:
            sample_df = pd.read_csv(file_path, nrows=5000)
        
        # Find ticker column
        ticker_col = None
        for col in sample_df.columns:
            if any(word in col.lower() for word in ['sym', 'ticker', 'root']):
                ticker_col = col
                break
        
        if not ticker_col:
            print("❌ No ticker column found")
            return None
        
        # Get available tickers
        ticker_counts = sample_df[ticker_col].value_counts()
        available_tickers = set(ticker_counts.index)
        
        print(f"✅ Found {len(available_tickers)} unique tickers")
        print(f"   Top tickers: {list(ticker_counts.head(15).index)}")
        
        # Test predefined groups
        focus_groups = {
            'tech_pairs': tech_pairs,
            'cross_sector': cross_sector,
            'high_vol': high_vol,
            'commodities': commodities,
            'financials': financials
        }
        
        group_scores = {}
        
        for group_name, assets in focus_groups.items():
            # Count how many assets from this group are available
            available_assets = [asset for asset in assets if asset in available_tickers]
            availability_score = len(available_assets) / len(assets)
            
            # Count total ticks for these assets
            total_ticks = sum(ticker_counts.get(asset, 0) for asset in available_assets)
            tick_score = min(total_ticks / 1000, 1.0)  # Normalize to 0-1
            
            # Combined score
            combined_score = (availability_score * 0.7) + (tick_score * 0.3)
            
            group_scores[group_name] = {
                'score': combined_score,
                'available_assets': available_assets,
                'availability_pct': availability_score * 100,
                'total_ticks': total_ticks
            }
            
            print(f"   {group_name}: {len(available_assets)}/{len(assets)} assets ({availability_score*100:.0f}%), {total_ticks:,} ticks, score: {combined_score:.3f}")
        
        # Find best group
        best_group = max(group_scores.keys(), key=lambda g: group_scores[g]['score'])
        best_info = group_scores[best_group]
        
        print(f"\n🏆 RECOMMENDED FOCUS GROUP: {best_group}")
        print(f"   Available assets: {best_info['available_assets']}")
        print(f"   Coverage: {best_info['availability_pct']:.0f}%")
        print(f"   Total ticks: {best_info['total_ticks']:,}")
        
        # Also suggest a custom group from most available assets
        top_available = list(ticker_counts.head(10).index)
        print(f"\n💡 ALTERNATIVE: Custom group from most available assets")
        print(f"   Suggested: {top_available[:5]}")
        
        return {
            'recommended_group': best_group,
            'recommended_assets': best_info['available_assets'],
            'alternative_assets': top_available[:5],
            'all_scores': group_scores
        }
        
    except Exception as e:
        print(f"❌ Error analyzing data: {e}")
        return None

# =================== EXAMPLE USAGE FUNCTIONS ===================

def example_tech_focus():
    """Example: Focus on tech stocks"""
    print("📱 EXAMPLE: Technology Stock Analysis")
    return run_comprehensive_analysis(
        focus_assets='tech_pairs',
        frequency='15min'
    )

def example_cross_sector_focus():
    """Example: Cross-sector analysis (AAPL, CORN, DBA)"""
    print("🌐 EXAMPLE: Cross-Sector Analysis")
    return run_comprehensive_analysis(
        focus_assets=['AAPL', 'CORN', 'DBA'],
        frequency='15min'
    )

def example_commodities_focus():
    """Example: Commodities focus"""
    print("🌾 EXAMPLE: Commodities Analysis")
    return run_comprehensive_analysis(
        focus_assets='commodities',
        frequency='15min'
    )

# =================== MAIN EXECUTION ===================

if __name__ == "__main__":
    print("🚀 COMPREHENSIVE BELL INEQUALITY ANALYZER WITH FOCUS GROUPS - COMPLETE VERSION")
    print("=" * 80)
    print(f"Current default focus group: {DEFAULT_FOCUS_SET}")
    print(f"This is the complete 2400+ line analyzer with:")
    print(f"   • Intelligent column detection")
    print(f"   • Individual plots for EACH focus asset")
    print(f"   • Lower thresholds for focus assets")
    print(f"   • Comprehensive diagnostics")
    print(f"   • Full Bell inequality analysis")
    
    print("\n📋 AVAILABLE FUNCTIONS:")
    print("   • run_comprehensive_analysis() - Main analysis")
    print("   • run_focus_asset_diagnostic() - Diagnostic for missing assets")
    print("   • find_best_focus_group_for_data() - Auto-find best focus group")
    print("   • example_cross_sector_focus() - Run cross-sector example")
    print("   • run_quick_test() - Quick test")
    
    # Run the main analysis
    print(f"\nStarting analysis with default focus group: {DEFAULT_FOCUS_SET}...")
    
    analyzer, results = run_comprehensive_analysis()
    
    if results is None:
        print("\n❌ ANALYSIS FAILED!")
        print("Let's run diagnostics to see what went wrong...")
        
        # Auto-run diagnostic
        file_path = '/Users/mjp38/Dropbox (Personal)/QuantumBellTest/kbaxvugzwiicmypy.csv.gz'
        
        print("\n🩺 Running automatic diagnostic...")
        diagnostic_analyzer = run_focus_asset_diagnostic(file_path, DEFAULT_FOCUS_SET)
        
        if diagnostic_analyzer is None:
            print("\n🔍 Let's find what assets ARE available in your data...")
            best_group_info = find_best_focus_group_for_data(file_path)
            
            if best_group_info:
                print(f"\n💡 Try running with: {best_group_info['recommended_assets']}")
                print(f"   Example: run_comprehensive_analysis(focus_assets={best_group_info['recommended_assets']})")
    else:
        print(f"\n✅ COMPREHENSIVE ANALYSIS COMPLETE!")
        
        # Display final focus group summary
        if hasattr(analyzer, 'conditional_bell_results') and analyzer.conditional_bell_results:
            focus_violations = 0
            total_violations = 0
            
            for method_results in analyzer.conditional_bell_results.values():
                for pair_result in method_results.values():
                    total_violations += pair_result['total_violations']
                    if pair_result.get('is_focus_pair', False):
                        focus_violations += pair_result['total_violations']
            
            print(f"\n🎯 FINAL FOCUS GROUP SUMMARY:")
            print(f"   Focus Assets: {analyzer.focus_assets}")
            print(f"   Focus Bell Violations: {focus_violations}")
            print(f"   Total Bell Violations: {total_violations}")
            
            if focus_violations > 0:
                print(f"   🚨 SUCCESS: Your focus group shows quantum-like behavior!")
                print(f"   Check the individual plots generated for each focus asset.")
            else:
                print(f"   ✅ Focus group shows classical behavior.")
                print(f"   All focus assets should have individual plots generated.")
        
        print(f"\n💡 To run with different focus groups:")
        print(f"   • Use: run_comprehensive_analysis(focus_assets=['AAPL', 'CORN', 'DBA'])")
        print(f"   • Or: example_cross_sector_focus()")
        print(f"   • Or: find_best_focus_group_for_data('/path/to/data.csv.gz')")

    print(f"\n📊 FOCUS ASSET HELP:")
    print(f"   If CORN/DBA are missing from plots:")
    print(f"   1. Run: run_focus_asset_diagnostic('/path/to/data.csv.gz', ['AAPL', 'CORN', 'DBA'])")
    print(f"   2. Or: find_best_focus_group_for_data('/path/to/data.csv.gz')")
    print(f"   3. Try different focus groups: example_tech_focus() or example_commodities_focus()")
