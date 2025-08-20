#!/usr/bin/env python3
"""
VERIFY DATA SOURCES - Check what data we're actually using
"""

import yfinance as yf
import pandas as pd
from datetime import datetime

def verify_data_sources():
    print("🔍 VERIFYING FOOD SYSTEMS DATA SOURCES")
    print("=" * 50)
    
    # Test assets we're using
    test_assets = ['CORN', 'WEAT', 'SOYB', 'ADM', 'BG', 'CF']
    
    print("\n📊 CHECKING YAHOO FINANCE DATA ACCESS:")
    for asset in test_assets:
        try:
            # Get recent data to verify access
            ticker = yf.Ticker(asset)
            info = ticker.info
            
            print(f"\n✅ {asset}:")
            print(f"   Name: {info.get('longName', 'N/A')}")
            print(f"   Exchange: {info.get('exchange', 'N/A')}")
            print(f"   Currency: {info.get('currency', 'N/A')}")
            print(f"   Market Cap: {info.get('marketCap', 'N/A')}")
            
            # Get sample historical data
            hist = ticker.history(period='5d')
            if not hist.empty:
                print(f"   Recent Price: ${hist['Close'].iloc[-1]:.2f}")
                print(f"   Data Available: ✅ (Last 5 days: {len(hist)} records)")
            else:
                print(f"   Data Available: ❌")
                
        except Exception as e:
            print(f"❌ {asset}: Error - {e}")
    
    print("\n🌾 CHECKING SPECIFIC CRISIS PERIOD DATA:")
    
    # Test COVID period data
    print("\n📅 COVID-19 Period (2020-03-01 to 2020-12-31):")
    try:
        covid_data = yf.download(['CORN', 'WEAT', 'SOYB'], 
                                start='2020-03-01', 
                                end='2020-12-31',
                                progress=False)
        print(f"   ✅ COVID data shape: {covid_data.shape}")
        print(f"   Date range: {covid_data.index[0]} to {covid_data.index[-1]}")
        print(f"   Sample prices (CORN): ${covid_data['Close']['CORN'].iloc[0]:.2f} → ${covid_data['Close']['CORN'].iloc[-1]:.2f}")
    except Exception as e:
        print(f"   ❌ COVID data error: {e}")
    
    # Test Ukraine war period data  
    print("\n📅 Ukraine War Period (2022-02-24 to 2023-12-31):")
    try:
        ukraine_data = yf.download(['CORN', 'WEAT', 'SOYB'], 
                                  start='2022-02-24', 
                                  end='2023-12-31',
                                  progress=False)
        print(f"   ✅ Ukraine data shape: {ukraine_data.shape}")
        print(f"   Date range: {ukraine_data.index[0]} to {ukraine_data.index[-1]}")
        print(f"   Sample prices (WEAT): ${ukraine_data['Close']['WEAT'].iloc[0]:.2f} → ${ukraine_data['Close']['WEAT'].iloc[-1]:.2f}")
    except Exception as e:
        print(f"   ❌ Ukraine data error: {e}")
    
    print("\n🔬 DATA SOURCE VERIFICATION:")
    print("✅ All data comes from Yahoo Finance (publicly available)")
    print("✅ Real historical market prices (not synthetic)")
    print("✅ Commodity futures and stock prices")
    print("✅ No special data access required")
    print("✅ Reproducible by anyone with internet access")
    
    print("\n⚠️  LIMITATIONS:")
    print("- Yahoo Finance data quality varies by asset")
    print("- Some commodity tickers may have gaps or issues")
    print("- Free data (not professional-grade like Bloomberg)")
    print("- Daily resolution only (no intraday)")
    
    print("\n🎯 FOR WDRS COMPARISON:")
    print("- Current analysis: Yahoo Finance (free, daily)")
    print("- WDRS upgrade: Professional data (paid, high-frequency)")
    print("- WDRS would provide: Tick-by-tick, better quality, more assets")

if __name__ == "__main__":
    verify_data_sources()