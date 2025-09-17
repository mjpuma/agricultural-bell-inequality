#!/usr/bin/env python3
"""
Agricultural Universe Management System Demo

This script demonstrates how to use the Agricultural Universe Management System
for cross-sectoral Bell inequality analysis.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from agricultural_universe_manager import AgriculturalUniverseManager, MarketCapCategory, ExposureLevel, Tier


def demonstrate_agricultural_universe():
    """Demonstrate the Agricultural Universe Management System capabilities."""
    
    print("🌾 AGRICULTURAL UNIVERSE MANAGEMENT SYSTEM DEMO")
    print("=" * 60)
    
    # Initialize the manager
    manager = AgriculturalUniverseManager()
    
    # Show universe overview
    summary = manager.get_universe_summary()
    print(f"\n📊 Universe Overview:")
    print(f"   Total Companies: {summary['total_companies']}")
    print(f"   Agricultural Companies: {summary['by_tier']['agricultural']}")
    print(f"   Cross-Sector Companies: {summary['total_companies'] - summary['by_tier']['agricultural']}")
    
    # Demonstrate tier-based analysis
    print(f"\n🎯 Tier-Based Analysis:")
    print(f"   Tier 1 (Energy/Transport/Chemicals): {summary['by_tier']['tier_1_energy_transport_chemicals']} companies")
    print(f"   Tier 2 (Finance/Equipment): {summary['by_tier']['tier_2_finance_equipment']} companies")
    print(f"   Tier 3 (Policy-linked): {summary['by_tier']['tier_3_policy_linked']} companies")
    
    # Show key agricultural companies by exposure level
    print(f"\n🌱 Key Agricultural Companies by Exposure:")
    
    primary_ag = [ticker for ticker in manager.classify_by_tier(0) 
                  if manager.get_company_info(ticker).exposure == ExposureLevel.PRIMARY]
    
    print(f"   Primary Exposure ({len(primary_ag)} companies):")
    for ticker in primary_ag[:8]:  # Show top 8
        company = manager.get_company_info(ticker)
        print(f"     {ticker}: {company.name} ({company.sector})")
    
    # Demonstrate cross-sector pairing for Tier 1 analysis
    print(f"\n⚡ Tier 1 Cross-Sector Analysis Pairs:")
    tier1_pairs = manager.get_cross_sector_pairs(1)
    print(f"   Total Tier 1 pairs: {len(tier1_pairs)}")
    
    # Show examples of each transmission type
    energy_pairs = []
    transport_pairs = []
    chemical_pairs = []
    
    for pair in tier1_pairs:
        tier_company = manager.get_company_info(pair[1])
        if tier_company.sector == "Energy" and len(energy_pairs) < 3:
            energy_pairs.append(pair)
        elif tier_company.sector == "Transportation" and len(transport_pairs) < 3:
            transport_pairs.append(pair)
        elif tier_company.sector == "Chemicals" and len(chemical_pairs) < 3:
            chemical_pairs.append(pair)
    
    print(f"\n   Energy → Agriculture pairs:")
    for pair in energy_pairs:
        ag_company = manager.get_company_info(pair[0])
        tier_company = manager.get_company_info(pair[1])
        print(f"     {pair[0]} ({ag_company.sector}) ↔ {pair[1]} ({tier_company.name})")
    
    print(f"\n   Transportation → Agriculture pairs:")
    for pair in transport_pairs:
        ag_company = manager.get_company_info(pair[0])
        tier_company = manager.get_company_info(pair[1])
        print(f"     {pair[0]} ({ag_company.sector}) ↔ {pair[1]} ({tier_company.name})")
    
    print(f"\n   Chemicals → Agriculture pairs:")
    for pair in chemical_pairs:
        ag_company = manager.get_company_info(pair[0])
        tier_company = manager.get_company_info(pair[1])
        print(f"     {pair[0]} ({ag_company.sector}) ↔ {pair[1]} ({tier_company.name})")
    
    # Show transmission mechanisms
    print(f"\n🔄 Transmission Mechanisms:")
    for mechanism in manager.transmission_mechanisms[:5]:  # Show first 5
        print(f"   {mechanism.source_sector} → {mechanism.target_sector}")
        print(f"     Mechanism: {mechanism.mechanism}")
        print(f"     Expected lag: {mechanism.expected_lag} days ({mechanism.strength})")
        print()
    
    # Demonstrate market cap analysis
    print(f"\n💰 Market Cap Distribution:")
    large_cap = manager.classify_by_market_cap(MarketCapCategory.LARGE_CAP)
    mid_cap = manager.classify_by_market_cap(MarketCapCategory.MID_CAP)
    small_cap = manager.classify_by_market_cap(MarketCapCategory.SMALL_CAP)
    
    print(f"   Large-Cap (>$10B): {len(large_cap)} companies")
    print(f"   Mid-Cap ($2B-$10B): {len(mid_cap)} companies")
    print(f"   Small-Cap ($250M-$2B): {len(small_cap)} companies")
    
    # Show example usage for Bell inequality analysis
    print(f"\n🔬 Example Bell Inequality Analysis Setup:")
    print(f"   For crisis period analysis (COVID-19, Ukraine War, 2008 Food Crisis):")
    
    # Get high-priority pairs for analysis
    fertilizer_companies = [ticker for ticker in manager.classify_by_tier(0) 
                           if manager.get_company_info(ticker).sector == "Fertilizers"]
    energy_companies = [ticker for ticker in manager.classify_by_tier(1) 
                       if manager.get_company_info(ticker).sector == "Energy"]
    
    print(f"\n   Priority Analysis: Fertilizer ↔ Energy pairs")
    print(f"   Fertilizer companies: {fertilizer_companies}")
    print(f"   Energy companies: {energy_companies[:3]}...")  # Show first 3
    
    # Show expected transmission for CF (fertilizer) ↔ COP (natural gas)
    if "CF" in fertilizer_companies and "COP" in energy_companies:
        cf_cop_mechanisms = manager.get_transmission_mechanisms(("CF", "COP"))
        if cf_cop_mechanisms:
            print(f"\n   Example: CF (Fertilizer) ↔ COP (Natural Gas)")
            for mechanism in cf_cop_mechanisms:
                print(f"     Transmission: {mechanism.mechanism}")
                print(f"     Expected lag: {mechanism.expected_lag} days")
                print(f"     Crisis amplification: {mechanism.crisis_amplification}")
    
    print(f"\n✅ Agricultural Universe Management System Ready for Analysis!")
    print(f"   Use manager.get_cross_sector_pairs(tier) for analysis pairs")
    print(f"   Use manager.get_transmission_mechanisms(pair) for expected relationships")
    print(f"   Total analysis pairs available: {len(tier1_pairs)} (Tier 1)")
    
    return manager


if __name__ == "__main__":
    manager = demonstrate_agricultural_universe()
    
    # Export for use in analysis
    print(f"\n📁 Exporting universe data...")
    df = manager.export_universe_to_csv("../agricultural_universe.csv")
    print(f"   Exported to agricultural_universe.csv ({len(df)} companies)")
    print(f"   Ready for Bell inequality analysis!")