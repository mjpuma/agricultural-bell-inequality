"""
Agricultural Universe Management System

This module manages the comprehensive agricultural company database with tier classifications,
market cap categorizations, and exposure level classifications for cross-sectoral analysis.

Following the design from agricultural-cross-sector-analysis spec.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum
import json
import pandas as pd


class MarketCapCategory(Enum):
    LARGE_CAP = "Large-Cap"  # >$10B
    MID_CAP = "Mid-Cap"      # $2B-$10B
    SMALL_CAP = "Small-Cap"  # $250M-$2B


class ExposureLevel(Enum):
    PRIMARY = "Primary"      # Direct agricultural operations
    SECONDARY = "Secondary"  # Significant agricultural exposure
    TERTIARY = "Tertiary"    # Indirect exposure


class Tier(Enum):
    TIER_1 = 1  # Energy/Transport/Chemicals - Direct operational dependencies
    TIER_2 = 2  # Finance/Equipment - Major cost drivers
    TIER_3 = 3  # Policy-linked - Renewable Energy, Water Utilities


@dataclass
class CompanyInfo:
    """Company information model following the design specification."""
    ticker: str
    name: str
    sector: str
    subsector: str
    exposure: ExposureLevel
    market_cap_category: MarketCapCategory
    market_cap_approx: str
    tier: Optional[Tier]  # None for agricultural companies, Tier for cross-sector
    operational_dependencies: List[str]
    geographic_exposure: List[str]
    description: str


@dataclass
class TransmissionMechanism:
    """Transmission mechanism model for cross-sector relationships."""
    source_sector: str
    target_sector: str
    mechanism: str
    expected_lag: int  # Days
    strength: str  # Strong, Moderate, Weak
    crisis_amplification: bool


class AgriculturalUniverseManager:
    """
    Manages the comprehensive agricultural company database with tier classifications
    and cross-sectoral relationships for Bell inequality analysis.
    """
    
    def __init__(self):
        self.companies: Dict[str, CompanyInfo] = {}
        self.transmission_mechanisms: List[TransmissionMechanism] = []
        self._initialize_universe()
        self._initialize_transmission_mechanisms()
    
    def _initialize_universe(self):
        """Initialize the comprehensive 60+ company agricultural universe."""
        
        # Agricultural Companies - Core Universe
        agricultural_companies = [
            # Large-Cap Agricultural Companies (>$10B)
            CompanyInfo("ADM", "Archer Daniels Midland", "Agricultural Processing", "Grain Processing", 
                       ExposureLevel.PRIMARY, MarketCapCategory.LARGE_CAP, "$25B", None,
                       ["CORN", "SOYB", "WEAT"], ["North America", "South America", "Europe"],
                       "Global agricultural processor and food ingredient provider"),
            
            CompanyInfo("BG", "Bunge Limited", "Agricultural Processing", "Grain Trading", 
                       ExposureLevel.PRIMARY, MarketCapCategory.LARGE_CAP, "$15B", None,
                       ["CORN", "SOYB", "WEAT", "SUGA"], ["Global"], 
                       "Global agribusiness and food company"),
            
            CompanyInfo("CF", "CF Industries", "Fertilizers", "Nitrogen Fertilizers", 
                       ExposureLevel.PRIMARY, MarketCapCategory.LARGE_CAP, "$18B", None,
                       ["Natural Gas", "Ammonia"], ["North America", "Europe"],
                       "Leading nitrogen fertilizer producer"),
            
            CompanyInfo("NTR", "Nutrien Ltd", "Fertilizers", "Crop Nutrients", 
                       ExposureLevel.PRIMARY, MarketCapCategory.LARGE_CAP, "$25B", None,
                       ["Potash", "Phosphate", "Nitrogen"], ["Global"],
                       "World's largest provider of crop inputs and services"),
            
            CompanyInfo("MOS", "Mosaic Company", "Fertilizers", "Phosphate & Potash", 
                       ExposureLevel.PRIMARY, MarketCapCategory.LARGE_CAP, "$12B", None,
                       ["Phosphate", "Potash"], ["North America", "South America"],
                       "Leading producer of concentrated phosphate and potash"),
            
            CompanyInfo("DE", "Deere & Company", "Agricultural Equipment", "Farm Machinery", 
                       ExposureLevel.PRIMARY, MarketCapCategory.LARGE_CAP, "$110B", None,
                       ["Steel", "Technology"], ["Global"],
                       "Leading manufacturer of agricultural machinery"),
            
            # Mid-Cap Agricultural Companies ($2B-$10B)
            CompanyInfo("CAG", "ConAgra Brands", "Food Processing", "Packaged Foods", 
                       ExposureLevel.PRIMARY, MarketCapCategory.MID_CAP, "$8B", None,
                       ["CORN", "WEAT", "SOYB"], ["North America"],
                       "Consumer packaged goods food company"),
            
            CompanyInfo("TSN", "Tyson Foods", "Food Processing", "Meat Processing", 
                       ExposureLevel.PRIMARY, MarketCapCategory.MID_CAP, "$9B", None,
                       ["CORN", "SOYB", "LEAN"], ["North America"],
                       "Multinational corporation in food processing"),
            
            CompanyInfo("HRL", "Hormel Foods", "Food Processing", "Meat Products", 
                       ExposureLevel.PRIMARY, MarketCapCategory.MID_CAP, "$7B", None,
                       ["LEAN", "CORN"], ["North America"],
                       "Food processing company specializing in meat products"),
            
            CompanyInfo("SJM", "J.M. Smucker", "Food Processing", "Consumer Foods", 
                       ExposureLevel.PRIMARY, MarketCapCategory.MID_CAP, "$6B", None,
                       ["SUGA", "COFF"], ["North America"],
                       "Food and beverage manufacturer"),
            
            CompanyInfo("K", "Kellogg Company", "Food Processing", "Breakfast Foods", 
                       ExposureLevel.PRIMARY, MarketCapCategory.MID_CAP, "$5B", None,
                       ["CORN", "WEAT", "SUGA"], ["Global"],
                       "Multinational food manufacturing company"),
            
            CompanyInfo("GIS", "General Mills", "Food Processing", "Consumer Foods", 
                       ExposureLevel.PRIMARY, MarketCapCategory.MID_CAP, "$8B", None,
                       ["WEAT", "CORN", "SUGA"], ["Global"],
                       "Multinational manufacturer and marketer of branded consumer foods"),
            
            CompanyInfo("CPB", "Campbell Soup", "Food Processing", "Packaged Foods", 
                       ExposureLevel.PRIMARY, MarketCapCategory.MID_CAP, "$4B", None,
                       ["Vegetables", "WEAT"], ["North America"],
                       "American processed food and snack company"),
            
            CompanyInfo("AGCO", "AGCO Corporation", "Agricultural Equipment", "Farm Equipment", 
                       ExposureLevel.PRIMARY, MarketCapCategory.MID_CAP, "$8B", None,
                       ["Steel", "Technology"], ["Global"],
                       "Agricultural equipment manufacturer"),
            
            CompanyInfo("CNH", "CNH Industrial", "Agricultural Equipment", "Farm Machinery", 
                       ExposureLevel.PRIMARY, MarketCapCategory.MID_CAP, "$7B", None,
                       ["Steel", "Technology"], ["Global"],
                       "Multinational corporation in agricultural equipment"),
            
            # Small-Cap Agricultural Companies ($250M-$2B)
            CompanyInfo("FMC", "FMC Corporation", "Crop Protection", "Agricultural Chemicals", 
                       ExposureLevel.PRIMARY, MarketCapCategory.SMALL_CAP, "$1.5B", None,
                       ["Chemicals"], ["Global"],
                       "Agricultural sciences company providing crop protection"),
            
            CompanyInfo("SMG", "Scotts Miracle-Gro", "Lawn & Garden", "Consumer Lawn Care", 
                       ExposureLevel.SECONDARY, MarketCapCategory.SMALL_CAP, "$800M", None,
                       ["Fertilizers", "Seeds"], ["North America"],
                       "Lawn and garden care company"),
            
            CompanyInfo("CALM", "Cal-Maine Foods", "Food Production", "Egg Production", 
                       ExposureLevel.PRIMARY, MarketCapCategory.SMALL_CAP, "$1B", None,
                       ["CORN", "SOYB"], ["North America"],
                       "Shell egg producer and distributor"),
            
            CompanyInfo("SAFM", "Sanderson Farms", "Food Production", "Poultry", 
                       ExposureLevel.PRIMARY, MarketCapCategory.SMALL_CAP, "$900M", None,
                       ["CORN", "SOYB"], ["North America"],
                       "Poultry processing company"),
            
            CompanyInfo("LNDC", "Landec Corporation", "Food Technology", "Fresh Foods", 
                       ExposureLevel.SECONDARY, MarketCapCategory.SMALL_CAP, "$300M", None,
                       ["Vegetables", "Technology"], ["North America"],
                       "Food technology and fresh food company"),
        ]
        
        # Tier 1: Energy/Transport/Chemicals - Direct operational dependencies
        tier1_companies = [
            # Energy Companies
            CompanyInfo("XOM", "Exxon Mobil", "Energy", "Oil & Gas", 
                       ExposureLevel.TERTIARY, MarketCapCategory.LARGE_CAP, "$400B", Tier.TIER_1,
                       ["Crude Oil", "Natural Gas"], ["Global"],
                       "Multinational oil and gas corporation - diesel for farm equipment"),
            
            CompanyInfo("CVX", "Chevron", "Energy", "Oil & Gas", 
                       ExposureLevel.TERTIARY, MarketCapCategory.LARGE_CAP, "$300B", Tier.TIER_1,
                       ["Crude Oil", "Natural Gas"], ["Global"],
                       "Multinational energy corporation - fuel for agricultural operations"),
            
            CompanyInfo("COP", "ConocoPhillips", "Energy", "Oil & Gas", 
                       ExposureLevel.TERTIARY, MarketCapCategory.LARGE_CAP, "$150B", Tier.TIER_1,
                       ["Crude Oil", "Natural Gas"], ["Global"],
                       "Oil and gas exploration company - natural gas for fertilizer production"),
            
            # Transportation Companies
            CompanyInfo("UNP", "Union Pacific", "Transportation", "Rail Transport", 
                       ExposureLevel.SECONDARY, MarketCapCategory.LARGE_CAP, "$140B", Tier.TIER_1,
                       ["Rail Infrastructure"], ["North America"],
                       "Railroad operating company - grain transportation"),
            
            CompanyInfo("CSX", "CSX Corporation", "Transportation", "Rail Transport", 
                       ExposureLevel.SECONDARY, MarketCapCategory.LARGE_CAP, "$70B", Tier.TIER_1,
                       ["Rail Infrastructure"], ["North America"],
                       "Transportation company - agricultural commodity shipping"),
            
            CompanyInfo("NSC", "Norfolk Southern", "Transportation", "Rail Transport", 
                       ExposureLevel.SECONDARY, MarketCapCategory.LARGE_CAP, "$60B", Tier.TIER_1,
                       ["Rail Infrastructure"], ["North America"],
                       "Railroad company - grain and fertilizer transport"),
            
            # Chemical Companies
            CompanyInfo("DOW", "Dow Inc", "Chemicals", "Agricultural Chemicals", 
                       ExposureLevel.SECONDARY, MarketCapCategory.LARGE_CAP, "$40B", Tier.TIER_1,
                       ["Petrochemicals"], ["Global"],
                       "Chemical company - agricultural chemical inputs"),
            
            CompanyInfo("DD", "DuPont", "Chemicals", "Specialty Chemicals", 
                       ExposureLevel.SECONDARY, MarketCapCategory.LARGE_CAP, "$35B", Tier.TIER_1,
                       ["Chemicals", "Seeds"], ["Global"],
                       "Chemical company - crop protection and seeds"),
            
            CompanyInfo("LYB", "LyondellBasell", "Chemicals", "Petrochemicals", 
                       ExposureLevel.TERTIARY, MarketCapCategory.LARGE_CAP, "$30B", Tier.TIER_1,
                       ["Petrochemicals"], ["Global"],
                       "Chemical company - plastic packaging for agriculture"),
        ]
        
        # Tier 2: Finance/Equipment - Major cost drivers
        tier2_companies = [
            # Financial Companies
            CompanyInfo("JPM", "JPMorgan Chase", "Finance", "Banking", 
                       ExposureLevel.TERTIARY, MarketCapCategory.LARGE_CAP, "$450B", Tier.TIER_2,
                       ["Credit Markets"], ["Global"],
                       "Investment bank - agricultural lending and commodity financing"),
            
            CompanyInfo("BAC", "Bank of America", "Finance", "Banking", 
                       ExposureLevel.TERTIARY, MarketCapCategory.LARGE_CAP, "$250B", Tier.TIER_2,
                       ["Credit Markets"], ["Global"],
                       "Commercial bank - farm loans and agricultural credit"),
            
            CompanyInfo("GS", "Goldman Sachs", "Finance", "Investment Banking", 
                       ExposureLevel.TERTIARY, MarketCapCategory.LARGE_CAP, "$120B", Tier.TIER_2,
                       ["Commodity Trading"], ["Global"],
                       "Investment bank - commodity derivatives and agricultural trading"),
            
            # Heavy Equipment (beyond agricultural)
            CompanyInfo("CAT", "Caterpillar", "Equipment", "Heavy Machinery", 
                       ExposureLevel.SECONDARY, MarketCapCategory.LARGE_CAP, "$130B", Tier.TIER_2,
                       ["Steel", "Mining"], ["Global"],
                       "Heavy equipment manufacturer - construction equipment for farms"),
        ]
        
        # Tier 3: Policy-linked - Renewable Energy, Water Utilities
        tier3_companies = [
            # Renewable Energy
            CompanyInfo("NEE", "NextEra Energy", "Utilities", "Renewable Energy", 
                       ExposureLevel.TERTIARY, MarketCapCategory.LARGE_CAP, "$150B", Tier.TIER_3,
                       ["Solar", "Wind"], ["North America"],
                       "Renewable energy company - rural electrification for farms"),
            
            CompanyInfo("DUK", "Duke Energy", "Utilities", "Electric Utilities", 
                       ExposureLevel.TERTIARY, MarketCapCategory.LARGE_CAP, "$70B", Tier.TIER_3,
                       ["Natural Gas", "Nuclear"], ["North America"],
                       "Electric utility - power for agricultural operations"),
            
            # Water Utilities
            CompanyInfo("AWK", "American Water Works", "Utilities", "Water Utilities", 
                       ExposureLevel.SECONDARY, MarketCapCategory.LARGE_CAP, "$20B", Tier.TIER_3,
                       ["Water Infrastructure"], ["North America"],
                       "Water utility - irrigation water supply"),
            
            CompanyInfo("WTR", "Aqua America", "Utilities", "Water Utilities", 
                       ExposureLevel.SECONDARY, MarketCapCategory.MID_CAP, "$3B", Tier.TIER_3,
                       ["Water Infrastructure"], ["North America"],
                       "Water utility company - agricultural water services"),
        ]
        
        # Additional Agricultural Companies to reach 60+ total
        additional_agricultural = [
            CompanyInfo("INGR", "Ingredion", "Food Processing", "Food Ingredients", 
                       ExposureLevel.PRIMARY, MarketCapCategory.MID_CAP, "$6B", None,
                       ["CORN", "Starches"], ["Global"],
                       "Ingredient solutions company"),
            
            CompanyInfo("DAR", "Darling Ingredients", "Food Processing", "Food Ingredients", 
                       ExposureLevel.SECONDARY, MarketCapCategory.MID_CAP, "$4B", None,
                       ["Animal By-products"], ["Global"],
                       "Food ingredient and specialty products company"),
            
            CompanyInfo("CVGW", "Calavo Growers", "Food Production", "Fresh Produce", 
                       ExposureLevel.PRIMARY, MarketCapCategory.SMALL_CAP, "$400M", None,
                       ["Avocados", "Vegetables"], ["North America"],
                       "Fresh produce company"),
            
            CompanyInfo("JJSF", "J&J Snack Foods", "Food Processing", "Snack Foods", 
                       ExposureLevel.SECONDARY, MarketCapCategory.SMALL_CAP, "$1B", None,
                       ["WEAT", "CORN"], ["North America"],
                       "Snack food manufacturer"),
            
            CompanyInfo("SENEA", "Seneca Foods", "Food Processing", "Canned Foods", 
                       ExposureLevel.PRIMARY, MarketCapCategory.SMALL_CAP, "$500M", None,
                       ["Vegetables", "Fruits"], ["North America"],
                       "Canned and frozen vegetable processor"),
            
            CompanyInfo("FARM", "Farmer Bros", "Food Processing", "Coffee", 
                       ExposureLevel.SECONDARY, MarketCapCategory.SMALL_CAP, "$300M", None,
                       ["COFF"], ["North America"],
                       "Coffee roaster and distributor"),
            
            CompanyInfo("VITL", "Vital Farms", "Food Production", "Organic Eggs", 
                       ExposureLevel.PRIMARY, MarketCapCategory.SMALL_CAP, "$800M", None,
                       ["Organic Feed"], ["North America"],
                       "Pasture-raised egg producer"),
            
            CompanyInfo("APPH", "AppHarvest", "Food Production", "Controlled Agriculture", 
                       ExposureLevel.PRIMARY, MarketCapCategory.SMALL_CAP, "$200M", None,
                       ["Technology", "Water"], ["North America"],
                       "Controlled environment agriculture company"),
            
            # Additional Tier 1 Companies
            CompanyInfo("SLB", "Schlumberger", "Energy", "Oil Services", 
                       ExposureLevel.TERTIARY, MarketCapCategory.LARGE_CAP, "$60B", Tier.TIER_1,
                       ["Oil Services"], ["Global"],
                       "Oil services company - drilling equipment for energy production"),
            
            CompanyInfo("HAL", "Halliburton", "Energy", "Oil Services", 
                       ExposureLevel.TERTIARY, MarketCapCategory.LARGE_CAP, "$25B", Tier.TIER_1,
                       ["Oil Services"], ["Global"],
                       "Oil services company - energy infrastructure"),
            
            CompanyInfo("KMI", "Kinder Morgan", "Energy", "Pipeline", 
                       ExposureLevel.TERTIARY, MarketCapCategory.LARGE_CAP, "$35B", Tier.TIER_1,
                       ["Natural Gas Pipeline"], ["North America"],
                       "Pipeline company - natural gas transport for fertilizer production"),
            
            CompanyInfo("FDX", "FedEx", "Transportation", "Logistics", 
                       ExposureLevel.SECONDARY, MarketCapCategory.LARGE_CAP, "$60B", Tier.TIER_1,
                       ["Logistics"], ["Global"],
                       "Logistics company - agricultural product distribution"),
            
            CompanyInfo("UPS", "United Parcel Service", "Transportation", "Logistics", 
                       ExposureLevel.SECONDARY, MarketCapCategory.LARGE_CAP, "$140B", Tier.TIER_1,
                       ["Logistics"], ["Global"],
                       "Package delivery - agricultural supply chain logistics"),
            
            CompanyInfo("EMN", "Eastman Chemical", "Chemicals", "Specialty Chemicals", 
                       ExposureLevel.SECONDARY, MarketCapCategory.MID_CAP, "$8B", Tier.TIER_1,
                       ["Specialty Chemicals"], ["Global"],
                       "Chemical company - agricultural chemical inputs"),
            
            CompanyInfo("PPG", "PPG Industries", "Chemicals", "Coatings", 
                       ExposureLevel.TERTIARY, MarketCapCategory.LARGE_CAP, "$25B", Tier.TIER_1,
                       ["Coatings", "Chemicals"], ["Global"],
                       "Chemical company - agricultural equipment coatings"),
            
            # Additional Tier 2 Companies
            CompanyInfo("WFC", "Wells Fargo", "Finance", "Banking", 
                       ExposureLevel.TERTIARY, MarketCapCategory.LARGE_CAP, "$180B", Tier.TIER_2,
                       ["Agricultural Lending"], ["North America"],
                       "Commercial bank - agricultural lending and farm financing"),
            
            CompanyInfo("C", "Citigroup", "Finance", "Banking", 
                       ExposureLevel.TERTIARY, MarketCapCategory.LARGE_CAP, "$100B", Tier.TIER_2,
                       ["Commodity Trading"], ["Global"],
                       "Investment bank - commodity financing and trading"),
            
            CompanyInfo("MS", "Morgan Stanley", "Finance", "Investment Banking", 
                       ExposureLevel.TERTIARY, MarketCapCategory.LARGE_CAP, "$130B", Tier.TIER_2,
                       ["Commodity Trading"], ["Global"],
                       "Investment bank - agricultural commodity derivatives"),
            
            CompanyInfo("JCI", "Johnson Controls", "Equipment", "Industrial Equipment", 
                       ExposureLevel.SECONDARY, MarketCapCategory.LARGE_CAP, "$45B", Tier.TIER_2,
                       ["HVAC", "Controls"], ["Global"],
                       "Industrial equipment - agricultural facility climate control"),
            
            # Additional Tier 3 Companies
            CompanyInfo("SO", "Southern Company", "Utilities", "Electric Utilities", 
                       ExposureLevel.TERTIARY, MarketCapCategory.LARGE_CAP, "$70B", Tier.TIER_3,
                       ["Electric Grid"], ["North America"],
                       "Electric utility - rural electrification for agricultural operations"),
            
            CompanyInfo("D", "Dominion Energy", "Utilities", "Electric Utilities", 
                       ExposureLevel.TERTIARY, MarketCapCategory.LARGE_CAP, "$65B", Tier.TIER_3,
                       ["Natural Gas", "Electric"], ["North America"],
                       "Utility company - energy for agricultural operations"),
            
            CompanyInfo("XEL", "Xcel Energy", "Utilities", "Electric Utilities", 
                       ExposureLevel.TERTIARY, MarketCapCategory.LARGE_CAP, "$35B", Tier.TIER_3,
                       ["Wind", "Electric"], ["North America"],
                       "Electric utility - renewable energy for agricultural operations"),
            
            CompanyInfo("EXC", "Exelon", "Utilities", "Electric Utilities", 
                       ExposureLevel.TERTIARY, MarketCapCategory.LARGE_CAP, "$40B", Tier.TIER_3,
                       ["Nuclear", "Electric"], ["North America"],
                       "Electric utility - power for agricultural processing facilities"),
            
            # More Agricultural Companies
            CompanyInfo("UNFI", "United Natural Foods", "Food Distribution", "Organic Distribution", 
                       ExposureLevel.PRIMARY, MarketCapCategory.SMALL_CAP, "$1.5B", None,
                       ["Organic Foods"], ["North America"],
                       "Natural and organic food distributor"),
            
            CompanyInfo("SEB", "Seaboard Corporation", "Food Production", "Diversified Agriculture", 
                       ExposureLevel.PRIMARY, MarketCapCategory.MID_CAP, "$4B", None,
                       ["Pork", "Grain", "Sugar"], ["Global"],
                       "Diversified agribusiness and transportation company"),
            
            CompanyInfo("LANC", "Lancaster Colony", "Food Processing", "Specialty Foods", 
                       ExposureLevel.SECONDARY, MarketCapCategory.MID_CAP, "$3B", None,
                       ["Specialty Foods"], ["North America"],
                       "Specialty food manufacturer"),
            
            CompanyInfo("PFGC", "Performance Food Group", "Food Distribution", "Food Service", 
                       ExposureLevel.SECONDARY, MarketCapCategory.MID_CAP, "$8B", None,
                       ["Food Distribution"], ["North America"],
                       "Food distribution company"),
            
            CompanyInfo("USFD", "US Foods", "Food Distribution", "Food Service", 
                       ExposureLevel.SECONDARY, MarketCapCategory.LARGE_CAP, "$12B", None,
                       ["Food Distribution"], ["North America"],
                       "Food service distribution company"),
            
            CompanyInfo("SYY", "Sysco Corporation", "Food Distribution", "Food Service", 
                       ExposureLevel.SECONDARY, MarketCapCategory.LARGE_CAP, "$40B", None,
                       ["Food Distribution"], ["Global"],
                       "Global food service distribution company"),
            
            CompanyInfo("CHEF", "Chefs' Warehouse", "Food Distribution", "Specialty Foods", 
                       ExposureLevel.SECONDARY, MarketCapCategory.SMALL_CAP, "$1B", None,
                       ["Specialty Foods"], ["North America"],
                       "Specialty food distributor"),
            
            CompanyInfo("AVO", "Mission Produce", "Food Production", "Fresh Produce", 
                       ExposureLevel.PRIMARY, MarketCapCategory.SMALL_CAP, "$800M", None,
                       ["Avocados"], ["Global"],
                       "Avocado producer and distributor"),
            
            CompanyInfo("DOLE", "Dole plc", "Food Production", "Fresh Produce", 
                       ExposureLevel.PRIMARY, MarketCapCategory.SMALL_CAP, "$1.2B", None,
                       ["Fruits", "Vegetables"], ["Global"],
                       "Fresh fruit and vegetable producer"),
            
            CompanyInfo("FDP", "Fresh Del Monte", "Food Production", "Fresh Produce", 
                       ExposureLevel.PRIMARY, MarketCapCategory.SMALL_CAP, "$1.5B", None,
                       ["Fruits", "Vegetables"], ["Global"],
                       "Fresh produce company"),
        ]
        
        # Combine all companies
        all_companies = agricultural_companies + tier1_companies + tier2_companies + tier3_companies + additional_agricultural
        
        # Store in dictionary
        for company in all_companies:
            self.companies[company.ticker] = company
    
    def _initialize_transmission_mechanisms(self):
        """Initialize transmission mechanisms between sectors."""
        self.transmission_mechanisms = [
            # Energy → Agriculture
            TransmissionMechanism(
                "Energy", "Fertilizers", 
                "Natural gas prices → fertilizer production costs → crop input costs",
                30, "Strong", True
            ),
            TransmissionMechanism(
                "Energy", "Agricultural Equipment", 
                "Diesel prices → farm equipment operating costs → farming costs",
                15, "Strong", True
            ),
            TransmissionMechanism(
                "Energy", "Food Processing", 
                "Energy costs → food processing costs → food prices",
                45, "Moderate", True
            ),
            
            # Transportation → Agriculture
            TransmissionMechanism(
                "Transportation", "Grain Trading", 
                "Rail/shipping costs → commodity logistics costs → grain prices",
                20, "Strong", True
            ),
            TransmissionMechanism(
                "Transportation", "Food Processing", 
                "Transportation bottlenecks → supply chain disruptions → food prices",
                30, "Moderate", True
            ),
            
            # Chemicals → Agriculture
            TransmissionMechanism(
                "Chemicals", "Crop Protection", 
                "Chemical input costs → pesticide/herbicide costs → crop protection costs",
                25, "Strong", True
            ),
            TransmissionMechanism(
                "Chemicals", "Fertilizers", 
                "Chemical feedstock costs → fertilizer production costs → crop input costs",
                35, "Strong", True
            ),
            
            # Finance → Agriculture
            TransmissionMechanism(
                "Finance", "Agricultural Operations", 
                "Credit availability → farm financing → agricultural investment",
                60, "Moderate", True
            ),
            TransmissionMechanism(
                "Finance", "Commodity Trading", 
                "Credit conditions → commodity financing → trading volumes",
                15, "Strong", True
            ),
        ]
    
    def load_agricultural_universe(self) -> Dict[str, CompanyInfo]:
        """Load the complete agricultural universe."""
        return self.companies
    
    def classify_by_tier(self, tier: int) -> List[str]:
        """Get companies by tier classification."""
        if tier == 0:  # Agricultural companies (no tier)
            return [ticker for ticker, company in self.companies.items() if company.tier is None]
        else:
            target_tier = Tier(tier)
            return [ticker for ticker, company in self.companies.items() if company.tier == target_tier]
    
    def classify_by_market_cap(self, category: MarketCapCategory) -> List[str]:
        """Get companies by market cap category."""
        return [ticker for ticker, company in self.companies.items() 
                if company.market_cap_category == category]
    
    def classify_by_exposure(self, exposure: ExposureLevel) -> List[str]:
        """Get companies by exposure level."""
        return [ticker for ticker, company in self.companies.items() 
                if company.exposure == exposure]
    
    def get_cross_sector_pairs(self, tier: int) -> List[Tuple[str, str]]:
        """Get cross-sector pairs for analysis based on operational dependencies."""
        agricultural_companies = self.classify_by_tier(0)  # Agricultural companies
        tier_companies = self.classify_by_tier(tier)
        
        pairs = []
        
        # Create pairs based on operational dependencies
        for ag_ticker in agricultural_companies:
            ag_company = self.companies[ag_ticker]
            
            for tier_ticker in tier_companies:
                tier_company = self.companies[tier_ticker]
                
                # Check for operational dependencies
                if self._has_operational_dependency(ag_company, tier_company):
                    pairs.append((ag_ticker, tier_ticker))
        
        return pairs
    
    def _has_operational_dependency(self, ag_company: CompanyInfo, tier_company: CompanyInfo) -> bool:
        """Check if agricultural company has operational dependency on tier company."""
        
        # Energy dependencies
        if tier_company.sector == "Energy":
            # Fertilizer companies depend on natural gas
            if ag_company.sector == "Fertilizers" and "Natural Gas" in tier_company.operational_dependencies:
                return True
            # All agricultural operations depend on diesel/fuel
            if "Crude Oil" in tier_company.operational_dependencies:
                return True
        
        # Transportation dependencies
        if tier_company.sector == "Transportation":
            # Grain companies depend on rail transport
            if ag_company.sector in ["Agricultural Processing", "Grain Trading"]:
                return True
        
        # Chemical dependencies
        if tier_company.sector == "Chemicals":
            # Crop protection and fertilizer companies depend on chemicals
            if ag_company.sector in ["Crop Protection", "Fertilizers"]:
                return True
        
        # Financial dependencies
        if tier_company.sector == "Finance":
            # All agricultural companies need financing
            return True
        
        # Equipment dependencies
        if tier_company.sector == "Equipment":
            # All agricultural operations need equipment
            return True
        
        # Utility dependencies
        if tier_company.sector == "Utilities":
            # All agricultural operations need power and water
            return True
        
        return False
    
    def get_transmission_mechanisms(self, pair: Tuple[str, str]) -> List[TransmissionMechanism]:
        """Get transmission mechanisms for a specific pair."""
        ag_ticker, tier_ticker = pair
        ag_company = self.companies[ag_ticker]
        tier_company = self.companies[tier_ticker]
        
        relevant_mechanisms = []
        for mechanism in self.transmission_mechanisms:
            if (tier_company.sector == mechanism.source_sector and 
                (ag_company.sector == mechanism.target_sector or 
                 ag_company.subsector == mechanism.target_sector)):
                relevant_mechanisms.append(mechanism)
        
        return relevant_mechanisms
    
    def get_company_info(self, ticker: str) -> Optional[CompanyInfo]:
        """Get company information by ticker."""
        return self.companies.get(ticker)
    
    def get_universe_summary(self) -> Dict:
        """Get summary statistics of the universe."""
        total_companies = len(self.companies)
        
        # Count by tier
        agricultural = len(self.classify_by_tier(0))
        tier1 = len(self.classify_by_tier(1))
        tier2 = len(self.classify_by_tier(2))
        tier3 = len(self.classify_by_tier(3))
        
        # Count by market cap
        large_cap = len(self.classify_by_market_cap(MarketCapCategory.LARGE_CAP))
        mid_cap = len(self.classify_by_market_cap(MarketCapCategory.MID_CAP))
        small_cap = len(self.classify_by_market_cap(MarketCapCategory.SMALL_CAP))
        
        # Count by exposure
        primary = len(self.classify_by_exposure(ExposureLevel.PRIMARY))
        secondary = len(self.classify_by_exposure(ExposureLevel.SECONDARY))
        tertiary = len(self.classify_by_exposure(ExposureLevel.TERTIARY))
        
        return {
            "total_companies": total_companies,
            "by_tier": {
                "agricultural": agricultural,
                "tier_1_energy_transport_chemicals": tier1,
                "tier_2_finance_equipment": tier2,
                "tier_3_policy_linked": tier3
            },
            "by_market_cap": {
                "large_cap": large_cap,
                "mid_cap": mid_cap,
                "small_cap": small_cap
            },
            "by_exposure": {
                "primary": primary,
                "secondary": secondary,
                "tertiary": tertiary
            },
            "transmission_mechanisms": len(self.transmission_mechanisms)
        }
    
    def export_universe_to_csv(self, filename: str = "agricultural_universe.csv"):
        """Export the universe to CSV for analysis."""
        data = []
        for ticker, company in self.companies.items():
            data.append({
                "ticker": ticker,
                "name": company.name,
                "sector": company.sector,
                "subsector": company.subsector,
                "exposure": company.exposure.value,
                "market_cap_category": company.market_cap_category.value,
                "market_cap_approx": company.market_cap_approx,
                "tier": company.tier.value if company.tier else "Agricultural",
                "operational_dependencies": "|".join(company.operational_dependencies),
                "geographic_exposure": "|".join(company.geographic_exposure),
                "description": company.description
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        return df


if __name__ == "__main__":
    # Example usage
    manager = AgriculturalUniverseManager()
    
    # Print summary
    summary = manager.get_universe_summary()
    print("Agricultural Universe Summary:")
    print(f"Total Companies: {summary['total_companies']}")
    print(f"Agricultural: {summary['by_tier']['agricultural']}")
    print(f"Tier 1 (Energy/Transport/Chemicals): {summary['by_tier']['tier_1_energy_transport_chemicals']}")
    print(f"Tier 2 (Finance/Equipment): {summary['by_tier']['tier_2_finance_equipment']}")
    print(f"Tier 3 (Policy-linked): {summary['by_tier']['tier_3_policy_linked']}")
    
    # Show some cross-sector pairs
    tier1_pairs = manager.get_cross_sector_pairs(1)
    print(f"\nTier 1 Cross-Sector Pairs: {len(tier1_pairs)}")
    for pair in tier1_pairs[:5]:  # Show first 5
        ag_company = manager.get_company_info(pair[0])
        tier_company = manager.get_company_info(pair[1])
        print(f"  {pair[0]} ({ag_company.sector}) <-> {pair[1]} ({tier_company.sector})")