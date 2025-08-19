# 🎯 WDRS SYMBOL GUIDE FOR FOOD SYSTEMS ANALYSIS

## 📊 **CRITICAL QUESTION: WHAT SYMBOLS DOES WDRS USE?**

**I don't have access to WDRS's specific symbol conventions**, but I can guide you on what to ask for and what alternatives to consider.

## 🔍 **WHAT YOU NEED TO ASK WDRS:**

### **Stock Symbols (Likely Standard):**
These are probably the same as Yahoo Finance, but confirm:
- **ADM** → WDRS symbol for Archer-Daniels-Midland?
- **SJM** → WDRS symbol for J.M. Smucker?
- **CAG** → WDRS symbol for ConAgra Brands?
- **CPB** → WDRS symbol for Campbell Soup?
- **CF** → WDRS symbol for CF Industries?
- **NTR** → WDRS symbol for Nutrien?

### **Commodity Futures (Most Important to Verify):**
Different data providers use different conventions:

| Commodity | Common Symbols | Ask WDRS |
|-----------|----------------|----------|
| **Corn** | ZC, C, @C, CZ | "What symbol for corn futures?" |
| **Wheat** | ZW, W, @W, WZ | "What symbol for wheat futures?" |
| **Soybeans** | ZS, S, @S, SZ | "What symbol for soybean futures?" |

## 🎯 **EXACT QUESTIONS TO ASK WDRS:**

### **1. Symbol Verification:**
*"I need data for these assets for academic research. What are your exact symbols for:*
- *Archer-Daniels-Midland stock*
- *J.M. Smucker stock*
- *ConAgra Brands stock*
- *Campbell Soup stock*
- *Chicago corn futures (continuous front month)*
- *Chicago wheat futures (continuous front month)*
- *Chicago soybean futures (continuous front month)"*

### **2. Data Availability:**
*"Do you have complete daily data for these assets from 2020-2025 with no significant gaps?"*

### **3. Futures Contract Handling:**
*"For commodity futures, do you provide continuous contracts with proper rollover adjustments, or do I need to specify individual contract months?"*

### **4. Sample Data:**
*"Can you provide a small sample (1 week of data) for one asset to verify format and quality before placing the full order?"*

## 🔬 **WHY SYMBOL ACCURACY MATTERS:**

### **For Bell Inequality Analysis:**
- **Exact price matching** required between assets
- **No missing data points** (breaks the analysis)
- **Proper corporate action adjustments** for stocks
- **Continuous futures contracts** (no rollover gaps)

### **Common WDRS Symbol Patterns:**
Different providers use different conventions:
- **Bloomberg style:** ADM US Equity, C A Comdty
- **Reuters style:** ADM.N, Cc1
- **Simple style:** ADM, ZC
- **Exchange prefixes:** NYSE:ADM, CME:ZC

## 📋 **RECOMMENDED APPROACH:**

### **Step 1: Contact WDRS with Symbol Questions**
Use the questions above to get their exact symbol conventions.

### **Step 2: Request Sample Data**
Ask for 1 week of data for ADM and SJM to verify:
- File format
- Data quality
- Symbol accuracy
- Timestamp format

### **Step 3: Verify Against Yahoo Finance**
Compare the sample WDRS data with Yahoo Finance for the same period to ensure:
- Prices match (within reasonable bounds)
- No obvious data errors
- Proper corporate action adjustments

### **Step 4: Place Full Order**
Once symbols and quality are verified, order the full dataset.

## ⚠️ **POTENTIAL WDRS COMPLICATIONS:**

### **Futures Contracts:**
- **Individual months:** ZCZ24 (Dec 2024 corn) vs continuous ZC
- **Rollover dates:** When contracts expire and roll to next month
- **Price adjustments:** Back-adjusted vs non-adjusted continuous series

### **Stock Adjustments:**
- **Split adjustments:** Stock splits properly handled?
- **Dividend adjustments:** Total return vs price return?
- **Corporate actions:** Mergers, spinoffs properly adjusted?

### **Data Gaps:**
- **Holiday schedules:** US vs exchange-specific holidays
- **Trading halts:** Temporary suspensions included?
- **Delisting periods:** Any assets temporarily unavailable?

## 🎯 **FALLBACK STRATEGY:**

### **If WDRS Symbols Don't Work:**
1. **Ask for symbol lookup service:** "Can you help me find the right symbols?"
2. **Provide company names:** Full legal names instead of symbols
3. **Use CUSIP/ISIN codes:** Universal identifiers for stocks
4. **Request symbol mapping:** "What symbols do you use for NYSE:ADM?"

### **Alternative Data Sources:**
If WDRS doesn't have good coverage:
- **Quandl/Nasdaq Data Link:** Good for commodities
- **Alpha Vantage:** Good for stocks
- **IEX Cloud:** Good for recent stock data
- **FRED (Federal Reserve):** Some commodity data

## 💡 **KEY SUCCESS FACTORS:**

### **1. Clear Communication:**
Explain you're doing academic research on food systems and need high-quality, gap-free data.

### **2. Specific Requirements:**
- Daily frequency minimum
- 2020-2025 period for stocks
- COVID period (2020) for commodities
- Corporate action adjusted
- No missing data

### **3. Quality Verification:**
Always request sample data first to verify symbols and quality.

### **4. Backup Plan:**
Have alternative symbols ready in case primary ones don't work.

---

**Bottom Line:** Contact WDRS directly with the symbol verification questions above. Don't assume Yahoo Finance symbols will work - each data provider has their own conventions, especially for commodity futures.