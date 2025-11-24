# Data Sources and Methodology

This document describes the data sources, calibration methodology, and key assumptions for the technology parameters used in the Net-Zero Frontier framework.

## Korean Steel Sector (`korea_steel.csv`)

### Cost Parameters ($/ton steel)

| Technology | Cost | Source | Notes |
|------------|------|--------|-------|
| BF-BOF (Baseline) | $390 | Global Efficiency Intelligence (2024) | Global average LCOS |
| BF-BOF + CCUS | $465 | IEA (2024), CCS Institute | ~19% premium over baseline |
| Scrap-EAF | $415 | Columbia CKI (2024) | Includes scrap procurement |
| NG-DRI-EAF | $455 | DOE Liftoff Report (2024) | Natural gas at $3/MMBtu |
| HyREX H2-DRI | $616 | Transition Asia (2024) | At H2 price $5/kg |
| Hy-Cube H2-DRI | $600 | Estimated from HyREX | Similar technology pathway |
| EAF + Green H2 DRI | $680 | Transition Asia | Green premium ~$263/ton |
| Molten Oxide Electrolysis | $950 | MIT Technology Review | Early-stage technology |

### Abatement Parameters (tCO2/ton steel)

- **BF-BOF Baseline Emissions**: 2.2 tCO2/ton (industry standard)
- **Scrap-EAF Emissions**: 0.66 tCO2/ton (72% reduction)
- **H2-DRI Target**: 0.4 tCO2/ton (IEA near-zero threshold)

Sources:
- POSCO Sustainability Report 2023-2024
- Columbia Business School Climate Knowledge Initiative
- IEEFA Steel Fact Sheet (2022)

### Volatility Parameters (σ)

Volatility calibrated from:
- Historical cost data (Rubin et al., 2015)
- Technology maturity (TRL levels)
- Input price sensitivity

| Technology Type | σ Range | Rationale |
|-----------------|---------|-----------|
| Mature (BF-BOF) | 0.05 | Established supply chains |
| CCS | 0.12-0.15 | Storage cost uncertainty |
| EAF | 0.10 | Scrap price volatility |
| H2-DRI | 0.24-0.28 | Hydrogen price uncertainty |
| MOE | 0.40 | Early-stage R&D |

### Learning Rates (α)

| Technology | Learning Rate | Source |
|------------|---------------|--------|
| BF-BOF | 1% | McDonald & Schrattenholzer (2001) |
| CCUS | 6-7% | Global CCS Institute |
| EAF | 5% | Historical data |
| H2-DRI | 16-20% | Analogous to electrolyzer curves |
| MOE | 25% | Based on electrochemical systems |

## Global Steel Sector (`global_steel.csv`)

### European Projects
- **HYBRIT (SSAB/Sweden)**: Commercial 2026, cost $520/ton estimated
- **H2 Green Steel (Stegra)**: 5 Mt/year target by 2030
- **SALCOS (Salzgitter)**: 1 Mt/year H2-DRI by 2026
- **ThyssenKrupp**: H2 demonstration at Duisburg

### Asian Projects
- **POSCO HyREX**: 300kt demo by 2028, commercial 2030
- **Hyundai Hy-Cube**: Louisiana $6B plant announced 2025
- **HBIS (China)**: H2-DRI pilot operations

### US Market
- **Nucor**: Leading EAF producer, low emissions at 0.38 tCO2/ton
- **Hyundai Louisiana**: Blue H2 initially, green H2 post-2034

## Key Economic Assumptions

### Hydrogen Prices
| Scenario | Price ($/kg) | Timeframe |
|----------|--------------|-----------|
| Current (2024) | $3.50-6.00 | Global average |
| China (2024) | $3.00 | Declining rapidly |
| DOE Target | $1.00 | 2031 (Hydrogen Shot) |
| Cost Parity | $1.40-1.80 | Break-even with BF-BOF |

### Carbon Prices
| Region | Price ($/tCO2) | Notes |
|--------|----------------|-------|
| EU ETS | $52 | 2025 average |
| Korea ETS | $15-20 | Expected expansion |
| US (45Q) | $85 | Tax credit for CCS |

### CCS Costs
- Steel industry: $30-70/tCO2 (IEA)
- High-purity streams: $15-25/tCO2
- Dilute streams: $40-120/tCO2

## Primary Data Sources

1. **POSCO Holdings**
   - Sustainability Report 2023
   - HyREX Technology Roadmap
   - [newsroom.posco.com](https://newsroom.posco.com)

2. **Global Efficiency Intelligence**
   - Green Steel Economics Report (2024)
   - [globalefficiencyintel.com](https://www.globalefficiencyintel.com/green-steel-economics)

3. **Transition Asia**
   - Green Steel Economics Factsheets
   - [transitionasia.org](https://transitionasia.org/greensteeleconomics_es/)

4. **International Energy Agency (IEA)**
   - Global Hydrogen Review 2024
   - Iron and Steel Technology Roadmap

5. **Academic Sources**
   - Rubin et al. (2015) - Learning rates for electricity supply technologies
   - Nagy et al. (2013) - Statistical basis for predicting technological progress
   - Vogl et al. (2018) - Assessment of hydrogen direct reduction

6. **Industry Reports**
   - Columbia Business School CKI Steel Reports
   - IEEFA Steel Decarbonization Analysis
   - BloombergNEF New Energy Outlook

## Methodology Notes

### Abatement Calculation
Abatement $a_j$ = Baseline emissions - Technology emissions
- Baseline: BF-BOF at 2.2 tCO2/ton
- Measured relative to baseline (0 = no improvement)

### Correlation Matrix
Correlations inferred from:
- Shared input dependencies (H2, electricity)
- Technology families (H2-based: ρ ≈ 0.4-0.5)
- Cross-technology (ρ ≈ 0.1-0.3)

### Option Value
Embedded option value estimated via Black-Scholes for:
- Expansion options (modular technologies)
- Switching options (fuel flexibility)
- Abandonment options (salvage value)

## Data Update Schedule

This data reflects research conducted in November 2024. Key parameters should be updated:
- Quarterly: Hydrogen prices, carbon prices
- Annually: Technology costs, learning curves
- As announced: Major project milestones

## Contact

For questions about data methodology:
- Jinsu Park, PLANiT Institute
- jinsu.park@planit.institute
