# Net-Zero Frontier

**Portfolio Theory for Corporate Decarbonization: A Risk-Efficiency Framework for Net-Zero Investment under Uncertainty**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://nz-frontier.streamlit.app)

## Overview

This repository implements a rigorous theoretical framework for corporate decarbonization investment, extending modern portfolio theory (Markowitz, 1952) to the climate transition context. The framework helps firms select optimal technology portfolios to meet emissions reduction targets while minimizing transition risk.

### Key Features

- **Monte Carlo Efficient Frontier** with 90% confidence bands
- **Full Correlation Matrix** capturing technology interdependencies
- **Dynamic Real Options** valuation with learning curves
- **Stochastic Dominance Analysis** for portfolio comparison
- **Robust Optimization** under parameter uncertainty
- **Jump-Diffusion Cost Simulation** with breakthrough innovations

## Theoretical Foundation

The core optimization problem minimizes portfolio transition risk subject to meeting an abatement target:

```
min_w  R_P(w) = w'Σw + λh(w) - γg(w)
s.t.   Σ w_j a_j ≥ A*  (abatement constraint)
       Σ w_j c_j ≤ B   (budget constraint)
       w ≥ 0           (non-negativity)
```

Where:
- **w'Σw**: Cost volatility (Markowitz variance)
- **λh(w)**: Stranded asset risk (failure probability × loss + maturity risk)
- **γg(w)**: Option value (expansion, switching, abandonment options)

## Installation

```bash
git clone https://github.com/jinsupark4/nz_frontier.git
cd nz_frontier
pip install -r requirements.txt
```

## Quick Start

### Run the Streamlit App

```bash
streamlit run app.py
```

### Use as a Python Library

```python
from nz_frontier import (
    Technology,
    MonteCarloFrontier,
    RobustOptimizer,
    build_correlation_matrix,
)
import numpy as np

# Define technologies
technologies = [
    Technology(name="BF-BOF + CCS", a=1.4, c=720, sigma=0.18,
               learning_rate=0.08, tau=25),
    Technology(name="Scrap-EAF", a=1.0, c=580, sigma=0.12,
               learning_rate=0.05, tau=25),
    Technology(name="H2-DRI", a=1.9, c=800, sigma=0.30,
               learning_rate=0.20, tau=30),
]

# Build correlation matrix (infers from technology characteristics)
cov_matrix = build_correlation_matrix(technologies)

# Compute Monte Carlo efficient frontier
mc_frontier = MonteCarloFrontier(technologies, cov_matrix)
result = mc_frontier.compute_frontier(
    abatement_min=20,
    abatement_max=100,
    budget_constraint=5000,
    n_simulations=500,
    time_horizon=15
)

# Results include confidence bands
print(f"Mean Risk: {result.mean_risk}")
print(f"90% CI: [{result.percentile_5}, {result.percentile_95}]")
```

## Case Studies

### Korean Steel Industry (Updated 2024-2025)

The repository includes research-calibrated data for South Korea's steel sector based on POSCO Sustainability Reports, Global Efficiency Intelligence, and IEA data:

| Technology | Abatement (tCO₂/t) | Cost ($/t) | Volatility | Learning Rate |
|------------|-------------------|------------|------------|---------------|
| BF-BOF (Baseline) | 0.0 | 390 | 0.05 | 0.01 |
| BF-BOF + CCUS | 1.60 | 465 | 0.15 | 0.06 |
| Scrap-EAF | 1.54 | 415 | 0.10 | 0.05 |
| NG-DRI-EAF | 1.25 | 455 | 0.12 | 0.08 |
| HyREX H2-DRI (POSCO) | 1.96 | 616 | 0.25 | 0.18 |
| Hy-Cube H2-DRI (Hyundai) | 1.94 | 600 | 0.24 | 0.16 |

**Key 2024-2025 Developments:**
- **POSCO HyREX**: Pilot success (April 2024) producing 24 tons/day at 0.4 tCO₂/t; BHP partnership for 300kt demo plant (October 2025)
- **Hyundai Hy-Cube**: $6B Louisiana plant announced using blue → green H2 transition
- **Green Premium**: ~$263/ton at H2 price $5/kg; cost-competitive at $1.5/kg with $15/tCO₂ carbon price

### Global Steel Comparison

Includes major green steel initiatives: HYBRIT (Sweden), H2 Green Steel/Stegra, ArcelorMittal, SALCOS (Germany), Nucor (US), and Asian projects.

### Korean Energy Sector

Includes power generation technologies: coal, LNG, nuclear (APR1400), solar PV, offshore wind, and green hydrogen.

## Module Structure

```
nz_frontier/
├── __init__.py
├── types.py          # Technology, Portfolio, RiskBreakdown dataclasses
├── risk.py           # Risk model: R_P(w) = w'Σw + λh(w) - γg(w)
├── optimizer.py      # Constrained optimization (SLSQP)
├── frontier.py       # Efficient frontier computation
├── simulation.py     # Jump-diffusion cost simulation
├── valuation.py      # Black-Scholes option pricing + PDE solver
├── dynamic.py        # Multi-period MPC optimization
└── advanced.py       # Monte Carlo frontier, robust optimization,
                      # stochastic dominance, correlation matrix
```

## App Features (6 Tabs)

1. **Monte Carlo Frontier**: Stochastic efficient frontier with confidence bands
2. **Risk Decomposition**: Visual breakdown of cost volatility, stranded asset risk, option value
3. **Dynamic Transition**: Multi-period optimization with irreversibility constraints
4. **Stochastic Dominance**: First-order and second-order dominance testing
5. **Robust Optimization**: Min-max portfolio under parameter uncertainty
6. **Cost Simulation**: Jump-diffusion paths with learning curves

## CSV Data Format

```csv
name,a,c,sigma,rho,o,tau,jump_intensity,jump_size,learning_rate,failure_prob,loss_given_failure
Technology A,1.5,100,0.2,0.0,10,20,0.05,-0.1,0.15,0.05,80
Technology B,2.0,150,0.3,0.0,15,25,0.08,-0.2,0.20,0.08,120
```

| Column | Description |
|--------|-------------|
| `name` | Technology name |
| `a` | Abatement potential (tCO₂/unit) |
| `c` | Capital cost ($/unit) |
| `sigma` | Cost volatility |
| `tau` | Capital lifetime (years) |
| `learning_rate` | Wright's Law learning rate (α) |
| `jump_intensity` | Poisson intensity for breakthroughs |
| `jump_size` | Jump size (negative = cost reduction) |
| `failure_prob` | Technology failure probability |
| `loss_given_failure` | Loss given failure ($/unit) |

## Data Sources

Technology parameters are calibrated from peer-reviewed literature and industry reports. See [`data/DATA_SOURCES.md`](data/DATA_SOURCES.md) for complete methodology.

**Primary Sources:**
- [POSCO Sustainability Report 2023-2024](https://sustainability.posco.com)
- [Global Efficiency Intelligence - Green Steel Economics](https://www.globalefficiencyintel.com/green-steel-economics)
- [Transition Asia - Green Steel Factsheets](https://transitionasia.org/greensteeleconomics_es/)
- [IEA Global Hydrogen Review 2024](https://www.iea.org/reports/global-hydrogen-review-2024)
- [Columbia Business School CKI Steel Reports](https://business.columbia.edu/insights/climate/cki/steel)

## References

### Theoretical Foundations
- Markowitz, H. (1952). Portfolio Selection. *Journal of Finance*, 7(1), 77-91.
- Dixit, A. K., & Pindyck, R. S. (1994). *Investment under Uncertainty*. Princeton University Press.
- Arrow, K. J. (1962). The Economic Implications of Learning by Doing. *Review of Economic Studies*, 29(3), 155-173.

### Steel Decarbonization
- Vogl, V., Åhman, M., & Nilsson, L. J. (2018). Assessment of hydrogen direct reduction for fossil-free steelmaking. *Journal of Cleaner Production*, 203, 736-745.
- Rubin, E. S., et al. (2015). A review of learning rates for electricity supply technologies. *Energy Policy*, 86, 198-218.
- Nagy, B., et al. (2013). Statistical basis for predicting technological progress. *PLoS ONE*, 8(2), e52669.

### Industry Reports
- POSCO Newsroom (2024). [HyREX Pilot Success](https://newsroom.posco.com/en/from-ccus-to-hyrex-the-full-lineup-of-posco-groups-decarbonization-strategies-for-a-sustainable-steel-industry/)
- For Our Climate (2024). [Korea Steel Investment Gap Analysis](https://forourclimate.org/research/521)
- GMK Center (2024). [Green Steel in South Korea](https://gmk.center/en/posts/prospects-for-south-korean-green-steel/)

## Citation

```bibtex
@article{park2025portfolio,
  title={Portfolio Theory for Corporate Decarbonization: A Risk-Efficiency Framework for Net-Zero Investment under Uncertainty},
  author={Park, Jinsu},
  journal={Working Paper},
  year={2025},
  institution={PLANiT Institute}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Author

**Jinsu Park**
PLANiT Institute
jinsu.park@planit.institute
