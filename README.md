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

### Korean Steel Industry

The repository includes calibrated data for South Korea's steel sector:

| Technology | Abatement (tCO₂/t) | Cost ($/t) | Volatility | Learning Rate |
|------------|-------------------|------------|------------|---------------|
| BF-BOF (Baseline) | 0.0 | 605 | 0.05 | 0.01 |
| BF-BOF + CCUS | 1.4 | 720 | 0.18 | 0.08 |
| Scrap-EAF | 1.0 | 580 | 0.12 | 0.05 |
| HyREX H2-DRI (POSCO) | 1.9 | 800 | 0.30 | 0.20 |
| Hy-Cube H2-DRI (Hyundai) | 1.85 | 780 | 0.28 | 0.18 |

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

## References

- Markowitz, H. (1952). Portfolio Selection. *Journal of Finance*, 7(1), 77-91.
- Dixit, A. K., & Pindyck, R. S. (1994). *Investment under Uncertainty*. Princeton University Press.
- Arrow, K. J. (1962). The Economic Implications of Learning by Doing. *Review of Economic Studies*, 29(3), 155-173.
- Vogl, V., Åhman, M., & Nilsson, L. J. (2018). Assessment of hydrogen direct reduction for fossil-free steelmaking. *Journal of Cleaner Production*, 203, 736-745.

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
