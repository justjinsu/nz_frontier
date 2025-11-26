from dataclasses import dataclass
import numpy as np
from typing import List, Optional

@dataclass
class Technology:
    """
    Represents a low-carbon technology with its characteristics.

    Attributes:
        name: Name of the technology
        a: Abatement potential per unit capacity (tCO2/unit)
        c: Capital cost per unit capacity ($/unit)
        sigma: Cost volatility parameter
        rho: Technology-specific risk correlation (with other factors or market)
        o: Embedded option value
        tau: Technology maturity timeline (years)
        max_capacity: Maximum deployable capacity (Mt/year) - realistic constraint
    """
    name: str
    a: float
    c: float
    sigma: float
    rho: float = 0.0
    o: float = 0.0
    tau: float = 0.0
    # Jump-Diffusion Parameters
    jump_intensity: float = 0.0  # lambda for Poisson process
    jump_size: float = 0.0       # h for jump size
    # Option Parameters
    strike_price: float = 0.0    # K for option valuation
    # Deep Theory Parameters
    learning_rate: float = 0.0   # alpha for learning curve
    failure_prob: float = 0.0    # pi for stranded asset risk
    loss_given_failure: float = 0.0 # L for stranded asset risk
    # Capacity Constraints
    max_capacity: float = float('inf')  # Maximum capacity (Mt/year)

@dataclass
class Portfolio:
    """
    Represents a portfolio of technology deployments.
    """
    weights: np.ndarray  # Capacity of each technology deployed
    technologies: List[Technology]
    
    @property
    def total_abatement(self) -> float:
        return sum(w * t.a for w, t in zip(self.weights, self.technologies))
    
    @property
    def total_cost(self) -> float:
        return sum(w * t.c for w, t in zip(self.weights, self.technologies))

@dataclass
class RiskBreakdown:
    """
    Container for the risk decomposition R_P(w) = variance + lambda*h(w) - gamma*g(w).
    """
    cost_volatility: float
    stranded_asset: float
    option_value: float
    total: float

@dataclass
class FrontierPoint:
    """
    Represents one point on the efficient frontier.

    New Framework:
    - volatility: Cost volatility σ_P (uncertainty)
    - expected_cost: Expected portfolio cost C_P
    - abatement: Total abatement achieved

    Legacy fields (for backward compatibility):
    - risk: Old R_P measure (deprecated)
    """
    volatility: float  # σ_P: Cost volatility (2nd moment)
    expected_cost: float  # C_P: Expected cost (1st moment)
    abatement: float  # A_P: Total abatement
    portfolio: Portfolio
    breakdown: Optional[RiskBreakdown] = None

    # Deprecated field (kept for backward compatibility)
    risk: Optional[float] = None
