import numpy as np
from typing import List
from .types import Technology

from .valuation import OptionValuator

class RiskModel:
    def __init__(self, technologies: List[Technology], covariance_matrix: np.ndarray, r: float = 0.05):
        self.technologies = technologies
        self.covariance_matrix = covariance_matrix
        self.valuator = OptionValuator(r=r)
        
        if covariance_matrix.shape != (len(technologies), len(technologies)):
            raise ValueError("Covariance matrix shape must match number of technologies")

    def portfolio_variance(self, weights: np.ndarray) -> float:
        """
        Calculates the base portfolio variance: w^T * Sigma * w
        """
        return float(weights.T @ self.covariance_matrix @ weights)

    def stranded_asset_risk(self, weights: np.ndarray) -> float:
        """
        Implements the stranded asset risk function h(w).
        R_j = pi_j * L_j + ... + nu_j * S_j(T)
        
        Theorem 1 Decomposition:
        1. Technology Failure: pi_j * L_j
        2. Cost Uncertainty: Integral(sigma^2) -> handled in Variance part usually, but here we can add residual
        3. Stranded Value: nu_j * S_j(T) -> approximated by maturity risk
        """
        risk = 0.0
        for w, t in zip(weights, self.technologies):
            # 1. Failure Risk (Explicit)
            failure_risk = t.failure_prob * t.loss_given_failure
            
            # 2. Maturity/Lock-in Risk (Proxy for S_j(T))
            # Longer tau means capital is locked in longer, higher chance of being stranded by new tech
            maturity_risk = t.sigma * np.sqrt(max(t.tau, 1.0))
            
            # Total Stranded Risk per unit
            tech_risk = failure_risk + maturity_risk
            
            risk += w * tech_risk
        return risk

    def option_value(self, weights: np.ndarray) -> float:
        """
        Implements the option value function g(w).
        Calculates real option value for each technology and aggregates.
        """
        value = 0.0
        for w, t in zip(weights, self.technologies):
            # If technology has option characteristics (o > 0 or defined strike)
            if t.strike_price > 0 and t.tau > 0:
                # Assume current state S_0 is related to current cost or market price
                # For a "switch" option, S might be the cost of incumbent vs this tech.
                # Let's assume S_0 = t.c (current cost) and we have an option to switch if cost drops?
                # Or option to expand?
                # Paper says: O_j(S, T) = max(S - K, 0).
                # Let's use the pre-calculated 'o' if provided, otherwise calculate.
                if t.o > 0:
                    ov = t.o
                else:
                    # Calculate using PDE
                    # Assume S_0 = 100 (index), K = t.strike_price
                    ov = self.valuator.calculate_option_value(S_0=100.0, K=t.strike_price, T=t.tau, sigma=t.sigma)
                
                value += w * ov
            else:
                value += w * t.o
                
        return value

    def total_risk(self, weights: np.ndarray, lambda_param: float = 1.0, gamma_param: float = 1.0) -> float:
        """
        Calculates total portfolio transition risk R_P(w).
        R_P(w) = w^T * Sigma * w + lambda * h(w) - gamma * g(w)
        """
        base_risk = self.portfolio_variance(weights)
        stranded_risk = self.stranded_asset_risk(weights)
        opt_val = self.option_value(weights)
        
        return base_risk + lambda_param * stranded_risk - gamma_param * opt_val
