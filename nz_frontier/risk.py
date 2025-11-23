import numpy as np
from typing import List
from .types import Technology, RiskBreakdown

from .valuation import OptionValuator

class RiskModel:
    """
    Implements the risk function R_P(w) = w^T Σ w + λ h(w) − γ g(w) from paper_v2.tex.
    """

    def __init__(
        self,
        technologies: List[Technology],
        covariance_matrix: np.ndarray,
        r: float = 0.05,
        lambda_param: float = 1.0,
        gamma_param: float = 1.0,
    ):
        self.technologies = technologies
        self.covariance_matrix = covariance_matrix
        self.lambda_param = lambda_param
        self.gamma_param = gamma_param
        self.valuator = OptionValuator(r=r)

        if covariance_matrix.shape != (len(technologies), len(technologies)):
            raise ValueError("Covariance matrix shape must match number of technologies")
        self._validate_covariance()

    def _validate_covariance(self) -> None:
        """
        Ensures the covariance matrix is symmetric positive semi-definite, as required by Assumption 2.
        """
        if not np.allclose(self.covariance_matrix, self.covariance_matrix.T, atol=1e-10):
            raise ValueError("Covariance matrix must be symmetric")

        # Eigenvalues must be non-negative within tolerance
        min_eig = np.min(np.linalg.eigvalsh(self.covariance_matrix))
        if min_eig < -1e-10:
            raise ValueError("Covariance matrix must be positive semi-definite")

    def _validate_weights(self, weights: np.ndarray) -> np.ndarray:
        weights = np.asarray(weights, dtype=float)
        if weights.shape[0] != len(self.technologies):
            raise ValueError("Weight vector length must match number of technologies")
        return weights

    def portfolio_variance(self, weights: np.ndarray) -> float:
        """
        Base portfolio variance: w^T Σ w (cost volatility component).
        """
        w = self._validate_weights(weights)
        return float(w.T @ self.covariance_matrix @ w)

    def stranded_asset_risk(self, weights: np.ndarray) -> float:
        """
        Implements h(w) = Σ w_j [π_j L_j + σ_j sqrt(τ_j)] from Equation (4).
        """
        w = self._validate_weights(weights)
        risk = 0.0
        for weight, tech in zip(w, self.technologies):
            failure_risk = tech.failure_prob * tech.loss_given_failure
            maturity_risk = tech.sigma * np.sqrt(max(tech.tau, 0.0))
            tech_risk = failure_risk + maturity_risk
            risk += weight * tech_risk
        return risk

    def option_value(self, weights: np.ndarray) -> float:
        """
        Implements g(w) = Σ w_j o_j, using model-based option value if explicit o_j is absent.
        """
        w = self._validate_weights(weights)
        value = 0.0
        for weight, tech in zip(w, self.technologies):
            # Use explicit embedded value if provided
            if tech.o > 0:
                ov = tech.o
            elif tech.strike_price > 0 and tech.tau > 0:
                # Use Black-Scholes as the analytical solution highlighted in paper_v2.tex
                underlying = max(tech.c, 1e-6)  # prevent zero underlying
                ov = self.valuator.black_scholes_call(
                    S=underlying, K=tech.strike_price, T=tech.tau, sigma=max(tech.sigma, 1e-6)
                )
            else:
                ov = 0.0

            value += weight * ov
        return value

    def breakdown(self, weights: np.ndarray, lambda_param: float = None, gamma_param: float = None) -> RiskBreakdown:
        """
        Returns a RiskBreakdown dataclass with each component and total risk.
        """
        lam = self.lambda_param if lambda_param is None else lambda_param
        gam = self.gamma_param if gamma_param is None else gamma_param

        cost_vol = self.portfolio_variance(weights)
        stranded = self.stranded_asset_risk(weights)
        opt_val = self.option_value(weights)
        total = cost_vol + lam * stranded - gam * opt_val

        return RiskBreakdown(cost_volatility=cost_vol, stranded_asset=stranded, option_value=opt_val, total=total)

    def total_risk(self, weights: np.ndarray, lambda_param: float = None, gamma_param: float = None) -> float:
        """
        Calculates total portfolio transition risk R_P(w) = w^T Σ w + λ h(w) − γ g(w).
        """
        return self.breakdown(weights, lambda_param=lambda_param, gamma_param=gamma_param).total
