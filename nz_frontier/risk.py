import numpy as np
from typing import List
from .types import Technology, RiskBreakdown

from .valuation import OptionValuator

class RiskModel:
    """
    Implements the revised Net-Zero Efficient Frontier framework.

    Key Separation (Markowitz-style):
    - Cost Volatility σ_P(w): Pure uncertainty (2nd moment)
    - Expected Cost C_P(w): Risk-adjusted expected cost (1st moment)

    This replaces the old R_P(w) = w'Σw + λh(w) - γg(w) formulation,
    which incorrectly mixed variance with expected values.
    """

    def __init__(
        self,
        technologies: List[Technology],
        covariance_matrix: np.ndarray,
        r: float = 0.05,
    ):
        self.technologies = technologies
        self.covariance_matrix = covariance_matrix
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

    def cost_volatility(self, weights: np.ndarray) -> float:
        """
        Cost Volatility (Pure Uncertainty): σ_P = sqrt(w'Σw)

        Measures the standard deviation of portfolio costs due to:
        - Technology cost fluctuations
        - Correlated risks across technologies

        This is a pure second-moment measure, exactly analogous to
        portfolio volatility in Markowitz theory.

        Returns:
            float: Standard deviation of portfolio costs (in $/unit or %)
        """
        w = self._validate_weights(weights)
        variance = float(w.T @ self.covariance_matrix @ w)
        return np.sqrt(max(variance, 0.0))  # Ensure non-negative under numerical noise

    def portfolio_variance(self, weights: np.ndarray) -> float:
        """
        Portfolio Variance: w'Σw

        This is kept for backward compatibility and internal use.
        For user-facing analysis, use cost_volatility() which returns σ_P.
        """
        w = self._validate_weights(weights)
        return float(w.T @ self.covariance_matrix @ w)

    def _risk_adjusted_unit_cost(self, tech: Technology) -> float:
        """
        Risk-Adjusted Unit Cost: c̃_j = c_j + π_j·L_j - o_j

        Components:
        - c_j: Base capital cost per unit capacity
        - π_j·L_j: Expected stranded asset loss
        - o_j: Expected option value (flexibility benefit)

        All components are first moments (expected values), so they
        can be consistently summed.
        """
        # Base cost
        base_cost = tech.c

        # Expected stranded asset loss
        stranded_loss = tech.failure_prob * tech.loss_given_failure

        # Option value (use explicit if available, else compute)
        if tech.o > 0:
            option_val = tech.o
        elif tech.strike_price > 0 and tech.tau > 0:
            underlying = max(tech.c, 1e-6)
            option_val = self.valuator.black_scholes_call(
                S=underlying, K=tech.strike_price, T=tech.tau, sigma=max(tech.sigma, 1e-6)
            )
        else:
            option_val = 0.0

        return base_cost + stranded_loss - option_val

    def expected_cost(self, weights: np.ndarray) -> float:
        """
        Expected Portfolio Cost: C_P(w) = Σ w_j·c̃_j

        where c̃_j = c_j + π_j·L_j - o_j (risk-adjusted unit cost)

        This is a pure first-moment measure (expected value), consisting of:
        - Base capital costs
        - Expected stranded asset losses
        - Expected option values (subtracted as benefits)

        Returns:
            float: Expected total portfolio cost (in $M or appropriate units)
        """
        w = self._validate_weights(weights)
        total_cost = 0.0

        for weight, tech in zip(w, self.technologies):
            adjusted_cost = self._risk_adjusted_unit_cost(tech)
            total_cost += weight * adjusted_cost

        return total_cost

    def breakdown(self, weights: np.ndarray) -> RiskBreakdown:
        """
        Returns a RiskBreakdown with cost volatility and expected cost components.

        Note: In the new framework:
        - cost_volatility = σ_P (standard deviation)
        - stranded_asset = total stranded asset expected loss (component of C_P)
        - option_value = total option value (component of C_P)
        - total = C_P (expected cost)
        """
        w = self._validate_weights(weights)

        # Cost volatility (σ_P)
        cost_vol = self.cost_volatility(weights)

        # Expected cost components
        stranded = sum(w_j * tech.failure_prob * tech.loss_given_failure
                      for w_j, tech in zip(w, self.technologies))

        opt_val = sum(w_j * self._option_value_single(tech)
                     for w_j, tech in zip(w, self.technologies))

        # Total expected cost
        total = self.expected_cost(weights)

        return RiskBreakdown(
            cost_volatility=cost_vol,
            stranded_asset=stranded,
            option_value=opt_val,
            total=total
        )

    def _option_value_single(self, tech: Technology) -> float:
        """Helper to get option value for a single technology."""
        if tech.o > 0:
            return tech.o
        elif tech.strike_price > 0 and tech.tau > 0:
            underlying = max(tech.c, 1e-6)
            return self.valuator.black_scholes_call(
                S=underlying, K=tech.strike_price, T=tech.tau, sigma=max(tech.sigma, 1e-6)
            )
        return 0.0

    def objective_function(self, weights: np.ndarray, lambda_param: float) -> float:
        """
        Objective Function: J(w) = C_P(w) + (λ/2)·σ_P²(w)

        where:
        - C_P(w): Expected cost (1st moment)
        - σ_P(w): Cost volatility (2nd moment)
        - λ: Investor-specific risk aversion parameter

        This exactly parallels Markowitz's mean-variance objective:
            maximize: μ_P - (γ/2)σ_P²
        becomes in our context:
            minimize: C_P + (λ/2)σ_P²

        Different investors with different λ will trace different
        efficient portfolios.

        Args:
            weights: Portfolio weights (capacity allocations)
            lambda_param: Risk aversion (cost-volatility penalty)

        Returns:
            float: Total objective value to minimize
        """
        C_P = self.expected_cost(weights)
        sigma_P = self.cost_volatility(weights)

        return C_P + (lambda_param / 2.0) * (sigma_P ** 2)

    # ========================================================================
    # BACKWARD COMPATIBILITY METHODS
    # ========================================================================
    # These methods maintain compatibility with old code that expects
    # the R_P(w) formulation. They map to the new framework.

    def stranded_asset_risk(self, weights: np.ndarray) -> float:
        """
        DEPRECATED: Use expected_cost() components instead.

        Returns total expected stranded asset loss (part of C_P).
        """
        w = self._validate_weights(weights)
        return sum(w_j * tech.failure_prob * tech.loss_given_failure
                  for w_j, tech in zip(w, self.technologies))

    def option_value(self, weights: np.ndarray) -> float:
        """
        DEPRECATED: Use expected_cost() components instead.

        Returns total option value (part of C_P).
        """
        w = self._validate_weights(weights)
        return sum(w_j * self._option_value_single(tech)
                  for w_j, tech in zip(w, self.technologies))

    def total_risk(self, weights: np.ndarray, lambda_param: float = None, gamma_param: float = None) -> float:
        """
        DEPRECATED: Use objective_function() instead.

        For backward compatibility, this returns the old R_P(w) formulation.
        New code should use:
            - cost_volatility(w) for uncertainty
            - expected_cost(w) for expected cost
            - objective_function(w, λ) for optimization
        """
        if lambda_param is None:
            lambda_param = 1.0
        if gamma_param is None:
            gamma_param = 1.0

        variance = self.portfolio_variance(weights)
        stranded = self.stranded_asset_risk(weights)
        opt_val = self.option_value(weights)

        return variance + lambda_param * stranded - gamma_param * opt_val
