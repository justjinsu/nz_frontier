import numpy as np
from scipy.optimize import minimize
from typing import List, Optional, Tuple, Union
from .types import Technology, Portfolio, RiskBreakdown
from .risk import RiskModel

class OptimizationEngine:
    """
    Solves the Net-Zero Efficient Frontier optimization problem.

    New Framework:
        minimize: C_P(w) + (λ/2)·σ_P²(w)
        s.t.: A_P(w) ≥ A*  (abatement target)
              w_j ≥ 0, w_j ≤ w̄_j  (capacity bounds)

    This exactly parallels Markowitz mean-variance optimization:
        maximize: μ_P - (γ/2)σ_P²
        s.t.: budget constraint

    The key difference: firms choose λ (risk aversion), not the abatement
    target A* (which is externally determined by regulation/commitment).
    """

    def __init__(self, technologies: List[Technology], risk_model: RiskModel):
        self.technologies = technologies
        self.risk_model = risk_model
        self.n_tech = len(technologies)
        self.abatements = np.array([t.a for t in technologies])
        self.costs = np.array([t.c for t in technologies])
        self.max_capacities = np.array([t.max_capacity for t in technologies])

    def _build_constraints(self, target_abatement: float, budget_constraint: Optional[float] = None) -> List[dict]:
        """Build constraints for optimization problem."""
        constraints = [
            {"type": "ineq", "fun": lambda w: np.sum(w * self.abatements) - target_abatement}
        ]

        if budget_constraint is not None:
            constraints.append(
                {"type": "ineq", "fun": lambda w: budget_constraint - np.sum(w * self.costs)}
            )
        return constraints

    def solve_for_target(
        self,
        target_abatement: float,
        lambda_param: float = 1.0,
        budget_constraint: Optional[float] = None,
        return_breakdown: bool = False,
        initial_guess: Optional[np.ndarray] = None,
        # Deprecated parameters (kept for backward compatibility)
        gamma_param: Optional[float] = None,
    ) -> Union[Portfolio, Tuple[Portfolio, RiskBreakdown]]:
        """
        Solve for optimal portfolio at given abatement target.

        minimize: C_P(w) + (λ/2)·σ_P²(w)
        s.t.: A_P(w) ≥ A*
              w_j ∈ [0, w̄_j]

        Args:
            target_abatement: Minimum abatement target A* (Mt CO2)
            lambda_param: Risk aversion parameter λ (higher = more conservative)
            budget_constraint: Optional budget limit (deprecated, use capacity constraints)
            return_breakdown: If True, return (Portfolio, RiskBreakdown) tuple
            initial_guess: Optional starting point for optimization

        Returns:
            Portfolio object (or tuple with RiskBreakdown if requested)

        Example:
            >>> optimizer = OptimizationEngine(technologies, risk_model)
            >>> portfolio = optimizer.solve_for_target(
            ...     target_abatement=48.9,
            ...     lambda_param=1.5  # Conservative investor
            ... )
            >>> print(f"Expected cost: ${portfolio.total_cost:.1f}M")
            >>> print(f"Cost volatility: {risk_model.cost_volatility(portfolio.weights):.1f}%")
        """

        # Objective function: C_P + (λ/2)σ_P²
        def objective(w):
            return self.risk_model.objective_function(w, lambda_param=lambda_param)

        constraints = self._build_constraints(target_abatement, budget_constraint)

        # Bounds: 0 ≤ w_j ≤ w̄_j
        bounds = tuple((0, cap) for cap in self.max_capacities)

        # Initial guess
        if initial_guess is None:
            # Simple heuristic: distribute capacity proportional to abatement potential
            if np.sum(self.abatements) > 0:
                initial_guess = (target_abatement / np.sum(self.abatements)) * self.abatements
                # Clip to capacity bounds
                initial_guess = np.clip(initial_guess, 0, self.max_capacities)
            else:
                initial_guess = np.full(self.n_tech, 1.0)

        result = minimize(objective, initial_guess, method="SLSQP", bounds=bounds, constraints=constraints)

        if not result.success:
            raise RuntimeError(f"Optimization failed: {result.message}")

        portfolio = Portfolio(weights=result.x, technologies=self.technologies)

        if return_breakdown:
            breakdown = self.risk_model.breakdown(result.x)
            return portfolio, breakdown
        return portfolio

    def solve_for_volatility_constraint(
        self,
        target_abatement: float,
        max_volatility: float,
        budget_constraint: Optional[float] = None,
        return_breakdown: bool = False,
        initial_guess: Optional[np.ndarray] = None,
    ) -> Union[Portfolio, Tuple[Portfolio, RiskBreakdown]]:
        """
        Solve for minimum-cost portfolio with volatility constraint.

        minimize: C_P(w)
        s.t.: A_P(w) ≥ A*
              σ_P(w) ≤ σ_max
              w_j ∈ [0, w̄_j]

        This is the "Markowitz-style" formulation: minimize cost for given
        risk level, rather than trading off cost vs. volatility.

        Args:
            target_abatement: Minimum abatement target A*
            max_volatility: Maximum allowed cost volatility σ_max
            budget_constraint: Optional budget limit
            return_breakdown: If True, return breakdown
            initial_guess: Optional starting point

        Returns:
            Portfolio object (or tuple with breakdown)
        """

        # Objective: minimize expected cost only
        def objective(w):
            return self.risk_model.expected_cost(w)

        # Constraints
        constraints = [
            # Abatement constraint
            {"type": "ineq", "fun": lambda w: np.sum(w * self.abatements) - target_abatement},
            # Volatility constraint
            {"type": "ineq", "fun": lambda w: max_volatility - self.risk_model.cost_volatility(w)},
        ]

        if budget_constraint is not None:
            constraints.append(
                {"type": "ineq", "fun": lambda w: budget_constraint - np.sum(w * self.costs)}
            )

        bounds = tuple((0, cap) for cap in self.max_capacities)

        if initial_guess is None:
            if np.sum(self.abatements) > 0:
                initial_guess = (target_abatement / np.sum(self.abatements)) * self.abatements
                initial_guess = np.clip(initial_guess, 0, self.max_capacities)
            else:
                initial_guess = np.full(self.n_tech, 1.0)

        result = minimize(objective, initial_guess, method="SLSQP", bounds=bounds, constraints=constraints)

        if not result.success:
            # Volatility constraint might be infeasible - try without it
            print(f"Warning: Volatility constraint σ ≤ {max_volatility} may be infeasible. Trying without constraint...")
            constraints_relaxed = [c for c in constraints if 'volatility' not in str(c)]
            result = minimize(objective, initial_guess, method="SLSQP", bounds=bounds, constraints=constraints_relaxed)

            if not result.success:
                raise RuntimeError(f"Optimization failed: {result.message}")

        portfolio = Portfolio(weights=result.x, technologies=self.technologies)

        if return_breakdown:
            breakdown = self.risk_model.breakdown(result.x)
            return portfolio, breakdown
        return portfolio
