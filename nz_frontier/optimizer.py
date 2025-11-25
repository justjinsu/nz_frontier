import numpy as np
from scipy.optimize import minimize
from typing import List, Optional, Tuple, Union
from .types import Technology, Portfolio, RiskBreakdown
from .risk import RiskModel

class OptimizationEngine:
    """
    Solves the constrained risk minimization problem from Section 4 of paper_v2.tex.
    """

    def __init__(self, technologies: List[Technology], risk_model: RiskModel):
        self.technologies = technologies
        self.risk_model = risk_model
        self.n_tech = len(technologies)
        self.abatements = np.array([t.a for t in technologies])
        self.costs = np.array([t.c for t in technologies])
        self.max_capacities = np.array([t.max_capacity for t in technologies])

    def _build_constraints(self, target_abatement: float, budget_constraint: Optional[float]) -> List[dict]:
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
        budget_constraint: Optional[float] = None,
        lambda_param: Optional[float] = None,
        gamma_param: Optional[float] = None,
        return_breakdown: bool = False,
        initial_guess: Optional[np.ndarray] = None,
    ) -> Union[Portfolio, Tuple[Portfolio, RiskBreakdown]]:
        """
        Solves:
            min_w R_P(w)
            s.t. Σ w_j a_j ≥ A*
                 Σ w_j c_j ≤ B   (optional)
                 w_j ≥ 0
        """

        # Objective function
        def objective(w):
            return self.risk_model.total_risk(w, lambda_param=lambda_param, gamma_param=gamma_param)

        constraints = self._build_constraints(target_abatement, budget_constraint)

        # Bounds (0 <= w <= max_capacity for each technology)
        bounds = tuple((0, cap) for cap in self.max_capacities)

        # Initial guess (distribute target equally among techs, roughly)
        if initial_guess is None:
            initial_guess = np.full(self.n_tech, max(target_abatement / max(np.sum(self.abatements), 1e-6), 1.0))

        result = minimize(objective, initial_guess, method="SLSQP", bounds=bounds, constraints=constraints)

        if not result.success:
            raise RuntimeError(f"Optimization failed: {result.message}")

        portfolio = Portfolio(weights=result.x, technologies=self.technologies)
        if return_breakdown:
            breakdown = self.risk_model.breakdown(result.x, lambda_param=lambda_param, gamma_param=gamma_param)
            return portfolio, breakdown
        return portfolio
