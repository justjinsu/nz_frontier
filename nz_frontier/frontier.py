import numpy as np
from typing import List, Optional
from .types import Technology, FrontierPoint
from .risk import RiskModel
from .optimizer import OptimizationEngine

class EfficientFrontier:
    def __init__(self, technologies: List[Technology], covariance_matrix: np.ndarray):
        self.technologies = technologies
        self.risk_model = RiskModel(technologies, covariance_matrix)
        self.optimizer = OptimizationEngine(technologies, self.risk_model)

    def compute(
        self,
        abatement_min: float,
        abatement_max: float,
        n_points: int = 20,
        budget_constraint: Optional[float] = None,
        lambda_param: Optional[float] = None,
        gamma_param: Optional[float] = None,
        return_breakdown: bool = False,
    ) -> List[FrontierPoint]:
        """
        Computes the net-zero efficient frontier R_P^*(A) described in Section 4.

        Returns:
            List[FrontierPoint]
        """
        targets = np.linspace(abatement_min, abatement_max, n_points)
        results: List[FrontierPoint] = []

        for target in targets:
            try:
                result = self.optimizer.solve_for_target(
                    target,
                    budget_constraint=budget_constraint,
                    lambda_param=lambda_param,
                    gamma_param=gamma_param,
                    return_breakdown=return_breakdown,
                )

                if return_breakdown:
                    portfolio, breakdown = result  # type: ignore[misc]
                else:
                    portfolio = result  # type: ignore[assignment]
                    breakdown = None

                # Always compute risk to keep output consistent
                if breakdown is None:
                    breakdown = self.risk_model.breakdown(
                        portfolio.weights, lambda_param=lambda_param, gamma_param=gamma_param
                    )

                results.append(FrontierPoint(risk=breakdown.total, abatement=target, portfolio=portfolio, breakdown=breakdown))
            except RuntimeError as e:
                # Keep frontier construction robust to infeasible points
                print(f"Could not solve for target {target}: {e}")

        return results
