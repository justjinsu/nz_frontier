import numpy as np
from typing import List, Optional
from .types import Technology, FrontierPoint
from .risk import RiskModel
from .optimizer import OptimizationEngine

class EfficientFrontier:
    """
    Computes the Net-Zero Efficient Frontier in (σ, C) space.

    Framework:
    - For each abatement target A* ∈ [A_min, A_max]
    - Solve: min C_P(w) + (λ/2)σ_P²(w)  s.t. A_P(w) ≥ A*
    - Plot resulting (σ_P, C_P) points

    This parallels Markowitz:
    - Markowitz: Plot (σ_P, μ_P) for different return targets
    - Net-Zero: Plot (σ_P, C_P) for different abatement targets
    """

    def __init__(self, technologies: List[Technology], covariance_matrix: np.ndarray):
        self.technologies = technologies
        self.risk_model = RiskModel(technologies, covariance_matrix)
        self.optimizer = OptimizationEngine(technologies, self.risk_model)

    def compute(
        self,
        abatement_min: float,
        abatement_max: float,
        n_points: int = 20,
        lambda_param: float = 1.0,
        budget_constraint: Optional[float] = None,
        return_breakdown: bool = True,
        # Deprecated parameters (kept for backward compatibility)
        gamma_param: Optional[float] = None,
    ) -> List[FrontierPoint]:
        """
        Computes the efficient frontier by varying abatement target A*.

        For each A* in [A_min, A_max]:
            Solve: minimize C_P(w) + (λ/2)σ_P²(w)
                   s.t. A_P(w) ≥ A*

        Returns a list of FrontierPoint objects containing:
        - volatility (σ_P): Cost volatility
        - expected_cost (C_P): Expected cost
        - abatement (A_P): Abatement achieved
        - portfolio: Optimal portfolio weights

        Args:
            abatement_min: Minimum abatement target
            abatement_max: Maximum abatement target
            n_points: Number of frontier points to compute
            lambda_param: Risk aversion parameter (higher = prefer lower volatility)
            budget_constraint: Optional budget limit (deprecated)
            return_breakdown: If True, include risk breakdown in each point

        Returns:
            List[FrontierPoint]: Frontier points in (σ, C) space

        Example:
            >>> frontier = EfficientFrontier(technologies, cov_matrix)
            >>> points = frontier.compute(
            ...     abatement_min=10,
            ...     abatement_max=50,
            ...     n_points=20,
            ...     lambda_param=1.5
            ... )
            >>> # Plot in (σ, C) space
            >>> sigmas = [p.volatility for p in points]
            >>> costs = [p.expected_cost for p in points]
            >>> plt.plot(sigmas, costs)
        """
        targets = np.linspace(abatement_min, abatement_max, n_points)
        results: List[FrontierPoint] = []

        for target in targets:
            try:
                result = self.optimizer.solve_for_target(
                    target_abatement=target,
                    lambda_param=lambda_param,
                    budget_constraint=budget_constraint,
                    return_breakdown=return_breakdown,
                )

                if return_breakdown:
                    portfolio, breakdown = result  # type: ignore[misc]
                else:
                    portfolio = result  # type: ignore[assignment]
                    breakdown = None

                # Compute metrics for new framework
                if breakdown is None:
                    breakdown = self.risk_model.breakdown(portfolio.weights)

                volatility = self.risk_model.cost_volatility(portfolio.weights)
                expected_cost = self.risk_model.expected_cost(portfolio.weights)

                # For backward compatibility, also compute old "risk"
                risk_old = self.risk_model.total_risk(
                    portfolio.weights,
                    lambda_param=lambda_param if lambda_param else 1.0,
                    gamma_param=gamma_param if gamma_param else 1.0
                )

                results.append(
                    FrontierPoint(
                        volatility=volatility,
                        expected_cost=expected_cost,
                        abatement=target,
                        portfolio=portfolio,
                        breakdown=breakdown,
                        risk=risk_old,  # Deprecated
                    )
                )
            except RuntimeError as e:
                # Keep frontier construction robust to infeasible points
                print(f"Could not solve for target {target}: {e}")

        return results

    def compute_volatility_frontier(
        self,
        target_abatement: float,
        volatility_min: float,
        volatility_max: float,
        n_points: int = 20,
        budget_constraint: Optional[float] = None,
    ) -> List[FrontierPoint]:
        """
        Computes frontier by varying volatility constraint (Markowitz-style).

        For each σ_max in [σ_min, σ_max]:
            Solve: minimize C_P(w)
                   s.t. A_P(w) ≥ A*
                        σ_P(w) ≤ σ_max

        This gives the classic Markowitz interpretation:
        "What's the minimum cost to achieve A* with volatility ≤ σ?"

        Args:
            target_abatement: Fixed abatement target A*
            volatility_min: Minimum allowed volatility
            volatility_max: Maximum allowed volatility
            n_points: Number of points to compute
            budget_constraint: Optional budget limit

        Returns:
            List[FrontierPoint]: Frontier points showing cost vs. volatility tradeoff
        """
        volatilities = np.linspace(volatility_min, volatility_max, n_points)
        results: List[FrontierPoint] = []

        for vol_max in volatilities:
            try:
                portfolio, breakdown = self.optimizer.solve_for_volatility_constraint(
                    target_abatement=target_abatement,
                    max_volatility=vol_max,
                    budget_constraint=budget_constraint,
                    return_breakdown=True,
                )

                volatility = self.risk_model.cost_volatility(portfolio.weights)
                expected_cost = self.risk_model.expected_cost(portfolio.weights)

                results.append(
                    FrontierPoint(
                        volatility=volatility,
                        expected_cost=expected_cost,
                        abatement=target_abatement,
                        portfolio=portfolio,
                        breakdown=breakdown,
                    )
                )
            except RuntimeError as e:
                print(f"Could not solve for volatility {vol_max}: {e}")

        return results
