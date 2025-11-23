import numpy as np
from scipy.optimize import minimize
from typing import List, Tuple, Optional
from .types import Technology, Portfolio
from .risk import RiskModel

class OptimizationEngine:
    def __init__(self, technologies: List[Technology], risk_model: RiskModel):
        self.technologies = technologies
        self.risk_model = risk_model
        self.n_tech = len(technologies)

    def solve_for_target(self, 
                         target_abatement: float, 
                         budget_constraint: Optional[float] = None,
                         lambda_param: float = 1.0,
                         gamma_param: float = 1.0) -> Portfolio:
        """
        Solves the minimization problem for a specific abatement target A*.
        
        min R_P(w)
        s.t. sum(w_j * a_j) >= A*
             sum(w_j * c_j) <= B (if budget_constraint provided)
             w >= 0
        """
        
        # Objective function
        def objective(w):
            return self.risk_model.total_risk(w, lambda_param, gamma_param)
        
        # Constraints
        constraints = [
            {'type': 'ineq', 'fun': lambda w: np.sum(w * np.array([t.a for t in self.technologies])) - target_abatement}
        ]
        
        if budget_constraint is not None:
            constraints.append(
                {'type': 'ineq', 'fun': lambda w: budget_constraint - np.sum(w * np.array([t.c for t in self.technologies]))}
            )
            
        # Bounds (w >= 0)
        bounds = tuple((0, None) for _ in range(self.n_tech))
        
        # Initial guess (distribute target equally among techs, roughly)
        # A simple guess: 1.0 for each
        x0 = np.ones(self.n_tech)
        
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if not result.success:
            raise RuntimeError(f"Optimization failed: {result.message}")
            
        return Portfolio(weights=result.x, technologies=self.technologies)
