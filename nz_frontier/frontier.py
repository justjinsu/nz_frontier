import numpy as np
from typing import List, Tuple, Optional
from .types import Technology, Portfolio
from .risk import RiskModel
from .optimizer import OptimizationEngine

class EfficientFrontier:
    def __init__(self, technologies: List[Technology], covariance_matrix: np.ndarray):
        self.technologies = technologies
        self.risk_model = RiskModel(technologies, covariance_matrix)
        self.optimizer = OptimizationEngine(technologies, self.risk_model)
        
    def compute(self, 
                abatement_min: float, 
                abatement_max: float, 
                n_points: int = 20,
                budget_constraint: Optional[float] = None) -> List[Tuple[float, float, Portfolio]]:
        """
        Computes the efficient frontier.
        
        Returns:
            List of tuples (Risk, Abatement, Portfolio)
        """
        targets = np.linspace(abatement_min, abatement_max, n_points)
        results = []
        
        for target in targets:
            try:
                portfolio = self.optimizer.solve_for_target(target, budget_constraint)
                risk = self.risk_model.total_risk(portfolio.weights)
                results.append((risk, target, portfolio))
            except RuntimeError as e:
                print(f"Could not solve for target {target}: {e}")
                
        return results
