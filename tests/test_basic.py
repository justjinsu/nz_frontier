import unittest
import numpy as np
from nz_frontier import Technology, Portfolio, RiskModel, OptimizationEngine

class TestNZFrontier(unittest.TestCase):
    def setUp(self):
        self.tech1 = Technology(name="Tech A", a=1.0, c=10.0, sigma=0.1)
        self.tech2 = Technology(name="Tech B", a=2.0, c=20.0, sigma=0.2)
        self.technologies = [self.tech1, self.tech2]
        self.cov_matrix = np.array([[0.01, 0.0], [0.0, 0.04]])
        self.risk_model = RiskModel(self.technologies, self.cov_matrix)
        self.optimizer = OptimizationEngine(self.technologies, self.risk_model)

    def test_risk_calculation(self):
        weights = np.array([0.5, 0.5])
        risk = self.risk_model.total_risk(weights)
        # Variance: 0.5^2 * 0.01 + 0.5^2 * 0.04 = 0.25 * 0.01 + 0.25 * 0.04 = 0.0025 + 0.01 = 0.0125
        # Stranded risk: 0.5 * 0.1 * 0 + 0.5 * 0.2 * 0 = 0 (tau=0)
        self.assertAlmostEqual(risk, 0.0125)

    def test_optimization(self):
        target_abatement = 1.5
        portfolio = self.optimizer.solve_for_target(target_abatement)
        self.assertGreaterEqual(portfolio.total_abatement, target_abatement - 1e-6)

if __name__ == '__main__':
    unittest.main()
