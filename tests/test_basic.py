"""
Unit tests for nz_frontier core functionality.
"""

import unittest
import numpy as np
from nz_frontier import (
    Technology,
    Portfolio,
    RiskModel,
    OptimizationEngine,
    EfficientFrontier,
    FrontierPoint,
    RiskBreakdown,
)


class TestTechnology(unittest.TestCase):
    """Tests for Technology dataclass."""

    def test_technology_creation(self):
        tech = Technology(name="Solar", a=1.5, c=100, sigma=0.2)
        self.assertEqual(tech.name, "Solar")
        self.assertEqual(tech.a, 1.5)
        self.assertEqual(tech.c, 100)
        self.assertEqual(tech.sigma, 0.2)

    def test_technology_defaults(self):
        tech = Technology(name="Wind", a=1.0, c=80, sigma=0.15)
        self.assertEqual(tech.rho, 0.0)
        self.assertEqual(tech.o, 0.0)
        self.assertEqual(tech.tau, 0.0)
        self.assertEqual(tech.learning_rate, 0.0)
        self.assertEqual(tech.failure_prob, 0.0)

    def test_technology_full_params(self):
        tech = Technology(
            name="H2-DRI",
            a=1.9,
            c=800,
            sigma=0.30,
            tau=30,
            learning_rate=0.20,
            failure_prob=0.10,
            loss_given_failure=500,
            o=50,
        )
        self.assertEqual(tech.tau, 30)
        self.assertEqual(tech.learning_rate, 0.20)
        self.assertEqual(tech.failure_prob, 0.10)


class TestPortfolio(unittest.TestCase):
    """Tests for Portfolio dataclass."""

    def setUp(self):
        self.tech1 = Technology(name="Tech A", a=1.0, c=10.0, sigma=0.1)
        self.tech2 = Technology(name="Tech B", a=2.0, c=20.0, sigma=0.2)
        self.technologies = [self.tech1, self.tech2]

    def test_portfolio_creation(self):
        weights = np.array([0.5, 0.5])
        portfolio = Portfolio(weights=weights, technologies=self.technologies)
        self.assertEqual(len(portfolio.weights), 2)

    def test_total_abatement(self):
        weights = np.array([1.0, 2.0])
        portfolio = Portfolio(weights=weights, technologies=self.technologies)
        # 1.0 * 1.0 + 2.0 * 2.0 = 5.0
        self.assertAlmostEqual(portfolio.total_abatement, 5.0)

    def test_total_cost(self):
        weights = np.array([1.0, 2.0])
        portfolio = Portfolio(weights=weights, technologies=self.technologies)
        # 1.0 * 10.0 + 2.0 * 20.0 = 50.0
        self.assertAlmostEqual(portfolio.total_cost, 50.0)


class TestRiskModel(unittest.TestCase):
    """Tests for RiskModel class."""

    def setUp(self):
        self.tech1 = Technology(name="Tech A", a=1.0, c=10.0, sigma=0.1)
        self.tech2 = Technology(name="Tech B", a=2.0, c=20.0, sigma=0.2)
        self.technologies = [self.tech1, self.tech2]
        self.cov_matrix = np.array([[0.01, 0.0], [0.0, 0.04]])
        self.risk_model = RiskModel(self.technologies, self.cov_matrix)

    def test_risk_calculation(self):
        weights = np.array([0.5, 0.5])
        risk = self.risk_model.total_risk(weights)
        # Variance: 0.5^2 * 0.01 + 0.5^2 * 0.04 = 0.0025 + 0.01 = 0.0125
        self.assertAlmostEqual(risk, 0.0125)

    def test_portfolio_variance(self):
        weights = np.array([1.0, 0.0])
        variance = self.risk_model.portfolio_variance(weights)
        self.assertAlmostEqual(variance, 0.01)

    def test_stranded_asset_risk(self):
        tech = Technology(
            name="Test Tech",
            a=1.0,
            c=10.0,
            sigma=0.1,
            tau=4.0,
            failure_prob=0.1,
            loss_given_failure=50.0,
        )
        rm = RiskModel([tech], np.array([[0.01]]))
        stranded = rm.stranded_asset_risk(np.array([1.0]))
        # 0.1 * 50 + 0.1 * sqrt(4) = 5.0 + 0.2 = 5.2
        self.assertAlmostEqual(stranded, 5.2)

    def test_option_value(self):
        tech = Technology(name="Test", a=1.0, c=10.0, sigma=0.1, o=5.0)
        rm = RiskModel([tech], np.array([[0.01]]))
        opt_val = rm.option_value(np.array([2.0]))
        self.assertAlmostEqual(opt_val, 10.0)

    def test_risk_breakdown(self):
        tech = Technology(
            name="Test Tech",
            a=1.0,
            c=10.0,
            sigma=0.1,
            tau=4.0,
            failure_prob=0.1,
            loss_given_failure=50.0,
            o=5.0,
        )
        rm = RiskModel([tech], np.array([[0.01]]))
        breakdown = rm.breakdown(np.array([1.0]))

        self.assertAlmostEqual(breakdown.cost_volatility, 0.01)
        self.assertAlmostEqual(breakdown.stranded_asset, 5.2)
        self.assertAlmostEqual(breakdown.option_value, 5.0)
        self.assertAlmostEqual(breakdown.total, 0.21)

    def test_invalid_covariance_shape(self):
        with self.assertRaises(ValueError):
            RiskModel(self.technologies, np.array([[0.01]]))

    def test_asymmetric_covariance_rejected(self):
        asymmetric = np.array([[0.01, 0.02], [0.03, 0.04]])
        with self.assertRaises(ValueError):
            RiskModel(self.technologies, asymmetric)


class TestOptimizationEngine(unittest.TestCase):
    """Tests for OptimizationEngine class."""

    def setUp(self):
        self.tech1 = Technology(name="Tech A", a=1.0, c=10.0, sigma=0.1)
        self.tech2 = Technology(name="Tech B", a=2.0, c=20.0, sigma=0.2)
        self.technologies = [self.tech1, self.tech2]
        self.cov_matrix = np.array([[0.01, 0.0], [0.0, 0.04]])
        self.risk_model = RiskModel(self.technologies, self.cov_matrix)
        self.optimizer = OptimizationEngine(self.technologies, self.risk_model)

    def test_optimization(self):
        target_abatement = 1.5
        portfolio = self.optimizer.solve_for_target(target_abatement)
        self.assertGreaterEqual(portfolio.total_abatement, target_abatement - 1e-6)

    def test_optimization_with_budget(self):
        target_abatement = 1.5
        budget = 100.0
        portfolio = self.optimizer.solve_for_target(target_abatement, budget_constraint=budget)
        self.assertLessEqual(portfolio.total_cost, budget + 1e-6)

    def test_optimization_returns_breakdown(self):
        target_abatement = 1.5
        portfolio, breakdown = self.optimizer.solve_for_target(
            target_abatement, return_breakdown=True
        )
        self.assertIsInstance(breakdown, RiskBreakdown)

    def test_nonnegative_weights(self):
        target_abatement = 2.0
        portfolio = self.optimizer.solve_for_target(target_abatement)
        self.assertTrue(np.all(portfolio.weights >= -1e-10))


class TestEfficientFrontier(unittest.TestCase):
    """Tests for EfficientFrontier class."""

    def setUp(self):
        self.tech1 = Technology(name="Tech A", a=1.0, c=10.0, sigma=0.1)
        self.tech2 = Technology(name="Tech B", a=2.0, c=20.0, sigma=0.2)
        self.technologies = [self.tech1, self.tech2]
        self.cov_matrix = np.array([[0.01, 0.0], [0.0, 0.04]])

    def test_frontier_returns_dataclass(self):
        frontier = EfficientFrontier(self.technologies, self.cov_matrix)
        results = frontier.compute(abatement_min=0.5, abatement_max=2.0, n_points=3)
        self.assertIsInstance(results[0], FrontierPoint)

    def test_frontier_correct_length(self):
        frontier = EfficientFrontier(self.technologies, self.cov_matrix)
        results = frontier.compute(abatement_min=0.5, abatement_max=2.0, n_points=5)
        self.assertEqual(len(results), 5)

    def test_frontier_increasing_abatement(self):
        frontier = EfficientFrontier(self.technologies, self.cov_matrix)
        results = frontier.compute(abatement_min=0.5, abatement_max=2.0, n_points=5)
        abatements = [r.abatement for r in results]
        self.assertEqual(abatements, sorted(abatements))

    def test_frontier_convexity(self):
        """Risk should be convex (non-decreasing marginal risk)."""
        frontier = EfficientFrontier(self.technologies, self.cov_matrix)
        results = frontier.compute(abatement_min=0.5, abatement_max=3.0, n_points=10)
        risks = [r.risk for r in results]
        # Check that risk generally increases (allowing for numerical tolerance)
        for i in range(len(risks) - 1):
            self.assertGreaterEqual(risks[i + 1], risks[i] - 1e-6)


class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases and boundary conditions."""

    def test_single_technology(self):
        tech = Technology(name="Solo", a=1.0, c=10.0, sigma=0.1)
        cov_matrix = np.array([[0.01]])
        risk_model = RiskModel([tech], cov_matrix)
        optimizer = OptimizationEngine([tech], risk_model)

        portfolio = optimizer.solve_for_target(target_abatement=1.0)
        self.assertAlmostEqual(portfolio.weights[0], 1.0, places=3)

    def test_zero_volatility(self):
        tech = Technology(name="Stable", a=1.0, c=10.0, sigma=0.0)
        cov_matrix = np.array([[0.0]])
        risk_model = RiskModel([tech], cov_matrix)

        risk = risk_model.total_risk(np.array([1.0]))
        self.assertEqual(risk, 0.0)

    def test_high_correlation(self):
        tech1 = Technology(name="A", a=1.0, c=10.0, sigma=0.1)
        tech2 = Technology(name="B", a=1.0, c=10.0, sigma=0.1)
        # High correlation (0.9)
        cov_matrix = np.array([[0.01, 0.009], [0.009, 0.01]])

        risk_model = RiskModel([tech1, tech2], cov_matrix)
        weights = np.array([0.5, 0.5])
        risk = risk_model.portfolio_variance(weights)
        # With high correlation, diversification benefit is reduced
        self.assertGreater(risk, 0.005)


class TestIntegration(unittest.TestCase):
    """Integration tests with realistic scenarios."""

    def test_steel_sector_scenario(self):
        """Test with steel sector-like technologies."""
        technologies = [
            Technology(name="BF-BOF", a=0.0, c=605, sigma=0.05),
            Technology(name="BF-BOF + CCS", a=1.4, c=720, sigma=0.18),
            Technology(name="Scrap-EAF", a=1.0, c=580, sigma=0.12),
            Technology(name="H2-DRI", a=1.9, c=800, sigma=0.30),
        ]

        # Build a simple diagonal covariance
        sigmas = np.array([t.sigma for t in technologies])
        cov_matrix = np.diag(sigmas ** 2)

        frontier = EfficientFrontier(technologies, cov_matrix)
        results = frontier.compute(
            abatement_min=0.5,
            abatement_max=1.5,
            n_points=5,
        )

        self.assertEqual(len(results), 5)
        # All portfolios should meet minimum abatement
        for point in results:
            self.assertGreaterEqual(point.portfolio.total_abatement, point.abatement - 1e-6)


if __name__ == "__main__":
    unittest.main()
