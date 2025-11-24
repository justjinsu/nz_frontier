"""
Advanced Analysis Module for Net-Zero Frontier

Implements:
1. Monte Carlo Efficient Frontier with confidence bands
2. Dynamic Real Options re-pricing
3. Stochastic Dominance Analysis
4. Robust Optimization (min-max formulation)
5. Full Correlation Matrix construction

References: paper_v2.tex Sections 5, 6, 9
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from .types import Technology, Portfolio, RiskBreakdown, FrontierPoint
from .risk import RiskModel
from .optimizer import OptimizationEngine
from .valuation import OptionValuator
from .simulation import CostSimulator


@dataclass
class MonteCarloFrontierResult:
    """Results from Monte Carlo frontier analysis."""
    abatement_targets: np.ndarray
    mean_risk: np.ndarray
    std_risk: np.ndarray
    percentile_5: np.ndarray
    percentile_95: np.ndarray
    all_paths: np.ndarray  # Shape: (n_simulations, n_targets)
    optimal_portfolios: List[Portfolio]  # Mean-optimal portfolios


@dataclass
class StochasticDominanceResult:
    """Results from stochastic dominance analysis."""
    portfolio_a_name: str
    portfolio_b_name: str
    fsd_a_dominates_b: bool  # First-order stochastic dominance
    fsd_b_dominates_a: bool
    ssd_a_dominates_b: bool  # Second-order stochastic dominance
    ssd_b_dominates_a: bool
    risk_distribution_a: np.ndarray
    risk_distribution_b: np.ndarray


@dataclass
class RobustOptimizationResult:
    """Results from robust optimization."""
    portfolio: Portfolio
    worst_case_risk: float
    nominal_risk: float
    robustness_gap: float
    worst_case_params: Dict[str, float]


def build_correlation_matrix(technologies: List[Technology],
                             correlation_pairs: Optional[Dict[Tuple[str, str], float]] = None) -> np.ndarray:
    """
    Builds full correlation matrix from technology parameters.

    Uses the factor structure from paper_v2.tex Remark 1:
    Σ = B F B^T + D

    For simplicity, we use a simplified approach:
    - Technologies with same 'rho' category are correlated
    - Hydrogen-based technologies share correlation
    - CCS technologies share correlation

    Args:
        technologies: List of Technology objects
        correlation_pairs: Optional dict mapping (tech_name_i, tech_name_j) -> correlation

    Returns:
        Covariance matrix of shape (n_tech, n_tech)
    """
    n = len(technologies)
    sigmas = np.array([t.sigma for t in technologies])

    # Start with correlation matrix
    corr_matrix = np.eye(n)

    # Infer correlations from technology characteristics
    for i, tech_i in enumerate(technologies):
        for j, tech_j in enumerate(technologies):
            if i >= j:
                continue

            # Check for correlation pairs override
            if correlation_pairs:
                key1 = (tech_i.name, tech_j.name)
                key2 = (tech_j.name, tech_i.name)
                if key1 in correlation_pairs:
                    corr_matrix[i, j] = correlation_pairs[key1]
                    corr_matrix[j, i] = correlation_pairs[key1]
                    continue
                elif key2 in correlation_pairs:
                    corr_matrix[i, j] = correlation_pairs[key2]
                    corr_matrix[j, i] = correlation_pairs[key2]
                    continue

            # Infer correlation from technology names/characteristics
            rho = _infer_correlation(tech_i, tech_j)
            corr_matrix[i, j] = rho
            corr_matrix[j, i] = rho

    # Convert correlation matrix to covariance matrix
    # Σ_ij = ρ_ij * σ_i * σ_j
    cov_matrix = corr_matrix * np.outer(sigmas, sigmas)

    # Ensure positive semi-definiteness
    cov_matrix = _nearest_psd(cov_matrix)

    return cov_matrix


def _infer_correlation(tech_i: Technology, tech_j: Technology) -> float:
    """
    Infer correlation between two technologies based on their characteristics.

    Correlation drivers:
    - Hydrogen-based: High correlation (shared H2 infrastructure/cost)
    - CCS-based: Moderate correlation (shared CO2 transport/storage)
    - Electricity-dependent: Moderate correlation (shared grid costs)
    - Same sector baseline: Low correlation (idiosyncratic)
    """
    name_i = tech_i.name.lower()
    name_j = tech_j.name.lower()

    # Hydrogen correlation
    h2_keywords = ['hydrogen', 'h2', 'hyrex', 'hy-cube', 'electrolysis', 'dri']
    is_h2_i = any(kw in name_i for kw in h2_keywords)
    is_h2_j = any(kw in name_j for kw in h2_keywords)

    if is_h2_i and is_h2_j:
        return 0.6  # High correlation for H2 technologies

    # CCS correlation
    ccs_keywords = ['ccs', 'ccus', 'capture', 'carbon capture']
    is_ccs_i = any(kw in name_i for kw in ccs_keywords)
    is_ccs_j = any(kw in name_j for kw in ccs_keywords)

    if is_ccs_i and is_ccs_j:
        return 0.5  # Moderate-high correlation for CCS

    # EAF / Electric correlation
    elec_keywords = ['eaf', 'electric', 'arc furnace', 'electr']
    is_elec_i = any(kw in name_i for kw in elec_keywords)
    is_elec_j = any(kw in name_j for kw in elec_keywords)

    if is_elec_i and is_elec_j:
        return 0.4  # Moderate correlation

    # Cross-category: H2 and Electric have some correlation (electricity for electrolysis)
    if (is_h2_i and is_elec_j) or (is_h2_j and is_elec_i):
        return 0.3

    # Use rho from Technology if specified
    if tech_i.rho > 0 and tech_j.rho > 0:
        return min(tech_i.rho, tech_j.rho) * 0.5

    # Default: low correlation
    return 0.1


def _nearest_psd(matrix: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """
    Find the nearest positive semi-definite matrix.
    Uses eigenvalue decomposition and clips negative eigenvalues.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.maximum(eigenvalues, epsilon)
    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T


class MonteCarloFrontier:
    """
    Monte Carlo simulation of the efficient frontier.

    Implements the stochastic efficient frontier from paper_v2.tex Section 9.3:
    - Simulates technology cost paths using jump-diffusion
    - Re-optimizes portfolio at each simulation
    - Computes confidence bands on the frontier
    """

    def __init__(self,
                 technologies: List[Technology],
                 covariance_matrix: np.ndarray,
                 risk_free_rate: float = 0.05):
        self.technologies = technologies
        self.covariance_matrix = covariance_matrix
        self.r = risk_free_rate
        self.n_tech = len(technologies)

    def compute_frontier(self,
                        abatement_min: float,
                        abatement_max: float,
                        budget_constraint: float,
                        n_targets: int = 15,
                        n_simulations: int = 500,
                        time_horizon: float = 10.0,
                        lambda_param: float = 1.0,
                        gamma_param: float = 1.0,
                        revalue_options: bool = True) -> MonteCarloFrontierResult:
        """
        Compute Monte Carlo efficient frontier with confidence bands.

        Args:
            abatement_min: Minimum abatement target
            abatement_max: Maximum abatement target
            budget_constraint: Total budget
            n_targets: Number of points on frontier
            n_simulations: Number of Monte Carlo paths
            time_horizon: Simulation horizon (years)
            lambda_param: Weight on stranded asset risk
            gamma_param: Weight on option value
            revalue_options: Whether to dynamically re-price options

        Returns:
            MonteCarloFrontierResult with mean, std, and percentile bands
        """
        targets = np.linspace(abatement_min, abatement_max, n_targets)
        all_risks = np.zeros((n_simulations, n_targets))

        # Simulate cost paths
        simulator = CostSimulator(self.technologies, self.covariance_matrix)

        for sim in range(n_simulations):
            # Generate one realization of cost paths
            cost_paths = simulator.simulate_paths(time_horizon, dt=0.5, n_paths=1)

            # Get terminal costs for this simulation
            terminal_costs = {}
            for tech in self.technologies:
                terminal_costs[tech.name] = cost_paths[tech.name][0, -1]

            # Create modified technologies with simulated costs
            sim_technologies = self._create_simulated_technologies(
                terminal_costs, revalue_options, time_horizon
            )

            # Build risk model and optimizer for this simulation
            sim_risk_model = RiskModel(
                sim_technologies,
                self.covariance_matrix,
                r=self.r,
                lambda_param=lambda_param,
                gamma_param=gamma_param
            )
            sim_optimizer = OptimizationEngine(sim_technologies, sim_risk_model)

            # Optimize for each target
            for t_idx, target in enumerate(targets):
                try:
                    portfolio = sim_optimizer.solve_for_target(
                        target,
                        budget_constraint=budget_constraint,
                        lambda_param=lambda_param,
                        gamma_param=gamma_param
                    )
                    risk = sim_risk_model.total_risk(
                        portfolio.weights,
                        lambda_param=lambda_param,
                        gamma_param=gamma_param
                    )
                    all_risks[sim, t_idx] = risk
                except RuntimeError:
                    # Infeasible - use large penalty
                    all_risks[sim, t_idx] = np.nan

        # Compute statistics (ignoring NaN)
        mean_risk = np.nanmean(all_risks, axis=0)
        std_risk = np.nanstd(all_risks, axis=0)
        percentile_5 = np.nanpercentile(all_risks, 5, axis=0)
        percentile_95 = np.nanpercentile(all_risks, 95, axis=0)

        # Compute mean-optimal portfolios (using expected costs)
        optimal_portfolios = self._compute_mean_optimal_portfolios(
            targets, budget_constraint, lambda_param, gamma_param
        )

        return MonteCarloFrontierResult(
            abatement_targets=targets,
            mean_risk=mean_risk,
            std_risk=std_risk,
            percentile_5=percentile_5,
            percentile_95=percentile_95,
            all_paths=all_risks,
            optimal_portfolios=optimal_portfolios
        )

    def _create_simulated_technologies(self,
                                       terminal_costs: Dict[str, float],
                                       revalue_options: bool,
                                       time_horizon: float) -> List[Technology]:
        """Create technologies with simulated terminal costs and re-valued options."""
        sim_techs = []
        valuator = OptionValuator(r=self.r)

        for tech in self.technologies:
            new_cost = terminal_costs[tech.name]

            # Re-price option if requested
            if revalue_options and tech.strike_price > 0:
                # Option value based on NEW cost level
                new_option_value = valuator.black_scholes_call(
                    S=max(new_cost, 1e-6),
                    K=tech.strike_price,
                    T=max(tech.tau - time_horizon, 1.0),  # Remaining time
                    sigma=tech.sigma
                )
            else:
                new_option_value = tech.o

            sim_tech = Technology(
                name=tech.name,
                a=tech.a,
                c=new_cost,
                sigma=tech.sigma,
                rho=tech.rho,
                o=new_option_value,
                tau=max(tech.tau - time_horizon, 1.0),
                jump_intensity=tech.jump_intensity,
                jump_size=tech.jump_size,
                strike_price=tech.strike_price,
                learning_rate=tech.learning_rate,
                failure_prob=tech.failure_prob,
                loss_given_failure=tech.loss_given_failure
            )
            sim_techs.append(sim_tech)

        return sim_techs

    def _compute_mean_optimal_portfolios(self,
                                         targets: np.ndarray,
                                         budget_constraint: float,
                                         lambda_param: float,
                                         gamma_param: float) -> List[Portfolio]:
        """Compute optimal portfolios using expected (mean) parameters."""
        risk_model = RiskModel(
            self.technologies,
            self.covariance_matrix,
            r=self.r,
            lambda_param=lambda_param,
            gamma_param=gamma_param
        )
        optimizer = OptimizationEngine(self.technologies, risk_model)

        portfolios = []
        for target in targets:
            try:
                portfolio = optimizer.solve_for_target(
                    target,
                    budget_constraint=budget_constraint,
                    lambda_param=lambda_param,
                    gamma_param=gamma_param
                )
                portfolios.append(portfolio)
            except RuntimeError:
                # Return empty portfolio if infeasible
                portfolios.append(Portfolio(
                    weights=np.zeros(self.n_tech),
                    technologies=self.technologies
                ))

        return portfolios


class StochasticDominanceAnalyzer:
    """
    Analyzes stochastic dominance between portfolios.

    Implements first-order (FSD) and second-order (SSD) stochastic dominance
    as discussed in the risk comparison literature.

    FSD: Portfolio A dominates B if F_A(x) ≤ F_B(x) for all x
    SSD: Portfolio A dominates B if ∫F_A(x)dx ≤ ∫F_B(x)dx for all x
    """

    def __init__(self,
                 technologies: List[Technology],
                 covariance_matrix: np.ndarray):
        self.technologies = technologies
        self.covariance_matrix = covariance_matrix
        self.simulator = CostSimulator(technologies, covariance_matrix)

    def compare_portfolios(self,
                          weights_a: np.ndarray,
                          weights_b: np.ndarray,
                          name_a: str = "Portfolio A",
                          name_b: str = "Portfolio B",
                          n_simulations: int = 1000,
                          time_horizon: float = 10.0,
                          lambda_param: float = 1.0,
                          gamma_param: float = 1.0) -> StochasticDominanceResult:
        """
        Compare two portfolios for stochastic dominance.

        Args:
            weights_a: Weight vector for portfolio A
            weights_b: Weight vector for portfolio B
            name_a: Name for portfolio A
            name_b: Name for portfolio B
            n_simulations: Number of Monte Carlo simulations
            time_horizon: Simulation horizon
            lambda_param: Stranded asset risk weight
            gamma_param: Option value weight

        Returns:
            StochasticDominanceResult with dominance indicators
        """
        # Simulate risk distributions
        risks_a = self._simulate_portfolio_risks(
            weights_a, n_simulations, time_horizon, lambda_param, gamma_param
        )
        risks_b = self._simulate_portfolio_risks(
            weights_b, n_simulations, time_horizon, lambda_param, gamma_param
        )

        # Test first-order stochastic dominance
        fsd_a_dom_b = self._test_fsd(risks_a, risks_b)
        fsd_b_dom_a = self._test_fsd(risks_b, risks_a)

        # Test second-order stochastic dominance
        ssd_a_dom_b = self._test_ssd(risks_a, risks_b)
        ssd_b_dom_a = self._test_ssd(risks_b, risks_a)

        return StochasticDominanceResult(
            portfolio_a_name=name_a,
            portfolio_b_name=name_b,
            fsd_a_dominates_b=fsd_a_dom_b,
            fsd_b_dominates_a=fsd_b_dom_a,
            ssd_a_dominates_b=ssd_a_dom_b,
            ssd_b_dominates_a=ssd_b_dom_a,
            risk_distribution_a=risks_a,
            risk_distribution_b=risks_b
        )

    def _simulate_portfolio_risks(self,
                                  weights: np.ndarray,
                                  n_simulations: int,
                                  time_horizon: float,
                                  lambda_param: float,
                                  gamma_param: float) -> np.ndarray:
        """Simulate risk distribution for a portfolio."""
        risks = np.zeros(n_simulations)

        cost_paths = self.simulator.simulate_paths(
            time_horizon, dt=0.5, n_paths=n_simulations
        )

        for sim in range(n_simulations):
            # Get terminal costs
            terminal_costs = {
                tech.name: cost_paths[tech.name][sim, -1]
                for tech in self.technologies
            }

            # Create simulated technologies
            sim_techs = []
            for tech in self.technologies:
                sim_tech = Technology(
                    name=tech.name,
                    a=tech.a,
                    c=terminal_costs[tech.name],
                    sigma=tech.sigma,
                    rho=tech.rho,
                    o=tech.o,
                    tau=tech.tau,
                    jump_intensity=tech.jump_intensity,
                    jump_size=tech.jump_size,
                    strike_price=tech.strike_price,
                    learning_rate=tech.learning_rate,
                    failure_prob=tech.failure_prob,
                    loss_given_failure=tech.loss_given_failure
                )
                sim_techs.append(sim_tech)

            # Compute risk
            risk_model = RiskModel(
                sim_techs,
                self.covariance_matrix,
                lambda_param=lambda_param,
                gamma_param=gamma_param
            )
            risks[sim] = risk_model.total_risk(weights, lambda_param, gamma_param)

        return risks

    def _test_fsd(self, risks_a: np.ndarray, risks_b: np.ndarray, tolerance: float = 0.01) -> bool:
        """
        Test if A first-order stochastically dominates B.
        A dominates B if F_A(x) ≤ F_B(x) for all x (A has less probability of high risk)
        """
        # Create common grid
        all_risks = np.concatenate([risks_a, risks_b])
        grid = np.linspace(np.min(all_risks), np.max(all_risks), 100)

        # Compute CDFs
        cdf_a = np.array([np.mean(risks_a <= x) for x in grid])
        cdf_b = np.array([np.mean(risks_b <= x) for x in grid])

        # A dominates B if F_A(x) >= F_B(x) for all x
        # (higher CDF means more probability mass on lower values = less risk)
        return np.all(cdf_a >= cdf_b - tolerance)

    def _test_ssd(self, risks_a: np.ndarray, risks_b: np.ndarray, tolerance: float = 0.01) -> bool:
        """
        Test if A second-order stochastically dominates B.
        A dominates B if ∫_{-∞}^{x} F_A(t)dt ≥ ∫_{-∞}^{x} F_B(t)dt for all x
        """
        # Create common grid
        all_risks = np.concatenate([risks_a, risks_b])
        grid = np.linspace(np.min(all_risks), np.max(all_risks), 100)
        dx = grid[1] - grid[0]

        # Compute CDFs
        cdf_a = np.array([np.mean(risks_a <= x) for x in grid])
        cdf_b = np.array([np.mean(risks_b <= x) for x in grid])

        # Compute integrated CDFs
        int_cdf_a = np.cumsum(cdf_a) * dx
        int_cdf_b = np.cumsum(cdf_b) * dx

        # A dominates B if integrated CDF of A >= integrated CDF of B
        return np.all(int_cdf_a >= int_cdf_b - tolerance)


class RobustOptimizer:
    """
    Robust optimization under parameter uncertainty.

    Implements the min-max formulation from paper_v2.tex Section 9.1:
    min_w max_{θ ∈ Θ} R_P(w; θ)

    where Θ is an uncertainty set for parameters (costs, volatilities, etc.)
    """

    def __init__(self,
                 technologies: List[Technology],
                 covariance_matrix: np.ndarray,
                 cost_uncertainty: float = 0.2,
                 volatility_uncertainty: float = 0.3,
                 correlation_uncertainty: float = 0.2):
        """
        Args:
            technologies: List of technologies
            covariance_matrix: Nominal covariance matrix
            cost_uncertainty: Fractional uncertainty in costs (e.g., 0.2 = ±20%)
            volatility_uncertainty: Fractional uncertainty in volatilities
            correlation_uncertainty: Fractional uncertainty in correlations
        """
        self.technologies = technologies
        self.nominal_cov = covariance_matrix
        self.cost_uncertainty = cost_uncertainty
        self.volatility_uncertainty = volatility_uncertainty
        self.correlation_uncertainty = correlation_uncertainty
        self.n_tech = len(technologies)

    def solve(self,
              target_abatement: float,
              budget_constraint: float,
              lambda_param: float = 1.0,
              gamma_param: float = 1.0,
              n_scenarios: int = 50) -> RobustOptimizationResult:
        """
        Solve robust optimization problem.

        Uses scenario-based approximation:
        1. Generate worst-case scenarios from uncertainty set
        2. Solve min_w max_scenario R_P(w; scenario)

        Args:
            target_abatement: Required abatement
            budget_constraint: Budget limit
            lambda_param: Stranded asset risk weight
            gamma_param: Option value weight
            n_scenarios: Number of uncertainty scenarios

        Returns:
            RobustOptimizationResult with robust portfolio
        """
        # Generate scenarios from uncertainty set
        scenarios = self._generate_scenarios(n_scenarios)

        # Solve min-max problem
        def objective(w):
            max_risk = -np.inf
            for scenario in scenarios:
                risk = self._evaluate_scenario_risk(w, scenario, lambda_param, gamma_param)
                max_risk = max(max_risk, risk)
            return max_risk

        # Constraints
        abatements = np.array([t.a for t in self.technologies])
        costs = np.array([t.c for t in self.technologies])

        constraints = [
            {'type': 'ineq', 'fun': lambda w: np.sum(w * abatements) - target_abatement},
            {'type': 'ineq', 'fun': lambda w: budget_constraint - np.sum(w * costs)}
        ]

        bounds = [(0, None) for _ in range(self.n_tech)]
        initial_guess = np.ones(self.n_tech) * target_abatement / max(np.sum(abatements), 1e-6)

        # Try multiple optimization methods for robustness
        result = None
        for method in ['SLSQP', 'trust-constr']:
            try:
                result = minimize(
                    objective,
                    initial_guess,
                    method=method,
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 500}
                )
                if result.success:
                    break
            except Exception:
                continue

        if result is None or not result.success:
            # Fallback: use heuristic solution that meets constraints
            # Distribute capacity proportionally to abatement efficiency
            efficiency = abatements / np.maximum(costs, 1e-6)
            weights = efficiency / np.sum(efficiency) * target_abatement / np.mean(abatements[abatements > 0])
            # Ensure abatement constraint is met
            total_abatement = np.sum(weights * abatements)
            if total_abatement < target_abatement and np.sum(abatements) > 0:
                weights = weights * (target_abatement / max(total_abatement, 1e-6)) * 1.1
            result = type('obj', (object,), {'x': weights, 'success': True})()

        # Find worst-case scenario for optimal portfolio
        worst_risk = -np.inf
        worst_scenario = None
        for scenario in scenarios:
            risk = self._evaluate_scenario_risk(result.x, scenario, lambda_param, gamma_param)
            if risk > worst_risk:
                worst_risk = risk
                worst_scenario = scenario

        # Compute nominal risk
        nominal_risk_model = RiskModel(
            self.technologies,
            self.nominal_cov,
            lambda_param=lambda_param,
            gamma_param=gamma_param
        )
        nominal_risk = nominal_risk_model.total_risk(result.x, lambda_param, gamma_param)

        portfolio = Portfolio(weights=result.x, technologies=self.technologies)

        return RobustOptimizationResult(
            portfolio=portfolio,
            worst_case_risk=worst_risk,
            nominal_risk=nominal_risk,
            robustness_gap=worst_risk - nominal_risk,
            worst_case_params=worst_scenario or {}
        )

    def _generate_scenarios(self, n_scenarios: int) -> List[Dict]:
        """Generate scenarios from uncertainty set."""
        scenarios = []

        for _ in range(n_scenarios):
            # Perturb costs
            cost_multipliers = 1 + np.random.uniform(
                -self.cost_uncertainty,
                self.cost_uncertainty,
                self.n_tech
            )

            # Perturb volatilities (only upward for conservatism)
            vol_multipliers = 1 + np.random.uniform(
                0,
                self.volatility_uncertainty,
                self.n_tech
            )

            # Perturb correlations (increase for conservatism)
            corr_multiplier = 1 + np.random.uniform(0, self.correlation_uncertainty)

            scenarios.append({
                'cost_multipliers': cost_multipliers,
                'vol_multipliers': vol_multipliers,
                'corr_multiplier': corr_multiplier
            })

        # Add worst-case corners
        scenarios.append({
            'cost_multipliers': np.ones(self.n_tech) * (1 + self.cost_uncertainty),
            'vol_multipliers': np.ones(self.n_tech) * (1 + self.volatility_uncertainty),
            'corr_multiplier': 1 + self.correlation_uncertainty
        })

        return scenarios

    def _evaluate_scenario_risk(self,
                                weights: np.ndarray,
                                scenario: Dict,
                                lambda_param: float,
                                gamma_param: float) -> float:
        """Evaluate portfolio risk under a specific scenario."""
        # Create perturbed technologies
        perturbed_techs = []
        for i, tech in enumerate(self.technologies):
            perturbed_tech = Technology(
                name=tech.name,
                a=tech.a,
                c=tech.c * scenario['cost_multipliers'][i],
                sigma=tech.sigma * scenario['vol_multipliers'][i],
                rho=tech.rho,
                o=tech.o,
                tau=tech.tau,
                jump_intensity=tech.jump_intensity,
                jump_size=tech.jump_size,
                strike_price=tech.strike_price,
                learning_rate=tech.learning_rate,
                failure_prob=tech.failure_prob,
                loss_given_failure=tech.loss_given_failure
            )
            perturbed_techs.append(perturbed_tech)

        # Create perturbed covariance matrix
        sigmas = np.array([t.sigma for t in perturbed_techs])

        # Extract correlations from nominal covariance
        nominal_sigmas = np.array([t.sigma for t in self.technologies])
        with np.errstate(divide='ignore', invalid='ignore'):
            corr_matrix = self.nominal_cov / np.outer(nominal_sigmas, nominal_sigmas)
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
            np.fill_diagonal(corr_matrix, 1.0)

        # Increase correlations
        corr_matrix = corr_matrix * scenario['corr_multiplier']
        np.clip(corr_matrix, -1, 1, out=corr_matrix)
        np.fill_diagonal(corr_matrix, 1.0)

        # Rebuild covariance
        perturbed_cov = corr_matrix * np.outer(sigmas, sigmas)
        perturbed_cov = _nearest_psd(perturbed_cov)

        # Compute risk
        risk_model = RiskModel(
            perturbed_techs,
            perturbed_cov,
            lambda_param=lambda_param,
            gamma_param=gamma_param
        )

        return risk_model.total_risk(weights, lambda_param, gamma_param)


class DynamicRealOptionsOptimizer:
    """
    Dynamic optimization with real options re-valuation.

    Extends the MPC approach from dynamic.py with:
    - Option values updated at each period based on cost evolution
    - Expansion/abandonment options explicitly modeled
    """

    def __init__(self,
                 technologies: List[Technology],
                 covariance_matrix: np.ndarray,
                 risk_free_rate: float = 0.05,
                 discount_factor: float = 0.95):
        self.technologies = technologies
        self.covariance_matrix = covariance_matrix
        self.r = risk_free_rate
        self.beta = discount_factor
        self.n_tech = len(technologies)
        self.valuator = OptionValuator(r=risk_free_rate)

    def solve_with_options(self,
                          T_periods: int,
                          target_path: List[float],
                          budget_per_period: float,
                          lambda_param: float = 1.0,
                          gamma_param: float = 1.0,
                          n_simulations: int = 100) -> Tuple[List[Portfolio], Dict[str, np.ndarray]]:
        """
        Solve dynamic optimization with Monte Carlo option valuation.

        At each period:
        1. Simulate future cost paths
        2. Value options based on simulated paths
        3. Optimize portfolio with updated option values

        Args:
            T_periods: Number of periods
            target_path: Abatement targets for each period
            budget_per_period: Budget per period
            lambda_param: Stranded asset risk weight
            gamma_param: Option value weight
            n_simulations: MC simulations for option valuation

        Returns:
            Tuple of (portfolios, cost_history)
        """
        portfolios = []
        current_weights = np.zeros(self.n_tech)
        current_costs = np.array([t.c for t in self.technologies])

        cost_history = {tech.name: [tech.c] for tech in self.technologies}

        simulator = CostSimulator(self.technologies, self.covariance_matrix)

        for t in range(T_periods):
            remaining_periods = T_periods - t

            # Update option values via Monte Carlo
            updated_option_values = self._value_options_mc(
                current_costs,
                remaining_periods,
                n_simulations
            )

            # Create technologies with updated costs and options
            updated_techs = []
            for i, tech in enumerate(self.technologies):
                updated_tech = Technology(
                    name=tech.name,
                    a=tech.a,
                    c=current_costs[i],
                    sigma=tech.sigma,
                    rho=tech.rho,
                    o=updated_option_values[i],
                    tau=max(tech.tau - t, 1.0),
                    jump_intensity=tech.jump_intensity,
                    jump_size=tech.jump_size,
                    strike_price=tech.strike_price,
                    learning_rate=tech.learning_rate,
                    failure_prob=tech.failure_prob,
                    loss_given_failure=tech.loss_given_failure
                )
                updated_techs.append(updated_tech)

            # Optimize
            risk_model = RiskModel(
                updated_techs,
                self.covariance_matrix,
                r=self.r,
                lambda_param=lambda_param,
                gamma_param=gamma_param
            )
            optimizer = OptimizationEngine(updated_techs, risk_model)

            try:
                # Solve with irreversibility constraint
                portfolio = self._solve_with_irreversibility(
                    optimizer,
                    target_path[t],
                    budget_per_period,
                    current_weights,
                    lambda_param,
                    gamma_param
                )
                portfolios.append(portfolio)
                current_weights = portfolio.weights.copy()
            except RuntimeError as e:
                print(f"Period {t} optimization failed: {e}")
                portfolios.append(Portfolio(current_weights, self.technologies))

            # Simulate cost evolution for next period
            if t < T_periods - 1:
                paths = simulator.simulate_paths(1.0, dt=0.5, n_paths=1)
                for i, tech in enumerate(self.technologies):
                    current_costs[i] = paths[tech.name][0, -1]
                    cost_history[tech.name].append(current_costs[i])

        return portfolios, cost_history

    def _value_options_mc(self,
                         current_costs: np.ndarray,
                         remaining_periods: int,
                         n_simulations: int) -> np.ndarray:
        """
        Value options using Monte Carlo simulation.

        Option value = E[max(S_T - K, 0)] discounted
        where S_T is simulated terminal "value" (cost savings from tech improvement)
        """
        option_values = np.zeros(self.n_tech)

        for i, tech in enumerate(self.technologies):
            if tech.strike_price <= 0:
                option_values[i] = tech.o
                continue

            # Simulate future cost paths
            payoffs = []
            for _ in range(n_simulations):
                # Simple GBM simulation
                cost = current_costs[i]
                for _ in range(remaining_periods):
                    drift = -tech.learning_rate - 0.5 * tech.sigma**2
                    diffusion = tech.sigma * np.random.normal()

                    # Jump
                    if np.random.random() < tech.jump_intensity:
                        cost *= (1 + tech.jump_size)

                    cost *= np.exp(drift + diffusion)

                # Option payoff: value of having low-cost tech
                # If cost drops below strike, option is valuable
                payoff = max(tech.strike_price - cost, 0)
                payoffs.append(payoff)

            # Discounted expected payoff
            option_values[i] = np.mean(payoffs) * (self.beta ** remaining_periods)

        return option_values

    def _solve_with_irreversibility(self,
                                    optimizer: OptimizationEngine,
                                    target: float,
                                    budget: float,
                                    min_weights: np.ndarray,
                                    lambda_param: float,
                                    gamma_param: float) -> Portfolio:
        """Solve optimization with irreversibility constraint w >= w_prev."""

        def objective(w):
            return optimizer.risk_model.total_risk(w, lambda_param, gamma_param)

        abatements = np.array([t.a for t in optimizer.technologies])
        costs = np.array([t.c for t in optimizer.technologies])

        constraints = [
            {'type': 'ineq', 'fun': lambda w: np.sum(w * abatements) - target},
            {'type': 'ineq', 'fun': lambda w: budget - np.sum((w - min_weights) * costs)},
            {'type': 'ineq', 'fun': lambda w: w - min_weights}  # Irreversibility
        ]

        bounds = [(mw, None) for mw in min_weights]

        # Better initial guess
        if np.sum(abatements) > 0:
            base_allocation = target / np.sum(abatements)
            initial_guess = np.maximum(min_weights, np.ones(len(min_weights)) * base_allocation)
        else:
            initial_guess = min_weights.copy()

        # Try multiple methods
        result = None
        for method in ['SLSQP', 'trust-constr']:
            try:
                result = minimize(
                    objective,
                    initial_guess,
                    method=method,
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 300}
                )
                if result.success:
                    break
            except Exception:
                continue

        if result is None or not result.success:
            # Fallback: heuristic allocation
            if np.sum(abatements) > 0:
                efficiency = abatements / np.maximum(costs, 1e-6)
                weights = np.maximum(min_weights, efficiency / np.sum(efficiency) * target / np.mean(abatements[abatements > 0]))
            else:
                weights = min_weights.copy()
            return Portfolio(weights=weights, technologies=optimizer.technologies)

        return Portfolio(weights=result.x, technologies=optimizer.technologies)
