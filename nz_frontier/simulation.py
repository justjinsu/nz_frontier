import numpy as np
from typing import List, Dict, Optional
from .types import Technology

class CostSimulator:
    def __init__(self, technologies: List[Technology], covariance_matrix: Optional[np.ndarray] = None):
        """
        Args:
            covariance_matrix: Optional covariance for correlated Brownian shocks across technologies.
        """
        self.technologies = technologies
        self.covariance_matrix = covariance_matrix

    def simulate_paths(self, 
                       t_horizon: float, 
                       dt: float, 
                       n_paths: int) -> Dict[str, np.ndarray]:
        """
        Simulates cost paths using the jump-diffusion with learning described in Section 6.
        
        Returns:
            Dict mapping technology name to array of shape (n_paths, n_steps + 1)
        """
        n_steps = int(t_horizon / dt)
        n_tech = len(self.technologies)

        results: Dict[str, np.ndarray] = {tech.name: np.zeros((n_paths, n_steps + 1)) for tech in self.technologies}
        for tech in self.technologies:
            results[tech.name][:, 0] = tech.c

        # Precompute covariance factor for correlated diffusion shocks
        use_corr = self.covariance_matrix is not None
        if use_corr:
            if self.covariance_matrix.shape != (n_tech, n_tech):
                raise ValueError("Covariance matrix for simulation must have shape (n_tech, n_tech)")

        for t in range(n_steps):
            if use_corr:
                # Mean zero shocks with provided covariance
                dW_matrix = np.random.multivariate_normal(
                    mean=np.zeros(n_tech), cov=self.covariance_matrix, size=n_paths
                ) * np.sqrt(dt)
            else:
                dW_matrix = np.random.normal(0, np.sqrt(dt), size=(n_paths, n_tech))

            for j, tech in enumerate(self.technologies):
                paths = results[tech.name]

                # Jump Process (Poisson)
                jump_prob = tech.jump_intensity * dt
                dN = np.random.binomial(1, jump_prob, n_paths)
                jump_factor = (1 + tech.jump_size) ** dN

                effective_drift = -tech.learning_rate  # Learning lowers cost trajectory
                sigma = tech.sigma
                drift_term = (effective_drift - 0.5 * sigma**2) * dt
                diffusion_term = sigma * dW_matrix[:, j]

                paths[:, t + 1] = paths[:, t] * np.exp(drift_term + diffusion_term) * jump_factor

        return results
