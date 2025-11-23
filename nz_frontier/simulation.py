import numpy as np
from typing import List, Dict
from .types import Technology

class CostSimulator:
    def __init__(self, technologies: List[Technology]):
        self.technologies = technologies

    def simulate_paths(self, 
                       t_horizon: float, 
                       dt: float, 
                       n_paths: int) -> Dict[str, np.ndarray]:
        """
        Simulates cost paths using Jump-Diffusion process.
        
        Returns:
            Dict mapping technology name to array of shape (n_paths, n_steps)
        """
        n_steps = int(t_horizon / dt)
        results = {}
        
        for tech in self.technologies:
            # Initialize paths
            paths = np.zeros((n_paths, n_steps + 1))
            paths[:, 0] = tech.c
            
            # Parameters
            mu = 0.0 # Drift (could be learning rate dependent)
            sigma = tech.sigma
            
            # Jump-Diffusion Process with Learning Curve
            # mu_j(t) = -alpha_j * log(Q_j(t)/Q_0)  <-- Learning effect (drift becomes negative as capacity grows)
            # For simulation, we assume a simplified exogenous deployment path or just constant drift reduction
            # Let's assume 'mu' is the base drift, and learning_rate reduces cost further.
            # Effective drift = mu - learning_rate
            
            effective_drift = -tech.learning_rate # Assuming mu=0 base drift for simplicity
            sigma = tech.sigma
            
            for t in range(n_steps):
                # Brownian Motion
                dW = np.random.normal(0, np.sqrt(dt), n_paths)
                
                # Jump Process (Poisson)
                # Prob of jump in dt is lambda * dt
                # dN is 1 if jump occurs, 0 otherwise
                jump_prob = tech.jump_intensity * dt
                dN = np.random.binomial(1, jump_prob, n_paths)
                
                # Jump size impact: if jump happens, cost changes by factor (1 + jump_size)
                # e.g. jump_size = -0.5 means 50% drop
                jump_factor = (1 + tech.jump_size) ** dN
                
                drift_term = (effective_drift - 0.5 * sigma**2) * dt
                diffusion_term = sigma * dW
                
                paths[:, t+1] = paths[:, t] * np.exp(drift_term + diffusion_term) * jump_factor
                
            results[tech.name] = paths
            
        return results
