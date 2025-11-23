import numpy as np
import matplotlib.pyplot as plt
from nz_frontier import Technology, EfficientFrontier, DynamicOptimizer, CostSimulator
from nz_frontier.analysis import plot_frontier, plot_transition_paths

def run_demo():
    print("=== Running Corporate Decarbonization Model Demo ===")
    
    # 1. Define Technologies
    print("\n1. Defining Technologies...")
    solar = Technology(name="Solar", a=1.0, c=10.0, sigma=0.1)
    wind = Technology(name="Wind", a=1.2, c=12.0, sigma=0.15)
    ccs = Technology(name="CCS", a=5.0, c=50.0, sigma=0.3, 
                     jump_intensity=0.1, jump_size=-0.3, # Breakthrough potential
                     strike_price=40.0, tau=5.0) # Option to deploy later
    
    technologies = [solar, wind, ccs]
    cov_matrix = np.diag([t.sigma**2 for t in technologies])
    
    # 2. Static Efficient Frontier
    print("\n2. Computing Static Efficient Frontier...")
    frontier = EfficientFrontier(technologies, cov_matrix)
    results = frontier.compute(abatement_min=10, abatement_max=50, n_points=10, budget_constraint=1000)
    
    print(f"   Computed {len(results)} frontier points.")
    print(f"   Min Risk: {results[0].risk:.4f} at Abatement {results[0].abatement:.1f}")
    print(f"   Max Risk: {results[-1].risk:.4f} at Abatement {results[-1].abatement:.1f}")
    
    # 3. Dynamic Optimization
    print("\n3. Running Dynamic Optimization (10 periods)...")
    dyn_opt = DynamicOptimizer(technologies, cov_matrix)
    targets = np.linspace(10, 50, 10)
    portfolios = dyn_opt.solve_bellman(10, targets, budget_per_period=200)
    
    print("   Dynamic path computed.")
    print(f"   Final Portfolio Weights: {portfolios[-1].weights}")
    
    # 4. Cost Simulation
    print("\n4. Simulating Stochastic Costs (Jump-Diffusion)...")
    sim = CostSimulator(technologies)
    paths = sim.simulate_paths(t_horizon=10, dt=0.1, n_paths=100)
    
    print("   Simulation complete.")
    print(f"   CCS Final Mean Cost: {np.mean(paths['CCS'][:, -1]):.2f} (Initial: {ccs.c})")
    
    print("\n=== Demo Complete ===")

if __name__ == "__main__":
    run_demo()
