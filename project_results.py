import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from nz_frontier import Technology, EfficientFrontier, DynamicOptimizer

def load_technologies(csv_path):
    df = pd.read_csv(csv_path)
    techs = []
    for _, row in df.iterrows():
        t = Technology(
            name=row['name'],
            a=row['a'],
            c=row['c'],
            sigma=row['sigma'],
            rho=row.get('rho', 0.0),
            o=row.get('o', 0.0),
            tau=row.get('tau', 0.0),
            jump_intensity=row.get('jump_intensity', 0.0),
            jump_size=row.get('jump_size', 0.0),
            strike_price=row.get('strike_price', 0.0),
            learning_rate=row.get('learning_rate', 0.0),
            failure_prob=row.get('failure_prob', 0.0),
            loss_given_failure=row.get('loss_given_failure', 0.0)
        )
        techs.append(t)
    return techs

def run_projection(case_name, csv_path):
    print(f"--- Running Projection for {case_name} ---")
    technologies = load_technologies(csv_path)
    
    # Covariance
    sigmas = np.array([t.sigma for t in technologies])
    cov_matrix = np.diag(sigmas**2)
    
    # 1. Efficient Frontier
    print("Computing Efficient Frontier...")
    frontier = EfficientFrontier(technologies, cov_matrix)
    # Adjust targets based on case
    if "steel" in case_name.lower():
        min_target, max_target = 10, 100
        budget = 5000
    else:
        min_target, max_target = 10, 100
        budget = 5000
        
    results = frontier.compute(min_target, max_target, n_points=20, budget_constraint=budget)
    
    risks = [p.risk for p in results]
    abatements = [p.abatement for p in results]
    
    plt.figure(figsize=(10, 6))
    plt.plot(abatements, risks, 'o-', linewidth=2, color='#2E86C1')
    plt.xlabel('Abatement Target (tCO2)')
    plt.ylabel('Portfolio Transition Risk')
    plt.title(f'Efficient Frontier: {case_name}')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'results/{case_name.lower()}_frontier.png')
    plt.close()
    
    # 2. Dynamic Transition
    print("Computing Dynamic Transition...")
    dyn_opt = DynamicOptimizer(technologies, cov_matrix)
    time_horizon = 20
    targets = np.linspace(min_target, max_target, time_horizon)
    budget_per_period = budget / time_horizon * 2
    
    portfolios = dyn_opt.solve_bellman(time_horizon, targets, budget_per_period)
    
    weights_history = np.array([p.weights for p in portfolios])
    
    plt.figure(figsize=(10, 6))
    for i, tech in enumerate(technologies):
        plt.plot(range(time_horizon), weights_history[:, i], label=tech.name, linewidth=2)
    
    plt.xlabel('Year')
    plt.ylabel('Installed Capacity')
    plt.title(f'Transition Pathway: {case_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'results/{case_name.lower()}_transition.png')
    plt.close()
    
    print(f"Saved plots to results/{case_name.lower()}_frontier.png and results/{case_name.lower()}_transition.png")

if __name__ == "__main__":
    if os.path.exists("data/steel.csv"):
        run_projection("Steel", "data/steel.csv")
    if os.path.exists("data/petrochemical.csv"):
        run_projection("Petrochemical", "data/petrochemical.csv")
