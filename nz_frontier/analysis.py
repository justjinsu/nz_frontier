import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple
from .types import Portfolio

def plot_frontier(results: List[Tuple[float, float, Portfolio]], title: str = "Net-Zero Efficient Frontier"):
    """
    Plots the efficient frontier (Risk vs Abatement).
    """
    risks = [r[0] for r in results]
    abatements = [r[1] for r in results]
    
    plt.figure(figsize=(10, 6))
    plt.plot(abatements, risks, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Abatement Target (A*)')
    plt.ylabel('Portfolio Transition Risk (R_P)')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_transition_paths(portfolios: List[Portfolio], tech_names: List[str]):
    """
    Plots the evolution of technology capacities over time.
    """
    n_periods = len(portfolios)
    n_tech = len(tech_names)
    
    weights_history = np.zeros((n_periods, n_tech))
    for t, p in enumerate(portfolios):
        weights_history[t, :] = p.weights
        
    plt.figure(figsize=(10, 6))
    for i in range(n_tech):
        plt.plot(range(n_periods), weights_history[:, i], label=tech_names[i], linewidth=2)
        
    plt.xlabel('Period')
    plt.ylabel('Capacity (GW)')
    plt.title('Technology Transition Pathway')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
