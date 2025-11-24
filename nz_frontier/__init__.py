"""
Net-Zero Frontier: Portfolio Theory for Corporate Decarbonization

A risk-efficiency framework for net-zero investment under uncertainty,
extending modern portfolio theory (Markowitz, 1952) to climate transition.
"""

__version__ = "0.1.0"
__author__ = "Jinsu Park"

try:
    from .types import Technology, Portfolio, RiskBreakdown, FrontierPoint
    from .risk import RiskModel
    from .optimizer import OptimizationEngine
    from .simulation import CostSimulator
    from .frontier import EfficientFrontier
    from .dynamic import DynamicOptimizer
    from .advanced import (
        MonteCarloFrontier,
        MonteCarloFrontierResult,
        StochasticDominanceAnalyzer,
        StochasticDominanceResult,
        RobustOptimizer,
        RobustOptimizationResult,
        DynamicRealOptionsOptimizer,
        build_correlation_matrix,
    )
except ImportError as e:
    print(f"Error importing nz_frontier components: {e}")
    print("Please ensure all dependencies are installed: pip install -r requirements.txt")
    # Re-raise to prevent broken usage, but user sees the message first
    raise

__all__ = [
    # Version info
    "__version__",
    "__author__",
    # Core types
    "Technology",
    "Portfolio",
    "RiskBreakdown",
    "FrontierPoint",
    # Core modules
    "RiskModel",
    "OptimizationEngine",
    "CostSimulator",
    "EfficientFrontier",
    "DynamicOptimizer",
    # Advanced modules
    "MonteCarloFrontier",
    "MonteCarloFrontierResult",
    "StochasticDominanceAnalyzer",
    "StochasticDominanceResult",
    "RobustOptimizer",
    "RobustOptimizationResult",
    "DynamicRealOptionsOptimizer",
    "build_correlation_matrix",
]
