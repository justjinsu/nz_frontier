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
    "Technology",
    "Portfolio",
    "RiskModel",
    "OptimizationEngine",
    "CostSimulator",
    "EfficientFrontier",
    "DynamicOptimizer",
    "RiskBreakdown",
    "FrontierPoint",
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
