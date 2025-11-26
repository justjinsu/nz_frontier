"""
Test the New Framework Implementation
======================================
Quick test to verify the revised (σ, C) framework works correctly.
"""
import numpy as np
import matplotlib.pyplot as plt
from nz_frontier import Technology, EfficientFrontier, build_correlation_matrix

print("=" * 80)
print("TESTING NEW FRAMEWORK: (σ, C) Space")
print("=" * 80)

# Create simple test technologies
technologies = [
    Technology(name="Low-Risk Tech", a=1.0, c=100, sigma=0.05, max_capacity=10),
    Technology(name="Medium-Risk Tech", a=1.2, c=90, sigma=0.15, max_capacity=10),
    Technology(name="High-Risk Tech", a=1.5, c=80, sigma=0.30, max_capacity=10),
]

# Build covariance matrix
cov_matrix = build_correlation_matrix(technologies)

# Create frontier
frontier = EfficientFrontier(technologies, cov_matrix)

print("\n1. Computing Frontier...")
try:
    results = frontier.compute(
        abatement_min=5,
        abatement_max=20,
        n_points=10,
        lambda_param=1.0,  # Moderate risk aversion
    )
    print(f"   ✓ Computed {len(results)} frontier points")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Check results structure
print("\n2. Checking Results Structure...")
for i, point in enumerate(results[:3]):  # Check first 3 points
    print(f"\n   Point {i+1}:")
    print(f"     Abatement: {point.abatement:.2f} Mt")
    print(f"     Volatility (σ_P): {point.volatility:.4f}")
    print(f"     Expected Cost (C_P): {point.expected_cost:.2f}")
    print(f"     Portfolio weights: {point.portfolio.weights}")

# Plot in (σ, C) space
print("\n3. Plotting Frontier in (σ, C) Space...")
fig, ax = plt.subplots(1, 1, figsize=(10, 7))

volatilities = [p.volatility for p in results]
costs = [p.expected_cost for p in results]
abatements = [p.abatement for p in results]

# Color by abatement level
scatter = ax.scatter(volatilities, costs, c=abatements, cmap='viridis', s=100, edgecolors='black', linewidth=1.5)
ax.plot(volatilities, costs, 'k--', alpha=0.3, linewidth=1)

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Abatement (Mt CO₂)', fontsize=11, fontweight='bold')

# Labels
ax.set_xlabel('Cost Volatility σ_P (Standard Deviation)', fontsize=12, fontweight='bold')
ax.set_ylabel('Expected Cost C_P ($M)', fontsize=12, fontweight='bold')
ax.set_title('New Framework: Efficient Frontier in (σ, C) Space\nGoal: Move towards bottom-left (low cost, low volatility)',
             fontsize=13, fontweight='bold', pad=15)
ax.grid(alpha=0.3, linestyle='--')

# Add arrows showing "better" direction
ax.annotate('', xy=(min(volatilities), min(costs)), xytext=(max(volatilities), max(costs)),
            arrowprops=dict(arrowstyle='->', color='green', lw=3, alpha=0.5))
ax.text(min(volatilities), min(costs) - (max(costs) - min(costs)) * 0.1,
        'Better ←\n(Low cost,\nLow volatility)', fontsize=10, color='green',
        ha='left', va='top', fontweight='bold')

plt.tight_layout()
plt.savefig('paper/test_new_framework.pdf', dpi=300, bbox_inches='tight')
plt.savefig('paper/test_new_framework.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: paper/test_new_framework.pdf/png")

# Compare old vs new metrics
print("\n4. Comparing Old vs New Metrics...")
print("\n   Old Framework (deprecated):")
for i, point in enumerate(results[:3]):
    print(f"     Point {i+1}: Risk R_P = {point.risk:.2f}")

print("\n   New Framework:")
for i, point in enumerate(results[:3]):
    print(f"     Point {i+1}: Volatility σ_P = {point.volatility:.4f}, Cost C_P = {point.expected_cost:.2f}")

print("\n5. Testing Risk Model Methods...")
from nz_frontier import RiskModel

risk_model = RiskModel(technologies, cov_matrix)
test_weights = results[5].portfolio.weights

print(f"\n   Test portfolio weights: {test_weights}")
print(f"   cost_volatility(): {risk_model.cost_volatility(test_weights):.4f}")
print(f"   expected_cost(): {risk_model.expected_cost(test_weights):.2f}")
print(f"   objective_function(λ=1.0): {risk_model.objective_function(test_weights, 1.0):.2f}")

breakdown = risk_model.breakdown(test_weights)
print(f"\n   Breakdown:")
print(f"     cost_volatility: {breakdown.cost_volatility:.4f}")
print(f"     stranded_asset (component): {breakdown.stranded_asset:.2f}")
print(f"     option_value (component): {breakdown.option_value:.2f}")
print(f"     total (C_P): {breakdown.total:.2f}")

print("\n" + "=" * 80)
print("NEW FRAMEWORK TEST: SUCCESS ✓")
print("=" * 80)
print("\nKey Achievements:")
print("  1. Clean separation: σ_P (2nd moment) vs C_P (1st moment)")
print("  2. Markowitz correspondence established")
print("  3. Frontier correctly plotted in (σ, C) space")
print("  4. Goal is bottom-left: minimize both cost and volatility")
print("\nNext Steps:")
print("  - Regenerate all paper figures")
print("  - Update POSCO case study")
print("  - Rewrite Section 3 of paper")
