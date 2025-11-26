"""
POSCO Case Study Visualization: New (σ, C) Framework
====================================================
Simplified visualization showing POSCO's announced strategy vs optimal frontier.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from nz_frontier import Technology, EfficientFrontier, build_correlation_matrix, RiskModel, OptimizationEngine

plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.size'] = 10

print("=" * 80)
print("POSCO CASE STUDY: New (σ, C) Framework")
print("=" * 80)

# Load POSCO technologies
df = pd.read_csv('data/korea_steel.csv')
technologies = []
for _, row in df.iterrows():
    t = Technology(
        name=row['name'], a=row['a'], c=row['c'], sigma=row['sigma'],
        rho=row.get('rho', 0.0), o=row.get('o', 0.0), tau=row.get('tau', 20.0),
        failure_prob=row.get('failure_prob', 0.0),
        loss_given_failure=row.get('loss_given_failure', 0.0),
        max_capacity=row.get('max_capacity', float('inf'))
    )
    technologies.append(t)

# Build models
cov_matrix = build_correlation_matrix(technologies)
risk_model = RiskModel(technologies, cov_matrix)

# POSCO Announced 2030 Strategy
posco_announced = {
    'Scrap-EAF (Hyundai)': 5.0,
    'HyREX H2-DRI (POSCO)': 0.5,
    'FINEX + CCUS': 12.0,
    'NG-DRI-EAF': 5.0,
    'BF-BOF + CCUS (POSCO)': 10.0,
    'BF-BOF (Baseline)': 11.5,
}

announced_weights = np.zeros(len(technologies))
for i, tech in enumerate(technologies):
    for name, cap in posco_announced.items():
        if name in tech.name or tech.name in name:
            announced_weights[i] = cap
            break

# Calculate announced metrics
announced_abatement = sum(w * t.a for w, t in zip(announced_weights, technologies))
announced_volatility = risk_model.cost_volatility(announced_weights)
announced_cost = risk_model.expected_cost(announced_weights)

print(f"\nPOSCO Announced 2030 Strategy:")
print(f"  Abatement: {announced_abatement:.1f} Mt CO2")
print(f"  Cost Volatility (σ_P): {announced_volatility:.2f}")
print(f"  Expected Cost (C_P): ${announced_cost:.0f}M")

# Compute POSCO efficient frontier
print(f"\nComputing POSCO Efficient Frontier...")
frontier = EfficientFrontier(technologies, cov_matrix)
results = frontier.compute(
    abatement_min=10,
    abatement_max=50,
    n_points=25,
    lambda_param=1.0,
)

# Find optimal at announced abatement
optimizer = OptimizationEngine(technologies, risk_model)
optimal_portfolio, optimal_breakdown = optimizer.solve_for_target(
    target_abatement=announced_abatement,
    lambda_param=1.0,
    return_breakdown=True
)

optimal_volatility = risk_model.cost_volatility(optimal_portfolio.weights)
optimal_cost = risk_model.expected_cost(optimal_portfolio.weights)

print(f"\nPOSCO Optimal Portfolio (at {announced_abatement:.1f} Mt):")
print(f"  Cost Volatility (σ_P): {optimal_volatility:.2f}")
print(f"  Expected Cost (C_P): ${optimal_cost:.0f}M")

print(f"\nInefficiency Analysis:")
print(f"  Excess Volatility: {announced_volatility - optimal_volatility:.2f} ({(announced_volatility/optimal_volatility - 1)*100:.1f}%)")
print(f"  Excess Cost: ${announced_cost - optimal_cost:.0f}M ({(announced_cost/optimal_cost - 1)*100:.1f}%)")

# ============================================================================
# Visualization
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 7))

# Plot frontier
volatilities = [r.volatility for r in results]
costs = [r.expected_cost for r in results]
abatements = [r.abatement for r in results]

scatter = ax.scatter(volatilities, costs, c=abatements, cmap='viridis',
                     s=100, edgecolors='black', linewidth=1.5, zorder=3, alpha=0.8)
ax.plot(volatilities, costs, 'k--', alpha=0.3, linewidth=2, zorder=2)

# Colorbar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Abatement (Mt CO₂)', fontsize=11, fontweight='bold')

# Plot optimal point
ax.plot(optimal_volatility, optimal_cost, 'D', color='#06A77D',
        markersize=16, markeredgecolor='black', markeredgewidth=2,
        label=f'POSCO Optimal (A={announced_abatement:.1f} Mt)', zorder=10)

# Plot announced point
ax.plot(announced_volatility, announced_cost, '*', color='#C1121F',
        markersize=24, markeredgecolor='black', markeredgewidth=1.5,
        label='POSCO Announced 2030', zorder=10)

# Draw efficiency gap
ax.annotate('', xy=(announced_volatility, announced_cost),
            xytext=(optimal_volatility, optimal_cost),
            arrowprops=dict(arrowstyle='<-', color='red', lw=3, linestyle='--'))

# Add gap annotation
gap_text = f'Investment\nInefficiency\nΔσ = {announced_volatility - optimal_volatility:.2f}\nΔC = ${announced_cost - optimal_cost:.0f}M'
ax.text((announced_volatility + optimal_volatility) / 2,
        (announced_cost + optimal_cost) / 2,
        gap_text,
        fontsize=10, color='red', ha='center', va='bottom', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.7', facecolor='#FFE5E5', edgecolor='red', linewidth=2))

# Labels
ax.set_xlabel('Cost Volatility σ_P (Standard Deviation)', fontsize=12, fontweight='bold')
ax.set_ylabel('Expected Cost C_P ($M)', fontsize=12, fontweight='bold')
ax.set_title('POSCO Investment Efficiency Analysis\n' +
             'Announced Strategy vs Efficient Frontier',
             fontsize=14, fontweight='bold', pad=15)
ax.legend(loc='upper left', fontsize=11, framealpha=0.95)
ax.grid(alpha=0.3, linestyle='--')

# Add interpretation box
interpretation = f"""Key Finding:
POSCO's announced 2030 strategy is OFF the efficient frontier.

At the same abatement target ({announced_abatement:.1f} Mt),
POSCO could achieve:
  • {(1 - optimal_volatility/announced_volatility)*100:.0f}% lower volatility
  • ${announced_cost - optimal_cost:.0f}M lower cost

This represents sub-optimal investment allocation."""

ax.text(0.98, 0.02, interpretation,
        transform=ax.transAxes, fontsize=9, ha='right', va='bottom',
        family='monospace',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='#FFF9E5',
                  edgecolor='#E76F51', linewidth=2))

plt.tight_layout()
plt.savefig('paper/posco_simple_v2.pdf', dpi=300, bbox_inches='tight')
plt.savefig('paper/posco_simple_v2.png', dpi=300, bbox_inches='tight')

print("\n✓ Saved: paper/posco_simple_v2.pdf/png")
print("\n" + "=" * 80)
print("POSCO CASE STUDY COMPLETE")
print("=" * 80)
