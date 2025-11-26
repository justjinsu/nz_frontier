"""
Generate Figures for Revised Paper: (σ, C) Framework
=====================================================
New framework with clean separation of cost volatility and expected cost.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from nz_frontier import Technology, EfficientFrontier, build_correlation_matrix, RiskModel

# Set publication-quality style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 12

# Load POSCO technologies
df = pd.read_csv('data/korea_steel.csv')

technologies = []
for _, row in df.iterrows():
    t = Technology(
        name=row['name'],
        a=row['a'],
        c=row['c'],
        sigma=row['sigma'],
        rho=row.get('rho', 0.0),
        o=row.get('o', 0.0),
        tau=row.get('tau', 20.0),
        jump_intensity=row.get('jump_intensity', 0.0),
        jump_size=row.get('jump_size', 0.0),
        strike_price=row.get('strike_price', 0.0),
        learning_rate=row.get('learning_rate', 0.0),
        failure_prob=row.get('failure_prob', 0.0),
        loss_given_failure=row.get('loss_given_failure', 0.0),
        max_capacity=row.get('max_capacity', float('inf'))
    )
    technologies.append(t)

# Build correlation matrix and risk model
cov_matrix = build_correlation_matrix(technologies)
risk_model = RiskModel(technologies, cov_matrix)

print("=" * 80)
print("GENERATING REVISED PAPER FIGURES: (σ, C) Framework")
print("=" * 80)

# ============================================================================
# Figure 1: Efficient Frontier in (σ, C) Space
# ============================================================================
print("\n1. Figure 1: POSCO Efficient Frontier in (σ, C) Space...")

frontier = EfficientFrontier(technologies, cov_matrix)
results = frontier.compute(
    abatement_min=10,
    abatement_max=50,
    n_points=25,
    lambda_param=1.2,  # Moderate risk aversion
)

fig, ax = plt.subplots(figsize=(8, 6))

# Extract data
volatilities = [r.volatility for r in results]
costs = [r.expected_cost for r in results]
abatements = [r.abatement for r in results]

# Plot frontier
scatter = ax.scatter(volatilities, costs, c=abatements, cmap='viridis',
                     s=120, edgecolors='black', linewidth=1.5, zorder=3)
ax.plot(volatilities, costs, 'k--', alpha=0.3, linewidth=1.5, zorder=2)

# Colorbar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Abatement Target (Mt CO₂)', fontsize=11, fontweight='bold')

# Labels
ax.set_xlabel('Cost Volatility σ_P (Standard Deviation)', fontsize=12, fontweight='bold')
ax.set_ylabel('Expected Cost C_P ($M)', fontsize=12, fontweight='bold')
ax.set_title('Net-Zero Efficient Frontier: POSCO Steel Decarbonization\n' +
             'Goal: Minimize cost and uncertainty simultaneously',
             fontsize=13, fontweight='bold', pad=15)
ax.grid(alpha=0.3, linestyle='--')

# Add annotation showing "better" direction
ax.annotate('', xy=(min(volatilities), min(costs)),
            xytext=(max(volatilities)*0.8, max(costs)*0.8),
            arrowprops=dict(arrowstyle='->', color='green', lw=3, alpha=0.6))
ax.text(min(volatilities), min(costs) - (max(costs) - min(costs)) * 0.08,
        'Better\n(Lower cost,\nLower volatility)', fontsize=10, color='green',
        ha='left', va='top', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.3))

plt.tight_layout()
plt.savefig('paper/figure_frontier_v2.pdf', dpi=300, bbox_inches='tight')
plt.savefig('paper/figure_frontier_v2.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: paper/figure_frontier_v2.pdf/png")

# ============================================================================
# Figure 2: Risk-Return Tradeoff (Multiple λ values)
# ============================================================================
print("\n2. Figure 2: Risk Aversion Parameter Impact...")

fig, ax = plt.subplots(figsize=(8, 6))

lambda_values = [0.5, 1.0, 1.5, 2.0]
colors = ['#e63946', '#f77f00', '#06a77d', '#2a9d8f']

for lambda_val, color in zip(lambda_values, colors):
    results_lambda = frontier.compute(
        abatement_min=10,
        abatement_max=50,
        n_points=15,
        lambda_param=lambda_val,
    )

    vols = [r.volatility for r in results_lambda]
    costs_l = [r.expected_cost for r in results_lambda]

    ax.plot(vols, costs_l, 'o-', linewidth=2.5, markersize=7,
            color=color, label=f'λ = {lambda_val}', alpha=0.8)

ax.set_xlabel('Cost Volatility σ_P', fontsize=12, fontweight='bold')
ax.set_ylabel('Expected Cost C_P ($M)', fontsize=12, fontweight='bold')
ax.set_title('Impact of Risk Aversion Parameter λ\n' +
             'Higher λ → Prefer lower volatility (more conservative)',
             fontsize=13, fontweight='bold', pad=15)
ax.legend(title='Risk Aversion', fontsize=10, title_fontsize=11, loc='upper left')
ax.grid(alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('paper/figure_risk_aversion_v2.pdf', dpi=300, bbox_inches='tight')
plt.savefig('paper/figure_risk_aversion_v2.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: paper/figure_risk_aversion_v2.pdf/png")

# ============================================================================
# Figure 3: Portfolio Composition Along Frontier
# ============================================================================
print("\n3. Figure 3: Portfolio Composition Evolution...")

# Select 4 representative points
indices = [0, len(results)//3, 2*len(results)//3, -1]
selected_points = [results[i] for i in indices]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

tech_names_short = [t.name.split('(')[0].strip()[:15] for t in technologies]
colors_tech = plt.cm.Set3(np.linspace(0, 1, len(technologies)))

for idx, (point, ax) in enumerate(zip(selected_points, axes)):
    weights = point.portfolio.weights
    significant = weights > 0.01  # Only show meaningful allocations

    if np.sum(significant) > 0:
        ax.pie(weights[significant],
               labels=[tech_names_short[i] for i in range(len(technologies)) if significant[i]],
               colors=[colors_tech[i] for i in range(len(technologies)) if significant[i]],
               autopct='%1.1f%%', startangle=90, textprops={'fontsize': 9})

    ax.set_title(f'Point {idx+1}: A = {point.abatement:.1f} Mt\n' +
                 f'σ_P = {point.volatility:.2f}, C_P = ${point.expected_cost:.0f}M',
                 fontsize=11, fontweight='bold', pad=10)

plt.suptitle('Portfolio Composition Along Efficient Frontier\n' +
             'As abatement targets increase, portfolio mix evolves',
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('paper/figure_portfolios_v2.pdf', dpi=300, bbox_inches='tight')
plt.savefig('paper/figure_portfolios_v2.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: paper/figure_portfolios_v2.pdf/png")

# ============================================================================
# Figure 4: Technology-Level Analysis
# ============================================================================
print("\n4. Figure 4: Technology Risk-Cost Profile...")

fig, ax = plt.subplots(figsize=(10, 7))

# For each technology, compute unit risk-adjusted cost
tech_data = []
for tech in technologies:
    # Unit metrics
    unit_volatility = tech.sigma
    unit_cost = tech.c + tech.failure_prob * tech.loss_given_failure
    abatement = tech.a

    tech_data.append({
        'name': tech.name.split('(')[0].strip(),
        'volatility': unit_volatility,
        'cost': unit_cost,
        'abatement': abatement,
        'max_capacity': tech.max_capacity if tech.max_capacity < 1000 else 100
    })

df_tech = pd.DataFrame(tech_data)

# Scatter plot
scatter = ax.scatter(df_tech['volatility'], df_tech['cost'],
                     s=df_tech['max_capacity']*20,  # Size by capacity
                     c=df_tech['abatement'], cmap='RdYlGn',
                     alpha=0.7, edgecolors='black', linewidth=1.5)

# Labels
for i, row in df_tech.iterrows():
    ax.annotate(row['name'], (row['volatility'], row['cost']),
                xytext=(5, 5), textcoords='offset points', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

# Colorbar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Abatement Potential (tCO₂/t)', fontsize=11, fontweight='bold')

ax.set_xlabel('Technology Volatility σ_j', fontsize=12, fontweight='bold')
ax.set_ylabel('Risk-Adjusted Unit Cost c̃_j ($M/Mt)', fontsize=12, fontweight='bold')
ax.set_title('Technology Landscape: Risk-Cost Trade-offs\n' +
             'Bubble size = Maximum capacity',
             fontsize=13, fontweight='bold', pad=15)
ax.grid(alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('paper/figure_technologies_v2.pdf', dpi=300, bbox_inches='tight')
plt.savefig('paper/figure_technologies_v2.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: paper/figure_technologies_v2.pdf/png")

# ============================================================================
# Summary Statistics
# ============================================================================
print("\n" + "=" * 80)
print("FIGURE GENERATION SUMMARY")
print("=" * 80)

print(f"\nFrontier Statistics:")
print(f"  Abatement range: {min(abatements):.1f} - {max(abatements):.1f} Mt")
print(f"  Cost range: ${min(costs):.0f}M - ${max(costs):.0f}M")
print(f"  Volatility range: {min(volatilities):.2f} - {max(volatilities):.2f}")

print(f"\nTechnology Statistics:")
print(f"  Number of technologies: {len(technologies)}")
print(f"  Total capacity: {sum(t.max_capacity for t in technologies if t.max_capacity < 1000):.1f} Mt")
print(f"  Average unit cost: ${np.mean([t.c for t in technologies]):.0f}M/Mt")
print(f"  Average volatility: {np.mean([t.sigma for t in technologies]):.2f}")

print("\n✓ All figures generated successfully in (σ, C) space!")
print("\nKey Framework Features:")
print("  1. Clean separation: σ_P (2nd moment) vs C_P (1st moment)")
print("  2. Markowitz correspondence achieved")
print("  3. Intuitive interpretation: minimize both cost and volatility")
print("  4. Goal direction clear: bottom-left is better")
