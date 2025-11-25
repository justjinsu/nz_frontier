"""
Generate Three Visualization Options for Frontier Comparison
=============================================================
Creates mockups of 3 different approaches to visualizing the three-tier frontier.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from nz_frontier import Technology, EfficientFrontier, build_correlation_matrix, RiskModel

plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.size'] = 10

# ============================================================================
# LOAD DATA AND COMPUTE FRONTIERS
# ============================================================================

# Load Global Technologies
df_global = pd.read_csv('data/global_steel.csv')
global_techs = []
for _, row in df_global.iterrows():
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
    global_techs.append(t)

# Load POSCO Technologies
df_posco = pd.read_csv('data/korea_steel.csv')
posco_techs = []
for _, row in df_posco.iterrows():
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
    posco_techs.append(t)

# Adjust POSCO constraints (legacy asset risk)
for tech in posco_techs:
    if 'BF-BOF' in tech.name and 'Baseline' in tech.name:
        tech.failure_prob = 0.05  # Higher than global
        tech.loss_given_failure = 0.30

# Compute Global Frontier
cov_global = build_correlation_matrix(global_techs)
risk_model_global = RiskModel(global_techs, cov_global)
frontier_global = EfficientFrontier(global_techs, cov_global)
results_global = frontier_global.compute(
    abatement_min=10,
    abatement_max=100,
    n_points=30,
    lambda_param=1.2,
    gamma_param=0.8
)

# Compute POSCO Frontier
cov_posco = build_correlation_matrix(posco_techs)
risk_model_posco = RiskModel(posco_techs, cov_posco)
frontier_posco = EfficientFrontier(posco_techs, cov_posco)
results_posco = frontier_posco.compute(
    abatement_min=10,
    abatement_max=50,
    n_points=30,
    lambda_param=1.2,
    gamma_param=0.8
)

# POSCO Announced Strategy
posco_announced = {
    'Scrap-EAF (Hyundai)': 5.0,
    'HyREX H2-DRI (POSCO)': 0.5,
    'FINEX + CCUS': 12.0,
    'NG-DRI-EAF': 5.0,
    'BF-BOF + CCUS (POSCO)': 10.0,
    'BF-BOF (Baseline)': 11.5,
}

announced_weights = np.zeros(len(posco_techs))
for i, tech in enumerate(posco_techs):
    for announced_name, capacity in posco_announced.items():
        if announced_name in tech.name or tech.name in announced_name:
            announced_weights[i] = capacity
            break

announced_abatement = sum(w * t.a for w, t in zip(announced_weights, posco_techs))
announced_risk = risk_model_posco.total_risk(announced_weights, lambda_param=1.2, gamma_param=0.8)

# Extract data
abatement_global = [r.abatement for r in results_global]
risk_global = [r.risk for r in results_global]

abatement_posco = [r.abatement for r in results_posco]
risk_posco = [r.risk for r in results_posco]

# Find POSCO optimal at announced abatement
from nz_frontier import OptimizationEngine
optimizer_posco = OptimizationEngine(posco_techs, risk_model_posco)
optimal_portfolio, _ = optimizer_posco.solve_for_target(
    target_abatement=announced_abatement,
    lambda_param=1.2,
    gamma_param=0.8,
    return_breakdown=True
)
optimal_risk = risk_model_posco.total_risk(optimal_portfolio.weights, lambda_param=1.2, gamma_param=0.8)

print("Data Summary:")
print(f"  Global Risk Range: [{min(risk_global):.1f}, {max(risk_global):.1f}]")
print(f"  POSCO Risk Range: [{min(risk_posco):.1f}, {max(risk_posco):.1f}]")
print(f"  POSCO Announced: {announced_risk:.1f}")
print(f"  POSCO Optimal: {optimal_risk:.1f}")
print(f"  Gap: {announced_risk - optimal_risk:.1f}")

# ============================================================================
# OPTION A: ABSOLUTE RISK (Shift to Remove Negatives)
# ============================================================================

fig_a, ax = plt.subplots(1, 1, figsize=(10, 6))

# Shift to make all positive
shift = -min(risk_global) + 100  # Add buffer of 100
risk_global_shifted = [r + shift for r in risk_global]
risk_posco_shifted = [r + shift for r in risk_posco]
announced_risk_shifted = announced_risk + shift
optimal_risk_shifted = optimal_risk + shift

# Plot
ax.plot(abatement_global, risk_global_shifted, 'o-', color='#2A9D8F', linewidth=2.5,
        markersize=6, label='Global Industry Frontier (Theoretical Best)', alpha=0.9)
ax.plot(abatement_posco, risk_posco_shifted, 's-', color='#E76F51', linewidth=2.5,
        markersize=6, label='POSCO-Feasible Frontier (POSCO-Optimal)', alpha=0.9)
ax.plot(announced_abatement, announced_risk_shifted, '*', color='#C1121F',
        markersize=20, label='POSCO Announced 2030 Strategy', zorder=5)
ax.plot(announced_abatement, optimal_risk_shifted, 'D', color='#06A77D',
        markersize=10, label='POSCO Optimal at Target', zorder=5, markeredgecolor='black', markeredgewidth=1)

# Annotations
ax.annotate(f'Target: {announced_abatement:.1f} Mt\nRisk: {announced_risk_shifted:.0f}',
            xy=(announced_abatement, announced_risk_shifted),
            xytext=(announced_abatement + 5, announced_risk_shifted + 200),
            fontsize=9, ha='left',
            arrowprops=dict(arrowstyle='->', color='#C1121F', lw=1.5),
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#C1121F', linewidth=1.5))

# Gap visualization
ax.annotate('', xy=(announced_abatement, announced_risk_shifted),
            xytext=(announced_abatement, optimal_risk_shifted),
            arrowprops=dict(arrowstyle='<->', color='red', lw=2))
ax.text(announced_abatement + 1, (announced_risk_shifted + optimal_risk_shifted) / 2,
        f'Gap: {announced_risk - optimal_risk:.1f}\n(36% sub-optimal)',
        fontsize=10, color='red', ha='left', va='center',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFE5E5', edgecolor='red', linewidth=1))

# Shaded region (cost of POSCO constraints)
ax.fill_between(abatement_posco, risk_global_shifted[:len(abatement_posco)],
                risk_posco_shifted, alpha=0.15, color='orange',
                label='Cost of POSCO Constraints')

ax.set_xlabel('Total Abatement (Mt CO₂)', fontsize=12, fontweight='bold')
ax.set_ylabel('Adjusted Portfolio Transition Risk', fontsize=12, fontweight='bold')
ax.set_title('Option A: Absolute Risk Visualization\n(All values shifted to positive range)',
             fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper left', fontsize=10, framealpha=0.95)
ax.grid(alpha=0.3, linestyle='--')

# Add note explaining adjustment
ax.text(0.98, 0.02, f'Note: Risk values adjusted by +{shift:.0f} to remove negatives\n(original range: [{min(risk_global):.0f}, {max(risk_global):.0f}])',
        transform=ax.transAxes, fontsize=8, ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='gray', linewidth=0.5))

plt.tight_layout()
plt.savefig('paper/figure_three_tier_option_a.pdf', dpi=300, bbox_inches='tight')
plt.savefig('paper/figure_three_tier_option_a.png', dpi=300, bbox_inches='tight')
print("✓ Saved: Option A (Absolute Risk)")

# ============================================================================
# OPTION B: SEPARATE PLOTS (Dual Panel)
# ============================================================================

fig_b, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# LEFT PANEL: POSCO Comparison (Zoomed)
ax1.plot(abatement_posco, risk_posco, 's-', color='#E76F51', linewidth=2.5,
        markersize=6, label='POSCO-Feasible Frontier', alpha=0.9)
ax1.plot(announced_abatement, announced_risk, '*', color='#C1121F',
        markersize=20, label='POSCO Announced', zorder=5)
ax1.plot(announced_abatement, optimal_risk, 'D', color='#06A77D',
        markersize=10, label='POSCO Optimal', zorder=5, markeredgecolor='black', markeredgewidth=1)

# Gap visualization
ax1.annotate('', xy=(announced_abatement, announced_risk),
            xytext=(announced_abatement, optimal_risk),
            arrowprops=dict(arrowstyle='<->', color='red', lw=2.5))
ax1.text(announced_abatement - 2, (announced_risk + optimal_risk) / 2,
        f'Optimality Gap\n{announced_risk - optimal_risk:.1f} units\n(36% loss)',
        fontsize=11, color='red', ha='right', va='center',
        bbox=dict(boxstyle='round,pad=0.6', facecolor='#FFE5E5', edgecolor='red', linewidth=2))

ax1.set_xlabel('Total Abatement (Mt CO₂)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Portfolio Transition Risk', fontsize=12, fontweight='bold')
ax1.set_title('(a) POSCO Strategy Evaluation\n(Zoomed view showing sub-optimality)',
             fontsize=13, fontweight='bold', pad=15)
ax1.legend(loc='upper left', fontsize=10, framealpha=0.95)
ax1.grid(alpha=0.3, linestyle='--')
ax1.set_ylim([min(risk_posco) - 20, announced_risk + 50])

# RIGHT PANEL: Global vs POSCO Gap
ax2.plot(abatement_global, risk_global, 'o-', color='#2A9D8F', linewidth=2.5,
        markersize=6, label='Global Industry Frontier', alpha=0.9)
ax2.plot(abatement_posco, risk_posco, 's-', color='#E76F51', linewidth=2.5,
        markersize=6, label='POSCO-Feasible Frontier', alpha=0.9)

# Shaded region
ax2.fill_between(abatement_posco,
                [risk_global[i] for i in range(len(abatement_posco))],
                risk_posco, alpha=0.2, color='orange',
                label='Cost of POSCO Constraints')

# Find constraint cost at target
idx_target = min(range(len(abatement_posco)), key=lambda i: abs(abatement_posco[i] - announced_abatement))
risk_global_at_target = risk_global[idx_target]
constraint_cost = optimal_risk - risk_global_at_target

ax2.annotate(f'Constraint Cost:\n{constraint_cost:.1f} units',
            xy=(announced_abatement, optimal_risk),
            xytext=(announced_abatement + 10, optimal_risk + 500),
            fontsize=11, ha='left',
            arrowprops=dict(arrowstyle='->', color='orange', lw=2),
            bbox=dict(boxstyle='round,pad=0.6', facecolor='#FFF4E5', edgecolor='orange', linewidth=2))

ax2.set_xlabel('Total Abatement (Mt CO₂)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Portfolio Transition Risk', fontsize=12, fontweight='bold')
ax2.set_title('(b) Industry Benchmark Comparison\n(Showing cost of firm-level constraints)',
             fontsize=13, fontweight='bold', pad=15)
ax2.legend(loc='upper left', fontsize=10, framealpha=0.95)
ax2.grid(alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('paper/figure_three_tier_option_b.pdf', dpi=300, bbox_inches='tight')
plt.savefig('paper/figure_three_tier_option_b.png', dpi=300, bbox_inches='tight')
print("✓ Saved: Option B (Separate Plots)")

# ============================================================================
# OPTION C: COMPONENT STACKING (Bar Chart Decomposition)
# ============================================================================

# Compute risk breakdowns
from nz_frontier import RiskBreakdown

# Global at target
optimizer_global = OptimizationEngine(global_techs, risk_model_global)
global_portfolio, global_breakdown = optimizer_global.solve_for_target(
    target_abatement=announced_abatement,
    lambda_param=1.2,
    gamma_param=0.8,
    return_breakdown=True
)

# POSCO optimal breakdown
posco_breakdown = _

# POSCO announced breakdown
announced_breakdown = risk_model_posco.breakdown(announced_weights, lambda_param=1.2, gamma_param=0.8)

fig_c, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# LEFT: Component Stacking
portfolios = ['Global\nOptimal', 'POSCO\nOptimal', 'POSCO\nAnnounced']
x_pos = np.arange(len(portfolios))

# Component values (multiply by lambda/gamma)
volatility = [
    global_breakdown.cost_volatility,
    optimal_portfolio.breakdown.cost_volatility if hasattr(optimal_portfolio, 'breakdown') else 0,
    announced_breakdown.cost_volatility
]

stranded = [
    global_breakdown.stranded_asset * 1.2,
    optimal_portfolio.breakdown.stranded_asset * 1.2 if hasattr(optimal_portfolio, 'breakdown') else 0,
    announced_breakdown.stranded_asset * 1.2
]

option = [
    global_breakdown.option_value * (-0.8),
    optimal_portfolio.breakdown.option_value * (-0.8) if hasattr(optimal_portfolio, 'breakdown') else 0,
    announced_breakdown.option_value * (-0.8)
]

# Need to recalculate breakdowns properly
# Recalculate all breakdowns
global_breakdown_calc = risk_model_global.breakdown(global_portfolio.weights, lambda_param=1.2, gamma_param=0.8)
optimal_breakdown_calc = risk_model_posco.breakdown(optimal_portfolio.weights, lambda_param=1.2, gamma_param=0.8)

volatility = [
    global_breakdown_calc.cost_volatility,
    optimal_breakdown_calc.cost_volatility,
    announced_breakdown.cost_volatility
]

stranded = [
    global_breakdown_calc.stranded_asset * 1.2,
    optimal_breakdown_calc.stranded_asset * 1.2,
    announced_breakdown.stranded_asset * 1.2
]

option = [
    global_breakdown_calc.option_value * (-0.8),
    optimal_breakdown_calc.option_value * (-0.8),
    announced_breakdown.option_value * (-0.8)
]

total_risk_vals = [
    risk_model_global.total_risk(global_portfolio.weights, lambda_param=1.2, gamma_param=0.8),
    optimal_risk,
    announced_risk
]

# Stacked bar chart
width = 0.5
p1 = ax1.bar(x_pos, volatility, width, label='Cost Volatility (σ²)', color='#457B9D', edgecolor='black', linewidth=0.5)
p2 = ax1.bar(x_pos, stranded, width, bottom=volatility, label='Stranded Asset Risk (λ·h)', color='#E63946', edgecolor='black', linewidth=0.5)
p3 = ax1.bar(x_pos, option, width, bottom=[v+s for v,s in zip(volatility, stranded)],
            label='Option Value Loss (-γ·g)', color='#06A77D', edgecolor='black', linewidth=0.5)

# Add total risk line
ax1.plot(x_pos, total_risk_vals, 'o-', color='black', linewidth=2.5, markersize=12,
        label='Total Risk', zorder=10, markeredgecolor='white', markeredgewidth=2)

# Value labels
for i, (v, s, o, tot) in enumerate(zip(volatility, stranded, option, total_risk_vals)):
    ax1.text(i, tot + 20, f'{tot:.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax1.set_ylabel('Risk Component Value', fontsize=12, fontweight='bold')
ax1.set_title('(a) Risk Component Decomposition at Target Abatement (48.9 Mt)',
             fontsize=13, fontweight='bold', pad=15)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(portfolios, fontsize=11)
ax1.legend(loc='upper left', fontsize=10, framealpha=0.95)
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

# RIGHT: Gap Decomposition
gap_components = ['Cost\nVolatility', 'Stranded\nAsset', 'Option\nValue\nLoss']
gap_values = [
    announced_breakdown.cost_volatility - optimal_breakdown_calc.cost_volatility,
    (announced_breakdown.stranded_asset - optimal_breakdown_calc.stranded_asset) * 1.2,
    (announced_breakdown.option_value - optimal_breakdown_calc.option_value) * (-0.8)
]
gap_pct = [g / (announced_risk - optimal_risk) * 100 for g in gap_values]

colors_gap = ['#457B9D', '#E63946', '#06A77D']
bars = ax2.barh(gap_components, gap_values, color=colors_gap, edgecolor='black', linewidth=1)

# Value labels
for i, (val, pct) in enumerate(zip(gap_values, gap_pct)):
    ax2.text(val + 2, i, f'{val:.1f}\n({pct:.1f}%)', va='center', ha='left', fontsize=11, fontweight='bold')

ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax2.set_xlabel('Contribution to Optimality Gap', fontsize=12, fontweight='bold')
ax2.set_title(f'(b) Gap Decomposition: Why is POSCO {announced_risk - optimal_risk:.1f} units above optimal?',
             fontsize=13, fontweight='bold', pad=15)
ax2.grid(axis='x', alpha=0.3, linestyle='--')

# Add total gap annotation
total_gap = sum(gap_values)
ax2.text(0.98, 0.98, f'Total Gap: {total_gap:.1f}\n(100.0%)',
        transform=ax2.transAxes, fontsize=12, ha='right', va='top', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='#FFE5E5', edgecolor='red', linewidth=2))

plt.tight_layout()
plt.savefig('paper/figure_three_tier_option_c.pdf', dpi=300, bbox_inches='tight')
plt.savefig('paper/figure_three_tier_option_c.png', dpi=300, bbox_inches='tight')
print("✓ Saved: Option C (Component Stacking)")

print("\n" + "="*80)
print("ALL THREE OPTIONS GENERATED SUCCESSFULLY")
print("="*80)
print("\nFiles created:")
print("  • paper/figure_three_tier_option_a.pdf/png - Absolute Risk (shifted)")
print("  • paper/figure_three_tier_option_b.pdf/png - Separate Plots (dual panel)")
print("  • paper/figure_three_tier_option_c.pdf/png - Component Stacking (bar charts)")
