"""
Option A Updated: Absolute Risk with Same Abatement Target
===========================================================
Shows all three frontiers with emphasis on comparing at the SAME target (48.9 Mt)
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from nz_frontier import Technology, EfficientFrontier, build_correlation_matrix, RiskModel, OptimizationEngine

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
optimizer_posco = OptimizationEngine(posco_techs, risk_model_posco)
optimal_portfolio, optimal_breakdown = optimizer_posco.solve_for_target(
    target_abatement=announced_abatement,
    lambda_param=1.2,
    gamma_param=0.8,
    return_breakdown=True
)
optimal_risk = risk_model_posco.total_risk(optimal_portfolio.weights, lambda_param=1.2, gamma_param=0.8)

# Find GLOBAL optimal at SAME abatement target (48.9 Mt)
optimizer_global = OptimizationEngine(global_techs, risk_model_global)
global_portfolio_at_target, global_breakdown_at_target = optimizer_global.solve_for_target(
    target_abatement=announced_abatement,
    lambda_param=1.2,
    gamma_param=0.8,
    return_breakdown=True
)
global_risk_at_target = risk_model_global.total_risk(global_portfolio_at_target.weights, lambda_param=1.2, gamma_param=0.8)

print("="*80)
print("THREE-TIER COMPARISON AT SAME TARGET (48.9 Mt CO2)")
print("="*80)
print(f"\nTarget Abatement: {announced_abatement:.1f} Mt CO2")
print(f"\n  Global Optimal Risk:      {global_risk_at_target:.1f}")
print(f"  POSCO Optimal Risk:       {optimal_risk:.1f}")
print(f"  POSCO Announced Risk:     {announced_risk:.1f}")
print(f"\nGaps:")
print(f"  POSCO vs Global:          {optimal_risk - global_risk_at_target:.1f} (cost of POSCO constraints)")
print(f"  Announced vs POSCO:       {announced_risk - optimal_risk:.1f} (POSCO sub-optimality)")
print(f"  Announced vs Global:      {announced_risk - global_risk_at_target:.1f} (total gap)")

# ============================================================================
# OPTION A UPDATED: ABSOLUTE RISK WITH EMPHASIS ON SAME TARGET
# ============================================================================

fig, ax = plt.subplots(1, 1, figsize=(12, 7))

# Shift to make all positive
shift = -min(risk_global) + 100  # Add buffer of 100
risk_global_shifted = [r + shift for r in risk_global]
risk_posco_shifted = [r + shift for r in risk_posco]
announced_risk_shifted = announced_risk + shift
optimal_risk_shifted = optimal_risk + shift
global_risk_at_target_shifted = global_risk_at_target + shift

# Plot frontiers
ax.plot(abatement_global, risk_global_shifted, 'o-', color='#2A9D8F', linewidth=2.5,
        markersize=6, label='Global Industry Frontier (Theoretical Best)', alpha=0.9, zorder=3)
ax.plot(abatement_posco, risk_posco_shifted, 's-', color='#E76F51', linewidth=2.5,
        markersize=6, label='POSCO-Feasible Frontier (POSCO-Optimal)', alpha=0.9, zorder=3)

# Mark the THREE key points at same target
ax.plot(announced_abatement, global_risk_at_target_shifted, 'D', color='#2A9D8F',
        markersize=14, label='Global Optimal at Target', zorder=10,
        markeredgecolor='black', markeredgewidth=2)
ax.plot(announced_abatement, optimal_risk_shifted, 'D', color='#E76F51',
        markersize=14, label='POSCO Optimal at Target', zorder=10,
        markeredgecolor='black', markeredgewidth=2)
ax.plot(announced_abatement, announced_risk_shifted, '*', color='#C1121F',
        markersize=22, label='POSCO Announced 2030 Strategy', zorder=10,
        markeredgecolor='black', markeredgewidth=1)

# Vertical line at target
ax.axvline(x=announced_abatement, color='gray', linestyle='--', linewidth=1.5, alpha=0.5, zorder=1)
ax.text(announced_abatement, ax.get_ylim()[1] * 0.95, f'Target: {announced_abatement:.1f} Mt',
        ha='center', va='top', fontsize=11, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='gray', linewidth=1))

# Gap 1: POSCO Constraint Cost (Global → POSCO Optimal)
ax.annotate('', xy=(announced_abatement + 0.3, optimal_risk_shifted),
            xytext=(announced_abatement + 0.3, global_risk_at_target_shifted),
            arrowprops=dict(arrowstyle='<->', color='orange', lw=3))
ax.text(announced_abatement + 2, (optimal_risk_shifted + global_risk_at_target_shifted) / 2,
        f'Constraint Cost\n{optimal_risk - global_risk_at_target:.1f} units',
        fontsize=10, color='orange', ha='left', va='center', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF4E5', edgecolor='orange', linewidth=2))

# Gap 2: POSCO Sub-optimality (POSCO Optimal → Announced)
ax.annotate('', xy=(announced_abatement - 0.3, announced_risk_shifted),
            xytext=(announced_abatement - 0.3, optimal_risk_shifted),
            arrowprops=dict(arrowstyle='<->', color='red', lw=3))
ax.text(announced_abatement - 2, (announced_risk_shifted + optimal_risk_shifted) / 2,
        f'Sub-optimality Gap\n{announced_risk - optimal_risk:.1f} units\n(26% above optimal)',
        fontsize=10, color='red', ha='right', va='center', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFE5E5', edgecolor='red', linewidth=2))

# Shaded region (cost of POSCO constraints)
# Find matching abatement points for shading
matching_abatements = []
for ap in abatement_posco:
    if ap <= max(abatement_global):
        matching_abatements.append(ap)

# Get corresponding global risks
global_risks_for_shading = []
for ap in matching_abatements:
    # Find closest global point
    idx = min(range(len(abatement_global)), key=lambda i: abs(abatement_global[i] - ap))
    global_risks_for_shading.append(risk_global_shifted[idx])

posco_risks_for_shading = [risk_posco_shifted[abatement_posco.index(ap)] for ap in matching_abatements]

ax.fill_between(matching_abatements, global_risks_for_shading, posco_risks_for_shading,
                alpha=0.15, color='orange', zorder=2,
                label='Cost of POSCO Constraints')

ax.set_xlabel('Total Abatement (Mt CO₂)', fontsize=13, fontweight='bold')
ax.set_ylabel('Adjusted Portfolio Transition Risk', fontsize=13, fontweight='bold')
ax.set_title('Three-Tier Efficient Frontier Comparison at Same Abatement Target\n' +
             f'Global vs POSCO vs Announced Strategy (Target: {announced_abatement:.1f} Mt CO₂)',
             fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper left', fontsize=10, framealpha=0.95, ncol=1)
ax.grid(alpha=0.3, linestyle='--')
ax.set_xlim([5, 105])

# Add summary box
summary_text = f"""At Target {announced_abatement:.1f} Mt CO₂:
• Global Optimal: {global_risk_at_target_shifted:.0f}
• POSCO Optimal: {optimal_risk_shifted:.0f}
• POSCO Announced: {announced_risk_shifted:.0f}

Total Gap: {announced_risk - global_risk_at_target:.1f} units
  = {optimal_risk - global_risk_at_target:.1f} (constraints)
  + {announced_risk - optimal_risk:.1f} (sub-optimality)"""

ax.text(0.98, 0.30, summary_text,
        transform=ax.transAxes, fontsize=9.5, ha='right', va='top', family='monospace',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='#F0F8FF', edgecolor='#2A9D8F', linewidth=2))

# Add note explaining adjustment
ax.text(0.02, 0.02, f'Note: Risk values adjusted by +{shift:.0f} for visualization\n(Global range: [{min(risk_global):.0f}, {max(risk_global):.0f}] → POSCO range: [{min(risk_posco):.0f}, {max(risk_posco):.0f}])',
        transform=ax.transAxes, fontsize=8, ha='left', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='gray', linewidth=0.5, alpha=0.8))

plt.tight_layout()
plt.savefig('paper/figure_three_tier_option_a_updated.pdf', dpi=300, bbox_inches='tight')
plt.savefig('paper/figure_three_tier_option_a_updated.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: paper/figure_three_tier_option_a_updated.pdf/png")

print("\n" + "="*80)
print("OPTION A UPDATED GENERATED SUCCESSFULLY")
print("="*80)
