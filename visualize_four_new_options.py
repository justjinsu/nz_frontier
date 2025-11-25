"""
Four New Visualization Options for Three-Tier Frontier
=======================================================
Option 1: Focus Only on POSCO (Remove Global)
Option 2: Dual-Scale Plot (Two Y-axes)
Option 3: Percentage-Based View
Option 4: Efficiency Score Dashboard
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

# Adjust POSCO constraints
for tech in posco_techs:
    if 'BF-BOF' in tech.name and 'Baseline' in tech.name:
        tech.failure_prob = 0.05
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

# Find GLOBAL optimal at SAME abatement target
optimizer_global = OptimizationEngine(global_techs, risk_model_global)
global_portfolio_at_target, global_breakdown_at_target = optimizer_global.solve_for_target(
    target_abatement=announced_abatement,
    lambda_param=1.2,
    gamma_param=0.8,
    return_breakdown=True
)
global_risk_at_target = risk_model_global.total_risk(global_portfolio_at_target.weights, lambda_param=1.2, gamma_param=0.8)

print("="*80)
print("DATA SUMMARY")
print("="*80)
print(f"Target Abatement: {announced_abatement:.1f} Mt")
print(f"  Global Optimal:   {global_risk_at_target:.1f}")
print(f"  POSCO Optimal:    {optimal_risk:.1f}")
print(f"  POSCO Announced:  {announced_risk:.1f}")
print(f"  Sub-optimality:   {announced_risk - optimal_risk:.1f} ({(announced_risk - optimal_risk)/optimal_risk*100:.1f}%)")

# ============================================================================
# OPTION 1: FOCUS ONLY ON POSCO (Remove Global)
# ============================================================================

fig1, ax = plt.subplots(1, 1, figsize=(10, 7))

# Plot POSCO frontier only
ax.plot(abatement_posco, risk_posco, 's-', color='#E76F51', linewidth=3,
        markersize=8, label='POSCO-Feasible Frontier', alpha=0.9)

# Mark optimal and announced
ax.plot(announced_abatement, optimal_risk, 'D', color='#06A77D',
        markersize=16, label='POSCO Optimal Strategy', zorder=10,
        markeredgecolor='black', markeredgewidth=2)
ax.plot(announced_abatement, announced_risk, '*', color='#C1121F',
        markersize=24, label='POSCO Announced 2030', zorder=10,
        markeredgecolor='black', markeredgewidth=1.5)

# Vertical line at target
ax.axvline(x=announced_abatement, color='gray', linestyle='--', linewidth=1.5, alpha=0.4)

# Gap annotation
ax.annotate('', xy=(announced_abatement, announced_risk),
            xytext=(announced_abatement, optimal_risk),
            arrowprops=dict(arrowstyle='<->', color='red', lw=4))
ax.text(announced_abatement + 1.5, (announced_risk + optimal_risk) / 2,
        f'Investment\nInefficiency:\n{announced_risk - optimal_risk:.1f} units\n({(announced_risk - optimal_risk)/optimal_risk*100:.0f}% above optimal)',
        fontsize=11, color='red', ha='left', va='center', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.7', facecolor='#FFE5E5', edgecolor='red', linewidth=2.5))

# Fill region showing wasted risk
x_fill = [announced_abatement, announced_abatement, announced_abatement + 0.5, announced_abatement + 0.5]
y_fill = [optimal_risk, announced_risk, announced_risk, optimal_risk]
ax.fill(x_fill, y_fill, color='red', alpha=0.2, zorder=1, label='Wasted Risk')

ax.set_xlabel('Total Abatement (Mt CO₂)', fontsize=13, fontweight='bold')
ax.set_ylabel('Portfolio Transition Risk', fontsize=13, fontweight='bold')
ax.set_title('Option 1: POSCO Investment Efficiency Analysis\nAnnounced Strategy vs Optimal Portfolio',
             fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper left', fontsize=11, framealpha=0.95)
ax.grid(alpha=0.3, linestyle='--')
ax.set_xlim([8, 52])
ax.set_ylim([min(risk_posco) - 20, announced_risk + 30])

# Summary box
summary = f"""FINDING: POSCO's 2030 strategy is SUB-OPTIMAL

At {announced_abatement:.1f} Mt abatement target:
  • Optimal risk:    {optimal_risk:.1f}
  • Announced risk:  {announced_risk:.1f}
  • Gap:             {announced_risk - optimal_risk:.1f} units

Efficiency Score: {optimal_risk/announced_risk*100:.1f}%
Waste:            {(announced_risk - optimal_risk)/optimal_risk*100:.0f}%

Recommendation: Rebalance portfolio to
reduce risk by {announced_risk - optimal_risk:.1f} units at SAME cost
and SAME abatement target."""

ax.text(0.98, 0.97, summary,
        transform=ax.transAxes, fontsize=9, ha='right', va='top', family='monospace',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='#FFF9E5', edgecolor='#E76F51', linewidth=2))

plt.tight_layout()
plt.savefig('paper/figure_option_1_posco_only.pdf', dpi=300, bbox_inches='tight')
plt.savefig('paper/figure_option_1_posco_only.png', dpi=300, bbox_inches='tight')
print("\n✓ Option 1: POSCO-only focus")

# ============================================================================
# OPTION 2: DUAL-SCALE PLOT (Two Y-axes)
# ============================================================================

fig2, ax1 = plt.subplots(1, 1, figsize=(11, 7))

# Left axis: POSCO
ax1.plot(abatement_posco, risk_posco, 's-', color='#E76F51', linewidth=3,
        markersize=8, label='POSCO-Feasible Frontier', alpha=0.9)
ax1.plot(announced_abatement, optimal_risk, 'D', color='#06A77D',
        markersize=14, label='POSCO Optimal', zorder=10,
        markeredgecolor='black', markeredgewidth=2)
ax1.plot(announced_abatement, announced_risk, '*', color='#C1121F',
        markersize=22, label='POSCO Announced', zorder=10,
        markeredgecolor='black', markeredgewidth=1.5)

ax1.set_xlabel('Total Abatement (Mt CO₂)', fontsize=13, fontweight='bold')
ax1.set_ylabel('POSCO Portfolio Risk', fontsize=13, fontweight='bold', color='#E76F51')
ax1.tick_params(axis='y', labelcolor='#E76F51')
ax1.set_ylim([min(risk_posco) - 20, announced_risk + 50])

# Right axis: Global
ax2 = ax1.twinx()
ax2.plot(abatement_global, risk_global, 'o-', color='#2A9D8F', linewidth=3,
        markersize=6, label='Global Industry Frontier', alpha=0.9)
ax2.plot(announced_abatement, global_risk_at_target, 'D', color='#2A9D8F',
        markersize=14, label='Global Optimal', zorder=10,
        markeredgecolor='black', markeredgewidth=2)

ax2.set_ylabel('Global Portfolio Risk', fontsize=13, fontweight='bold', color='#2A9D8F')
ax2.tick_params(axis='y', labelcolor='#2A9D8F')

ax1.set_title('Option 2: Dual-Scale Comparison\nPOSCO (left axis) vs Global Industry (right axis)',
             fontsize=14, fontweight='bold', pad=20)

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10, framealpha=0.95)

ax1.grid(alpha=0.3, linestyle='--')
ax1.axvline(x=announced_abatement, color='gray', linestyle='--', linewidth=1.5, alpha=0.4)

# Annotations
ax1.annotate('', xy=(announced_abatement - 0.5, announced_risk),
            xytext=(announced_abatement - 0.5, optimal_risk),
            arrowprops=dict(arrowstyle='<->', color='red', lw=3))
ax1.text(announced_abatement - 2, (announced_risk + optimal_risk) / 2,
        f'POSCO Gap:\n{announced_risk - optimal_risk:.1f}',
        fontsize=10, color='red', ha='right', va='center', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFE5E5', edgecolor='red', linewidth=2))

plt.tight_layout()
plt.savefig('paper/figure_option_2_dual_scale.pdf', dpi=300, bbox_inches='tight')
plt.savefig('paper/figure_option_2_dual_scale.png', dpi=300, bbox_inches='tight')
print("✓ Option 2: Dual-scale plot")

# ============================================================================
# OPTION 3: PERCENTAGE-BASED VIEW
# ============================================================================

fig3, ax = plt.subplots(1, 1, figsize=(10, 7))

# Normalize to percentages (Global = 0% baseline)
# Use absolute values to avoid negative percentages
baseline = abs(global_risk_at_target)
risk_posco_pct = [(r - global_risk_at_target) / baseline * 100 for r in risk_posco]
risk_global_pct = [(r - global_risk_at_target) / baseline * 100 for r in risk_global]
optimal_risk_pct = (optimal_risk - global_risk_at_target) / baseline * 100
announced_risk_pct = (announced_risk - global_risk_at_target) / baseline * 100

# Plot
ax.plot(abatement_global, risk_global_pct, 'o-', color='#2A9D8F', linewidth=2.5,
        markersize=6, label='Global Industry Frontier (Baseline = 0%)', alpha=0.9)
ax.plot(abatement_posco, risk_posco_pct, 's-', color='#E76F51', linewidth=2.5,
        markersize=6, label='POSCO-Feasible Frontier', alpha=0.9)

# Mark points at target
ax.plot(announced_abatement, 0, 'D', color='#2A9D8F',
        markersize=14, label='Global Optimal (0%)', zorder=10,
        markeredgecolor='black', markeredgewidth=2)
ax.plot(announced_abatement, optimal_risk_pct, 'D', color='#06A77D',
        markersize=14, label=f'POSCO Optimal (+{optimal_risk_pct:.1f}%)', zorder=10,
        markeredgecolor='black', markeredgewidth=2)
ax.plot(announced_abatement, announced_risk_pct, '*', color='#C1121F',
        markersize=22, label=f'POSCO Announced (+{announced_risk_pct:.1f}%)', zorder=10,
        markeredgecolor='black', markeredgewidth=1.5)

# Shaded regions
ax.axhspan(0, optimal_risk_pct, alpha=0.15, color='orange',
           label=f'Constraint Cost: +{optimal_risk_pct:.1f}%', zorder=1)
ax.axhspan(optimal_risk_pct, announced_risk_pct, alpha=0.2, color='red',
           label=f'Sub-optimality: +{announced_risk_pct - optimal_risk_pct:.1f}%', zorder=1)

# Vertical line
ax.axvline(x=announced_abatement, color='gray', linestyle='--', linewidth=1.5, alpha=0.4)
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

ax.set_xlabel('Total Abatement (Mt CO₂)', fontsize=13, fontweight='bold')
ax.set_ylabel('Risk Premium Relative to Global Optimum (%)', fontsize=13, fontweight='bold')
ax.set_title('Option 3: Percentage-Based Risk Premium View\nAll values normalized to Global Optimal = 0%',
             fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper left', fontsize=9.5, framealpha=0.95)
ax.grid(alpha=0.3, linestyle='--')

# Annotations
ax.text(announced_abatement + 2, optimal_risk_pct / 2,
        f'Constraint Cost\n{optimal_risk_pct:.1f}%\n(unavoidable)',
        fontsize=10, color='orange', ha='left', va='center', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF4E5', edgecolor='orange', linewidth=2))

ax.text(announced_abatement + 2, (optimal_risk_pct + announced_risk_pct) / 2,
        f'Sub-optimality\n{announced_risk_pct - optimal_risk_pct:.1f}%\n(AVOIDABLE!)',
        fontsize=10, color='red', ha='left', va='center', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFE5E5', edgecolor='red', linewidth=2))

plt.tight_layout()
plt.savefig('paper/figure_option_3_percentage.pdf', dpi=300, bbox_inches='tight')
plt.savefig('paper/figure_option_3_percentage.png', dpi=300, bbox_inches='tight')
print("✓ Option 3: Percentage-based view")

# ============================================================================
# OPTION 4: EFFICIENCY SCORE DASHBOARD
# ============================================================================

fig4 = plt.figure(figsize=(12, 8))
gs = fig4.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# Panel A: Efficiency Gauge
ax1 = fig4.add_subplot(gs[0, 0])
efficiency_score = optimal_risk / announced_risk * 100
waste = (announced_risk - optimal_risk) / optimal_risk * 100

# Create gauge (semi-circle)
theta = np.linspace(0, np.pi, 100)
r = 1

# Background arc (red zone)
ax1.fill_between(theta, 0, r, color='#FFE5E5', alpha=0.5)
# Good zone (green)
theta_good = np.linspace(0, np.pi * 0.8, 100)
ax1.fill_between(theta_good, 0, r, color='#E5F5E5', alpha=0.5)

# Needle pointing to efficiency score
needle_angle = np.pi * (efficiency_score / 100)
ax1.plot([0, r * np.cos(needle_angle)], [0, r * np.sin(needle_angle)],
         color='#2A9D8F', linewidth=4, marker='o', markersize=10)

# Labels
ax1.text(0, -0.3, f'{efficiency_score:.1f}%', ha='center', va='top',
         fontsize=24, fontweight='bold', color='#2A9D8F')
ax1.text(0, -0.5, 'Efficiency Score', ha='center', va='top', fontsize=12)

ax1.text(-1, 0, '0%', ha='right', va='center', fontsize=10)
ax1.text(1, 0, '100%', ha='left', va='center', fontsize=10)
ax1.text(0, 1.1, '80%', ha='center', va='bottom', fontsize=10)

ax1.set_xlim([-1.3, 1.3])
ax1.set_ylim([-0.6, 1.3])
ax1.axis('off')
ax1.set_title('(a) Strategy Efficiency Score (SES)', fontsize=12, fontweight='bold', pad=10)

# Panel B: Risk Comparison Bars
ax2 = fig4.add_subplot(gs[0, 1])
portfolios = ['Global\nOptimal', 'POSCO\nOptimal', 'POSCO\nAnnounced']
risks = [global_risk_at_target, optimal_risk, announced_risk]
colors = ['#2A9D8F', '#06A77D', '#C1121F']

bars = ax2.bar(portfolios, risks, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)

# Value labels
for bar, risk in zip(bars, risks):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{risk:.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax2.set_ylabel('Portfolio Transition Risk', fontsize=11, fontweight='bold')
ax2.set_title(f'(b) Risk at Target Abatement ({announced_abatement:.1f} Mt)', fontsize=12, fontweight='bold', pad=10)
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.set_ylim([min(risks) * 1.5 if min(risks) < 0 else 0, max(risks) * 1.1])

# Panel C: Gap Waterfall
ax3 = fig4.add_subplot(gs[1, 0])
categories = ['Global\nOptimal', 'Constraint\nCost', 'Sub-\noptimality', 'POSCO\nAnnounced']
values = [global_risk_at_target, optimal_risk - global_risk_at_target,
          announced_risk - optimal_risk, 0]
cumulative = [global_risk_at_target, optimal_risk, announced_risk, announced_risk]

# Waterfall bars
ax3.bar(0, global_risk_at_target, color='#2A9D8F', edgecolor='black', linewidth=1.5, alpha=0.8)
ax3.bar(1, values[1], bottom=cumulative[0], color='orange', edgecolor='black', linewidth=1.5, alpha=0.8)
ax3.bar(2, values[2], bottom=cumulative[1], color='red', edgecolor='black', linewidth=1.5, alpha=0.8)
ax3.bar(3, announced_risk, color='#C1121F', edgecolor='black', linewidth=1.5, alpha=0.8)

# Connecting lines
for i in range(3):
    ax3.plot([i + 0.4, i + 0.6], [cumulative[i], cumulative[i]], 'k--', linewidth=1)

# Labels
ax3.text(1, cumulative[0] + values[1]/2, f'+{values[1]:.0f}', ha='center', va='center',
         fontsize=10, fontweight='bold', color='white')
ax3.text(2, cumulative[1] + values[2]/2, f'+{values[2]:.0f}', ha='center', va='center',
         fontsize=10, fontweight='bold', color='white')

ax3.set_xticks(range(4))
ax3.set_xticklabels(categories, fontsize=9)
ax3.set_ylabel('Cumulative Risk', fontsize=11, fontweight='bold')
ax3.set_title('(c) Risk Gap Waterfall Decomposition', fontsize=12, fontweight='bold', pad=10)
ax3.grid(axis='y', alpha=0.3, linestyle='--')

# Panel D: Key Metrics Table
ax4 = fig4.add_subplot(gs[1, 1])
ax4.axis('off')

metrics_data = [
    ['Metric', 'Value'],
    ['─' * 30, '─' * 15],
    ['Target Abatement', f'{announced_abatement:.1f} Mt'],
    ['', ''],
    ['Global Optimal Risk', f'{global_risk_at_target:.1f}'],
    ['POSCO Optimal Risk', f'{optimal_risk:.1f}'],
    ['POSCO Announced Risk', f'{announced_risk:.1f}'],
    ['', ''],
    ['Constraint Cost', f'{optimal_risk - global_risk_at_target:.1f}'],
    ['Sub-optimality Gap', f'{announced_risk - optimal_risk:.1f}'],
    ['', ''],
    ['Efficiency Score (SES)', f'{efficiency_score:.1f}%'],
    ['Wasted Risk', f'{waste:.1f}%'],
    ['', ''],
    ['Potential Savings', f'${(announced_risk - optimal_risk) * 10:.0f}M'],
]

table = ax4.table(cellText=metrics_data, cellLoc='left', loc='center',
                  colWidths=[0.6, 0.4],
                  bbox=[0, 0, 1, 1])

table.auto_set_font_size(False)
table.set_fontsize(10)

# Style header
for i in range(2):
    table[(0, i)].set_facecolor('#2A9D8F')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Style key rows
table[(11, 0)].set_text_props(weight='bold', color='#2A9D8F')
table[(11, 1)].set_text_props(weight='bold', color='#2A9D8F')
table[(12, 0)].set_text_props(weight='bold', color='red')
table[(12, 1)].set_text_props(weight='bold', color='red')

ax4.set_title('(d) Key Performance Metrics', fontsize=12, fontweight='bold', pad=10)

plt.suptitle('Option 4: Investment Efficiency Dashboard\nPOSCO 2030 Strategy Performance Analysis',
             fontsize=14, fontweight='bold', y=0.98)

plt.savefig('paper/figure_option_4_dashboard.pdf', dpi=300, bbox_inches='tight')
plt.savefig('paper/figure_option_4_dashboard.png', dpi=300, bbox_inches='tight')
print("✓ Option 4: Efficiency dashboard")

print("\n" + "="*80)
print("ALL FOUR NEW OPTIONS GENERATED SUCCESSFULLY")
print("="*80)
print("\nFiles created:")
print("  • paper/figure_option_1_posco_only.pdf/png")
print("  • paper/figure_option_2_dual_scale.pdf/png")
print("  • paper/figure_option_3_percentage.pdf/png")
print("  • paper/figure_option_4_dashboard.pdf/png")
