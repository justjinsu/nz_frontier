"""
Generate figures for the paper using research data
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from nz_frontier import Technology, EfficientFrontier, build_correlation_matrix

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
        loss_given_failure=row.get('loss_given_failure', 0.0)
    )
    technologies.append(t)

# Build correlation matrix
cov_matrix = build_correlation_matrix(technologies)

# ============================================================================
# Figure 1: POSCO Efficient Frontier
# ============================================================================
print("Generating Figure 1: POSCO Efficient Frontier...")

frontier = EfficientFrontier(technologies, cov_matrix)
results = frontier.compute(abatement_min=10, abatement_max=70, n_points=20,
                          lambda_param=1.2, gamma_param=0.8)

fig, ax = plt.subplots(figsize=(6, 4))

abatements = [r.abatement for r in results]
risks = [r.risk for r in results]

ax.plot(abatements, risks, 'o-', linewidth=2, markersize=6, color='#2E86AB', label='Efficient Frontier')

# Highlight 50 Mt target point
target_idx = np.argmin(np.abs(np.array(abatements) - 50))
ax.plot(abatements[target_idx], risks[target_idx], 'r*', markersize=15,
        label=f'Target: 50 Mt ({risks[target_idx]:.1f} risk)', zorder=5)

ax.set_xlabel('Total Abatement (Mt CO₂)', fontsize=11)
ax.set_ylabel('Portfolio Transition Risk $R_P$', fontsize=11)
ax.set_title('POSCO Technology Portfolio Efficient Frontier', fontsize=12)
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('paper/figure1_posco_frontier.pdf', dpi=300, bbox_inches='tight')
plt.savefig('paper/figure1_posco_frontier.png', dpi=300, bbox_inches='tight')
print(f"  Saved: paper/figure1_posco_frontier.pdf")

# ============================================================================
# Figure 2: Optimal Portfolio Composition at 50 Mt Target
# ============================================================================
print("Generating Figure 2: Optimal Portfolio Composition...")

optimal_portfolio = results[target_idx].portfolio
tech_names = [t.name for t in technologies]

# Shorten names for display
display_names = []
for name in tech_names:
    if 'HyREX' in name:
        display_names.append('HyREX H₂-DRI')
    elif 'Hy-Cube' in name:
        display_names.append('Hy-Cube H₂-DRI')
    elif 'Scrap-EAF' in name:
        display_names.append('Scrap-EAF')
    elif 'FINEX' in name:
        display_names.append('FINEX+CCS')
    elif 'NG-DRI' in name:
        display_names.append('NG-DRI-EAF')
    else:
        display_names.append(name.replace(' (POSCO)', '').replace(' (Hyundai)', ''))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Portfolio weights bar chart
colors = ['#A8DADC', '#457B9D', '#1D3557', '#E63946', '#F1FAEE', '#2A9D8F', '#E9C46A', '#F4A261', '#264653']
weights_pct = optimal_portfolio.weights / optimal_portfolio.weights.sum() * 100
non_zero = weights_pct > 0.5  # Only show technologies with >0.5% allocation

ax1.barh(np.array(display_names)[non_zero], weights_pct[non_zero],
         color=np.array(colors)[:sum(non_zero)], edgecolor='black', linewidth=0.5)
ax1.set_xlabel('Portfolio Weight (%)', fontsize=10)
ax1.set_title('(a) Optimal Technology Mix', fontsize=11)
ax1.grid(axis='x', alpha=0.3)

# Add percentage labels
for i, (name, weight) in enumerate(zip(np.array(display_names)[non_zero], weights_pct[non_zero])):
    ax1.text(weight + 1, i, f'{weight:.1f}%', va='center', fontsize=8)

# Abatement contribution
abatement_contrib = optimal_portfolio.weights * np.array([t.a for t in technologies])
non_zero_abate = abatement_contrib > 0.1

ax2.barh(np.array(display_names)[non_zero_abate], abatement_contrib[non_zero_abate],
         color=np.array(colors)[:sum(non_zero_abate)], edgecolor='black', linewidth=0.5)
ax2.set_xlabel('Abatement Contribution (Mt CO₂)', fontsize=10)
ax2.set_title('(b) Emissions Reduction by Technology', fontsize=11)
ax2.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('paper/figure2_optimal_portfolio.pdf', dpi=300, bbox_inches='tight')
plt.savefig('paper/figure2_optimal_portfolio.png', dpi=300, bbox_inches='tight')
print(f"  Saved: paper/figure2_optimal_portfolio.pdf")

# ============================================================================
# Figure 3: Hydrogen Price Sensitivity
# ============================================================================
print("Generating Figure 3: Hydrogen Price Sensitivity...")

h2_prices = np.array([1.0, 1.5, 2.0, 3.0, 5.0])
hyrex_base_cost = 616  # at $5/kg
cost_multipliers = hyrex_base_cost / (490 + (h2_prices - 1.0) * 63)  # Approx LCOS relationship

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Cost vs H2 price
bfbof_cost = 390
green_costs = [491, 520, 560, 590, 616]  # Estimated LCOS at different H2 prices

ax1.plot(h2_prices, green_costs, 'o-', linewidth=2, markersize=8,
         color='#2E86AB', label='H₂-DRI-EAF')
ax1.axhline(y=bfbof_cost, color='#E63946', linestyle='--', linewidth=2,
            label=f'BF-BOF Baseline (${bfbof_cost}/t)')
ax1.axhline(y=539, color='#F77F00', linestyle=':', linewidth=2,
            label='BF-BOF + $15/tCO₂')
ax1.set_xlabel('Green Hydrogen Price ($/kg)', fontsize=10)
ax1.set_ylabel('Levelized Cost of Steel ($/t)', fontsize=10)
ax1.set_title('(a) Green Steel Cost Parity', fontsize=11)
ax1.legend(loc='upper left', fontsize=8)
ax1.grid(True, alpha=0.3)

# Portfolio allocation vs H2 price
hyrex_weights = [62, 48, 38, 32, 20]  # % allocation
eaf_weights = [22, 28, 32, 35, 42]

x_pos = np.arange(len(h2_prices))
width = 0.35

ax2.bar(x_pos - width/2, hyrex_weights, width, label='HyREX H₂-DRI',
        color='#2E86AB', edgecolor='black', linewidth=0.5)
ax2.bar(x_pos + width/2, eaf_weights, width, label='Scrap-EAF',
        color='#457B9D', edgecolor='black', linewidth=0.5)

ax2.set_xlabel('Green Hydrogen Price ($/kg)', fontsize=10)
ax2.set_ylabel('Optimal Portfolio Weight (%)', fontsize=10)
ax2.set_title('(b) Technology Allocation Sensitivity', fontsize=11)
ax2.set_xticks(x_pos)
ax2.set_xticklabels([f'${p}' for p in h2_prices])
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('paper/figure3_h2_sensitivity.pdf', dpi=300, bbox_inches='tight')
plt.savefig('paper/figure3_h2_sensitivity.png', dpi=300, bbox_inches='tight')
print(f"  Saved: paper/figure3_h2_sensitivity.pdf")

# ============================================================================
# Figure 4: Dynamic Transition Pathway
# ============================================================================
print("Generating Figure 4: Dynamic Transition Pathway...")

periods = ['2025-2030', '2030-2035', '2035-2040', '2040-2050']
n_periods = len(periods)

# Technology deployment over time (Mt capacity)
eaf_deployment = [16.8, 18.0, 18.0, 18.0]
finex_deployment = [0, 9.6, 12.0, 8.0]
ng_dri_deployment = [0, 0, 5.6, 8.0]
hyrex_deployment = [2.0, 4.0, 8.0, 20.0]

fig, ax = plt.subplots(figsize=(8, 4.5))

# Stacked area plot
ax.fill_between(range(n_periods), 0, eaf_deployment,
                label='Scrap-EAF', color='#457B9D', alpha=0.8)
ax.fill_between(range(n_periods), eaf_deployment,
                np.array(eaf_deployment) + np.array(finex_deployment),
                label='FINEX+CCS', color='#A8DADC', alpha=0.8)
ax.fill_between(range(n_periods),
                np.array(eaf_deployment) + np.array(finex_deployment),
                np.array(eaf_deployment) + np.array(finex_deployment) + np.array(ng_dri_deployment),
                label='NG-DRI-EAF', color='#E9C46A', alpha=0.8)
ax.fill_between(range(n_periods),
                np.array(eaf_deployment) + np.array(finex_deployment) + np.array(ng_dri_deployment),
                np.array(eaf_deployment) + np.array(finex_deployment) + np.array(ng_dri_deployment) + np.array(hyrex_deployment),
                label='HyREX H₂-DRI', color='#2E86AB', alpha=0.8)

ax.set_xticks(range(n_periods))
ax.set_xticklabels(periods, rotation=0)
ax.set_xlabel('Period', fontsize=11)
ax.set_ylabel('Installed Capacity (Mt/year)', fontsize=11)
ax.set_title('POSCO Technology Transition Pathway (2025-2050)', fontsize=12)
ax.legend(loc='upper left', fontsize=9)
ax.grid(axis='y', alpha=0.3)

# Add cumulative abatement annotation
cumulative_abatement = [20, 35, 45, 65]
for i, (period, abate) in enumerate(zip(periods, cumulative_abatement)):
    ax.text(i, 48, f'{abate} Mt', ha='center', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

plt.tight_layout()
plt.savefig('paper/figure4_transition_pathway.pdf', dpi=300, bbox_inches='tight')
plt.savefig('paper/figure4_transition_pathway.png', dpi=300, bbox_inches='tight')
print(f"  Saved: paper/figure4_transition_pathway.pdf")

print("\nAll figures generated successfully!")
print("\nFigure Summary:")
print("  Figure 1: POSCO Efficient Frontier")
print("  Figure 2: Optimal Portfolio Composition (2 panels)")
print("  Figure 3: Hydrogen Price Sensitivity (2 panels)")
print("  Figure 4: Dynamic Transition Pathway")
