"""
Net-Zero Frontier: Advanced Risk-Efficiency Analysis for Corporate Decarbonization

This application implements the complete theoretical framework from paper_v2.tex:
- Monte Carlo Efficient Frontier with confidence bands
- Full correlation matrix construction
- Dynamic real options valuation
- Stochastic dominance analysis
- Robust optimization under parameter uncertainty

Author: Jinsu Park, PLANiT Institute
"""

import sys
import os

# Add the current directory to path for Streamlit Cloud deployment
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')

from nz_frontier import (
    Technology,
    EfficientFrontier,
    DynamicOptimizer,
    CostSimulator,
    MonteCarloFrontier,
    StochasticDominanceAnalyzer,
    RobustOptimizer,
    DynamicRealOptionsOptimizer,
    build_correlation_matrix,
    RiskModel,
)

st.set_page_config(page_title="Net-Zero Frontier", layout="wide")

st.title("Portfolio Theory for Corporate Decarbonization")
st.markdown("""
### Risk-Efficiency Framework for Net-Zero Investment under Uncertainty

This application implements the complete theoretical framework from **Park (2024)**, extending
Markowitz portfolio theory to corporate decarbonization with:

- **Monte Carlo Efficient Frontier** with 90% confidence bands (Section 9.3)
- **Full Correlation Matrix** capturing technology interdependencies (Remark 1)
- **Dynamic Real Options** re-valued at each simulation step (Section 5)
- **Stochastic Dominance Analysis** for portfolio comparison (Extension)
- **Robust Optimization** under parameter uncertainty (Section 9.1)

**Risk Function:** $R_P(\\mathbf{w}) = \\mathbf{w}^T \\Sigma \\mathbf{w} + \\lambda h(\\mathbf{w}) - \\gamma g(\\mathbf{w})$
""")

# =============================================================================
# Sidebar Configuration
# =============================================================================
st.sidebar.header("Configuration")

# 1. Data Source
data_source = st.sidebar.radio(
    "Data Source",
    ["Preset Case: Korea Steel", "Preset Case: Global Steel",
     "Preset Case: Korea Energy", "Preset Case: Steel",
     "Preset Case: Petrochemical", "Upload CSV"]
)

df = None
if data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload Technology Data (CSV)", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
elif data_source == "Preset Case: Steel":
    df = pd.read_csv("data/steel.csv")
elif data_source == "Preset Case: Petrochemical":
    df = pd.read_csv("data/petrochemical.csv")
elif data_source == "Preset Case: Korea Steel":
    df = pd.read_csv("data/korea_steel.csv")
elif data_source == "Preset Case: Global Steel":
    df = pd.read_csv("data/global_steel.csv")
elif data_source == "Preset Case: Korea Energy":
    df = pd.read_csv("data/korea_energy.csv")

# 2. Model Parameters
st.sidebar.subheader("Model Parameters")
abatement_target_min = st.sidebar.number_input("Min Abatement Target", value=10.0, step=5.0)
abatement_target_max = st.sidebar.number_input("Max Abatement Target", value=100.0, step=10.0)
budget_constraint = st.sidebar.number_input("Budget Constraint ($M)", value=2000.0, step=100.0)

# 3. Risk Preference Parameters (from Equation 4)
st.sidebar.subheader("Risk Preferences (Eq. 4)")
lambda_param = st.sidebar.slider(
    "λ (Stranded Asset Risk Weight)",
    min_value=0.0, max_value=3.0, value=1.0, step=0.1,
    help="Weight on stranded asset risk h(w) in the risk function"
)
gamma_param = st.sidebar.slider(
    "γ (Option Value Weight)",
    min_value=0.0, max_value=3.0, value=1.0, step=0.1,
    help="Weight on option value g(w) in the risk function"
)

# 4. Advanced Settings
with st.sidebar.expander("Advanced Settings"):
    risk_free_rate = st.number_input("Risk-Free Rate", value=0.05, step=0.01)
    time_horizon = st.number_input("Time Horizon (Years)", value=20, step=1)
    n_mc_simulations = st.number_input("Monte Carlo Simulations", value=200, step=50,
                                        help="More simulations = smoother confidence bands but slower")
    n_frontier_points = st.number_input("Frontier Points", value=15, step=5)

    st.markdown("**Uncertainty Parameters (Robust Optimization)**")
    cost_uncertainty = st.slider("Cost Uncertainty (±%)", 0, 50, 20) / 100
    vol_uncertainty = st.slider("Volatility Uncertainty (±%)", 0, 50, 30) / 100


def load_technologies(df: pd.DataFrame) -> list:
    """Load technologies from DataFrame."""
    techs = []
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
        techs.append(t)
    return techs


def plot_monte_carlo_frontier(mc_result, title="Monte Carlo Efficient Frontier"):
    """Plot frontier with confidence bands."""
    fig, ax = plt.subplots(figsize=(10, 6))

    targets = mc_result.abatement_targets
    mean_risk = mc_result.mean_risk
    p5 = mc_result.percentile_5
    p95 = mc_result.percentile_95

    # Confidence band
    ax.fill_between(targets, p5, p95, alpha=0.3, color='steelblue', label='90% Confidence Band')

    # Mean frontier
    ax.plot(targets, mean_risk, 'o-', linewidth=2, markersize=6, color='darkblue', label='Expected Frontier')

    ax.set_xlabel('Abatement Target (tCO₂)', fontsize=12)
    ax.set_ylabel('Portfolio Transition Risk $R_P$', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    return fig


def plot_risk_decomposition(breakdown, tech_names):
    """Plot risk decomposition bar chart."""
    fig, ax = plt.subplots(figsize=(8, 5))

    components = ['Cost Volatility\n$w^T\\Sigma w$',
                  f'Stranded Asset\n$\\lambda h(w)$',
                  f'Option Value\n$-\\gamma g(w)$']
    values = [breakdown.cost_volatility,
              breakdown.stranded_asset,
              -breakdown.option_value]  # Negative because it reduces risk
    colors = ['#E74C3C', '#F39C12', '#27AE60']

    bars = ax.bar(components, values, color=colors, edgecolor='black', linewidth=1.2)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_ylabel('Risk Contribution', fontsize=11)
    ax.set_title(f'Risk Decomposition (Total: {breakdown.total:.2f})', fontsize=12)

    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(f'{val:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height >= 0 else -15),
                    textcoords="offset points",
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=10)

    return fig


def plot_stochastic_dominance(sd_result):
    """Plot risk distribution comparison for stochastic dominance."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram comparison
    ax1 = axes[0]
    ax1.hist(sd_result.risk_distribution_a, bins=30, alpha=0.6,
             label=sd_result.portfolio_a_name, color='steelblue', density=True)
    ax1.hist(sd_result.risk_distribution_b, bins=30, alpha=0.6,
             label=sd_result.portfolio_b_name, color='coral', density=True)
    ax1.set_xlabel('Portfolio Risk', fontsize=11)
    ax1.set_ylabel('Density', fontsize=11)
    ax1.set_title('Risk Distribution Comparison', fontsize=12)
    ax1.legend()

    # CDF comparison
    ax2 = axes[1]
    sorted_a = np.sort(sd_result.risk_distribution_a)
    sorted_b = np.sort(sd_result.risk_distribution_b)
    cdf_a = np.arange(1, len(sorted_a) + 1) / len(sorted_a)
    cdf_b = np.arange(1, len(sorted_b) + 1) / len(sorted_b)

    ax2.plot(sorted_a, cdf_a, label=sd_result.portfolio_a_name, linewidth=2, color='steelblue')
    ax2.plot(sorted_b, cdf_b, label=sd_result.portfolio_b_name, linewidth=2, color='coral')
    ax2.set_xlabel('Portfolio Risk', fontsize=11)
    ax2.set_ylabel('Cumulative Probability', fontsize=11)
    ax2.set_title('CDF Comparison (Stochastic Dominance)', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_dynamic_pathway(portfolios, technologies, cost_history=None):
    """Plot dynamic transition pathway."""
    n_periods = len(portfolios)
    n_tech = len(technologies)
    tech_names = [t.name for t in technologies]

    weights_history = np.array([p.weights for p in portfolios])

    fig, axes = plt.subplots(1, 2 if cost_history else 1, figsize=(14 if cost_history else 10, 6))

    if cost_history:
        ax1, ax2 = axes
    else:
        ax1 = axes

    # Stacked area for capacity
    ax1.stackplot(range(n_periods), weights_history.T, labels=tech_names, alpha=0.8)
    ax1.set_xlabel('Period', fontsize=11)
    ax1.set_ylabel('Installed Capacity', fontsize=11)
    ax1.set_title('Technology Transition Pathway', fontsize=12)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Cost evolution
    if cost_history:
        for tech_name, costs in cost_history.items():
            ax2.plot(range(len(costs)), costs, label=tech_name, linewidth=2)
        ax2.set_xlabel('Period', fontsize=11)
        ax2.set_ylabel('Technology Cost ($/unit)', fontsize=11)
        ax2.set_title('Cost Evolution with Learning & Jumps', fontsize=12)
        ax2.legend(loc='upper right', fontsize=9)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# =============================================================================
# Main Application
# =============================================================================

if df is not None:
    st.write(f"### Loaded Technologies ({data_source})")
    st.dataframe(df, use_container_width=True)

    try:
        technologies = load_technologies(df)
        n_tech = len(technologies)

        # Build FULL correlation matrix (not just diagonal!)
        st.info("Building full correlation matrix from technology characteristics...")
        cov_matrix = build_correlation_matrix(technologies)

        # Show correlation matrix
        with st.expander("View Correlation Matrix"):
            sigmas = np.array([t.sigma for t in technologies])
            with np.errstate(divide='ignore', invalid='ignore'):
                corr_matrix = cov_matrix / np.outer(sigmas, sigmas)
                corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
                np.fill_diagonal(corr_matrix, 1.0)

            corr_df = pd.DataFrame(
                corr_matrix,
                index=[t.name for t in technologies],
                columns=[t.name for t in technologies]
            )
            st.dataframe(corr_df.style.background_gradient(cmap='RdYlGn', vmin=-1, vmax=1).format("{:.2f}"))

        # =============================================================================
        # Tab Layout
        # =============================================================================
        tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Case Study",
            "Monte Carlo Frontier",
            "Risk Decomposition",
            "Dynamic Transition",
            "Stochastic Dominance",
            "Robust Optimization",
            "Cost Simulation"
        ])

        # ---------------------------------------------------------------------
        # Tab 0: Case Study Context
        # ---------------------------------------------------------------------
        with tab0:
            if "Korea Steel" in data_source:
                st.subheader("🇰🇷 Case Study: South Korea's Steel Sector Decarbonization")

                st.markdown("""
                ### Industry Context

                South Korea's steel industry accounts for **15% of national carbon emissions** and **40% of industrial emissions**,
                making it critical for the country's 2050 net-zero commitment. The sector is dominated by two major players:

                | Company | Production | Emissions (2024) | Intensity |
                |---------|------------|------------------|-----------|
                | **POSCO** | ~40 Mt/year | 71.07 Mt CO₂ | 2.02 tCO₂/tcs |
                | **Hyundai Steel** | ~20 Mt/year | ~29 Mt CO₂ | 1.43 tCO₂/tcs |

                ### Key Technologies (2024-2025 Data)
                """)

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("""
                    #### 🔵 POSCO HyREX
                    - **April 2024**: Pilot success at Pohang - 24 tons/day at 0.4 tCO₂/t
                    - **October 2025**: BHP partnership for 300kt demo plant
                    - **Target**: Commercial scale by 2030
                    - **Cost**: $616/ton (at H₂ price $5/kg)
                    """)

                with col2:
                    st.markdown("""
                    #### 🔴 Hyundai Hy-Cube
                    - **Hy-Arc** electric furnace technology
                    - **2025**: $6B Louisiana plant announced
                    - Blue H₂ initially → Green H₂ post-2034
                    - **Cost**: $600/ton estimated
                    """)

                st.markdown("""
                ### Investment Gap Analysis

                | Region | Government Funding | Production |
                |--------|-------------------|------------|
                | **Korea** | KRW 268.5B ($198M) | 55 Mt/year |
                | **Germany** | ~$7.5B | 26 Mt/year |
                | **Ratio** | **38× less** per ton | - |

                POSCO estimates **KRW 20 trillion ($14.8B)** needed for HyREX commercialization.

                ### Green Premium Economics
                """)

                # Green premium visualization
                h2_prices = [1.0, 1.5, 2.0, 3.0, 5.0]
                bfbof_cost = 390
                green_costs = [491, 539, 587, 616, 653]  # Approximate LCOS at different H2 prices

                fig, ax = plt.subplots(figsize=(10, 5))
                ax.bar([f"${p}/kg" for p in h2_prices], green_costs, color='green', alpha=0.7, label='H₂-DRI-EAF')
                ax.axhline(y=bfbof_cost, color='red', linestyle='--', linewidth=2, label=f'BF-BOF Baseline (${bfbof_cost}/t)')
                ax.axhline(y=539, color='orange', linestyle=':', linewidth=2, label='BF-BOF + $15/tCO₂ carbon price')
                ax.set_xlabel('Hydrogen Price', fontsize=11)
                ax.set_ylabel('Levelized Cost of Steel ($/ton)', fontsize=11)
                ax.set_title('Green Steel Cost Parity Analysis', fontsize=12)
                ax.legend()
                ax.grid(True, alpha=0.3, axis='y')
                st.pyplot(fig)

                st.markdown("""
                **Key Insight**: At H₂ price $1.5/kg with $15/tCO₂ carbon pricing, green H₂-DRI-EAF becomes
                **cost-competitive** with traditional BF-BOF steelmaking.

                ### Timeline Comparison
                """)

                timeline_data = {
                    'Project': ['HYBRIT (Sweden)', 'H2 Green Steel', 'SALCOS (Germany)', 'HyREX (Korea)', 'Hy-Cube Louisiana'],
                    'Company': ['SSAB/LKAB/Vattenfall', 'Stegra', 'Salzgitter', 'POSCO', 'Hyundai'],
                    'Commercial': ['2026', '2025-2030', '2026', '2030+', '2028+'],
                    'Scale': ['Fossil-free steel', '5 Mt/year', '1 Mt/year', '300kt demo', 'TBD']
                }
                st.dataframe(pd.DataFrame(timeline_data), use_container_width=True)

                st.warning("""
                ⚠️ **Timeline Risk**: Korea's HyREX lags European competitors by **4-6 years**,
                increasing stranded asset risk for existing blast furnaces.
                """)

            elif "Global Steel" in data_source:
                st.subheader("🌍 Case Study: Global Green Steel Race")

                st.markdown("""
                ### The Global Green Steel Landscape (2024-2025)

                The steel industry accounts for **8% of global CO₂ emissions**. Major steelmakers worldwide
                are racing to commercialize low-carbon production technologies.

                ### Technology Pathways

                | Route | Emissions | Cost Premium | TRL |
                |-------|-----------|--------------|-----|
                | **BF-BOF** (baseline) | 2.2 tCO₂/t | - | 9 |
                | **Scrap-EAF** | 0.66 tCO₂/t | ~6% | 9 |
                | **BF-BOF + CCS** | 0.8-1.0 tCO₂/t | ~19% | 7-8 |
                | **NG-DRI-EAF** | 1.0-1.2 tCO₂/t | ~17% | 8 |
                | **H₂-DRI-EAF** | 0.3-0.5 tCO₂/t | ~70%* | 6-7 |
                | **MOE** | ~0 tCO₂/t | TBD | 4-5 |

                *At current H₂ prices ($5/kg); drops to ~20% at $1.5/kg

                ### Leading Projects
                """)

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown("""
                    #### 🇸🇪 Sweden
                    - **HYBRIT**: World's first fossil-free steel (2021 pilot)
                    - **H2 Green Steel**: 5 Mt/year by 2030
                    - Advantage: Cheap hydro + wind
                    """)

                with col2:
                    st.markdown("""
                    #### 🇩🇪 Germany
                    - **SALCOS**: 1 Mt H₂-DRI by 2026
                    - **ThyssenKrupp**: Duisburg H₂ demo
                    - €7.5B government funding
                    """)

                with col3:
                    st.markdown("""
                    #### 🇺🇸 United States
                    - **Nucor**: Leading EAF (0.38 tCO₂/t)
                    - **Hyundai Louisiana**: $6B H₂ plant
                    - IRA incentives: $85/tCO₂ 45Q credit
                    """)

                st.markdown("""
                ### Hydrogen Cost Trajectory
                """)

                years = ['2024', '2025', '2030', '2035', '2050']
                h2_costs = [4.5, 3.5, 2.0, 1.5, 1.0]

                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(years, h2_costs, 'o-', linewidth=2, markersize=10, color='green')
                ax.axhline(y=1.4, color='red', linestyle='--', label='Cost parity threshold')
                ax.fill_between(years, h2_costs, alpha=0.3, color='green')
                ax.set_xlabel('Year', fontsize=11)
                ax.set_ylabel('Green Hydrogen Cost ($/kg)', fontsize=11)
                ax.set_title('Projected Green Hydrogen Cost Decline', fontsize=12)
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)

            elif "Korea Energy" in data_source:
                st.subheader("🇰🇷 Case Study: South Korea's Energy Transition")

                st.markdown("""
                ### Power Sector Context

                South Korea's power sector faces unique challenges:
                - **Renewables**: Only 10.5% of electricity (2024) vs OECD avg ~30%
                - **Nuclear**: 30% of generation (APR1400 fleet)
                - **CCS Dependence**: BloombergNEF projects 41% of abatement via CCS by 2050

                ### 11th Basic Plan for Long-Term Electricity Supply (Feb 2025)
                - Target: **121.9 GW** renewable capacity by 2038
                - Nuclear expansion approved
                - RE100 compliance challenges: Korean members source only 12% renewables vs 53% globally

                ### Technology Comparison

                | Technology | LCOE ($/MWh) | Emissions | Dispatchability |
                |------------|--------------|-----------|-----------------|
                | Coal | 45 | High | Baseload |
                | LNG CCGT | 65 | Medium | Flexible |
                | Nuclear | 80 | Very Low | Baseload |
                | Solar PV | 42 | Zero | Variable |
                | Offshore Wind | 85 | Zero | Variable |
                | Green H₂ | 150 | Zero | Storage |
                """)

            else:
                st.subheader("📊 Case Study Overview")

                st.markdown("""
                Select a preset case study from the sidebar to see detailed industry context:

                - **🇰🇷 Korea Steel**: POSCO HyREX, Hyundai Hy-Cube, investment gap analysis
                - **🌍 Global Steel**: HYBRIT, H2 Green Steel, international comparison
                - **🇰🇷 Korea Energy**: Power sector transition, nuclear + renewables mix

                Or upload your own CSV data to analyze custom technology portfolios.

                ### Framework Overview

                This application implements **Portfolio Theory for Corporate Decarbonization**, extending
                Markowitz (1952) portfolio optimization to climate transition investment.

                **Key Innovation**: The risk function captures three components:

                $$R_P(\\mathbf{w}) = \\underbrace{\\mathbf{w}^T \\Sigma \\mathbf{w}}_{\\text{Cost Volatility}}
                + \\lambda \\underbrace{h(\\mathbf{w})}_{\\text{Stranded Asset Risk}}
                - \\gamma \\underbrace{g(\\mathbf{w})}_{\\text{Option Value}}$$

                Use the other tabs to:
                1. **Monte Carlo Frontier**: Generate efficient frontier with confidence bands
                2. **Risk Decomposition**: Analyze risk components
                3. **Dynamic Transition**: Plan multi-period pathways
                4. **Stochastic Dominance**: Compare portfolios
                5. **Robust Optimization**: Account for parameter uncertainty
                6. **Cost Simulation**: Visualize technology cost evolution
                """)

        # ---------------------------------------------------------------------
        # Tab 1: Monte Carlo Efficient Frontier
        # ---------------------------------------------------------------------
        with tab1:
            st.subheader("Monte Carlo Efficient Frontier")
            st.markdown("""
            Computes the efficient frontier via Monte Carlo simulation:
            1. Simulate $N$ cost paths using jump-diffusion with learning (Eq. 22)
            2. Re-optimize portfolio for each simulation
            3. Compute mean frontier with 90% confidence bands
            """)

            col1, col2 = st.columns([1, 1])
            with col1:
                revalue_options = st.checkbox("Re-value options dynamically", value=True,
                                              help="Update option values at each MC step based on simulated costs")

            if st.button("Generate Monte Carlo Frontier", type="primary"):
                with st.spinner(f"Running {n_mc_simulations} Monte Carlo simulations..."):
                    mc_frontier = MonteCarloFrontier(
                        technologies,
                        cov_matrix,
                        risk_free_rate=risk_free_rate
                    )

                    mc_result = mc_frontier.compute_frontier(
                        abatement_min=abatement_target_min,
                        abatement_max=abatement_target_max,
                        budget_constraint=budget_constraint,
                        n_targets=n_frontier_points,
                        n_simulations=n_mc_simulations,
                        time_horizon=time_horizon,
                        lambda_param=lambda_param,
                        gamma_param=gamma_param,
                        revalue_options=revalue_options
                    )

                    # Store in session state
                    st.session_state['mc_result'] = mc_result
                    st.session_state['technologies'] = technologies
                    st.session_state['cov_matrix'] = cov_matrix

                # Plot
                fig = plot_monte_carlo_frontier(mc_result)
                st.pyplot(fig)

                # Statistics
                st.markdown("#### Frontier Statistics")
                stats_df = pd.DataFrame({
                    'Abatement': mc_result.abatement_targets,
                    'Mean Risk': mc_result.mean_risk,
                    'Std Dev': mc_result.std_risk,
                    '5th Percentile': mc_result.percentile_5,
                    '95th Percentile': mc_result.percentile_95
                })
                st.dataframe(stats_df.style.format("{:.2f}"), use_container_width=True)

                # Portfolio composition for selected target
                st.markdown("#### Optimal Portfolio Composition")
                target_select = st.select_slider(
                    "Select Abatement Target",
                    options=mc_result.abatement_targets.tolist(),
                    value=mc_result.abatement_targets[len(mc_result.abatement_targets)//2]
                )

                idx = np.argmin(np.abs(mc_result.abatement_targets - target_select))
                selected_portfolio = mc_result.optimal_portfolios[idx]

                fig2, ax = plt.subplots(figsize=(10, 5))
                colors = plt.cm.Set3(np.linspace(0, 1, n_tech))
                bars = ax.bar([t.name for t in technologies], selected_portfolio.weights, color=colors, edgecolor='black')
                ax.set_ylabel('Capacity (Units)', fontsize=11)
                ax.set_title(f'Optimal Portfolio for Target = {target_select:.1f}', fontsize=12)
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig2)

        # ---------------------------------------------------------------------
        # Tab 2: Risk Decomposition
        # ---------------------------------------------------------------------
        with tab2:
            st.subheader("Risk Decomposition Analysis")
            st.markdown("""
            Decomposes total portfolio risk into three components (Equation 4):

            $$R_P(\\mathbf{w}) = \\underbrace{\\mathbf{w}^T \\Sigma \\mathbf{w}}_{\\text{Cost Volatility}}
            + \\lambda \\underbrace{h(\\mathbf{w})}_{\\text{Stranded Asset Risk}}
            - \\gamma \\underbrace{g(\\mathbf{w})}_{\\text{Option Value}}$$
            """)

            if 'mc_result' in st.session_state:
                mc_result = st.session_state['mc_result']

                target_for_breakdown = st.select_slider(
                    "Select Abatement Target for Decomposition",
                    options=mc_result.abatement_targets.tolist(),
                    value=mc_result.abatement_targets[len(mc_result.abatement_targets)//2],
                    key="breakdown_target"
                )

                idx = np.argmin(np.abs(mc_result.abatement_targets - target_for_breakdown))
                portfolio = mc_result.optimal_portfolios[idx]

                # Compute breakdown
                risk_model = RiskModel(
                    technologies, cov_matrix,
                    r=risk_free_rate,
                    lambda_param=lambda_param,
                    gamma_param=gamma_param
                )
                breakdown = risk_model.breakdown(portfolio.weights, lambda_param, gamma_param)

                col1, col2 = st.columns([1, 1])

                with col1:
                    fig = plot_risk_decomposition(breakdown, [t.name for t in technologies])
                    st.pyplot(fig)

                with col2:
                    st.markdown("#### Component Details")
                    st.metric("Cost Volatility (w^T Σ w)", f"{breakdown.cost_volatility:.3f}")
                    st.metric(f"Stranded Asset Risk (λ={lambda_param})", f"{lambda_param * breakdown.stranded_asset:.3f}")
                    st.metric(f"Option Value (γ={gamma_param})", f"-{gamma_param * breakdown.option_value:.3f}")
                    st.metric("**Total Risk**", f"{breakdown.total:.3f}")

                    st.markdown("---")
                    st.markdown("#### Portfolio Weights")
                    weights_df = pd.DataFrame({
                        'Technology': [t.name for t in technologies],
                        'Weight': portfolio.weights,
                        'Abatement Contribution': portfolio.weights * np.array([t.a for t in technologies])
                    })
                    st.dataframe(weights_df.style.format({'Weight': '{:.2f}', 'Abatement Contribution': '{:.2f}'}))
            else:
                st.warning("Please generate the Monte Carlo Frontier first (Tab 1)")

        # ---------------------------------------------------------------------
        # Tab 3: Dynamic Transition with Real Options
        # ---------------------------------------------------------------------
        with tab3:
            st.subheader("Dynamic Transition Pathway with Real Options")
            st.markdown("""
            Multi-period optimization using Model Predictive Control (Section 6) with:
            - **Irreversibility constraint**: $\\mathbf{w}_t \\geq \\mathbf{w}_{t-1}$
            - **Dynamic option re-valuation**: Options priced via Monte Carlo at each period
            - **Learning curve feedback**: Costs evolve with jump-diffusion
            """)

            col1, col2 = st.columns(2)
            with col1:
                n_periods = st.number_input("Number of Periods", value=min(15, time_horizon), step=1)
            with col2:
                budget_per_period = st.number_input("Budget per Period ($M)", value=budget_constraint / n_periods * 1.5, step=50.0)

            if st.button("Run Dynamic Optimization with Real Options", type="primary"):
                with st.spinner("Solving dynamic optimization with Monte Carlo option valuation..."):
                    dyn_optimizer = DynamicRealOptionsOptimizer(
                        technologies,
                        cov_matrix,
                        risk_free_rate=risk_free_rate,
                        discount_factor=0.95
                    )

                    # Linear ramp-up of targets
                    targets = np.linspace(abatement_target_min, abatement_target_max, int(n_periods)).tolist()

                    portfolios, cost_history = dyn_optimizer.solve_with_options(
                        T_periods=int(n_periods),
                        target_path=targets,
                        budget_per_period=budget_per_period,
                        lambda_param=lambda_param,
                        gamma_param=gamma_param,
                        n_simulations=50  # Fewer for speed
                    )

                    st.session_state['dyn_portfolios'] = portfolios
                    st.session_state['cost_history'] = cost_history

                fig = plot_dynamic_pathway(portfolios, technologies, cost_history)
                st.pyplot(fig)

                # Show final portfolio
                st.markdown("#### Final Period Portfolio")
                final_weights = portfolios[-1].weights
                final_df = pd.DataFrame({
                    'Technology': [t.name for t in technologies],
                    'Final Capacity': final_weights,
                    'Abatement': final_weights * np.array([t.a for t in technologies])
                })
                st.dataframe(final_df.style.format({'Final Capacity': '{:.2f}', 'Abatement': '{:.2f}'}))

        # ---------------------------------------------------------------------
        # Tab 4: Stochastic Dominance
        # ---------------------------------------------------------------------
        with tab4:
            st.subheader("Stochastic Dominance Analysis")
            st.markdown("""
            Compare two portfolios for stochastic dominance:
            - **First-Order (FSD)**: A dominates B if $F_A(x) \\geq F_B(x)$ for all $x$
            - **Second-Order (SSD)**: A dominates B if $\\int F_A(x)dx \\geq \\int F_B(x)dx$ for all $x$

            FSD implies all risk-averse investors prefer A. SSD implies all risk-averse investors
            with decreasing absolute risk aversion prefer A.
            """)

            st.markdown("#### Define Two Portfolios to Compare")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Portfolio A**")
                weights_a = []
                for tech in technologies:
                    w = st.number_input(f"{tech.name}", value=0.0, step=1.0, key=f"a_{tech.name}")
                    weights_a.append(w)
                weights_a = np.array(weights_a)
                name_a = st.text_input("Portfolio A Name", value="Conservative")

            with col2:
                st.markdown("**Portfolio B**")
                weights_b = []
                for tech in technologies:
                    w = st.number_input(f"{tech.name}", value=0.0, step=1.0, key=f"b_{tech.name}")
                    weights_b.append(w)
                weights_b = np.array(weights_b)
                name_b = st.text_input("Portfolio B Name", value="Aggressive")

            if st.button("Analyze Stochastic Dominance", type="primary"):
                if np.sum(weights_a) == 0 or np.sum(weights_b) == 0:
                    st.error("Both portfolios must have non-zero weights")
                else:
                    with st.spinner("Running stochastic dominance analysis..."):
                        analyzer = StochasticDominanceAnalyzer(technologies, cov_matrix)
                        sd_result = analyzer.compare_portfolios(
                            weights_a, weights_b,
                            name_a=name_a, name_b=name_b,
                            n_simulations=500,
                            time_horizon=time_horizon,
                            lambda_param=lambda_param,
                            gamma_param=gamma_param
                        )

                    # Results
                    st.markdown("#### Dominance Results")
                    col1, col2 = st.columns(2)

                    with col1:
                        if sd_result.fsd_a_dominates_b:
                            st.success(f"✅ {name_a} FIRST-ORDER dominates {name_b}")
                        elif sd_result.fsd_b_dominates_a:
                            st.success(f"✅ {name_b} FIRST-ORDER dominates {name_a}")
                        else:
                            st.info("No first-order dominance relationship")

                    with col2:
                        if sd_result.ssd_a_dominates_b:
                            st.success(f"✅ {name_a} SECOND-ORDER dominates {name_b}")
                        elif sd_result.ssd_b_dominates_a:
                            st.success(f"✅ {name_b} SECOND-ORDER dominates {name_a}")
                        else:
                            st.info("No second-order dominance relationship")

                    # Plot
                    fig = plot_stochastic_dominance(sd_result)
                    st.pyplot(fig)

                    # Statistics
                    st.markdown("#### Distribution Statistics")
                    stats = pd.DataFrame({
                        'Metric': ['Mean Risk', 'Std Dev', 'VaR (5%)', 'CVaR (5%)'],
                        name_a: [
                            np.mean(sd_result.risk_distribution_a),
                            np.std(sd_result.risk_distribution_a),
                            np.percentile(sd_result.risk_distribution_a, 95),
                            np.mean(sd_result.risk_distribution_a[sd_result.risk_distribution_a >= np.percentile(sd_result.risk_distribution_a, 95)])
                        ],
                        name_b: [
                            np.mean(sd_result.risk_distribution_b),
                            np.std(sd_result.risk_distribution_b),
                            np.percentile(sd_result.risk_distribution_b, 95),
                            np.mean(sd_result.risk_distribution_b[sd_result.risk_distribution_b >= np.percentile(sd_result.risk_distribution_b, 95)])
                        ]
                    })
                    st.dataframe(stats.style.format({name_a: '{:.3f}', name_b: '{:.3f}'}))

        # ---------------------------------------------------------------------
        # Tab 5: Robust Optimization
        # ---------------------------------------------------------------------
        with tab5:
            st.subheader("Robust Optimization under Uncertainty")
            st.markdown("""
            Solves the min-max problem (Section 9.1):

            $$\\min_{\\mathbf{w}} \\max_{\\theta \\in \\Theta} R_P(\\mathbf{w}; \\theta)$$

            where $\\Theta$ is an uncertainty set for costs, volatilities, and correlations.
            This provides protection against estimation error and model misspecification.
            """)

            st.markdown("#### Uncertainty Parameters")
            col1, col2, col3 = st.columns(3)
            with col1:
                robust_cost_unc = st.slider("Cost Uncertainty (±%)", 0, 50, int(cost_uncertainty*100), key="robust_cost") / 100
            with col2:
                robust_vol_unc = st.slider("Volatility Uncertainty (±%)", 0, 50, int(vol_uncertainty*100), key="robust_vol") / 100
            with col3:
                robust_corr_unc = st.slider("Correlation Uncertainty (±%)", 0, 50, 20, key="robust_corr") / 100

            target_for_robust = st.number_input(
                "Abatement Target for Robust Optimization",
                value=(abatement_target_min + abatement_target_max) / 2,
                step=5.0
            )

            if st.button("Solve Robust Optimization", type="primary"):
                with st.spinner("Solving robust optimization (evaluating worst-case scenarios)..."):
                    robust_optimizer = RobustOptimizer(
                        technologies,
                        cov_matrix,
                        cost_uncertainty=robust_cost_unc,
                        volatility_uncertainty=robust_vol_unc,
                        correlation_uncertainty=robust_corr_unc
                    )

                    robust_result = robust_optimizer.solve(
                        target_abatement=target_for_robust,
                        budget_constraint=budget_constraint,
                        lambda_param=lambda_param,
                        gamma_param=gamma_param,
                        n_scenarios=100
                    )

                # Results
                st.markdown("#### Robust Optimization Results")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Nominal Risk", f"{robust_result.nominal_risk:.3f}")
                with col2:
                    st.metric("Worst-Case Risk", f"{robust_result.worst_case_risk:.3f}")
                with col3:
                    st.metric("Robustness Gap", f"{robust_result.robustness_gap:.3f}",
                             help="Additional risk under worst-case scenario")

                # Portfolio
                st.markdown("#### Robust Portfolio")
                robust_df = pd.DataFrame({
                    'Technology': [t.name for t in technologies],
                    'Robust Weight': robust_result.portfolio.weights,
                    'Abatement': robust_result.portfolio.weights * np.array([t.a for t in technologies])
                })
                st.dataframe(robust_df.style.format({'Robust Weight': '{:.2f}', 'Abatement': '{:.2f}'}))

                # Comparison with nominal
                st.markdown("#### Nominal vs Robust Comparison")

                # Compute nominal solution
                risk_model = RiskModel(technologies, cov_matrix, lambda_param=lambda_param, gamma_param=gamma_param)
                from nz_frontier import OptimizationEngine
                nominal_optimizer = OptimizationEngine(technologies, risk_model)
                try:
                    nominal_portfolio = nominal_optimizer.solve_for_target(
                        target_for_robust, budget_constraint, lambda_param, gamma_param
                    )

                    fig, ax = plt.subplots(figsize=(10, 5))
                    x = np.arange(n_tech)
                    width = 0.35

                    ax.bar(x - width/2, nominal_portfolio.weights, width, label='Nominal', color='steelblue')
                    ax.bar(x + width/2, robust_result.portfolio.weights, width, label='Robust', color='coral')

                    ax.set_ylabel('Capacity')
                    ax.set_title('Nominal vs Robust Portfolio Allocation')
                    ax.set_xticks(x)
                    ax.set_xticklabels([t.name for t in technologies], rotation=45, ha='right')
                    ax.legend()
                    plt.tight_layout()
                    st.pyplot(fig)
                except Exception as e:
                    st.warning(f"Could not compute nominal portfolio: {e}")

        # ---------------------------------------------------------------------
        # Tab 6: Cost Simulation
        # ---------------------------------------------------------------------
        with tab6:
            st.subheader("Technology Cost Simulation")
            st.markdown("""
            Simulates cost evolution using the jump-diffusion with learning (Equation 22):

            $$\\frac{dc_j}{c_j} = (-\\alpha_j \\cdot \\iota_j) dt + \\sigma_j dW_j + h_j dN_j$$

            - **Learning**: Costs decline at rate $\\alpha_j$ (Wright's Law)
            - **Diffusion**: Random shocks with volatility $\\sigma_j$
            - **Jumps**: Breakthrough innovations via Poisson process
            """)

            n_sim_paths = st.number_input("Number of Simulation Paths", value=100, step=20)

            if st.button("Simulate Cost Paths", type="primary"):
                with st.spinner("Simulating cost evolution..."):
                    sim = CostSimulator(technologies, cov_matrix)
                    paths = sim.simulate_paths(time_horizon, dt=0.5, n_paths=int(n_sim_paths))

                tech_select = st.selectbox("Select Technology to View", [t.name for t in technologies])

                if tech_select:
                    tech_paths = paths[tech_select]

                    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

                    # Path plot
                    ax1 = axes[0]
                    time_steps = np.arange(tech_paths.shape[1]) * 0.5

                    # Plot sample paths
                    for i in range(min(30, tech_paths.shape[0])):
                        ax1.plot(time_steps, tech_paths[i, :], alpha=0.2, color='gray', linewidth=0.8)

                    # Mean and percentiles
                    mean_path = np.mean(tech_paths, axis=0)
                    p10 = np.percentile(tech_paths, 10, axis=0)
                    p90 = np.percentile(tech_paths, 90, axis=0)

                    ax1.fill_between(time_steps, p10, p90, alpha=0.3, color='steelblue', label='80% CI')
                    ax1.plot(time_steps, mean_path, 'r-', linewidth=2.5, label='Expected Cost')

                    ax1.set_xlabel('Years', fontsize=11)
                    ax1.set_ylabel('Cost ($/unit)', fontsize=11)
                    ax1.set_title(f'Cost Evolution: {tech_select}', fontsize=12)
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)

                    # Terminal distribution
                    ax2 = axes[1]
                    terminal_costs = tech_paths[:, -1]
                    ax2.hist(terminal_costs, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
                    ax2.axvline(np.mean(terminal_costs), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(terminal_costs):.1f}')
                    ax2.axvline(np.percentile(terminal_costs, 10), color='orange', linestyle=':', linewidth=2, label=f'10th %ile: {np.percentile(terminal_costs, 10):.1f}')
                    ax2.set_xlabel('Terminal Cost ($/unit)', fontsize=11)
                    ax2.set_ylabel('Frequency', fontsize=11)
                    ax2.set_title(f'Terminal Cost Distribution (Year {time_horizon})', fontsize=12)
                    ax2.legend()

                    plt.tight_layout()
                    st.pyplot(fig)

                    # Statistics
                    st.markdown("#### Simulation Statistics")
                    initial_cost = tech_paths[0, 0]
                    stats = {
                        'Initial Cost': initial_cost,
                        'Expected Terminal Cost': np.mean(terminal_costs),
                        'Cost Reduction (%)': (1 - np.mean(terminal_costs) / initial_cost) * 100,
                        'Terminal Std Dev': np.std(terminal_costs),
                        'Terminal 10th Percentile': np.percentile(terminal_costs, 10),
                        'Terminal 90th Percentile': np.percentile(terminal_costs, 90),
                    }
                    stats_df = pd.DataFrame([stats]).T
                    stats_df.columns = ['Value']
                    st.dataframe(stats_df.style.format({'Value': '{:.2f}'}))

    except Exception as e:
        st.error(f"Error processing data: {e}")
        import traceback
        st.code(traceback.format_exc())

else:
    st.info("Please upload a CSV file or select a preset case to begin.")

    st.markdown("""
    ### CSV Format
    Your CSV should include the following columns:

    | Column | Description | Required |
    |--------|-------------|----------|
    | `name` | Technology name | Yes |
    | `a` | Abatement potential (tCO₂/unit) | Yes |
    | `c` | Capital cost ($/unit) | Yes |
    | `sigma` | Cost volatility | Yes |
    | `rho` | Correlation factor | No |
    | `o` | Embedded option value | No |
    | `tau` | Capital lifetime (years) | No |
    | `jump_intensity` | Poisson intensity for breakthroughs | No |
    | `jump_size` | Jump size (negative = cost reduction) | No |
    | `learning_rate` | Learning curve rate (α) | No |
    | `failure_prob` | Technology failure probability | No |
    | `loss_given_failure` | Loss given failure ($/unit) | No |
    """)
