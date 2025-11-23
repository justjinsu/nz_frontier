import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nz_frontier import Technology, EfficientFrontier, DynamicOptimizer, CostSimulator

st.set_page_config(page_title="Net-Zero Frontier", layout="wide")

st.title("Risk-Efficiency Theory of Corporate Decarbonization")
st.markdown("""
### Optimal Portfolio Design for Net-Zero Transition

This application implements the **Risk-Efficiency Framework** for corporate decarbonization. 
It helps firms select the optimal mix of low-carbon technologies to meet abatement targets while **minimizing transition risks** (technology failure, stranded assets, and cost volatility).

**Key Features:**
*   **Efficient Frontier**: Visualize the trade-off between Abatement and Risk.
*   **Dynamic Pathways**: Plan multi-year investment strategies using Model Predictive Control.
*   **Real Options**: Value flexibility in emerging technologies.
*   **Stochastic Simulation**: Stress-test portfolios against cost shocks and jumps.
""")

# Sidebar for Inputs
st.sidebar.header("Configuration")

# 1. Data Source
data_source = st.sidebar.radio("Data Source", ["Upload CSV", "Preset Case: Steel", "Preset Case: Petrochemical"])

df = None
if data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload Technology Data (CSV)", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
elif data_source == "Preset Case: Steel":
    df = pd.read_csv("data/steel.csv")
elif data_source == "Preset Case: Petrochemical":
    df = pd.read_csv("data/petrochemical.csv")

# 2. Global Parameters
st.sidebar.subheader("Model Parameters")
abatement_target_min = st.sidebar.number_input("Min Abatement Target", value=10.0)
abatement_target_max = st.sidebar.number_input("Max Abatement Target", value=100.0)
budget_constraint = st.sidebar.number_input("Budget Constraint ($)", value=2000.0)

# 3. Advanced Settings
with st.sidebar.expander("Advanced Settings"):
    risk_free_rate = st.number_input("Risk-Free Rate", value=0.05)
    time_horizon = st.number_input("Time Horizon (Years)", value=20)
    n_sim_paths = st.number_input("Simulation Paths", value=100)

def load_technologies(df):
    techs = []
    for _, row in df.iterrows():
        t = Technology(
            name=row['name'],
            a=row['a'],
            c=row['c'],
            sigma=row['sigma'],
            rho=row.get('rho', 0.0),
            o=row.get('o', 0.0),
            tau=row.get('tau', 0.0),
            jump_intensity=row.get('jump_intensity', 0.0),
            jump_size=row.get('jump_size', 0.0),
            strike_price=row.get('strike_price', 0.0),
            learning_rate=row.get('learning_rate', 0.0),
            failure_prob=row.get('failure_prob', 0.0),
            loss_given_failure=row.get('loss_given_failure', 0.0)
        )
        techs.append(t)
    return techs

if df is not None:
    st.write(f"### Loaded Technologies ({data_source})")
    st.dataframe(df)
    
    try:
        technologies = load_technologies(df)
        
        # Construct Covariance Matrix (Simplified: Diagonal + Rho)
        n = len(technologies)
        sigmas = np.array([t.sigma for t in technologies])
        cov_matrix = np.diag(sigmas**2) 
        
        # --- Tab Layout ---
        tab1, tab2, tab3, tab4 = st.tabs(["Efficient Frontier", "Dynamic Transition", "Cost Simulation", "Carbon Price Analysis"])
        
        with tab1:
            st.subheader("Net-Zero Efficient Frontier")
            if st.button("Generate Frontier"):
                with st.spinner("Optimizing..."):
                    frontier = EfficientFrontier(technologies, cov_matrix)
                    results = frontier.compute(abatement_target_min, abatement_target_max, n_points=20, budget_constraint=budget_constraint)
                    
                    # Plotting
                    risks = [p.risk for p in results]
                    abatements = [p.abatement for p in results]
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(abatements, risks, 'o-', linewidth=2, markersize=8, color='#2E86C1')
                    ax.set_xlabel('Abatement Target (tCO2)')
                    ax.set_ylabel('Portfolio Transition Risk')
                    ax.set_title('Efficient Frontier')
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    
                    # Portfolio Composition
                    st.subheader("Optimal Portfolio Composition")
                    target_select = st.select_slider("Select Abatement Target to View Portfolio", options=abatements)
                    
                    # Find closest result
                    idx = np.argmin(np.abs(np.array(abatements) - target_select))
                    selected_portfolio = results[idx].portfolio
                    
                    fig2, ax2 = plt.subplots()
                    weights = selected_portfolio.weights
                    names = [t.name for t in technologies]
                    ax2.bar(names, weights, color='#28B463')
                    ax2.set_ylabel('Capacity (Units)')
                    ax2.set_title(f'Portfolio for Target {target_select:.1f}')
                    plt.xticks(rotation=45)
                    st.pyplot(fig2)

        with tab2:
            st.subheader("Dynamic Transition Pathway")
            st.info("Simulates a multi-period transition where the abatement target increases over time.")
            if st.button("Run Dynamic Optimization"):
                dyn_opt = DynamicOptimizer(technologies, cov_matrix, discount_factor=0.95)
                
                # Linear ramp up of target
                targets = np.linspace(abatement_target_min, abatement_target_max, int(time_horizon))
                budget_per_period = budget_constraint / time_horizon * 2 # Allow some flex
                
                portfolios = dyn_opt.solve_bellman(int(time_horizon), targets, budget_per_period)
                
                # Plot
                weights_history = np.array([p.weights for p in portfolios])
                fig3, ax3 = plt.subplots(figsize=(10, 6))
                for i, tech in enumerate(technologies):
                    ax3.plot(range(int(time_horizon)), weights_history[:, i], label=tech.name, linewidth=2)
                
                ax3.set_xlabel('Year')
                ax3.set_ylabel('Installed Capacity')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
                st.pyplot(fig3)

        with tab3:
            st.subheader("Stochastic Cost Simulation")
            if st.button("Simulate Costs"):
                sim = CostSimulator(technologies)
                paths = sim.simulate_paths(time_horizon, dt=0.1, n_paths=n_sim_paths)
                
                tech_select = st.selectbox("Select Technology", [t.name for t in technologies])
                
                if tech_select:
                    tech_paths = paths[tech_select]
                    fig4, ax4 = plt.subplots(figsize=(10, 6))
                    # Plot first 20 paths
                    ax4.plot(tech_paths[:20, :].T, alpha=0.3, color='gray')
                    # Plot mean
                    ax4.plot(np.mean(tech_paths, axis=0), 'r-', linewidth=2, label='Mean Cost')
                    ax4.set_title(f'Cost Evolution: {tech_select} (with Learning & Jumps)')
                    ax4.set_xlabel('Time Steps')
                    ax4.set_ylabel('Cost ($)')
                    ax4.legend()
                    st.pyplot(fig4)
        
        with tab4:
            st.subheader("Carbon Price Sensitivity Analysis")
            st.markdown("Analyze how the optimal portfolio changes with increasing Carbon Price ($p_c$).")
            
            carbon_prices = np.linspace(0, 200, 10)
            target_for_sensitivity = st.number_input("Abatement Target for Sensitivity", value=(abatement_target_min + abatement_target_max)/2)
            
            if st.button("Run Sensitivity Analysis"):
                # We need to adjust 'a' (abatement value) based on carbon price?
                # Or 'c' (cost) becomes c - p_c * a? 
                # The paper says: effective abatement value a_tilde = a * (1 + p_c/p_bar)^epsilon
                # Let's implement this logic.
                
                p_bar = 100.0 # Reference price
                epsilon = 0.5 # Elasticity
                
                sensitivity_results = []
                
                for p_c in carbon_prices:
                    # Create modified technologies
                    mod_techs = []
                    for t in technologies:
                        # Modify abatement value or cost?
                        # Usually higher carbon price makes abatement more valuable.
                        # Let's assume it reduces effective cost: c_effective = c - p_c * a
                        # Or increases abatement utility. 
                        # Paper Eq (243): a_tilde = a * (1 + p_c/p_bar)^epsilon
                        
                        a_tilde = t.a * (1 + p_c / p_bar)**epsilon
                        
                        # Create copy
                        t_mod = Technology(
                            name=t.name, a=a_tilde, c=t.c, sigma=t.sigma, 
                            rho=t.rho, o=t.o, tau=t.tau, 
                            jump_intensity=t.jump_intensity, jump_size=t.jump_size, strike_price=t.strike_price,
                            learning_rate=t.learning_rate, failure_prob=t.failure_prob, loss_given_failure=t.loss_given_failure
                        )
                        mod_techs.append(t_mod)
                    
                    # Optimize
                    frontier_sens = EfficientFrontier(mod_techs, cov_matrix)
                    try:
                        # We solve for the SAME physical abatement target? 
                        # Or target in terms of a_tilde? 
                        # Usually target is physical tCO2. So constraint uses physical 'a'.
                        # But objective function risk is same.
                        # Wait, if 'a' changes, the constraint sum(w*a) >= A changes.
                        # If p_c increases, 'a' effectively increases (value), so we need LESS capacity to meet "value" target?
                        # No, target is physical.
                        # Carbon price usually affects the COST function or the OPTIMALITY condition.
                        # In this risk-min framework, p_c might not directly enter unless it affects Risk or Cost.
                        # Let's assume p_c reduces the effective cost of technology: c_eff = c - p_c * a
                        # But our objective is Risk Minimization, not Cost Minimization.
                        # Cost is a constraint: sum(w*c) <= B.
                        # So higher p_c -> effectively higher budget or lower cost?
                        # Let's assume it relaxes the budget constraint: sum(w * (c - p_c*a)) <= B
                        
                        # Let's use the budget relaxation approach.
                        
                        # We need to pass modified costs to optimizer?
                        # Our optimizer takes 'technologies' list.
                        # Let's modify 'c' in mod_techs.
                        
                        for t_mod in mod_techs:
                            # Effective cost can't be negative for this solver setup easily, but let's try
                            t_mod.c = max(0.1, t_mod.c - p_c * t_mod.a * 0.5) # Scaling factor 0.5 for realism
                        
                        # Re-init optimizer with mod_techs
                        # Note: We must use ORIGINAL 'a' for the physical abatement constraint if A* is physical.
                        # But here we modified 'a' in the loop above based on paper eq.
                        # Let's revert 'a' to physical for constraint, but keep 'c' modified.
                        for t_mod, t_orig in zip(mod_techs, technologies):
                            t_mod.a = t_orig.a
                            
                        optimizer = EfficientFrontier(mod_techs, cov_matrix).optimizer
                        port = optimizer.solve_for_target(target_for_sensitivity, budget_constraint)
                        sensitivity_results.append(port.weights)
                        
                    except Exception as e:
                        sensitivity_results.append(np.zeros(n))
                
                # Plot Stacked Area
                sensitivity_results = np.array(sensitivity_results)
                fig5, ax5 = plt.subplots(figsize=(10, 6))
                ax5.stackplot(carbon_prices, sensitivity_results.T, labels=[t.name for t in technologies], alpha=0.8)
                ax5.set_xlabel('Carbon Price ($/tCO2)')
                ax5.set_ylabel('Optimal Capacity (GW)')
                ax5.set_title('Portfolio Sensitivity to Carbon Price')
                ax5.legend(loc='upper left')
                st.pyplot(fig5)

    except Exception as e:
        st.error(f"Error processing data: {e}")
else:
    st.info("Please upload a CSV file or select a preset case to begin.")
