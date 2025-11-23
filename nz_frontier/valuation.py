import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import norm

class OptionValuator:
    def __init__(self, r: float = 0.05, delta: float = 0.0):
        """
        Args:
            r: Risk-free rate
            delta: Dividend yield / convenience yield
        """
        self.r = r
        self.delta = delta

    def black_scholes_call(self, S: float, K: float, T: float, sigma: float) -> float:
        """
        Closed-form European call valuation used in paper_v2.tex (Section 5).
        """
        if T <= 0:
            return max(S - K, 0.0)
        if sigma <= 0:
            return max(S - K * np.exp(-self.r * T), 0.0)

        sqrtT = np.sqrt(T)
        d1 = (np.log(S / K) + (self.r - self.delta + 0.5 * sigma**2) * T) / (sigma * sqrtT)
        d2 = d1 - sigma * sqrtT
        return float(
            S * np.exp(-self.delta * T) * norm.cdf(d1) - K * np.exp(-self.r * T) * norm.cdf(d2)
        )

    def solve_pde(self, 
                  S_max: float, 
                  K: float, 
                  T: float, 
                  sigma: float, 
                  M: int = 100, 
                  N: int = 1000) -> float:
        """
        Solves the PDE for Option Value using Implicit Finite Difference Method.
        
        PDE: 0.5*sigma^2*S^2*O_SS + (r-delta)*S*O_S - r*O + O_t = 0
        Note: Usually O_t is -O_tau where tau is time to maturity. 
        Standard Black-Scholes PDE: O_t + 0.5*sigma^2*S^2*O_SS + r*S*O_S - r*O = 0
        
        We solve backwards from t=T to t=0.
        
        Args:
            S_max: Maximum value of state variable S
            K: Strike price
            T: Time to maturity
            sigma: Volatility
            M: Number of asset steps
            N: Number of time steps
            
        Returns:
            Option value at S = current_S (assumed to be K for ATM or defined externally, 
            here we return the interpolation function or value at a specific S).
            For simplicity, let's assume we want the value at S_0.
            But to be generic, let's return the value at S = K (At-The-Money) as a default proxy 
            if S_0 isn't specified, or add S_0 as arg.
        """
        # Grid
        dS = S_max / M
        dt = T / N
        S = np.linspace(0, S_max, M+1)
        t = np.linspace(0, T, N+1)
        
        # Grid for V[i, j] where i is S index, j is time index
        # We solve for V[i, j] where j goes from N (maturity) down to 0 (today)
        V = np.zeros((M+1, N+1))
        
        # Boundary Conditions
        # Terminal condition: V(S, T) = max(S - K, 0)
        V[:, N] = np.maximum(S - K, 0)
        
        # Boundary conditions at S=0 and S=S_max
        # S=0: V(0, t) = 0
        V[0, :] = 0
        # S=S_max: V(S_max, t) = S_max - K * exp(-r(T-t))
        V[M, :] = S_max - K * np.exp(-self.r * (T - t))
        
        # Coefficients for Implicit Scheme
        # a_i * V[i-1, j] + b_i * V[i, j] + c_i * V[i+1, j] = V[i, j+1]
        # We are solving A * V_j = V_{j+1}
        
        # Discretization:
        # dV/dt approx (V[i, j+1] - V[i, j]) / dt
        # dV/dS approx (V[i+1, j] - V[i-1, j]) / (2*dS)
        # d2V/dS2 approx (V[i+1, j] - 2*V[i, j] + V[i-1, j]) / (dS^2)
        
        # Actually, for backward induction:
        # (V[i, j+1] - V[i, j])/dt + ... = 0
        # V[i, j] = V[i, j+1] + dt * (...)
        # Implicit method is more stable.
        
        # Let's use standard coefficients for fully implicit:
        # -0.5*dt*(sigma^2*i^2 - (r-delta)*i) * V[i-1, j] 
        # + (1 + dt*(sigma^2*i^2 + r)) * V[i, j]
        # + -0.5*dt*(sigma^2*i^2 + (r-delta)*i) * V[i+1, j]
        # = V[i, j+1]
        
        alpha = 0.5 * dt * (sigma**2 * np.arange(1, M)**2 - (self.r - self.delta) * np.arange(1, M))
        beta = 1 + dt * (sigma**2 * np.arange(1, M)**2 + self.r)
        gamma = 0.5 * dt * (sigma**2 * np.arange(1, M)**2 + (self.r - self.delta) * np.arange(1, M))
        
        # Tridiagonal matrix construction
        A_diag = np.diag(beta)
        A_lower = np.diag(-alpha[1:], k=-1)
        A_upper = np.diag(-gamma[:-1], k=1)
        A = A_diag + A_lower + A_upper
        
        # Time stepping
        for j in range(N-1, -1, -1):
            B = V[1:M, j+1].copy()
            # Adjust for boundary conditions
            B[0] += alpha[0] * V[0, j] # V[0, j] is 0
            B[-1] += gamma[-1] * V[M, j] # V[M, j] is known boundary
            
            # Solve linear system
            V[1:M, j] = np.linalg.solve(A, B)
            
        return V[:, 0], S

    def calculate_option_value(self, 
                               S_0: float, 
                               K: float, 
                               T: float, 
                               sigma: float) -> float:
        """
        Calculates option value for a specific current state S_0.
        """
        if T <= 0:
            return max(S_0 - K, 0.0)
            
        # Heuristic for S_max
        S_max = max(S_0 * 3, K * 3)
        
        values, S_grid = self.solve_pde(S_max, K, T, sigma)
        
        # Interpolate to find value at S_0
        f = interp1d(S_grid, values, kind='cubic')
        return float(f(S_0))
