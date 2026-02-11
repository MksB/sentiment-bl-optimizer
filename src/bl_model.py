"""
Black-Litterman Model Implementation
Based on Meucci (2008): "The Black-Litterman Approach: Original Model and Extensions"
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1117574

This implementation follows the market formulation (Section 3) which is more intuitive
and handles null-confidence and full-confidence limits correctly.
"""

import numpy as np
from typing import Optional, Tuple, Union
import warnings


class BlackLittermanModel:
    """
    Black-Litterman portfolio optimization model.
    
    This class implements both the original BL formulation (Section 2) and
    the market-based formulation (Section 3) from Meucci's paper.
    
    The market formulation is recommended as it:
    1. Eliminates the need for the tau parameter in posterior calculations
    2. Correctly handles limiting cases (null and full confidence)
    3. Integrates naturally with scenario analysis
    
    Attributes:
        pi (np.ndarray): Prior expected returns (equilibrium or reference)
        sigma (np.ndarray): Covariance matrix of asset returns
        tau (float): Uncertainty scalar for prior (only used in original formulation)
        use_market_formulation (bool): If True, uses market-based BL (recommended)
    """
    
    def __init__(
        self,
        pi: np.ndarray,
        sigma: np.ndarray,
        tau: float = 0.025,
        use_market_formulation: bool = True
    ):
        """
        Initialize the Black-Litterman model.
        
        Args:
            pi: Prior expected returns (N x 1 array or N-length vector)
                Typically derived from equilibrium (CAPM) as pi = 2*lambda*Sigma*w_eq
            sigma: Covariance matrix of returns (N x N)
            tau: Scalar representing uncertainty in prior (typically 0.01 to 0.05)
                 Meucci suggests tau ≈ 1/T where T is the time series length
                 Only used in original formulation
            use_market_formulation: If True, uses market-based formulation (Section 3)
                                   If False, uses original formulation (Section 2)
        """
        # Convert to numpy arrays and validate inputs
        self.pi = np.asarray(pi).flatten()
        self.sigma = np.asarray(sigma)
        self.tau = tau
        self.use_market_formulation = use_market_formulation
        
        # Validate dimensions
        n_assets = len(self.pi)
        if self.sigma.shape != (n_assets, n_assets):
            raise ValueError(
                f"Sigma shape {self.sigma.shape} incompatible with pi length {n_assets}"
            )
        
        # Check if sigma is symmetric positive definite
        if not np.allclose(self.sigma, self.sigma.T):
            warnings.warn("Covariance matrix is not symmetric. Symmetrizing.")
            self.sigma = (self.sigma + self.sigma.T) / 2
        
        # Check positive definiteness
        eigenvalues = np.linalg.eigvalsh(self.sigma)
        if np.any(eigenvalues <= 0):
            warnings.warn(
                f"Covariance matrix has non-positive eigenvalues: {eigenvalues[eigenvalues <= 0]}"
            )
    
    @staticmethod
    def compute_equilibrium_returns(
        w_eq: np.ndarray,
        sigma: np.ndarray,
        risk_aversion: float = 2.5
    ) -> np.ndarray:
        """
        Compute equilibrium expected returns from market weights.
        
        Based on equation (5) in Meucci:
        π = 2λΣw_eq
        
        where λ is the market risk aversion parameter.
        
        Args:
            w_eq: Market equilibrium weights (N x 1 or N-length)
            sigma: Covariance matrix (N x N)
            risk_aversion: Market risk aversion parameter (λ)
                          Typical values: 2-3 for equity markets
                          Black-Litterman suggest λ ≈ 2.5
        
        Returns:
            Equilibrium expected returns (N x 1 array)
        """
        w_eq = np.asarray(w_eq).flatten()
        sigma = np.asarray(sigma)
        
        # Equation (5): π = 2λΣw_eq
        pi = 2 * risk_aversion * sigma @ w_eq
        
        return pi
    
    def compute_posterior_original(
        self,
        P: np.ndarray,
        Q: np.ndarray,
        Omega: Optional[np.ndarray] = None,
        confidence: float = 1.0,
        relative_confidence: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute posterior distribution using ORIGINAL Black-Litterman formulation.
        
        This implements equations (20) and (21) from Section 2:
        
        μ_BL = π + τΣP'(τPΣP' + Ω)^(-1)(Q - Pπ)
        Σ_BL = (1+τ)Σ - τ²ΣP'(τPΣP' + Ω)^(-1)PΣ
        
        Args:
            P: View pick matrix (K x N) where K = number of views
               Each row defines a portfolio for one view
            Q: View expected returns (K x 1 or K-length)
               Expected returns on the view portfolios
            Omega: Uncertainty matrix for views (K x K)
                   If None, computed from equation (12) or (13)
            confidence: Overall confidence level c in equations (12-13)
                       Higher values = more confident in views
            relative_confidence: Relative confidence for each view (K-length)
                                Used in equation (13) as vector u
        
        Returns:
            mu_bl: Posterior expected returns (N x 1)
            sigma_bl: Posterior covariance matrix (N x N)
        """
        # Convert inputs to numpy arrays
        P = np.asarray(P)
        Q = np.asarray(Q).flatten()
        
        K, N = P.shape
        if len(Q) != K:
            raise ValueError(f"Q length {len(Q)} must match P rows {K}")
        if N != len(self.pi):
            raise ValueError(f"P columns {N} must match asset count {len(self.pi)}")
        
        # Compute Omega if not provided
        if Omega is None:
            Omega = self._compute_omega(P, confidence, relative_confidence)
        else:
            Omega = np.asarray(Omega)
            # Handle scalar Omega for single view
            if Omega.ndim == 0 and K == 1:
                Omega = Omega.reshape((1, 1))
            elif Omega.shape != (K, K):
                raise ValueError(f"Omega shape {Omega.shape} must be ({K}, {K})")
        
        # Compute intermediate term: τPΣP' + Ω
        tau_P_sigma_Pt = self.tau * P @ self.sigma @ P.T
        M = tau_P_sigma_Pt + Omega
        
        # Compute M^(-1) using stable inversion
        try:
            M_inv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            warnings.warn("Matrix M is singular, using pseudo-inverse")
            M_inv = np.linalg.pinv(M)
        
        # Equation (20): μ_BL = π + τΣP'(τPΣP' + Ω)^(-1)(Q - Pπ)
        view_adjustment = Q - P @ self.pi
        mu_bl = self.pi + self.tau * self.sigma @ P.T @ M_inv @ view_adjustment
        
        # Equation (21): Σ_BL = (1+τ)Σ - τ²ΣP'(τPΣP' + Ω)^(-1)PΣ
        sigma_bl = (1 + self.tau) * self.sigma
        sigma_bl -= self.tau**2 * self.sigma @ P.T @ M_inv @ P @ self.sigma
        
        return mu_bl, sigma_bl
    
    def compute_posterior_market(
        self,
        P: np.ndarray,
        Q: np.ndarray,
        Omega: Optional[np.ndarray] = None,
        confidence: float = 1.0,
        relative_confidence: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute posterior distribution using MARKET-BASED formulation.
        
        This implements equations (32) and (33) from Section 3:
        
        μ_BL^m = π + ΣP'(PΣP' + Ω)^(-1)(Q - Pπ)
        Σ_BL^m = Σ - ΣP'(PΣP' + Ω)^(-1)PΣ
        
        This formulation:
        - Does not require tau in posterior computation
        - Correctly reduces to prior when Ω → ∞ (no confidence)
        - Correctly reduces to conditional when Ω → 0 (full confidence)
        - Integrates naturally with scenario analysis
        
        Args:
            P: View pick matrix (K x N)
            Q: View expected returns (K x 1 or K-length)
            Omega: Uncertainty matrix for views (K x K)
            confidence: Overall confidence level
            relative_confidence: Relative confidence for each view
        
        Returns:
            mu_bl: Posterior expected returns (N x 1)
            sigma_bl: Posterior covariance matrix (N x N)
        """
        # Convert inputs to numpy arrays
        P = np.asarray(P)
        Q = np.asarray(Q).flatten()
        
        K, N = P.shape
        if len(Q) != K:
            raise ValueError(f"Q length {len(Q)} must match P rows {K}")
        if N != len(self.pi):
            raise ValueError(f"P columns {N} must match asset count {len(self.pi)}")
        
        # Compute Omega if not provided
        if Omega is None:
            Omega = self._compute_omega(P, confidence, relative_confidence)
        else:
            Omega = np.asarray(Omega)
            # Handle scalar Omega for single view
            if Omega.ndim == 0 and K == 1:
                Omega = Omega.reshape((1, 1))
            elif Omega.shape != (K, K):
                raise ValueError(f"Omega shape {Omega.shape} must be ({K}, {K})")
        
        # Compute intermediate term: PΣP' + Ω
        P_sigma_Pt = P @ self.sigma @ P.T
        M = P_sigma_Pt + Omega
        
        # Compute M^(-1) using stable inversion
        try:
            M_inv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            warnings.warn("Matrix M is singular, using pseudo-inverse")
            M_inv = np.linalg.pinv(M)
        
        # Equation (32): μ_BL^m = π + ΣP'(PΣP' + Ω)^(-1)(Q - Pπ)
        view_adjustment = Q - P @ self.pi
        mu_bl = self.pi + self.sigma @ P.T @ M_inv @ view_adjustment
        
        # Equation (33): Σ_BL^m = Σ - ΣP'(PΣP' + Ω)^(-1)PΣ
        sigma_bl = self.sigma - self.sigma @ P.T @ M_inv @ P @ self.sigma
        
        return mu_bl, sigma_bl
    
    def compute_posterior(
        self,
        P: np.ndarray,
        Q: np.ndarray,
        Omega: Optional[np.ndarray] = None,
        confidence: float = 1.0,
        relative_confidence: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute posterior distribution using the selected formulation.
        
        This is the main interface method that delegates to either the original
        or market-based formulation based on initialization.
        
        Args:
            P: View pick matrix (K x N)
               Example: [[0, 1, 0, 0]] for a view on asset 2
                       [[1, -1, 0, 0]] for a view that asset 1 outperforms asset 2
            Q: View expected returns (K x 1 or K-length)
               Example: [0.05] for 5% expected return on the view
            Omega: Uncertainty matrix for views (K x K)
                   If None, computed automatically
            confidence: Overall confidence level (higher = more confident)
                       Typical range: 0.1 to 10
            relative_confidence: Relative confidence for each view (K-length)
                                Can be used to express different confidence levels
        
        Returns:
            mu_bl: Posterior expected returns (N x 1)
            sigma_bl: Posterior covariance matrix (N x N)
        
        Example:
            >>> # Single view: Asset 2 will return 5%
            >>> P = np.array([[0, 1, 0, 0]])
            >>> Q = np.array([0.05])
            >>> mu_bl, sigma_bl = bl.compute_posterior(P, Q, confidence=2.0)
        """
        if self.use_market_formulation:
            return self.compute_posterior_market(
                P, Q, Omega, confidence, relative_confidence
            )
        else:
            return self.compute_posterior_original(
                P, Q, Omega, confidence, relative_confidence
            )
    
    def _compute_omega(
        self,
        P: np.ndarray,
        confidence: float = 1.0,
        relative_confidence: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute the view uncertainty matrix Omega.
        
        Implements equations (12) and (13) from the paper:
        
        Simple version (12): Ω = (1/c) * PΣP'
        
        Advanced version (13): Ω = (1/c) * diag(u) * PΣP' * diag(u)
        
        where c is the overall confidence and u is the relative confidence vector.
        
        Args:
            P: View pick matrix (K x N)
            confidence: Overall confidence parameter c
            relative_confidence: Relative confidence vector u (K-length)
        
        Returns:
            Omega: View uncertainty matrix (K x K)
        """
        # Compute base uncertainty from market volatilities
        P_sigma_Pt = P @ self.sigma @ P.T
        
        if relative_confidence is not None:
            # Equation (13): Ω = (1/c) * diag(u) * PΣP' * diag(u)
            u = np.asarray(relative_confidence).flatten()
            if len(u) != P.shape[0]:
                raise ValueError(
                    f"relative_confidence length {len(u)} must match views {P.shape[0]}"
                )
            u_diag = np.diag(u)
            Omega = (1 / confidence) * u_diag @ P_sigma_Pt @ u_diag
        else:
            # Equation (12): Ω = (1/c) * PΣP'
            Omega = (1 / confidence) * P_sigma_Pt
        
        return Omega
    
    def set_qualitative_views(
        self,
        P: np.ndarray,
        view_types: list,
        alpha: float = 1.0,
        beta: float = 2.0
    ) -> np.ndarray:
        """
        Convert qualitative views to quantitative Q vector.
        
        Implements equation (11) from the paper:
        Q_k = (Pπ)_k + η_k * sqrt((PΣP')_{k,k})
        
        where η ∈ {-β, -α, +α, +β} for:
        - "very bearish" (-β)
        - "bearish" (-α)
        - "bullish" (+α)
        - "very bullish" (+β)
        
        Args:
            P: View pick matrix (K x N)
            view_types: List of strings from:
                       ['very_bearish', 'bearish', 'bullish', 'very_bullish']
            alpha: Parameter for bullish/bearish (typically 1.0)
            beta: Parameter for very bullish/bearish (typically 2.0)
        
        Returns:
            Q: Quantitative view vector (K x 1)
        
        Example:
            >>> P = np.array([[1, -1, 0], [0, 1, 0]])
            >>> types = ['bullish', 'very_bullish']
            >>> Q = bl.set_qualitative_views(P, types)
        """
        P = np.asarray(P)
        K = P.shape[0]
        
        if len(view_types) != K:
            raise ValueError(
                f"view_types length {len(view_types)} must match views {K}"
            )
        
        # Map view types to eta values
        eta_map = {
            'very_bearish': -beta,
            'bearish': -alpha,
            'bullish': alpha,
            'very_bullish': beta
        }
        
        # Compute base expectations from prior
        P_pi = P @ self.pi
        
        # Compute volatility of view portfolios
        P_sigma_Pt = P @ self.sigma @ P.T
        view_volatilities = np.sqrt(np.diag(P_sigma_Pt))
        
        # Equation (11): Q_k = (Pπ)_k + η_k * sqrt((PΣP')_{k,k})
        Q = np.zeros(K)
        for k, view_type in enumerate(view_types):
            if view_type not in eta_map:
                raise ValueError(
                    f"view_type '{view_type}' not recognized. "
                    f"Use one of: {list(eta_map.keys())}"
                )
            eta = eta_map[view_type]
            Q[k] = P_pi[k] + eta * view_volatilities[k]
        
        return Q
    
    def validate_posterior(
        self,
        sigma_bl: np.ndarray,
        check_psd: bool = True,
        tolerance: float = 1e-8
    ) -> dict:
        """
        Validate the posterior covariance matrix.
        
        Checks:
        1. Symmetry
        2. Positive semi-definiteness
        3. Reasonable conditioning
        
        Args:
            sigma_bl: Posterior covariance matrix
            check_psd: If True, check positive semi-definiteness
            tolerance: Numerical tolerance for checks
        
        Returns:
            Dictionary with validation results
        """
        results = {
            'is_symmetric': False,
            'is_psd': None,
            'min_eigenvalue': None,
            'condition_number': None,
            'warnings': []
        }
        
        # Check symmetry
        if np.allclose(sigma_bl, sigma_bl.T, atol=tolerance):
            results['is_symmetric'] = True
        else:
            results['warnings'].append("Covariance matrix is not symmetric")
        
        if check_psd:
            # Check positive semi-definiteness
            eigenvalues = np.linalg.eigvalsh(sigma_bl)
            results['min_eigenvalue'] = np.min(eigenvalues)
            
            if np.all(eigenvalues >= -tolerance):
                results['is_psd'] = True
            else:
                results['is_psd'] = False
                results['warnings'].append(
                    f"Covariance has negative eigenvalues: min = {results['min_eigenvalue']}"
                )
            
            # Check condition number
            max_eig = np.max(eigenvalues)
            if results['min_eigenvalue'] > tolerance:
                results['condition_number'] = max_eig / results['min_eigenvalue']
                if results['condition_number'] > 1e10:
                    results['warnings'].append(
                        f"Poor conditioning: {results['condition_number']:.2e}"
                    )
        
        return results


def create_view_matrix(
    n_assets: int,
    absolute_views: Optional[dict] = None,
    relative_views: Optional[list] = None
) -> np.ndarray:
    """
    Helper function to create the P matrix from intuitive view specifications.
    
    Args:
        n_assets: Total number of assets
        absolute_views: Dict mapping asset indices to weights
                       Example: {1: 1.0} for a view on asset 1
        relative_views: List of (asset_i, asset_j, weight_i, weight_j) tuples
                       Example: [(0, 1, 1.0, -1.0)] for asset 0 outperforms asset 1
    
    Returns:
        P matrix (K x N) where K is total number of views
    
    Example:
        >>> # View 1: Asset 2 will return X%
        >>> # View 2: Asset 0 will outperform Asset 1
        >>> P = create_view_matrix(
        ...     n_assets=4,
        ...     absolute_views={2: 1.0},
        ...     relative_views=[(0, 1, 1.0, -1.0)]
        ... )
    """
    views = []
    
    # Add absolute views
    if absolute_views:
        for asset_idx, weight in absolute_views.items():
            if asset_idx >= n_assets or asset_idx < 0:
                raise ValueError(f"Asset index {asset_idx} out of range [0, {n_assets})")
            view = np.zeros(n_assets)
            view[asset_idx] = weight
            views.append(view)
    
    # Add relative views
    if relative_views:
        for asset_i, asset_j, weight_i, weight_j in relative_views:
            if asset_i >= n_assets or asset_i < 0:
                raise ValueError(f"Asset index {asset_i} out of range [0, {n_assets})")
            if asset_j >= n_assets or asset_j < 0:
                raise ValueError(f"Asset index {asset_j} out of range [0, {n_assets})")
            view = np.zeros(n_assets)
            view[asset_i] = weight_i
            view[asset_j] = weight_j
            views.append(view)
    
    if not views:
        raise ValueError("No views specified")
    
    return np.array(views)


if __name__ == "__main__":
    # Example usage from Meucci's paper (Section 2, international stock fund)
    print("=" * 80)
    print("Black-Litterman Model - Example from Meucci (2008)")
    print("=" * 80)
    
    # Market data (6 assets: Italy, Spain, Switzerland, Canada, US, Germany)
    # Annualized volatilities
    volatilities = np.array([0.21, 0.24, 0.24, 0.25, 0.29, 0.31])
    
    # Correlation matrix (equation 9)
    C = np.array([
        [1.00, 0.54, 0.62, 0.25, 0.41, 0.59],
        [0.54, 1.00, 0.69, 0.29, 0.36, 0.83],
        [0.62, 0.69, 1.00, 0.15, 0.46, 0.65],
        [0.25, 0.29, 0.15, 1.00, 0.47, 0.39],
        [0.41, 0.36, 0.46, 0.47, 1.00, 0.38],
        [0.59, 0.83, 0.65, 0.39, 0.38, 1.00]
    ])
    
    # Convert to covariance matrix
    D = np.diag(volatilities)
    Sigma = D @ C @ D
    
    # Market equilibrium weights
    w_eq = np.array([0.04, 0.04, 0.05, 0.08, 0.71, 0.08])
    
    # Compute equilibrium returns (using risk aversion = 2.5)
    pi = BlackLittermanModel.compute_equilibrium_returns(w_eq, Sigma, risk_aversion=2.5)
    
    print("\nPrior (Equilibrium) Expected Returns:")
    print(f"  {pi * 100}%")
    print(f"  (Paper values: [6%, 7%, 9%, 8%, 17%, 10%])")
    
    # Initialize model with market formulation (recommended)
    bl_market = BlackLittermanModel(pi, Sigma, tau=0.4, use_market_formulation=True)
    
    # Define views from the paper:
    # View 1: Spanish index will rise 12% (annualized)
    # View 2: US-Germany spread will be -10% (annualized)
    P = np.array([
        [0, 1, 0, 0, 0, 0],  # Spain
        [0, 0, 0, 0, 1, -1]  # US - Germany
    ])
    Q = np.array([0.12, -0.10])
    
    print("\nViews:")
    print("  View 1: Spain will return 12%")
    print("  View 2: US - Germany spread will be -10%")
    
    # Compute posterior with moderate confidence
    mu_bl, sigma_bl = bl_market.compute_posterior(P, Q, confidence=1.0)
    
    print("\nPosterior Expected Returns (Market Formulation):")
    print(f"  {mu_bl * 100}%")
    
    # Validate posterior
    validation = bl_market.validate_posterior(sigma_bl)
    print("\nPosterior Validation:")
    print(f"  Symmetric: {validation['is_symmetric']}")
    print(f"  Positive Semi-Definite: {validation['is_psd']}")
    print(f"  Min Eigenvalue: {validation['min_eigenvalue']:.6f}")
    if validation['warnings']:
        print("  Warnings:")
        for warning in validation['warnings']:
            print(f"    - {warning}")
    
    # Compare with original formulation
    bl_original = BlackLittermanModel(pi, Sigma, tau=0.4, use_market_formulation=False)
    mu_bl_orig, sigma_bl_orig = bl_original.compute_posterior(P, Q, confidence=1.0)
    
    print("\nPosterior Expected Returns (Original Formulation):")
    print(f"  {mu_bl_orig * 100}%")
    
    print("\nDifference in Expected Returns:")
    print(f"  {(mu_bl - mu_bl_orig) * 100}%")
    
    print("\n" + "=" * 80)
