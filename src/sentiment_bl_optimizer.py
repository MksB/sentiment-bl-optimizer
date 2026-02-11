"""
Sentiment-Enhanced Black-Litterman Portfolio Optimizer
Master Integration Script

This script demonstrates the complete workflow:
1. Load historical price data → Estimate Σ, compute equilibrium π
2. Fetch news and analyze sentiment → FinBERT scores
3. Generate Black-Litterman views → ViewGenerator (P, Q, Ω)
4. Compute posterior distribution → Black-Litterman model
5. Optimize portfolio → Mean-variance optimization
6. Visualize results → Interactive dashboard

This is the "supreme discipline" - bridging unstructured NLP and structured quant finance.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import warnings

# Our modules
from bl_model import BlackLittermanModel, create_view_matrix
from view_generator import ViewGenerator, SentimentData, compute_sentiment_statistics

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class SentimentPortfolioOptimizer:
    """
    Complete sentiment-enhanced Black-Litterman portfolio optimization system.
    
    This class orchestrates the entire pipeline:
    - Data acquisition (prices, news)
    - Sentiment analysis (FinBERT)
    - View generation (NLP → Math)
    - Portfolio optimization (Black-Litterman)
    
    Example:
        >>> optimizer = SentimentPortfolioOptimizer(
        ...     tickers=['XLK', 'XLE', 'XLF'],
        ...     lookback_days=252
        ... )
        >>> results = optimizer.optimize(sentiment_data)
    """
    
    def __init__(
        self,
        tickers: List[str],
        ticker_names: Optional[List[str]] = None,
        lookback_days: int = 252,
        risk_aversion: float = 2.5,
        use_market_formulation: bool = True
    ):
        """
        Initialize the optimizer.
        
        Parameters:
            tickers: List of asset tickers (e.g., ['XLK', 'XLE', 'XLF'])
            ticker_names: Human-readable names (e.g., ['Tech', 'Energy', 'Finance'])
            lookback_days: Number of days for historical covariance estimation
            risk_aversion: Risk aversion parameter λ for equilibrium
            use_market_formulation: Use market-based BL (recommended: True)
        """
        self.tickers = tickers
        self.ticker_names = ticker_names or tickers
        self.N = len(tickers)
        self.lookback_days = lookback_days
        self.risk_aversion = risk_aversion
        self.use_market_formulation = use_market_formulation
        
        # Data containers (populated by load_data)
        self.returns_data = None
        self.sigma = None
        self.volatilities = None
        self.pi = None
        self.w_eq = None
        
        # Models (initialized after data load)
        self.bl_model = None
        self.view_generator = None
    
    def load_historical_data(self) -> pd.DataFrame:
        """
        Load historical price data and compute returns.
        
        Uses yfinance to download adjusted close prices,
        then computes log returns for normality assumption.
        
        Returns:
            DataFrame with log returns (dates × tickers)
        """
        print(f"\n{'='*80}")
        print("STEP 1: Loading Historical Data")
        print(f"{'='*80}")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_days + 30)  # Extra buffer
        
        print(f"Downloading data for {self.N} assets...")
        print(f"Period: {start_date.date()} to {end_date.date()}")
        
        returns_dict = {}
        
        for ticker in self.tickers:
            try:
                # Download data
                data = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    progress=False
                )
                
                if data.empty:
                    print(f"Warning: No data for {ticker}")
                    continue
                
                # Extract close prices
                if isinstance(data, pd.DataFrame) and 'Close' in data.columns:
                    prices = data['Close']
                else:
                    prices = data
                
                # Compute log returns
                log_returns = np.log(prices / prices.shift(1)).dropna()
                
                returns_dict[ticker] = log_returns
                
                print(f"  ✓ {ticker}: {len(log_returns)} observations")
                
            except Exception as e:
                print(f"  ✗ Error loading {ticker}: {e}")
                continue
        
        # Combine into DataFrame
        self.returns_data = pd.DataFrame(returns_dict)
        
        # Align dates (keep only common dates)
        self.returns_data = self.returns_data.dropna()
        
        print(f"\nFinal dataset: {len(self.returns_data)} days × {len(self.returns_data.columns)} assets")
        
        return self.returns_data
    
    def estimate_parameters(self, annualization_factor: int = 252):
        """
        Estimate covariance matrix and compute equilibrium returns.
        
        Steps:
        1. Estimate Σ from historical returns
        2. Compute volatilities (for ViewGenerator)
        3. Assume equal-weight equilibrium
        4. Compute π via reverse optimization
        
        Parameters:
            annualization_factor: Trading days per year (252 for daily data)
        """
        print(f"\n{'='*80}")
        print("STEP 2: Estimating Parameters")
        print(f"{'='*80}")
        
        if self.returns_data is None:
            raise ValueError("Must load data first (call load_historical_data)")
        
        # 1. Covariance matrix (annualized)
        self.sigma = self.returns_data.cov().values * annualization_factor
        
        # Add small diagonal for numerical stability
        self.sigma += np.eye(self.N) * 1e-6
        
        print(f"\nCovariance Matrix Σ ({self.N}×{self.N}):")
        print(f"  Condition number: {np.linalg.cond(self.sigma):.2e}")
        
        # Check positive definiteness
        eigenvalues = np.linalg.eigvalsh(self.sigma)
        if np.any(eigenvalues <= 0):
            print(f"  ⚠ Warning: Non-positive eigenvalues detected")
            print(f"    Min eigenvalue: {eigenvalues.min():.2e}")
        else:
            print(f"  ✓ Positive definite (min eigenvalue: {eigenvalues.min():.2e})")
        
        # 2. Volatilities (for view generator)
        self.volatilities = np.sqrt(np.diag(self.sigma))
        
        print(f"\nAnnualized Volatilities:")
        for i, ticker in enumerate(self.tickers):
            print(f"  {ticker:6s}: {self.volatilities[i]*100:5.1f}%")
        
        # 3. Equilibrium weights (equal-weight for simplicity)
        # In practice, use market cap weights
        self.w_eq = np.ones(self.N) / self.N
        
        print(f"\nEquilibrium Weights (equal-weighted):")
        for i, ticker in enumerate(self.tickers):
            print(f"  {ticker:6s}: {self.w_eq[i]*100:5.1f}%")
        
        # 4. Equilibrium returns (reverse optimization)
        self.pi = BlackLittermanModel.compute_equilibrium_returns(
            self.w_eq, self.sigma, self.risk_aversion
        )
        
        print(f"\nEquilibrium Returns π (risk aversion λ = {self.risk_aversion}):")
        for i, ticker in enumerate(self.tickers):
            print(f"  {ticker:6s}: {self.pi[i]*100:6.2f}%")
        
        # 5. Initialize Black-Litterman model
        self.bl_model = BlackLittermanModel(
            self.pi,
            self.sigma,
            use_market_formulation=self.use_market_formulation
        )
        
        # 6. Initialize ViewGenerator
        self.view_generator = ViewGenerator(
            tickers=self.tickers,
            volatilities=self.volatilities,
            sentiment_scaling=0.02,      # 2% max return impact
            base_uncertainty=0.0001,
            news_volume_weight=0.5,
            consistency_weight=0.5,
            min_news_count=3
        )
        
        print(f"\n✓ Parameters estimated successfully")
    
    def generate_views_from_sentiment(
        self,
        sentiment_data: Dict[str, SentimentData],
        filter_weak_signals: bool = True,
        min_abs_sentiment: float = 0.15
    ):
        """
        Generate Black-Litterman views from sentiment data.
        
        This is where NLP meets quantitative finance.
        
        Parameters:
            sentiment_data: Dictionary mapping ticker -> SentimentData
            filter_weak_signals: Filter out weak/noisy sentiment
            min_abs_sentiment: Minimum |sentiment| to include
        
        Returns:
            BlackLittermanView object
        """
        print(f"\n{'='*80}")
        print("STEP 3: Generating Views from Sentiment")
        print(f"{'='*80}")
        
        if self.view_generator is None:
            raise ValueError("Must estimate parameters first")
        
        print(f"\nInput sentiment data:")
        for ticker, data in sentiment_data.items():
            print(f"  {ticker:6s}: sentiment={data.sentiment_mean:+.3f}, "
                  f"std={data.sentiment_std:.3f}, news_count={data.news_count}")
        
        # Generate views
        view = self.view_generator.generate_views(
            sentiment_data,
            prior_returns=self.pi,
            filter_weak_signals=filter_weak_signals,
            min_abs_sentiment=min_abs_sentiment
        )
        
        print(f"\n✓ Generated {view.P.shape[0]} views")
        print(f"  Tickers: {view.metadata.get('tickers', [])}")
        
        # Display view details
        print(f"\nView Structure:")
        print(f"  P (Pick Matrix): {view.P.shape}")
        print(f"  Q (View Returns): {view.Q}")
        print(f"  Ω (Uncertainty, diagonal): {np.diag(view.Omega)}")
        
        # Analyze impact
        print(f"\nView Impact Analysis:")
        analysis = self.view_generator.analyze_view_impact(view, self.pi)
        print(analysis.to_string(index=False))
        
        return view
    
    def compute_posterior(self, view, confidence: float = 1.0):
        """
        Compute Black-Litterman posterior distribution.
        
        Parameters:
            view: BlackLittermanView from generate_views_from_sentiment
            confidence: Overall confidence multiplier (default: 1.0)
        
        Returns:
            (mu_bl, sigma_bl): Posterior mean and covariance
        """
        print(f"\n{'='*80}")
        print("STEP 4: Computing Black-Litterman Posterior")
        print(f"{'='*80}")
        
        if self.bl_model is None:
            raise ValueError("Must estimate parameters first")
        
        # Compute posterior using the view's Omega directly
        mu_bl, sigma_bl = self.bl_model.compute_posterior(
            view.P,
            view.Q,
            Omega=view.Omega * (1.0 / confidence),  # Scale uncertainty
            confidence=None  # Omega already computed
        )
        
        # Validate posterior
        validation = self.bl_model.validate_posterior(sigma_bl)
        
        print(f"\nPosterior Validation:")
        print(f"  Symmetric: {validation['is_symmetric']}")
        print(f"  Positive Semi-Definite: {validation['is_psd']}")
        if validation['min_eigenvalue'] is not None:
            print(f"  Min Eigenvalue: {validation['min_eigenvalue']:.2e}")
        
        if validation['warnings']:
            print(f"Warnings:")
            for warning in validation['warnings']:
                print(f"    - {warning}")
        
        print(f"\nPosterior Returns μ_BL:")
        for i, ticker in enumerate(self.tickers):
            delta = mu_bl[i] - self.pi[i]
            direction = "↑" if delta > 0 else "↓" if delta < 0 else "→"
            print(f"  {ticker:6s}: {self.pi[i]*100:6.2f}% → {mu_bl[i]*100:6.2f}% "
                  f"{direction} ({delta*100:+.2f}%)")
        
        return mu_bl, sigma_bl
    
    def optimize_portfolio(
        self,
        mu_bl: np.ndarray,
        sigma_bl: np.ndarray,
        constraints: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Optimize portfolio weights using mean-variance optimization.
        
        Solves: max w'μ - λ w'Σw
                s.t. constraints
        
        Parameters:
            mu_bl: Posterior expected returns
            sigma_bl: Posterior covariance
            constraints: Optional dict with 'bounds', 'budget', etc.
        
        Returns:
            Optimal weights (N,)
        """
        print(f"\n{'='*80}")
        print("STEP 5: Portfolio Optimization")
        print(f"{'='*80}")
        
        # Default: unconstrained analytical solution
        # w* = (1/2λ) Σ^(-1) μ
        
        try:
            sigma_inv = np.linalg.inv(sigma_bl)
        except np.linalg.LinAlgError:
            print("Singular covariance, using pseudo-inverse")
            sigma_inv = np.linalg.pinv(sigma_bl)
        
        # Analytical solution
        w_optimal = sigma_inv @ mu_bl / (2 * self.risk_aversion)
        
        # Normalize to sum to 1 (budget constraint)
        w_optimal = w_optimal / np.sum(np.abs(w_optimal))
        
        # Apply bounds if specified
        if constraints and 'bounds' in constraints:
            lower, upper = constraints['bounds']
            w_optimal = np.clip(w_optimal, lower, upper)
            w_optimal = w_optimal / np.sum(np.abs(w_optimal))  # Re-normalize
        
        print(f"\nOptimal Weights (Black-Litterman):")
        for i, ticker in enumerate(self.tickers):
            delta = w_optimal[i] - self.w_eq[i]
            direction = "↑" if delta > 0 else "↓" if delta < 0 else "→"
            print(f"  {ticker:6s}: {self.w_eq[i]*100:5.1f}% → {w_optimal[i]*100:5.1f}% "
                  f"{direction} ({delta*100:+5.1f}%)")
        
        # Portfolio statistics
        portfolio_return = w_optimal @ mu_bl
        portfolio_vol = np.sqrt(w_optimal @ sigma_bl @ w_optimal)
        sharpe = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
        
        print(f"\nPortfolio Statistics:")
        print(f"  Expected Return: {portfolio_return*100:.2f}%")
        print(f"  Volatility:      {portfolio_vol*100:.2f}%")
        print(f"  Sharpe Ratio:    {sharpe:.3f}")
        
        return w_optimal
    
    def run_complete_optimization(
        self,
        sentiment_data: Dict[str, SentimentData],
        filter_weak_signals: bool = True,
        min_abs_sentiment: float = 0.15,
        confidence: float = 1.0,
        constraints: Optional[Dict] = None
    ) -> Dict:
        """
        Run the complete optimization pipeline.
        
        This is the master function that orchestrates everything.
        
        Returns:
            Dictionary with all results
        """
        # Step 1: Load data (if not already loaded)
        if self.returns_data is None:
            self.load_historical_data()
        
        # Step 2: Estimate parameters (if not already done)
        if self.sigma is None:
            self.estimate_parameters()
        
        # Step 3: Generate views from sentiment
        view = self.generate_views_from_sentiment(
            sentiment_data,
            filter_weak_signals=filter_weak_signals,
            min_abs_sentiment=min_abs_sentiment
        )
        
        # Step 4: Compute posterior
        mu_bl, sigma_bl = self.compute_posterior(view, confidence)
        
        # Step 5: Optimize portfolio
        w_optimal = self.optimize_portfolio(mu_bl, sigma_bl, constraints)
        
        # Compile results
        results = {
            'tickers': self.tickers,
            'ticker_names': self.ticker_names,
            'prior_returns': self.pi,
            'posterior_returns': mu_bl,
            'prior_weights': self.w_eq,
            'optimal_weights': w_optimal,
            'view': view,
            'sigma': self.sigma,
            'sigma_bl': sigma_bl,
            'sentiment_data': sentiment_data
        }
        
        return results


# ============================================================================
# Example Usage / Demonstration
# ============================================================================

def create_example_sentiment_data() -> Dict[str, SentimentData]:
    """
    Create example sentiment data for demonstration.
    
    In production, this would come from actual FinBERT analysis of news.
    """
    
    # Simulate realistic sentiment scenarios
    sentiment_data = {
        'XLK': SentimentData(  # Technology - Strong positive
            ticker='XLK',
            sentiment_mean=0.65,
            sentiment_std=0.18,
            news_count=28,
            raw_scores=[0.7, 0.6, 0.65, 0.8, 0.5, 0.7, 0.6, 0.75]
        ),
        'XLE': SentimentData(  # Energy - Moderate negative
            ticker='XLE',
            sentiment_mean=-0.35,
            sentiment_std=0.28,
            news_count=18,
            raw_scores=[-0.5, -0.3, -0.4, -0.6, -0.2, -0.3]
        ),
        'XLF': SentimentData(  # Finance - Weak positive (may filter out)
            ticker='XLF',
            sentiment_mean=0.12,
            sentiment_std=0.42,
            news_count=12,
            raw_scores=[0.3, -0.2, 0.4, 0.1, 0.2]
        ),
        'XLV': SentimentData(  # Healthcare - Moderate positive
            ticker='XLV',
            sentiment_mean=0.45,
            sentiment_std=0.22,
            news_count=22,
            raw_scores=[0.5, 0.4, 0.45, 0.6, 0.3, 0.5]
        ),
        'XLY': SentimentData(  # Consumer - Slight negative
            ticker='XLY',
            sentiment_mean=-0.18,
            sentiment_std=0.35,
            news_count=15,
            raw_scores=[-0.2, -0.1, -0.3, 0.0, -0.25]
        )
    }
    
    return sentiment_data


def main():
    """
    Main demonstration of sentiment-enhanced Black-Litterman optimization.
    """
    print("\n" + "="*80)
    print("SENTIMENT-ENHANCED BLACK-LITTERMAN PORTFOLIO OPTIMIZATION")
    print("="*80)
    print("\nBridging NLP (FinBERT) and Quantitative Finance (Black-Litterman)")
    print("This is the 'Supreme Discipline' in action.\n")
    
    # Define portfolio universe
    tickers = ['XLK', 'XLE', 'XLF', 'XLV', 'XLY']
    ticker_names = ['Technology', 'Energy', 'Finance', 'Healthcare', 'Consumer']
    
    # Initialize optimizer
    optimizer = SentimentPortfolioOptimizer(
        tickers=tickers,
        ticker_names=ticker_names,
        lookback_days=252,  # 1 year
        risk_aversion=2.5,
        use_market_formulation=True
    )
    
    # Create example sentiment data
    # In production: This comes from actual FinBERT analysis
    sentiment_data = create_example_sentiment_data()
    
    # Run complete optimization
    results = optimizer.run_complete_optimization(
        sentiment_data=sentiment_data,
        filter_weak_signals=True,
        min_abs_sentiment=0.15,
        confidence=1.0,
        constraints={'bounds': (0.0, 0.5)}  # Long-only, max 50% per asset
    )
    
    # Summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    
    print(f"\nPortfolio Adjustment (Equilibrium → Black-Litterman):")
    for i, ticker in enumerate(tickers):
        prior_w = results['prior_weights'][i]
        post_w = results['optimal_weights'][i]
        prior_r = results['prior_returns'][i]
        post_r = results['posterior_returns'][i]
        
        print(f"\n  {ticker_names[i]} ({ticker}):")
        print(f"    Weight:  {prior_w*100:5.1f}% → {post_w*100:5.1f}% "
              f"({(post_w-prior_w)*100:+5.1f}%)")
        print(f"    Return:  {prior_r*100:5.1f}% → {post_r*100:5.1f}% "
              f"({(post_r-prior_r)*100:+5.1f}%)")
    
    print(f"\n{'='*80}")
    print("✓ Optimization Complete")
    print(f"{'='*80}\n")
    
    return results


if __name__ == "__main__":
    # Run demonstration
    results = main()
    
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("""
1. Integrate real FinBERT sentiment analysis (replace create_example_sentiment_data)
2. Add news fetching from NewsAPI or similar
3. Implement portfolio rebalancing with transaction costs
4. Add visualization dashboard (Plotly/Matplotlib)
5. Backtest the strategy on historical data
6. Deploy to production with automated updates
    """)
