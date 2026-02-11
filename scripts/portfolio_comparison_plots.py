"""
========================================================================================
PORTFOLIO COMPARISON: Market-Cap vs Mean-Variance vs Sentiment Black-Litterman
Interactive Plotly Visualization
========================================================================================

This script compares three portfolio strategies:
1. Market-Cap Weighted (Benchmark)
2. Classic Mean-Variance (Markowitz - often unstable)
3. Sentiment-Enhanced Black-Litterman (Our approach)

Features:
- Interactive Plotly charts
- Efficient frontier visualization
- Rolling performance comparison
- Risk-return scatter
- Weight evolution over time

========================================================================================
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings

# Import our framework (assumes previous code is available)
# For standalone use, you can paste the classes here

warnings.filterwarnings('ignore')

# ============================================================================
# PART 1: PORTFOLIO STRATEGIES
# ============================================================================

class PortfolioComparison:
    """Compare different portfolio optimization strategies."""
    
    def __init__(self, tickers: List[str], lookback_days: int = 252):
        """
        Initialize portfolio comparison.
        
        Parameters:
            tickers: List of stock tickers
            lookback_days: Historical data period
        """
        self.tickers = tickers
        self.N = len(tickers)
        self.lookback_days = lookback_days
        
        # Data containers
        self.prices_df = None
        self.returns_df = None
        self.mu = None  # Historical mean returns
        self.sigma = None  # Covariance matrix
        
        # Portfolio weights
        self.w_market_cap = None
        self.w_mean_variance = None
        self.w_black_litterman = None
        
    def load_data(self, end_date: datetime = None):
        """Load historical market data."""
        if end_date is None:
            end_date = datetime.now()
        
        start_date = end_date - timedelta(days=self.lookback_days + 30)
        
        print(f"Loading data for {len(self.tickers)} tickers...")
        print(f"Period: {start_date.date()} to {end_date.date()}")
        
        # Download data
        data = yf.download(
            self.tickers,
            start=start_date,
            end=end_date,
            group_by='ticker',
            progress=False
        )
        
        # Extract prices
        prices_dict = {}
        for ticker in self.tickers:
            try:
                if len(self.tickers) == 1:
                    prices_dict[ticker] = data['Close']
                else:
                    prices_dict[ticker] = data[ticker]['Close']
                print(f"  ✓ {ticker}: {len(prices_dict[ticker])} days")
            except Exception as e:
                print(f"  ✗ {ticker}: {e}")
        
        # Create DataFrame
        self.prices_df = pd.DataFrame(prices_dict).dropna()
        self.returns_df = np.log(self.prices_df / self.prices_df.shift(1)).dropna()
        
        print(f"\nFinal dataset: {len(self.returns_df)} days × {len(self.returns_df.columns)} assets")
        
        # Estimate parameters
        self.mu = self.returns_df.mean().values * 252  # Annualized
        self.sigma = self.returns_df.cov().values * 252  # Annualized
        self.sigma += np.eye(self.N) * 1e-6  # Regularization
        
        return self
    
    def compute_market_cap_weights(self):
        """
        Strategy 1: Market-Cap Weighted (Benchmark)
        
        Equal-weighted as proxy (real market-cap requires additional data).
        In practice, you'd use actual market capitalizations.
        """
        self.w_market_cap = np.ones(self.N) / self.N
        
        print("\n" + "="*80)
        print("STRATEGY 1: MARKET-CAP WEIGHTED (Benchmark)")
        print("="*80)
        print("\nWeights (equal-weighted proxy):")
        for i, ticker in enumerate(self.tickers):
            print(f"  {ticker}: {self.w_market_cap[i]*100:5.1f}%")
        
        return self.w_market_cap
    
    def compute_mean_variance_weights(self, risk_aversion: float = 2.5):
        """
        Strategy 2: Classic Mean-Variance (Markowitz)
        
        This is often UNSTABLE due to estimation error in μ.
        
        Formula: w* = (1/2λ) Σ^(-1) μ
        """
        print("\n" + "="*80)
        print("STRATEGY 2: MEAN-VARIANCE (Markowitz - Often Unstable)")
        print("="*80)
        
        try:
            sigma_inv = np.linalg.inv(self.sigma)
        except:
            print("   Singular covariance, using pseudo-inverse")
            sigma_inv = np.linalg.pinv(self.sigma)
        
        # Unconstrained solution
        w_mv = sigma_inv @ self.mu / (2 * risk_aversion)
        
        # Check for extreme weights (sign of instability)
        if np.any(np.abs(w_mv) > 2):
            print("   INSTABILITY DETECTED: Extreme weights (|w| > 200%)")
            print("     This is typical for mean-variance optimization!")
        
        # Long-only constraint
        w_mv = np.maximum(w_mv, 0)
        
        # Check if all weights are zero (another instability sign)
        if np.sum(w_mv) < 0.01:
            print("   INSTABILITY: All weights near zero, using equal weights")
            w_mv = np.ones(self.N) / self.N
        else:
            w_mv = w_mv / np.sum(w_mv)
        
        self.w_mean_variance = w_mv
        
        print("\nWeights:")
        for i, ticker in enumerate(self.tickers):
            print(f"  {ticker}: {self.w_mean_variance[i]*100:5.1f}%")
        
        # Show concentration
        concentration = np.max(self.w_mean_variance)
        print(f"\nConcentration (max weight): {concentration*100:.1f}%")
        if concentration > 0.6:
            print("   High concentration - sign of instability")
        
        return self.w_mean_variance
    
    def compute_black_litterman_weights(
        self,
        sentiment_data: Dict,
        risk_aversion: float = 2.5,
        sentiment_scaling: float = 0.02
    ):
        """
        Strategy 3: Sentiment-Enhanced Black-Litterman
        
        This is STABLE and incorporates views from sentiment.
        """
        print("\n" + "="*80)
        print("STRATEGY 3: SENTIMENT BLACK-LITTERMAN (Stable)")
        print("="*80)
        
        # Import classes (assuming they're available)
        from complete_bl_finbert_colab_FINAL import (
            BlackLittermanModel, ViewGenerator, SentimentData
        )
        
        # Equilibrium returns
        pi = BlackLittermanModel.compute_equilibrium_returns(
            self.w_market_cap, self.sigma, risk_aversion
        )
        
        print("\nEquilibrium Returns:")
        for i, ticker in enumerate(self.tickers):
            print(f"  {ticker}: {pi[i]*100:5.1f}%")
        
        # Volatilities for view calibration
        volatilities = np.sqrt(np.diag(self.sigma))
        
        # Generate views
        generator = ViewGenerator(
            tickers=self.tickers,
            volatilities=volatilities,
            sentiment_scaling=sentiment_scaling
        )
        
        view = generator.generate_views(
            sentiment_data,
            prior_returns=pi,
            filter_weak_signals=True,
            min_abs_sentiment=0.10
        )
        
        print(f"\n✓ Generated {view.P.shape[0]} views from sentiment")
        
        # Black-Litterman posterior
        bl = BlackLittermanModel(pi, self.sigma)
        mu_bl, sigma_bl = bl.compute_posterior(view.P, view.Q, view.Omega)
        
        print("\nPosterior Returns:")
        for i, ticker in enumerate(self.tickers):
            print(f"  {ticker}: {mu_bl[i]*100:5.1f}% (Δ{(mu_bl[i]-pi[i])*100:+.1f}%)")
        
        # Optimize
        try:
            sigma_bl_inv = np.linalg.inv(sigma_bl)
        except:
            sigma_bl_inv = np.linalg.pinv(sigma_bl)
        
        w_bl = sigma_bl_inv @ mu_bl / (2 * risk_aversion)
        w_bl = np.maximum(w_bl, 0)
        w_bl = w_bl / np.sum(w_bl)
        
        self.w_black_litterman = w_bl
        
        print("\nWeights:")
        for i, ticker in enumerate(self.tickers):
            print(f"  {ticker}: {self.w_black_litterman[i]*100:5.1f}%")
        
        return self.w_black_litterman
    
    def compute_portfolio_statistics(self, weights: np.ndarray) -> Dict:
        """Compute portfolio statistics."""
        portfolio_return = weights @ self.mu
        portfolio_vol = np.sqrt(weights @ self.sigma @ weights)
        sharpe = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
        
        return {
            'return': portfolio_return,
            'volatility': portfolio_vol,
            'sharpe': sharpe,
            'weights': weights
        }
    
    def create_comparison_table(self) -> pd.DataFrame:
        """Create comparison table of all strategies."""
        strategies = {
            'Market-Cap': self.w_market_cap,
            'Mean-Variance': self.w_mean_variance,
            'Black-Litterman': self.w_black_litterman
        }
        
        data = []
        for strategy_name, weights in strategies.items():
            if weights is not None:
                stats = self.compute_portfolio_statistics(weights)
                data.append({
                    'Strategy': strategy_name,
                    'Expected Return': f"{stats['return']*100:.2f}%",
                    'Volatility': f"{stats['volatility']*100:.2f}%",
                    'Sharpe Ratio': f"{stats['sharpe']:.3f}"
                })
        
        return pd.DataFrame(data)


# ============================================================================
# PART 2: VISUALIZATION FUNCTIONS
# ============================================================================

def create_efficient_frontier_plot(comparison: PortfolioComparison):
    """
    Create efficient frontier plot with all three strategies.
    
    Shows:
    - Efficient frontier curve
    - Three portfolio strategies as points
    - Individual assets
    """
    print("\nGenerating efficient frontier plot...")
    
    # Generate efficient frontier
    n_points = 50
    target_returns = np.linspace(
        np.min(comparison.mu) * 0.8,
        np.max(comparison.mu) * 1.2,
        n_points
    )
    
    frontier_vols = []
    for target_ret in target_returns:
        # Minimize variance subject to target return
        try:
            # Simplified: scan over possible portfolios
            # In practice, use quadratic programming
            best_vol = float('inf')
            for _ in range(100):
                w = np.random.dirichlet(np.ones(comparison.N))
                ret = w @ comparison.mu
                if abs(ret - target_ret) < 0.01:
                    vol = np.sqrt(w @ comparison.sigma @ w)
                    if vol < best_vol:
                        best_vol = vol
            frontier_vols.append(best_vol if best_vol < float('inf') else np.nan)
        except:
            frontier_vols.append(np.nan)
    
    # Create plot
    fig = go.Figure()
    
    # Efficient frontier
    fig.add_trace(go.Scatter(
        x=np.array(frontier_vols) * 100,
        y=target_returns * 100,
        mode='lines',
        name='Efficient Frontier',
        line=dict(color='lightgray', width=2, dash='dash')
    ))
    
    # Individual assets
    asset_vols = np.sqrt(np.diag(comparison.sigma))
    fig.add_trace(go.Scatter(
        x=asset_vols * 100,
        y=comparison.mu * 100,
        mode='markers',
        name='Individual Assets',
        marker=dict(size=12, color='lightblue', symbol='diamond'),
        text=comparison.tickers,
        hovertemplate='%{text}<br>Vol: %{x:.1f}%<br>Return: %{y:.1f}%'
    ))
    
    # Portfolio strategies
    strategies = [
        ('Market-Cap', comparison.w_market_cap, 'green', 'circle'),
        ('Mean-Variance', comparison.w_mean_variance, 'orange', 'square'),
        ('Black-Litterman', comparison.w_black_litterman, 'red', 'star')
    ]
    
    for name, weights, color, symbol in strategies:
        if weights is not None:
            stats = comparison.compute_portfolio_statistics(weights)
            fig.add_trace(go.Scatter(
                x=[stats['volatility'] * 100],
                y=[stats['return'] * 100],
                mode='markers',
                name=name,
                marker=dict(size=15, color=color, symbol=symbol, line=dict(width=2, color='white')),
                hovertemplate=f'{name}<br>Vol: %{{x:.2f}}%<br>Return: %{{y:.2f}}%<br>Sharpe: {stats["sharpe"]:.3f}'
            ))
    
    # Layout
    fig.update_layout(
        title='Efficient Frontier: Portfolio Strategies Comparison',
        xaxis_title='Volatility (Annualized %)',
        yaxis_title='Expected Return (Annualized %)',
        hovermode='closest',
        template='plotly_white',
        height=600,
        showlegend=True,
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)')
    )
    
    return fig


def create_weights_comparison_plot(comparison: PortfolioComparison):
    """
    Bar chart comparing portfolio weights across strategies.
    """
    print("\nGenerating weights comparison plot...")
    
    fig = go.Figure()
    
    strategies = [
        ('Market-Cap', comparison.w_market_cap, 'green'),
        ('Mean-Variance', comparison.w_mean_variance, 'orange'),
        ('Black-Litterman', comparison.w_black_litterman, 'red')
    ]
    
    x_positions = np.arange(comparison.N)
    bar_width = 0.25
    
    for i, (name, weights, color) in enumerate(strategies):
        if weights is not None:
            fig.add_trace(go.Bar(
                x=x_positions + i * bar_width,
                y=weights * 100,
                name=name,
                marker_color=color,
                width=bar_width,
                hovertemplate='%{y:.1f}%'
            ))
    
    fig.update_layout(
        title='Portfolio Weights Comparison',
        xaxis_title='Assets',
        yaxis_title='Weight (%)',
        xaxis=dict(
            tickmode='array',
            tickvals=x_positions + bar_width,
            ticktext=comparison.tickers
        ),
        template='plotly_white',
        height=500,
        barmode='group',
        showlegend=True
    )
    
    return fig


def create_risk_return_metrics_plot(comparison: PortfolioComparison):
    """
    Create grouped bar chart for return, volatility, and Sharpe.
    """
    print("\nGenerating risk-return metrics plot...")
    
    strategies = [
        ('Market-Cap', comparison.w_market_cap),
        ('Mean-Variance', comparison.w_mean_variance),
        ('Black-Litterman', comparison.w_black_litterman)
    ]
    
    strategy_names = []
    returns = []
    vols = []
    sharpes = []
    
    for name, weights in strategies:
        if weights is not None:
            stats = comparison.compute_portfolio_statistics(weights)
            strategy_names.append(name)
            returns.append(stats['return'] * 100)
            vols.append(stats['volatility'] * 100)
            sharpes.append(stats['sharpe'])
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Expected Return', 'Volatility', 'Sharpe Ratio'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}, {'type': 'bar'}]]
    )
    
    # Returns
    fig.add_trace(
        go.Bar(
            x=strategy_names,
            y=returns,
            marker_color=['green', 'orange', 'red'],
            name='Return',
            showlegend=False,
            hovertemplate='%{y:.2f}%'
        ),
        row=1, col=1
    )
    
    # Volatility
    fig.add_trace(
        go.Bar(
            x=strategy_names,
            y=vols,
            marker_color=['green', 'orange', 'red'],
            name='Volatility',
            showlegend=False,
            hovertemplate='%{y:.2f}%'
        ),
        row=1, col=2
    )
    
    # Sharpe
    fig.add_trace(
        go.Bar(
            x=strategy_names,
            y=sharpes,
            marker_color=['green', 'orange', 'red'],
            name='Sharpe',
            showlegend=False,
            hovertemplate='%{y:.3f}'
        ),
        row=1, col=3
    )
    
    fig.update_layout(
        title_text='Portfolio Performance Metrics Comparison',
        template='plotly_white',
        height=400,
        showlegend=False
    )
    
    fig.update_yaxes(title_text='Return (%)', row=1, col=1)
    fig.update_yaxes(title_text='Volatility (%)', row=1, col=2)
    fig.update_yaxes(title_text='Sharpe Ratio', row=1, col=3)
    
    return fig


def create_comprehensive_dashboard(comparison: PortfolioComparison):
    """
    Create comprehensive dashboard with multiple visualizations.
    """
    print("\nGenerating comprehensive dashboard...")
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Efficient Frontier',
            'Portfolio Weights',
            'Expected Returns',
            'Risk Metrics'
        ),
        specs=[
            [{'type': 'scatter'}, {'type': 'bar'}],
            [{'type': 'bar'}, {'type': 'bar'}]
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # 1. Efficient Frontier (simplified)
    strategies = [
        ('Market-Cap', comparison.w_market_cap, 'green', 'circle'),
        ('Mean-Variance', comparison.w_mean_variance, 'orange', 'square'),
        ('Black-Litterman', comparison.w_black_litterman, 'red', 'star')
    ]
    
    for name, weights, color, symbol in strategies:
        if weights is not None:
            stats = comparison.compute_portfolio_statistics(weights)
            fig.add_trace(
                go.Scatter(
                    x=[stats['volatility'] * 100],
                    y=[stats['return'] * 100],
                    mode='markers',
                    name=name,
                    marker=dict(size=12, color=color, symbol=symbol),
                    showlegend=True
                ),
                row=1, col=1
            )
    
    # 2. Weights comparison
    x_positions = np.arange(comparison.N)
    for i, (name, weights, color, _) in enumerate(strategies):
        if weights is not None:
            fig.add_trace(
                go.Bar(
                    x=comparison.tickers,
                    y=weights * 100,
                    name=name,
                    marker_color=color,
                    showlegend=False
                ),
                row=1, col=2
            )
    
    # 3. Expected Returns
    returns_list = []
    names_list = []
    colors_list = []
    
    for name, weights, color, _ in strategies:
        if weights is not None:
            stats = comparison.compute_portfolio_statistics(weights)
            returns_list.append(stats['return'] * 100)
            names_list.append(name)
            colors_list.append(color)
    
    fig.add_trace(
        go.Bar(
            x=names_list,
            y=returns_list,
            marker_color=colors_list,
            showlegend=False
        ),
        row=2, col=1
    )
    
    # 4. Sharpe Ratios
    sharpe_list = []
    for name, weights, color, _ in strategies:
        if weights is not None:
            stats = comparison.compute_portfolio_statistics(weights)
            sharpe_list.append(stats['sharpe'])
    
    fig.add_trace(
        go.Bar(
            x=names_list,
            y=sharpe_list,
            marker_color=colors_list,
            showlegend=False
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_xaxes(title_text='Volatility (%)', row=1, col=1)
    fig.update_yaxes(title_text='Return (%)', row=1, col=1)
    
    fig.update_xaxes(title_text='Assets', row=1, col=2)
    fig.update_yaxes(title_text='Weight (%)', row=1, col=2)
    
    fig.update_yaxes(title_text='Return (%)', row=2, col=1)
    fig.update_yaxes(title_text='Sharpe Ratio', row=2, col=2)
    
    fig.update_layout(
        title_text='Portfolio Strategies: Comprehensive Comparison Dashboard',
        template='plotly_white',
        height=800,
        showlegend=True,
        legend=dict(x=0.02, y=0.98)
    )
    
    return fig


# ============================================================================
# PART 3: MAIN EXECUTION
# ============================================================================

def run_portfolio_comparison(
    tickers: List[str],
    sentiment_data: Dict,
    save_html: bool = True
):
    """
    Run complete portfolio comparison.
    
    Parameters:
        tickers: List of stock tickers
        sentiment_data: Sentiment data for Black-Litterman
        save_html: Save interactive plots as HTML
    
    Returns:
        PortfolioComparison object with results
    """
    print("="*80)
    print("PORTFOLIO STRATEGIES COMPARISON")
    print("="*80)
    print(f"Tickers: {', '.join(tickers)}")
    print("="*80)
    
    # Initialize
    comparison = PortfolioComparison(tickers)
    
    # Load data
    comparison.load_data()
    
    # Compute all strategies
    comparison.compute_market_cap_weights()
    comparison.compute_mean_variance_weights(risk_aversion=2.5)
    comparison.compute_black_litterman_weights(
        sentiment_data=sentiment_data,
        risk_aversion=2.5,
        sentiment_scaling=0.02
    )
    
    # Print comparison table
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON")
    print("="*80)
    table = comparison.create_comparison_table()
    print("\n" + table.to_string(index=False))
    print("="*80)
    
    # Create visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    # 1. Efficient Frontier
    fig1 = create_efficient_frontier_plot(comparison)
    fig1.show()
    if save_html:
        fig1.write_html("efficient_frontier_comparison.html")
        print("\n✓ Saved: efficient_frontier_comparison.html")
    
    # 2. Weights Comparison
    fig2 = create_weights_comparison_plot(comparison)
    fig2.show()
    if save_html:
        fig2.write_html("weights_comparison.html")
        print("✓ Saved: weights_comparison.html")
    
    # 3. Risk-Return Metrics
    fig3 = create_risk_return_metrics_plot(comparison)
    fig3.show()
    if save_html:
        fig3.write_html("risk_return_metrics.html")
        print("✓ Saved: risk_return_metrics.html")
    
    # 4. Comprehensive Dashboard
    fig4 = create_comprehensive_dashboard(comparison)
    fig4.show()
    if save_html:
        fig4.write_html("portfolio_dashboard.html")
        print("✓ Saved: portfolio_dashboard.html")
    
    print("\n" + "="*80)
    print("✓ VISUALIZATION COMPLETE")
    print("="*80)
    
    return comparison


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Example: Run portfolio comparison with simulated sentiment.
    """
    
    # Portfolio universe
    TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    
    # Simulated sentiment data (in practice, use real FinBERT output)
    from complete_bl_finbert_colab_FINAL import SentimentData
    
    np.random.seed(42)
    sentiment_data = {
        'AAPL': SentimentData('AAPL', 0.45, 0.18, 20, []),
        'MSFT': SentimentData('MSFT', 0.35, 0.15, 18, []),
        'GOOGL': SentimentData('GOOGL', 0.25, 0.22, 15, []),
        'TSLA': SentimentData('TSLA', -0.20, 0.30, 12, []),
        'NVDA': SentimentData('NVDA', 0.60, 0.12, 25, [])
    }
    
    # Run comparison
    comparison = run_portfolio_comparison(
        tickers=TICKERS,
        sentiment_data=sentiment_data,
        save_html=True
    )
    
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    print("""
1. MARKET-CAP (Green):
   - Simple, transparent benchmark
   - Equal weights (or market-cap in production)
   - Moderate risk-return

2. MEAN-VARIANCE (Orange):
   - Often UNSTABLE due to estimation error
   - Extreme weights or high concentration
   - Sensitive to input changes

3. BLACK-LITTERMAN (Red):
   - STABLE through Bayesian shrinkage
   - Incorporates sentiment views
   - Better risk-adjusted returns
   - More diversified than Mean-Variance
    """)
