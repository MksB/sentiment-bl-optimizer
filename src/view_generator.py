"""
ViewGenerator: Bridge between NLP Sentiment and Quantitative Finance

This module translates unstructured sentiment data (from FinBERT) into the structured
mathematical framework of Black-Litterman portfolio optimization.

Mathematical Foundation:
- Pick Matrix P: Maps sentiment to assets
- View Vector Q: Sentiment-scaled expected returns
- Uncertainty Matrix Ω: Confidence based on news volume and consistency

Author: Implementation following Meucci (2008) framework
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings


@dataclass
class SentimentData:
    """
    Container for sentiment analysis results for a single asset.
    
    Attributes:
        ticker: Asset identifier (e.g., 'XLK', 'AAPL')
        sentiment_mean: Average sentiment score [-1, +1]
        sentiment_std: Standard deviation of sentiment scores
        news_count: Number of news articles analyzed
        raw_scores: Individual sentiment scores (for validation)
    """
    ticker: str
    sentiment_mean: float
    sentiment_std: float
    news_count: int
    raw_scores: List[float]
    
    def __post_init__(self):
        """Validate sentiment data"""
        if not -1.0 <= self.sentiment_mean <= 1.0:
            warnings.warn(
                f"Sentiment mean {self.sentiment_mean} outside [-1, 1] for {self.ticker}"
            )
        if self.sentiment_std < 0:
            raise ValueError(f"Sentiment std must be non-negative, got {self.sentiment_std}")
        if self.news_count < 0:
            raise ValueError(f"News count must be non-negative, got {self.news_count}")


@dataclass
class BlackLittermanView:
    """
    Complete Black-Litterman view specification.
    
    Attributes:
        P: Pick matrix (K × N) - which assets the views refer to
        Q: View vector (K,) - expected returns based on sentiment
        Omega: Uncertainty matrix (K × K) - confidence in each view
        metadata: Additional information for tracking/debugging
    """
    P: np.ndarray
    Q: np.ndarray
    Omega: np.ndarray
    metadata: Dict[str, any]
    
    def validate(self) -> Dict[str, bool]:
        """
        Validate the mathematical structure of the view.
        
        Returns:
            Dictionary with validation results
        """
        K, N = self.P.shape
        
        results = {
            'P_valid': True,
            'Q_valid': True,
            'Omega_valid': True,
            'dimensions_consistent': True,
        }
        
        # Check dimensions
        if len(self.Q) != K:
            results['dimensions_consistent'] = False
            results['Q_valid'] = False
        
        if self.Omega.shape != (K, K):
            results['dimensions_consistent'] = False
            results['Omega_valid'] = False
        
        # Check Omega properties
        if not np.allclose(self.Omega, self.Omega.T):
            results['Omega_valid'] = False
            warnings.warn("Omega is not symmetric")
        
        eigenvalues = np.linalg.eigvalsh(self.Omega)
        if np.any(eigenvalues < -1e-10):
            results['Omega_valid'] = False
            warnings.warn(f"Omega has negative eigenvalues: {eigenvalues[eigenvalues < 0]}")
        
        return results


class ViewGenerator:
    """
    Generates Black-Litterman views from sentiment data.
    
    This class implements the critical bridge between:
    - Unstructured: FinBERT sentiment scores from news
    - Structured: Mathematical view specification (P, Q, Ω)
    
    The calibration follows four key principles:
    1. Sentiment magnitude → View strength (via volatility scaling)
    2. News volume → Confidence (more news = lower uncertainty)
    3. Sentiment consistency → Confidence (low variance = higher confidence)
    4. Market volatility → Return scaling (high-vol assets get larger adjustments)
    
    Example:
        >>> sentiment_data = {
        ...     'XLK': SentimentData('XLK', 0.65, 0.20, 25, [...]),
        ...     'XLE': SentimentData('XLE', -0.40, 0.30, 15, [...])
        ... }
        >>> generator = ViewGenerator(
        ...     tickers=['XLK', 'XLE', 'XLF'],
        ...     volatilities=np.array([0.25, 0.30, 0.20]),
        ...     sentiment_scaling=0.02
        ... )
        >>> view = generator.generate_views(sentiment_data)
    """
    
    def __init__(
        self,
        tickers: List[str],
        volatilities: np.ndarray,
        sentiment_scaling: float = 0.02,
        base_uncertainty: float = 0.0001,
        news_volume_weight: float = 0.5,
        consistency_weight: float = 0.5,
        min_news_count: int = 3
    ):
        """
        Initialize the ViewGenerator.
        
        Parameters:
            tickers: List of asset tickers in order (defines N)
            volatilities: Annualized volatility for each asset (N,)
            sentiment_scaling: Maximum return impact of full sentiment
                              Default 0.02 = 2% expected return change for sentiment = 1.0
            base_uncertainty: Minimum uncertainty level (prevents division by zero)
            news_volume_weight: Weight for news count in uncertainty [0, 1]
            consistency_weight: Weight for sentiment variance in uncertainty [0, 1]
            min_news_count: Minimum news articles required to generate view
        
        Mathematical Intuition:
            - sentiment_scaling controls Q magnitude: Q ≈ sentiment × volatility × scaling
            - base_uncertainty prevents over-confidence when news_count → ∞
            - Weights control whether we trust "many articles" vs "consistent articles"
        """
        self.tickers = tickers
        self.volatilities = np.asarray(volatilities)
        self.N = len(tickers)
        
        if len(volatilities) != self.N:
            raise ValueError(
                f"Volatilities length {len(volatilities)} must match tickers {self.N}"
            )
        
        self.sentiment_scaling = sentiment_scaling
        self.base_uncertainty = base_uncertainty
        self.news_volume_weight = news_volume_weight
        self.consistency_weight = consistency_weight
        self.min_news_count = min_news_count
        
        # Create ticker -> index mapping
        self.ticker_to_idx = {t: i for i, t in enumerate(tickers)}
    
    def generate_views(
        self,
        sentiment_data: Dict[str, SentimentData],
        prior_returns: Optional[np.ndarray] = None,
        filter_weak_signals: bool = True,
        min_abs_sentiment: float = 0.1
    ) -> BlackLittermanView:
        """
        Generate Black-Litterman view specification from sentiment data.
        
        This is the CORE function that translates NLP → Math.
        
        Process:
        1. Filter: Keep only assets with sufficient news and strong sentiment
        2. Build P: Create pick matrix (identity matrix for absolute views)
        3. Compute Q: Scale sentiment by volatility and sentiment_scaling
        4. Compute Ω: Derive uncertainty from news volume and consistency
        
        Parameters:
            sentiment_data: Dictionary mapping ticker -> SentimentData
            prior_returns: Prior expected returns (N,) - used for relative views
            filter_weak_signals: If True, ignore weak/noisy sentiment
            min_abs_sentiment: Minimum |sentiment| to generate view
        
        Returns:
            BlackLittermanView with P, Q, Ω and metadata
        
        Mathematical Formulas:
            P[k, i] = 1 for asset i in view k (absolute views)
            Q[k] = sentiment[k] × σ[i] × scaling_factor
            Ω[k,k] = base + (1 - volume_factor) × volume_weight 
                          + variance_factor × consistency_weight
        """
        # Step 1: Filter views based on news quality
        valid_views = self._filter_sentiment_data(
            sentiment_data, 
            filter_weak_signals, 
            min_abs_sentiment
        )
        
        if not valid_views:
            # No valid views - return null view (won't affect posterior)
            return self._create_null_view()
        
        K = len(valid_views)
        
        # Step 2: Build Pick Matrix P
        P = self._build_pick_matrix(valid_views)
        
        # Step 3: Compute View Vector Q
        Q = self._compute_view_vector(valid_views, prior_returns)
        
        # Step 4: Compute Uncertainty Matrix Ω
        Omega = self._compute_uncertainty_matrix(valid_views)
        
        # Create metadata for tracking
        metadata = {
            'tickers': [v.ticker for v in valid_views],
            'sentiments': [v.sentiment_mean for v in valid_views],
            'news_counts': [v.news_count for v in valid_views],
            'generation_timestamp': pd.Timestamp.now(),
            'parameters': {
                'sentiment_scaling': self.sentiment_scaling,
                'base_uncertainty': self.base_uncertainty,
                'news_volume_weight': self.news_volume_weight,
                'consistency_weight': self.consistency_weight
            }
        }
        
        view = BlackLittermanView(P=P, Q=Q, Omega=Omega, metadata=metadata)
        
        # Validate before returning
        validation = view.validate()
        if not all(validation.values()):
            warnings.warn(f"View validation failed: {validation}")
        
        return view
    
    def _filter_sentiment_data(
        self,
        sentiment_data: Dict[str, SentimentData],
        filter_weak_signals: bool,
        min_abs_sentiment: float
    ) -> List[SentimentData]:
        """
        Filter sentiment data to keep only high-quality views.
        
        Quality criteria:
        1. Ticker must be in our universe
        2. Sufficient news volume (>= min_news_count)
        3. Strong enough sentiment signal (if filter_weak_signals)
        
        Returns:
            List of SentimentData that pass quality filters
        """
        valid_views = []
        
        for ticker, data in sentiment_data.items():
            # Check 1: Ticker in universe
            if ticker not in self.ticker_to_idx:
                warnings.warn(f"Ticker {ticker} not in portfolio universe, skipping")
                continue
            
            # Check 2: Sufficient news
            if data.news_count < self.min_news_count:
                warnings.warn(
                    f"Insufficient news for {ticker}: {data.news_count} < {self.min_news_count}"
                )
                continue
            
            # Check 3: Strong signal (if filtering enabled)
            if filter_weak_signals:
                if abs(data.sentiment_mean) < min_abs_sentiment:
                    warnings.warn(
                        f"Weak sentiment for {ticker}: {data.sentiment_mean:.3f}"
                    )
                    continue
            
            valid_views.append(data)
        
        return valid_views
    
    def _build_pick_matrix(self, valid_views: List[SentimentData]) -> np.ndarray:
        """
        Build the Pick Matrix P.
        
        For absolute views (our case), P is a sparse matrix with one 1 per row.
        
        Mathematical structure:
            P[k, i] = 1 if view k refers to asset i
            P[k, j] = 0 otherwise
        
        Example (3 assets, 2 views on assets 0 and 2):
            P = [[1, 0, 0],
                 [0, 0, 1]]
        
        Returns:
            Pick matrix (K × N)
        """
        K = len(valid_views)
        P = np.zeros((K, self.N))
        
        for k, view_data in enumerate(valid_views):
            asset_idx = self.ticker_to_idx[view_data.ticker]
            P[k, asset_idx] = 1.0
        
        return P
    
    def _compute_view_vector(
        self,
        valid_views: List[SentimentData],
        prior_returns: Optional[np.ndarray]
    ) -> np.ndarray:
        """
        Compute the View Vector Q from sentiment.
        
        This is where we translate sentiment scores into expected returns.
        
        Formula:
            Q[k] = π[i] + sentiment[k] × σ[i] × scaling
        
        Where:
            - π[i]: Prior expected return (if provided, else 0)
            - sentiment[k]: FinBERT score ∈ [-1, +1]
            - σ[i]: Annualized volatility of asset i
            - scaling: sentiment_scaling parameter (default 0.02)
        
        Intuition:
            - High volatility assets get larger sentiment impact
            - scaling = 0.02 means full positive sentiment (+1.0) adds 2% return
            - Sentiment of 0.5 on 25% vol asset: Δret = 0.5 × 0.25 × 0.02 = 0.25%
        
        Returns:
            View vector Q (K,)
        """
        K = len(valid_views)
        Q = np.zeros(K)
        
        for k, view_data in enumerate(valid_views):
            asset_idx = self.ticker_to_idx[view_data.ticker]
            
            # Base return (prior if available)
            base_return = prior_returns[asset_idx] if prior_returns is not None else 0.0
            
            # Sentiment-driven adjustment
            # Formula: Δreturn = sentiment × volatility × scaling
            sentiment_impact = (
                view_data.sentiment_mean 
                * self.volatilities[asset_idx] 
                * self.sentiment_scaling
            )
            
            Q[k] = base_return + sentiment_impact
        
        return Q
    
    def _compute_uncertainty_matrix(
        self,
        valid_views: List[SentimentData]
    ) -> np.ndarray:
        """
        Compute the Uncertainty Matrix Ω.
        
        This is THE MOST CRITICAL function for model stability.
        
        Ω controls how much we trust each view:
        - Small Ω[k,k] → High confidence → View strongly affects posterior
        - Large Ω[k,k] → Low confidence → View weakly affects posterior
        
        We derive uncertainty from TWO sources:
        
        1. NEWS VOLUME: More articles → Lower uncertainty
           volume_factor = 1 / sqrt(news_count)
           
        2. SENTIMENT CONSISTENCY: Low variance → Lower uncertainty
           consistency_factor = sentiment_std
        
        Formula:
            Ω[k,k] = base_uncertainty 
                     + volume_weight × volume_factor
                     + consistency_weight × consistency_factor²
        
        Intuition:
            - 50 consistent articles (std=0.1) → Low Ω → High confidence
            - 5 conflicting articles (std=0.8) → High Ω → Low confidence
        
        Returns:
            Uncertainty matrix Ω (K × K), diagonal
        """
        K = len(valid_views)
        Omega = np.zeros((K, K))
        
        for k, view_data in enumerate(valid_views):
            # Component 1: Volume-based uncertainty
            # More news → lower uncertainty
            # Use sqrt to prevent over-confidence with many articles
            volume_uncertainty = 1.0 / np.sqrt(max(view_data.news_count, 1))
            
            # Component 2: Consistency-based uncertainty
            # High variance → high uncertainty
            # Square to penalize inconsistency more
            consistency_uncertainty = view_data.sentiment_std ** 2
            
            # Combined uncertainty (weighted sum)
            total_uncertainty = (
                self.base_uncertainty
                + self.news_volume_weight * volume_uncertainty
                + self.consistency_weight * consistency_uncertainty
            )
            
            Omega[k, k] = total_uncertainty
        
        return Omega
    
    def _create_null_view(self) -> BlackLittermanView:
        """
        Create a null view (no impact on posterior).
        
        Used when no valid sentiment data is available.
        The view is mathematically valid but has infinite uncertainty,
        so it doesn't affect the posterior.
        
        Returns:
            BlackLittermanView with trivial P, Q, and large Ω
        """
        P = np.zeros((1, self.N))
        P[0, 0] = 1.0  # Arbitrary asset
        Q = np.zeros(1)
        Omega = np.eye(1) * 1e10  # Infinite uncertainty
        
        metadata = {
            'null_view': True,
            'reason': 'No valid sentiment data',
            'generation_timestamp': pd.Timestamp.now()
        }
        
        return BlackLittermanView(P=P, Q=Q, Omega=Omega, metadata=metadata)
    
    def analyze_view_impact(
        self,
        view: BlackLittermanView,
        prior_returns: np.ndarray
    ) -> pd.DataFrame:
        """
        Analyze the expected impact of views on portfolio.
        
        This is a diagnostic tool to understand what the views are doing.
        
        Parameters:
            view: Generated BlackLittermanView
            prior_returns: Prior expected returns (N,)
        
        Returns:
            DataFrame with view analysis
        """
        K = view.P.shape[0]
        
        analysis_data = []
        
        for k in range(K):
            # Find which asset this view refers to
            asset_idx = np.argmax(np.abs(view.P[k]))
            ticker = self.tickers[asset_idx]
            
            # Extract metadata
            sentiment = view.metadata.get('sentiments', [None])[k]
            news_count = view.metadata.get('news_counts', [None])[k]
            
            # Compute impact
            prior_ret = prior_returns[asset_idx]
            view_ret = view.Q[k]
            return_adjustment = view_ret - prior_ret
            
            # Uncertainty
            uncertainty = view.Omega[k, k]
            
            # Effective confidence (inverse of uncertainty)
            confidence_score = 1.0 / (1.0 + uncertainty)
            
            analysis_data.append({
                'ticker': ticker,
                'sentiment': sentiment,
                'news_count': news_count,
                'prior_return': prior_ret,
                'view_return': view_ret,
                'return_adjustment': return_adjustment,
                'uncertainty': uncertainty,
                'confidence_score': confidence_score
            })
        
        df = pd.DataFrame(analysis_data)
        
        # Sort by absolute return adjustment (biggest impact first)
        df = df.sort_values('return_adjustment', key=abs, ascending=False)
        
        return df


def compute_sentiment_statistics(
    raw_scores: List[float]
) -> Tuple[float, float]:
    """
    Compute mean and standard deviation from raw sentiment scores.
    
    Helper function for creating SentimentData objects.
    
    Parameters:
        raw_scores: List of FinBERT scores from individual articles
    
    Returns:
        (mean, std) tuple
    """
    if not raw_scores:
        return 0.0, 0.5  # Neutral sentiment, moderate uncertainty
    
    scores = np.array(raw_scores)
    mean = float(np.mean(scores))
    std = float(np.std(scores)) if len(scores) > 1 else 0.5
    
    return mean, std


# Example usage and demonstration
if __name__ == "__main__":
    print("=" * 80)
    print("ViewGenerator: Sentiment → Black-Litterman Integration")
    print("=" * 80)
    
    # Example: Technology sector portfolio
    tickers = ['XLK', 'XLE', 'XLF', 'XLV', 'XLY']
    ticker_names = ['Tech', 'Energy', 'Finance', 'Healthcare', 'Consumer']
    
    # Historical volatilities (annualized)
    volatilities = np.array([0.25, 0.30, 0.22, 0.20, 0.28])
    
    # Prior returns (from equilibrium or historical)
    prior_returns = np.array([0.10, 0.08, 0.09, 0.085, 0.095])
    
    # Initialize ViewGenerator
    generator = ViewGenerator(
        tickers=tickers,
        volatilities=volatilities,
        sentiment_scaling=0.02,  # 2% max impact
        news_volume_weight=0.5,
        consistency_weight=0.5
    )
    
    # Simulated sentiment data (from FinBERT)
    # Scenario: Positive tech news, negative energy news
    sentiment_data = {
        'XLK': SentimentData(
            ticker='XLK',
            sentiment_mean=0.65,  # Positive sentiment
            sentiment_std=0.20,    # Fairly consistent
            news_count=25,         # Good volume
            raw_scores=[0.7, 0.6, 0.65, 0.8, 0.5, 0.7]  # Example scores
        ),
        'XLE': SentimentData(
            ticker='XLE',
            sentiment_mean=-0.40,  # Negative sentiment
            sentiment_std=0.30,     # More scattered
            news_count=15,          # Moderate volume
            raw_scores=[-0.5, -0.3, -0.4, -0.6, -0.2]
        ),
        'XLF': SentimentData(
            ticker='XLF',
            sentiment_mean=0.15,   # Slightly positive
            sentiment_std=0.45,     # Inconsistent (will likely filter out)
            news_count=8,           # Low volume
            raw_scores=[0.3, -0.2, 0.4, 0.1]
        )
    }
    
    # Generate views
    print("\n1. Generating views from sentiment data...")
    view = generator.generate_views(
        sentiment_data,
        prior_returns=prior_returns,
        filter_weak_signals=True,
        min_abs_sentiment=0.2  # Filter out weak signals
    )
    
    print(f"\nGenerated {view.P.shape[0]} views")
    print(f"Tickers with views: {view.metadata['tickers']}")
    
    # Display view structure
    print("\n2. View Structure:")
    print("\nPick Matrix P:")
    print(view.P)
    print(f"\nView Vector Q (expected returns):")
    print(view.Q)
    print(f"\nUncertainty Matrix Ω (diagonal):")
    print(np.diag(view.Omega))
    
    # Analyze impact
    print("\n3. View Impact Analysis:")
    analysis = generator.analyze_view_impact(view, prior_returns)
    print(analysis.to_string(index=False))
    
    # Interpretation
    print("\n4. Interpretation:")
    for _, row in analysis.iterrows():
        direction = "increases" if row['return_adjustment'] > 0 else "decreases"
        print(f"  {row['ticker']}: Sentiment {direction} expected return by "
              f"{abs(row['return_adjustment'])*100:.2f}% "
              f"(confidence: {row['confidence_score']:.1%})")
    
    print("\n" + "=" * 80)
    print("✓ ViewGenerator demonstration complete")
    print("=" * 80)
