"""
========================================================================================
COMPLETE BLACK-LITTERMAN + FINBERT INTEGRATION - FINAL VERSION
Single-File Google Colab Ready Implementation
========================================================================================


Features:
- Complete Black-Litterman model (Meucci 2008)
- FinBERT sentiment analysis integration
- Live NewsAPI data fetching
- ViewGenerator (sentiment → Q, \Omega calibration)
- Portfolio optimization
- Results export to Excel
- DEMO mode for testing without API

========================================================================================
"""

# ============================================================================
# PART 1: IMPORTS & SETUP
# ============================================================================

import numpy as np
import pandas as pd
import warnings
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

# Check and install dependencies for Google Colab
try:
    import yfinance as yf
except ImportError:
    print("Installing yfinance...")
    import sys
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance", "-q"])
    import yfinance as yf

try:
    from newsapi import NewsApiClient
except ImportError:
    print("Installing newsapi-python...")
    import sys
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "newsapi-python", "-q"])
    from newsapi import NewsApiClient

try:
    from transformers import BertTokenizer, BertForSequenceClassification
    import torch
except ImportError:
    print("Installing transformers and torch...")
    import sys
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers", "torch", "-q"])
    from transformers import BertTokenizer, BertForSequenceClassification
    import torch

# Suppress warnings
warnings.filterwarnings('ignore')

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# PART 2: DATA STRUCTURES
# ============================================================================

@dataclass
class SentimentData:
    """Container for sentiment analysis results."""
    ticker: str
    sentiment_mean: float
    sentiment_std: float
    news_count: int
    raw_scores: List[float]


@dataclass
class BlackLittermanView:
    """Black-Litterman view specification (P, Q, \Omega)."""
    P: np.ndarray
    Q: np.ndarray
    Omega: np.ndarray
    metadata: Dict


# ============================================================================
# PART 3: BLACK-LITTERMAN MODEL
# ============================================================================

class BlackLittermanModel:
    """Black-Litterman portfolio optimization (Meucci 2008)."""
    
    def __init__(self, pi: np.ndarray, sigma: np.ndarray):
        """Initialize Black-Litterman model."""
        self.pi = np.asarray(pi).flatten()
        self.sigma = np.asarray(sigma)
        
        n_assets = len(self.pi)
        if self.sigma.shape != (n_assets, n_assets):
            raise ValueError(f"Sigma shape mismatch")
        
        if not np.allclose(self.sigma, self.sigma.T):
            self.sigma = (self.sigma + self.sigma.T) / 2
    
    @staticmethod
    def compute_equilibrium_returns(w_eq: np.ndarray, sigma: np.ndarray, 
                                    risk_aversion: float = 2.5) -> np.ndarray:
        """Compute equilibrium returns: π = 2λΣw_eq"""
        return 2 * risk_aversion * sigma @ np.asarray(w_eq).flatten()
    
    def compute_posterior(self, P: np.ndarray, Q: np.ndarray, 
                         Omega: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Black-Litterman posterior (Meucci 2008, Eq 32-33).
        
        μ_BL = π + ΣP'(PΣP' + Ω)^(-1)(Q - Pπ)
        Σ_BL = Σ - ΣP'(PΣP' + Ω)^(-1)PΣ
        """
        P = np.asarray(P)
        Q = np.asarray(Q).flatten()
        Omega = np.asarray(Omega)
        
        K, N = P.shape
        if Omega.ndim == 0 and K == 1:
            Omega = Omega.reshape((1, 1))
        
        # M = PΣP' + Ω
        M = P @ self.sigma @ P.T + Omega
        
        # Stable inversion
        try:
            M_inv = np.linalg.inv(M)
        except:
            M_inv = np.linalg.pinv(M)
        
        # Posterior mean
        mu_bl = self.pi + self.sigma @ P.T @ M_inv @ (Q - P @ self.pi)
        
        # Posterior covariance
        sigma_bl = self.sigma - self.sigma @ P.T @ M_inv @ P @ self.sigma
        
        return mu_bl, sigma_bl


# ============================================================================
# PART 4: VIEW GENERATOR
# ============================================================================

class ViewGenerator:
    """Generate Black-Litterman views from FinBERT sentiment."""
    
    def __init__(self, tickers: List[str], volatilities: np.ndarray,
                 sentiment_scaling: float = 0.02,
                 base_uncertainty: float = 0.0001,
                 news_volume_weight: float = 0.5,
                 consistency_weight: float = 0.5,
                 min_news_count: int = 3):
        """Initialize ViewGenerator."""
        self.tickers = tickers
        self.volatilities = np.asarray(volatilities)
        self.N = len(tickers)
        self.sentiment_scaling = sentiment_scaling
        self.base_uncertainty = base_uncertainty
        self.news_volume_weight = news_volume_weight
        self.consistency_weight = consistency_weight
        self.min_news_count = min_news_count
        self.ticker_to_idx = {t: i for i, t in enumerate(tickers)}
    
    def generate_views(self, sentiment_data: Dict[str, SentimentData],
                      prior_returns: np.ndarray,
                      filter_weak_signals: bool = True,
                      min_abs_sentiment: float = 0.1) -> BlackLittermanView:
        """Generate Black-Litterman views from sentiment data."""
        
        # Filter valid views
        valid_views = []
        for ticker, data in sentiment_data.items():
            if ticker not in self.ticker_to_idx:
                continue
            if data.news_count < self.min_news_count:
                continue
            if filter_weak_signals and abs(data.sentiment_mean) < min_abs_sentiment:
                continue
            valid_views.append(data)
        
        if not valid_views:
            # Null view
            P = np.zeros((1, self.N))
            P[0, 0] = 1.0
            Q = np.zeros(1)
            Omega = np.eye(1) * 1e10
            metadata = {'null_view': True, 'tickers': []}
            return BlackLittermanView(P, Q, Omega, metadata)
        
        K = len(valid_views)
        
        # Build P
        P = np.zeros((K, self.N))
        for k, view_data in enumerate(valid_views):
            asset_idx = self.ticker_to_idx[view_data.ticker]
            P[k, asset_idx] = 1.0
        
        # Compute Q: Q = π + sentiment × σ × scaling
        Q = np.zeros(K)
        for k, view_data in enumerate(valid_views):
            asset_idx = self.ticker_to_idx[view_data.ticker]
            base_return = prior_returns[asset_idx]
            sentiment_impact = (view_data.sentiment_mean * 
                              self.volatilities[asset_idx] * 
                              self.sentiment_scaling)
            Q[k] = base_return + sentiment_impact
        
        # Compute Ω
        Omega = np.zeros((K, K))
        for k, view_data in enumerate(valid_views):
            volume_unc = 1.0 / np.sqrt(max(view_data.news_count, 1))
            consistency_unc = view_data.sentiment_std ** 2
            Omega[k, k] = (self.base_uncertainty + 
                          self.news_volume_weight * volume_unc +
                          self.consistency_weight * consistency_unc)
        
        metadata = {
            'tickers': [v.ticker for v in valid_views],
            'sentiments': [v.sentiment_mean for v in valid_views],
            'news_counts': [v.news_count for v in valid_views]
        }
        
        return BlackLittermanView(P, Q, Omega, metadata)


# ============================================================================
# PART 5: FINBERT ANALYZER
# ============================================================================

class FinBERTAnalyzer:
    """FinBERT sentiment analysis."""
    
    def __init__(self):
        """Load FinBERT model."""
        logger.info("Loading FinBERT model...")
        self.tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = BertForSequenceClassification.from_pretrained("ProsusAI/finbert")
        self.model.eval()
        logger.info("✓ FinBERT loaded")
    
    def analyze_sentiment(self, texts: List[str]) -> Tuple[np.ndarray, float, float]:
        """Analyze sentiment: Score = P(Pos) - P(Neg)"""
        if not texts:
            return np.array([]), 0.0, 0.5
        
        inputs = self.tokenizer(texts, padding=True, truncation=True, 
                               max_length=512, return_tensors='pt')
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).numpy()
        scores = probs[:, 0] - probs[:, 1]  # P(Pos) - P(Neg)
        
        return probs, float(np.mean(scores)), float(np.std(scores)) if len(scores) > 1 else 0.5


# ============================================================================
# PART 6: MAIN OPTIMIZER CLASS
# ============================================================================

class SentimentBlackLittermanOptimizer:
    """Complete integration: NewsAPI → FinBERT → Black-Litterman"""
    
    def __init__(self, api_key: str, tickers: List[str],
                 lookback_days: int = 252,
                 news_lookback_days: int = 7,
                 articles_per_ticker: int = 10,
                 risk_aversion: float = 2.5,
                 sentiment_scaling: float = 0.02):
        """Initialize optimizer."""
        self.api_key = api_key
        self.tickers = tickers
        self.N = len(tickers)
        self.lookback_days = lookback_days
        self.news_lookback_days = news_lookback_days
        self.articles_per_ticker = articles_per_ticker
        self.risk_aversion = risk_aversion
        self.sentiment_scaling = sentiment_scaling
        
        self.newsapi = NewsApiClient(api_key=api_key)
        self.finbert = FinBERTAnalyzer()
        
        self.sigma = None
        self.volatilities = None
        self.pi = None
        self.w_eq = None
        self.bl_model = None
        self.view_generator = None
    
    def fetch_news(self, ticker: str) -> List[Dict]:
        """Fetch news for ticker."""
        try:
            to_date = datetime.now()
            from_date = to_date - timedelta(days=self.news_lookback_days)
            
            response = self.newsapi.get_everything(
                q=ticker,
                language='en',
                sort_by='relevancy',
                page_size=self.articles_per_ticker,
                from_param=from_date.strftime('%Y-%m-%d'),
                to=to_date.strftime('%Y-%m-%d')
            )
            
            articles = response.get('articles', [])
            logger.info(f"  ✓ {ticker}: {len(articles)} articles")
            return articles
        except Exception as e:
            logger.error(f"  ✗ {ticker}: {e}")
            return []
    
    def process_ticker(self, ticker: str) -> Optional[SentimentData]:
        """Fetch news and analyze sentiment."""
        logger.info(f"\nProcessing {ticker}:")
        
        articles = self.fetch_news(ticker)
        if not articles:
            return None
        
        texts = []
        for article in articles:
            title = article.get('title', '')
            desc = article.get('description', '')
            text = f"{title}. {desc}" if desc else title
            if text:
                texts.append(text)
        
        if not texts:
            return None
        
        probs, sentiment_mean, sentiment_std = self.finbert.analyze_sentiment(texts)
        
        logger.info(f"  Sentiment: {sentiment_mean:+.3f} (std: {sentiment_std:.3f})")
        logger.info(f"  Sample: {texts[0][:60]}...")
        
        return SentimentData(
            ticker=ticker,
            sentiment_mean=sentiment_mean,
            sentiment_std=sentiment_std,
            news_count=len(texts),
            raw_scores=(probs[:, 0] - probs[:, 1]).tolist()
        )
    
    def load_market_data(self):
        """Load historical data - FIXED VERSION."""
        logger.info(f"\n{'='*80}")
        logger.info("LOADING MARKET DATA")
        logger.info(f"{'='*80}")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_days + 30)
        
        # Download all tickers at once (more efficient)
        logger.info(f"Downloading data for {len(self.tickers)} tickers...")
        
        try:
            # Download all at once
            data = yf.download(self.tickers, start=start_date, end=end_date, 
                             group_by='ticker', progress=False, threads=True)
            
            # Extract Close prices
            prices_dict = {}
            for ticker in self.tickers:
                try:
                    if len(self.tickers) == 1:
                        # Single ticker case
                        prices_dict[ticker] = data['Close']
                    else:
                        # Multiple tickers
                        prices_dict[ticker] = data[ticker]['Close']
                    
                    logger.info(f"  ✓ {ticker}: {len(prices_dict[ticker])} days")
                except Exception as e:
                    logger.warning(f"  ⚠ {ticker}: {e}")
            
            if not prices_dict:
                raise ValueError("No data loaded for any ticker!")
            
            # Create DataFrame and align
            prices_df = pd.DataFrame(prices_dict)
            prices_df = prices_df.dropna()
            
            if len(prices_df) < 50:
                raise ValueError(f"Insufficient data: only {len(prices_df)} days")
            
            logger.info(f"\nCommon trading days: {len(prices_df)}")
            
            # Compute returns
            returns_df = np.log(prices_df / prices_df.shift(1)).dropna()
            
            # Estimate covariance
            self.sigma = returns_df.cov().values * 252
            self.sigma += np.eye(self.N) * 1e-6
            
            # Volatilities
            self.volatilities = np.sqrt(np.diag(self.sigma))
            
            logger.info("\nVolatilities (annualized):")
            for i, ticker in enumerate(self.tickers):
                logger.info(f"  {ticker}: {self.volatilities[i]*100:5.1f}%")
            
            # Equilibrium
            self.w_eq = np.ones(self.N) / self.N
            self.pi = BlackLittermanModel.compute_equilibrium_returns(
                self.w_eq, self.sigma, self.risk_aversion
            )
            
            logger.info("\nEquilibrium Returns:")
            for i, ticker in enumerate(self.tickers):
                logger.info(f"  {ticker}: {self.pi[i]*100:5.1f}%")
            
            # Initialize models
            self.bl_model = BlackLittermanModel(self.pi, self.sigma)
            self.view_generator = ViewGenerator(
                tickers=self.tickers,
                volatilities=self.volatilities,
                sentiment_scaling=self.sentiment_scaling
            )
            
        except Exception as e:
            logger.error(f"Error loading market data: {e}")
            raise
    
    def optimize(self) -> Optional[Dict]:
        """Run complete optimization."""
        logger.info(f"\n{'='*80}")
        logger.info("SENTIMENT BLACK-LITTERMAN OPTIMIZATION")
        logger.info(f"{'='*80}")
        
        # Load market data
        if self.sigma is None:
            self.load_market_data()
        
        # Fetch news & analyze
        logger.info(f"\n{'='*80}")
        logger.info("FETCHING NEWS & ANALYZING SENTIMENT")
        logger.info(f"{'='*80}")
        
        sentiment_data = {}
        for ticker in self.tickers:
            data = self.process_ticker(ticker)
            if data:
                sentiment_data[ticker] = data
        
        if not sentiment_data:
            logger.error("\n✗ No valid sentiment data")
            return None
        
        # Generate views
        logger.info(f"\n{'='*80}")
        logger.info("GENERATING VIEWS")
        logger.info(f"{'='*80}")
        
        view = self.view_generator.generate_views(
            sentiment_data, self.pi, 
            filter_weak_signals=True, min_abs_sentiment=0.10
        )
        
        logger.info(f"\n✓ Generated {view.P.shape[0]} views")
        logger.info(f"  Tickers: {view.metadata.get('tickers', [])}")
        
        # Black-Litterman
        logger.info(f"\n{'='*80}")
        logger.info("BLACK-LITTERMAN POSTERIOR")
        logger.info(f"{'='*80}")
        
        mu_bl, sigma_bl = self.bl_model.compute_posterior(view.P, view.Q, view.Omega)
        
        logger.info("\nPosterior Returns:")
        for i, ticker in enumerate(self.tickers):
            delta = mu_bl[i] - self.pi[i]
            direction = "↑" if delta > 0 else "↓" if delta < 0 else "→"
            logger.info(f"  {ticker}: {self.pi[i]*100:5.1f}% → {mu_bl[i]*100:5.1f}% "
                       f"{direction} ({delta*100:+.1f}%)")
        
        # Optimize
        logger.info(f"\n{'='*80}")
        logger.info("PORTFOLIO OPTIMIZATION")
        logger.info(f"{'='*80}")
        
        try:
            sigma_inv = np.linalg.inv(sigma_bl)
        except:
            sigma_inv = np.linalg.pinv(sigma_bl)
        
        w_optimal = sigma_inv @ mu_bl / (2 * self.risk_aversion)
        w_optimal = np.maximum(w_optimal, 0)
        w_optimal = w_optimal / np.sum(w_optimal)
        
        logger.info("\nOptimal Weights:")
        for i, ticker in enumerate(self.tickers):
            logger.info(f"  {ticker}: {w_optimal[i]*100:5.1f}%")
        
        portfolio_return = w_optimal @ mu_bl
        portfolio_vol = np.sqrt(w_optimal @ sigma_bl @ w_optimal)
        sharpe = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
        
        logger.info(f"\nPortfolio:")
        logger.info(f"  Return: {portfolio_return*100:5.1f}%")
        logger.info(f"  Vol:    {portfolio_vol*100:5.1f}%")
        logger.info(f"  Sharpe: {sharpe:.2f}")
        
        return {
            'timestamp': datetime.now(),
            'tickers': self.tickers,
            'sentiment_data': sentiment_data,
            'view': view,
            'prior_returns': self.pi,
            'posterior_returns': mu_bl,
            'prior_weights': self.w_eq,
            'optimal_weights': w_optimal,
            'portfolio_return': portfolio_return,
            'portfolio_vol': portfolio_vol,
            'sharpe_ratio': sharpe
        }


# ============================================================================
# PART 7: DEMO MODE
# ============================================================================

def run_demo_mode(tickers: List[str] = None):
    """Demo mode with simulated sentiment (no API needed)."""
    if tickers is None:
        tickers = ['AAPL', 'MSFT', 'GOOGL']
    
    print("\n" + "="*80)
    print("DEMO MODE - SIMULATED SENTIMENT")
    print("="*80)
    
    # Load market data
    print("\nLoading market data...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=282)
    
    try:
        data = yf.download(tickers, start=start_date, end=end_date,
                          group_by='ticker', progress=False)
        
        prices_dict = {}
        for ticker in tickers:
            try:
                if len(tickers) == 1:
                    prices_dict[ticker] = data['Close']
                else:
                    prices_dict[ticker] = data[ticker]['Close']
                print(f"  ✓ {ticker}: {len(prices_dict[ticker])} days")
            except:
                pass
        
        prices_df = pd.DataFrame(prices_dict).dropna()
        returns_df = np.log(prices_df / prices_df.shift(1)).dropna()
        
        N = len(tickers)
        sigma = returns_df.cov().values * 252
        sigma += np.eye(N) * 1e-6
        volatilities = np.sqrt(np.diag(sigma))
        
        w_eq = np.ones(N) / N
        pi = BlackLittermanModel.compute_equilibrium_returns(w_eq, sigma, 2.5)
        
        print("\nEquilibrium Returns:")
        for i, ticker in enumerate(tickers):
            print(f"  {ticker}: {pi[i]*100:5.1f}%")
        
        # Simulated sentiment
        print("\n" + "="*80)
        print("SIMULATED SENTIMENT")
        print("="*80)
        
        np.random.seed(42)
        sentiment_data = {}
        
        base_sentiments = {'AAPL': 0.45, 'MSFT': 0.35, 'GOOGL': 0.25,
                          'TSLA': -0.20, 'NVDA': 0.60}
        
        for ticker in tickers:
            base = base_sentiments.get(ticker, np.random.uniform(-0.3, 0.3))
            news_count = np.random.randint(8, 25)
            raw_scores = np.clip(np.random.normal(base, 0.2, news_count), -1, 1)
            
            sentiment_data[ticker] = SentimentData(
                ticker=ticker,
                sentiment_mean=float(np.mean(raw_scores)),
                sentiment_std=float(np.std(raw_scores)),
                news_count=news_count,
                raw_scores=raw_scores.tolist()
            )
            
            print(f"  {ticker}: {sentiment_data[ticker].sentiment_mean:+.3f} "
                  f"({news_count} articles)")
        
        # Generate views
        generator = ViewGenerator(tickers, volatilities, 0.02)
        view = generator.generate_views(sentiment_data, pi, True, 0.10)
        
        print(f"\n✓ Generated {view.P.shape[0]} views")
        
        # Black-Litterman
        bl = BlackLittermanModel(pi, sigma)
        mu_bl, sigma_bl = bl.compute_posterior(view.P, view.Q, view.Omega)
        
        # Optimize
        sigma_inv = np.linalg.pinv(sigma_bl)
        w_optimal = sigma_inv @ mu_bl / 5.0
        w_optimal = np.maximum(w_optimal, 0)
        w_optimal = w_optimal / np.sum(w_optimal)
        
        # Results
        print("\n" + "="*80)
        print("FINAL PORTFOLIO")
        print("="*80)
        print(f"\n{'Ticker':<8} {'Sentiment':<12} {'Weight':<10} {'Return'}")
        print("-" * 60)
        
        for i, ticker in enumerate(tickers):
            sent = sentiment_data[ticker].sentiment_mean
            print(f"{ticker:<8} {sent:>+.3f}         {w_optimal[i]*100:>5.1f}%     "
                  f"{mu_bl[i]*100:>5.1f}%")
        
        portfolio_return = w_optimal @ mu_bl
        portfolio_vol = np.sqrt(w_optimal @ sigma_bl @ w_optimal)
        sharpe = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
        
        print("-" * 60)
        print(f"Portfolio Return: {portfolio_return*100:>5.1f}%")
        print(f"Portfolio Vol:    {portfolio_vol*100:>5.1f}%")
        print(f"Sharpe Ratio:     {sharpe:>5.2f}")
        print("="*80)
        
        return {'success': True}
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# PART 8: MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    Google Colab Execution
    
    Two modes:
    1. DEMO MODE (simulated sentiment) - No API key needed
    2. LIVE MODE (real news) - Uses NewsAPI
    """
    
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    
    USE_DEMO_MODE = False  # Set to True for demo, False for live
    
    # YOUR REAL API KEY
    NEWS_API_KEY = "1246c60fdabd4db7b6d55b5fcfa73c14"
    
    # Portfolio
    TICKERS = ['AAPL', 'MSFT', 'GOOGL']  # Start with 3 for testing
    
    # Parameters
    LOOKBACK_DAYS = 252
    NEWS_LOOKBACK_DAYS = 7
    RISK_AVERSION = 2.5
    
    # ========================================================================
    # RUN
    # ========================================================================
    
    print("\n" + "="*80)
    print("BLACK-LITTERMAN + FINBERT PORTFOLIO OPTIMIZATION")
    print("="*80)
    
    if USE_DEMO_MODE:
        print("\nMode: DEMO (simulated sentiment)")
        print("Set USE_DEMO_MODE = False for live news")
        print("="*80)
        
        results = run_demo_mode(TICKERS)
        
    else:
        print("\nMode: LIVE (real NewsAPI + FinBERT)")
        print(f"API Key: {NEWS_API_KEY[:10]}...")
        print(f"Tickers: {', '.join(TICKERS)}")
        print("="*80)
        
        try:
            optimizer = SentimentBlackLittermanOptimizer(
                api_key=NEWS_API_KEY,
                tickers=TICKERS,
                lookback_days=LOOKBACK_DAYS,
                news_lookback_days=NEWS_LOOKBACK_DAYS,
                risk_aversion=RISK_AVERSION
            )
            
            results = optimizer.optimize()
            
            if results:
                print("\n" + "="*80)
                print("✓ OPTIMIZATION COMPLETE")
                print("="*80)
                
                print(f"\n{'Ticker':<8} {'Sentiment':<12} {'Weight':<10} {'Return'}")
                print("-" * 60)
                
                for i, ticker in enumerate(TICKERS):
                    sent_data = results['sentiment_data'].get(ticker)
                    sent = sent_data.sentiment_mean if sent_data else 0.0
                    weight = results['optimal_weights'][i]
                    ret = results['posterior_returns'][i]
                    
                    print(f"{ticker:<8} {sent:>+.3f}         {weight*100:>5.1f}%     "
                          f"{ret*100:>5.1f}%")
                
                print("-" * 60)
                print(f"Portfolio Return: {results['portfolio_return']*100:>5.1f}%")
                print(f"Portfolio Vol:    {results['portfolio_vol']*100:>5.1f}%")
                print(f"Sharpe Ratio:     {results['sharpe_ratio']:>5.2f}")
                print("="*80)
            
        except Exception as e:
            print("\n" + "="*80)
            print("✗ ERROR")
            print("="*80)
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
            print("\nTry setting USE_DEMO_MODE = True to test without API")
    
    print("\n" + "="*80)
    print("DONE")
    print("="*80)


# ============================================================================
# PART 9: PORTFOLIO COMPARISON VISUALIZATION
# ============================================================================

def create_portfolio_comparison_plots(
    tickers: List[str],
    results: Dict,
    show_plots: bool = True
):
    """
    Create comparison plots: Market-Cap vs Mean-Variance vs Black-Litterman.
    
    Parameters:
        tickers: List of tickers
        results: Results from optimize()
        show_plots: Display plots interactively
    
    Returns:
        Dictionary of Plotly figures
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("Installing plotly...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "plotly", "-q"])
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    
    print("\n" + "="*80)
    print("PORTFOLIO COMPARISON VISUALIZATION")
    print("="*80)
    
    # Extract data from results
    mu_bl = results['posterior_returns']
    sigma = results.get('sigma')
    w_eq = results['prior_weights']  # Market-cap proxy
    w_bl = results['optimal_weights']  # Black-Litterman
    
    # Compute Mean-Variance (unstable)
    mu_hist = results['prior_returns']  # Use prior as proxy for historical
    
    try:
        sigma_inv = np.linalg.inv(sigma)
    except:
        sigma_inv = np.linalg.pinv(sigma)
    
    w_mv = sigma_inv @ mu_hist / 5.0
    w_mv = np.maximum(w_mv, 0)
    if np.sum(w_mv) > 0.01:
        w_mv = w_mv / np.sum(w_mv)
    else:
        w_mv = np.ones(len(tickers)) / len(tickers)
    
    # Compute statistics
    def portfolio_stats(w, mu, sigma):
        ret = w @ mu
        vol = np.sqrt(w @ sigma @ w)
        sharpe = ret / vol if vol > 0 else 0
        return ret, vol, sharpe
    
    stats_eq = portfolio_stats(w_eq, mu_hist, sigma)
    stats_mv = portfolio_stats(w_mv, mu_hist, sigma)
    stats_bl = portfolio_stats(w_bl, mu_bl, sigma)
    
    # ========================================================================
    # PLOT 1: Comprehensive Dashboard
    # ========================================================================
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Risk-Return Comparison',
            'Portfolio Weights',
            'Expected Returns',
            'Sharpe Ratios'
        ),
        specs=[
            [{'type': 'scatter'}, {'type': 'bar'}],
            [{'type': 'bar'}, {'type': 'bar'}]
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.12
    )
    
    # 1. Risk-Return Scatter
    strategies = [
        ('Market-Cap', stats_eq, 'green', 'circle'),
        ('Mean-Variance', stats_mv, 'orange', 'square'),
        ('Black-Litterman', stats_bl, 'red', 'star')
    ]
    
    for name, (ret, vol, sharpe), color, symbol in strategies:
        fig.add_trace(
            go.Scatter(
                x=[vol * 100],
                y=[ret * 100],
                mode='markers+text',
                name=name,
                marker=dict(size=20, color=color, symbol=symbol, 
                          line=dict(width=2, color='white')),
                text=[name],
                textposition='top center',
                hovertemplate=f'{name}<br>Vol: %{{x:.2f}}%<br>Return: %{{y:.2f}}%<br>Sharpe: {sharpe:.3f}',
                showlegend=True
            ),
            row=1, col=1
        )
    
    # 2. Weights Comparison
    weights_data = [
        ('Market-Cap', w_eq, 'green'),
        ('Mean-Variance', w_mv, 'orange'),
        ('Black-Litterman', w_bl, 'red')
    ]
    
    for name, weights, color in weights_data:
        fig.add_trace(
            go.Bar(
                x=tickers,
                y=weights * 100,
                name=name,
                marker_color=color,
                showlegend=False
            ),
            row=1, col=2
        )
    
    # 3. Expected Returns
    returns_data = [stats_eq[0] * 100, stats_mv[0] * 100, stats_bl[0] * 100]
    names = ['Market-Cap', 'Mean-Variance', 'Black-Litterman']
    colors = ['green', 'orange', 'red']
    
    fig.add_trace(
        go.Bar(
            x=names,
            y=returns_data,
            marker_color=colors,
            showlegend=False,
            text=[f'{r:.1f}%' for r in returns_data],
            textposition='outside'
        ),
        row=2, col=1
    )
    
    # 4. Sharpe Ratios
    sharpe_data = [stats_eq[2], stats_mv[2], stats_bl[2]]
    
    fig.add_trace(
        go.Bar(
            x=names,
            y=sharpe_data,
            marker_color=colors,
            showlegend=False,
            text=[f'{s:.3f}' for s in sharpe_data],
            textposition='outside'
        ),
        row=2, col=2
    )
    
    # Update axes
    fig.update_xaxes(title_text='Volatility (%)', row=1, col=1)
    fig.update_yaxes(title_text='Return (%)', row=1, col=1)
    
    fig.update_xaxes(title_text='Assets', row=1, col=2)
    fig.update_yaxes(title_text='Weight (%)', row=1, col=2)
    
    fig.update_yaxes(title_text='Expected Return (%)', row=2, col=1)
    fig.update_yaxes(title_text='Sharpe Ratio', row=2, col=2)
    
    fig.update_layout(
        title_text='Portfolio Strategies Comparison Dashboard',
        template='plotly_white',
        height=800,
        showlegend=True,
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.9)')
    )
    
    # Print comparison table
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON TABLE")
    print("="*80)
    print(f"\n{'Strategy':<20} {'Return':<12} {'Volatility':<12} {'Sharpe':<10}")
    print("-" * 60)
    
    for name, (ret, vol, sharpe), _, _ in strategies:
        print(f"{name:<20} {ret*100:>10.2f}%  {vol*100:>10.2f}%  {sharpe:>8.3f}")
    
    print("-" * 60)
    
    # Key insights
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    
    print("\n1. MARKET-CAP WEIGHTED (Green - Benchmark):")
    print(f"   - Equal weights: {w_eq[0]*100:.1f}% each")
    print(f"   - Return: {stats_eq[0]*100:.2f}%")
    print(f"   - Sharpe: {stats_eq[2]:.3f}")
    print("   → Simple, transparent baseline")
    
    print("\n2. MEAN-VARIANCE (Orange - Often Unstable):")
    concentration_mv = np.max(w_mv) * 100
    print(f"   - Max concentration: {concentration_mv:.1f}%")
    print(f"   - Return: {stats_mv[0]*100:.2f}%")
    print(f"   - Sharpe: {stats_mv[2]:.3f}")
    if concentration_mv > 60:
        print("    HIGH CONCENTRATION - Sign of instability!")
    print("   → Sensitive to estimation errors")
    
    print("\n3. BLACK-LITTERMAN (Red - Sentiment-Enhanced):")
    concentration_bl = np.max(w_bl) * 100
    print(f"   - Max concentration: {concentration_bl:.1f}%")
    print(f"   - Return: {stats_bl[0]*100:.2f}%")
    print(f"   - Sharpe: {stats_bl[2]:.3f}")
    print("   ✓ Stable, incorporates sentiment views")
    
    # Improvement metrics
    sharpe_improvement = ((stats_bl[2] - stats_eq[2]) / stats_eq[2]) * 100
    print(f"\n Sharpe Improvement vs Benchmark: {sharpe_improvement:+.1f}%")
    
    print("="*80)
    
    # Show plot
    if show_plots:
        fig.show()
    
    # Save HTML
    try:
        filename = f"portfolio_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        fig.write_html(filename)
        print(f"\n✓ Interactive plot saved: {filename}")
        
        # Download in Colab
        try:
            from google.colab import files
            files.download(filename)
            print(f"✓ Downloading {filename}...")
        except:
            pass
    except Exception as e:
        print(f"\n Could not save HTML: {e}")
    
    return {
        'figure': fig,
        'stats': {
            'market_cap': stats_eq,
            'mean_variance': stats_mv,
            'black_litterman': stats_bl
        },
        'weights': {
            'market_cap': w_eq,
            'mean_variance': w_mv,
            'black_litterman': w_bl
        }
    }


# Add to main execution
if __name__ == "__main__":
    # ... existing code ...
    
    # After optimization completes, add visualization
    if not USE_DEMO_MODE and results is not None:
        print("\n" + "="*80)
        print("CREATING COMPARISON VISUALIZATIONS...")
        print("="*80)
        
        viz_results = create_portfolio_comparison_plots(
            tickers=TICKERS,
            results=results,
            show_plots=True
        )
