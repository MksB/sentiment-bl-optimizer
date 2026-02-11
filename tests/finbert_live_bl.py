"""
Live FinBERT-to-Meucci Integration
Real-time News → Sentiment → Black-Litterman Portfolio Optimization

This module connects your existing FinBERT implementation with the production
Black-Litterman framework, using live NewsAPI data.

Key Innovation: FinBERT Score → (Q, Ω) calibration
- Sentiment Score = P(Positive) - P(Negative) ∈ [-1, +1]
- Q: Expected returns calibrated via volatility scaling
- Ω: Confidence from article count and sentiment variance
"""

import yfinance as yf
from newsapi import NewsApiClient
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
import logging

# Our production framework
from bl_model import BlackLittermanModel
from view_generator import ViewGenerator, SentimentData

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LiveFinBERTBlackLitterman:
    """
    Live integration: NewsAPI → FinBERT → Meucci Black-Litterman
    
    This class orchestrates the complete real-time pipeline:
    1. Fetch live news from NewsAPI
    2. Analyze sentiment with FinBERT
    3. Calibrate views (Q, Ω) using Meucci framework
    4. Optimize portfolio using Black-Litterman
    
    Example:
        >>> optimizer = LiveFinBERTBlackLitterman(
        ...     api_key="YOUR_NEWS_API_KEY",
        ...     tickers=['AAPL', 'TSLA', 'GOOGL']
        ... )
        >>> results = optimizer.run_live_optimization()
    """
    
    def __init__(
        self,
        api_key: str,
        tickers: List[str],
        lookback_days: int = 252,
        news_lookback_days: int = 7,
        articles_per_ticker: int = 10,
        risk_aversion: float = 2.5,
        sentiment_scaling: float = 0.02
    ):
        """
        Initialize live FinBERT-BL optimizer.
        
        Parameters:
            api_key: NewsAPI key
            tickers: List of stock tickers (e.g., ['AAPL', 'TSLA'])
            lookback_days: Historical data for covariance estimation
            news_lookback_days: How far back to fetch news (days)
            articles_per_ticker: Max articles per ticker
            risk_aversion: Portfolio risk aversion λ
            sentiment_scaling: Max return impact from sentiment
        """
        self.api_key = api_key
        self.tickers = tickers
        self.N = len(tickers)
        self.lookback_days = lookback_days
        self.news_lookback_days = news_lookback_days
        self.articles_per_ticker = articles_per_ticker
        self.risk_aversion = risk_aversion
        self.sentiment_scaling = sentiment_scaling
        
        # Initialize NewsAPI client
        self.newsapi = NewsApiClient(api_key=api_key)
        
        # Initialize FinBERT
        logger.info("Loading FinBERT model...")
        self.tokenizer, self.model = self._load_finbert()
        logger.info("✓ FinBERT loaded successfully")
        
        # Data containers (populated during optimization)
        self.prices_data = None
        self.returns_data = None
        self.sigma = None
        self.volatilities = None
        self.pi = None
        self.w_eq = None
        
        # Models
        self.bl_model = None
        self.view_generator = None
    
    def _load_finbert(self) -> Tuple:
        """
        Load FinBERT model and tokenizer.
        
        Returns:
            (tokenizer, model) tuple
        """
        tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")
        model = BertForSequenceClassification.from_pretrained("ProsusAI/finbert")
        model.eval()  # Set to evaluation mode
        return tokenizer, model
    
    def fetch_live_news(self, ticker: str) -> List[Dict]:
        """
        Fetch live news for a ticker from NewsAPI.
        
        Parameters:
            ticker: Stock ticker symbol
        
        Returns:
            List of article dictionaries
        """
        try:
            # Get company info for better search
            company_info = yf.Ticker(ticker).info
            company_name = company_info.get('longName', ticker)
            
            # Search query (ticker OR company name)
            query = f'{ticker} OR "{company_name}"'
            
            # Date range
            to_date = datetime.now()
            from_date = to_date - timedelta(days=self.news_lookback_days)
            
            logger.info(f"Fetching news for {ticker} ({company_name})...")
            
            # Fetch articles
            response = self.newsapi.get_everything(
                q=query,
                language='en',
                sort_by='relevancy',
                page_size=self.articles_per_ticker,
                from_param=from_date.strftime('%Y-%m-%d'),
                to=to_date.strftime('%Y-%m-%d')
            )
            
            articles = response.get('articles', [])
            
            logger.info(f"  ✓ Found {len(articles)} articles for {ticker}")
            
            return articles
            
        except Exception as e:
            logger.error(f"  ✗ Error fetching news for {ticker}: {e}")
            return []
    
    def analyze_sentiment_finbert(
        self,
        texts: List[str]
    ) -> Tuple[np.ndarray, float, float]:
        """
        Analyze sentiment using FinBERT.
        
        This is the CORE FinBERT integration that computes:
        - Sentiment Score = P(Positive) - P(Negative) ∈ [-1, +1]
        
        Parameters:
            texts: List of text strings (headlines, descriptions)
        
        Returns:
            (probs, sentiment_mean, sentiment_std)
            - probs: Raw probability array (N × 3) [positive, negative, neutral]
            - sentiment_mean: Mean sentiment score
            - sentiment_std: Std deviation of sentiment scores
        """
        if not texts:
            return np.array([]), 0.0, 0.5
        
        # Tokenize (FinBERT max length = 512)
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Softmax to get probabilities
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).numpy()
        
        # FinBERT output: [positive, negative, neutral]
        # Compute sentiment scores for each article
        sentiment_scores = []
        for prob in probs:
            p_pos = prob[0]     # Positive probability
            p_neg = prob[1]     # Negative probability
            # p_neu = prob[2]   # Neutral (not used in score)
            
            # Sentiment Score Formula
            score = p_pos - p_neg  # ∈ [-1, +1]
            sentiment_scores.append(score)
        
        sentiment_scores = np.array(sentiment_scores)
        
        # Statistics
        sentiment_mean = float(np.mean(sentiment_scores))
        sentiment_std = float(np.std(sentiment_scores)) if len(sentiment_scores) > 1 else 0.5
        
        return probs, sentiment_mean, sentiment_std
    
    def process_ticker_news(self, ticker: str) -> Optional[SentimentData]:
        """
        Complete pipeline for one ticker: Fetch news → Analyze sentiment.
        
        Parameters:
            ticker: Stock ticker
        
        Returns:
            SentimentData object or None if insufficient data
        """
        logger.info(f"\nProcessing {ticker}:")
        logger.info("=" * 60)
        
        # 1. Fetch news
        articles = self.fetch_live_news(ticker)
        
        if not articles:
            logger.warning(f"No articles found for {ticker}")
            return None
        
        # 2. Extract text (title + description)
        texts = []
        for article in articles:
            title = article.get('title', '')
            description = article.get('description', '')
            
            # Combine title and description
            text = f"{title}. {description}" if description else title
            if text:
                texts.append(text)
        
        if not texts:
            logger.warning(f"No text content for {ticker}")
            return None
        
        logger.info(f"  Analyzing {len(texts)} articles with FinBERT...")
        
        # 3. Analyze sentiment
        probs, sentiment_mean, sentiment_std = self.analyze_sentiment_finbert(texts)
        
        # 4. Display sample
        logger.info(f"\n  Sample Articles:")
        for i, (text, prob) in enumerate(zip(texts[:3], probs[:3])):
            score = prob[0] - prob[1]  # P(pos) - P(neg)
            logger.info(f"    [{i+1}] Score: {score:+.3f} | {text[:80]}...")
        
        # 5. Summary statistics
        logger.info(f"\n  Sentiment Statistics:")
        logger.info(f"    Mean:  {sentiment_mean:+.3f}")
        logger.info(f"    Std:   {sentiment_std:.3f}")
        logger.info(f"    Count: {len(texts)}")
        
        # 6. Create SentimentData object
        sentiment_data = SentimentData(
            ticker=ticker,
            sentiment_mean=sentiment_mean,
            sentiment_std=sentiment_std,
            news_count=len(texts),
            raw_scores=probs[:, 0] - probs[:, 1]  # All scores
        )
        
        return sentiment_data
    
    def load_market_data(self):
        """
        Load historical price data and estimate parameters.
        
        This computes:
        - Σ: Covariance matrix
        - π: Equilibrium returns
        - σ: Volatilities (for view calibration)
        """
        logger.info(f"\n{'='*80}")
        logger.info("LOADING MARKET DATA")
        logger.info(f"{'='*80}")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_days + 30)
        
        logger.info(f"Period: {start_date.date()} to {end_date.date()}")
        
        # Download prices
        prices_dict = {}
        
        for ticker in self.tickers:
            try:
                logger.info(f"  Downloading {ticker}...")
                data = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    progress=False
                )
                
                if data.empty:
                    logger.warning(f"No data for {ticker}")
                    continue
                
                prices_dict[ticker] = data['Close']
                logger.info(f"     {len(data)} days")
                
            except Exception as e:
                logger.error(f"    Error: {e}")
                continue
        
        # Create DataFrame
        self.prices_data = pd.DataFrame(prices_dict)
        self.prices_data = self.prices_data.dropna()
        
        # Compute log returns
        self.returns_data = np.log(self.prices_data / self.prices_data.shift(1)).dropna()
        
        logger.info(f"\nFinal dataset: {len(self.returns_data)} days × {len(self.returns_data.columns)} tickers")
        
        # Estimate covariance (annualized)
        self.sigma = self.returns_data.cov().values * 252
        self.sigma += np.eye(self.N) * 1e-6  # Numerical stability
        
        # Volatilities
        self.volatilities = np.sqrt(np.diag(self.sigma))
        
        logger.info(f"\nAnnualized Volatilities:")
        for i, ticker in enumerate(self.tickers):
            logger.info(f"  {ticker:6s}: {self.volatilities[i]*100:5.1f}%")
        
        # Equilibrium returns (equal-weight for simplicity)
        self.w_eq = np.ones(self.N) / self.N
        self.pi = BlackLittermanModel.compute_equilibrium_returns(
            self.w_eq, self.sigma, self.risk_aversion
        )
        
        logger.info(f"\nEquilibrium Returns (λ = {self.risk_aversion}):")
        for i, ticker in enumerate(self.tickers):
            logger.info(f"  {ticker:6s}: {self.pi[i]*100:6.2f}%")
        
        # Initialize models
        self.bl_model = BlackLittermanModel(
            self.pi,
            self.sigma,
            use_market_formulation=True
        )
        
        self.view_generator = ViewGenerator(
            tickers=self.tickers,
            volatilities=self.volatilities,
            sentiment_scaling=self.sentiment_scaling,
            base_uncertainty=0.0001,
            news_volume_weight=0.5,
            consistency_weight=0.5,
            min_news_count=3
        )
    
    def run_live_optimization(
        self,
        min_abs_sentiment: float = 0.10,
        filter_weak_signals: bool = True
    ) -> Dict:
        """
        Run complete live optimization pipeline.
        
        This is the MASTER function:
        1. Load market data (Σ, π)
        2. Fetch live news for all tickers
        3. Analyze sentiment with FinBERT
        4. Generate Black-Litterman views (Q, Ω)
        5. Compute posterior (μ_BL, Σ_BL)
        6. Optimize portfolio
        
        Parameters:
            min_abs_sentiment: Filter views below this threshold
            filter_weak_signals: Remove weak/noisy signals
        
        Returns:
            Dictionary with complete results
        """
        logger.info(f"\n{'='*80}")
        logger.info("LIVE FINBERT → BLACK-LITTERMAN OPTIMIZATION")
        logger.info(f"{'='*80}")
        logger.info(f"Tickers: {', '.join(self.tickers)}")
        logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Step 1: Load market data
        if self.sigma is None:
            self.load_market_data()
        
        # Step 2: Fetch news and analyze sentiment for all tickers
        logger.info(f"\n{'='*80}")
        logger.info("FETCHING NEWS & ANALYZING SENTIMENT")
        logger.info(f"{'='*80}")
        
        sentiment_data_dict = {}
        
        for ticker in self.tickers:
            sentiment_data = self.process_ticker_news(ticker)
            if sentiment_data is not None:
                sentiment_data_dict[ticker] = sentiment_data
        
        if not sentiment_data_dict:
            logger.error("\n✗ No valid sentiment data - cannot optimize")
            return None
        
        # Step 3: Generate Black-Litterman views
        logger.info(f"\n{'='*80}")
        logger.info("GENERATING BLACK-LITTERMAN VIEWS")
        logger.info(f"{'='*80}")
        
        view = self.view_generator.generate_views(
            sentiment_data_dict,
            prior_returns=self.pi,
            filter_weak_signals=filter_weak_signals,
            min_abs_sentiment=min_abs_sentiment
        )
        
        logger.info(f"\n✓ Generated {view.P.shape[0]} views")
        logger.info(f"  Tickers: {view.metadata.get('tickers', [])}")
        
        # Display view details
        logger.info(f"\nView Specification:")
        logger.info(f"  Pick Matrix P: {view.P.shape}")
        logger.info(f"  View Returns Q: {view.Q}")
        logger.info(f"  Uncertainty Ω (diag): {np.diag(view.Omega)}")
        
        # View impact analysis
        impact_df = self.view_generator.analyze_view_impact(view, self.pi)
        
        logger.info(f"\nView Impact Analysis:")
        logger.info("\n" + impact_df.to_string(index=False))
        
        # Step 4: Compute Black-Litterman posterior
        logger.info(f"\n{'='*80}")
        logger.info("BLACK-LITTERMAN POSTERIOR COMPUTATION")
        logger.info(f"{'='*80}")
        
        mu_bl, sigma_bl = self.bl_model.compute_posterior(
            view.P,
            view.Q,
            Omega=view.Omega
        )
        
        # Validate
        validation = self.bl_model.validate_posterior(sigma_bl)
        logger.info(f"\nPosterior Validation:")
        logger.info(f"  Symmetric: {validation['is_symmetric']}")
        logger.info(f"  PSD: {validation['is_psd']}")
        if validation['warnings']:
            for warning in validation['warnings']:
                logger.warning(f"{warning}")
        
        # Display posterior returns
        logger.info(f"\nPosterior Returns:")
        for i, ticker in enumerate(self.tickers):
            prior = self.pi[i]
            posterior = mu_bl[i]
            delta = posterior - prior
            direction = "↑" if delta > 0 else "↓" if delta < 0 else "→"
            
            logger.info(
                f"  {ticker:6s}: {prior*100:6.2f}% → {posterior*100:6.2f}% "
                f"{direction} ({delta*100:+.2f}%)"
            )
        
        # Step 5: Optimize portfolio
        logger.info(f"\n{'='*80}")
        logger.info("PORTFOLIO OPTIMIZATION")
        logger.info(f"{'='*80}")
        
        # Mean-variance optimization
        try:
            sigma_inv = np.linalg.inv(sigma_bl)
        except np.linalg.LinAlgError:
            logger.warning("Singular covariance, using pseudo-inverse")
            sigma_inv = np.linalg.pinv(sigma_bl)
        
        # Unconstrained solution
        w_optimal = sigma_inv @ mu_bl / (2 * self.risk_aversion)
        
        # Normalize (long-only, sum to 1)
        w_optimal = np.maximum(w_optimal, 0)  # Long-only
        w_optimal = w_optimal / np.sum(w_optimal)  # Normalize
        
        logger.info(f"\nOptimal Weights:")
        for i, ticker in enumerate(self.tickers):
            prior_w = self.w_eq[i]
            optimal_w = w_optimal[i]
            delta = optimal_w - prior_w
            direction = "↑" if delta > 0 else "↓" if delta < 0 else "→"
            
            logger.info(
                f"  {ticker:6s}: {prior_w*100:5.1f}% → {optimal_w*100:5.1f}% "
                f"{direction} ({delta*100:+5.1f}%)"
            )
        
        # Portfolio statistics
        portfolio_return = w_optimal @ mu_bl
        portfolio_vol = np.sqrt(w_optimal @ sigma_bl @ w_optimal)
        sharpe = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
        
        logger.info(f"\nPortfolio Statistics:")
        logger.info(f"  Expected Return: {portfolio_return*100:6.2f}%")
        logger.info(f"  Volatility:      {portfolio_vol*100:6.2f}%")
        logger.info(f"  Sharpe Ratio:    {sharpe:.3f}")
        
        # Compile results
        results = {
            'timestamp': datetime.now(),
            'tickers': self.tickers,
            'sentiment_data': sentiment_data_dict,
            'view': view,
            'prior_returns': self.pi,
            'posterior_returns': mu_bl,
            'prior_weights': self.w_eq,
            'optimal_weights': w_optimal,
            'portfolio_return': portfolio_return,
            'portfolio_vol': portfolio_vol,
            'sharpe_ratio': sharpe,
            'sigma': self.sigma,
            'sigma_bl': sigma_bl
        }
        
        logger.info(f"\n{'='*80}")
        logger.info("✓ OPTIMIZATION COMPLETE")
        logger.info(f"{'='*80}\n")
        
        return results
    
    def export_results(self, results: Dict, filename: Optional[str] = None):
        """
        Export results to Excel.
        
        Parameters:
            results: Results dictionary from run_live_optimization
            filename: Output filename (auto-generated if None)
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"FinBERT_BL_Results_{timestamp}.xlsx"
        
        # Create summary DataFrame
        summary_data = []
        
        for i, ticker in enumerate(results['tickers']):
            # Get sentiment data if available
            sent_data = results['sentiment_data'].get(ticker)
            sentiment = sent_data.sentiment_mean if sent_data else None
            news_count = sent_data.news_count if sent_data else None
            
            summary_data.append({
                'Ticker': ticker,
                'Sentiment': sentiment,
                'News_Count': news_count,
                'Prior_Return': results['prior_returns'][i],
                'Posterior_Return': results['posterior_returns'][i],
                'Return_Adjustment': results['posterior_returns'][i] - results['prior_returns'][i],
                'Prior_Weight': results['prior_weights'][i],
                'Optimal_Weight': results['optimal_weights'][i],
                'Weight_Change': results['optimal_weights'][i] - results['prior_weights'][i]
            })
        
        df = pd.DataFrame(summary_data)
        
        # Write to Excel
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Format percentages
            worksheet = writer.sheets['Summary']
            for row in range(2, len(df) + 2):
                for col in ['D', 'E', 'F', 'G', 'H', 'I']:
                    cell = f'{col}{row}'
                    worksheet[cell].number_format = '0.00%'
        
        logger.info(f"✓ Results exported to {filename}")
        
        return filename


# ============================================================================
# Example Usage / Main Execution
# ============================================================================

def main():
    """
    Main execution: Live FinBERT → Meucci Black-Litterman
    """
    # Configuration
    NEWS_API_KEY = "YOUR_NEWS_API_KEY"  # ← CHANGE THIS
    
    # Portfolio universe
    TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    
    # Initialize optimizer
    optimizer = LiveFinBERTBlackLitterman(
        api_key=NEWS_API_KEY,
        tickers=TICKERS,
        lookback_days=252,          # 1 year historical data
        news_lookback_days=7,       # Last week's news
        articles_per_ticker=10,     # Max 10 articles per ticker
        risk_aversion=2.5,
        sentiment_scaling=0.02      # 2% max return impact
    )
    
    # Run live optimization
    results = optimizer.run_live_optimization(
        min_abs_sentiment=0.10,     # Filter weak signals
        filter_weak_signals=True
    )
    
    if results:
        # Export to Excel
        optimizer.export_results(results)
        
        # Print final summary
        print("\n" + "="*80)
        print("FINAL PORTFOLIO ALLOCATION")
        print("="*80)
        
        for i, ticker in enumerate(TICKERS):
            weight = results['optimal_weights'][i]
            ret = results['posterior_returns'][i]
            
            sent_data = results['sentiment_data'].get(ticker)
            sentiment = sent_data.sentiment_mean if sent_data else 0.0
            
            print(f"{ticker:6s}: {weight*100:5.1f}% | "
                  f"Return: {ret*100:5.1f}% | "
                  f"Sentiment: {sentiment:+.2f}")
        
        print("\nPortfolio Metrics:")
        print(f"  Expected Return: {results['portfolio_return']*100:.2f}%")
        print(f"  Volatility:      {results['portfolio_vol']*100:.2f}%")
        print(f"  Sharpe Ratio:    {results['sharpe_ratio']:.3f}")
        print("="*80)
    
    return results


if __name__ == "__main__":
    # Run live optimization
    results = main()
