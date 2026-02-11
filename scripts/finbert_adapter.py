"""
Adapter: Existing FinBERT.py → Production Black-Litterman Framework

This script shows how to integrate your existing FinBERT.py implementation
with the production Meucci Black-Litterman framework with MINIMAL changes.

Key Bridge:
    FinBERT Score = P(Positive) - P(Negative) 
    → ViewGenerator → (Q, Ω)
    → Black-Litterman → Optimal Weights
"""

import yfinance as yf
from newsapi import NewsApiClient
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Production framework
from bl_model import BlackLittermanModel
from view_generator import ViewGenerator, SentimentData

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# Step 1: Your existing FinBERT functions (from FinBERT.py)
# ============================================================================

def load_finbert():
    """Your existing function - unchanged"""
    tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")
    model = BertForSequenceClassification.from_pretrained("ProsusAI/finbert")
    return tokenizer, model


def fetch_stock_news(symbol, api_key, days_back=7):
    """
    Enhanced version of your function with date range.
    
    Original only got latest news, this adds time window.
    """
    newsapi = NewsApiClient(api_key=api_key)
    
    # Date range
    to_date = datetime.now()
    from_date = to_date - timedelta(days=days_back)
    
    articles_response = newsapi.get_everything(
        q=symbol,
        language='en',
        sort_by='relevancy',
        page_size=10,  # Increased from 5
        from_param=from_date.strftime('%Y-%m-%d'),
        to=to_date.strftime('%Y-%m-%d')
    )
    
    return articles_response['articles']


def analyze_sentiment(texts, tokenizer, model):
    """
    Your existing function - unchanged.
    
    Returns probability matrix (N × 3): [positive, negative, neutral]
    """
    if not texts:
        return None
    
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1).numpy()
    return probs


# ============================================================================
# Step 2: NEW BRIDGE FUNCTIONS (FinBERT → Meucci)
# ============================================================================

def finbert_to_sentiment_data(
    symbol: str,
    probs: np.ndarray,
    headlines: List[str]
) -> SentimentData:
    """
    Bridge: Convert FinBERT probabilities to SentimentData object.
    
    This is THE KEY FUNCTION that connects your FinBERT implementation
    to the production Black-Litterman framework.
    
    Parameters:
        symbol: Stock ticker
        probs: FinBERT probability matrix (N × 3)
        headlines: Original headlines (for reference)
    
    Returns:
        SentimentData object ready for ViewGenerator
    """
    # Compute sentiment scores: P(Positive) - P(Negative)
    # This is your existing formula!
    scores = probs[:, 0] - probs[:, 1]  # Shape: (N,)
    
    # Statistics
    sentiment_mean = float(np.mean(scores))
    sentiment_std = float(np.std(scores)) if len(scores) > 1 else 0.5
    news_count = len(scores)
    
    # Create SentimentData object
    sentiment_data = SentimentData(
        ticker=symbol,
        sentiment_mean=sentiment_mean,
        sentiment_std=sentiment_std,
        news_count=news_count,
        raw_scores=scores.tolist()
    )
    
    return sentiment_data


# ============================================================================
# Step 3: PORTFOLIO OPTIMIZATION WRAPPER
# ============================================================================

def run_finbert_black_litterman(
    symbols: List[str],
    api_key: str,
    lookback_days: int = 252,
    risk_aversion: float = 2.5,
    sentiment_scaling: float = 0.02
):
    """
    Complete pipeline: FinBERT → Black-Litterman → Portfolio Weights
    
    This replaces your manual aggregation with production framework.
    
    Parameters:
        symbols: List of tickers (e.g., ['AAPL', 'TSLA', 'GOOGL'])
        api_key: NewsAPI key
        lookback_days: Historical data for covariance
        risk_aversion: Portfolio risk aversion λ
        sentiment_scaling: Max return impact from sentiment
    
    Returns:
        Dictionary with complete results
    """
    logger.info("="*80)
    logger.info("FINBERT → BLACK-LITTERMAN PORTFOLIO OPTIMIZATION")
    logger.info("="*80)
    
    # Initialize FinBERT
    logger.info("\n1. Loading FinBERT model...")
    tokenizer, model = load_finbert()
    logger.info("   ✓ FinBERT ready")
    
    # ========================================================================
    # PHASE 1: Fetch News & Analyze Sentiment (Your existing code)
    # ========================================================================
    
    logger.info("\n2. Fetching news and analyzing sentiment...")
    sentiment_data_dict = {}
    
    for symbol in symbols:
        logger.info(f"\n   Processing {symbol}:")
        
        # A. Fetch news (your function)
        raw_articles = fetch_stock_news(symbol, api_key)
        
        if not raw_articles:
            logger.warning(f"No articles for {symbol}")
            continue
        
        # B. Extract headlines (your code)
        headlines = [art['title'] for art in raw_articles]
        logger.info(f"      Found {len(headlines)} articles")
        
        # C. Analyze sentiment (your function)
        probs = analyze_sentiment(headlines, tokenizer, model)
        
        if probs is None:
            continue
        
        # D. Convert to production format (NEW)
        sentiment_data = finbert_to_sentiment_data(symbol, probs, headlines)
        sentiment_data_dict[symbol] = sentiment_data
        
        # E. Display results
        logger.info(f"      Sentiment: {sentiment_data.sentiment_mean:+.3f}")
        logger.info(f"      Std Dev:   {sentiment_data.sentiment_std:.3f}")
        logger.info(f"      Sample: {headlines[0][:60]}...")
    
    if not sentiment_data_dict:
        logger.error("\n✗ No valid sentiment data")
        return None
    
    # ========================================================================
    # PHASE 2: Load Market Data & Estimate Parameters (NEW)
    # ========================================================================
    
    logger.info("\n3. Loading market data...")
    
    # Download historical prices
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days + 30)
    
    returns_dict = {}
    for symbol in symbols:
        try:
            data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            if not data.empty:
                returns_dict[symbol] = np.log(data['Close'] / data['Close'].shift(1)).dropna()
                logger.info(f"   ✓ {symbol}: {len(returns_dict[symbol])} days")
        except Exception as e:
            logger.error(f"   ✗ {symbol}: {e}")
    
    # Create returns DataFrame
    returns_df = pd.DataFrame(returns_dict).dropna()
    
    # Estimate covariance (annualized)
    N = len(symbols)
    sigma = returns_df.cov().values * 252
    sigma += np.eye(N) * 1e-6  # Numerical stability
    
    # Volatilities
    volatilities = np.sqrt(np.diag(sigma))
    
    logger.info("\n   Volatilities (annualized):")
    for i, symbol in enumerate(symbols):
        logger.info(f"      {symbol}: {volatilities[i]*100:5.1f}%")
    
    # Equilibrium returns (equal-weight)
    w_eq = np.ones(N) / N
    pi = BlackLittermanModel.compute_equilibrium_returns(w_eq, sigma, risk_aversion)
    
    logger.info("\n   Equilibrium Returns:")
    for i, symbol in enumerate(symbols):
        logger.info(f"      {symbol}: {pi[i]*100:5.1f}%")
    
    # ========================================================================
    # PHASE 3: Generate Black-Litterman Views (NEW)
    # ========================================================================
    
    logger.info("\n4. Generating Black-Litterman views...")
    
    # Initialize ViewGenerator
    generator = ViewGenerator(
        tickers=symbols,
        volatilities=volatilities,
        sentiment_scaling=sentiment_scaling,
        base_uncertainty=0.0001,
        news_volume_weight=0.5,
        consistency_weight=0.5,
        min_news_count=3
    )
    
    # Generate views from sentiment
    view = generator.generate_views(
        sentiment_data_dict,
        prior_returns=pi,
        filter_weak_signals=True,
        min_abs_sentiment=0.10
    )
    
    logger.info(f"\n   ✓ Generated {view.P.shape[0]} views")
    logger.info(f"   Tickers: {view.metadata.get('tickers', [])}")
    
    # View impact analysis
    impact_df = generator.analyze_view_impact(view, pi)
    logger.info("\n   View Impact:")
    logger.info("\n" + impact_df.to_string(index=False))
    
    # ========================================================================
    # PHASE 4: Black-Litterman Posterior (NEW)
    # ========================================================================
    
    logger.info("\n5. Computing Black-Litterman posterior...")
    
    # Initialize BL model
    bl = BlackLittermanModel(pi, sigma, use_market_formulation=True)
    
    # Compute posterior
    mu_bl, sigma_bl = bl.compute_posterior(view.P, view.Q, Omega=view.Omega)
    
    logger.info("\n   Posterior Returns:")
    for i, symbol in enumerate(symbols):
        delta = mu_bl[i] - pi[i]
        direction = "↑" if delta > 0 else "↓" if delta < 0 else "→"
        logger.info(
            f"      {symbol}: {pi[i]*100:5.1f}% → {mu_bl[i]*100:5.1f}% "
            f"{direction} ({delta*100:+.1f}%)"
        )
    
    # ========================================================================
    # PHASE 5: Portfolio Optimization (NEW)
    # ========================================================================
    
    logger.info("\n6. Optimizing portfolio...")
    
    # Mean-variance optimization
    try:
        sigma_inv = np.linalg.inv(sigma_bl)
    except:
        sigma_inv = np.linalg.pinv(sigma_bl)
    
    # Analytical solution: w* = (1/2λ) Σ^(-1) μ
    w_optimal = sigma_inv @ mu_bl / (2 * risk_aversion)
    
    # Long-only constraint
    w_optimal = np.maximum(w_optimal, 0)
    w_optimal = w_optimal / np.sum(w_optimal)
    
    logger.info("\n   Optimal Weights:")
    for i, symbol in enumerate(symbols):
        logger.info(f"      {symbol}: {w_optimal[i]*100:5.1f}%")
    
    # Portfolio statistics
    portfolio_return = w_optimal @ mu_bl
    portfolio_vol = np.sqrt(w_optimal @ sigma_bl @ w_optimal)
    sharpe = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
    
    logger.info("\n   Portfolio Statistics:")
    logger.info(f"      Expected Return: {portfolio_return*100:5.1f}%")
    logger.info(f"      Volatility:      {portfolio_vol*100:5.1f}%")
    logger.info(f"      Sharpe Ratio:    {sharpe:.2f}")
    
    # ========================================================================
    # Return Results
    # ========================================================================
    
    results = {
        'symbols': symbols,
        'sentiment_data': sentiment_data_dict,
        'view': view,
        'prior_returns': pi,
        'posterior_returns': mu_bl,
        'equilibrium_weights': w_eq,
        'optimal_weights': w_optimal,
        'portfolio_return': portfolio_return,
        'portfolio_vol': portfolio_vol,
        'sharpe_ratio': sharpe,
        'sigma': sigma,
        'sigma_bl': sigma_bl
    }
    
    logger.info("\n" + "="*80)
    logger.info("✓ OPTIMIZATION COMPLETE")
    logger.info("="*80 + "\n")
    
    return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Example: Using your existing FinBERT code with production BL framework
    """
    # CONFIGURATION
    NEWS_API_KEY = 'YOUR_API_KEY'  # ← Your free API key here  https://newsapi.org/
    
    # Your portfolio (from your FinBERT.py)
    symbols = ['AAPL', 'TSLA', 'MSFT', 'GOOGL', 'NVDA']
    
    print("\n" + "="*80)
    print("FINBERT.PY → PRODUCTION BLACK-LITTERMAN INTEGRATION")
    print("="*80)
    print("\nThis script uses:")
    print("  - Your existing FinBERT sentiment analysis")
    print("  - Production Meucci Black-Litterman framework")
    print("  - Automated portfolio optimization")
    
    # Run complete pipeline
    results = run_finbert_black_litterman(
        symbols=symbols,
        api_key=NEWS_API_KEY,
        lookback_days=252,
        risk_aversion=2.5,
        sentiment_scaling=0.02
    )
    
    if results:
        # Export to Excel
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"FinBERT_BL_Results_{timestamp}.xlsx"
        
        # Create DataFrame
        data = []
        for i, symbol in enumerate(results['symbols']):
            sent_data = results['sentiment_data'].get(symbol)
            
            data.append({
                'Ticker': symbol,
                'Sentiment': sent_data.sentiment_mean if sent_data else None,
                'News_Count': sent_data.news_count if sent_data else None,
                'Prior_Return': results['prior_returns'][i],
                'Posterior_Return': results['posterior_returns'][i],
                'Equilibrium_Weight': results['equilibrium_weights'][i],
                'Optimal_Weight': results['optimal_weights'][i],
                'Weight_Change': results['optimal_weights'][i] - results['equilibrium_weights'][i]
            })
        
        df = pd.DataFrame(data)
        df.to_excel(filename, index=False)
        
        print(f"\n✓ Results exported to: {filename}")
        
        # Display final allocation
        print("\n" + "="*80)
        print("FINAL PORTFOLIO ALLOCATION")
        print("="*80)
        print(f"\n{'Ticker':<8} {'Sentiment':<12} {'Weight':<10} {'Expected Return'}")
        print("-" * 60)
        
        for i, symbol in enumerate(results['symbols']):
            sent_data = results['sentiment_data'].get(symbol)
            sentiment = sent_data.sentiment_mean if sent_data else 0.0
            weight = results['optimal_weights'][i]
            ret = results['posterior_returns'][i]
            
            print(f"{symbol:<8} {sentiment:>+.3f}         {weight*100:>5.1f}%     {ret*100:>5.1f}%")
        
        print("-" * 60)
        print(f"Portfolio Return: {results['portfolio_return']*100:>5.1f}%")
        print(f"Portfolio Vol:    {results['portfolio_vol']*100:>5.1f}%")
        print(f"Sharpe Ratio:     {results['sharpe_ratio']:>5.2f}")
        print("="*80 + "\n")
    
    return results


if __name__ == "__main__":
    # Check if required modules are available
    try:
        from bl_model import BlackLittermanModel
        from view_generator import ViewGenerator, SentimentData
        
        # Run main
        results = main()
        
    except ImportError as e:
        print("\n✗ Error: Missing required modules")
        print("\nPlease ensure these files are in the same directory:")
        print("  - bl_model.py")
        print("  - view_generator.py")
        print("\nOr install from the provided package.")
        print(f"\nError details: {e}")
