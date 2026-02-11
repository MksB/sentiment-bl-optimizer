# Sentiment-Enhanced Black-Litterman Portfolio Optimizer

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **Production-ready implementation of the Black-Litterman model enhanced with FinBERT sentiment analysis for institutional portfolio optimization**

## Overview

This repository provides a **complete, production-grade framework** that bridges **Natural Language Processing** and **Quantitative Finance** by integrating:

- **Meucci's (2008) market-based Black-Litterman formulation** for stable portfolio optimization
- **FinBERT sentiment analysis** (Araci, 2019) for extracting actionable views from financial news
- **Real-time NewsAPI integration** for live sentiment-driven rebalancing
- **Interactive Plotly visualizations** comparing Market-Cap, Mean-Variance, and Black-Litterman strategies

### Key Innovation: The Sentiment → View Calibration Bridge

Traditional Mean-Variance optimization suffers from estimation error, leading to unstable portfolios with extreme weights. Black-Litterman mitigates this through Bayesian shrinkage, but requires manually specified views. **We automate view generation** from unstructured news data:

```
Financial News → FinBERT → Sentiment Scores → Black-Litterman Views → Optimal Portfolio
```

**Mathematical Rigor:**
- Volatility-scaled view returns: `Q[k] = π[i] + sentiment × σ[i] × α`
- Dual-factor uncertainty: `Ω[k,k] = β + w_vol/√n + w_cons × std²`
- Meucci's posterior (Equations 32-33): No hyperparameter τ, correct limiting behavior

---

## Use Cases

- **Asset Management**: Systematic sentiment integration for equity portfolios
- **Risk Management**: Solvency II/IFRS 17 compliant portfolio construction
- **Quantitative Research**: Backtesting sentiment-driven strategies
- **Academic Research**: Reproducible implementation of Meucci (2008) + FinBERT

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/sentiment-bl-optimizer.git
cd sentiment-bl-optimizer

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from sentiment_bl_optimizer import SentimentBlackLittermanOptimizer

# Initialize with your NewsAPI key
optimizer = SentimentBlackLittermanOptimizer(
    api_key="YOUR_NEWS_API_KEY",
    tickers=['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA'],
    lookback_days=252,
    risk_aversion=2.5
)

# Run complete optimization pipeline
results = optimizer.optimize()

# Display results
print(f"Optimal Weights: {results['optimal_weights']}")
print(f"Expected Return: {results['portfolio_return']*100:.2f}%")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.3f}")
```

### Google Colab (No Installation Required)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/sentiment-bl-optimizer/blob/main/notebooks/demo.ipynb)

```python
# Simply run in Colab - dependencies auto-install
!wget https://raw.githubusercontent.com/yourusername/sentiment-bl-optimizer/main/complete_bl_finbert_colab_FINAL.py

# Execute with your API key
NEWS_API_KEY = "your_key_here"
TICKERS = ['AAPL', 'MSFT', 'GOOGL']

# Run optimization
%run complete_bl_finbert_colab_FINAL.py
```

---

##  Mathematical Foundation

### Black-Litterman Model (Meucci 2008)

**Prior Distribution (CAPM Equilibrium):**
```
π = δΣw_mkt
```
where δ is risk aversion (typically 2.5), Σ is covariance matrix, w_mkt is market-cap weights.

**Investor Views:**
```
P X ~ N(Q, Ω)
```
- P (K×N): Pick matrix selecting assets
- Q (K×1): Expected view returns  
- Ω (K×K): View uncertainty matrix

**Posterior Distribution (Market Formulation):**
```
μ_BL = π + ΣP'(PΣP' + Ω)⁻¹(Q - Pπ)
Σ_BL = Σ - ΣP'(PΣP' + Ω)⁻¹PΣ
```
*(Meucci 2008, Equations 32-33)*

**Advantages over original formulation:**
- No hyperparameter τ
- Correct limiting behavior (Ω→∞ ⇒ posterior→prior)
- Cleaner interpretation

### FinBERT Sentiment Analysis

**Sentiment Score:**
```
score = P(Positive) - P(Negative) ∈ [-1, +1]
```

**View Calibration:**

1. **View Returns (Volatility-Scaled):**
   ```
   Q[k] = π[i] + sentiment[k] × σ[i] × α
   ```
   - σ[i]: Asset volatility (anchors magnitude)
   - α: Scaling factor (default 0.02 = 2% max impact)
   
   *Rationale*: High-vol assets receive proportionally larger adjustments

2. **Uncertainty Matrix (Dual-Factor):**
   ```
   Ω[k,k] = β + w_vol × (1/√n) + w_cons × std²
   ```
   - β: Base uncertainty (0.0001)
   - n: Number of news articles
   - std: Sentiment consistency
   - w_vol = w_cons = 0.5 (default weights)
   
   *Rationale*: More news + low variance → high confidence → low Ω

**Example Calibration:**

| Asset | Sentiment | News Count | Std Dev | σ | Q (Prior 10%) | Ω |
|-------|-----------|------------|---------|---|---------------|------|
| AAPL | +0.65 | 25 | 0.18 | 28% | 10.36% | 0.116 |
| TSLA | -0.40 | 8 | 0.35 | 35% | 9.72% | 0.274 |

---

## Architecture

### Core Components

```
sentiment-bl-optimizer/
│
├── src/
│   ├── bl_model.py                    # Black-Litterman core (Meucci 2008)
│   ├── view_generator.py              # Sentiment → (P, Q, Ω) calibration
│   └── sentiment_bl_optimizer.py      # FinBERT integration, Complete optimization pipeline    # 
│
├── notebooks/
│   ├── Data_Aquise_Views_BL.ipynb     		    # Data-Akquise & Sentiment Engine
│   ├── Complete_Finbert_BL_Tutorial.ipynb  	# Complete FinBERT BL-Tutorial und Colab-capable
│   └── Black_Litterman_Model_Demo_Cases.ipynb  # BL-Example demonstrtes use cases and features
│
├── scripts/
│   ├── complete_bl_finbert_colab_FINAL.py  # Standalone Colab version
│   ├── portfolio_comparison_plots.py       # Visualization suite
│   └── finbert_adapter.py                  # Legacy integration
│
├── docs/
│   ├── Technical_Whitepaper.pdf       # Complete mathematical derivation
│   ├-─ ssrn-1117574.pdf               # Meucci: The Black-Litterman Approach
│   ├-─ 1908.10063v1.pdf  			   # Araci: FINBERT: Financial Sentiment Analysis 
│     
├── tests/
│   ├── test_bl_model.py               # Unit tests (26 passing)
│   └── bl_examples.py
│   
│
├── requirements.txt
└── README.md
```

### Class Hierarchy

```python
# Core Black-Litterman Model
class BlackLittermanModel:
    """
    Implements Meucci (2008) market-based formulation.
    
    Methods:
    - compute_equilibrium_returns(w_eq, Sigma, delta) → π
    - compute_posterior(P, Q, Omega) → (μ_BL, Σ_BL)
    """

# Sentiment → View Translation
class ViewGenerator:
    """
    Calibrates FinBERT sentiment into Black-Litterman views.
    
    Methods:
    - generate_views(sentiment_data, prior_returns) → BlackLittermanView
    - analyze_view_impact(view, prior) → pd.DataFrame
    """

# FinBERT Sentiment Analysis
class FinBERTAnalyzer:
    """
    Sentiment extraction using ProsusAI/finbert.
    
    Methods:
    - analyze_sentiment(texts: List[str]) → (probs, mean, std)
    """

# Complete Pipeline
class SentimentBlackLittermanOptimizer:
    """
    End-to-end: NewsAPI → FinBERT → Black-Litterman → Portfolio.
    
    Methods:
    - load_market_data() → Estimates Σ, π
    - fetch_news(ticker) → Live articles from NewsAPI
    - process_ticker(ticker) → SentimentData
    - optimize() → Results dictionary
    """
```

---

## Visualization & Analysis

### Portfolio Comparison Dashboard

The framework automatically generates interactive Plotly visualizations comparing three strategies:

1. **Market-Cap Weighted** (Green) - Benchmark
2. **Mean-Variance** (Orange) - Classic Markowitz (often unstable)
3. **Sentiment Black-Litterman** (Red) - Our approach

**Example Output:**

```
================================================================================
PERFORMANCE COMPARISON TABLE
================================================================================

Strategy             Return       Volatility    Sharpe    
------------------------------------------------------------
Market-Cap             9.20%       20.50%      0.449
Mean-Variance         10.80%       23.10%      0.468
Black-Litterman       10.30%       19.80%      0.520  ⭐ BEST
------------------------------------------------------------

KEY INSIGHTS:

 BLACK-LITTERMAN:
   - Sharpe improvement vs benchmark: +15.8%
   - Max concentration: 36.2% (diversified)
   - Stable across rebalancing

 MEAN-VARIANCE:
   - Max concentration: 78.3% (UNSTABLE)
   - Higher volatility than BL despite similar return
   - Sensitive to input perturbations
```

**Interactive Features:**
- Efficient frontier with all three strategies
- Weight evolution over time
- Risk-return scatter
- Sharpe ratio comparison
- HTML export for sharing

---

##  Configuration & Parameters

### Key Parameters

```python
# Portfolio Universe
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']

# Historical Data
LOOKBACK_DAYS = 252  # 1 year for covariance estimation

# News Sentiment
NEWS_LOOKBACK_DAYS = 7      # Last week's news
ARTICLES_PER_TICKER = 10    # Max articles to fetch
MIN_NEWS_COUNT = 3          # Min articles to generate view
MIN_ABS_SENTIMENT = 0.15    # Filter weak signals

# Black-Litterman
RISK_AVERSION = 2.5         # δ parameter (2.0-3.5 typical)

# Sentiment Calibration
SENTIMENT_SCALING = 0.02    # α: Max return impact (1-5%)
BASE_UNCERTAINTY = 0.0001   # β: Minimum Ω
NEWS_VOLUME_WEIGHT = 0.5    # w_vol in Ω formula
CONSISTENCY_WEIGHT = 0.5    # w_cons in Ω formula
```

### Parameter Tuning Guidelines

| Parameter | Conservative | Standard | Aggressive |
|-----------|--------------|----------|------------|
| `sentiment_scaling` | 0.01 (1%) | 0.02 (2%) | 0.05 (5%) |
| `risk_aversion` | 3.5 | 2.5 | 1.5 |
| `min_abs_sentiment` | 0.25 | 0.15 | 0.10 |
| `news_volume_weight` | 0.7 | 0.5 | 0.3 |
| `consistency_weight` | 0.3 | 0.5 | 0.7 |

**Recommendation:** Start with standard values, then backtest different configurations.

---

##  Academic References


1. **Meucci, A. (2008).** *The Black-Litterman Approach: Original Model and Extensions.* SSRN:1117574.
   - Market-based formulation (Equations 32-33)
   - Theoretical foundation of this implementation

2. **Araci, D. (2019).** *FinBERT: Financial Sentiment Analysis with Pre-Trained Language Models.* arXiv:1908.10063.
   - FinBERT architecture and training
   - Sentiment score interpretation

3. **Black, F., & Litterman, R. (1992).** *Global Portfolio Optimization.* Financial Analysts Journal, 48(5), 28–43.
   - Original Black-Litterman model
   - CAPM equilibrium as prior

4. **Meucci, A. (2005).** *Risk and Asset Allocation.* Springer.
   - Comprehensive portfolio theory
   - Covariance estimation techniques

5. **Ledoit, O., & Wolf, M. (2004).** *Honey, I Shrunk the Sample Covariance Matrix.* Journal of Portfolio Management.
   - Covariance shrinkage methodology

6. **Markowitz, H. (1952).** *Portfolio Selection.* Journal of Finance, 7(1), 77–91.
   - Foundation of modern portfolio theory

---

## 🔬 Validation & Testing

### Unit Tests

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

**Test Coverage:**
- `test_bl_model.py`: 26 tests covering equilibrium, posterior, edge cases
- `test_view_generator.py`: 15 tests for sentiment calibration
- `test_integration.py`: End-to-end pipeline validation

### Numerical Validation

**Posterior Properties (Validated):**
```python
# 1. Symmetry: Σ_BL = Σ_BL'
assert np.allclose(Sigma_BL, Sigma_BL.T)

# 2. Positive Semi-Definite
eigenvalues = np.linalg.eigvalsh(Sigma_BL)
assert np.all(eigenvalues >= -1e-10)

# 3. Limiting Behavior
# Ω → ∞: μ_BL → π (views ignored)
# Ω → 0: μ_BL → Q (views fully trusted)
```

---

## Known Limitations & Future Work

### Current Limitations

1. **Equity Focus**: Currently optimized for equity portfolios
2. **Normal Distribution Assumption**: Returns assumed Gaussian
3. **Single-Period Optimization**: No multi-period dynamics
4. **English News Only**: NewsAPI limited to English articles

### Roadmap

- [ ] Multi-asset class support (Equities + Bonds + Alternatives)
- [ ] Non-normal distributions (Student-t, copulas)
- [ ] Multi-period optimization with transaction costs
- [ ] Ensemble sentiment (Twitter + News + SEC filings)
- [ ] Live deployment template (Docker + Kubernetes)
- [ ] ESG integration (sustainability views)

---


### Development Setup

```bash
git clone https://github.com/yourusername/sentiment-bl-optimizer.git
cd sentiment-bl-optimizer
python -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt
pre-commit install
pytest tests/ -v
```

---

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **Attilio Meucci** for the rigorous mathematical framework (2008)
- **Fischer Black & Robert Litterman** for the original model (1992)
- **Dogu Araci** for FinBERT (2019)
- **ProsusAI** for the pre-trained FinBERT model
- **NewsAPI** for real-time financial news access

---

## Contact & Support


- **Email**: marksquant@gmail.com

---

**Built with efficiency for quantitative finance and institutional asset management**

*Last Updated: February 10, 2026*
