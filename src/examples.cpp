/**
 * @file examples.cpp
 * @brief Usage examples for Black-Litterman C++ implementation
 * 
 * Compile with MSVC 2026:
 *   cl /std:c++17 /EHsc /I"path\to\eigen" examples.cpp /Fe:bl_examples.exe
 * 
 * Or with CMake:
 *   cmake -G "Visual Studio 17 2026" ..
 *   cmake --build . --config Release
 * 
 * Run:
 *   bl_examples.exe
 */

#include "BlackLittermanModel.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>

using namespace portfolio;
using namespace Eigen;

/**
 * @brief Print vector with label
 */
void printVector(const std::string& label, const VectorXd& vec) {
    std::cout << label << ":\n";
    for (int i = 0; i < vec.size(); ++i) {
        std::cout << "  [" << i << "] " << std::fixed << std::setprecision(4) 
                  << vec(i) * 100 << "%\n";
    }
    std::cout << "\n";
}

/**
 * @brief Print matrix with label
 */
void printMatrix(const std::string& label, const MatrixXd& mat) {
    std::cout << label << ":\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << mat << "\n\n";
}

/**
 * @brief Print separator line
 */
void printSeparator(char c = '=', int width = 80) {
    std::cout << std::string(width, c) << "\n";
}

// ============================================================================
// EXAMPLE 1: Basic Black-Litterman with Single View
// ============================================================================

void example1_basic_single_view() {
    printSeparator();
    std::cout << "EXAMPLE 1: Basic Black-Litterman with Single View\n";
    printSeparator();
    std::cout << "\n";
    
    // Setup: 3 assets
    const int N = 3;
    std::vector<std::string> tickers = {"AAPL", "MSFT", "GOOGL"};
    
    // Covariance matrix (annualized)
    MatrixXd sigma(N, N);
    sigma << 0.08, 0.04, 0.05,
             0.04, 0.06, 0.03,
             0.05, 0.03, 0.07;
    
    std::cout << "Covariance Matrix (annualized):\n" << sigma << "\n\n";
    
    // Market equilibrium weights (equal-weighted as proxy)
    VectorXd w_eq(N);
    w_eq << 1.0/3, 1.0/3, 1.0/3;
    
    std::cout << "Equilibrium Weights:\n" << w_eq.transpose() << "\n\n";
    
    // Compute equilibrium returns
    double risk_aversion = 2.5;
    VectorXd pi = BlackLittermanModel::computeEquilibriumReturns(
        w_eq, sigma, risk_aversion
    );
    
    printVector("Equilibrium Returns (pi)", pi);
    
    // Initialize model
    BlackLittermanModel bl_model(pi, sigma);
    
    // Create view: "AAPL will outperform by 5%"
    BlackLittermanView view;
    view.P = MatrixXd(1, N);
    view.P << 1, 0, 0;  // Pick AAPL
    
    view.Q = VectorXd(1);
    view.Q << 0.15;  // 15% expected return
    
    view.Omega = MatrixXd(1, 1);
    view.Omega << 0.01;  // 1% uncertainty
    
    std::cout << "View Specification:\n";
    std::cout << "  Asset: AAPL (index 0)\n";
    std::cout << "  Expected Return: " << view.Q(0)*100 << "%\n";
    std::cout << "  Uncertainty: " << view.Omega(0,0)*100 << "%\n\n";
    
    // Compute posterior
    auto posterior = bl_model.computePosterior(view);
    VectorXd mu_BL = posterior.first;
    MatrixXd Sigma_BL = posterior.second;
    
    printVector("Posterior Returns (mu_BL)", mu_BL);
    std::cout << "Posterior Covariance (Sigma_BL):\n" << Sigma_BL << "\n\n";
    
    // Validate
    bool valid = BlackLittermanModel::validatePosterior(Sigma_BL);
    std::cout << "Posterior validation: " << (valid ? "PASS" : "FAIL") << "\n\n";
    
    // Optimize portfolio
    VectorXd w_optimal = BlackLittermanModel::optimizePortfolio(
        mu_BL, Sigma_BL, risk_aversion, true
    );
    
    std::cout << "Optimal Portfolio Weights:\n";
    for (int i = 0; i < N; ++i) {
        std::cout << "  " << tickers[i] << ": " << std::fixed 
                  << std::setprecision(1) << w_optimal(i)*100 << "%\n";
    }
    std::cout << "\n";
    
    // Portfolio statistics
    auto stats = BlackLittermanModel::computePortfolioStats(
        w_optimal, mu_BL, Sigma_BL
    );
    double ret = std::get<0>(stats);
    double vol = std::get<1>(stats);
    double sharpe = std::get<2>(stats);
    
    std::cout << "Portfolio Statistics:\n";
    std::cout << "  Expected Return: " << ret*100 << "%\n";
    std::cout << "  Volatility:      " << vol*100 << "%\n";
    std::cout << "  Sharpe Ratio:    " << sharpe << "\n\n";
}

// ============================================================================
// EXAMPLE 2: Multiple Views
// ============================================================================

void example2_multiple_views() {
    printSeparator();
    std::cout << "EXAMPLE 2: Black-Litterman with Multiple Views\n";
    printSeparator();
    std::cout << "\n";
    
    // Setup: 5 tech stocks
    const int N = 5;
    std::vector<std::string> tickers = {"AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"};
    
    // Covariance matrix (simplified for example)
    MatrixXd sigma = MatrixXd::Identity(N, N) * 0.06;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (i != j) {
                sigma(i, j) = 0.03;  // Correlation
            }
        }
    }
    
    // Equilibrium
    VectorXd w_eq = VectorXd::Constant(N, 1.0 / N);
    VectorXd pi = BlackLittermanModel::computeEquilibriumReturns(w_eq, sigma, 2.5);
    
    printVector("Prior Returns (pi)", pi);
    
    // Create multiple views
    const int K = 3;  // 3 views
    
    BlackLittermanView view;
    view.P = MatrixXd::Zero(K, N);
    view.Q = VectorXd(K);
    view.Omega = MatrixXd::Zero(K, K);
    
    // View 1: AAPL will return 12%
    view.P(0, 0) = 1.0;
    view.Q(0) = 0.12;
    view.Omega(0, 0) = 0.005;
    
    // View 2: TSLA will return 8%
    view.P(1, 3) = 1.0;
    view.Q(1) = 0.08;
    view.Omega(1, 1) = 0.02;  // Lower confidence
    
    // View 3: NVDA will outperform GOOGL by 3%
    view.P(2, 4) = 1.0;   // NVDA
    view.P(2, 2) = -1.0;  // GOOGL
    view.Q(2) = 0.03;
    view.Omega(2, 2) = 0.01;
    
    std::cout << "View Specifications:\n";
    std::cout << "  1. AAPL: 12% (Omega=0.5%)\n";
    std::cout << "  2. TSLA: 8% (Omega=2.0%)\n";
    std::cout << "  3. NVDA - GOOGL: 3% (Omega=1.0%)\n\n";
    
    // Black-Litterman
    BlackLittermanModel bl_model(pi, sigma);
    auto posterior = bl_model.computePosterior(view);
    VectorXd mu_BL = posterior.first;
    MatrixXd Sigma_BL = posterior.second;
    
    std::cout << "Prior vs Posterior Returns:\n";
    std::cout << std::setw(10) << "Ticker" << std::setw(15) << "Prior" 
              << std::setw(15) << "Posterior" << std::setw(15) << "Change\n";
    std::cout << std::string(55, '-') << "\n";
    
    for (int i = 0; i < N; ++i) {
        double change = mu_BL(i) - pi(i);
        std::cout << std::setw(10) << tickers[i]
                  << std::setw(14) << std::fixed << std::setprecision(2) 
                  << pi(i)*100 << "%"
                  << std::setw(14) << mu_BL(i)*100 << "%"
                  << std::setw(14) << (change >= 0 ? "+" : "") << change*100 << "%\n";
    }
    std::cout << "\n";
    
    // Optimize
    VectorXd w_optimal = BlackLittermanModel::optimizePortfolio(mu_BL, Sigma_BL);
    
    std::cout << "Optimal Allocation:\n";
    for (int i = 0; i < N; ++i) {
        std::cout << "  " << tickers[i] << ": " 
                  << std::setw(5) << std::fixed << std::setprecision(1) 
                  << w_optimal(i)*100 << "%\n";
    }
    std::cout << "\n";
}

// ============================================================================
// EXAMPLE 3: Sentiment-Driven Views (Simulation)
// ============================================================================

void example3_sentiment_views() {
    printSeparator();
    std::cout << "EXAMPLE 3: Sentiment-Driven Black-Litterman\n";
    printSeparator();
    std::cout << "\n";
    
    const int N = 4;
    std::vector<std::string> tickers = {"XLK", "XLE", "XLF", "XLV"};
    
    // Simulated market data
    MatrixXd sigma(N, N);
    sigma << 0.09, 0.02, 0.03, 0.02,
             0.02, 0.12, 0.04, 0.03,
             0.03, 0.04, 0.08, 0.02,
             0.02, 0.03, 0.02, 0.06;
    
    VectorXd w_eq = VectorXd::Constant(N, 0.25);
    VectorXd pi = BlackLittermanModel::computeEquilibriumReturns(w_eq, sigma);
    
    // Simulated sentiment scores (from FinBERT)
    struct SentimentData {
        std::string ticker;
        double sentiment_mean;    // [-1, +1]
        double sentiment_std;
        int news_count;
    };
    
    std::vector<SentimentData> sentiments = {
        {"XLK",  0.65, 0.18, 25},  // Tech: Positive
        {"XLE", -0.40, 0.35, 8},   // Energy: Negative
        {"XLF",  0.12, 0.45, 5},   // Finance: Weak positive
        {"XLV",  0.30, 0.20, 15}   // Health: Moderate positive
    };
    
    std::cout << "Sentiment Analysis:\n";
    std::cout << std::setw(10) << "Ticker" << std::setw(15) << "Sentiment" 
              << std::setw(12) << "StdDev" << std::setw(12) << "Articles\n";
    std::cout << std::string(49, '-') << "\n";
    
    for (const auto& s : sentiments) {
        std::cout << std::setw(10) << s.ticker
                  << std::setw(14) << std::fixed << std::setprecision(3) 
                  << (s.sentiment_mean >= 0 ? "+" : "") << s.sentiment_mean
                  << std::setw(12) << s.sentiment_std
                  << std::setw(12) << s.news_count << "\n";
    }
    std::cout << "\n";
    
    // Generate views from sentiment
    // Q = pi + sentiment × σ × scaling
    const double sentiment_scaling = 0.02;
    const double base_uncertainty = 0.0001;
    const double vol_weight = 0.5;
    const double cons_weight = 0.5;
    
    BlackLittermanView view;
    view.P = MatrixXd::Identity(N, N);
    view.Q = VectorXd(N);
    view.Omega = MatrixXd::Zero(N, N);
    
    std::cout << "View Calibration:\n";
    for (int i = 0; i < N; ++i) {
        const auto& s = sentiments[i];
        
        // Volatility-scaled adjustment
        double vol_i = std::sqrt(sigma(i, i));
        double adjustment = s.sentiment_mean * vol_i * sentiment_scaling;
        view.Q(i) = pi(i) + adjustment;
        
        // Dual-factor uncertainty
        double vol_unc = 1.0 / std::sqrt(static_cast<double>(s.news_count));
        double cons_unc = s.sentiment_std * s.sentiment_std;
        view.Omega(i, i) = base_uncertainty + 
                          vol_weight * vol_unc + 
                          cons_weight * cons_unc;
        
        std::cout << "  " << s.ticker << ":\n";
        std::cout << "    Q = " << pi(i)*100 << "% + " << adjustment*100 
                  << "% = " << view.Q(i)*100 << "%\n";
        std::cout << "    Omega = " << view.Omega(i,i) << "\n";
    }
    std::cout << "\n";
    
    // Black-Litterman
    BlackLittermanModel bl_model(pi, sigma);
    auto posterior = bl_model.computePosterior(view);
    VectorXd mu_BL = posterior.first;
    MatrixXd Sigma_BL = posterior.second;
    
    // Optimize
    VectorXd w_optimal = BlackLittermanModel::optimizePortfolio(mu_BL, Sigma_BL);
    
    std::cout << "Results:\n";
    std::cout << std::setw(10) << "Ticker" << std::setw(12) << "Sentiment" 
              << std::setw(15) << "Post. Return" << std::setw(12) << "Weight\n";
    std::cout << std::string(49, '-') << "\n";
    
    for (int i = 0; i < N; ++i) {
        std::cout << std::setw(10) << tickers[i]
                  << std::setw(11) << std::fixed << std::setprecision(2)
                  << (sentiments[i].sentiment_mean >= 0 ? "+" : "") 
                  << sentiments[i].sentiment_mean
                  << std::setw(14) << mu_BL(i)*100 << "%"
                  << std::setw(11) << w_optimal(i)*100 << "%\n";
    }
    std::cout << "\n";
}

// ============================================================================
// EXAMPLE 4: Comparing Prior vs Posterior Portfolios
// ============================================================================

void example4_comparison() {
    printSeparator();
    std::cout << "EXAMPLE 4: Prior vs Posterior Portfolio Comparison\n";
    printSeparator();
    std::cout << "\n";
    
    const int N = 3;
    std::vector<std::string> tickers = {"AAPL", "MSFT", "GOOGL"};
    
    // Setup
    MatrixXd sigma(N, N);
    sigma << 0.08, 0.04, 0.05,
             0.04, 0.06, 0.03,
             0.05, 0.03, 0.07;
    
    VectorXd w_eq = VectorXd::Constant(N, 1.0/N);
    VectorXd pi = BlackLittermanModel::computeEquilibriumReturns(w_eq, sigma);
    
    // View: AAPL = 15%, high confidence
    BlackLittermanView view;
    view.P = MatrixXd(1, N);
    view.P << 1, 0, 0;
    view.Q = VectorXd(1);
    view.Q << 0.15;
    view.Omega = MatrixXd(1, 1);
    view.Omega << 0.005;  // High confidence (low Omega)
    
    // Compute
    BlackLittermanModel bl_model(pi, sigma);
    auto posterior = bl_model.computePosterior(view);
    VectorXd mu_BL = posterior.first;
    MatrixXd Sigma_BL = posterior.second;
    
    VectorXd w_prior = BlackLittermanModel::optimizePortfolio(pi, sigma);
    VectorXd w_posterior = BlackLittermanModel::optimizePortfolio(mu_BL, Sigma_BL);
    
    auto stats_prior = BlackLittermanModel::computePortfolioStats(w_prior, pi, sigma);
    auto stats_post = BlackLittermanModel::computePortfolioStats(w_posterior, mu_BL, Sigma_BL);
    
    double ret_prior = std::get<0>(stats_prior);
    double vol_prior = std::get<1>(stats_prior);
    double sharpe_prior = std::get<2>(stats_prior);
    
    double ret_post = std::get<0>(stats_post);
    double vol_post = std::get<1>(stats_post);
    double sharpe_post = std::get<2>(stats_post);
    
    std::cout << "Portfolio Comparison:\n\n";
    
    std::cout << std::setw(15) << "Metric" << std::setw(15) << "Prior" 
              << std::setw(15) << "Posterior" << std::setw(15) << "Change\n";
    std::cout << std::string(60, '-') << "\n";
    
    std::cout << std::setw(15) << "Return"
              << std::setw(14) << std::fixed << std::setprecision(2) 
              << ret_prior*100 << "%"
              << std::setw(14) << ret_post*100 << "%"
              << std::setw(14) << (ret_post-ret_prior)*100 << "%\n";
    
    std::cout << std::setw(15) << "Volatility"
              << std::setw(14) << vol_prior*100 << "%"
              << std::setw(14) << vol_post*100 << "%"
              << std::setw(14) << (vol_post-vol_prior)*100 << "%\n";
    
    std::cout << std::setw(15) << "Sharpe Ratio"
              << std::setw(14) << std::setprecision(3) << sharpe_prior
              << std::setw(14) << sharpe_post
              << std::setw(14) << sharpe_post-sharpe_prior << "\n\n";
    
    std::cout << "Weights:\n";
    for (int i = 0; i < N; ++i) {
        double change = w_posterior(i) - w_prior(i);
        std::cout << "  " << tickers[i] << ": " 
                  << std::setw(6) << std::fixed << std::setprecision(1)
                  << w_prior(i)*100 << "% -> " 
                  << std::setw(6) << w_posterior(i)*100 << "% ("
                  << (change >= 0 ? "+" : "") << change*100 << "%)\n";
    }
    std::cout << "\n";
}

// ============================================================================
// MAIN
// ============================================================================

int main() {
    try {
        example1_basic_single_view();
        example2_multiple_views();
        example3_sentiment_views();
        example4_comparison();
        
        std::cout << "All examples completed successfully!\n";
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
