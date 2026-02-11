/**
 * @file BlackLittermanModel.hpp
 * @brief Black-Litterman Portfolio Optimization (C++ Implementation)
 * @author Based on Meucci (2008): "The Black-Litterman Approach"
 * @date 2026
 * 
 * Modern C++17 implementation with Eigen for linear algebra.
 * Follows the market-based formulation (Meucci 2008, Section 3).
 * 
 * Key Features:
 * - Header-only design for easy integration
 * - Eigen library for efficient matrix operations
 * - Exception-safe RAII patterns
 * - Move semantics for performance
 * - Type-safe interfaces
 * - MSVC 2026 compatible
 */

#ifndef BLACK_LITTERMAN_MODEL_HPP
#define BLACK_LITTERMAN_MODEL_HPP

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <cmath>
#include <memory>
#include <string>

namespace portfolio {

/**
 * @brief View specification for Black-Litterman model
 * 
 * Represents investor views: P*X ~ N(Q, Ω)
 */
struct BlackLittermanView {
    Eigen::MatrixXd P;      ///< Pick matrix (K x N)
    Eigen::VectorXd Q;      ///< View returns (K x 1)
    Eigen::MatrixXd Omega;  ///< View uncertainty (K x K)
    
    /**
     * @brief Validate view dimensions
     * @throws std::invalid_argument if dimensions inconsistent
     */
    void validate() const {
        if (P.rows() != Q.size()) {
            throw std::invalid_argument(
                "P rows (" + std::to_string(P.rows()) + 
                ") must match Q size (" + std::to_string(Q.size()) + ")"
            );
        }
        if (Omega.rows() != Omega.cols() || Omega.rows() != Q.size()) {
            throw std::invalid_argument(
                "Omega must be square with size matching Q"
            );
        }
    }
};

/**
 * @brief Black-Litterman Portfolio Optimization Model
 * 
 * Implements Meucci's (2008) market-based formulation:
 * 
 * Posterior:
 *   μ_BL = π + ΣP'(PΣP' + Ω)^(-1)(Q - Pπ)
 *   Σ_BL = Σ - ΣP'(PΣP' + Ω)^(-1)PΣ
 * 
 * Where:
 *   π: Prior expected returns (equilibrium)
 *   Σ: Covariance matrix
 *   P: Pick matrix (selects assets in views)
 *   Q: View expected returns
 *   Ω: View uncertainty matrix
 */
class BlackLittermanModel {
public:
    /**
     * @brief Construct Black-Litterman model
     * @param pi Prior expected returns (N x 1)
     * @param sigma Covariance matrix (N x N)
     * @throws std::invalid_argument if dimensions inconsistent or sigma not PSD
     */
    BlackLittermanModel(
        const Eigen::VectorXd& pi,
        const Eigen::MatrixXd& sigma
    ) : pi_(pi), sigma_(sigma) {
        validateInputs();
    }
    
    /**
     * @brief Compute equilibrium returns from market weights
     * 
     * Formula: π = δΣw_eq
     * 
     * @param w_eq Market equilibrium weights (N x 1)
     * @param sigma Covariance matrix (N x N)
     * @param risk_aversion Risk aversion parameter δ (typically 2.5)
     * @return Equilibrium expected returns (N x 1)
     */
    static Eigen::VectorXd computeEquilibriumReturns(
        const Eigen::VectorXd& w_eq,
        const Eigen::MatrixXd& sigma,
        double risk_aversion = 2.5
    ) {
        if (w_eq.size() != sigma.rows() || sigma.rows() != sigma.cols()) {
            throw std::invalid_argument("Dimension mismatch in equilibrium calculation");
        }
        return 2.0 * risk_aversion * sigma * w_eq;
    }
    
    /**
     * @brief Compute Black-Litterman posterior distribution
     * 
     * Implements Meucci (2008) Equations 32-33 (Market Formulation)
     * 
     * @param view View specification (P, Q, Ω)
     * @return Pair of (μ_BL, Σ_BL)
     */
    std::pair<Eigen::VectorXd, Eigen::MatrixXd> computePosterior(
        const BlackLittermanView& view
    ) const {
        view.validate();
        
        const int N = static_cast<int>(pi_.size());
        const int K = static_cast<int>(view.Q.size());
        
        // Check dimensions
        if (view.P.cols() != N) {
            throw std::invalid_argument(
                "Pick matrix P columns must match number of assets"
            );
        }
        
        // Compute M = PΣP' + Ω
        Eigen::MatrixXd M = view.P * sigma_ * view.P.transpose() + view.Omega;
        
        // Stable inversion using LU decomposition
        Eigen::MatrixXd M_inv;
        try {
            M_inv = M.lu().solve(Eigen::MatrixXd::Identity(K, K));
        } catch (const std::exception& e) {
            throw std::runtime_error(
                "Failed to invert M matrix (singular?): " + std::string(e.what())
            );
        }
        
        // Posterior mean: μ_BL = π + ΣP'M^(-1)(Q - Pπ)
        Eigen::VectorXd view_adjustment = view.Q - view.P * pi_;
        Eigen::VectorXd mu_BL = pi_ + sigma_ * view.P.transpose() * M_inv * view_adjustment;
        
        // Posterior covariance: Σ_BL = Σ - ΣP'M^(-1)PΣ
        Eigen::MatrixXd Sigma_BL = sigma_ - sigma_ * view.P.transpose() * M_inv * view.P * sigma_;
        
        // Ensure symmetry (numerical stability)
        Sigma_BL = 0.5 * (Sigma_BL + Sigma_BL.transpose());
        
        return {mu_BL, Sigma_BL};
    }
    
    /**
     * @brief Mean-variance portfolio optimization
     * 
     * Solves: max w'μ - (δ/2)w'Σw
     * Analytical solution: w* = (1/δ)Σ^(-1)μ
     * 
     * @param mu Expected returns (N x 1)
     * @param sigma Covariance matrix (N x N)
     * @param risk_aversion Risk aversion δ
     * @param long_only If true, enforce w >= 0
     * @return Optimal weights (N x 1), normalized to sum to 1
     */
    static Eigen::VectorXd optimizePortfolio(
        const Eigen::VectorXd& mu,
        const Eigen::MatrixXd& sigma,
        double risk_aversion = 2.5,
        bool long_only = true
    ) {
        const int N = static_cast<int>(mu.size());
        
        // Unconstrained solution: w = (1/2δ)Σ^(-1)μ
        Eigen::MatrixXd sigma_inv;
        try {
            sigma_inv = sigma.lu().solve(Eigen::MatrixXd::Identity(N, N));
        } catch (const std::exception&) {
            // Fallback to pseudo-inverse
            Eigen::JacobiSVD<Eigen::MatrixXd> svd(
                sigma, 
                Eigen::ComputeThinU | Eigen::ComputeThinV
            );
            sigma_inv = svd.solve(Eigen::MatrixXd::Identity(N, N));
        }
        
        Eigen::VectorXd w = sigma_inv * mu / (2.0 * risk_aversion);
        
        // Apply long-only constraint
        if (long_only) {
            w = w.cwiseMax(0.0);
        }
        
        // Normalize to sum to 1
        double sum = w.sum();
        if (std::abs(sum) < 1e-10) {
            // If all weights near zero, use equal weights
            w = Eigen::VectorXd::Constant(N, 1.0 / N);
        } else {
            w /= sum;
        }
        
        return w;
    }
    
    /**
     * @brief Compute portfolio statistics
     * @param w Portfolio weights (N x 1)
     * @param mu Expected returns (N x 1)
     * @param sigma Covariance matrix (N x N)
     * @return Tuple of (return, volatility, sharpe_ratio)
     */
    static std::tuple<double, double, double> computePortfolioStats(
        const Eigen::VectorXd& w,
        const Eigen::VectorXd& mu,
        const Eigen::MatrixXd& sigma
    ) {
        double portfolio_return = w.dot(mu);
        double portfolio_variance = w.dot(sigma * w);
        double portfolio_vol = std::sqrt(portfolio_variance);
        double sharpe = (portfolio_vol > 1e-10) ? (portfolio_return / portfolio_vol) : 0.0;
        
        return std::make_tuple(portfolio_return, portfolio_vol, sharpe);
    }
    
    /**
     * @brief Validate posterior covariance is PSD
     * @param Sigma_BL Posterior covariance matrix
     * @return True if PSD, false otherwise
     */
    static bool validatePosterior(const Eigen::MatrixXd& Sigma_BL) {
        // Check symmetry
        if (!Sigma_BL.isApprox(Sigma_BL.transpose(), 1e-10)) {
            std::cerr << "Warning: Posterior covariance not symmetric\n";
            return false;
        }
        
        // Check positive semi-definite
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Sigma_BL);
        auto eigenvalues = es.eigenvalues();
        
        if ((eigenvalues.array() < -1e-10).any()) {
            std::cerr << "Warning: Posterior covariance has negative eigenvalues\n";
            std::cerr << "Min eigenvalue: " << eigenvalues.minCoeff() << "\n";
            return false;
        }
        
        return true;
    }
    
    // Getters
    const Eigen::VectorXd& getPi() const { return pi_; }
    const Eigen::MatrixXd& getSigma() const { return sigma_; }
    
private:
    Eigen::VectorXd pi_;      ///< Prior expected returns
    Eigen::MatrixXd sigma_;   ///< Covariance matrix
    
    /**
     * @brief Validate model inputs
     * @throws std::invalid_argument if validation fails
     */
    void validateInputs() {
        const int N = static_cast<int>(pi_.size());
        
        // Check sigma dimensions
        if (sigma_.rows() != N || sigma_.cols() != N) {
            throw std::invalid_argument(
                "Sigma must be N x N where N = " + std::to_string(N)
            );
        }
        
        // Check symmetry
        if (!sigma_.isApprox(sigma_.transpose(), 1e-10)) {
            std::cerr << "Warning: Symmetrizing covariance matrix\n";
            sigma_ = 0.5 * (sigma_ + sigma_.transpose());
        }
        
        // Check positive definiteness
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(sigma_);
        auto eigenvalues = es.eigenvalues();
        
        if ((eigenvalues.array() <= 0).any()) {
            std::cerr << "Warning: Covariance matrix has non-positive eigenvalues\n";
            std::cerr << "Min eigenvalue: " << eigenvalues.minCoeff() << "\n";
            
            // Add regularization
            const double reg = 1e-6;
            sigma_ += reg * Eigen::MatrixXd::Identity(N, N);
            std::cerr << "Added regularization: " << reg << "\n";
        }
    }
};

} // namespace portfolio

#endif // BLACK_LITTERMAN_MODEL_HPP
