"""
Black-Litterman Model Examples
Demonstrates various use cases and features
"""

import numpy as np
import matplotlib.pyplot as plt
from bl_model import BlackLittermanModel, create_view_matrix


def example_1_basic_usage():
    """
    Example 1: Basic usage with absolute and relative views
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Basic Black-Litterman Usage")
    print("=" * 80)
    
    # Simple 4-asset portfolio
    n_assets = 4
    asset_names = ['Stocks', 'Bonds', 'Gold', 'Real Estate']
    
    # Prior expected returns (could be from equilibrium or other model)
    pi = np.array([0.08, 0.04, 0.05, 0.06])
    
    # Covariance matrix (annualized)
    sigma = np.array([
        [0.040, 0.008, 0.002, 0.010],
        [0.008, 0.010, 0.001, 0.004],
        [0.002, 0.001, 0.015, 0.002],
        [0.010, 0.004, 0.002, 0.025]
    ])
    
    # Initialize model (using market formulation)
    bl = BlackLittermanModel(pi, sigma, use_market_formulation=True)
    
    # Express views:
    # View 1: Stocks will return 10%
    # View 2: Bonds will outperform Gold by 2%
    P = create_view_matrix(
        n_assets=n_assets,
        absolute_views={0: 1.0},  # Stocks
        relative_views=[(1, 2, 1.0, -1.0)]  # Bonds - Gold
    )
    Q = np.array([0.10, 0.02])
    
    print("\nPrior Expected Returns:")
    for i, name in enumerate(asset_names):
        print(f"  {name:15s}: {pi[i]*100:6.2f}%")
    
    print("\nViews:")
    print("  1. Stocks will return 10.0%")
    print("  2. Bonds will outperform Gold by 2.0%")
    
    # Compute posterior with different confidence levels
    for conf in [0.5, 1.0, 2.0, 5.0]:
        mu_bl, sigma_bl = bl.compute_posterior(P, Q, confidence=conf)
        print(f"\nPosterior Returns (confidence = {conf}):")
        for i, name in enumerate(asset_names):
            print(f"  {name:15s}: {mu_bl[i]*100:6.2f}%")


def example_2_qualitative_views():
    """
    Example 2: Using qualitative views (bullish/bearish)
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Qualitative Views")
    print("=" * 80)
    
    # 3-asset portfolio
    n_assets = 3
    asset_names = ['Tech', 'Energy', 'Finance']
    
    pi = np.array([0.12, 0.08, 0.10])
    sigma = np.array([
        [0.09, 0.02, 0.03],
        [0.02, 0.06, 0.02],
        [0.03, 0.02, 0.07]
    ])
    
    bl = BlackLittermanModel(pi, sigma, use_market_formulation=True)
    
    # Express qualitative views
    # View 1: Very bullish on Tech
    # View 2: Bearish on Energy
    # View 3: Tech will outperform Finance (bullish on spread)
    P = create_view_matrix(
        n_assets=n_assets,
        absolute_views={0: 1.0, 1: 1.0},  # Tech, Energy
        relative_views=[(0, 2, 1.0, -1.0)]  # Tech - Finance
    )
    
    view_types = ['very_bullish', 'bearish', 'bullish']
    Q = bl.set_qualitative_views(P, view_types, alpha=1.0, beta=2.0)
    
    print("\nPrior Expected Returns:")
    for i, name in enumerate(asset_names):
        print(f"  {name:15s}: {pi[i]*100:6.2f}%")
    
    print("\nQualitative Views:")
    print("  1. Very bullish on Tech")
    print("  2. Bearish on Energy")
    print("  3. Bullish: Tech will outperform Finance")
    
    print(f"\nQuantified Views (Q vector): {Q*100}%")
    
    mu_bl, sigma_bl = bl.compute_posterior(P, Q, confidence=1.5)
    
    print("\nPosterior Expected Returns:")
    for i, name in enumerate(asset_names):
        print(f"  {name:15s}: {mu_bl[i]*100:6.2f}%")


def example_3_scenario_analysis():
    """
    Example 3: Scenario analysis (full confidence limit)
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Scenario Analysis (Full Confidence)")
    print("=" * 80)
    
    # 3-asset portfolio
    asset_names = ['Asset A', 'Asset B', 'Asset C']
    
    pi = np.array([0.07, 0.05, 0.06])
    sigma = np.array([
        [0.04, 0.01, 0.01],
        [0.01, 0.03, 0.01],
        [0.01, 0.01, 0.05]
    ])
    
    bl = BlackLittermanModel(pi, sigma, use_market_formulation=True)
    
    # Scenario: Asset A returns 10% and Asset B returns 4%
    P = create_view_matrix(
        n_assets=3,
        absolute_views={0: 1.0, 1: 1.0}
    )
    Q = np.array([0.10, 0.04])
    
    print("\nPrior Expected Returns:")
    for i, name in enumerate(asset_names):
        print(f"  {name:15s}: {pi[i]*100:6.2f}%")
    
    print("\nScenario (Full Confidence):")
    print("  Asset A: 10.0%")
    print("  Asset B:  4.0%")
    
    # Very high confidence approximates scenario analysis
    mu_bl, sigma_bl = bl.compute_posterior(P, Q, confidence=1000.0)
    
    print("\nPosterior Expected Returns:")
    for i, name in enumerate(asset_names):
        print(f"  {name:15s}: {mu_bl[i]*100:6.2f}%")
    
    print("\nNote: With very high confidence, Asset A ≈ 10% and Asset B ≈ 4%")
    print("      Asset C is adjusted through correlation structure")


def example_4_confidence_sensitivity():
    """
    Example 4: Sensitivity to confidence levels
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Confidence Level Sensitivity")
    print("=" * 80)
    
    # Simple 2-asset case for clarity
    asset_names = ['Asset 1', 'Asset 2']
    
    pi = np.array([0.08, 0.06])
    sigma = np.array([
        [0.04, 0.01],
        [0.01, 0.03]
    ])
    
    bl = BlackLittermanModel(pi, sigma, use_market_formulation=True)
    
    # View: Asset 1 will return 12%
    P = np.array([[1, 0]])
    Q = np.array([0.12])
    
    print("\nPrior: Asset 1 = 8%, Asset 2 = 6%")
    print("View:  Asset 1 = 12%")
    print("\nPosterior Asset 1 returns at different confidence levels:")
    
    confidences = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0]
    results = []
    
    for conf in confidences:
        mu_bl, _ = bl.compute_posterior(P, Q, confidence=conf)
        results.append(mu_bl[0])
        print(f"  Confidence = {conf:6.1f}: {mu_bl[0]*100:6.3f}%")
    
    print("\nObservation:")
    print("  - Low confidence  → stays close to prior (8%)")
    print("  - High confidence → approaches view (12%)")


def example_5_formulation_comparison():
    """
    Example 5: Compare original vs market formulation
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Original vs Market Formulation")
    print("=" * 80)
    
    # 3-asset portfolio
    pi = np.array([0.08, 0.06, 0.07])
    sigma = np.array([
        [0.04, 0.01, 0.01],
        [0.01, 0.03, 0.01],
        [0.01, 0.01, 0.05]
    ])
    
    # View: Asset 1 will return 10%
    P = np.array([[1, 0, 0]])
    Q = np.array([0.10])
    
    print("\nView: Asset 1 will return 10%")
    print(f"Prior: {pi*100}%\n")
    
    # Original formulation
    bl_orig = BlackLittermanModel(pi, sigma, tau=0.025, use_market_formulation=False)
    mu_orig, sigma_orig = bl_orig.compute_posterior(P, Q, confidence=1.0)
    
    # Market formulation
    bl_mkt = BlackLittermanModel(pi, sigma, use_market_formulation=True)
    mu_mkt, sigma_mkt = bl_mkt.compute_posterior(P, Q, confidence=1.0)
    
    print("Original Formulation (with tau=0.025):")
    print(f"  Expected Returns: {mu_orig*100}%")
    print(f"  Posterior Variance (Asset 1): {sigma_orig[0,0]*100:.4f}%²")
    
    print("\nMarket Formulation (no tau needed):")
    print(f"  Expected Returns: {mu_mkt*100}%")
    print(f"  Posterior Variance (Asset 1): {sigma_mkt[0,0]*100:.4f}%²")
    
    print("\nDifference:")
    print(f"  Expected Returns: {(mu_orig - mu_mkt)*100}%")
    print(f"  Variance: {(sigma_orig[0,0] - sigma_mkt[0,0])*100:.4f}%²")
    
    print("\nNote: Market formulation is cleaner (no tau parameter in posterior)")
    print("      and has better limiting behavior for scenario analysis")


def example_6_equilibrium_returns():
    """
    Example 6: Computing equilibrium returns from market weights
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Equilibrium Returns from Market Capitalization")
    print("=" * 80)
    
    # Example: Global equity markets
    asset_names = ['US', 'Europe', 'Japan', 'Emerging']
    
    # Market cap weights (example)
    w_eq = np.array([0.55, 0.25, 0.10, 0.10])
    
    # Historical covariance (example, annualized)
    sigma = np.array([
        [0.040, 0.025, 0.020, 0.030],
        [0.025, 0.035, 0.018, 0.028],
        [0.020, 0.018, 0.045, 0.025],
        [0.030, 0.028, 0.025, 0.060]
    ])
    
    print("\nMarket Capitalization Weights:")
    for i, name in enumerate(asset_names):
        print(f"  {name:15s}: {w_eq[i]*100:5.1f}%")
    
    # Compute equilibrium returns for different risk aversion levels
    print("\nImplied Equilibrium Returns:")
    
    for risk_aversion in [1.5, 2.5, 3.5]:
        pi = BlackLittermanModel.compute_equilibrium_returns(
            w_eq, sigma, risk_aversion
        )
        print(f"\n  Risk Aversion λ = {risk_aversion}:")
        for i, name in enumerate(asset_names):
            print(f"    {name:15s}: {pi[i]*100:6.2f}%")
    
    print("\nNote: Higher risk aversion → higher implied returns")
    print("      (investors require more return to hold risky assets)")


def example_7_relative_confidence():
    """
    Example 7: Using relative confidence levels for different views
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 7: Relative Confidence Levels")
    print("=" * 80)
    
    # 3-asset portfolio
    asset_names = ['Stock A', 'Stock B', 'Stock C']
    
    pi = np.array([0.08, 0.07, 0.06])
    sigma = np.array([
        [0.04, 0.01, 0.01],
        [0.01, 0.03, 0.01],
        [0.01, 0.01, 0.05]
    ])
    
    bl = BlackLittermanModel(pi, sigma, use_market_formulation=True)
    
    # Two views:
    # View 1: Stock A will return 12% (HIGH confidence)
    # View 2: Stock B will return 9%  (LOW confidence)
    P = create_view_matrix(
        n_assets=3,
        absolute_views={0: 1.0, 1: 1.0}
    )
    Q = np.array([0.12, 0.09])
    
    print("\nPrior Expected Returns:")
    for i, name in enumerate(asset_names):
        print(f"  {name:15s}: {pi[i]*100:6.2f}%")
    
    print("\nViews:")
    print("  1. Stock A = 12% (HIGH confidence)")
    print("  2. Stock B =  9% (LOW confidence)")
    
    # Equal confidence
    print("\n--- Equal Confidence ---")
    mu_equal, _ = bl.compute_posterior(P, Q, confidence=2.0)
    for i, name in enumerate(asset_names):
        print(f"  {name:15s}: {mu_equal[i]*100:6.2f}%")
    
    # Relative confidence: View 1 is 3x more confident than View 2
    print("\n--- Relative Confidence (3:1) ---")
    relative_conf = np.array([1.0, 3.0])  # Higher value = LESS confident
    mu_relative, _ = bl.compute_posterior(
        P, Q, confidence=2.0, relative_confidence=relative_conf
    )
    for i, name in enumerate(asset_names):
        print(f"  {name:15s}: {mu_relative[i]*100:6.2f}%")
    
    print("\nObservation:")
    print("  Stock A moves more toward 12% (higher confidence)")
    print("  Stock B moves less toward 9% (lower confidence)")


if __name__ == "__main__":
    # Run all examples
    example_1_basic_usage()
    example_2_qualitative_views()
    example_3_scenario_analysis()
    example_4_confidence_sensitivity()
    example_5_formulation_comparison()
    example_6_equilibrium_returns()
    example_7_relative_confidence()
    
    print("\n" + "=" * 80)
    print("All examples completed successfully!")
    print("=" * 80 + "\n")
