"""
Unit Tests for Black-Litterman Model
Validates implementation against known results and edge cases
"""

import numpy as np
import unittest
from bl_model import BlackLittermanModel, create_view_matrix


class TestBlackLittermanModel(unittest.TestCase):
    """Test suite for Black-Litterman implementation"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Simple 3-asset test case
        self.pi = np.array([0.08, 0.06, 0.07])
        self.sigma = np.array([
            [0.04, 0.01, 0.01],
            [0.01, 0.03, 0.01],
            [0.01, 0.01, 0.05]
        ])
        self.n_assets = 3
    
    def test_initialization(self):
        """Test model initialization"""
        bl = BlackLittermanModel(self.pi, self.sigma)
        
        np.testing.assert_array_equal(bl.pi, self.pi)
        np.testing.assert_array_equal(bl.sigma, self.sigma)
        self.assertTrue(bl.use_market_formulation)
        
    def test_initialization_dimension_mismatch(self):
        """Test that dimension mismatch raises error"""
        wrong_sigma = np.array([[1, 2], [2, 3]])
        
        with self.assertRaises(ValueError):
            BlackLittermanModel(self.pi, wrong_sigma)
    
    def test_equilibrium_returns(self):
        """Test equilibrium return computation"""
        w_eq = np.array([0.5, 0.3, 0.2])
        risk_aversion = 2.5
        
        pi = BlackLittermanModel.compute_equilibrium_returns(
            w_eq, self.sigma, risk_aversion
        )
        
        # Should be: 2 * lambda * Sigma * w_eq
        expected = 2 * risk_aversion * self.sigma @ w_eq
        np.testing.assert_array_almost_equal(pi, expected)
    
    def test_no_views_returns_prior(self):
        """Test that with no confidence, posterior equals prior"""
        bl = BlackLittermanModel(self.pi, self.sigma, use_market_formulation=True)
        
        # Single view with very low confidence
        P = np.array([[1, 0, 0]])
        Q = np.array([0.10])
        
        mu_bl, sigma_bl = bl.compute_posterior(P, Q, confidence=0.0001)
        
        # Should be very close to prior
        np.testing.assert_array_almost_equal(mu_bl, self.pi, decimal=2)
        np.testing.assert_array_almost_equal(sigma_bl, self.sigma, decimal=2)
    
    def test_full_confidence_scenario(self):
        """Test that full confidence approaches deterministic scenario"""
        bl = BlackLittermanModel(self.pi, self.sigma, use_market_formulation=True)
        
        # View: Asset 0 = 10%
        P = np.array([[1, 0, 0]])
        Q = np.array([0.10])
        
        mu_bl, sigma_bl = bl.compute_posterior(P, Q, confidence=10000.0)
        
        # Asset 0 should be very close to 10%
        self.assertAlmostEqual(mu_bl[0], 0.10, places=3)
        
        # Variance of asset 0 should be reduced significantly
        self.assertLess(sigma_bl[0, 0], self.sigma[0, 0])
    
    def test_view_matrix_creation_absolute(self):
        """Test view matrix creation for absolute views"""
        P = create_view_matrix(
            n_assets=3,
            absolute_views={0: 1.0, 2: 1.0}
        )
        
        expected = np.array([
            [1, 0, 0],
            [0, 0, 1]
        ])
        
        np.testing.assert_array_equal(P, expected)
    
    def test_view_matrix_creation_relative(self):
        """Test view matrix creation for relative views"""
        P = create_view_matrix(
            n_assets=3,
            relative_views=[(0, 1, 1.0, -1.0)]
        )
        
        expected = np.array([[1, -1, 0]])
        
        np.testing.assert_array_equal(P, expected)
    
    def test_view_matrix_creation_mixed(self):
        """Test view matrix creation for mixed views"""
        P = create_view_matrix(
            n_assets=4,
            absolute_views={1: 1.0},
            relative_views=[(0, 2, 1.0, -1.0)]
        )
        
        expected = np.array([
            [0, 1, 0, 0],
            [1, 0, -1, 0]
        ])
        
        np.testing.assert_array_equal(P, expected)
    
    def test_qualitative_views(self):
        """Test qualitative view conversion"""
        bl = BlackLittermanModel(self.pi, self.sigma)
        
        P = np.array([[1, 0, 0], [0, 1, 0]])
        view_types = ['bullish', 'bearish']
        
        Q = bl.set_qualitative_views(P, view_types, alpha=1.0, beta=2.0)
        
        # Bullish should be above prior, bearish below
        self.assertGreater(Q[0], self.pi[0])
        self.assertLess(Q[1], self.pi[1])
    
    def test_omega_computation_simple(self):
        """Test Omega matrix computation"""
        bl = BlackLittermanModel(self.pi, self.sigma)
        
        P = np.array([[1, 0, 0]])
        confidence = 2.0
        
        Omega = bl._compute_omega(P, confidence)
        
        # Should be (1/c) * P * Sigma * P'
        expected = (1 / confidence) * P @ self.sigma @ P.T
        
        np.testing.assert_array_almost_equal(Omega, expected)
    
    def test_omega_computation_relative_confidence(self):
        """Test Omega with relative confidence"""
        bl = BlackLittermanModel(self.pi, self.sigma)
        
        P = np.array([[1, 0, 0], [0, 1, 0]])
        confidence = 2.0
        relative_conf = np.array([1.0, 2.0])
        
        Omega = bl._compute_omega(P, confidence, relative_conf)
        
        # Should have different diagonal elements
        self.assertNotAlmostEqual(Omega[0, 0], Omega[1, 1])
    
    def test_posterior_symmetry(self):
        """Test that posterior covariance is symmetric"""
        bl = BlackLittermanModel(self.pi, self.sigma)
        
        P = np.array([[1, -1, 0]])
        Q = np.array([0.02])
        
        mu_bl, sigma_bl = bl.compute_posterior(P, Q, confidence=1.0)
        
        # Check symmetry
        np.testing.assert_array_almost_equal(sigma_bl, sigma_bl.T)
    
    def test_posterior_psd(self):
        """Test that posterior covariance is positive semi-definite"""
        bl = BlackLittermanModel(self.pi, self.sigma)
        
        P = np.array([[1, 0, 0]])
        Q = np.array([0.09])
        
        mu_bl, sigma_bl = bl.compute_posterior(P, Q, confidence=2.0)
        
        # Check positive semi-definiteness
        eigenvalues = np.linalg.eigvalsh(sigma_bl)
        self.assertTrue(np.all(eigenvalues >= -1e-10))
    
    def test_market_vs_original_formulation(self):
        """Test consistency between formulations at moderate confidence"""
        P = np.array([[1, 0, 0]])
        Q = np.array([0.09])
        confidence = 1.0
        
        # Market formulation
        bl_market = BlackLittermanModel(
            self.pi, self.sigma, use_market_formulation=True
        )
        mu_market, sigma_market = bl_market.compute_posterior(P, Q, confidence)
        
        # Original formulation
        bl_orig = BlackLittermanModel(
            self.pi, self.sigma, tau=0.025, use_market_formulation=False
        )
        mu_orig, sigma_orig = bl_orig.compute_posterior(P, Q, confidence)
        
        # They should be different but both valid
        self.assertFalse(np.allclose(mu_market, mu_orig))
        
        # But both should be between prior and view
        self.assertTrue(self.pi[0] < mu_market[0] < Q[0])
        self.assertTrue(self.pi[0] < mu_orig[0] < Q[0])
    
    def test_validation_symmetric(self):
        """Test validation detects non-symmetric matrices"""
        bl = BlackLittermanModel(self.pi, self.sigma)
        
        # Create slightly asymmetric matrix
        sigma_asymm = self.sigma.copy()
        sigma_asymm[0, 1] += 0.001
        
        validation = bl.validate_posterior(sigma_asymm)
        self.assertFalse(validation['is_symmetric'])
        self.assertTrue(len(validation['warnings']) > 0)
    
    def test_validation_psd(self):
        """Test validation detects non-PSD matrices"""
        bl = BlackLittermanModel(self.pi, self.sigma)
        
        # Create non-PSD matrix
        sigma_neg = np.array([
            [1, 2, 0],
            [2, 1, 0],
            [0, 0, 1]
        ])
        
        validation = bl.validate_posterior(sigma_neg, check_psd=True)
        self.assertFalse(validation['is_psd'])
    
    def test_confidence_monotonicity(self):
        """Test that higher confidence moves posterior closer to view"""
        bl = BlackLittermanModel(self.pi, self.sigma)
        
        P = np.array([[1, 0, 0]])
        Q = np.array([0.12])
        
        # Low confidence
        mu_low, _ = bl.compute_posterior(P, Q, confidence=0.5)
        
        # High confidence
        mu_high, _ = bl.compute_posterior(P, Q, confidence=5.0)
        
        # High confidence should be closer to view
        dist_low = abs(mu_low[0] - Q[0])
        dist_high = abs(mu_high[0] - Q[0])
        
        self.assertLess(dist_high, dist_low)
    
    def test_multiple_views(self):
        """Test model with multiple views"""
        bl = BlackLittermanModel(self.pi, self.sigma)
        
        P = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [1, -1, 0]
        ])
        Q = np.array([0.10, 0.08, 0.02])
        
        mu_bl, sigma_bl = bl.compute_posterior(P, Q, confidence=1.5)
        
        # Should not raise errors
        self.assertEqual(len(mu_bl), self.n_assets)
        self.assertEqual(sigma_bl.shape, (self.n_assets, self.n_assets))
    
    def test_zero_confidence_error_handling(self):
        """Test handling of extreme parameters"""
        bl = BlackLittermanModel(self.pi, self.sigma)
        
        P = np.array([[1, 0, 0]])
        Q = np.array([0.10])
        
        # Very low confidence should still work
        mu_bl, sigma_bl = bl.compute_posterior(P, Q, confidence=1e-6)
        
        # Should be close to prior
        np.testing.assert_array_almost_equal(mu_bl, self.pi, decimal=2)
    
    def test_view_dimension_mismatch(self):
        """Test error handling for dimension mismatches"""
        bl = BlackLittermanModel(self.pi, self.sigma)
        
        P = np.array([[1, 0, 0], [0, 1, 0]])
        Q = np.array([0.10])  # Wrong size!
        
        with self.assertRaises(ValueError):
            bl.compute_posterior(P, Q, confidence=1.0)
    
    def test_invalid_view_type(self):
        """Test error handling for invalid qualitative view types"""
        bl = BlackLittermanModel(self.pi, self.sigma)
        
        P = np.array([[1, 0, 0]])
        view_types = ['invalid_type']
        
        with self.assertRaises(ValueError):
            bl.set_qualitative_views(P, view_types)
    
    def test_reproducibility(self):
        """Test that results are reproducible"""
        bl = BlackLittermanModel(self.pi, self.sigma)
        
        P = np.array([[1, -1, 0]])
        Q = np.array([0.03])
        
        # Run twice
        mu1, sigma1 = bl.compute_posterior(P, Q, confidence=1.0)
        mu2, sigma2 = bl.compute_posterior(P, Q, confidence=1.0)
        
        np.testing.assert_array_equal(mu1, mu2)
        np.testing.assert_array_equal(sigma1, sigma2)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions"""
    
    def test_single_asset(self):
        """Test with single asset"""
        pi = np.array([0.08])
        sigma = np.array([[0.04]])
        
        bl = BlackLittermanModel(pi, sigma)
        
        P = np.array([[1]])
        Q = np.array([0.10])
        
        mu_bl, sigma_bl = bl.compute_posterior(P, Q, confidence=1.0)
        
        self.assertEqual(len(mu_bl), 1)
        self.assertEqual(sigma_bl.shape, (1, 1))
    
    def test_many_assets(self):
        """Test with many assets"""
        n = 20
        pi = np.random.rand(n) * 0.1
        sigma = np.eye(n) * 0.04 + 0.01
        
        bl = BlackLittermanModel(pi, sigma)
        
        # View on first asset
        P = np.zeros((1, n))
        P[0, 0] = 1
        Q = np.array([0.12])
        
        mu_bl, sigma_bl = bl.compute_posterior(P, Q, confidence=1.0)
        
        self.assertEqual(len(mu_bl), n)
        self.assertEqual(sigma_bl.shape, (n, n))
    
    def test_highly_correlated_assets(self):
        """Test with highly correlated assets"""
        pi = np.array([0.08, 0.08])
        sigma = np.array([
            [0.04, 0.039],  # Correlation ≈ 0.975
            [0.039, 0.04]
        ])
        
        bl = BlackLittermanModel(pi, sigma)
        
        P = np.array([[1, 0]])
        Q = np.array([0.10])
        
        mu_bl, sigma_bl = bl.compute_posterior(P, Q, confidence=1.0)
        
        # High correlation means view on asset 0 should affect asset 1
        self.assertNotAlmostEqual(mu_bl[1], pi[1], places=3)
    
    def test_uncorrelated_assets(self):
        """Test with uncorrelated assets"""
        pi = np.array([0.08, 0.06, 0.07])
        sigma = np.diag([0.04, 0.03, 0.05])  # Diagonal = uncorrelated
        
        bl = BlackLittermanModel(pi, sigma)
        
        P = np.array([[1, 0, 0]])
        Q = np.array([0.10])
        
        mu_bl, sigma_bl = bl.compute_posterior(P, Q, confidence=1.0)
        
        # View on asset 0 shouldn't affect others much
        self.assertAlmostEqual(mu_bl[1], pi[1], places=2)
        self.assertAlmostEqual(mu_bl[2], pi[2], places=2)


def run_tests():
    """Run all tests and print results"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestBlackLittermanModel))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed")
    
    return result


if __name__ == "__main__":
    run_tests()
