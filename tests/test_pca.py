"""
Unit Tests for Principal Component Analysis (PCA) Implementation

This module contains comprehensive tests for the PCA class,
including tests for:
- Model initialization
- Fitting and transformation
- Principal components extraction
- Explained variance calculation
- Dimensionality reduction
- Inverse transformation (reconstruction)
- Whitening
- Covariance matrix computation
- Comparison with scikit-learn
"""

import numpy as np
import pytest
from sklearn.datasets import load_iris, make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as SklearnPCA

# Import the PCA model to test
from algorithms.unsupervised.dimensionality_reduction import PCA


class TestPCAInitialization:
    """Test PCA initialization."""
    
    def test_default_initialization(self):
        """Test default initialization."""
        pca = PCA()
        
        assert pca.n_components is None
        assert pca.whiten is False
        assert pca.components is None
        assert pca.explained_variance is None
        assert pca.explained_variance_ratio is None
        assert pca.mean is None
        assert pca.n_features is None
    
    def test_custom_initialization(self):
        """Test custom initialization."""
        pca = PCA(n_components=3, whiten=True)
        
        assert pca.n_components == 3
        assert pca.whiten is True
    
    def test_different_n_components(self):
        """Test initialization with different n_components."""
        for n_comp in [1, 2, 5, 10]:
            pca = PCA(n_components=n_comp)
            assert pca.n_components == n_comp


class TestPCAFitting:
    """Test PCA fitting."""
    
    @pytest.fixture
    def simple_data(self):
        """Generate simple 2D data."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        return X
    
    @pytest.fixture
    def iris_data(self):
        """Load iris dataset."""
        iris = load_iris()
        X = iris.data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled
    
    def test_basic_fitting(self, simple_data):
        """Test basic fitting."""
        X = simple_data
        pca = PCA(n_components=2)
        
        result = pca.fit(X)
        
        assert result is pca
        assert pca.components is not None
        assert pca.explained_variance is not None
        assert pca.explained_variance_ratio is not None
        assert pca.mean is not None
        assert pca.n_features == 5
    
    def test_components_shape(self, simple_data):
        """Test that components have correct shape."""
        X = simple_data
        pca = PCA(n_components=3)
        pca.fit(X)
        
        # Components should be (n_components, n_features)
        assert pca.components.shape == (3, 5)
    
    def test_explained_variance_shape(self, simple_data):
        """Test explained variance shape."""
        X = simple_data
        pca = PCA(n_components=3)
        pca.fit(X)
        
        assert pca.explained_variance.shape == (3,)
        assert pca.explained_variance_ratio.shape == (3,)
    
    def test_explained_variance_ratio_sums_to_less_than_one(self, simple_data):
        """Test that explained variance ratio sums to <= 1."""
        X = simple_data
        pca = PCA(n_components=3)
        pca.fit(X)
        
        total_variance = np.sum(pca.explained_variance_ratio)
        
        assert 0 < total_variance <= 1.0
    
    def test_explained_variance_descending(self, simple_data):
        """Test that explained variance is in descending order."""
        X = simple_data
        pca = PCA(n_components=4)
        pca.fit(X)
        
        # Explained variance should be sorted descending
        assert all(pca.explained_variance[i] >= pca.explained_variance[i+1] 
                  for i in range(len(pca.explained_variance)-1))
    
    def test_mean_calculation(self, simple_data):
        """Test that mean is calculated correctly."""
        X = simple_data
        pca = PCA(n_components=2)
        pca.fit(X)
        
        expected_mean = np.mean(X, axis=0)
        
        np.testing.assert_array_almost_equal(pca.mean, expected_mean)
    
    def test_fit_with_none_components(self, simple_data):
        """Test fitting with n_components=None."""
        X = simple_data
        pca = PCA(n_components=None)
        pca.fit(X)
        
        # Should use min(n_samples, n_features)
        expected_n_comp = min(X.shape)
        assert pca.n_components == expected_n_comp


class TestPCATransformation:
    """Test PCA transformation."""
    
    @pytest.fixture
    def fitted_pca(self):
        """Create fitted PCA model."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        pca = PCA(n_components=2)
        pca.fit(X)
        return pca, X
    
    def test_transform_shape(self, fitted_pca):
        """Test transform output shape."""
        pca, X = fitted_pca
        
        X_transformed = pca.transform(X)
        
        assert X_transformed.shape == (100, 2)
    
    def test_transform_single_sample(self, fitted_pca):
        """Test transform on single sample."""
        pca, _ = fitted_pca
        
        X_single = np.random.randn(1, 5)
        X_transformed = pca.transform(X_single)
        
        assert X_transformed.shape == (1, 2)
    
    def test_fit_transform(self):
        """Test fit_transform method."""
        np.random.seed(42)
        X = np.random.randn(50, 4)
        
        pca = PCA(n_components=2)
        X_transformed = pca.fit_transform(X)
        
        assert X_transformed.shape == (50, 2)
        assert pca.components is not None
    
    def test_transform_reduces_dimensionality(self):
        """Test that transform reduces dimensionality."""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        
        pca = PCA(n_components=3)
        X_transformed = pca.fit_transform(X)
        
        # Original: 10 dimensions, transformed: 3 dimensions
        assert X.shape[1] == 10
        assert X_transformed.shape[1] == 3


class TestPCAInverseTransform:
    """Test PCA inverse transformation."""
    
    @pytest.fixture
    def fitted_pca_with_data(self):
        """Create fitted PCA with data."""
        np.random.seed(42)
        X = np.random.randn(50, 5)
        pca = PCA(n_components=3)
        X_transformed = pca.fit_transform(X)
        return pca, X, X_transformed
    
    def test_inverse_transform_shape(self, fitted_pca_with_data):
        """Test inverse transform output shape."""
        pca, X, X_transformed = fitted_pca_with_data
        
        X_reconstructed = pca.inverse_transform(X_transformed)
        
        assert X_reconstructed.shape == X.shape
    
    def test_inverse_transform_reconstruction(self, fitted_pca_with_data):
        """Test that inverse transform reconstructs data reasonably."""
        pca, X, X_transformed = fitted_pca_with_data
        
        X_reconstructed = pca.inverse_transform(X_transformed)
        
        # Reconstruction error should be small if using all components
        # (here we use 3 out of 5, so error will exist)
        reconstruction_error = np.mean((X - X_reconstructed) ** 2)
        
        # Should have some reconstruction error but not too large
        assert reconstruction_error < 10.0
    
    def test_perfect_reconstruction_all_components(self):
        """Test perfect reconstruction when using all components."""
        np.random.seed(42)
        X = np.random.randn(30, 4)
        
        pca = PCA(n_components=4)
        X_transformed = pca.fit_transform(X)
        X_reconstructed = pca.inverse_transform(X_transformed)
        
        # Should reconstruct almost perfectly with all components
        np.testing.assert_array_almost_equal(X, X_reconstructed, decimal=10)


class TestPCAWhitening:
    """Test PCA whitening."""
    
    def test_whitening_enabled(self):
        """Test PCA with whitening enabled."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        
        pca = PCA(n_components=3, whiten=True)
        X_transformed = pca.fit_transform(X)
        
        # With whitening, components should have unit variance
        variances = np.var(X_transformed, axis=0)
        
        # Variances should be close to 1
        np.testing.assert_array_almost_equal(variances, np.ones(3), decimal=1)
    
    def test_whitening_disabled(self):
        """Test PCA with whitening disabled."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        
        pca = PCA(n_components=3, whiten=False)
        X_transformed = pca.fit_transform(X)
        
        # Without whitening, variances should match explained variance
        variances = np.var(X_transformed, axis=0, ddof=1)
        
        # Should be proportional to explained variance
        assert variances.shape == (3,)


class TestPCACovariance:
    """Test covariance computation."""
    
    def test_get_covariance_shape(self):
        """Test covariance matrix shape."""
        np.random.seed(42)
        X = np.random.randn(50, 5)
        
        pca = PCA(n_components=3)
        pca.fit(X)
        
        covariance = pca.get_covariance()
        
        # Covariance should be (n_features, n_features)
        assert covariance.shape == (5, 5)
    
    def test_get_covariance_symmetric(self):
        """Test that covariance matrix is symmetric."""
        np.random.seed(42)
        X = np.random.randn(50, 5)
        
        pca = PCA(n_components=3)
        pca.fit(X)
        
        covariance = pca.get_covariance()
        
        # Covariance should be symmetric
        np.testing.assert_array_almost_equal(covariance, covariance.T)


class TestPCAGetParams:
    """Test get_params method."""
    
    def test_get_params_before_fitting(self):
        """Test get_params before fitting."""
        pca = PCA(n_components=2)
        
        params = pca.get_params()
        
        assert params['n_components'] == 2
        assert params['n_features'] is None
        assert params['components_shape'] is None
    
    def test_get_params_after_fitting(self):
        """Test get_params after fitting."""
        np.random.seed(42)
        X = np.random.randn(50, 5)
        
        pca = PCA(n_components=3)
        pca.fit(X)
        
        params = pca.get_params()
        
        assert params['n_components'] == 3
        assert params['n_features'] == 5
        assert params['components_shape'] == (3, 5)
        assert params['explained_variance'] is not None
        assert params['explained_variance_ratio'] is not None
        assert params['cumulative_variance_ratio'] is not None
    
    def test_cumulative_variance_ratio(self):
        """Test cumulative variance ratio calculation."""
        np.random.seed(42)
        X = np.random.randn(50, 5)
        
        pca = PCA(n_components=3)
        pca.fit(X)
        
        params = pca.get_params()
        cumulative = params['cumulative_variance_ratio']
        
        # Should be cumulative sum
        expected = np.cumsum(pca.explained_variance_ratio)
        np.testing.assert_array_almost_equal(cumulative, expected)
        
        # Last value should be sum of all ratios
        assert cumulative[-1] == np.sum(pca.explained_variance_ratio)


class TestPCAVarianceExplained:
    """Test variance explained calculations."""
    
    def test_variance_explained_positive(self):
        """Test that explained variance is positive."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        
        pca = PCA(n_components=4)
        pca.fit(X)
        
        assert all(pca.explained_variance > 0)
    
    def test_variance_ratio_between_zero_and_one(self):
        """Test that variance ratio values are between 0 and 1."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        
        pca = PCA(n_components=4)
        pca.fit(X)
        
        assert all(0 <= ratio <= 1 for ratio in pca.explained_variance_ratio)
    
    def test_first_component_explains_most(self):
        """Test that first component explains most variance."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        
        pca = PCA(n_components=5)
        pca.fit(X)
        
        # First component should explain most variance
        assert pca.explained_variance_ratio[0] == max(pca.explained_variance_ratio)


class TestPCADimensionalityReduction:
    """Test dimensionality reduction capabilities."""
    
    def test_reduce_from_high_to_low_dimension(self):
        """Test reducing high-dimensional data to low dimension."""
        np.random.seed(42)
        X = np.random.randn(100, 50)
        
        pca = PCA(n_components=5)
        X_reduced = pca.fit_transform(X)
        
        assert X.shape == (100, 50)
        assert X_reduced.shape == (100, 5)
    
    def test_preserve_samples(self):
        """Test that number of samples is preserved."""
        np.random.seed(42)
        X = np.random.randn(75, 10)
        
        pca = PCA(n_components=3)
        X_reduced = pca.fit_transform(X)
        
        assert X.shape[0] == X_reduced.shape[0]
    
    def test_dimensionality_reduction_iris(self):
        """Test dimensionality reduction on iris dataset."""
        iris = load_iris()
        X = iris.data
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(X_scaled)
        
        assert X.shape == (150, 4)
        assert X_reduced.shape == (150, 2)
        
        # Should explain reasonable amount of variance
        total_variance = np.sum(pca.explained_variance_ratio)
        assert total_variance > 0.8  # At least 80%


class TestPCAComparison:
    """Test PCA against scikit-learn."""
    
    @pytest.fixture
    def dataset(self):
        """Generate dataset."""
        np.random.seed(42)
        X, _ = make_classification(
            n_samples=100,
            n_features=20,
            n_informative=15,
            random_state=42
        )
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled
    
    def test_explained_variance_comparison(self, dataset):
        """Compare explained variance with scikit-learn."""
        X = dataset
        
        # Our implementation
        pca_ours = PCA(n_components=5)
        pca_ours.fit(X)
        
        # Scikit-learn
        pca_sklearn = SklearnPCA(n_components=5)
        pca_sklearn.fit(X)
        
        # Explained variance should be similar
        np.testing.assert_array_almost_equal(
            pca_ours.explained_variance,
            pca_sklearn.explained_variance_,
            decimal=5
        )
    
    def test_explained_variance_ratio_comparison(self, dataset):
        """Compare explained variance ratio with scikit-learn."""
        X = dataset
        
        # Our implementation
        pca_ours = PCA(n_components=5)
        pca_ours.fit(X)
        
        # Scikit-learn
        pca_sklearn = SklearnPCA(n_components=5)
        pca_sklearn.fit(X)
        
        # Explained variance ratio should be similar
        np.testing.assert_array_almost_equal(
            pca_ours.explained_variance_ratio,
            pca_sklearn.explained_variance_ratio_,
            decimal=5
        )
    
    def test_transformation_comparison(self, dataset):
        """Compare transformation with scikit-learn."""
        X = dataset
        
        # Our implementation
        pca_ours = PCA(n_components=5)
        X_transformed_ours = pca_ours.fit_transform(X)
        
        # Scikit-learn
        pca_sklearn = SklearnPCA(n_components=5)
        X_transformed_sklearn = pca_sklearn.fit_transform(X)
        
        # Transformations should have same shape
        assert X_transformed_ours.shape == X_transformed_sklearn.shape
        
        # Values should be similar (possibly with sign flip)
        for i in range(5):
            correlation = np.corrcoef(
                X_transformed_ours[:, i],
                X_transformed_sklearn[:, i]
            )[0, 1]
            # Correlation should be close to 1 or -1
            assert abs(abs(correlation) - 1.0) < 0.01


class TestPCAEdgeCases:
    """Test edge cases."""
    
    def test_single_component(self):
        """Test PCA with single component."""
        np.random.seed(42)
        X = np.random.randn(50, 5)
        
        pca = PCA(n_components=1)
        X_transformed = pca.fit_transform(X)
        
        assert X_transformed.shape == (50, 1)
    
    def test_more_components_than_features(self):
        """Test requesting more components than features."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        
        # Request 10 components but only have 5 features
        pca = PCA(n_components=10)
        pca.fit(X)
        
        # Should only get min(n_samples, n_features) components
        assert pca.components.shape[0] <= min(X.shape)
    
    def test_more_components_than_samples(self):
        """Test requesting more components than samples."""
        np.random.seed(42)
        X = np.random.randn(10, 20)
        
        # 10 samples, 20 features
        pca = PCA(n_components=15)
        pca.fit(X)
        
        # Should only get min(n_samples, n_features) = 10 components
        assert pca.components.shape[0] <= min(X.shape)
    
    def test_square_data(self):
        """Test PCA on square data (n_samples = n_features)."""
        np.random.seed(42)
        X = np.random.randn(50, 50)
        
        pca = PCA(n_components=10)
        X_transformed = pca.fit_transform(X)
        
        assert X_transformed.shape == (50, 10)
    
    def test_all_zeros_feature(self):
        """Test PCA with a feature that's all zeros."""
        np.random.seed(42)
        X = np.random.randn(50, 5)
        X[:, 2] = 0  # Make one feature all zeros
        
        pca = PCA(n_components=3)
        X_transformed = pca.fit_transform(X)
        
        # Should still work
        assert X_transformed.shape == (50, 3)
    
    def test_constant_feature(self):
        """Test PCA with a constant feature."""
        np.random.seed(42)
        X = np.random.randn(50, 5)
        X[:, 1] = 5.0  # Make one feature constant
        
        pca = PCA(n_components=3)
        X_transformed = pca.fit_transform(X)
        
        # Should still work (constant feature has zero variance)
        assert X_transformed.shape == (50, 3)


class TestPCAOrthogonality:
    """Test orthogonality of principal components."""
    
    def test_components_orthogonal(self):
        """Test that principal components are orthogonal."""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        
        pca = PCA(n_components=5)
        pca.fit(X)
        
        # Compute dot products between all pairs of components
        for i in range(5):
            for j in range(i+1, 5):
                dot_product = np.dot(pca.components[i], pca.components[j])
                # Should be close to zero (orthogonal)
                assert abs(dot_product) < 1e-10
    
    def test_components_unit_vectors(self):
        """Test that components are unit vectors."""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        
        pca = PCA(n_components=5)
        pca.fit(X)
        
        # Each component should have norm close to 1
        for i in range(5):
            norm = np.linalg.norm(pca.components[i])
            assert abs(norm - 1.0) < 1e-10


class TestPCANumericalStability:
    """Test numerical stability of PCA."""
    
    def test_eigenvalues_real(self):
        """Test that eigenvalues are real (no complex numbers)."""
        np.random.seed(42)
        X = np.random.randn(50, 10)
        
        pca = PCA(n_components=5)
        pca.fit(X)
        
        # Explained variance (eigenvalues) should be real
        assert np.all(np.isreal(pca.explained_variance))
    
    def test_no_nan_values(self):
        """Test that PCA doesn't produce NaN values."""
        np.random.seed(42)
        X = np.random.randn(50, 10)
        
        pca = PCA(n_components=5)
        X_transformed = pca.fit_transform(X)
        
        assert not np.any(np.isnan(X_transformed))
        assert not np.any(np.isnan(pca.components))
        assert not np.any(np.isnan(pca.explained_variance))


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])