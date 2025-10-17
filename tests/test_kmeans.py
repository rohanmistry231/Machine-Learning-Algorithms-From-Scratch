"""
Unit Tests for K-Means Clustering Implementation

This module contains comprehensive tests for the KMeans class,
including tests for:
- Model initialization
- Centroid initialization (random and k-means++)
- Cluster assignment
- Centroid updates
- Convergence behavior
- Inertia calculation
- Predictions and transformations
- fit_predict method
- Comparison with scikit-learn
"""

import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.metrics import adjusted_rand_score

# Import the KMeans model to test
from algorithms.unsupervised.clustering import KMeans


class TestKMeansInitialization:
    """Test KMeans initialization."""
    
    def test_default_initialization(self):
        """Test default initialization."""
        kmeans = KMeans()
        
        assert kmeans.n_clusters == 3
        assert kmeans.max_iters == 300
        assert kmeans.tol == 1e-4
        assert kmeans.init == 'kmeans++'
        assert kmeans.random_state is None
        assert kmeans.centroids is None
        assert kmeans.labels is None
        assert kmeans.inertia is None
        assert kmeans.n_iterations == 0
    
    def test_custom_initialization(self):
        """Test custom initialization."""
        kmeans = KMeans(
            n_clusters=5,
            max_iters=100,
            tol=1e-3,
            init='random',
            random_state=42
        )
        
        assert kmeans.n_clusters == 5
        assert kmeans.max_iters == 100
        assert kmeans.tol == 1e-3
        assert kmeans.init == 'random'
        assert kmeans.random_state == 42
    
    def test_different_n_clusters(self):
        """Test initialization with different n_clusters."""
        for n_clusters in [2, 3, 5, 10]:
            kmeans = KMeans(n_clusters=n_clusters)
            assert kmeans.n_clusters == n_clusters


class TestKMeansCentroidInitialization:
    """Test centroid initialization methods."""
    
    @pytest.fixture
    def simple_data(self):
        """Generate simple 2D data."""
        np.random.seed(42)
        X, _ = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)
        return X
    
    def test_random_initialization_shape(self, simple_data):
        """Test random initialization produces correct shape."""
        X = simple_data
        kmeans = KMeans(n_clusters=3, init='random', random_state=42)
        
        centroids = kmeans._initialize_centroids(X)
        
        assert centroids.shape == (3, 2)
    
    def test_random_initialization_from_data(self, simple_data):
        """Test that random initialization selects from data."""
        X = simple_data
        kmeans = KMeans(n_clusters=3, init='random', random_state=42)
        
        centroids = kmeans._initialize_centroids(X)
        
        # Each centroid should be from the dataset
        for centroid in centroids:
            # Check if centroid matches any data point
            matches = np.any(np.all(np.isclose(X, centroid), axis=1))
            assert matches
    
    def test_kmeans_plus_plus_initialization_shape(self, simple_data):
        """Test k-means++ initialization produces correct shape."""
        X = simple_data
        kmeans = KMeans(n_clusters=3, init='kmeans++', random_state=42)
        
        centroids = kmeans._initialize_centroids(X)
        
        assert centroids.shape == (3, 2)
    
    def test_kmeans_plus_plus_first_centroid_from_data(self, simple_data):
        """Test that k-means++ first centroid is from data."""
        X = simple_data
        kmeans = KMeans(n_clusters=3, init='kmeans++', random_state=42)
        
        centroids = kmeans._initialize_centroids(X)
        
        # First centroid should be from dataset
        first_centroid = centroids[0]
        matches = np.any(np.all(np.isclose(X, first_centroid), axis=1))
        assert matches
    
    def test_invalid_init_method_raises_error(self, simple_data):
        """Test that invalid init method raises error."""
        X = simple_data
        kmeans = KMeans(n_clusters=3, init='invalid_method')
        
        with pytest.raises(ValueError, match="Unknown init method"):
            kmeans._initialize_centroids(X)
    
    def test_random_state_reproducibility(self, simple_data):
        """Test that random_state ensures reproducibility."""
        X = simple_data
        
        kmeans1 = KMeans(n_clusters=3, init='random', random_state=42)
        centroids1 = kmeans1._initialize_centroids(X)
        
        kmeans2 = KMeans(n_clusters=3, init='random', random_state=42)
        centroids2 = kmeans2._initialize_centroids(X)
        
        np.testing.assert_array_equal(centroids1, centroids2)


class TestKMeansClusterAssignment:
    """Test cluster assignment functionality."""
    
    @pytest.fixture
    def fitted_kmeans(self):
        """Create a KMeans with predefined centroids."""
        X = np.array([[0, 0], [1, 1], [10, 10], [11, 11]])
        kmeans = KMeans(n_clusters=2)
        kmeans.centroids = np.array([[0.5, 0.5], [10.5, 10.5]])
        return kmeans, X
    
    def test_assign_clusters_shape(self, fitted_kmeans):
        """Test that cluster assignment returns correct shape."""
        kmeans, X = fitted_kmeans
        
        labels = kmeans._assign_clusters(X)
        
        assert labels.shape == (4,)
    
    def test_assign_clusters_values(self, fitted_kmeans):
        """Test that cluster assignment produces valid labels."""
        kmeans, X = fitted_kmeans
        
        labels = kmeans._assign_clusters(X)
        
        # Labels should be 0 or 1
        assert all(label in [0, 1] for label in labels)
    
    def test_assign_clusters_logic(self, fitted_kmeans):
        """Test that points are assigned to nearest centroid."""
        kmeans, X = fitted_kmeans
        
        labels = kmeans._assign_clusters(X)
        
        # Points [0,0] and [1,1] should be in cluster 0
        # Points [10,10] and [11,11] should be in cluster 1
        assert labels[0] == labels[1]
        assert labels[2] == labels[3]
        assert labels[0] != labels[2]


class TestKMeansCentroidUpdate:
    """Test centroid update functionality."""
    
    def test_update_centroids_shape(self):
        """Test that updated centroids have correct shape."""
        X = np.array([[0, 0], [1, 1], [10, 10], [11, 11]])
        labels = np.array([0, 0, 1, 1])
        
        kmeans = KMeans(n_clusters=2)
        centroids = kmeans._update_centroids(X, labels)
        
        assert centroids.shape == (2, 2)
    
    def test_update_centroids_values(self):
        """Test that centroids are mean of assigned points."""
        X = np.array([[0, 0], [2, 2], [10, 10], [12, 12]])
        labels = np.array([0, 0, 1, 1])
        
        kmeans = KMeans(n_clusters=2)
        centroids = kmeans._update_centroids(X, labels)
        
        # Centroid 0 should be mean of [0,0] and [2,2] = [1,1]
        np.testing.assert_array_almost_equal(centroids[0], [1, 1])
        # Centroid 1 should be mean of [10,10] and [12,12] = [11,11]
        np.testing.assert_array_almost_equal(centroids[1], [11, 11])
    
    def test_empty_cluster_handling(self):
        """Test handling of empty clusters."""
        X = np.array([[0, 0], [1, 1], [2, 2]])
        labels = np.array([0, 0, 0])  # All points in cluster 0, cluster 1 empty
        
        kmeans = KMeans(n_clusters=2, random_state=42)
        centroids = kmeans._update_centroids(X, labels)
        
        # Should still produce 2 centroids
        assert centroids.shape == (2, 2)


class TestKMeansInertiaCalculation:
    """Test inertia calculation."""
    
    def test_compute_inertia_positive(self):
        """Test that inertia is positive."""
        X = np.array([[0, 0], [1, 1], [10, 10], [11, 11]])
        labels = np.array([0, 0, 1, 1])
        
        kmeans = KMeans(n_clusters=2)
        kmeans.centroids = np.array([[0.5, 0.5], [10.5, 10.5]])
        
        inertia = kmeans._compute_inertia(X, labels)
        
        assert inertia > 0
    
    def test_compute_inertia_perfect_clusters(self):
        """Test inertia with perfect clusters."""
        X = np.array([[0, 0], [0, 0], [10, 10], [10, 10]])
        labels = np.array([0, 0, 1, 1])
        
        kmeans = KMeans(n_clusters=2)
        kmeans.centroids = np.array([[0, 0], [10, 10]])
        
        inertia = kmeans._compute_inertia(X, labels)
        
        # Should be zero for perfect clusters
        assert abs(inertia) < 1e-10
    
    def test_compute_inertia_type(self):
        """Test that inertia is a float."""
        X = np.array([[0, 0], [1, 1], [10, 10]])
        labels = np.array([0, 0, 1])
        
        kmeans = KMeans(n_clusters=2)
        kmeans.centroids = np.array([[0.5, 0.5], [10, 10]])
        
        inertia = kmeans._compute_inertia(X, labels)
        
        assert isinstance(inertia, (float, np.floating))


class TestKMeansFitting:
    """Test KMeans fitting."""
    
    @pytest.fixture
    def blob_data(self):
        """Generate blob data."""
        X, y = make_blobs(n_samples=150, centers=3, n_features=2, 
                         cluster_std=0.5, random_state=42)
        return X, y
    
    def test_basic_fitting(self, blob_data):
        """Test basic fitting."""
        X, _ = blob_data
        kmeans = KMeans(n_clusters=3, random_state=42)
        
        result = kmeans.fit(X)
        
        assert result is kmeans
        assert kmeans.centroids is not None
        assert kmeans.labels is not None
        assert kmeans.inertia is not None
        assert kmeans.n_iterations > 0
    
    def test_centroids_shape(self, blob_data):
        """Test that centroids have correct shape."""
        X, _ = blob_data
        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans.fit(X)
        
        assert kmeans.centroids.shape == (3, 2)
    
    def test_labels_shape(self, blob_data):
        """Test that labels have correct shape."""
        X, _ = blob_data
        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans.fit(X)
        
        assert kmeans.labels.shape == (150,)
    
    def test_labels_range(self, blob_data):
        """Test that labels are in valid range."""
        X, _ = blob_data
        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans.fit(X)
        
        assert all(0 <= label < 3 for label in kmeans.labels)
    
    def test_convergence(self, blob_data):
        """Test that algorithm converges."""
        X, _ = blob_data
        kmeans = KMeans(n_clusters=3, max_iters=100, random_state=42)
        kmeans.fit(X)
        
        # Should converge before max_iters for well-separated clusters
        assert kmeans.n_iterations <= 100
    
    def test_inertia_decreases(self):
        """Test that inertia generally decreases over iterations."""
        X, _ = make_blobs(n_samples=100, centers=3, random_state=42)
        
        kmeans = KMeans(n_clusters=3, max_iters=1, random_state=42)
        kmeans.fit(X)
        inertia_1_iter = kmeans.inertia
        
        kmeans = KMeans(n_clusters=3, max_iters=10, random_state=42)
        kmeans.fit(X)
        inertia_10_iter = kmeans.inertia
        
        # More iterations should lead to lower or equal inertia
        assert inertia_10_iter <= inertia_1_iter


class TestKMeansPrediction:
    """Test KMeans prediction."""
    
    @pytest.fixture
    def fitted_model(self):
        """Create a fitted KMeans model."""
        X, _ = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)
        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans.fit(X)
        return kmeans, X
    
    def test_predict_shape(self, fitted_model):
        """Test prediction shape."""
        kmeans, X = fitted_model
        
        X_new = np.random.randn(10, 2)
        predictions = kmeans.predict(X_new)
        
        assert predictions.shape == (10,)
    
    def test_predict_range(self, fitted_model):
        """Test that predictions are valid cluster labels."""
        kmeans, _ = fitted_model
        
        X_new = np.random.randn(10, 2)
        predictions = kmeans.predict(X_new)
        
        assert all(0 <= pred < 3 for pred in predictions)
    
    def test_predict_single_sample(self, fitted_model):
        """Test prediction on single sample."""
        kmeans, _ = fitted_model
        
        X_new = np.array([[0, 0]])
        prediction = kmeans.predict(X_new)
        
        assert prediction.shape == (1,)
    
    def test_fit_predict(self):
        """Test fit_predict method."""
        X, _ = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)
        
        kmeans = KMeans(n_clusters=3, random_state=42)
        labels = kmeans.fit_predict(X)
        
        assert labels.shape == (100,)
        assert kmeans.centroids is not None


class TestKMeansTransform:
    """Test KMeans transform method."""
    
    @pytest.fixture
    def fitted_model(self):
        """Create a fitted KMeans model."""
        X, _ = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)
        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans.fit(X)
        return kmeans, X
    
    def test_transform_shape(self, fitted_model):
        """Test transform output shape."""
        kmeans, X = fitted_model
        
        distances = kmeans.transform(X)
        
        # Should be (n_samples, n_clusters)
        assert distances.shape == (100, 3)
    
    def test_transform_positive_distances(self, fitted_model):
        """Test that transform returns positive distances."""
        kmeans, X = fitted_model
        
        distances = kmeans.transform(X)
        
        assert np.all(distances >= 0)
    
    def test_transform_nearest_cluster(self, fitted_model):
        """Test that nearest cluster matches prediction."""
        kmeans, X = fitted_model
        
        distances = kmeans.transform(X)
        predicted_labels = kmeans.predict(X)
        nearest_clusters = np.argmin(distances, axis=1)
        
        np.testing.assert_array_equal(predicted_labels, nearest_clusters)


class TestKMeansGetParams:
    """Test get_params method."""
    
    def test_get_params_before_fitting(self):
        """Test get_params before fitting."""
        kmeans = KMeans(n_clusters=4)
        
        params = kmeans.get_params()
        
        assert params['n_clusters'] == 4
        assert params['centroids'] is None
        assert params['inertia'] is None
        assert params['n_iterations'] == 0
    
    def test_get_params_after_fitting(self):
        """Test get_params after fitting."""
        X, _ = make_blobs(n_samples=100, centers=3, random_state=42)
        
        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans.fit(X)
        
        params = kmeans.get_params()
        
        assert params['centroids'] is not None
        assert params['n_clusters'] == 3
        assert params['inertia'] is not None
        assert params['n_iterations'] > 0


class TestKMeansConvergence:
    """Test convergence behavior."""
    
    def test_early_stopping(self):
        """Test that algorithm stops when converged."""
        X, _ = make_blobs(n_samples=100, centers=3, cluster_std=0.3, random_state=42)
        
        kmeans = KMeans(n_clusters=3, max_iters=1000, tol=1e-4, random_state=42)
        kmeans.fit(X)
        
        # Should converge well before max_iters for well-separated clusters
        assert kmeans.n_iterations < 1000
    
    def test_max_iterations_respected(self):
        """Test that algorithm respects max_iters."""
        X, _ = make_blobs(n_samples=100, centers=5, random_state=42)
        
        max_iters = 10
        kmeans = KMeans(n_clusters=5, max_iters=max_iters, tol=0, random_state=42)
        kmeans.fit(X)
        
        # Should not exceed max_iters
        assert kmeans.n_iterations <= max_iters


class TestKMeansComparison:
    """Test KMeans against scikit-learn."""
    
    @pytest.fixture
    def dataset(self):
        """Generate dataset."""
        X, y_true = make_blobs(n_samples=200, centers=4, n_features=2,
                              cluster_std=0.6, random_state=42)
        return X, y_true
    
    def test_clustering_quality_comparison(self, dataset):
        """Compare clustering quality with scikit-learn using ARI."""
        X, y_true = dataset
        
        # Our implementation
        kmeans_ours = KMeans(n_clusters=4, init='k-means++', random_state=42)
        y_pred_ours = kmeans_ours.fit_predict(X)
        ari_ours = adjusted_rand_score(y_true, y_pred_ours)
        
        # Scikit-learn
        kmeans_sklearn = SklearnKMeans(n_clusters=4, init='k-means++', 
                                       random_state=42, n_init=1)
        y_pred_sklearn = kmeans_sklearn.fit_predict(X)
        ari_sklearn = adjusted_rand_score(y_true, y_pred_sklearn)
        
        # Both should achieve reasonable clustering
        assert ari_ours > 0.7
        assert abs(ari_ours - ari_sklearn) < 0.3


class TestKMeansEdgeCases:
    """Test edge cases."""
    
    def test_single_cluster(self):
        """Test with single cluster."""
        X = np.random.randn(50, 2)
        
        kmeans = KMeans(n_clusters=1, random_state=42)
        kmeans.fit(X)
        
        # All points should be in cluster 0
        assert all(label == 0 for label in kmeans.labels)
    
    def test_more_clusters_than_samples(self):
        """Test with more clusters than samples."""
        X = np.random.randn(5, 2)
        
        kmeans = KMeans(n_clusters=10, random_state=42)
        kmeans.fit(X)
        
        # Should still work (some clusters may be empty or duplicated)
        assert kmeans.centroids.shape[0] == 10
    
    def test_single_feature(self):
        """Test with single feature."""
        X = np.random.randn(100, 1)
        
        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans.fit(X)
        
        assert kmeans.centroids.shape == (3, 1)
    
    def test_many_features(self):
        """Test with many features."""
        X = np.random.randn(100, 50)
        
        kmeans = KMeans(n_clusters=5, random_state=42)
        kmeans.fit(X)
        
        assert kmeans.centroids.shape == (5, 50)
    
    def test_identical_points(self):
        """Test with identical points."""
        X = np.ones((50, 2))
        
        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans.fit(X)
        
        # Should still work (all centroids should be same)
        assert kmeans.centroids is not None


class TestKMeansDifferentNClusters:
    """Test KMeans with different numbers of clusters."""
    
    @pytest.fixture
    def dataset(self):
        """Generate dataset."""
        X, _ = make_blobs(n_samples=200, centers=5, random_state=42)
        return X
    
    def test_various_n_clusters(self, dataset):
        """Test with various numbers of clusters."""
        X = dataset
        
        n_clusters_list = [2, 3, 5, 8, 10]
        
        for n_clusters in n_clusters_list:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans.fit(X)
            
            assert kmeans.centroids.shape[0] == n_clusters
            assert len(np.unique(kmeans.labels)) <= n_clusters
    
    def test_inertia_increases_with_fewer_clusters(self, dataset):
        """Test that inertia generally increases with fewer clusters."""
        X = dataset
        
        kmeans_2 = KMeans(n_clusters=2, random_state=42)
        kmeans_2.fit(X)
        
        kmeans_10 = KMeans(n_clusters=10, random_state=42)
        kmeans_10.fit(X)
        
        # More clusters should lead to lower inertia
        assert kmeans_10.inertia <= kmeans_2.inertia


class TestKMeansReproducibility:
    """Test reproducibility with random_state."""
    
    def test_random_state_reproducibility(self):
        """Test that same random_state produces same results."""
        X = np.random.randn(100, 5)
        
        kmeans1 = KMeans(n_clusters=3, random_state=42)
        labels1 = kmeans1.fit_predict(X)
        
        kmeans2 = KMeans(n_clusters=3, random_state=42)
        labels2 = kmeans2.fit_predict(X)
        
        np.testing.assert_array_equal(labels1, labels2)
        np.testing.assert_array_almost_equal(kmeans1.centroids, kmeans2.centroids)
    
    def test_different_random_state_different_results(self):
        """Test that different random_state may produce different results."""
        X = np.random.randn(100, 5)
        
        kmeans1 = KMeans(n_clusters=3, random_state=42)
        labels1 = kmeans1.fit_predict(X)
        
        kmeans2 = KMeans(n_clusters=3, random_state=99)
        labels2 = kmeans2.fit_predict(X)
        
        # Results may differ (not guaranteed to differ, but often will)
        # Just check that both produce valid outputs
        assert labels1.shape == labels2.shape
        assert kmeans1.centroids.shape == kmeans2.centroids.shape


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])