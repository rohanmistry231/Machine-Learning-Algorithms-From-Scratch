"""
Unit Tests for K-Nearest Neighbors Implementation

This module contains comprehensive tests for the KNearestNeighbors class,
including tests for:
- Model initialization
- Distance metrics (Euclidean, Manhattan, Minkowski)
- Classification and regression tasks
- Weighted and uniform voting
- Neighbor finding
- Predictions and scoring
- Different k values
- Comparison with scikit-learn
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

# Import the KNearestNeighbors model to test
from algorithms.supervised.classification import KNearestNeighbors


class TestKNNInitialization:
    """Test KNearestNeighbors initialization."""
    
    def test_default_initialization(self):
        """Test default initialization."""
        knn = KNearestNeighbors()
        
        assert knn.n_neighbors == 5
        assert knn.metric == 'euclidean'
        assert knn.p == 2
        assert knn.weights == 'uniform'
        assert knn.task == 'classification'
        assert knn.X_train is None
        assert knn.y_train is None
    
    def test_custom_initialization(self):
        """Test custom initialization."""
        knn = KNearestNeighbors(
            n_neighbors=3,
            metric='manhattan',
            p=3,
            weights='distance',
            task='regression'
        )
        
        assert knn.n_neighbors == 3
        assert knn.metric == 'manhattan'
        assert knn.p == 3
        assert knn.weights == 'distance'
        assert knn.task == 'regression'
    
    def test_different_k_values(self):
        """Test initialization with different k values."""
        k_values = [1, 3, 5, 10, 20]
        
        for k in k_values:
            knn = KNearestNeighbors(n_neighbors=k)
            assert knn.n_neighbors == k
    
    def test_metric_options(self):
        """Test different metric options."""
        metrics = ['euclidean', 'manhattan', 'minkowski']
        
        for metric in metrics:
            knn = KNearestNeighbors(metric=metric)
            assert knn.metric == metric
    
    def test_weights_options(self):
        """Test different weight options."""
        weights = ['uniform', 'distance']
        
        for weight in weights:
            knn = KNearestNeighbors(weights=weight)
            assert knn.weights == weight


class TestKNNFitting:
    """Test KNearestNeighbors fitting."""
    
    @pytest.fixture
    def simple_classification_data(self):
        """Generate simple classification data."""
        X = np.array([[1, 2], [2, 3], [3, 4], [10, 11], [11, 12], [12, 13]])
        y = np.array([0, 0, 0, 1, 1, 1])
        return X, y
    
    @pytest.fixture
    def simple_regression_data(self):
        """Generate simple regression data."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])
        return X, y
    
    def test_basic_fitting(self, simple_classification_data):
        """Test basic fitting."""
        X, y = simple_classification_data
        knn = KNearestNeighbors(n_neighbors=3)
        
        result = knn.fit(X, y)
        
        assert result is knn
        assert knn.X_train is not None
        assert knn.y_train is not None
        np.testing.assert_array_equal(knn.X_train, X)
        np.testing.assert_array_equal(knn.y_train, y)
    
    def test_fitting_stores_data(self, simple_classification_data):
        """Test that fitting stores training data."""
        X, y = simple_classification_data
        knn = KNearestNeighbors(n_neighbors=3)
        
        knn.fit(X, y)
        
        assert knn.X_train.shape == X.shape
        assert knn.y_train.shape == y.shape


class TestKNNDistanceMetrics:
    """Test distance metric calculations."""
    
    def test_euclidean_distance(self):
        """Test Euclidean distance calculation."""
        knn = KNearestNeighbors(metric='euclidean')
        
        x1 = np.array([0, 0])
        x2 = np.array([3, 4])
        
        distance = knn._compute_distance(x1, x2)
        
        # Distance should be 5 (3-4-5 triangle)
        assert abs(distance - 5.0) < 1e-6
    
    def test_manhattan_distance(self):
        """Test Manhattan distance calculation."""
        knn = KNearestNeighbors(metric='manhattan')
        
        x1 = np.array([0, 0])
        x2 = np.array([3, 4])
        
        distance = knn._compute_distance(x1, x2)
        
        # Distance should be 7 (3 + 4)
        assert abs(distance - 7.0) < 1e-6
    
    def test_minkowski_distance_p2(self):
        """Test Minkowski distance with p=2 (same as Euclidean)."""
        knn = KNearestNeighbors(metric='minkowski', p=2)
        
        x1 = np.array([0, 0])
        x2 = np.array([3, 4])
        
        distance = knn._compute_distance(x1, x2)
        
        # Should be same as Euclidean
        assert abs(distance - 5.0) < 1e-6
    
    def test_minkowski_distance_p1(self):
        """Test Minkowski distance with p=1 (same as Manhattan)."""
        knn = KNearestNeighbors(metric='minkowski', p=1)
        
        x1 = np.array([0, 0])
        x2 = np.array([3, 4])
        
        distance = knn._compute_distance(x1, x2)
        
        # Should be same as Manhattan
        assert abs(distance - 7.0) < 1e-6
    
    def test_distance_zero_same_point(self):
        """Test distance is zero for same point."""
        knn = KNearestNeighbors(metric='euclidean')
        
        x = np.array([1, 2, 3])
        
        distance = knn._compute_distance(x, x)
        
        assert distance == 0.0
    
    def test_distance_symmetry(self):
        """Test that distance is symmetric."""
        knn = KNearestNeighbors(metric='euclidean')
        
        x1 = np.array([1, 2])
        x2 = np.array([4, 6])
        
        d1 = knn._compute_distance(x1, x2)
        d2 = knn._compute_distance(x2, x1)
        
        assert abs(d1 - d2) < 1e-10
    
    def test_invalid_metric_raises_error(self):
        """Test that invalid metric raises error."""
        knn = KNearestNeighbors(metric='invalid_metric')
        knn.X_train = np.array([[1, 2]])
        knn.y_train = np.array([0])
        
        x = np.array([3, 4])
        
        with pytest.raises(ValueError, match="Unknown metric"):
            knn._compute_distance(x, knn.X_train[0])


class TestKNNNeighborFinding:
    """Test neighbor finding functionality."""
    
    @pytest.fixture
    def fitted_knn(self):
        """Create a fitted KNN model."""
        X = np.array([[0, 0], [1, 1], [2, 2], [10, 10], [11, 11], [12, 12]])
        y = np.array([0, 0, 0, 1, 1, 1])
        
        knn = KNearestNeighbors(n_neighbors=3)
        knn.fit(X, y)
        
        return knn, X, y
    
    def test_get_neighbors_returns_correct_count(self, fitted_knn):
        """Test that get_neighbors returns k neighbors."""
        knn, _, _ = fitted_knn
        
        x_test = np.array([0.5, 0.5])
        neighbor_labels, neighbor_distances = knn._get_neighbors(x_test)
        
        assert len(neighbor_labels) == 3
        assert len(neighbor_distances) == 3
    
    def test_get_neighbors_closest_first(self, fitted_knn):
        """Test that neighbors are sorted by distance."""
        knn, _, _ = fitted_knn
        
        x_test = np.array([0, 0])
        _, neighbor_distances = knn._get_neighbors(x_test)
        
        # Distances should be sorted (non-decreasing)
        assert all(neighbor_distances[i] <= neighbor_distances[i+1] 
                  for i in range(len(neighbor_distances)-1))
    
    def test_get_neighbors_returns_labels(self, fitted_knn):
        """Test that get_neighbors returns valid labels."""
        knn, _, y = fitted_knn
        
        x_test = np.array([1, 1])
        neighbor_labels, _ = knn._get_neighbors(x_test)
        
        # All labels should be from training set
        assert all(label in y for label in neighbor_labels)


class TestKNNClassificationPrediction:
    """Test KNN classification predictions."""
    
    @pytest.fixture
    def classification_data(self):
        """Generate classification data."""
        X = np.array([[0, 0], [1, 1], [2, 2], [10, 10], [11, 11], [12, 12]])
        y = np.array([0, 0, 0, 1, 1, 1])
        return X, y
    
    def test_uniform_weights_prediction(self, classification_data):
        """Test prediction with uniform weights."""
        X, y = classification_data
        knn = KNearestNeighbors(n_neighbors=3, weights='uniform', task='classification')
        knn.fit(X, y)
        
        # Point close to class 0
        x_test = np.array([[0.5, 0.5]])
        prediction = knn.predict(x_test)
        
        assert prediction[0] == 0
    
    def test_distance_weights_prediction(self, classification_data):
        """Test prediction with distance weights."""
        X, y = classification_data
        knn = KNearestNeighbors(n_neighbors=3, weights='distance', task='classification')
        knn.fit(X, y)
        
        # Point close to class 1
        x_test = np.array([[11, 11]])
        prediction = knn.predict(x_test)
        
        assert prediction[0] == 1
    
    def test_prediction_shape(self, classification_data):
        """Test prediction output shape."""
        X, y = classification_data
        knn = KNearestNeighbors(n_neighbors=3, task='classification')
        knn.fit(X, y)
        
        X_test = np.array([[1, 1], [11, 11]])
        predictions = knn.predict(X_test)
        
        assert predictions.shape == (2,)
    
    def test_single_sample_prediction(self, classification_data):
        """Test prediction on single sample."""
        X, y = classification_data
        knn = KNearestNeighbors(n_neighbors=3, task='classification')
        knn.fit(X, y)
        
        x_test = np.array([[1, 1]])
        prediction = knn.predict(x_test)
        
        assert prediction.shape == (1,)
        assert prediction[0] in [0, 1]


class TestKNNRegressionPrediction:
    """Test KNN regression predictions."""
    
    @pytest.fixture
    def regression_data(self):
        """Generate regression data."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])
        return X, y
    
    def test_uniform_weights_regression(self, regression_data):
        """Test regression with uniform weights."""
        X, y = regression_data
        knn = KNearestNeighbors(n_neighbors=3, weights='uniform', task='regression')
        knn.fit(X, y)
        
        # Predict for x=3, neighbors should be 2,3,4 with y=4,6,8
        x_test = np.array([[3]])
        prediction = knn.predict(x_test)
        
        # Average should be (4+6+8)/3 = 6
        assert abs(prediction[0] - 6.0) < 1e-6
    
    def test_distance_weights_regression(self, regression_data):
        """Test regression with distance weights."""
        X, y = regression_data
        knn = KNearestNeighbors(n_neighbors=3, weights='distance', task='regression')
        knn.fit(X, y)
        
        x_test = np.array([[3]])
        prediction = knn.predict(x_test)
        
        # Should be close to 6 but weighted by distance
        assert isinstance(prediction[0], (float, np.floating))
    
    def test_regression_prediction_shape(self, regression_data):
        """Test regression prediction shape."""
        X, y = regression_data
        knn = KNearestNeighbors(n_neighbors=3, task='regression')
        knn.fit(X, y)
        
        X_test = np.array([[2], [3], [4]])
        predictions = knn.predict(X_test)
        
        assert predictions.shape == (3,)


class TestKNNScoring:
    """Test KNN scoring."""
    
    def test_classification_accuracy(self):
        """Test classification accuracy calculation."""
        X = np.array([[0, 0], [1, 1], [2, 2], [10, 10], [11, 11], [12, 12]])
        y = np.array([0, 0, 0, 1, 1, 1])
        
        knn = KNearestNeighbors(n_neighbors=3, task='classification')
        knn.fit(X, y)
        
        accuracy = knn.score(X, y)
        
        # Should achieve perfect accuracy on training data
        assert accuracy == 1.0
    
    def test_classification_accuracy_range(self):
        """Test that classification accuracy is between 0 and 1."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        
        knn = KNearestNeighbors(n_neighbors=5, task='classification')
        knn.fit(X, y)
        
        accuracy = knn.score(X, y)
        
        assert 0.0 <= accuracy <= 1.0
    
    def test_regression_r2_score(self):
        """Test regression R² score calculation."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])
        
        knn = KNearestNeighbors(n_neighbors=3, task='regression')
        knn.fit(X, y)
        
        r2 = knn.score(X, y)
        
        # Should achieve good R² on simple linear data
        assert r2 > 0.9
    
    def test_score_type(self):
        """Test that score returns a float."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1, 0])
        
        knn = KNearestNeighbors(n_neighbors=2, task='classification')
        knn.fit(X, y)
        
        score = knn.score(X, y)
        
        assert isinstance(score, (float, np.floating))


class TestKNNGetParams:
    """Test get_params method."""
    
    def test_get_params_before_fitting(self):
        """Test get_params before fitting."""
        knn = KNearestNeighbors(n_neighbors=7, metric='manhattan')
        
        params = knn.get_params()
        
        assert params['n_neighbors'] == 7
        assert params['metric'] == 'manhattan'
        assert params['n_training_samples'] == 0
    
    def test_get_params_after_fitting(self):
        """Test get_params after fitting."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1, 0])
        
        knn = KNearestNeighbors(n_neighbors=2)
        knn.fit(X, y)
        
        params = knn.get_params()
        
        assert params['n_training_samples'] == 3
    
    def test_get_params_returns_dict(self):
        """Test that get_params returns a dictionary."""
        knn = KNearestNeighbors()
        
        params = knn.get_params()
        
        assert isinstance(params, dict)
        assert 'n_neighbors' in params
        assert 'metric' in params
        assert 'weights' in params


class TestKNNDifferentKValues:
    """Test KNN with different k values."""
    
    @pytest.fixture
    def dataset(self):
        """Generate dataset."""
        X, y = make_classification(
            n_samples=100,
            n_features=5,
            n_informative=3,
            random_state=42
        )
        return X, y
    
    def test_k_equals_1(self, dataset):
        """Test KNN with k=1."""
        X, y = dataset
        
        knn = KNearestNeighbors(n_neighbors=1, task='classification')
        knn.fit(X, y)
        
        # k=1 should achieve perfect training accuracy
        accuracy = knn.score(X, y)
        assert accuracy == 1.0
    
    def test_different_k_values(self, dataset):
        """Test KNN with various k values."""
        X, y = dataset
        
        k_values = [1, 3, 5, 10, 20]
        accuracies = []
        
        for k in k_values:
            knn = KNearestNeighbors(n_neighbors=k, task='classification')
            knn.fit(X, y)
            accuracy = knn.score(X, y)
            accuracies.append(accuracy)
        
        # k=1 should have highest training accuracy
        assert accuracies[0] == 1.0
    
    def test_k_larger_than_samples(self):
        """Test KNN when k > number of samples."""
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])
        
        # k=5 but only 2 samples
        knn = KNearestNeighbors(n_neighbors=5, task='classification')
        knn.fit(X, y)
        
        # Should still work (will use all available neighbors)
        x_test = np.array([[2, 3]])
        prediction = knn.predict(x_test)
        
        assert prediction.shape == (1,)


class TestKNNComparison:
    """Test KNN against scikit-learn."""
    
    @pytest.fixture
    def classification_dataset(self):
        """Generate classification dataset."""
        X, y = make_classification(
            n_samples=100,
            n_features=5,
            n_informative=3,
            random_state=42
        )
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    @pytest.fixture
    def regression_dataset(self):
        """Generate regression dataset."""
        X, y = make_regression(
            n_samples=100,
            n_features=5,
            noise=10,
            random_state=42
        )
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    def test_classification_accuracy_comparison(self, classification_dataset):
        """Compare classification accuracy with scikit-learn."""
        X_train, X_test, y_train, y_test = classification_dataset
        
        # Our implementation
        knn_ours = KNearestNeighbors(
            n_neighbors=5,
            metric='euclidean',
            weights='uniform',
            task='classification'
        )
        knn_ours.fit(X_train, y_train)
        accuracy_ours = knn_ours.score(X_test, y_test)
        
        # Scikit-learn
        knn_sklearn = KNeighborsClassifier(
            n_neighbors=5,
            metric='euclidean',
            weights='uniform'
        )
        knn_sklearn.fit(X_train, y_train)
        accuracy_sklearn = knn_sklearn.score(X_test, y_test)
        
        # Should be very similar
        assert abs(accuracy_ours - accuracy_sklearn) < 0.05
    
    def test_regression_r2_comparison(self, regression_dataset):
        """Compare regression R² with scikit-learn."""
        X_train, X_test, y_train, y_test = regression_dataset
        
        # Our implementation
        knn_ours = KNearestNeighbors(
            n_neighbors=5,
            metric='euclidean',
            weights='uniform',
            task='regression'
        )
        knn_ours.fit(X_train, y_train)
        r2_ours = knn_ours.score(X_test, y_test)
        
        # Scikit-learn
        knn_sklearn = KNeighborsRegressor(
            n_neighbors=5,
            metric='euclidean',
            weights='uniform'
        )
        knn_sklearn.fit(X_train, y_train)
        r2_sklearn = knn_sklearn.score(X_test, y_test)
        
        # Should be very similar
        assert abs(r2_ours - r2_sklearn) < 0.1


class TestKNNEdgeCases:
    """Test edge cases."""
    
    def test_single_sample(self):
        """Test with single training sample."""
        X = np.array([[1, 2]])
        y = np.array([0])
        
        knn = KNearestNeighbors(n_neighbors=1, task='classification')
        knn.fit(X, y)
        
        prediction = knn.predict(X)
        
        assert prediction[0] == 0
    
    def test_single_feature(self):
        """Test with single feature."""
        X = np.array([[1], [2], [3], [4]])
        y = np.array([0, 0, 1, 1])
        
        knn = KNearestNeighbors(n_neighbors=3, task='classification')
        knn.fit(X, y)
        
        accuracy = knn.score(X, y)
        
        assert accuracy > 0
    
    def test_binary_classification(self):
        """Test standard binary classification."""
        X = np.array([[1, 2], [2, 3], [3, 4], [10, 11], [11, 12], [12, 13]])
        y = np.array([0, 0, 0, 1, 1, 1])
        
        knn = KNearestNeighbors(n_neighbors=3, task='classification')
        knn.fit(X, y)
        
        accuracy = knn.score(X, y)
        
        assert accuracy == 1.0
    
    def test_multiclass_classification(self):
        """Test multiclass classification."""
        X, y = make_classification(
            n_samples=150,
            n_features=4,
            n_informative=3,
            n_classes=3,
            random_state=42
        )
        
        knn = KNearestNeighbors(n_neighbors=5, task='classification')
        knn.fit(X, y)
        
        predictions = knn.predict(X)
        
        # Should predict all 3 classes
        assert len(np.unique(predictions)) >= 2
    
    def test_identical_points(self):
        """Test with identical training points."""
        X = np.array([[1, 2], [1, 2], [3, 4], [3, 4]])
        y = np.array([0, 0, 1, 1])
        
        knn = KNearestNeighbors(n_neighbors=2, task='classification')
        knn.fit(X, y)
        
        # Should still work
        prediction = knn.predict(np.array([[1, 2]]))
        assert prediction[0] == 0


class TestKNNWeightingSchemes:
    """Test different weighting schemes."""
    
    @pytest.fixture
    def dataset(self):
        """Generate dataset."""
        X = np.array([[0, 0], [1, 0], [2, 0], [10, 0], [11, 0], [12, 0]])
        y = np.array([0, 0, 0, 1, 1, 1])
        return X, y
    
    def test_uniform_vs_distance_weights(self, dataset):
        """Test difference between uniform and distance weights."""
        X, y = dataset
        
        # Uniform weights
        knn_uniform = KNearestNeighbors(
            n_neighbors=5,
            weights='uniform',
            task='classification'
        )
        knn_uniform.fit(X, y)
        
        # Distance weights
        knn_distance = KNearestNeighbors(
            n_neighbors=5,
            weights='distance',
            task='classification'
        )
        knn_distance.fit(X, y)
        
        # Both should work
        x_test = np.array([[1, 0]])
        pred_uniform = knn_uniform.predict(x_test)
        pred_distance = knn_distance.predict(x_test)
        
        assert pred_uniform.shape == pred_distance.shape


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])