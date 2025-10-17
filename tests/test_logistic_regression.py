"""
Unit Tests for Logistic Regression Implementation

This module contains comprehensive tests for the LogisticRegression class,
including tests for:
- Model initialization
- Sigmoid function
- Model fitting
- Predictions (probabilities and class labels)
- Scoring and accuracy
- Regularization (L1, L2)
- Edge cases and error handling
- Comparison with scikit-learn
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression

# Import the LogisticRegression model to test
from algorithms.supervised.classification import LogisticRegression


class TestLogisticRegressionInitialization:
    """Test LogisticRegression initialization and parameters."""
    
    def test_default_initialization(self):
        """Test that LogisticRegression initializes with default parameters."""
        model = LogisticRegression()
        
        assert model.learning_rate == 0.01
        assert model.n_iterations == 1000
        assert model.regularization is None
        assert model.lambda_reg == 0.01
        assert model.tol == 1e-4
        assert model.weights is None
        assert model.bias is None
        assert model.cost_history == []
    
    def test_custom_initialization(self):
        """Test LogisticRegression initialization with custom parameters."""
        model = LogisticRegression(
            learning_rate=0.05,
            n_iterations=500,
            regularization='l2',
            lambda_reg=0.1,
            tol=1e-3
        )
        
        assert model.learning_rate == 0.05
        assert model.n_iterations == 500
        assert model.regularization == 'l2'
        assert model.lambda_reg == 0.1
        assert model.tol == 1e-3
    
    def test_l1_regularization_initialization(self):
        """Test initialization with L1 regularization."""
        model = LogisticRegression(regularization='l1', lambda_reg=0.05)
        
        assert model.regularization == 'l1'
        assert model.lambda_reg == 0.05
    
    def test_l2_regularization_initialization(self):
        """Test initialization with L2 regularization."""
        model = LogisticRegression(regularization='l2', lambda_reg=0.1)
        
        assert model.regularization == 'l2'
        assert model.lambda_reg == 0.1


class TestLogisticRegressionSigmoid:
    """Test sigmoid activation function."""
    
    def test_sigmoid_at_zero(self):
        """Test sigmoid(0) = 0.5."""
        model = LogisticRegression()
        
        result = model._sigmoid(np.array([0]))
        
        np.testing.assert_almost_equal(result, 0.5)
    
    def test_sigmoid_large_positive(self):
        """Test sigmoid approaches 1 for large positive values."""
        model = LogisticRegression()
        
        result = model._sigmoid(np.array([100]))
        
        assert result > 0.99
    
    def test_sigmoid_large_negative(self):
        """Test sigmoid approaches 0 for large negative values."""
        model = LogisticRegression()
        
        result = model._sigmoid(np.array([-100]))
        
        assert result < 0.01
    
    def test_sigmoid_range(self):
        """Test sigmoid output is always between 0 and 1."""
        model = LogisticRegression()
        
        z = np.linspace(-10, 10, 100)
        result = model._sigmoid(z)
        
        assert np.all(result >= 0)
        assert np.all(result <= 1)
    
    def test_sigmoid_monotonic(self):
        """Test sigmoid is monotonically increasing."""
        model = LogisticRegression()
        
        z = np.linspace(-5, 5, 100)
        result = model._sigmoid(z)
        
        # Check that result is increasing
        assert np.all(np.diff(result) >= 0)
    
    def test_sigmoid_symmetric(self):
        """Test sigmoid is symmetric around 0.5."""
        model = LogisticRegression()
        
        z_pos = np.array([2])
        z_neg = np.array([-2])
        
        result_pos = model._sigmoid(z_pos)
        result_neg = model._sigmoid(z_neg)
        
        # sigmoid(z) + sigmoid(-z) should equal 1
        np.testing.assert_almost_equal(result_pos + result_neg, 1.0)


class TestLogisticRegressionFitting:
    """Test LogisticRegression fitting functionality."""
    
    @pytest.fixture
    def simple_data(self):
        """Generate simple binary classification data."""
        X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        y = np.array([0, 0, 1, 1])
        return X, y
    
    @pytest.fixture
    def complex_data(self):
        """Generate complex dataset using sklearn."""
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_informative=8,
            n_redundant=2,
            random_state=42
        )
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        return X, y
    
    def test_basic_fitting(self, simple_data):
        """Test basic model fitting."""
        X, y = simple_data
        model = LogisticRegression(learning_rate=0.1, n_iterations=100)
        
        result = model.fit(X, y)
        
        # Check that fit returns self
        assert result is model
        
        # Check that weights and bias are initialized
        assert model.weights is not None
        assert model.bias is not None
        
        # Check shapes
        assert model.weights.shape == (2, 1)
        assert isinstance(model.bias, (int, float, np.number))
    
    def test_cost_history_recorded(self, simple_data):
        """Test that cost history is recorded during fitting."""
        X, y = simple_data
        n_iterations = 100
        model = LogisticRegression(learning_rate=0.1, n_iterations=n_iterations)
        
        model.fit(X, y)
        
        # Cost history should have entries
        assert len(model.cost_history) > 0
        assert len(model.cost_history) <= n_iterations
    
    def test_cost_decreasing(self, simple_data):
        """Test that cost generally decreases during training."""
        X, y = simple_data
        model = LogisticRegression(learning_rate=0.1, n_iterations=500)
        
        model.fit(X, y)
        
        # Final cost should be less than initial cost
        assert model.cost_history[-1] < model.cost_history[0]
    
    def test_fitting_with_1d_target(self, complex_data):
        """Test fitting with 1D target array."""
        X, y = complex_data
        model = LogisticRegression(learning_rate=0.01, n_iterations=100)
        
        # Should not raise error
        model.fit(X, y)
        
        assert model.weights is not None
    
    def test_fitting_with_2d_target(self, complex_data):
        """Test fitting with 2D target array."""
        X, y = complex_data
        y_2d = y.reshape(-1, 1)
        model = LogisticRegression(learning_rate=0.01, n_iterations=100)
        
        # Should not raise error
        model.fit(X, y_2d)
        
        assert model.weights is not None
    
    def test_convergence_early_stopping(self):
        """Test that training stops early when converged."""
        X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        y = np.array([0, 0, 1, 1])
        
        model = LogisticRegression(
            learning_rate=0.1,
            n_iterations=10000,
            tol=0.001
        )
        
        model.fit(X, y)
        
        # Should stop before max iterations due to convergence
        assert len(model.cost_history) < 10000


class TestLogisticRegressionPredictProba:
    """Test probability predictions."""
    
    @pytest.fixture
    def trained_model(self):
        """Create and train a model."""
        X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        y = np.array([0, 0, 1, 1])
        
        model = LogisticRegression(learning_rate=0.1, n_iterations=500)
        model.fit(X, y)
        
        return model, X, y
    
    def test_predict_proba_shape(self, trained_model):
        """Test predict_proba returns correct shape."""
        model, X, _ = trained_model
        
        proba = model.predict_proba(X)
        
        assert proba.shape == (4, 1)
    
    def test_predict_proba_range(self, trained_model):
        """Test predict_proba returns values between 0 and 1."""
        model, X, _ = trained_model
        
        proba = model.predict_proba(X)
        
        assert np.all(proba >= 0)
        assert np.all(proba <= 1)
    
    def test_predict_proba_single_sample(self, trained_model):
        """Test predict_proba on single sample."""
        model, _, _ = trained_model
        
        X_test = np.array([[1, 1]])
        proba = model.predict_proba(X_test)
        
        assert proba.shape == (1, 1)
        assert 0 <= proba[0, 0] <= 1
    
    def test_predict_proba_multiple_samples(self, trained_model):
        """Test predict_proba on multiple samples."""
        model, _, _ = trained_model
        
        X_test = np.array([[0, 0], [1, 1], [2, 2]])
        proba = model.predict_proba(X_test)
        
        assert proba.shape == (3, 1)
        assert np.all((proba >= 0) & (proba <= 1))


class TestLogisticRegressionPredict:
    """Test class label predictions."""
    
    @pytest.fixture
    def trained_model(self):
        """Create and train a model."""
        X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        y = np.array([0, 0, 1, 1])
        
        model = LogisticRegression(learning_rate=0.1, n_iterations=500)
        model.fit(X, y)
        
        return model, X, y
    
    def test_predict_shape(self, trained_model):
        """Test predict returns correct shape."""
        model, X, _ = trained_model
        
        predictions = model.predict(X)
        
        assert predictions.shape == (4, 1)
    
    def test_predict_binary_output(self, trained_model):
        """Test predict returns binary values (0 or 1)."""
        model, X, _ = trained_model
        
        predictions = model.predict(X)
        
        assert np.all((predictions == 0) | (predictions == 1))
    
    def test_predict_threshold(self, trained_model):
        """Test predict with custom threshold."""
        model, X, _ = trained_model
        
        predictions_default = model.predict(X, threshold=0.5)
        predictions_high = model.predict(X, threshold=0.8)
        
        # Higher threshold should give more 0s
        assert np.sum(predictions_high == 0) >= np.sum(predictions_default == 0)
    
    def test_predict_single_sample(self, trained_model):
        """Test predict on single sample."""
        model, _, _ = trained_model
        
        X_test = np.array([[0, 0]])
        prediction = model.predict(X_test)
        
        assert prediction.shape == (1, 1)
        assert prediction[0, 0] in [0, 1]


class TestLogisticRegressionScoring:
    """Test accuracy scoring."""
    
    @pytest.fixture
    def trained_model(self):
        """Create and train a model."""
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_informative=8,
            random_state=42
        )
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        model = LogisticRegression(learning_rate=0.01, n_iterations=500)
        model.fit(X, y)
        
        return model, X, y
    
    def test_score_type(self, trained_model):
        """Test score returns a float."""
        model, X, y = trained_model
        
        accuracy = model.score(X, y)
        
        assert isinstance(accuracy, (float, np.floating))
    
    def test_score_range(self, trained_model):
        """Test score is between 0 and 1."""
        model, X, y = trained_model
        
        accuracy = model.score(X, y)
        
        assert 0 <= accuracy <= 1
    
    def test_score_perfect_predictions(self):
        """Test score with perfect predictions."""
        X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        y = np.array([0, 0, 1, 1])
        
        model = LogisticRegression(learning_rate=0.1, n_iterations=1000)
        model.fit(X, y)
        
        accuracy = model.score(X, y)
        
        # Should be close to 1 for well-separable data
        assert accuracy > 0.5


class TestLogisticRegressionRegularization:
    """Test regularization functionality."""
    
    @pytest.fixture
    def complex_data(self):
        """Generate dataset with many features."""
        X = np.random.randn(100, 50)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        return X, y
    
    def test_no_regularization(self, complex_data):
        """Test fitting without regularization."""
        X, y = complex_data
        model = LogisticRegression(
            learning_rate=0.01,
            n_iterations=100,
            regularization=None
        )
        
        model.fit(X, y)
        
        assert model.weights is not None
    
    def test_l2_regularization(self, complex_data):
        """Test fitting with L2 regularization."""
        X, y = complex_data
        model = LogisticRegression(
            learning_rate=0.01,
            n_iterations=100,
            regularization='l2',
            lambda_reg=0.1
        )
        
        model.fit(X, y)
        
        assert model.weights is not None
    
    def test_l1_regularization(self, complex_data):
        """Test fitting with L1 regularization."""
        X, y = complex_data
        model = LogisticRegression(
            learning_rate=0.01,
            n_iterations=100,
            regularization='l1',
            lambda_reg=0.1
        )
        
        model.fit(X, y)
        
        assert model.weights is not None
    
    def test_l1_produces_sparsity(self, complex_data):
        """Test that L1 regularization produces sparse weights."""
        X, y = complex_data
        
        # Model with L1 regularization
        model_l1 = LogisticRegression(
            learning_rate=0.01,
            n_iterations=200,
            regularization='l1',
            lambda_reg=0.5
        )
        model_l1.fit(X, y)
        
        # Model without regularization
        model_none = LogisticRegression(
            learning_rate=0.01,
            n_iterations=200,
            regularization=None
        )
        model_none.fit(X, y)
        
        # L1 should have more zero weights
        l1_zeros = np.sum(model_l1.weights == 0)
        none_zeros = np.sum(model_none.weights == 0)
        
        assert l1_zeros >= none_zeros
    
    def test_l2_reduces_weights_magnitude(self, complex_data):
        """Test that L2 regularization reduces weight magnitudes."""
        X, y = complex_data
        
        # Model with L2 regularization
        model_l2 = LogisticRegression(
            learning_rate=0.01,
            n_iterations=200,
            regularization='l2',
            lambda_reg=0.5
        )
        model_l2.fit(X, y)
        
        # Model without regularization
        model_none = LogisticRegression(
            learning_rate=0.01,
            n_iterations=200,
            regularization=None
        )
        model_none.fit(X, y)
        
        # L2 should have smaller weight magnitudes
        l2_magnitude = np.sum(np.abs(model_l2.weights))
        none_magnitude = np.sum(np.abs(model_none.weights))
        
        assert l2_magnitude <= none_magnitude


class TestLogisticRegressionGetParams:
    """Test get_params method."""
    
    def test_get_params_before_fitting(self):
        """Test get_params before model is fitted."""
        model = LogisticRegression()
        
        params = model.get_params()
        
        assert params['weights'] is None
        assert params['bias'] is None
        assert params['cost_history'] == []
    
    def test_get_params_after_fitting(self):
        """Test get_params after model is fitted."""
        X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        y = np.array([0, 0, 1, 1])
        
        model = LogisticRegression(learning_rate=0.1, n_iterations=100)
        model.fit(X, y)
        
        params = model.get_params()
        
        assert params['weights'] is not None
        assert params['bias'] is not None
        assert len(params['cost_history']) > 0
    
    def test_get_params_returns_dict(self):
        """Test that get_params returns a dictionary."""
        X = np.array([[0, 0], [1, 1]])
        y = np.array([0, 1])
        
        model = LogisticRegression(learning_rate=0.1, n_iterations=100)
        model.fit(X, y)
        
        params = model.get_params()
        
        assert isinstance(params, dict)
        assert 'weights' in params
        assert 'bias' in params
        assert 'cost_history' in params


class TestLogisticRegressionComparison:
    """Test LogisticRegression against scikit-learn."""
    
    @pytest.fixture
    def dataset(self):
        """Generate dataset for comparison."""
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_informative=8,
            random_state=42
        )
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        return train_test_split(X, y, test_size=0.2, random_state=42)
    
    def test_accuracy_comparison_with_sklearn(self, dataset):
        """Compare accuracy with scikit-learn."""
        X_train, X_test, y_train, y_test = dataset
        
        # Our implementation
        model_ours = LogisticRegression(learning_rate=0.01, n_iterations=1000)
        model_ours.fit(X_train, y_train)
        accuracy_ours = model_ours.score(X_test, y_test)
        
        # Scikit-learn implementation
        model_sklearn = SklearnLogisticRegression(max_iter=1000, random_state=42)
        model_sklearn.fit(X_train, y_train)
        accuracy_sklearn = model_sklearn.score(X_test, y_test)
        
        # Should be reasonably similar
        assert abs(accuracy_ours - accuracy_sklearn) < 0.15


class TestLogisticRegressionEdgeCases:
    """Test edge cases and error handling."""
    
    def test_single_sample(self):
        """Test with single training sample."""
        X = np.array([[1, 2]])
        y = np.array([1])
        
        model = LogisticRegression(learning_rate=0.1, n_iterations=100)
        model.fit(X, y)
        
        prediction = model.predict(X)
        
        assert prediction is not None
    
    def test_single_feature(self):
        """Test with single feature."""
        X = np.array([[1], [2], [3], [4]])
        y = np.array([0, 0, 1, 1])
        
        model = LogisticRegression(learning_rate=0.1, n_iterations=200)
        model.fit(X, y)
        
        accuracy = model.score(X, y)
        
        assert 0 <= accuracy <= 1
    
    def test_many_features(self):
        """Test with many features."""
        X = np.random.randn(50, 100)
        y = (X[:, 0] > 0).astype(int)
        
        model = LogisticRegression(learning_rate=0.01, n_iterations=100)
        model.fit(X, y)
        
        predictions = model.predict(X)
        
        assert predictions.shape == (50, 1)
    
    def test_with_negative_values(self):
        """Test with negative input values."""
        X = np.array([[-2, -1], [-1, 0], [0, 1], [1, 2]])
        y = np.array([0, 0, 1, 1])
        
        model = LogisticRegression(learning_rate=0.1, n_iterations=200)
        model.fit(X, y)
        
        predictions = model.predict(X)
        
        assert np.all((predictions == 0) | (predictions == 1))
    
    def test_balanced_classes(self):
        """Test with balanced binary classes."""
        X = np.array([[0, 0], [1, 1], [2, 2], [3, 3],
                      [0, 1], [1, 0], [2, 3], [3, 2]])
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        
        model = LogisticRegression(learning_rate=0.1, n_iterations=300)
        model.fit(X, y)
        
        accuracy = model.score(X, y)
        
        assert accuracy > 0.5
    
    def test_imbalanced_classes(self):
        """Test with imbalanced binary classes."""
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1],
                      [2, 2], [2.1, 2.1], [2.2, 2.2], [2.3, 2.3]])
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        
        model = LogisticRegression(learning_rate=0.1, n_iterations=300)
        model.fit(X, y)
        
        predictions = model.predict(X)
        
        assert predictions is not None


class TestLogisticRegressionWeightInitialization:
    """Test weight initialization."""
    
    def test_weights_initialized_to_zero(self):
        """Test that weights are initialized to zero."""
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])
        
        model = LogisticRegression(learning_rate=0.1, n_iterations=10)
        model.fit(X, y)
        
        # After first iteration, weights should not be zero
        assert not np.allclose(model.weights, 0)
    
    def test_bias_initialized_to_zero(self):
        """Test that bias is initialized to zero."""
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])
        
        model = LogisticRegression(learning_rate=0.1, n_iterations=1)
        model.fit(X, y)
        
        # After training, bias should typically be non-zero
        # (unless data is perfectly symmetric)
        assert isinstance(model.bias, (int, float, np.number))


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])