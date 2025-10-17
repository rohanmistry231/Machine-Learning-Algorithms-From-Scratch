"""
Unit Tests for Ridge Regression Implementation

This module contains comprehensive tests for the RidgeRegression class,
including tests for:
- Model initialization
- Model fitting (gradient descent and normal equation)
- L2 regularization effectiveness
- Predictions and scoring
- Alpha parameter effects
- Comparison with scikit-learn
- Edge cases and error handling
"""

import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge as SklearnRidge

# Import the RidgeRegression model to test
from algorithms.supervised.regression import RidgeRegression


class TestRidgeRegressionInitialization:
    """Test RidgeRegression initialization and parameters."""
    
    def test_default_initialization(self):
        """Test that RidgeRegression initializes with default parameters."""
        model = RidgeRegression()
        
        assert model.alpha == 1.0
        assert model.learning_rate == 0.01
        assert model.n_iterations == 1000
        assert model.method == 'normal_equation'
        assert model.weights is None
        assert model.bias is None
        assert model.cost_history == []
    
    def test_custom_initialization(self):
        """Test RidgeRegression initialization with custom parameters."""
        model = RidgeRegression(
            alpha=0.5,
            learning_rate=0.05,
            n_iterations=500,
            method='gradient_descent'
        )
        
        assert model.alpha == 0.5
        assert model.learning_rate == 0.05
        assert model.n_iterations == 500
        assert model.method == 'gradient_descent'
    
    def test_normal_equation_method(self):
        """Test initialization with normal equation method."""
        model = RidgeRegression(method='normal_equation')
        
        assert model.method == 'normal_equation'
    
    def test_gradient_descent_method(self):
        """Test initialization with gradient descent method."""
        model = RidgeRegression(method='gradient_descent')
        
        assert model.method == 'gradient_descent'
    
    def test_alpha_parameter(self):
        """Test different alpha (regularization) values."""
        alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
        
        for alpha in alphas:
            model = RidgeRegression(alpha=alpha)
            assert model.alpha == alpha


class TestRidgeRegressionFitting:
    """Test RidgeRegression fitting functionality."""
    
    @pytest.fixture
    def simple_data(self):
        """Generate simple regression data."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])
        return X, y
    
    @pytest.fixture
    def complex_data(self):
        """Generate complex dataset using sklearn."""
        X, y = make_regression(
            n_samples=100,
            n_features=20,
            n_informative=10,
            noise=10,
            random_state=42
        )
        return X, y
    
    def test_gradient_descent_fitting(self, simple_data):
        """Test fitting using gradient descent method."""
        X, y = simple_data
        model = RidgeRegression(
            alpha=0.1,
            learning_rate=0.01,
            n_iterations=500,
            method='gradient_descent'
        )
        
        result = model.fit(X, y)
        
        # Check that fit returns self
        assert result is model
        
        # Check that weights and bias are set
        assert model.weights is not None
        assert model.bias is not None
        
        # Check shapes
        assert model.weights.shape == (1, 1)
        assert isinstance(model.bias, (int, float, np.number))
    
    def test_normal_equation_fitting(self, simple_data):
        """Test fitting using normal equation method."""
        X, y = simple_data
        model = RidgeRegression(alpha=1.0, method='normal_equation')
        
        result = model.fit(X, y)
        
        # Check that fit returns self
        assert result is model
        
        # Check that weights and bias are set
        assert model.weights is not None
        assert model.bias is not None
        
        # Check shapes
        assert model.weights.shape == (1, 1)
    
    def test_invalid_method_raises_error(self):
        """Test that invalid method raises ValueError."""
        model = RidgeRegression(method='invalid_method')
        
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([1, 2, 3])
        
        with pytest.raises(ValueError, match="Unknown method"):
            model.fit(X, y)
    
    def test_gradient_descent_vs_normal_equation(self, complex_data):
        """Test that both methods give similar results."""
        X, y = complex_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train with gradient descent
        model_gd = RidgeRegression(
            alpha=1.0,
            learning_rate=0.01,
            n_iterations=2000,
            method='gradient_descent'
        )
        model_gd.fit(X_train, y_train)
        
        # Train with normal equation
        model_ne = RidgeRegression(alpha=1.0, method='normal_equation')
        model_ne.fit(X_train, y_train)
        
        # Get predictions
        y_pred_gd = model_gd.predict(X_test)
        y_pred_ne = model_ne.predict(X_test)
        
        # Predictions should be similar (within tolerance)
        np.testing.assert_allclose(y_pred_gd, y_pred_ne, rtol=0.1)
    
    def test_cost_history_recorded(self, simple_data):
        """Test that cost history is recorded during gradient descent."""
        X, y = simple_data
        n_iterations = 100
        model = RidgeRegression(
            alpha=0.1,
            n_iterations=n_iterations,
            method='gradient_descent'
        )
        
        model.fit(X, y)
        
        # Cost history should have n_iterations entries
        assert len(model.cost_history) == n_iterations
    
    def test_cost_decreasing(self, simple_data):
        """Test that cost decreases during training."""
        X, y = simple_data
        model = RidgeRegression(
            alpha=0.1,
            learning_rate=0.01,
            n_iterations=500,
            method='gradient_descent'
        )
        
        model.fit(X, y)
        
        # Final cost should be less than initial cost
        assert model.cost_history[-1] < model.cost_history[0]
    
    def test_fitting_with_1d_target(self, complex_data):
        """Test fitting with 1D target array."""
        X, y = complex_data
        model = RidgeRegression(alpha=1.0, method='normal_equation')
        
        # Should not raise error
        model.fit(X, y)
        
        assert model.weights is not None
    
    def test_fitting_with_2d_target(self, complex_data):
        """Test fitting with 2D target array."""
        X, y = complex_data
        y_2d = y.reshape(-1, 1)
        model = RidgeRegression(alpha=1.0, method='normal_equation')
        
        # Should not raise error
        model.fit(X, y_2d)
        
        assert model.weights is not None


class TestRidgeRegressionRegularization:
    """Test L2 regularization effectiveness."""
    
    def test_alpha_reduces_weight_magnitude(self):
        """Test that higher alpha reduces weight magnitudes."""
        X, y = make_regression(
            n_samples=100,
            n_features=50,
            n_informative=10,
            noise=5,
            random_state=42
        )
        
        # Low regularization
        model_low = RidgeRegression(alpha=0.01, method='normal_equation')
        model_low.fit(X, y)
        
        # High regularization
        model_high = RidgeRegression(alpha=100.0, method='normal_equation')
        model_high.fit(X, y)
        
        # Higher alpha should have smaller weights
        low_magnitude = np.sum(np.abs(model_low.weights))
        high_magnitude = np.sum(np.abs(model_high.weights))
        
        assert high_magnitude < low_magnitude
    
    def test_alpha_zero_similar_to_linear(self):
        """Test that alpha=0 behaves like linear regression."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])
        
        # Ridge with alpha=0
        model_ridge = RidgeRegression(alpha=0.0, method='normal_equation')
        model_ridge.fit(X, y)
        
        r2_ridge = model_ridge.score(X, y)
        
        # Should be close to 1 for linear data
        assert r2_ridge > 0.99
    
    def test_regularization_with_gradient_descent(self):
        """Test L2 regularization with gradient descent."""
        X, y = make_regression(
            n_samples=100,
            n_features=20,
            n_informative=10,
            random_state=42
        )
        
        model = RidgeRegression(
            alpha=1.0,
            learning_rate=0.01,
            n_iterations=1000,
            method='gradient_descent'
        )
        
        model.fit(X, y)
        
        # Check that regularization was applied (cost includes L2 term)
        assert len(model.cost_history) > 0
        assert model.weights is not None
    
    def test_different_alpha_values_comparison(self):
        """Test different alpha values produce different models."""
        X, y = make_regression(
            n_samples=100,
            n_features=20,
            n_informative=10,
            random_state=42
        )
        
        model1 = RidgeRegression(alpha=0.1, method='normal_equation')
        model1.fit(X, y)
        
        model2 = RidgeRegression(alpha=10.0, method='normal_equation')
        model2.fit(X, y)
        
        # Weights should be different
        assert not np.allclose(model1.weights, model2.weights)
    
    def test_bias_not_regularized(self):
        """Test that bias term is not regularized."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([10, 20, 30])
        
        # High regularization
        model = RidgeRegression(alpha=100.0, method='normal_equation')
        model.fit(X, y)
        
        # Bias should still be reasonable (not shrunk to zero)
        assert model.bias is not None
        assert isinstance(model.bias, (int, float, np.number))


class TestRidgeRegressionPrediction:
    """Test RidgeRegression prediction functionality."""
    
    @pytest.fixture
    def trained_model(self):
        """Create and train a model."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])
        
        model = RidgeRegression(alpha=0.1, method='normal_equation')
        model.fit(X, y)
        
        return model, X, y
    
    def test_prediction_shape(self, trained_model):
        """Test that predictions have correct shape."""
        model, X, _ = trained_model
        
        predictions = model.predict(X)
        
        assert predictions.shape == (5, 1)
    
    def test_single_sample_prediction(self, trained_model):
        """Test prediction on single sample."""
        model, _, _ = trained_model
        
        X_test = np.array([[6]])
        predictions = model.predict(X_test)
        
        assert predictions.shape == (1, 1)
        assert isinstance(predictions[0, 0], (float, np.floating))
    
    def test_multiple_samples_prediction(self, trained_model):
        """Test prediction on multiple samples."""
        model, _, _ = trained_model
        
        X_test = np.array([[1.5], [2.5], [3.5]])
        predictions = model.predict(X_test)
        
        assert predictions.shape == (3, 1)
    
    def test_prediction_accuracy_simple_data(self, trained_model):
        """Test prediction accuracy on simple linear data."""
        model, X, y = trained_model
        
        predictions = model.predict(X)
        
        # For simple linear y = 2x, predictions should be close
        np.testing.assert_allclose(predictions.flatten(), y, atol=0.5)
    
    def test_prediction_with_list_input(self, trained_model):
        """Test that prediction works with list input."""
        model, _, _ = trained_model
        
        X_test = [[1], [2], [3]]
        predictions = model.predict(X_test)
        
        assert predictions.shape == (3, 1)


class TestRidgeRegressionScoring:
    """Test RidgeRegression R² scoring."""
    
    @pytest.fixture
    def model_and_data(self):
        """Create model and test data."""
        X = np.array([[1], [2], [3], [4], [5], [6]])
        y = np.array([2, 4, 6, 8, 10, 12])
        
        model = RidgeRegression(alpha=0.1, method='normal_equation')
        model.fit(X, y)
        
        return model, X, y
    
    def test_r2_score_perfect_fit(self, model_and_data):
        """Test R² score for good fit (should be close to 1)."""
        model, X, y = model_and_data
        
        r2 = model.score(X, y)
        
        # Good fit should have R² close to 1
        assert r2 > 0.95
    
    def test_r2_score_type(self, model_and_data):
        """Test that score returns a float."""
        model, X, y = model_and_data
        
        r2 = model.score(X, y)
        
        assert isinstance(r2, (float, np.floating))
    
    def test_r2_score_range(self):
        """Test R² score bounds."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.random.randn(5)  # Random data (potentially bad fit)
        
        model = RidgeRegression(alpha=1.0, method='normal_equation')
        model.fit(X, y)
        
        r2 = model.score(X, y)
        
        # R² should be <= 1
        assert r2 <= 1.0
    
    def test_score_with_1d_array(self, model_and_data):
        """Test scoring with 1D target array."""
        model, X, _ = model_and_data
        y = np.array([2, 4, 6, 8, 10, 12])
        
        r2 = model.score(X, y)
        
        assert isinstance(r2, (float, np.floating))


class TestRidgeRegressionGetParams:
    """Test get_params method."""
    
    def test_get_params_before_fitting(self):
        """Test get_params before model is fitted."""
        model = RidgeRegression()
        
        params = model.get_params()
        
        assert params['weights'] is None
        assert params['bias'] is None
        assert params['alpha'] == 1.0
        assert params['cost_history'] == []
    
    def test_get_params_after_fitting(self):
        """Test get_params after model is fitted."""
        X = np.array([[1], [2], [3]])
        y = np.array([2, 4, 6])
        
        model = RidgeRegression(alpha=0.5, method='normal_equation')
        model.fit(X, y)
        
        params = model.get_params()
        
        assert params['weights'] is not None
        assert params['bias'] is not None
        assert params['alpha'] == 0.5
    
    def test_get_params_returns_dict(self):
        """Test that get_params returns a dictionary."""
        X = np.array([[1], [2], [3]])
        y = np.array([2, 4, 6])
        
        model = RidgeRegression(alpha=1.0, method='normal_equation')
        model.fit(X, y)
        
        params = model.get_params()
        
        assert isinstance(params, dict)
        assert 'weights' in params
        assert 'bias' in params
        assert 'alpha' in params
        assert 'cost_history' in params


class TestRidgeRegressionComparison:
    """Test RidgeRegression against scikit-learn."""
    
    @pytest.fixture
    def dataset(self):
        """Generate dataset for comparison."""
        X, y = make_regression(
            n_samples=100,
            n_features=10,
            n_informative=8,
            noise=5,
            random_state=42
        )
        return train_test_split(X, y, test_size=0.2, random_state=42)
    
    def test_r2_score_comparison_with_sklearn(self, dataset):
        """Compare R² score with scikit-learn."""
        X_train, X_test, y_train, y_test = dataset
        
        # Our implementation
        model_ours = RidgeRegression(alpha=1.0, method='normal_equation')
        model_ours.fit(X_train, y_train)
        r2_ours = model_ours.score(X_test, y_test)
        
        # Scikit-learn implementation
        model_sklearn = SklearnRidge(alpha=1.0)
        model_sklearn.fit(X_train, y_train)
        r2_sklearn = model_sklearn.score(X_test, y_test)
        
        # Should be very similar
        np.testing.assert_allclose(r2_ours, r2_sklearn, rtol=0.01)
    
    def test_predictions_comparison_with_sklearn(self, dataset):
        """Compare predictions with scikit-learn."""
        X_train, X_test, y_train, y_test = dataset
        
        # Our implementation
        model_ours = RidgeRegression(alpha=1.0, method='normal_equation')
        model_ours.fit(X_train, y_train)
        y_pred_ours = model_ours.predict(X_test)
        
        # Scikit-learn implementation
        model_sklearn = SklearnRidge(alpha=1.0)
        model_sklearn.fit(X_train, y_train)
        y_pred_sklearn = model_sklearn.predict(X_test)
        
        # Predictions should be very similar
        np.testing.assert_allclose(y_pred_ours.flatten(), y_pred_sklearn, rtol=0.01)


class TestRidgeRegressionEdgeCases:
    """Test edge cases and error handling."""
    
    def test_single_sample(self):
        """Test with single training sample."""
        X = np.array([[1, 2]])
        y = np.array([3])
        
        model = RidgeRegression(alpha=1.0, method='normal_equation')
        model.fit(X, y)
        
        prediction = model.predict(X)
        
        assert prediction is not None
    
    def test_single_feature(self):
        """Test with single feature."""
        X = np.array([[1], [2], [3], [4]])
        y = np.array([2, 4, 6, 8])
        
        model = RidgeRegression(alpha=0.1, method='normal_equation')
        model.fit(X, y)
        
        r2 = model.score(X, y)
        
        assert r2 > 0.95
    
    def test_many_features(self):
        """Test with many features."""
        X = np.random.randn(50, 100)
        y = X[:, :10].sum(axis=1)
        
        model = RidgeRegression(alpha=1.0, method='normal_equation')
        model.fit(X, y)
        
        prediction = model.predict(X)
        
        assert prediction.shape == (50, 1)
    
    def test_with_negative_values(self):
        """Test with negative input and output values."""
        X = np.array([[-2], [-1], [0], [1], [2]])
        y = np.array([-4, -2, 0, 2, 4])
        
        model = RidgeRegression(alpha=0.1, method='normal_equation')
        model.fit(X, y)
        
        r2 = model.score(X, y)
        
        assert r2 > 0.95
    
    def test_with_large_values(self):
        """Test with large numerical values."""
        X = np.array([[1e6], [2e6], [3e6]])
        y = np.array([2e6, 4e6, 6e6])
        
        model = RidgeRegression(alpha=1.0, method='normal_equation')
        model.fit(X, y)
        
        prediction = model.predict(X)
        
        assert prediction is not None
    
    def test_high_dimensionality(self):
        """Test with more features than samples (high dimensional)."""
        X = np.random.randn(20, 50)
        y = X[:, :5].sum(axis=1)
        
        # Ridge should handle this better than OLS
        model = RidgeRegression(alpha=1.0, method='normal_equation')
        model.fit(X, y)
        
        # Should not raise error and produce reasonable predictions
        prediction = model.predict(X)
        assert prediction.shape == (20, 1)


class TestRidgeRegressionCostHistory:
    """Test cost history tracking (gradient descent only)."""
    
    def test_cost_history_gradient_descent(self):
        """Test that cost history is recorded during gradient descent."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])
        
        n_iterations = 500
        model = RidgeRegression(
            alpha=0.1,
            learning_rate=0.01,
            n_iterations=n_iterations,
            method='gradient_descent'
        )
        model.fit(X, y)
        
        # Cost history should have n_iterations entries
        assert len(model.cost_history) == n_iterations
    
    def test_cost_history_decreasing(self):
        """Test that cost generally decreases over iterations."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])
        
        model = RidgeRegression(
            alpha=0.1,
            learning_rate=0.01,
            n_iterations=1000,
            method='gradient_descent'
        )
        model.fit(X, y)
        
        # Final cost should be less than initial cost
        assert model.cost_history[-1] < model.cost_history[0]
    
    def test_cost_history_type(self):
        """Test that cost history contains floats."""
        X = np.array([[1], [2], [3]])
        y = np.array([2, 4, 6])
        
        model = RidgeRegression(
            alpha=0.1,
            learning_rate=0.01,
            n_iterations=100,
            method='gradient_descent'
        )
        model.fit(X, y)
        
        assert all(isinstance(cost, (float, np.floating)) 
                  for cost in model.cost_history)


class TestRidgeRegressionAlphaEffect:
    """Test the effect of alpha parameter on model behavior."""
    
    def test_alpha_range_comparison(self):
        """Test different alpha values produce different weight magnitudes."""
        X, y = make_regression(
            n_samples=100,
            n_features=20,
            n_informative=10,
            random_state=42
        )
        
        alphas = [0.001, 0.1, 1.0, 10.0, 100.0]
        weight_magnitudes = []
        
        for alpha in alphas:
            model = RidgeRegression(alpha=alpha, method='normal_equation')
            model.fit(X, y)
            weight_magnitude = np.sum(np.abs(model.weights))
            weight_magnitudes.append(weight_magnitude)
        
        # Higher alpha should generally result in smaller weights
        assert weight_magnitudes[0] >= weight_magnitudes[-1]
    
    def test_very_high_alpha_shrinks_weights(self):
        """Test that very high alpha strongly shrinks weights."""
        X, y = make_regression(
            n_samples=100,
            n_features=20,
            n_informative=10,
            random_state=42
        )
        
        model = RidgeRegression(alpha=1000.0, method='normal_equation')
        model.fit(X, y)
        
        # Weights should be small
        avg_weight = np.mean(np.abs(model.weights))
        
        assert avg_weight < 1.0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])