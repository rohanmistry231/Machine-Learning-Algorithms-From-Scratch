"""
Unit Tests for Lasso Regression Implementation

This module contains comprehensive tests for the LassoRegression class,
including tests for:
- Model initialization
- Feature normalization
- Model fitting (coordinate descent and subgradient methods)
- Predictions and scoring
- L1 regularization and feature selection
- Soft-thresholding operator
- Cost history tracking
- Comparison with scikit-learn
"""

import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso as SklearnLasso

# Import the LassoRegression model to test
from algorithms.supervised.regression import LassoRegression


class TestLassoRegressionInitialization:
    """Test LassoRegression initialization and parameters."""
    
    def test_default_initialization(self):
        """Test that LassoRegression initializes with default parameters."""
        model = LassoRegression()
        
        assert model.alpha == 1.0
        assert model.learning_rate == 0.01
        assert model.n_iterations == 1000
        assert model.method == 'coordinate_descent'
        assert model.tol == 1e-4
        assert model.weights is None
        assert model.bias is None
        assert model.cost_history == []
    
    def test_custom_initialization(self):
        """Test LassoRegression initialization with custom parameters."""
        model = LassoRegression(
            alpha=0.5,
            learning_rate=0.05,
            n_iterations=500,
            method='subgradient',
            tol=1e-3
        )
        
        assert model.alpha == 0.5
        assert model.learning_rate == 0.05
        assert model.n_iterations == 500
        assert model.method == 'subgradient'
        assert model.tol == 1e-3
    
    def test_coordinate_descent_method(self):
        """Test initialization with coordinate descent method."""
        model = LassoRegression(method='coordinate_descent')
        
        assert model.method == 'coordinate_descent'
    
    def test_subgradient_method(self):
        """Test initialization with subgradient method."""
        model = LassoRegression(method='subgradient')
        
        assert model.method == 'subgradient'
    
    def test_alpha_parameter(self):
        """Test different alpha (regularization) values."""
        alphas = [0.01, 0.1, 1.0, 10.0]
        
        for alpha in alphas:
            model = LassoRegression(alpha=alpha)
            assert model.alpha == alpha


class TestLassoRegressionSoftThreshold:
    """Test soft-thresholding operator."""
    
    def test_soft_threshold_positive(self):
        """Test soft-thresholding with positive value."""
        model = LassoRegression()
        
        result = model._soft_threshold(5.0, 2.0)
        
        assert result == 3.0
    
    def test_soft_threshold_negative(self):
        """Test soft-thresholding with negative value."""
        model = LassoRegression()
        
        result = model._soft_threshold(-5.0, 2.0)
        
        assert result == -3.0
    
    def test_soft_threshold_zero(self):
        """Test soft-thresholding shrinks small values to zero."""
        model = LassoRegression()
        
        result = model._soft_threshold(0.5, 2.0)
        
        assert result == 0
    
    def test_soft_threshold_boundary_positive(self):
        """Test soft-thresholding at positive boundary."""
        model = LassoRegression()
        
        result = model._soft_threshold(2.0, 2.0)
        
        assert result == 0
    
    def test_soft_threshold_boundary_negative(self):
        """Test soft-thresholding at negative boundary."""
        model = LassoRegression()
        
        result = model._soft_threshold(-2.0, 2.0)
        
        assert result == 0
    
    def test_soft_threshold_large_values(self):
        """Test soft-thresholding with large values."""
        model = LassoRegression()
        
        result_pos = model._soft_threshold(100.0, 5.0)
        result_neg = model._soft_threshold(-100.0, 5.0)
        
        assert result_pos == 95.0
        assert result_neg == -95.0


class TestLassoRegressionFitting:
    """Test LassoRegression fitting functionality."""
    
    @pytest.fixture
    def simple_data(self):
        """Generate simple regression data."""
        X = np.array([[1, 0], [2, 0], [3, 0], [4, 0], [5, 0]])
        y = np.array([2, 4, 6, 8, 10])
        return X, y
    
    @pytest.fixture
    def complex_data(self):
        """Generate complex dataset using sklearn."""
        X, y = make_regression(
            n_samples=100,
            n_features=20,
            n_informative=5,
            noise=10,
            random_state=42
        )
        return X, y
    
    def test_coordinate_descent_fitting(self, simple_data):
        """Test fitting using coordinate descent method."""
        X, y = simple_data
        model = LassoRegression(
            alpha=0.1,
            n_iterations=100,
            method='coordinate_descent'
        )
        
        result = model.fit(X, y)
        
        # Check that fit returns self
        assert result is model
        
        # Check that weights and bias are set
        assert model.weights is not None
        assert model.bias is not None
        
        # Check shapes
        assert model.weights.shape == (2, 1)
        assert isinstance(model.bias, (int, float, np.number))
    
    def test_subgradient_fitting(self, simple_data):
        """Test fitting using subgradient method."""
        X, y = simple_data
        model = LassoRegression(
            alpha=0.1,
            learning_rate=0.01,
            n_iterations=100,
            method='subgradient'
        )
        
        result = model.fit(X, y)
        
        # Check that fit returns self
        assert result is model
        
        # Check that weights and bias are set
        assert model.weights is not None
        assert model.bias is not None
    
    def test_invalid_method_raises_error(self):
        """Test that invalid method raises ValueError."""
        model = LassoRegression(method='invalid_method')
        
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([1, 2, 3])
        
        with pytest.raises(ValueError, match="Unknown method"):
            model.fit(X, y)
    
    def test_cost_history_recorded(self, simple_data):
        """Test that cost history is recorded during fitting."""
        X, y = simple_data
        n_iterations = 50
        model = LassoRegression(
            alpha=0.1,
            n_iterations=n_iterations,
            method='coordinate_descent'
        )
        
        model.fit(X, y)
        
        # Cost history should have entries
        assert len(model.cost_history) > 0
        assert len(model.cost_history) <= n_iterations
    
    def test_cost_generally_decreasing(self, simple_data):
        """Test that cost generally decreases during training."""
        X, y = simple_data
        model = LassoRegression(
            alpha=0.01,
            n_iterations=200,
            method='coordinate_descent'
        )
        
        model.fit(X, y)
        
        # Final cost should be less than initial cost
        assert model.cost_history[-1] < model.cost_history[0]
    
    def test_convergence_early_stopping(self, simple_data):
        """Test early stopping when converged."""
        X, y = simple_data
        model = LassoRegression(
            alpha=0.1,
            n_iterations=10000,
            method='coordinate_descent',
            tol=0.0001
        )
        
        model.fit(X, y)
        
        # Should stop before max iterations due to convergence
        assert len(model.cost_history) < 10000
    
    def test_feature_normalization_applied(self, complex_data):
        """Test that features are normalized internally."""
        X, y = complex_data
        model = LassoRegression(alpha=0.1, n_iterations=50)
        
        model.fit(X, y)
        
        # Check that normalization parameters are stored
        assert hasattr(model, 'X_mean')
        assert hasattr(model, 'X_std')
        assert model.X_mean.shape == (X.shape[1],)
        assert model.X_std.shape == (X.shape[1],)


class TestLassoRegressionFeatureSelection:
    """Test Lasso's feature selection capability (sparsity)."""
    
    def test_high_alpha_produces_sparsity(self):
        """Test that high alpha produces sparse weights (many zeros)."""
        X, y = make_regression(
            n_samples=100,
            n_features=50,
            n_informative=10,
            noise=5,
            random_state=42
        )
        
        # High regularization
        model_high = LassoRegression(alpha=1.0, n_iterations=200)
        model_high.fit(X, y)
        
        # Low regularization
        model_low = LassoRegression(alpha=0.01, n_iterations=200)
        model_low.fit(X, y)
        
        # High alpha should have more zero weights
        high_zeros = np.sum(np.abs(model_high.weights) < 1e-6)
        low_zeros = np.sum(np.abs(model_low.weights) < 1e-6)
        
        assert high_zeros > low_zeros
    
    def test_zero_weights_are_sparse(self):
        """Test that Lasso produces some zero coefficients."""
        X, y = make_regression(
            n_samples=100,
            n_features=30,
            n_informative=10,
            noise=5,
            random_state=42
        )
        
        model = LassoRegression(alpha=0.5, n_iterations=200)
        model.fit(X, y)
        
        # Should have at least some zero weights
        n_zeros = np.sum(np.abs(model.weights) < 1e-6)
        
        assert n_zeros > 0
    
    def test_feature_selection_comparison(self):
        """Compare feature selection between different alpha values."""
        X, y = make_regression(
            n_samples=100,
            n_features=50,
            n_informative=5,
            noise=5,
            random_state=42
        )
        
        alphas = [0.01, 0.1, 1.0, 10.0]
        n_nonzero_list = []
        
        for alpha in alphas:
            model = LassoRegression(alpha=alpha, n_iterations=200)
            model.fit(X, y)
            n_nonzero = np.sum(np.abs(model.weights) > 1e-6)
            n_nonzero_list.append(n_nonzero)
        
        # Higher alpha should result in fewer non-zero coefficients
        assert n_nonzero_list[0] >= n_nonzero_list[1]
        assert n_nonzero_list[1] >= n_nonzero_list[2]


class TestLassoRegressionPrediction:
    """Test LassoRegression prediction functionality."""
    
    @pytest.fixture
    def trained_model(self):
        """Create and train a model."""
        X = np.array([[1, 0], [2, 0], [3, 0], [4, 0], [5, 0]])
        y = np.array([2, 4, 6, 8, 10])
        
        model = LassoRegression(alpha=0.1, n_iterations=100)
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
        
        X_test = np.array([[2.5, 0]])
        predictions = model.predict(X_test)
        
        assert predictions.shape == (1, 1)
        assert isinstance(predictions[0, 0], (float, np.floating))
    
    def test_multiple_samples_prediction(self, trained_model):
        """Test prediction on multiple samples."""
        model, _, _ = trained_model
        
        X_test = np.array([[1.5, 0], [2.5, 0], [3.5, 0]])
        predictions = model.predict(X_test)
        
        assert predictions.shape == (3, 1)
    
    def test_prediction_with_list_input(self, trained_model):
        """Test that prediction works with list input."""
        model, _, _ = trained_model
        
        X_test = [[1, 0], [2, 0], [3, 0]]
        predictions = model.predict(X_test)
        
        assert predictions.shape == (3, 1)


class TestLassoRegressionScoring:
    """Test LassoRegression R² scoring."""
    
    @pytest.fixture
    def model_and_data(self):
        """Create model and test data."""
        X = np.array([[1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0]])
        y = np.array([2, 4, 6, 8, 10, 12])
        
        model = LassoRegression(alpha=0.01, n_iterations=200)
        model.fit(X, y)
        
        return model, X, y
    
    def test_r2_score_type(self, model_and_data):
        """Test that score returns a float."""
        model, X, y = model_and_data
        
        r2 = model.score(X, y)
        
        assert isinstance(r2, (float, np.floating))
    
    def test_r2_score_reasonable(self, model_and_data):
        """Test R² score on training data is reasonable."""
        model, X, y = model_and_data
        
        r2 = model.score(X, y)
        
        # Should be positive for well-fitting data
        assert r2 > 0
    
    def test_score_with_1d_array(self, model_and_data):
        """Test scoring with 1D target array."""
        model, X, _ = model_and_data
        y = np.array([2, 4, 6, 8, 10, 12])
        
        r2 = model.score(X, y)
        
        assert isinstance(r2, (float, np.floating))


class TestLassoRegressionGetParams:
    """Test get_params method."""
    
    def test_get_params_before_fitting(self):
        """Test get_params before model is fitted."""
        model = LassoRegression()
        
        params = model.get_params()
        
        assert params['weights'] is None
        assert params['bias'] is None
        assert params['alpha'] == 1.0
        assert params['cost_history'] == []
    
    def test_get_params_after_fitting(self):
        """Test get_params after model is fitted."""
        X = np.array([[1, 0], [2, 0], [3, 0]])
        y = np.array([2, 4, 6])
        
        model = LassoRegression(alpha=0.1, n_iterations=100)
        model.fit(X, y)
        
        params = model.get_params()
        
        assert params['weights'] is not None
        assert params['bias'] is not None
        assert params['alpha'] == 0.1
        assert len(params['cost_history']) > 0
    
    def test_get_params_nonzero_weights(self):
        """Test that get_params reports non-zero weights."""
        X, y = make_regression(
            n_samples=50,
            n_features=30,
            n_informative=5,
            random_state=42
        )
        
        model = LassoRegression(alpha=0.5, n_iterations=200)
        model.fit(X, y)
        
        params = model.get_params()
        
        assert 'n_nonzero_weights' in params
        assert params['n_nonzero_weights'] > 0


class TestLassoRegressionComparison:
    """Test LassoRegression against scikit-learn."""
    
    @pytest.fixture
    def dataset(self):
        """Generate dataset for comparison."""
        X, y = make_regression(
            n_samples=100,
            n_features=20,
            n_informative=10,
            noise=5,
            random_state=42
        )
        return train_test_split(X, y, test_size=0.2, random_state=42)
    
    def test_r2_score_comparison_with_sklearn(self, dataset):
        """Compare R² score with scikit-learn."""
        X_train, X_test, y_train, y_test = dataset
        
        # Our implementation
        model_ours = LassoRegression(alpha=0.1, n_iterations=500)
        model_ours.fit(X_train, y_train)
        r2_ours = model_ours.score(X_test, y_test)
        
        # Scikit-learn implementation
        model_sklearn = SklearnLasso(alpha=0.1, max_iter=1000)
        model_sklearn.fit(X_train, y_train)
        r2_sklearn = model_sklearn.score(X_test, y_test)
        
        # Should be reasonably similar
        assert abs(r2_ours - r2_sklearn) < 0.2


class TestLassoRegressionEdgeCases:
    """Test edge cases and error handling."""
    
    def test_single_sample(self):
        """Test with single training sample."""
        X = np.array([[1, 2]])
        y = np.array([3])
        
        model = LassoRegression(alpha=0.1, n_iterations=100)
        model.fit(X, y)
        
        prediction = model.predict(X)
        
        assert prediction is not None
    
    def test_single_feature(self):
        """Test with single feature."""
        X = np.array([[1], [2], [3], [4]])
        y = np.array([2, 4, 6, 8])
        
        model = LassoRegression(alpha=0.1, n_iterations=200)
        model.fit(X, y)
        
        r2 = model.score(X, y)
        
        assert r2 > 0
    
    def test_many_features(self):
        """Test with many features."""
        X = np.random.randn(50, 100)
        y = X[:, :5].sum(axis=1)
        
        model = LassoRegression(alpha=0.1, n_iterations=100)
        model.fit(X, y)
        
        prediction = model.predict(X)
        
        assert prediction.shape == (50, 1)
    
    def test_with_negative_values(self):
        """Test with negative input and output values."""
        X = np.array([[-2, -1], [-1, 0], [0, 1], [1, 2]])
        y = np.array([-4, -2, 0, 2])
        
        model = LassoRegression(alpha=0.1, n_iterations=200)
        model.fit(X, y)
        
        r2 = model.score(X, y)
        
        assert r2 > 0
    
    def test_with_large_values(self):
        """Test with large numerical values."""
        X = np.array([[1e6, 0], [2e6, 0], [3e6, 0]])
        y = np.array([2e6, 4e6, 6e6])
        
        model = LassoRegression(alpha=0.1, n_iterations=100)
        model.fit(X, y)
        
        prediction = model.predict(X)
        
        assert prediction is not None


class TestLassoRegressionCostHistory:
    """Test cost history tracking."""
    
    def test_cost_history_length(self):
        """Test that cost history is recorded."""
        X = np.array([[1, 0], [2, 0], [3, 0]])
        y = np.array([2, 4, 6])
        
        n_iterations = 50
        model = LassoRegression(
            alpha=0.1,
            n_iterations=n_iterations,
            method='coordinate_descent'
        )
        model.fit(X, y)
        
        # Cost history should have entries
        assert len(model.cost_history) > 0
    
    def test_cost_history_type(self):
        """Test that cost history contains floats."""
        X = np.array([[1, 0], [2, 0], [3, 0]])
        y = np.array([2, 4, 6])
        
        model = LassoRegression(alpha=0.1, n_iterations=50)
        model.fit(X, y)
        
        assert all(isinstance(cost, (float, np.floating)) 
                  for cost in model.cost_history)


class TestLassoRegressionAlphaEffect:
    """Test the effect of alpha parameter on model behavior."""
    
    def test_alpha_affects_weights_magnitude(self):
        """Test that higher alpha reduces weight magnitudes."""
        X, y = make_regression(
            n_samples=100,
            n_features=20,
            n_informative=10,
            random_state=42
        )
        
        # Low regularization
        model_low = LassoRegression(alpha=0.01, n_iterations=200)
        model_low.fit(X, y)
        
        # High regularization
        model_high = LassoRegression(alpha=1.0, n_iterations=200)
        model_high.fit(X, y)
        
        # Higher alpha should have smaller weights
        low_magnitude = np.sum(np.abs(model_low.weights))
        high_magnitude = np.sum(np.abs(model_high.weights))
        
        assert high_magnitude <= low_magnitude
    
    def test_different_alpha_values(self):
        """Test different alpha values produce different models."""
        X, y = make_regression(
            n_samples=100,
            n_features=10,
            n_informative=5,
            random_state=42
        )
        
        model1 = LassoRegression(alpha=0.1, n_iterations=100)
        model1.fit(X, y)
        
        model2 = LassoRegression(alpha=1.0, n_iterations=100)
        model2.fit(X, y)
        
        # Weights should be different
        assert not np.allclose(model1.weights, model2.weights)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])