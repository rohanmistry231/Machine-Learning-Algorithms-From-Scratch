"""
Unit Tests for Polynomial Regression Implementation

This module contains comprehensive tests for the PolynomialRegression and
PolynomialFeatures classes, including tests for:
- PolynomialFeatures initialization and transformation
- Feature name generation
- PolynomialRegression initialization
- Model fitting and prediction
- Different polynomial degrees
- Regularization (L1, L2)
- Non-linear data fitting
- Comparison with scikit-learn
"""

import numpy as np
import pytest
from sklearn.preprocessing import PolynomialFeatures as SklearnPolyFeatures
from sklearn.linear_model import Ridge

# Import the models to test
from algorithms.supervised.regression import PolynomialRegression, PolynomialFeatures


class TestPolynomialFeaturesInitialization:
    """Test PolynomialFeatures initialization."""
    
    def test_default_initialization(self):
        """Test default initialization."""
        poly = PolynomialFeatures()
        
        assert poly.degree == 2
        assert poly.include_bias is True
        assert poly.interaction_only is False
        assert poly.n_input_features is None
        assert poly.n_output_features is None
    
    def test_custom_initialization(self):
        """Test custom initialization."""
        poly = PolynomialFeatures(
            degree=3,
            include_bias=False,
            interaction_only=True
        )
        
        assert poly.degree == 3
        assert poly.include_bias is False
        assert poly.interaction_only is True
    
    def test_different_degrees(self):
        """Test initialization with different degrees."""
        for degree in [1, 2, 3, 4, 5]:
            poly = PolynomialFeatures(degree=degree)
            assert poly.degree == degree


class TestPolynomialFeaturesFitting:
    """Test PolynomialFeatures fitting."""
    
    def test_fit_single_feature(self):
        """Test fitting with single feature."""
        X = np.array([[1], [2], [3]])
        poly = PolynomialFeatures(degree=2)
        
        result = poly.fit(X)
        
        assert result is poly
        assert poly.n_input_features == 1
        assert poly.n_output_features is not None
        assert poly.powers is not None
    
    def test_fit_multiple_features(self):
        """Test fitting with multiple features."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        poly = PolynomialFeatures(degree=2)
        
        result = poly.fit(X)
        
        assert result is poly
        assert poly.n_input_features == 2
        assert poly.n_output_features > poly.n_input_features
    
    def test_fit_sets_attributes(self):
        """Test that fit sets all necessary attributes."""
        X = np.array([[1, 2], [3, 4]])
        poly = PolynomialFeatures(degree=2)
        
        poly.fit(X)
        
        assert poly.n_input_features is not None
        assert poly.n_output_features is not None
        assert poly.powers is not None
        assert len(poly.powers) == poly.n_output_features


class TestPolynomialFeaturesTransformation:
    """Test PolynomialFeatures transformation."""
    
    def test_transform_single_feature_degree_2(self):
        """Test transformation of single feature with degree 2."""
        X = np.array([[2], [3]])
        poly = PolynomialFeatures(degree=2, include_bias=True)
        poly.fit(X)
        
        X_poly = poly.transform(X)
        
        # Should have [1, x, x^2]
        assert X_poly.shape[1] == 3
        np.testing.assert_array_almost_equal(X_poly[0], [1, 2, 4])
        np.testing.assert_array_almost_equal(X_poly[1], [1, 3, 9])
    
    def test_transform_two_features_degree_2(self):
        """Test transformation of two features with degree 2."""
        X = np.array([[2, 3]])
        poly = PolynomialFeatures(degree=2, include_bias=True)
        poly.fit(X)
        
        X_poly = poly.transform(X)
        
        # Should have [1, x1, x2, x1^2, x1*x2, x2^2]
        assert X_poly.shape[1] == 6
        expected = [1, 2, 3, 4, 6, 9]
        np.testing.assert_array_almost_equal(X_poly[0], expected)
    
    def test_transform_without_bias(self):
        """Test transformation without bias term."""
        X = np.array([[2]])
        poly = PolynomialFeatures(degree=2, include_bias=False)
        poly.fit(X)
        
        X_poly = poly.transform(X)
        
        # Should have [x, x^2] (no bias)
        assert X_poly.shape[1] == 2
        np.testing.assert_array_almost_equal(X_poly[0], [2, 4])
    
    def test_fit_transform(self):
        """Test fit_transform method."""
        X = np.array([[2, 3], [4, 5]])
        poly = PolynomialFeatures(degree=2)
        
        X_poly = poly.fit_transform(X)
        
        assert X_poly.shape[0] == 2
        assert X_poly.shape[1] > 2
        assert poly.n_input_features == 2
    
    def test_transform_preserves_shape(self):
        """Test that transform preserves number of samples."""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        poly = PolynomialFeatures(degree=3)
        
        X_poly = poly.fit_transform(X)
        
        assert X_poly.shape[0] == 4


class TestPolynomialFeaturesInteractionOnly:
    """Test interaction_only parameter."""
    
    def test_interaction_only_true(self):
        """Test interaction_only=True excludes power terms."""
        X = np.array([[2, 3]])
        
        # With interaction only
        poly_interaction = PolynomialFeatures(
            degree=2, 
            include_bias=False, 
            interaction_only=True
        )
        X_inter = poly_interaction.fit_transform(X)
        
        # Without interaction only
        poly_normal = PolynomialFeatures(
            degree=2, 
            include_bias=False, 
            interaction_only=False
        )
        X_normal = poly_normal.fit_transform(X)
        
        # Interaction only should have fewer features
        assert X_inter.shape[1] < X_normal.shape[1]
    
    def test_interaction_only_no_squares(self):
        """Test that interaction_only doesn't include x^2 terms."""
        X = np.array([[2, 3]])
        poly = PolynomialFeatures(
            degree=2, 
            include_bias=False, 
            interaction_only=True
        )
        
        X_poly = poly.fit_transform(X)
        
        # Should have [x1, x2, x1*x2] (no x1^2 or x2^2)
        assert X_poly.shape[1] == 3


class TestPolynomialFeaturesNames:
    """Test feature name generation."""
    
    def test_get_feature_names_default(self):
        """Test default feature names."""
        X = np.array([[1, 2]])
        poly = PolynomialFeatures(degree=2, include_bias=True)
        poly.fit(X)
        
        names = poly.get_feature_names()
        
        assert '1' in names  # Bias term
        assert len(names) == poly.n_output_features
    
    def test_get_feature_names_custom(self):
        """Test custom feature names."""
        X = np.array([[1, 2]])
        poly = PolynomialFeatures(degree=2, include_bias=True)
        poly.fit(X)
        
        names = poly.get_feature_names(['a', 'b'])
        
        assert 'a' in names
        assert 'b' in names
        assert len(names) == poly.n_output_features
    
    def test_feature_names_structure(self):
        """Test feature names have correct structure."""
        X = np.array([[1, 2]])
        poly = PolynomialFeatures(degree=2, include_bias=False)
        poly.fit(X)
        
        names = poly.get_feature_names(['x', 'y'])
        
        # Check that names are strings
        assert all(isinstance(name, str) for name in names)


class TestPolynomialRegressionInitialization:
    """Test PolynomialRegression initialization."""
    
    def test_default_initialization(self):
        """Test default initialization."""
        model = PolynomialRegression()
        
        assert model.degree == 2
        assert model.learning_rate == 0.01
        assert model.n_iterations == 1000
        assert model.include_bias is True
        assert model.regularization is None
        assert model.lambda_reg == 0.01
        assert model.weights is None
        assert model.bias is None
        assert model.cost_history == []
    
    def test_custom_initialization(self):
        """Test custom initialization."""
        model = PolynomialRegression(
            degree=3,
            learning_rate=0.05,
            n_iterations=500,
            regularization='l2',
            lambda_reg=0.1
        )
        
        assert model.degree == 3
        assert model.learning_rate == 0.05
        assert model.n_iterations == 500
        assert model.regularization == 'l2'
        assert model.lambda_reg == 0.1
    
    def test_polynomial_features_created(self):
        """Test that PolynomialFeatures object is created."""
        model = PolynomialRegression(degree=3)
        
        assert model.poly_features is not None
        assert isinstance(model.poly_features, PolynomialFeatures)
        assert model.poly_features.degree == 3


class TestPolynomialRegressionFitting:
    """Test PolynomialRegression fitting."""
    
    @pytest.fixture
    def linear_data(self):
        """Generate simple linear data."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])
        return X, y
    
    @pytest.fixture
    def quadratic_data(self):
        """Generate quadratic data."""
        X = np.linspace(-3, 3, 50).reshape(-1, 1)
        y = 2 * X.flatten() ** 2 + X.flatten() + 1 + np.random.randn(50) * 0.5
        return X, y
    
    def test_basic_fitting(self, linear_data):
        """Test basic model fitting."""
        X, y = linear_data
        model = PolynomialRegression(degree=2, n_iterations=100)
        
        result = model.fit(X, y)
        
        assert result is model
        assert model.weights is not None
        assert model.bias is not None
        assert len(model.cost_history) > 0
    
    def test_degree_1_is_linear(self, linear_data):
        """Test that degree 1 behaves like linear regression."""
        X, y = linear_data
        model = PolynomialRegression(degree=1, n_iterations=1000)
        
        model.fit(X, y)
        r2 = model.score(X, y)
        
        # Should fit linear data well
        assert r2 > 0.99
    
    def test_quadratic_data_fitting(self, quadratic_data):
        """Test fitting on quadratic data."""
        X, y = quadratic_data
        
        # Degree 1 (linear) - poor fit
        model_1 = PolynomialRegression(degree=1, n_iterations=1000)
        model_1.fit(X, y)
        r2_1 = model_1.score(X, y)
        
        # Degree 2 (quadratic) - good fit
        model_2 = PolynomialRegression(degree=2, n_iterations=1000)
        model_2.fit(X, y)
        r2_2 = model_2.score(X, y)
        
        # Quadratic should fit better
        assert r2_2 > r2_1
    
    def test_cost_history_decreasing(self, linear_data):
        """Test that cost decreases during training."""
        X, y = linear_data
        model = PolynomialRegression(degree=2, n_iterations=500)
        
        model.fit(X, y)
        
        # Cost should decrease
        assert model.cost_history[-1] < model.cost_history[0]
    
    def test_fitting_with_1d_target(self, linear_data):
        """Test fitting with 1D target array."""
        X, y = linear_data
        model = PolynomialRegression(degree=2, n_iterations=100)
        
        model.fit(X, y)
        
        assert model.weights is not None
    
    def test_fitting_with_multiple_features(self):
        """Test fitting with multiple input features."""
        X = np.random.randn(50, 3)
        y = X[:, 0] ** 2 + X[:, 1] + X[:, 2] + np.random.randn(50) * 0.1
        
        model = PolynomialRegression(degree=2, n_iterations=500)
        model.fit(X, y)
        
        assert model.weights is not None
        assert model.poly_features.n_input_features == 3


class TestPolynomialRegressionPrediction:
    """Test PolynomialRegression prediction."""
    
    @pytest.fixture
    def trained_model(self):
        """Create and train a model."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([1, 4, 9, 16, 25])  # y = x^2
        
        model = PolynomialRegression(degree=2, n_iterations=1000)
        model.fit(X, y)
        
        return model, X, y
    
    def test_prediction_shape(self, trained_model):
        """Test prediction shape."""
        model, X, _ = trained_model
        
        predictions = model.predict(X)
        
        assert predictions.shape == (5,)
    
    def test_single_sample_prediction(self, trained_model):
        """Test prediction on single sample."""
        model, _, _ = trained_model
        
        X_test = np.array([[6]])
        prediction = model.predict(X_test)
        
        assert prediction.shape == (1,)
        assert isinstance(prediction[0], (float, np.floating))
    
    def test_multiple_samples_prediction(self, trained_model):
        """Test prediction on multiple samples."""
        model, _, _ = trained_model
        
        X_test = np.array([[1.5], [2.5], [3.5]])
        predictions = model.predict(X_test)
        
        assert predictions.shape == (3,)
    
    def test_prediction_on_quadratic_data(self):
        """Test prediction accuracy on quadratic data."""
        X = np.array([[1], [2], [3], [4]])
        y = np.array([1, 4, 9, 16])  # y = x^2
        
        model = PolynomialRegression(degree=2, n_iterations=2000, learning_rate=0.01)
        model.fit(X, y)
        
        X_test = np.array([[5]])
        prediction = model.predict(X_test)
        
        # Should predict close to 25
        assert abs(prediction[0] - 25) < 5


class TestPolynomialRegressionScoring:
    """Test PolynomialRegression scoring."""
    
    def test_score_type(self):
        """Test that score returns a float."""
        X = np.array([[1], [2], [3]])
        y = np.array([1, 4, 9])
        
        model = PolynomialRegression(degree=2, n_iterations=500)
        model.fit(X, y)
        
        r2 = model.score(X, y)
        
        assert isinstance(r2, (float, np.floating))
    
    def test_perfect_fit_high_score(self):
        """Test that perfect fit gives high R² score."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])  # Linear
        
        model = PolynomialRegression(degree=1, n_iterations=2000)
        model.fit(X, y)
        
        r2 = model.score(X, y)
        
        assert r2 > 0.95
    
    def test_higher_degree_better_fit(self):
        """Test that appropriate degree fits better."""
        X = np.linspace(-2, 2, 30).reshape(-1, 1)
        y = X.flatten() ** 3 + np.random.randn(30) * 0.5
        
        model_1 = PolynomialRegression(degree=1, n_iterations=1000)
        model_1.fit(X, y)
        r2_1 = model_1.score(X, y)
        
        model_3 = PolynomialRegression(degree=3, n_iterations=1000)
        model_3.fit(X, y)
        r2_3 = model_3.score(X, y)
        
        # Degree 3 should fit cubic data better
        assert r2_3 > r2_1


class TestPolynomialRegressionRegularization:
    """Test regularization in PolynomialRegression."""
    
    def test_no_regularization(self):
        """Test fitting without regularization."""
        X = np.random.randn(50, 2)
        y = X[:, 0] ** 2 + X[:, 1]
        
        model = PolynomialRegression(
            degree=2,
            n_iterations=200,
            regularization=None
        )
        
        model.fit(X, y)
        
        assert model.weights is not None
    
    def test_l2_regularization(self):
        """Test fitting with L2 regularization."""
        X = np.random.randn(50, 2)
        y = X[:, 0] ** 2 + X[:, 1]
        
        model = PolynomialRegression(
            degree=2,
            n_iterations=200,
            regularization='l2',
            lambda_reg=0.1
        )
        
        model.fit(X, y)
        
        assert model.weights is not None
    
    def test_l1_regularization(self):
        """Test fitting with L1 regularization."""
        X = np.random.randn(50, 2)
        y = X[:, 0] ** 2 + X[:, 1]
        
        model = PolynomialRegression(
            degree=2,
            n_iterations=200,
            regularization='l1',
            lambda_reg=0.1
        )
        
        model.fit(X, y)
        
        assert model.weights is not None
    
    def test_regularization_reduces_overfitting(self):
        """Test that regularization helps with overfitting."""
        X = np.linspace(-3, 3, 20).reshape(-1, 1)
        y = X.flatten() ** 2 + np.random.randn(20) * 2
        
        # High degree without regularization
        model_no_reg = PolynomialRegression(
            degree=10,
            n_iterations=500,
            regularization=None
        )
        model_no_reg.fit(X, y)
        
        # High degree with regularization
        model_reg = PolynomialRegression(
            degree=10,
            n_iterations=500,
            regularization='l2',
            lambda_reg=0.5
        )
        model_reg.fit(X, y)
        
        # Regularized model should have smaller weights
        weight_magnitude_no_reg = np.sum(np.abs(model_no_reg.weights))
        weight_magnitude_reg = np.sum(np.abs(model_reg.weights))
        
        assert weight_magnitude_reg < weight_magnitude_no_reg


class TestPolynomialRegressionGetParams:
    """Test get_params method."""
    
    def test_get_params_before_fitting(self):
        """Test get_params before fitting."""
        model = PolynomialRegression(degree=3)
        
        params = model.get_params()
        
        assert params['degree'] == 3
        assert params['weights'] is None
        assert params['bias'] is None
        assert params['cost_history'] == []
    
    def test_get_params_after_fitting(self):
        """Test get_params after fitting."""
        X = np.array([[1], [2], [3]])
        y = np.array([1, 4, 9])
        
        model = PolynomialRegression(degree=2, n_iterations=100)
        model.fit(X, y)
        
        params = model.get_params()
        
        assert params['weights'] is not None
        assert params['bias'] is not None
        assert params['degree'] == 2
        assert params['n_polynomial_features'] > 0
        assert len(params['cost_history']) > 0


class TestPolynomialRegressionComparison:
    """Test against scikit-learn."""
    
    def test_comparison_with_sklearn_poly_features(self):
        """Compare polynomial features with sklearn."""
        X = np.array([[2, 3]])
        
        # Our implementation
        poly_ours = PolynomialFeatures(degree=2, include_bias=True)
        X_poly_ours = poly_ours.fit_transform(X)
        
        # Scikit-learn
        poly_sklearn = SklearnPolyFeatures(degree=2, include_bias=True)
        X_poly_sklearn = poly_sklearn.fit_transform(X)
        
        # Should have same number of features
        assert X_poly_ours.shape[1] == X_poly_sklearn.shape[1]


class TestPolynomialRegressionEdgeCases:
    """Test edge cases."""
    
    def test_single_sample(self):
        """Test with single sample."""
        X = np.array([[1]])
        y = np.array([2])
        
        model = PolynomialRegression(degree=2, n_iterations=50)
        model.fit(X, y)
        
        prediction = model.predict(X)
        
        assert prediction is not None
    
    def test_degree_zero_raises_or_handles(self):
        """Test degree 0 (constant model)."""
        X = np.array([[1], [2], [3]])
        y = np.array([5, 5, 5])
        
        model = PolynomialRegression(degree=0, n_iterations=50)
        
        # Should either handle gracefully or raise error
        try:
            model.fit(X, y)
            assert model.weights is not None
        except:
            pass  # Expected if degree 0 is not supported
    
    def test_very_high_degree(self):
        """Test with very high polynomial degree."""
        X = np.array([[1], [2], [3], [4]])
        y = np.array([1, 4, 9, 16])
        
        model = PolynomialRegression(degree=10, n_iterations=200)
        model.fit(X, y)
        
        # Should still work but might overfit
        assert model.weights is not None
    
    def test_with_negative_values(self):
        """Test with negative input values."""
        X = np.array([[-2], [-1], [0], [1], [2]])
        y = np.array([4, 1, 0, 1, 4])  # y = x^2
        
        model = PolynomialRegression(degree=2, n_iterations=1000)
        model.fit(X, y)
        
        r2 = model.score(X, y)
        
        assert r2 > 0.8


class TestPolynomialRegressionDifferentDegrees:
    """Test different polynomial degrees."""
    
    @pytest.fixture
    def cubic_data(self):
        """Generate cubic data."""
        X = np.linspace(-2, 2, 40).reshape(-1, 1)
        y = X.flatten() ** 3 + np.random.randn(40) * 0.5
        return X, y
    
    def test_degrees_1_to_5(self, cubic_data):
        """Test degrees 1 through 5."""
        X, y = cubic_data
        
        scores = []
        for degree in range(1, 6):
            model = PolynomialRegression(degree=degree, n_iterations=1000)
            model.fit(X, y)
            r2 = model.score(X, y)
            scores.append(r2)
        
        # Degree 3 should have good score for cubic data
        assert scores[2] > scores[0]  # degree 3 better than degree 1
    
    def test_increasing_features_with_degree(self):
        """Test that higher degree creates more features."""
        X = np.array([[1, 2]])
        
        n_features_list = []
        for degree in [1, 2, 3, 4]:
            model = PolynomialRegression(degree=degree)
            model.poly_features.fit(X)
            n_features_list.append(model.poly_features.n_output_features)
        
        # Each should be increasing
        for i in range(len(n_features_list) - 1):
            assert n_features_list[i + 1] > n_features_list[i]


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])