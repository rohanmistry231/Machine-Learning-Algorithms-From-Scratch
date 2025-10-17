"""
Polynomial Regression Implementation from Scratch

Polynomial Regression extends linear regression by adding polynomial features,
allowing it to model non-linear relationships.

Mathematical Foundation:
    y = β₀ + β₁x + β₂x² + β₃x³ + ... + βₙxⁿ
    
    This is still linear in parameters (β), so we can use linear regression
    after transforming features to polynomial form.
    
    For multiple features:
    - Degree 2: [1, x₁, x₂, x₁², x₁x₂, x₂²]
    - Degree 3: [1, x₁, x₂, x₁², x₁x₂, x₂², x₁³, x₁²x₂, x₁x₂², x₂³]
"""

import numpy as np
from itertools import combinations_with_replacement
from typing import List, Tuple


class PolynomialFeatures:
    """
    Generate polynomial features.
    
    Parameters
    ----------
    degree : int, default=2
        Degree of polynomial features
    include_bias : bool, default=True
        Whether to include bias column (column of ones)
    interaction_only : bool, default=False
        If True, only interaction features (products of distinct features)
        
    Attributes
    ----------
    n_input_features : int
        Number of input features
    n_output_features : int
        Number of output features after transformation
    powers : List[Tuple]
        List of feature power combinations
    """
    
    def __init__(
        self,
        degree: int = 2,
        include_bias: bool = True,
        interaction_only: bool = False
    ):
        self.degree = degree
        self.include_bias = include_bias
        self.interaction_only = interaction_only
        self.n_input_features = None
        self.n_output_features = None
        self.powers = None
    
    def fit(self, X: np.ndarray) -> 'PolynomialFeatures':
        """
        Compute number of output features.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Data to fit
            
        Returns
        -------
        self : PolynomialFeatures
            Fitted transformer
        """
        X = np.array(X)
        n_samples, n_features = X.shape
        self.n_input_features = n_features
        
        # Generate all combinations of powers
        self.powers = []
        
        # Generate combinations for each degree
        for deg in range(0 if self.include_bias else 1, self.degree + 1):
            if self.interaction_only and deg > 1:
                # Only interaction terms (no x^2, x^3, etc.)
                for combo in combinations_with_replacement(range(n_features), deg):
                    # Check if all features are different (interaction only)
                    if len(set(combo)) == len(combo):
                        powers = [0] * n_features
                        for feature_idx in combo:
                            powers[feature_idx] += 1
                        self.powers.append(tuple(powers))
            else:
                # All combinations including powers
                for combo in combinations_with_replacement(range(n_features), deg):
                    powers = [0] * n_features
                    for feature_idx in combo:
                        powers[feature_idx] += 1
                    self.powers.append(tuple(powers))
        
        self.n_output_features = len(self.powers)
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data to polynomial features.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Data to transform
            
        Returns
        -------
        X_poly : np.ndarray of shape (n_samples, n_output_features)
            Transformed polynomial features
        """
        X = np.array(X)
        n_samples = X.shape[0]
        
        X_poly = np.zeros((n_samples, self.n_output_features))
        
        for i, power_combo in enumerate(self.powers):
            # Compute each polynomial feature
            X_poly[:, i] = np.prod(X ** power_combo, axis=1)
        
        return X_poly
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit and transform data.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Data to transform
            
        Returns
        -------
        X_poly : np.ndarray of shape (n_samples, n_output_features)
            Transformed polynomial features
        """
        self.fit(X)
        return self.transform(X)
    
    def get_feature_names(self, input_features: List[str] = None) -> List[str]:
        """
        Get feature names for output features.
        
        Parameters
        ----------
        input_features : List[str] or None
            Names of input features
            
        Returns
        -------
        feature_names : List[str]
            Names of output features
        """
        if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_input_features)]
        
        feature_names = []
        for powers in self.powers:
            if sum(powers) == 0:
                # Bias term
                feature_names.append("1")
            else:
                # Build feature name
                terms = []
                for idx, power in enumerate(powers):
                    if power > 0:
                        if power == 1:
                            terms.append(input_features[idx])
                        else:
                            terms.append(f"{input_features[idx]}^{power}")
                feature_names.append(" ".join(terms))
        
        return feature_names


class PolynomialRegression:
    """
    Polynomial Regression using polynomial feature transformation.
    
    Parameters
    ----------
    degree : int, default=2
        Degree of polynomial features
    learning_rate : float, default=0.01
        Learning rate for gradient descent
    n_iterations : int, default=1000
        Number of iterations for gradient descent
    include_bias : bool, default=True
        Whether to include bias term in polynomial features
    regularization : str or None, default=None
        Regularization type: None, 'l1', or 'l2'
    lambda_reg : float, default=0.01
        Regularization parameter
        
    Attributes
    ----------
    poly_features : PolynomialFeatures
        Polynomial feature transformer
    weights : np.ndarray
        Model coefficients
    bias : float
        Bias term
    cost_history : list
        Training cost history
    """
    
    def __init__(
        self,
        degree: int = 2,
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        include_bias: bool = True,
        regularization: str = None,
        lambda_reg: float = 0.01
    ):
        self.degree = degree
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.include_bias = include_bias
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        
        self.poly_features = PolynomialFeatures(degree=degree, include_bias=False)
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'PolynomialRegression':
        """
        Fit polynomial regression model.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data
        y : np.ndarray of shape (n_samples,)
            Target values
            
        Returns
        -------
        self : PolynomialRegression
            Fitted estimator
        """
        X = np.array(X)
        y = np.array(y)
        
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        # Transform features to polynomial
        X_poly = self.poly_features.fit_transform(X)
        
        n_samples, n_features = X_poly.shape
        
        # Initialize parameters
        self.weights = np.zeros((n_features, 1))
        self.bias = 0
        
        # Gradient descent
        for i in range(self.n_iterations):
            # Forward pass
            y_pred = np.dot(X_poly, self.weights) + self.bias
            
            # Compute cost
            mse_cost = (1 / (2 * n_samples)) * np.sum((y_pred - y) ** 2)
            
            # Add regularization to cost
            if self.regularization == 'l2':
                reg_cost = (self.lambda_reg / (2 * n_samples)) * np.sum(self.weights ** 2)
                cost = mse_cost + reg_cost
            elif self.regularization == 'l1':
                reg_cost = (self.lambda_reg / n_samples) * np.sum(np.abs(self.weights))
                cost = mse_cost + reg_cost
            else:
                cost = mse_cost
            
            self.cost_history.append(cost)
            
            # Compute gradients
            dw = (1 / n_samples) * np.dot(X_poly.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            # Add regularization to gradients
            if self.regularization == 'l2':
                dw += (self.lambda_reg / n_samples) * self.weights
            elif self.regularization == 'l1':
                dw += (self.lambda_reg / n_samples) * np.sign(self.weights)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data
            
        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted values
        """
        X = np.array(X)
        X_poly = self.poly_features.transform(X)
        predictions = np.dot(X_poly, self.weights) + self.bias
        return predictions.flatten()
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate R² score.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test data
        y : np.ndarray of shape (n_samples,)
            True values
            
        Returns
        -------
        r2 : float
            R² score
        """
        y_pred = self.predict(X)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        ss_res = np.sum((y - y_pred) ** 2)
        return 1 - (ss_res / ss_tot)
    
    def get_params(self) -> dict:
        """Get model parameters."""
        return {
            'degree': self.degree,
            'weights': self.weights,
            'bias': self.bias,
            'n_polynomial_features': self.poly_features.n_output_features,
            'cost_history': self.cost_history
        }


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Generate non-linear data
    np.random.seed(42)
    X = np.linspace(-3, 3, 100).reshape(-1, 1)
    y = 0.5 * X.flatten() ** 3 - 2 * X.flatten() ** 2 + X.flatten() + np.random.randn(100) * 2
    
    print("=== Polynomial Regression with Different Degrees ===")
    
    degrees = [1, 2, 3, 5]
    
    for degree in degrees:
        print(f"\n--- Degree {degree} ---")
        
        # Fit polynomial regression
        poly_reg = PolynomialRegression(degree=degree, learning_rate=0.01,
                                        n_iterations=1000, regularization=None)
        poly_reg.fit(X, y)
        
        # Predictions
        y_pred = poly_reg.predict(X)
        
        print(f"R² Score: {poly_reg.score(X, y):.4f}")
        print(f"Number of polynomial features: {poly_reg.get_params()['n_polynomial_features']}")
        print(f"Final cost: {poly_reg.cost_history[-1]:.4f}")
    
    print("\n=== Polynomial Features Transformation Example ===")
    # Example of polynomial features
    X_simple = np.array([[2, 3]])
    
    for deg in [2, 3]:
        poly = PolynomialFeatures(degree=deg, include_bias=True)
        X_poly = poly.fit_transform(X_simple)
        feature_names = poly.get_feature_names(['x1', 'x2'])
        
        print(f"\nDegree {deg}:")
        print(f"Original: {X_simple[0]}")
        print(f"Polynomial features: {X_poly[0]}")
        print(f"Feature names: {feature_names}")