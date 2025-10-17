"""
Linear Regression Implementation from Scratch

This module implements Linear Regression using Ordinary Least Squares (OLS)
with gradient descent optimization.

Mathematical Foundation:
    y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε
    
    Cost Function (MSE): J(β) = (1/2m) * Σ(h(x⁽ⁱ⁾) - y⁽ⁱ⁾)²
    
    Gradient: ∂J/∂β = (1/m) * Xᵀ(Xβ - y)
"""

import numpy as np
from typing import Optional


class LinearRegression:
    """
    Linear Regression using Gradient Descent.
    
    Parameters
    ----------
    learning_rate : float, default=0.01
        Learning rate for gradient descent
    n_iterations : int, default=1000
        Number of iterations for gradient descent
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model
    method : str, default='gradient_descent'
        Method to use: 'gradient_descent' or 'normal_equation'
        
    Attributes
    ----------
    weights : np.ndarray
        Coefficients of the linear model
    bias : float
        Intercept term
    cost_history : list
        History of cost function values during training
    """
    
    def __init__(
        self, 
        learning_rate: float = 0.01, 
        n_iterations: int = 1000,
        fit_intercept: bool = True,
        method: str = 'gradient_descent'
    ):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.fit_intercept = fit_intercept
        self.method = method
        self.weights = None
        self.bias = None
        self.cost_history = []
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegression':
        """
        Fit the linear regression model.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data
        y : np.ndarray of shape (n_samples,)
            Target values
            
        Returns
        -------
        self : LinearRegression
            Fitted estimator
        """
        # Convert to numpy arrays if needed
        X = np.array(X)
        y = np.array(y)
        
        # Reshape y if needed
        if y.ndim == 1:
            y = y.reshape(-1, 1)
            
        n_samples, n_features = X.shape
        
        if self.method == 'gradient_descent':
            self._fit_gradient_descent(X, y, n_samples, n_features)
        elif self.method == 'normal_equation':
            self._fit_normal_equation(X, y)
        else:
            raise ValueError(f"Unknown method: {self.method}")
            
        return self
    
    def _fit_gradient_descent(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        n_samples: int, 
        n_features: int
    ) -> None:
        """Fit using gradient descent optimization."""
        # Initialize parameters
        self.weights = np.zeros((n_features, 1))
        self.bias = 0
        
        # Gradient descent
        for i in range(self.n_iterations):
            # Forward pass: compute predictions
            y_pred = self.predict(X)
            
            # Compute cost (MSE)
            cost = (1 / (2 * n_samples)) * np.sum((y_pred - y) ** 2)
            self.cost_history.append(cost)
            
            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
    def _fit_normal_equation(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit using the normal equation (closed-form solution).
        
        Normal Equation: β = (XᵀX)⁻¹Xᵀy
        """
        if self.fit_intercept:
            # Add bias column
            X_with_bias = np.c_[np.ones((X.shape[0], 1)), X]
        else:
            X_with_bias = X
            
        # Compute parameters using normal equation
        # Adding small value to diagonal for numerical stability
        theta = np.linalg.inv(X_with_bias.T @ X_with_bias + 1e-8 * np.eye(X_with_bias.shape[1])) @ X_with_bias.T @ y
        
        if self.fit_intercept:
            self.bias = theta[0, 0]
            self.weights = theta[1:].reshape(-1, 1)
        else:
            self.bias = 0
            self.weights = theta.reshape(-1, 1)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the linear model.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data
            
        Returns
        -------
        y_pred : np.ndarray of shape (n_samples, 1)
            Predicted values
        """
        X = np.array(X)
        return np.dot(X, self.weights) + self.bias
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate R² (coefficient of determination).
        
        R² = 1 - (SS_res / SS_tot)
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test data
        y : np.ndarray of shape (n_samples,)
            True values
            
        Returns
        -------
        score : float
            R² score
        """
        y = np.array(y).reshape(-1, 1)
        y_pred = self.predict(X)
        
        # Total sum of squares
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        # Residual sum of squares
        ss_res = np.sum((y - y_pred) ** 2)
        
        # R² score
        r2 = 1 - (ss_res / ss_tot)
        
        return r2
    
    def get_params(self) -> dict:
        """
        Get model parameters.
        
        Returns
        -------
        params : dict
            Dictionary containing weights and bias
        """
        return {
            'weights': self.weights,
            'bias': self.bias,
            'cost_history': self.cost_history
        }


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    
    # Generate sample data
    X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model using gradient descent
    model_gd = LinearRegression(learning_rate=0.01, n_iterations=1000, method='gradient_descent')
    model_gd.fit(X_train, y_train)
    
    print("Gradient Descent Method:")
    print(f"R² Score: {model_gd.score(X_test, y_test):.4f}")
    print(f"Weights: {model_gd.weights.flatten()}")
    print(f"Bias: {model_gd.bias:.4f}")
    print(f"Final Cost: {model_gd.cost_history[-1]:.4f}\n")
    
    # Train model using normal equation
    model_ne = LinearRegression(method='normal_equation')
    model_ne.fit(X_train, y_train)
    
    print("Normal Equation Method:")
    print(f"R² Score: {model_ne.score(X_test, y_test):.4f}")
    print(f"Weights: {model_ne.weights.flatten()}")
    print(f"Bias: {model_ne.bias:.4f}")