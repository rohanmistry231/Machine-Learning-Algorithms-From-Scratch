"""
Lasso Regression (L1 Regularization) Implementation from Scratch

Lasso Regression adds L1 regularization penalty to linear regression.
It can perform feature selection by shrinking some coefficients to zero.

Mathematical Foundation:
    Cost Function: J(β) = (1/2m) * Σ(h(x⁽ⁱ⁾) - y⁽ⁱ⁾)² + (λ/m) * Σ|βⱼ|
    
    where λ is the regularization parameter
    
    Uses coordinate descent or subgradient methods for optimization
    (no closed-form solution exists due to L1 penalty)
"""

import numpy as np


class LassoRegression:
    """
    Lasso Regression with L1 regularization.
    
    Parameters
    ----------
    alpha : float, default=1.0
        Regularization strength (λ). Must be positive.
        Larger values specify stronger regularization.
    learning_rate : float, default=0.01
        Learning rate for gradient descent
    n_iterations : int, default=1000
        Number of iterations
    method : str, default='coordinate_descent'
        Method to use: 'coordinate_descent' or 'subgradient'
    tol : float, default=1e-4
        Tolerance for convergence
        
    Attributes
    ----------
    weights : np.ndarray
        Coefficients of the linear model
    bias : float
        Intercept term
    cost_history : list
        History of cost function values during training
    """
    
    def _init_(
        self,
        alpha: float = 1.0,
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        method: str = 'coordinate_descent',
        tol: float = 1e-4
    ):
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.method = method
        self.tol = tol
        self.weights = None
        self.bias = None
        self.cost_history = []
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LassoRegression':
        """
        Fit the lasso regression model.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data
        y : np.ndarray of shape (n_samples,)
            Target values
            
        Returns
        -------
        self : LassoRegression
            Fitted estimator
        """
        X = np.array(X)
        y = np.array(y)
        
        if y.ndim == 1:
            y = y.reshape(-1, 1)
            
        n_samples, n_features = X.shape
        
        # Standardize features for better convergence
        self.X_mean = np.mean(X, axis=0)
        self.X_std = np.std(X, axis=0) + 1e-8
        X_normalized = (X - self.X_mean) / self.X_std
        
        if self.method == 'coordinate_descent':
            self._fit_coordinate_descent(X_normalized, y, n_samples, n_features)
        elif self.method == 'subgradient':
            self._fit_subgradient(X_normalized, y, n_samples, n_features)
        else:
            raise ValueError(f"Unknown method: {self.method}")
            
        return self
    
    def _soft_threshold(self, rho: float, lambda_val: float) -> float:
        """
        Soft-thresholding operator for Lasso.
        
        S(ρ, λ) = sign(ρ) * max(|ρ| - λ, 0)
        """
        if rho < -lambda_val:
            return rho + lambda_val
        elif rho > lambda_val:
            return rho - lambda_val
        else:
            return 0
    
    def _fit_coordinate_descent(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_samples: int,
        n_features: int
    ) -> None:
        """
        Fit using coordinate descent algorithm.
        
        This is the standard algorithm for Lasso regression.
        Updates one coefficient at a time while holding others fixed.
        """
        # Initialize parameters
        self.weights = np.zeros((n_features, 1))
        self.bias = np.mean(y)
        
        for iteration in range(self.n_iterations):
            weights_old = self.weights.copy()
            
            # Update bias
            residuals = y - (np.dot(X, self.weights) + self.bias)
            self.bias = np.mean(residuals)
            
            # Update each weight using coordinate descent
            for j in range(n_features):
                # Compute residual without feature j
                X_j = X[:, j].reshape(-1, 1)
                residuals = y - (np.dot(X, self.weights) + self.bias - np.dot(X_j, self.weights[j]))
                
                # Compute ρⱼ = Σ xⱼ(y - ŷ₍₋ⱼ₎)
                rho = np.dot(X_j.T, residuals)[0, 0]
                
                # Apply soft-thresholding
                lambda_val = self.alpha * n_samples
                self.weights[j] = self._soft_threshold(rho, lambda_val) / (np.sum(X_j ** 2))
            
            # Compute cost
            cost = self._compute_cost(X, y, n_samples)
            self.cost_history.append(cost)
            
            # Check convergence
            if np.sum(np.abs(self.weights - weights_old)) < self.tol:
                break
    
    def _fit_subgradient(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_samples: int,
        n_features: int
    ) -> None:
        """
        Fit using subgradient descent.
        
        Uses subgradient of L1 norm for optimization.
        """
        # Initialize parameters
        self.weights = np.zeros((n_features, 1))
        self.bias = 0
        
        for i in range(self.n_iterations):
            # Forward pass
            y_pred = self.predict_normalized(X)
            
            # Compute cost
            cost = self._compute_cost(X, y, n_samples)
            self.cost_history.append(cost)
            
            # Compute subgradient
            # For L1: subgradient is sign(w) when w ≠ 0, and [-1, 1] when w = 0
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            dw += (self.alpha / n_samples) * np.sign(self.weights)
            
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def _compute_cost(self, X: np.ndarray, y: np.ndarray, n_samples: int) -> float:
        """Compute Lasso cost function."""
        y_pred = self.predict_normalized(X)
        mse_cost = (1 / (2 * n_samples)) * np.sum((y_pred - y) ** 2)
        l1_cost = (self.alpha / n_samples) * np.sum(np.abs(self.weights))
        return mse_cost + l1_cost
    
    def predict_normalized(self, X: np.ndarray) -> np.ndarray:
        """Predict using normalized features."""
        return np.dot(X, self.weights) + self.bias
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the lasso regression model.
        
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
        X_normalized = (X - self.X_mean) / self.X_std
        return self.predict_normalized(X_normalized)
    
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
        score : float
            R² score
        """
        y = np.array(y).reshape(-1, 1)
        y_pred = self.predict(X)
        
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        ss_res = np.sum((y - y_pred) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        return r2
    
    def get_params(self) -> dict:
        """Get model parameters."""
        return {
            'weights': self.weights,
            'bias': self.bias,
            'alpha': self.alpha,
            'cost_history': self.cost_history,
            'n_nonzero_weights': np.sum(np.abs(self.weights) > 1e-6)
        }


if _name_ == "_main_":
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    
    # Generate sample data
    X, y = make_regression(n_samples=100, n_features=10, n_informative=5, 
                          noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = LassoRegression(alpha=0.1, n_iterations=1000, method='coordinate_descent')
    model.fit(X_train, y_train)
    
    print("Lasso Regression Results:")
    print(f"R² Score: {model.score(X_test, y_test):.4f}")
    print(f"Number of non-zero weights: {model.get_params()['n_nonzero_weights']}")
    print(f"Total features: {X_train.shape[1]}")
    print(f"Alpha (regularization): {model.alpha}")