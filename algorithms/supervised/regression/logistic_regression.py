"""
Logistic Regression Implementation from Scratch

Logistic Regression is used for binary classification problems.
It uses the sigmoid function to map predictions to probabilities.

Mathematical Foundation:
    Hypothesis: h(x) = σ(wᵀx + b) where σ(z) = 1 / (1 + e⁻ᶻ)
    
    Cost Function (Binary Cross-Entropy):
    J(w,b) = -(1/m) * Σ[y⁽ⁱ⁾log(h(x⁽ⁱ⁾)) + (1-y⁽ⁱ⁾)log(1-h(x⁽ⁱ⁾))]
    
    Gradients:
    ∂J/∂w = (1/m) * Xᵀ(h(X) - y)
    ∂J/∂b = (1/m) * Σ(h(x⁽ⁱ⁾) - y⁽ⁱ⁾)
"""

import numpy as np
from typing import Optional


class LogisticRegression:
    """
    Logistic Regression for binary classification.
    
    Parameters
    ----------
    learning_rate : float, default=0.01
        Learning rate for gradient descent
    n_iterations : int, default=1000
        Number of iterations for gradient descent
    regularization : str or None, default=None
        Type of regularization: None, 'l1', or 'l2'
    lambda_reg : float, default=0.01
        Regularization parameter
    tol : float, default=1e-4
        Tolerance for stopping criterion
        
    Attributes
    ----------
    weights : np.ndarray
        Coefficients of the logistic model
    bias : float
        Intercept term
    cost_history : list
        History of cost function values during training
    """
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        regularization: Optional[str] = None,
        lambda_reg: float = 0.01,
        tol: float = 1e-4
    ):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        self.tol = tol
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Sigmoid activation function.
        
        σ(z) = 1 / (1 + e⁻ᶻ)
        
        Clips z to prevent overflow.
        """
        z = np.clip(z, -500, 500)  # Prevent overflow
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegression':
        """
        Fit the logistic regression model.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data
        y : np.ndarray of shape (n_samples,)
            Target values (0 or 1)
            
        Returns
        -------
        self : LogisticRegression
            Fitted estimator
        """
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros((n_features, 1))
        self.bias = 0
        
        # Gradient descent
        for i in range(self.n_iterations):
            # Forward pass: compute predictions
            linear_output = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear_output)
            
            # Compute cost
            cost = self._compute_cost(y, y_pred, n_samples)
            self.cost_history.append(cost)
            
            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            # Add regularization to gradients
            if self.regularization == 'l2':
                dw += (self.lambda_reg / n_samples) * self.weights
            elif self.regularization == 'l1':
                dw += (self.lambda_reg / n_samples) * np.sign(self.weights)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Check convergence
            if i > 0 and abs(self.cost_history[-1] - self.cost_history[-2]) < self.tol:
                break
        
        return self
    
    def _compute_cost(self, y: np.ndarray, y_pred: np.ndarray, n_samples: int) -> float:
        """
        Compute binary cross-entropy cost with optional regularization.
        """
        # Clip predictions to prevent log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # Binary cross-entropy
        cost = -(1 / n_samples) * np.sum(
            y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)
        )
        
        # Add regularization term
        if self.regularization == 'l2':
            cost += (self.lambda_reg / (2 * n_samples)) * np.sum(self.weights ** 2)
        elif self.regularization == 'l1':
            cost += (self.lambda_reg / n_samples) * np.sum(np.abs(self.weights))
        
        return cost
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data
            
        Returns
        -------
        probabilities : np.ndarray of shape (n_samples, 1)
            Predicted probabilities for the positive class
        """
        X = np.array(X)
        linear_output = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_output)
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict class labels.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data
        threshold : float, default=0.5
            Decision threshold for classification
            
        Returns
        -------
        predictions : np.ndarray of shape (n_samples, 1)
            Predicted class labels (0 or 1)
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate accuracy score.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test data
        y : np.ndarray of shape (n_samples,)
            True labels
            
        Returns
        -------
        accuracy : float
            Accuracy score
        """
        y = np.array(y).reshape(-1, 1)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
    
    def get_params(self) -> dict:
        """Get model parameters."""
        return {
            'weights': self.weights,
            'bias': self.bias,
            'cost_history': self.cost_history
        }


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=5,
                               n_redundant=2, random_state=42)
    
    # Split and standardize
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train model
    model = LogisticRegression(learning_rate=0.1, n_iterations=1000, 
                               regularization='l2', lambda_reg=0.01)
    model.fit(X_train, y_train)
    
    print("Logistic Regression Results:")
    print(f"Training Accuracy: {model.score(X_train, y_train):.4f}")
    print(f"Test Accuracy: {model.score(X_test, y_test):.4f}")
    print(f"Final Cost: {model.cost_history[-1]:.4f}")
    print(f"Number of iterations: {len(model.cost_history)}")