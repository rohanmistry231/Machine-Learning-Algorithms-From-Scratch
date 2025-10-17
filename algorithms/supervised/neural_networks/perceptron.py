"""
Perceptron Implementation from Scratch

The Perceptron is the simplest neural network unit and forms
the foundation for more complex neural networks.

Mathematical Foundation:
    Activation: ŷ = sign(wᵀx + b)
    
    Update rule (for misclassified samples):
    w := w + η * y * x
    b := b + η * y
    
    where:
    - η is the learning rate
    - y is the true label (-1 or 1)
    - x is the input vector
"""

import numpy as np


class Perceptron:
    """
    Perceptron classifier.
    
    Binary classifier that learns a linear decision boundary.
    
    Parameters
    ----------
    learning_rate : float, default=0.01
        Learning rate for weight updates
    n_iterations : int, default=1000
        Maximum number of passes over the training data
    random_state : int or None, default=None
        Random seed for reproducibility
        
    Attributes
    ----------
    weights : np.ndarray
        Weights after fitting
    bias : float
        Bias term after fitting
    errors : list
        Number of misclassifications in each epoch
    """
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        random_state: int = None
    ):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.weights = None
        self.bias = None
        self.errors = []
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'Perceptron':
        """
        Fit the Perceptron model.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data
        y : np.ndarray of shape (n_samples,)
            Target labels (must be -1 or 1)
            
        Returns
        -------
        self : Perceptron
            Fitted estimator
        """
        X = np.array(X)
        y = np.array(y)
        
        # Convert labels to -1 and 1 if needed
        unique_labels = np.unique(y)
        if not np.array_equal(unique_labels, np.array([-1, 1])):
            y = np.where(y == unique_labels[0], -1, 1)
        
        n_samples, n_features = X.shape
        
        # Initialize weights and bias
        if self.random_state:
            np.random.seed(self.random_state)
        
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Training loop
        for _ in range(self.n_iterations):
            errors = 0
            
            for idx, x_i in enumerate(X):
                # Compute prediction
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_pred = self._activation(linear_output)
                
                # Update weights if misclassified
                update = self.learning_rate * (y[idx] - y_pred)
                
                if update != 0:
                    self.weights += update * x_i
                    self.bias += update
                    errors += 1
            
            self.errors.append(errors)
            
            # Early stopping if no errors
            if errors == 0:
                break
        
        return self
    
    def _activation(self, x: float) -> int:
        """
        Step activation function.
        
        Returns 1 if x >= 0, else -1
        """
        return np.where(x >= 0, 1, -1)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples in X.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data
            
        Returns
        -------
        predictions : np.ndarray of shape (n_samples,)
            Predicted class labels (-1 or 1)
        """
        X = np.array(X)
        linear_output = np.dot(X, self.weights) + self.bias
        return self._activation(linear_output)
    
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
        y = np.array(y)
        
        # Convert labels to -1 and 1 if needed