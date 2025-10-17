"""
Multi-Layer Perceptron (MLP) Implementation from Scratch

MLP is a feedforward neural network with one or more hidden layers.
It uses backpropagation algorithm for training.

Mathematical Foundation:
    Forward Pass:
    - z⁽ˡ⁾ = W⁽ˡ⁾a⁽ˡ⁻¹⁾ + b⁽ˡ⁾
    - a⁽ˡ⁾ = σ(z⁽ˡ⁾)
    
    Backward Pass (Backpropagation):
    - δ⁽ᴸ⁾ = (a⁽ᴸ⁾ - y) ⊙ σ'(z⁽ᴸ⁾)
    - δ⁽ˡ⁾ = (W⁽ˡ⁺¹⁾)ᵀδ⁽ˡ⁺¹⁾ ⊙ σ'(z⁽ˡ⁾)
    
    Weight Updates:
    - W⁽ˡ⁾ := W⁽ˡ⁾ - η * δ⁽ˡ⁾(a⁽ˡ⁻¹⁾)ᵀ
    - b⁽ˡ⁾ := b⁽ˡ⁾ - η * δ⁽ˡ⁾
"""

import numpy as np
from typing import List, Tuple


class MLP:
    """
    Multi-Layer Perceptron neural network.
    
    Parameters
    ----------
    hidden_layers : List[int], default=[64, 32]
        List of hidden layer sizes
    learning_rate : float, default=0.01
        Learning rate for gradient descent
    n_iterations : int, default=1000
        Number of training iterations
    activation : str, default='relu'
        Activation function: 'relu', 'sigmoid', 'tanh'
    batch_size : int or None, default=32
        Size of mini-batches. If None, use batch gradient descent
    random_state : int or None, default=None
        Random seed for reproducibility
    task : str, default='classification'
        Task type: 'classification' or 'regression'
        
    Attributes
    ----------
    weights : List[np.ndarray]
        Weight matrices for each layer
    biases : List[np.ndarray]
        Bias vectors for each layer
    loss_history : list
        Training loss history
    """
    
    def __init__(
        self,
        hidden_layers: List[int] = [64, 32],
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        activation: str = 'relu',
        batch_size: int = 32,
        random_state: int = None,
        task: str = 'classification'
    ):
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.activation = activation
        self.batch_size = batch_size
        self.random_state = random_state
        self.task = task
        self.weights = []
        self.biases = []
        self.loss_history = []
    
    def _initialize_parameters(self, n_features: int, n_outputs: int) -> None:
        """Initialize weights and biases using He initialization."""
        if self.random_state:
            np.random.seed(self.random_state)
        
        layer_sizes = [n_features] + self.hidden_layers + [n_outputs]
        
        self.weights = []
        self.biases = []
        
        for i in range(len(layer_sizes) - 1):
            # He initialization for weights
            weight = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * \
                     np.sqrt(2.0 / layer_sizes[i])
            bias = np.zeros((1, layer_sizes[i + 1]))
            
            self.weights.append(weight)
            self.biases.append(bias)
    
    def _activation_function(self, z: np.ndarray) -> np.ndarray:
        """Apply activation function."""
        if self.activation == 'relu':
            return np.maximum(0, z)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        elif self.activation == 'tanh':
            return np.tanh(z)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
    
    def _activation_derivative(self, z: np.ndarray) -> np.ndarray:
        """Compute derivative of activation function."""
        if self.activation == 'relu':
            return (z > 0).astype(float)
        elif self.activation == 'sigmoid':
            s = self._activation_function(z)
            return s * (1 - s)
        elif self.activation == 'tanh':
            return 1 - np.tanh(z) ** 2
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
    
    def _forward_pass(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Perform forward pass through the network.
        
        Returns
        -------
        activations : List[np.ndarray]
            Activations for each layer
        z_values : List[np.ndarray]
            Pre-activation values for each layer
        """
        activations = [X]
        z_values = []
        
        # Forward through hidden layers
        for i in range(len(self.weights) - 1):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            z_values.append(z)
            a = self._activation_function(z)
            activations.append(a)
        
        # Output layer
        z_out = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        z_values.append(z_out)
        
        if self.task == 'classification':
            # Sigmoid for binary, softmax could be added for multi-class
            a_out = 1 / (1 + np.exp(-np.clip(z_out, -500, 500)))
        else:
            # Linear activation for regression
            a_out = z_out
        
        activations.append(a_out)
        
        return activations, z_values
    
    def _backward_pass(
        self,
        X: np.ndarray,
        y: np.ndarray,
        activations: List[np.ndarray],
        z_values: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Perform backward pass (backpropagation).
        
        Returns
        -------
        weight_gradients : List[np.ndarray]
            Gradients for weights
        bias_gradients : List[np.ndarray]
            Gradients for biases
        """
        m = X.shape[0]
        
        weight_gradients = []
        bias_gradients = []
        
        # Output layer error
        if self.task == 'classification':
            delta = activations[-1] - y
        else:
            delta = activations[-1] - y
        
        # Backpropagate through layers
        for i in range(len(self.weights) - 1, -1, -1):
            # Compute gradients
            dW = (1 / m) * np.dot(activations[i].T, delta)
            db = (1 / m) * np.sum(delta, axis=0, keepdims=True)
            
            weight_gradients.insert(0, dW)
            bias_gradients.insert(0, db)
            
            # Propagate error to previous layer
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * \
                       self._activation_derivative(z_values[i - 1])
        
        return weight_gradients, bias_gradients
    
    def _compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute loss function."""
        m = y_true.shape[0]
        
        if self.task == 'classification':
            # Binary cross-entropy
            epsilon = 1e-15
            y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
            loss = -(1 / m) * np.sum(
                y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
            )
        else:
            # Mean squared error
            loss = (1 / (2 * m)) * np.sum((y_pred - y_true) ** 2)
        
        return loss
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MLP':
        """
        Fit the MLP model.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data
        y : np.ndarray of shape (n_samples,) or (n_samples, n_outputs)
            Target values
            
        Returns
        -------
        self : MLP
            Fitted estimator
        """
        X = np.array(X)
        y = np.array(y)
        
        # Reshape y if needed
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        n_samples, n_features = X.shape
        n_outputs = y.shape[1]
        
        # Initialize parameters
        self._initialize_parameters(n_features, n_outputs)
        
        # Training loop
        for iteration in range(self.n_iterations):
            # Mini-batch gradient descent
            if self.batch_size is not None and self.batch_size < n_samples:
                indices = np.random.permutation(n_samples)
                
                for start_idx in range(0, n_samples, self.batch_size):
                    end_idx = min(start_idx + self.batch_size, n_samples)
                    batch_indices = indices[start_idx:end_idx]
                    
                    X_batch = X[batch_indices]
                    y_batch = y[batch_indices]
                    
                    # Forward and backward pass
                    activations, z_values = self._forward_pass(X_batch)
                    weight_grads, bias_grads = self._backward_pass(
                        X_batch, y_batch, activations, z_values
                    )
                    
                    # Update parameters
                    for i in range(len(self.weights)):
                        self.weights[i] -= self.learning_rate * weight_grads[i]
                        self.biases[i] -= self.learning_rate * bias_grads[i]
            else:
                # Batch gradient descent
                activations, z_values = self._forward_pass(X)
                weight_grads, bias_grads = self._backward_pass(X, y, activations, z_values)
                
                # Update parameters
                for i in range(len(self.weights)):
                    self.weights[i] -= self.learning_rate * weight_grads[i]
                    self.biases[i] -= self.learning_rate * bias_grads[i]
            
            # Compute and store loss
            if iteration % 10 == 0:
                activations, _ = self._forward_pass(X)
                loss = self._compute_loss(y, activations[-1])
                self.loss_history.append(loss)
        
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
        predictions : np.ndarray
            Predicted values or class labels
        """
        X = np.array(X)
        activations, _ = self._forward_pass(X)
        predictions = activations[-1]
        
        if self.task == 'classification':
            return (predictions >= 0.5).astype(int).flatten()
        else:
            return predictions.flatten()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities (classification only)."""
        if self.task != 'classification':
            raise ValueError("predict_proba only available for classification")
        
        X = np.array(X)
        activations, _ = self._forward_pass(X)
        return activations[-1]
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate score (accuracy for classification, R² for regression).
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test data
        y : np.ndarray of shape (n_samples,)
            True labels/values
            
        Returns
        -------
        score : float
            Accuracy (classification) or R² (regression)
        """
        y_pred = self.predict(X)
        
        if self.task == 'classification':
            return np.mean(y_pred == y)
        else:
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            ss_res = np.sum((y - y_pred) ** 2)
            return 1 - (ss_res / ss_tot)


if __name__ == "__main__":
    from sklearn.datasets import make_classification, make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    print("=== Classification Example ===")
    X_clf, y_clf = make_classification(n_samples=500, n_features=20, 
                                       n_informative=15, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    mlp_clf = MLP(hidden_layers=[64, 32], learning_rate=0.01, 
                  n_iterations=500, activation='relu', batch_size=32,
                  task='classification', random_state=42)
    mlp_clf.fit(X_train_scaled, y_train)
    
    print(f"Training Accuracy: {mlp_clf.score(X_train_scaled, y_train):.4f}")
    print(f"Test Accuracy: {mlp_clf.score(X_test_scaled, y_test):.4f}")
    print(f"Final Loss: {mlp_clf.loss_history[-1]:.4f}")
    
    print("\n=== Regression Example ===")
    X_reg, y_reg = make_regression(n_samples=500, n_features=20, noise=10,
                                   random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    mlp_reg = MLP(hidden_layers=[64, 32], learning_rate=0.01,
                  n_iterations=500, activation='relu', batch_size=32,
                  task='regression', random_state=42)
    mlp_reg.fit(X_train_scaled, y_train)
    
    print(f"Training R²: {mlp_reg.score(X_train_scaled, y_train):.4f}")
    print(f"Test R²: {mlp_reg.score(X_test_scaled, y_test):.4f}")
    print(f"Final Loss: {mlp_reg.loss_history[-1]:.4f}")