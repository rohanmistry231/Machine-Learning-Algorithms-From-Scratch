"""
Backpropagation Algorithm Implementation from Scratch

Backpropagation is the algorithm used to train neural networks by computing
gradients of the loss function with respect to weights using the chain rule.

Mathematical Foundation:
    Forward Pass:
    - Layer l: z⁽ˡ⁾ = W⁽ˡ⁾a⁽ˡ⁻¹⁾ + b⁽ˡ⁾
    - Activation: a⁽ˡ⁾ = σ(z⁽ˡ⁾)
    
    Backward Pass (Chain Rule):
    - Output layer: δ⁽ᴸ⁾ = ∂L/∂z⁽ᴸ⁾ = (a⁽ᴸ⁾ - y) ⊙ σ'(z⁽ᴸ⁾)
    - Hidden layer: δ⁽ˡ⁾ = [(W⁽ˡ⁺¹⁾)ᵀδ⁽ˡ⁺¹⁾] ⊙ σ'(z⁽ˡ⁾)
    
    Gradients:
    - ∂L/∂W⁽ˡ⁾ = δ⁽ˡ⁾(a⁽ˡ⁻¹⁾)ᵀ
    - ∂L/∂b⁽ˡ⁾ = δ⁽ˡ⁾
    
This is a detailed, educational implementation with step-by-step calculations.
"""

import numpy as np
from typing import List, Dict, Tuple, Callable


class ActivationFunctions:
    """Collection of activation functions and their derivatives."""
    
    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        """Sigmoid activation: σ(z) = 1 / (1 + e^(-z))"""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    @staticmethod
    def sigmoid_derivative(z: np.ndarray) -> np.ndarray:
        """Derivative of sigmoid: σ'(z) = σ(z)(1 - σ(z))"""
        s = ActivationFunctions.sigmoid(z)
        return s * (1 - s)
    
    @staticmethod
    def tanh(z: np.ndarray) -> np.ndarray:
        """Tanh activation: tanh(z)"""
        return np.tanh(z)
    
    @staticmethod
    def tanh_derivative(z: np.ndarray) -> np.ndarray:
        """Derivative of tanh: 1 - tanh²(z)"""
        return 1 - np.tanh(z) ** 2
    
    @staticmethod
    def relu(z: np.ndarray) -> np.ndarray:
        """ReLU activation: max(0, z)"""
        return np.maximum(0, z)
    
    @staticmethod
    def relu_derivative(z: np.ndarray) -> np.ndarray:
        """Derivative of ReLU: 1 if z > 0, else 0"""
        return (z > 0).astype(float)
    
    @staticmethod
    def leaky_relu(z: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        """Leaky ReLU: max(αz, z)"""
        return np.where(z > 0, z, alpha * z)
    
    @staticmethod
    def leaky_relu_derivative(z: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        """Derivative of Leaky ReLU: 1 if z > 0, else α"""
        return np.where(z > 0, 1, alpha)
    
    @staticmethod
    def softmax(z: np.ndarray) -> np.ndarray:
        """Softmax activation for multi-class classification"""
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)


class LossFunctions:
    """Collection of loss functions and their derivatives."""
    
    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Squared Error"""
        return np.mean((y_pred - y_true) ** 2)
    
    @staticmethod
    def mse_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Derivative of MSE"""
        return (y_pred - y_true) / y_true.shape[0]
    
    @staticmethod
    def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Binary Cross-Entropy Loss"""
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    @staticmethod
    def binary_cross_entropy_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Derivative of Binary Cross-Entropy"""
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -(y_true / y_pred - (1 - y_true) / (1 - y_pred)) / y_true.shape[0]
    
    @staticmethod
    def categorical_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Categorical Cross-Entropy Loss"""
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))


class BackpropagationNN:
    """
    Neural Network with detailed backpropagation implementation.
    
    This implementation shows step-by-step calculations for educational purposes.
    
    Parameters
    ----------
    layer_sizes : List[int]
        List of layer sizes including input and output layers
        Example: [784, 128, 64, 10] for 784 inputs, two hidden layers, 10 outputs
    activation : str, default='relu'
        Activation function: 'sigmoid', 'tanh', 'relu', 'leaky_relu'
    output_activation : str, default='sigmoid'
        Output layer activation: 'sigmoid', 'softmax', 'linear'
    loss_function : str, default='mse'
        Loss function: 'mse', 'binary_cross_entropy', 'categorical_cross_entropy'
    learning_rate : float, default=0.01
        Learning rate
    batch_size : int, default=32
        Mini-batch size
    epochs : int, default=100
        Number of training epochs
    random_state : int or None, default=None
        Random seed
    verbose : bool, default=True
        Whether to print training progress
        
    Attributes
    ----------
    weights : List[np.ndarray]
        Weight matrices for each layer
    biases : List[np.ndarray]
        Bias vectors for each layer
    training_loss_history : List[float]
        Loss history during training
    """
    
    def __init__(
        self,
        layer_sizes: List[int],
        activation: str = 'relu',
        output_activation: str = 'sigmoid',
        loss_function: str = 'mse',
        learning_rate: float = 0.01,
        batch_size: int = 32,
        epochs: int = 100,
        random_state: int = None,
        verbose: bool = True
    ):
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.output_activation = output_activation
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.random_state = random_state
        self.verbose = verbose
        
        self.weights = []
        self.biases = []
        self.training_loss_history = []
        
        # Initialize activation and loss functions
        self._setup_functions()
        
        # Initialize network parameters
        self._initialize_parameters()
    
    def _setup_functions(self) -> None:
        """Setup activation and loss function references."""
        act_funcs = ActivationFunctions()
        
        # Hidden layer activation
        if self.activation == 'sigmoid':
            self.activation_func = act_funcs.sigmoid
            self.activation_derivative = act_funcs.sigmoid_derivative
        elif self.activation == 'tanh':
            self.activation_func = act_funcs.tanh
            self.activation_derivative = act_funcs.tanh_derivative
        elif self.activation == 'relu':
            self.activation_func = act_funcs.relu
            self.activation_derivative = act_funcs.relu_derivative
        elif self.activation == 'leaky_relu':
            self.activation_func = act_funcs.leaky_relu
            self.activation_derivative = act_funcs.leaky_relu_derivative
        
        # Output layer activation
        if self.output_activation == 'sigmoid':
            self.output_func = act_funcs.sigmoid
            self.output_derivative = act_funcs.sigmoid_derivative
        elif self.output_activation == 'softmax':
            self.output_func = act_funcs.softmax
            self.output_derivative = None  # Special handling for softmax
        elif self.output_activation == 'linear':
            self.output_func = lambda z: z
            self.output_derivative = lambda z: np.ones_like(z)
        
        # Loss function
        loss_funcs = LossFunctions()
        if self.loss_function == 'mse':
            self.loss_func = loss_funcs.mse
            self.loss_derivative = loss_funcs.mse_derivative
        elif self.loss_function == 'binary_cross_entropy':
            self.loss_func = loss_funcs.binary_cross_entropy
            self.loss_derivative = loss_funcs.binary_cross_entropy_derivative
        elif self.loss_function == 'categorical_cross_entropy':
            self.loss_func = loss_funcs.categorical_cross_entropy
            # For categorical cross-entropy with softmax, use simplified derivative
            self.loss_derivative = None
    
    def _initialize_parameters(self) -> None:
        """Initialize weights and biases using He initialization."""
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        for i in range(len(self.layer_sizes) - 1):
            # He initialization for weights
            limit = np.sqrt(2.0 / self.layer_sizes[i])
            weight = np.random.randn(self.layer_sizes[i], self.layer_sizes[i + 1]) * limit
            bias = np.zeros((1, self.layer_sizes[i + 1]))
            
            self.weights.append(weight)
            self.biases.append(bias)
    
    def _forward_pass(self, X: np.ndarray) -> Dict[str, List[np.ndarray]]:
        """
        Perform forward pass through the network.
        
        Returns dictionary with:
        - 'z_values': pre-activation values for each layer
        - 'activations': post-activation values for each layer
        """
        cache = {
            'z_values': [],
            'activations': [X]  # Input layer activation
        }
        
        current_activation = X
        
        # Forward pass through hidden layers
        for i in range(len(self.weights) - 1):
            z = np.dot(current_activation, self.weights[i]) + self.biases[i]
            cache['z_values'].append(z)
            
            a = self.activation_func(z)
            cache['activations'].append(a)
            
            current_activation = a
        
        # Output layer
        z_out = np.dot(current_activation, self.weights[-1]) + self.biases[-1]
        cache['z_values'].append(z_out)
        
        a_out = self.output_func(z_out)
        cache['activations'].append(a_out)
        
        return cache
    
    def _backward_pass(
        self,
        y_true: np.ndarray,
        cache: Dict[str, List[np.ndarray]]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Perform backward pass (backpropagation).
        
        Returns gradients for weights and biases.
        """
        m = y_true.shape[0]  # Batch size
        
        weight_gradients = []
        bias_gradients = []
        
        # Output layer error
        y_pred = cache['activations'][-1]
        
        # Special case: Categorical cross-entropy with softmax
        if self.loss_function == 'categorical_cross_entropy' and \
           self.output_activation == 'softmax':
            delta = y_pred - y_true
        else:
            # General case
            loss_grad = self.loss_derivative(y_true, y_pred)
            if self.output_derivative is not None:
                delta = loss_grad * self.output_derivative(cache['z_values'][-1])
            else:
                delta = loss_grad
        
        # Backpropagate through layers
        for i in range(len(self.weights) - 1, -1, -1):
            # Compute gradients
            dW = np.dot(cache['activations'][i].T, delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m
            
            weight_gradients.insert(0, dW)
            bias_gradients.insert(0, db)
            
            # Propagate error to previous layer (if not input layer)
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * \
                       self.activation_derivative(cache['z_values'][i - 1])
        
        return weight_gradients, bias_gradients
    
    def _update_parameters(
        self,
        weight_gradients: List[np.ndarray],
        bias_gradients: List[np.ndarray]
    ) -> None:
        """Update weights and biases using gradient descent."""
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * weight_gradients[i]
            self.biases[i] -= self.learning_rate * bias_gradients[i]
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BackpropagationNN':
        """
        Train the neural network using backpropagation.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data
        y : np.ndarray of shape (n_samples,) or (n_samples, n_outputs)
            Target values
            
        Returns
        -------
        self : BackpropagationNN
            Fitted neural network
        """
        X = np.array(X)
        y = np.array(y)
        
        # Reshape y if needed
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        n_samples = X.shape[0]
        
        # Training loop
        for epoch in range(self.epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            n_batches = 0
            
            # Mini-batch training
            for start_idx in range(0, n_samples, self.batch_size):
                end_idx = min(start_idx + self.batch_size, n_samples)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Forward pass
                cache = self._forward_pass(X_batch)
                
                # Compute loss
                batch_loss = self.loss_func(y_batch, cache['activations'][-1])
                epoch_loss += batch_loss
                n_batches += 1
                
                # Backward pass
                weight_grads, bias_grads = self._backward_pass(y_batch, cache)
                
                # Update parameters
                self._update_parameters(weight_grads, bias_grads)
            
            # Record average epoch loss
            avg_loss = epoch_loss / n_batches
            self.training_loss_history.append(avg_loss)
            
            # Print progress
            if self.verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.6f}")
        
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
            Predicted values
        """
        X = np.array(X)
        cache = self._forward_pass(X)
        predictions = cache['activations'][-1]
        
        # For classification, return class labels
        if self.output_activation == 'softmax':
            return np.argmax(predictions, axis=1)
        elif self.output_activation == 'sigmoid' and predictions.shape[1] == 1:
            return (predictions >= 0.5).astype(int).flatten()
        
        return predictions.flatten()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities (for classification)."""
        X = np.array(X)
        cache = self._forward_pass(X)
        return cache['activations'][-1]
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate accuracy (classification) or R² (regression).
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test data
        y : np.ndarray of shape (n_samples,)
            True values
            
        Returns
        -------
        score : float
            Accuracy or R² score
        """
        if self.output_activation in ['sigmoid', 'softmax']:
            # Classification accuracy
            y_pred = self.predict(X)
            return np.mean(y_pred == y)
        else:
            # Regression R²
            y_pred = self.predict(X)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            ss_res = np.sum((y - y_pred) ** 2)
            return 1 - (ss_res / ss_tot)
    
    def get_params(self) -> dict:
        """Get model parameters and architecture."""
        return {
            'layer_sizes': self.layer_sizes,
            'n_parameters': sum(w.size + b.size for w, b in zip(self.weights, self.biases)),
            'activation': self.activation,
            'output_activation': self.output_activation,
            'loss_function': self.loss_function,
            'training_loss_history': self.training_loss_history
        }


if __name__ == "__main__":
    from sklearn.datasets import load_digits, make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    print("=== Binary Classification Example ===")
    # Generate binary classification data
    X_bin, y_bin = make_classification(n_samples=1000, n_features=20,
                                       n_informative=15, n_classes=2,
                                       random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_bin, y_bin, test_size=0.2, random_state=42
    )
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train binary classifier
    nn_binary = BackpropagationNN(
        layer_sizes=[20, 16, 8, 1],
        activation='relu',
        output_activation='sigmoid',
        loss_function='binary_cross_entropy',
        learning_rate=0.01,
        batch_size=32,
        epochs=100,
        verbose=False,
        random_state=42
    )
    
    nn_binary.fit(X_train_scaled, y_train)
    
    print(f"Training Accuracy: {nn_binary.score(X_train_scaled, y_train):.4f}")
    print(f"Test Accuracy: {nn_binary.score(X_test_scaled, y_test):.4f}")
    print(f"Final Loss: {nn_binary.training_loss_history[-1]:.6f}")
    print(f"Total Parameters: {nn_binary.get_params()['n_parameters']}")
    
    print("\n=== Multi-class Classification Example ===")
    # Load digits dataset (10 classes)
    digits = load_digits()
    X_digits = digits.data
    y_digits = digits.target
    
    # One-hot encode targets for softmax
    n_classes = 10
    y_digits_onehot = np.eye(n_classes)[y_digits]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_digits, y_digits_onehot, test_size=0.2, random_state=42
    )
    y_train_labels = np.argmax(y_train, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train multi-class classifier
    nn_multi = BackpropagationNN(
        layer_sizes=[64, 32, 16, 10],
        activation='relu',
        output_activation='softmax',
        loss_function='categorical_cross_entropy',
        learning_rate=0.01,
        batch_size=32,
        epochs=100,
        verbose=False,
        random_state=42
    )
    
    nn_multi.fit(X_train_scaled, y_train)
    
    print(f"Training Accuracy: {nn_multi.score(X_train_scaled, y_train_labels):.4f}")
    print(f"Test Accuracy: {nn_multi.score(X_test_scaled, y_test_labels):.4f}")
    print(f"Final Loss: {nn_multi.training_loss_history[-1]:.6f}")
    print(f"Network Architecture: {nn_multi.get_params()['layer_sizes']}")