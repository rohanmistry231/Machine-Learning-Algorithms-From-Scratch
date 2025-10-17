"""
Optimization Algorithms Implementation from Scratch

This module provides various gradient-based optimization algorithms
commonly used in machine learning.
"""

import numpy as np
from typing import Optional, Callable, Tuple


class GradientDescent:
    """
    Standard Gradient Descent optimizer.
    
    Update rule: θ := θ - η * ∇J(θ)
    
    Parameters
    ----------
    learning_rate : float, default=0.01
        Learning rate (step size)
    """
    
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
    
    def update(self, params: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        """
        Update parameters using gradient descent.
        
        Parameters
        ----------
        params : np.ndarray
            Current parameters
        gradients : np.ndarray
            Gradients of parameters
            
        Returns
        -------
        updated_params : np.ndarray
            Updated parameters
        """
        return params - self.learning_rate * gradients


class MomentumOptimizer:
    """
    Gradient Descent with Momentum.
    
    Update rules:
    v := β * v + η * ∇J(θ)
    θ := θ - v
    
    Parameters
    ----------
    learning_rate : float, default=0.01
        Learning rate
    momentum : float, default=0.9
        Momentum coefficient (β)
    """
    
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = None
    
    def update(self, params: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        """Update parameters using momentum."""
        if self.velocity is None:
            self.velocity = np.zeros_like(params)
        
        self.velocity = self.momentum * self.velocity + self.learning_rate * gradients
        return params - self.velocity
    
    def reset(self):
        """Reset velocity."""
        self.velocity = None


class NesterovMomentum:
    """
    Nesterov Accelerated Gradient (NAG).
    
    Update rules:
    v := β * v + η * ∇J(θ - β * v)
    θ := θ - v
    
    Parameters
    ----------
    learning_rate : float, default=0.01
        Learning rate
    momentum : float, default=0.9
        Momentum coefficient
    """
    
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = None
    
    def update(self, params: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        """Update parameters using Nesterov momentum."""
        if self.velocity is None:
            self.velocity = np.zeros_like(params)
        
        velocity_prev = self.velocity.copy()
        self.velocity = self.momentum * self.velocity + self.learning_rate * gradients
        
        # Nesterov update
        return params - self.momentum * velocity_prev - (1 + self.momentum) * self.velocity
    
    def reset(self):
        """Reset velocity."""
        self.velocity = None


class AdaGrad:
    """
    Adaptive Gradient Algorithm (AdaGrad).
    
    Update rules:
    G := G + (∇J(θ))²
    θ := θ - (η / √(G + ε)) * ∇J(θ)
    
    Parameters
    ----------
    learning_rate : float, default=0.01
        Learning rate
    epsilon : float, default=1e-8
        Small value to avoid division by zero
    """
    
    def __init__(self, learning_rate: float = 0.01, epsilon: float = 1e-8):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.accumulated_grad = None
    
    def update(self, params: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        """Update parameters using AdaGrad."""
        if self.accumulated_grad is None:
            self.accumulated_grad = np.zeros_like(params)
        
        # Accumulate squared gradients
        self.accumulated_grad += gradients ** 2
        
        # Adaptive learning rate
        adapted_lr = self.learning_rate / (np.sqrt(self.accumulated_grad) + self.epsilon)
        
        return params - adapted_lr * gradients
    
    def reset(self):
        """Reset accumulated gradients."""
        self.accumulated_grad = None


class RMSprop:
    """
    Root Mean Square Propagation (RMSprop).
    
    Update rules:
    E[g²] := β * E[g²] + (1 - β) * (∇J(θ))²
    θ := θ - (η / √(E[g²] + ε)) * ∇J(θ)
    
    Parameters
    ----------
    learning_rate : float, default=0.001
        Learning rate
    decay_rate : float, default=0.9
        Decay rate for moving average (β)
    epsilon : float, default=1e-8
        Small value to avoid division by zero
    """
    
    def __init__(
        self,
        learning_rate: float = 0.001,
        decay_rate: float = 0.9,
        epsilon: float = 1e-8
    ):
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.squared_grad = None
    
    def update(self, params: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        """Update parameters using RMSprop."""
        if self.squared_grad is None:
            self.squared_grad = np.zeros_like(params)
        
        # Moving average of squared gradients
        self.squared_grad = (self.decay_rate * self.squared_grad +
                            (1 - self.decay_rate) * gradients ** 2)
        
        # Adaptive learning rate
        adapted_lr = self.learning_rate / (np.sqrt(self.squared_grad) + self.epsilon)
        
        return params - adapted_lr * gradients
    
    def reset(self):
        """Reset squared gradients."""
        self.squared_grad = None


class Adam:
    """
    Adaptive Moment Estimation (Adam).
    
    Combines ideas from Momentum and RMSprop.
    
    Update rules:
    m := β₁ * m + (1 - β₁) * ∇J(θ)
    v := β₂ * v + (1 - β₂) * (∇J(θ))²
    m̂ := m / (1 - β₁ᵗ)
    v̂ := v / (1 - β₂ᵗ)
    θ := θ - η * m̂ / (√v̂ + ε)
    
    Parameters
    ----------
    learning_rate : float, default=0.001
        Learning rate (α)
    beta1 : float, default=0.9
        Exponential decay rate for first moment (β₁)
    beta2 : float, default=0.999
        Exponential decay rate for second moment (β₂)
    epsilon : float, default=1e-8
        Small value to avoid division by zero
    """
    
    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8
    ):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None  # First moment
        self.v = None  # Second moment
        self.t = 0     # Time step
    
    def update(self, params: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        """Update parameters using Adam."""
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        
        self.t += 1
        
        # Update biased first moment estimate
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradients
        
        # Update biased second moment estimate
        self.v = self.beta2 * self.v + (1 - self.beta2) * gradients ** 2
        
        # Bias correction
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        
        # Update parameters
        return params - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
    
    def reset(self):
        """Reset moment estimates."""
        self.m = None
        self.v = None
        self.t = 0


class AdaMax:
    """
    AdaMax optimizer (variant of Adam based on infinity norm).
    
    Update rules:
    m := β₁ * m + (1 - β₁) * ∇J(θ)
    u := max(β₂ * u, |∇J(θ)|)
    θ := θ - (η / (1 - β₁ᵗ)) * m / u
    
    Parameters
    ----------
    learning_rate : float, default=0.002
        Learning rate
    beta1 : float, default=0.9
        Exponential decay rate for first moment
    beta2 : float, default=0.999
        Exponential decay rate for infinity norm
    epsilon : float, default=1e-8
        Small value for numerical stability
    """
    
    def __init__(
        self,
        learning_rate: float = 0.002,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8
    ):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.u = None
        self.t = 0
    
    def update(self, params: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        """Update parameters using AdaMax."""
        if self.m is None:
            self.m = np.zeros_like(params)
            self.u = np.zeros_like(params)
        
        self.t += 1
        
        # Update first moment
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradients
        
        # Update infinity norm
        self.u = np.maximum(self.beta2 * self.u, np.abs(gradients))
        
        # Bias correction and update
        m_hat = self.m / (1 - self.beta1 ** self.t)
        
        return params - self.learning_rate * m_hat / (self.u + self.epsilon)
    
    def reset(self):
        """Reset moment estimates."""
        self.m = None
        self.u = None
        self.t = 0


class Nadam:
    """
    Nesterov-accelerated Adaptive Moment Estimation (Nadam).
    
    Combines Adam and Nesterov momentum.
    
    Parameters
    ----------
    learning_rate : float, default=0.001
        Learning rate
    beta1 : float, default=0.9
        Exponential decay rate for first moment
    beta2 : float, default=0.999
        Exponential decay rate for second moment
    epsilon : float, default=1e-8
        Small value to avoid division by zero
    """
    
    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8
    ):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0
    
    def update(self, params: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        """Update parameters using Nadam."""
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        
        self.t += 1
        
        # Update moments
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradients
        self.v = self.beta2 * self.v + (1 - self.beta2) * gradients ** 2
        
        # Bias correction
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        
        # Nesterov update
        m_bar = self.beta1 * m_hat + (1 - self.beta1) * gradients / (1 - self.beta1 ** self.t)
        
        return params - self.learning_rate * m_bar / (np.sqrt(v_hat) + self.epsilon)
    
    def reset(self):
        """Reset moment estimates."""
        self.m = None
        self.v = None
        self.t = 0


# ==================== LEARNING RATE SCHEDULES ====================

class LearningRateSchedule:
    """Base class for learning rate schedules."""
    
    def __call__(self, epoch: int) -> float:
        """Return learning rate for given epoch."""
        raise NotImplementedError


class StepDecay(LearningRateSchedule):
    """
    Step decay learning rate schedule.
    
    lr = initial_lr * drop_rate^floor(epoch / epochs_drop)
    """
    
    def __init__(self, initial_lr: float = 0.01, drop_rate: float = 0.5, epochs_drop: int = 10):
        self.initial_lr = initial_lr
        self.drop_rate = drop_rate
        self.epochs_drop = epochs_drop
    
    def __call__(self, epoch: int) -> float:
        return self.initial_lr * (self.drop_rate ** (epoch // self.epochs_drop))


class ExponentialDecay(LearningRateSchedule):
    """
    Exponential decay learning rate schedule.
    
    lr = initial_lr * exp(-decay_rate * epoch)
    """
    
    def __init__(self, initial_lr: float = 0.01, decay_rate: float = 0.1):
        self.initial_lr = initial_lr
        self.decay_rate = decay_rate
    
    def __call__(self, epoch: int) -> float:
        return self.initial_lr * np.exp(-self.decay_rate * epoch)


class CosineAnnealing(LearningRateSchedule):
    """
    Cosine annealing learning rate schedule.
    
    lr = min_lr + (max_lr - min_lr) * 0.5 * (1 + cos(π * epoch / max_epochs))
    """
    
    def __init__(self, max_lr: float = 0.01, min_lr: float = 0.0001, max_epochs: int = 100):
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.max_epochs = max_epochs
    
    def __call__(self, epoch: int) -> float:
        return self.min_lr + (self.max_lr - self.min_lr) * 0.5 * \
               (1 + np.cos(np.pi * epoch / self.max_epochs))


if __name__ == "__main__":
    print("=== Optimizer Examples ===\n")
    
    # Simulated parameters and gradients
    params = np.array([1.0, 2.0, 3.0])
    gradients = np.array([0.1, 0.2, 0.15])
    
    print("Initial parameters:", params)
    print("Gradients:", gradients)
    print()
    
    # Test each optimizer
    optimizers = {
        'SGD': GradientDescent(learning_rate=0.1),
        'Momentum': MomentumOptimizer(learning_rate=0.1, momentum=0.9),
        'AdaGrad': AdaGrad(learning_rate=0.1),
        'RMSprop': RMSprop(learning_rate=0.1),
        'Adam': Adam(learning_rate=0.1),
        'AdaMax': AdaMax(learning_rate=0.1),
        'Nadam': Nadam(learning_rate=0.1)
    }
    
    for name, optimizer in optimizers.items():
        updated_params = optimizer.update(params.copy(), gradients)
        print(f"{name:10s}: {updated_params}")
    
    print("\n=== Learning Rate Schedule Examples ===\n")
    
    schedules = {
        'Step Decay': StepDecay(initial_lr=0.1, drop_rate=0.5, epochs_drop=10),
        'Exponential': ExponentialDecay(initial_lr=0.1, decay_rate=0.1),
        'Cosine': CosineAnnealing(max_lr=0.1, min_lr=0.001, max_epochs=50)
    }
    
    epochs = [0, 10, 20, 30, 40, 50]
    
    for name, schedule in schedules.items():
        print(f"{name}:")
        for epoch in epochs:
            lr = schedule(epoch)
            print(f"  Epoch {epoch:2d}: lr = {lr:.6f}")
        print()