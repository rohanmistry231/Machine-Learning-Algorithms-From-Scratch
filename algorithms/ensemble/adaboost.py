"""
AdaBoost (Adaptive Boosting) Implementation from Scratch

AdaBoost is an ensemble method that combines multiple weak learners
(typically decision stumps) to create a strong classifier by adaptively
weighting training samples based on classification errors.

Mathematical Foundation:
    For m = 1 to M:
        1. Train weak learner hₘ on weighted data
        
        2. Compute weighted error:
           εₘ = Σᵢ wᵢ · I(hₘ(xᵢ) ≠ yᵢ) / Σᵢ wᵢ
        
        3. Compute classifier weight:
           αₘ = (1/2) · ln((1 - εₘ) / εₘ)
        
        4. Update sample weights:
           wᵢ := wᵢ · exp(-αₘ · yᵢ · hₘ(xᵢ))
        
        5. Normalize weights
    
    Final prediction: H(x) = sign(Σₘ αₘ · hₘ(x))
"""

import numpy as np
from typing import List, Optional


class DecisionStump:
    """
    Decision Stump (single-level decision tree) for AdaBoost.
    
    A decision stump makes a prediction based on a single feature threshold.
    """
    
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.polarity = 1  # Direction of inequality (1 or -1)
        self.alpha = None  # Weight of this stump
    
    def fit(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
        """
        Find best decision stump for weighted data.
        
        Returns
        -------
        min_error : float
            Minimum weighted error achieved
        """
        n_samples, n_features = X.shape
        min_error = float('inf')
        
        # Try each feature
        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            thresholds = np.unique(feature_values)
            
            # Try each threshold
            for threshold in thresholds:
                # Try both polarities
                for polarity in [1, -1]:
                    predictions = np.ones(n_samples)
                    if polarity == 1:
                        predictions[feature_values < threshold] = -1
                    else:
                        predictions[feature_values >= threshold] = -1
                    
                    # Calculate weighted error
                    misclassified = predictions != y
                    error = np.sum(weights[misclassified])
                    
                    # Update if best so far
                    if error < min_error:
                        min_error = error
                        self.feature_index = feature_idx
                        self.threshold = threshold
                        self.polarity = polarity
        
        return min_error
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the decision stump."""
        n_samples = X.shape[0]
        predictions = np.ones(n_samples)
        
        feature_values = X[:, self.feature_index]
        
        if self.polarity == 1:
            predictions[feature_values < self.threshold] = -1
        else:
            predictions[feature_values >= self.threshold] = -1
        
        return predictions


class AdaBoost:
    """
    AdaBoost classifier using decision stumps.
    
    Parameters
    ----------
    n_estimators : int, default=50
        Number of weak learners (decision stumps)
    learning_rate : float, default=1.0
        Weight applied to each classifier. Shrinks contribution of each stump
    random_state : int or None, default=None
        Random seed
        
    Attributes
    ----------
    stumps : List[DecisionStump]
        List of trained decision stumps
    alphas : List[float]
        Weights for each stump
    training_errors : List[float]
        Training error at each iteration
    """
    
    def __init__(
        self,
        n_estimators: int = 50,
        learning_rate: float = 1.0,
        random_state: Optional[int] = None
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.stumps = []
        self.alphas = []
        self.training_errors = []
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'AdaBoost':
        """
        Fit AdaBoost classifier.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data
        y : np.ndarray of shape (n_samples,)
            Target labels (must be -1 or 1)
            
        Returns
        -------
        self : AdaBoost
            Fitted estimator
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        X = np.array(X)
        y = np.array(y)
        
        # Convert labels to -1 and 1 if needed
        unique_labels = np.unique(y)
        if not np.array_equal(sorted(unique_labels), [-1, 1]):
            y = np.where(y == unique_labels[0], -1, 1)
        
        n_samples, n_features = X.shape
        
        # Initialize weights uniformly
        weights = np.ones(n_samples) / n_samples
        
        self.stumps = []
        self.alphas = []
        
        # Train weak learners
        for _ in range(self.n_estimators):
            # Train decision stump
            stump = DecisionStump()
            min_error = stump.fit(X, y, weights)
            
            # Avoid division by zero
            min_error = np.clip(min_error, 1e-10, 1 - 1e-10)
            
            # Calculate stump weight (alpha)
            alpha = 0.5 * np.log((1 - min_error) / min_error)
            alpha *= self.learning_rate  # Apply learning rate
            
            # Get predictions
            predictions = stump.predict(X)
            
            # Update weights
            weights *= np.exp(-alpha * y * predictions)
            # Normalize weights
            weights /= np.sum(weights)
            
            # Store stump and its weight
            stump.alpha = alpha
            self.stumps.append(stump)
            self.alphas.append(alpha)
            
            # Calculate training error
            ensemble_predictions = self._predict_ensemble(X, len(self.stumps))
            training_error = np.mean(ensemble_predictions != y)
            self.training_errors.append(training_error)
        
        return self
    
    def _predict_ensemble(self, X: np.ndarray, n_stumps: int) -> np.ndarray:
        """
        Make predictions using first n_stumps.
        
        Used for tracking training progress.
        """
        # Initialize predictions
        ensemble_pred = np.zeros(X.shape[0])
        
        # Add weighted predictions from each stump
        for i in range(n_stumps):
            ensemble_pred += self.alphas[i] * self.stumps[i].predict(X)
        
        # Return sign
        return np.sign(ensemble_pred)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
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
        return self._predict_ensemble(X, len(self.stumps))
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute decision function (weighted sum before sign).
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data
            
        Returns
        -------
        scores : np.ndarray of shape (n_samples,)
            Decision scores
        """
        X = np.array(X)
        scores = np.zeros(X.shape[0])
        
        for alpha, stump in zip(self.alphas, self.stumps):
            scores += alpha * stump.predict(X)
        
        return scores
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data
            
        Returns
        -------
        probabilities : np.ndarray of shape (n_samples, 2)
            Probability estimates for each class
        """
        scores = self.decision_function(X)
        
        # Convert to probabilities using sigmoid-like transformation
        proba_positive = 1 / (1 + np.exp(-2 * scores))
        proba_negative = 1 - proba_positive
        
        return np.column_stack([proba_negative, proba_positive])
    
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
        unique_labels = np.unique(y)
        if not np.array_equal(sorted(unique_labels), [-1, 1]):
            y = np.where(y == unique_labels[0], -1, 1)
        
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
    
    def staged_score(self, X: np.ndarray, y: np.ndarray) -> List[float]:
        """
        Return accuracy at each stage of boosting.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test data
        y : np.ndarray of shape (n_samples,)
            True labels
            
        Returns
        -------
        scores : List[float]
            Accuracy at each boosting iteration
        """
        y = np.array(y)
        
        # Convert labels to -1 and 1 if needed
        unique_labels = np.unique(y)
        if not np.array_equal(sorted(unique_labels), [-1, 1]):
            y = np.where(y == unique_labels[0], -1, 1)
        
        scores = []
        for i in range(1, len(self.stumps) + 1):
            y_pred = self._predict_ensemble(X, i)
            scores.append(np.mean(y_pred == y))
        
        return scores
    
    def get_params(self) -> dict:
        """Get model parameters."""
        return {
            'n_estimators': self.n_estimators,
            'n_stumps_trained': len(self.stumps),
            'learning_rate': self.learning_rate,
            'final_training_error': self.training_errors[-1] if self.training_errors else None,
            'stump_weights_range': (min(self.alphas), max(self.alphas)) if self.alphas else None
        }


if __name__ == "__main__":
    from sklearn.datasets import make_classification, load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt
    
    print("=== AdaBoost on Synthetic Data ===")
    # Generate binary classification data
    X, y = make_classification(n_samples=500, n_features=10, n_informative=5,
                               n_redundant=2, n_classes=2, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train AdaBoost
    adaboost = AdaBoost(n_estimators=50, learning_rate=1.0, random_state=42)
    adaboost.fit(X_train_scaled, y_train)
    
    print(f"Training Accuracy: {adaboost.score(X_train_scaled, y_train):.4f}")
    print(f"Test Accuracy: {adaboost.score(X_test_scaled, y_test):.4f}")
    print(f"Number of stumps: {adaboost.get_params()['n_stumps_trained']}")
    print(f"Final training error: {adaboost.get_params()['final_training_error']:.4f}")
    print(f"Stump weights range: {adaboost.get_params()['stump_weights_range']}")
    
    print("\n=== AdaBoost on Breast Cancer Dataset ===")
    # Load breast cancer dataset
    cancer = load_breast_cancer()
    X_cancer = cancer.data
    y_cancer = cancer.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_cancer, y_cancer, test_size=0.2, random_state=42
    )
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train with different learning rates
    learning_rates = [0.5, 1.0, 1.5]
    
    for lr in learning_rates:
        adaboost_lr = AdaBoost(n_estimators=50, learning_rate=lr, random_state=42)
        adaboost_lr.fit(X_train_scaled, y_train)
        
        train_acc = adaboost_lr.score(X_train_scaled, y_train)
        test_acc = adaboost_lr.score(X_test_scaled, y_test)
        
        print(f"\nLearning Rate: {lr}")
        print(f"  Training Accuracy: {train_acc:.4f}")
        print(f"  Test Accuracy: {test_acc:.4f}")
    
    print("\n=== Staged Predictions (Error vs. Iterations) ===")
    adaboost_staged = AdaBoost(n_estimators=50, learning_rate=1.0, random_state=42)
    adaboost_staged.fit(X_train_scaled, y_train)
    
    train_scores = adaboost_staged.staged_score(X_train_scaled, y_train)
    test_scores = adaboost_staged.staged_score(X_test_scaled, y_test)
    
    print("Iteration | Train Acc | Test Acc")
    print("-" * 35)
    for i in [0, 9, 19, 29, 39, 49]:
        if i < len(train_scores):
            print(f"{i+1:9d} | {train_scores[i]:9.4f} | {test_scores[i]:8.4f}")