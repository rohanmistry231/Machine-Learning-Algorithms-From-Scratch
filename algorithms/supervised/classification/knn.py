"""
K-Nearest Neighbors (KNN) Implementation from Scratch

KNN is a non-parametric algorithm that makes predictions based on
the k nearest training examples in the feature space.

Mathematical Foundation:
    For classification: Predict the most common class among k nearest neighbors
    For regression: Predict the average of k nearest neighbors
    
    Distance metrics:
    - Euclidean: d(x,y) = √(Σ(xᵢ - yᵢ)²)
    - Manhattan: d(x,y) = Σ|xᵢ - yᵢ|
    - Minkowski: d(x,y) = (Σ|xᵢ - yᵢ|ᵖ)^(1/p)
"""

import numpy as np
from collections import Counter
from typing import Union


class KNearestNeighbors:
    """
    K-Nearest Neighbors for classification and regression.
    
    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to use
    metric : str, default='euclidean'
        Distance metric: 'euclidean', 'manhattan', or 'minkowski'
    p : int, default=2
        Power parameter for Minkowski distance
    weights : str, default='uniform'
        Weight function: 'uniform' or 'distance'
    task : str, default='classification'
        Task type: 'classification' or 'regression'
        
    Attributes
    ----------
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training labels
    """
    
    def _init_(
        self,
        n_neighbors: int = 5,
        metric: str = 'euclidean',
        p: int = 2,
        weights: str = 'uniform',
        task: str = 'classification'
    ):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.p = p
        self.weights = weights
        self.task = task
        self.X_train = None
        self.y_train = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'KNearestNeighbors':
        """
        Fit the KNN model (store training data).
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data
        y : np.ndarray of shape (n_samples,)
            Target values
            
        Returns
        -------
        self : KNearestNeighbors
            Fitted estimator
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        return self
    
    def _compute_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Compute distance between two points based on specified metric.
        """
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        elif self.metric == 'minkowski':
            return np.power(np.sum(np.abs(x1 - x2) ** self.p), 1 / self.p)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
    
    def _get_neighbors(self, x: np.ndarray) -> tuple:
        """
        Find k nearest neighbors for a single point.
        
        Returns
        -------
        neighbor_labels : np.ndarray
            Labels of k nearest neighbors
        neighbor_distances : np.ndarray
            Distances to k nearest neighbors
        """
        # Compute distances to all training points
        distances = np.array([
            self._compute_distance(x, x_train) 
            for x_train in self.X_train
        ])
        
        # Get indices of k nearest neighbors
        k_indices = np.argsort(distances)[:self.n_neighbors]
        
        # Get labels and distances of k nearest neighbors
        neighbor_labels = self.y_train[k_indices]
        neighbor_distances = distances[k_indices]
        
        return neighbor_labels, neighbor_distances
    
    def _predict_single_classification(self, x: np.ndarray) -> Union[int, float]:
        """
        Predict class for a single sample.
        """
        neighbor_labels, neighbor_distances = self._get_neighbors(x)
        
        if self.weights == 'uniform':
            # Simple majority vote
            most_common = Counter(neighbor_labels).most_common(1)
            return most_common[0][0]
        
        elif self.weights == 'distance':
            # Weighted vote based on inverse distance
            # Add small epsilon to avoid division by zero
            weights = 1 / (neighbor_distances + 1e-10)
            
            # Weight each class by distance
            unique_labels = np.unique(neighbor_labels)
            weighted_votes = {}
            
            for label in unique_labels:
                mask = neighbor_labels == label
                weighted_votes[label] = np.sum(weights[mask])
            
            return max(weighted_votes, key=weighted_votes.get)
    
    def _predict_single_regression(self, x: np.ndarray) -> float:
        """
        Predict value for a single sample (regression).
        """
        neighbor_labels, neighbor_distances = self._get_neighbors(x)
        
        if self.weights == 'uniform':
            # Simple average
            return np.mean(neighbor_labels)
        
        elif self.weights == 'distance':
            # Weighted average based on inverse distance
            weights = 1 / (neighbor_distances + 1e-10)
            return np.sum(weights * neighbor_labels) / np.sum(weights)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions for input data.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data
            
        Returns
        -------
        predictions : np.ndarray of shape (n_samples,)
            Predicted values
        """
        X = np.array(X)
        
        if self.task == 'classification':
            predictions = [self._predict_single_classification(x) for x in X]
        else:  # regression
            predictions = [self._predict_single_regression(x) for x in X]
        
        return np.array(predictions)
    
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
        else:  # regression
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            ss_res = np.sum((y - y_pred) ** 2)
            return 1 - (ss_res / ss_tot)
    
    def get_params(self) -> dict:
        """Get model parameters."""
        return {
            'n_neighbors': self.n_neighbors,
            'metric': self.metric,
            'weights': self.weights,
            'task': self.task,
            'n_training_samples': len(self.X_train) if self.X_train is not None else 0
        }


if _name_ == "_main_":
    from sklearn.datasets import make_classification, make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    print("=== Classification Example ===")
    # Generate classification data
    X_clf, y_clf = make_classification(n_samples=200, n_features=5, 
                                       n_informative=3, n_redundant=1, 
                                       random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train KNN classifier
    knn_clf = KNearestNeighbors(n_neighbors=5, metric='euclidean', 
                                weights='distance', task='classification')
    knn_clf.fit(X_train_scaled, y_train)
    
    print(f"Training Accuracy: {knn_clf.score(X_train_scaled, y_train):.4f}")
    print(f"Test Accuracy: {knn_clf.score(X_test_scaled, y_test):.4f}")
    
    print("\n=== Regression Example ===")
    # Generate regression data
    X_reg, y_reg = make_regression(n_samples=200, n_features=5, noise=10, 
                                   random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )
    
    # Standardize features
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train KNN regressor
    knn_reg = KNearestNeighbors(n_neighbors=5, metric='euclidean', 
                               weights='distance', task='regression')
    knn_reg.fit(X_train_scaled, y_train)
    
    print(f"Training R²: {knn_reg.score(X_train_scaled, y_train):.4f}")
    print(f"Test R²: {knn_reg.score(X_test_scaled, y_test):.4f}")