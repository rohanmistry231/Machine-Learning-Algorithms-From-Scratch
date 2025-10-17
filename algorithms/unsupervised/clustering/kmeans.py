"""
K-Means Clustering Implementation from Scratch

K-Means is an unsupervised clustering algorithm that partitions
data into K clusters by minimizing within-cluster variance.

Mathematical Foundation:
    Objective: Minimize Σᵢ Σₓ∈Cᵢ ||x - μᵢ||²
    
    where:
    - Cᵢ is cluster i
    - μᵢ is the centroid of cluster i
    
    Algorithm:
    1. Initialize K centroids randomly
    2. Assign each point to nearest centroid
    3. Update centroids as mean of assigned points
    4. Repeat steps 2-3 until convergence
"""

import numpy as np
from typing import Optional, Tuple


class KMeans:
    """
    K-Means clustering algorithm.
    
    Parameters
    ----------
    n_clusters : int, default=3
        Number of clusters to form
    max_iters : int, default=300
        Maximum number of iterations
    tol : float, default=1e-4
        Tolerance for convergence (change in centroids)
    init : str, default='kmeans++'
        Initialization method: 'random' or 'kmeans++'
    random_state : int or None, default=None
        Random seed for reproducibility
        
    Attributes
    ----------
    centroids : np.ndarray
        Coordinates of cluster centers
    labels : np.ndarray
        Labels of each point
    inertia : float
        Sum of squared distances to closest cluster center
    n_iterations : int
        Number of iterations run
    """
    
    def __init__(
        self,
        n_clusters: int = 3,
        max_iters: int = 300,
        tol: float = 1e-4,
        init: str = 'kmeans++',
        random_state: Optional[int] = None
    ):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.init = init
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        self.inertia = None
        self.n_iterations = 0
    
    def _initialize_centroids(self, X: np.ndarray) -> np.ndarray:
        """
        Initialize centroids using specified method.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        n_samples, n_features = X.shape
        
        if self.init == 'random':
            # Randomly select k samples as initial centroids
            indices = np.random.choice(n_samples, self.n_clusters, replace=False)
            centroids = X[indices]
        
        elif self.init == 'kmeans++':
            # K-Means++ initialization
            centroids = np.zeros((self.n_clusters, n_features))
            
            # Choose first centroid randomly
            centroids[0] = X[np.random.choice(n_samples)]
            
            # Choose remaining centroids
            for i in range(1, self.n_clusters):
                # Compute distances to nearest existing centroid
                distances = np.array([
                    min([np.linalg.norm(x - c) ** 2 for c in centroids[:i]])
                    for x in X
                ])
                
                # Choose next centroid with probability proportional to distance²
                probabilities = distances / distances.sum()
                cumulative_probs = probabilities.cumsum()
                r = np.random.rand()
                
                for j, p in enumerate(cumulative_probs):
                    if r < p:
                        centroids[i] = X[j]
                        break
        else:
            raise ValueError(f"Unknown init method: {self.init}")
        
        return centroids
    
    def _assign_clusters(self, X: np.ndarray) -> np.ndarray:
        """
        Assign each sample to the nearest centroid.
        
        Returns
        -------
        labels : np.ndarray
            Cluster label for each sample
        """
        distances = np.zeros((X.shape[0], self.n_clusters))
        
        for i, centroid in enumerate(self.centroids):
            distances[:, i] = np.linalg.norm(X - centroid, axis=1)
        
        return np.argmin(distances, axis=1)
    
    def _update_centroids(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Update centroids as mean of assigned samples.
        
        Returns
        -------
        centroids : np.ndarray
            Updated centroids
        """
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        
        for i in range(self.n_clusters):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                centroids[i] = cluster_points.mean(axis=0)
            else:
                # If cluster is empty, reinitialize randomly
                centroids[i] = X[np.random.choice(X.shape[0])]
        
        return centroids
    
    def _compute_inertia(self, X: np.ndarray, labels: np.ndarray) -> float:
        """
        Compute sum of squared distances to closest cluster center.
        """
        inertia = 0
        for i in range(self.n_clusters):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - self.centroids[i]) ** 2)
        return inertia
    
    def fit(self, X: np.ndarray) -> 'KMeans':
        """
        Fit K-Means clustering.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data
            
        Returns
        -------
        self : KMeans
            Fitted estimator
        """
        X = np.array(X)
        
        # Initialize centroids
        self.centroids = self._initialize_centroids(X)
        
        # K-Means iterations
        for iteration in range(self.max_iters):
            # Assign clusters
            labels = self._assign_clusters(X)
            
            # Update centroids
            new_centroids = self._update_centroids(X, labels)
            
            # Check for convergence
            centroid_shift = np.linalg.norm(new_centroids - self.centroids)
            
            self.centroids = new_centroids
            self.n_iterations = iteration + 1
            
            if centroid_shift < self.tol:
                break
        
        # Final assignment
        self.labels = self._assign_clusters(X)
        self.inertia = self._compute_inertia(X, self.labels)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for samples.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            New data to predict
            
        Returns
        -------
        labels : np.ndarray of shape (n_samples,)
            Cluster labels
        """
        X = np.array(X)
        return self._assign_clusters(X)
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fit K-Means and return cluster labels.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data
            
        Returns
        -------
        labels : np.ndarray of shape (n_samples,)
            Cluster labels
        """
        self.fit(X)
        return self.labels
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform X to cluster-distance space.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            New data to transform
            
        Returns
        -------
        distances : np.ndarray of shape (n_samples, n_clusters)
            Distances to each cluster center
        """
        X = np.array(X)
        distances = np.zeros((X.shape[0], self.n_clusters))
        
        for i, centroid in enumerate(self.centroids):
            distances[:, i] = np.linalg.norm(X - centroid, axis=1)
        
        return distances
    
    def get_params(self) -> dict:
        """Get model parameters."""
        return {
            'centroids': self.centroids,
            'n_clusters': self.n_clusters,
            'inertia': self.inertia,
            'n_iterations': self.n_iterations
        }


if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    import matplotlib.pyplot as plt
    
    # Generate sample data
    X, y_true = make_blobs(n_samples=300, centers=4, n_features=2,
                          cluster_std=0.6, random_state=42)
    
    # Fit K-Means
    kmeans = KMeans(n_clusters=4, init='kmeans++', random_state=42)
    y_pred = kmeans.fit_predict(X)
    
    print("K-Means Clustering Results:")
    print(f"Number of iterations: {kmeans.n_iterations}")
    print(f"Inertia: {kmeans.inertia:.2f}")
    print(f"Centroids shape: {kmeans.centroids.shape}")
    
    # Visualize results
    plt.figure(figsize=(12, 5))
    
    # True labels
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', alpha=0.6)
    plt.title('True Labels')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    # Predicted labels
    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', alpha=0.6)
    plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1],
               c='red', marker='X', s=200, edgecolors='black', label='Centroids')
    plt.title('K-Means Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('kmeans_clustering.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved as 'kmeans_clustering.png'")