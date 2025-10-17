"""
DBSCAN (Density-Based Spatial Clustering) Implementation from Scratch

DBSCAN groups together points that are closely packed (density-based),
marking points in low-density regions as outliers.

Mathematical Foundation:
    Core Point: Point with at least min_samples within epsilon radius
    Border Point: Point within epsilon of a core point but not itself core
    Noise Point: Point that is neither core nor border
    
    Directly Density-Reachable: Point q is directly density-reachable from p
    if p is a core point and q is within epsilon distance
    
    Density-Reachable: Point q is density-reachable from p if there exists
    a chain of directly density-reachable points from p to q
"""

import numpy as np
from typing import List, Set


class DBSCAN:
    """
    DBSCAN clustering algorithm.
    
    Parameters
    ----------
    eps : float, default=0.5
        Maximum distance between two samples for one to be considered
        as in the neighborhood of the other (epsilon)
    min_samples : int, default=5
        Minimum number of samples in a neighborhood for a point to be
        considered as a core point
    metric : str, default='euclidean'
        Distance metric: 'euclidean' or 'manhattan'
        
    Attributes
    ----------
    labels : np.ndarray of shape (n_samples,)
        Cluster labels for each sample. Noisy samples are given label -1
    core_sample_indices : np.ndarray
        Indices of core samples
    n_clusters : int
        Number of clusters found (excluding noise)
    """
    
    def _init_(
        self,
        eps: float = 0.5,
        min_samples: int = 5,
        metric: str = 'euclidean'
    ):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.labels = None
        self.core_sample_indices = None
        self.n_clusters = 0
    
    def _compute_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute distance between two points."""
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
    
    def _get_neighbors(self, X: np.ndarray, point_idx: int) -> List[int]:
        """
        Find all neighbors within eps distance of a point.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Data
        point_idx : int
            Index of the point
            
        Returns
        -------
        neighbors : List[int]
            Indices of neighboring points
        """
        neighbors = []
        
        for idx in range(len(X)):
            if self._compute_distance(X[point_idx], X[idx]) <= self.eps:
                neighbors.append(idx)
        
        return neighbors
    
    def _expand_cluster(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        point_idx: int,
        neighbors: List[int],
        cluster_id: int
    ) -> None:
        """
        Expand cluster by adding density-reachable points.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Data
        labels : np.ndarray of shape (n_samples,)
            Current cluster labels
        point_idx : int
            Index of the seed point
        neighbors : List[int]
            Initial neighbors of the seed point
        cluster_id : int
            ID of current cluster
        """
        # Add seed point to cluster
        labels[point_idx] = cluster_id
        
        # Process neighbors using a queue
        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]
            
            # If neighbor was noise, add it to cluster
            if labels[neighbor_idx] == -1:
                labels[neighbor_idx] = cluster_id
            
            # If neighbor is unprocessed
            if labels[neighbor_idx] == 0:
                labels[neighbor_idx] = cluster_id
                
                # Find neighbors of this neighbor
                neighbor_neighbors = self._get_neighbors(X, neighbor_idx)
                
                # If it's a core point, add its neighbors to queue
                if len(neighbor_neighbors) >= self.min_samples:
                    neighbors.extend(neighbor_neighbors)
            
            i += 1
    
    def fit(self, X: np.ndarray) -> 'DBSCAN':
        """
        Perform DBSCAN clustering.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data
            
        Returns
        -------
        self : DBSCAN
            Fitted estimator
        """
        X = np.array(X)
        n_samples = X.shape[0]
        
        # Initialize all points as unprocessed (label 0)
        labels = np.zeros(n_samples, dtype=int)
        cluster_id = 0
        core_samples = []
        
        # Process each point
        for point_idx in range(n_samples):
            # Skip if already processed
            if labels[point_idx] != 0:
                continue
            
            # Find neighbors
            neighbors = self._get_neighbors(X, point_idx)
            
            # Check if core point
            if len(neighbors) < self.min_samples:
                # Mark as noise (will be reconsidered if reached from core point)
                labels[point_idx] = -1
            else:
                # Start new cluster
                cluster_id += 1
                core_samples.append(point_idx)
                self._expand_cluster(X, labels, point_idx, neighbors, cluster_id)
        
        self.labels = labels
        self.core_sample_indices = np.array(core_samples)
        self.n_clusters = cluster_id
        
        return self
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Perform clustering and return labels.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data
            
        Returns
        -------
        labels : np.ndarray of shape (n_samples,)
            Cluster labels (-1 for noise)
        """
        self.fit(X)
        return self.labels
    
    def get_params(self) -> dict:
        """Get model parameters."""
        return {
            'eps': self.eps,
            'min_samples': self.min_samples,
            'n_clusters': self.n_clusters,
            'n_core_samples': len(self.core_sample_indices) if self.core_sample_indices is not None else 0,
            'n_noise_samples': np.sum(self.labels == -1) if self.labels is not None else 0
        }


if _name_ == "_main_":
    from sklearn.datasets import make_moons, make_blobs
    import matplotlib.pyplot as plt
    
    print("=== DBSCAN on Moon-shaped Data ===")
    # Generate moon-shaped data (challenging for K-means)
    X_moons, _ = make_moons(n_samples=300, noise=0.05, random_state=42)
    
    # Fit DBSCAN
    dbscan = DBSCAN(eps=0.3, min_samples=5)
    labels = dbscan.fit_predict(X_moons)
    
    print(f"Number of clusters: {dbscan.n_clusters}")
    print(f"Number of noise points: {dbscan.get_params()['n_noise_samples']}")
    print(f"Number of core samples: {dbscan.get_params()['n_core_samples']}")
    
    print("\n=== DBSCAN on Blobs with Varying Density ===")
    # Generate blobs with different densities
    X_blobs, _ = make_blobs(n_samples=300, centers=3, n_features=2,
                            cluster_std=[1.0, 2.5, 0.5], random_state=42)
    
    dbscan_blobs = DBSCAN(eps=0.5, min_samples=5)
    labels_blobs = dbscan_blobs.fit_predict(X_blobs)
    
    print(f"Number of clusters: {dbscan_blobs.n_clusters}")
    print(f"Number of noise points: {dbscan_blobs.get_params()['n_noise_samples']}")
    print(f"Number of core samples: {dbscan_blobs.get_params()['n_core_samples']}")
    
    # Show unique labels
    unique_labels = np.unique(labels_blobs)
    print(f"Unique cluster labels: {unique_labels}")