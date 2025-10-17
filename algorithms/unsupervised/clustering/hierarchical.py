"""
Hierarchical Clustering Implementation from Scratch

Hierarchical clustering creates a tree of clusters (dendrogram) by
iteratively merging or splitting clusters based on distance metrics.

Mathematical Foundation:
    Agglomerative (Bottom-up):
    1. Start with each point as its own cluster
    2. Merge the two closest clusters
    3. Repeat until single cluster or desired number reached
    
    Linkage methods:
    - Single: min distance between any two points in clusters
    - Complete: max distance between any two points in clusters
    - Average: average distance between all pairs of points
    - Ward: minimizes within-cluster variance
"""

import numpy as np
from typing import Optional, List, Tuple


class HierarchicalClustering:
    """
    Agglomerative Hierarchical Clustering.
    
    Parameters
    ----------
    n_clusters : int, default=2
        Number of clusters to find
    linkage : str, default='average'
        Linkage criterion: 'single', 'complete', 'average', or 'ward'
    metric : str, default='euclidean'
        Distance metric: 'euclidean' or 'manhattan'
        
    Attributes
    ----------
    labels : np.ndarray of shape (n_samples,)
        Cluster labels for each sample
    n_clusters : int
        Number of clusters
    linkage_matrix : list
        Linkage matrix showing merge history
    """
    
    def _init_(
        self,
        n_clusters: int = 2,
        linkage: str = 'average',
        metric: str = 'euclidean'
    ):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.metric = metric
        self.labels = None
        self.linkage_matrix = []
    
    def _compute_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute distance between two points."""
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
    
    def _compute_cluster_distance(
        self,
        X: np.ndarray,
        cluster1: List[int],
        cluster2: List[int]
    ) -> float:
        """
        Compute distance between two clusters based on linkage method.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Data
        cluster1 : List[int]
            Indices of points in cluster 1
        cluster2 : List[int]
            Indices of points in cluster 2
            
        Returns
        -------
        distance : float
            Distance between clusters
        """
        if self.linkage == 'single':
            # Minimum distance between any two points
            min_dist = float('inf')
            for i in cluster1:
                for j in cluster2:
                    dist = self._compute_distance(X[i], X[j])
                    min_dist = min(min_dist, dist)
            return min_dist
        
        elif self.linkage == 'complete':
            # Maximum distance between any two points
            max_dist = 0
            for i in cluster1:
                for j in cluster2:
                    dist = self._compute_distance(X[i], X[j])
                    max_dist = max(max_dist, dist)
            return max_dist
        
        elif self.linkage == 'average':
            # Average distance between all pairs of points
            total_dist = 0
            count = 0
            for i in cluster1:
                for j in cluster2:
                    total_dist += self._compute_distance(X[i], X[j])
                    count += 1
            return total_dist / count if count > 0 else 0
        
        elif self.linkage == 'ward':
            # Ward's method: minimize within-cluster variance
            # Calculate centroids
            centroid1 = np.mean(X[cluster1], axis=0)
            centroid2 = np.mean(X[cluster2], axis=0)
            
            # Calculate merged centroid
            n1 = len(cluster1)
            n2 = len(cluster2)
            merged_centroid = (n1 * centroid1 + n2 * centroid2) / (n1 + n2)
            
            # Calculate increase in variance
            var_increase = 0
            for i in cluster1:
                var_increase += np.sum((X[i] - merged_centroid) ** 2)
            for j in cluster2:
                var_increase += np.sum((X[j] - merged_centroid) ** 2)
            
            # Subtract current within-cluster variance
            for i in cluster1:
                var_increase -= np.sum((X[i] - centroid1) ** 2)
            for j in cluster2:
                var_increase -= np.sum((X[j] - centroid2) ** 2)
            
            return var_increase
        
        else:
            raise ValueError(f"Unknown linkage: {self.linkage}")
    
    def fit(self, X: np.ndarray) -> 'HierarchicalClustering':
        """
        Perform hierarchical clustering.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data
            
        Returns
        -------
        self : HierarchicalClustering
            Fitted estimator
        """
        X = np.array(X)
        n_samples = X.shape[0]
        
        # Initialize: each point is its own cluster
        clusters = [[i] for i in range(n_samples)]
        
        # Merge clusters until we have n_clusters
        while len(clusters) > self.n_clusters:
            # Find the two closest clusters
            min_distance = float('inf')
            merge_i, merge_j = 0, 1
            
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    distance = self._compute_cluster_distance(X, clusters[i], clusters[j])
                    
                    if distance < min_distance:
                        min_distance = distance
                        merge_i, merge_j = i, j
            
            # Record merge in linkage matrix
            self.linkage_matrix.append({
                'clusters': (merge_i, merge_j),
                'distance': min_distance,
                'n_clusters': len(clusters)
            })
            
            # Merge the two closest clusters
            clusters[merge_i].extend(clusters[merge_j])
            clusters.pop(merge_j)
        
        # Assign cluster labels
        self.labels = np.zeros(n_samples, dtype=int)
        for cluster_id, cluster_indices in enumerate(clusters):
            for idx in cluster_indices:
                self.labels[idx] = cluster_id
        
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
            Cluster labels
        """
        self.fit(X)
        return self.labels
    
    def get_params(self) -> dict:
        """Get model parameters."""
        return {
            'n_clusters': self.n_clusters,
            'linkage': self.linkage,
            'metric': self.metric,
            'n_merges': len(self.linkage_matrix)
        }


if _name_ == "_main_":
    from sklearn.datasets import make_blobs
    import matplotlib.pyplot as plt
    
    print("=== Hierarchical Clustering with Different Linkages ===")
    
    # Generate sample data
    X, y_true = make_blobs(n_samples=150, centers=3, n_features=2,
                           cluster_std=0.5, random_state=42)
    
    linkages = ['single', 'complete', 'average', 'ward']
    
    for linkage_method in linkages:
        print(f"\n--- {linkage_method.upper()} Linkage ---")
        
        # Fit hierarchical clustering
        hc = HierarchicalClustering(n_clusters=3, linkage=linkage_method)
        labels = hc.fit_predict(X)
        
        # Check cluster distribution
        unique, counts = np.unique(labels, return_counts=True)
        print(f"Cluster sizes: {dict(zip(unique, counts))}")
        print(f"Number of merges performed: {hc.get_params()['n_merges']}")
    
    print("\n=== Hierarchical Clustering with Different n_clusters ===")
    for n_clust in [2, 3, 4, 5]:
        hc = HierarchicalClustering(n_clusters=n_clust, linkage='average')
        labels = hc.fit_predict(X)
        
        unique, counts = np.unique(labels, return_counts=True)
        print(f"\nn_clusters={n_clust}")
        print(f"Cluster sizes: {dict(zip(unique, counts))}")