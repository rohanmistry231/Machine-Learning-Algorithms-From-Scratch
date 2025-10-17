"""
Principal Component Analysis (PCA) Implementation from Scratch

PCA is a dimensionality reduction technique that transforms data to a new
coordinate system where the greatest variance lies on the first coordinate
(principal component), the second greatest variance on the second coordinate, etc.

Mathematical Foundation:
    1. Standardize the data: X_std = (X - μ) / σ
    2. Compute covariance matrix: Σ = (1/(n-1)) * X_stdᵀ * X_std
    3. Compute eigenvalues and eigenvectors of Σ
    4. Sort eigenvectors by eigenvalues (descending)
    5. Project data: X_transformed = X_std * W
    
    where W contains the top k eigenvectors (principal components)
"""

import numpy as np
from typing import Optional


class PCA:
    """
    Principal Component Analysis for dimensionality reduction.
    
    Parameters
    ----------
    n_components : int or None, default=None
        Number of components to keep. If None, keep all components
    whiten : bool, default=False
        When True, multiply outputs by sqrt(n_samples) and divide by
        singular values to ensure uncorrelated outputs with unit variance
        
    Attributes
    ----------
    components : np.ndarray of shape (n_components, n_features)
        Principal axes in feature space (eigenvectors)
    explained_variance : np.ndarray of shape (n_components,)
        Amount of variance explained by each component (eigenvalues)
    explained_variance_ratio : np.ndarray of shape (n_components,)
        Percentage of variance explained by each component
    mean : np.ndarray of shape (n_features,)
        Per-feature empirical mean
    n_components : int
        Number of components
    n_features : int
        Number of features in training data
    """
    
    def _init_(
        self,
        n_components: Optional[int] = None,
        whiten: bool = False
    ):
        self.n_components = n_components
        self.whiten = whiten
        self.components = None
        self.explained_variance = None
        self.explained_variance_ratio = None
        self.mean = None
        self.n_features = None
    
    def fit(self, X: np.ndarray) -> 'PCA':
        """
        Fit PCA model.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data
            
        Returns
        -------
        self : PCA
            Fitted estimator
        """
        X = np.array(X)
        n_samples, n_features = X.shape
        
        self.n_features = n_features
        
        # Set n_components if not specified
        if self.n_components is None:
            self.n_components = min(n_samples, n_features)
        
        # Center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        # Compute covariance matrix
        # Cov = (1/(n-1)) * X^T * X
        cov_matrix = np.cov(X_centered, rowvar=False)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # Ensure eigenvalues are real (numerical stability)
        eigenvalues = np.real(eigenvalues)
        eigenvectors = np.real(eigenvectors)
        
        # Sort eigenvectors by eigenvalues in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Select top n_components
        self.components = eigenvectors[:, :self.n_components].T
        self.explained_variance = eigenvalues[:self.n_components]
        
        # Calculate explained variance ratio
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio = self.explained_variance / total_variance
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data to principal component space.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Data to transform
            
        Returns
        -------
        X_transformed : np.ndarray of shape (n_samples, n_components)
            Transformed data
        """
        X = np.array(X)
        
        # Center the data
        X_centered = X - self.mean
        
        # Project data onto principal components
        X_transformed = np.dot(X_centered, self.components.T)
        
        if self.whiten:
            # Whiten the data: divide by sqrt(eigenvalues)
            X_transformed /= np.sqrt(self.explained_variance)
        
        return X_transformed
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit PCA and transform data.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data
            
        Returns
        -------
        X_transformed : np.ndarray of shape (n_samples, n_components)
            Transformed data
        """
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        """
        Transform data back to original space.
        
        Parameters
        ----------
        X_transformed : np.ndarray of shape (n_samples, n_components)
            Transformed data
            
        Returns
        -------
        X_reconstructed : np.ndarray of shape (n_samples, n_features)
            Reconstructed data in original space
        """
        X_transformed = np.array(X_transformed)
        
        if self.whiten:
            # Reverse whitening
            X_transformed = X_transformed * np.sqrt(self.explained_variance)
        
        # Project back to original space
        X_reconstructed = np.dot(X_transformed, self.components) + self.mean
        
        return X_reconstructed
    
    def get_covariance(self) -> np.ndarray:
        """
        Compute data covariance with the generative model.
        
        Returns
        -------
        covariance : np.ndarray of shape (n_features, n_features)
            Estimated covariance of data
        """
        components = self.components
        explained_variance = self.explained_variance
        
        # Covariance = components^T * diag(explained_variance) * components
        covariance = np.dot(
            components.T * explained_variance,
            components
        )
        
        return covariance
    
    def get_params(self) -> dict:
        """Get model parameters."""
        return {
            'n_components': self.n_components,
            'n_features': self.n_features,
            'explained_variance': self.explained_variance,
            'explained_variance_ratio': self.explained_variance_ratio,
            'cumulative_variance_ratio': np.cumsum(self.explained_variance_ratio),
            'components_shape': self.components.shape if self.components is not None else None
        }


if _name_ == "_main_":
    from sklearn.datasets import load_iris, load_digits
    from sklearn.preprocessing import StandardScaler
    
    print("=== PCA on Iris Dataset ===")
    # Load iris dataset
    iris = load_iris()
    X_iris = iris.data
    y_iris = iris.target
    
    # Standardize features
    scaler = StandardScaler()
    X_iris_scaled = scaler.fit_transform(X_iris)
    
    # Apply PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_iris_scaled)
    
    print(f"Original shape: {X_iris.shape}")
    print(f"Transformed shape: {X_pca.shape}")
    print(f"Explained variance ratio: {pca.explained_variance_ratio}")
    print(f"Cumulative variance explained: {np.sum(pca.explained_variance_ratio):.4f}")
    
    print("\n=== PCA on Digits Dataset ===")
    # Load digits dataset (64 features)
    digits = load_digits()
    X_digits = digits.data
    
    # Standardize features
    X_digits_scaled = scaler.fit_transform(X_digits)
    
    # Apply PCA to reduce to 10 components
    pca_digits = PCA(n_components=10)
    X_digits_pca = pca_digits.fit_transform(X_digits_scaled)
    
    print(f"Original shape: {X_digits.shape}")
    print(f"Transformed shape: {X_digits_pca.shape}")
    print(f"Explained variance by top 10 components: {np.sum(pca_digits.explained_variance_ratio):.4f}")
    
    # Reconstruction
    X_reconstructed = pca_digits.inverse_transform(X_digits_pca)
    reconstruction_error = np.mean((X_digits_scaled - X_reconstructed) ** 2)
    print(f"Reconstruction error (MSE): {reconstruction_error:.6f}")
    
    print("\n=== Component-wise Variance ===")
    for i, var_ratio in enumerate(pca_digits.explained_variance_ratio):
        print(f"PC{i+1}: {var_ratio:.4f} ({var_ratio*100:.2f}%)")