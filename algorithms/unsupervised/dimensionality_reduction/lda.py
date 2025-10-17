"""
Linear Discriminant Analysis (LDA) Implementation from Scratch

LDA is a supervised dimensionality reduction technique that projects data
to maximize class separability. Unlike PCA, LDA uses class labels.

Mathematical Foundation:
    Objective: Maximize the ratio of between-class variance to within-class variance
    
    Between-class scatter matrix: S_B = Σᵢ nᵢ(μᵢ - μ)(μᵢ - μ)ᵀ
    Within-class scatter matrix: S_W = Σᵢ Σₓ∈Cᵢ (x - μᵢ)(x - μᵢ)ᵀ
    
    Find eigenvectors of S_W⁻¹S_B
    
    Maximum components: min(n_classes - 1, n_features)
"""

import numpy as np
from typing import Optional


class LDA:
    """
    Linear Discriminant Analysis for dimensionality reduction.
    
    Parameters
    ----------
    n_components : int or None, default=None
        Number of components to keep. If None, keep min(n_classes-1, n_features)
        
    Attributes
    ----------
    components : np.ndarray of shape (n_components, n_features)
        Linear discriminants (projection vectors)
    explained_variance_ratio : np.ndarray of shape (n_components,)
        Percentage of variance explained by each discriminant
    means : np.ndarray of shape (n_classes, n_features)
        Class means
    priors : np.ndarray of shape (n_classes,)
        Class priors
    classes : np.ndarray of shape (n_classes,)
        Unique class labels
    """
    
    def __init__(self, n_components: Optional[int] = None):
        self.n_components = n_components
        self.components = None
        self.explained_variance_ratio = None
        self.means = None
        self.priors = None
        self.classes = None
        self.overall_mean = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LDA':
        """
        Fit LDA model.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data
        y : np.ndarray of shape (n_samples,)
            Target labels
            
        Returns
        -------
        self : LDA
            Fitted estimator
        """
        X = np.array(X)
        y = np.array(y)
        
        n_samples, n_features = X.shape
        
        # Get unique classes
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
        # Set n_components if not specified
        if self.n_components is None:
            self.n_components = min(n_classes - 1, n_features)
        
        # Ensure n_components is valid
        max_components = min(n_classes - 1, n_features)
        if self.n_components > max_components:
            raise ValueError(
                f"n_components ({self.n_components}) cannot be larger than "
                f"min(n_classes - 1, n_features) = {max_components}"
            )
        
        # Calculate overall mean
        self.overall_mean = np.mean(X, axis=0)
        
        # Calculate class means and priors
        self.means = np.zeros((n_classes, n_features))
        self.priors = np.zeros(n_classes)
        
        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.means[idx] = np.mean(X_c, axis=0)
            self.priors[idx] = len(X_c) / n_samples
        
        # Calculate within-class scatter matrix (S_W)
        S_W = np.zeros((n_features, n_features))
        
        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            # Center class data
            X_c_centered = X_c - self.means[idx]
            # Add to within-class scatter
            S_W += np.dot(X_c_centered.T, X_c_centered)
        
        # Calculate between-class scatter matrix (S_B)
        S_B = np.zeros((n_features, n_features))
        
        for idx, c in enumerate(self.classes):
            n_c = np.sum(y == c)
            mean_diff = (self.means[idx] - self.overall_mean).reshape(-1, 1)
            S_B += n_c * np.dot(mean_diff, mean_diff.T)
        
        # Solve generalized eigenvalue problem: S_B * v = λ * S_W * v
        # This is equivalent to: S_W^(-1) * S_B * v = λ * v
        
        # Add small value to diagonal for numerical stability
        S_W_inv = np.linalg.inv(S_W + 1e-8 * np.eye(n_features))
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(np.dot(S_W_inv, S_B))
        
        # Ensure eigenvalues are real
        eigenvalues = np.real(eigenvalues)
        eigenvectors = np.real(eigenvectors)
        
        # Sort eigenvectors by eigenvalues in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Select top n_components
        self.components = eigenvectors[:, :self.n_components].T
        
        # Calculate explained variance ratio
        total_variance = np.sum(eigenvalues)
        if total_variance > 0:
            self.explained_variance_ratio = eigenvalues[:self.n_components] / total_variance
        else:
            self.explained_variance_ratio = np.zeros(self.n_components)
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data to discriminant space.
        
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
        
        # Project data onto discriminant vectors
        X_transformed = np.dot(X, self.components.T)
        
        return X_transformed
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Fit LDA and transform data.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data
        y : np.ndarray of shape (n_samples,)
            Target labels
            
        Returns
        -------
        X_transformed : np.ndarray of shape (n_samples, n_components)
            Transformed data
        """
        self.fit(X, y)
        return self.transform(X)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels using nearest centroid.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test data
            
        Returns
        -------
        predictions : np.ndarray of shape (n_samples,)
            Predicted class labels
        """
        X = np.array(X)
        
        # Transform data
        X_transformed = self.transform(X)
        
        # Transform class means
        means_transformed = self.transform(self.means)
        
        # Predict using nearest centroid
        predictions = []
        for x in X_transformed:
            distances = np.linalg.norm(means_transformed - x, axis=1)
            predictions.append(self.classes[np.argmin(distances)])
        
        return np.array(predictions)
    
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
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
    
    def get_params(self) -> dict:
        """Get model parameters."""
        return {
            'n_components': self.n_components,
            'n_classes': len(self.classes),
            'explained_variance_ratio': self.explained_variance_ratio,
            'cumulative_variance_ratio': np.cumsum(self.explained_variance_ratio),
            'components_shape': self.components.shape if self.components is not None else None
        }


if __name__ == "__main__":
    from sklearn.datasets import load_iris, load_wine
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    print("=== LDA on Iris Dataset ===")
    # Load iris dataset (3 classes)
    iris = load_iris()
    X_iris = iris.data
    y_iris = iris.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_iris, y_iris, test_size=0.3, random_state=42
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply LDA
    lda = LDA(n_components=2)
    X_train_lda = lda.fit_transform(X_train_scaled, y_train)
    X_test_lda = lda.transform(X_test_scaled)
    
    print(f"Original shape: {X_train_scaled.shape}")
    print(f"Transformed shape: {X_train_lda.shape}")
    print(f"Explained variance ratio: {lda.explained_variance_ratio}")
    print(f"Cumulative variance explained: {np.sum(lda.explained_variance_ratio):.4f}")
    
    # Classification accuracy
    train_accuracy = lda.score(X_train_scaled, y_train)
    test_accuracy = lda.score(X_test_scaled, y_test)
    print(f"Training accuracy: {train_accuracy:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    print("\n=== LDA on Wine Dataset ===")
    # Load wine dataset (3 classes, 13 features)
    wine = load_wine()
    X_wine = wine.data
    y_wine = wine.target
    
    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(
        X_wine, y_wine, test_size=0.3, random_state=42
    )
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply LDA (max 2 components for 3 classes)
    lda_wine = LDA(n_components=2)
    X_train_lda = lda_wine.fit_transform(X_train_scaled, y_train)
    
    print(f"Original shape: {X_train_scaled.shape}")
    print(f"Transformed shape: {X_train_lda.shape}")
    print(f"Explained variance ratio: {lda_wine.explained_variance_ratio}")
    print(f"Training accuracy: {lda_wine.score(X_train_scaled, y_train):.4f}")
    print(f"Test accuracy: {lda_wine.score(X_test_scaled, y_test):.4f}")