"""
Gaussian Mixture Model (GMM) Implementation from Scratch

GMM assumes data is generated from a mixture of Gaussian distributions
and uses the Expectation-Maximization (EM) algorithm to fit parameters.

Mathematical Foundation:
    P(x) = Σₖ πₖ * N(x | μₖ, Σₖ)
    
    where:
    - πₖ is the mixing coefficient (weight) of component k
    - N(x | μₖ, Σₖ) is a Gaussian distribution with mean μₖ and covariance Σₖ
    
    EM Algorithm:
    E-step: Calculate responsibilities (posterior probabilities)
        γₖ(xᵢ) = πₖ * N(xᵢ | μₖ, Σₖ) / Σⱼ πⱼ * N(xᵢ | μⱼ, Σⱼ)
    
    M-step: Update parameters
        πₖ = (1/n) * Σᵢ γₖ(xᵢ)
        μₖ = Σᵢ γₖ(xᵢ) * xᵢ / Σᵢ γₖ(xᵢ)
        Σₖ = Σᵢ γₖ(xᵢ) * (xᵢ - μₖ)(xᵢ - μₖ)ᵀ / Σᵢ γₖ(xᵢ)
"""

import numpy as np
from typing import Optional


class GaussianMixtureModel:
    """
    Gaussian Mixture Model for clustering.
    
    Parameters
    ----------
    n_components : int, default=3
        Number of Gaussian components
    max_iterations : int, default=100
        Maximum number of EM iterations
    tol : float, default=1e-3
        Convergence threshold
    covariance_type : str, default='full'
        Type of covariance: 'full', 'diag', or 'spherical'
    random_state : int or None, default=None
        Random seed for reproducibility
        
    Attributes
    ----------
    weights : np.ndarray of shape (n_components,)
        Mixing coefficients (πₖ)
    means : np.ndarray of shape (n_components, n_features)
        Component means (μₖ)
    covariances : np.ndarray
        Component covariances (Σₖ)
    converged : bool
        Whether the algorithm converged
    n_iterations : int
        Number of iterations run
    """
    
    def __init__(
        self,
        n_components: int = 3,
        max_iterations: int = 100,
        tol: float = 1e-3,
        covariance_type: str = 'full',
        random_state: Optional[int] = None
    ):
        self.n_components = n_components
        self.max_iterations = max_iterations
        self.tol = tol
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.weights = None
        self.means = None
        self.covariances = None
        self.converged = False
        self.n_iterations = 0
    
    def _initialize_parameters(self, X: np.ndarray) -> None:
        """Initialize GMM parameters."""
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        n_samples, n_features = X.shape
        
        # Initialize weights uniformly
        self.weights = np.ones(self.n_components) / self.n_components
        
        # Initialize means randomly from data
        indices = np.random.choice(n_samples, self.n_components, replace=False)
        self.means = X[indices]
        
        # Initialize covariances
        if self.covariance_type == 'full':
            # Full covariance matrix for each component
            self.covariances = np.array([np.cov(X.T) for _ in range(self.n_components)])
        elif self.covariance_type == 'diag':
            # Diagonal covariance matrix
            self.covariances = np.array([np.var(X, axis=0) for _ in range(self.n_components)])
        elif self.covariance_type == 'spherical':
            # Single variance value for all dimensions
            self.covariances = np.array([np.mean(np.var(X, axis=0))] * self.n_components)
        else:
            raise ValueError(f"Unknown covariance_type: {self.covariance_type}")
    
    def _gaussian_pdf(self, X: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
        """
        Calculate Gaussian probability density function.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Data points
        mean : np.ndarray of shape (n_features,)
            Mean of Gaussian
        cov : np.ndarray
            Covariance of Gaussian
            
        Returns
        -------
        pdf : np.ndarray of shape (n_samples,)
            Probability density values
        """
        n_features = X.shape[1]
        
        if self.covariance_type == 'full':
            # Full covariance matrix
            cov_det = np.linalg.det(cov)
            cov_inv = np.linalg.inv(cov)
            
            # Add small value to prevent singular matrix
            if cov_det < 1e-10:
                cov += 1e-6 * np.eye(n_features)
                cov_det = np.linalg.det(cov)
                cov_inv = np.linalg.inv(cov)
            
            diff = X - mean
            exponent = -0.5 * np.sum(np.dot(diff, cov_inv) * diff, axis=1)
            
        elif self.covariance_type == 'diag':
            # Diagonal covariance
            cov_det = np.prod(cov)
            diff = X - mean
            exponent = -0.5 * np.sum((diff ** 2) / (cov + 1e-6), axis=1)
            
        elif self.covariance_type == 'spherical':
            # Spherical covariance
            cov_det = cov ** n_features
            diff = X - mean
            exponent = -0.5 * np.sum(diff ** 2, axis=1) / (cov + 1e-6)
        
        # Calculate PDF
        normalization = 1.0 / np.sqrt((2 * np.pi) ** n_features * cov_det)
        pdf = normalization * np.exp(exponent)
        
        return pdf
    
    def _e_step(self, X: np.ndarray) -> np.ndarray:
        """
        E-step: Calculate responsibilities.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Data
            
        Returns
        -------
        responsibilities : np.ndarray of shape (n_samples, n_components)
            Posterior probabilities
        """
        n_samples = X.shape[0]
        responsibilities = np.zeros((n_samples, self.n_components))
        
        # Calculate weighted probabilities for each component
        for k in range(self.n_components):
            responsibilities[:, k] = self.weights[k] * \
                                    self._gaussian_pdf(X, self.means[k], self.covariances[k])
        
        # Normalize to get responsibilities
        # Add small epsilon to avoid division by zero
        responsibilities_sum = np.sum(responsibilities, axis=1, keepdims=True) + 1e-10
        responsibilities /= responsibilities_sum
        
        return responsibilities
    
    def _m_step(self, X: np.ndarray, responsibilities: np.ndarray) -> None:
        """
        M-step: Update parameters.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Data
        responsibilities : np.ndarray of shape (n_samples, n_components)
            Responsibilities from E-step
        """
        n_samples, n_features = X.shape
        
        # Effective number of points assigned to each component
        N_k = np.sum(responsibilities, axis=0)
        
        for k in range(self.n_components):
            # Update weights
            self.weights[k] = N_k[k] / n_samples
            
            # Update means
            self.means[k] = np.sum(responsibilities[:, k:k+1] * X, axis=0) / (N_k[k] + 1e-10)
            
            # Update covariances
            diff = X - self.means[k]
            
            if self.covariance_type == 'full':
                # Full covariance matrix
                weighted_diff = responsibilities[:, k:k+1] * diff
                self.covariances[k] = np.dot(weighted_diff.T, diff) / (N_k[k] + 1e-10)
                # Add small value to diagonal for numerical stability
                self.covariances[k] += 1e-6 * np.eye(n_features)
                
            elif self.covariance_type == 'diag':
                # Diagonal covariance
                weighted_sq_diff = responsibilities[:, k:k+1] * (diff ** 2)
                self.covariances[k] = np.sum(weighted_sq_diff, axis=0) / (N_k[k] + 1e-10)
                self.covariances[k] += 1e-6  # Numerical stability
                
            elif self.covariance_type == 'spherical':
                # Spherical covariance
                weighted_sq_diff = responsibilities[:, k:k+1] * (diff ** 2)
                self.covariances[k] = np.sum(weighted_sq_diff) / (N_k[k] * n_features + 1e-10)
                self.covariances[k] += 1e-6  # Numerical stability
    
    def _compute_log_likelihood(self, X: np.ndarray) -> float:
        """
        Compute log-likelihood of data.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Data
            
        Returns
        -------
        log_likelihood : float
            Log-likelihood value
        """
        n_samples = X.shape[0]
        log_likelihood = 0
        
        for i in range(n_samples):
            sample_likelihood = 0
            for k in range(self.n_components):
                sample_likelihood += self.weights[k] * \
                                   self._gaussian_pdf(X[i:i+1], self.means[k], self.covariances[k])[0]
            log_likelihood += np.log(sample_likelihood + 1e-10)
        
        return log_likelihood
    
    def fit(self, X: np.ndarray) -> 'GaussianMixtureModel':
        """
        Fit GMM using EM algorithm.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data
            
        Returns
        -------
        self : GaussianMixtureModel
            Fitted estimator
        """
        X = np.array(X)
        
        # Initialize parameters
        self._initialize_parameters(X)
        
        prev_log_likelihood = -np.inf
        
        # EM iterations
        for iteration in range(self.max_iterations):
            # E-step
            responsibilities = self._e_step(X)
            
            # M-step
            self._m_step(X, responsibilities)
            
            # Check convergence
            log_likelihood = self._compute_log_likelihood(X)
            
            if abs(log_likelihood - prev_log_likelihood) < self.tol:
                self.converged = True
                self.n_iterations = iteration + 1
                break
            
            prev_log_likelihood = log_likelihood
            self.n_iterations = iteration + 1
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Data to predict
            
        Returns
        -------
        labels : np.ndarray of shape (n_samples,)
            Cluster labels (component with highest responsibility)
        """
        X = np.array(X)
        responsibilities = self._e_step(X)
        return np.argmax(responsibilities, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict posterior probabilities.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Data to predict
            
        Returns
        -------
        probabilities : np.ndarray of shape (n_samples, n_components)
            Posterior probabilities (responsibilities)
        """
        X = np.array(X)
        return self._e_step(X)
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fit GMM and predict cluster labels.
        
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
        return self.predict(X)
    
    def sample(self, n_samples: int = 1) -> np.ndarray:
        """
        Generate random samples from fitted GMM.
        
        Parameters
        ----------
        n_samples : int, default=1
            Number of samples to generate
            
        Returns
        -------
        X_sampled : np.ndarray of shape (n_samples, n_features)
            Generated samples
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # Sample component indices based on weights
        component_indices = np.random.choice(
            self.n_components,
            size=n_samples,
            p=self.weights
        )
        
        samples = []
        for k in component_indices:
            if self.covariance_type == 'full':
                sample = np.random.multivariate_normal(self.means[k], self.covariances[k])
            elif self.covariance_type == 'diag':
                sample = np.random.normal(self.means[k], np.sqrt(self.covariances[k]))
            elif self.covariance_type == 'spherical':
                n_features = len(self.means[k])
                sample = np.random.normal(self.means[k], np.sqrt(self.covariances[k]), size=n_features)
            samples.append(sample)
        
        return np.array(samples)
    
    def score(self, X: np.ndarray) -> float:
        """
        Compute average log-likelihood per sample.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Data
            
        Returns
        -------
        score : float
            Average log-likelihood per sample
        """
        X = np.array(X)
        return self._compute_log_likelihood(X) / X.shape[0]
    
    def get_params(self) -> dict:
        """Get model parameters."""
        return {
            'n_components': self.n_components,
            'weights': self.weights,
            'means': self.means,
            'covariances': self.covariances,
            'converged': self.converged,
            'n_iterations': self.n_iterations,
            'covariance_type': self.covariance_type
        }


if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    import matplotlib.pyplot as plt
    
    print("=== GMM Clustering ===")
    
    # Generate sample data with 3 clusters
    X, y_true = make_blobs(n_samples=300, centers=3, n_features=2,
                           cluster_std=0.6, random_state=42)
    
    print("\n--- Full Covariance ---")
    # Fit GMM with full covariance
    gmm_full = GaussianMixtureModel(n_components=3, covariance_type='full',
                                    max_iterations=100, random_state=42)
    gmm_full.fit(X)
    labels_full = gmm_full.predict(X)
    
    print(f"Converged: {gmm_full.converged}")
    print(f"Number of iterations: {gmm_full.n_iterations}")
    print(f"Log-likelihood per sample: {gmm_full.score(X):.4f}")
    print(f"Weights: {gmm_full.weights}")
    
    # Check cluster distribution
    unique, counts = np.unique(labels_full, return_counts=True)
    print(f"Cluster sizes: {dict(zip(unique, counts))}")
    
    print("\n--- Diagonal Covariance ---")
    # Fit GMM with diagonal covariance
    gmm_diag = GaussianMixtureModel(n_components=3, covariance_type='diag',
                                    max_iterations=100, random_state=42)
    gmm_diag.fit(X)
    labels_diag = gmm_diag.predict(X)
    
    print(f"Converged: {gmm_diag.converged}")
    print(f"Number of iterations: {gmm_diag.n_iterations}")
    print(f"Log-likelihood per sample: {gmm_diag.score(X):.4f}")
    
    print("\n--- Spherical Covariance ---")
    # Fit GMM with spherical covariance
    gmm_sph = GaussianMixtureModel(n_components=3, covariance_type='spherical',
                                   max_iterations=100, random_state=42)
    gmm_sph.fit(X)
    labels_sph = gmm_sph.predict(X)
    
    print(f"Converged: {gmm_sph.converged}")
    print(f"Number of iterations: {gmm_sph.n_iterations}")
    print(f"Log-likelihood per sample: {gmm_sph.score(X):.4f}")
    
    # Predict probabilities
    print("\n--- Soft Clustering (Probabilities) ---")
    probabilities = gmm_full.predict_proba(X[:5])
    print("Probabilities for first 5 samples:")
    print(probabilities)
    
    # Generate new samples
    print("\n--- Sample Generation ---")
    X_generated = gmm_full.sample(n_samples=10)
    print(f"Generated samples shape: {X_generated.shape}")
    print(f"Sample means: {np.mean(X_generated, axis=0)}")