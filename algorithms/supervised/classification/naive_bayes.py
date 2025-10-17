"""
Naive Bayes Classifier Implementation from Scratch

Naive Bayes is based on Bayes' theorem with the "naive" assumption
of conditional independence between features.

Mathematical Foundation:
    P(y|X) = P(X|y) * P(y) / P(X)
    
    With naive independence assumption:
    P(X|y) = P(x₁|y) * P(x₂|y) * ... * P(xₙ|y)
    
    Classification: argmax_y P(y|X) = argmax_y P(X|y) * P(y)
"""

import numpy as np


class GaussianNaiveBayes:
    """
    Gaussian Naive Bayes classifier.
    
    Assumes features follow a normal (Gaussian) distribution.
    Best for continuous features.
    
    Parameters
    ----------
    var_smoothing : float, default=1e-9
        Portion of the largest variance added to all variances
        for calculation stability
        
    Attributes
    ----------
    classes : np.ndarray
        Unique class labels
    class_priors : dict
        Prior probabilities for each class
    means : dict
        Mean of features for each class
    variances : dict
        Variance of features for each class
    """
    
    def _init_(self, var_smoothing: float = 1e-9):
        self.var_smoothing = var_smoothing
        self.classes = None
        self.class_priors = {}
        self.means = {}
        self.variances = {}
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GaussianNaiveBayes':
        """
        Fit Gaussian Naive Bayes classifier.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data
        y : np.ndarray of shape (n_samples,)
            Target labels
            
        Returns
        -------
        self : GaussianNaiveBayes
            Fitted estimator
        """
        X = np.array(X)
        y = np.array(y)
        
        self.classes = np.unique(y)
        n_samples = X.shape[0]
        
        # Calculate class priors, means, and variances
        for c in self.classes:
            X_c = X[y == c]
            self.class_priors[c] = len(X_c) / n_samples
            self.means[c] = np.mean(X_c, axis=0)
            self.variances[c] = np.var(X_c, axis=0) + self.var_smoothing
        
        return self
    
    def _calculate_likelihood(self, x: np.ndarray, mean: np.ndarray, var: np.ndarray) -> float:
        """
        Calculate Gaussian probability density function.
        
        P(x|μ,σ²) = (1/√(2πσ²)) * exp(-(x-μ)²/(2σ²))
        
        Using log for numerical stability:
        log P(x|μ,σ²) = -0.5 * [log(2π) + log(σ²) + (x-μ)²/σ²]
        """
        # Calculate log likelihood for numerical stability
        log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * var))
        log_likelihood -= 0.5 * np.sum(((x - mean) ** 2) / var)
        return log_likelihood
    
    def _calculate_posterior(self, x: np.ndarray) -> dict:
        """
        Calculate posterior probability for each class.
        
        Returns log posterior for numerical stability.
        """
        posteriors = {}
        
        for c in self.classes:
            # Log prior
            log_prior = np.log(self.class_priors[c])
            
            # Log likelihood
            log_likelihood = self._calculate_likelihood(
                x, self.means[c], self.variances[c]
            )
            
            # Log posterior (unnormalized)
            posteriors[c] = log_prior + log_likelihood
        
        return posteriors
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples in X.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data
            
        Returns
        -------
        predictions : np.ndarray of shape (n_samples,)
            Predicted class labels
        """
        X = np.array(X)
        predictions = []
        
        for x in X:
            posteriors = self._calculate_posterior(x)
            # Select class with maximum posterior probability
            predicted_class = max(posteriors, key=posteriors.get)
            predictions.append(predicted_class)
        
        return np.array(predictions)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for samples in X.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data
            
        Returns
        -------
        probabilities : np.ndarray of shape (n_samples, n_classes)
            Predicted class probabilities
        """
        X = np.array(X)
        probabilities = []
        
        for x in X:
            posteriors = self._calculate_posterior(x)
            
            # Convert log posteriors to probabilities using softmax
            log_posteriors = np.array([posteriors[c] for c in self.classes])
            # Subtract max for numerical stability
            log_posteriors_stable = log_posteriors - np.max(log_posteriors)
            exp_posteriors = np.exp(log_posteriors_stable)
            probs = exp_posteriors / np.sum(exp_posteriors)
            
            probabilities.append(probs)
        
        return np.array(probabilities)
    
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
            'classes': self.classes,
            'class_priors': self.class_priors,
            'means': self.means,
            'variances': self.variances
        }


class MultinomialNaiveBayes:
    """
    Multinomial Naive Bayes classifier.
    
    Suitable for discrete features (e.g., word counts).
    Often used for text classification.
    
    Parameters
    ----------
    alpha : float, default=1.0
        Additive (Laplace/Lidstone) smoothing parameter
        
    Attributes
    ----------
    classes : np.ndarray
        Unique class labels
    class_priors : dict
        Prior probabilities for each class
    feature_probs : dict
        Feature probabilities for each class
    """
    
    def _init_(self, alpha: float = 1.0):
        self.alpha = alpha
        self.classes = None
        self.class_priors = {}
        self.feature_probs = {}
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MultinomialNaiveBayes':
        """
        Fit Multinomial Naive Bayes classifier.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data (should be non-negative)
        y : np.ndarray of shape (n_samples,)
            Target labels
            
        Returns
        -------
        self : MultinomialNaiveBayes
            Fitted estimator
        """
        X = np.array(X)
        y = np.array(y)
        
        self.classes = np.unique(y)
        n_samples, n_features = X.shape
        
        for c in self.classes:
            X_c = X[y == c]
            
            # Calculate class prior
            self.class_priors[c] = len(X_c) / n_samples
            
            # Calculate feature probabilities with Laplace smoothing
            # P(xᵢ|c) = (count(xᵢ, c) + α) / (count(c) + α * n_features)
            feature_counts = np.sum(X_c, axis=0)
            total_count = np.sum(feature_counts)
            
            self.feature_probs[c] = (feature_counts + self.alpha) / \
                                   (total_count + self.alpha * n_features)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples in X.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data
            
        Returns
        -------
        predictions : np.ndarray of shape (n_samples,)
            Predicted class labels
        """
        X = np.array(X)
        predictions = []
        
        for x in X:
            posteriors = {}
            
            for c in self.classes:
                # Log prior
                log_prior = np.log(self.class_priors[c])
                
                # Log likelihood using multinomial distribution
                log_likelihood = np.sum(x * np.log(self.feature_probs[c]))
                
                # Log posterior
                posteriors[c] = log_prior + log_likelihood
            
            predicted_class = max(posteriors, key=posteriors.get)
            predictions.append(predicted_class)
        
        return np.array(predictions)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate accuracy score."""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


if _name_ == "_main_":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    print("=== Gaussian Naive Bayes Example ===")
    # Generate continuous data
    X, y = make_classification(n_samples=500, n_features=5, n_informative=3,
                               n_redundant=1, n_classes=3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Gaussian NB
    gnb = GaussianNaiveBayes()
    gnb.fit(X_train, y_train)
    
    print(f"Training Accuracy: {gnb.score(X_train, y_train):.4f}")
    print(f"Test Accuracy: {gnb.score(X_test, y_test):.4f}")
    
    print("\n=== Multinomial Naive Bayes Example ===")
    # Generate count data (simulate word counts)
    X_count = np.random.randint(0, 20, size=(500, 10))
    y_count = np.random.randint(0, 2, size=500)
    X_train, X_test, y_train, y_test = train_test_split(
        X_count, y_count, test_size=0.2, random_state=42
    )
    
    # Train Multinomial NB
    mnb = MultinomialNaiveBayes(alpha=1.0)
    mnb.fit(X_train, y_train)
    
    print(f"Training Accuracy: {mnb.score(X_train, y_train):.4f}")
    print(f"Test Accuracy: {mnb.score(X_test, y_test):.4f}")