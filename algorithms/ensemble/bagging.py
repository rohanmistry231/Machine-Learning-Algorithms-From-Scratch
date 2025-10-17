"""
Bagging (Bootstrap Aggregating) Implementation from Scratch

Bagging is an ensemble technique that trains multiple models on different
bootstrap samples of the training data and combines their predictions.

Mathematical Foundation:
    For b = 1 to B:
        1. Create bootstrap sample: Dᵦ by sampling n samples with replacement
        
        2. Train base learner fᵦ on Dᵦ
    
    Prediction:
    - Classification: Majority vote across all models
      ŷ = mode{f₁(x), f₂(x), ..., fᵦ(x)}
    
    - Regression: Average across all models
      ŷ = (1/B) * Σᵦ fᵦ(x)
    
    Out-of-Bag (OOB) Score:
    - For each sample, predict using only models that didn't see it in training
    - Provides unbiased estimate of generalization error
"""

import numpy as np
from typing import List, Optional, Any
from collections import Counter


class BaggingClassifier:
    """
    Bagging classifier that can use any base estimator.
    
    Parameters
    ----------
    base_estimator : object, default=None
        Base estimator to fit on random subsets. If None, uses decision tree
    n_estimators : int, default=10
        Number of base estimators in the ensemble
    max_samples : float or int, default=1.0
        Number/fraction of samples to draw for each base estimator
    max_features : float or int, default=1.0
        Number/fraction of features to draw for each base estimator
    bootstrap : bool, default=True
        Whether samples are drawn with replacement
    bootstrap_features : bool, default=False
        Whether features are drawn with replacement
    oob_score : bool, default=False
        Whether to use out-of-bag samples to estimate generalization error
    random_state : int or None, default=None
        Random seed
        
    Attributes
    ----------
    estimators_ : List
        Collection of fitted base estimators
    estimators_features_ : List[np.ndarray]
        Features used by each estimator
    oob_score_ : float or None
        Out-of-bag score
    oob_decision_function_ : np.ndarray or None
        OOB predictions for each sample
    """
    
    def __init__(
        self,
        base_estimator = None,
        n_estimators: int = 10,
        max_samples: float = 1.0,
        max_features: float = 1.0,
        bootstrap: bool = True,
        bootstrap_features: bool = False,
        oob_score: bool = False,
        random_state: Optional[int] = None
    ):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.oob_score = oob_score
        self.random_state = random_state
        
        self.estimators_ = []
        self.estimators_features_ = []
        self.oob_score_ = None
        self.oob_decision_function_ = None
    
    def _get_n_samples_features(self, n_samples: int, n_features: int) -> tuple:
        """Calculate number of samples and features to use."""
        if isinstance(self.max_samples, float):
            n_samples_bootstrap = int(n_samples * self.max_samples)
        else:
            n_samples_bootstrap = min(self.max_samples, n_samples)
        
        if isinstance(self.max_features, float):
            n_features_bootstrap = int(n_features * self.max_features)
        else:
            n_features_bootstrap = min(self.max_features, n_features)
        
        return n_samples_bootstrap, n_features_bootstrap
    
    def _make_sampler(self, n_samples: int, n_samples_bootstrap: int) -> tuple:
        """Create sample indices and out-of-bag mask."""
        if self.bootstrap:
            # Sample with replacement
            indices = np.random.choice(n_samples, n_samples_bootstrap, replace=True)
            # OOB samples are those not selected
            oob_mask = np.ones(n_samples, dtype=bool)
            oob_mask[indices] = False
        else:
            # Sample without replacement
            indices = np.random.choice(n_samples, n_samples_bootstrap, replace=False)
            oob_mask = None
        
        return indices, oob_mask
    
    def _get_base_estimator(self):
        """Get a copy of base estimator or default decision tree."""
        if self.base_estimator is None:
            # Use simple decision tree as default
            from copy import deepcopy
            # Simplified decision tree stub
            class SimpleDecisionTree:
                def __init__(self):
                    self.feature_importances_ = None
                    self.tree_ = None
                
                def fit(self, X, y):
                    # Simple majority class predictor as fallback
                    self.majority_class_ = Counter(y).most_common(1)[0][0]
                    return self
                
                def predict(self, X):
                    return np.full(X.shape[0], self.majority_class_)
            
            return SimpleDecisionTree()
        else:
            # Try to import and copy the estimator
            try:
                from copy import deepcopy
                return deepcopy(self.base_estimator)
            except:
                # If deepcopy fails, try to create new instance
                return self.base_estimator.__class__()
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaggingClassifier':
        """
        Fit bagging classifier.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data
        y : np.ndarray of shape (n_samples,)
            Target labels
            
        Returns
        -------
        self : BaggingClassifier
            Fitted estimator
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        X = np.array(X)
        y = np.array(y)
        
        n_samples, n_features = X.shape
        
        n_samples_bootstrap, n_features_bootstrap = self._get_n_samples_features(
            n_samples, n_features
        )
        
        self.estimators_ = []
        self.estimators_features_ = []
        
        # For OOB score calculation
        if self.oob_score:
            oob_predictions = [[] for _ in range(n_samples)]
        
        # Train each estimator
        for i in range(self.n_estimators):
            # Sample data
            sample_indices, oob_mask = self._make_sampler(n_samples, n_samples_bootstrap)
            
            # Sample features
            if self.bootstrap_features:
                feature_indices = np.random.choice(
                    n_features, n_features_bootstrap, replace=True
                )
            else:
                feature_indices = np.random.choice(
                    n_features, n_features_bootstrap, replace=False
                )
            
            feature_indices = np.sort(feature_indices)
            
            # Get bootstrap sample
            X_bootstrap = X[sample_indices][:, feature_indices]
            y_bootstrap = y[sample_indices]
            
            # Train estimator
            estimator = self._get_base_estimator()
            estimator.fit(X_bootstrap, y_bootstrap)
            
            self.estimators_.append(estimator)
            self.estimators_features_.append(feature_indices)
            
            # OOB predictions
            if self.oob_score and oob_mask is not None:
                oob_indices = np.where(oob_mask)[0]
                if len(oob_indices) > 0:
                    X_oob = X[oob_indices][:, feature_indices]
                    oob_pred = estimator.predict(X_oob)
                    
                    for idx, pred in zip(oob_indices, oob_pred):
                        oob_predictions[idx].append(pred)
        
        # Calculate OOB score
        if self.oob_score:
            # Get samples that have OOB predictions
            valid_oob = [i for i, preds in enumerate(oob_predictions) if len(preds) > 0]
            
            if len(valid_oob) > 0:
                oob_y_pred = np.array([
                    Counter(oob_predictions[i]).most_common(1)[0][0]
                    for i in valid_oob
                ])
                
                self.oob_score_ = np.mean(oob_y_pred == y[valid_oob])
                self.oob_decision_function_ = oob_predictions
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using majority vote.
        
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
        n_samples = X.shape[0]
        
        # Collect predictions from all estimators
        all_predictions = []
        
        for estimator, feature_indices in zip(self.estimators_, self.estimators_features_):
            X_subset = X[:, feature_indices]
            predictions = estimator.predict(X_subset)
            all_predictions.append(predictions)
        
        # Majority vote
        all_predictions = np.array(all_predictions).T
        final_predictions = np.array([
            Counter(all_predictions[i]).most_common(1)[0][0]
            for i in range(n_samples)
        ])
        
        return final_predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data
            
        Returns
        -------
        probabilities : np.ndarray of shape (n_samples, n_classes)
            Class probabilities
        """
        X = np.array(X)
        n_samples = X.shape[0]
        
        # Collect predictions
        all_predictions = []
        
        for estimator, feature_indices in zip(self.estimators_, self.estimators_features_):
            X_subset = X[:, feature_indices]
            predictions = estimator.predict(X_subset)
            all_predictions.append(predictions)
        
        all_predictions = np.array(all_predictions).T
        
        # Calculate probabilities based on vote proportions
        unique_classes = np.unique(np.concatenate(all_predictions))
        n_classes = len(unique_classes)
        
        probabilities = np.zeros((n_samples, n_classes))
        
        for i in range(n_samples):
            vote_counts = Counter(all_predictions[i])
            for j, cls in enumerate(unique_classes):
                probabilities[i, j] = vote_counts.get(cls, 0) / len(self.estimators_)
        
        return probabilities
    
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
            'n_estimators': self.n_estimators,
            'n_estimators_fitted': len(self.estimators_),
            'max_samples': self.max_samples,
            'max_features': self.max_features,
            'bootstrap': self.bootstrap,
            'oob_score': self.oob_score_
        }


class BaggingRegressor:
    """
    Bagging regressor that averages predictions from multiple base estimators.
    
    Similar to BaggingClassifier but for regression tasks.
    Uses averaging instead of voting for predictions.
    
    Parameters are same as BaggingClassifier.
    """
    
    def __init__(
        self,
        base_estimator = None,
        n_estimators: int = 10,
        max_samples: float = 1.0,
        max_features: float = 1.0,
        bootstrap: bool = True,
        bootstrap_features: bool = False,
        oob_score: bool = False,
        random_state: Optional[int] = None
    ):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.oob_score = oob_score
        self.random_state = random_state
        
        self.estimators_ = []
        self.estimators_features_ = []
        self.oob_score_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaggingRegressor':
        """Fit bagging regressor (similar to classifier)."""
        # Implementation similar to BaggingClassifier
        # Would follow same pattern but for regression
        pass
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using average of all estimators."""
        pass
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate R² score."""
        pass


if __name__ == "__main__":
    from sklearn.datasets import make_classification, load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    
    print("=== Bagging Classifier Example ===")
    # Generate classification data
    X, y = make_classification(n_samples=500, n_features=20, n_informative=15,
                               n_redundant=3, n_classes=2, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Use sklearn's DecisionTree as base estimator
    base_tree = DecisionTreeClassifier(max_depth=5, random_state=42)
    
    print("\n--- Bagging with 10 estimators ---")
    bagging = BaggingClassifier(
        base_estimator=base_tree,
        n_estimators=10,
        max_samples=1.0,
        max_features=1.0,
        bootstrap=True,
        oob_score=True,
        random_state=42
    )
    bagging.fit(X_train, y_train)
    
    print(f"Training Accuracy: {bagging.score(X_train, y_train):.4f}")
    print(f"Test Accuracy: {bagging.score(X_test, y_test):.4f}")
    print(f"OOB Score: {bagging.oob_score_:.4f}")
    print(f"Number of estimators: {bagging.get_params()['n_estimators_fitted']}")
    
    print("\n--- Bagging with 50 estimators ---")
    bagging_50 = BaggingClassifier(
        base_estimator=base_tree,
        n_estimators=50,
        max_samples=0.8,
        max_features=0.8,
        bootstrap=True,
        oob_score=True,
        random_state=42
    )
    bagging_50.fit(X_train, y_train)
    
    print(f"Training Accuracy: {bagging_50.score(X_train, y_train):.4f}")
    print(f"Test Accuracy: {bagging_50.score(X_test, y_test):.4f}")
    print(f"OOB Score: {bagging_50.oob_score_:.4f}")
    
    print("\n=== Comparison: Single Tree vs Bagging ===")
    # Train single decision tree
    single_tree = DecisionTreeClassifier(max_depth=5, random_state=42)
    single_tree.fit(X_train, y_train)
    
    single_train_acc = single_tree.score(X_train, y_train)
    single_test_acc = single_tree.score(X_test, y_test)
    
    print(f"Single Tree - Train: {single_train_acc:.4f}, Test: {single_test_acc:.4f}")
    print(f"Bagging (50) - Train: {bagging_50.score(X_train, y_train):.4f}, "
          f"Test: {bagging_50.score(X_test, y_test):.4f}")
    print(f"\nImprovement in test accuracy: {bagging_50.score(X_test, y_test) - single_test_acc:.4f}")
    
    print("\n=== Iris Dataset Multi-class Classification ===")
    iris = load_iris()
    X_iris = iris.data
    y_iris = iris.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_iris, y_iris, test_size=0.3, random_state=42
    )
    
    bagging_iris = BaggingClassifier(
        base_estimator=DecisionTreeClassifier(max_depth=3),
        n_estimators=30,
        max_samples=0.8,
        bootstrap=True,
        oob_score=True,
        random_state=42
    )
    bagging_iris.fit(X_train, y_train)
    
    print(f"Training Accuracy: {bagging_iris.score(X_train, y_train):.4f}")
    print(f"Test Accuracy: {bagging_iris.score(X_test, y_test):.4f}")
    print(f"OOB Score: {bagging_iris.oob_score_:.4f}")
    
    # Predict probabilities
    probas = bagging_iris.predict_proba(X_test[:5])
    print(f"\nProbability predictions for first 5 test samples:")
    print(probas)
    print(f"True labels: {y_test[:5]}")
    print(f"Predicted labels: {bagging_iris.predict(X_test[:5])}")