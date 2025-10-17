"""
Stacking (Stacked Generalization) Implementation from Scratch

Stacking combines multiple heterogeneous base learners with a meta-learner
(also called blender) that learns how to best combine the predictions
of the base learners.

Mathematical Foundation:
    Level 0: Train base learners L₁, L₂, ..., Lₖ on training data
    
    Level 1: 
        - Generate meta-features: predictions of base learners on training data
        - Meta-features matrix: X_meta of shape (n_samples, k)
        - Train meta-learner M on (X_meta, y)
    
    Prediction:
        - Get predictions from all base learners: [ŷ₁, ŷ₂, ..., ŷₖ]
        - Pass to meta-learner: ŷ = M([ŷ₁, ŷ₂, ..., ŷₖ])
"""

import numpy as np
from typing import List, Optional, Any


class StackingClassifier:
    """
    Stacking classifier that combines multiple base classifiers
    with a meta-learner.
    
    Parameters
    ----------
    base_learners : List[tuple]
        List of (name, estimator) tuples for base learners
    meta_learner : object
        Meta-learner estimator with fit and predict methods
    use_proba : bool, default=False
        Whether to use predict_proba from base learners instead of predict
    cv : int or None, default=None
        Number of folds for cross-validation when generating meta-features.
        If None, uses full training data.
    random_state : int or None, default=None
        Random seed
        
    Attributes
    ----------
    base_learners_ : List[tuple]
        Fitted base learners
    meta_learner_ : object
        Fitted meta-learner
    base_learner_names_ : List[str]
        Names of base learners
    """
    
    def __init__(
        self,
        base_learners: List[tuple],
        meta_learner: Any,
        use_proba: bool = False,
        cv: Optional[int] = None,
        random_state: Optional[int] = None
    ):
        self.base_learners = base_learners
        self.meta_learner = meta_learner
        self.use_proba = use_proba
        self.cv = cv
        self.random_state = random_state
        
        self.base_learners_ = []
        self.meta_learner_ = None
        self.base_learner_names_ = [name for name, _ in base_learners]
    
    def _get_meta_features(self, X: np.ndarray, y: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        Generate meta-features using base learners.
        
        Parameters
        ----------
        X : np.ndarray
            Training data
        y : np.ndarray
            Target values
        fit : bool
            Whether to fit base learners
            
        Returns
        -------
        meta_features : np.ndarray
            Predictions from base learners
        """
        n_samples = X.shape[0]
        n_base_learners = len(self.base_learners)
        
        if self.use_proba:
            # Use probability predictions
            meta_features = np.zeros((n_samples, n_base_learners))
        else:
            # Use class predictions
            meta_features = np.zeros((n_samples, n_base_learners))
        
        # Generate meta-features with or without cross-validation
        if self.cv is None:
            # Use full training data
            for i, (name, learner) in enumerate(self.base_learners):
                if fit:
                    learner.fit(X, y)
                    self.base_learners_.append((name, learner))
                
                if self.use_proba:
                    if hasattr(learner, 'predict_proba'):
                        # For binary classification, use positive class probability
                        proba = learner.predict_proba(X)
                        if proba.shape[1] == 2:
                            meta_features[:, i] = proba[:, 1]
                        else:
                            # For multi-class, average probabilities or use argmax
                            meta_features[:, i] = np.max(proba, axis=1)
                    else:
                        meta_features[:, i] = learner.predict(X)
                else:
                    meta_features[:, i] = learner.predict(X)
        
        else:
            # Use cross-validation
            if self.random_state is not None:
                np.random.seed(self.random_state)
            
            fold_size = n_samples // self.cv
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            
            for fold in range(self.cv):
                # Split into train and validation
                val_start = fold * fold_size
                val_end = val_start + fold_size if fold < self.cv - 1 else n_samples
                
                val_indices = indices[val_start:val_end]
                train_indices = np.concatenate([indices[:val_start], indices[val_end:]])
                
                X_train = X[train_indices]
                y_train = y[train_indices]
                X_val = X[val_indices]
                
                # Train base learners on this fold
                for i, (name, learner) in enumerate(self.base_learners):
                    learner.fit(X_train, y_train)
                    
                    if self.use_proba:
                        if hasattr(learner, 'predict_proba'):
                            proba = learner.predict_proba(X_val)
                            if proba.shape[1] == 2:
                                meta_features[val_indices, i] = proba[:, 1]
                            else:
                                meta_features[val_indices, i] = np.max(proba, axis=1)
                        else:
                            meta_features[val_indices, i] = learner.predict(X_val)
                    else:
                        meta_features[val_indices, i] = learner.predict(X_val)
        
        return meta_features
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'StackingClassifier':
        """
        Fit stacking classifier.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data
        y : np.ndarray of shape (n_samples,)
            Target values
            
        Returns
        -------
        self : StackingClassifier
            Fitted estimator
        """
        X = np.array(X)
        y = np.array(y)
        
        # Generate meta-features
        meta_features = self._get_meta_features(X, y, fit=True)
        
        # Train meta-learner on meta-features
        self.meta_learner_ = self.meta_learner.__class__(**self.meta_learner.get_params()) \
                              if hasattr(self.meta_learner, 'get_params') \
                              else self.meta_learner
        self.meta_learner_.fit(meta_features, y)
        
        return self
    
    def _get_meta_features_predict(self, X: np.ndarray) -> np.ndarray:
        """Generate meta-features for prediction."""
        n_samples = X.shape[0]
        n_base_learners = len(self.base_learners_)
        
        meta_features = np.zeros((n_samples, n_base_learners))
        
        for i, (name, learner) in enumerate(self.base_learners_):
            if self.use_proba:
                if hasattr(learner, 'predict_proba'):
                    proba = learner.predict_proba(X)
                    if proba.shape[1] == 2:
                        meta_features[:, i] = proba[:, 1]
                    else:
                        meta_features[:, i] = np.max(proba, axis=1)
                else:
                    meta_features[:, i] = learner.predict(X)
            else:
                meta_features[:, i] = learner.predict(X)
        
        return meta_features
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using stacked model.
        
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
        
        # Get meta-features
        meta_features = self._get_meta_features_predict(X)
        
        # Predict with meta-learner
        return self.meta_learner_.predict(meta_features)
    
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
            Predicted probabilities
        """
        X = np.array(X)
        
        # Get meta-features
        meta_features = self._get_meta_features_predict(X)
        
        # Predict probabilities with meta-learner
        if hasattr(self.meta_learner_, 'predict_proba'):
            return self.meta_learner_.predict_proba(meta_features)
        else:
            # Fallback: convert predictions to one-hot
            predictions = self.meta_learner_.predict(meta_features)
            unique_classes = np.unique(predictions)
            n_classes = len(unique_classes)
            
            proba = np.zeros((X.shape[0], n_classes))
            for i, cls in enumerate(unique_classes):
                proba[predictions == cls, i] = 1.0
            
            return proba
    
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
            'n_base_learners': len(self.base_learners_),
            'base_learner_names': self.base_learner_names_,
            'use_proba': self.use_proba,
            'cv': self.cv,
            'meta_learner_type': type(self.meta_learner_).__name__
        }


class StackingRegressor:
    """
    Stacking regressor that combines multiple base regressors
    with a meta-learner.
    
    Similar to StackingClassifier but for regression tasks.
    
    Parameters
    ----------
    base_learners : List[tuple]
        List of (name, estimator) tuples for base regressors
    meta_learner : object
        Meta-learner estimator with fit and predict methods
    cv : int or None, default=None
        Number of folds for cross-validation
    random_state : int or None, default=None
        Random seed
    """
    
    def __init__(
        self,
        base_learners: List[tuple],
        meta_learner: Any,
        cv: Optional[int] = None,
        random_state: Optional[int] = None
    ):
        self.base_learners = base_learners
        self.meta_learner = meta_learner
        self.cv = cv
        self.random_state = random_state
        
        self.base_learners_ = []
        self.meta_learner_ = None
        self.base_learner_names_ = [name for name, _ in base_learners]
    
    def _get_meta_features(self, X: np.ndarray, y: np.ndarray, fit: bool = True) -> np.ndarray:
        """Generate meta-features using base learners."""
        n_samples = X.shape[0]
        n_base_learners = len(self.base_learners)
        
        meta_features = np.zeros((n_samples, n_base_learners))
        
        if self.cv is None:
            # Use full training data
            for i, (name, learner) in enumerate(self.base_learners):
                if fit:
                    learner.fit(X, y)
                    self.base_learners_.append((name, learner))
                
                meta_features[:, i] = learner.predict(X)
        
        else:
            # Use cross-validation
            if self.random_state is not None:
                np.random.seed(self.random_state)
            
            fold_size = n_samples // self.cv
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            
            for fold in range(self.cv):
                val_start = fold * fold_size
                val_end = val_start + fold_size if fold < self.cv - 1 else n_samples
                
                val_indices = indices[val_start:val_end]
                train_indices = np.concatenate([indices[:val_start], indices[val_end:]])
                
                X_train = X[train_indices]
                y_train = y[train_indices]
                X_val = X[val_indices]
                
                for i, (name, learner) in enumerate(self.base_learners):
                    learner.fit(X_train, y_train)
                    meta_features[val_indices, i] = learner.predict(X_val)
        
        return meta_features
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'StackingRegressor':
        """Fit stacking regressor."""
        X = np.array(X)
        y = np.array(y)
        
        # Generate meta-features
        meta_features = self._get_meta_features(X, y, fit=True)
        
        # Train meta-learner
        self.meta_learner_ = self.meta_learner.__class__(**self.meta_learner.get_params()) \
                              if hasattr(self.meta_learner, 'get_params') \
                              else self.meta_learner
        self.meta_learner_.fit(meta_features, y)
        
        return self
    
    def _get_meta_features_predict(self, X: np.ndarray) -> np.ndarray:
        """Generate meta-features for prediction."""
        n_samples = X.shape[0]
        n_base_learners = len(self.base_learners_)
        
        meta_features = np.zeros((n_samples, n_base_learners))
        
        for i, (name, learner) in enumerate(self.base_learners_):
            meta_features[:, i] = learner.predict(X)
        
        return meta_features
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using stacked model."""
        X = np.array(X)
        meta_features = self._get_meta_features_predict(X)
        return self.meta_learner_.predict(meta_features)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate R² score.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test data
        y : np.ndarray of shape (n_samples,)
            True values
            
        Returns
        -------
        r2_score : float
            R² score
        """
        y_pred = self.predict(X)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        ss_res = np.sum((y - y_pred) ** 2)
        return 1 - (ss_res / ss_tot)
    
    def get_params(self) -> dict:
        """Get model parameters."""
        return {
            'n_base_learners': len(self.base_learners_),
            'base_learner_names': self.base_learner_names_,
            'cv': self.cv,
            'meta_learner_type': type(self.meta_learner_).__name__
        }


if __name__ == "__main__":
    from sklearn.datasets import make_classification, make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    
    print("=== Stacking Classifier Example ===\n")
    
    # Generate classification data
    X, y = make_classification(n_samples=500, n_features=20, n_informative=15,
                               n_redundant=3, n_classes=2, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Define base learners
    base_learners = [
        ('Decision Tree', DecisionTreeClassifier(max_depth=5, random_state=42)),
        ('Random Forest', RandomForestClassifier(n_estimators=10, random_state=42)),
        ('Logistic Regression', LogisticRegression(max_iter=1000, random_state=42))
    ]
    
    # Define meta-learner
    meta_learner = LogisticRegression(max_iter=1000, random_state=42)
    
    # Create and train stacking classifier
    stacking_clf = StackingClassifier(
        base_learners=base_learners,
        meta_learner=meta_learner,
        use_proba=False,
        cv=None,
        random_state=42
    )
    
    print("Training Stacking Classifier...")
    stacking_clf.fit(X_train, y_train)
    
    # Evaluate
    train_acc = stacking_clf.score(X_train, y_train)
    test_acc = stacking_clf.score(X_test, y_test)
    
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Compare with individual base learners
    print("\n=== Individual Base Learner Performance ===")
    for name, learner in base_learners:
        learner.fit(X_train, y_train)
        acc = np.mean(learner.predict(X_test) == y_test)
        print(f"{name:25s}: Test Accuracy = {acc:.4f}")
    
    print("\n=== Stacking Regressor Example ===\n")
    
    # Generate regression data
    X_reg, y_reg = make_regression(n_samples=500, n_features=20, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )
    
    from sklearn.linear_model import Ridge, Lasso
    
    # Define base regressors
    base_regressors = [
        ('Ridge', Ridge(alpha=1.0)),
        ('Lasso', Lasso(alpha=0.1)),
        ('Decision Tree', DecisionTreeClassifier(max_depth=5))
    ]
    
    # Meta-learner for regression
    meta_regressor = Ridge(alpha=1.0)
    
    # Create and train stacking regressor
    stacking_reg = StackingRegressor(
        base_learners=base_regressors,
        meta_learner=meta_regressor,
        cv=None,
        random_state=42
    )
    
    print("Training Stacking Regressor...")
    stacking_reg.fit(X_train, y_train)
    
    # Evaluate
    train_r2 = stacking_reg.score(X_train, y_train)
    test_r2 = stacking_reg.score(X_test, y_test)
    
    print(f"Training R²: {train_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    
    print("\n=== Stacking Parameters ===")
    print(f"Stacking Classifier params: {stacking_clf.get_params()}")
    print(f"Stacking Regressor params: {stacking_reg.get_params()}")