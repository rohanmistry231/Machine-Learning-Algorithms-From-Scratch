"""
Unit Tests for Decision Tree Implementation

This module contains comprehensive tests for the DecisionTree and Node classes,
including tests for:
- Node class functionality
- DecisionTree initialization
- Tree building and splitting
- Classification tasks
- Regression tasks
- Impurity measures (Gini, Entropy, MSE)
- Hyperparameters (max_depth, min_samples_split, min_samples_leaf)
- Predictions and scoring
- Tree depth calculation
- Comparison with scikit-learn
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# Import the models to test
from algorithms.supervised.classification import DecisionTree, Node


class TestNode:
    """Test Node class functionality."""
    
    def test_node_initialization_internal(self):
        """Test internal node initialization."""
        node = Node(feature_index=0, threshold=5.0)
        
        assert node.feature_index == 0
        assert node.threshold == 5.0
        assert node.left is None
        assert node.right is None
        assert node.value is None
    
    def test_node_initialization_leaf(self):
        """Test leaf node initialization."""
        node = Node(value=1)
        
        assert node.feature_index is None
        assert node.threshold is None
        assert node.left is None
        assert node.right is None
        assert node.value == 1
    
    def test_node_is_leaf_true(self):
        """Test is_leaf returns True for leaf nodes."""
        node = Node(value=1)
        
        assert node.is_leaf() is True
    
    def test_node_is_leaf_false(self):
        """Test is_leaf returns False for internal nodes."""
        node = Node(feature_index=0, threshold=5.0)
        
        assert node.is_leaf() is False
    
    def test_node_with_children(self):
        """Test node with left and right children."""
        left_child = Node(value=0)
        right_child = Node(value=1)
        parent = Node(feature_index=0, threshold=5.0, left=left_child, right=right_child)
        
        assert parent.left == left_child
        assert parent.right == right_child
        assert parent.is_leaf() is False


class TestDecisionTreeInitialization:
    """Test DecisionTree initialization."""
    
    def test_default_initialization_classification(self):
        """Test default initialization for classification."""
        tree = DecisionTree()
        
        assert tree.max_depth is None
        assert tree.min_samples_split == 2
        assert tree.min_samples_leaf == 1
        assert tree.criterion == 'gini'
        assert tree.task == 'classification'
        assert tree.max_features is None
        assert tree.root is None
        assert tree.n_features is None
    
    def test_custom_initialization(self):
        """Test custom initialization."""
        tree = DecisionTree(
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            criterion='entropy',
            task='regression',
            max_features=3
        )
        
        assert tree.max_depth == 5
        assert tree.min_samples_split == 10
        assert tree.min_samples_leaf == 5
        assert tree.criterion == 'entropy'
        assert tree.task == 'regression'
        assert tree.max_features == 3
    
    def test_criterion_options(self):
        """Test different criterion options."""
        criteria = ['gini', 'entropy', 'mse']
        
        for criterion in criteria:
            tree = DecisionTree(criterion=criterion)
            assert tree.criterion == criterion


class TestDecisionTreeClassificationFitting:
    """Test DecisionTree fitting for classification."""
    
    @pytest.fixture
    def simple_classification_data(self):
        """Generate simple classification data."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5],
                      [10, 10], [11, 11], [12, 12], [13, 13]])
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        return X, y
    
    @pytest.fixture
    def complex_classification_data(self):
        """Generate complex classification data."""
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_informative=8,
            n_redundant=2,
            random_state=42
        )
        return X, y
    
    def test_basic_fitting(self, simple_classification_data):
        """Test basic tree fitting."""
        X, y = simple_classification_data
        tree = DecisionTree(max_depth=3, task='classification')
        
        result = tree.fit(X, y)
        
        assert result is tree
        assert tree.root is not None
        assert tree.n_features == 2
    
    def test_perfect_fit_simple_data(self, simple_classification_data):
        """Test tree achieves perfect fit on simple separable data."""
        X, y = simple_classification_data
        tree = DecisionTree(max_depth=10, task='classification')
        
        tree.fit(X, y)
        accuracy = tree.score(X, y)
        
        assert accuracy == 1.0
    
    def test_gini_criterion(self, complex_classification_data):
        """Test fitting with Gini criterion."""
        X, y = complex_classification_data
        tree = DecisionTree(
            max_depth=5,
            criterion='gini',
            task='classification'
        )
        
        tree.fit(X, y)
        
        assert tree.root is not None
    
    def test_entropy_criterion(self, complex_classification_data):
        """Test fitting with entropy criterion."""
        X, y = complex_classification_data
        tree = DecisionTree(
            max_depth=5,
            criterion='entropy',
            task='classification'
        )
        
        tree.fit(X, y)
        
        assert tree.root is not None
    
    def test_max_depth_constraint(self, complex_classification_data):
        """Test that max_depth constraint is respected."""
        X, y = complex_classification_data
        
        max_depths = [1, 3, 5, 10]
        
        for max_depth in max_depths:
            tree = DecisionTree(max_depth=max_depth, task='classification')
            tree.fit(X, y)
            actual_depth = tree.get_depth()
            
            assert actual_depth <= max_depth
    
    def test_min_samples_split_constraint(self, complex_classification_data):
        """Test min_samples_split constraint."""
        X, y = complex_classification_data
        
        tree = DecisionTree(
            min_samples_split=50,
            task='classification'
        )
        
        tree.fit(X, y)
        
        assert tree.root is not None
    
    def test_min_samples_leaf_constraint(self, complex_classification_data):
        """Test min_samples_leaf constraint."""
        X, y = complex_classification_data
        
        tree = DecisionTree(
            min_samples_leaf=20,
            task='classification'
        )
        
        tree.fit(X, y)
        
        assert tree.root is not None


class TestDecisionTreeRegressionFitting:
    """Test DecisionTree fitting for regression."""
    
    @pytest.fixture
    def simple_regression_data(self):
        """Generate simple regression data."""
        X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])
        y = np.array([2, 4, 6, 8, 10, 12, 14, 16])
        return X, y
    
    @pytest.fixture
    def complex_regression_data(self):
        """Generate complex regression data."""
        X, y = make_regression(
            n_samples=200,
            n_features=10,
            n_informative=8,
            noise=10,
            random_state=42
        )
        return X, y
    
    def test_basic_regression_fitting(self, simple_regression_data):
        """Test basic regression tree fitting."""
        X, y = simple_regression_data
        tree = DecisionTree(
            max_depth=5,
            criterion='mse',
            task='regression'
        )
        
        result = tree.fit(X, y)
        
        assert result is tree
        assert tree.root is not None
        assert tree.n_features == 1
    
    def test_mse_criterion(self, complex_regression_data):
        """Test fitting with MSE criterion."""
        X, y = complex_regression_data
        tree = DecisionTree(
            max_depth=5,
            criterion='mse',
            task='regression'
        )
        
        tree.fit(X, y)
        
        assert tree.root is not None
    
    def test_regression_perfect_fit(self, simple_regression_data):
        """Test that regression tree can achieve good R² on training data."""
        X, y = simple_regression_data
        tree = DecisionTree(
            max_depth=10,
            criterion='mse',
            task='regression'
        )
        
        tree.fit(X, y)
        r2 = tree.score(X, y)
        
        assert r2 > 0.9


class TestDecisionTreeImpurityMeasures:
    """Test impurity calculation methods."""
    
    def test_gini_impurity_pure(self):
        """Test Gini impurity on pure node."""
        tree = DecisionTree()
        y = np.array([1, 1, 1, 1, 1])
        
        gini = tree._gini_impurity(y)
        
        assert gini == 0.0
    
    def test_gini_impurity_mixed(self):
        """Test Gini impurity on mixed node."""
        tree = DecisionTree()
        y = np.array([0, 0, 1, 1])
        
        gini = tree._gini_impurity(y)
        
        # Gini = 1 - (0.5^2 + 0.5^2) = 0.5
        assert abs(gini - 0.5) < 1e-6
    
    def test_gini_impurity_empty(self):
        """Test Gini impurity on empty array."""
        tree = DecisionTree()
        y = np.array([])
        
        gini = tree._gini_impurity(y)
        
        assert gini == 0.0
    
    def test_entropy_pure(self):
        """Test entropy on pure node."""
        tree = DecisionTree()
        y = np.array([1, 1, 1, 1, 1])
        
        entropy = tree._entropy(y)
        
        assert entropy == 0.0
    
    def test_entropy_mixed(self):
        """Test entropy on mixed node."""
        tree = DecisionTree()
        y = np.array([0, 0, 1, 1])
        
        entropy = tree._entropy(y)
        
        # Entropy = -0.5*log2(0.5) - 0.5*log2(0.5) = 1.0
        assert abs(entropy - 1.0) < 1e-6
    
    def test_entropy_empty(self):
        """Test entropy on empty array."""
        tree = DecisionTree()
        y = np.array([])
        
        entropy = tree._entropy(y)
        
        assert entropy == 0.0
    
    def test_mse_constant(self):
        """Test MSE on constant values."""
        tree = DecisionTree()
        y = np.array([5.0, 5.0, 5.0, 5.0])
        
        mse = tree._mse(y)
        
        assert mse == 0.0
    
    def test_mse_varied(self):
        """Test MSE on varied values."""
        tree = DecisionTree()
        y = np.array([1.0, 2.0, 3.0, 4.0])
        
        mse = tree._mse(y)
        
        # Mean = 2.5, MSE = ((1-2.5)^2 + (2-2.5)^2 + (3-2.5)^2 + (4-2.5)^2) / 4 = 1.25
        assert abs(mse - 1.25) < 1e-6
    
    def test_mse_empty(self):
        """Test MSE on empty array."""
        tree = DecisionTree()
        y = np.array([])
        
        mse = tree._mse(y)
        
        assert mse == 0.0


class TestDecisionTreePrediction:
    """Test DecisionTree prediction functionality."""
    
    @pytest.fixture
    def trained_classifier(self):
        """Create and train a classifier."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5],
                      [10, 10], [11, 11], [12, 12], [13, 13]])
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        
        tree = DecisionTree(max_depth=5, task='classification')
        tree.fit(X, y)
        
        return tree, X, y
    
    @pytest.fixture
    def trained_regressor(self):
        """Create and train a regressor."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])
        
        tree = DecisionTree(max_depth=5, criterion='mse', task='regression')
        tree.fit(X, y)
        
        return tree, X, y
    
    def test_classification_prediction_shape(self, trained_classifier):
        """Test prediction shape for classification."""
        tree, X, _ = trained_classifier
        
        predictions = tree.predict(X)
        
        assert predictions.shape == (8,)
    
    def test_classification_prediction_values(self, trained_classifier):
        """Test that predictions are valid class labels."""
        tree, X, y = trained_classifier
        
        predictions = tree.predict(X)
        
        # Predictions should be from the set of class labels
        assert all(pred in np.unique(y) for pred in predictions)
    
    def test_regression_prediction_shape(self, trained_regressor):
        """Test prediction shape for regression."""
        tree, X, _ = trained_regressor
        
        predictions = tree.predict(X)
        
        assert predictions.shape == (5,)
    
    def test_regression_prediction_type(self, trained_regressor):
        """Test that regression predictions are numeric."""
        tree, X, _ = trained_regressor
        
        predictions = tree.predict(X)
        
        assert all(isinstance(pred, (int, float, np.number)) for pred in predictions)
    
    def test_single_sample_prediction(self, trained_classifier):
        """Test prediction on single sample."""
        tree, _, _ = trained_classifier
        
        X_test = np.array([[1, 2]])
        prediction = tree.predict(X_test)
        
        assert prediction.shape == (1,)


class TestDecisionTreeScoring:
    """Test DecisionTree scoring."""
    
    def test_classification_accuracy_perfect(self):
        """Test accuracy calculation with perfect predictions."""
        X = np.array([[1], [2], [3], [4]])
        y = np.array([0, 0, 1, 1])
        
        tree = DecisionTree(max_depth=5, task='classification')
        tree.fit(X, y)
        
        accuracy = tree.score(X, y)
        
        assert accuracy == 1.0
    
    def test_classification_accuracy_type(self):
        """Test that accuracy returns a float."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        
        tree = DecisionTree(max_depth=5, task='classification')
        tree.fit(X, y)
        
        accuracy = tree.score(X, y)
        
        assert isinstance(accuracy, (float, np.floating))
    
    def test_classification_accuracy_range(self):
        """Test that accuracy is between 0 and 1."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        
        tree = DecisionTree(max_depth=3, task='classification')
        tree.fit(X, y)
        
        accuracy = tree.score(X, y)
        
        assert 0.0 <= accuracy <= 1.0
    
    def test_regression_r2_score(self):
        """Test R² score calculation for regression."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])
        
        tree = DecisionTree(max_depth=5, criterion='mse', task='regression')
        tree.fit(X, y)
        
        r2 = tree.score(X, y)
        
        assert r2 > 0.9


class TestDecisionTreeDepth:
    """Test tree depth calculation."""
    
    def test_get_depth_single_node(self):
        """Test depth of single node tree."""
        X = np.array([[1], [1], [1]])
        y = np.array([0, 0, 0])
        
        tree = DecisionTree(max_depth=1, task='classification')
        tree.fit(X, y)
        
        depth = tree.get_depth()
        
        assert depth == 0
    
    def test_get_depth_increases_with_max_depth(self):
        """Test that depth increases with max_depth parameter."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        
        depths = []
        for max_depth in [1, 3, 5, 10]:
            tree = DecisionTree(max_depth=max_depth, task='classification')
            tree.fit(X, y)
            depths.append(tree.get_depth())
        
        # Depths should generally increase (or stay same)
        assert depths[-1] >= depths[0]
    
    def test_get_depth_respects_max_depth(self):
        """Test that actual depth doesn't exceed max_depth."""
        X, y = make_classification(n_samples=200, n_features=10, random_state=42)
        
        max_depth = 5
        tree = DecisionTree(max_depth=max_depth, task='classification')
        tree.fit(X, y)
        
        actual_depth = tree.get_depth()
        
        assert actual_depth <= max_depth


class TestDecisionTreeMaxFeatures:
    """Test max_features parameter."""
    
    def test_max_features_limits_splits(self):
        """Test that max_features limits features considered for splits."""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        
        tree = DecisionTree(max_features=3, task='classification')
        tree.fit(X, y)
        
        # Tree should be built (just with limited features)
        assert tree.root is not None
    
    def test_max_features_none(self):
        """Test max_features=None uses all features."""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        
        tree = DecisionTree(max_features=None, task='classification')
        tree.fit(X, y)
        
        assert tree.max_features == 10


class TestDecisionTreeComparison:
    """Test DecisionTree against scikit-learn."""
    
    @pytest.fixture
    def classification_dataset(self):
        """Generate classification dataset."""
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_informative=8,
            random_state=42
        )
        return train_test_split(X, y, test_size=0.2, random_state=42)
    
    @pytest.fixture
    def regression_dataset(self):
        """Generate regression dataset."""
        X, y = make_regression(
            n_samples=200,
            n_features=10,
            noise=10,
            random_state=42
        )
        return train_test_split(X, y, test_size=0.2, random_state=42)
    
    def test_classification_accuracy_comparison(self, classification_dataset):
        """Compare classification accuracy with scikit-learn."""
        X_train, X_test, y_train, y_test = classification_dataset
        
        # Our implementation
        tree_ours = DecisionTree(
            max_depth=10,
            criterion='gini',
            task='classification',
            random_state=42
        )
        tree_ours.fit(X_train, y_train)
        accuracy_ours = tree_ours.score(X_test, y_test)
        
        # Scikit-learn
        tree_sklearn = DecisionTreeClassifier(
            max_depth=10,
            criterion='gini',
            random_state=42
        )
        tree_sklearn.fit(X_train, y_train)
        accuracy_sklearn = tree_sklearn.score(X_test, y_test)
        
        # Should be reasonably similar
        assert abs(accuracy_ours - accuracy_sklearn) < 0.2


class TestDecisionTreeEdgeCases:
    """Test edge cases."""
    
    def test_single_sample(self):
        """Test with single sample."""
        X = np.array([[1, 2]])
        y = np.array([0])
        
        tree = DecisionTree(max_depth=5, task='classification')
        tree.fit(X, y)
        
        prediction = tree.predict(X)
        
        assert prediction[0] == y[0]
    
    def test_single_feature(self):
        """Test with single feature."""
        X = np.array([[1], [2], [3], [4]])
        y = np.array([0, 0, 1, 1])
        
        tree = DecisionTree(max_depth=5, task='classification')
        tree.fit(X, y)
        
        accuracy = tree.score(X, y)
        
        assert accuracy > 0
    
    def test_all_same_labels(self):
        """Test with all same labels."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([1, 1, 1])
        
        tree = DecisionTree(max_depth=5, task='classification')
        tree.fit(X, y)
        
        predictions = tree.predict(X)
        
        assert all(pred == 1 for pred in predictions)
    
    def test_binary_classification(self):
        """Test standard binary classification."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5],
                      [10, 10], [11, 11], [12, 12], [13, 13]])
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        
        tree = DecisionTree(max_depth=10, task='classification')
        tree.fit(X, y)
        
        accuracy = tree.score(X, y)
        
        assert accuracy == 1.0
    
    def test_multiclass_classification(self):
        """Test multiclass classification."""
        X, y = make_classification(
            n_samples=150,
            n_features=4,
            n_informative=3,
            n_classes=3,
            random_state=42
        )
        
        tree = DecisionTree(max_depth=10, task='classification')
        tree.fit(X, y)
        
        predictions = tree.predict(X)
        
        # Should have all 3 classes
        assert len(np.unique(predictions)) > 1


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])