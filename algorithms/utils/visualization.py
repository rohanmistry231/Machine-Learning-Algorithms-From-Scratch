"""
Visualization Utilities for Machine Learning

This module provides plotting functions for visualizing ML models,
data, and results. Requires matplotlib.
"""

import numpy as np
from typing import Optional, Tuple, List

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import ListedColormap
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Visualization functions will not work.")


def check_matplotlib():
    """Check if matplotlib is available."""
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization functions")


# ==================== CLASSIFICATION VISUALIZATIONS ====================

def plot_decision_boundary(
    model,
    X: np.ndarray,
    y: np.ndarray,
    title: str = "Decision Boundary",
    xlabel: str = "Feature 1",
    ylabel: str = "Feature 2",
    figsize: Tuple[int, int] = (10, 6),
    resolution: int = 100,
    alpha: float = 0.4
):
    """
    Plot decision boundary for 2D classification problem.
    
    Parameters
    ----------
    model : object
        Trained classifier with predict method
    X : np.ndarray of shape (n_samples, 2)
        2D feature data
    y : np.ndarray
        True labels
    """
    check_matplotlib()
    
    X = np.array(X)
    y = np.array(y)
    
    if X.shape[1] != 2:
        raise ValueError("X must have exactly 2 features for 2D visualization")
    
    # Create mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )
    
    # Predict on mesh
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot
    plt.figure(figsize=figsize)
    plt.contourf(xx, yy, Z, alpha=alpha, cmap=plt.cm.RdYlBu)
    
    # Plot data points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu,
                         edgecolors='black', s=50)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.colorbar(scatter)
    plt.tight_layout()
    
    return plt.gcf()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    cmap: str = "Blues",
    figsize: Tuple[int, int] = (8, 6),
    normalize: bool = False
):
    """
    Plot confusion matrix as heatmap.
    
    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix
    class_names : List[str] or None
        Names of classes
    normalize : bool
        Whether to normalize values
    """
    check_matplotlib()
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    n_classes = cm.shape[0]
    
    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]
    
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    
    # Set ticks
    ax.set(xticks=np.arange(n_classes),
           yticks=np.arange(n_classes),
           xticklabels=class_names,
           yticklabels=class_names,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, format(cm[i, j], fmt),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    title: str = "ROC Curve",
    figsize: Tuple[int, int] = (8, 6)
):
    """
    Plot ROC (Receiver Operating Characteristic) curve.
    
    Parameters
    ----------
    y_true : np.ndarray
        True binary labels
    y_scores : np.ndarray
        Predicted scores/probabilities
    """
    check_matplotlib()
    
    # Sort by scores
    desc_score_indices = np.argsort(y_scores)[::-1]
    y_scores = y_scores[desc_score_indices]
    y_true = y_true[desc_score_indices]
    
    # Calculate TPR and FPR
    tps = np.cumsum(y_true)
    fps = np.cumsum(1 - y_true)
    
    total_positives = tps[-1]
    total_negatives = fps[-1]
    
    tpr = tps / total_positives
    fpr = fps / total_negatives
    
    # Add (0,0) point
    tpr = np.concatenate([[0], tpr])
    fpr = np.concatenate([[0], fpr])
    
    # Calculate AUC
    auc = np.trapz(tpr, fpr)
    
    # Plot
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf()


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    title: str = "Precision-Recall Curve",
    figsize: Tuple[int, int] = (8, 6)
):
    """
    Plot Precision-Recall curve.
    """
    check_matplotlib()
    
    # Sort by scores
    desc_score_indices = np.argsort(y_scores)[::-1]
    y_scores = y_scores[desc_score_indices]
    y_true = y_true[desc_score_indices]
    
    # Calculate precision and recall
    tps = np.cumsum(y_true)
    fps = np.cumsum(1 - y_true)
    
    precision = tps / (tps + fps)
    recall = tps / tps[-1]
    
    # Plot
    plt.figure(figsize=figsize)
    plt.plot(recall, precision, linewidth=2)
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.tight_layout()
    
    return plt.gcf()


# ==================== REGRESSION VISUALIZATIONS ====================

def plot_regression_results(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Regression Results",
    figsize: Tuple[int, int] = (12, 5)
):
    """
    Plot regression predictions vs actual values.
    
    Creates two subplots:
    1. Predicted vs Actual scatter plot
    2. Residual plot
    """
    check_matplotlib()
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Predicted vs Actual
    axes[0].scatter(y_true, y_pred, alpha=0.5, edgecolors='k')
    axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
                'r--', lw=2, label='Perfect Prediction')
    axes[0].set_xlabel('True Values')
    axes[0].set_ylabel('Predicted Values')
    axes[0].set_title('Predicted vs Actual')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Residual plot
    axes[1].scatter(y_pred, residuals, alpha=0.5, edgecolors='k')
    axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Predicted Values')
    axes[1].set_ylabel('Residuals')
    axes[1].set_title('Residual Plot')
    axes[1].grid(alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    return fig


def plot_learning_curve(
    train_sizes: np.ndarray,
    train_scores: np.ndarray,
    val_scores: np.ndarray,
    title: str = "Learning Curve",
    xlabel: str = "Training Examples",
    ylabel: str = "Score",
    figsize: Tuple[int, int] = (8, 6)
):
    """
    Plot learning curve showing training and validation scores.
    """
    check_matplotlib()
    
    plt.figure(figsize=figsize)
    
    plt.plot(train_sizes, train_scores, 'o-', label='Training Score', linewidth=2)
    plt.plot(train_sizes, val_scores, 'o-', label='Validation Score', linewidth=2)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc='best')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf()


# ==================== CLUSTERING VISUALIZATIONS ====================

def plot_clusters(
    X: np.ndarray,
    labels: np.ndarray,
    centers: Optional[np.ndarray] = None,
    title: str = "Cluster Visualization",
    figsize: Tuple[int, int] = (10, 6)
):
    """
    Plot 2D clustering results.
    
    Parameters
    ----------
    X : np.ndarray of shape (n_samples, 2)
        2D feature data
    labels : np.ndarray
        Cluster labels
    centers : np.ndarray or None
        Cluster centers to plot
    """
    check_matplotlib()
    
    X = np.array(X)
    labels = np.array(labels)
    
    if X.shape[1] != 2:
        raise ValueError("X must have exactly 2 features for 2D visualization")
    
    plt.figure(figsize=figsize)
    
    # Plot points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis',
                         alpha=0.6, edgecolors='k', s=50)
    
    # Plot centers if provided
    if centers is not None:
        plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X',
                   s=200, edgecolors='black', linewidths=2, label='Centers')
        plt.legend()
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.colorbar(scatter, label='Cluster')
    plt.tight_layout()
    
    return plt.gcf()


def plot_dendrogram(
    linkage_matrix: list,
    title: str = "Hierarchical Clustering Dendrogram",
    figsize: Tuple[int, int] = (10, 6)
):
    """
    Plot dendrogram for hierarchical clustering.
    
    Note: This is a simplified version. For full dendrograms,
    use scipy.cluster.hierarchy.dendrogram
    """
    check_matplotlib()
    
    plt.figure(figsize=figsize)
    
    # Simple bar plot showing merge history
    n_merges = len(linkage_matrix)
    iterations = list(range(1, n_merges + 1))
    distances = [merge['distance'] for merge in linkage_matrix]
    
    plt.bar(iterations, distances, alpha=0.7)
    plt.xlabel('Merge Step')
    plt.ylabel('Distance')
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf()


# ==================== DIMENSIONALITY REDUCTION VISUALIZATIONS ====================

def plot_pca_variance(
    explained_variance_ratio: np.ndarray,
    title: str = "PCA Explained Variance",
    figsize: Tuple[int, int] = (10, 6)
):
    """
    Plot explained variance ratio for PCA components.
    """
    check_matplotlib()
    
    n_components = len(explained_variance_ratio)
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Individual variance
    axes[0].bar(range(1, n_components + 1), explained_variance_ratio, alpha=0.7)
    axes[0].set_xlabel('Principal Component')
    axes[0].set_ylabel('Explained Variance Ratio')
    axes[0].set_title('Variance per Component')
    axes[0].grid(alpha=0.3)
    
    # Cumulative variance
    axes[1].plot(range(1, n_components + 1), cumulative_variance, 'bo-', linewidth=2)
    axes[1].axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
    axes[1].set_xlabel('Number of Components')
    axes[1].set_ylabel('Cumulative Explained Variance')
    axes[1].set_title('Cumulative Variance')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    return fig


def plot_2d_embedding(
    X_embedded: np.ndarray,
    y: Optional[np.ndarray] = None,
    title: str = "2D Embedding",
    xlabel: str = "Dimension 1",
    ylabel: str = "Dimension 2",
    figsize: Tuple[int, int] = (10, 8)
):
    """
    Plot 2D embedding (e.g., from PCA, t-SNE, LDA).
    
    Parameters
    ----------
    X_embedded : np.ndarray of shape (n_samples, 2)
        2D embedded data
    y : np.ndarray or None
        Labels for coloring points
    """
    check_matplotlib()
    
    plt.figure(figsize=figsize)
    
    if y is not None:
        scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y,
                            cmap='viridis', alpha=0.6, edgecolors='k', s=50)
        plt.colorbar(scatter, label='Class')
    else:
        plt.scatter(X_embedded[:, 0], X_embedded[:, 1],
                   alpha=0.6, edgecolors='k', s=50)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf()


# ==================== FEATURE IMPORTANCE ====================

def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    title: str = "Feature Importance",
    figsize: Tuple[int, int] = (10, 6),
    top_n: Optional[int] = None
):
    """
    Plot feature importance as horizontal bar chart.
    
    Parameters
    ----------
    feature_names : List[str]
        Names of features
    importances : np.ndarray
        Importance scores
    top_n : int or None
        Show only top N features
    """
    check_matplotlib()
    
    # Sort by importance
    indices = np.argsort(importances)[::-1]
    
    if top_n is not None:
        indices = indices[:top_n]
    
    sorted_importances = importances[indices]
    sorted_names = [feature_names[i] for i in indices]
    
    plt.figure(figsize=figsize)
    plt.barh(range(len(sorted_names)), sorted_importances, alpha=0.7)
    plt.yticks(range(len(sorted_names)), sorted_names)
    plt.xlabel('Importance')
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.grid(alpha=0.3, axis='x')
    plt.tight_layout()
    
    return plt.gcf()


# ==================== TRAINING HISTORY ====================

def plot_training_history(
    history: dict,
    title: str = "Training History",
    figsize: Tuple[int, int] = (12, 5)
):
    """
    Plot training history (loss and metrics over epochs).
    
    Parameters
    ----------
    history : dict
        Dictionary with keys like 'loss', 'accuracy', 'val_loss', 'val_accuracy'
    """
    check_matplotlib()
    
    metrics = [key for key in history.keys() if not key.startswith('val_')]
    n_metrics = len(metrics)
    
    if n_metrics == 0:
        raise ValueError("No metrics found in history")
    
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    
    if n_metrics == 1:
        axes = [axes]
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # Plot training metric
        ax.plot(history[metric], label=f'Training {metric}', linewidth=2)
        
        # Plot validation metric if available
        val_metric = f'val_{metric}'
        if val_metric in history:
            ax.plot(history[val_metric], label=f'Validation {metric}', linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'{metric.capitalize()} over Epochs')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    return fig


# ==================== DATA DISTRIBUTION ====================

def plot_data_distribution(
    X: np.ndarray,
    feature_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 8)
):
    """
    Plot distribution of features using histograms.
    
    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Feature data
    feature_names : List[str] or None
        Names of features
    """
    check_matplotlib()
    
    X = np.array(X)
    n_features = X.shape[1]
    
    if feature_names is None:
        feature_names = [f'Feature {i+1}' for i in range(n_features)]
    
    # Calculate grid size
    n_cols = min(4, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_features > 1 else [axes]
    
    for i in range(n_features):
        axes[i].hist(X[:, i], bins=30, alpha=0.7, edgecolor='black')
        axes[i].set_xlabel(feature_names[i])
        axes[i].set_ylabel('Frequency')
        axes[i].set_title(f'Distribution: {feature_names[i]}')
        axes[i].grid(alpha=0.3)
    
    # Hide extra subplots
    for i in range(n_features, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    return fig


def plot_correlation_matrix(
    X: np.ndarray,
    feature_names: Optional[List[str]] = None,
    title: str = "Correlation Matrix",
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = "coolwarm"
):
    """
    Plot correlation matrix heatmap.
    
    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Feature data
    feature_names : List[str] or None
        Names of features
    """
    check_matplotlib()
    
    X = np.array(X)
    n_features = X.shape[1]
    
    if feature_names is None:
        feature_names = [f'F{i+1}' for i in range(n_features)]
    
    # Calculate correlation matrix
    corr_matrix = np.corrcoef(X.T)
    
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(corr_matrix, cmap=cmap, vmin=-1, vmax=1, aspect='auto')
    
    # Set ticks
    ax.set_xticks(np.arange(n_features))
    ax.set_yticks(np.arange(n_features))
    ax.set_xticklabels(feature_names)
    ax.set_yticklabels(feature_names)
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Correlation", rotation=-90, va="bottom")
    
    # Add text annotations
    for i in range(n_features):
        for j in range(n_features):
            text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    ax.set_title(title)
    plt.tight_layout()
    
    return fig


# ==================== MODEL COMPARISON ====================

def plot_model_comparison(
    model_names: List[str],
    scores: List[float],
    metric_name: str = "Accuracy",
    title: str = "Model Comparison",
    figsize: Tuple[int, int] = (10, 6)
):
    """
    Plot bar chart comparing different models.
    
    Parameters
    ----------
    model_names : List[str]
        Names of models
    scores : List[float]
        Performance scores
    metric_name : str
        Name of the metric
    """
    check_matplotlib()
    
    plt.figure(figsize=figsize)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(model_names)))
    bars = plt.bar(model_names, scores, alpha=0.7, color=colors, edgecolor='black')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    plt.xlabel('Model')
    plt.ylabel(metric_name)
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    
    return plt.gcf()


if __name__ == "__main__":
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available. Install it to use visualization functions.")
        print("pip install matplotlib")
    else:
        print("=== Visualization Module Examples ===\n")
        
        # Generate sample data
        np.random.seed(42)
        
        print("1. Confusion Matrix")
        cm = np.array([[50, 10], [5, 35]])
        fig = plot_confusion_matrix(cm, class_names=['Class 0', 'Class 1'])
        plt.savefig('confusion_matrix_example.png', dpi=100, bbox_inches='tight')
        plt.close()
        print("   Saved: confusion_matrix_example.png")
        
        print("\n2. Regression Results")
        y_true = np.random.randn(100) * 10 + 50
        y_pred = y_true + np.random.randn(100) * 5
        fig = plot_regression_results(y_true, y_pred)
        plt.savefig('regression_results_example.png', dpi=100, bbox_inches='tight')
        plt.close()
        print("   Saved: regression_results_example.png")
        
        print("\n3. Learning Curve")
        train_sizes = np.array([50, 100, 150, 200, 250])
        train_scores = np.array([0.7, 0.75, 0.78, 0.8, 0.82])
        val_scores = np.array([0.65, 0.72, 0.74, 0.76, 0.77])
        fig = plot_learning_curve(train_sizes, train_scores, val_scores)
        plt.savefig('learning_curve_example.png', dpi=100, bbox_inches='tight')
        plt.close()
        print("   Saved: learning_curve_example.png")
        
        print("\n4. Model Comparison")
        model_names = ['Linear Reg', 'Ridge', 'Lasso', 'Random Forest']
        scores = [0.85, 0.87, 0.86, 0.92]
        fig = plot_model_comparison(model_names, scores, metric_name='R² Score')
        plt.savefig('model_comparison_example.png', dpi=100, bbox_inches='tight')
        plt.close()
        print("   Saved: model_comparison_example.png")
        
        print("\nAll example plots saved successfully!")