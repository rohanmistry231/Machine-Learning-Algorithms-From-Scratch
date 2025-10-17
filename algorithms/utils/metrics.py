"""
Machine Learning Metrics Implementation from Scratch

This module provides various evaluation metrics for classification,
regression, and clustering tasks.
"""

import numpy as np
from typing import Optional, Tuple


# ==================== CLASSIFICATION METRICS ====================

def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate accuracy: fraction of correct predictions.
    
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(y_true == y_pred)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Returns
    -------
    matrix : np.ndarray of shape (n_classes, n_classes)
        Confusion matrix where element [i, j] is the number of observations
        known to be in class i but predicted to be in class j
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)
    
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    
    for i, true_class in enumerate(classes):
        for j, pred_class in enumerate(classes):
            matrix[i, j] = np.sum((y_true == true_class) & (y_pred == pred_class))
    
    return matrix


def precision_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = 'binary',
    pos_label: int = 1
) -> float:
    """
    Calculate precision: TP / (TP + FP)
    
    Parameters
    ----------
    average : str
        'binary', 'macro', 'micro', 'weighted'
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if average == 'binary':
        tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
        fp = np.sum((y_true != pos_label) & (y_pred == pos_label))
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    elif average in ['macro', 'weighted', 'micro']:
        classes = np.unique(y_true)
        precisions = []
        weights = []
        
        for cls in classes:
            tp = np.sum((y_true == cls) & (y_pred == cls))
            fp = np.sum((y_true != cls) & (y_pred == cls))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            precisions.append(precision)
            weights.append(np.sum(y_true == cls))
        
        if average == 'macro':
            return np.mean(precisions)
        elif average == 'weighted':
            return np.average(precisions, weights=weights)
        elif average == 'micro':
            tp_total = np.sum([np.sum((y_true == cls) & (y_pred == cls)) for cls in classes])
            fp_total = np.sum([np.sum((y_true != cls) & (y_pred == cls)) for cls in classes])
            return tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0


def recall_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = 'binary',
    pos_label: int = 1
) -> float:
    """
    Calculate recall (sensitivity): TP / (TP + FN)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if average == 'binary':
        tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
        fn = np.sum((y_true == pos_label) & (y_pred != pos_label))
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    elif average in ['macro', 'weighted', 'micro']:
        classes = np.unique(y_true)
        recalls = []
        weights = []
        
        for cls in classes:
            tp = np.sum((y_true == cls) & (y_pred == cls))
            fn = np.sum((y_true == cls) & (y_pred != cls))
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            recalls.append(recall)
            weights.append(np.sum(y_true == cls))
        
        if average == 'macro':
            return np.mean(recalls)
        elif average == 'weighted':
            return np.average(recalls, weights=weights)
        elif average == 'micro':
            tp_total = np.sum([np.sum((y_true == cls) & (y_pred == cls)) for cls in classes])
            fn_total = np.sum([np.sum((y_true == cls) & (y_pred != cls)) for cls in classes])
            return tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0


def f1_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = 'binary',
    pos_label: int = 1
) -> float:
    """
    Calculate F1 score: harmonic mean of precision and recall.
    
    F1 = 2 * (precision * recall) / (precision + recall)
    """
    precision = precision_score(y_true, y_pred, average, pos_label)
    recall = recall_score(y_true, y_pred, average, pos_label)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)


def roc_auc_score(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Calculate Area Under ROC Curve.
    
    Uses trapezoidal rule for numerical integration.
    """
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    # Sort by scores
    desc_score_indices = np.argsort(y_scores)[::-1]
    y_scores = y_scores[desc_score_indices]
    y_true = y_true[desc_score_indices]
    
    # Calculate TPR and FPR at each threshold
    tps = np.cumsum(y_true)
    fps = np.cumsum(1 - y_true)
    
    total_positives = tps[-1]
    total_negatives = fps[-1]
    
    if total_positives == 0 or total_negatives == 0:
        return 0.5
    
    tpr = tps / total_positives
    fpr = fps / total_negatives
    
    # Add (0,0) and (1,1) points
    tpr = np.concatenate([[0], tpr])
    fpr = np.concatenate([[0], fpr])
    
    # Calculate AUC using trapezoidal rule
    auc = np.trapz(tpr, fpr)
    
    return auc


def log_loss(y_true: np.ndarray, y_pred_proba: np.ndarray, eps: float = 1e-15) -> float:
    """
    Calculate log loss (binary cross-entropy).
    
    Log Loss = -1/n * Σ[y*log(p) + (1-y)*log(1-p)]
    """
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    
    # Clip probabilities to avoid log(0)
    y_pred_proba = np.clip(y_pred_proba, eps, 1 - eps)
    
    return -np.mean(y_true * np.log(y_pred_proba) + (1 - y_true) * np.log(1 - y_pred_proba))


# ==================== REGRESSION METRICS ====================

def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Squared Error.
    
    MSE = (1/n) * Σ(y_true - y_pred)²
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean((y_true - y_pred) ** 2)


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Root Mean Squared Error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error.
    
    MAE = (1/n) * Σ|y_true - y_pred|
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate R² (coefficient of determination).
    
    R² = 1 - (SS_res / SS_tot)
    where SS_res = Σ(y_true - y_pred)²
          SS_tot = Σ(y_true - mean(y_true))²
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if ss_tot == 0:
        return 0.0
    
    return 1 - (ss_res / ss_tot)


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error.
    
    MAPE = (100/n) * Σ|((y_true - y_pred) / y_true)|
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Avoid division by zero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def explained_variance_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate explained variance.
    
    EV = 1 - Var(y_true - y_pred) / Var(y_true)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    var_residual = np.var(y_true - y_pred)
    var_y = np.var(y_true)
    
    if var_y == 0:
        return 0.0
    
    return 1 - (var_residual / var_y)


# ==================== CLUSTERING METRICS ====================

def silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculate Silhouette Coefficient.
    
    s(i) = (b(i) - a(i)) / max(a(i), b(i))
    
    where a(i) = avg distance to points in same cluster
          b(i) = avg distance to points in nearest cluster
    """
    X = np.array(X)
    labels = np.array(labels)
    
    unique_labels = np.unique(labels)
    n_samples = X.shape[0]
    
    if len(unique_labels) == 1:
        return 0.0
    
    silhouette_vals = []
    
    for i in range(n_samples):
        cluster_i = labels[i]
        
        # Calculate a(i): mean distance to points in same cluster
        same_cluster_mask = labels == cluster_i
        same_cluster_points = X[same_cluster_mask]
        
        if len(same_cluster_points) > 1:
            a_i = np.mean([np.linalg.norm(X[i] - x) for x in same_cluster_points if not np.array_equal(x, X[i])])
        else:
            a_i = 0
        
        # Calculate b(i): min mean distance to points in other clusters
        b_i = float('inf')
        for cluster_j in unique_labels:
            if cluster_j != cluster_i:
                other_cluster_mask = labels == cluster_j
                other_cluster_points = X[other_cluster_mask]
                mean_dist = np.mean([np.linalg.norm(X[i] - x) for x in other_cluster_points])
                b_i = min(b_i, mean_dist)
        
        # Calculate silhouette for sample i
        if max(a_i, b_i) > 0:
            s_i = (b_i - a_i) / max(a_i, b_i)
        else:
            s_i = 0
        
        silhouette_vals.append(s_i)
    
    return np.mean(silhouette_vals)


def calinski_harabasz_score(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculate Calinski-Harabasz Index (Variance Ratio Criterion).
    
    CH = (SSB / SSW) * ((n - k) / (k - 1))
    
    where SSB = between-cluster dispersion
          SSW = within-cluster dispersion
          n = number of samples
          k = number of clusters
    """
    X = np.array(X)
    labels = np.array(labels)
    
    unique_labels = np.unique(labels)
    n_samples = X.shape[0]
    n_clusters = len(unique_labels)
    
    if n_clusters == 1 or n_clusters == n_samples:
        return 0.0
    
    # Calculate overall centroid
    overall_centroid = np.mean(X, axis=0)
    
    # Calculate SSB (between-cluster dispersion)
    ssb = 0
    for label in unique_labels:
        cluster_points = X[labels == label]
        cluster_size = len(cluster_points)
        cluster_centroid = np.mean(cluster_points, axis=0)
        ssb += cluster_size * np.sum((cluster_centroid - overall_centroid) ** 2)
    
    # Calculate SSW (within-cluster dispersion)
    ssw = 0
    for label in unique_labels:
        cluster_points = X[labels == label]
        cluster_centroid = np.mean(cluster_points, axis=0)
        ssw += np.sum((cluster_points - cluster_centroid) ** 2)
    
    if ssw == 0:
        return 0.0
    
    ch_score = (ssb / ssw) * ((n_samples - n_clusters) / (n_clusters - 1))
    
    return ch_score


def davies_bouldin_score(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculate Davies-Bouldin Index.
    
    Lower values indicate better clustering.
    
    DB = (1/k) * Σᵢ max_j≠i [(sᵢ + sⱼ) / dᵢⱼ]
    
    where sᵢ = avg distance of points in cluster i to centroid i
          dᵢⱼ = distance between centroids i and j
    """
    X = np.array(X)
    labels = np.array(labels)
    
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    if n_clusters == 1:
        return 0.0
    
    # Calculate centroids and average distances
    centroids = []
    avg_distances = []
    
    for label in unique_labels:
        cluster_points = X[labels == label]
        centroid = np.mean(cluster_points, axis=0)
        centroids.append(centroid)
        
        # Average distance to centroid
        avg_dist = np.mean([np.linalg.norm(point - centroid) for point in cluster_points])
        avg_distances.append(avg_dist)
    
    centroids = np.array(centroids)
    avg_distances = np.array(avg_distances)
    
    # Calculate DB index
    db_values = []
    for i in range(n_clusters):
        max_ratio = 0
        for j in range(n_clusters):
            if i != j:
                centroid_distance = np.linalg.norm(centroids[i] - centroids[j])
                if centroid_distance > 0:
                    ratio = (avg_distances[i] + avg_distances[j]) / centroid_distance
                    max_ratio = max(max_ratio, ratio)
        db_values.append(max_ratio)
    
    return np.mean(db_values)


# ==================== UTILITY FUNCTIONS ====================

def classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[list] = None
) -> str:
    """
    Generate a classification report with precision, recall, F1, and support.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    classes = np.unique(y_true)
    
    if target_names is None:
        target_names = [f"Class {cls}" for cls in classes]
    
    report = "              precision    recall  f1-score   support\n\n"
    
    for i, cls in enumerate(classes):
        precision = precision_score(y_true, y_pred, average='binary', pos_label=cls)
        recall = recall_score(y_true, y_pred, average='binary', pos_label=cls)
        f1 = f1_score(y_true, y_pred, average='binary', pos_label=cls)
        support = np.sum(y_true == cls)
        
        report += f"{target_names[i]:>12}  {precision:>9.2f}  {recall:>7.2f}  {f1:>8.2f}  {support:>8d}\n"
    
    # Add macro and weighted averages
    macro_precision = precision_score(y_true, y_pred, average='macro')
    macro_recall = recall_score(y_true, y_pred, average='macro')
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    
    weighted_precision = precision_score(y_true, y_pred, average='weighted')
    weighted_recall = recall_score(y_true, y_pred, average='weighted')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    
    total_support = len(y_true)
    
    report += f"\n   macro avg  {macro_precision:>9.2f}  {macro_recall:>7.2f}  {macro_f1:>8.2f}  {total_support:>8d}\n"
    report += f"weighted avg  {weighted_precision:>9.2f}  {weighted_recall:>7.2f}  {weighted_f1:>8.2f}  {total_support:>8d}\n"
    
    return report


if __name__ == "__main__":
    print("=== Classification Metrics Example ===")
    y_true = np.array([0, 1, 1, 0, 1, 1, 0, 0, 1, 0])
    y_pred = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 0])
    
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_true, y_pred):.4f}")
    print(f"\nConfusion Matrix:\n{confusion_matrix(y_true, y_pred)}")
    
    print("\n=== Regression Metrics Example ===")
    y_true_reg = np.array([3.0, -0.5, 2.0, 7.0])
    y_pred_reg = np.array([2.5, 0.0, 2.0, 8.0])
    
    print(f"MSE: {mean_squared_error(y_true_reg, y_pred_reg):.4f}")
    print(f"RMSE: {root_mean_squared_error(y_true_reg, y_pred_reg):.4f}")
    print(f"MAE: {mean_absolute_error(y_true_reg, y_pred_reg):.4f}")
    print(f"R² Score: {r2_score(y_true_reg, y_pred_reg):.4f}")
    
    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred, target_names=['Negative', 'Positive']))