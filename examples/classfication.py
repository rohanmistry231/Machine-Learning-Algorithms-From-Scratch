"""
Complete Classification Example

This example demonstrates how to:
1. Generate classification data
2. Preprocess and scale features
3. Train multiple classification models
4. Evaluate and compare models
5. Visualize results
6. Perform cross-validation
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Import classification algorithms
from algorithms.supervised.classification import (
    KNN,
    NaiveBayes,
    DecisionTree,
    SVM,
    RandomForest,
    GradientBoosting
)

# Import utilities
from algorithms.utils.preprocessing import StandardScaler, train_test_split as custom_split
from algorithms.utils.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from algorithms.utils.visualization import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_classification_results
)


def main():
    """Main example function."""
    
    print("=" * 70)
    print("CLASSIFICATION EXAMPLE: Comparing Multiple Models")
    print("=" * 70)
    
    # ============================================================================
    # 1. GENERATE CLASSIFICATION DATA
    # ============================================================================
    print("\n[1] Generating classification data...")
    X, y = make_classification(
        n_samples=500,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        n_classes=2,
        random_state=42,
        shuffle=True
    )
    
    print(f"Data shape: X={X.shape}, y={y.shape}")
    print(f"Class distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for cls, count in zip(unique, counts):
        percentage = (count / len(y)) * 100
        print(f"  Class {cls}: {count} samples ({percentage:.1f}%)")
    
    # ============================================================================
    # 2. TRAIN-TEST SPLIT
    # ============================================================================
    print("\n[2] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # ============================================================================
    # 3. PREPROCESS: SCALE FEATURES
    # ============================================================================
    print("\n[3] Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Original mean: {X_train.mean():.4f}, std: {X_train.std():.4f}")
    print(f"Scaled mean: {X_train_scaled.mean():.4f}, std: {X_train_scaled.std():.4f}")
    
    # ============================================================================
    # 4. TRAIN MULTIPLE CLASSIFICATION MODELS
    # ============================================================================
    print("\n[4] Training classification models...")
    
    models = {
        'K-Nearest Neighbors': KNN(k=5, distance_metric='euclidean'),
        
        'Naive Bayes': NaiveBayes(model_type='gaussian'),
        
        'Decision Tree': DecisionTree(
            max_depth=10,
            min_samples_split=5,
            criterion='gini'
        ),
        
        'Support Vector Machine': SVM(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            random_state=42
        ),
        
        'Random Forest': RandomForest(
            n_estimators=10,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        ),
        
        'Gradient Boosting': GradientBoosting(
            n_estimators=10,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
    }
    
    trained_models = {}
    
    for name, model in models.items():
        print(f"\n  Training {name}...")
        try:
            model.fit(X_train_scaled, y_train)
            trained_models[name] = model
            print(f"  ✓ {name} trained successfully")
        except Exception as e:
            print(f"  ✗ Error training {name}: {str(e)}")
    
    # ============================================================================
    # 5. EVALUATE MODELS
    # ============================================================================
    print("\n[5] Evaluating models...")
    print("\n" + "-" * 90)
    print(f"{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 90)
    
    results = {}
    
    for name, model in trained_models.items():
        # Make predictions
        try:
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'y_pred': y_pred,
                'model': model
            }
            
            print(f"{name:<25} {accuracy:<12.4f} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f}")
            
        except Exception as e:
            print(f"{name:<25} Error: {str(e)}")
    
    print("-" * 90)
    
    # ============================================================================
    # 6. COMPARE MODELS
    # ============================================================================
    print("\n[6] Model Comparison:")
    
    if results:
        # Find best model by accuracy
        best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
        worst_model = min(results.items(), key=lambda x: x[1]['accuracy'])
        
        print(f"\n  Best Model (by Accuracy): {best_model[0]}")
        print(f"    Accuracy: {best_model[1]['accuracy']:.4f}")
        print(f"    F1-Score: {best_model[1]['f1']:.4f}")
        
        print(f"\n  Model with Lowest Accuracy: {worst_model[0]}")
        print(f"    Accuracy: {worst_model[1]['accuracy']:.4f}")
        print(f"    F1-Score: {worst_model[1]['f1']:.4f}")
        
        # ====================================================================
        # 7. DETAILED ANALYSIS OF BEST MODEL
        # ====================================================================
        print("\n[7] Detailed Analysis of Best Model...")
        
        best_name = best_model[0]
        y_pred_best = best_model[1]['y_pred']
        
        print(f"\n  Model: {best_name}")
        print(f"\n  Classification Report:")
        print(classification_report(y_test, y_pred_best))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred_best)
        print(f"\n  Confusion Matrix:")
        print(cm)
        
        # Calculate metrics from confusion matrix
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        print(f"\n  Additional Metrics:")
        print(f"    True Negatives: {tn}")
        print(f"    False Positives: {fp}")
        print(f"    False Negatives: {fn}")
        print(f"    True Positives: {tp}")
        print(f"    Sensitivity (Recall): {sensitivity:.4f}")
        print(f"    Specificity: {specificity:.4f}")
        
        # ====================================================================
        # 8. VISUALIZATIONS
        # ====================================================================
        print("\n[8] Creating visualizations...")
        
        # Plot confusion matrix for best model
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        plot_confusion_matrix(y_test, y_pred_best, ax=ax,
                             title=f"Confusion Matrix: {best_name}")
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=100, bbox_inches='tight')
        print("  ✓ Saved: confusion_matrix.png")
        plt.close()
        
        # Model comparison bar chart
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        model_names = list(results.keys())
        accuracies = [results[m]['accuracy'] for m in model_names]
        f1_scores = [results[m]['f1'] for m in model_names]
        
        # Accuracy comparison
        axes[0].barh(model_names, accuracies, color='steelblue')
        axes[0].set_xlabel('Accuracy')
        axes[0].set_title('Model Accuracy Comparison')
        axes[0].set_xlim([0, 1])
        for i, v in enumerate(accuracies):
            axes[0].text(v + 0.01, i, f'{v:.3f}', va='center')
        
        # F1-Score comparison
        axes[1].barh(model_names, f1_scores, color='coral')
        axes[1].set_xlabel('F1-Score')
        axes[1].set_title('Model F1-Score Comparison')
        axes[1].set_xlim([0, 1])
        for i, v in enumerate(f1_scores):
            axes[1].text(v + 0.01, i, f'{v:.3f}', va='center')
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=100, bbox_inches='tight')
        print("  ✓ Saved: model_comparison.png")
        plt.close()
        
        # ====================================================================
        # 9. PREDICT ON NEW DATA
        # ====================================================================
        print("\n[9] Making predictions on new data...")
        
        # Generate new data
        X_new = np.random.randn(5, 10) * X_train_scaled.std() + X_train_scaled.mean()
        
        best_model_obj = best_model[1]['model']
        predictions_new = best_model_obj.predict(X_new)
        
        print(f"\n  New samples predictions ({best_name}):")
        for i, pred in enumerate(predictions_new):
            print(f"    Sample {i+1}: Class {pred}")
        
        # ====================================================================
        # 10. CROSS-VALIDATION (MANUAL K-FOLD)
        # ====================================================================
        print("\n[10] Cross-Validation (3-Fold) for Best Model...")
        
        from algorithms.utils.preprocessing import k_fold_split
        
        fold_scores = []
        
        for fold, (train_idx, test_idx) in enumerate(
            k_fold_split(X_train_scaled, y_train, n_splits=3)
        ):
            X_fold_train = X_train_scaled[train_idx]
            X_fold_test = X_train_scaled[test_idx]
            y_fold_train = y_train[train_idx]
            y_fold_test = y_train[test_idx]
            
            # Use same model type as best model
            if best_name == 'K-Nearest Neighbors':
                model = KNN(k=5)
            elif best_name == 'Naive Bayes':
                model = NaiveBayes(model_type='gaussian')
            elif best_name == 'Decision Tree':
                model = DecisionTree(max_depth=10)
            else:
                model = best_model_obj.__class__()
            
            model.fit(X_fold_train, y_fold_train)
            y_pred_fold = model.predict(X_fold_test)
            
            fold_accuracy = accuracy_score(y_fold_test, y_pred_fold)
            fold_scores.append(fold_accuracy)
            
            print(f"  Fold {fold + 1}: Accuracy = {fold_accuracy:.4f}")
        
        print(f"\n  Mean CV Accuracy: {np.mean(fold_scores):.4f} (+/- {np.std(fold_scores):.4f})")
    
    # ============================================================================
    # SUMMARY
    # ============================================================================
    print("\n" + "=" * 70)
    print("EXAMPLE COMPLETE!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  1. Always scale features before training")
    print("  2. Use stratified split for imbalanced datasets")
    print("  3. Compare multiple models to find the best")
    print("  4. Use appropriate metrics (Accuracy, Precision, Recall, F1)")
    print("  5. Visualize confusion matrix to understand model errors")
    print("  6. Cross-validation provides robust estimates")
    print("  7. Consider business context when choosing metrics")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()