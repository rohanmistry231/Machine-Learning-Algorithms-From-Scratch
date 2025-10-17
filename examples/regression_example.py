"""
Complete Regression Example

This example demonstrates how to:
1. Generate regression data
2. Preprocess and scale features
3. Train multiple regression models
4. Evaluate and compare models
5. Visualize results
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Import regression algorithms
from algorithms.supervised.regression import (
    LinearRegression,
    RidgeRegression,
    LassoRegression,
    PolynomialRegression
)

# Import utilities
from algorithms.utils.preprocessing import StandardScaler, train_test_split as custom_split
from algorithms.utils.metrics import (
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
    r2_score
)
from algorithms.utils.visualization import plot_regression_results, plot_learning_curve


def main():
    """Main example function."""
    
    print("=" * 70)
    print("REGRESSION EXAMPLE: Comparing Multiple Models")
    print("=" * 70)
    
    # ============================================================================
    # 1. GENERATE REGRESSION DATA
    # ============================================================================
    print("\n[1] Generating regression data...")
    X, y = make_regression(
        n_samples=300,
        n_features=10,
        n_informative=8,
        noise=20,
        random_state=42
    )
    
    print(f"Data shape: X={X.shape}, y={y.shape}")
    print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")
    
    # ============================================================================
    # 2. TRAIN-TEST SPLIT
    # ============================================================================
    print("\n[2] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
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
    # 4. TRAIN MULTIPLE REGRESSION MODELS
    # ============================================================================
    print("\n[4] Training regression models...")
    
    models = {
        'Linear Regression': LinearRegression(
            learning_rate=0.01, 
            n_iterations=1000,
            method='gradient_descent'
        ),
        'Ridge Regression': RidgeRegression(
            alpha=1.0,
            learning_rate=0.01,
            n_iterations=1000,
            method='normal_equation'
        ),
        'Lasso Regression': LassoRegression(
            alpha=0.1,
            learning_rate=0.01,
            n_iterations=1000,
            method='coordinate_descent'
        ),
        'Polynomial (degree=2)': PolynomialRegression(
            degree=2,
            learning_rate=0.01,
            n_iterations=1000
        ),
    }
    
    trained_models = {}
    
    for name, model in models.items():
        print(f"\n  Training {name}...")
        model.fit(X_train_scaled, y_train)
        trained_models[name] = model
        print(f"  ✓ {name} trained successfully")
    
    # ============================================================================
    # 5. EVALUATE MODELS
    # ============================================================================
    print("\n[5] Evaluating models...")
    print("\n" + "-" * 70)
    print(f"{'Model':<25} {'MSE':<12} {'RMSE':<12} {'MAE':<12} {'R²':<12}")
    print("-" * 70)
    
    results = {}
    
    for name, model in trained_models.items():
        # Make predictions
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred_test)
        rmse = root_mean_squared_error(y_test, y_pred_test)
        mae = mean_absolute_error(y_test, y_pred_test)
        r2 = r2_score(y_test, y_pred_test)
        
        results[name] = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'y_pred': y_pred_test
        }
        
        print(f"{name:<25} {mse:<12.4f} {rmse:<12.4f} {mae:<12.4f} {r2:<12.4f}")
    
    print("-" * 70)
    
    # ============================================================================
    # 6. COMPARE MODELS
    # ============================================================================
    print("\n[6] Model Comparison:")
    
    # Find best model by R²
    best_model = max(results.items(), key=lambda x: x[1]['r2'])
    worst_model = min(results.items(), key=lambda x: x[1]['r2'])
    
    print(f"\n  Best Model (by R²): {best_model[0]}")
    print(f"    R² Score: {best_model[1]['r2']:.4f}")
    
    print(f"\n  Model with Highest Error: {worst_model[0]}")
    print(f"    R² Score: {worst_model[1]['r2']:.4f}")
    
    # ============================================================================
    # 7. DETAILED ANALYSIS OF BEST MODEL
    # ============================================================================
    print("\n[7] Detailed Analysis of Best Model...")
    
    best_name = best_model[0]
    best_model_obj = trained_models[best_name]
    
    y_pred_best = best_model[1]['y_pred']
    
    # Calculate residuals
    residuals = y_test - y_pred_best
    
    print(f"\n  Model: {best_name}")
    print(f"  Mean Residual: {np.mean(residuals):.4f}")
    print(f"  Std Residual: {np.std(residuals):.4f}")
    print(f"  Min Residual: {np.min(residuals):.4f}")
    print(f"  Max Residual: {np.max(residuals):.4f}")
    
    # ============================================================================
    # 8. VISUALIZATIONS
    # ============================================================================
    print("\n[8] Creating visualizations...")
    
    # Plot regression results for best model
    fig = plot_regression_results(
        y_test, y_pred_best,
        title=f"Regression Results: {best_name}",
        figsize=(12, 5)
    )
    plt.tight_layout()
    plt.savefig('regression_results.png', dpi=100, bbox_inches='tight')
    print("  ✓ Saved: regression_results.png")
    plt.close()
    
    # ============================================================================
    # 9. PREDICT ON NEW DATA
    # ============================================================================
    print("\n[9] Making predictions on new data...")
    
    # Generate new data
    X_new = np.random.randn(5, 10) * X_train_scaled.std() + X_train_scaled.mean()
    
    predictions_new = best_model_obj.predict(X_new)
    
    print(f"\n  New samples predictions:")
    for i, pred in enumerate(predictions_new):
        print(f"    Sample {i+1}: {pred:.2f}")
    
    # ============================================================================
    # 10. CROSS-VALIDATION (MANUAL K-FOLD)
    # ============================================================================
    print("\n[10] Cross-Validation (3-Fold)...")
    
    from algorithms.utils.preprocessing import k_fold_split
    
    fold_scores = []
    
    for fold, (train_idx, test_idx) in enumerate(k_fold_split(X_train_scaled, y_train, n_splits=3)):
        X_fold_train = X_train_scaled[train_idx]
        X_fold_test = X_train_scaled[test_idx]
        y_fold_train = y_train[train_idx]
        y_fold_test = y_train[test_idx]
        
        model = LinearRegression(learning_rate=0.01, n_iterations=1000)
        model.fit(X_fold_train, y_fold_train)
        
        fold_r2 = model.score(X_fold_test, y_fold_test)
        fold_scores.append(fold_r2)
        
        print(f"  Fold {fold + 1}: R² = {fold_r2:.4f}")
    
    print(f"\n  Mean CV R²: {np.mean(fold_scores):.4f} (+/- {np.std(fold_scores):.4f})")
    
    # ============================================================================
    # SUMMARY
    # ============================================================================
    print("\n" + "=" * 70)
    print("EXAMPLE COMPLETE!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  1. Always scale features before training")
    print("  2. Compare multiple models to find the best")
    print("  3. Use appropriate metrics for evaluation")
    print("  4. Cross-validation provides robust estimates")
    print("  5. Visualize results to understand model behavior")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()