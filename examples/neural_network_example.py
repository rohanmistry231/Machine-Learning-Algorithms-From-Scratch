"""
Complete Neural Network Example

This example demonstrates how to:
1. Generate neural network data
2. Build custom neural network architectures
3. Train networks with different configurations
4. Visualize learning curves
5. Evaluate and compare networks
6. Make predictions
7. Analyze network behavior
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression, make_blobs
from sklearn.model_selection import train_test_split

# Import neural network algorithms
from algorithms.supervised.neural_networks import (
    Perceptron,
    MultiLayerPerceptron,
    Backpropagation
)

# Import utilities
from algorithms.utils.preprocessing import StandardScaler
from algorithms.utils.metrics import (
    accuracy_score,
    mean_squared_error,
    confusion_matrix,
    classification_report
)
from algorithms.utils.visualization import (
    plot_decision_boundary,
    plot_learning_curve
)


def main():
    """Main example function."""
    
    print("=" * 70)
    print("NEURAL NETWORK EXAMPLE: Building and Training Networks")
    print("=" * 70)
    
    # ============================================================================
    # 1. PERCEPTRON - BINARY CLASSIFICATION
    # ============================================================================
    print("\n" + "=" * 70)
    print("PART 1: PERCEPTRON - Simple Binary Classification")
    print("=" * 70)
    
    print("\n[1.1] Generating binary classification data...")
    X_bin, y_bin = make_classification(
        n_samples=200,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_classes=2,
        random_state=42,
        shuffle=True
    )
    
    X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
        X_bin, y_bin, test_size=0.2, random_state=42
    )
    
    scaler_bin = StandardScaler()
    X_train_bin_scaled = scaler_bin.fit_transform(X_train_bin)
    X_test_bin_scaled = scaler_bin.transform(X_test_bin)
    
    print(f"Training set: {X_train_bin_scaled.shape}")
    print(f"Test set: {X_test_bin_scaled.shape}")
    
    print("\n[1.2] Training Perceptron...")
    perceptron = Perceptron(
        learning_rate=0.01,
        n_iterations=100,
        random_state=42
    )
    perceptron.fit(X_train_bin_scaled, y_train_bin)
    
    y_pred_perceptron = perceptron.predict(X_test_bin_scaled)
    accuracy_perceptron = accuracy_score(y_test_bin, y_pred_perceptron)
    
    print(f"✓ Perceptron trained")
    print(f"  Accuracy: {accuracy_perceptron:.4f}")
    
    # Visualize perceptron decision boundary
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_decision_boundary(
        X_test_bin_scaled, y_test_bin, perceptron,
        ax=ax, title="Perceptron Decision Boundary"
    )
    plt.tight_layout()
    plt.savefig('perceptron_decision_boundary.png', dpi=100, bbox_inches='tight')
    print(f"  ✓ Saved: perceptron_decision_boundary.png")
    plt.close()
    
    # ============================================================================
    # 2. MULTI-LAYER PERCEPTRON - CLASSIFICATION
    # ============================================================================
    print("\n" + "=" * 70)
    print("PART 2: MULTI-LAYER PERCEPTRON - Complex Classification")
    print("=" * 70)
    
    print("\n[2.1] Generating classification data (non-linearly separable)...")
    from sklearn.datasets import make_moons
    X_cls, y_cls = make_moons(n_samples=300, noise=0.1, random_state=42)
    
    X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
        X_cls, y_cls, test_size=0.2, random_state=42
    )
    
    scaler_cls = StandardScaler()
    X_train_cls_scaled = scaler_cls.fit_transform(X_train_cls)
    X_test_cls_scaled = scaler_cls.transform(X_test_cls)
    
    print(f"Training set: {X_train_cls_scaled.shape}")
    print(f"Test set: {X_test_cls_scaled.shape}")
    
    # Train multiple MLP architectures
    print("\n[2.2] Training MLPs with different architectures...")
    
    mlp_configs = {
        'Shallow (64)': [64],
        'Medium (128-64)': [128, 64],
        'Deep (256-128-64)': [256, 128, 64],
        'Wide (512)': [512]
    }
    
    mlp_results = {}
    
    for config_name, hidden_layers in mlp_configs.items():
        print(f"\n  Training MLP: {config_name}")
        print(f"  Architecture: {[X_train_cls_scaled.shape[1]]} -> {hidden_layers} -> [2]")
        
        mlp = MultiLayerPerceptron(
            hidden_layers=hidden_layers,
            activation='relu',
            output_activation='sigmoid',
            learning_rate=0.01,
            n_iterations=500,
            batch_size=32,
            random_state=42,
            verbose=False
        )
        
        mlp.fit(X_train_cls_scaled, y_train_cls)
        
        y_pred_mlp = mlp.predict(X_test_cls_scaled)
        accuracy_mlp = accuracy_score(y_test_cls, y_pred_mlp)
        
        mlp_results[config_name] = {
            'model': mlp,
            'accuracy': accuracy_mlp,
            'y_pred': y_pred_mlp,
            'loss_history': mlp.loss_history_
        }
        
        print(f"  ✓ {config_name} trained")
        print(f"    Accuracy: {accuracy_mlp:.4f}")
        print(f"    Final Loss: {mlp.loss_history_[-1]:.4f}")
    
    # Compare MLP architectures
    print("\n[2.3] Architecture Comparison:")
    print("-" * 50)
    print(f"{'Architecture':<20} {'Accuracy':<15} {'Final Loss':<15}")
    print("-" * 50)
    
    best_mlp = max(mlp_results.items(), key=lambda x: x[1]['accuracy'])
    
    for config_name, result in mlp_results.items():
        print(f"{config_name:<20} {result['accuracy']:<15.4f} {result['loss_history'][-1]:<15.6f}")
    
    print("-" * 50)
    print(f"\n  Best Architecture: {best_mlp[0]} (Accuracy: {best_mlp[1]['accuracy']:.4f})")
    
    # Plot learning curves
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    
    for idx, (config_name, result) in enumerate(mlp_results.items()):
        if idx < 4:
            ax = axes[idx]
            loss_history = result['loss_history']
            ax.plot(loss_history, linewidth=2, color='steelblue')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title(f"Learning Curve: {config_name}\n(Final Accuracy: {result['accuracy']:.4f})")
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mlp_learning_curves.png', dpi=100, bbox_inches='tight')
    print(f"  ✓ Saved: mlp_learning_curves.png")
    plt.close()
    
    # Visualize best MLP decision boundary
    best_mlp_obj = best_mlp[1]['model']
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_decision_boundary(
        X_test_cls_scaled, y_test_cls, best_mlp_obj,
        ax=ax, title=f"MLP Decision Boundary: {best_mlp[0]}"
    )
    plt.tight_layout()
    plt.savefig('mlp_decision_boundary.png', dpi=100, bbox_inches='tight')
    print(f"  ✓ Saved: mlp_decision_boundary.png")
    plt.close()
    
    # ============================================================================
    # 3. REGRESSION WITH NEURAL NETWORK
    # ============================================================================
    print("\n" + "=" * 70)
    print("PART 3: NEURAL NETWORK - REGRESSION")
    print("=" * 70)
    
    print("\n[3.1] Generating regression data...")
    X_reg, y_reg = make_regression(
        n_samples=200,
        n_features=5,
        n_informative=5,
        noise=20,
        random_state=42
    )
    
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )
    
    scaler_reg = StandardScaler()
    X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
    X_test_reg_scaled = scaler_reg.transform(X_test_reg)
    
    # Normalize targets
    y_train_reg_mean = y_train_reg.mean()
    y_train_reg_std = y_train_reg.std()
    y_train_reg_normalized = (y_train_reg - y_train_reg_mean) / y_train_reg_std
    y_test_reg_normalized = (y_test_reg - y_train_reg_mean) / y_train_reg_std
    
    print(f"Training set: {X_train_reg_scaled.shape}")
    print(f"Test set: {X_test_reg_scaled.shape}")
    
    print("\n[3.2] Training MLPs for Regression...")
    
    regression_configs = {
        'Simple (32)': [32],
        'Medium (64-32)': [64, 32],
        'Deep (128-64-32)': [128, 64, 32]
    }
    
    reg_results = {}
    
    for config_name, hidden_layers in regression_configs.items():
        print(f"\n  Training: {config_name}")
        
        mlp_reg = MultiLayerPerceptron(
            hidden_layers=hidden_layers,
            activation='relu',
            output_activation='linear',
            learning_rate=0.01,
            n_iterations=500,
            batch_size=16,
            random_state=42,
            verbose=False
        )
        
        mlp_reg.fit(X_train_reg_scaled, y_train_reg_normalized)
        
        y_pred_reg = mlp_reg.predict(X_test_reg_scaled)
        mse = mean_squared_error(y_test_reg_normalized, y_pred_reg)
        
        reg_results[config_name] = {
            'model': mlp_reg,
            'mse': mse,
            'y_pred': y_pred_reg,
            'loss_history': mlp_reg.loss_history_
        }
        
        print(f"  ✓ {config_name} trained")
        print(f"    MSE: {mse:.4f}")
    
    print("\n[3.3] Regression Architecture Comparison:")
    print("-" * 50)
    print(f"{'Architecture':<20} {'MSE':<15} {'Final Loss':<15}")
    print("-" * 50)
    
    best_reg = min(reg_results.items(), key=lambda x: x[1]['mse'])
    
    for config_name, result in reg_results.items():
        print(f"{config_name:<20} {result['mse']:<15.4f} {result['loss_history'][-1]:<15.6f}")
    
    print("-" * 50)
    print(f"\n  Best Architecture: {best_reg[0]} (MSE: {best_reg[1]['mse']:.4f})")
    
    # Plot regression learning curves
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    for idx, (config_name, result) in enumerate(reg_results.items()):
        ax = axes[idx]
        loss_history = result['loss_history']
        ax.plot(loss_history, linewidth=2, color='coral')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f"Regression Learning Curve: {config_name}\n(Final MSE: {result['mse']:.4f})")
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mlp_regression_learning_curves.png', dpi=100, bbox_inches='tight')
    print(f"  ✓ Saved: mlp_regression_learning_curves.png")
    plt.close()
    
    # Plot regression predictions
    best_reg_obj = best_reg[1]['model']
    best_reg_pred = best_reg[1]['y_pred']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Sort by true values for better visualization
    sort_idx = np.argsort(y_test_reg_normalized)
    ax.plot(y_test_reg_normalized[sort_idx], 'o-', label='True Values', linewidth=2, markersize=8)
    ax.plot(best_reg_pred[sort_idx], 's-', label='Predictions', linewidth=2, markersize=6, alpha=0.7)
    ax.fill_between(range(len(sort_idx)), 
                     y_test_reg_normalized[sort_idx], 
                     best_reg_pred[sort_idx], 
                     alpha=0.2, color='gray', label='Error')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Normalized Target Value')
    ax.set_title(f"Regression Predictions: {best_reg[0]}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mlp_regression_predictions.png', dpi=100, bbox_inches='tight')
    print(f"  ✓ Saved: mlp_regression_predictions.png")
    plt.close()
    
    # ============================================================================
    # 4. BACKPROPAGATION ANALYSIS
    # ============================================================================
    print("\n" + "=" * 70)
    print("PART 4: BACKPROPAGATION ANALYSIS")
    print("=" * 70)
    
    print("\n[4.1] Training network with different learning rates...")
    
    learning_rates = [0.001, 0.01, 0.1, 0.5]
    lr_results = {}
    
    for lr in learning_rates:
        print(f"\n  Training with learning_rate={lr}...")
        
        mlp_lr = MultiLayerPerceptron(
            hidden_layers=[64, 32],
            activation='relu',
            output_activation='sigmoid',
            learning_rate=lr,
            n_iterations=300,
            batch_size=32,
            random_state=42,
            verbose=False
        )
        
        mlp_lr.fit(X_train_cls_scaled, y_train_cls)
        
        y_pred_lr = mlp_lr.predict(X_test_cls_scaled)
        accuracy_lr = accuracy_score(y_test_cls, y_pred_lr)
        
        lr_results[lr] = {
            'accuracy': accuracy_lr,
            'loss_history': mlp_lr.loss_history_
        }
        
        print(f"  ✓ Learning rate {lr}: Accuracy = {accuracy_lr:.4f}")
    
    print("\n[4.2] Learning Rate Impact on Convergence:")
    print("-" * 50)
    print(f"{'Learning Rate':<20} {'Final Accuracy':<20} {'Final Loss':<20}")
    print("-" * 50)
    
    for lr, result in lr_results.items():
        print(f"{lr:<20} {result['accuracy']:<20.4f} {result['loss_history'][-1]:<20.6f}")
    
    print("-" * 50)
    
    # Plot learning rate comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for lr, result in lr_results.items():
        loss_history = result['loss_history']
        ax.plot(loss_history, linewidth=2, label=f'LR={lr}', marker='o', markevery=20)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Effect of Learning Rate on Training Loss')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('learning_rate_comparison.png', dpi=100, bbox_inches='tight')
    print(f"  ✓ Saved: learning_rate_comparison.png")
    plt.close()
    
    # ============================================================================
    # 5. NETWORK PREDICTIONS AND ANALYSIS
    # ============================================================================
    print("\n" + "=" * 70)
    print("PART 5: PREDICTIONS AND DETAILED ANALYSIS")
    print("=" * 70)
    
    print("\n[5.1] Making predictions on test data (Best Classification Model)...")
    
    best_mlp_final = best_mlp[1]['model']
    y_pred_final = best_mlp[1]['y_pred']
    
    # Classification report
    print(f"\nModel: {best_mlp[0]}")
    print("\nClassification Report:")
    print(classification_report(y_test_cls, y_pred_final))
    
    # Confusion matrix
    cm = confusion_matrix(y_test_cls, y_pred_final)
    print(f"\nConfusion Matrix:")
    print(cm)
    
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    
    print(f"\nAdditional Metrics:")
    print(f"  Sensitivity: {sensitivity:.4f}")
    print(f"  Specificity: {specificity:.4f}")
    print(f"  False Positive Rate: {fp / (fp + tn):.4f}")
    print(f"  False Negative Rate: {fn / (fn + tp):.4f}")
    
    print("\n[5.2] Generating predictions for new data...")
    
    # Generate new data
    X_new = np.random.randn(5, 2) * X_train_cls_scaled.std() + X_train_cls_scaled.mean()
    
    predictions_new = best_mlp_final.predict(X_new)
    
    print(f"\nPredictions for 5 new samples ({best_mlp[0]}):")
    for i, pred in enumerate(predictions_new):
        print(f"  Sample {i+1}: Class {pred}")
    
    # ============================================================================
    # 6. NETWORK ARCHITECTURE VISUALIZATION
    # ============================================================================
    print("\n[6] Creating network architecture summary...")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create text summary of architectures
    summary_text = "NEURAL NETWORK ARCHITECTURES COMPARED\n" + "=" * 50 + "\n\n"
    
    summary_text += "CLASSIFICATION TASK (Non-linearly separable data):\n"
    summary_text += "-" * 50 + "\n"
    for config_name, result in mlp_results.items():
        summary_text += f"{config_name}:\n"
        summary_text += f"  Accuracy: {result['accuracy']:.4f}\n"
        summary_text += f"  Final Loss: {result['loss_history'][-1]:.6f}\n\n"
    
    summary_text += "\nREGRESSION TASK:\n"
    summary_text += "-" * 50 + "\n"
    for config_name, result in reg_results.items():
        summary_text += f"{config_name}:\n"
        summary_text += f"  MSE: {result['mse']:.4f}\n"
        summary_text += f"  Final Loss: {result['loss_history'][-1]:.6f}\n\n"
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('network_architecture_summary.png', dpi=100, bbox_inches='tight')
    print("  ✓ Saved: network_architecture_summary.png")
    plt.close()
    
    # ============================================================================
    # SUMMARY
    # ============================================================================
    print("\n" + "=" * 70)
    print("EXAMPLE COMPLETE!")
    print("=" * 70)
    
    print("\nKey Findings:")
    print(f"\n1. CLASSIFICATION (Non-linear separable data):")
    print(f"   - Best Architecture: {best_mlp[0]}")
    print(f"   - Accuracy: {best_mlp[1]['accuracy']:.4f}")
    print(f"   - Note: Deeper networks needed for complex decision boundaries")
    
    print(f"\n2. REGRESSION (Continuous output):")
    print(f"   - Best Architecture: {best_reg[0]}")
    print(f"   - MSE: {best_reg[1]['mse']:.4f}")
    print(f"   - Note: Output activation changed to linear for regression")
    
    print(f"\n3. LEARNING RATES:")
    best_lr = max(lr_results.items(), key=lambda x: x[1]['accuracy'])
    print(f"   - Best Learning Rate: {best_lr[0]}")
    print(f"   - Accuracy: {best_lr[1]['accuracy']:.4f}")
    print(f"   - Note: Learning rate significantly affects convergence speed")
    
    print("\nKey Takeaways:")
    print("  1. Always scale features before training neural networks")
    print("  2. Use appropriate activation functions:")
    print("     - ReLU for hidden layers (non-linearity)")
    print("     - Sigmoid for binary classification output")
    print("     - Linear for regression output")
    print("  3. Network depth matters for non-linear data")
    print("  4. Learning rate is crucial for convergence")
    print("  5. Monitor loss curves to detect:")
    print("     - Overfitting (loss increases on validation)")
    print("     - Underfitting (high loss throughout)")
    print("     - Learning rate too high (loss diverges)")
    print("  6. Use batch processing for efficient training")
    print("  7. Normalize/standardize inputs for better convergence")
    print("  8. Start with shallow networks, increase complexity if needed")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()