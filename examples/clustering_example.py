"""
Complete Clustering Example

This example demonstrates how to:
1. Generate clustering data
2. Preprocess and scale features
3. Train multiple clustering models
4. Evaluate and compare clusters
5. Visualize results
6. Find optimal number of clusters
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
from sklearn.model_selection import train_test_split

# Import clustering algorithms
from algorithms.unsupervised.clustering import (
    KMeans,
    DBSCAN,
    HierarchicalClustering,
    GaussianMixtureModel
)

# Import utilities
from algorithms.utils.preprocessing import StandardScaler
from algorithms.utils.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score
)
from algorithms.utils.visualization import (
    plot_clusters,
    plot_silhouette_analysis,
    plot_dendrogram
)


def main():
    """Main example function."""
    
    print("=" * 70)
    print("CLUSTERING EXAMPLE: Comparing Multiple Models")
    print("=" * 70)
    
    # ============================================================================
    # 1. GENERATE CLUSTERING DATA
    # ============================================================================
    print("\n[1] Generating clustering data...")
    
    # Generate synthetic clustering data with clear clusters
    X, y_true = make_blobs(
        n_samples=500,
        centers=4,
        n_features=2,
        cluster_std=0.60,
        random_state=42
    )
    
    print(f"Data shape: X={X.shape}")
    print(f"Number of true clusters: {len(np.unique(y_true))}")
    print(f"Feature ranges: X1=[{X[:, 0].min():.2f}, {X[:, 0].max():.2f}]")
    print(f"                X2=[{X[:, 1].min():.2f}, {X[:, 1].max():.2f}]")
    
    # ============================================================================
    # 2. PREPROCESS: SCALE FEATURES
    # ============================================================================
    print("\n[2] Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"Original mean: {X.mean():.4f}, std: {X.std():.4f}")
    print(f"Scaled mean: {X_scaled.mean():.4f}, std: {X_scaled.std():.4f}")
    
    # ============================================================================
    # 3. FIND OPTIMAL NUMBER OF CLUSTERS (ELBOW METHOD)
    # ============================================================================
    print("\n[3] Finding optimal number of clusters (Elbow Method)...")
    
    inertias = []
    silhouette_scores = []
    k_range = range(2, 11)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        kmeans.fit(X_scaled)
        
        inertias.append(kmeans.inertia_)
        silhouette = silhouette_score(X_scaled, kmeans.labels_)
        silhouette_scores.append(silhouette)
        
        print(f"  k={k}: Inertia={kmeans.inertia_:.4f}, Silhouette={silhouette:.4f}")
    
    # Plot elbow curve
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Number of Clusters (k)')
    axes[0].set_ylabel('Inertia (Within-cluster sum of squares)')
    axes[0].set_title('Elbow Method for Optimal k')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(k_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Number of Clusters (k)')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].set_title('Silhouette Score vs Number of Clusters')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('elbow_silhouette_analysis.png', dpi=100, bbox_inches='tight')
    print("\n  ✓ Saved: elbow_silhouette_analysis.png")
    plt.close()
    
    # ============================================================================
    # 4. TRAIN MULTIPLE CLUSTERING MODELS
    # ============================================================================
    print("\n[4] Training clustering models...")
    
    optimal_k = k_range[np.argmax(silhouette_scores)]
    print(f"\n  Optimal k (by silhouette score): {optimal_k}")
    
    models = {
        'K-Means': KMeans(
            n_clusters=optimal_k,
            n_init=10,
            max_iter=300,
            random_state=42
        ),
        
        'Gaussian Mixture Model': GaussianMixtureModel(
            n_clusters=optimal_k,
            n_init=10,
            max_iter=300,
            random_state=42
        ),
        
        'Hierarchical Clustering': HierarchicalClustering(
            n_clusters=optimal_k,
            linkage='ward',
            metric='euclidean'
        ),
        
        'DBSCAN': DBSCAN(
            eps=0.5,
            min_samples=5,
            metric='euclidean'
        )
    }
    
    trained_models = {}
    
    for name, model in models.items():
        print(f"\n  Training {name}...")
        try:
            model.fit(X_scaled)
            trained_models[name] = model
            
            # Get cluster labels
            labels = model.labels_ if hasattr(model, 'labels_') else model.predict(X_scaled)
            n_clusters = len(np.unique(labels[labels != -1]))  # Exclude noise points (-1 in DBSCAN)
            
            print(f"  ✓ {name} trained successfully")
            print(f"    Number of clusters found: {n_clusters}")
            
        except Exception as e:
            print(f"  ✗ Error training {name}: {str(e)}")
    
    # ============================================================================
    # 5. EVALUATE CLUSTERING MODELS
    # ============================================================================
    print("\n[5] Evaluating clustering models...")
    print("\n" + "-" * 90)
    print(f"{'Model':<25} {'Silhouette':<15} {'Davies-Bouldin':<15} {'Calinski-H':<15}")
    print("-" * 90)
    
    results = {}
    
    for name, model in trained_models.items():
        try:
            # Get predictions
            if hasattr(model, 'labels_'):
                y_pred = model.labels_
            else:
                y_pred = model.predict(X_scaled)
            
            # Filter out noise points for evaluation
            valid_mask = y_pred != -1
            if np.sum(valid_mask) < len(y_pred):
                y_pred_eval = y_pred[valid_mask]
                X_eval = X_scaled[valid_mask]
            else:
                y_pred_eval = y_pred
                X_eval = X_scaled
            
            # Calculate metrics
            silhouette = silhouette_score(X_eval, y_pred_eval)
            davies_bouldin = davies_bouldin_score(X_eval, y_pred_eval)
            calinski = calinski_harabasz_score(X_eval, y_pred_eval)
            
            results[name] = {
                'silhouette': silhouette,
                'davies_bouldin': davies_bouldin,
                'calinski': calinski,
                'labels': y_pred,
                'model': model
            }
            
            print(f"{name:<25} {silhouette:<15.4f} {davies_bouldin:<15.4f} {calinski:<15.4f}")
            
        except Exception as e:
            print(f"{name:<25} Error: {str(e)}")
    
    print("-" * 90)
    print("Note: Higher Silhouette & Calinski-Harabasz, Lower Davies-Bouldin is better")
    
    # ============================================================================
    # 6. COMPARE MODELS
    # ============================================================================
    print("\n[6] Model Comparison:")
    
    if results:
        # Find best model by silhouette score
        best_model = max(results.items(), key=lambda x: x[1]['silhouette'])
        
        print(f"\n  Best Model (by Silhouette Score): {best_model[0]}")
        print(f"    Silhouette Score: {best_model[1]['silhouette']:.4f}")
        print(f"    Davies-Bouldin Index: {best_model[1]['davies_bouldin']:.4f}")
        print(f"    Calinski-Harabasz Index: {best_model[1]['calinski']:.4f}")
        
        # ====================================================================
        # 7. DETAILED ANALYSIS OF BEST MODEL
        # ====================================================================
        print("\n[7] Detailed Analysis of Best Model...")
        
        best_name = best_model[0]
        y_pred_best = best_model[1]['labels']
        best_model_obj = best_model[1]['model']
        
        print(f"\n  Model: {best_name}")
        print(f"  Number of clusters: {len(np.unique(y_pred_best[y_pred_best != -1]))}")
        
        # Cluster sizes
        unique, counts = np.unique(y_pred_best, return_counts=True)
        print(f"\n  Cluster distribution:")
        for cluster_id, count in zip(unique, counts):
            if cluster_id != -1:  # Skip noise points
                percentage = (count / len(y_pred_best)) * 100
                print(f"    Cluster {cluster_id}: {count} points ({percentage:.1f}%)")
            else:
                print(f"    Noise points: {count}")
        
        # ====================================================================
        # 8. VISUALIZATIONS
        # ====================================================================
        print("\n[8] Creating visualizations...")
        
        # Plot all clustering results (2D visualization)
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.ravel()
        
        for idx, (name, result) in enumerate(results.items()):
            if idx < 4:
                ax = axes[idx]
                labels = result['labels']
                
                # Plot points colored by cluster
                scatter = ax.scatter(
                    X_scaled[:, 0], X_scaled[:, 1],
                    c=labels, cmap='viridis', s=50, alpha=0.6,
                    edgecolors='k', linewidth=0.5
                )
                
                # Plot cluster centers if available
                if hasattr(result['model'], 'cluster_centers_'):
                    centers = result['model'].cluster_centers_
                    ax.scatter(centers[:, 0], centers[:, 1], c='red', s=300,
                             alpha=0.8, marker='*', edgecolors='black', linewidth=2,
                             label='Centroids')
                
                ax.set_xlabel('Feature 1')
                ax.set_ylabel('Feature 2')
                ax.set_title(f"{name}\n(Silhouette: {result['silhouette']:.3f})")
                ax.grid(True, alpha=0.3)
                if hasattr(result['model'], 'cluster_centers_'):
                    ax.legend()
                
                plt.colorbar(scatter, ax=ax, label='Cluster')
        
        plt.tight_layout()
        plt.savefig('clustering_comparison.png', dpi=100, bbox_inches='tight')
        print("  ✓ Saved: clustering_comparison.png")
        plt.close()
        
        # Silhouette analysis for best model
        fig, ax = plt.subplots(figsize=(10, 6))
        
        plot_silhouette_analysis(X_scaled, y_pred_best, ax=ax,
                                title=f"Silhouette Analysis: {best_name}")
        
        plt.tight_layout()
        plt.savefig('silhouette_analysis_best_model.png', dpi=100, bbox_inches='tight')
        print("  ✓ Saved: silhouette_analysis_best_model.png")
        plt.close()
        
        # Model comparison bar chart
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        model_names = list(results.keys())
        silhouettes = [results[m]['silhouette'] for m in model_names]
        davies_bouldin = [results[m]['davies_bouldin'] for m in model_names]
        calinski = [results[m]['calinski'] for m in model_names]
        
        # Silhouette comparison
        axes[0].barh(model_names, silhouettes, color='steelblue')
        axes[0].set_xlabel('Silhouette Score')
        axes[0].set_title('Silhouette Score Comparison\n(Higher is Better)')
        for i, v in enumerate(silhouettes):
            axes[0].text(v + 0.01, i, f'{v:.3f}', va='center')
        
        # Davies-Bouldin comparison
        axes[1].barh(model_names, davies_bouldin, color='coral')
        axes[1].set_xlabel('Davies-Bouldin Index')
        axes[1].set_title('Davies-Bouldin Index Comparison\n(Lower is Better)')
        for i, v in enumerate(davies_bouldin):
            axes[1].text(v + 0.01, i, f'{v:.3f}', va='center')
        
        # Calinski-Harabasz comparison
        axes[2].barh(model_names, calinski, color='lightgreen')
        axes[2].set_xlabel('Calinski-Harabasz Index')
        axes[2].set_title('Calinski-Harabasz Index Comparison\n(Higher is Better)')
        for i, v in enumerate(calinski):
            axes[2].text(v + 0.01, i, f'{v:.1f}', va='center')
        
        plt.tight_layout()
        plt.savefig('clustering_metrics_comparison.png', dpi=100, bbox_inches='tight')
        print("  ✓ Saved: clustering_metrics_comparison.png")
        plt.close()
        
        # ====================================================================
        # 9. PREDICT ON NEW DATA
        # ====================================================================
        print("\n[9] Predicting cluster assignments for new data...")
        
        # Generate new data
        X_new = np.random.randn(5, 2) * X_scaled.std() + X_scaled.mean()
        
        try:
            if hasattr(best_model_obj, 'predict'):
                predictions_new = best_model_obj.predict(X_new)
                print(f"\n  New samples cluster assignments ({best_name}):")
                for i, cluster in enumerate(predictions_new):
                    print(f"    Sample {i+1}: Cluster {cluster}")
        except Exception as e:
            print(f"  Cannot predict for new data: {str(e)}")
        
        # ====================================================================
        # 10. TEMPORAL CLUSTERING ANALYSIS
        # ====================================================================
        print("\n[10] Within-cluster analysis for best model...")
        
        # Calculate within-cluster distances
        valid_mask = y_pred_best != -1
        y_pred_valid = y_pred_best[valid_mask]
        X_valid = X_scaled[valid_mask]
        
        intra_cluster_distances = []
        for cluster_id in np.unique(y_pred_valid):
            cluster_mask = y_pred_valid == cluster_id
            cluster_points = X_valid[cluster_mask]
            
            if len(cluster_points) > 1:
                # Calculate mean distance from cluster center
                cluster_center = cluster_points.mean(axis=0)
                distances = np.linalg.norm(cluster_points - cluster_center, axis=1)
                intra_cluster_distances.append(distances.mean())
        
        print(f"\n  Average intra-cluster distances:")
        for i, dist in enumerate(intra_cluster_distances):
            print(f"    Cluster {i}: {dist:.4f}")
    
    # ============================================================================
    # SUMMARY
    # ============================================================================
    print("\n" + "=" * 70)
    print("EXAMPLE COMPLETE!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  1. Always scale features before clustering")
    print("  2. Use elbow method and silhouette score to find optimal k")
    print("  3. Compare multiple clustering algorithms")
    print("  4. Use appropriate evaluation metrics:")
    print("     - Silhouette Score (higher is better)")
    print("     - Davies-Bouldin Index (lower is better)")
    print("     - Calinski-Harabasz Index (higher is better)")
    print("  5. Visualize clusters in 2D/3D when possible")
    print("  6. Consider domain knowledge when choosing algorithm")
    print("  7. DBSCAN is good for irregular shapes and noise detection")
    print("  8. K-Means is fast and works well for spherical clusters")
    print("  9. Hierarchical clustering provides dendrogram for inspection")
    print(" 10. GMM provides probabilistic cluster assignments")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()