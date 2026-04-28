from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import hdbscan
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
import glob
import json
from collections import Counter
import pickle
from pathlib import Path
import matplotlib.patches as mpatches
import re


def normalize_features(image_features, text_features, text_weight=1.0):
    """Normalize features and combine with proper weighting"""
    # Normalize to unit length
    image_features = image_features / (np.linalg.norm(image_features, axis=1, keepdims=True) + 1e-8)
    text_features = text_features / (np.linalg.norm(text_features, axis=1, keepdims=True) + 1e-8)

    # Combine with weighting
    combined_features = np.concatenate([image_features, text_features * text_weight], axis=1)
    return combined_features


def apply_pca_with_scaling(features, n_components=100, scale=True):
    """Apply PCA with more components for better representation"""
    if scale:
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
    else:
        scaler = None

    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(features)

    print(f"PCA explained variance ratio (first 10): {pca.explained_variance_ratio_[:10]}")
    print(f"Total explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    return reduced_features, pca, scaler


def apply_kmeans(features, num_clusters=200):
    """K-Means with more clusters for detailed fault categorization"""
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto', max_iter=500)
    cluster_labels = kmeans.fit_predict(features)

    # Print cluster size statistics
    unique, counts = np.unique(cluster_labels, return_counts=True)
    print(f"K-Means cluster sizes: min={counts.min()}, max={counts.max()}, mean={counts.mean():.1f}")

    return cluster_labels


def apply_hdbscan_tuned(features, min_cluster_size=50, min_samples=10, cluster_selection_epsilon=0.1):
    """HDBSCAN with better parameters to find more meaningful clusters"""
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,  # Allow smaller density variations
        metric='euclidean',
        cluster_selection_method='eom',
        alpha=1.0,
        core_dist_n_jobs=-1  # Use all available cores
    )
    cluster_labels = clusterer.fit_predict(features)

    # Analyze results
    unique_labels = np.unique(cluster_labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = np.sum(cluster_labels == -1) if -1 in cluster_labels else 0

    print(f"HDBSCAN found {n_clusters} clusters with {n_noise} noise points ({n_noise / len(features) * 100:.1f}%)")
    if n_clusters > 0:
        valid_clusters = cluster_labels[cluster_labels >= 0]
        unique, counts = np.unique(valid_clusters, return_counts=True)
        print(f"Valid cluster sizes: min={counts.min()}, max={counts.max()}, mean={counts.mean():.1f}")

    return cluster_labels


def apply_hierarchical_clustering(features, n_clusters=200):
    """Hierarchical clustering with more clusters"""
    clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    cluster_labels = clusterer.fit_predict(features)

    # Print cluster statistics
    unique, counts = np.unique(cluster_labels, return_counts=True)
    print(f"Hierarchical cluster sizes: min={counts.min()}, max={counts.max()}, mean={counts.mean():.1f}")

    return cluster_labels


def apply_dbscan_tuned(features, eps='auto', min_samples=20):
    """DBSCAN with better parameter tuning"""
    # Auto-tune eps if needed
    if eps == 'auto':
        from sklearn.neighbors import NearestNeighbors
        print("Auto-tuning DBSCAN eps parameter...")
        neighbors = NearestNeighbors(n_neighbors=min_samples, n_jobs=-1)
        neighbors_fit = neighbors.fit(features)
        distances, _ = neighbors_fit.kneighbors(features)
        distances = np.sort(distances[:, -1])  # k-th nearest neighbor distances

        # Use elbow method to find optimal eps
        eps = np.percentile(distances, 85)  # Slightly more conservative than 90th percentile
        print(f"Auto-tuned eps: {eps:.4f}")

    clusterer = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    cluster_labels = clusterer.fit_predict(features)

    # Analyze results
    unique_labels = np.unique(cluster_labels)
    n_clusters = len(unique_labels) - (1 if -1 in cluster_labels else 0)
    n_noise = np.sum(cluster_labels == -1) if -1 in cluster_labels else 0

    print(f"DBSCAN found {n_clusters} clusters with {n_noise} noise points ({n_noise / len(features) * 100:.1f}%)")
    if n_clusters > 0:
        valid_clusters = cluster_labels[cluster_labels >= 0]
        unique, counts = np.unique(valid_clusters, return_counts=True)
        print(f"Valid cluster sizes: min={counts.min()}, max={counts.max()}, mean={counts.mean():.1f}")

    return cluster_labels


def create_multicolumn_legend(unique_labels, n_cols=3, max_labels_per_col=25):
    """Create a multi-column legend that fits better in the visualization"""
    # Handle noise label specially
    noise_label = None
    clean_labels = []
    for label in unique_labels:
        if label == -1:
            noise_label = label
        else:
            clean_labels.append(label)

    # Sort clean labels
    clean_labels = sorted(clean_labels)

    # Add noise label at the end if it exists
    if noise_label is not None:
        clean_labels.append(noise_label)

    # Truncate if too many labels
    if len(clean_labels) > n_cols * max_labels_per_col:
        clean_labels = clean_labels[:n_cols * max_labels_per_col]
        truncated = True
    else:
        truncated = False

    return clean_labels, truncated


def plot_tsne_visualization_improved(features, labels, output_path, fault_metadata=None,
                                     title_suffix="", perplexity=50, max_legend_items=75):
    """Improved t-SNE visualization with better legend handling"""

    # Adjust perplexity based on data size
    n_samples = len(features)
    perplexity = min(perplexity, max(5, n_samples // 4))

    print(f"Computing t-SNE with perplexity={perplexity}...")
    tsne = TSNE(n_components=2, perplexity=perplexity, metric="cosine",
                n_jobs=1, random_state=42, verbose=True, max_iter=1000)
    embedding = tsne.fit_transform(features)

    # Create figure with adjusted size
    plt.figure(figsize=(24, 16))

    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

    # Create legend labels with truncation if needed
    legend_labels, truncated = create_multicolumn_legend(unique_labels, max_labels_per_col=max_legend_items // 3)

    # Color mapping
    if -1 in unique_labels:
        # Special handling for noise points
        colors = []
        cmap = plt.get_cmap("tab20", min(20, n_clusters))  # Use tab20 for better distinction
        if n_clusters > 20:
            cmap = plt.get_cmap("turbo", n_clusters)

        color_map = {}
        color_idx = 0
        for label in unique_labels:
            if label == -1:
                color_map[label] = 'lightgrey'
            else:
                color_map[label] = cmap(color_idx)
                color_idx += 1

        colors = [color_map[label] for label in labels]
        scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=colors, alpha=0.6, s=3)

    else:
        # No noise points
        cmap = plt.get_cmap("tab20", min(20, len(unique_labels)))
        if len(unique_labels) > 20:
            cmap = plt.get_cmap("turbo", len(unique_labels))

        colors = [unique_labels.tolist().index(l) for l in labels]
        scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=colors, cmap=cmap, alpha=0.6, s=3)

    plt.title(f't-SNE Visualization of Multimodal Fault Clusters {title_suffix}', fontsize=20)
    plt.xlabel('t-SNE Dimension 1', fontsize=14)
    plt.ylabel('t-SNE Dimension 2', fontsize=14)

    # Create legend with limited items
    if len(legend_labels) <= max_legend_items:
        handles = []
        legend_names = []

        for label in legend_labels[:max_legend_items]:
            if label == -1:
                handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                          markerfacecolor='lightgrey', markersize=8))
                legend_names.append("Noise")
            else:
                if -1 in unique_labels:
                    color_idx = [l for l in unique_labels if l != -1].index(label)
                else:
                    color_idx = unique_labels.tolist().index(label)

                if n_clusters <= 20:
                    color = plt.get_cmap("tab20")(color_idx)
                else:
                    color = plt.get_cmap("turbo")(color_idx)

                handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                          markerfacecolor=color, markersize=8))
                legend_names.append(f"Cluster {label}")

        # Create multi-column legend
        n_cols = min(3, (len(handles) + 24) // 25)  # 25 items per column max
        legend = plt.legend(handles, legend_names, title="Fault Categories",
                            bbox_to_anchor=(1.02, 1), loc='upper left',
                            ncol=n_cols, fontsize=8, title_fontsize=10)
        legend.set_frame_on(True)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.8)

        if truncated:
            # Add note about truncation
            plt.text(1.02, 0.02, f"Showing {len(handles)} of {len(unique_labels)} clusters",
                     transform=plt.gca().transAxes, fontsize=8, style='italic')

    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()

    # Save with high DPI
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved visualization: {output_path}")


def extract_fault_metadata_from_paths(sample_folder_paths):
    """Extract fault metadata from the sample folder paths"""
    fault_metadata = []

    for i, folder_path in enumerate(sample_folder_paths):
        path_obj = Path(folder_path)

        # Extract fault UUID from folder path
        fault_id_match = re.search(r'([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})', str(path_obj))
        fault_id = fault_id_match.group(1) if fault_id_match else f"unknown_fault_{i}"

        # Try to extract unit_id and fault_category from the path structure
        # The path structure should be: .../unit_id/fault_category/fault_uuid/...
        path_parts = path_obj.parts

        unit_id = "unknown_unit"
        fault_category = "unknown_category"

        # Look for the fault UUID in the path and extract parent directories
        for j, part in enumerate(path_parts):
            if fault_id_match and fault_id in part:
                if j >= 2:  # We have at least 2 parent directories
                    fault_category = path_parts[j - 1]
                    unit_id = path_parts[j - 2]
                elif j >= 1:  # We have at least 1 parent directory
                    fault_category = path_parts[j - 1]
                break

        fault_metadata.append({
            'fault_index': i,
            'fault_id': fault_id,
            'unit_id': unit_id,
            'fault_category': fault_category,
            'sample_folder_path': folder_path
        })

    return fault_metadata


def save_clustering_results(labels_dict, fault_metadata, output_dir="/data1/detectron2/feature"):
    """Save detailed clustering results for fault-based analysis"""
    os.makedirs(output_dir, exist_ok=True)

    # Create DataFrame with all fault information
    results_data = []

    for meta in fault_metadata:
        row = meta.copy()

        # Add clustering results for this fault
        for method_name, labels in labels_dict.items():
            fault_idx = meta['fault_index']
            if fault_idx < len(labels):  # Safety check
                row[f'{method_name}_cluster'] = int(labels[fault_idx])

        results_data.append(row)

    df = pd.DataFrame(results_data)

    # Save comprehensive results
    df.to_csv(os.path.join(output_dir, 'fault_clustering_results_complete.csv'), index=False)

    # Save summary statistics for each method
    summary_stats = {}
    for method_name, labels in labels_dict.items():
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = np.sum(labels == -1) if -1 in labels else 0

        summary_stats[method_name] = {
            'n_clusters': int(n_clusters),
            'n_noise': int(n_noise),
            'noise_percentage': float(n_noise / len(labels) * 100),
            'cluster_sizes': [int(x) for x in np.bincount(labels[labels >= 0])] if n_clusters > 0 else []
        }

        # Analyze clustering quality by fault category
        method_df = df[df[f'{method_name}_cluster'] >= 0].copy() if -1 in labels else df.copy()
        if len(method_df) > 0:
            category_analysis = method_df.groupby('fault_category')[f'{method_name}_cluster'].agg([
                'nunique',  # Number of different clusters per category
                lambda x: x.mode().iloc[0] if len(x) > 0 else -1,  # Most common cluster
                'count'  # Number of faults in category
            ]).reset_index()
            category_analysis.columns = ['fault_category', 'clusters_per_category', 'dominant_cluster', 'fault_count']

            # Calculate purity - percentage of faults in category that go to dominant cluster
            dominant_cluster_counts = []
            for _, row in category_analysis.iterrows():
                category_df = method_df[method_df['fault_category'] == row['fault_category']]
                dominant_count = len(category_df[category_df[f'{method_name}_cluster'] == row['dominant_cluster']])
                purity = dominant_count / row['fault_count'] * 100
                dominant_cluster_counts.append(purity)

            category_analysis['category_purity_percent'] = dominant_cluster_counts

            # Save category analysis
            category_analysis.to_csv(
                os.path.join(output_dir, f'{method_name}_fault_category_analysis.csv'),
                index=False
            )

            # Analyze clustering quality by unit
            unit_analysis = method_df.groupby('unit_id')[f'{method_name}_cluster'].agg([
                'nunique',  # Number of different clusters per unit
                'count'  # Number of faults per unit
            ]).reset_index()
            unit_analysis.columns = ['unit_id', 'clusters_per_unit', 'fault_count']

            # Save unit analysis
            unit_analysis.to_csv(
                os.path.join(output_dir, f'{method_name}_unit_analysis.csv'),
                index=False
            )

    # Save summary statistics
    with open(os.path.join(output_dir, 'fault_clustering_summary.json'), 'w') as f:
        json.dump(summary_stats, f, indent=2)

    print(f"Saved detailed fault clustering results to {output_dir}")
    print(f"Total faults analyzed: {len(fault_metadata)}")
    print(f"Unique units: {len(set(meta['unit_id'] for meta in fault_metadata))}")
    print(f"Unique fault categories: {len(set(meta['fault_category'] for meta in fault_metadata))}")

    return df


def cluster_data_comprehensive(image_features, text_features, fault_metadata, num_clusters=200,
                               text_weight=1.0, pca_components=100):
    """Comprehensive clustering with improved parameters for fault-based analysis"""

    n_faults = len(fault_metadata)
    n_units = len(set(meta['unit_id'] for meta in fault_metadata))
    n_categories = len(set(meta['fault_category'] for meta in fault_metadata))

    print(f"Starting comprehensive fault clustering analysis...")
    print(f"Dataset: {n_faults} fault instances, {n_units} units, {n_categories} fault categories")

    # Step 1: Normalize and combine features
    combined_features = normalize_features(image_features, text_features, text_weight)
    print(f"Combined features shape: {combined_features.shape}")

    # Step 2: Apply PCA for dimensionality reduction
    reduced_features, pca, scaler = apply_pca_with_scaling(
        combined_features,
        n_components=pca_components,
        scale=True
    )

    # Save PCA components for later analysis
    np.save('/data1/detectron2/feature/fault_pca_reduced_features.npy', reduced_features)

    # Step 3: Apply different clustering methods with improved parameters
    clustering_results = {}

    print(f"\n{'=' * 20} K-Means Clustering {'=' * 20}")
    clustering_results['kmeans'] = apply_kmeans(reduced_features, num_clusters=num_clusters)

    print(f"\n{'=' * 20} HDBSCAN Clustering {'=' * 20}")
    # Adaptive parameters based on dataset size
    min_cluster_size = max(10, n_faults // 500)  # Smaller minimum for fault-level analysis
    clustering_results['hdbscan'] = apply_hdbscan_tuned(
        reduced_features,
        min_cluster_size=min_cluster_size,
        min_samples=max(3, min_cluster_size // 4),
        cluster_selection_epsilon=0.05
    )

    print(f"\n{'=' * 20} Hierarchical Clustering {'=' * 20}")
    clustering_results['hierarchical'] = apply_hierarchical_clustering(
        reduced_features,
        n_clusters=num_clusters
    )

    print(f"\n{'=' * 20} DBSCAN Clustering {'=' * 20}")
    clustering_results['dbscan'] = apply_dbscan_tuned(
        reduced_features,
        eps='auto',
        min_samples=max(5, n_faults // 2000)  # Adaptive min_samples for faults
    )

    return {
        'clustering_results': clustering_results,
        'features': reduced_features,
        'pca': pca,
        'scaler': scaler
    }


# Main execution
if __name__ == "__main__":
    print("Starting multimodal fault clustering analysis...")

    # Load features extracted by the multimodal feature extraction script
    print("Loading fault features...")
    feature_dir = "/data1/detectron2/feature"

    image_features = np.load(os.path.join(feature_dir, 'image_features.npy'))
    text_features = np.load(os.path.join(feature_dir, 'text_features.npy'))

    # Load sample folder paths and prompts
    with open(os.path.join(feature_dir, 'sample_folder_paths.json'), 'r') as f:
        sample_folder_paths = json.load(f)

    with open(os.path.join(feature_dir, 'prompts.json'), 'r') as f:
        prompts = json.load(f)

    print(f"Loaded features for {len(image_features)} fault instances")
    print(f"Image features shape: {image_features.shape}")
    print(f"Text features shape: {text_features.shape}")
    print(f"Sample folder paths: {len(sample_folder_paths)}")
    print(f"Prompts: {len(prompts)}")

    # Ensure all arrays have the same length
    min_len = min(len(image_features), len(text_features), len(sample_folder_paths), len(prompts))
    if min_len < max(len(image_features), len(text_features), len(sample_folder_paths), len(prompts)):
        print(f"Warning: Mismatched array lengths. Truncating to {min_len} samples.")
        image_features = image_features[:min_len]
        text_features = text_features[:min_len]
        sample_folder_paths = sample_folder_paths[:min_len]
        prompts = prompts[:min_len]

    # Extract fault metadata from sample folder paths
    print("Extracting fault metadata...")
    fault_metadata = extract_fault_metadata_from_paths(sample_folder_paths)

    # Display dataset statistics
    unique_units = set(meta['unit_id'] for meta in fault_metadata)
    unique_categories = set(meta['fault_category'] for meta in fault_metadata)

    print(f"\nDataset Statistics:")
    print(f"- Total fault instances: {len(fault_metadata)}")
    print(f"- Unique units: {len(unique_units)}")
    print(f"- Unique fault categories: {len(unique_categories)}")
    print(f"- Sample fault categories: {list(unique_categories)[:10]}...")

    # Run comprehensive clustering with parameters suitable for fault analysis
    results = cluster_data_comprehensive(
        image_features,
        text_features,
        fault_metadata,
        num_clusters=min(200, len(unique_categories) * 3),  # Adaptive cluster count
        text_weight=1.5,  # Higher text weight for semantic understanding
        pca_components=min(150, image_features.shape[1] + text_features.shape[1] - 50)  # Adaptive PCA
    )

    clustering_results = results['clustering_results']
    features = results['features']

    # Save detailed clustering results with fault metadata
    clustering_df = save_clustering_results(clustering_results, fault_metadata)

    # Create visualizations with improved legend handling
    print("\nGenerating visualizations...")
    methods = ['kmeans', 'hdbscan', 'hierarchical', 'dbscan']

    for method in methods:
        labels = clustering_results[method]
        n_clusters = len(np.unique(labels)) - (1 if -1 in labels else 0)
        n_noise = np.sum(labels == -1) if -1 in labels else 0

        print(f"\nProcessing {method.upper()}: {n_clusters} clusters, {n_noise} noise points")

        output_path = f'/data1/detectron2/feature/{method}_fault_clustering_improved.png'
        plot_tsne_visualization_improved(
            features,
            labels,
            output_path,
            fault_metadata=fault_metadata,
            title_suffix=f"({method.upper()}, {n_clusters} clusters)",
            perplexity=min(50, len(features) // 20),
            max_legend_items=60
        )

    print("\nFault clustering analysis complete!")
    print("\nOutput files:")
    print("- fault_clustering_results_complete.csv: Detailed results for all fault instances")
    print("- fault_clustering_summary.json: Summary statistics for each method")
    print("- *_fault_category_analysis.csv: How clusters align with true fault categories")
    print("- *_unit_analysis.csv: Analysis of clustering by unit")
    print("- *_fault_clustering_improved.png: Improved visualizations")
    print("\nNote: Each row in the results represents one fault instance (which may contain multiple images)")
    print("      that was processed through stitching, cropping, and feature extraction.")