# filename: multimodal_clustering.py
import argparse
import json
import re
import shutil
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from openTSNE import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE as SklearnTSNE
import hdbscan

# Fallback options if UMAP doesn't work
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("UMAP not available, will use PCA for dimensionality reduction")


def analyze_cluster_names(prompts, labels, num_clusters, outlier_label=-1):
    cluster_names = {}
    print("Analyzing cluster contents to generate names...")
    unique_labels = sorted(list(set(labels)))

    for i in unique_labels:
        if i == outlier_label:
            cluster_names[i] = "Cluster_outliers_noise"
            continue

        cluster_prompts = [prompts[j] for j, label in enumerate(labels) if label == i]
        if not cluster_prompts:
            cluster_names[i] = f"Cluster_{i}_empty"
            continue

        word_counts = Counter()
        stopwords = {'a', 'of', 'on', 'the', 'photo', 'detailed', 'in', 'or', 'and', 'showing', 'at'}
        for p in cluster_prompts:
            words = [word for word in re.split(r'\W+', p.lower()) if
                     word and word not in stopwords and not word.isdigit()]
            word_counts.update(words)

        top_words = [word for word, count in word_counts.most_common(3)]
        name = "_".join(top_words)
        cluster_names[i] = f"Cluster_{i}_{name}" if name else f"Cluster_{i}"
    return cluster_names


def plot_tsne_visualization(features, labels, output_path, label_map, perplexity=50, outlier_label=-1):
    print("Generating t-SNE visualization...")
    tsne = TSNE(n_components=2, perplexity=perplexity, metric="cosine", n_jobs=-1, random_state=42, verbose=True)
    embedding = tsne.fit(features)

    print("Plotting results...")
    plt.figure(figsize=(22, 16))
    unique_labels = sorted(list(set(labels)))

    # Handle outliers: give them a distinct color (e.g., gray) and place them first in the legend
    has_outliers = outlier_label in unique_labels
    if has_outliers:
        unique_labels.remove(outlier_label)
        unique_labels.insert(0, outlier_label)

    num_unique_labels = len(unique_labels)
    cmap = plt.get_cmap("turbo", num_unique_labels)

    colors = [unique_labels.index(l) for l in labels]
    scatter_colors = [cmap(i) for i in colors]

    # Set outlier color to gray
    if has_outliers:
        outlier_idx_in_list = unique_labels.index(outlier_label)
        scatter_colors = [(0.5, 0.5, 0.5, 0.3) if l == outlier_label else cmap(unique_labels.index(l)) for l in labels]

    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=scatter_colors, alpha=0.6, s=8)

    plt.title('t-SNE Visualization of Multimodal Fault Clusters', fontsize=24)
    plt.xlabel('t-SNE Dimension 1', fontsize=16)
    plt.ylabel('t-SNE Dimension 2', fontsize=16)

    handles = [plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=cmap(i) if unique_labels[i] != outlier_label else (0.5, 0.5, 0.5),
                          markersize=10) for i in range(num_unique_labels)]
    legend_labels = [label_map.get(l, f"Cluster {l}") for l in unique_labels]

    plt.legend(handles, legend_labels, title="Fault Categories", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(output_path, dpi=150)
    print(f"t-SNE plot saved to {output_path}")
    plt.close()


def reduce_dimensions(features, method='pca', n_components=10, random_state=42):
    """
    Reduce dimensionality of features using various methods
    """
    print(f"Reducing dimensions using {method}...")
    
    if method == 'umap' and UMAP_AVAILABLE:
        try:
            reducer = umap.UMAP(n_components=n_components, random_state=random_state, 
                              n_jobs=1, low_memory=True)  # Use single thread and low memory
            embedding = reducer.fit_transform(features)
            return embedding
        except Exception as e:
            print(f"UMAP failed with error: {e}")
            print("Falling back to PCA...")
            method = 'pca'
    
    if method == 'pca':
        reducer = PCA(n_components=n_components, random_state=random_state)
        embedding = reducer.fit_transform(features)
        return embedding
    
    elif method == 'tsne':
        # Use sklearn's TSNE for dimensionality reduction (different from visualization)
        reducer = SklearnTSNE(n_components=n_components, random_state=random_state, 
                             metric='cosine', n_jobs=1)
        embedding = reducer.fit_transform(features)
        return embedding
    
    else:
        raise ValueError(f"Unknown dimensionality reduction method: {method}")


def main():
    parser = argparse.ArgumentParser('Multimodal Clustering and Reclassification')
    parser.add_argument('--feature_dir', default='/data1/detectron2/feature',
                        help='Directory with extracted .npy and .json files.')
    parser.add_argument('--output_dir', default='/data1/detectron2/feature/multimodal_hdb/reclassified',
                        help='Directory to save reclassified folders.')
    parser.add_argument('--algorithm', default='hdbscan', choices=['kmeans', 'dbscan', 'hdbscan'],
                        help='Clustering algorithm to use.')
    parser.add_argument('--text_weight', default=1.0, type=float,
                        help='Weight to apply to text features before clustering.')
    parser.add_argument('--dim_reduction', default='pca', choices=['umap', 'pca', 'tsne'],
                        help='Dimensionality reduction method to use before clustering.')
    parser.add_argument('--n_components', default=10, type=int,
                        help='Number of components for dimensionality reduction.')
    # K-Means specific
    parser.add_argument('--num_clusters', default=100, type=int, help='Number of clusters for K-Means.')
    # DBSCAN/HDBSCAN specific
    parser.add_argument('--eps', default=0.5, type=float, help='Epsilon for DBSCAN.')
    parser.add_argument('--min_samples', default=2, type=int, help='min_samples for DBSCAN/HDBSCAN.')
    args = parser.parse_args()

    feature_path = Path(args.feature_dir)
    image_features = np.load(feature_path / 'image_features.npy')
    text_features = np.load(feature_path / 'text_features.npy')
    with open(feature_path / 'sample_folder_paths.json', 'r') as f:
        paths = json.load(f)
    with open(feature_path / 'prompts.json', 'r') as f:
        prompts = json.load(f)

    print(f"Loaded {len(paths)} samples.")
    print(f"Combining features with text_weight={args.text_weight}...")
    # L2 normalize the combined features for better distance calculations
    combined_features = np.concatenate([image_features, text_features * args.text_weight], axis=1)
    norm = np.linalg.norm(combined_features, axis=1, keepdims=True)
    combined_features = combined_features / norm

    # Reduce dimensionality before clustering (especially important for HDBSCAN)
    embedding = reduce_dimensions(combined_features, method=args.dim_reduction, 
                                n_components=args.n_components)

    print(f"Performing clustering with algorithm: {args.algorithm}...")
    if args.algorithm == 'kmeans':
        clusterer = KMeans(n_clusters=args.num_clusters, random_state=42, n_init='auto')
        cluster_labels = clusterer.fit_predict(embedding)
        outlier_label = -1  # Not applicable for KMeans
    elif args.algorithm == 'dbscan':
        # For reduced dimensions, use euclidean distance
        clusterer = DBSCAN(eps=args.eps, min_samples=args.min_samples, metric='euclidean', n_jobs=-1)
        cluster_labels = clusterer.fit_predict(embedding)
        outlier_label = -1
    elif args.algorithm == 'hdbscan':
        print("Performing clustering with HDBSCAN on reduced embedding...")
        clusterer = hdbscan.HDBSCAN(min_cluster_size=args.min_samples, 
                                    metric='euclidean',
                                    core_dist_n_jobs=-1,
                                    cluster_selection_epsilon=0.01)
        cluster_labels = clusterer.fit_predict(embedding)
        outlier_label = -1

    num_found_clusters = len(set(cluster_labels))
    print(f"Found {num_found_clusters} clusters (including outliers).")

    cluster_names = analyze_cluster_names(prompts, cluster_labels, num_found_clusters, outlier_label)

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Use the original combined features for visualization (better representation)
    plot_tsne_visualization(combined_features, cluster_labels, 
                           output_path / f'{args.algorithm}_{args.dim_reduction}_clustering.png',
                           cluster_names)

    print("Organizing sample folders into new, named cluster directories...")
    for i, folder_path_str in enumerate(tqdm(paths, desc="Copying folders")):
        label_id = cluster_labels[i]
        category_name = cluster_names.get(label_id, f"Cluster_{label_id}")
        category_dir = output_path / category_name
        category_dir.mkdir(exist_ok=True)
        source_folder = Path(folder_path_str)
        if source_folder.exists():
            grandparent = source_folder.parent.parent.name
            parent = source_folder.parent.name
            new_name = f'{grandparent}_{parent}_{source_folder.name}'
            try:
                shutil.copytree(source_folder, category_dir / new_name, dirs_exist_ok=True)
            except Exception as e:
                print(f"Could not copy {source_folder}, error: {e}")

    print(f"\nReclassification complete! Folders organized in '{output_path}'.")


if __name__ == "__main__":
    main()