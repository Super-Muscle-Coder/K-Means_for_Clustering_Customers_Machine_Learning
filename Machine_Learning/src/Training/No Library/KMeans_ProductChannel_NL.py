"""
================================================================================
K-MEANS CLUSTERING - PRODUCT CHANNEL SEGMENTATION (NO LIBRARY IMPLEMENTATION)
================================================================================

Module: Simplified K-Means for Product Channel Customer Segmentation
Strategy: Product_HHI, Store_Preference, Web_Engagement, PC1_Total_TotalPurchases
Post-hoc: Wine_Preference, Meat_Preference, Fish_Preference, Fruit_Preference,
          Sweet_Preference, Gold_Preference, Dominant_Product, Top_Product_Share

Features:
- Custom K-Means implementation (no sklearn.cluster.KMeans)
- Random initialization (default) + K-Means++ (optional toggle)
- Optimal K selection via Elbow + Silhouette (2-method voting)
- Flexible K adjustment (default K=2, adjustable based on voting)
- PCA visualization (2D + 3D)
- Line chart visualization for cluster characteristics
- Minimal console output, focus on visualizations

Input:  Customer_Behavior_ProductChannel_Robust_scaled.csv
Output: Optimized clustering + clean visualizations
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")


# ================================================================================
# CUSTOM K-MEANS IMPLEMENTATION
# ================================================================================

class CustomKMeans:
    """Self-implemented K-Means clustering algorithm."""
    def __init__(self, n_clusters=2, init_method='random', max_iter=300,
                 n_init=10, tol=1e-4, random_state=42):
        self.n_clusters = n_clusters
        self.init_method = init_method
        self.max_iter = max_iter
        self.n_init = n_init
        self.tol = tol
        self.random_state = random_state

        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0

    def _init_centroids_random(self, X):
        np.random.seed(self.random_state)
        random_indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        return X[random_indices].copy()

    def _init_centroids_kmeans_plus_plus(self, X):
        np.random.seed(self.random_state)
        n_samples = X.shape[0]
        centroids = [X[np.random.randint(n_samples)]]
        for _ in range(1, self.n_clusters):
            distances = np.array([
                min([np.linalg.norm(x - c)**2 for c in centroids])
                for x in X
            ])
            probabilities = distances / distances.sum()
            cumulative_probs = probabilities.cumsum()
            r = np.random.rand()
            for idx, cum_prob in enumerate(cumulative_probs):
                if r < cum_prob:
                    centroids.append(X[idx])
                    break
        return np.array(centroids)

    def _assign_clusters(self, X, centroids):
        distances = np.zeros((X.shape[0], self.n_clusters))
        for k in range(self.n_clusters):
            distances[:, k] = np.linalg.norm(X - centroids[k], axis=1)
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X, labels):
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                centroids[k] = cluster_points.mean(axis=0)
            else:
                centroids[k] = X[np.random.randint(X.shape[0])]
        return centroids

    def _calculate_inertia(self, X, labels, centroids):
        inertia = 0
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - centroids[k])**2)
        return inertia

    def fit(self, X):
        X = np.array(X)
        best_inertia = np.inf
        best_centroids = None
        best_labels = None
        best_n_iter = 0
        for init_run in range(self.n_init):
            if self.init_method == 'kmeans++':
                centroids = self._init_centroids_kmeans_plus_plus(X)
            else:
                centroids = self._init_centroids_random(X)
            for iteration in range(self.max_iter):
                labels = self._assign_clusters(X, centroids)
                new_centroids = self._update_centroids(X, labels)
                centroid_shift = np.linalg.norm(new_centroids - centroids)
                centroids = new_centroids
                if centroid_shift < self.tol:
                    break
            inertia = self._calculate_inertia(X, labels, centroids)
            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = centroids
                best_labels = labels
                best_n_iter = iteration + 1
        self.cluster_centers_ = best_centroids
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter
        return self

    def predict(self, X):
        X = np.array(X)
        return self._assign_clusters(X, self.cluster_centers_)


# ================================================================================
# OPTIMAL K SELECTOR (ELBOW + SILHOUETTE)
# ================================================================================

class OptimalKSelector:
    """Select optimal K using Elbow + Silhouette voting."""
    def __init__(self, X, k_range=(2, 11), init_method='random', random_state=42):
        self.X = np.array(X)
        self.k_range = k_range
        self.init_method = init_method
        self.random_state = random_state
        self.wcss_scores = {}
        self.silhouette_scores = {}
        self.votes = {}

    def evaluate_all_k(self):
        k_values = range(self.k_range[0], self.k_range[1])
        for k in k_values:
            kmeans = CustomKMeans(
                n_clusters=k,
                init_method=self.init_method,
                random_state=self.random_state
            )
            kmeans.fit(self.X)
            labels = kmeans.labels_
            self.wcss_scores[k] = kmeans.inertia_
            self.silhouette_scores[k] = silhouette_score(self.X, labels)

    def vote_optimal_k(self):
        elbow_k = self._find_elbow_point()
        silhouette_k = max(self.silhouette_scores, key=self.silhouette_scores.get)
        votes = [elbow_k, silhouette_k]
        for k in votes:
            self.votes[k] = self.votes.get(k, 0) + 1
        optimal_k = max(self.votes, key=self.votes.get)
        return optimal_k, elbow_k, silhouette_k

    def _find_elbow_point(self):
        k_values = sorted(self.wcss_scores.keys())
        wcss_values = [self.wcss_scores[k] for k in k_values]
        wcss_norm = (wcss_values - np.min(wcss_values)) / (np.max(wcss_values) - np.min(wcss_values))
        k_norm = (np.array(k_values) - np.min(k_values)) / (np.max(k_values) - np.min(k_values))
        line_vec = np.array([k_norm[-1] - k_norm[0], wcss_norm[-1] - wcss_norm[0]])
        line_vec_norm = line_vec / np.linalg.norm(line_vec)
        distances = []
        for i in range(len(k_values)):
            point_vec = np.array([k_norm[i] - k_norm[0], wcss_norm[i] - wcss_norm[0]])
            distance = np.abs(np.cross(line_vec_norm, point_vec))
            distances.append(distance)
        elbow_idx = np.argmax(distances)
        return k_values[elbow_idx]


# ================================================================================
# PRODUCT CHANNEL K-MEANS ANALYZER
# ================================================================================

class ProductChannelKMeansAnalyzer:
    """Complete K-Means analysis pipeline for Product Channel segmentation."""
    def __init__(self, input_csv, output_graph_dir, k=2, init_kmeans_plus_plus=False,
                 use_voting=True, random_state=42):
        self.input_csv = input_csv
        self.output_graph_dir = output_graph_dir
        self.k = k
        self.init_method = 'kmeans++' if init_kmeans_plus_plus else 'random'
        self.use_voting = use_voting
        self.random_state = random_state

        os.makedirs(output_graph_dir, exist_ok=True)

        self.df = None
        self.X_clustering = None
        self.kmeans = None
        self.optimal_k_selector = None
        self.optimal_k = None
        self.elbow_k = None
        self.silhouette_k = None
        self.pca_2d = None
        self.pca_3d = None
        self.X_pca_2d = None
        self.X_pca_3d = None

        # Clustering features
        self.CLUSTERING_FEATURES = [
            'Product_HHI', 'Store_Preference', 'Web_Engagement', 'PC1_Total_TotalPurchases'
        ]
        # Post-hoc (reference) features for behavioral analysis
        self.POSTHOC_FEATURES = [
            'Wine_Preference', 'Meat_Preference', 'Fish_Preference', 'Fruit_Preference',
            'Sweet_Preference', 'Gold_Preference', 'Dominant_Product', 'Top_Product_Share'
        ]

    def load_data(self):
        """Load and validate dataset."""
        self.df = pd.read_csv(self.input_csv)
        missing = [f for f in self.CLUSTERING_FEATURES + self.POSTHOC_FEATURES if f not in self.df.columns]
        if len(missing) > 0:
            raise ValueError(f"Missing required columns: {missing}")
        self.X_clustering = self.df[self.CLUSTERING_FEATURES].values

    def find_optimal_k(self):
        """Find optimal K using Elbow + Silhouette voting."""
        if not self.use_voting:
            return
        self.optimal_k_selector = OptimalKSelector(
            self.X_clustering,
            k_range=(2, 11),
            init_method=self.init_method,
            random_state=self.random_state
        )
        self.optimal_k_selector.evaluate_all_k()
        self.optimal_k, self.elbow_k, self.silhouette_k = self.optimal_k_selector.vote_optimal_k()

    def fit_kmeans(self):
        """Fit K-Means with selected K."""
        self.kmeans = CustomKMeans(
            n_clusters=self.k,
            init_method=self.init_method,
            max_iter=300,
            n_init=10,
            random_state=self.random_state
        )
        self.kmeans.fit(self.X_clustering)
        self.df['Cluster'] = self.kmeans.labels_

    def compute_pca(self):
        """Compute PCA for 2D and 3D visualization."""
        self.pca_2d = PCA(n_components=2, random_state=self.random_state)
        self.X_pca_2d = self.pca_2d.fit_transform(self.X_clustering)
        self.pca_3d = PCA(n_components=3, random_state=self.random_state)
        self.X_pca_3d = self.pca_3d.fit_transform(self.X_clustering)

    # --------------------------------------------------------------------------
    # Visualizations
    # --------------------------------------------------------------------------

    def visualize_optimal_k(self):
        """Visualize Elbow + Silhouette with voting suggestions."""
        if self.optimal_k_selector is None:
            return

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Optimal K Selection - Elbow & Silhouette Methods', fontsize=16, fontweight='bold')

        k_values = sorted(self.optimal_k_selector.wcss_scores.keys())

        # Elbow
        ax = axes[0]
        wcss_vals = [self.optimal_k_selector.wcss_scores[k] for k in k_values]
        ax.plot(k_values, wcss_vals, 'bo-', linewidth=2.5, markersize=8, label='WCSS')
        ax.axvline(self.elbow_k, color='red', linestyle='--', linewidth=2.5, label=f'Elbow K={self.elbow_k}')
        ax.set_xlabel('Number of Clusters (K)', fontsize=12, fontweight='bold')
        ax.set_ylabel('WCSS (Within-Cluster Sum of Squares)', fontsize=12, fontweight='bold')
        ax.set_title('Elbow Method', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)

        # Silhouette
        ax = axes[1]
        sil_vals = [self.optimal_k_selector.silhouette_scores[k] for k in k_values]
        ax.plot(k_values, sil_vals, 'go-', linewidth=2.5, markersize=8, label='Silhouette')
        ax.axvline(self.silhouette_k, color='red', linestyle='--', linewidth=2.5, label=f'Silhouette K={self.silhouette_k}')
        ax.set_xlabel('Number of Clusters (K)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')
        ax.set_title('Silhouette Score (Higher is Better)', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)

        plt.tight_layout()
        filepath = os.path.join(self.output_graph_dir, "01_Optimal_K_Evaluation.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()

    def visualize_pca_2d(self):
        """Visualize clusters using 2D PCA."""
        fig, ax = plt.subplots(figsize=(13, 8))
        colors = plt.cm.tab10(np.linspace(0, 1, self.k))

        for cluster_id in range(self.k):
            cluster_mask = self.df['Cluster'] == cluster_id
            ax.scatter(
                self.X_pca_2d[cluster_mask, 0],
                self.X_pca_2d[cluster_mask, 1],
                label=f'Cluster {cluster_id} (n={(cluster_mask).sum():,})',
                alpha=0.65,
                s=60,
                color=colors[cluster_id],
                edgecolors='black',
                linewidth=0.5
            )

        # PCA centroids
        pca_centroids_2d = self.pca_2d.transform(self.kmeans.cluster_centers_)
        ax.scatter(
            pca_centroids_2d[:, 0],
            pca_centroids_2d[:, 1],
            marker='X',
            s=400,
            c='red',
            edgecolors='black',
            linewidth=2.5,
            label='Centroids',
            zorder=10
        )

        var_1 = self.pca_2d.explained_variance_ratio_[0] * 100
        var_2 = self.pca_2d.explained_variance_ratio_[1] * 100
        ax.set_xlabel(f'PC1 ({var_1:.1f}% variance)', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'PC2 ({var_2:.1f}% variance)', fontsize=12, fontweight='bold')
        ax.set_title(f'K-Means Clustering (K={self.k}) - PCA 2D Projection', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        filepath = os.path.join(self.output_graph_dir, "02_PCA_2D.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()

    def visualize_pca_3d(self):
        """Visualize clusters using 3D PCA."""
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        colors = plt.cm.tab10(np.linspace(0, 1, self.k))

        for cluster_id in range(self.k):
            cluster_mask = self.df['Cluster'] == cluster_id
            ax.scatter(
                self.X_pca_3d[cluster_mask, 0],
                self.X_pca_3d[cluster_mask, 1],
                self.X_pca_3d[cluster_mask, 2],
                label=f'Cluster {cluster_id} (n={(cluster_mask).sum():,})',
                c=[colors[cluster_id]],
                alpha=0.65,
                s=40,
                edgecolors='black',
                linewidth=0.3
            )

        pca_centroids_3d = self.pca_3d.transform(self.kmeans.cluster_centers_)
        ax.scatter(
            pca_centroids_3d[:, 0],
            pca_centroids_3d[:, 1],
            pca_centroids_3d[:, 2],
            marker='X',
            s=400,
            c='red',
            edgecolors='black',
            linewidth=2.5,
            label='Centroids',
            zorder=10
        )

        var_1 = self.pca_3d.explained_variance_ratio_[0] * 100
        var_2 = self.pca_3d.explained_variance_ratio_[1] * 100
        var_3 = self.pca_3d.explained_variance_ratio_[2] * 100
        ax.set_xlabel(f'PC1 ({var_1:.1f}%)', fontsize=11, fontweight='bold')
        ax.set_ylabel(f'PC2 ({var_2:.1f}%)', fontsize=11, fontweight='bold')
        ax.set_zlabel(f'PC3 ({var_3:.1f}%)', fontsize=11, fontweight='bold')
        ax.set_title(f'K-Means Clustering (K={self.k}) - PCA 3D Projection', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)

        plt.tight_layout()
        filepath = os.path.join(self.output_graph_dir, "03_PCA_3D.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()

    def visualize_clusters_2d(self):
        """Visualize clusters in 2D using Store_Preference vs Web_Engagement."""
        fig, ax = plt.subplots(figsize=(13, 8))
        colors = plt.cm.tab10(np.linspace(0, 1, self.k))

        for cluster_id in range(self.k):
            cluster_data = self.df[self.df['Cluster'] == cluster_id]
            ax.scatter(
                cluster_data['Store_Preference'],
                cluster_data['Web_Engagement'],
                label=f'Cluster {cluster_id} (n={len(cluster_data):,})',
                alpha=0.65,
                s=60,
                color=colors[cluster_id],
                edgecolors='black',
                linewidth=0.5
            )

        centroids = self.kmeans.cluster_centers_
        # Indices: Store_Preference (1), Web_Engagement (2)
        ax.scatter(
            centroids[:, 1],
            centroids[:, 2],
            marker='X',
            s=400,
            c='red',
            edgecolors='black',
            linewidth=2.5,
            label='Centroids',
            zorder=10
        )

        ax.set_xlabel('Store_Preference (Scaled)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Web_Engagement (Scaled)', fontsize=12, fontweight='bold')
        ax.set_title(f'K-Means Clustering (K={self.k}, init={self.init_method}) - Store vs Web', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        filepath = os.path.join(self.output_graph_dir, "04_Clusters_2D_Store_Web.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()

    def visualize_clusters_3d(self):
        """Visualize clusters in 3D: Product_HHI, Store_Preference, Web_Engagement."""
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        colors = plt.cm.tab10(np.linspace(0, 1, self.k))

        for cluster_id in range(self.k):
            cluster_data = self.df[self.df['Cluster'] == cluster_id]
            ax.scatter(
                cluster_data['Product_HHI'],
                cluster_data['Store_Preference'],
                cluster_data['Web_Engagement'],
                label=f'Cluster {cluster_id} (n={len(cluster_data):,})',
                c=[colors[cluster_id]],
                alpha=0.65,
                s=40,
                edgecolors='black',
                linewidth=0.3
            )

        centroids = self.kmeans.cluster_centers_
        ax.scatter(
            centroids[:, 0],
            centroids[:, 1],
            centroids[:, 2],
            marker='X',
            s=400,
            c='red',
            edgecolors='black',
            linewidth=2.5,
            label='Centroids',
            zorder=10
        )

        ax.set_xlabel('Product_HHI (Scaled)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Store_Preference (Scaled)', fontsize=11, fontweight='bold')
        ax.set_zlabel('Web_Engagement (Scaled)', fontsize=11, fontweight='bold')
        ax.set_title(f'K-Means Clustering (K={self.k}) - 3D View', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)

        plt.tight_layout()
        filepath = os.path.join(self.output_graph_dir, "05_Clusters_3D_HHI_Store_Web.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()

    def visualize_posthoc_preferences(self):
        """Visualize product preferences per cluster (mean values)."""
        fig, ax = plt.subplots(figsize=(14, 8))
        pref_cols = [
            'Wine_Preference', 'Meat_Preference', 'Fish_Preference', 'Fruit_Preference',
            'Sweet_Preference', 'Gold_Preference'
        ]
        x = np.arange(len(pref_cols))
        width = 0.8 / self.k

        for cluster_id in range(self.k):
            cluster_data = self.df[self.df['Cluster'] == cluster_id]
            means = [cluster_data[c].mean() for c in pref_cols]
            offset = width * cluster_id
            ax.bar(x + offset, means, width, label=f'Cluster {cluster_id}', alpha=0.85)

        ax.set_xlabel('Preference Dimensions', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean (Scaled)', fontsize=12, fontweight='bold')
        ax.set_title('Product Preferences per Cluster (Post-Hoc)', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * (self.k - 1) / 2)
        ax.set_xticklabels(pref_cols, rotation=20, ha='right', fontsize=11)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        filepath = os.path.join(self.output_graph_dir, "06_PostHoc_Preferences.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()

    def visualize_dominant_product_distribution(self, top_n=8):
        """Visualize Dominant_Product distribution per cluster."""
        fig, ax = plt.subplots(figsize=(15, 8))

        # Get global top categories to keep chart readable
        top_categories = (
            self.df['Dominant_Product']
            .value_counts()
            .head(top_n)
            .index
            .tolist()
        )
        x = np.arange(len(top_categories))
        width = 0.8 / self.k

        for cluster_id in range(self.k):
            cluster_data = self.df[self.df['Cluster'] == cluster_id]
            counts = cluster_data['Dominant_Product'].value_counts()
            pcts = [(counts.get(cat, 0) / len(cluster_data) * 100) if len(cluster_data) > 0 else 0 for cat in top_categories]
            offset = width * cluster_id
            ax.bar(x + offset, pcts, width, label=f'Cluster {cluster_id}', alpha=0.85)

        ax.set_xlabel('Dominant Product (Top categories)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
        ax.set_title('Dominant Product Distribution per Cluster', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * (self.k - 1) / 2)
        ax.set_xticklabels(top_categories, rotation=25, ha='right', fontsize=11)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        filepath = os.path.join(self.output_graph_dir, "07_PostHoc_DominantProduct.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()

    def visualize_top_product_share(self):
        """Visualize Top_Product_Share mean per cluster."""
        fig, ax = plt.subplots(figsize=(13, 7))
        means = []
        labels = []
        for cluster_id in range(self.k):
            cluster_data = self.df[self.df['Cluster'] == cluster_id]
            means.append(cluster_data['Top_Product_Share'].mean())
            labels.append(f'Cluster {cluster_id}')

        ax.bar(labels, means, color=plt.cm.tab10(np.linspace(0, 1, self.k)), alpha=0.85, edgecolor='black')
        ax.set_ylabel('Mean Top_Product_Share (Scaled)', fontsize=12, fontweight='bold')
        ax.set_title('Top Product Share by Cluster', fontsize=14, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)

        plt.tight_layout()
        filepath = os.path.join(self.output_graph_dir, "08_PostHoc_TopProductShare.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()

    def visualize_cluster_characteristics_line(self):
        """Line chart of cluster characteristics across all clustering + numeric post-hoc features."""
        fig, ax = plt.subplots(figsize=(15, 8))
        # Numeric post-hoc only (exclude Dominant_Product categorical)
        numeric_posthoc = [
            'Wine_Preference', 'Meat_Preference', 'Fish_Preference', 'Fruit_Preference',
            'Sweet_Preference', 'Gold_Preference', 'Top_Product_Share'
        ]
        feature_names = self.CLUSTERING_FEATURES + numeric_posthoc

        for cluster_id in range(self.k):
            cluster_data = self.df[self.df['Cluster'] == cluster_id]
            means = [cluster_data[c].mean() for c in feature_names]
            ax.plot(feature_names, means, 'o-', linewidth=2.5, markersize=8,
                    label=f'Cluster {cluster_id}', alpha=0.85)

        ax.set_ylabel('Mean Value (Scaled)', fontsize=12, fontweight='bold')
        ax.set_title('Cluster Characteristics Profile (Line Chart)', fontsize=14, fontweight='bold')
        ax.set_xticklabels(feature_names, rotation=25, ha='right', fontsize=11)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        filepath = os.path.join(self.output_graph_dir, "09_Cluster_Characteristics_Line.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()

    # --------------------------------------------------------------------------
    # Summary
    # --------------------------------------------------------------------------

    def print_summary(self):
        """Print concise summary of clustering results."""
        silhouette_avg = silhouette_score(self.X_clustering, self.kmeans.labels_)
        
        print(f"\n{'='*90}")
        print(f"K-MEANS CLUSTERING SUMMARY (K={self.k})".center(90))
        print(f"{'='*90}")
        print(f"Library: Custom Implementation (No Library)")
        print(f"Initialization: {self.init_method}")
        print(f"Inertia (WCSS): {self.kmeans.inertia_:.2f}")
        print(f"Iterations: {self.kmeans.n_iter_}")
        
        print(f"\nEvaluation Metrics:")
        print(f"  Silhouette Score: {silhouette_avg:.4f}")
        
        if self.optimal_k is not None:
            print(f"\nVoting Suggestion: K={self.optimal_k}")
            print(f"  - Elbow Method: K={self.elbow_k}")
            print(f"  - Silhouette: K={self.silhouette_k}")
        
        print(f"\nCluster Distribution:")
        cluster_counts = self.df['Cluster'].value_counts().sort_index()
        for cluster_id, count in cluster_counts.items():
            pct = count / len(self.df) * 100
            print(f"  Cluster {cluster_id}: {count:,} ({pct:.1f}%)")
        
        print(f"\nCentroid Locations (Scaled Features):")
        print(f"  Shape: {self.kmeans.cluster_centers_.shape} (n_clusters={self.k}, n_features=4)")
        for cluster_id in range(self.k):
            centroid = self.kmeans.cluster_centers_[cluster_id]
            print(f"  Cluster {cluster_id}: Product_HHI={centroid[0]:.6f}, Store_Pref={centroid[1]:.6f}, Web_Engage={centroid[2]:.6f}, PC1_Total={centroid[3]:.6f}")
        
        print(f"\nPCA Explained Variance:")
        print(f"  2D: PC1={self.pca_2d.explained_variance_ratio_[0]*100:.1f}%, PC2={self.pca_2d.explained_variance_ratio_[1]*100:.1f}%")
        print(f"  3D: PC1={self.pca_3d.explained_variance_ratio_[0]*100:.1f}%, PC2={self.pca_3d.explained_variance_ratio_[1]*100:.1f}%, PC3={self.pca_3d.explained_variance_ratio_[2]*100:.1f}%")
        
        print(f"\nVisualizations saved to: {self.output_graph_dir}")
        print(f"{'='*90}\n")

    def run_complete_analysis(self):
        """Run complete K-Means analysis pipeline."""
        self.load_data()
        self.compute_pca()

        if self.use_voting:
            self.find_optimal_k()

        self.fit_kmeans()
        self.print_summary()

        # Visualizations
        if self.optimal_k_selector is not None:
            self.visualize_optimal_k()
        self.visualize_pca_2d()
        self.visualize_pca_3d()
        self.visualize_clusters_2d()
        self.visualize_clusters_3d()
        self.visualize_posthoc_preferences()
        self.visualize_dominant_product_distribution()
        self.visualize_top_product_share()
        self.visualize_cluster_characteristics_line()


# ================================================================================
# MAIN EXECUTION
# ================================================================================

def main():
    """Main execution function."""

    INPUT_CSV = r"C:\Project\Machine_Learning\Machine_Learning\dataset\Customer_Behavior_ProductChannel_Robust_scaled.csv"
    OUTPUT_GRAPH_DIR = r"C:\Project\Machine_Learning\Machine_Learning\graph\Training\No Library\ProductChannel_NL"

    # Configuration: K=3 (adjustable), K-Means++ init (recommended), Use voting
    analyzer = ProductChannelKMeansAnalyzer(
        input_csv=INPUT_CSV,
        output_graph_dir=OUTPUT_GRAPH_DIR,
        k=4,  # ← Adjust K based on voting suggestions after first run
        init_kmeans_plus_plus=True,  # ← Use K-Means++ for stability
        use_voting=True,
        random_state=42
    )
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()
