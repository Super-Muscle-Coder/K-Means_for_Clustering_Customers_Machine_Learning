"""
================================================================================
K-MEANS CLUSTERING - DEMOGRAPHIC SEGMENTATION (WITH SKLEARN LIBRARY)
================================================================================

Module: K-Means for Demographic Customer Segmentation using Scikit-Learn
Strategy: Age, Income, Dependency_Ratio clustering with Elbow + Silhouette
Post-hoc: Education_ord, Life_Stage for deeper customer understanding

Features:
- sklearn.cluster.KMeans implementation
- Optimal K selection via Elbow + Silhouette (2-method voting)
- Flexible K adjustment (default K=2, adjustable based on voting)
- PCA visualization (2D + 3D)
- Line chart visualization for cluster characteristics
- Minimal console output, focus on visualizations
- Comparison metrics (Inertia, Silhouette Score, Davies-Bouldin Index)
- DISTINCT color scheme (Set2) for clear differentiation from No Library version

Input:  Customer_Behavior_Demographic_Robust_scaled.csv
Output: Optimized clustering + clean visualizations
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")


# ================================================================================
# COLOR PALETTE - SET2 (DISTINCT FROM NO LIBRARY VERSION)
# ================================================================================

def get_cluster_colors(n_clusters):
    """
    Get distinct colors using Set2 colormap.
    Set2: Qualitative palette with pastel colors (vibrant & easily distinguishable)
    Different from NL version which uses tab10 (standard colors)
    """
    if n_clusters <= 8:
        return plt.cm.Set2(np.linspace(0, 1, n_clusters))
    else:
        # Extend beyond 8 using Set3 for more clusters
        colors_set2 = plt.cm.Set2(np.linspace(0, 1, 8))
        colors_set3 = plt.cm.Set3(np.linspace(0, 1, n_clusters - 8))
        return np.vstack([colors_set2, colors_set3])


# ================================================================================
# OPTIMAL K SELECTOR (ELBOW + SILHOUETTE)
# ================================================================================

class OptimalKSelector:
    """Select optimal K using Elbow + Silhouette voting with sklearn."""
    
    def __init__(self, X, k_range=(2, 11), init='k-means++', random_state=42):
        """
        Parameters:
        -----------
        X : array-like
            Feature matrix
        k_range : tuple
            Range of K values to evaluate (inclusive, exclusive)
        init : str
            Initialization method ('k-means++' or 'random')
        random_state : int
            Random seed
        """
        self.X = np.array(X)
        self.k_range = k_range
        self.init = init
        self.random_state = random_state
        
        self.wcss_scores = {}
        self.silhouette_scores = {}
        self.davies_bouldin_scores = {}
        self.votes = {}
        
    def evaluate_all_k(self):
        """Evaluate all K values in range."""
        k_values = range(self.k_range[0], self.k_range[1])
        
        for k in k_values:
            kmeans = KMeans(
                n_clusters=k,
                init=self.init,
                n_init=10,
                max_iter=300,
                random_state=self.random_state
            )
            kmeans.fit(self.X)
            labels = kmeans.labels_
            
            self.wcss_scores[k] = kmeans.inertia_
            self.silhouette_scores[k] = silhouette_score(self.X, labels)
            self.davies_bouldin_scores[k] = davies_bouldin_score(self.X, labels)
    
    def vote_optimal_k(self):
        """Vote for optimal K using Elbow + Silhouette."""
        # Method 1: Elbow
        elbow_k = self._find_elbow_point()
        
        # Method 2: Silhouette (highest score)
        silhouette_k = max(self.silhouette_scores, key=self.silhouette_scores.get)
        
        # Count votes
        votes = [elbow_k, silhouette_k]
        for k in votes:
            self.votes[k] = self.votes.get(k, 0) + 1
        
        optimal_k = max(self.votes, key=self.votes.get)
        return optimal_k, elbow_k, silhouette_k
    
    def _find_elbow_point(self):
        """Find elbow point using normalized distance method."""
        k_values = sorted(self.wcss_scores.keys())
        wcss_values = np.array([self.wcss_scores[k] for k in k_values])
        
        # Normalize both axes to [0, 1]
        wcss_norm = (wcss_values - wcss_values.min()) / (wcss_values.max() - wcss_values.min())
        k_norm = (np.array(k_values) - k_values[0]) / (k_values[-1] - k_values[0])
        
        # Line from first to last point
        line_vec = np.array([k_norm[-1] - k_norm[0], wcss_norm[-1] - wcss_norm[0]])
        line_vec_norm = line_vec / np.linalg.norm(line_vec)
        
        # Calculate perpendicular distances
        distances = []
        for i in range(len(k_values)):
            point_vec = np.array([k_norm[i] - k_norm[0], wcss_norm[i] - wcss_norm[0]])
            distance = np.abs(np.cross(line_vec_norm, point_vec))
            distances.append(distance)
        
        elbow_idx = np.argmax(distances)
        return k_values[elbow_idx]


# ================================================================================
# DEMOGRAPHIC K-MEANS ANALYZER (SKLEARN VERSION)
# ================================================================================

class DemographicKMeansAnalyzer:
    """Complete K-Means analysis pipeline using sklearn."""
    
    def __init__(self, input_csv, output_graph_dir, k=2, init='k-means++', 
                 use_voting=True, random_state=42):
        """
        Parameters:
        -----------
        input_csv : str
            Path to Robust-scaled CSV
        output_graph_dir : str
            Directory to save visualizations
        k : int, default=2
            Initial number of clusters (adjustable)
        init : str, default='k-means++'
            Initialization method ('k-means++' or 'random')
        use_voting : bool, default=True
            Use Elbow+Silhouette voting to suggest optimal K
        random_state : int, default=42
            Random seed
        """
        self.input_csv = input_csv
        self.output_graph_dir = output_graph_dir
        self.k = k
        self.init = init
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
        self.colors = None
        
        self.CLUSTERING_FEATURES = ['Age', 'Income', 'Dependency_Ratio']
        
    def load_data(self):
        """Load and validate dataset."""
        self.df = pd.read_csv(self.input_csv)
        self.X_clustering = self.df[self.CLUSTERING_FEATURES].values
        self.colors = get_cluster_colors(self.k)
    
    def find_optimal_k(self):
        """Find optimal K using Elbow + Silhouette voting."""
        if not self.use_voting:
            return
        
        self.optimal_k_selector = OptimalKSelector(
            self.X_clustering,
            k_range=(2, 11),
            init=self.init,
            random_state=self.random_state
        )
        
        self.optimal_k_selector.evaluate_all_k()
        self.optimal_k, self.elbow_k, self.silhouette_k = self.optimal_k_selector.vote_optimal_k()
    
    def fit_kmeans(self):
        """Fit K-Means with selected K using sklearn."""
        self.kmeans = KMeans(
            n_clusters=self.k,
            init=self.init,
            n_init=10,
            max_iter=300,
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
    
    def visualize_optimal_k(self):
        """Visualize Elbow + Silhouette with voting suggestions."""
        if self.optimal_k_selector is None:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Optimal K Selection - Elbow & Silhouette Methods (Sklearn)', 
                    fontsize=16, fontweight='bold', color='#2E7D32')
        
        k_values = sorted(self.optimal_k_selector.wcss_scores.keys())
        
        # 1. Elbow Method
        ax = axes[0]
        wcss_vals = [self.optimal_k_selector.wcss_scores[k] for k in k_values]
        ax.plot(k_values, wcss_vals, 'o-', linewidth=2.5, markersize=8, 
               label='WCSS', color='#1976D2', markerfacecolor='#64B5F6')
        ax.axvline(self.elbow_k, color='#D32F2F', linestyle='--', linewidth=2.5, 
                  label=f'Elbow K={self.elbow_k}')
        ax.set_xlabel('Number of Clusters (K)', fontsize=12, fontweight='bold')
        ax.set_ylabel('WCSS (Within-Cluster Sum of Squares)', fontsize=12, fontweight='bold')
        ax.set_title('Elbow Method', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        
        # 2. Silhouette Score
        ax = axes[1]
        sil_vals = [self.optimal_k_selector.silhouette_scores[k] for k in k_values]
        ax.plot(k_values, sil_vals, 'o-', linewidth=2.5, markersize=8, 
               label='Silhouette', color='#F57C00', markerfacecolor='#FFB74D')
        ax.axvline(self.silhouette_k, color='#D32F2F', linestyle='--', linewidth=2.5, 
                  label=f'Silhouette K={self.silhouette_k}')
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
        
        for cluster_id in range(self.k):
            cluster_mask = self.df['Cluster'] == cluster_id
            ax.scatter(
                self.X_pca_2d[cluster_mask, 0],
                self.X_pca_2d[cluster_mask, 1],
                label=f'Cluster {cluster_id} (n={(cluster_mask).sum():,})',
                alpha=0.7,
                s=70,
                color=self.colors[cluster_id],
                edgecolors='#333333',
                linewidth=0.6
            )
        
        # Plot PCA centroids
        pca_centroids_2d = self.pca_2d.transform(self.kmeans.cluster_centers_)
        ax.scatter(
            pca_centroids_2d[:, 0],
            pca_centroids_2d[:, 1],
            marker='*',
            s=800,
            c='#FFD700',
            edgecolors='#333333',
            linewidth=2,
            label='Centroids',
            zorder=10
        )
        
        var_1 = self.pca_2d.explained_variance_ratio_[0] * 100
        var_2 = self.pca_2d.explained_variance_ratio_[1] * 100
        
        ax.set_xlabel(f'PC1 ({var_1:.1f}% variance)', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'PC2 ({var_2:.1f}% variance)', fontsize=12, fontweight='bold')
        ax.set_title(f'K-Means Clustering (K={self.k}) - PCA 2D Projection [Sklearn]', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath = os.path.join(self.output_graph_dir, "02_PCA_2D.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
    
    def visualize_pca_3d(self):
        """Visualize clusters using 3D PCA."""
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        for cluster_id in range(self.k):
            cluster_mask = self.df['Cluster'] == cluster_id
            ax.scatter(
                self.X_pca_3d[cluster_mask, 0],
                self.X_pca_3d[cluster_mask, 1],
                self.X_pca_3d[cluster_mask, 2],
                label=f'Cluster {cluster_id} (n={(cluster_mask).sum():,})',
                c=[self.colors[cluster_id]],
                alpha=0.7,
                s=50,
                edgecolors='#333333',
                linewidth=0.4
            )
        
        # Plot PCA centroids
        pca_centroids_3d = self.pca_3d.transform(self.kmeans.cluster_centers_)
        ax.scatter(
            pca_centroids_3d[:, 0],
            pca_centroids_3d[:, 1],
            pca_centroids_3d[:, 2],
            marker='*',
            s=800,
            c='#FFD700',
            edgecolors='#333333',
            linewidth=2,
            label='Centroids',
            zorder=10
        )
        
        var_1 = self.pca_3d.explained_variance_ratio_[0] * 100
        var_2 = self.pca_3d.explained_variance_ratio_[1] * 100
        var_3 = self.pca_3d.explained_variance_ratio_[2] * 100
        
        ax.set_xlabel(f'PC1 ({var_1:.1f}%)', fontsize=11, fontweight='bold')
        ax.set_ylabel(f'PC2 ({var_2:.1f}%)', fontsize=11, fontweight='bold')
        ax.set_zlabel(f'PC3 ({var_3:.1f}%)', fontsize=11, fontweight='bold')
        ax.set_title(f'K-Means Clustering (K={self.k}) - PCA 3D Projection [Sklearn]', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        
        plt.tight_layout()
        filepath = os.path.join(self.output_graph_dir, "03_PCA_3D.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
    
    def visualize_clusters_2d(self):
        """Visualize clusters in 2D (Age vs Income)."""
        fig, ax = plt.subplots(figsize=(13, 8))
        
        for cluster_id in range(self.k):
            cluster_data = self.df[self.df['Cluster'] == cluster_id]
            ax.scatter(
                cluster_data['Age'], 
                cluster_data['Income'],
                label=f'Cluster {cluster_id} (n={len(cluster_data):,})',
                alpha=0.7,
                s=70,
                color=self.colors[cluster_id],
                edgecolors='#333333',
                linewidth=0.6
            )
        
        # Plot centroids
        centroids = self.kmeans.cluster_centers_
        ax.scatter(
            centroids[:, 0],
            centroids[:, 1],
            marker='*',
            s=800,
            c='#FFD700',
            edgecolors='#333333',
            linewidth=2,
            label='Centroids',
            zorder=10
        )
        
        ax.set_xlabel('Age (Robust Scaled)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Income (Robust Scaled)', fontsize=12, fontweight='bold')
        ax.set_title(f'K-Means Clustering (K={self.k}, {self.init}) - Age vs Income [Sklearn]', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath = os.path.join(self.output_graph_dir, "04_Clusters_2D.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
    
    def visualize_clusters_3d(self):
        """Visualize clusters in 3D."""
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        for cluster_id in range(self.k):
            cluster_data = self.df[self.df['Cluster'] == cluster_id]
            ax.scatter(
                cluster_data['Age'],
                cluster_data['Income'],
                cluster_data['Dependency_Ratio'],
                label=f'Cluster {cluster_id} (n={len(cluster_data):,})',
                c=[self.colors[cluster_id]],
                alpha=0.7,
                s=50,
                edgecolors='#333333',
                linewidth=0.4
            )
        
        centroids = self.kmeans.cluster_centers_
        ax.scatter(
            centroids[:, 0],
            centroids[:, 1],
            centroids[:, 2],
            marker='*',
            s=800,
            c='#FFD700',
            edgecolors='#333333',
            linewidth=2,
            label='Centroids',
            zorder=10
        )
        
        ax.set_xlabel('Age (Scaled)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Income (Scaled)', fontsize=11, fontweight='bold')
        ax.set_zlabel('Dependency_Ratio (Scaled)', fontsize=11, fontweight='bold')
        ax.set_title(f'K-Means Clustering (K={self.k}) - 3D View [Sklearn]', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        
        plt.tight_layout()
        filepath = os.path.join(self.output_graph_dir, "05_Clusters_3D.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
    
    def visualize_posthoc_education(self):
        """Visualize Education distribution per cluster."""
        fig, ax = plt.subplots(figsize=(13, 7))
        
        edu_labels = ['None/Basic', '2nd Cycle', 'Graduation', 'Master', 'PhD']
        x = np.arange(len(edu_labels))
        width = 0.8 / self.k
        
        for cluster_id in range(self.k):
            cluster_data = self.df[self.df['Cluster'] == cluster_id]
            edu_counts = cluster_data['Education_ord'].value_counts().sort_index()
            
            edu_pcts = []
            for edu_level in range(5):
                count = edu_counts.get(edu_level, 0)
                pct = (count / len(cluster_data)) * 100
                edu_pcts.append(pct)
            
            offset = width * cluster_id
            ax.bar(x + offset, edu_pcts, width, label=f'Cluster {cluster_id}', 
                  color=self.colors[cluster_id], alpha=0.85, edgecolor='#333333', linewidth=0.7)
        
        ax.set_xlabel('Education Level', fontsize=12, fontweight='bold')
        ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
        ax.set_title('Education Distribution per Cluster (Post-Hoc) [Sklearn]', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * (self.k - 1) / 2)
        ax.set_xticklabels(edu_labels, rotation=15, ha='right', fontsize=11)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        filepath = os.path.join(self.output_graph_dir, "06_PostHoc_Education.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
    
    def visualize_posthoc_lifestage(self):
        """Visualize Life Stage distribution per cluster."""
        fig, ax = plt.subplots(figsize=(13, 7))
        
        stage_names = ['Young Single', 'Young Family', 'Mature Family', 'Empty Nest', 'Single Parent']
        x = np.arange(len(stage_names))
        width = 0.8 / self.k
        
        for cluster_id in range(self.k):
            cluster_data = self.df[self.df['Cluster'] == cluster_id]
            stage_counts = cluster_data['Life_Stage'].value_counts().sort_index()
            
            stage_pcts = []
            for stage in range(5):
                count = stage_counts.get(stage, 0)
                pct = (count / len(cluster_data)) * 100
                stage_pcts.append(pct)
            
            offset = width * cluster_id
            ax.bar(x + offset, stage_pcts, width, label=f'Cluster {cluster_id}', 
                  color=self.colors[cluster_id], alpha=0.85, edgecolor='#333333', linewidth=0.7)
        
        ax.set_xlabel('Life Stage', fontsize=12, fontweight='bold')
        ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
        ax.set_title('Life Stage Distribution per Cluster (Post-Hoc) [Sklearn]', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * (self.k - 1) / 2)
        ax.set_xticklabels(stage_names, rotation=25, ha='right', fontsize=11)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        filepath = os.path.join(self.output_graph_dir, "07_PostHoc_LifeStage.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
    
    def visualize_cluster_characteristics_line(self):
        """Line chart of cluster characteristics (all features)."""
        fig, ax = plt.subplots(figsize=(13, 7))
        
        feature_names = ['Age', 'Income', 'Dependency_Ratio', 'Education_ord', 'Life_Stage']
        markers = ['o', 's', '^', 'D', 'v']
        
        for cluster_id in range(self.k):
            cluster_data = self.df[self.df['Cluster'] == cluster_id]
            means = [
                cluster_data['Age'].mean(),
                cluster_data['Income'].mean(),
                cluster_data['Dependency_Ratio'].mean(),
                cluster_data['Education_ord'].mean(),
                cluster_data['Life_Stage'].mean()
            ]
            
            ax.plot(feature_names, means, 'o-', linewidth=2.5, markersize=9, 
                   label=f'Cluster {cluster_id}', color=self.colors[cluster_id], 
                   marker=markers[cluster_id % len(markers)], alpha=0.85)
        
        ax.set_ylabel('Mean Value (Scaled)', fontsize=12, fontweight='bold')
        ax.set_title('Cluster Characteristics Profile (Line Chart) [Sklearn]', 
                    fontsize=14, fontweight='bold')
        ax.set_xticklabels(feature_names, rotation=25, ha='right', fontsize=11)
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath = os.path.join(self.output_graph_dir, "08_Cluster_Characteristics_Line.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
    
    def print_summary(self):
        """Print concise summary of clustering results."""
        silhouette_avg = silhouette_score(self.X_clustering, self.kmeans.labels_)
        davies_bouldin = davies_bouldin_score(self.X_clustering, self.kmeans.labels_)
        
        print(f"\n{'='*80}")
        print(f"K-MEANS CLUSTERING SUMMARY (K={self.k}) - SKLEARN VERSION".center(80))
        print(f"{'='*80}")
        print(f"Library: scikit-learn (With Library Implementation)")
        print(f"Initialization: {self.init}")
        print(f"Inertia (WCSS): {self.kmeans.inertia_:.2f}")
        print(f"Iterations: {self.kmeans.n_iter_}")
        
        print(f"\nEvaluation Metrics:")
        print(f"  Silhouette Score: {silhouette_avg:.4f}")
        print(f"  Davies-Bouldin Index: {davies_bouldin:.4f}")
        
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
        print(f"  Shape: {self.kmeans.cluster_centers_.shape} (n_clusters={self.k}, n_features=3)")
        for cluster_id in range(self.k):
            centroid = self.kmeans.cluster_centers_[cluster_id]
            print(f"  Cluster {cluster_id}: Age={centroid[0]:.6f}, Income={centroid[1]:.6f}, Dependency_Ratio={centroid[2]:.6f}")
        
        print(f"\nPCA Explained Variance:")
        print(f"  2D: PC1={self.pca_2d.explained_variance_ratio_[0]*100:.1f}%, " \
              f"PC2={self.pca_2d.explained_variance_ratio_[1]*100:.1f}%")
        print(f"  3D: PC1={self.pca_3d.explained_variance_ratio_[0]*100:.1f}%, " \
              f"PC2={self.pca_3d.explained_variance_ratio_[1]*100:.1f}%, " \
              f"PC3={self.pca_3d.explained_variance_ratio_[2]*100:.1f}%")
        
        print(f"\nVisualizations saved to: {self.output_graph_dir}")
        print(f"{'='*80}\n")
    
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
        self.visualize_posthoc_education()
        self.visualize_posthoc_lifestage()
        self.visualize_cluster_characteristics_line()


# ================================================================================
# MAIN EXECUTION
# ================================================================================

def main():
    """Main execution function."""
    
    INPUT_CSV = r"C:\Project\Machine_Learning\Machine_Learning\dataset\Customer_Behavior_Demographic_Robust_scaled.csv"
    OUTPUT_GRAPH_DIR = r"C:\Project\Machine_Learning\Machine_Learning\graph\Training\WIth Library\Demographic_WL"
    
    # Configuration: K=3 (adjustable), k-means++ init (default), Use voting
    analyzer = DemographicKMeansAnalyzer(
        input_csv=INPUT_CSV,
        output_graph_dir=OUTPUT_GRAPH_DIR,
        k=3,  # ← Adjust K based on voting suggestions
        init='k-means++',  # ← K-Means++ initialization (default)
        use_voting=True,  # ← Use Elbow + Silhouette voting
        random_state=42
    )
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()