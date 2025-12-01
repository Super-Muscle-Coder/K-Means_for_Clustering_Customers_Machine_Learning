import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import warnings

from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")



class CustomKMeans:
    """Self-implemented K-Means clustering algorithm."""
    
    def __init__(self, n_clusters=2, init_method='random', max_iter=300, 
                 n_init=10, tol=1e-4, random_state=42):
        self.n_clusters = n_clusters # Số cụm K
        self.init_method = init_method # Phương pháp khởi tạo ('random' hoặc 'kmeans++')
        self.max_iter = max_iter # Số vòng lặp tối đa
        self.n_init = n_init # Số lần khởi tạo khác nhau
        self.tol = tol # Ngưỡng hội tụ
        self.random_state = random_state 
        
        self.cluster_centers_ = None # Tọa độ tâm cụm
        self.labels_ = None # Nhãn cụm cho mỗi điểm dữ liệu
        self.inertia_ = None # Tổng WCSS
        self.n_iter_ = 0 # Số vòng lặp thực tế
        
    def _init_centroids_random(self, X): # Khởi tạo tâm cụm ngẫu nhiên
        """Initialize centroids randomly."""
        np.random.seed(self.random_state) # Đặt seed ngẫu nhiên
        random_indices = np.random.choice(X.shape[0], self.n_clusters, replace=False) # Chọn ngẫu nhiên K điểm dữ liệu
        return X[random_indices].copy() # Trả về tọa độ tâm cụm
    
    def _init_centroids_kmeans_plus_plus(self, X): # Khởi tạo tâm cụm bằng K-Means++
        """Initialize centroids using K-Means++ algorithm."""
        np.random.seed(self.random_state) # Đặt seed ngẫu nhiên
        n_samples = X.shape[0] # Số mẫu dữ liệu
        
        centroids = [X[np.random.randint(n_samples)]] # Chọn ngẫu nhiên tâm cụm đầu tiên
        
        for _ in range(1, self.n_clusters): # Lặp để chọn các tâm cụm tiếp theo
            distances = np.array([ 
                min([np.linalg.norm(x - c)**2 for c in centroids]) # Khoảng cách nhỏ nhất từ điểm x đến các tâm cụm đã chọn
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
        """Assign each sample to nearest centroid."""
        distances = np.zeros((X.shape[0], self.n_clusters)) # Ma trận khoảng cách
        for k in range(self.n_clusters):
            distances[:, k] = np.linalg.norm(X - centroids[k], axis=1) # Tính khoảng cách từ mỗi điểm đến tâm cụm k
        return np.argmin(distances, axis=1) # Trả về nhãn cụm cho mỗi điểm dữ liệu
    
    def _update_centroids(self, X, labels):
        """Update centroids as mean of assigned samples."""
        centroids = np.zeros((self.n_clusters, X.shape[1])) # Ma trận tọa độ tâm cụm mới
        for k in range(self.n_clusters): 
            cluster_points = X[labels == k] # Lấy các điểm thuộc cụm k
            if len(cluster_points) > 0: 
                centroids[k] = cluster_points.mean(axis=0) # Cập nhật tâm cụm là trung bình của các điểm trong cụm
            else:
                centroids[k] = X[np.random.randint(X.shape[0])] # Nếu cụm rỗng, chọn ngẫu nhiên một điểm dữ liệu làm tâm cụm
        return centroids
    
    def _calculate_inertia(self, X, labels, centroids):
        """Calculate within-cluster sum of squares."""
        inertia = 0
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]  # Lấy các điểm thuộc cụm k
            if len(cluster_points) > 0: # Nếu cụm không rỗng
                inertia += np.sum((cluster_points - centroids[k])**2) # Cộng WCSS của cụm k
        return inertia
    
    def fit(self, X):
        """Fit K-Means clustering."""
        X = np.array(X)
        best_inertia = np.inf
        best_centroids = None
        best_labels = None
        best_n_iter = 0
        
        for init_run in range(self.n_init):
            if self.init_method == 'kmeans++':
                centroids = self._init_centroids_kmeans_plus_plus(X) # Khởi tạo tâm cụm bằng K-Means++
            else:
                centroids = self._init_centroids_random(X) # Khởi tạo tâm cụm ngẫu nhiên
            
            for iteration in range(self.max_iter): # Lặp tối đa
                labels = self._assign_clusters(X, centroids) # Gán nhãn cụm
                new_centroids = self._update_centroids(X, labels) # Cập nhật tâm cụm
                
                centroid_shift = np.linalg.norm(new_centroids - centroids) # Tính độ dịch chuyển của tâm cụm
                centroids = new_centroids # Cập nhật tâm cụm
                
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
        """Predict cluster labels for new data."""
        X = np.array(X)
        return self._assign_clusters(X, self.cluster_centers_)


# ================================================================================
# Lựa chọn số cụm tối ưu (ELBOW + SILHOUETTE)
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
        """Evaluate all K values in range."""
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
        """Vote for optimal K using Elbow + Silhouette."""
        # Method 1: Elbow
        elbow_k = self._find_elbow_point()
        
        # Method 2: Silhouette
        silhouette_k = max(self.silhouette_scores, key=self.silhouette_scores.get)
        
        # Count votes
        votes = [elbow_k, silhouette_k]
        for k in votes:
            self.votes[k] = self.votes.get(k, 0) + 1
        
        optimal_k = max(self.votes, key=self.votes.get)
        return optimal_k, elbow_k, silhouette_k
    
    def _find_elbow_point(self):
        """Find elbow point using angle method."""
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
# DEMOGRAPHIC K-MEANS ANALYZER
# ================================================================================

class DemographicKMeansAnalyzer:
    """Complete K-Means analysis pipeline for Demographic segmentation."""
    
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
        self.pca = None
        self.pca_2d = None
        self.pca_3d = None
        
        self.CLUSTERING_FEATURES = ['Age', 'Income', 'Dependency_Ratio']
        self.POSTHOC_FEATURES = ['Education_ord', 'Life_Stage']
        
    def load_data(self):
        """Load and validate dataset."""
        self.df = pd.read_csv(self.input_csv)
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
    
    def visualize_optimal_k(self):
        """Visualize Elbow + Silhouette with voting suggestions."""
        if self.optimal_k_selector is None:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Optimal K Selection - Elbow & Silhouette Methods', fontsize=16, fontweight='bold')
        
        k_values = sorted(self.optimal_k_selector.wcss_scores.keys())
        
        # 1. Elbow Method
        ax = axes[0]
        wcss_vals = [self.optimal_k_selector.wcss_scores[k] for k in k_values]
        ax.plot(k_values, wcss_vals, 'bo-', linewidth=2.5, markersize=8, label='WCSS')
        ax.axvline(self.elbow_k, color='red', linestyle='--', linewidth=2.5, label=f'Elbow K={self.elbow_k}')
        ax.set_xlabel('Number of Clusters (K)', fontsize=12, fontweight='bold')
        ax.set_ylabel('WCSS (Within-Cluster Sum of Squares)', fontsize=12, fontweight='bold')
        ax.set_title('Elbow Method', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        
        # 2. Silhouette Score
        ax = axes[1]
        sil_vals = [self.optimal_k_selector.silhouette_scores[k] for k in k_values]
        ax.plot(k_values, sil_vals, 'go-', linewidth=2.5, markersize=8, label='Silhouette')
        ax.axvline(self.silhouette_k, color='red', linestyle='--', linewidth=2.5, 
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
        
        # Plot PCA centroids
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
        from mpl_toolkits.mplot3d import Axes3D
        
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
        
        # Plot PCA centroids
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
        """Visualize clusters in 2D (Age vs Income)."""
        fig, ax = plt.subplots(figsize=(13, 8))
        
        colors = plt.cm.tab10(np.linspace(0, 1, self.k))
        
        for cluster_id in range(self.k):
            cluster_data = self.df[self.df['Cluster'] == cluster_id]
            ax.scatter(
                cluster_data['Age'], 
                cluster_data['Income'],
                label=f'Cluster {cluster_id} (n={len(cluster_data):,})',
                alpha=0.65,
                s=60,
                color=colors[cluster_id],
                edgecolors='black',
                linewidth=0.5
            )
        
        # Plot centroids
        centroids = self.kmeans.cluster_centers_
        ax.scatter(
            centroids[:, 0],
            centroids[:, 1],
            marker='X',
            s=400,
            c='red',
            edgecolors='black',
            linewidth=2.5,
            label='Centroids',
            zorder=10
        )
        
        ax.set_xlabel('Age (Robust Scaled)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Income (Robust Scaled)', fontsize=12, fontweight='bold')
        ax.set_title(f'K-Means Clustering (K={self.k}, init={self.init_method}) - Age vs Income', 
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
        
        colors = plt.cm.tab10(np.linspace(0, 1, self.k))
        
        for cluster_id in range(self.k):
            cluster_data = self.df[self.df['Cluster'] == cluster_id]
            ax.scatter(
                cluster_data['Age'],
                cluster_data['Income'],
                cluster_data['Dependency_Ratio'],
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
        
        ax.set_xlabel('Age (Scaled)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Income (Scaled)', fontsize=11, fontweight='bold')
        ax.set_zlabel('Dependency_Ratio (Scaled)', fontsize=11, fontweight='bold')
        ax.set_title(f'K-Means Clustering (K={self.k}) - 3D View', fontsize=14, fontweight='bold')
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
            ax.bar(x + offset, edu_pcts, width, label=f'Cluster {cluster_id}', alpha=0.8)
        
        ax.set_xlabel('Education Level', fontsize=12, fontweight='bold')
        ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
        ax.set_title('Education Distribution per Cluster (Post-Hoc)', fontsize=14, fontweight='bold')
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
            ax.bar(x + offset, stage_pcts, width, label=f'Cluster {cluster_id}', alpha=0.8)
        
        ax.set_xlabel('Life Stage', fontsize=12, fontweight='bold')
        ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
        ax.set_title('Life Stage Distribution per Cluster (Post-Hoc)', fontsize=14, fontweight='bold')
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
        
        for cluster_id in range(self.k):
            cluster_data = self.df[self.df['Cluster'] == cluster_id]
            means = [
                cluster_data['Age'].mean(),
                cluster_data['Income'].mean(),
                cluster_data['Dependency_Ratio'].mean(),
                cluster_data['Education_ord'].mean(),
                cluster_data['Life_Stage'].mean()
            ]
            
            ax.plot(feature_names, means, 'o-', linewidth=2.5, markersize=8, 
                   label=f'Cluster {cluster_id}', alpha=0.8)
        
        ax.set_ylabel('Mean Value (Scaled)', fontsize=12, fontweight='bold')
        ax.set_title('Cluster Characteristics Profile (Line Chart)', fontsize=14, fontweight='bold')
        ax.set_xticklabels(feature_names, rotation=25, ha='right', fontsize=11)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath = os.path.join(self.output_graph_dir, "08_Cluster_Characteristics_Line.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
    
    def print_summary(self):
        """Print concise summary of clustering results."""
        silhouette_avg = silhouette_score(self.X_clustering, self.kmeans.labels_)
        
        print(f"\n{'='*80}")
        print(f"K-MEANS CLUSTERING SUMMARY (K={self.k})".center(80))
        print(f"{'='*80}")
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
        print(f"  Shape: {self.kmeans.cluster_centers_.shape} (n_clusters={self.k}, n_features=3)")
        for cluster_id in range(self.k):
            centroid = self.kmeans.cluster_centers_[cluster_id]
            print(f"  Cluster {cluster_id}: Age={centroid[0]:.6f}, Income={centroid[1]:.6f}, Dependency_Ratio={centroid[2]:.6f}")
        
        print(f"\nPCA Explained Variance:")
        print(f"  2D: PC1={self.pca_2d.explained_variance_ratio_[0]*100:.1f}%, PC2={self.pca_2d.explained_variance_ratio_[1]*100:.1f}%")
        print(f"  3D: PC1={self.pca_3d.explained_variance_ratio_[0]*100:.1f}%, PC2={self.pca_3d.explained_variance_ratio_[1]*100:.1f}%, PC3={self.pca_3d.explained_variance_ratio_[2]*100:.1f}%")
        
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
    OUTPUT_GRAPH_DIR = r"C:\Project\Machine_Learning\Machine_Learning\graph\Training\No Library\DemoGraphic_NL"
    
    # Configuration: K=3 (adjustable), Random init (default), Use voting
    analyzer = DemographicKMeansAnalyzer(
        input_csv=INPUT_CSV,
        output_graph_dir=OUTPUT_GRAPH_DIR,
        k=3,  # ← Adjust K based on voting suggestions
        init_kmeans_plus_plus=True,  # ← Random initialization (default)
        use_voting=True,  # ← Use Elbow + Silhouette voting
        random_state=42
    )
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()