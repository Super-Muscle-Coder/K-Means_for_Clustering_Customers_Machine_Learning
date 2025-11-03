import os, sys, logging
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns

# =========================
# Cấu hình đầu vào
# =========================
dataset_path = r"C:\Project\Machine_Learning\Machine_Learning\dataset\Customer_Behavior_ProductChannel_robust_scaled.csv"
output_dir   = r"C:\Project\Machine_Learning\Machine_Learning\graph\Training\No Library\ProductChannel_NL\\"
report_dir   = r"C:\Project\Machine_Learning\Machine_Learning\report\Training\No Library\\"

K_MIN, K_MAX = 2, 10
RANDOM_STATE, MAX_ITER, TOL = 42, 300, 1e-4

# =========================
# Khởi tạo thư mục + logging
# =========================
os.makedirs(output_dir, exist_ok=True)
os.makedirs(report_dir, exist_ok=True)

log_file = os.path.join(report_dir, "ProductChannel_NL.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

rng = np.random.default_rng(RANDOM_STATE)

# =========================
# Tiện ích
# =========================
def load_dataset(path):
    df = pd.read_csv(path)
    feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not feature_cols:
        raise ValueError("No numeric features found.")
    return df, df[feature_cols].values, feature_cols

def euclidean_distances(A, B):
    return np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)

def kmeans_plus_plus_init(X, k):
    n_samples = X.shape[0]
    first_idx = rng.integers(0, n_samples)
    centroids = [X[first_idx]]
    for _ in range(1, k):
        dists = euclidean_distances(X, np.array(centroids))
        closest_dist_sq = np.min(dists**2, axis=1)
        probs = closest_dist_sq / np.sum(closest_dist_sq)
        next_idx = rng.choice(n_samples, p=probs)
        centroids.append(X[next_idx])
    return np.array(centroids)

def assign_clusters(X, centroids):
    return np.argmin(euclidean_distances(X, centroids), axis=1)

def update_centroids(X, labels, k):
    d = X.shape[1]
    new_centroids = np.zeros((k, d))
    for i in range(k):
        cluster_points = X[labels == i]
        if cluster_points.shape[0] == 0:
            idx = rng.integers(0, X.shape[0])
            new_centroids[i] = X[idx]
        else:
            new_centroids[i] = cluster_points.mean(axis=0)
    return new_centroids

def inertia(X, centroids, labels):
    diffs = X - centroids[labels]
    return np.sum(np.sum(diffs**2, axis=1))

def silhouette_score_custom(X, labels):
    n = X.shape[0]
    unique_labels = np.unique(labels)
    if unique_labels.size < 2: return np.nan
    D = euclidean_distances(X, X)
    s_vals = np.zeros(n)
    for i in range(n):
        li = labels[i]
        same = np.where(labels == li)[0]
        same = same[same != i]
        a = np.mean(D[i, same]) if same.size > 0 else 0.0
        b = np.inf
        for lj in unique_labels:
            if lj == li: continue
            other = np.where(labels == lj)[0]
            if other.size > 0:
                b = min(b, np.mean(D[i, other]))
        denom = max(a, b)
        s_vals[i] = (b - a) / denom if denom > 0 else 0.0
    return float(np.mean(s_vals))

def davies_bouldin_index(X, labels, centroids):
    k = centroids.shape[0]
    S = np.zeros(k)
    for i in range(k):
        Xi = X[labels == i]
        S[i] = np.mean(np.linalg.norm(Xi - centroids[i], axis=1)) if Xi.shape[0] > 0 else 0.0
    M = euclidean_distances(centroids, centroids)
    R = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            if i != j:
                R[i, j] = (S[i] + S[j]) / (M[i, j] + 1e-12)
    return float(np.mean(np.max(R, axis=1)))

def calinski_harabasz_index(X, labels, centroids):
    n, d = X.shape
    k = centroids.shape[0]
    overall_mean = X.mean(axis=0)
    W, B = 0.0, 0.0
    for i in range(k):
        Xi = X[labels == i]
        if Xi.shape[0] > 0:
            W += np.sum(np.sum((Xi - centroids[i])**2, axis=1))
            diff = centroids[i] - overall_mean
            B += Xi.shape[0] * np.sum(diff**2)
    return float((B / (k - 1)) / (W / (n - k))) if k > 1 and n > k else np.nan

# =========================
# K-Means chính
# =========================
def kmeans_nolib(X, k, max_iter=MAX_ITER, tol=TOL):
    centroids = kmeans_plus_plus_init(X, k)
    for _ in range(max_iter):
        labels = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, labels, k)
        if np.linalg.norm(new_centroids - centroids) < tol: break
        centroids = new_centroids
    inert = inertia(X, centroids, labels)
    sil = silhouette_score_custom(X, labels)
    dbi = davies_bouldin_index(X, labels, centroids)
    chi = calinski_harabasz_index(X, labels, centroids)
    return labels, centroids, inert, sil, dbi, chi

# =========================
# Vẽ biểu đồ cho từng k
# =========================
def save_heatmap_centroids(centroids, feature_cols, out_dir, k):
    centroids_df = pd.DataFrame(centroids, columns=feature_cols)
    heat_df = centroids_df.transpose()
    fig = plt.figure(figsize=(max(6, len(feature_cols)*0.5), max(4, centroids.shape[0]*0.3)))
    sns.heatmap(heat_df, cmap='vlag', annot=True, fmt=".2f")
    plt.title(f'Cluster Centroids Heatmap (k={k})')
    fname = os.path.join(out_dir, f"centroid_heatmap_k{k}.png")
    fig.savefig(fname, bbox_inches='tight', dpi=150)
    plt.close(fig)
    logger.info("Saved heatmap: %s", fname)

def save_centroid_plot(centroids, feature_cols, out_dir, k):
    fig = plt.figure(figsize=(10,6))
    for i in range(centroids.shape[0]):
        plt.plot(feature_cols, centroids[i], marker='o', label=f'Cluster {i}')
    plt.xticks(rotation=45, ha='right')
    plt.title(f'Cluster Centroid Plot (k={k})')
    plt.legend()
    fname = os.path.join(out_dir, f"centroid_plot_k{k}.png")
    fig.savefig(fname, bbox_inches='tight', dpi=150)
    plt.close(fig)
    logger.info("Saved centroid plot: %s", fname)

def save_pairplot(df, labels, feature_cols, out_dir, k):
    variances = df[feature_cols].var().sort_values(ascending=False)
    pairplot_cols = variances.index[:4].tolist()
    pair_df = df[pairplot_cols].copy()
    pair_df['cluster'] = labels
    sns.set(style="ticks")
    g = sns.pairplot(pair_df, hue='cluster', diag_kind='kde', plot_kws={'s':20})
    fname = os.path.join(out_dir, f"pairplot_k{k}.png")
    g.fig.savefig(fname, bbox_inches='tight', dpi=150)
    plt.close('all')
    logger.info("Saved pairplot: %s", fname)

# =========================
# Chạy toàn bộ vòng lặp k
# =========================
def main():
    df, X, feature_cols = load_dataset(dataset_path)
    results = []

    for k in range(K_MIN, K_MAX + 1):
        logger.info("Running custom K-Means for k=%d", k)
        labels, centroids, inert, sil, dbi, chi = kmeans_nolib(X, k)
        results.append((k, inert, sil, dbi, chi))

        # Lưu biểu đồ cho từng k
        save_heatmap_centroids(centroids, feature_cols, output_dir, k)
        save_centroid_plot(centroids, feature_cols, output_dir, k)
        try:
            save_pairplot(df, labels, feature_cols, output_dir, k)
        except Exception as e:
            logger.warning("Pairplot failed for k=%d: %s", k, str(e))

        # Log kết quả từng k
        logger.info("- k=%d: inertia=%.4f, silhouette=%.4f, calinski_harabasz=%.4f, davies_bouldin=%.4f",
                    k, inert, sil, chi, dbi)

    # Tóm tắt kết quả
    res_df = pd.DataFrame(results, columns=['k', 'inertia', 'silhouette', 'davies_bouldin', 'calinski_harabasz'])
    best_by_sil = res_df.loc[res_df['silhouette'].idxmax()]
    worst_by_sil = res_df.loc[res_df['silhouette'].idxmin()]
    inertias = res_df['inertia'].values
    ks = res_df['k'].values
    deltas = np.diff(inertias)
    elbow_k = ks[np.argmin(deltas) + 1] if len(deltas) > 0 else ks[0]

    logger.info("\n================ SUMMARY ================")
    logger.info("All k results:\n%s", res_df.to_string(index=False))
    logger.info("\nBest k by silhouette:\n%s", best_by_sil.to_string(index=False))
    logger.info("\nWorst k by silhouette:\n%s", worst_by_sil.to_string(index=False))
    logger.info("\nHeuristic elbow k: %d", elbow_k)

if __name__ == "__main__":
    main()
