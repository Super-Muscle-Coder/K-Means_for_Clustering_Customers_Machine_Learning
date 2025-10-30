"""
Workflow: K-Means Clustering

Quy trình:
1. Load dữ liệu đã chuẩn hóa (feature-scaled) từ CSV.
2. Kiểm tra chất lượng dữ liệu (missing, dtype, thống kê cơ bản).
3. Tính toán thống kê trước khi phân cụm (mean/std/min/max).
4. Tìm K tối ưu bằng Elbow method + Silhouette score.
   - Elbow: quan sát inertia (within-cluster sum-of-squares).
   - Silhouette: đánh giá nội tại ([-1,1]), giá trị lớn hơn thể hiện phân cụm tốt.
   - Kết hợp để chọn K cân bằng giữa compactness và separation.
5. Tính toán Gap Statistic để bổ sung quyết định chọn K.
6. Khởi tạo KMeans với init='k-means++' và chạy phân cụm.
7. Đánh giá nội tại sau khi chạy: inertia, silhouette score, cluster sizes, centroid.
8. Xuất các biểu đồ: Elbow, Silhouette (score vs K), Gap Statistic, Heatmap (feature vs cluster centroid),
   Scatter (2D bằng PCA), Cluster Centroid Plot (centroid values theo feature).
9. Lưu kết quả nhãn (cluster labels) nếu cần, và xuất báo cáo log chi tiết.

Kiến thức cần biết về K-Means (tóm tắt):
- K-Means tối ưu hóa inertia (tổng bình phương khoảng cách điểm tới tâm cụm).
- K cần được chọn trước (không có nhãn).
- Do dùng khoảng cách Euclidean, dữ liệu cần được scaling trước.
- K-Means++ giúp khởi tạo tâm cụm tốt hơn, giảm khả năng rơi vào cực trị địa phương.
- Đánh giá nội tại: inertia, silhouette, calinski_harabasz, davies_bouldin. Ở đây dùng silhouette + inertia + gap statistic.

Tiêu chí đánh giá (internal only):
- Silhouette score trung bình càng cao càng tốt.
- Inertia càng thấp càng tốt nhưng phải so sánh theo K.
- Gap Statistic: K tối ưu khi gap(k) >= gap(k+1) - s_{k+1} (theo Tibshirani).
- Kết hợp các chỉ số để đưa ra K hợp lý.

Hướng dẫn sử dụng:
- Thay đường dẫn dataset_path và output_* tương ứng cho từng dataset (Demographic/ProductChannel/RFM).
- Chạy script: python kmeans_pipeline.py
"""

import os
import sys
import logging
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
import math

# -----------------------------
# Cấu hình (chỉnh đường dẫn tương ứng trước khi chạy)
# -----------------------------
# Thay 1 biến dưới đây tương ứng cho dataset Demographic/ProductChannel/RFM trước khi chạy.
dataset_path = r"C:\Project\Machine_Learning\Machine_Learning\dataset\Customer_Behavior_RFM_robust_scaled.csv"

# Thư mục lưu biểu đồ (tùy dataset, set tương ứng)
graph_output_dir = r"C:\Project\Machine_Learning\Machine_Learning\graph\Training\RFM\\"

# Thư mục lưu report log
report_dir = r"C:\Project\Machine_Learning\Machine_Learning\report\Traning\\"

# Tên file log (tự động thêm timestamp)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(report_dir, f"RFM.log")  # đổi tên phù hợp nếu cần

# Các tham số tìm K
K_MIN = 2
K_MAX = 10  # bạn có thể tăng nếu dataset lớn / cần thử nhiều K hơn
RANDOM_STATE = 42
N_INIT = 10
MAX_ITER = 300       

# Gap statistic params
GAP_B = 10  # số mẫu tham chiếu; tăng giá trị để ổn định hơn nhưng tốn thời gian

# -----------------------------
# Kiểm tra và tạo thư mục cần thiết
# -----------------------------
os.makedirs(graph_output_dir, exist_ok=True)
os.makedirs(report_dir, exist_ok=True)

# -----------------------------
# Thiết lập logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# -----------------------------
# Hàm phụ trợ
# -----------------------------
def load_dataset(path):
    """Nạp dataset từ CSV, trả về DataFrame (chưa đổi nhãn)"""
    logger.info("Loading dataset: %s", path)
    df = pd.read_csv(path)
    logger.info("Loaded shape: %s", df.shape)
    return df

def validate_dataframe(df):
    """Kiểm tra nhanh quality: missing, dtypes, basic stats"""
    logger.info("Validating dataframe...")
    missing = df.isnull().sum()
    total_missing = missing.sum()
    if total_missing > 0:
        logger.warning("Missing values detected:")
        for col, cnt in missing.items():
            if cnt > 0:
                logger.warning("  %s : %d", col, cnt)
    else:
        logger.info("No missing values detected.")

    logger.info("Data types:")
    logger.info(df.dtypes.to_string())

    logger.info("Basic describe():")
    logger.info(df.describe().to_string())

def compute_pre_stats(df):
    """Trích thống kê trước khi scaling / trước clustering"""
    logger.info("Pre-clustering statistics (per feature):")
    stats = df.describe().transpose()
    logger.info("\n" + stats.to_string())

def save_plot(fig, fname):
    path = os.path.join(graph_output_dir, fname)
    fig.savefig(path, bbox_inches='tight', dpi=150)
    logger.info("Saved plot: %s", path)
    plt.close(fig)

def compute_gap_statistic(X, refs=None, nrefs=GAP_B, max_k=K_MAX):
    """
    Compute Gap Statistic for k=1..max_k
    Returns: gaps, sk, ref_inertia
    Implementation: Tibshirani et al.
    """
    logger.info("Computing Gap Statistic with %d reference datasets...", nrefs)
    shape = X.shape
    if refs is None:
        # create nrefs reference datasets by sampling from uniform distribution within feature-wise min-max
        tops = X.max(axis=0)
        bots = X.min(axis=0)
        dists = np.diag(tops - bots)
        rands = np.random.RandomState(RANDOM_STATE).rand(nrefs, shape[0], shape[1])
        refs = np.zeros_like(rands)
        for i in range(nrefs):
            refs[i] = rands[i] * (tops - bots) + bots

    # compute Wk for reference and original data
    def wk(data, k):
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=N_INIT, random_state=RANDOM_STATE)
        kmeans.fit(data)
        # inertia is sum of squared distances to closest cluster center
        return kmeans.inertia_

    gaps = np.zeros(max_k)
    sk = np.zeros(max_k)
    wk_orig = np.zeros(max_k)
    wk_refs = np.zeros((nrefs, max_k))

    for k in range(1, max_k + 1):
        wk_orig[k-1] = wk(X, k)
        for i in range(nrefs):
            wk_refs[i, k-1] = wk(refs[i], k)
        log_wk_refs = np.log(wk_refs[:, k-1])
        gap = np.mean(log_wk_refs) - np.log(wk_orig[k-1])
        gaps[k-1] = gap
        sk[k-1] = np.sqrt(np.mean((log_wk_refs - np.mean(log_wk_refs))**2)) * np.sqrt(1 + 1.0 / nrefs)
        logger.info("Gap k=%d: gap=%.4f, sk=%.4f, log(Wk)=%.4f", k, gaps[k-1], sk[k-1], np.log(wk_orig[k-1]))
    return gaps, sk

# -----------------------------
# Main pipeline
# -----------------------------
def run_kmeans_pipeline(dataset_path, graph_output_dir):
    # 1. Load
    df = load_dataset(dataset_path)
    # Nếu dataframe có cột ID hoặc non-feature, loại bỏ nếu cần.
    # Giả sử các cột đặc trưng là tất cả cột numeric; giữ nguyên tên cột để dùng trong heatmap/centroid.
    feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(feature_cols) == 0:
        logger.critical("No numeric features found in dataset. Exiting.")
        return
    X = df[feature_cols].values

    # 2. Validate
    validate_dataframe(df)
    compute_pre_stats(df[feature_cols])

    # 3. Tìm K tối ưu: Elbow + Silhouette
    inertias = []
    silhouette_scores = []
    ch_scores = []
    db_scores = []
    K_range = list(range(K_MIN, K_MAX + 1))
    logger.info("Evaluating K from %d to %d", K_MIN, K_MAX)

    for k in K_range:
        logger.info("Fitting KMeans for k=%d", k)
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=N_INIT, random_state=RANDOM_STATE, max_iter=MAX_ITER)
        labels = kmeans.fit_predict(X)
        inertia = kmeans.inertia_
        inertias.append(inertia)
        logger.info("  inertia: %.4f", inertia)
        if k > 1:
            sil = silhouette_score(X, labels)
            silhouette_scores.append(sil)
            ch = calinski_harabasz_score(X, labels)
            db = davies_bouldin_score(X, labels)
            ch_scores.append(ch)
            db_scores.append(db)
            logger.info("  silhouette: %.4f, calinski_harabasz: %.4f, davies_bouldin: %.4f", sil, ch, db)
        else:
            silhouette_scores.append(np.nan)
            ch_scores.append(np.nan)
            db_scores.append(np.nan)

    # 3a. Vẽ Elbow plot
    fig = plt.figure(figsize=(6,4))
    plt.plot(K_range, inertias, 'o-', color='tab:blue')
    plt.xlabel('K')
    plt.ylabel('Inertia')
    plt.title('Elbow Plot (Inertia vs K)')
    save_plot(fig, "elbow_plot.png")

    # 3b. Vẽ Silhouette score vs K
    fig = plt.figure(figsize=(6,4))
    plt.plot(K_range, silhouette_scores, 'o-', color='tab:green')
    plt.xlabel('K')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score vs K')
    save_plot(fig, "silhouette_vs_k.png")

    # 4. Gap statistic
    try:
        gaps, sk = compute_gap_statistic(X, nrefs=GAP_B, max_k=K_MAX)
        # Plot Gap
        fig = plt.figure(figsize=(6,4))
        ks = list(range(1, K_MAX+1))
        plt.plot(ks, gaps, 'o-', color='tab:purple')
        plt.fill_between(ks, gaps - sk, gaps + sk, color='lavender', alpha=0.4)
        plt.xlabel('K')
        plt.ylabel('Gap Statistic')
        plt.title('Gap Statistic vs K')
        save_plot(fig, "gap_statistic.png")
    except Exception as e:
        logger.warning("Gap Statistic computation failed: %s", str(e))
        gaps = None
        sk = None

    # 5. Quyết định K tối ưu (tự động đề xuất; người dùng có thể điều chỉnh)
    # Heuristic: chọn K với silhouette cao nhất (k>=2), đồng thời kiểm tra elbow (giảm nhiều inertia)
    best_k_by_sil = K_range[1 + int(np.nanargmax(silhouette_scores[1:]))] if len(K_range) > 1 else K_range[0]
    logger.info("Best K by silhouette: %d", best_k_by_sil)

    # Nếu có gap statistic, áp dụng Tibshirani rule:
    chosen_k = best_k_by_sil
    if gaps is not None:
        # tìm k* = smallest k such that gap(k) >= gap(k+1) - s_{k+1}
        k_star = None
        for k in range(1, len(gaps)):
            if gaps[k-1] >= gaps[k] - sk[k]:
                k_star = k
                break
        if k_star is not None:
            k_star = max(k_star, K_MIN)  # đảm bảo >= K_MIN
            chosen_k = k_star
            logger.info("Gap statistic suggests k = %d", chosen_k)
        else:
            logger.info("Gap statistic did not suggest a clear k; keep silhouette suggestion: %d", chosen_k)
    else:
        logger.info("No gap statistic result; choose K by silhouette: %d", chosen_k)

    # 6. Chạy KMeans cuối cùng với init='k-means++'
    logger.info("Running final KMeans with K=%d", chosen_k)
    kmeans_final = KMeans(n_clusters=chosen_k, init='k-means++', n_init=50, random_state=RANDOM_STATE, max_iter=MAX_ITER)
    labels_final = kmeans_final.fit_predict(X)
    centroids = kmeans_final.cluster_centers_
    inertia_final = kmeans_final.inertia_
    sil_final = silhouette_score(X, labels_final) if chosen_k > 1 else float('nan')
    ch_final = calinski_harabasz_score(X, labels_final) if chosen_k > 1 else float('nan')
    db_final = davies_bouldin_score(X, labels_final) if chosen_k > 1 else float('nan')

    logger.info("Final inertia: %.4f", inertia_final)
    logger.info("Final silhouette: %.4f", sil_final)
    logger.info("Final calinski_harabasz: %.4f", ch_final)
    logger.info("Final davies_bouldin: %.4f", db_final)

    # Cluster sizes
    unique, counts = np.unique(labels_final, return_counts=True)
    logger.info("Cluster sizes:")
    for u, c in zip(unique, counts):
        logger.info("  Cluster %d : %d", int(u), int(c))

    # 7. Heatmap of centroids (features x clusters)
    centroids_df = pd.DataFrame(centroids, columns=feature_cols)
    # transpose for heatmap (features as rows)
    heat_df = centroids_df.transpose()
    fig = plt.figure(figsize=(max(6, len(feature_cols)*0.5), max(4, chosen_k*0.3)))
    sns.heatmap(heat_df, cmap='vlag', annot=True, fmt=".2f")
    plt.xlabel('Cluster')
    plt.ylabel('Feature')
    plt.title('Cluster Centroids Heatmap (features x cluster)')
    save_plot(fig, "centroid_heatmap.png")

    # 8. Cluster centroid plot (centroid values per feature)
    fig = plt.figure(figsize=(10,6))
    for i in range(centroids.shape[0]):
        plt.plot(feature_cols, centroids[i], marker='o', label=f'Cluster {i}')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Feature')
    plt.ylabel('Centroid value')
    plt.title('Cluster Centroid Plot')
    plt.legend()
    save_plot(fig, "centroid_plot.png")

    # 9. 2D scatter by PCA (visualization)
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X)
    df_vis = pd.DataFrame({
        'pc1': X_pca[:,0],
        'pc2': X_pca[:,1],
        'cluster': labels_final
    })
    fig = plt.figure(figsize=(8,6))
    palette = sns.color_palette("tab10", n_colors=chosen_k)
    sns.scatterplot(data=df_vis, x='pc1', y='pc2', hue='cluster', palette=palette, s=30, alpha=0.8)
    plt.title('Clusters visualized by PCA (2D)')
    save_plot(fig, "pca_scatter.png")

    # 10. Scatter matrix (pairplot) with cluster hue - nếu feature số lượng lớn, chỉ chọn top 4 features by variance
    num_features = len(feature_cols)
    max_pairplot_features = 4
    if num_features <= max_pairplot_features:
        pairplot_cols = feature_cols
    else:
        variances = df[feature_cols].var().sort_values(ascending=False)
        pairplot_cols = variances.index[:max_pairplot_features].tolist()
    try:
        sns.set(style="ticks")
        pair_df = df[pairplot_cols].copy()
        pair_df['cluster'] = labels_final
        g = sns.pairplot(pair_df, hue='cluster', palette=palette, diag_kind='kde', plot_kws={'s':20})
        fname = os.path.join(graph_output_dir, "pairplot.png")
        g.fig.savefig(fname, bbox_inches='tight', dpi=150)
        logger.info("Saved plot: %s", fname)
        plt.close('all')
    except Exception as e:
        logger.warning("Pairplot failed: %s", str(e))

    # 11. Save labels appended to original df if desired
    out_labels_path = os.path.join(graph_output_dir, "cluster_labels.csv")
    df_out = df.copy()
    df_out['cluster'] = labels_final
    df_out.to_csv(out_labels_path, index=False)
    logger.info("Saved cluster labels: %s", out_labels_path)

    # 12. Export final report summary (printed to log as well)
    logger.info("\n" + "="*60)
    logger.info("FINAL REPORT SUMMARY")
    logger.info("="*60)
    logger.info("Dataset: %s", dataset_path)
    logger.info("Features used (%d): %s", len(feature_cols), ", ".join(feature_cols))
    logger.info("Chosen K: %d", chosen_k)
    logger.info("Inertia: %.4f", inertia_final)
    logger.info("Silhouette: %.4f", sil_final)
    logger.info("Calinski-Harabasz: %.4f", ch_final)
    logger.info("Davies-Bouldin: %.4f", db_final)
    logger.info("Cluster sizes:")
    for u, c in zip(unique, counts):
        logger.info("  Cluster %d : %d", int(u), int(c))
    logger.info("Outputs (graphs) saved to: %s", graph_output_dir)
    logger.info("Cluster labels saved to: %s", out_labels_path)
    logger.info("Log file: %s", log_file)
    logger.info("="*60)

# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    try:
        logger.info("K-Means pipeline started.")
        run_kmeans_pipeline(dataset_path, graph_output_dir)
        logger.info("K-Means pipeline finished successfully.")
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    except Exception as e:
        logger.exception("Unhandled exception: %s", str(e))
        raise