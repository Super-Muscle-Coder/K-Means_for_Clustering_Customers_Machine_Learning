"""
================================================================================
PHÂN CỤM K-MEANS - PRODUCT+CHANNEL CLUSTERING (DUAL DATASET COMPARISON)
================================================================================

Quy trình thực hiện:
1. Load CẢ 2 dữ liệu đã chuẩn hóa:
   - Customer_Behavior_ProductChannel_Standard_scaled.csv (StandardScaler ONLY)
   - Customer_Behavior_ProductChannel_Robust_scaled.csv (RobustScaler ONLY)
2. Chạy K-Means độc lập trên từng dataset
3. So sánh hiệu quả giữa 2 phương pháp scaling
4. Xuất báo cáo tổng hợp + biểu đồ so sánh

CLUSTERING FEATURES (4):
- Product_HHI (Product diversity - Herfindahl-Hirschman Index)
- Store_Preference (Store channel preference ratio 0-1)
- Web_Engagement (Web visits count)
- PC1_Total_TotalPurchases (PCA composite: Total_Spent_Transformed x TotalPurchases)

REFERENCE FEATURES (8 - for post-hoc interpretation):
- Wine_Preference, Meat_Preference, Fish_Preference, Fruit_Preference, 
  Sweet_Preference, Gold_Preference
- Dominant_Product (categorical)
- Top_Product_Share (numeric)

Các chỉ số đánh giá:
- Inertia: Tổng bình phương khoảng cách đến tâm cụm (càng thấp càng tốt)
- Silhouette Score: Đo độ tách biệt giữa các cụm [-1, 1] (càng cao càng tốt)
- Calinski-Harabasz: Tỷ lệ phương sai giữa các cụm (càng cao càng tốt)
- Davies-Bouldin: Độ tương đồng giữa các cụm (càng thấp càng tốt)
- Gap Statistic: So sánh với phân bố ngẫu nhiên (chọn K khi gap cao nhất)
================================================================================
"""

import os
import sys
import logging
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, silhouette_samples
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
import warnings

warnings.filterwarnings('ignore')

# ================================================================================
# CẤU HÌNH ĐƯỜNG DẪN
# ================================================================================
BASE_DIR = r"C:\Project\Machine_Learning\Machine_Learning"

# Input datasets (CẢ 2 VERSIONS - INDEPENDENT SCALING)
# Dataset 1: StandardScaler ONLY
DATASET_STANDARD_ONLY = os.path.join(BASE_DIR, "dataset", "Customer_Behavior_ProductChannel_Standard_scaled.csv")

# Dataset 2: RobustScaler ONLY (NOT sequential)
DATASET_ROBUST_ONLY = os.path.join(BASE_DIR, "dataset", "Customer_Behavior_ProductChannel_Robust_scaled.csv")

# Original data (for post-hoc analysis)
ENGINEERED_CSV = os.path.join(BASE_DIR, "dataset", "Customer_Behavior_ProductChannel.csv")

# Output directories
GRAPH_BASE_DIR = os.path.join(BASE_DIR, "graph", "Training", "With Library", "ProductChannel_WL")
REPORT_DIR = os.path.join(BASE_DIR, "report", "Training", "With Library")

# Output files
LOG_FILE = os.path.join(REPORT_DIR, "ProductChannel_KMeans.log")
RESULT_CSV_STANDARD_ONLY = os.path.join(REPORT_DIR, "ProductChannel_KMeans_StandardOnly_Results.csv")
RESULT_CSV_ROBUST_ONLY = os.path.join(REPORT_DIR, "ProductChannel_KMeans_RobustOnly_Results.csv")
COMPARISON_CSV = os.path.join(REPORT_DIR, "ProductChannel_KMeans_Comparison_StandardVsRobust.csv")

# ================================================================================
# CLUSTERING & REFERENCE FEATURE CONFIG (AUTO-DETECT)
# ================================================================================

def auto_detect_clustering_features(df):
    """
    Auto-detect clustering features từ dataset.
    Xử lý cả đặc trưng chuẩn và PC1_* từ PCA.
    
    Returns:
        list: Danh sách các clustering features có trong dataset
    """
    clustering_features = []
    
    # Standard features (có thể có hoặc không)
    standard_features = [
        'Product_HHI',
        'Store_Preference',
        'Web_Engagement',
        'Total_Spent_Transformed',
        'TotalPurchases'
    ]
    
    # Thêm standard features có trong dataset
    for feat in standard_features:
        if feat in df.columns:
            clustering_features.append(feat)
    
    # Thêm PCA components (nếu có)
    for col in df.columns:
        if col.startswith('PC1_') and col not in clustering_features:
            clustering_features.append(col)
    
    return clustering_features


def get_reference_features(df):
    """
    Lấy reference features có trong dataset.
    
    Returns:
        list: Danh sách reference features
    """
    expected_reference = [
        'Wine_Preference',
        'Meat_Preference',
        'Fish_Preference',
        'Fruit_Preference',
        'Sweet_Preference',
        'Gold_Preference',
        'Dominant_Product',
        'Top_Product_Share'
    ]
    
    return [feat for feat in expected_reference if feat in df.columns]


# ================================================================================
# THAM SỐ THUẬT TOÁN
# ================================================================================
K_MIN = 2
K_MAX = 6
RANDOM_STATE = 42
N_INIT = 10
MAX_ITER = 300
GAP_B = 10

# ================================================================================
# KHỞI TẠO THƯ MỤC VÀ LOGGING
# ================================================================================
os.makedirs(GRAPH_BASE_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ================================================================================
# HÀM TIỆN ÍCH
# ================================================================================

def print_section_header(title, width=100, char='='):
    """In tiêu đề phần với đường viền"""
    logger.info("\n" + char * width)
    logger.info(title.center(width))
    logger.info(char * width + "\n")

def load_dataset(path, dataset_name):
    """Đọc dữ liệu từ file CSV"""
    logger.info("Dang tai du lieu: %s", dataset_name)
    
    if not os.path.exists(path):
        logger.error("LOI: Tep khong ton tai: %s", path)
        raise FileNotFoundError(f"Dataset not found: {path}")
    
    df = pd.read_csv(path)
    logger.info("Kich thuoc du lieu: %d hang x %d cot", df.shape[0], df.shape[1])
    logger.info("Bo nho: %.2f KB\n", df.memory_usage(deep=True).sum() / 1024)
    return df

def validate_dataframe(df, dataset_name):
    """Kiểm tra chất lượng dữ liệu"""
    print_section_header(f"KIEM TRA CHAT LUONG DU LIEU - {dataset_name.upper()}")
    
    # Kiểm tra giá trị bị thiếu
    missing = df.isnull().sum()
    if missing.sum() == 0:
        logger.info("Khong co gia tri bi thieu")
    else:
        logger.warning("Phat hien gia tri thieu:")
        for col, cnt in missing[missing > 0].items():
            logger.warning("  - %s: %d gia tri (%.2f%%)", col, cnt, cnt/len(df)*100)
    
    # Kiểm tra giá trị vô hạn
    numeric_df = df.select_dtypes(include=[np.number])
    inf_count = np.isinf(numeric_df).sum().sum()
    if inf_count == 0:
        logger.info("Khong co gia tri vo han")
    else:
        logger.warning("Phat hien %d gia tri vo han", inf_count)
    
    # Kiểm tra kiểu dữ liệu
    logger.info("\nKieu du lieu cac cot:")
    for col in df.columns:
        logger.info("  - %-35s: %s", col, df[col].dtype)
    
    # Thống kê mô tả
    logger.info("\nThong ke mo ta (chi cac cot so):")
    logger.info("\n%s\n", numeric_df.describe().to_string())

def save_plot(fig, fname, subdir=""):
    """Lưu biểu đồ"""
    if subdir:
        output_dir = os.path.join(GRAPH_BASE_DIR, subdir)
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, fname)
    else:
        path = os.path.join(GRAPH_BASE_DIR, fname)
    
    fig.savefig(path, bbox_inches='tight', dpi=150)
    logger.info("  Luu bieu do: %s", fname)
    plt.close(fig)

def compute_gap_statistic(X, nrefs=GAP_B, max_k=K_MAX):
    """Tính Gap Statistic để xác định K tối ưu"""
    logger.info("\nDang tinh Gap Statistic voi %d mau tham chieu...", nrefs)
    shape = X.shape
    
    tops = X.max(axis=0)
    bots = X.min(axis=0)
    rands = np.random.RandomState(RANDOM_STATE).rand(nrefs, shape[0], shape[1])
    refs = np.zeros_like(rands)
    for i in range(nrefs):
        refs[i] = rands[i] * (tops - bots) + bots
    
    def compute_wk(data, k):
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=N_INIT, 
                       random_state=RANDOM_STATE, max_iter=MAX_ITER)
        kmeans.fit(data)
        return kmeans.inertia_
    
    gaps = np.zeros(max_k)
    sk = np.zeros(max_k)
    wk_orig = np.zeros(max_k)
    
    for k in range(1, max_k + 1):
        wk_orig[k-1] = compute_wk(X, k)
        
        wk_refs = np.zeros(nrefs)
        for i in range(nrefs):
            wk_refs[i] = compute_wk(refs[i], k)
        
        log_wk_refs = np.log(wk_refs)
        gap = np.mean(log_wk_refs) - np.log(wk_orig[k-1])
        gaps[k-1] = gap
        sk[k-1] = np.sqrt(np.mean((log_wk_refs - np.mean(log_wk_refs))**2)) * np.sqrt(1 + 1.0 / nrefs)
    
    logger.info("Gap Statistic tinh xong\n")
    return gaps, sk

def compute_advanced_metrics(X, labels, centroids):
    """Tính toán các chỉ số đánh giá nâng cao"""
    n_clusters = len(np.unique(labels))
    silhouette_vals = silhouette_samples(X, labels)
    
    cluster_silhouettes = {}
    for i in range(n_clusters):
        cluster_mask = (labels == i)
        cluster_silhouettes[i] = {
            'mean': np.mean(silhouette_vals[cluster_mask]),
            'min': np.min(silhouette_vals[cluster_mask]),
            'max': np.max(silhouette_vals[cluster_mask]),
            'negative_count': np.sum(silhouette_vals[cluster_mask] < 0)
        }
    
    within_cluster_variance = {}
    for i in range(n_clusters):
        cluster_mask = (labels == i)
        cluster_points = X[cluster_mask]
        if len(cluster_points) > 1:
            variance = np.var(cluster_points, axis=0).mean()
            within_cluster_variance[i] = variance
        else:
            within_cluster_variance[i] = 0.0
    
    centroid_distances = squareform(pdist(centroids, metric='euclidean'))
    np.fill_diagonal(centroid_distances, np.inf)
    min_centroid_dist = np.min(centroid_distances) if np.any(centroid_distances != np.inf) else 0
    avg_centroid_dist = np.mean(centroid_distances[centroid_distances != np.inf]) if np.any(centroid_distances != np.inf) else 0
    
    cluster_sizes = np.bincount(labels)
    balance_ratio = cluster_sizes.min() / cluster_sizes.max() if len(cluster_sizes) > 0 else 0
    
    avg_within_var = np.mean(list(within_cluster_variance.values())) if within_cluster_variance else 0
    compactness_separation = avg_centroid_dist / (avg_within_var + 1e-10)
    
    return {
        'cluster_silhouettes': cluster_silhouettes,
        'within_cluster_variance': within_cluster_variance,
        'min_centroid_distance': min_centroid_dist,
        'avg_centroid_distance': avg_centroid_dist,
        'cluster_balance_ratio': balance_ratio,
        'compactness_separation_score': compactness_separation
    }

def vote_for_k(silhouette_scores, ch_scores, db_scores, K_range, gaps=None, sk=None):
    """Chọn K bằng voting từ nhiều metrics"""
    votes = {}
    
    # Silhouette: higher is better
    valid_sil = [(k, s) for k, s in zip(K_range, silhouette_scores) if not np.isnan(s)]
    if valid_sil:
        k_sil = max(valid_sil, key=lambda x: x[1])[0]
        votes['silhouette'] = k_sil
    
    # Calinski-Harabasz: higher is better
    valid_ch = [(k, s) for k, s in zip(K_range, ch_scores) if not np.isnan(s)]
    if valid_ch:
        k_ch = max(valid_ch, key=lambda x: x[1])[0]
        votes['calinski_harabasz'] = k_ch
    
    # Davies-Bouldin: lower is better
    valid_db = [(k, s) for k, s in zip(K_range, db_scores) if not np.isnan(s)]
    if valid_db:
        k_db = min(valid_db, key=lambda x: x[1])[0]
        votes['davies_bouldin'] = k_db
    
    # Gap Statistic
    if gaps is not None and sk is not None:
        for k in range(K_MIN, len(gaps)):
            if gaps[k-1] >= gaps[k] - sk[k]:
                votes['gap_statistic'] = k
                break
    
    # Tính số phiếu cho mỗi K
    vote_counts = {}
    for k in votes.values():
        vote_counts[k] = vote_counts.get(k, 0) + 1
    
    max_votes = max(vote_counts.values()) if vote_counts else 0
    winners = [k for k, cnt in vote_counts.items() if cnt == max_votes]
    
    # Chọn K tốt nhất
    if len(winners) == 1:
        chosen_k = winners[0]
    else:
        # Ưu tiên Silhouette > Calinski-Harabasz
        if 'silhouette' in votes and votes['silhouette'] in winners:
            chosen_k = votes['silhouette']
        elif 'calinski_harabasz' in votes and votes['calinski_harabasz'] in winners:
            chosen_k = votes['calinski_harabasz']
        else:
            chosen_k = min(winners)
    
    logger.info("\nKET QUA VOTING CHON K TOI UU")
    logger.info("-" * 60)
    
    for metric, k in sorted(votes.items()):
        marker = "CHON" if k == chosen_k else "    "
        logger.info("%s %-25s -> K = %d", marker, metric, k)
    
    logger.info("\nK duoc chon theo da so: %d (%d/%d phieu)", chosen_k, vote_counts.get(chosen_k, 0), len(votes))
    
    return chosen_k, votes

# ================================================================================
# HÀM CHẠY K-MEANS CHO 1 DATASET (CORE PIPELINE)
# ================================================================================

def run_kmeans_single_dataset(df_scaled, dataset_name, dataset_type, subdir, df_orig=None):
    """
    Chạy toàn bộ quy trình K-Means cho 1 dataset
    
    Args:
        df_scaled: DataFrame đã scaled
        dataset_name: Tên hiển thị (VD: "Standard Only", "Robust Only")
        dataset_type: Loại dataset (VD: "StandardOnly", "RobustOnly") - dùng cho tên file
        subdir: Thư mục con để lưu biểu đồ
        df_orig: DataFrame gốc (raw data) cho phân tích post-hoc
    
    Returns:
        dict: Kết quả clustering
    """
    print_section_header(f"PHAN TICH DATASET: {dataset_name.upper()}")
    
    # Validate data
    validate_dataframe(df_scaled, dataset_name)
    
    # Auto-detect clustering features
    clustering_features = auto_detect_clustering_features(df_scaled)
    reference_features = get_reference_features(df_scaled)
    
    logger.info("\nAUTO-DETECTED FEATURES:")
    logger.info("Clustering features (%d): %s", len(clustering_features), clustering_features)
    logger.info("Reference features (%d): %s\n", len(reference_features), reference_features)
    
    # Kiểm tra có đủ clustering features không
    if not clustering_features:
        logger.critical("LOI: Khong tim thay bat ky clustering features nao trong dataset")
        logger.critical("Danh sach cac cot: %s", list(df_scaled.columns))
        return None

    X = df_scaled[clustering_features].values
    logger.info("\nSo dac trung su dung cho clustering: %d", len(clustering_features))
    logger.info("Danh sach: %s\n", ", ".join(clustering_features))
    
    # ============================================================================
    # ĐÁNH GIÁ K TỐI ƯU
    # ============================================================================
    print_section_header("DANH GIA K TOI UU")
    
    inertias = []
    silhouette_scores = []
    ch_scores = []
    db_scores = []
    K_range = list(range(K_MIN, K_MAX + 1))
    
    logger.info("Dang danh gia K tu %d den %d...\n", K_MIN, K_MAX)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=N_INIT, 
                       random_state=RANDOM_STATE, max_iter=MAX_ITER)
        labels = kmeans.fit_predict(X)
        inertias.append(kmeans.inertia_)
        
        if k > 1:
            silhouette_scores.append(silhouette_score(X, labels))
            ch_scores.append(calinski_harabasz_score(X, labels))
            db_scores.append(davies_bouldin_score(X, labels))
        else:
            silhouette_scores.append(np.nan)
            ch_scores.append(np.nan)
            db_scores.append(np.nan)
    
    # ============================================================================
    # VẼ ELBOW PLOT
    # ============================================================================
    logger.info("ELBOW METHOD - INERTIA")
    logger.info("%-5s | %-12s | %-12s", "K", "Inertia", "Giam (%)")
    logger.info("-" * 40)
    for i, k in enumerate(K_range):
        if i == 0:
            logger.info("%-5d | %12.4f | %12s", k, inertias[i], "---")
        else:
            decrease_pct = (inertias[i-1] - inertias[i]) / inertias[i-1] * 100
            logger.info("%-5d | %12.4f | %11.2f%%", k, inertias[i], decrease_pct)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(K_range, inertias, 'o-', color='steelblue', linewidth=2, markersize=10, label='Inertia')
    ax.set_xlabel('So cum (K)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Inertia (Sum of Squared Distances)', fontsize=13, fontweight='bold')
    ax.set_title(f'Phuong phap Elbow - {dataset_name}\n(K toi uu o diem gap)', fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xticks(K_range)
    ax.legend(fontsize=11)
    plt.tight_layout()
    save_plot(fig, f"elbow_plot_{dataset_type.lower()}.png", subdir)
    
    # ============================================================================
    # VẼ SILHOUETTE PLOT
    # ============================================================================
    logger.info("\nSILHOUETTE SCORE")
    logger.info("%-5s | %-12s | %-20s", "K", "Silhouette", "Danh gia")
    logger.info("-" * 50)
    for k, score in zip(K_range, silhouette_scores):
        if np.isnan(score):
            evaluation = "Khong ap dung"
        elif score > 0.7:
            evaluation = "Xuat sac"
        elif score > 0.5:
            evaluation = "Tot"
        elif score > 0.3:
            evaluation = "Trung binh"
        elif score > 0.2:
            evaluation = "Yeu"
        else:
            evaluation = "Kem"
        logger.info("%-5d | %12.4f | %-20s", k, score if not np.isnan(score) else 0, evaluation)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    valid_k = [k for k, s in zip(K_range, silhouette_scores) if not np.isnan(s)]
    valid_scores = [s for s in silhouette_scores if not np.isnan(s)]
    if valid_k:
        ax.plot(valid_k, valid_scores, 'o-', color='seagreen', linewidth=2, markersize=10, label='Silhouette Score')
        ax.fill_between(valid_k, valid_scores, alpha=0.3, color='seagreen')
    ax.set_xlabel('So cum (K)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Silhouette Score', fontsize=13, fontweight='bold')
    ax.set_title(f'Diem Silhouette - {dataset_name}\n(Cao hon = Tot hon)', fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xticks(K_range)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.6, linewidth=2, label='Nguong tot (0.5)')
    ax.set_ylim([min(valid_scores) - 0.1 if valid_scores else 0, 1.0])
    ax.legend(fontsize=11)
    plt.tight_layout()
    save_plot(fig, f"silhouette_score_{dataset_type.lower()}.png", subdir)
    
    # ============================================================================
    # VẼ OTHER METRICS
    # ============================================================================
    logger.info("\nCALINSKI-HARABASZ VA DAVIES-BOULDIN")
    logger.info("%-5s | %-18s | %-18s", "K", "Calinski-Harabasz", "Davies-Bouldin")
    logger.info("-" * 50)
    for k, ch, db in zip(K_range, ch_scores, db_scores):
        if np.isnan(ch):
            logger.info("%-5d | %18s | %18s", k, "---", "---")
        else:
            logger.info("%-5d | %18.4f | %18.4f", k, ch, db)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Calinski-Harabasz (higher is better)
    valid_k = [k for k, ch in zip(K_range, ch_scores) if not np.isnan(ch)]
    valid_ch = [ch for ch in ch_scores if not np.isnan(ch)]
    if valid_k:
        ax1.plot(valid_k, valid_ch, 'o-', color='darkorange', linewidth=2, markersize=10, label='Calinski-Harabasz')
        ax1.fill_between(valid_k, valid_ch, alpha=0.3, color='darkorange')
    ax1.set_xlabel('So cum (K)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Calinski-Harabasz Score', fontsize=13, fontweight='bold')
    ax1.set_title(f'Diem Calinski-Harabasz - {dataset_name}\n(Cao hon = Tot hon)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xticks(K_range)
    ax1.legend(fontsize=11)
    
    # Davies-Bouldin (lower is better)
    valid_db = [db for db in db_scores if not np.isnan(db)]
    if valid_k:
        ax2.plot(valid_k, valid_db, 'o-', color='crimson', linewidth=2, markersize=10, label='Davies-Bouldin')
        ax2.fill_between(valid_k, valid_db, alpha=0.3, color='crimson')
    ax2.set_xlabel('So cum (K)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Davies-Bouldin Score', fontsize=13, fontweight='bold')
    ax2.set_title(f'Diem Davies-Bouldin - {dataset_name}\n(Thap hon = Tot hon)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xticks(K_range)
    ax2.legend(fontsize=11)
    
    plt.tight_layout()
    save_plot(fig, f"other_metrics_{dataset_type.lower()}.png", subdir)
    
    # ============================================================================
    # GAP STATISTIC
    # ============================================================================
    gaps = None
    sk_vals = None
    k_optimal_gap = None
    
    try:
        gaps, sk_vals = compute_gap_statistic(X, nrefs=GAP_B, max_k=K_MAX)
        
        logger.info("GAP STATISTIC")
        logger.info("%-5s | %-12s | %-12s | %-15s", "K", "Gap", "Std Error", "Gap - SE")
        logger.info("-" * 60)
        for k in range(1, len(gaps) + 1):
            gap_minus_se = gaps[k-1] - sk_vals[k-1]
            logger.info("%-5d | %12.4f | %12.4f | %15.4f", k, gaps[k-1], sk_vals[k-1], gap_minus_se)
        
        for k in range(1, len(gaps)):
            if gaps[k-1] >= gaps[k] - sk_vals[k]:
                k_optimal_gap = k
                break
        
        if k_optimal_gap:
            logger.info("\nK toi uu theo Gap Statistic: %d", k_optimal_gap)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ks = list(range(1, len(gaps) + 1))
        ax.plot(ks, gaps, 'o-', color='purple', linewidth=2, markersize=10, label='Gap Statistic')
        ax.fill_between(ks, gaps - sk_vals, gaps + sk_vals, color='lavender', alpha=0.4, label='Gap +/- SE')
        if k_optimal_gap:
            ax.axvline(x=k_optimal_gap, color='red', linestyle='--', linewidth=2.5, label=f'K toi uu = {k_optimal_gap}')
        ax.set_xlabel('So cum (K)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Gap Statistic', fontsize=13, fontweight='bold')
        ax.set_title(f'Gap Statistic - {dataset_name}\n(Chon K khi gap giam manh)', fontsize=15, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=11)
        plt.tight_layout()
        save_plot(fig, f"gap_statistic_{dataset_type.lower()}.png", subdir)
        
    except Exception as e:
        logger.warning("Khong the tinh Gap Statistic: %s", str(e))
    
    # ============================================================================
    # CHỌN K TỐI ƯU
    # ============================================================================
    print_section_header("QUYET DINH K TOI UU")
    
    chosen_k, votes = vote_for_k(silhouette_scores, ch_scores, db_scores, K_range, gaps, sk_vals)
    logger.info("\nK CUOI CUNG DUOC CHON CHO %s: %d\n", dataset_name.upper(), chosen_k)
    
    # ============================================================================
    # CHẠY K-MEANS CUỐI CÙNG
    # ============================================================================
    print_section_header("CHAY K-MEANS CUOI CUNG")
    
    logger.info("Dang chay K-Means voi K=%d (50 lan khoi tao)...", chosen_k)
    kmeans_final = KMeans(n_clusters=chosen_k, init='k-means++', n_init=50, 
                         random_state=RANDOM_STATE, max_iter=MAX_ITER)
    labels_final = kmeans_final.fit_predict(X)
    centroids = kmeans_final.cluster_centers_
    
    inertia_final = kmeans_final.inertia_
    sil_final = silhouette_score(X, labels_final) if chosen_k > 1 else float('nan')
    ch_final = calinski_harabasz_score(X, labels_final) if chosen_k > 1 else float('nan')
    db_final = davies_bouldin_score(X, labels_final) if chosen_k > 1 else float('nan')
    
    logger.info("\nCac chi so co ban:")
    logger.info("  Inertia:             %10.4f", inertia_final)
    logger.info("  Silhouette:          %10.4f", sil_final if not np.isnan(sil_final) else 0)
    logger.info("  Calinski-Harabasz:   %10.4f", ch_final if not np.isnan(ch_final) else 0)
    logger.info("  Davies-Bouldin:      %10.4f", db_final if not np.isnan(db_final) else 0)
    
    # Tính toán advanced metrics
    logger.info("\nDang tinh toan cac chi so nang cao...")
    advanced_metrics = compute_advanced_metrics(X, labels_final, centroids)
    
    logger.info("\nCHI SO DANH GIA NANG CAO:")
    logger.info("1. SILHOUETTE THEO TUNG CLUSTER:")
    logger.info("   %-10s | %-10s | %-15s | %-15s", "Cluster", "Trung binh", "Khoang", "So diem am")
    logger.info("   " + "-" * 65)
    
    for cluster_id in sorted(advanced_metrics['cluster_silhouettes'].keys()):
        stats_data = advanced_metrics['cluster_silhouettes'][cluster_id]
        logger.info("   %-10d | %10.4f | [%6.4f, %6.4f] | %15d", 
                   cluster_id, stats_data['mean'], stats_data['min'], stats_data['max'], stats_data['negative_count'])
        if stats_data['mean'] < 0.2:
            logger.warning("   Canh bao: Cluster %d co chat luong thap (silhouette < 0.2)", cluster_id)
    
    logger.info("\n2. PHUONG SAI TRONG CLUSTER:")
    for cluster_id in sorted(advanced_metrics['within_cluster_variance'].keys()):
        variance = advanced_metrics['within_cluster_variance'][cluster_id]
        logger.info("   Cluster %d: %.4f", cluster_id, variance)
    
    logger.info("\n3. KHOANG CACH GIUA CAC TAM CLUSTER:")
    logger.info("   Khoang cach toi thieu: %.4f", advanced_metrics['min_centroid_distance'])
    logger.info("   Khoang cach trung binh: %.4f", advanced_metrics['avg_centroid_distance'])
    
    logger.info("\n4. TY LE CAN BANG CLUSTER:")
    logger.info("   Ty le: %.4f", advanced_metrics['cluster_balance_ratio'])
    if advanced_metrics['cluster_balance_ratio'] < 0.3:
        logger.warning("   Canh bao: Cac cluster bi mat can bang")
    
    logger.info("\n5. DIEM COMPACTNESS-SEPARATION:")
    logger.info("   Diem so: %.4f", advanced_metrics['compactness_separation_score'])
    
    # ============================================================================
    # SILHOUETTE ANALYSIS
    # ============================================================================
    fig, ax = plt.subplots(figsize=(12, 8))
    silhouette_vals = silhouette_samples(X, labels_final)
    silhouette_avg = silhouette_score(X, labels_final)
    
    y_lower = 10
    for i in range(chosen_k):
        cluster_silhouette_vals = silhouette_vals[labels_final == i]
        cluster_silhouette_vals.sort()
        
        size_cluster_i = cluster_silhouette_vals.shape[0]
        if size_cluster_i == 0:
            continue
        y_upper = y_lower + size_cluster_i
        
        color = plt.cm.nipy_spectral(float(i) / chosen_k)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_vals,
                         facecolor=color, edgecolor=color, alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, f'Cluster {i}')
        y_lower = y_upper + 10
    
    ax.set_title(f'Phan tich Silhouette - {dataset_name} (K={chosen_k})', fontsize=15, fontweight='bold')
    ax.set_xlabel('He so Silhouette', fontsize=13, fontweight='bold')
    ax.set_ylabel('Cluster', fontsize=13, fontweight='bold')
    ax.axvline(x=silhouette_avg, color="red", linestyle="--", linewidth=2.5, 
               label=f'Trung binh: {silhouette_avg:.3f}')
    ax.set_yticks([])
    ax.set_xlim([-0.3, 1])
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3, axis='x', linestyle='--')
    plt.tight_layout()
    save_plot(fig, f"silhouette_analysis_{dataset_type.lower()}.png", subdir)
    
    # ============================================================================
    # CLUSTER QUALITY SUMMARY
    # ============================================================================
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    cluster_ids = sorted(list(advanced_metrics['cluster_silhouettes'].keys()))
    
    # 1. Silhouette Means
    silhouette_means = [advanced_metrics['cluster_silhouettes'][i]['mean'] for i in cluster_ids]
    bars1 = axes[0, 0].bar(cluster_ids, silhouette_means, color='steelblue', alpha=0.7, edgecolor='black', linewidth=1.5)
    axes[0, 0].axhline(y=0.2, color='red', linestyle='--', linewidth=2, label='Nguong yeu (0.2)')
    axes[0, 0].set_title('Silhouette Score theo Cluster', fontsize=13, fontweight='bold')
    axes[0, 0].set_xlabel('Cluster', fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel('Silhouette trung binh', fontsize=11, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3, axis='y', linestyle='--')
    axes[0, 0].set_xticks(cluster_ids)
    for i, (cid, val) in enumerate(zip(cluster_ids, silhouette_means)):
        axes[0, 0].text(cid, val + 0.02, f'{val:.3f}', ha='center', fontsize=9, fontweight='bold')
    
    # 2. Within-Cluster Variance
    variances = [advanced_metrics['within_cluster_variance'][i] for i in cluster_ids]
    bars2 = axes[0, 1].bar(cluster_ids, variances, color='coral', alpha=0.7, edgecolor='black', linewidth=1.5)
    axes[0, 1].set_title('Phuong sai trong Cluster', fontsize=13, fontweight='bold')
    axes[0, 1].set_xlabel('Cluster', fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel('Phuong sai trung binh', fontsize=11, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y', linestyle='--')
    axes[0, 1].set_xticks(cluster_ids)
    for cid, val in zip(cluster_ids, variances):
        axes[0, 1].text(cid, val + max(variances)*0.02, f'{val:.3f}', ha='center', fontsize=9, fontweight='bold')
    
    # 3. Misclassified Points
    neg_counts = [advanced_metrics['cluster_silhouettes'][i]['negative_count'] for i in cluster_ids]
    bars3 = axes[1, 0].bar(cluster_ids, neg_counts, color='indianred', alpha=0.7, edgecolor='black', linewidth=1.5)
    axes[1, 0].set_title('So diem phan loai sai (Silhouette < 0)', fontsize=13, fontweight='bold')
    axes[1, 0].set_xlabel('Cluster', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('So luong', fontsize=11, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y', linestyle='--')
    axes[1, 0].set_xticks(cluster_ids)
    for cid, val in zip(cluster_ids, neg_counts):
        axes[1, 0].text(cid, val + max(neg_counts)*0.02, str(int(val)), ha='center', fontsize=9, fontweight='bold')
    
    # 4. Summary Metrics
    summary_labels = ['Ty le\ncan bang', 'KC tam\ntoi thieu', 'Diem C-S\n(x10)']
    summary_values = [
        advanced_metrics['cluster_balance_ratio'],
        advanced_metrics['min_centroid_distance'],
        advanced_metrics['compactness_separation_score'] / 10
    ]
    colors_summary = ['#2ecc71', '#9b59b6', '#e67e22']
    bars4 = axes[1, 1].bar(summary_labels, summary_values, color=colors_summary, alpha=0.7, edgecolor='black', linewidth=1.5)
    axes[1, 1].set_title('Cac chi so tong hop', fontsize=13, fontweight='bold')
    axes[1, 1].set_ylabel('Gia tri', fontsize=11, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y', linestyle='--')
    for val, bar in zip(summary_values, bars4):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + max(summary_values)*0.02,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.suptitle(f'Tom tat chat luong Cluster - {dataset_name}', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    save_plot(fig, f"cluster_quality_summary_{dataset_type.lower()}.png", subdir)
    
    # ============================================================================
    # CLUSTER SIZES
    # ============================================================================
    unique, counts = np.unique(labels_final, return_counts=True)
    logger.info("\nPHAN BO CAC CLUSTER:")
    logger.info("%-10s | %-10s | %-10s", "Cluster", "So luong", "Ty le (%)")
    logger.info("-" * 35)
    for u, c in zip(unique, counts):
        logger.info("%-10d | %-10d | %9.2f%%", int(u), int(c), (c/len(X)*100))
    
    total_misclassified = sum(stats_item['negative_count'] 
                              for stats_item in advanced_metrics['cluster_silhouettes'].values())
    logger.info("\nSo diem bi phan loai sai (silhouette < 0): %d (%.2f%%)", 
               total_misclassified, (total_misclassified / len(X)) * 100)
    
    # ============================================================================
    # CENTROIDS ANALYSIS
    # ============================================================================
    logger.info("\nPHAN TICH TAM CLUSTER (CENTROIDS):")
    centroids_df = pd.DataFrame(centroids, columns=clustering_features)
    logger.info("\n%s", centroids_df.to_string())
    
    # Heatmap Centroids
    heat_df = centroids_df.transpose()
    fig = plt.figure(figsize=(max(10, chosen_k * 1.5), max(6, len(clustering_features) * 0.8)))
    sns.heatmap(heat_df, cmap='RdYlGn', annot=True, fmt=".3f", 
                cbar_kws={'label': 'Gia tri chuan hoa'}, linewidths=2, linecolor='black')
    plt.xlabel('Cluster', fontsize=13, fontweight='bold')
    plt.ylabel('Dac trung', fontsize=13, fontweight='bold')
    plt.title(f'Ban do nhiet: Gia tri tam cluster - {dataset_name}', fontsize=15, fontweight='bold')
    plt.tight_layout()
    save_plot(fig, f"centroid_heatmap_{dataset_type.lower()}.png", subdir)
    
    # Line plot Centroids
    fig, ax = plt.subplots(figsize=(12, 6))
    for i in range(centroids.shape[0]):
        ax.plot(clustering_features, centroids[i], marker='o', linewidth=2.5, markersize=10, 
               label=f'Cluster {i}', alpha=0.7)
    ax.set_xticks(range(len(clustering_features)))
    ax.set_xticklabels(clustering_features, fontsize=11, fontweight='bold')
    ax.set_xlabel('Dac trung', fontsize=13, fontweight='bold')
    ax.set_ylabel('Gia tri tam cluster (chuan hoa)', fontsize=13, fontweight='bold')
    ax.set_title(f'Bieu do gia tri tam cluster - {dataset_name}', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    save_plot(fig, f"centroid_plot_{dataset_type.lower()}.png", subdir)
    
    # ============================================================================
    # PCA VISUALIZATION
    # ============================================================================
    logger.info("\nDang thuc hien PCA de truc quan hoa...")
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X)
    
    explained_var = pca.explained_variance_ratio_
    logger.info("Phuong sai giai thich boi 2 PC dau tien: %.2f%%", sum(explained_var) * 100)
    logger.info("  PC1: %.2f%%", explained_var[0] * 100)
    logger.info("  PC2: %.2f%%", explained_var[1] * 100)
    
    df_vis = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'cluster': labels_final
    })
    
    fig, ax = plt.subplots(figsize=(12, 8))
    palette = sns.color_palette("husl", n_colors=chosen_k)
    sns.scatterplot(data=df_vis, x='PC1', y='PC2', hue='cluster', palette=palette, 
                   s=80, alpha=0.6, edgecolor='black', linewidth=0.7, ax=ax)
    ax.set_xlabel(f'PC1 ({explained_var[0]*100:.1f}%)', fontsize=13, fontweight='bold')
    ax.set_ylabel(f'PC2 ({explained_var[1]*100:.1f}%)', fontsize=13, fontweight='bold')
    ax.set_title(f'Truc quan hoa Cluster qua PCA - {dataset_name}', fontsize=15, fontweight='bold')
    ax.legend(title='Cluster', fontsize=11, title_fontsize=12, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    save_plot(fig, f"pca_scatter_{dataset_type.lower()}.png", subdir)
    
    logger.info("PCA visualization chart saved\n")
    
    # ============================================================================
    # PAIRPLOT
    # ============================================================================
    try:
        logger.info("Dang tao pairplot...")
        sns.set(style="ticks")
        pair_df = df_scaled[clustering_features].copy()
        pair_df['cluster'] = labels_final
        g = sns.pairplot(pair_df, hue='cluster', palette=palette, 
                        diag_kind='kde', plot_kws={'s': 40, 'alpha': 0.6})
        g.fig.suptitle(f'Ma tran phan tan - {dataset_name}', y=1.002, fontsize=15, fontweight='bold')
        fname = os.path.join(GRAPH_BASE_DIR, subdir, f"pairplot_{dataset_type.lower()}.png")
        g.fig.savefig(fname, bbox_inches='tight', dpi=150)
        logger.info("Pairplot chart saved\n")
        plt.close('all')
    except Exception as e:
        logger.warning("Khong the tao pairplot: %s\n", str(e))
    
    # ============================================================================
    # PHÂN TÍCH RAW DATA (NẾU CÓ)
    # ============================================================================
    if df_orig is not None:
        print_section_header("PHAN TICH DU LIEU GOC (RAW DATA)")
        
        df = df_orig.copy()
        df['cluster'] = labels_final
        
        logger.info("1. THONG KE CAC DAC TRUNG CLUSTERING (GIA TRI GOC):")
        logger.info("%-10s | %-8s | %-18s | %-15s | %-12s | %-15s | %-15s",
                   "Cluster", "N", "Product_HHI", "Store_Pref", "Web_Engagement", "PC1", "Count")
        logger.info("-" * 115)
        
        for cluster_id in sorted(df['cluster'].unique()):
            sub = df[df['cluster'] == cluster_id]
            pc1_mean = sub['PC1_Total_TotalPurchases'].mean() if 'PC1_Total_TotalPurchases' in sub.columns else 0
            logger.info("%-10d | %8d | %18.4f | %15.4f | %12.2f | %15.4f | %15d",
                       cluster_id, len(sub),
                       sub['Product_HHI'].mean() if 'Product_HHI' in sub.columns else 0,
                       sub['Store_Preference'].mean() if 'Store_Preference' in sub.columns else 0,
                       sub['Web_Engagement'].mean() if 'Web_Engagement' in sub.columns else 0,
                       pc1_mean,
                       len(sub))
        
        logger.info("\nRaw data analysis complete\n")
        
        # Product Preferences analysis
        if all(f in df.columns for f in ['Wine_Preference', 'Meat_Preference', 'Fish_Preference', 
                                          'Fruit_Preference', 'Sweet_Preference', 'Gold_Preference']):
            logger.info("2. PHAN BO PRODUCT PREFERENCES THEO CLUSTER (%):")
            logger.info("%-10s | %-8s | %-8s | %-8s | %-8s | %-8s | %-8s",
                       "Cluster", "Wine", "Meat", "Fish", "Fruit", "Sweet", "Gold")
            logger.info("-" * 70)
            
            for cluster_id in sorted(df['cluster'].unique()):
                sub = df[df['cluster'] == cluster_id]
                logger.info("%-10d | %7.2f%% | %7.2f%% | %7.2f%% | %7.2f%% | %7.2f%% | %7.2f%%",
                           cluster_id,
                           sub['Wine_Preference'].mean() * 100,
                           sub['Meat_Preference'].mean() * 100,
                           sub['Fish_Preference'].mean() * 100,
                           sub['Fruit_Preference'].mean() * 100,
                           sub['Sweet_Preference'].mean() * 100,
                           sub['Gold_Preference'].mean() * 100)
            
            # Heatmap
            try:
                product_features = ['Wine_Preference', 'Meat_Preference', 'Fish_Preference',
                                  'Fruit_Preference', 'Sweet_Preference', 'Gold_Preference']
                
                cluster_means = []
                for cluster_id in sorted(df['cluster'].unique()):
                    sub = df[df['cluster'] == cluster_id]
                    means = [sub[feat].mean() for feat in product_features]
                    cluster_means.append(means)
                
                product_labels = [f.replace('_Preference', '') for f in product_features]
                
                fig, ax = plt.subplots(figsize=(10, max(6, chosen_k * 0.8)))
                sns.heatmap(cluster_means, annot=True, fmt='.3f', cmap='YlOrRd',
                           xticklabels=product_labels,
                           yticklabels=[f'Cluster {i}' for i in sorted(df['cluster'].unique())],
                           ax=ax, cbar_kws={'label': 'Preference (ratio)'})
                
                ax.set_title(f'Product Preferences Heatmap - {dataset_name}',
                            fontsize=15, fontweight='bold')
                ax.set_xlabel('Product Category', fontsize=13, fontweight='bold')
                ax.set_ylabel('Cluster', fontsize=13, fontweight='bold')
                
                plt.tight_layout()
                save_plot(fig, f"product_heatmap_{dataset_type.lower()}.png", subdir)
                
            except Exception as e:
                logger.warning("Khong the tao heatmap: %s", str(e))
        
        # Channel Behavior scatter
        if all(f in df.columns for f in ['Store_Preference', 'Web_Engagement']):
            try:
                fig, ax = plt.subplots(figsize=(12, 8))
                
                sns.scatterplot(data=df, x='Store_Preference', y='Web_Engagement',
                               hue='cluster', palette=palette, s=80, ax=ax,
                               alpha=0.6, edgecolor='k', linewidth=0.5)
                
                ax.set_xlabel('Store Preference (ratio)', fontsize=13, fontweight='bold')
                ax.set_ylabel('Web Engagement (visits)', fontsize=13, fontweight='bold')
                ax.set_title(f'Shopping Channel Behavior - {dataset_name}',
                            fontsize=15, fontweight='bold')
                ax.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
                ax.grid(True, alpha=0.3, linestyle='--')
                
                plt.tight_layout()
                save_plot(fig, f"channel_behavior_{dataset_type.lower()}.png", subdir)
                
            except Exception as e:
                logger.warning("Khong the tao scatter plot: %s", str(e))
    
    # ============================================================================
    # TỔNG HỢP KẾT QUẢ
    # ============================================================================
    logger.info("Hoan tat phan tich dataset: %s\n", dataset_name)
    
    return {
        'dataset_name': dataset_name,
        'dataset_type': dataset_type,
        'chosen_k': int(chosen_k),
        'inertia': float(inertia_final),
        'silhouette': float(sil_final) if not np.isnan(sil_final) else None,
        'calinski_harabasz': float(ch_final) if not np.isnan(ch_final) else None,
        'davies_bouldin': float(db_final) if not np.isnan(db_final) else None,
        'cluster_balance_ratio': float(advanced_metrics['cluster_balance_ratio']),
        'compactness_separation_score': float(advanced_metrics['compactness_separation_score']),
        'misclassified_count': int(total_misclassified),
        'misclassified_percent': float(total_misclassified / len(X) * 100),
        'cluster_sizes': counts.tolist(),
        'total_points': int(len(X)),
        'clustering_features_used': clustering_features
    }

# ================================================================================
# HÀM SO SÁNH 2 DATASETS
# ================================================================================

def compare_results(result_standard_only, result_robust_only):
    """So sánh kết quả giữa 2 datasets và chọn dataset tốt nhất"""
    print_section_header("SO SANH KET QUA GIUA 2 DATASETS")
    
    # Tạo bảng so sánh
    comparison_data = {
        'Metric': [
            'K toi uu',
            'Inertia',
            'Silhouette Score',
            'Calinski-Harabasz',
            'Davies-Bouldin',
            'Cluster Balance Ratio',
            'Compactness-Separation',
            'Misclassified (%)'
        ],
        'StandardOnly': [
            result_standard_only['chosen_k'],
            f"{result_standard_only['inertia']:.4f}",
            f"{result_standard_only['silhouette']:.4f}" if result_standard_only['silhouette'] else "N/A",
            f"{result_standard_only['calinski_harabasz']:.4f}" if result_standard_only['calinski_harabasz'] else "N/A",
            f"{result_standard_only['davies_bouldin']:.4f}" if result_standard_only['davies_bouldin'] else "N/A",
            f"{result_standard_only['cluster_balance_ratio']:.4f}",
            f"{result_standard_only['compactness_separation_score']:.4f}",
            f"{result_standard_only['misclassified_percent']:.2f}%"
        ],
        'RobustOnly': [
            result_robust_only['chosen_k'],
            f"{result_robust_only['inertia']:.4f}",
            f"{result_robust_only['silhouette']:.4f}" if result_robust_only['silhouette'] else "N/A",
            f"{result_robust_only['calinski_harabasz']:.4f}" if result_robust_only['calinski_harabasz'] else "N/A",
            f"{result_robust_only['davies_bouldin']:.4f}" if result_robust_only['davies_bouldin'] else "N/A",
            f"{result_robust_only['cluster_balance_ratio']:.4f}",
            f"{result_robust_only['compactness_separation_score']:.4f}",
            f"{result_robust_only['misclassified_percent']:.2f}%"
        ]
    }
    
    df_comparison = pd.DataFrame(comparison_data)
    logger.info("\n%s\n", df_comparison.to_string(index=False))
    
    # Voting để chọn dataset tốt nhất
    votes = {'StandardOnly': 0, 'RobustOnly': 0}
    
    logger.info("KET QUA VOTING CHON DATASET TOT NHAT")
    logger.info("-" * 80)
        
    # 1. Silhouette (higher is better)
    if result_standard_only['silhouette'] and result_robust_only['silhouette']:
        if result_standard_only['silhouette'] > result_robust_only['silhouette']:
            votes['StandardOnly'] += 1
            logger.info("Silhouette: StandardOnly THANG (%.4f vs %.4f)", 
                       result_standard_only['silhouette'], result_robust_only['silhouette'])
        else:
            votes['RobustOnly'] += 1
            logger.info("Silhouette: RobustOnly THANG (%.4f vs %.4f)", 
                       result_robust_only['silhouette'], result_standard_only['silhouette'])
    
    # 2. Calinski-Harabasz (higher is better)
    if result_standard_only['calinski_harabasz'] and result_robust_only['calinski_harabasz']:
        if result_standard_only['calinski_harabasz'] > result_robust_only['calinski_harabasz']:
            votes['StandardOnly'] += 1
            logger.info("Calinski-Harabasz: StandardOnly THANG (%.4f vs %.4f)", 
                       result_standard_only['calinski_harabasz'], result_robust_only['calinski_harabasz'])
        else:
            votes['RobustOnly'] += 1
            logger.info("Calinski-Harabasz: RobustOnly THANG (%.4f vs %.4f)", 
                       result_robust_only['calinski_harabasz'], result_standard_only['calinski_harabasz'])
    
    # 3. Davies-Bouldin (lower is better)
    if result_standard_only['davies_bouldin'] and result_robust_only['davies_bouldin']:
        if result_standard_only['davies_bouldin'] < result_robust_only['davies_bouldin']:
            votes['StandardOnly'] += 1
            logger.info("Davies-Bouldin: StandardOnly THANG (%.4f vs %.4f)", 
                       result_standard_only['davies_bouldin'], result_robust_only['davies_bouldin'])
        else:
            votes['RobustOnly'] += 1
            logger.info("Davies-Bouldin: RobustOnly THANG (%.4f vs %.4f)", 
                       result_robust_only['davies_bouldin'], result_standard_only['davies_bouldin'])
    
    # 4. Cluster Balance (higher is better)
    if result_standard_only['cluster_balance_ratio'] > result_robust_only['cluster_balance_ratio']:
        votes['StandardOnly'] += 1
        logger.info("Cluster Balance: StandardOnly THANG (%.4f vs %.4f)", 
                   result_standard_only['cluster_balance_ratio'], result_robust_only['cluster_balance_ratio'])
    else:
        votes['RobustOnly'] += 1
        logger.info("Cluster Balance: RobustOnly THANG (%.4f vs %.4f)", 
                   result_robust_only['cluster_balance_ratio'], result_standard_only['cluster_balance_ratio'])
    
    # 5. Misclassified (lower is better)
    if result_standard_only['misclassified_percent'] < result_robust_only['misclassified_percent']:
        votes['StandardOnly'] += 1
        logger.info("Misclassified: StandardOnly THANG (%.2f%% vs %.2f%%)", 
                   result_standard_only['misclassified_percent'], result_robust_only['misclassified_percent'])
    else:
        votes['RobustOnly'] += 1
        logger.info("Misclassified: RobustOnly THANG (%.2f%% vs %.2f%%)", 
                   result_robust_only['misclassified_percent'], result_standard_only['misclassified_percent'])
    
    # Xác định winner
    winner = max(votes, key=votes.get)
    
    logger.info("\n" + "=" * 80)
    logger.info("KET QUA VOTING".center(80))
    logger.info("=" * 80)
    logger.info("\nStandardOnly: %d phieu", votes['StandardOnly'])
    logger.info("RobustOnly: %d phieu", votes['RobustOnly'])
    logger.info("\nDATASET TOT NHAT: %s\n", winner.upper())
    logger.info("=" * 80 + "\n")
    
    # Luu comparison CSV
    df_comparison.to_csv(COMPARISON_CSV, index=False)
    logger.info("Da luu bang so sanh: %s\n", COMPARISON_CSV)
    
    return winner, df_comparison

# ================================================================================
# PIPELINE CHINH
# ================================================================================

def run_dual_kmeans_pipeline():
    """Chay toan bo quy trinh K-Means cho CA 2 datasets"""
    
    print_section_header("K-MEANS CLUSTERING - PRODUCT+CHANNEL (DUAL DATASET COMPARISON)")
    logger.info("Bat dau quy trinh phan cum...")
    logger.info("Thoi gian: %s\n", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    # ============================================================================
    # LOAD DU LIEU
    # ============================================================================
    print_section_header("LOAD DU LIEU")
    
    logger.info("Dang tai 2 datasets...\n")
    df_standard_only = load_dataset(DATASET_STANDARD_ONLY, "StandardOnly")
    df_robust_only = load_dataset(DATASET_ROBUST_ONLY, "RobustOnly")
    
    # Load du lieu goc (cho phan tich post-hoc)
    df_orig = None
    if os.path.exists(ENGINEERED_CSV):
        try:
            df_orig = pd.read_csv(ENGINEERED_CSV)
            logger.info("Da tai du lieu goc cho phan tich post-hoc: %s\n", ENGINEERED_CSV)
        except Exception as e:
            logger.warning("Khong the tai du lieu goc: %s\n", str(e))
    
    # ============================================================================
    # CHAY K-MEANS CHO TUNG DATASET
    # ============================================================================
    result_standard_only = run_kmeans_single_dataset(
        df_standard_only, 
        "Standard Only", 
        "StandardOnly",
        "Standard_Robust",
        df_orig
    )
    
    # KIEM SOAT LOI: Neu StandardOnly that bai, dung ngay
    if result_standard_only is None:
        logger.critical("DUNG LAI: StandardOnly dataset bi loi - khong co cac clustering features can thiet")
        logger.critical("Kiem tra file: %s", DATASET_STANDARD_ONLY)
        logger.critical("Cac feature can co: %s", str(auto_detect_clustering_features(df_standard_only)))
        raise ValueError("StandardOnly dataset processing failed")
    
    result_robust_only = run_kmeans_single_dataset(
        df_robust_only, 
        "Robust Only", 
        "RobustOnly",
        "Standard_Robust",
        df_orig
    )
    
    # KIEM SOAT LOI: Neu RobustOnly that bai, dung ngay
    if result_robust_only is None:
        logger.critical("DUNG LAI: RobustOnly dataset bi loi - khong co cac clustering features can thiet")
        logger.critical("Kiem tra file: %s", DATASET_ROBUST_ONLY)
        logger.critical("Cac feature can co: %s", str(auto_detect_clustering_features(df_robust_only)))
        raise ValueError("RobustOnly dataset processing failed")
    
    # ============================================================================
    # SO SANH KET QUA
    # ============================================================================
    winner, df_comparison = compare_results(result_standard_only, result_robust_only)
    
    # ============================================================================
    # LUU KET QUA
    # ============================================================================
    print_section_header("LUU KET QUA")
    
    # Luu ket qua rieng cho tung dataset
    df_result_standard_only = pd.DataFrame([result_standard_only])
    df_result_standard_only.to_csv(RESULT_CSV_STANDARD_ONLY, index=False)
    logger.info("Da luu ket qua StandardOnly: %s", RESULT_CSV_STANDARD_ONLY)
    
    df_result_robust_only = pd.DataFrame([result_robust_only])
    df_result_robust_only.to_csv(RESULT_CSV_ROBUST_ONLY, index=False)
    logger.info("Da luu ket qua RobustOnly: %s\n", RESULT_CSV_ROBUST_ONLY)
    
    # ============================================================================
    # TONG KET
    # ============================================================================
    print_section_header("TONG KET KET QUA")
    
    logger.info("Dataset tot nhat: %s", winner.upper())
    logger.info("\nDuong dan ket qua:")
    logger.info("  - Bieu do chung: %s", os.path.join(GRAPH_BASE_DIR, "Standard_Robust"))
    logger.info("  - Bao cao StandardOnly: %s", RESULT_CSV_STANDARD_ONLY)
    logger.info("  - Bao cao RobustOnly: %s", RESULT_CSV_ROBUST_ONLY)
    logger.info("  - Bang so sanh: %s", COMPARISON_CSV)
    logger.info("  - Log file: %s", LOG_FILE)
    logger.info("\n")

# ================================================================================
# MAIN ENTRY POINT
# ================================================================================

if __name__ == "__main__":
    try:
        run_dual_kmeans_pipeline()
        
        print_section_header("HOAN THANH")
        logger.info("Quy trinh da hoan tat thanh cong")
        
    except KeyboardInterrupt:
        logger.info("\nDa dung boi nguoi dung.")
    except Exception as e:
        logger.exception("Loi khong xu ly duoc: %s", str(e))
        raise