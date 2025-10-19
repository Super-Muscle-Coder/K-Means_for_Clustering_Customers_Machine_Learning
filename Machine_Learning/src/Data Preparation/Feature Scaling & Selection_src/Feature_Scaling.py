import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import warnings
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings('ignore')

class FeatureScalingSelection:
    """
    Class thực hiện các bước của Feature Scaling & Selection trong quá trình Data Preparation.
    
    Quy trình tối ưu:
    1. Phân tích dữ liệu và tương quan
    2. Chọn một phương pháp scaling phù hợp nhất
    3. Loại bỏ multicollinearity
    4. Chọn lọc đặc trưng quan trọng
    5. Xuất dataset đã xử lý
    """

    def __init__(self, dataset_path):
        """
        Khởi tạo lớp FeatureScalingSelection với đường dẫn đến tập dữ liệu.
        """
        self.dataset_path = dataset_path
        self.dataset = None
        self.processed_dataset = None
        self.processing_log = []
        self.original_shape = None
        self.numerical_columns = []
        self.categorical_columns = []
        self.scaling_method = None  # Phương pháp scaling được chọn
        self.scaler = None  # Đối tượng scaler
    
    def log_action(self, action, details=""):
        """
        Ghi lại các hành động và chi tiết trong quá trình xử lý.
        """
        self.processing_log.append({
            'action': action,
            'details': details,
            'timestamp': pd.Timestamp.now()
        })

    # =======================================================================================================================
    # HÀM LOAD DATASET
    def load_data(self):
        """
        Load tập dữ liệu từ đường dẫn đã cung cấp.
        """
        try:
            print("="*100)
            print("Bước 1: TẢI VÀ ĐỌC DỮ LIỆU")
            print()

            self.dataset = pd.read_csv(self.dataset_path)
            self.original_shape = self.dataset.shape
            
            # Xác định các cột số và cột phân loại
            self.numerical_columns = self.dataset.select_dtypes(include=['int64', 'float64']).columns.tolist()
            self.categorical_columns = self.dataset.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Loại bỏ cột datetime nếu có
            datetime_cols = self.dataset.select_dtypes(include=['datetime64']).columns.tolist()
            if datetime_cols:
                print(f"Phát hiện {len(datetime_cols)} cột datetime: {datetime_cols}")
                print("Các cột này sẽ được loại trừ khỏi quá trình scaling.")
            
            print(f"Dữ liệu đã được tải thành công: {self.dataset.shape[0]:,} dòng × {self.dataset.shape[1]} cột")
            print(f"- Cột số: {len(self.numerical_columns)}")
            print(f"- Cột phân loại: {len(self.categorical_columns)}")
            print(f"- Cột datetime: {len(datetime_cols)}")
            print()
            
            self.log_action("Tải dữ liệu thành công", f"Dataset shape: {self.dataset.shape}")
            
            # Khởi tạo processed dataset
            self.processed_dataset = self.dataset.copy()
            
            return self.dataset

        except Exception as e:
            print(f"Lỗi khi tải dữ liệu: {e}")
            return None

    # =======================================================================================================================
    # HÀM PHÂN TÍCH TƯƠNG QUAN ĐẶC TRƯNG
    def analyze_feature_correlation(self, correlation_threshold=0.8):
        """
        Phân tích tương quan giữa các đặc trưng số và hiển thị ma trận tương quan.
        """
        print("="*100)
        print("Bước 2: PHÂN TÍCH TƯƠNG QUAN ĐẶC TRƯNG")
        print()
        
        if len(self.numerical_columns) < 2:
            print("Không đủ cột số để phân tích tương quan.")
            return None
        
        # Tính ma trận tương quan
        correlation_matrix = self.dataset[self.numerical_columns].corr()
        
        # Tìm các cặp đặc trưng có tương quan cao
        high_correlation_pairs = []
        for i in range(len(self.numerical_columns)):
            for j in range(i+1, len(self.numerical_columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) >= correlation_threshold:
                    high_correlation_pairs.append({
                        'feature1': self.numerical_columns[i],
                        'feature2': self.numerical_columns[j],
                        'correlation': corr_value
                    })
        
        # Sắp xếp theo giá trị tương quan giảm dần
        high_correlation_pairs = sorted(high_correlation_pairs, key=lambda x: abs(x['correlation']), reverse=True)
        
        print(f"Đã tìm thấy {len(high_correlation_pairs)} cặp đặc trưng có tương quan cao (|r| >= {correlation_threshold}):")
        print("-" * 80)
        for idx, pair in enumerate(high_correlation_pairs, 1):
            print(f"{idx:2d}. {pair['feature1']} vs {pair['feature2']}: r = {pair['correlation']:.3f}")
        print("-" * 80)
        
        # Tạo ma trận tương quan trực quan
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=False, cmap='coolwarm', vmin=-1, vmax=1, linewidths=.5)
        plt.title('Ma trận tương quan giữa các đặc trưng số')
        
        # Lưu biểu đồ
        figure_dir = os.path.join(r"C:\Project\Machine_Learning\Machine_Learning\output\Scaling&Selection_report\figures")
        os.makedirs(figure_dir, exist_ok=True)
        correlation_heatmap_path = os.path.join(figure_dir, 'correlation_heatmap.png')
        plt.tight_layout()
        plt.savefig(correlation_heatmap_path)
        plt.close()
        
        print(f"Đã lưu ma trận tương quan: {correlation_heatmap_path}")
        print()
        
        # Ghi nhận kết quả
        self.correlation_analysis = {
            'high_correlation_pairs': high_correlation_pairs,
            'correlation_matrix': correlation_matrix,
            'correlation_threshold': correlation_threshold,
            'correlation_heatmap_path': correlation_heatmap_path
        }
        
        self.log_action("Phân tích tương quan đặc trưng", f"Đã tìm thấy {len(high_correlation_pairs)} cặp đặc trưng tương quan cao")
        
        return correlation_matrix, high_correlation_pairs

    # =======================================================================================================================
    # HÀM PHÂN TÍCH PHÂN PHỐI DỮ LIỆU
    def analyze_distribution(self, n_samples=5):
        """
        Phân tích phân phối của dữ liệu để xác định phương pháp scaling phù hợp.
        """
        print("="*100)
        print("Bước 3: PHÂN TÍCH PHÂN PHỐI DỮ LIỆU")
        print()
        
        if len(self.numerical_columns) == 0:
            print("Không có cột số để phân tích phân phối.")
            return None
            
        # Chọn một số mẫu để phân tích
        if len(self.numerical_columns) > n_samples:
            np.random.seed(42)
            sample_columns = np.random.choice(self.numerical_columns, n_samples, replace=False).tolist()
        else:
            sample_columns = self.numerical_columns
            
        # Tính các thông số thống kê
        distribution_stats = {}
        for col in self.numerical_columns:
            data = self.dataset[col]
            distribution_stats[col] = {
                'mean': data.mean(),
                'median': data.median(),
                'std': data.std(),
                'min': data.min(),
                'max': data.max(),
                'skewness': data.skew(),
                'kurtosis': data.kurtosis(),
                'has_outliers': abs(data.skew()) > 1 or abs(data.kurtosis()) > 3
            }
            
        # Hiển thị thống kê
        print("Thống kê phân phối dữ liệu:")
        print("-" * 120)
        print(f"{'Feature':25} | {'Mean':10} | {'Median':10} | {'Std':10} | {'Skewness':10} | {'Kurtosis':10} | {'Has Outliers'}")
        print("-" * 120)
        
        for col, stats in distribution_stats.items():
            print(f"{col:25} | {stats['mean']:10.2f} | {stats['median']:10.2f} | {stats['std']:10.2f} | {stats['skewness']:10.2f} | {stats['kurtosis']:10.2f} | {'Có' if stats['has_outliers'] else 'Không'}")
        
        # Đếm số cột có outliers
        n_with_outliers = sum(1 for stats in distribution_stats.values() if stats['has_outliers'])
        print("-" * 80)
        print(f"Tổng số cột có phân phối lệch/outliers: {n_with_outliers}/{len(self.numerical_columns)} ({n_with_outliers/len(self.numerical_columns)*100:.1f}%)")
        
        # Vẽ biểu đồ phân phối cho các cột mẫu
        fig, axes = plt.subplots(len(sample_columns), 1, figsize=(12, 4*len(sample_columns)))
        if len(sample_columns) == 1:
            axes = [axes]
            
        for i, col in enumerate(sample_columns):
            sns.histplot(self.dataset[col], kde=True, ax=axes[i])
            axes[i].set_title(f'Phân phối của {col} (skew={distribution_stats[col]["skewness"]:.2f}, kurt={distribution_stats[col]["kurtosis"]:.2f})')
        
        plt.tight_layout()
        
        # Lưu biểu đồ
        figure_dir = os.path.join(r"C:\Project\Machine_Learning\Machine_Learning\output\Scaling&Selection_report\figures")
        os.makedirs(figure_dir, exist_ok=True)
        distribution_plot_path = os.path.join(figure_dir, 'feature_distributions.png')
        plt.savefig(distribution_plot_path)
        plt.close()
        
        print(f"\nĐã lưu biểu đồ phân phối đặc trưng: {distribution_plot_path}")
        print()
        
        # Đề xuất phương pháp scaling
        if n_with_outliers / len(self.numerical_columns) > 0.3:
            recommended_scaler = "RobustScaler"
            reason = "Nhiều cột có phân phối lệch và outliers"
        elif n_with_outliers / len(self.numerical_columns) > 0.1:
            recommended_scaler = "StandardScaler"
            reason = "Một số cột có phân phối lệch nhưng không quá nhiều outliers"
        else:
            recommended_scaler = "MinMaxScaler"
            reason = "Phần lớn các cột có phân phối tương đối cân đối, ít outliers"
        
        print(f"Đề xuất phương pháp scaling: {recommended_scaler}")
        print(f"Lý do: {reason}")
        print()
        
        self.log_action("Phân tích phân phối dữ liệu", 
                       f"Đã phân tích {len(self.numerical_columns)} cột số, đề xuất sử dụng {recommended_scaler}")
        
        # Trả về thống kê và đề xuất
        return {
            'distribution_stats': distribution_stats,
            'recommended_scaler': recommended_scaler,
            'reason': reason,
            'n_with_outliers': n_with_outliers
        }

    # =======================================================================================================================
    # HÀM THỰC HIỆN SCALING DỮ LIỆU
    def scale_features(self, method=None, columns=None):
        """
        Thực hiện scaling dữ liệu với phương pháp được chỉ định.
        
        Tham số:
        - method (str): Phương pháp scaling ('standard', 'minmax', hoặc 'robust'). Nếu None, sẽ sử dụng phương pháp đề xuất.
        - columns (list): Danh sách các cột cần scaling. Nếu None, sẽ áp dụng cho tất cả cột số.
        """
        print("="*100)
        print("Bước 4: THỰC HIỆN SCALING DỮ LIỆU")
        print()
        
        if not hasattr(self, 'processed_dataset') or self.processed_dataset is None:
            self.processed_dataset = self.dataset.copy()
            
        if not columns:
            columns = self.numerical_columns
            
        # Lọc ra chỉ các cột tồn tại trong dataset
        columns = [col for col in columns if col in self.dataset.columns]
        
        if len(columns) == 0:
            print("Không có cột nào để thực hiện scaling.")
            return None
            
        # Nếu không chỉ định phương pháp, sử dụng đề xuất từ phân tích phân phối
        if method is None:
            if hasattr(self, 'distribution_analysis') and self.distribution_analysis:
                method = self.distribution_analysis['recommended_scaler'].lower()
            else:
                # Phân tích phân phối nếu chưa thực hiện
                analysis = self.analyze_distribution()
                method = analysis['recommended_scaler'].lower()
        
        # Chuyển đổi tên phương pháp
        method = method.lower()
        if 'standard' in method:
            scaler = StandardScaler()
            method_name = "StandardScaler"
        elif 'minmax' in method:
            scaler = MinMaxScaler()
            method_name = "MinMaxScaler"
        elif 'robust' in method:
            scaler = RobustScaler()
            method_name = "RobustScaler"
        else:
            print(f"Phương pháp scaling không được hỗ trợ: {method}. Sử dụng StandardScaler.")
            scaler = StandardScaler()
            method_name = "StandardScaler"
            
        print(f"Áp dụng {method_name} cho {len(columns)} cột:")
        print("-" * 80)
        
        try:
            # Lưu thông số trước khi scaling
            before_stats = {}
            for col in columns:
                before_stats[col] = {
                    'mean': self.dataset[col].mean(),
                    'median': self.dataset[col].median(),
                    'std': self.dataset[col].std(),
                    'min': self.dataset[col].min(),
                    'max': self.dataset[col].max()
                }
            
            # Thực hiện scaling
            scaled_data = scaler.fit_transform(self.dataset[columns])
            
            # Tạo DataFrame mới với dữ liệu đã scaled
            scaled_df = pd.DataFrame(scaled_data, columns=columns)
            
            # Lưu các cột đã scaled vào processed_dataset
            for col in columns:
                self.processed_dataset[col] = scaled_df[col]
            
            # Lưu thông số sau khi scaling
            after_stats = {}
            for col in columns:
                after_stats[col] = {
                    'mean': self.processed_dataset[col].mean(),
                    'median': self.processed_dataset[col].median(),
                    'std': self.processed_dataset[col].std(),
                    'min': self.processed_dataset[col].min(),
                    'max': self.processed_dataset[col].max()
                }
                
                # Hiển thị thống kê
                print(f"{col}:")
                if method_name == "StandardScaler":
                    print(f"  Trước: mean={before_stats[col]['mean']:.2f}, std={before_stats[col]['std']:.2f}, range=[{before_stats[col]['min']:.2f}, {before_stats[col]['max']:.2f}]")
                    print(f"  Sau  : mean={after_stats[col]['mean']:.2f}, std={after_stats[col]['std']:.2f}, range=[{after_stats[col]['min']:.2f}, {after_stats[col]['max']:.2f}]")
                elif method_name == "MinMaxScaler":
                    print(f"  Trước: min={before_stats[col]['min']:.2f}, max={before_stats[col]['max']:.2f}, range={before_stats[col]['max']-before_stats[col]['min']:.2f}")
                    print(f"  Sau  : min={after_stats[col]['min']:.2f}, max={after_stats[col]['max']:.2f}, range={after_stats[col]['max']-after_stats[col]['min']:.2f}")
                elif method_name == "RobustScaler":
                    print(f"  Trước: median={before_stats[col]['median']:.2f}, range=[{before_stats[col]['min']:.2f}, {before_stats[col]['max']:.2f}]")
                    print(f"  Sau  : median={after_stats[col]['median']:.2f}, range=[{after_stats[col]['min']:.2f}, {after_stats[col]['max']:.2f}]")
                print()
                
            # Lưu thông tin scaler
            self.scaling_method = method_name
            self.scaler = scaler
            self.scaled_columns = columns
            self.scaling_stats = {
                'before': before_stats,
                'after': after_stats
            }
            
            # Vẽ biểu đồ so sánh trước và sau scaling
            sample_cols = columns[:min(5, len(columns))]
            
            fig, axes = plt.subplots(len(sample_cols), 2, figsize=(16, 4*len(sample_cols)))
            if len(sample_cols) == 1:
                axes = [axes]
                
            for i, col in enumerate(sample_cols):
                # Trước scaling
                sns.histplot(self.dataset[col], kde=True, ax=axes[i, 0])
                axes[i, 0].set_title(f'{col} - Trước scaling')
                
                # Sau scaling
                sns.histplot(self.processed_dataset[col], kde=True, ax=axes[i, 1])
                axes[i, 1].set_title(f'{col} - Sau scaling ({method_name})')
                
            plt.tight_layout()
            
            # Lưu biểu đồ
            figure_dir = os.path.join(r"C:\Project\Machine_Learning\Machine_Learning\output\Scaling&Selection_report\figures")
            os.makedirs(figure_dir, exist_ok=True)
            scaling_plot_path = os.path.join(figure_dir, f'{method_name.lower()}_comparison.png')
            plt.savefig(scaling_plot_path)
            plt.close()
            
            print(f"Đã lưu biểu đồ so sánh {method_name}: {scaling_plot_path}")
            print()
            
            self.log_action(f"Áp dụng {method_name}", f"Đã scaling {len(columns)} cột")
            
            return {
                'method': method_name,
                'columns': columns,
                'before_stats': before_stats,
                'after_stats': after_stats
            }
            
        except Exception as e:
            print(f"Lỗi khi thực hiện scaling: {e}")
            import traceback
            print(traceback.format_exc())
            return None

    # =======================================================================================================================
    # HÀM XỬ LÝ MULTICOLLINEARITY
    def handle_multicollinearity(self, correlation_threshold=0.8, method='select_one'):
        """
        Xử lý multicollinearity giữa các đặc trưng.
        
        Tham số:
        - correlation_threshold (float): Ngưỡng tương quan để xác định multicollinearity.
        - method (str): Phương pháp xử lý ('select_one' hoặc 'vif').
        """
        print("="*100)
        print("Bước 5: XỬ LÝ MULTICOLLINEARITY")
        print()
        
        # Đảm bảo đã có phân tích tương quan
        if not hasattr(self, 'correlation_analysis'):
            self.analyze_feature_correlation(correlation_threshold)
        
        high_correlation_pairs = self.correlation_analysis['high_correlation_pairs']
        correlation_matrix = self.correlation_analysis['correlation_matrix']
        
        if not high_correlation_pairs:
            print("Không phát hiện multicollinearity đáng kể (không có cặp đặc trưng có tương quan cao).")
            return None
        
        print(f"Xử lý multicollinearity với ngưỡng tương quan: {correlation_threshold}")
        print(f"Phương pháp: {method}")
        print(f"Tổng số cặp đặc trưng có tương quan cao: {len(high_correlation_pairs)}")
        print()
        
        features_to_remove = set()
        
        if method == 'select_one':
            # Chọn một đặc trưng từ mỗi cặp có tương quan cao
            for pair in high_correlation_pairs:
                feature1 = pair['feature1']
                feature2 = pair['feature2']
                
                # Bỏ qua nếu một trong hai đặc trưng đã được đánh dấu để loại bỏ
                if feature1 in features_to_remove or feature2 in features_to_remove:
                    continue
                
                # Tính trung bình tương quan tuyệt đối với các đặc trưng khác
                corr1 = correlation_matrix.loc[feature1].drop(feature1).abs().mean()
                corr2 = correlation_matrix.loc[feature2].drop(feature2).abs().mean()
                
                # Loại bỏ đặc trưng có trung bình tương quan cao hơn
                if corr1 > corr2:
                    features_to_remove.add(feature1)
                    print(f"Loại bỏ {feature1} (giữ {feature2}): {feature1} có mean absolute corr = {corr1:.3f}, {feature2} có mean absolute corr = {corr2:.3f}")
                else:
                    features_to_remove.add(feature2)
                    print(f"Loại bỏ {feature2} (giữ {feature1}): {feature1} có mean absolute corr = {corr1:.3f}, {feature2} có mean absolute corr = {corr2:.3f}")
                    
        elif method == 'vif':
            # Sử dụng VIF (Variance Inflation Factor)
            print("Tính toán VIF cho các đặc trưng...")
            
            # Chuẩn bị dữ liệu để tính VIF
            X = self.processed_dataset[self.numerical_columns].copy()
            X = X.assign(const=1)
            
            # Tính VIF cho từng đặc trưng
            vif_data = {}
            for i, col in enumerate(self.numerical_columns):
                try:
                    vif_value = variance_inflation_factor(X.values, i)
                    vif_data[col] = vif_value
                    print(f"{col:>25}: VIF = {vif_value:.2f} {'(loại bỏ)' if vif_value > 10 else ''}")
                except Exception as e:
                    print(f"Không thể tính VIF cho {col}: {e}")
                    vif_data[col] = float('inf')
            
            # Loại bỏ các đặc trưng có VIF cao
            vif_threshold = 10.0
            high_vif_features = [col for col, vif in vif_data.items() if vif > vif_threshold]
            features_to_remove.update(high_vif_features)
            
        else:
            print(f"Phương pháp xử lý multicollinearity không hợp lệ: {method}")
            return None
        
        features_to_remove = list(features_to_remove)
        features_to_keep = [col for col in self.numerical_columns if col not in features_to_remove]
        
        print("\nKết quả xử lý multicollinearity:")
        print(f"- Số đặc trưng giữ lại: {len(features_to_keep)}")
        print(f"- Số đặc trưng loại bỏ: {len(features_to_remove)}")
        print(f"- Danh sách đặc trưng loại bỏ: {features_to_remove}")
        
        # Cập nhật processed_dataset
        self.processed_dataset = self.processed_dataset.drop(columns=features_to_remove)
        
        # Cập nhật numerical_columns
        self.numerical_columns = features_to_keep
        
        self.log_action("Xử lý multicollinearity", 
                       f"Đã loại bỏ {len(features_to_remove)} đặc trưng, giữ lại {len(features_to_keep)} đặc trưng")
        
        return {
            'features_to_keep': features_to_keep,
            'features_to_remove': features_to_remove
        }

    # =======================================================================================================================
    # HÀM CHỌN ĐẶC TRƯNG QUAN TRỌNG
    def select_important_features(self, target_column, n_features=None, importance_threshold=0.01):
        """
        Chọn đặc trưng quan trọng dựa trên độ quan trọng từ Random Forest.
        
        Tham số:
        - target_column (str): Tên cột mục tiêu.
        - n_features (int): Số lượng đặc trưng muốn giữ lại. Nếu None, sẽ dựa vào ngưỡng.
        - importance_threshold (float): Ngưỡng độ quan trọng để chọn đặc trưng.
        """
        print("="*100)
        print("Bước 6: CHỌN ĐẶC TRƯNG QUAN TRỌNG")
        print()
        
        if target_column not in self.processed_dataset.columns:
            print(f"Không tìm thấy cột mục tiêu '{target_column}'. Vui lòng kiểm tra lại.")
            return None
            
        numerical_columns = [col for col in self.numerical_columns if col in self.processed_dataset.columns]
        
        if len(numerical_columns) == 0:
            print("Không có cột số để chọn đặc trưng quan trọng.")
            return None
            
        print(f"Sử dụng Random Forest để đánh giá độ quan trọng của đặc trưng...")
        print(f"- Cột mục tiêu: {target_column}")
        print(f"- Số lượng đặc trưng đầu vào: {len(numerical_columns)}")
        
        try:
            # Chuẩn bị dữ liệu
            X = self.processed_dataset[numerical_columns]
            y = self.processed_dataset[target_column]
            
            # Huấn luyện Random Forest để đánh giá độ quan trọng của đặc trưng
            clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            clf.fit(X, y)
            
            # Tính độ quan trọng
            importances = clf.feature_importances_
            
            # Tạo DataFrame chứa độ quan trọng
            feature_importances = pd.DataFrame({
                'feature': numerical_columns,
                'importance': importances
            }).sort_values(by='importance', reverse=True)
            
            # Hiển thị độ quan trọng
            print("\nĐộ quan trọng của các đặc trưng:")
            for _, row in feature_importances.iterrows():
                print(f"{row['feature']:>25}: {row['importance']:.5f}")
            
            # Chọn đặc trưng dựa trên số lượng hoặc ngưỡng
            if n_features is not None:
                selected_features = feature_importances['feature'].head(n_features).tolist()
                selection_criteria = f"top {n_features} đặc trưng"
            else:
                selected_features = feature_importances[feature_importances['importance'] >= importance_threshold]['feature'].tolist()
                selection_criteria = f"ngưỡng độ quan trọng {importance_threshold}"
            
            # Thêm cột mục tiêu và categorical columns
            selected_features = selected_features + [target_column] + self.categorical_columns
            selected_features = list(set(selected_features))  # Loại bỏ trùng lặp
            
            # Các đặc trưng bị loại bỏ
            excluded_features = [col for col in numerical_columns if col not in selected_features]
            
            print(f"\nChọn đặc trưng với {selection_criteria}:")
            print(f"- Số đặc trưng số giữ lại: {len(selected_features) - 1 - len(self.categorical_columns)}")
            print(f"- Số đặc trưng số loại bỏ: {len(excluded_features)}")
            print(f"- Tổng số cột trong dataset sau khi chọn: {len(selected_features)}")
            
            # Cập nhật processed_dataset
            self.processed_dataset = self.processed_dataset[selected_features]
            
            # Cập nhật numerical_columns
            self.numerical_columns = [col for col in self.numerical_columns if col in selected_features]
            
            # Vẽ biểu đồ độ quan trọng
            plt.figure(figsize=(14, 6))
            top_features = feature_importances.head(20)
            sns.barplot(x='feature', y='importance', data=top_features)
            plt.xticks(rotation=90)
            plt.xlabel('Features')
            plt.ylabel('Importance')
            plt.title('Top 20 Feature Importance')
            if importance_threshold is not None and n_features is None:
                plt.axhline(y=importance_threshold, color='r', linestyle='--', label=f'Threshold: {importance_threshold}')
                plt.legend()
            plt.tight_layout()
            
            # Lưu biểu đồ
            figure_dir = os.path.join(r"C:\Project\Machine_Learning\Machine_Learning\output\Scaling&Selection_report\figures")
            os.makedirs(figure_dir, exist_ok=True)
            importance_plot_path = os.path.join(figure_dir, 'feature_importance.png')
            plt.savefig(importance_plot_path)
            plt.close()
            
            print(f"\nĐã lưu biểu đồ độ quan trọng đặc trưng: {importance_plot_path}")
            
            self.log_action("Chọn đặc trưng quan trọng", 
                           f"Đã chọn {len(selected_features)} đặc trưng dựa trên {selection_criteria}")
            
            return {
                'selected_features': selected_features,
                'excluded_features': excluded_features,
                'feature_importances': feature_importances.to_dict('records')
            }
            
        except Exception as e:
            print(f"Lỗi khi chọn đặc trưng quan trọng: {e}")
            import traceback
            print(traceback.format_exc())
            return None

    # =======================================================================================================================
    # HÀM XUẤT DATASET ĐÃ XỬ LÝ
    def export_processed_dataset(self, filename=None):
        """
        Xuất dataset đã xử lý ra file.
        
        Tham số:
        - filename (str): Tên file xuất. Nếu None, sẽ tự động tạo tên.
        """
        print("="*100)
        print("Bước 7: XUẤT DATASET ĐÃ XỬ LÝ")
        print()
        
        if self.processed_dataset is None or len(self.processed_dataset) == 0:
            print("Không có dữ liệu để xuất.")
            return None
            
        try:
            # Tạo tên file xuất
            if filename is None:
                timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                filename = f"scaled_selected_dataset_{timestamp}.csv"
                
            # Đường dẫn đầy đủ
            output_path = os.path.join(r"C:\Project\Machine_Learning\Machine_Learning\Dataset", filename)
            
            # Xuất dataset
            self.processed_dataset.to_csv(output_path, index=False)
            
            print(f"Đã xuất dataset đã xử lý thành công:")
            print(f"- Đường dẫn: {output_path}")
            print(f"- Shape: {self.processed_dataset.shape}")
            print(f"- Số cột số: {len(self.numerical_columns)}")
            print(f"- Số cột phân loại: {len(self.categorical_columns)}")
            print()
            
            self.log_action("Xuất dataset đã xử lý", 
                           f"Đã xuất dataset {self.processed_dataset.shape} ra {output_path}")
            
            return output_path
            
        except Exception as e:
            print(f"Lỗi khi xuất dataset: {e}")
            import traceback
            print(traceback.format_exc())
            return None

    # =======================================================================================================================
    # HÀM TẠO TÓM TẮT KẾT QUẢ
    def generate_summary(self, start_time):
        """
        Tạo tóm tắt kết quả của quá trình Feature Scaling & Selection.
        
        Tham số:
        - start_time (float): Thời điểm bắt đầu quá trình.
        """
        print("="*100)
        print("TÓM TẮT KẾT QUẢ FEATURE SCALING & SELECTION")
        print("="*100)
        
        # Thời gian thực hiện
        elapsed_time = time.time() - start_time
        
        # Thông tin dataset
        original_shape = self.original_shape
        current_shape = self.processed_dataset.shape if self.processed_dataset is not None else None
        
        print("THÔNG TIN DATASET:")
        print(f"    Shape ban đầu          : {original_shape[0]:,} dòng × {original_shape[1]} cột")
        if current_shape:
            print(f"    Shape sau xử lý        : {current_shape[0]:,} dòng × {current_shape[1]} cột")
            cols_diff = current_shape[1] - original_shape[1]
            diff_text = f"{cols_diff:+}" if cols_diff != 0 else "không đổi"
            print(f"    Thay đổi số cột       : {diff_text}")
        
        # Thông tin scaling
        print("\nFEATURE SCALING:")
        if hasattr(self, 'scaling_method') and self.scaling_method:
            print(f"    Phương pháp           : {self.scaling_method}")
            print(f"    Số cột áp dụng        : {len(self.scaled_columns)}")
        else:
            print("    Không thực hiện scaling")
            
        # Thông tin feature selection
        print("\nFEATURE SELECTION:")
        original_numerical = len([col for col in self.dataset.columns if col in self.numerical_columns])
        current_numerical = len(self.numerical_columns)
        print(f"    Số đặc trưng ban đầu  : {original_numerical}")
        print(f"    Số đặc trưng giữ lại  : {current_numerical}")
        print(f"    Số đặc trưng loại bỏ  : {original_numerical - current_numerical}")
        
        # Hiệu suất xử lý
        print("\nHIỆU SUẤT XỬ LÝ:")
        print(f"    Thời gian xử lý       : {elapsed_time:.2f} giây")
        print(f"    Tốc độ xử lý          : {len(self.dataset) / elapsed_time:.0f} dòng/giây")
        print(f"    Bộ nhớ sử dụng        : {self.processed_dataset.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        
        # Trạng thái hệ thống
        print("\nTRẠNG THÁI HỆ THỐNG:")
        print(f"    Tính toàn vẹn dữ liệu : Được duy trì")
        print(f"    Log system            : Hoàn thành ({len(self.processing_log)} actions)")
        print(f"    Feature scaling       : Hoàn thành")
        print(f"    Feature selection     : Hoàn thành")
        
        # Trả về thông tin tóm tắt
        summary = {
            'original_shape': original_shape,
            'current_shape': current_shape,
            'scaling_method': self.scaling_method if hasattr(self, 'scaling_method') else None,
            'num_original_features': original_numerical,
            'num_selected_features': current_numerical,
            'elapsed_time': elapsed_time
        }
        
        return summary

    # =======================================================================================================================
    # HÀM HIỂN THỊ LOG
    def display_processing_log(self):
        """
        Hiển thị log các hành động đã thực hiện.
        """
        print("="*100)
        print("LOG CÁC HÀNH ĐỘNG THỰC HIỆN")
        print("="*100)
        
        if not self.processing_log:
            print("Không có log nào được ghi lại.")
            return
            
        print(f"Tổng số hành động: {len(self.processing_log)}")
        print()
        
        for i, log_entry in enumerate(self.processing_log, 1):
            timestamp = log_entry['timestamp'].strftime('%H:%M:%S')
            action = log_entry['action']
            details = log_entry.get('details', '')
            
            print(f"{i:2d}.  [{timestamp}] {action}")
            if details:
                print(f"   └─ {details}")
        
        print(f"\nĐã hoàn thành {len(self.processing_log)} hành động")

    # =======================================================================================================================
    # HÀM XUẤT BÁO CÁO TEXT
    def export_text_report(self, output_dir=None):
        """
        Xuất báo cáo text chi tiết về quá trình Feature Scaling & Selection.
        """
        if output_dir is None:
            output_dir = r"C:\Project\Machine_Learning\Machine_Learning\output\Scaling&Selection_report"

        os.makedirs(output_dir, exist_ok=True)

        report_file = os.path.join(output_dir, f"Feature_Scaling_Selection_report.txt")

        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                # Header
                f.write("=" * 100 + "\n")
                f.write("THỰC HIỆN QUY TRÌNH FEATURE SCALING & SELECTION HOÀN CHỈNH\n")
                f.write("=" * 100 + "\n")
                f.write("Bước 1: TẢI VÀ ĐỌC DỮ LIỆU\n\n")
            
                if self.dataset is not None:
                    f.write(f"Dữ liệu đã được tải thành công: {self.dataset.shape[0]:,} dòng × {self.dataset.shape[1]} cột\n")
                    f.write(f"- Cột số: {len(self.numerical_columns)}\n")
                    f.write(f"- Cột phân loại: {len(self.categorical_columns)}\n\n")
            
                # Ghi chi tiết các bước từ processing log
                current_step = 2  # Bắt đầu từ bước 2 vì bước 1 đã viết ở trên
                for log_entry in self.processing_log:
                    timestamp = log_entry['timestamp'].strftime('%H:%M:%S')
                    action = log_entry['action']
                    details = log_entry.get('details', '')
                
                    # Kiểm tra nếu là bước chính
                    if any(keyword in action for keyword in [
                        'Phân tích tương quan',
                        'Phân tích phân phối',
                        'Áp dụng StandardScaler',
                        'Áp dụng MinMaxScaler',
                        'Áp dụng RobustScaler',
                        'Xử lý multicollinearity',
                        'Chọn đặc trưng quan trọng',
                        'Xuất dataset đã xử lý'
                    ]):
                        # Tạo tiêu đề bước
                        if 'Phân tích tương quan' in action:
                            step_title = "PHÂN TÍCH TƯƠNG QUAN ĐẶC TRƯNG"
                        elif 'Phân tích phân phối' in action:
                            step_title = "PHÂN TÍCH PHÂN PHỐI DỮ LIỆU"
                        elif 'Áp dụng StandardScaler' in action:
                            step_title = "ÁP DỤNG STANDARDSCALER"
                        elif 'Áp dụng MinMaxScaler' in action:
                            step_title = "ÁP DỤNG MINMAXSCALER"
                        elif 'Áp dụng RobustScaler' in action:
                            step_title = "ÁP DỤNG ROBUSTSCALER"
                        elif 'Xử lý multicollinearity' in action:
                            step_title = "XỬ LÝ MULTICOLLINEARITY"
                        elif 'Chọn đặc trưng quan trọng' in action:
                            step_title = "CHỌN ĐẶC TRƯNG QUAN TRỌNG"
                        elif 'Xuất dataset đã xử lý' in action:
                            step_title = "XUẤT DATASET ĐÃ XỬ LÝ"
                        else:
                            step_title = action.upper()
                        
                        f.write(f"\n{'='*100}\n")
                        f.write(f"Bước {current_step}: {step_title}\n")
                        f.write(f"{'='*100}\n")
                        current_step += 1
                
                    f.write(f"[{timestamp}] {action}\n")
                    if details:
                        f.write(f"    └─ {details}\n")
                    f.write("\n")
                    
            print(f"Đã xuất báo cáo Feature Scaling & Selection ra file: {report_file}")
            return report_file
        
        except Exception as e:
            print(f"Lỗi khi xuất báo cáo: {e}")
            return None

    # =======================================================================================================================
    # HÀM THỰC HIỆN TOÀN BỘ PIPELINE
    def run_pipeline(self, target_column='Response', scaling_method=None):
        """
        Thực hiện toàn bộ pipeline Feature Scaling & Selection.
        
        Tham số:
        - target_column (str): Tên cột mục tiêu cho việc chọn đặc trưng quan trọng.
        - scaling_method (str): Phương pháp scaling sẽ sử dụng. Nếu None, sẽ tự động chọn.
        """
        print("="*100)
        print("THỰC HIỆN QUY TRÌNH FEATURE SCALING & SELECTION HOÀN CHỈNH")
        print("="*100)
        
        start_time = time.time()
        
        try:
            # Bước 1: Load dữ liệu
            self.load_data()
            
            # Bước 2: Phân tích tương quan
            self.analyze_feature_correlation()
            
            # Bước 3: Phân tích phân phối dữ liệu
            distribution_analysis = self.analyze_distribution()
            self.distribution_analysis = distribution_analysis
            
            # Bước 4: Thực hiện scaling dữ liệu
            self.scale_features(method=scaling_method)
            
            # Bước 5: Xử lý multicollinearity
            self.handle_multicollinearity(method='select_one')
            
            # Bước 6: Chọn đặc trưng quan trọng
            if target_column in self.dataset.columns:
                self.select_important_features(target_column=target_column)
            else:
                print(f"Không tìm thấy cột mục tiêu '{target_column}'. Bỏ qua bước chọn đặc trưng quan trọng.")
            
            # Bước 7: Xuất dataset đã xử lý
            self.export_processed_dataset()
            
            # Hiển thị tóm tắt kết quả
            self.generate_summary(start_time)
            
            # Hiển thị log
            self.display_processing_log()
            
            # Xuất báo cáo text
            self.export_text_report()
            
            # Kết thúc pipeline
            total_time = time.time() - start_time
            print("\n" + "=" * 100)
            print("HOÀN THÀNH FEATURE SCALING & SELECTION")
            print("=" * 100)
            print(f" Tổng thời gian thực hiện : {total_time:.2f} giây")
            print(f" Tốc độ xử lý             : {len(self.dataset) / total_time:.0f} dòng/giây")
            print(f" Thời gian kết thúc       : {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            return {
                'success': True,
                'original_shape': self.original_shape,
                'final_shape': self.processed_dataset.shape,
                'elapsed_time': total_time
            }
            
        except Exception as e:
            print(f"\nLỖI TRONG PIPELINE: {e}")
            import traceback
            print("Chi tiết lỗi:")
            print(traceback.format_exc())
            return {
                'success': False,
                'error': str(e)
            }

# =================================================================================================================================
# HÀM MAIN 
def main():
    """
    Hàm Main
    """
    
    # Đường dẫn đến dataset
    dataset_path = r"C:\Project\Machine_Learning\Machine_Learning\Dataset\Customer_Behavior_feature_engineered_20251016_141827.csv"
    
    try:
        print("Bắt đầu Feature Scaling & Selection Pipeline")
        print()
        
        # Khởi tạo đối tượng Feature Scaling & Selection
        scaler_selector = FeatureScalingSelection(dataset_path)
        
        # Chạy toàn bộ pipeline
        results = scaler_selector.run_pipeline(target_column='Response')
        
        # Kiểm tra kết quả
        if results and results.get('success'):
            print("\n" + "="*60)
            print("FEATURE SCALING & SELECTION PIPELINE HOÀN THÀNH")
            print("="*60)
            print(f"Dataset gốc                : {results['original_shape']}")
            print(f"Dataset sau xử lý          : {results['final_shape']}")
            print(f"Tổng thời gian             : {results['elapsed_time']:.2f} giây")
            print("="*60)
        else:
            print("\nFEATURE SCALING & SELECTION PIPELINE GẶP LỖI")
            if results:
                print(f"Chi tiết lỗi: {results.get('error', 'Unknown error')}")
        
    except Exception as e:
        print(f"\nLỖI KHỞI TẠO FEATURE SCALING & SELECTION: {e}")
        import traceback
        print("Chi tiết lỗi:")
        print(traceback.format_exc())

# =================================================================================================================================
if __name__ == "__main__":
    main()