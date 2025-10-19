import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from datetime import datetime
from pathlib import Path

class EngineeredDatasetAnalyzer:
    """
    Class phân tích và đánh giá dataset sau khi đã qua cả hai giai đoạn:
    1. Data Processing & Wrangling
    2. Feature Extraction & Engineering
    
    Tập trung vào kiểm định chất lượng dữ liệu, đánh giá các đặc trưng mới,
    và chuẩn bị cho giai đoạn Feature Scaling & Selection.
    """

    def __init__(self, dataset_path, output_dir=None):
        """
        Khởi tạo analyzer với đường dẫn dataset đã qua feature engineering.
        
        Args:
            dataset_path (str): Đường dẫn đến file CSV đã feature engineered
            output_dir (str, optional): Thư mục xuất báo cáo
        """
        self.dataset_path = dataset_path
        self.dataset = None
        
        # Thiết lập thư mục output
        if output_dir is None:
            project_root = Path(__file__).parent.parent.parent.parent
            self.output_dir = project_root / "Machine_Learning" / "output" / "After_each_stage_report"
        else:
            self.output_dir = Path(output_dir)
            
        os.makedirs(self.output_dir, exist_ok=True)
        
        # File báo cáo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.report_file = self.output_dir / f"S2_report_{timestamp}.txt"
        
    #===============================================================================================================================
    # HÀM IN VÀ GHI LOG
    def print_and_log(self, text):
        """In ra console và ghi vào file báo cáo."""
        print(text)
        with open(self.report_file, "a", encoding="utf-8") as f:
            f.write(text + "\n")
    
    #===============================================================================================================================
    # HÀM TẢI DỮ LIỆU
    def load_data(self):
        """Tải dữ liệu từ file CSV đã qua feature engineering."""
        try:
            self.print_and_log("=" * 100)
            self.print_and_log("PHÂN TÍCH DATASET SAU KHI FEATURE ENGINEERING")
            self.print_and_log("=" * 100)
            self.print_and_log(f"Đọc dữ liệu từ: {self.dataset_path}")
            
            # Đọc dữ liệu với parse datetime cho Dt_Customer
            self.dataset = pd.read_csv(self.dataset_path, parse_dates=['Dt_Customer'])
            
            self.print_and_log(f"Tải dữ liệu thành công: {self.dataset.shape[0]:,} dòng × {self.dataset.shape[1]} cột\n")
            return True
            
        except Exception as e:
            self.print_and_log(f"Lỗi khi tải dữ liệu: {e}")
            return False
    
    #===============================================================================================================================
    # HÀM PHÂN TÍCH THÔNG TIN CƠ BẢN
    def analyze_basic_info(self):
        """Phân tích thông tin cơ bản của dataset sau feature engineering."""
        self.print_and_log("1. THÔNG TIN CƠ BẢN SAU FEATURE ENGINEERING")
        self.print_and_log("=" * 100)
        
        # Shape
        self.print_and_log(f"Dataset shape: {self.dataset.shape}")
        self.print_and_log(f"- Số dòng: {self.dataset.shape[0]:,}")
        self.print_and_log(f"- Số cột: {self.dataset.shape[1]}")
        
        # Kiểu dữ liệu
        self.print_and_log(f"\nPhân bố kiểu dữ liệu:")
        dtype_counts = self.dataset.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            self.print_and_log(f"- {dtype}: {count} cột")
        
        # Thống kê bộ nhớ
        memory_usage = self.dataset.memory_usage(deep=True).sum() / 1024 / 1024
        self.print_and_log(f"\nBộ nhớ sử dụng: {memory_usage:.2f} MB")
        self.print_and_log("")

    #===============================================================================================================================
    # HÀM PHÂN TÍCH CHẤT LƯỢNG DỮ LIỆU
    def analyze_data_quality(self):
        """Kiểm tra chất lượng dữ liệu sau feature engineering."""
        self.print_and_log("2. ĐÁNH GIÁ CHẤT LƯỢNG DỮ LIỆU")
        self.print_and_log("=" * 100)
        
        # Missing values
        missing_count = self.dataset.isnull().sum().sum()
        self.print_and_log(f"Missing values: {missing_count}")
        
        if missing_count > 0:
            missing_cols = self.dataset.isnull().sum()
            missing_cols = missing_cols[missing_cols > 0]
            for col, count in missing_cols.items():
                pct = count / len(self.dataset) * 100
                self.print_and_log(f"  - {col}: {count} ({pct:.2f}%)")
        else:
            self.print_and_log("  Không có missing values")
        
        # Duplicates
        duplicate_count = self.dataset.duplicated().sum()
        duplicate_pct = duplicate_count / len(self.dataset) * 100
        self.print_and_log(f"\nDuplicate rows: {duplicate_count} ({duplicate_pct:.2f}%)")
        
        if duplicate_count == 0:
            self.print_and_log("  Không có dòng trùng lặp")
        
        # Infinite values
        inf_count = 0
        inf_cols = []
        numeric_cols = self.dataset.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            col_inf = np.isinf(self.dataset[col]).sum()
            if col_inf > 0:
                inf_count += col_inf
                inf_cols.append(col)
        
        self.print_and_log(f"\nInfinite values: {inf_count}")
        if inf_count == 0:
            self.print_and_log("  Không có giá trị vô cực")
        else:
            for col in inf_cols:
                col_inf = np.isinf(self.dataset[col]).sum()
                self.print_and_log(f"  - {col}: {col_inf} infinite values")
        
        self.print_and_log("")

    #===============================================================================================================================
    # HÀM PHÂN TÍCH CÁC ĐẶC TRƯNG MỚI
    def analyze_engineered_features(self):
        """Phân tích các đặc trưng mới được tạo trong feature engineering."""
        self.print_and_log("3. PHÂN TÍCH CÁC ĐẶC TRƯNG MỚI")
        self.print_and_log("=" * 100)
        
        # Đặc trưng derivation - các đặc trưng được tạo mới từ các đặc trưng gốc
        derived_features = ['Age', 'Tenure', 'Total_Spent', 'TotalPurchases', 'AvgPerPurchase']
        derived_features = [f for f in derived_features if f in self.dataset.columns]
        
        if derived_features:
            self.print_and_log("A. ĐẶC TRƯNG DẪN XUẤT (Derived Features):")
            for feature in derived_features:
                stats = self.dataset[feature].describe()
                self.print_and_log(f"\n{feature}:")
                self.print_and_log(f"  - Range: [{stats['min']:.2f}, {stats['max']:.2f}]")
                self.print_and_log(f"  - Mean: {stats['mean']:.2f}, Median: {stats['50%']:.2f}")
                self.print_and_log(f"  - Std Dev: {stats['std']:.2f}")
                
                # Kiểm tra outliers đơn giản
                if feature == 'Age':
                    very_old = (self.dataset[feature] > 100).sum()
                    very_young = (self.dataset[feature] < 18).sum()
                    if very_old > 0:
                        self.print_and_log(f"  - Tuổi > 100: {very_old} cases")
                    if very_young > 0:
                        self.print_and_log(f"  - Tuổi < 18: {very_young} cases")
                
                elif feature == 'AvgPerPurchase':
                    zero_purchases = (self.dataset['TotalPurchases'] == 0).sum()
                    if zero_purchases > 0:
                        self.print_and_log(f"  - Khách hàng chưa mua gì: {zero_purchases} ({zero_purchases/len(self.dataset)*100:.1f}%)")

        # Đặc trưng log-transformed
        log_features = [col for col in self.dataset.columns if col.endswith('_Log')]
        
        if log_features:
            self.print_and_log(f"\nB. ĐẶC TRƯNG LOG-TRANSFORMED ({len(log_features)} features):")
            for feature in log_features:
                original_feature = feature.replace('_Log', '')
                if original_feature in self.dataset.columns:
                    original_skew = self.dataset[original_feature].skew()
                    log_skew = self.dataset[feature].skew()
                    improvement = abs(original_skew) - abs(log_skew)
                    
                    self.print_and_log(f"  - {feature}:")
                    self.print_and_log(f"    Original skewness: {original_skew:.3f}")
                    self.print_and_log(f"    Log skewness: {log_skew:.3f}")
                    self.print_and_log(f"    Improvement: {improvement:.3f} {'✓' if improvement > 0 else '✗'}")

        # Đặc trưng binned/grouped
        binned_features = [col for col in self.dataset.columns if 'Group' in col]
        
        if binned_features:
            self.print_and_log(f"\nC. ĐẶC TRƯNG PHÂN NHÓM ({len(binned_features)} features):")
            for feature in binned_features:
                value_counts = self.dataset[feature].value_counts()
                self.print_and_log(f"\n  {feature} ({len(value_counts)} nhóm):")
                for group, count in value_counts.items():
                    pct = count / len(self.dataset) * 100
                    self.print_and_log(f"    - {group}: {count} ({pct:.1f}%)")

        # Đặc trưng datetime
        datetime_features = [col for col in self.dataset.columns if col.startswith('Customer_')]
        
        if datetime_features:
            self.print_and_log(f"\nD. ĐẶC TRƯNG THỜI GIAN ({len(datetime_features)} features):")
            for feature in datetime_features:
                if feature in self.dataset.columns:
                    unique_count = self.dataset[feature].nunique()
                    self.print_and_log(f"  - {feature}: {unique_count} unique values")
                    
                    if feature == 'Customer_Season':
                        season_dist = self.dataset[feature].value_counts()
                        for season, count in season_dist.items():
                            pct = count / len(self.dataset) * 100
                            self.print_and_log(f"    * {season}: {count} ({pct:.1f}%)")

        self.print_and_log("")

    #===============================================================================================================================
    # HÀM PHÂN TÍCH TƯƠNG QUAN
    def analyze_correlation_patterns(self):
        """Phân tích mẫu tương quan giữa các đặc trưng."""
        self.print_and_log("4. PHÂN TÍCH TƯƠNG QUAN GIỮA CÁC ĐẶC TRƯNG")
        self.print_and_log("=" * 100)
        
        # Lấy các cột số
        numeric_cols = self.dataset.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) > 1:
            # Tính ma trận tương quan
            correlation_matrix = self.dataset[numeric_cols].corr()
            
            # Tìm các cặp tương quan cao
            high_corr_pairs = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_value = correlation_matrix.iloc[i, j]
                    if abs(corr_value) >= 0.8:
                        high_corr_pairs.append({
                            'feature1': correlation_matrix.columns[i],
                            'feature2': correlation_matrix.columns[j],
                            'correlation': corr_value
                        })
            
            self.print_and_log(f"Tổng số đặc trưng số: {len(numeric_cols)}")
            
            if high_corr_pairs:
                self.print_and_log(f"\nCác cặp đặc trưng có tương quan cao (|r| >= 0.8):")
                for pair in sorted(high_corr_pairs, key=lambda x: abs(x['correlation']), reverse=True):
                    self.print_and_log(f"  - {pair['feature1']} vs {pair['feature2']}: r = {pair['correlation']:.3f}")
                
                # Nhóm đặc trưng theo tương quan
                self.print_and_log(f"\nGợi ý cho Feature Selection:")
                self.print_and_log("  - Các cặp tương quan cao có thể gây multicollinearity")
                self.print_and_log("  - Xem xét loại bỏ một trong các cặp đặc trưng tương quan cao")
                
            else:
                self.print_and_log(f"\nKhông có cặp đặc trưng nào có tương quan quá cao (|r| >= 0.8)")
                self.print_and_log("  - Tốt cho việc tránh multicollinearity")

            # Phân tích nhóm đặc trưng
            spending_features = [col for col in numeric_cols if 'Mnt' in col and '_Log' not in col]
            log_spending_features = [col for col in numeric_cols if 'Mnt' in col and '_Log' in col]
            
            if len(spending_features) > 1:
                spending_corr = correlation_matrix.loc[spending_features, spending_features]
                mean_spending_corr = spending_corr.mean().mean()
                self.print_and_log(f"\nTương quan trung bình nhóm spending gốc: {mean_spending_corr:.3f}")
            
            if len(log_spending_features) > 1:
                log_spending_corr = correlation_matrix.loc[log_spending_features, log_spending_features]
                mean_log_corr = log_spending_corr.mean().mean()
                self.print_and_log(f"Tương quan trung bình nhóm spending log: {mean_log_corr:.3f}")

        self.print_and_log("")

    #===============================================================================================================================
    # HÀM PHÂN TÍCH CÁC BIẾN QUAN TRỌNG SAU FEATURE ENGINEERING
    def analyze_key_engineered_variables(self):
        """Phân tích các biến quan trọng sau feature engineering."""
        self.print_and_log("5. CÁC BIẾN QUAN TRỌNG SAU FEATURE ENGINEERING")
        self.print_and_log("=" * 100)
        
        # Các biến quan trọng để phân tích
        key_vars = ['Age', 'Income', 'Total_Spent', 'AvgPerPurchase', 'Tenure', 
                   'Age_Group', 'Income_Group', 'TotalChildren', 'HasChildren']
        
        for var in key_vars:
            if var in self.dataset.columns:
                self.print_and_log(f"{var}:")
                
                if pd.api.types.is_numeric_dtype(self.dataset[var]):
                    stats = self.dataset[var].describe()
                    self.print_and_log(f"  - Range: [{stats['min']:.2f}, {stats['max']:.2f}]")
                    self.print_and_log(f"  - Mean: {stats['mean']:.2f}, Median: {stats['50%']:.2f}")
                    self.print_and_log(f"  - Std Dev: {stats['std']:.2f}")
                    self.print_and_log(f"  - Skewness: {self.dataset[var].skew():.3f}")
                    
                    # Phân tích đặc biệt cho từng biến
                    if var == 'Age':
                        age_ranges = {
                            'Young (18-35)': (self.dataset[var] <= 35).sum(),
                            'Middle (36-55)': ((self.dataset[var] > 35) & (self.dataset[var] <= 55)).sum(),
                            'Senior (56+)': (self.dataset[var] > 55).sum()
                        }
                        for range_name, count in age_ranges.items():
                            pct = count / len(self.dataset) * 100
                            self.print_and_log(f"    * {range_name}: {count} ({pct:.1f}%)")
                    
                    elif var == 'Total_Spent':
                        quartiles = self.dataset[var].quantile([0.25, 0.5, 0.75])
                        self.print_and_log(f"    * Q1: {quartiles[0.25]:.0f}, Q2: {quartiles[0.5]:.0f}, Q3: {quartiles[0.75]:.0f}")
                        
                    elif var == 'HasChildren':
                        has_children = (self.dataset[var] == 1).sum()
                        no_children = (self.dataset[var] == 0).sum()
                        self.print_and_log(f"    * Có con: {has_children} ({has_children/len(self.dataset)*100:.1f}%)")
                        self.print_and_log(f"    * Không con: {no_children} ({no_children/len(self.dataset)*100:.1f}%)")
                        
                else:
                    # Categorical variable
                    value_counts = self.dataset[var].value_counts()
                    self.print_and_log(f"  - Categories: {len(value_counts)}")
                    for val, count in value_counts.head(3).items():
                        pct = count / len(self.dataset) * 100
                        self.print_and_log(f"    * '{val}': {count} ({pct:.1f}%)")
                    if len(value_counts) > 3:
                        self.print_and_log(f"    * ... và {len(value_counts) - 3} categories khác")
                
                self.print_and_log("")

    #===============================================================================================================================
    # HÀM TẠO BIỂU ĐỒ TRỰC QUAN
    def create_visualizations(self):
        """Tạo các biểu đồ trực quan cho dataset đã feature engineering."""
        self.print_and_log("6. TẠO BIỂU ĐỒ TRỰC QUAN")
        self.print_and_log("=" * 100)
        
        # Tạo thư mục figures
        fig_dir = self.output_dir / "figures"
        os.makedirs(fig_dir, exist_ok=True)
        
        try:
            # 1. Phân phối Age và Age_Group
            if 'Age' in self.dataset.columns:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Age distribution
                ax1.hist(self.dataset['Age'], bins=30, alpha=0.7, edgecolor='black', color='skyblue')
                ax1.set_title('Phân phối Age sau Feature Engineering')
                ax1.set_xlabel('Age')
                ax1.set_ylabel('Tần suất')
                ax1.grid(True, alpha=0.3)
                
                # Age_Group distribution (if exists)
                if 'Age_Group' in self.dataset.columns:
                    age_group_counts = self.dataset['Age_Group'].value_counts()
                    ax2.bar(range(len(age_group_counts)), age_group_counts.values, color='lightcoral')
                    ax2.set_title('Phân phối Age Groups')
                    ax2.set_xlabel('Age Groups')
                    ax2.set_ylabel('Số lượng')
                    ax2.set_xticks(range(len(age_group_counts)))
                    ax2.set_xticklabels(age_group_counts.index, rotation=45)
                    ax2.grid(True, alpha=0.3, axis='y')
                
                plt.tight_layout()
                age_file = fig_dir / "age_analysis.png"
                plt.savefig(age_file, dpi=150, bbox_inches='tight')
                plt.close()
                self.print_and_log(f"  Đã tạo biểu đồ phân tích tuổi: {age_file}")

            # 2. So sánh Original vs Log-transformed spending
            spending_cols = [col for col in self.dataset.columns if 'Mnt' in col and '_Log' not in col]
            log_spending_cols = [col for col in self.dataset.columns if 'Mnt' in col and '_Log' in col]
            
            if spending_cols and log_spending_cols:
                fig, axes = plt.subplots(2, 3, figsize=(18, 12))
                axes = axes.ravel()
                
                for i, (original, log_col) in enumerate(zip(spending_cols[:3], log_spending_cols[:3])):
                    # Original distribution
                    axes[i].hist(self.dataset[original], bins=30, alpha=0.7, color='lightblue', edgecolor='black')
                    axes[i].set_title(f'{original} (Original)')
                    axes[i].set_ylabel('Tần suất')
                    axes[i].grid(True, alpha=0.3)
                    
                    # Log-transformed distribution
                    axes[i+3].hist(self.dataset[log_col], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
                    axes[i+3].set_title(f'{log_col} (Log-transformed)')
                    axes[i+3].set_ylabel('Tần suất')
                    axes[i+3].grid(True, alpha=0.3)
                
                plt.tight_layout()
                spending_file = fig_dir / "spending_transformation_comparison.png"
                plt.savefig(spending_file, dpi=150, bbox_inches='tight')
                plt.close()
                self.print_and_log(f"  Đã tạo biểu đồ so sánh spending transformation: {spending_file}")

            # 3. Total_Spent vs AvgPerPurchase scatter plot
            if 'Total_Spent' in self.dataset.columns and 'AvgPerPurchase' in self.dataset.columns:
                plt.figure(figsize=(10, 8))
                
                # Scatter plot với màu sắc theo Income_Group
                if 'Income_Group' in self.dataset.columns:
                    groups = self.dataset['Income_Group'].unique()
                    colors = plt.cm.Set1(np.linspace(0, 1, len(groups)))
                    
                    for i, group in enumerate(groups):
                        group_data = self.dataset[self.dataset['Income_Group'] == group]
                        plt.scatter(group_data['Total_Spent'], group_data['AvgPerPurchase'], 
                                  alpha=0.6, c=[colors[i]], label=group)
                    plt.legend()
                else:
                    plt.scatter(self.dataset['Total_Spent'], self.dataset['AvgPerPurchase'], alpha=0.6)
                
                plt.xlabel('Total_Spent')
                plt.ylabel('AvgPerPurchase')
                plt.title('Total Spent vs Average Per Purchase')
                plt.grid(True, alpha=0.3)
                
                spending_scatter_file = fig_dir / "spending_patterns.png"
                plt.savefig(spending_scatter_file, dpi=150, bbox_inches='tight')
                plt.close()
                self.print_and_log(f"  Đã tạo biểu đồ mẫu chi tiêu: {spending_scatter_file}")

            # 4. Correlation heatmap (top features)
            numeric_cols = self.dataset.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) > 1:
                # Chọn các đặc trưng quan trọng để hiển thị
                important_features = ['Age', 'Income', 'Total_Spent', 'AvgPerPurchase', 'Tenure', 'TotalPurchases', 'TotalChildren']
                important_features = [f for f in important_features if f in numeric_cols]
                
                if len(important_features) > 2:
                    plt.figure(figsize=(12, 10))
                    correlation_matrix = self.dataset[important_features].corr()
                    
                    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                               square=True, linewidths=0.5, cbar_kws={"shrink": .8}, fmt='.2f')
                    plt.title('Ma trận tương quan - Các đặc trưng quan trọng')
                    plt.tight_layout()
                    
                    corr_file = fig_dir / "correlation_heatmap.png"
                    plt.savefig(corr_file, dpi=150, bbox_inches='tight')
                    plt.close()
                    self.print_and_log(f"  Đã tạo ma trận tương quan: {corr_file}")

            # 5. Feature Engineering impact visualization
            if 'Age_Group' in self.dataset.columns and 'Total_Spent' in self.dataset.columns:
                plt.figure(figsize=(12, 8))
                
                # Boxplot Total_Spent theo Age_Group
                sns.boxplot(x='Age_Group', y='Total_Spent', data=self.dataset)
                plt.title('Total Spent theo Age Groups')
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3, axis='y')
                plt.tight_layout()
                
                boxplot_file = fig_dir / "spending_by_age_groups.png"
                plt.savefig(boxplot_file, dpi=150, bbox_inches='tight')
                plt.close()
                self.print_and_log(f"  Đã tạo boxplot chi tiêu theo nhóm tuổi: {boxplot_file}")

        except Exception as e:
            self.print_and_log(f"  Lỗi khi tạo biểu đồ: {e}")
        
        self.print_and_log("")

    #===============================================================================================================================
    # HÀM TÓM TẮT VÀ ĐÁNH GIÁ
    def summarize_analysis(self):
        """Tóm tắt kết quả phân tích và đánh giá sẵn sàng cho bước tiếp theo."""
        self.print_and_log("7. TÓM TẮT & ĐÁNH GIÁ CUỐI CÙNG")
        self.print_and_log("=" * 100)
        
        # Thống kê tổng quan
        total_cols = len(self.dataset.columns)
        numeric_cols = len(self.dataset.select_dtypes(include=[np.number]).columns)
        categorical_cols = len(self.dataset.select_dtypes(include=['object', 'category']).columns)
        
        self.print_and_log(f"Dataset sau Feature Engineering:")
        self.print_and_log(f"  - {self.dataset.shape[0]:,} dòng × {total_cols} cột")
        self.print_and_log(f"  - {numeric_cols} cột số, {categorical_cols} cột phân loại")
        
        # Đánh giá chất lượng
        missing_count = self.dataset.isnull().sum().sum()
        duplicate_count = self.dataset.duplicated().sum()
        
        self.print_and_log(f"\nChất lượng dữ liệu:")
        self.print_and_log(f"  - Missing values: {'✓ Sạch' if missing_count == 0 else f'⚠ {missing_count} values'}")
        self.print_and_log(f"  - Duplicates: {'✓ Sạch' if duplicate_count == 0 else f'⚠ {duplicate_count} rows'}")
        
        # Đánh giá đặc trưng mới
        derived_features = [f for f in ['Age', 'Tenure', 'Total_Spent', 'AvgPerPurchase'] if f in self.dataset.columns]
        log_features = [col for col in self.dataset.columns if col.endswith('_Log')]
        binned_features = [col for col in self.dataset.columns if 'Group' in col]
        datetime_features = [col for col in self.dataset.columns if col.startswith('Customer_')]
        
        self.print_and_log(f"\nĐặc trưng đã được engineering:")
        self.print_and_log(f"  - Derived features: {len(derived_features)} đặc trưng")
        self.print_and_log(f"  - Log-transformed: {len(log_features)} đặc trưng")
        self.print_and_log(f"  - Binned/Grouped: {len(binned_features)} đặc trưng")
        self.print_and_log(f"  - Datetime features: {len(datetime_features)} đặc trưng")
        
        # Kiểm tra tương quan cao
        numeric_only = self.dataset.select_dtypes(include=[np.number]).columns.tolist()
        high_corr_count = 0
        
        if len(numeric_only) > 1:
            correlation_matrix = self.dataset[numeric_only].corr()
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    if abs(correlation_matrix.iloc[i, j]) >= 0.8:
                        high_corr_count += 1
        
        self.print_and_log(f"\nTương quan:")
        if high_corr_count > 0:
            self.print_and_log(f"  - Phát hiện {high_corr_count} cặp đặc trưng có tương quan cao (|r| >= 0.8)")
            self.print_and_log(f"  - Khuyến nghị: Cân nhắc feature selection để giảm multicollinearity")
        else:
            self.print_and_log(f"  - Không có cặp đặc trưng nào có tương quan quá cao")
        
        # Đánh giá sẵn sàng cho bước tiếp theo
        self.print_and_log(f"\nTrạng thái sẵn sàng cho Feature Scaling & Selection:")
        
        ready_indicators = []
        if missing_count == 0:
            ready_indicators.append("✓ Không có missing values")
        if duplicate_count == 0:
            ready_indicators.append("Không có duplicates")
        if len(derived_features) > 0:
            ready_indicators.append("Có đặc trưng derived")
        if len(log_features) > 0:
            ready_indicators.append("Có log transformation")
        
        if len(ready_indicators) >= 3:
            self.print_and_log("  Dataset sẵn sàng cho Feature Scaling & Selection")
        else:
            self.print_and_log("  Dataset cần kiểm tra thêm trước khi tiếp tục")
        
        for indicator in ready_indicators:
            self.print_and_log(f"    {indicator}")
        
        # Các bước tiếp theo được khuyến nghị
        self.print_and_log(f"\nCác bước tiếp theo được khuyến nghị:")
        self.print_and_log("  1. Feature Scaling (StandardScaler/MinMaxScaler)")
        self.print_and_log("  2. Feature Selection (correlation/importance-based)")
        
        if high_corr_count > 0:
            self.print_and_log("  3. Multicollinearity handling (VIF analysis)")
        
        self.print_and_log("  4. Dimensionality Reduction (PCA nếu cần)")
        self.print_and_log("  5. Model Training & Validation")
        
        self.print_and_log("")

    #===============================================================================================================================
    # HÀM CHẠY TOÀN BỘ PHÂN TÍCH
    def run_analysis(self):
        """Chạy toàn bộ quá trình phân tích dataset sau feature engineering."""
        start_time = time.time()
        
        # Tải dữ liệu
        if not self.load_data():
            return False
        
        try:
            # Chạy các bước phân tích
            self.analyze_basic_info()
            self.analyze_data_quality()
            self.analyze_engineered_features()
            self.analyze_correlation_patterns()
            self.analyze_key_engineered_variables()
            self.create_visualizations()
            self.summarize_analysis()
            
            # Thông tin thời gian thực hiện
            elapsed_time = time.time() - start_time
            self.print_and_log("=" * 100)
            self.print_and_log("HOÀN THÀNH PHÂN TÍCH DATASET SAU FEATURE ENGINEERING")
            self.print_and_log("=" * 100)
            self.print_and_log(f"Thời gian thực hiện  : {elapsed_time:.2f} giây")
            self.print_and_log(f"Báo cáo được lưu tại : {self.report_file}")
            
            return True
            
        except Exception as e:
            self.print_and_log(f"\nLỗi trong quá trình phân tích: {e}")
            import traceback
            self.print_and_log(traceback.format_exc())
            return False

#===================================================================================================================================
def main():
    """Hàm chính để chạy phân tích dataset sau feature engineering."""
    
    # Đường dẫn đến dataset đã qua feature engineering
    dataset_path = r"C:\Project\Machine_Learning\Machine_Learning\Dataset\Customer_Behavior_feature_engineered_20251016_141827.csv"
    
    try:
        print("Bắt đầu phân tích dataset sau Feature Engineering")
        
        # Khởi tạo analyzer
        analyzer = EngineeredDatasetAnalyzer(dataset_path)
        
        # Chạy phân tích
        success = analyzer.run_analysis()
        
        if success:
            print(f"\nPhân tích hoàn tất thành công")
            print(f"Báo cáo chi tiết: {analyzer.report_file}")
            print(f"Thư mục output: {analyzer.output_dir}")
        else:
            print("\nPhân tích gặp lỗi. Vui lòng kiểm tra báo cáo.")
            
    except Exception as e:
        print(f"\nLỗi khởi tạo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()