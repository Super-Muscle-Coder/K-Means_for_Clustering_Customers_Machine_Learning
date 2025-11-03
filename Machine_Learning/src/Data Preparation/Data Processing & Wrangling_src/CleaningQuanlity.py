import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from datetime import datetime
from pathlib import Path
import matplotlib.gridspec as gridspec
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'  
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'Segoe UI']
matplotlib.rcParams['axes.unicode_minus'] = False

class DatasetComparator:
    """
    Class để phân tích và so sánh dataset trước và sau khi làm sạch.
    Cung cấp trực quan hóa và báo cáo chi tiết về sự thay đổi.
    """
# ==================================================================================================================

    def __init__(self, original_dataset_path, cleaned_dataset_path):
        """
        Khởi tạo comparator với đường dẫn đến dataset gốc và dataset đã làm sạch.
        
        Args:
            original_dataset_path (str): Đường dẫn đến file CSV gốc
            cleaned_dataset_path (str): Đường dẫn đến file CSV đã làm sạch
        """
        self.original_dataset_path = original_dataset_path
        self.cleaned_dataset_path = cleaned_dataset_path
        self.original_data = None
        self.cleaned_data = None
        
        # Thiết lập style cho các biểu đồ
        plt.style.use('seaborn-v0_8-whitegrid')
        self.colors = {
            'original': '#FF9999',   # Màu đỏ nhạt cho dataset gốc
            'cleaned': '#66B2FF',    # Màu xanh cho dataset đã làm sạch
            'highlight': '#FF5733',  # Màu highlight
            'accent': '#2E86C1'      # Màu accent
        }
        
        # Thiết lập thư mục output
        self.output_dir = Path(r"C:\Project\Machine_Learning\Machine_Learning\report\Data Processing & Wrangling_report")
        
        # Tạo thư mục cho figures
        self.fig_dir = Path(r"C:\Project\Machine_Learning\Machine_Learning\graph\Data Processing & Wrangling_graph\Cleaning_Quanlity")
        os.makedirs(self.fig_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Xuất File báo cáo
        self.report_file = self.output_dir / "Cleaning_Quanlity.log"
        # Xóa báo cáo cũ nếu có
        if os.path.exists(self.report_file):
            os.remove(self.report_file)
# ==================================================================================================================

    def print_and_log(self, text, bold=False):
        """In ra console và ghi vào file báo cáo."""
        if bold:
            print("\033[1m" + text + "\033[0m")  
            formatted_text = "=" * 100 + "\n" + text + "\n" + "=" * 100
        else:
            print(text)
            formatted_text = text
            
        with open(self.report_file, "a", encoding="utf-8") as f:
            f.write(formatted_text + "\n")
# ==================================================================================================================

    def load_data(self):
        """Tải cả dataset gốc và dataset đã làm sạch."""
        try:
            self.print_and_log("BÁO CÁO SO SÁNH DATASET TRƯỚC VÀ SAU KHI LÀM SẠCH", bold=True)
            
            # Tải dataset gốc
            self.print_and_log(f"Đọc dataset gốc từ: {self.original_dataset_path}")
            self.original_data = pd.read_csv(self.original_dataset_path, sep='\t')
            
            # Tải dataset đã làm sạch
            self.print_and_log(f"Đọc dataset đã làm sạch từ: {self.cleaned_dataset_path}")
            self.cleaned_data = pd.read_csv(self.cleaned_dataset_path)
            
            # Parse datetime nếu 'Dt_Customer' là cột datetime
            if 'Dt_Customer' in self.cleaned_data.columns:
                try:
                    self.cleaned_data['Dt_Customer'] = pd.to_datetime(self.cleaned_data['Dt_Customer'])
                except Exception:
                    self.print_and_log("  Lưu ý: Không thể chuyển đổi 'Dt_Customer' sang datetime")
            
            self.print_and_log("\nTải dữ liệu thành công:")
            self.print_and_log(f"- Dataset gốc: {self.original_data.shape[0]:,} dòng × {self.original_data.shape[1]} cột")
            self.print_and_log(f"- Dataset đã làm sạch: {self.cleaned_data.shape[0]:,} dòng × {self.cleaned_data.shape[1]} cột")
            rows_removed = self.original_data.shape[0] - self.cleaned_data.shape[0]
            self.print_and_log(f"- Dòng đã loại bỏ: {rows_removed:,} ({rows_removed/self.original_data.shape[0]*100:.2f}%)")
            
            cols_diff = self.cleaned_data.shape[1] - self.original_data.shape[1]
            if cols_diff > 0:
                self.print_and_log(f"- Cột mới thêm: {cols_diff}")
            else:
                self.print_and_log(f"- Cột đã loại bỏ: {abs(cols_diff)}")
            
            return True
        except Exception as e:
            self.print_and_log(f"Lỗi khi tải dữ liệu: {e}")
            return False

# ==================================================================================================================

    def compare_basic_info(self):
        """So sánh thông tin cơ bản giữa hai dataset."""
        self.print_and_log("\n1. SO SÁNH THÔNG TIN CƠ BẢN", bold=True)
        
        # So sánh cột giữa hai dataset
        orig_cols = set(self.original_data.columns)
        clean_cols = set(self.cleaned_data.columns)
        
        # Tìm cột chỉ có trong mỗi dataset
        only_orig_cols = orig_cols - clean_cols
        only_clean_cols = clean_cols - orig_cols
        common_cols = orig_cols.intersection(clean_cols)
        
        self.print_and_log("\nCấu trúc cột:")
        self.print_and_log(f"- Số cột chung giữa hai dataset: {len(common_cols)}")
        self.print_and_log(f"- Cột chỉ có trong dataset gốc: {len(only_orig_cols)} {list(only_orig_cols) if len(only_orig_cols) <= 10 else '(quá nhiều để hiển thị)'}")
        self.print_and_log(f"- Cột mới trong dataset đã làm sạch: {len(only_clean_cols)} {list(only_clean_cols)}")
        
        # So sánh kiểu dữ liệu
        self.print_and_log("\nSo sánh kiểu dữ liệu:")
        orig_dtypes = self.original_data.dtypes.value_counts().to_dict()
        clean_dtypes = self.cleaned_data.dtypes.value_counts().to_dict()
        
        all_dtypes = set(orig_dtypes.keys()) | set(clean_dtypes.keys())
        
        dtype_table = []
        for dtype in all_dtypes:
            orig_count = orig_dtypes.get(dtype, 0)
            clean_count = clean_dtypes.get(dtype, 0)
            change = clean_count - orig_count
            change_str = f"{change:+d}" if change else "0"
            dtype_table.append([str(dtype), orig_count, clean_count, change_str])
        
        # Format as table
        self.print_and_log(f"{'Kiểu dữ liệu':<20} | {'Gốc':<8} | {'Đã làm sạch':<12} | {'Thay đổi':<8}")
        self.print_and_log("-" * 55)
        for row in dtype_table:
            self.print_and_log(f"{row[0]:<20} | {row[1]:<8} | {row[2]:<12} | {row[3]:<8}")
        
        # Kiểm tra thay đổi kiểu dữ liệu cho các cột chung
        dtype_changes = []
        for col in common_cols:
            orig_type = self.original_data[col].dtype
            clean_type = self.cleaned_data[col].dtype
            
            if orig_type != clean_type:
                dtype_changes.append((col, orig_type, clean_type))
        
        if dtype_changes:
            self.print_and_log("\nThay đổi kiểu dữ liệu cụ thể:")
            for col, orig_type, clean_type in dtype_changes:
                self.print_and_log(f"- {col}: {orig_type} → {clean_type}")
        
        # So sánh bộ nhớ
        orig_mem = self.original_data.memory_usage(deep=True).sum() / (1024 * 1024)
        clean_mem = self.cleaned_data.memory_usage(deep=True).sum() / (1024 * 1024)
        mem_change = clean_mem - orig_mem
        mem_change_pct = (mem_change / orig_mem) * 100
        
        self.print_and_log(f"\nBộ nhớ sử dụng:")
        self.print_and_log(f"- Original: {orig_mem:.2f} MB")
        self.print_and_log(f"- Cleaned : {clean_mem:.2f} MB")
        self.print_and_log(f"- Thay đổi: {mem_change:+.2f} MB ({mem_change_pct:+.2f}%)")

        # Tạo biểu đồ so sánh cơ bản
        self._create_basic_comparison_charts()
# ==================================================================================================================

    def compare_data_quality(self):
        """So sánh chất lượng dữ liệu giữa hai dataset."""
        self.print_and_log("\n2. SO SÁNH CHẤT LƯỢNG DỮ LIỆU", bold=True)
    
        # Missing values
        orig_missing = self.original_data.isnull().sum().sum()
        clean_missing = self.cleaned_data.isnull().sum().sum()
    
        orig_missing_pct = orig_missing / (self.original_data.shape[0] * self.original_data.shape[1]) * 100
        clean_missing_pct = clean_missing / (self.cleaned_data.shape[0] * self.cleaned_data.shape[1]) * 100
    
        self.print_and_log("\nMissing values:")
        self.print_and_log(f"- Original: {orig_missing:,} giá trị ({orig_missing_pct:.4f}%)")
        self.print_and_log(f"- Cleaned : {clean_missing:,} giá trị ({clean_missing_pct:.4f}%)")
        missing_diff = orig_missing - clean_missing
        if missing_diff > 0:
            self.print_and_log(f"- Đã xử lý: {missing_diff:,} giá trị missing")
    
        # =====================================
        # PHÂN TÍCH TRÙNG LẶP CHI TIẾT
        # =====================================
        self.print_and_log("\nPhân tích Duplicates:")
    
        # 1. Kiểm tra duplicates với ID (full-row)
        orig_dups_with_id = self.original_data.duplicated().sum()
        clean_dups = self.cleaned_data.duplicated().sum()
    
        # 2. Kiểm tra duplicates KHÔNG có ID (feature duplicates)
        if 'ID' in self.original_data.columns:
            feature_cols = [col for col in self.original_data.columns if col != 'ID']
            orig_feature_dups = self.original_data.duplicated(subset=feature_cols).sum()
        else:
            orig_feature_dups = orig_dups_with_id
    
        # 3. Số duplicates "ẩn" bởi ID
        hidden_dups = orig_feature_dups - orig_dups_with_id
    
        self.print_and_log("\nKiểm tra VỚI ID (full-row duplicates):")
        self.print_and_log(f"- Original: {orig_dups_with_id:,} dòng")
        self.print_and_log(f"- Cleaned : {clean_dups:,} dòng")
    
        self.print_and_log("\nKiểm tra KHÔNG CÓ ID (feature duplicates):")
        self.print_and_log(f"- Original: {orig_feature_dups:,} dòng")
        self.print_and_log(f"- Lý do   : Khi bỏ ID, phát hiện thêm {hidden_dups} duplicates ẩn")
    
        if orig_feature_dups > 0:
            self.print_and_log(f"\nQUAN TRỌNG: Dataset gốc có {orig_feature_dups} dòng trùng lặp")
            self.print_and_log(f"  → Chỉ phát hiện được khi loại bỏ cột ID")
            self.print_and_log(f"  → Các dòng này có features giống nhau nhưng ID khác nhau")
    
        # Thông tin chi tiết từ báo cáo
        self.print_and_log("\nThông tin dòng bị loại bỏ (từ báo cáo làm sạch):")
        self.print_and_log(f"- {orig_feature_dups} dòng do trùng lặp feature (phát hiện khi bỏ ID)")
        self.print_and_log("- 21 dòng do trùng lặp full-row (sau khi đã loại ID)")
        self.print_and_log("- 3 dòng do là outliers (Income: 1, Year_Birth: 2)")
        self.print_and_log("- 24 dòng được tách do có missing values")
    
        # Missing values theo cột
        orig_missing_cols = self.original_data.isnull().sum()
        orig_missing_cols = orig_missing_cols[orig_missing_cols > 0]
    
        if len(orig_missing_cols) > 0:
            self.print_and_log("\nMissing values theo cột (Original):")
            for col, count in orig_missing_cols.items():
                pct = count / len(self.original_data) * 100
                self.print_and_log(f"  - {col}: {count:,} ({pct:.2f}%)")
    
        # Tạo biểu đồ với thông tin duplicates đầy đủ
        self._create_data_quality_charts(
            orig_dups_with_id=orig_dups_with_id,
            orig_feature_dups=orig_feature_dups, 
            hidden_dups=hidden_dups,
            clean_dups=clean_dups
        )
# ==================================================================================================================

    def analyze_categorical_standardization(self):
        """Phân tích việc chuẩn hóa các biến phân loại."""
        self.print_and_log("\n3. CHUẨN HÓA BIẾN PHÂN LOẠI", bold=True)
        
        # Phân tích Marital_Status
        if 'Marital_Status' in self.original_data.columns and 'Marital_Status' in self.cleaned_data.columns:
            orig_marital = self.original_data['Marital_Status'].value_counts()
            clean_marital = self.cleaned_data['Marital_Status'].value_counts()
            
            orig_categories = set(self.original_data['Marital_Status'].unique())
            clean_categories = set(self.cleaned_data['Marital_Status'].unique())
            
            self.print_and_log("\nChuẩn hóa Marital_Status:")
            self.print_and_log(f"- Số danh mục trước: {len(orig_categories)}")
            self.print_and_log(f"- Số danh mục sau: {len(clean_categories)}")
            
            # Các danh mục chỉ có trong dataset gốc
            only_orig_cats = orig_categories - clean_categories
            if only_orig_cats:
                self.print_and_log(f"- Danh mục đã được map: {', '.join(only_orig_cats)}")
                
            # Tạo bảng so sánh
            self.print_and_log("\nSo sánh phân phối Marital_Status:")
            self.print_and_log(f"{'Giá trị':<20} | {'Original':<12} | {'Cleaned':<12} | {'Thay đổi':<12}")
            self.print_and_log("-" * 60)
            
            all_cats = sorted(orig_categories.union(clean_categories))
            for cat in all_cats:
                orig_count = orig_marital.get(cat, 0)
                clean_count = clean_marital.get(cat, 0)
                
                orig_pct = orig_count / len(self.original_data) * 100 if orig_count > 0 else 0
                clean_pct = clean_count / len(self.cleaned_data) * 100 if clean_count > 0 else 0
                
                diff = clean_count - orig_count
                diff_pct = clean_pct - orig_pct
                
                self.print_and_log(f"{cat:<20} | {orig_count:>5} ({orig_pct:>5.2f}%) | {clean_count:>5} ({clean_pct:>5.2f}%) | {diff:+d} ({diff_pct:+.2f}%)")
            
            # Tạo biểu đồ so sánh Marital Status
            self._create_categorical_standardization_chart()

# =================================================================================================================
    def analyze_outliers_distribution(self):
        """Phân tích và so sánh phân phối các biến có outliers."""
        self.print_and_log("\n2.5. PHÂN TÍCH PHÂN PHỐI CÁC BIẾN CÓ OUTLIERS", bold=True)

        outlier_vars = [
            ('Income', 'THU NHẬP', 1),
            ('Year_Birth', 'NĂM SINH', 2),
            ('MntMeatProducts', 'CHI TIÊU THỊT', 29),
            ('MntFruits', 'CHI TIÊU TRÁI CÂY', 96),
            ('MntSweetProducts', 'CHI TIÊU ĐỒ NGỌT', 105),
            ('MntGoldProds', 'CHI TIÊU VÀNG BẠC', 49),
            ('MntFishProducts', 'CHI TIÊU HẢI SẢN', 72),
            ('NumWebPurchases', 'MUA HÀNG ONLINE', 4),
            ('NumCatalogPurchases', 'MUA QUA CATALOG', 4),
            ('NumDealsPurchases', 'MUA KHUYẾN MÃI', 24),
            ('NumWebVisitsMonth', 'TRUY CẬP WEB/THÁNG', 3)
        ]

        self.print_and_log(f"\nPhát hiện {len(outlier_vars)} biến có outliers:")
        for var, var_vn, count in outlier_vars:
            self.print_and_log(f"  - {var} ({var_vn}): {count} outliers")

        self.print_and_log("\nTạo biểu đồ so sánh phân phối")

        for idx, (var, var_vn, outlier_count) in enumerate(outlier_vars, 1): 
            if var in self.original_data.columns and var in self.cleaned_data.columns:
                print(f"DEBUG: Creating chart for {var} with idx={idx}")  
                self._create_outlier_distribution_chart(var, var_vn, outlier_count, idx)

        self.print_and_log(f"\n  Đã tạo {len(outlier_vars)} biểu đồ phân phối outliers")

# ==================================================================================================================

    def analyze_encoding_transformations(self):
        """Phân tích các biến đổi encoding - TÁCH RIÊNG Ordinal và One-hot."""
        self.print_and_log("\n4. PHÂN TÍCH CÁC BIẾN ĐỔI ENCODING", bold=True)
    
        # =====================================
        # PHẦN 1: ORDINAL ENCODING (Education)
        # =====================================
        self.print_and_log("\n4.1. ORDINAL ENCODING - Education")
    
        if 'Education_ord' in self.cleaned_data.columns and 'Education' in self.cleaned_data.columns:
            mapping = self.cleaned_data.groupby('Education')['Education_ord'].first().sort_values()
        
            self.print_and_log("\nMapping Education → Education_ord (thấp → cao):")
            self.print_and_log(f"{'Giá trị':<15} | {'Ordinal':<8} | {'Số lượng':<10} | {'%':<8}")
            self.print_and_log("-" * 50)
        
            for edu, ord_val in mapping.items():
                count = (self.cleaned_data['Education'] == edu).sum()
                pct = count / len(self.cleaned_data) * 100
                self.print_and_log(f"{edu:<15} | {ord_val:<8} | {count:<10} | {pct:.1f}%")
        
            self.print_and_log("\nPhương pháp: Ordinal Encoding")
            self.print_and_log("- Lý do   : Education có thứ bậc rõ ràng")
            self.print_and_log("- Kết quả : 1 cột mới (Education_ord)")
        
            # Tạo biểu đồ Ordinal
            self._create_ordinal_encoding_chart()
    
        # =====================================
        # PHẦN 2: ONE-HOT ENCODING (Marital_Status)
        # =====================================
        self.print_and_log("\n4.2. ONE-HOT ENCODING - Marital_Status")
    
        dummy_cols = [col for col in self.cleaned_data.columns if col.startswith('Marital_') and col != 'Marital_Status_Grouped']
    
        if dummy_cols:
            self.print_and_log(f"\nKết quả: {len(dummy_cols)} cột dummy")
            self.print_and_log(f"Danh sách: {', '.join(dummy_cols)}")
        
            self.print_and_log("\nPhân phối các cột dummy:")
            self.print_and_log(f"{'Cột':<25} | {'Count':<8} | {'%':<10}")
            self.print_and_log("-" * 50)
        
            for col in sorted(dummy_cols):
                if pd.api.types.is_numeric_dtype(self.cleaned_data[col]):
                    count = (self.cleaned_data[col] == 1).sum()
                    pct = count / len(self.cleaned_data) * 100
                    self.print_and_log(f"{col:<25} | {count:<8} | {pct:.2f}%")
        
            self.print_and_log("\nPhương pháp: One-Hot Encoding")
            self.print_and_log("- Lý do   : Marital_Status không có thứ bậc")
            self.print_and_log("- Kết quả : 6 cột dummy (binary 0/1)")
        
            # Tạo biểu đồ One-hot
            self._create_onehot_encoding_chart(dummy_cols)
    
        # =====================================
        # PHẦN 3: DATA TYPE CONVERSION
        # =====================================
        if 'Dt_Customer' in self.cleaned_data.columns:
            if pd.api.types.is_datetime64_dtype(self.cleaned_data['Dt_Customer']):
                self.print_and_log("\n4.3. CHUYỂN ĐỔI KIỂU DỮ LIỆU - Dt_Customer")
                self.print_and_log("- Từ: string (object)")
                self.print_and_log("- Đến: datetime64[ns]")
            
                dt_years = self.cleaned_data['Dt_Customer'].dt.year.value_counts().sort_index()
            
                self.print_and_log("\nPhân phối theo năm:")
                for year, count in dt_years.items():
                    pct = count / len(self.cleaned_data) * 100
                    self.print_and_log(f"  - {year}: {count} ({pct:.2f}%)")
            
                self._create_datetime_distribution_chart()
# ==================================================================================================================

    def analyze_key_variables(self):
        """So sánh phân phối của các biến quan trọng trước và sau khi làm sạch."""
        self.print_and_log("\n5. SO SÁNH PHÂN PHỐI BIẾN QUAN TRỌNG", bold=True)
        
        # Chọn các biến số quan trọng để so sánh
        numeric_vars = ['Income', 'Year_Birth', 'MntWines', 'MntMeatProducts']
        
        for var in numeric_vars:
            if var in self.original_data.columns and var in self.cleaned_data.columns:
                orig_data = self.original_data[var].dropna()
                clean_data = self.cleaned_data[var].dropna()
                
                # Tính toán thống kê
                orig_stats = orig_data.describe()
                clean_stats = clean_data.describe()
                
                self.print_and_log(f"\n{var}:")
                self.print_and_log(f"{'Thống kê':<10} | {'Original':<15} | {'Cleaned':<15} | {'Thay đổi':<15}")
                self.print_and_log("-" * 60)
                
                for stat in ['min', 'max', 'mean', 'std', '50%']:
                    orig_val = orig_stats[stat]
                    clean_val = clean_stats[stat]
                    diff = clean_val - orig_val
                    diff_pct = (diff / orig_val) * 100 if orig_val != 0 else float('inf')
                    
                    self.print_and_log(f"{stat:<10} | {orig_val:<15.2f} | {clean_val:<15.2f} | {diff:+.2f} ({diff_pct:+.2f}%)")
                
                # Kiểm tra outliers
                if var == 'Income':
                    orig_high = (orig_data >= 100000).sum()
                    clean_high = (clean_data >= 100000).sum()
                    self.print_and_log(f"\nOutliers ({var}):")
                    self.print_and_log(f"- Giá trị cao (>=100k): {orig_high} → {clean_high} ({orig_high-clean_high:+d})")
                    
                    orig_extreme = (orig_data >= 500000).sum()
                    clean_extreme = (clean_data >= 500000).sum()
                    self.print_and_log(f"- Giá trị cực cao (>=500k): {orig_extreme} → {clean_extreme} ({orig_extreme-clean_extreme:+d})")
                
                elif var == 'Year_Birth':
                    orig_old = (orig_data < 1900).sum()
                    clean_old = (clean_data < 1900).sum()
                    self.print_and_log(f"\nOutliers ({var}):")
                    self.print_and_log(f"- Năm sinh cũ (< 1900): {orig_old} → {clean_old} ({orig_old-clean_old:+d})")
                
# ==================================================================================================================
# HÀM TỔNG KẾT CUỐI CÙNG
    def summarize_changes(self):
        """Tổng kết các thay đổi từ data wrangling."""
        self.print_and_log("\n6. TÓM TẮT CÁC THAY ĐỔI", bold=True)
        
        # Thống kê tổng thể
        orig_rows = self.original_data.shape[0]
        orig_cols = self.original_data.shape[1]
        clean_rows = self.cleaned_data.shape[0]
        clean_cols = self.cleaned_data.shape[1]
        
        rows_removed = orig_rows - clean_rows
        rows_removed_pct = rows_removed / orig_rows * 100
        
        self.print_and_log("\nThống kê tổng quan:")
        self.print_and_log(f"- Dataset gốc: {orig_rows:,} dòng × {orig_cols} cột")
        self.print_and_log(f"- Dataset đã làm sạch: {clean_rows:,} dòng × {clean_cols} cột")
        self.print_and_log(f"- Đã loại bỏ: {rows_removed:,} dòng ({rows_removed_pct:.2f}%)")
        
        # Tóm tắt các thay đổi chính
        self.print_and_log("\nCác thay đổi chính đã thực hiện:")
        self.print_and_log("1. Tách 24 dòng có missing values thành mini dataset")
        self.print_and_log("2. Loại bỏ 182 dòng trùng lặp feature")
        self.print_and_log("3. Loại bỏ 3 dòng outliers (Income: 1, Year_Birth: 2)")
        self.print_and_log("4. Chuẩn hóa Marital_Status (thay đổi 7 giá trị)")
        self.print_and_log("5. Chuyển đổi Dt_Customer từ object sang datetime64[ns]")
        self.print_and_log("6. Ordinal encoding Education thành Education_ord")
        self.print_and_log("7. One-hot encoding Marital_Status thành 6 cột dummy")
        self.print_and_log("8. Loại bỏ 21 dòng trùng lặp hoàn toàn (sau khi loại ID)")
        self.print_and_log("9. Loại bỏ 10 cột không phục vụ nhân khẩu học")
        
        # Tổng hợp cải thiện chất lượng dữ liệu
        orig_missing = self.original_data.isnull().sum().sum()
        clean_missing = self.cleaned_data.isnull().sum().sum()
        
        orig_dups = self.original_data.duplicated().sum()
        clean_dups = self.cleaned_data.duplicated().sum()
        
        self.print_and_log("\nCải thiện chất lượng dữ liệu:")
        self.print_and_log(f"- Missing values : {orig_missing:,} → {clean_missing:,} ({orig_missing-clean_missing:+,})")
        self.print_and_log(f"- Duplicated rows: {orig_dups:,} → {clean_dups:,} ({orig_dups-clean_dups:+,})")
        
        # Tạo biểu đồ tổng hợp
        self._create_summary_improvement_chart()
# ==================================================================================================================
# BIỂU ĐỒ SO SÁNH CƠ BẢN
    def _create_basic_comparison_charts(self):
        """Tạo các biểu đồ so sánh cơ bản."""
        try:
            # So sánh kích thước dataset
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Biểu đồ 1: So sánh số dòng
            labels = ['Original', 'Cleaned']
            values = [self.original_data.shape[0], self.cleaned_data.shape[0]]
            bars = ax1.bar(labels, values, color=[self.colors['original'], self.colors['cleaned']])
            ax1.set_title('So sánh số dòng', fontweight='bold')
            ax1.set_ylabel('Số dòng')
            
            # Thêm labels và % giảm
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + values[0]*0.01,
                        f'{height:,}',
                        ha='center', va='bottom')

            # Biểu đồ 2: So sánh số cột
            values = [self.original_data.shape[1], self.cleaned_data.shape[1]]
            bars = ax2.bar(labels, values, color=[self.colors['original'], self.colors['cleaned']])
            ax2.set_title('So sánh số cột', fontweight='bold')
            ax2.set_ylabel('Số cột')
            
            # Thêm labels
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                        f'{height}',
                        ha='center', va='bottom')
            
            
            plt.suptitle('SO SÁNH KÍCH THƯỚC DATASET', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            # Lưu biểu đồ
            chart_path = self.fig_dir / "1_dataset_size_comparison.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # So sánh kiểu dữ liệu
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Tìm tất cả các kiểu dữ liệu
            orig_dtypes = self.original_data.dtypes.value_counts().to_dict()
            clean_dtypes = self.cleaned_data.dtypes.value_counts().to_dict()
            
            all_dtypes = sorted(set(orig_dtypes.keys()) | set(clean_dtypes.keys()), key=str)
            
            # Chuẩn bị dữ liệu
            orig_counts = [orig_dtypes.get(dtype, 0) for dtype in all_dtypes]
            clean_counts = [clean_dtypes.get(dtype, 0) for dtype in all_dtypes]
            
            x = np.arange(len(all_dtypes))
            width = 0.35
            
            # Tạo bars
            rects1 = ax.bar(x - width/2, orig_counts, width, label='Original', color=self.colors['original'])
            rects2 = ax.bar(x + width/2, clean_counts, width, label='Cleaned', color=self.colors['cleaned'])
            
            # Thêm labels và text
            ax.set_title('So sánh số cột theo kiểu dữ liệu', fontweight='bold')
            ax.set_ylabel('Số cột')
            ax.set_xticks(x)
            ax.set_xticklabels([str(dtype) for dtype in all_dtypes])
            ax.legend()
            
            # Thêm số liệu trên bars
            def autolabel(rects):
                for rect in rects:
                    height = rect.get_height()
                    if height > 0:
                        ax.annotate(f'{height}',
                                    xy=(rect.get_x() + rect.get_width() / 2, height),
                                    xytext=(0, 3),
                                    textcoords="offset points",
                                    ha='center', va='bottom')
            
            autolabel(rects1)
            autolabel(rects2)
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Lưu biểu đồ
            chart_path = self.fig_dir / "2_dtype_comparison.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.print_and_log(f"\n  Đã tạo biểu đồ so sánh cơ bản (2).")
            
        except Exception as e:
            self.print_and_log(f"  Lỗi khi tạo biểu đồ so sánh cơ bản: {e}")
# ==================================================================================================================

    def _create_data_quality_charts(self, orig_dups_with_id, orig_feature_dups, hidden_dups, clean_dups):
        """Tạo các biểu đồ so sánh chất lượng dữ liệu - TÁCH RIÊNG Missing và Duplicates."""
        try:
            # =====================================
            # BIỂU ĐỒ 1: MISSING VALUES
            # =====================================
            fig, ax = plt.subplots(figsize=(10, 7))
        
            orig_missing = self.original_data.isnull().sum().sum()
            clean_missing = self.cleaned_data.isnull().sum().sum()
        
            labels = ['Original', 'Cleaned']
            values = [orig_missing, clean_missing]
            colors = [self.colors['original'], self.colors['cleaned']]
        
            bars = ax.bar(labels, values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
            ax.set_title('SO SÁNH MISSING VALUES TRƯỚC VÀ SAU LÀM SẠCH', fontweight='bold', fontsize=14)
            ax.set_ylabel('Số Missing Values', fontweight='bold', fontsize=12)
        
            # Thêm số liệu
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.02,
                        f'{int(val):,}', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        
            plt.tight_layout()
            chart_path = self.fig_dir / "3_missing_values_comparison.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
        
            # =====================================
            # BIỂU ĐỒ 2: DUPLICATES (CHI TIẾT)
            # =====================================
            fig = plt.figure(figsize=(16, 8))
            gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.3)
        
            # SUBPLOT 1: So sánh WITH vs WITHOUT ID
            ax1 = fig.add_subplot(gs[0, 0])
        
            categories = ['Original\n(Với ID)', 'Original\n(Không ID)', 'Ẩn bởi ID', 'Cleaned']
            values = [orig_dups_with_id, orig_feature_dups, hidden_dups, clean_dups]
            colors_dup = ['#FFB6C1', '#FF6B6B', '#DC143C', '#90EE90']
        
            bars = ax1.bar(categories, values, color=colors_dup, alpha=0.8, 
                           edgecolor='black', linewidth=1.5)
            ax1.set_ylabel('Số dòng trùng lặp', fontweight='bold', fontsize=11)
            ax1.set_title('Phân tích Feature Duplicates: Ảnh hưởng của ID', 
                          fontweight='bold', fontsize=12)
        
            # Thêm số liệu
            for bar, val in zip(bars, values):
                if val > 0:
                    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.02,
                            f'{int(val)}', ha='left', va='bottom', fontweight='bold', fontsize=11)
        
        
            # SUBPLOT 2: Tổng hợp loại bỏ (182 + 21)
            ax2 = fig.add_subplot(gs[0, 1])
        
            # Dữ liệu tổng hợp
            removal_stages = ['Feature\nDuplicates\n(Bước 2)', 'Full-row\nDuplicates\n(Bước 8)', 'Tổng cộng\nđã loại bỏ']
            removal_counts = [182, 21, 203]
            removal_colors = ['#FF6B6B', '#FF8C94', '#8B0000']
        
            bars2 = ax2.bar(removal_stages, removal_counts, color=removal_colors, alpha=0.8,
                           edgecolor='black', linewidth=1.5)
            ax2.set_ylabel('Số dòng đã loại bỏ', fontweight='bold', fontsize=11)
            ax2.set_title('Tổng hợp Loại bỏ Duplicates (2 giai đoạn)', 
                          fontweight='bold', fontsize=12)
        
            # Thêm số liệu
            for bar, val in zip(bars2, removal_counts):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(removal_counts)*0.02,
                        f'{val}', ha='left', va='bottom', fontweight='bold', fontsize=12)
       
        
            plt.suptitle('PHÂN TÍCH CHI TIẾT DUPLICATES', fontsize=16, fontweight='bold')
        
            chart_path = self.fig_dir / "4_duplicates_detailed_analysis.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
        
            self.print_and_log(f"\n  Đã tạo biểu đồ Missing Values")
            self.print_and_log(f"  Đã tạo biểu đồ Duplicates ")
        
        except Exception as e:
            self.print_and_log(f"  Lỗi khi tạo biểu đồ chất lượng dữ liệu: {e}")
            import traceback
            self.print_and_log(f"  Chi tiết: {traceback.format_exc()}")

# ==================================================================================================================
# BIỂU ĐỒ ONE-HOT ENCODING
    def _create_onehot_encoding_chart(self, dummy_cols):
        """Vẽ biểu đồ cho One-Hot Encoding (Marital_Status)."""
        try:
            counts = []
            for col in dummy_cols:
                if pd.api.types.is_numeric_dtype(self.cleaned_data[col]):
                    count = (self.cleaned_data[col] == 1).sum()
                    pct = count / len(self.cleaned_data) * 100
                    counts.append((col, count, pct))
        
            counts.sort(key=lambda x: x[1], reverse=True)
        
            fig, ax = plt.subplots(figsize=(12, 7))
        
            cols = [item[0] for item in counts]
            values = [item[1] for item in counts]
        
            bars = ax.bar(range(len(cols)), values, color=self.colors['cleaned'], 
                          alpha=0.8, edgecolor='black', linewidth=1.5)
        
            ax.set_title('ONE-HOT ENCODING: Marital_Status → Dummy Variables', 
                         fontweight='bold', fontsize=14)
            ax.set_ylabel('Số lượng (giá trị = 1)', fontweight='bold')
            ax.set_xticks(range(len(cols)))
            ax.set_xticklabels(cols, rotation=45, ha='right')
        
            # Thêm labels
            for i, (col, count, pct) in enumerate(counts):
                ax.text(i, count + max(values)*0.01, f'{count}\n({pct:.1f}%)',
                       ha='left', va='bottom', fontweight='bold')
        
            plt.tight_layout()
        
            chart_path = self.fig_dir / "7_onehot_encoding_marital.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
        
            self.print_and_log(f"  Đã tạo biểu đồ One-Hot Encoding")
        
        except Exception as e:
            self.print_and_log(f"  Lỗi vẽ biểu đồ One-Hot: {e}")
# ==================================================================================================================
# BIỂU ĐỒ CHUẨN HÓA CATEGORICAL
    def _create_categorical_standardization_chart(self):
        """Tạo biểu đồ so sánh phân phối của biến Marital_Status trước và sau khi chuẩn hóa."""
        try:
            if 'Marital_Status' in self.original_data.columns and 'Marital_Status' in self.cleaned_data.columns:
                # Lấy phân phối
                orig_counts = self.original_data['Marital_Status'].value_counts()
                clean_counts = self.cleaned_data['Marital_Status'].value_counts()
                
                # Lấy tất cả các giá trị
                all_categories = list(set(orig_counts.index) | set(clean_counts.index))
                all_categories.sort()  # Sắp xếp theo bảng chữ cái
                
                # Chuẩn bị dữ liệu
                orig_data = [orig_counts.get(cat, 0) for cat in all_categories]
                clean_data = [clean_counts.get(cat, 0) for cat in all_categories]
                
                # Tạo biểu đồ
                fig, ax = plt.subplots(figsize=(12, 7))
                
                x = np.arange(len(all_categories))
                width = 0.4
                
                # Vẽ bars
                bars1 = ax.bar(x - width/2, orig_data, width, label='Original', color=self.colors['original'])
                bars2 = ax.bar(x + width/2, clean_data, width, label='Cleaned', color=self.colors['cleaned'])
                
                # Thêm labels và tiêu đề
                ax.set_title('So sánh phân phối Marital_Status trước và sau chuẩn hóa', fontweight='bold')
                ax.set_xlabel('Marital_Status')
                ax.set_ylabel('Số lượng')
                ax.set_xticks(x)
                ax.set_xticklabels(all_categories, rotation=45, ha='right')
                ax.legend()
                
                # Thêm số liệu trên các cột
                def autolabel(rects):
                    for rect in rects:
                        height = rect.get_height()
                        if height > 0:
                            ax.annotate(f'{int(height)}',
                                      xy=(rect.get_x() + rect.get_width()/2, height),
                                      xytext=(0, 3),
                                      textcoords="offset points",
                                      ha='center', va='bottom')
                
                autolabel(bars1)
                autolabel(bars2)
                
                plt.tight_layout()
                
                # Lưu biểu đồ
                chart_path = self.fig_dir / "5_marital_status_standardization.png"
                plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                self.print_and_log(f"\n  Đã tạo biểu đồ chuẩn hóa Marital_Status.")
                
        except Exception as e:
            self.print_and_log(f"  Lỗi khi tạo biểu đồ chuẩn hóa Marital_Status: {e}")
# ==================================================================================================================
# BIỂU ĐỒ ORDINAL ENCODING
    def _create_ordinal_encoding_chart(self):
        """Tạo biểu đồ cho ordinal encoding (Education)."""
        try:
            if 'Education' in self.cleaned_data.columns and 'Education_ord' in self.cleaned_data.columns:
                # Lấy mapping
                mapping = self.cleaned_data.groupby('Education')['Education_ord'].first().sort_values()
            
                # Tạo DataFrame cho biểu đồ
                mapping_df = pd.DataFrame({
                    'Education': mapping.index,
                    'Ordinal': mapping.values
                })
            
                # Thêm cột count để vẽ size
                edu_counts = self.cleaned_data['Education'].value_counts()
                mapping_df['Count'] = mapping_df['Education'].map(edu_counts)
            
                # Tạo biểu đồ
                fig, ax = plt.subplots(figsize=(10, 6))
            
                # Vẽ scatter plot với kích thước biểu thị số lượng
                scatter = ax.scatter(mapping_df['Ordinal'], range(len(mapping_df)), 
                          s=mapping_df['Count']/10,  # Chia cho 10 để scale phù hợp
                          alpha=0.6, color=self.colors['cleaned'])
            
                # Thêm annotation
                for i, row in mapping_df.iterrows():
                    ax.text(row['Ordinal'] + 0.1, i, 
                            f"{row['Education']} ({row['Count']} dòng)",
                            va='center')
            
                ax.set_title('Education: Ordinal Encoding', fontweight='bold')
                ax.set_xlabel('Giá trị Ordinal')
                ax.set_yticks([])
                ax.set_ylim(-1, len(mapping_df))
                ax.grid(axis='x', linestyle='--', alpha=0.7)
            
                # Thêm giải thích
                ax.text(0.5, -0.15, 
                       'Kích thước điểm biểu thị số lượng dòng dữ liệu',
                       transform=ax.transAxes, ha='center')
            
                plt.tight_layout()
            
                # Lưu biểu đồ
                chart_path = self.fig_dir / "6_education_ordinal_encoding.png"
                plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                plt.close()
            
                self.print_and_log(f"  Đã tạo biểu đồ ordinal encoding cho Education.")
            
        except Exception as e:
            self.print_and_log(f"  Lỗi khi tạo biểu đồ ordinal encoding: {e}")
# ==================================================================================================================
# TẠO BIỂU ĐỒ BIẾN DUMMY
    def _create_dummy_vars_chart(self, dummy_cols):
        """Tạo biểu đồ cho các biến dummy."""
        try:
            # Tính tần suất cho các biến dummy
            counts = []
            for col in dummy_cols:
                if pd.api.types.is_numeric_dtype(self.cleaned_data[col]):
                    count = (self.cleaned_data[col] == 1).sum()
                    pct = count / len(self.cleaned_data) * 100
                    counts.append((col, count, pct))
        
            # Sắp xếp theo tần suất
            counts.sort(key=lambda x: x[1], reverse=True)
        
            # Tạo biểu đồ
            fig, ax = plt.subplots(figsize=(10, 6))
        
            cols = [item[0] for item in counts]
            values = [item[1] for item in counts]
        
            bars = ax.bar(range(len(cols)), values, color=self.colors['cleaned'])
        
            ax.set_title('Phân phối các biến dummy (One-hot encoding)', fontweight='bold')
            ax.set_ylabel('Số lượng (giá trị = 1)')
            ax.set_xticks(range(len(cols)))
            ax.set_xticklabels(cols, rotation=45, ha='right')
        
            # Thêm số liệu và %
            for i, (col, count, pct) in enumerate(counts):
                ax.text(i, count + max(values)*0.01, f'{count} ({pct:.1f}%)',
                       ha='center', va='bottom')
        
            plt.tight_layout()
        
            # Lưu biểu đồ
            chart_path = self.fig_dir / "7_dummy_variables.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
        
            self.print_and_log(f"  Đã tạo biểu đồ phân phối biến dummy.")
        
        except Exception as e:
            self.print_and_log(f"  Lỗi khi tạo biểu đồ phân phối biến dummy: {e}")
# ==================================================================================================================
    def _create_outlier_distribution_chart(self, var, var_vn, outlier_count, idx):  
        """Vẽ biểu đồ so sánh phân phối trước/sau loại bỏ outliers."""
        try:
            orig_data = self.original_data[var].dropna()
            clean_data = self.cleaned_data[var].dropna()
    
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
            # SUBPLOT 1: Histogram + KDE
            sns.histplot(orig_data, kde=True, ax=ax1, color=self.colors['original'], 
                         label=f'Original ({len(orig_data)})', alpha=0.6, bins=50)
            sns.histplot(clean_data, kde=True, ax=ax1, color=self.colors['cleaned'], 
                         label=f'Cleaned ({len(clean_data)})', alpha=0.6, bins=50)
    
            ax1.set_title(f'Phân phối {var_vn}', fontweight='bold')
            ax1.set_xlabel(var, fontweight='bold')
            ax1.legend()
    
            # SUBPLOT 2: Boxplot
            boxplot_data = pd.DataFrame({
                'Original': orig_data,
                'Cleaned': clean_data
            })
            sns.boxplot(data=boxplot_data, ax=ax2, palette=[self.colors['original'], self.colors['cleaned']])
            ax2.set_title(f'Boxplot {var_vn}', fontweight='bold')
    
            # SUBPLOT 3: Thống kê
            ax3.axis('off')
    
            orig_stats = orig_data.describe()
            clean_stats = clean_data.describe()
    
            summary_text = f"""
    THỐNG KÊ SO SÁNH - {var_vn}

    OUTLIERS PHÁT HIỆN: {outlier_count} giá trị

    TRƯỚC LÀM SẠCH:
        • Count  : {len(orig_data):,}
        • Mean   : {orig_stats['mean']:.2f}
        • Std    : {orig_stats['std']:.2f}
        • Min    : {orig_stats['min']:.2f}
        • Max    : {orig_stats['max']:.2f}
        • Q1     : {orig_stats['25%']:.2f}
        • Median : {orig_stats['50%']:.2f}
        • Q3     : {orig_stats['75%']:.2f}

    SAU LÀM SẠCH:
        • Count  : {len(clean_data):,}
        • Mean   : {clean_stats['mean']:.2f}
        • Std    : {clean_stats['std']:.2f}
        • Min    : {clean_stats['min']:.2f}
        • Max    : {clean_stats['max']:.2f}
        • Q1     : {clean_stats['25%']:.2f}
        • Median : {clean_stats['50%']:.2f}
        • Q3     : {clean_stats['75%']:.2f}

    THAY ĐỔI:
        • Mean   : {clean_stats['mean'] - orig_stats['mean']:+.2f}
        • Std    : {clean_stats['std'] - orig_stats['std']:+.2f}
        • Deleted: {len(orig_data) - len(clean_data)} giá trị
    """
    
            ax3.text(0.05, 0.95, summary_text, transform=ax3.transAxes,
                    fontsize=9, verticalalignment='top', family='sans-serif',
                    bbox=dict(boxstyle='round,pad=0.8', facecolor='lightblue', 
                             alpha=0.9, edgecolor='blue', linewidth=1.5))
    
            plt.suptitle(f'SO SÁNH PHÂN PHỐI: {var_vn} ({var})', 
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
    
            # Lưu với idx
            safe_name = var.replace('_', '-').lower()
            chart_path = self.fig_dir / f"5_{idx:02d}_outlier_{safe_name}.png" 
            plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
    
        except Exception as e:
            self.print_and_log(f"  Lỗi vẽ biểu đồ {var}: {e}")
# ==================================================================================================================
    def _create_datetime_distribution_chart(self):
        """Tạo biểu đồ phân phối cho biến datetime."""
        try:
            if 'Dt_Customer' in self.cleaned_data.columns and pd.api.types.is_datetime64_dtype(self.cleaned_data['Dt_Customer']):
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
                # Phân phối theo năm
                year_counts = self.cleaned_data['Dt_Customer'].dt.year.value_counts().sort_index()
                years = year_counts.index
                counts = year_counts.values
            
                ax1.bar(years, counts, color=self.colors['cleaned'])
                ax1.set_title('Phân phối theo năm', fontweight='bold')
                ax1.set_xlabel('Năm')
                ax1.set_ylabel('Số lượng khách hàng')
            
                # Thêm labels
                for i, count in enumerate(counts):
                    ax1.text(years[i], count + max(counts)*0.01, f'{count}',
                           ha='start', va='bottom')
            
                # Phân phối theo tháng
                month_counts = self.cleaned_data['Dt_Customer'].dt.month.value_counts().sort_index()
            
                # Tạo dict map từ số tháng sang tên tháng
                month_names = {
                    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
                }
            
                # Tạo labels cho biểu đồ
                month_labels = [month_names[m] for m in month_counts.index]
                month_values = month_counts.values
            
                ax2.bar(month_labels, month_values, color=self.colors['accent'])
                ax2.set_title('Phân phối theo tháng', fontweight='bold')
                ax2.set_xlabel('Tháng')
                ax2.set_ylabel('Số lượng khách hàng')
            
                # Thêm labels
                for i, count in enumerate(month_values):
                    ax2.text(i, count + max(month_values)*0.01, f'{count}',
                            ha='left', va='bottom')
            
                plt.suptitle('Phân phối Dt_Customer sau khi chuyển đổi kiểu dữ liệu', fontsize=16, fontweight='bold')
                plt.tight_layout()
            
                # Lưu biểu đồ
                chart_path = self.fig_dir / "9_datetime_distribution.png"
                plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                plt.close()
            
                self.print_and_log(f"  Đã tạo biểu đồ phân phối datetime.")
            
        except Exception as e:
            self.print_and_log(f"  Lỗi khi tạo biểu đồ phân phối datetime: {e}")

# ==================================================================================================================

# ==================================================================================================================
    def _create_summary_improvement_chart(self):
        """Tạo biểu đồ tổng hợp cải thiện."""
        try:
            # Tạo biểu đồ so sánh các metric trước và sau
            metrics = ['Dòng dữ liệu', 'Missing Values', 'Giá trị trùng lặp', 'Cột dữ liệu']
        
            # Tính duplicates KHÔNG CÓ ID (feature duplicates)
            if 'ID' in self.original_data.columns:
                feature_cols = [col for col in self.original_data.columns if col != 'ID']
                orig_feature_dups = self.original_data.duplicated(subset=feature_cols).sum()
            else:
                orig_feature_dups = self.original_data.duplicated().sum()
        
            clean_dups = self.cleaned_data.duplicated().sum()
        
            orig_values = [
                self.original_data.shape[0],
                self.original_data.isnull().sum().sum(),
                orig_feature_dups,  
                self.original_data.shape[1]
            ]
        
            clean_values = [
                self.cleaned_data.shape[0],
                self.cleaned_data.isnull().sum().sum(),
                clean_dups,
                self.cleaned_data.shape[1]
            ]
        
            # Tính % thay đổi
            pct_changes = []
            for i, (ov, cv) in enumerate(zip(orig_values, clean_values)):
                if ov == 0:
                    pct_changes.append(0)
                else:
                    change = ((cv - ov) / ov) * 100
                    pct_changes.append(change)
        
            # Tạo biểu đồ
            fig, ax = plt.subplots(figsize=(12, 7))
        
            x = np.arange(len(metrics))
            width = 0.35
        
            rects1 = ax.bar(x - width/2, orig_values, width, label='Original', color=self.colors['original'])
            rects2 = ax.bar(x + width/2, clean_values, width, label='Cleaned', color=self.colors['cleaned'])
        
            # Thêm labels và tiêu đề
            ax.set_title('Tổng hợp thay đổi sau quá trình làm sạch dữ liệu', fontweight='bold')
            ax.set_ylabel('Số lượng')
            ax.set_xticks(x)
            ax.set_xticklabels(metrics)
            ax.legend()
        
            # Thêm labels trên bars
            def autolabel(rects):
                for rect in rects:
                    height = rect.get_height()
                    if height > 1000:
                        text = f'{int(height):,}'
                    else:
                        text = f'{int(height)}'
                
                    # CHỈ HIỂN THỊ NẾU height > 0
                    if height > 0:
                        ax.annotate(text,
                                  xy=(rect.get_x() + rect.get_width()/2, height),
                                  xytext=(0, 3),
                                  textcoords="offset points",
                                  ha='center', va='bottom',
                                  fontweight='bold')  # ← Thêm bold
        
            autolabel(rects1)
            autolabel(rects2)
        
            plt.tight_layout()
        
            # Lưu biểu đồ
            chart_path = self.fig_dir / "8_summary_improvement.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
        
            self.print_and_log(f"\n  Đã tạo biểu đồ tổng hợp cải thiện.")
        
        except Exception as e:
            self.print_and_log(f"  Lỗi khi tạo biểu đồ tổng hợp cải thiện: {e}")
            import traceback
            traceback.print_exc()
# ==================================================================================================================
    def run_analysis(self):
        """Chạy toàn bộ quá trình phân tích."""
        start_time = time.time()

        if not self.load_data():
            return False

        try:
            # 1. So sánh thông tin cơ bản
            self.compare_basic_info()
    
            # 2. So sánh chất lượng dữ liệu
            self.compare_data_quality()
        
            # 2.5. Phân tích Outliers Distribution 
            self.analyze_outliers_distribution()
    
            # 3. Phân tích chuẩn hóa biến phân loại
            self.analyze_categorical_standardization()
    
            # 4. Phân tích encoding
            self.analyze_encoding_transformations()
        
            # 5. So sánh biến quan trọng
            self.analyze_key_variables()
        
            # 6. Tóm tắt
            self.summarize_changes()
    
            # Hoàn thành
            elapsed_time = time.time() - start_time
            self.print_and_log("\n" + "=" * 100, bold=True)
            self.print_and_log("HOÀN THÀNH PHÂN TÍCH SO SÁNH", bold=True)
            self.print_and_log(f"Thời gian thực hiện: {elapsed_time:.2f} giây")
            self.print_and_log(f"Báo cáo được lưu tại: {self.report_file}")
            self.print_and_log(f"Biểu đồ được lưu tại: {self.fig_dir}")
    
            return True
    
        except Exception as e:
            self.print_and_log(f"\nLỗi: {e}")
            import traceback
            self.print_and_log(traceback.format_exc())
            return False

# ==================================================================================================================
def main():
    """Hàm chính để chạy phân tích so sánh dataset trước và sau làm sạch."""
    
    # Đường dẫn đến các dataset
    original_dataset_path = r"C:\Project\Machine_Learning\Machine_Learning\Dataset\Customer_Behavior.csv"
    cleaned_dataset_path = r"C:\Project\Machine_Learning\Machine_Learning\Dataset\Customer_Behavior_cleaned.csv"
    
    try:
        print("Bắt đầu phân tích so sánh dataset trước và sau khi làm sạch.")
        
        # Khởi tạo comparator
        comparator = DatasetComparator(original_dataset_path, cleaned_dataset_path)
        
        # Chạy phân tích
        success = comparator.run_analysis()
        
        if success:
            print(f"\nPhân tích hoàn tất thành công")
            print(f"Báo cáo chi tiết: {comparator.report_file}")
            print(f"Biểu đồ đã được lưu tại: {comparator.fig_dir}")
        else:
            print("\nPhân tích gặp lỗi. Vui lòng kiểm tra báo cáo.")
            
    except Exception as e:
        print(f"\nLỗi khởi tạo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()