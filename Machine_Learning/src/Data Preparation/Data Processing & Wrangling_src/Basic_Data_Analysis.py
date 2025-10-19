from multiprocessing.reduction import duplicate
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from scipy import stats
import matplotlib.pyplot as plt
import time
import os
import matplotlib.patches as mpatches 

warnings.filterwarnings('ignore') # Bỏ qua các cảnh báo

class DataProcessingWrangling:
    #=================================================================================================================================
    # Hàm __init__ khởi tạo đối tượng với các tham số
    def __init__(self, dataset_path):
        # Đường dẫn đến file dữ liệu
        self.dataset_path = dataset_path 

        # Khởi tạo biến dataset để lưu trữ dữ liệu
        self.dataset = None 

        # Khởi tạo biến processed_dataset để lưu trữ dữ liệu đã xử lý
        self.processed_dataset = None 

        # Khởi tạo biến processing_log để lưu trữ log xử lý dữ liệu
        self.processing_log = [] 

    # =================================================================================================================================
    # Hàm load_data để tải dữ liệu từ file CSV
    def load_data(self): 
        try:
            # Đọc dữ liệu từ file CSV:
            print("="*100)
            print("Bước 1: TẢI VÀ ĐỌC DỮ LIỆU")
            self.dataset = pd.read_csv(
                self.dataset_path, sep="\t"
                #parse_dates=['Dt_Customer']  # Thêm dòng này để tự động parse cột datetime
            )

            if self.dataset is not None: 
                print("\nDữ liệu đã được tải thành công.")
            else: 
                print("\nKhông thể tải dữ liệu, vui lòng kiểm tra lại.")
                return None

            self.log_action("Tải dữ liệu thành công")
        
            return self.dataset

        except Exception as e:
            print(f"Lỗi khi tải dữ liệu: {e}")
            return None

    # ==================================================================================================================
    # HÀM TẠO THƯ MỤC OUTPUT
    def _create_output_directory(self):
        """
        Tạo thư mục để lưu biểu đồ outlier
        """
        output_dir = r"C:\Project\Machine_Learning\Machine_Learning\graph\Data Processing & Wrangling_graph\Basic_Data_Analysis"
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    # =================================================================================================================
    # HÀM XUẤT BÁO CÁO TEXT
    def export_text_report(self, output_dir=None):
        """
        Xuất báo cáo text chi tiết về quá trình phân tích dữ liệu
        """
        if output_dir is None:
            output_dir = r"C:\Project\Machine_Learning\Machine_Learning\report\Data Processing & Wrangling_report"

        os.makedirs(output_dir, exist_ok=True)

        report_file = os.path.join(output_dir, f"Basic_Data_Analysic_report.txt")

        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                # Header
                f.write("=" * 100 + "\n")
                f.write("THỰC HIỆN QUY TRÌNH PHÂN TÍCH DỮ LIỆU HOÀN CHỈNH\n")
                f.write("=" * 100 + "\n")
                f.write("Bước 1: TẢI VÀ ĐỌC DỮ LIỆU\n\n")
        
                if self.dataset is not None:
                    f.write("Dữ liệu đã được tải thành công\n")
                    f.write(f"Tải dữ liệu thành công: {self.dataset.shape[0]:,} dòng × {self.dataset.shape[1]} cột\n\n")
        
                # Ghi chi tiết các bước từ processing log
                current_step = 2  # Bắt đầu từ bước 2 vì bước 1 đã viết ở trên
                for log_entry in self.processing_log:
                    timestamp = log_entry['timestamp'].strftime('%H:%M:%S')
                    action = log_entry['action']
                    details = log_entry.get('details', '')
            
                    # Kiểm tra nếu là bước chính
                    if any(keyword in action for keyword in [
                        'Phân tích thông tin cơ bản', 
                        'Phân tích missing values', 
                        'Phân tích trùng lặp', 
                        'Phân tích outlier', 
                        'Phân tích tính nhất quán',
                        'hoàn tất'
                    ]):
                        # Tạo tiêu đề bước dựa trên action
                        if 'Phân tích thông tin cơ bản' in action:
                            step_title = "PHÂN TÍCH THÔNG TIN CƠ BẢN"
                        elif 'Phân tích missing values' in action:
                            step_title = "PHÂN TÍCH GIÁ TRỊ BỊ THIẾU"
                        elif 'Phân tích trùng lặp bao gồm ID' in action:
                            step_title = "PHÂN TÍCH TRÙNG LẶP BAO GỒM ID"
                        elif 'Phân tích trùng lặp loại trừ ID' in action:
                            step_title = "PHÂN TÍCH TRÙNG LẶP LOẠI TRỪ ID"
                        elif 'Phân tích trùng lặp ID' in action:
                            step_title = "PHÂN TÍCH TRÙNG LẶP ID"
                        elif 'Phân tích outlier nhóm Binary' in action:
                            step_title = "PHÂN TÍCH OUTLIER NHÓM BINARY"
                        elif 'Phân tích outlier nhóm Count' in action:
                            step_title = "PHÂN TÍCH OUTLIER NHÓM COUNT"
                        elif 'Phân tích outlier nhóm Spending' in action:
                            step_title = "PHÂN TÍCH OUTLIER NHÓM SPENDING"
                        elif 'Phân tích outlier nhóm Income' in action:
                            step_title = "PHÂN TÍCH OUTLIER NHÓM INCOME"
                        elif 'Phân tích outlier nhóm Special' in action:
                            step_title = "PHÂN TÍCH OUTLIER NHÓM SPECIAL"
                        elif 'Phân tích outlier nhóm Constant' in action:
                            step_title = "PHÂN TÍCH OUTLIER NHÓM CONSTANT"
                        elif 'Phân tích outliers theo ngữ cảnh' in action:
                            step_title = "PHÂN TÍCH OUTLIERS THEO NGỮ CẢNH"
                        elif 'Phân tích tính nhất quán hoàn tất' in action:
                            step_title = "PHÂN TÍCH TÍNH NHẤT QUÁN DỮ LIỆU PHÂN LOẠI"
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
                
            print(f"Đã xuất báo cáo Basic Data Analysis ra file: {report_file}")
            return report_file
    
        except Exception as e:
            print(f"Lỗi khi xuất báo cáo: {e}")
            return None
    #=================================================================================================================================
    # HÀM GHI LOG HÀNH ĐỘNG
    def log_action(self, action, details=""): 
        self.processing_log.append({ 
            'action': action,   # Hành động đã thực hiện
            'details': details, # Chi tiết về hành động
            'timestamp': pd.Timestamp.now() # Thời gian thực hiện hành động ( mặc định là thời gian lúc thực hiện hành động )
        })
    
    #=================================================================================================================================   
    # HÀM CHÍNH KIỂM TRA SƠ BỘ CHẤT LƯỢNG DỮ LIỆU (Nơi tổng hợp hàm chức năng ấy)
    def initial_data_inspection(self):
        """
        Thực hiện kiểm tra sơ bộ chất lượng dữ liệu bao gồm:
        - Phân tích thông tin cơ bản về dataset
        - Phân tích giá trị bị thiếu (missing values)
        - Phân tích giá trị trùng lặp (duplicates)
        - Phân tích giá trị ngoại lai (outliers)
        - Phân tích tính nhất quán của dữ liệu phân loại (categorical consistency)
        
        Returns:
            dict: Dictionary chứa kết quả phân tích tổng hợp
        """

        print("="*100)
        print("Bước 2: KIỂM TRA SƠ BỘ DỮ LIỆU\n")

        # 1. Phân tích thông tin cơ bản
        self._analyze_basic_info()

        # 2. Phân tích giá trị bị thiếu
        missing_data = self._analyze_missing_values()

        # 3. Phân tích giá trị trùng lặp
        duplicates = self._analyze_duplicates()

        # 4. Phân tích giá trị ngoại lai
        outliers = self._analyze_outliers()

        # 5. Phân tích tính nhất quán của dữ liệu phân loại
        consistency_issues = self._analyze_categorical_consistency()

        # 6. Tổng hợp kết quả
        numerical_col = self.dataset.select_dtypes(include=[np.number]).columns
        numerical_col = [col for col in numerical_col if col != 'ID']
        categorical_col = self.dataset.select_dtypes(include=['object']).columns

        # Tạo kết quả tổng hợp
        results = {
            'missing_data': missing_data,
            'duplicates': duplicates,
            'outliers': outliers,
            'consistency_issues': consistency_issues,
            'summary': {
                'total_rows': len(self.dataset),
                'total_columns': len(self.dataset.columns),
                'numerical_columns': len(numerical_col),
                'categorical_columns': len(categorical_col)
            }
        }
       
        
        return results

    #=================================================================================================================================
    # HÀM CON PHÂN TÍCH THÔNG TIN CƠ BẢN 
    def _analyze_basic_info(self):
        """
        Hiển thị thông tin cơ bản về dataset: shape, info, describe, head
        Đơn giản, ngắn gọn, đủ dùng
        """
       
        #pd.set_option('display.max_columns', None)  # Hiển thị tất cả các cột khi print

        # ================================
        # 1. SHAPE - KÍCH CỠ DATASET
        # ================================
        print(f"SHAPE: {self.dataset.shape}")
        print(f"- {self.dataset.shape[0]:,} dòng")
        print(f"- {self.dataset.shape[1]} cột\n")
    
        # ================================
        # 2. INFO - THÔNG TIN CÁC CỘT
        # ================================
        print("INFO:")
        self.dataset.info()
        print()
    
        # ================================
        # 3. DESCRIBE - THỐNG KÊ MÔ TẢ
        # ================================
        print("DESCRIBE:")
        print(self.dataset.describe(include='all'))
    
        # ================================
        # 4. HEAD - MẪU DỮ LIỆU
        # ================================
        print("HEAD:")
        print(self.dataset.head())
        print()
    
        # ================================
        # 5. LOGGING
        # ================================
        self.log_action("Phân tích thông tin cơ bản", 
                       f"Dataset shape: {self.dataset.shape}")
    
        return {
            'shape': self.dataset.shape,
            'columns': list(self.dataset.columns),
            'dtypes': dict(self.dataset.dtypes)
        }
    #=================================================================================================================================
    # HÀM CON PHÂN TÍCH GIÁ TRỊ BỊ THIẾU 
    def _analyze_missing_values(self):
        """
        Phát hiện và thông báo các giá trị bị thiếu trong dataset
        Chỉ hiển thị thông tin, không xử lý gì thêm
        """
    
        print("="*100)
        print("Bước 3: PHÂN TÍCH GIÁ TRỊ BỊ THIẾU\n")
        # Tính missing values cho từng cột
        missing_count = self.dataset.isnull().sum() # Số lượng missing values
        missing_percent = (missing_count / len(self.dataset)) * 100 # Tỷ lệ phần trăm missing values
    
        # Lọc các cột có missing values
        cols_with_missing = missing_count[missing_count > 0]
    
        if cols_with_missing.empty:
            print("Không có giá trị bị thiếu trong dataset\n")
            self.log_action("Phân tích missing values", "Không có giá trị bị thiếu")
            return None
    
        # Hiển thị kết quả
        print(f"Phát hiện {len(cols_with_missing)} cột có giá trị bị thiếu:")
        print("-" * 35)
    
        for col in cols_with_missing.index:
            count = missing_count[col]
            percent = missing_percent[col]
            print(f"{col:<20} : {count:>4} ({percent:>5.1f}%)")
    
        print("-" * 35) 
    
        # Tính toán overall percentage
        total_missing = cols_with_missing.sum()
        total_cells = len(self.dataset) * len(self.dataset.columns)
        overall_percent = (total_missing / total_cells) * 100
    
        print(f"{'TỔNG CỘNG':<20} : {total_missing:>4} ({overall_percent:>5.2f}%)\n")
    
        # Logging
        self.log_action("Phân tích missing values", 
                       f"Phát hiện {len(cols_with_missing)} cột có missing values")
    
        # Trả về thông tin cơ bản
        return {
            'columns_with_missing': list(cols_with_missing.index),
            'missing_counts': dict(cols_with_missing),
            'total_missing': int(total_missing),
            'overall_percentage': round(overall_percent, 2)
        }

    #=================================================================================================================================
    # HÀM CON PHÂN TÍCH GIÁ TRỊ TRÙNG LẶP
    def _analyze_duplicates(self):
        """
        Phát hiện và thông báo các giá trị trùng lặp trong dataset với 2 cách tiếp cận:
        1. Kiểm tra duplicate bao gồm cột ID (exact full-row duplicates)
        2. Kiểm tra duplicate loại bỏ cột ID (identical feature vectors)
    
        Returns:
            dict: Kết quả phân tích duplicates chi tiết
        """

        print("="*100)
        print("Bước 4: PHÂN TÍCH GIÁ TRỊ TRÙNG LẶP\n")

        total_rows = len(self.dataset)
        id_present = 'ID' in self.dataset.columns

        # ================================
        # PHẦN 1: TRÙNG LẶP BAO GỒM ID
        # ================================
        print("PHẦN 1: TRÙNG LẶP TOÀN BỘ ROW (bao gồm ID)")
        print("-" * 50)
    
        # Full-row duplicates (bao gồm ID)
        dup_include_id_keepfirst = self.dataset.duplicated(keep='first').sum()
        dup_include_id_keepfalse = self.dataset.duplicated(keep=False).sum()
        dup_include_id_percent = (dup_include_id_keepfirst / total_rows) * 100

        print(f"Exact duplicates (keep='first'): {dup_include_id_keepfirst} rows ({dup_include_id_percent:.2f}%)")
        print(f"All rows in duplicate groups: {dup_include_id_keepfalse} rows")

        if dup_include_id_keepfirst > 0:
            # Hiển thị một vài mẫu duplicate
            duplicate_rows = self.dataset[self.dataset.duplicated(keep=False)]
            print(f"Mẫu duplicate rows (hiển thị 3 dòng đầu):")
            print(duplicate_rows.head(3).to_string(index=False))
            self.log_action("Phân tích trùng lặp bao gồm ID", 
                           f"Phát hiện {dup_include_id_keepfirst} bản ghi trùng lặp hoàn toàn")
        else:
            print("Không có bản ghi trùng lặp hoàn toàn")
            self.log_action("Phân tích trùng lặp bao gồm ID", "Không có bản ghi trùng lặp hoàn toàn")

        # ================================
        # PHẦN 2: TRÙNG LẶP LOẠI TRỪ ID
        # ================================
        print(f"\nPHẦN 2: TRÙNG LẶP FEATURE VECTOR (loại trừ ID)")
        print("-" * 50)
        dup_groups = []  # ← THÊM DÒNG NÀY để khởi tạo

        if id_present:
            # Tạo dataframe không có ID
            df_no_id = self.dataset.drop(columns=['ID'])
        
            # Tính duplicates trên dataframe không có ID
            dup_exclude_id_keepfirst = df_no_id.duplicated(keep='first').sum()
            dup_exclude_id_keepfalse = df_no_id.duplicated(keep=False).sum()
            dup_exclude_id_percent = (dup_exclude_id_keepfirst / total_rows) * 100

            print(f"Feature duplicates (keep='first'): {dup_exclude_id_keepfirst} rows ({dup_exclude_id_percent:.2f}%)")
            print(f"All rows in feature duplicate groups: {dup_exclude_id_keepfalse} rows")

            if dup_exclude_id_keepfirst > 0:
                # Phân tích chi tiết các nhóm duplicate
                grouped = df_no_id.groupby(list(df_no_id.columns), dropna=False).size().reset_index(name='group_size')
                dup_groups = grouped[grouped['group_size'] > 1].sort_values('group_size', ascending=False)
            
                print(f"Số nhóm duplicate: {len(dup_groups)}")
            
                # Phân bố kích thước nhóm
                size_distribution = dup_groups['group_size'].value_counts().sort_index()
                print("Phân bố kích thước nhóm:")
                for size, count in size_distribution.items():
                    total_rows_in_size = size * count
                    print(f"   - Nhóm size {size}: {count} nhóm ({total_rows_in_size} rows)")

                # Hiển thị mẫu các nhóm duplicate lớn nhất
                print(f"\nMẫu các nhóm duplicate lớn nhất:")
                for i, (_, group_info) in enumerate(dup_groups.head(3).iterrows()):
                    size = int(group_info['group_size'])
                    # Tìm các ID tương ứng với nhóm này
                    group_key = {col: group_info[col] for col in df_no_id.columns}
                    matching_rows = df_no_id.merge(pd.DataFrame([group_key]), on=list(df_no_id.columns), how='inner')
                    corresponding_ids = self.dataset.loc[matching_rows.index, 'ID'].tolist()
                
                    print(f"   Group {i+1} (size={size}): IDs = {corresponding_ids[:5]}{'...' if len(corresponding_ids) > 5 else ''}")

                self.log_action("Phân tích trùng lặp loại trừ ID", 
                               f"Phát hiện {dup_exclude_id_keepfirst} feature duplicates trong {len(dup_groups)} nhóm")
            else:
                print("Không có feature vector trùng lặp")
                self.log_action("Phân tích trùng lặp loại trừ ID", "Không có feature vector trùng lặp")
        else:
            print("Dataset không có cột ID - chỉ phân tích được duplicate toàn bộ row")
            dup_exclude_id_keepfirst = dup_include_id_keepfirst
            dup_exclude_id_keepfalse = dup_include_id_keepfalse
            dup_exclude_id_percent = dup_include_id_percent

        # ================================
        # PHẦN 3: TRÙNG LẶP ID (NẾU CÓ)
        # ================================
        print(f"\nPHẦN 3: TRÙNG LẶP ID")
        print("-" * 50)

        ID_duplicated_count = 0
        ID_duplicated_percent = 0

        if id_present:
            ID_duplicated_count = self.dataset['ID'].duplicated().sum()
            ID_duplicated_percent = (ID_duplicated_count / total_rows) * 100
    
            if ID_duplicated_count > 0:
                duplicate_ids = self.dataset[self.dataset['ID'].duplicated(keep=False)]['ID'].unique()
                print(f"Phát hiện {ID_duplicated_count} ID trùng lặp ({ID_duplicated_percent:.2f}%)")
                print(f"Các ID bị trùng (hiển thị 10 đầu): {duplicate_ids[:10].tolist()}")
                self.log_action("Phân tích trùng lặp ID", 
                               f"Phát hiện {ID_duplicated_count} ID trùng lặp")
            else:
                print("Không có ID trùng lặp")
                self.log_action("Phân tích trùng lặp ID", "Không có ID trùng lặp")
        else:
            print("Dataset không có cột ID để kiểm tra")

        # ================================
        # TÓM TẮT CUỐI
        # ================================
        print(f"\nTÓM TẮT:")
        print(f"- Tổng rows: {total_rows:,}")
        print(f"- Full-row duplicates: {dup_include_id_keepfirst} ({dup_include_id_percent:.2f}%)")
        if id_present:
            print(f"- Feature duplicates: {dup_exclude_id_keepfirst} ({dup_exclude_id_percent:.2f}%)")
            print(f"- ID duplicates: {ID_duplicated_count} ({ID_duplicated_percent:.2f}%)")
        print()

        # ================================
        # RETURN RESULTS
        # ================================
        return {
            'total_rows': total_rows,
            'id_present': id_present,
            # Include ID results
            'include_id': {
                'duplicates_keepfirst': dup_include_id_keepfirst,
                'duplicates_keepfalse': dup_include_id_keepfalse,
                'percentage': dup_include_id_percent
            },
            # Exclude ID results
            'exclude_id': {
                'duplicates_keepfirst': dup_exclude_id_keepfirst,
                'duplicates_keepfalse': dup_exclude_id_keepfalse,
                'percentage': dup_exclude_id_percent,
                'groups_count': len(dup_groups) if id_present and dup_exclude_id_keepfirst > 0 else 0
            },
            # ID-specific results
            'id_duplicates': {
                'count': ID_duplicated_count,
                'percentage': ID_duplicated_percent
            },
            # Legacy compatibility
            'total_duplicates': dup_include_id_keepfirst,
            'total_percentage': dup_include_id_percent,
            'id_duplicates_percentage': ID_duplicated_percent
        }

    #=================================================================================================================================
    # HÀM CON PHÂN TÍCH GIÁ TRỊ NGOẠI LAI
    def _analyze_outliers(self):
        """
        Phân tích giá trị ngoại lai với logic cải tiến theo ngữ cảnh business:
        - Phân loại biến theo ý nghĩa thực tế: binary, count, measure, special
        - Áp dụng phương pháp phát hiện outlier phù hợp cho từng loại
        - Đảm bảo lower_bound >= 0 cho biến không âm
        - Detect outliers trong biến binary (ngoài 0,1)
        Returns:
            list: Danh sách kết quả phân tích outliers cho mỗi cột số
        """
        print("="*100)
        print("Bước 5: PHÂN TÍCH GIÁ TRỊ NGOẠI LAI\n")

        # Lọc các cột số, loại bỏ ID
        numerical_col = self.dataset.select_dtypes(include=[np.number]).columns
        numerical_col = [col for col in numerical_col if col != 'ID']

        if len(numerical_col) == 0:
            print("Không có cột số nào để phân tích giá trị ngoại lai")
            self.log_action("Phân tích giá trị ngoại lai", "Không có cột số nào để phân tích")
            return []

        # ================================
        # PHÂN LOẠI CÁC CỘT THEO NGỮ CẢNH
        # ================================
        binary_cols = [
            'AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5',
            'Complain', 'Response'
        ]
        count_cols = [
            'Kidhome', 'Teenhome', 'Recency', 'NumDealsPurchases', 
            'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth'
        ]
        spending_cols = [
            'MntWines', 'MntFruits', 'MntMeatProducts', 
            'MntFishProducts', 'MntSweetProducts', 'MntGoldProds'
        ]
        income_cols = ['Income']
        special_cols = ['Year_Birth']
        constant_cols = ['Z_CostContact', 'Z_Revenue']

        # Lọc chỉ các cột tồn tại trong dataset
        binary_cols = [col for col in binary_cols if col in numerical_col] 
        count_cols = [col for col in count_cols if col in numerical_col]
        spending_cols = [col for col in spending_cols if col in numerical_col]
        income_cols = [col for col in income_cols if col in numerical_col]
        special_cols = [col for col in special_cols if col in numerical_col]
        constant_cols = [col for col in constant_cols if col in numerical_col]

        print("PHÂN LOẠI CÁC BIẾN THEO NGỮ CẢNH:")
        print(f"- Binary    : {binary_cols}\n")
        print(f"- Count     : {count_cols}\n")
        print(f"- Spending  : {spending_cols}\n")
        print(f"- Income    : {income_cols}\n")
        print(f"- Special   : {special_cols}\n")
        print(f"- Constant  : {constant_cols}\n")

        # ================================
        # PHÂN TÍCH TỪNG NHÓM
        # ================================
        outlier_summary = []
        outlier_count_by_type = {}

        # 1. Phân tích biến nhị phân
        for col in binary_cols:
            stats = self._analyze_binary_contextual(col)
            if stats:
                outlier_summary.append(stats)
                outlier_count_by_type['Binary'] = outlier_count_by_type.get('Binary', 0) + stats.get('Outlier_Count', 0)
        if binary_cols:
            self.log_action("Phân tích outlier nhóm Binary", f"Tổng outlier: {outlier_count_by_type.get('Binary', 0)}")

        # 2. Phân tích biến đếm
        for col in count_cols: 
            stats = self._analyze_count_contextual(col)
            if stats:
                outlier_summary.append(stats)
                outlier_count_by_type['Count'] = outlier_count_by_type.get('Count', 0) + stats.get('Outlier_Count', 0)
        if count_cols:
            self.log_action("Phân tích outlier nhóm Count", f"Tổng outlier: {outlier_count_by_type.get('Count', 0)}")

        # 3. Phân tích biến chi tiêu
        for col in spending_cols:
            stats = self._analyze_spending_contextual(col)
            if stats:
                outlier_summary.append(stats)
                outlier_count_by_type['Spending'] = outlier_count_by_type.get('Spending', 0) + stats.get('Outlier_Count', 0)
        if spending_cols:
            self.log_action("Phân tích outlier nhóm Spending", f"Tổng outlier: {outlier_count_by_type.get('Spending', 0)}")

        # 4. Phân tích biến thu nhập
        for col in income_cols:
            stats = self._analyze_income_contextual(col)
            if stats: 
                outlier_summary.append(stats)
                outlier_count_by_type['Income'] = outlier_count_by_type.get('Income', 0) + stats.get('Outlier_Count', 0)
        if income_cols:
            self.log_action("Phân tích outlier nhóm Income", f"Tổng outlier: {outlier_count_by_type.get('Income', 0)}")

        # 5. Phân tích biến đặc biệt
        for col in special_cols:
            stats = self._analyze_special_contextual(col)
            if stats:
                outlier_summary.append(stats)
                outlier_count_by_type['Special'] = outlier_count_by_type.get('Special', 0) + stats.get('Outlier_Count', 0)
        if special_cols:
            self.log_action("Phân tích outlier nhóm Special", f"Tổng outlier: {outlier_count_by_type.get('Special', 0)}")

        # 6. Phân tích biến constant
        for col in constant_cols:
            stats = self._analyze_constant_variable(col)
            if stats:
                outlier_summary.append(stats)
                outlier_count_by_type['Constant'] = outlier_count_by_type.get('Constant', 0) + 0
        if constant_cols:
            self.log_action("Phân tích outlier nhóm Constant", f"Tổng outlier: {outlier_count_by_type.get('Constant', 0)}")

        # Tổng kết
        total_outliers = sum(outlier_count_by_type.values())
        self.log_action(
            "Phân tích outliers theo ngữ cảnh", 
            f"Đã phân tích {len(numerical_col)} cột số theo 6 nhóm. Tổng outlier: {total_outliers}, Phân bố: {outlier_count_by_type}"
        )

        return outlier_summary
    #=================================================================================================================================
    # HÀM PHÂN TÍCH BIẾN NHỊ PHÂN THEO NGỮ CẢNH
    def _analyze_binary_contextual(self, col):
        """Phân tích biến nhị phân - chỉ nên có 2 giá trị là 0 và 1"""
        data = self.dataset[col].dropna() # Loại bỏ giá trị missing

        if len(data) == 0: # Nếu không có dữ liệu sau khi loại bỏ missing
            return None

        unique_vals = [int(x) for x in sorted(data.unique())]
        unique_count = len(unique_vals)

        print(f"{col} (Binary)")
        print(f"Unique values : {unique_vals}") # Hiển thị giá trị duy nhất
        print(f"Count         : {len(data)}")   # Hiển thị số lượng giá trị không missing

        # Kiểm tra outliers (giá trị khác 0,1)
        expected_values = {0, 1}
        actual_values = set(unique_vals)
        outliers = actual_values - expected_values

        outlier_count = 0
        if outliers:
            outlier_count = sum(data.isin(outliers)) # Đếm số lượng outliers
            outlier_percent = (outlier_count / len(data)) * 100 # Tính phần trăm outliers
            print(f"Outliers      : {list(outliers)} ({outlier_count} values, {outlier_percent:.2f}%)")
        else:
            print(f"Valid binary  : chỉ có 0 và 1")

        # Phân bố
        value_counts = data.value_counts() 
        for val, count in value_counts.items():
            percent = (count / len(data)) * 100
            print(f"          {val}   : {count} ({percent:.1f}%)")

        print()

        # TẠO KẾT QUẢ TRƯỚC KHI RETURN
        result = {
            'Column': col,
            'Type': 'Binary',
            'Count': len(data),
            'Unique_Values': unique_vals,
            'Outlier_Values': list(outliers) if outliers else [],
            'Outlier_Count': outlier_count,
            'Outlier_Percentage': (outlier_count / len(data) * 100) if outlier_count > 0 else 0
        }
    
        # VẼ BIỂU ĐỒ
        self._plot_binary_outliers(col, result)

        return result  # DI CHUYỂN RETURN VỀ CUỐI
    #=================================================================================================================================
    # HÀM PHÂN TÍCH BIẾN ĐẾM THEO NGỮ CẢNH  
    def _analyze_count_contextual(self, col):
        """Phân tích biến đếm - sử dụng IQR với lower_bound >= 0"""
        data = self.dataset[col].dropna()

        if len(data) == 0:
            return None

        print(f"{col} (Count)")
        print(f"   Range          : [{data.min()}, {data.max()}]")
        print(f"   Mean           : {data.mean():.1f}, Median: {data.median():.1f}")

        # IQR method với điều chỉnh
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1

        # Đảm bảo lower_bound >= 0 cho biến đếm ( Vì biến đếm không thể âm )
        lower_bound = max(Q1 - 3.0 * IQR, 0)
        upper_bound = Q3 + 3.0 * IQR

        # Tìm outliers dựa trên IQR
        outliers = data[(data < lower_bound) | (data > upper_bound)] # Phát hiện ra outlier dựa trên ngưỡng của IQR 
        outlier_count = len(outliers)  # Đếm số lượng outlier 
        outlier_percent = (outlier_count / len(data)) * 100 # Tính phần trăm của outlier 

        print(f"   IQR boundaries : [{lower_bound:.1f}, {upper_bound:.1f}]")
        print(f"   Outliers       : {outlier_count} values ({outlier_percent:.2f}%)")

        if outlier_count > 0 and outlier_count <= 10:
             clean_outliers = [int(x) for x in sorted(outliers.unique())]
             print(f"   Outlier values : {clean_outliers}")

        print()

        # TẠO KẾT QUẢ TRƯỚC KHI RETURN
        result = {
            'Column': col,
            'Type': 'Count', 
            'Count': len(data),
            'IQR_Lower': lower_bound,
            'IQR_Upper': upper_bound,
            'Outlier_Count': outlier_count,
            'Outlier_Percentage': outlier_percent
        }

        # VẼ BIỂU ĐỒ
        self._plot_numerical_outliers(col, result, 'Count')

        return result  # DI CHUYỂN RETURN VỀ CUỐI
    #=================================================================================================================================  
    # HÀM PHÂN TÍCH BIẾN CHI TIÊU THEO NGỮ CẢNH
    def _analyze_spending_contextual(self, col):
        """Phân tích biến chi tiêu - thường skewed, cần xử lý đặc biệt"""
        data = self.dataset[col].dropna()

        if len(data) == 0:
            return None

        print(f"{col} (Spending)")
        print(f"   Range          : [{data.min()}, {data.max()}]") 
        print(f"   Mean           : {data.mean():.0f}, Median: {data.median():.0f}")

        # IQR method với điều chỉnh cho biến chi tiêu
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1

        # Spending không thể âm
        lower_bound = max(Q1 - 3.0 * IQR, 0)
        upper_bound = Q3 + 3.0 * IQR

        # Tìm outliers
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        outlier_count = len(outliers)
        outlier_percent = (outlier_count / len(data)) * 100

        print(f"   IQR boundaries : [{lower_bound:.0f}, {upper_bound:.0f}]")
        print(f"   Outliers       : {outlier_count} values ({outlier_percent:.2f}%)")

        # Hiển thị top extreme values
        if outlier_count > 0:
            high_outliers = data[data > upper_bound]
            if not high_outliers.empty:
                top_3 = high_outliers.nlargest(3)
                print(f"   Top extreme    : {list(top_3)}")

        print()

        # TẠO KẾT QUẢ TRƯỚC KHI RETURN (SỬA KEY 'Count:' THÀNH 'Count')
        result = {
            'Column': col,
            'Type': 'Spending',
            'Count': len(data),  # SỬA TỪ 'Count:' THÀNH 'Count'
            'IQR_Lower': lower_bound,
            'IQR_Upper': upper_bound, 
            'Outlier_Count': outlier_count,
            'Outlier_Percentage': outlier_percent
        }

        # VẼ BIỂU ĐỒ
        self._plot_numerical_outliers(col, result, 'Spending')

        return result  # DI CHUYỂN RETURN VỀ CUỐI
    #=================================================================================================================================
    # HÀM PHÂN TÍCH BIẾN THU NHẬP THEO NGỮ CẢNH
    def _analyze_income_contextual(self, col):
        """Phân tích biến thu nhập - có thể có outliers rất lớn"""
        data = self.dataset[col].dropna()

        if len(data) == 0:
            return None

        print(f"{col} (Income)")
        print(f"   Range          : [{data.min():,.0f}, {data.max():,.0f}]")
        print(f"   Mean           : {data.mean():,.0f}, Median: {data.median():,.0f}")

        # Kiểm tra giá trị đáng ngờ
        suspicious_values = data[data >= 500000]

        if not suspicious_values.empty:
            clean_suspicious = [int(x) for x in sorted(suspicious_values.unique())]
            print(f"Suspicious values : {len(suspicious_values)}")
            print(f"Values            : {clean_suspicious}")
            # Log suspicious values
            self.log_action(f"Phát hiện {col} suspicious values", 
                           f"{len(suspicious_values)} values >= 500000: {clean_suspicious}")

        # IQR method
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75) 
        IQR = Q3 - Q1

        lower_bound = max(Q1 - 3.0 * IQR, 0)
        upper_bound = Q3 + 3.0 * IQR

        outliers = data[(data < lower_bound) | (data > upper_bound)]
        outlier_count = len(outliers)
        outlier_percent = (outlier_count / len(data)) * 100

        print(f"   IQR boundaries : [{lower_bound:,.0f}, {upper_bound:,.0f}]")
        print(f"   Outliers       : {outlier_count} values ({outlier_percent:.2f}%)")

        # Log kết quả phân tích
        self.log_action(f"Phân tích {col} outliers", 
                       f"IQR method: {outlier_count} outliers ({outlier_percent:.2f}%)")

        print()

        # TẠO KẾT QUẢ TRƯỚC KHI RETURN
        result = {
            'Column': col,
            'Type': 'Income',
            'Count': len(data),
            'Suspicious_Count': len(suspicious_values),
            'IQR_Lower': lower_bound,
            'IQR_Upper': upper_bound,
            'Outlier_Count': outlier_count, 
            'Outlier_Percentage': outlier_percent
        }

        # VẼ BIỂU ĐỒ (SỬA TỪ 'Count' THÀNH 'Income')
        self._plot_numerical_outliers(col, result, 'Income')

        return result  # DI CHUYỂN RETURN VỀ CUỐI

    #=================================================================================================================================
    # HÀM PHÂN TÍCH BIẾN ĐẶC BIỆT THEO NGỮ CẢNH
    def _analyze_special_contextual(self, col):
        """Phân tích biến đặc biệt như Year_Birth - sử dụng domain rules"""
        data = self.dataset[col].dropna()

        if len(data) == 0:
            return None

        print(f"{col} (Special)")
        print(f"   Range        : [{data.min()}, {data.max()}]")

        if col == 'Year_Birth':
            # Domain-specific rules cho năm sinh
            current_year = 2024  # Năm phân tích
    
            too_old = data[data < 1900]        # Sinh trước 1900 (>124 tuổi)
            too_young = data[data > 2006]      # Sinh sau 2006 (<18 tuổi)
    
            outlier_count = len(too_old) + len(too_young)
            outlier_percent = (outlier_count / len(data)) * 100
    
            print(f"   Valid range  : [1900, 2006] (18-124 tuổi)")
    
            if not too_old.empty:
                old_values = [int(x) for x in sorted(too_old.unique())]
                print(f"Too old         : {len(too_old)} values {old_values}")
            if not too_young.empty:
                 young_values = [int(x) for x in sorted(too_young.unique())]
                 print(f"Too young       : {len(too_young)} values {young_values}")
    
                 print(f"   Total outliers: {outlier_count} ({outlier_percent:.2f}%)")

        print()

        # TẠO KẾT QUẢ TRƯỚC KHI RETURN
        result = {
            'Column': col,
            'Type': 'Special',
            'Count': len(data),
            'Outlier_Count': outlier_count,
            'Outlier_Percentage': outlier_percent
        }

        # VẼ BIỂU ĐỒ
        self._plot_special_outliers(col, result)

        return result  # DI CHUYỂN RETURN VỀ CUỐI

    
    #=================================================================================================================================
    # HÀM PHÂN TÍCH BIẾN CONSTANT
    def _analyze_constant_variable(self, col):
            """Phân tích biến constant - không có giá trị phân biệt"""
            data = self.dataset[col].dropna()

            if len(data) == 0:
                return None

            unique_vals = data.unique()
            unique_count = len(unique_vals)

            print(f"{col} (Constant)")
            print(f"   Unique values: {unique_vals}")
            print(f"   Unique count: {unique_count}")

            if unique_count == 1:
                print(f"Variable is constant - no variance for analysis")
            else:
                print(f"Variable has variance")

            print()

            # TẠO KẾT QUẢ TRƯỚC KHI RETURN
            result = {
                'Column': col,
                'Type': 'Constant',
                'Count': len(data),
                'Unique_Count': unique_count,
                'Unique_Values': list(unique_vals),
                'Is_Constant': unique_count == 1
            }

            # VẼ BIỂU ĐỒ
            self._plot_constant_variable(col, result)

            return result  # DI CHUYỂN RETURN VỀ CUỐI
    #=================================================================================================================================
    # HÀM VẼ BIỂU ĐỒ CHO BIẾN NHỊ PHÂN
    def _plot_binary_outliers(self, col, stats_result):
        """
        Vẽ biểu đồ phân tích outliers cho biến nhị phân
        
        Args:
            col (str): Tên cột biến nhị phân cần phân tích
            stats_result (dict): Kết quả phân tích từ _analyze_binary_contextual
        
        Returns:
            None: Lưu biểu đồ vào file PNG trong thư mục output
        """
        try:
            # Tạo thư mục output để lưu biểu đồ
            output_dir = self._create_output_directory()
            
            # Lấy dữ liệu và loại bỏ giá trị missing
            data = self.dataset[col].dropna()
            
            # Kiểm tra nếu không có dữ liệu
            if len(data) == 0:
                print(f"    Không có dữ liệu hợp lệ cho cột {col}")
                return
            
            # =====================================
            # TẠO FIGURE VỚI 2 SUBPLOT
            # =====================================
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # =====================================
            # SUBPLOT 1: BAR CHART PHÂN BỐ GIÁ TRỊ
            # =====================================
            value_counts = data.value_counts().sort_index()
            colors = ['lightblue' if val in [0, 1] else 'red' for val in value_counts.index]
            
            bars = ax1.bar(range(len(value_counts)), value_counts.values, 
                          color=colors, alpha=0.7, edgecolor='black', linewidth=1.2)
            
            # Thiết lập labels và title cho subplot 1
            ax1.set_xlabel('Giá trị', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Số lượng', fontsize=12, fontweight='bold')
            ax1.set_title(f'{col} - Phân phối giá trị nhị phân', fontsize=14, fontweight='bold')
            ax1.set_xticks(range(len(value_counts)))
            ax1.set_xticklabels(value_counts.index, fontsize=11)
            ax1.grid(True, alpha=0.3, linestyle='--')
            
            # Thêm số liệu lên các cột
            for i, (bar, count) in enumerate(zip(bars, value_counts.values)):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2, height + count*0.01, 
                        f'{count:,}', ha='center', va='bottom', 
                        fontweight='bold', fontsize=10)
            
            # Tạo legend cho subplot 1
            legend_elements = [
                mpatches.Patch(color='lightblue', label='Giá trị hợp lệ (0, 1)'),
                mpatches.Patch(color='red', label='Giá trị ngoại lai')
            ]
            ax1.legend(handles=legend_elements, loc='upper right', fontsize=10)
            
            # =====================================
            # SUBPLOT 2: BẢNG THÔNG TIN CHI TIẾT
            # =====================================
            ax2.axis('off')
            
            # Tạo nội dung thông tin chi tiết
            expected_values = [0, 1]
            actual_values = stats_result.get('Unique_Values', [])
            outlier_count = stats_result.get('Outlier_Count', 0)
            outlier_percentage = stats_result.get('Outlier_Percentage', 0)
            total_records = stats_result.get('Count', 0)
            
            # Xác định trạng thái
            status = ' Hợp lệ' if outlier_count == 0 else ' Có outliers'
            status_color = 'green' if outlier_count == 0 else 'red'
            
            summary_text = f"""
PHÂN TÍCH BIẾN NHỊ PHÂN: {col}

THÔNG TIN CƠ BẢN:
   • Tổng số bản ghi: {total_records:,}
   • Giá trị mong đợi: {expected_values}
   • Giá trị thực tế: {actual_values}

PHÂN TÍCH OUTLIERS:
   • Số lượng outliers: {outlier_count:,}
   • Tỷ lệ outliers: {outlier_percentage:.2f}%
   • Trạng thái: {status}

 PHÂN BỐ GIÁ TRỊ:"""
            
            # Thêm phân bố chi tiết
            for val, count in value_counts.items():
                percentage = (count / len(data)) * 100
                summary_text += f"\n   • Giá trị {val}: {count:,} ({percentage:.1f}%)"
            
            # Thêm thông tin outliers nếu có
            outlier_values = stats_result.get('Outlier_Values', [])
            if outlier_values:
                summary_text += f"\n\n GIÁ TRỊ NGOẠI LAI:\n   • {outlier_values}"
            
            # Hiển thị text summary
            ax2.text(0.05, 0.95, summary_text, transform=ax2.transAxes, 
                    fontsize=11, verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.8', facecolor='lightgray', alpha=0.8))
            
            # =====================================
            # THIẾT LẬP FIGURE CHÍNH
            # =====================================
            plt.suptitle(f' PHÂN TÍCH OUTLIERS - BIẾN NHỊ PHÂN: {col}', 
                        fontsize=16, fontweight='bold', y=0.98)
            plt.tight_layout()
            
            # =====================================
            # LƯU BIỂU ĐỒ
            # =====================================
            filename = f"binary_outliers_{col}.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close()
            
            # Thông báo thành công
            print(f"    Đã lưu biểu đồ Binary: {filename}")
            print(f"    Đường dẫn: {filepath}")
            print("-" * 60)
            
        except Exception as e:
            print(f"    Lỗi khi vẽ biểu đồ Binary cho cột '{col}': {e}")
            import traceback
            print(f"    Chi tiết lỗi: {traceback.format_exc()}")


    #=================================================================================================================================
        #=================================================================================================================================
    # HÀM VẼ BIỂU ĐỒ CHO BIẾN SỐ (COUNT, SPENDING, INCOME)
    def _plot_numerical_outliers(self, col, stats_result, var_type):
        """
        Vẽ biểu đồ phân tích outliers cho các biến số sử dụng phương pháp IQR
        
        Args:
            col (str): Tên cột biến số cần phân tích
            stats_result (dict): Kết quả phân tích outliers
            var_type (str): Loại biến ('Count', 'Spending', 'Income')
        
        Returns:
            None: Lưu biểu đồ vào file PNG trong thư mục output
        """
        try:
            # Tạo thư mục output để lưu biểu đồ
            output_dir = self._create_output_directory()
            
            # Lấy dữ liệu và loại bỏ giá trị missing
            data = self.dataset[col].dropna()
            
            # Kiểm tra nếu không có dữ liệu
            if len(data) == 0:
                print(f"    Không có dữ liệu hợp lệ cho cột {col}")
                return
            
            # =====================================
            # TẠO FIGURE VỚI 3 SUBPLOT
            # =====================================
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 7))
            
            # Lấy thông tin IQR từ kết quả phân tích
            lower_bound = stats_result.get('IQR_Lower', 0)
            upper_bound = stats_result.get('IQR_Upper', data.max())
            
            # Xác định outliers và dữ liệu bình thường
            outliers_mask = (data < lower_bound) | (data > upper_bound)
            outliers_data = data[outliers_mask]
            normal_data = data[~outliers_mask]
            
            # =====================================
            # SUBPLOT 1: BOXPLOT VỚI IQR BOUNDARIES
            # =====================================
            box_plot = ax1.boxplot(data.values, vert=True, patch_artist=True, 
                                  showfliers=True, 
                                  flierprops={'marker': 'o', 'markersize': 5, 
                                            'markerfacecolor': 'red', 'markeredgecolor': 'darkred'})
            
            # Tô màu cho boxplot
            box_plot['boxes'][0].set_facecolor('lightblue')
            box_plot['boxes'][0].set_alpha(0.7)
            box_plot['boxes'][0].set_edgecolor('navy')
            box_plot['boxes'][0].set_linewidth(2)
            
            # Thêm đường IQR boundaries
            ax1.axhline(y=lower_bound, color='orange', linestyle='--', linewidth=2.5, 
                       label=f'Ngưỡng dưới IQR: {lower_bound:.1f}')
            ax1.axhline(y=upper_bound, color='orange', linestyle='--', linewidth=2.5, 
                       label=f'Ngưỡng trên IQR: {upper_bound:.1f}')
            
            # Thiết lập labels và title cho subplot 1
            ax1.set_ylabel('Giá trị', fontsize=12, fontweight='bold')
            ax1.set_title(f'{col}\nBoxplot với ngưỡng IQR', fontsize=13, fontweight='bold')
            ax1.legend(fontsize=9, loc='upper right')
            ax1.grid(True, alpha=0.3, linestyle='--')
            
            # =====================================
            # SUBPLOT 2: HISTOGRAM VỚI OUTLIERS HIGHLIGHTED
            # =====================================
            # Vẽ histogram cho dữ liệu bình thường
            ax2.hist(normal_data, bins=30, alpha=0.7, color='skyblue', 
                    edgecolor='black', linewidth=1, label='Dữ liệu bình thường')
            
            # Vẽ histogram cho outliers nếu có
            if len(outliers_data) > 0:
                ax2.hist(outliers_data, bins=20, alpha=0.8, color='red', 
                        edgecolor='darkred', linewidth=1.2, 
                        label=f'Outliers ({len(outliers_data):,})')
            
            # Thêm đường ngưỡng IQR
            ax2.axvline(x=lower_bound, color='orange', linestyle='--', 
                       linewidth=2.5, alpha=0.8, label='Ngưỡng IQR')
            ax2.axvline(x=upper_bound, color='orange', linestyle='--', 
                       linewidth=2.5, alpha=0.8)
            
            # Thiết lập labels và title cho subplot 2
            ax2.set_xlabel('Giá trị', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Tần số', fontsize=12, fontweight='bold')
            ax2.set_title(f'{col}\nPhân phối với outliers', fontsize=13, fontweight='bold')
            ax2.legend(fontsize=9, loc='upper right')
            ax2.grid(True, alpha=0.3, linestyle='--')
            
            # =====================================
            # SUBPLOT 3: BẢNG THÔNG TIN THỐNG KÊ CHI TIẾT
            # =====================================
            ax3.axis('off')
            
            # Lấy thông tin từ kết quả phân tích
            total_count = stats_result.get('Count', 0)
            outlier_count = stats_result.get('Outlier_Count', 0)
            outlier_percentage = stats_result.get('Outlier_Percentage', 0)
            
            # Tạo nội dung thông tin chi tiết
            var_type_vn = {
                'Count': 'ĐẾM', 
                'Spending': 'CHI TIÊU', 
                'Income': 'THU NHẬP'
            }.get(var_type, var_type.upper())
            
            summary_text = f"""
PHÂN TÍCH BIẾN {var_type_vn}: {col}

THÔNG TIN CƠ BẢN:
   • Tổng số điểm dữ liệu: {total_count:,}
   • Khoảng giá trị: [{data.min():,.0f}, {data.max():,.0f}]
   • Kiểu biến: {var_type_vn}

NGƯỠNG IQR (k=3.0):
   • Ngưỡng dưới: {lower_bound:.1f}
   • Ngưỡng trên: {upper_bound:.1f}
   • Phương pháp: Interquartile Range

PHÂN TÍCH OUTLIERS:
   • Tổng outliers: {outlier_count:,}
   • Tỷ lệ outliers: {outlier_percentage:.2f}%
   • Dữ liệu bình thường: {total_count - outlier_count:,}

THỐNG KÊ MÔ TẢ:
   • Trung bình: {data.mean():,.1f}
   • Trung vị: {data.median():,.1f}
   • Độ lệch chuẩn: {data.std():,.1f}
   • Q1: {data.quantile(0.25):,.1f}
   • Q3: {data.quantile(0.75):,.1f}
"""
            
            # Hiển thị text summary với box styling
            ax3.text(0.05, 0.95, summary_text, transform=ax3.transAxes, 
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.8', facecolor='lightgray', alpha=0.9))
            
            # =====================================
            # THIẾT LẬP FIGURE CHÍNH
            # =====================================
            plt.suptitle(f' PHÂN TÍCH OUTLIERS - BIẾN {var_type_vn}: {col}', 
                        fontsize=16, fontweight='bold', y=0.98)
            plt.tight_layout()
            
            # =====================================
            # LƯU BIỂU ĐỒ
            # =====================================
            filename = f"{var_type.lower()}_outliers_{col}.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            plt.close()
            
            # Thông báo thành công
            print(f"    Đã lưu biểu đồ {var_type}: {filename}")
            print(f"    Đường dẫn: {filepath}")
            print("-" * 60)
            
        except Exception as e:
            print(f"    Lỗi khi vẽ biểu đồ {var_type} cho cột '{col}': {e}")
            import traceback
            print(f"    Chi tiết lỗi: {traceback.format_exc()}")

    #=================================================================================================================================
    # HÀM VẼ BIỂU ĐỒ CHO BIẾN ĐẶC BIỆT (YEAR_BIRTH)
    def _plot_special_outliers(self, col, stats_result):
        """
        Vẽ biểu đồ phân tích outliers cho biến đặc biệt như Year_Birth
        
        Args:
            col (str): Tên cột biến đặc biệt cần phân tích
            stats_result (dict): Kết quả phân tích từ _analyze_special_contextual
        
        Returns:
            None: Lưu biểu đồ vào file PNG trong thư mục output
        """
        try:
            # Tạo thư mục output để lưu biểu đồ
            output_dir = self._create_output_directory()
            
            # Lấy dữ liệu và loại bỏ giá trị missing
            data = self.dataset[col].dropna()
            
            # Kiểm tra nếu không có dữ liệu
            if len(data) == 0:
                print(f"    Không có dữ liệu hợp lệ cho cột {col}")
                return
            
            # =====================================
            # TẠO FIGURE VỚI 2 SUBPLOT
            # =====================================
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
            
            # Xử lý riêng cho Year_Birth
            if col == 'Year_Birth':
                current_year = 2024
                
                # Phân loại dữ liệu theo độ tuổi
                too_old = data[data < 1900]        # Quá già (>124 tuổi)
                too_young = data[data > 2006]      # Quá trẻ (<18 tuổi)
                normal_data = data[(data >= 1900) & (data <= 2006)]  # Bình thường
                
                # =====================================
                # SUBPLOT 1: HISTOGRAM PHÂN BỐ THEO ĐỘ TUỔI
                # =====================================
                # Vẽ dữ liệu bình thường
                ax1.hist(normal_data, bins=40, alpha=0.7, color='skyblue', 
                        edgecolor='black', linewidth=1, label=f'Bình thường (1900-2006): {len(normal_data):,}')
                
                # Vẽ outliers quá già
                if len(too_old) > 0:
                    ax1.hist(too_old, bins=15, alpha=0.8, color='red', 
                            edgecolor='darkred', linewidth=1.2, 
                            label=f'Quá già (<1900): {len(too_old):,}')
                
                # Vẽ outliers quá trẻ
                if len(too_young) > 0:
                    ax1.hist(too_young, bins=15, alpha=0.8, color='orange', 
                            edgecolor='darkorange', linewidth=1.2, 
                            label=f'Quá trẻ (>2006): {len(too_young):,}')
                
                # Thêm đường ngưỡng độ tuổi hợp lệ
                ax1.axvline(x=1900, color='green', linestyle='-', linewidth=3, 
                           alpha=0.8, label='Ngưỡng tuổi hợp lệ')
                ax1.axvline(x=2006, color='green', linestyle='-', linewidth=3, alpha=0.8)
                
                # Thiết lập labels và title cho subplot 1
                ax1.set_xlabel('Năm sinh', fontsize=12, fontweight='bold')
                ax1.set_ylabel('Tần số', fontsize=12, fontweight='bold')
                ax1.set_title(f'{col}\nPhân phối năm sinh với outliers về độ tuổi', 
                             fontsize=13, fontweight='bold')
                ax1.legend(fontsize=10, loc='upper right')
                ax1.grid(True, alpha=0.3, linestyle='--')
                
                # =====================================
                # SUBPLOT 2: BẢNG THÔNG TIN CHI TIẾT
                # =====================================
                ax2.axis('off')
                
                # Lấy thông tin từ kết quả phân tích
                total_count = stats_result.get('Count', 0)
                outlier_count = stats_result.get('Outlier_Count', 0)
                outlier_percentage = stats_result.get('Outlier_Percentage', 0)
                
                # Tạo nội dung thông tin chi tiết
                summary_text = f"""
PHÂN TÍCH BIẾN ĐẶC BIỆT: {col}

THÔNG TIN CƠ BẢN:
   • Tổng số bản ghi: {total_count:,}
   • Khoảng dữ liệu: [{data.min()}, {data.max()}]
   • Khoảng hợp lệ: [1900, 2006] (Tuổi 18-124)

PHÂN TÍCH OUTLIERS:
   • Tổng outliers: {outlier_count:,}
   • Tỷ lệ outliers: {outlier_percentage:.2f}%
   • Dữ liệu bình thường: {len(normal_data):,}

PHÂN LOẠI OUTLIERS:
   • Quá già (<1900): {len(too_old):,} người
   • Quá trẻ (>2006): {len(too_young):,} người

THỐNG KÊ DỮ LIỆU BÌNH THƯỜNG:
   • Năm sinh TB: {normal_data.mean():.0f}
   • Năm sinh giữa: {normal_data.median():.0f}
   • Tuổi trung bình: {current_year - normal_data.mean():.0f} tuổi
   • Tuổi giữa: {current_year - normal_data.median():.0f} tuổi

ĐÁNH GIÁ CHẤT LƯỢNG:
   • Tỷ lệ dữ liệu hợp lệ: {(len(normal_data)/total_count)*100:.1f}%
   • Trạng thái: {'✅ Tốt' if outlier_percentage < 5 else '⚠️ Cần xem xét'}
"""
                
                # Hiển thị text summary
                ax2.text(0.05, 0.95, summary_text, transform=ax2.transAxes, 
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round,pad=0.8', facecolor='lightgray', alpha=0.9))
            
            # =====================================
            # THIẾT LẬP FIGURE CHÍNH
            # =====================================
            plt.suptitle(f' PHÂN TÍCH OUTLIERS - BIẾN ĐẶC BIỆT: {col}', 
                        fontsize=16, fontweight='bold', y=0.98)
            plt.tight_layout()
            
            # =====================================
            # LƯU BIỂU ĐỒ
            # =====================================
            filename = f"special_outliers_{col}.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            plt.close()
            
            # Thông báo thành công
            print(f"    Đã lưu biểu đồ Special: {filename}")
            print(f"    Đường dẫn: {filepath}")
            print("-" * 60)
            
        except Exception as e:
            print(f"    Lỗi khi vẽ biểu đồ Special cho cột '{col}': {e}")
            import traceback
            print(f"    Chi tiết lỗi: {traceback.format_exc()}")

        #=================================================================================================================================
    # HÀM VẼ BIỂU ĐỒ CHO BIẾN HẰNG SỐ
    def _plot_constant_variable(self, col, stats_result):
        """
        Vẽ biểu đồ phân tích cho biến hằng số (constant variable)
        
        Args:
            col (str): Tên cột biến hằng số cần phân tích
            stats_result (dict): Kết quả phân tích từ _analyze_constant_variable
        
        Returns:
            None: Lưu biểu đồ vào file PNG trong thư mục output
        """
        try:
            # Tạo thư mục output để lưu biểu đồ
            output_dir = self._create_output_directory()
            
            # Lấy dữ liệu và loại bỏ giá trị missing
            data = self.dataset[col].dropna()
            
            # Kiểm tra nếu không có dữ liệu
            if len(data) == 0:
                print(f"    Không có dữ liệu hợp lệ cho cột {col}")
                return
            
            # =====================================
            # TẠO FIGURE VỚI 2 SUBPLOT
            # =====================================
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Lấy thông tin từ kết quả phân tích
            unique_values = stats_result.get('Unique_Values', [])
            unique_count = stats_result.get('Unique_Count', 0)
            is_constant = stats_result.get('Is_Constant', False)
            
            # =====================================
            # SUBPLOT 1: BAR CHART GIÁ TRỊ DUY NHẤT
            # =====================================
            value_counts = data.value_counts().sort_index()
            
            # Chọn màu dựa trên số lượng giá trị duy nhất
            bar_color = 'lightgreen' if is_constant else 'lightblue'
            edge_color = 'darkgreen' if is_constant else 'darkblue'
            
            bars = ax1.bar(range(len(value_counts)), value_counts.values, 
                          color=bar_color, alpha=0.8, edgecolor=edge_color, linewidth=1.5)
            
            # Thiết lập labels và title cho subplot 1
            ax1.set_xlabel('Giá trị duy nhất', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Số lượng', fontsize=12, fontweight='bold')
            ax1.set_title(f'{col}\nPhân bố giá trị của biến hằng số', fontsize=13, fontweight='bold')
            ax1.set_xticks(range(len(value_counts)))
            
            # Xử lý labels cho trục x
            if len(value_counts) <= 10:
                labels = [str(val) for val in value_counts.index]
            else:
                labels = [str(val) if i % 2 == 0 else '' for i, val in enumerate(value_counts.index)]
            
            ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
            ax1.grid(True, alpha=0.3, linestyle='--')
            
            # Thêm số liệu lên các cột
            for i, (bar, count) in enumerate(zip(bars, value_counts.values)):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2, height + count*0.01, 
                        f'{count:,}', ha='center', va='bottom', 
                        fontweight='bold', fontsize=10)
            
            # =====================================
            # SUBPLOT 2: BẢNG THÔNG TIN CHI TIẾT
            # =====================================
            ax2.axis('off')
            
            # Lấy thông tin từ kết quả phân tích
            total_count = stats_result.get('Count', 0)
            
            # Tạo nội dung thông tin chi tiết
            status_text = 'BIẾN HẰNG SỐ' if is_constant else 'BIẾN CÓ PHƯƠNG SAI'
            
            summary_text = f"""
PHÂN TÍCH BIẾN HẰNG SỐ: {col}

THÔNG TIN CƠ BẢN:
   • Tổng số bản ghi: {total_count:,}
   • Số giá trị duy nhất: {unique_count:,}
   • Trạng thái: {status_text}

 DANH SÁCH GIÁ TRỊ:
"""
            
            # Thêm danh sách các giá trị duy nhất
            if len(unique_values) <= 20:
                for i, val in enumerate(unique_values):
                    count = value_counts.get(val, 0)
                    percentage = (count / total_count) * 100
                    summary_text += f"\n   • {val}: {count:,} ({percentage:.1f}%)"
            else:
                summary_text += f"\n   • Quá nhiều giá trị để hiển thị ({unique_count:,} giá trị)"
                # Hiển thị top 10
                summary_text += "\n   • Top 10 giá trị phổ biến:"
                for val, count in value_counts.head(10).items():
                    percentage = (count / total_count) * 100
                    summary_text += f"\n     - {val}: {count:,} ({percentage:.1f}%)"
            
            # Thêm kết luận phân tích
            summary_text += f"""

KẾT QUẢ PHÂN TÍCH:
   • Loại biến: {'Hằng số' if is_constant else 'Có biến thiên'}
   • Độ đa dạng: {'Thấp' if unique_count <= 5 else 'Cao'}
   • Đánh giá: {'Có thể loại bỏ' if is_constant else 'Nên giữ lại'}
"""
            
            # Hiển thị text summary
            ax2.text(0.05, 0.95, summary_text, transform=ax2.transAxes, 
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.8', facecolor='lightgray', alpha=0.9))
            
            # =====================================
            # THIẾT LẬP FIGURE CHÍNH
            # =====================================
            plt.suptitle(f' PHÂN TÍCH BIẾN HẰNG SỐ: {col}', 
                        fontsize=16, fontweight='bold', y=0.98)
            plt.tight_layout()
            
            # =====================================
            # LƯU BIỂU ĐỒ
            # =====================================
            filename = f"constant_variable_{col}.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            plt.close()
            
            # Thông báo thành công
            print(f"    Đã lưu biểu đồ Constant: {filename}")
            print(f"    Đường dẫn: {filepath}")
            print("-" * 60)
            
        except Exception as e:
            print(f"    Lỗi khi vẽ biểu đồ Constant cho cột '{col}': {e}")
            import traceback
            print(f"    Chi tiết lỗi: {traceback.format_exc()}")

    #=================================================================================================================================
    # HÀM CON PHÂN TÍCH TÍNH NHẤT QUÁN CỦA DỮ LIỆU PHÂN LOẠI
    def _analyze_categorical_consistency(self):
        """
        Phân tích tính nhất quán của dữ liệu phân loại 
        Chỉ hiển thị thống kê cơ bản, không phân tích chuyên sâu
        
        Returns:
            list: Danh sách các cột phân loại đã phân tích
        """

        print("="*100)
        print("Bước 6: PHÂN TÍCH TÍNH NHẤT QUÁN DỮ LIỆU PHÂN LOẠI")

        # Lấy danh sách các cột phân loại
        categorical_col = self.dataset.select_dtypes(include=['object']).columns

        if len(categorical_col) == 0:
            print("Không có cột phân loại nào trong dataset để phân tích tính nhất quán")
            self.log_action("Phân tích tính nhất quán dữ liệu phân loại", "Không có cột phân loại")
            return []

        # Phân tích từng cột một cách đơn giản
        for col in categorical_col:
            self._analyze_single_categorical_simple(col)

        # Tóm tắt đơn giản
        print("TÓM TẮT PHÂN TÍCH TÍNH NHẤT QUÁN\n")
        print(f"Đã phân tích {len(categorical_col)} cột mang kiểu object")
        
        self.log_action("Phân tích tính nhất quán hoàn tất", 
                      f"Phân tích {len(categorical_col)} cột phân loại")
        
        print(f"\nPhân tích tính nhất quán dữ liệu phân loại hoàn tất")
        
        # Trả về danh sách rỗng vì không cần track issues nữa
        return []

    #=================================================================================================================================
    # HÀM PHÂN TÍCH ĐơN GIẢN CHO MỘT CỘT PHÂN LOẠI
    def _analyze_single_categorical_simple(self, col):
        """
        Phân tích đơn giản một cột phân loại:
        - Hiển thị thống kê cơ bản
        - Hiển thị phân bố giá trị
        - Phân loại độ phức tạp
        Không có phân tích chuyên sâu
        """
        print(f"\nPhân tích cột: {col}")

        # Lấy thông tin cơ bản
        unique_values = self.dataset[col].value_counts()
        unique_count = len(unique_values)
        
        print(f"Số lượng giá trị duy nhất: {unique_count}")
        print(f"Tổng số dòng: {len(self.dataset)}")

        # Hiển thị phân bố giá trị
        print(f"\nPHÂN BỐ GIÁ TRỊ:")
        if unique_count <= 15:  # Hiển thị tất cả nếu ít giá trị
            for value, count in unique_values.items():
                percentage = (count / len(self.dataset)) * 100
                print(f"      '{value}': {count:,} lần ({percentage:.2f}%)")
        else:  # Hiển thị top 10 nếu quá nhiều
            print("Top 10 giá trị phổ biến nhất:")
            for value, count in unique_values.head(10).items():
                percentage = (count / len(self.dataset)) * 100
                print(f"      '{value}': {count:,} lần ({percentage:.2f}%)")
            print(f"      ... và {unique_count - 10} giá trị khác")

        # Phân loại độ phức tạp
        complexity = self._classify_categorical_complexity_simple(unique_count)
        print(f"Độ phức tạp: {complexity}\n")

    #=================================================================================================================================
    # HÀM PHÂN LOẠI ĐỘ PHỨC TẠP ĐƠN GIẢN
    def _classify_categorical_complexity_simple(self, unique_count):
        """
        Phân loại độ phức tạp đơn giản dựa trên số lượng giá trị duy nhất
        """
        cardinality_ratio = unique_count / len(self.dataset)
        
        if cardinality_ratio > 0.9:
            return "HIGH CARDINALITY"
        elif cardinality_ratio > 0.5:
            return "MEDIUM CARDINALITY"
        elif cardinality_ratio > 0.1:
            return "LOW CARDINALITY"
        elif unique_count == 1:
            return "CONSTANT"
        else:
            return "CATEGORICAL"

    #=================================================================================================================================
    # HÀM CHÍNH TỔNG HỢP - CHẠY TOÀN BỘ PIPELINE DATA ANALYSIS
    def run_complete_data_analysis(self):
        """
        Hàm chính tổng hợp - chạy toàn bộ pipeline phân tích dữ liệu từ đầu đến cuối:
        1. Tải dữ liệu
        2. Kiểm tra sơ bộ chất lượng dữ liệu (5 bước phân tích)
        3. Hiển thị tóm tắt kết quả cuối cùng
        4. Lưu log các hành động đã thực hiện
        
        Returns:
            dict: Kết quả tổng hợp toàn bộ quá trình phân tích
        """
        import time
        start_time = time.time()
        
        try:            
            # ==========================================
            # BƯỚC 1: TẢI DỮ LIỆU
            # ==========================================
            data = self.load_data() 
            
            if data is None:
                print("Không thể tải dữ liệu")
                return None
            
            print(f"Tải dữ liệu thành công: {data.shape[0]:,} dòng × {data.shape[1]} cột\n")
            
            # ==========================================
            # BƯỚC 2: KIỂM TRA SƠ BỘ CHẤT LƯỢNG DỮ LIỆU
            # ==========================================
            analysis_results = self.initial_data_inspection() # Gọi hàm initial_data_inspection để tiến hành lấy mọi thông tin sơ bộ của dataset 
            
            # ==========================================
            # BƯỚC 3: TÓM TẮT KẾT QUẢ CUỐI CÙNG
            # ==========================================
            final_summary = self._generate_final_summary(analysis_results, start_time) # Gọi hàm tóm tắt kết quả
            
            # ==========================================
            # BƯỚC 4: HIỂN THỊ LOG ACTIONS
            # ==========================================
            self._display_processing_log() # Gọi hàm để lấy log 
            
            # ==========================================
            # KẾT THÚC PIPELINE
            # ==========================================
            total_time = time.time() - start_time
            print("\n" + "=" * 100)
            print("HOÀN THÀNH PHÂN TÍCH DỮ LIỆU")
            print("=" * 100)
            print(f"Tổng thời gian thực hiện : {total_time:.2f} giây")
            print(f"Tốc độ xử lý             : {data.shape[0] / total_time:.0f} dòng/giây")
            print(f"Thời gian kết thúc       : {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Xuất báo cáo text
            try:
                self.export_text_report()
            except Exception as e:
                print(f"Lỗi khi xuất báo cáo text: {e}")

            return {
                'success': True,
                'dataset_shape': data.shape,
                'analysis_results': analysis_results,
                'final_summary': final_summary,
                'total_time': total_time,
                'processing_log': self.processing_log
            }
            
        except Exception as e:
            print(f"\nLỖI TRONG PIPELINE: {e}")
            import traceback
            print("Chi tiết lỗi:")
            print(traceback.format_exc())
            return {
                'success': False,
                'error': str(e),
                'processing_log': self.processing_log
            }

    #=================================================================================================================================
    # HÀM TẠO TÓM TẮT KẾT QUẢ CUỐI CÙNG
    def _generate_final_summary(self, analysis_results, start_time):
        """
        Tạo tóm tắt kết quả cuối cùng của toàn bộ quá trình phân tích
        
        Args:
            analysis_results (dict): Kết quả từ initial_data_inspection()
            start_time (float): Thời gian bắt đầu pipeline
            
        Returns:
            dict: Tóm tắt kết quả cuối cùng
        """
        print("=" * 100)
        print("TÓM TẮT KẾT QUẢ CUỐI CÙNG ")
        print("=" * 100)
        
        if not analysis_results:
            print(" Không có kết quả phân tích để tóm tắt")
            return None
        
        # Lấy thông tin từ analysis_results
        summary = analysis_results.get('summary', {})
        missing_data = analysis_results.get('missing_data')
        duplicates = analysis_results.get('duplicates', {})
        outliers = analysis_results.get('outliers', [])
        
        # ==========================================
        # THÔNG TIN DATASET
        # ==========================================
        print("THÔNG TIN DATASET:")
        print(f"    Tổng số dòng           : {summary.get('total_rows', 0):,}")
        print(f"    Tổng số cột            : {summary.get('total_columns', 0)}")
        print(f"    Cột số                 : {summary.get('numerical_columns', 0)}")
        print(f"    Cột phân loại          : {summary.get('categorical_columns', 0)}")
        
        # ==========================================
        # CHẤT LƯỢNG DỮ LIỆU
        # ==========================================
        print(f"\nCHẤT LƯỢNG DỮ LIỆU:")
        
        # Missing values
        if missing_data:
            missing_cols = len(missing_data.get('columns_with_missing', []))
            total_missing = missing_data.get('total_missing', 0)
            overall_pct = missing_data.get('overall_percentage', 0)
            print(f"    Missing values         : {missing_cols} cột ({total_missing:,} cells, {overall_pct:.2f}%)")
            if missing_cols > 0:
                cols_list = missing_data.get('columns_with_missing', [])
                print(f"    Cột có missing         : {cols_list}")
        else:
            print(f"    Missing values         : Không có")
        
        # Duplicates
        total_dups = duplicates.get('total_duplicates', 0)
        total_dups_pct = duplicates.get('total_percentage', 0)
        feature_dups = duplicates.get('exclude_id', {}).get('duplicates_keepfirst', 0)
        feature_dups_pct = duplicates.get('exclude_id', {}).get('percentage', 0)
        id_dups = duplicates.get('id_duplicates', {}).get('count', 0)
        id_dups_pct = duplicates.get('id_duplicates', {}).get('percentage', 0)

        print(f"    Dòng trùng lặp toàn bộ : {total_dups} ({total_dups_pct:.2f}%)")
        print(f"    Dòng trùng lặp feature : {feature_dups} ({feature_dups_pct:.2f}%)")
        print(f"    ID trùng lặp           : {id_dups} ({id_dups_pct:.2f}%)")

        # Outliers
        if outliers:
            total_outliers = sum(item.get('Outlier_Count', 0) for item in outliers)
            print(f"    Outliers               : {len(outliers)} cột đã phân tích ({total_outliers:,} values)")
             
            # Phân loại outliers theo type
            outlier_by_type = {}
            for item in outliers:
                item_type = item.get('Type', 'Unknown')
                outlier_by_type[item_type] = outlier_by_type.get(item_type, 0) + 1
            
            print(f"    Phân loại              : {dict(outlier_by_type)}")
        else:
            print(f"    Outliers               : Không có dữ liệu để phân tích")
        
        # Categorical consistency
        print(f"    Categorical data       : Đã phân tích và hiển thị chi tiết")
        
        # ==========================================
        # HIỆU SUẤT XỬ LÝ
        # ==========================================
        current_time = time.time()
        elapsed_time = current_time - start_time
        total_rows = summary.get('total_rows', 0)
        
        print(f"\nHIỆU SUẤT XỬ LÝ:")
        print(f"    Thời gian xử lý        : {elapsed_time:.2f} giây")
        print(f"    Tốc độ xử lý           : {total_rows / elapsed_time:.0f} dòng/giây" if elapsed_time > 0 else "   🚀 Tốc độ xử lý       : N/A")
        print(f"    Bộ nhớ sử dụng         : {self.dataset.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        
        # ==========================================
        # TRẠNG THÁI HỆ THỐNG
        # ==========================================
        print(f"\nTRẠNG THÁI HỆ THỐNG:")
        print(f"    Tính toàn vẹn dữ liệu  : Được duy trì")
        print(f"    Log system             : Hoạt động ({len(self.processing_log)} actions)")
        print(f"    Error handling         : Ổn định")
       
        
        # Tạo summary dict để trả về
        final_summary = {
            'dataset_info': summary,
            'data_quality': {
                'missing_info': missing_data,
                'duplicates_info': duplicates,
                'outliers_count': len(outliers) if outliers else 0
            },
            'performance': {
                'processing_time': elapsed_time,
                'processing_rate': total_rows / elapsed_time if elapsed_time > 0 else 0,
                'memory_usage_mb': self.dataset.memory_usage(deep=True).sum() / 1024 / 1024
            },
            'system_status': 'healthy'
        }
        
        return final_summary

    #=================================================================================================================================
    # HÀM HIỂN THỊ LOG XỬ LÝ
    def _display_processing_log(self):
        """
        Hiển thị log các hành động đã thực hiện trong quá trình xử lý
        """
        print("=" * 100)
        print("LOG CÁC HÀNH ĐỘNG THỰC HIỆN")
        print("=" * 100)
        
        if not self.processing_log:
            print("Không có log nào được ghi lại")
            return
        
        print(f"Tổng số hành động: {len(self.processing_log)}")
        print()
        
        for i, log_entry in enumerate(self.processing_log, 1):
            timestamp = log_entry['timestamp'].strftime('%H:%M:%S')
            action = log_entry['action']
            details = log_entry.get('details', '')
            
            print(f"{i:2d}. [{timestamp}] {action}")
            if details:
                print(f"    └─ {details}")
        
        print(f"\nĐã hoàn thành {len(self.processing_log)} hành động")

# =================================================================================================================================
# HÀM MAIN 
def main():
    """
    Hàm main 
    """
    
    # Đường dẫn đến dataset
    dataset_path = r"C:\Project\Machine_Learning\Machine_Learning\Dataset\Customer_Behavior.csv"
    
    try:
        # Khởi tạo processor
        processor = DataProcessingWrangling(dataset_path) # Khởi tạo đối tượng 
        
        # Chạy toàn bộ pipeline
        results = processor.run_complete_data_analysis()
        
        # Kiểm tra kết quả
        if results and results.get('success'):
            print("\nPipeline đã hoàn thành")
            print(f"Dataset đã được phân tích: {results['dataset_shape']}")
            print(f"Tổng thời gian: {results['total_time']:.2f} giây")
        else:
            print("\nPipeline gặp lỗi")
            if results:
                print(f"Lỗi: {results.get('error', 'Unknown error')}")
        
    except Exception as e:
        print(f"\nLỖI KHỞI TẠO: {e}")
        import traceback
        print("Chi tiết lỗi:")
        print(traceback.format_exc())
    

# =================================================================================================================================
if __name__ == "__main__":
    main()



























