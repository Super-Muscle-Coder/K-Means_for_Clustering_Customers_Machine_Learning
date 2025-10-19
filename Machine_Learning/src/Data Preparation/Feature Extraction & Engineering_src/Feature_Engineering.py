from datetime import datetime  
import pandas as pd
import numpy as np 
import warnings
import os
import time 
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, LabelEncoder
import matplotlib.pyplot as plt 
import seaborn as sns
warnings.filterwarnings('ignore')

class FeatureEngineering:
    """
    Class thực hiện các bước của Feature Extraction & Engineering trong quá trình Data Preparation.

    Các chức năng chính:
    - Feature Derivation: Tạo các đặc trưng mới từ các đặc trưng hiện có.
    - Feature Transformation: Biến đổi các đặc trưng để phù hợp với mô hình học máy.
    - Feature Combination: Kết hợp các đặc trưng để tạo ra các đặc trưng mới.
    - Feature Engineering Logging: Ghi lại các bước và thay đổi trong quá trình Feature Engineering.
    """

    def __init__(self, dataset_path): 
        """
        Khởi tạo lớp FeatureEngineering với đường dẫn đến tập dữ liệu.

        Tham số:
        - dataset_path (str): Đường dẫn đến tập dữ liệu.
        """
        self.dataset_path = dataset_path
        self.dataset = None
        self.engineered_dataset = None
        self.processing_log = []
        self.feature_stats = {}
        self.new_features = []
        self.original_shape = None  
        self.current_year = datetime.now().year
    
    def log_action(self, action, details=""):
        """
        Ghi lại các hành động và chi tiết trong quá trình Feature Engineering.
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
            self.original_shape = self.dataset.shape  # Lưu shape gốc ngay khi load

            # Chuyển đổi sang kiểu datetime cho cột 'Dt_Customer'
            if 'Dt_Customer' in self.dataset.columns:
                if self.dataset['Dt_Customer'].dtype == 'object':
                    self.dataset['Dt_Customer'] = pd.to_datetime(self.dataset['Dt_Customer'])
            
            print(f"Dữ liệu đã được tải thành công: {self.dataset.shape[0]:,} dòng × {self.dataset.shape[1]} cột")
            print()
            
            self.log_action("Tải dữ liệu thành công", f"Dataset shape: {self.dataset.shape}")
            return self.dataset

        except Exception as e:
            print(f"Lỗi khi tải dữ liệu: {e}")
            return None

    # =================================================================================================================
    # HÀM XUẤT BÁO CÁO TEXT 
    def export_text_report(self, output_dir=None):
        """
        Xuất báo cáo text chi tiết về quá trình Feature Engineering
        """
        if output_dir is None:
            output_dir = r"C:\Project\Machine_Learning\Machine_Learning\output\Extraction&Engineering_report"

        os.makedirs(output_dir, exist_ok=True)

        report_file = os.path.join(output_dir, f"Feature_Engineering_report.txt")

        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                # Header
                f.write("=" * 100 + "\n")
                f.write("THỰC HIỆN QUY TRÌNH FEATURE ENGINEERING HOÀN CHỈNH\n")
                f.write("=" * 100 + "\n")
                f.write("Bước 1: TẢI VÀ ĐỌC DỮ LIỆU\n\n")
        
                if self.dataset is not None:
                    f.write("Dữ liệu đã được tải thành công\n")
                    f.write(f"Dữ liệu đã được tải thành công: {self.dataset.shape[0]:,} dòng × {self.dataset.shape[1]} cột\n\n")
        
                # Ghi chi tiết các bước từ processing log
                current_step = 2  # Bắt đầu từ bước 2 vì bước 1 đã viết ở trên
                for log_entry in self.processing_log:
                    timestamp = log_entry['timestamp'].strftime('%H:%M:%S')
                    action = log_entry['action']
                    details = log_entry.get('details', '')
            
                    # Kiểm tra nếu là bước chính
                    if any(keyword in action for keyword in [
                        'Tạo đặc trưng Age', 
                        'Tạo đặc trưng Tenure', 
                        'Tạo đặc trưng Total_Spent',
                        'Tạo đặc trưng AvgPerPurchase',
                        'Log transform biến chi tiêu',
                        'Tạo đặc trưng phân nhóm tuổi',
                        'Tạo bins cho thu nhập',
                        'Trích xuất thông tin datetime',
                        'Tạo đặc trưng TotalChildren',
                        'Xuất dữ liệu feature engineered'
                    ]):
                        # Tạo tiêu đề bước dựa trên action
                        if 'Tạo đặc trưng Age' in action:
                            step_title = "TẠO ĐẶC TRƯNG TUỔI DỰA VÀO NĂM SINH CỦA KHÁCH HÀNG"
                        elif 'Tạo đặc trưng Tenure' in action:
                            step_title = "TẠO ĐẶC TRƯNG TENURE DỰA VÀO NGÀY ĐĂNG KÝ KHÁCH HÀNG"
                        elif 'Tạo đặc trưng Total_Spent' in action:
                            step_title = "TẠO ĐẶC TRƯNG TỔNG CHI TIÊU"
                        elif 'Tạo đặc trưng AvgPerPurchase' in action:
                            step_title = "TẠO ĐẶC TRƯNG GIÁ TRỊ TRUNG BÌNH MỖI LẦN MUA HÀNG"
                        elif 'Log transform biến chi tiêu' in action:
                            step_title = "ÁP DỤNG LOG TRANSFORM CHO CÁC BIẾN CHI TIÊU"
                        elif 'Tạo đặc trưng phân nhóm tuổi' in action:
                            step_title = "TẠO ĐẶC TRƯNG PHÂN NHÓM TUỔI"
                        elif 'Tạo bins cho thu nhập' in action:
                            step_title = "TẠO ĐẶC TRƯNG PHÂN NHÓM THU NHẬP"
                        elif 'Trích xuất thông tin datetime' in action:
                            step_title = "TRÍCH XUẤT THÔNG TIN THỜI GIAN TỪ NGÀY ĐĂNG KÝ"
                        elif 'Tạo đặc trưng TotalChildren' in action:
                            step_title = "TẠO ĐẶC TRƯNG TỔNG SỐ CON"
                        elif 'Xuất dữ liệu feature engineered' in action:
                            step_title = "XUẤT DỮ LIỆU ĐÃ FEATURE ENGINEERING"
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
                
            print(f"Đã xuất báo cáo Feature Engineering ra file: {report_file}")
            return report_file
    
        except Exception as e:
            print(f"Lỗi khi xuất báo cáo: {e}")
            return None
    # =================================================================================================================================
    # FEATURE DERIVATION - TẠO ĐẶC TRƯNG MỚI TỪ DỮ LIỆU GỐC
    def create_age_feature(self):
        """
        Tạo đặc trưng 'Age' từ cột 'Year_Birth'.
        """
        print("="*100)
        print("Bước 2: TẠO ĐẶC TRƯNG TUỔI DỰA VÀO NĂM SINH CỦA KHÁCH HÀNG")
        print()

        if 'Year_Birth' not in self.dataset.columns:
            print("Cột 'Year_Birth' không tồn tại trong dữ liệu. Bỏ qua bước này.")
            return

        # Tạo cột 'Age'
        self.dataset['Age'] = self.current_year - self.dataset['Year_Birth']
        print(f"Đặc trưng 'Age' đã được tạo thành công từ cột 'Year_Birth'")
        print(f"   Công thức: Age = {self.current_year} - Year_Birth")
        print()

        # Thống kê dành cho đặc trưng 'Age'
        age_stats = {
            'min': self.dataset['Age'].min(),
            'max': self.dataset['Age'].max(),
            'mean': self.dataset['Age'].mean(),
            'median': self.dataset['Age'].median(),
            'std': self.dataset['Age'].std()
        }
        
        print("THỐNG KÊ ĐẶC TRƯNG AGE:")
        print(f"- Tuổi nhỏ nhất       : {age_stats['min']:.0f}")
        print(f"- Tuổi lớn nhất       : {age_stats['max']:.0f}")
        print(f"- Tuổi trung bình     : {age_stats['mean']:.1f}")
        print(f"- Tuổi trung vị       : {age_stats['median']:.1f}")
        print(f"- Độ lệch chuẩn       : {age_stats['std']:.1f}")
        print()

        # Ghi nhận đặc trưng mới
        self.new_features.append('Age')
        self.feature_stats['Age'] = age_stats
        self.log_action("Tạo đặc trưng Age", 
                       f"Age = {self.current_year} - Year_Birth, Range: [{age_stats['min']:.0f}, {age_stats['max']:.0f}]")

    # ================================================================================================================================
    # TẠO ĐẶC TRƯNG TENURE TỪ CỘT 'Dt_Customer'
    def create_tenure_feature(self):
        """
        Tạo đặc trưng 'Tenure' từ cột 'Dt_Customer'.
        """
        print("="*100)
        print("Bước 3: TẠO ĐẶC TRƯNG TENURE DỰA VÀO NGÀY ĐĂNG KÝ KHÁCH HÀNG")
        print()

        if 'Dt_Customer' not in self.dataset.columns:
            print("Cột 'Dt_Customer' không tồn tại trong dữ liệu. Bỏ qua bước này.")
            return

        # Tạo cột 'Tenure' (tính bằng năm)
        current_date = pd.Timestamp.now()
        self.dataset['Tenure'] = (current_date - self.dataset['Dt_Customer']).dt.days // 365
        
        print(f"Đặc trưng 'Tenure' đã được tạo thành công từ cột 'Dt_Customer'")
        print(f"   Ngày tham chiếu: {current_date.strftime('%Y-%m-%d')}")
        print(f"   Công thức: Tenure = (Ngày hiện tại - Dt_Customer) / 365 ngày")
        print()

        # Thống kê dành cho đặc trưng 'Tenure'
        tenure_stats = {
            'min': self.dataset['Tenure'].min(),
            'max': self.dataset['Tenure'].max(), 
            'mean': self.dataset['Tenure'].mean(),
            'median': self.dataset['Tenure'].median(),
            'std': self.dataset['Tenure'].std()
        }

        print("THỐNG KÊ ĐẶC TRƯNG TENURE:")
        print(f"- Tenure nhỏ nhất     : {tenure_stats['min']:.0f} năm")
        print(f"- Tenure lớn nhất     : {tenure_stats['max']:.0f} năm")
        print(f"- Tenure trung bình   : {tenure_stats['mean']:.1f} năm")
        print(f"- Tenure trung vị     : {tenure_stats['median']:.1f} năm")
        print(f"- Độ lệch chuẩn       : {tenure_stats['std']:.1f} năm")
        print()

        # Ghi nhận đặc trưng mới
        self.new_features.append('Tenure')
        self.feature_stats['Tenure'] = tenure_stats
        self.log_action("Tạo đặc trưng Tenure", 
                       f"Tenure từ Dt_Customer, Range: [{tenure_stats['min']:.0f}, {tenure_stats['max']:.0f}] năm")

    # ================================================================================================================================
    # TẠO ĐẶC TRƯNG TOTAL_SPENT TỪ CÁC CỘT CHI TIÊU
    def create_total_spending_feature(self):
        """
        Tạo đặc trưng 'Total_Spent' từ các cột chi tiêu.
        """
        print("="*100)
        print("Bước 4: TẠO ĐẶC TRƯNG TỔNG CHI TIÊU")
        print()

        # Danh sách các cột chi tiêu
        spending_cols = [col for col in self.dataset.columns if col.startswith('Mnt')]

        if not spending_cols:
            print("Không tìm thấy cột chi tiêu nào (Mnt*). Bỏ qua bước này.")
            return

        # Tạo cột 'Total_Spent'
        self.dataset['Total_Spent'] = self.dataset[spending_cols].sum(axis=1)
        
        print(f"Đặc trưng 'Total_Spent' đã được tạo thành công")
        print(f"   Các cột được sử dụng: {', '.join(spending_cols)}")
        print(f"   Công thức: Total_Spent = tổng của {len(spending_cols)} cột Mnt*")
        print()
        
        # Thống kê dành cho đặc trưng 'Total_Spent'
        spending_stats = {
            'min': self.dataset['Total_Spent'].min(),
            'max': self.dataset['Total_Spent'].max(),
            'mean': self.dataset['Total_Spent'].mean(),
            'median': self.dataset['Total_Spent'].median(),
            'std': self.dataset['Total_Spent'].std()
        }

        print("THỐNG KÊ ĐẶC TRƯNG TOTAL_SPENT:")
        print(f"- Chi tiêu nhỏ nhất   : {spending_stats['min']:,.0f}")
        print(f"- Chi tiêu lớn nhất   : {spending_stats['max']:,.0f}")
        print(f"- Chi tiêu trung bình : {spending_stats['mean']:,.0f}")
        print(f"- Chi tiêu trung vị   : {spending_stats['median']:,.0f}")
        print(f"- Độ lệch chuẩn       : {spending_stats['std']:,.0f}")
        print()

        # Ghi nhận đặc trưng mới
        self.new_features.append('Total_Spent')
        self.feature_stats['Total_Spent'] = spending_stats
        self.log_action("Tạo đặc trưng Total_Spent", 
                       f"Tổng từ {len(spending_cols)} cột Mnt*, Range: [{spending_stats['min']:,.0f}, {spending_stats['max']:,.0f}]")

    # ================================================================================================================================
    # TẠO ĐẶC TRƯNG AvgPerPurchase = Total_Spent / TotalPurchases 
    def create_avg_per_purchase_feature(self):
        """
        Tạo đặc trưng 'AvgPerPurchase' = Total_Spent / TotalPurchases.
        """
        print("="*100)
        print("Bước 5: TẠO ĐẶC TRƯNG GIÁ TRỊ TRUNG BÌNH MỖI LẦN MUA HÀNG")
        print()

        # Kiểm tra cột Total_Spent
        if 'Total_Spent' not in self.dataset.columns:
            print("Cột 'Total_Spent' không tồn tại. Vui lòng tạo đặc trưng 'Total_Spent' trước.")
            return

        # Tìm các cột số lần mua hàng
        purchase_cols = [col for col in self.dataset.columns if col.startswith('Num') and 'Purchases' in col]

        if not purchase_cols:
            print("Không tìm thấy cột số lần mua hàng nào (Num*Purchases). Bỏ qua bước này.")
            return
        
        # Tạo cột 'TotalPurchases'
        self.dataset['TotalPurchases'] = self.dataset[purchase_cols].sum(axis=1)

        # Tạo cột 'AvgPerPurchase'
        self.dataset['AvgPerPurchase'] = np.where(
            self.dataset['TotalPurchases'] > 0,
            self.dataset['Total_Spent'] / self.dataset['TotalPurchases'],
            0
        )
        
        print(f"Đặc trưng 'AvgPerPurchase' đã được tạo thành công")
        print(f"   Công thức: AvgPerPurchase = Total_Spent / TotalPurchases")
        print(f"   Các cột số lần mua: {purchase_cols}")
        print()

        # Thống kê AvgPerPurchase
        non_zero_avg = self.dataset[self.dataset['AvgPerPurchase'] > 0]['AvgPerPurchase']
        avg_stats = {
            'min': non_zero_avg.min() if len(non_zero_avg) > 0 else 0,
            'max': non_zero_avg.max() if len(non_zero_avg) > 0 else 0,
            'mean': non_zero_avg.mean() if len(non_zero_avg) > 0 else 0,
            'median': non_zero_avg.median() if len(non_zero_avg) > 0 else 0,
            'zero_purchases': (self.dataset['TotalPurchases'] == 0).sum()
        }
    
        print("THỐNG KÊ ĐẶC TRƯNG AVERAGE PER PURCHASE:")
        print(f"- Giá trị nhỏ nhất            : {avg_stats['min']:,.0f}")
        print(f"- Giá trị lớn nhất            : {avg_stats['max']:,.0f}")
        print(f"- Trung bình                  : {avg_stats['mean']:,.0f}")
        print(f"- Trung vị                    : {avg_stats['median']:,.0f}")
        print(f"- Khách hàng chưa mua gì      : {avg_stats['zero_purchases']} ({avg_stats['zero_purchases']/len(self.dataset)*100:.1f}%)")
        print()

        # Ghi nhận đặc trưng mới
        self.new_features.extend(['TotalPurchases', 'AvgPerPurchase'])
        self.feature_stats['AvgPerPurchase'] = avg_stats
        self.log_action("Tạo đặc trưng AvgPerPurchase", 
                       f"AvgPerPurchase = Total_Spent / TotalPurchases, Range: [{avg_stats['min']:,.0f}, {avg_stats['max']:,.0f}]")

    # ================================================================================================================================
    # FEATURE TRANSFORMATION - BIẾN ĐỔI ĐẶC TRƯNG
    def apply_log_transform_spending(self):
        """
        Áp dụng biến đổi log cho các cột chi tiêu để giảm độ lệch.
        """
        print("=" * 100)
        print("Bước 6: ÁP DỤNG LOG TRANSFORM CHO CÁC BIẾN CHI TIÊU")
        print()

        # Danh sách các cột chi tiêu
        spending_cols = [col for col in self.dataset.columns if col.startswith('Mnt')]

        if not spending_cols:
            print("Không tìm thấy cột chi tiêu nào để áp dụng log transform. Bỏ qua bước này.")
            return

        print(f"Áp dụng log transform cho {len(spending_cols)} cột chi tiêu:")
        print()
        
        transformed_cols = []
        for col in spending_cols:
            # Tạo cột log transform (sử dụng log1p để tránh log(0))
            log_col_name = f"{col}_Log"
            self.dataset[log_col_name] = np.log1p(self.dataset[col])
            transformed_cols.append(log_col_name)

            # Thống kê trước và sau transform
            original_skew = self.dataset[col].skew()
            transformed_skew = self.dataset[log_col_name].skew()
            improvement = original_skew - transformed_skew

            print(f"Log transform cho {col}:")
            print(f"   - Skewness trước transform: {original_skew:>7.3f}")
            print(f"   - Skewness sau transform  : {transformed_skew:>7.3f}")
            print(f"   - Cải thiện skewness      : {improvement:>7.3f} {'✅' if improvement > 0 else '⚠️'}")
            print()

        print(f"Đã tạo thành công {len(transformed_cols)} cột log-transformed:")
        for i, col in enumerate(transformed_cols, 1):
            print(f"   {i:2d}. {col}")
        print()
        
        # Ghi nhận đặc trưng mới
        self.new_features.extend(transformed_cols)
        self.log_action("Log transform biến chi tiêu", 
                       f"Đã transform {len(spending_cols)} cột spending, tạo {len(transformed_cols)} cột mới")

    # =======================================================================================================================
    # TẠO BINS CHO ĐẶC TRƯNG 'Age'
    def create_bin_age_feature(self):
        """
        Tạo đặc trưng phân nhóm 'Age_Group' từ đặc trưng 'Age'.
        """
        print("="*100)
        print("Bước 7: TẠO ĐẶC TRƯNG PHÂN NHÓM TUỔI")
        print()

        if 'Age' not in self.dataset.columns:
            print("Cột 'Age' không tồn tại trong dữ liệu. Bỏ qua bước này.")
            return

        # Định nghĩa các bin và nhãn cho nhóm tuổi
        age_bins = [0, 25, 35, 45, 55, 65, np.inf]
        age_labels = ['Gen Z (18-25)', 'Millennials (26-35)', 'Gen X (36-45)', 
                      'Gen X+ (46-55)', 'Baby Boomers (56-65)', 'Seniors (65+)']
    
        # Tạo age groups
        self.dataset['Age_Group'] = pd.cut(self.dataset['Age'], bins=age_bins, labels=age_labels, right=False)
        
        print(f"Đặc trưng 'Age_Group' đã được tạo thành công từ cột 'Age'")
        print(f"   Phân chia thành {len(age_labels)} nhóm tuổi")
        print()

        # Thống kê phân phối nhóm tuổi
        age_distribution = self.dataset['Age_Group'].value_counts().sort_index()

        print("PHÂN BỐ KHÁCH HÀNG THEO NHÓM TUỔI:")
        for group, count in age_distribution.items():
            pct = count / len(self.dataset) * 100
            bar = "█" * int(pct / 2)  # Tạo biểu đồ thanh đơn giản
            print(f"   {group:<22}: {count:>4} khách hàng ({pct:>5.1f}%) {bar}")
        print()

        # Ghi nhận đặc trưng mới
        self.new_features.append('Age_Group')
        self.log_action("Tạo đặc trưng phân nhóm tuổi", 
                       f"Từ cột 'Age' thành 'Age_Group' với {len(age_labels)} nhóm tuổi")

    # ===========================================================================================================================
    # TẠO BINS CHO ĐẶC TRƯNG 'Income'
    def create_bin_income_feature(self):
        """
        Tạo bins cho thu nhập (Income) dựa trên quartiles.
        """
        print("=" * 100)
        print("Bước 8: TẠO ĐẶC TRƯNG PHÂN NHÓM THU NHẬP")
        print()
        
        if 'Income' not in self.dataset.columns:
            print("Cột 'Income' không tồn tại trong dữ liệu. Bỏ qua bước này.")
            return

        # Tính percentiles cho income
        income_percentiles = self.dataset['Income'].quantile([0.25, 0.5, 0.75]).values
    
        # Định nghĩa bins cho income
        income_bins = [0, income_percentiles[0], income_percentiles[1], income_percentiles[2], float('inf')]
        income_labels = ['Thu nhập thấp', 'Thu nhập trung bình thấp', 'Thu nhập trung bình cao', 'Thu nhập cao']
    
        # Tạo income groups
        self.dataset['Income_Group'] = pd.cut(self.dataset['Income'], bins=income_bins, labels=income_labels, right=False)
    
        print(f"Đặc trưng 'Income_Group' đã được tạo thành công")
        print(f"   Phân chia dựa trên quartiles thành {len(income_labels)} nhóm thu nhập")
        print()

        print("NGƯỠNG PHÂN CHIA THU NHẬP:")
        print(f"   - Thu nhập thấp           : < {income_percentiles[0]:>8,.0f}")
        print(f"   - Thu nhập trung bình thấp: {income_percentiles[0]:>8,.0f} - {income_percentiles[1]:>8,.0f}")
        print(f"   - Thu nhập trung bình cao : {income_percentiles[1]:>8,.0f} - {income_percentiles[2]:>8,.0f}")
        print(f"   - Thu nhập cao            : > {income_percentiles[2]:>8,.0f}")
        print()
        
        # Thống kê phân bố
        income_distribution = self.dataset['Income_Group'].value_counts()
    
        print("PHÂN BỐ KHÁCH HÀNG THEO NHÓM THU NHẬP:")
        for group, count in income_distribution.items():
            pct = count / len(self.dataset) * 100
            bar = "█" * int(pct / 2)
            print(f"   {group:<27}: {count:>4} khách hàng ({pct:>5.1f}%) {bar}")
        print()

        # Ghi nhận đặc trưng mới
        self.new_features.append('Income_Group')
        self.log_action("Tạo bins cho thu nhập", f"4 nhóm thu nhập dựa trên quartiles")

    # ==========================================================================================================================
    # TRÍCH XUẤT THÔNG TIN TỪ NGÀY THÁNG (Dt_Customer)
    def extract_datetime_features(self):
        """
        Trích xuất thông tin từ ngày tháng (Dt_Customer).
        """
        print("=" * 100)
        print("Bước 9: TRÍCH XUẤT THÔNG TIN THỜI GIAN TỪ NGÀY ĐĂNG KÝ")
        print()
        
        if 'Dt_Customer' not in self.dataset.columns:
            print("Cột 'Dt_Customer' không tồn tại trong dữ liệu. Bỏ qua bước này.")
            return
        
        # Trích xuất các thông tin từ ngày tháng
        self.dataset['Customer_Year'] = self.dataset['Dt_Customer'].dt.year
        self.dataset['Customer_Month'] = self.dataset['Dt_Customer'].dt.month
        self.dataset['Customer_Quarter'] = self.dataset['Dt_Customer'].dt.quarter
        self.dataset['Customer_DayOfWeek'] = self.dataset['Dt_Customer'].dt.dayofweek
        
        # Tạo mùa trong năm
        def get_season(month):
            if month in [12, 1, 2]:
                return 'Winter'
            elif month in [3, 4, 5]:
                return 'Spring' 
            elif month in [6, 7, 8]:
                return 'Summer'
            else:
                return 'Fall'
        
        self.dataset['Customer_Season'] = self.dataset['Customer_Month'].apply(get_season)
        
        datetime_features = ['Customer_Year', 'Customer_Month', 'Customer_Quarter', 'Customer_DayOfWeek', 'Customer_Season']
        
        print(f"Đã trích xuất thành công {len(datetime_features)} đặc trưng thời gian:")
        print(f"   - Customer_Year      : Năm đăng ký ({self.dataset['Customer_Year'].min()} - {self.dataset['Customer_Year'].max()})")
        print(f"   - Customer_Month     : Tháng đăng ký (1-12)")
        print(f"   - Customer_Quarter   : Quý đăng ký (1-4)")
        print(f"   - Customer_DayOfWeek : Ngày trong tuần (0=Monday, 6=Sunday)")
        print(f"   - Customer_Season    : Mùa đăng ký (Spring/Summer/Fall/Winter)")
        print()
        
        # Phân bố theo mùa
        season_dist = self.dataset['Customer_Season'].value_counts()
        print("PHÂN BỐ KHÁCH HÀNG THEO MÙA ĐĂNG KÝ:")
        for season, count in season_dist.items():
            pct = count / len(self.dataset) * 100
            bar = "█" * int(pct / 2)
            print(f"   {season:<12}: {count:>4} khách hàng ({pct:>5.1f}%) {bar}")
        print()
        
        # Phân bố theo năm
        year_dist = self.dataset['Customer_Year'].value_counts().sort_index()
        print("PHÂN BỐ KHÁCH HÀNG THEO NĂM ĐĂNG KÝ:")
        for year, count in year_dist.items():
            pct = count / len(self.dataset) * 100
            bar = "█" * int(pct / 2)
            print(f"   {year}: {count:>4} khách hàng ({pct:>5.1f}%) {bar}")
        print()
        
        self.new_features.extend(datetime_features)
        self.log_action("Trích xuất thông tin datetime", f"Đã tạo {len(datetime_features)} đặc trưng từ Dt_Customer")
        
    # ==========================================================================================================================
    # FEATURE COMBINATION - KẾT HỢP ĐẶC TRƯNG
    def create_total_children_feature(self):
        """
        Tạo đặc trưng TotalChildren = Kidhome + Teenhome.
        """
        print("=" * 100)
        print("Bước 10: TẠO ĐẶC TRƯNG TỔNG SỐ CON")
        print()
        
        if 'Kidhome' not in self.dataset.columns or 'Teenhome' not in self.dataset.columns:
            print("Không tìm thấy cột 'Kidhome' hoặc 'Teenhome'. Bỏ qua bước này.")
            return
        
        # Tạo TotalChildren
        self.dataset['TotalChildren'] = self.dataset['Kidhome'] + self.dataset['Teenhome']
        
        print(f"Đặc trưng 'TotalChildren' đã được tạo thành công")
        print(f"   Công thức: TotalChildren = Kidhome + Teenhome")
        print()
        
        # Thống kê TotalChildren
        children_dist = self.dataset['TotalChildren'].value_counts().sort_index()
        
        print("PHÂN BỐ SỐ CON TRONG GIA ĐÌNH:")
        for num_children, count in children_dist.items():
            pct = count / len(self.dataset) * 100
            bar = "█" * int(pct / 4)
            print(f"   {num_children} con: {count:>4} gia đình ({pct:>5.1f}%) {bar}")
        print()
        
        # Tạo nhóm có con / không có con
        self.dataset['HasChildren'] = (self.dataset['TotalChildren'] > 0).astype(int)
        has_children_dist = self.dataset['HasChildren'].value_counts()
        
        print("PHÂN BỐ CÓ CON / KHÔNG CÓ CON:")
        labels = ['Không có con', 'Có con']
        for i, (value, count) in enumerate(has_children_dist.items()):
            pct = count / len(self.dataset) * 100
            bar = "█" * int(pct / 4)
            print(f"   {labels[value]:<15}: {count:>4} gia đình ({pct:>5.1f}%) {bar}")
        print()
        
        self.new_features.extend(['TotalChildren', 'HasChildren'])
        self.log_action("Tạo đặc trưng TotalChildren", 
                       f"Kidhome + Teenhome, {has_children_dist[1]} gia đình có con")
    
    # =================================================================================================================================
    # HÀM TỔNG HỢP - THỰC HIỆN TOÀN BỘ FEATURE ENGINEERING
    def run_complete_feature_engineering(self):
        """
        Hàm chính thực hiện toàn bộ quá trình Feature Engineering.
        """
        start_time = time.time()
        
        try:
            # ==========================================
            # BƯỚC 1: TẢI DỮ LIỆU
            # ==========================================
            data = self.load_data()
            if data is None:
                print("Không thể tải dữ liệu")
                return None
            
            # Sao chép dataset để xử lý
            self.engineered_dataset = self.dataset.copy()
            
            # ==========================================
            # BƯỚC 2-5: FEATURE DERIVATION
            # ==========================================
            self.create_age_feature()
            self.create_tenure_feature()
            self.create_total_spending_feature()
            self.create_avg_per_purchase_feature()
            
            # ==========================================
            # BƯỚC 6-9: FEATURE TRANSFORMATION
            # ==========================================
            self.apply_log_transform_spending()
            self.create_bin_age_feature()
            self.create_bin_income_feature()
            self.extract_datetime_features()
            
            # ==========================================
            # BƯỚC 10: FEATURE COMBINATION
            # ==========================================
            self.create_total_children_feature()
            
            # ==========================================
            # BƯỚC 11: TÓM TẮT KẾT QUẢ
            # ==========================================
            final_summary = self._generate_final_summary(start_time)
            
            # ==========================================
            # BƯỚC 12: XUẤT DỮ LIỆU
            # ==========================================
            self._export_engineered_data()
            
            # ==========================================
            # BƯỚC 13: HIỂN THỊ LOG
            # ==========================================
            self._display_processing_log()
            
            # ==========================================
            # BƯỚC 14: XUẤT BÁO CÁO TEXT
            # ==========================================
            try:
                self.export_text_report()
            except Exception as e:
                print(f"Lỗi khi xuất báo cáo text: {e}")

            # ==========================================
            # KẾT THÚC
            # ==========================================
            total_time = time.time() - start_time
            print("\n" + "=" * 100)
            print("HOÀN THÀNH FEATURE ENGINEERING")
            print("=" * 100)
            print(f" Tổng thời gian thực hiện : {total_time:.2f} giây")
            print(f" Tốc độ xử lý             : {len(self.dataset) / total_time:.0f} dòng/giây")
            print(f" Thời gian kết thúc       : {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            return {
                'success': True,
                'original_shape': self.original_shape,  
                'final_shape': self.dataset.shape,    
                'new_features': self.new_features,
                'feature_stats': self.feature_stats,
                'final_summary': final_summary,
                'total_time': total_time,
                'processing_log': self.processing_log
            }
            
        except Exception as e:
            print(f"\nLỖI TRONG FEATURE ENGINEERING PIPELINE: {e}")
            import traceback
            print("Chi tiết lỗi:")
            print(traceback.format_exc())
            return {
                'success': False,
                'error': str(e),
                'processing_log': self.processing_log
            }
    
    # =================================================================================================================================
    # HÀM TẠO TÓM TẮT KẾT QUẢ
    def _generate_final_summary(self, start_time):
        """Tạo tóm tắt kết quả feature engineering."""
        print("=" * 100)
        print("TÓM TẮT KẾT QUẢ FEATURE ENGINEERING")
        print("=" * 100)
        
        # ✅ Sửa logic tính toán số cột ban đầu
        original_cols = self.original_shape[1] if self.original_shape else 0
        current_cols = len(self.dataset.columns)
        new_features_count = len(self.new_features)
        
        print("THÔNG TIN DATASET:")
        print(f"    Số cột ban đầu         : {original_cols}")
        print(f"    Số cột sau engineering : {current_cols}")
        print(f"    Số đặc trưng mới tạo   : {new_features_count}")
        print(f"    Tổng số dòng           : {len(self.dataset):,}")
        
        print(f"\nCÁC ĐẶC TRƯNG MỚI ĐÃ TẠO ({new_features_count} đặc trưng):")
        
        # Nhóm theo loại đặc trưng (sửa logic phân nhóm)
        derivation_features = []
        transformation_features = []
        combination_features = []
        
        for feature in self.new_features:
            if feature in ['Age', 'Tenure', 'Total_Spent', 'TotalPurchases', 'AvgPerPurchase']:
                derivation_features.append(feature)
            elif '_Log' in feature or '_Group' in feature or 'Customer_' in feature:
                transformation_features.append(feature)
            elif feature in ['TotalChildren', 'HasChildren']:
                combination_features.append(feature)
        
        if derivation_features:
            print("   + Feature Derivation:")
            for i, feature in enumerate(derivation_features, 1):
                print(f"      {i:2d}. {feature}")
        
        if transformation_features:
            print("   + Feature Transformation:")
            for i, feature in enumerate(transformation_features, 1):
                print(f"      {i:2d}. {feature}")
        
        if combination_features:
            print("   + Feature Combination:")
            for i, feature in enumerate(combination_features, 1):
                print(f"      {i:2d}. {feature}")
        
        # Hiệu suất xử lý
        elapsed_time = time.time() - start_time
        print(f"\nHIỆU SUẤT XỬ LÝ:")
        print(f"      Thời gian xử lý        : {elapsed_time:.2f} giây")
        print(f"      Tốc độ xử lý           : {len(self.dataset) / elapsed_time:.0f} dòng/giây")
        print(f"      Bộ nhớ sử dụng         : {self.dataset.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        
        print(f"\nTRẠNG THÁI HỆ THỐNG:")
        print(f"      Tính toàn vẹn dữ liệu  : Được duy trì")
        print(f"      Log system             : Hoạt động ({len(self.processing_log)} actions)")
        print(f"      Feature engineering    : Hoàn thành")
        
        return {
            'original_columns': original_cols,
            'final_columns': current_cols,
            'new_features_count': new_features_count,
            'processing_time': elapsed_time,
            'new_features_list': self.new_features
        }
    
    # =================================================================================================================================
    # HÀM XUẤT DỮ LIỆU
    def _export_engineered_data(self):
        """Xuất dataset đã được feature engineering: CSV + Parquet + dtype metadata (JSON)."""
        import os, json
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        filename = f"Customer_Behavior_feature_engineered_{timestamp}.csv"
        output_path = f"C:\\Project\\Machine_Learning\\Machine_Learning\\Dataset\\{filename}"
        
        try:
            # --- Safe casts for storage ---
            # dummy / binary-like
            uint8_cols = [
                'Marital_Divorced','Marital_Married','Marital_Other',
                'Marital_Single','Marital_Together','Marital_Widow',
                'HasChildren'
            ]
            for c in uint8_cols:
                if c in self.dataset.columns:
                    if self.dataset[c].isna().sum() == 0:
                        try:
                            self.dataset[c] = self.dataset[c].astype('uint8')
                        except Exception:
                            # ensure 0/1 then cast
                            self.dataset[c] = self.dataset[c].apply(lambda x: 1 if x in (1, True) else 0).astype('uint8')

            # categorical reductions
            cat_cols = ['Marital_Status_Grouped','Age_Group','Income_Group','Customer_Season','Education','Marital_Status']
            for c in cat_cols:
                if c in self.dataset.columns:
                    try:
                        self.dataset[c] = self.dataset[c].astype('category')
                    except Exception:
                        pass

            # small integers
            small_int_cols = ['Age','Tenure','TotalPurchases','NumWebVisitsMonth','Kidhome','Teenhome']
            for c in small_int_cols:
                if c in self.dataset.columns:
                    if self.dataset[c].isna().sum() == 0:
                        try:
                            # choose uint8 if fits
                            if int(self.dataset[c].max()) <= 255 and int(self.dataset[c].min()) >= 0:
                                self.dataset[c] = self.dataset[c].astype('uint8')
                            else:
                                self.dataset[c] = self.dataset[c].astype('uint16')
                        except Exception:
                            pass

            # --- Export CSV ---
            self.dataset.to_csv(output_path, index=False)
            print("=" * 100)
            print("XUẤT DỮ LIỆU ĐÃ FEATURE ENGINEERING")
            print("=" * 100)
            print(f"Đã xuất dữ liệu thành công (CSV):")
            print(f"    File: {filename}")
            print(f"    Shape: {self.dataset.shape}")
            print(f"    Đường dẫn đầy đủ: {output_path}\n")
            self.log_action("Xuất dữ liệu feature engineered (CSV)", f"File: {filename}, Shape: {self.dataset.shape}")

            # --- Export Parquet to preserve dtypes (best practice) ---
            parquet_path = os.path.splitext(output_path)[0] + '.parquet'
            try:
                self.dataset.to_parquet(parquet_path, index=False)
                print(f"Đã xuất dữ liệu thành công (Parquet): {parquet_path}")
                self.log_action("Xuất dữ liệu feature engineered (Parquet)", f"File: {parquet_path}")
            except Exception as e:
                self.log_action("Warning parquet export failed", str(e))

            # --- Export dtype metadata ---
            dtype_map = {col: str(self.dataset[col].dtype) for col in self.dataset.columns}
            dtype_json_path = os.path.splitext(output_path)[0] + '_dtypes.json'
            try:
                with open(dtype_json_path, 'w', encoding='utf-8') as f:
                    json.dump(dtype_map, f, indent=2)
                print(f"Đã lưu metadata dtypes: {dtype_json_path}")
                self.log_action("Export dtype metadata", f"Saved dtype mapping to {dtype_json_path}")
            except Exception as e:
                self.log_action("Warning write dtype_map failed", str(e))

        except Exception as e:
            print(f"Lỗi khi xuất dữ liệu: {e}")
            self.log_action("Error export_engineered_data", str(e))
    
    # =================================================================================================================================
    # HÀM HIỂN THỊ LOG
    def _display_processing_log(self):
        """Hiển thị log các hành động đã thực hiện."""
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
            
            print(f"{i:2d}.  [{timestamp}] {action}")
            if details:
                print(f"   └─ {details}")
        
        print(f"\nĐã hoàn thành {len(self.processing_log)} hành động")


# =================================================================================================================================
# HÀM MAIN
def main():
    """
    Hàm main 
    """
    
    # Đường dẫn đến dataset đã được làm sạch
    dataset_path = r"C:\Project\Machine_Learning\Machine_Learning\Dataset\Customer_Behavior_cleaned_20251016_140301.csv"
    
    try:
        print("Bắt đầu Feature Engineering Pipeline")
        print()
        
        # Khởi tạo feature engineer
        feature_engineer = FeatureEngineering(dataset_path)
        
        # Chạy toàn bộ pipeline feature engineering
        results = feature_engineer.run_complete_feature_engineering()
        
        # Kiểm tra kết quả
        if results and results.get('success'):
            print("\n" + "="*60)
            print("FEATURE ENGINEERING PIPELINE HOÀN THÀNH!")
            print("="*60)
            print(f"Dataset gốc                : {results['original_shape']}")
            print(f"Dataset sau engineering    : {results['final_shape']}")
            print(f"Số đặc trưng mới           : {len(results['new_features'])}")
            print(f"Tổng thời gian             : {results['total_time']:.2f} giây")
            print("="*60)
        else:
            print("\nFEATURE ENGINEERING PIPELINE GẶP LỖI")
            if results:
                print(f"Chi tiết lỗi: {results.get('error', 'Unknown error')}")
        
    except Exception as e:
        print(f"\nLỖI KHỞI TẠO FEATURE ENGINEERING: {e}")
        import traceback
        print("Chi tiết lỗi:")
        print(traceback.format_exc())


# =================================================================================================================================
if __name__ == "__main__":
    main()

