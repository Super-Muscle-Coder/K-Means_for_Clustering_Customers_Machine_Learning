import pandas as pd
import numpy as np 
import time 
import os 
import warnings 
import json
from scipy import stats
import tempfile
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

warnings.filterwarnings('ignore')

class DataCleaning:
    """
    Class này chứa các phương thức để làm sạch và xử lý dữ liệu:
    - Xử lý giá trị bị thiếu ( missing values ) 
    - Xử lý giá trị ngoại lai ( outliers )
    - Chuẩn hóa dữ liệu ( normalization )
    - Mã hóa biến phân loại ( categorical encoding )
    - Giảm chiều dữ liệu ( dimensionality reduction )
    - Chuyển đổi kiểu dữ liệu ( data type conversion )
    """
    # Hàm khởi tạo đối tượng DataCleaning
    def __init__ (self, dataset_path = None, dataframe = None ):
        """
        Khởi tạo đối tượng DataCleaning với đường dẫn đến tập dữ liệu hoặc DataFrame.

        Args: 
            dataset_path (str): Đường dẫn đến tập dữ liệu (CSV, Excel, v.v.)
            dataframe (pd.DataFrame): DataFrame đã được tải sẵn
        """
        # Khởi tạo các biến:
        self.dataset_path = dataset_path 
        self.raw_data = None # Dữ liệu thô
        self.cleaned_data = None # Dữ liệu đã được làm sạch
        self.processing_log = [] # Nhật ký xử lý
        self.encoders = {} # Lưu trữ các encoders

        # Nếu truyền vào DataFrame:
        if dataframe is not None:
            self.raw_data = dataframe
            self.cleaned_data = dataframe.copy()
            self.log_action("Khởi tạo từ DataFrame")

        # Nếu truyền vào đường dẫn tập dữ liệu:
        elif dataset_path is not None:
            self.load_data() # Gọi hàm load_data để tải dữ liệu

    # ==================================================================================================================
    # Hàm load_data để tải dữ liệu từ file CSV
    def load_data(self): 
        try:
            # Đọc dữ liệu từ file CSV:
            print("="*100)
            print("Bước 1: TẢI VÀ ĐỌC DỮ LIỆU")
            self.dataset = pd.read_csv(self.dataset_path, sep="\t") # Nhớ bật sep='\t' nếu file là TSV

            if self.dataset is not None: # Kiểm tra nếu dữ liệu đã được tải thành công
                print("\nDữ liệu đã được tải thành công\n")
            else: # Nếu không thể tải dữ liệu
                print("\nKhông thể tải dữ liệu, vui lòng kiểm tra lại.")
                return None

            self.log_action("Tải dữ liệu thành công")
            
            self.raw_data = self.dataset # Lưu dữ liệu thô
            self.cleaned_data = self.dataset.copy() # Tạo bản sao để làm sạch
            return self.dataset # Trả về dữ liệu đã tải
        except Exception as e:
            print(f"Lỗi khi tải dữ liệu: {e}")
            return None
    # ==================================================================================================================
    def _create_output_directory(self):
        """
        Tạo thư mục để lưu biểu đồ Data Cleaning
        """
        output_dir = r"C:\Project\Machine_Learning\Machine_Learning\graph\Data Processing & Wrangling_graph\Data_Cleaning"
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    # ==================================================================================================================
    # HÀM GHI LOG HÀNH ĐỘNG
    def log_action(self, action, details=""):
        self.processing_log.append({ 
            'action': action,   # Hành động đã thực hiện
            'details': details, # Chi tiết về hành động
            'timestamp': pd.Timestamp.now() # Thời gian thực hiện hành động
        })
    # =================================================================================================================
    # HÀM XUẤT BÁO CÁO TEXT 
    def export_text_report(self, output_dir=None):
        """
        Xuất báo cáo text chi tiết về quá trình làm sạch dữ liệu
        """
        if output_dir is None:
            output_dir = r"C:\Project\Machine_Learning\Machine_Learning\report\Data Processing & Wrangling_report"
    
        os.makedirs(output_dir, exist_ok=True)
    
        report_file = os.path.join(output_dir, f"Data_Cleaning_report.log")
    
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                # Header
                f.write("=" * 100 + "\n")
                f.write("THỰC HIỆN QUY TRÌNH LÀM SẠCH DỮ LIỆU HOÀN CHỈNH\n")
                f.write("=" * 100 + "\n")
                f.write("Bước 1: TẢI VÀ ĐỌC DỮ LIỆU\n\n")
            
                if self.raw_data is not None:
                    f.write("Dữ liệu đã được tải thành công\n\n")
            
                # Ghi chi tiết các bước từ processing log
                current_step = 1
                for log_entry in self.processing_log:
                    timestamp = log_entry['timestamp'].strftime('%H:%M:%S')
                    action = log_entry['action']
                    details = log_entry.get('details', '')
                
                    # Kiểm tra nếu là bước chính
                    if any(keyword in action for keyword in ['Bước', 'Step', 'Tách dataset', 'Loại bỏ', 'Chuẩn hóa', 'Chuyển đổi', 'Encode', 'Xác thực']):
                        f.write(f"\n{'='*100}\n")
                        f.write(f"Bước {current_step}: {action.upper()}\n")
                        f.write(f"{'='*100}\n")
                        current_step += 1
                
                    f.write(f"[{timestamp}] {action}\n")
                    if details:
                        f.write(f"    └─ {details}\n")
                    f.write("\n")
                    
            print(f"Đã xuất báo cáo Data Cleaning ra file: {report_file}")
            return report_file
        
        except Exception as e:
            print(f"Lỗi khi xuất báo cáo: {e}")
            return None

        # =================================================================================================================
    # HÀM XỬ LÝ GIÁ TRỊ BỊ THIẾU
    def handle_missing_value(self):
        """
        Hàm xử lý giá trị bị thiếu bằng cách loại bỏ trực tiếp các dòng có missing values.
        
        Returns:
            dict: Thông tin về quá trình xử lý
        """
        print("="*100)
        print("Bước 2: XỬ LÝ GIÁ TRỊ BỊ THIẾU")

        if self.cleaned_data is None:
            print("Dữ liệu chưa được tải. Vui lòng tải dữ liệu trước khi xử lý.")
            return None

        # =====================================
        # BƯỚC 1: THỐNG KÊ MISSING VALUES
        # =====================================
        initial_missing_count = self.cleaned_data.isnull().sum().sum() # Tổng số giá trị bị thiếu ban đầu
        print(f"\nTổng số giá trị bị thiếu ban đầu: {initial_missing_count}")

        if initial_missing_count == 0:
            print("Không có giá trị bị thiếu trong dữ liệu.")
            self.log_action("Xử lý missing values", "Không có giá trị bị thiếu")
            return {"status": "no_missing", "missing_values_count": 0}

        # Phân tích chi tiết missing values
        missing_columns = self.cleaned_data.columns[self.cleaned_data.isnull().any().tolist()]
        print(f"\nCác cột có giá trị bị thiếu:")
        print("-" * 50)
    
        missing_info = {}
        for col in missing_columns:
            missing_count = self.cleaned_data[col].isnull().sum()
            missing_percent = (missing_count / len(self.cleaned_data)) * 100
            print(f"- {col:<20}: {missing_count:>4} giá trị ({missing_percent:>5.2f}%)")
            missing_info[col] = {
                'count': missing_count,
                'percentage': missing_percent
            }
        print("-" * 50)

        # =====================================
        # BƯỚC 2: XÁC ĐỊNH PHƯƠNG ÁN XỬ LÝ
        # =====================================
        print(f"\nPHƯƠNG ÁN XỬ LÝ:")
        print("Sẽ loại bỏ tất cả các dòng có missing values")

        # =====================================
        # BƯỚC 3: LOẠI BỎ CÁC DÒNG CÓ MISSING VALUES
        # =====================================
        
        before_rows = len(self.cleaned_data)
        rows_with_missing = self.cleaned_data.isnull().any(axis=1)
        missing_rows_count = rows_with_missing.sum()
        
        # Loại bỏ các dòng có missing values
        self.cleaned_data = self.cleaned_data.dropna()
        after_rows = len(self.cleaned_data)

        print(f"\nLoại bỏ thành công")
        print(f"  - Đã loại bỏ: {missing_rows_count:,} dòng")
        print(f"  - Dataset còn lại: {after_rows:,} dòng")

        # =====================================
        # BƯỚC 4: THỐNG KÊ KẾT QUẢ
        # =====================================
        print(f"\nTHỐNG KÊ KẾT QUẢ XỬ LÝ:")
        print("=" * 60)
        print(f"Dataset gốc              : {len(self.raw_data):,} dòng")
        print(f"Dòng có missing values   : {missing_rows_count:,} dòng")
        print(f"Dataset sau loại bỏ      : {after_rows:,} dòng")
        print(f"Tỷ lệ dữ liệu còn lại    : {after_rows/len(self.raw_data)*100:.2f}%")
        print("=" * 60)

        # =====================================
        # BƯỚC 5: GHI LOG VÀ TRẢ VỀ KẾT QUẢ  
        # =====================================
        self.log_action("Loại bỏ các dòng có missing values", 
                       f"Đã loại bỏ {missing_rows_count} dòng")
    
        result = {
            'status': 'success',
            'original_rows': len(self.raw_data),
            'missing_rows': missing_rows_count,
            'cleaned_rows': after_rows,
            'cleaned_percentage': after_rows/len(self.raw_data)*100,
            'missing_info': missing_info
        }

        # Tạo biểu đồ nếu có missing values
        if missing_rows_count > 0:
            print(f"\nTẠO BIỂU ĐỒ MISSING VALUES:")
            self._plot_missing_values_analysis(missing_info, missing_rows_count)

        print(f"\nHoàn thành xử lý missing values\n")        
        return result
        return result

    # =================================================================================================================
    # HÀM LẤY DATASET ĐÃ LÀM SẠCH  
    def get_cleaned_dataset(self):
        """
        Trả về dataset đã làm sạch (không có missing values)
        
        Returns:
            pd.DataFrame: Dataset đã làm sạch
        """
        if self.cleaned_data is None:
            print("Chưa có dataset đã làm sạch. Hãy chạy handle_missing_value() trước.")
            return None
        
        print(f"Dataset đã làm sạch: {self.cleaned_data.shape}")
        print(f"Missing values: {self.cleaned_data.isnull().sum().sum()}")
        
        return self.cleaned_data

    # =================================================================================================================
    # HÀM LẤY MINI DATASET  
    def get_missing_subset(self):
        """
        Trả về mini dataset chứa các dòng có missing values
        
        Returns:
            pd.DataFrame: Mini dataset chứa missing values
        """
        if not hasattr(self, 'missing_data_subset') or self.missing_data_subset is None:
            print("Chưa có mini dataset missing values. Hãy chạy handle_missing_value() trước.")
            return None
            
        print(f"Mini dataset missing : {self.missing_data_subset.shape}")
        print(f"Missing values       : {self.missing_data_subset.isnull().sum().sum()}")
        
        return self.missing_data_subset

    # =================================================================================================================
    # HÀM VẼ BIỂU ĐỒ PHÂN TÍCH MISSING VALUES
    def _plot_missing_values_analysis(self, missing_info, missing_rows_count):
        """
        Vẽ biểu đồ phân tích missing values
        """
        try:
            output_dir = self._create_output_directory()
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 7))
        
            # SUBPLOT 1: Bar chart missing count by column
            if missing_info:
                cols = list(missing_info.keys())
                counts = [missing_info[col]['count'] for col in cols]
            
                bars = ax1.bar(range(len(cols)), counts, color='salmon', alpha=0.7)
                ax1.set_xlabel('Các cột có missing', fontweight='bold')
                ax1.set_ylabel('Số lượng missing', fontweight='bold')
                ax1.set_title('Phân bố Missing Values theo cột', fontweight='bold')
                ax1.set_xticks(range(len(cols)))
                ax1.set_xticklabels(cols, rotation=45, ha='right')
            
                # Thêm số liệu lên bars
                for bar, count in zip(bars, counts):
                    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + count*0.01,
                            f'{count}', ha='center', va='bottom', fontweight='bold')
        
            # SUBPLOT 2: Pie chart Clean vs Missing ratio
            clean_rows = len(self.cleaned_data)
            labels = ['Dữ liệu sạch', 'Có missing values']
            sizes = [clean_rows, missing_rows_count]
            colors = ['lightgreen', 'lightcoral']
        
            ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                    startangle=90, textprops={'fontweight': 'bold'})
            ax2.set_title('Tỷ lệ dữ liệu sạch vs Missing', fontweight='bold')
        
            # SUBPLOT 3: Summary table
            ax3.axis('off')
            summary_text = f"""
TỔNG QUAN XỬ LÝ MISSING VALUES

THỐNG KÊ:
    • Dataset gốc: {len(self.raw_data):,} dòng
    • Dòng có missing: {missing_rows_count:,}
    • Dòng sạch: {clean_rows:,}
    • Tỷ lệ sạch: {clean_rows/len(self.raw_data)*100:.1f}%

CHI TIẾT MISSING:
    • Số cột có missing: {len(missing_info) if missing_info else 0}
    • Tổng missing cells: {sum(missing_info[col]['count'] for col in missing_info) if missing_info else 0:,}

PHƯƠNG PHÁP XỬ LÝ:
    • Tách dataset thành 2 phần
    • Dataset chính: Chỉ dòng hoàn chỉnh
    • Mini dataset: Lưu dòng có missing
    • Không fill/drop → Giữ nguyên dữ liệu
"""
        
            ax3.text(0.05, 0.95, summary_text, transform=ax3.transAxes,
                    fontsize=11, verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.8', facecolor='lightblue', alpha=0.8))
        
            plt.suptitle(' PHÂN TÍCH VÀ XỬ LÝ MISSING VALUES', 
                        fontsize=16, fontweight='bold', y=0.98)
            plt.tight_layout()
        
            # Lưu biểu đồ
            filename = "missing_values_analysis.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
        
            print(f"     Đã lưu biểu đồ Missing Values: {filename}")
        
        except Exception as e:
            print(f"     Lỗi khi vẽ biểu đồ missing values: {e}")
    # =================================================================================================================
    # HÀM LOẠI BỎ DÒNG TRÙNG LẶP FEATURE (LOẠI TRỪ ID)
    def remove_feature_duplicates(self):
        """
        Loại bỏ các dòng trùng lặp feature (loại trừ cột ID), giữ lại dòng đầu tiên của mỗi nhóm.
        Returns:
            dict: Thông tin về số dòng đã loại bỏ và dataset sau khi loại bỏ
        """
        print("="*100)
        print("Bước 3: LOẠI BỎ DÒNG TRÙNG LẶP FEATURE\n")

        if self.cleaned_data is None:
            print("Dữ liệu chưa được tải hoặc làm sạch.")
            return None

        if 'ID' not in self.cleaned_data.columns:
            print("Không tìm thấy cột ID trong dữ liệu.")
            return None

        before_rows = len(self.cleaned_data)
        # Loại bỏ các dòng trùng lặp feature (giữ lại dòng đầu tiên)
        feature_cols = [col for col in self.cleaned_data.columns if col != 'ID']
        duplicated_mask = self.cleaned_data.duplicated(subset=feature_cols, keep='first')
        num_duplicates = duplicated_mask.sum()

        if num_duplicates == 0:
            print("Không có dòng trùng lặp feature để loại bỏ.")
            self.log_action("Loại bỏ dòng trùng lặp feature", "Không có dòng trùng lặp")
            return {"removed": 0, "after_rows": before_rows}

        # Loại bỏ các dòng bị trùng lặp
        self.cleaned_data = self.cleaned_data[~duplicated_mask].copy()
        after_rows = len(self.cleaned_data)

        print(f"Đã loại bỏ {num_duplicates} dòng trùng lặp feature.")
        print(f"Dataset sau khi loại bỏ: {after_rows:,} dòng.\n")

        self.log_action("Loại bỏ dòng trùng lặp feature",
                        f"Đã loại bỏ {num_duplicates} dòng, còn lại {after_rows} dòng")
        # Thêm sau khi loại bỏ duplicates (trước return):
        print(f"\nTẠO BIỂU ĐỒ DUPLICATE ANALYSIS:")
        self._plot_duplicate_analysis(before_rows, after_rows, num_duplicates)

        return {"removed": int(num_duplicates), "after_rows": after_rows}


    # =================================================================================================================
    # HÀM VẼ BIỂU ĐỒ PHÂN TÍCH LOẠI BỎ DÒNG TRÙNG LẶP FEATURE
    def _plot_duplicate_analysis(self, before_rows, after_rows, num_duplicates):
        """
        Vẽ biểu đồ phân tích loại bỏ duplicates
        """
        try:
            output_dir = self._create_output_directory()
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
            # SUBPLOT 1: Before/After comparison
            categories = ['Trước loại bỏ', 'Sau loại bỏ', 'Đã loại bỏ']
            values = [before_rows, after_rows, num_duplicates]
            colors = ['lightblue', 'lightgreen', 'salmon']
        
            bars = ax1.bar(categories, values, color=colors, alpha=0.8)
            ax1.set_ylabel('Số lượng dòng', fontweight='bold')
            ax1.set_title('So sánh trước/sau loại bỏ duplicates', fontweight='bold')
        
            # Thêm labels
            for bar, val in zip(bars, values):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + val*0.01,
                        f'{val:,}', ha='center', va='bottom', fontweight='bold')
        
            # SUBPLOT 2: Summary info
            ax2.axis('off')
            retention_rate = (after_rows / before_rows) * 100
        
            summary_text = f"""
LOẠI BỎ FEATURE DUPLICATES

THỐNG KÊ:
    • Dòng ban đầu: {before_rows:,}
    • Dòng sau loại bỏ: {after_rows:,}
    • Đã loại bỏ: {num_duplicates:,} dòng
    • Tỷ lệ giữ lại: {retention_rate:.1f}%

PHƯƠNG PHÁP:
    • Loại trừ cột ID khi so sánh
    • Giữ lại dòng đầu tiên (keep='first')
    • Chỉ loại bỏ trùng lặp về features
    • ID khác nhau nhưng features giống nhau

KẾT QUẢ:
    • {'Không có duplicates' if num_duplicates == 0 else f'Đã làm sạch {num_duplicates:,} duplicates'}
    • Dataset đã được tối ưu hóa
    • Giữ nguyên tính đại diện của dữ liệu
"""
        
            ax2.text(0.05, 0.95, summary_text, transform=ax2.transAxes,
                    fontsize=11, verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.8', facecolor='lightyellow', alpha=0.8))
        
            plt.suptitle(' PHÂN TÍCH LOẠI BỎ FEATURE DUPLICATES', 
                        fontsize=16, fontweight='bold', y=0.98)
            plt.tight_layout()
        
            # Lưu biểu đồ  
            filename = "duplicate_removal_analysis.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
        
            print(f"     Đã lưu biểu đồ Duplicates: {filename}")
        
        except Exception as e:
            print(f"     Lỗi khi vẽ biểu đồ duplicates: {e}")
    # =================================================================================================================
    # HÀM CHUẨN HÓA DỮ LIỆU PHÂN LOẠI (Làm việc với Marital_Status)
    def clean_categorical_data(self):
        """
        Chuẩn hóa các giá trị phân loại:
        - Marital_Status: 'Alone', 'YOLO' -> 'Single'; 'Absurd' -> 'Other'
        
        Returns:
            dict: Thông tin về các thay đổi đã thực hiện
        """
        print("="*100)
        print("Bước 5: CHUẨN HÓA DỮ LIỆU PHÂN LOẠI")

        if self.cleaned_data is None:
            print("Dữ liệu chưa được tải. Vui lòng tải dữ liệu trước khi xử lý.")
            return None
            
        changes_made = 0 # Số lượng thay đổi đã thực hiện
        categorical_changes = {} # Lưu thông tin về các thay đổi phân loại
        
        # Chuẩn hóa Marital_Status
        if 'Marital_Status' in self.cleaned_data.columns:
            print("\nChuẩn hóa cột Marital_Status:")
            
            # Đếm số lượng trước khi thay đổi
            before_counts = self.cleaned_data['Marital_Status'].value_counts()
            print("Giá trị ban đầu:")
            for val, count in before_counts.items():
                print(f"- '{val}': {count} ({count/len(self.cleaned_data)*100:.2f}%)")
            
            # Mapping các giá trị cần thay đổi
            marital_mapping = {
                'Alone': 'Single', # Chuyển 'Alone' thành 'Single'
                'YOLO': 'Single',  # Chuyển 'YOLO' thành 'Single' ( Dù không biết ý nghĩa gì )
                'Absurd': 'Other'  # Chuyển 'Absurd' thành 'Other'
            }
            
            # Lưu mapping cho reporting
            self.encoders['marital_mapping'] = marital_mapping
            
            # Lưu lại số lượng giá trị cần thay đổi
            changes = {}
            for old_val, new_val in marital_mapping.items():
                count = (self.cleaned_data['Marital_Status'] == old_val).sum()
                if count > 0:
                    changes[old_val] = {'count': count, 'new_value': new_val}
            
            # Thực hiện thay đổi
            self.cleaned_data['Marital_Status'] = self.cleaned_data['Marital_Status'].replace(marital_mapping)
            
            # Cập nhật số lượng thay đổi 
            changes_made = sum(item['count'] for item in changes.values())
            categorical_changes['Marital_Status'] = changes
            
            # Đếm số lượng sau khi thay đổi
            after_counts = self.cleaned_data['Marital_Status'].value_counts()
            print("\nGiá trị sau khi chuẩn hóa:")                                                                                                                                             
            for val, count in after_counts.items():
                print(f"- '{val}': {count} ({count/len(self.cleaned_data)*100:.2f}%)")
            
            # In thông tin về những thay đổi
            if changes:
                print("\nCác thay đổi đã thực hiện:")
                for old_val, details in changes.items():
                    print(f"- '{old_val}' → '{details['new_value']}': {details['count']} giá trị")
            
            # Ghi log
            self.log_action("Chuẩn hóa Marital_Status", 
                          f"Đã thay đổi {changes_made} giá trị")
        
        print(f"\nTổng cộng: {changes_made} giá trị đã được chuẩn hóa\n")

        # Thêm sau khi thay đổi categorical
        if changes_made > 0:
            print(f"\nTẠO BIỂU ĐỒ CATEGORICAL CLEANING:")
            self._plot_categorical_cleaning(before_counts, after_counts, changes)

        return {
            'changes_made': changes_made,
            'details': categorical_changes
        }
     
    
    # =================================================================================================================
    # HÀM VẼ BIỂU ĐỒ PHÂN TÍCH CHUẨN HÓA CATEGORICAL DATA
    def _plot_categorical_cleaning(self, before_counts, after_counts, changes):
        """
        Vẽ biểu đồ trước/sau khi chuẩn hóa categorical data
        """
        try:
            output_dir = self._create_output_directory()
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
            # SUBPLOT 1: Trước cleaning
            before_vals = list(before_counts.keys())
            before_cnts = list(before_counts.values())
        
            ax1.bar(range(len(before_vals)), before_cnts, color='lightcoral', alpha=0.7)
            ax1.set_xlabel('Giá trị Marital_Status', fontweight='bold')
            ax1.set_ylabel('Số lượng', fontweight='bold')
            ax1.set_title('TRƯỚC chuẩn hóa', fontweight='bold')
            ax1.set_xticks(range(len(before_vals)))
            ax1.set_xticklabels(before_vals, rotation=45, ha='right')
        
            # SUBPLOT 2: Sau cleaning
            after_vals = list(after_counts.keys())
            after_cnts = list(after_counts.values())
        
            ax2.bar(range(len(after_vals)), after_cnts, color='lightgreen', alpha=0.7)
            ax2.set_xlabel('Giá trị Marital_Status', fontweight='bold')
            ax2.set_ylabel('Số lượng', fontweight='bold')
            ax2.set_title('SAU chuẩn hóa', fontweight='bold')
            ax2.set_xticks(range(len(after_vals)))
            ax2.set_xticklabels(after_vals, rotation=45, ha='right')
        
            # SUBPLOT 3: Summary info
            ax3.axis('off')
        
            changes_detail = ""
            total_changed = 0
            for old_val, details in changes.items():
                count = details['count']
                new_val = details['new_value'] 
                changes_detail += f"\n   • '{old_val}' → '{new_val}': {count} giá trị"
                total_changed += count
        
            summary_text = f"""
CHUẨN HÓA DỮ LIỆU PHÂN LOẠI

MAPPING RULES:
    • 'Alone' → 'Single'
    • 'YOLO' → 'Single' 
    • 'Absurd' → 'Other'

CÁC THAY ĐỔI:{changes_detail}

TỔNG KẾT:
    • Tổng giá trị đã thay đổi: {total_changed}
    • Số categories trước: {len(before_vals)}
    • Số categories sau: {len(after_vals)}
    • Trạng thái:  Đã chuẩn hóa thành công
"""
        
            ax3.text(0.05, 0.95, summary_text, transform=ax3.transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.8', facecolor='lightcyan', alpha=0.8))
        
            plt.suptitle(' CHUẨN HÓA DỮ LIỆU PHÂN LOẠI - MARITAL STATUS', 
                        fontsize=16, fontweight='bold', y=0.98)
            plt.tight_layout()
        
            # Lưu biểu đồ
            filename = "categorical_cleaning_analysis.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
        
            print(f"     Đã lưu biểu đồ Categorical: {filename}")
        
        except Exception as e:
            print(f"     Lỗi khi vẽ biểu đồ categorical: {e}")
    # =================================================================================================================
    # HÀM CHUYỂN ĐỔI KIỂU DỮ LIỆU VÀ MÃ HÓA DỮ LIỆU
    def convert_data_types(self):
        """
        Chuyển đổi kiểu dữ liệu:
        - Dt_Customer: object → datetime
        - Đồng thời thực hiện encoding cho Education và Marital_Status
        
        Trả về dict báo cáo các chuyển đổi và encoding.
        """
        print("="*100)
        print("Bước 6: CHUYỂN ĐỔI KIỂU DỮ LIỆU VÀ MÃ HÓA ĐẶC TRƯNG PHÂN LOẠI\n")

        if self.cleaned_data is None:
            print("Dữ liệu chưa được tải. Vui lòng tải dữ liệu trước khi xử lý.")
            return None

        conversions = {}  # Lưu thông tin về các chuyển đổi đã thực hiện
        columns_added = 0 # Số cột đã thêm (dành cho encoding Martial_Status do Martial_Status là kiểu phân loại không có thứ bậc)

        # Chuyển đổi Dt_Customer thành datetime
        if 'Dt_Customer' in self.cleaned_data.columns:
            original_dtype = self.cleaned_data['Dt_Customer'].dtype
            try:
                self.cleaned_data['Dt_Customer'] = pd.to_datetime(
                    self.cleaned_data['Dt_Customer'],
                    format='%d-%m-%Y',   # Thêm dòng này để chỉ định đúng định dạng
                    errors='raise'
                )
                new_dtype = self.cleaned_data['Dt_Customer'].dtype
                conversions['Dt_Customer'] = {
                    'original_dtype': str(original_dtype),
                    'new_dtype': str(new_dtype),
                    'success': True
                }
                self.log_action("Chuyển đổi kiểu dữ liệu", f"Dt_Customer: {original_dtype} → {new_dtype}")
                print(f"- Dt_Customer: {original_dtype} → {new_dtype}")
            except Exception as e:
                conversions['Dt_Customer'] = {
                    'original_dtype': str(original_dtype),
                    'success': False,
                    'error': str(e)
                }
                print(f"- LỖI khi chuyển Dt_Customer: {e}")
        else:
            print("- Không tìm thấy cột 'Dt_Customer' để chuyển đổi")

        # Thực hiện encoding cho Education và Marital_Status
        encoding_report = {'education': None, 'marital_status': None}

        # Education (giải quyết theo kiểu ordinal - có thứ tự phân loại)
        if 'Education' in self.cleaned_data.columns:
            try:
                edu_res = self.encode_education_ordinal(
                    col='Education', # Cột gốc
                    new_col='Education_ord', # Cột mới sau khi encoding
                    mapping=['Basic', '2n Cycle', 'Graduation', 'Master', 'PhD'] # Thứ tự từ thấp đến cao
                )
                encoding_report['education'] = edu_res
                columns_added += 1
            except Exception as e:
                encoding_report['education'] = {'status': 'error', 'error': str(e)}
                self.log_action("Error encoding Education", str(e))

        # Marital_Status (giải quyết theo kiểu one-hotcode - không có thứ tự phân loại)
        if 'Marital_Status' in self.cleaned_data.columns:
            try:
                mar_res = self.encode_marital_status_onehot(
                    col='Marital_Status', # Cột gốc
                    prefix='Marital',     # Tiền tố cho cột mới
                    group_threshold=0.01, # Ngưỡng nhóm các giá trị hiếm
                    drop_original=False   
                )
                encoding_report['marital_status'] = mar_res # Lưu kết quả encoding vào báo cáo
                
                # Tính số cột dummy được thêm vào
                if mar_res and 'dummy_columns' in mar_res:
                    columns_added += len(mar_res['dummy_columns'])
                    
                    # Chuyển các cột dummy sang kiểu uint8 để tiết kiệm bộ nhớ
                    dummy_cols = mar_res['dummy_columns']
                    self.cleaned_data[dummy_cols] = self.cleaned_data[dummy_cols].astype('uint8')
                    print(f"\nĐã chuyển {len(dummy_cols)} cột dummy sang kiểu uint8")
                
                # Thêm cột Marital_Status_Grouped
                columns_added += 1
                    
            except Exception as e:
                encoding_report['marital_status'] = {'status': 'error', 'error': str(e)}
                self.log_action("Error encoding Marital_Status", str(e))

        conversions['encoding'] = encoding_report
        conversions['columns_added'] = columns_added

        print(f"\nHoàn tất chuyển đổi kiểu dữ liệu và mã hóa phân loại")
        print(f"- Đã thêm {columns_added} cột mới (encoding)\n")
        
        # Ensure binary dtypes for all dummy/binary columns
        try:
            # prefer explicitly provided dummy list if available
            if mar_res and 'dummy_columns' in mar_res:
                self.ensure_binary_dtypes(mar_res['dummy_columns'])
            else:
                self.ensure_binary_dtypes()  # auto-detect
        except Exception as e:
            self.log_action("Warning ensure_binary_dtypes failed", str(e))

        # Export conversion report so mappings are persisted for production reuse
        try:
            self.export_conversion_report(conversion_result=conversions)
        except Exception as e:
            self.log_action("Warning export_conversion_report failed", str(e))

        return conversions
    # =================================================================================================================
    # HÀM TẠO BIỂU ĐỒ TRỰC QUAN HÓA QUÁ TRÌNH LOẠI BỎ OUTLIERS
    def _create_outlier_removal_visualizations(self, data_before, data_after, removed_outliers, income_outliers, age_outliers):
        """
        Tạo biểu đồ trực quan hóa quá trình loại bỏ outliers cho các biến số chính
        """
        try:
            output_dir = self._create_output_directory()
        
            # Danh sách các biến số chính cần trực quan hóa
            numerical_variables = [
                ('Income', 'THU NHẬP'), # 1 outlier
                ('Year_Birth', 'NĂM SINH'), # 2 outliers
                ('MntMeatProducts', 'CHI TIÊU THỊT'), # 29 outliers
                ('MntFruits', 'CHI TIÊU TRÁI CÂY'), # 96 outliers
                ('MntSweetProducts','CHI TIÊU ĐỒ NGỌT'), # 105 outliers
                ('MntGoldProds', 'CHI TIÊU VÀNG BẠC'), # 49 outliers
                ('MntFishProducts', 'CHI TIÊU HẢI SẢN'), # 72 outliers
                ('NumWebPurchases', 'MUA HÀNG ONLINE'), # 4 outliers 
                ('NumCatalogPurchases', 'MUA HÀNG QUA CATALOG'), # 4 outliers
                ('NumDealsPurchases', 'MUA HÀNG KHUYẾN MÃI'), # 24 outliers
                ('NumWebVisitsMonth', 'SỐ LẦN TRUY CẬP WEB TRONG THÁNG'), # 3 outliers
                
            ]
        
            for col, col_vn in numerical_variables:
                if col in data_before.columns:
                    self._plot_outlier_removal_comparison(
                        col, col_vn, data_before, data_after, 
                        removed_outliers, income_outliers, age_outliers, 
                        output_dir
                    )
        
            print("Đã tạo xong các biểu đồ trực quan hóa")
                
        except Exception as e:
            print(f"Lỗi khi tạo biểu đồ: {e}")
            import traceback
            print(f"Chi tiết lỗi: {traceback.format_exc()}")
    # =================================================================================================================
    # HÀM VẼ BIỂU ĐỒ SO SÁNH TRƯỚC VÀ SAU KHI LOẠI BỎ OUTLIERS
    def _plot_outlier_removal_comparison(self, col, col_vn, data_before, data_after, 
                                       removed_outliers, income_outliers, age_outliers, output_dir):
        """
        Vẽ biểu đồ so sánh trước và sau khi loại bỏ outliers cho một biến cụ thể
        """
        try:
            # =====================================
            # TẠO FIGURE VỚI 3 SUBPLOT
            # =====================================
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 7))
        
            # Lấy dữ liệu cho biến hiện tại
            before_data = data_before[col].dropna()
            after_data = data_after[col].dropna()
        
            # =====================================
            # SUBPLOT 1: HISTOGRAM TRƯỚC KHI LOẠI BỎ
            # =====================================
            ax1.hist(before_data, bins=50, alpha=0.7, color='lightcoral', 
                    edgecolor='black', linewidth=0.5, label=f'Trước loại bỏ ({len(before_data):,})')
        
            # Highlight outliers nếu có
            if col == 'Income' and len(income_outliers) > 0:
                outlier_values = income_outliers[col].values
                ax1.hist(outlier_values, bins=20, alpha=0.9, color='red', 
                        edgecolor='darkred', linewidth=1.2, label=f'Outliers bị loại ({len(outlier_values)})')
            elif col == 'Year_Birth' and len(age_outliers) > 0:
                outlier_values = age_outliers[col].values
                ax1.hist(outlier_values, bins=10, alpha=0.9, color='red', 
                        edgecolor='darkred', linewidth=1.2, label=f'Outliers bị loại ({len(outlier_values)})')
        
            ax1.set_xlabel(f'{col}', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Tần số', fontsize=12, fontweight='bold')
            ax1.set_title(f'{col_vn}\nTrước khi loại bỏ outliers', fontsize=13, fontweight='bold')
            ax1.legend(fontsize=10)
            ax1.grid(True, alpha=0.3, linestyle='--')
        
            # =====================================
            # SUBPLOT 2: HISTOGRAM SAU KHI LOẠI BỎ
            # =====================================
            ax2.hist(after_data, bins=50, alpha=0.7, color='lightgreen', 
                    edgecolor='black', linewidth=0.5, label=f'Sau loại bỏ ({len(after_data):,})')
        
            ax2.set_xlabel(f'{col}', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Tần số', fontsize=12, fontweight='bold')
            ax2.set_title(f'{col_vn}\nSau khi loại bỏ outliers', fontsize=13, fontweight='bold')
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3, linestyle='--')
        
            # =====================================
            # SUBPLOT 3: BẢNG THÔNG TIN CHI TIẾT
            # =====================================
            ax3.axis('off')
        
            # Tạo nội dung thông tin
            removed_count = len(before_data) - len(after_data)
            removal_reason = self._get_removal_reason(col, income_outliers, age_outliers)
        
            summary_text = f"""
    QUÁ TRÌNH LOẠI BỎ OUTLIERS - {col_vn}

    THỐNG KÊ TRƯỚC/SAU:
       • Trước loại bỏ    : {len(before_data):,} giá trị
       • Sau loại bỏ     : {len(after_data):,} giá trị  
       • Đã loại bỏ      : {removed_count} giá trị
       • Tỷ lệ còn lại   : {len(after_data)/len(before_data)*100:.1f}%

    THỐNG KÊ MÔ TẢ:
       Trước loại bỏ:
       • Min/Max: {before_data.min():,.0f} / {before_data.max():,.0f}
       • Mean   : {before_data.mean():,.1f}
       • Median : {before_data.median():,.1f}
   
       Sau loại bỏ:
       • Min/Max: {after_data.min():,.0f} / {after_data.max():,.0f}
       • Mean   : {after_data.mean():,.1f}
       • Median : {after_data.median():,.1f}

    LÍ DO LOẠI BỎ:
    {removal_reason}

    KẾT QUẢ:
       Đã loại bỏ thành công các outliers
       cực đoan, giữ lại phân bố chính.
    """
        
            # Hiển thị text summary
            ax3.text(0.05, 0.95, summary_text, transform=ax3.transAxes, 
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.8', facecolor='lightblue', alpha=0.8))
        
            # =====================================
            # THIẾT LẬP FIGURE CHÍNH
            # =====================================
            # Escape underscore in column name for mathtext
            col_math = col.replace('_', r'\_')
            title = f'LOẠI BỎ OUTLIERS - {col_vn}: ' + r'$\mathbf{' + f'{col_math}' + '}$'
            plt.suptitle(title, fontsize=16, fontweight='regular', y=0.98)
            plt.tight_layout()
        
            # =====================================
            # LƯU BIỂU ĐỒ (GHI ĐÈ)
            # =====================================
            filename = f"outlier_removal_{col}.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            plt.close()
        
            print(f"    Đã lưu biểu đồ: {filename}")
        
        except Exception as e:
            print(f"    Lỗi khi vẽ biểu đồ cho {col}: {e}")

    def _get_removal_reason(self, col, income_outliers, age_outliers):
        """
        Trả về lý do loại bỏ outliers cho từng biến
        """
        if col == 'Income':
            if len(income_outliers) > 0:
                max_income = income_outliers['Income'].max()
                return f"""   • Loại bỏ {len(income_outliers)} giá trị Income > 500,000
       • Giá trị cao nhất: {max_income:,.0f}
       • Lý do: Tách biệt hoàn toàn khỏi phân phối chính
       • Có thể là lỗi nhập liệu hoặc outlier thực sự"""
            else:
                return "   • Không có outliers Income cần loại bỏ"
    
        elif col == 'Year_Birth':
            if len(age_outliers) > 0:
                min_year = age_outliers['Year_Birth'].min()
                max_age = 2024 - min_year
                return f"""   • Loại bỏ {len(age_outliers)} giá trị Year_Birth < 1900
       • Năm sinh nhỏ nhất: {min_year}
       • Tuổi tương ứng: ~{max_age} tuổi
       • Lý do: Khách hàng quá lớn tuổi"""
            else:
                return "   • Không có outliers Year_Birth cần loại bỏ"
    
        else:
            # Cho các biến khác
            return f"""   • Biến {col} không bị loại bỏ outliers
       • Lý do: Outliers vẫn trong ngưỡng chấp nhận được
       • Cách các giá trị bình thường không quá xa
       • Ngưỡng IQR hiện tại có thể chưa đủ rộng"""

        

    # =================================================================================================================
    # HÀM LOẠI BỎ OUTLIERS CỤ THỂ
    def remove_specific_outliers(self):
        """
        Loại bỏ các outliers cụ thể:
        - 1 dòng có Income bất thường (>= 500000) 
        - 2 dòng có Year_Birth bất thường (< 1900 - quá già)
    
        Sau khi loại bỏ sẽ tạo biểu đồ trực quan hóa cho các biến số chính
    
        Returns:
            dict: Thông tin về các dòng đã loại bỏ
        """
        print("="*100)
        print("Bước 4: LOẠI BỎ OUTLIERS CỤ THỂ")
        print()

        if self.cleaned_data is None:
            print("Dữ liệu chưa được tải. Vui lòng tải dữ liệu trước khi xử lý.")
            return None

        original_rows = len(self.cleaned_data)
        removed_records = []

        # =====================================
        # BƯỚC 1: XÁC ĐỊNH CÁC OUTLIERS
        # =====================================

        print("Xác định các outliers cần loại bỏ:")

        # Tìm dòng có Income bất thường (>= 500000)
        income_outliers = self.cleaned_data[self.cleaned_data['Income'] >= 500000]
        print(f"- Income >= 500000: {len(income_outliers)} dòng")

        if len(income_outliers) > 0:
            print("  Chi tiết Income outliers:")
            for idx, row in income_outliers.iterrows():
                id_val = row.get('ID', idx)
                income_val = row['Income']
                print(f"    ID {id_val}: Income = {income_val:,.0f}")
                removed_records.append({
                    'ID': id_val,
                    'reason': 'Income_outlier',
                    'Income': income_val,
                    'Year_Birth': row.get('Year_Birth', 'N/A')
                })

        # Tìm dòng có Year_Birth bất thường (< 1900 - quá già)  
        age_outliers = self.cleaned_data[self.cleaned_data['Year_Birth'] < 1900]
        print(f"- Year_Birth < 1900: {len(age_outliers)} dòng")

        if len(age_outliers) > 0:
            print("  Chi tiết Year_Birth outliers:")
            for idx, row in age_outliers.iterrows():
                id_val = row.get('ID', idx)
                year_birth = row['Year_Birth']
                age = 2024 - year_birth
                print(f"    ID {id_val}: Year_Birth = {year_birth} (tuổi ~{age})")
                removed_records.append({
                    'ID': id_val,
                    'reason': 'Year_Birth_outlier',
                    'Income': row.get('Income', 'N/A'),
                    'Year_Birth': year_birth
                })

        # =====================================
        # BƯỚC 2: LOẠI BỎ CÁC OUTLIERS
        # =====================================

        total_outliers = len(income_outliers) + len(age_outliers)

        if total_outliers == 0:
            print("\nKhông có outliers nào cần loại bỏ.")
            self.log_action("Loại bỏ outliers", "Không có outliers")
            return {"removed_count": 0, "outliers": []}

        print(f"\nTiến hành loại bỏ {total_outliers} dòng outliers:")

        # Lưu dữ liệu trước khi loại bỏ để so sánh
        data_before_removal = self.cleaned_data.copy()

        # Tạo mask để loại bỏ các outliers
        income_mask = self.cleaned_data['Income'] >= 500000
        age_mask = self.cleaned_data['Year_Birth'] < 1900
        outlier_mask = income_mask | age_mask

        # Lưu thông tin các dòng bị loại bỏ vào bộ nhớ (không xuất file)
        self.removed_outliers = self.cleaned_data[outlier_mask].copy()

        # Loại bỏ các outliers
        self.cleaned_data = self.cleaned_data[~outlier_mask].copy()

        after_rows = len(self.cleaned_data)
        removed_count = original_rows - after_rows

        print(f"- Đã loại bỏ {removed_count} dòng outliers")
        print(f"- Dataset còn lại: {after_rows:,} dòng")
        print(f"- Outliers đã lưu trong bộ nhớ: {len(self.removed_outliers)} dòng")

        # =====================================
        # BƯỚC 3: TẠO BIỂU ĐỒ TRỰC QUAN HÓA
        # =====================================
        print(f"\nTẠO BIỂU ĐỒ TRỰC QUAN HÓA:")
        print("-" * 40)

        # Tạo biểu đồ cho các biến số chính
        self._create_outlier_removal_visualizations(
            data_before=data_before_removal,
            data_after=self.cleaned_data,
            removed_outliers=self.removed_outliers,
            income_outliers=income_outliers,
            age_outliers=age_outliers
        )

        # =====================================
        # BƯỚC 4: THỐNG KÊ KẾT QUẢ
        # =====================================

        print(f"\nTHỐNG KÊ LOẠI BỎ OUTLIERS:")
        print("=" * 60)
        print(f"Dataset ban đầu            : {original_rows:,} dòng")
        print(f"Income outliers loại bỏ    : {len(income_outliers)} dòng")
        print(f"Year_Birth outliers loại bỏ: {len(age_outliers)} dòng")
        print(f"Tổng dòng loại bỏ          : {removed_count} dòng")
        print(f"Dataset sau loại bỏ        : {after_rows:,} dòng")
        print(f"Tỷ lệ dữ liệu còn lại      : {after_rows/original_rows*100:.2f}%")
        print("=" * 60)

        # =====================================
        # BƯỚC 5: GHI LOG VÀ TRẢ VỀ KẾT QUẢ
        # =====================================

        self.log_action("Loại bỏ outliers cụ thể", 
                       f"Đã loại bỏ {removed_count} dòng (Income: {len(income_outliers)}, Year_Birth: {len(age_outliers)})")

        result = {
            'removed_count': removed_count,
            'original_rows': original_rows,
            'after_rows': after_rows,
            'income_outliers_removed': len(income_outliers),
            'age_outliers_removed': len(age_outliers),
            'outliers': removed_records
        }

        print(f"\nHoàn thành loại bỏ outliers\n")
        return result
    # =================================================================================================================
    # HÀM XUẤT CONVERSION REPORT RA FILE JSON
    def export_conversion_report(self, output_path=None, conversion_result=None):
        """
        Xuất báo cáo chuyển đổi ra file JSON
        
        Args:
            output_path (str, optional): Đường dẫn file xuất. Nếu None, tạo đường dẫn mặc định.
            conversion_result (dict, optional): Kết quả chuyển đổi. Nếu None, lấy từ self.encoders.
            
        Returns:
            str: Đường dẫn đến file đã xuất
        """
        # Tạo báo cáo để xuất ra file
        report = {}
        
        # Lấy thông tin về encoding từ self.encoders
        if hasattr(self, 'encoders'):
            report.update(self.encoders)
              
        # Lấy thông tin từ conversion_result nếu có
        if conversion_result and isinstance(conversion_result, dict):
            # Thêm thông tin về encoding
            if 'encoding' in conversion_result:
                encoding_info = conversion_result['encoding']
                
                # Education mapping
                if 'education' in encoding_info and encoding_info['education']:
                    edu_info = encoding_info['education']
                    if 'mapping' in edu_info:
                        report['education_mapping'] = {
                            'order': edu_info['mapping'],
                            'column': edu_info.get('encoded_col', 'Education_ord')
                        }
                
                # Marital_Status one-hot
                if 'marital_status' in encoding_info and encoding_info['marital_status']:
                    mar_info = encoding_info['marital_status']
                    if 'dummy_columns' in mar_info:
                        report['marital_status_encoding'] = {
                            'dummy_columns': mar_info['dummy_columns'],
                            'rare_categories': mar_info.get('rare_categories', []),
                            'grouped_column': mar_info.get('original_col', 'Marital_Status') + '_Grouped'
                        }
        
        # Kiểm tra xem có dữ liệu để xuất không
        if not report:
            print("Không có dữ liệu để xuất báo cáo chuyển đổi")
            return None
            
        # Tạo đường dẫn xuất
        if output_path is None:
            # Nếu không cung cấp đường dẫn, tạo đường dẫn mặc định
            if self.dataset_path:
                dir_path = os.path.dirname(self.dataset_path)
                file_name = os.path.basename(self.dataset_path)
                base_name = os.path.splitext(file_name)[0]
            else:
                dir_path = "."
                base_name = "dataset"
                
            output_path = os.path.join(dir_path, f"{base_name}_conversion_report.json")
            
        try:
            # Xuất ra file JSON
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, default=str)  # default=str để xử lý các kiểu dữ liệu không serialize được
            
            print(f"Đã xuất báo cáo chuyển đổi ra file: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"Lỗi khi xuất báo cáo chuyển đổi: {e}")
            return None
        
    # =================================================================================================================
    # HÀM KIỂM TRA VÀ XÁC THỰC DỮ LIỆU
    def validate_cleaned_data(self):
        """
        Xác thực dữ liệu sau làm sạch.
        - Missing / categorical / data types -> fail if issues.
        - Year_Birth / Income anomalies recorded as warnings (do not fail).
        """
        print("="*100)
        print("Bước 8: KIỂM TRA VÀ XÁC THỰC DỮ LIỆU")

        if self.cleaned_data is None:
            print("Dữ liệu chưa được tải. Vui lòng tải dữ liệu trước khi kiểm tra.")
            return None

        validation = {
            'missing_values': {'passed': False},
            'categorical_data': {'passed': False},
            'data_types': {'passed': False},
            'special_values': {'passed': True, 'warnings': {}}
        }

        # Missing values
        print("\nKiểm tra missing values:")
        missing_count = int(self.cleaned_data.isnull().sum().sum())
        if missing_count > 0:
            missing_cols = self.cleaned_data.columns[self.cleaned_data.isnull().any()].tolist()
            validation['missing_values'] = {
                'passed': False,
                'count': missing_count,
                'columns': missing_cols
            }
            print(f"- Found {missing_count} missing values in columns: {missing_cols}")
        else:
            validation['missing_values'] = {'passed': True, 'count': 0}
            print("- No missing values")

        # Categorical consistency
        print("\nKiểm tra dữ liệu phân loại:")
        if 'Marital_Status' in self.cleaned_data.columns:
            irregular_mask = self.cleaned_data['Marital_Status'].isin(['Alone', 'YOLO', 'Absurd'])
            irregular_count = int(irregular_mask.sum())
            if irregular_count > 0:
                validation['categorical_data'] = {
                    'passed': False,
                    'issues': {
                        'Marital_Status': {
                            'count': irregular_count,
                            'values': self.cleaned_data.loc[irregular_mask, 'Marital_Status'].unique().tolist()
                        }
                    }
                }
                print(f"- Marital_Status: {irregular_count} irregular values detected")
            else:
                validation['categorical_data'] = {'passed': True}
                print("- Marital_Status: OK")
        else:
            validation['categorical_data'] = {'passed': True, 'note': 'No Marital_Status column'}
            print("- Marital_Status: not present")

        # Data types
        print("\nKiểm tra kiểu dữ liệu:")
        data_type_issues = {}
        if 'Dt_Customer' in self.cleaned_data.columns:
            if pd.api.types.is_datetime64_dtype(self.cleaned_data['Dt_Customer']):
                print("- Dt_Customer: datetime64 OK")
            else:
                data_type_issues['Dt_Customer'] = {
                    'expected': 'datetime64',
                    'actual': str(self.cleaned_data['Dt_Customer'].dtype)
                }
                print(f"- Dt_Customer: expected datetime64 but got {self.cleaned_data['Dt_Customer'].dtype}")

        if data_type_issues:
            validation['data_types'] = {'passed': False, 'issues': data_type_issues}
        else:
            validation['data_types'] = {'passed': True}

        # Special-value checks
        print("\nKiểm tra giá trị đặc biệt:")
        special_warnings = {}

        if 'Year_Birth' in self.cleaned_data.columns:
            too_old = int((self.cleaned_data['Year_Birth'] < 1900).sum())
            too_young = int((self.cleaned_data['Year_Birth'] > 2006).sum())
            if too_old > 0 or too_young > 0:
                special_warnings['Year_Birth'] = {
                    'too_old': too_old,
                    'too_young': too_young,
                    'note': 'Detected but not modified'
                }
                print(f"- Warning Year_Birth: {too_old} <1900, {too_young} >2006 (kept)")

        if 'Income' in self.cleaned_data.columns:
            suspicious = int((self.cleaned_data['Income'] >= 500000).sum())
            if suspicious > 0:
                special_warnings['Income'] = {
                    'suspicious_count': suspicious,
                    'note': 'Detected but not modified'
                }
                print(f"- Warning Income: {suspicious} values >=500000 (kept)")

        validation['special_values'] = {'passed': True, 'warnings': special_warnings}

        # Summary
        all_passed = validation['missing_values']['passed'] and validation['categorical_data']['passed'] and validation['data_types']['passed']
        print("\nKẾT QUẢ KIỂM TRA:")
        if all_passed:
            print("Dữ liệu đã sạch\n")
        else:
            failed = [k for k, v in validation.items() if k in ('missing_values','categorical_data','data_types') and not v['passed']]
            print(f"CÓ VẤN ĐỀ - {len(failed)} kiểm tra không thành công: {', '.join(failed)}")

        self.log_action("Xác thực dữ liệu", f"{'Tất cả kiểm tra đều thành công' if all_passed else 'Có kiểm tra không thành công'}")
        return {'all_passed': all_passed, 'details': validation}
        
    # =================================================================================================================
    # HÀM XUẤT DỮ LIỆU ĐÃ LÀM SẠCH
    def export_cleaned_data(self, output_path=None, include_timestamp=False):
        """
        Xuất dữ liệu đã làm sạch ra file CSV.

        Behavior:
          - Nếu output_path không được cung cấp: mặc định sẽ ghi đè lên
            "<base_name>_cleaned.csv" trong cùng thư mục dataset (atomic replace).
          - Nếu include_timestamp=True: hàm sẽ tạo thêm một bản có timestamp
            "<base_name>_cleaned_<YYYYmmdd_HHMMSS>.csv" (vẫn ghi đè bản latest).
          - Trả về đường dẫn của file 'latest' (bản bị ghi đè).
        """
        print("=" * 100)
        print("Bước 9: XUẤT DỮ LIỆU ĐÃ LÀM SẠCH")

        if self.cleaned_data is None:
            print("Không có dữ liệu để xuất")
            return None

        # Xác định đường dẫn mặc định nếu không cung cấp
        if output_path is None:
            if self.dataset_path:
                dir_path = os.path.dirname(self.dataset_path)
                file_name = os.path.basename(self.dataset_path)
                base_name = os.path.splitext(file_name)[0]
            else:
                dir_path = "."
                base_name = "dataset"
            latest_path = os.path.join(dir_path, f"{base_name}_cleaned.csv")
        else:
            # Nếu caller cung cấp output_path, dùng nó làm 'latest' (ghi đè)
            latest_path = output_path
            dir_path = os.path.dirname(latest_path) or "."

        os.makedirs(dir_path, exist_ok=True)

        try:
            # Ghi atomic: viết vào temp file trước, sau đó replace
            with tempfile.NamedTemporaryFile(delete=False, dir=dir_path, suffix=".csv") as tmp:
                tmp_path = tmp.name
            # Write dataframe to temporary path
            self.cleaned_data.to_csv(tmp_path, index=False)

            # Nếu muốn có bản timestamped lưu trữ, tạo nó
            if include_timestamp:
                ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                timestamped_path = os.path.join(dir_path, f"{base_name}_cleaned_{ts}.csv")
                # Move temp -> timestamped (atomic replace)
                os.replace(tmp_path, timestamped_path)
                # Then copy timestamped to latest_path (overwrite)
                # Use os.replace to ensure atomic overwrite of latest
                os.replace(timestamped_path, latest_path)
                saved_path = latest_path
                print(f"\nĐã lưu bản timestamped : {timestamped_path}")
                print(f"Đã cập ghi đè            : {latest_path}")
            else:
                # Move temp -> latest (overwrite)
                os.replace(tmp_path, latest_path)
                saved_path = latest_path
                print(f"\nĐã xuất dữ liệu thành công (ghi đè):")
                print(f"- File: {latest_path}")

            print(f"- Shape: {self.cleaned_data.shape}\n")
            self.log_action("Xuất dữ liệu", f"Đã xuất ra file {saved_path}")
            return saved_path

        except Exception as e:
            print(f"\nLỗi khi xuất dữ liệu: {e}")
            # cleanup temp if exists
            try:
                if 'tmp_path' in locals() and os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            return None
      
    # =================================================================================================================
    # HÀM LOẠI BỎ VÀ LƯU TRỮ CÁC ĐẶC TRƯNG KHÔNG CẦN THIẾT
    def remove_non_demographic_features(self):
        """
        Loại bỏ các cột không phù hợp cho phân tích nhân khẩu học và lưu trữ chúng trong bộ nhớ:
        - Binary features: AcceptedCmp1-5, Complain, Response (cho campaign analysis)
        - Identifiers/Constants: ID, Z_CostContact, Z_Revenue
        - Redundant categorical: Education, Marital_Status, Marital_Status_Grouped (đã có encoded versions)

        Returns:
            dict: Thông tin về các cột đã loại bỏ
        """
        print("="*100)
        print("Bước 7: LOẠI BỎ CÁC ĐẶC TRƯNG KHÔNG CẦN THIẾT")

        if self.cleaned_data is None:
            print("Dữ liệu chưa được tải. Vui lòng tải dữ liệu trước khi xử lý.")
            return None

        # =====================================
        # BƯỚC 1: XÁC ĐỊNH CÁC CỘT CẦN LOẠI BỎ
        # =====================================

        # Binary campaign features - để dành cho campaign analysis
        binary_features = [
            'AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5',
            'Complain', 'Response'
        ]

        # Identifier và constant features - hằng số và biến định danh, không có giá trị phân tích
        identifier_constant_features = [
            'ID', 'Z_CostContact', 'Z_Revenue'
        ]
    
        # Redundant categorical features - đã có encoded versions (Education_ord, Marital_* dummies)
        redundant_categorical_features = [
            'Education',              # Đã có Education_ord
            'Marital_Status',         # Đã có Marital_* dummies
            'Marital_Status_Grouped'  # Đã có Marital_* dummies
        ]

        # Tổng hợp tất cả cột cần loại bỏ
        columns_to_remove = binary_features + identifier_constant_features + redundant_categorical_features

        # Lọc chỉ các cột thực tế tồn tại trong dataset
        existing_columns_to_remove = [col for col in columns_to_remove if col in self.cleaned_data.columns]

        if not existing_columns_to_remove:
            print("Không có cột nào cần loại bỏ.")
            return {"status": "no_columns_removed", "removed_columns": []}

        print(f"\nCác cột sẽ được loại bỏ khỏi dataset chính:")
        print("-" * 60)

        removed_info = {}

        # Phân loại và hiển thị thông tin các cột sẽ loại bỏ
        binary_to_remove = [col for col in binary_features if col in existing_columns_to_remove]
        identifier_to_remove = [col for col in identifier_constant_features if col in existing_columns_to_remove]
        redundant_to_remove = [col for col in redundant_categorical_features if col in existing_columns_to_remove]

        if binary_to_remove:
            print("BINARY AND CAMPAIGN FEATURES (dành cho campaign analysis):")
            for col in binary_to_remove:
                unique_vals = self.cleaned_data[col].unique()
                unique_count = len(unique_vals)
                print(f"  - {col:<20}: {unique_count} unique values {sorted(unique_vals)}")
                removed_info[col] = {
                    'type': 'binary_campaign',
                    'unique_values': sorted(unique_vals.tolist()),
                    'purpose': 'campaign_analysis'
                }

        if identifier_to_remove:
            print("\nIDENTIFIER AND CONSTANT FEATURES (hằng số và biến định danh, không có giá trị phân tích):")
            for col in identifier_to_remove:
                unique_vals = self.cleaned_data[col].unique()
                unique_count = len(unique_vals)
                print(f"  - {col:<20}: {unique_count} unique values")
                removed_info[col] = {
                    'type': 'identifier_constant',
                    'unique_values': unique_vals.tolist() if unique_count <= 5 else f"{unique_count} unique values",
                    'purpose': 'identifier_or_constant'
                }
    
        if redundant_to_remove:
            print("\nREDUNDANT CATEGORICAL FEATURES (đã có encoded versions):")
            for col in redundant_to_remove:
                unique_vals = self.cleaned_data[col].unique()
                unique_count = len(unique_vals)
                print(f"  - {col:<20}: {unique_count} unique values")
                removed_info[col] = {
                    'type': 'redundant_categorical',
                    'unique_values': unique_vals.tolist() if unique_count <= 10 else f"{unique_count} unique values",
                    'purpose': 'already_encoded'
                }

        print("-" * 60)

        # =========================================
        # BƯỚC 2: TẠO ARCHIVED DATASET TRONG BỘ NHỚ
        # =========================================
        print(f"\nTẠO ARCHIVED DATASET")

        # Tạo dataset chứa các cột bị loại bỏ (bao gồm cả ID để mapping)
        archived_columns = existing_columns_to_remove.copy()
        if 'ID' not in archived_columns and 'ID' in self.cleaned_data.columns:
            archived_columns.insert(0, 'ID')  # Thêm ID vào đầu để mapping

        self.archived_features = self.cleaned_data[archived_columns].copy()

        print(f"- Archived dataset: {self.archived_features.shape}")
        print(f"  Columns: {list(self.archived_features.columns)}")

        # ==========================================
        # BƯỚC 3: LOẠI BỎ CÁC CỘT KHỎI DATASET CHÍNH
        # ==========================================
        print(f"\nLOẠI BỎ CÁC CỘT KHỎI DATASET CHÍNH")

        original_shape = self.cleaned_data.shape

        # Loại bỏ các cột (giữ lại ID nếu nó không trong danh sách loại bỏ ban đầu)
        columns_to_drop = [col for col in existing_columns_to_remove if col != 'ID' or 'ID' in identifier_constant_features]

        # Lưu danh sách các cột bị drop
        self.encoders['archived_columns'] = columns_to_drop

        # Thực hiện drop
        self.cleaned_data = self.cleaned_data.drop(columns=columns_to_drop)

        # KIỂM TRA VÀ LOẠI BỎ FULL-ROW DUPLICATES SAU KHI ĐÃ DROP ID
        print(f"\nKIỂM TRA VÀ LOẠI BỎ FULL-ROW DUPLICATES SAU KHI DROP ID:")
        before_full_dup = len(self.cleaned_data)
        full_dup_mask = self.cleaned_data.duplicated(keep='first')
        num_full_dups = full_dup_mask.sum()

        if num_full_dups > 0:
            self.cleaned_data = self.cleaned_data[~full_dup_mask].copy()
            after_full_dup = len(self.cleaned_data)
            print(f"  - Đã loại bỏ {num_full_dups} dòng trùng lặp hoàn toàn")
            print(f"  - Dataset còn lại: {after_full_dup} dòng")
            self.log_action("Loại bỏ full-row duplicates sau drop ID", 
                           f"Đã loại bỏ {num_full_dups} dòng")
        else:
            print(f"  - Không có dòng trùng lặp hoàn toàn")

        new_shape = self.cleaned_data.shape

        print(f"Dataset chính sau khi loại bỏ:")
        print(f"  - Shape      : {original_shape} → {new_shape}")
        print(f"  - Đã loại bỏ : {len(columns_to_drop)} cột")
        print(f"  - Còn lại    : {list(self.cleaned_data.columns)}")

        # =====================================
        # BƯỚC 4: THỐNG KÊ KẾT QUẢ
        # =====================================
        print(f"\nTHỐNG KÊ KẾT QUẢ:")
        print("=" * 70)
        print(f"Dataset gốc                    : {original_shape[0]:,} dòng × {original_shape[1]} cột")
        print(f"Dataset sau loại bỏ            : {new_shape[0]:,} dòng × {new_shape[1]} cột")
        print(f"Archived dataset (bộ nhớ)      : {self.archived_features.shape[0]:,} dòng × {self.archived_features.shape[1]} cột")
        print(f"Số cột đã loại bỏ              : {len(columns_to_drop)}")
        print("=" * 70)

        print(f"\nCHI TIẾT CÁC CỘT ĐƯỢC GIỮ LẠI:")
        remaining_cols = list(self.cleaned_data.columns)
        for i, col in enumerate(remaining_cols, 1):
            dtype = str(self.cleaned_data[col].dtype)
            print(f"{i:2d}. {col:<20} ({dtype})")

        # =====================================
        # BƯỚC 5: GHI LOG VÀ TRẢ VỀ KẾT QUẢ
        # =====================================
        self.log_action("Loại bỏ features không phục vụ nhân khẩu học + redundant categorical", 
                       f"Đã loại bỏ {len(columns_to_drop)} cột và lưu vào archived dataset")

        result = {
            'status': 'success',
            'original_shape': original_shape,
            'new_shape': new_shape,
            'removed_columns': columns_to_drop,
            'remaining_columns': remaining_cols,
            'binary_features_removed': binary_to_remove,
            'identifier_features_removed': identifier_to_remove,
            'redundant_categorical_removed': redundant_to_remove,
            'removed_info': removed_info
        }

        print(f"\nHoàn thành loại bỏ các đặc trưng không cần thiết\n")        
        return result

    # =================================================================================================================
    # HÀM LẤY ARCHIVED FEATURES DATASET
    def get_archived_features(self):
        """
        Trả về dataset chứa các features đã được archived
        
        Returns:
            pd.DataFrame: Dataset chứa các features đã loại bỏ
        """
        if not hasattr(self, 'archived_features') or self.archived_features is None:
            print("Chưa có archived features. Hãy chạy remove_non_demographic_features() trước.")
            return None
            
        print(f"Archived features dataset: {self.archived_features.shape}")
        print(f"Columns: {list(self.archived_features.columns)}")
        
        return self.archived_features

    # =================================================================================================================
    # HÀM HIỂN THỊ THỐNG KÊ ARCHIVED FEATURES
    def show_archived_features_info(self):
        """
        Hiển thị thông tin chi tiết về archived features dataset
        """
        if not hasattr(self, 'archived_features') or self.archived_features is None:
            print("Chưa có archived features. Hãy chạy remove_non_demographic_features() trước.")
            return None
            
        print("="*100)
        print("THÔNG TIN CHI TIẾT ARCHIVED FEATURES DATASET")
        print("="*100)
        
        print(f"Shape: {self.archived_features.shape}")
        print(f"Columns: {list(self.archived_features.columns)}")
        
        print(f"\nThống kê mô tả:")
        print(self.archived_features.describe(include='all'))
        
        print(f"\nMẫu dữ liệu đầu tiên:")
        print(self.archived_features.head())
        
        return self.archived_features

    # =================================================================================================================
    # HÀM ENCODE EDUCATION (ORDERED CATEGORICAL → ORDINAL)
    def encode_education_ordinal(self, col='Education', new_col='Education_Encoded', mapping=None):
        """
        Chuyển cột Education (có thứ bậc) thành ordinal numeric codes.
        """
        print("="*100)
        print("MÃ HÓA EDUCATION (ORDINAL ENCODING)")
    
        if self.cleaned_data is None:
            print("Dữ liệu chưa được tải.")
            self.log_action("Lỗi encode Education", "Dữ liệu chưa được tải")
            return None
        
        if col not in self.cleaned_data.columns:
            print(f"Không tìm thấy cột '{col}'")
            self.log_action("Lỗi encode Education", f"Không tìm thấy cột '{col}'")
            return None
    
        # Mapping mặc định (thứ tự từ thấp → cao)
        if mapping is None:
            mapping = ['Basic', '2n Cycle', 'Graduation', 'Master', 'PhD']
    
        # Lưu mapping để tái sử dụng
        self.encoders['education_mapping'] = mapping
    
        print(f"\nMapping thứ bậc: {mapping}")
        print(f"Mã hóa: 0 (thấp nhất) → {len(mapping)-1} (cao nhất)\n")
    
        # Log bắt đầu encoding
        self.log_action("Bắt đầu encode Education", 
                       f"Cột: {col} → {new_col}, Mapping: {mapping}")
    
        # Tạo ordered categorical
        try:
            cat_type = pd.CategoricalDtype(categories=mapping, ordered=True)
            self.cleaned_data[col] = self.cleaned_data[col].astype(cat_type)
        
            # Lấy codes (-1 cho NaN/unmapped → chuyển thành NaN)
            codes = self.cleaned_data[col].cat.codes.replace(-1, pd.NA)
            self.cleaned_data[new_col] = codes.astype('Int64')  # nullable integer
        
            # Kiểm tra unmapped
            unmapped_mask = self.cleaned_data[new_col].isna()
            unmapped_count = unmapped_mask.sum()
        
            if unmapped_count > 0:
                print(f"Phát hiện {unmapped_count} giá trị không khớp mapping → set NaN")
                unmapped_vals = self.cleaned_data.loc[unmapped_mask, col].unique()
                print(f"   Giá trị unmapped: {list(unmapped_vals)}")
                self.log_action("Cảnh báo encode Education", 
                               f"{unmapped_count} giá trị unmapped: {list(unmapped_vals)}")
            else:
                print(f"Tất cả giá trị đã được mapping")
                self.log_action("Encode Education thành công", "Tất cả giá trị đã được mapping")
       
            # Thống kê
            print(f"\nThống kê sau encoding:")
            print(self.cleaned_data[[col, new_col]].value_counts().sort_index())
        
            # Ghi log thành công
            self.log_action(f"Hoàn thành encode Education", 
                           f"Cột mới: {new_col}, Unmapped: {unmapped_count}")
        
            return {
                'status': 'success',
                'original_col': col,
                'encoded_col': new_col,
                'mapping': mapping,
                'unmapped_count': int(unmapped_count)
            }
        
        except Exception as e:
            print(f"✗ Lỗi khi encode {col}: {e}")
            self.log_action("Lỗi encode Education", str(e))
            return {'status': 'error', 'error': str(e)}

    # =================================================================================================================
    # HÀM ENCODE MARITAL_STATUS (NOMINAL → ONE-HOT)
    def encode_marital_status_onehot(self, col='Marital_Status', prefix='Marital', 
                                  group_threshold=0.01, drop_original=False):
        """
        One-hot encode cột Marital_Status (không có thứ bậc).
        Tự động gom các nhãn hiếm (< threshold) vào 'Other' trước khi encode.
        - đảm bảo dummy dtype = uint8
        - lưu metadata vào self.encoders['marital_status_encoding']
        """
        print("="*100)
        print("MÃ HÓA MARITAL_STATUS (ONE-HOT ENCODING)")
    
        if self.cleaned_data is None:
            print("Dữ liệu chưa được tải.")
            return None
        
        if col not in self.cleaned_data.columns:
            print(f"Không tìm thấy cột '{col}'")
            return None
    
        # Bước 1: Gom rare categories
        print(f"\n1. Gom các nhãn hiếm (tần suất < {group_threshold*100:.1f}%)...)")

        value_counts = self.cleaned_data[col].value_counts()
        freq = value_counts / len(self.cleaned_data)
    
        rare_cats = freq[freq < group_threshold].index.tolist()
    
        grouped_col = f"{col}_Grouped"
        if rare_cats:
            print(f"   Nhãn hiếm: {rare_cats}")
            self.cleaned_data[grouped_col] = self.cleaned_data[col].replace(
                {cat: 'Other' for cat in rare_cats}
            )
            print(f"Đã gom vào '{grouped_col}'")
            encode_col = grouped_col
        else:
            print(f"Không có nhãn hiếm")
            # still create grouped_col as copy to have consistent pipeline
            self.cleaned_data[grouped_col] = self.cleaned_data[col].copy()
            encode_col = grouped_col
    
        # Bước 2: One-hot encode với dtype explicitly uint8
        print(f"\n2. One-hot encoding cột '{encode_col}'...")
    
        try:
            # pd.get_dummies supports dtype parameter
            dummies = pd.get_dummies(
                self.cleaned_data[encode_col],
                prefix=prefix,
                dummy_na=False,
                dtype='uint8'
            )
        
            # Avoid column overlap (rename if necessary)
            overlap = set(dummies.columns) & set(self.cleaned_data.columns)
            if overlap:
                dummies = dummies.rename(columns={c: f"{c}_enc" for c in overlap})
        
            # Concat into dataset
            self.cleaned_data = pd.concat([self.cleaned_data, dummies], axis=1)
        
            dummy_cols = list(dummies.columns)
            print(f"Đã tạo {len(dummy_cols)} dummy columns:")
            for dc in dummy_cols:
                print(f"     - {dc} ({self.cleaned_data[dc].dtype})")
        
            # Bước 3: Xóa cột gốc nếu cần
            if drop_original:
                self.cleaned_data = self.cleaned_data.drop(columns=[col])
                print(f"\n3.Đã xóa cột gốc '{col}'")
            else:
                print(f"\n3.Giữ cột gốc '{col}' để audit")
        
            # Lưu metadata để sử dụng cho transform dữ liệu mới
            self.encoders['marital_status_encoding'] = {
                'original_col': col,
                'grouped_col': grouped_col,
                'dummy_columns': dummy_cols,
                'rare_categories': rare_cats,
                'prefix': prefix,
                'dtype': 'uint8'
            }
        
            # Ghi log
            self.log_action(f"Encode {col} (one-hot)", 
                           f"Rare grouped: {len(rare_cats)}, Dummies: {len(dummy_cols)}")
        
            return {
                'status': 'success',
                'original_col': col,
                'grouped_col': grouped_col,
                'dummy_columns': dummy_cols,
                'rare_categories': rare_cats,
                'grouped_count': len(rare_cats)
            }
        
        except Exception as e:
            print(f"✗ Lỗi khi encode {col}: {e}")
            return {'status': 'error', 'error': str(e)}

    #==================================================================================================================
    # HÀM ĐẢM BẢO CÁC CỘT NHỊ PHÂN LÀ 'uint8'
    def ensure_binary_dtypes(self, binary_cols=None):
        """
        Đảm bảo các cột nhị phân được cast về dtype 'uint8'.
        Nếu binary_cols là None, tự động tìm các cột có duy nhất hai giá trị {0,1} hoặc {False,True}.
        Gọi trước khi export hoặc trước khi đưa dữ liệu vào thuật toán.
        """
        if self.cleaned_data is None:
            return None

        if binary_cols is None:
            # tự động phát hiện cột nhị phân (0/1, True/False)
            candidate_cols = []
            for col in self.cleaned_data.columns:
                if pd.api.types.is_integer_dtype(self.cleaned_data[col].dtype) or pd.api.types.is_bool_dtype(self.cleaned_data[col].dtype):
                    uniques = self.cleaned_data[col].dropna().unique()
                    if set(uniques).issubset({0,1,True,False}):
                        candidate_cols.append(col)
            binary_cols = candidate_cols

        # Cast to uint8
        for col in binary_cols:
            try:
                self.cleaned_data[col] = self.cleaned_data[col].astype('uint8')
            except Exception:
                # fallback: convert via .where then astype
                self.cleaned_data[col] = self.cleaned_data[col].apply(lambda x: 1 if x in (1, True) else 0).astype('uint8')

        # Log và trả về danh sách đã cast
        self.log_action("Ensure binary dtypes", f"Casted {len(binary_cols)} columns to uint8: {binary_cols}")
        return {'casted_columns': list(binary_cols)}

    # =================================================================================================================
    # HÀM THỰC HIỆN TẤT CẢ CÁC BƯỚC LÀM SẠCH
    def run_complete_cleaning(self, export=True):
        """
        Thực hiện toàn bộ quy trình làm sạch dữ liệu
        
        Args:
            export (bool, optional): Xuất dữ liệu ra file CSV. Mặc định là True.
            
        Returns:
            dict: Kết quả của quy trình làm sạch
        """
        print("="*100)
        print("THỰC HIỆN QUY TRÌNH LÀM SẠCH DỮ LIỆU HOÀN CHỈNH")
        
        start_time = time.time()
        
        # Đảm bảo dữ liệu đã được tải
        if self.raw_data is None:
            if self.dataset_path:
                self.load_data()
            else:
                print("Không có dữ liệu để xử lý")
                return None
                
        # Đảm bảo có bản sao để làm sạch
        if self.cleaned_data is None:
            self.cleaned_data = self.raw_data.copy()
            
        # Ghi nhớ shape ban đầu
        original_shape = self.cleaned_data.shape if self.cleaned_data is not None else None
        
        # Bước 1: Xử lý missing values
        missing_result = self.handle_missing_value()
        
        # Bước 2: Loại bỏ dòng trùng lặp feature (loại trừ ID)
        duplicate_removal_result = self.remove_feature_duplicates()

        # Bước 4: Loại bỏ outliers cụ thể 
        outliers_result = self.remove_specific_outliers()

        # Bước 5: Chuẩn hóa dữ liệu phân loại
        categorical_result = self.clean_categorical_data()
        
        # Bước 6: Chuyển đổi kiểu dữ liệu (bao gồm Dt_Customer) và MÃ HÓA phân loại (Martial_Status, Education)
        conversion_result = self.convert_data_types()

        # Lấy kết quả encoding đã thực hiện trong convert_data_types
        if conversion_result and isinstance(conversion_result, dict):
            encoding_result = conversion_result.get('encoding', {'education': None, 'marital_status': None})
        else:
            encoding_result = {'education': None, 'marital_status': None}

        # Bước 7: Loại bỏ features không cần thiết
        feature_removal_result = self.remove_non_demographic_features()
        
        # Bước 8: Kiểm tra và xác thực dữ liệu
        validation_result = self.validate_cleaned_data()
        
        # Bước 9: Xuất dữ liệu 
        output_file = None
        if export:
            output_file = self.export_cleaned_data()
            
        # Tổng kết kết quả
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        final_shape = self.cleaned_data.shape if self.cleaned_data is not None else None
        
        # Tóm tắt
        print("="*100)
        print("TÓM TẮT QUY TRÌNH LÀM SẠCH DỮ LIỆU")
        print("="*100)
        
        print(f"Thời gian xử lý          : {elapsed_time:.2f} giây")
        if original_shape and final_shape:
            print(f"Dataset ban đầu          : {original_shape[0]:,} dòng × {original_shape[1]} cột")
            print(f"Dataset sau làm sạch     : {final_shape[0]:,} dòng × {final_shape[1]} cột")
            rows_diff = original_shape[0] - final_shape[0]
            cols_diff = original_shape[1] - final_shape[1]
            if rows_diff != 0:
                print(f"Số dòng đã loại bỏ       : {rows_diff} ({rows_diff/original_shape[0]*100:.2f}%)")
            if cols_diff != 0:
                print(f"Số cột đã loại bỏ        : {cols_diff}")
        
        # Chi tiết thay đổi
        print("\nChi tiết thay đổi:")
        # Missing values
        if missing_result and 'missing_rows' in missing_result:
            print(f"- Missing values         : Đã tách {missing_result.get('missing_rows', 0)} dòng có missing")
        
        # Duplicate removal
        if duplicate_removal_result and duplicate_removal_result.get('removed', 0) > 0:
            print(f"- Loại bỏ trùng lặp      : {duplicate_removal_result['removed']} dòng")  
            
        # Categorical cleaning
        if categorical_result:
            print(f"- Chuẩn hóa phân loại    : {categorical_result.get('changes_made', 0)} giá trị")
            
        # Data type conversion
        if conversion_result:
            # Detect if Dt_Customer conversion succeeded
            dt_converted = 'Dt_Customer' in conversion_result and conversion_result['Dt_Customer'].get('success', False)

            conversions = []
            if dt_converted:
                conversions.append("Dt_Customer → datetime")

            if conversions:
                print(f"- Chuyển đổi / phát hiện : {', '.join(conversions)}")

        # Report encoding results (lấy từ convert_data_types)
        if encoding_result:
            edu_enc = encoding_result.get('education')
            mar_enc = encoding_result.get('marital_status')
            if edu_enc:
                unmapped = edu_enc.get('unmapped_count', 0)
                print(f"- Encode Education       : {'success' if edu_enc.get('status')=='success' else 'error'} (unmapped: {unmapped})")
            if mar_enc:
                dummies = mar_enc.get('dummy_columns', []) if isinstance(mar_enc, dict) else []
                grouped = mar_enc.get('grouped_count', 0) if isinstance(mar_enc, dict) else 0
                print(f"- Encode Marital_Status  : {'success' if mar_enc and mar_enc.get('status')=='success' else 'error'} (dummies: {len(dummies)}, grouped_rare: {grouped})")
        
        # Feature removal
        if feature_removal_result:
            removed_count = len(feature_removal_result.get('removed_columns', []))
            print(f"- Loại bỏ features       : {removed_count} cột (archived)")
        
        # Validation
        if validation_result:
            validation_status = "Đạt" if validation_result.get('all_passed', False) else "Chưa đạt"
            print(f"- Kết quả xác thực       : {validation_status}")
            
        if output_file:
            print(f"- File xuất              : {output_file}")
            
        if feature_removal_result and feature_removal_result.get('archived_file'):
            print(f"- Archived features      : {feature_removal_result['archived_file']}")
        
        # THÊM THÔNG TIN OUTLIERS FILE
        if outliers_result and outliers_result.get('outliers_file'):
            print(f"- Outliers removed       : {outliers_result['outliers_file']}")
        
        print("\n")

        # Xuất báo cáo text
        try:
            self.export_text_report()
        except Exception as e:
            print(f"Lỗi khi xuất báo cáo text: {e}")

        # Ghi log
        self.log_action("Hoàn thành quy trình làm sạch hoàn chỉnh", 
                      f"Thời gian: {elapsed_time:.2f}s, Kết quả: {'Đạt' if validation_result and validation_result.get('all_passed', False) else 'Chưa đạt'}")
        # Kết quả
        return {
            'success': True,
            'time_elapsed': elapsed_time,
            'original_shape': original_shape,
            'final_shape': final_shape,
            'missing_result': missing_result,
            'outliers_result': outliers_result, 
            'categorical_result': categorical_result,
            'conversion_result': conversion_result,
            'encoding_result': encoding_result,
            'feature_removal_result': feature_removal_result,
            'validation_result': validation_result,
            'output_file': output_file
        }
    # =================================================================================================================
    # HÀM HIỂN THỊ LOG XỬ LÝ
    def display_processing_log(self):
        """
        Hiển thị log các hành động đã thực hiện trong quá trình xử lý
        """
        print("=" * 100)
        print("LOG CÁC HÀNH ĐỘNG THỰC HIỆN")
        print("=" * 100)

        if not self.processing_log:
            print("Không có log nào được ghi lại")
            return

        print(f"Tổng số hành động: {len(self.processing_log)}\n")
        for i, log_entry in enumerate(self.processing_log, 1):
            timestamp = log_entry['timestamp'].strftime('%H:%M:%S')
            action = log_entry['action']
            details = log_entry.get('details', '')
            print(f" {i:2d}. [{timestamp}] {action}")
            if details:
              print(f"     └─ {details}")
        print(f"\nĐã hoàn thành {len(self.processing_log)} hành động\n")

# =================================================================================================================
# HÀM MAIN 
def main():
    """
    Hàm chính
    """
    # Đường dẫn đến dataset
    dataset_path = r"C:\Project\Machine_Learning\Machine_Learning\Dataset\Customer_Behavior.csv"
    
    try:
        # Khởi tạo đối tượng làm sạch dữ liệu
        cleaner = DataCleaning(dataset_path)
        
        # Thực hiện quy trình làm sạch dữ liệu
        result = cleaner.run_complete_cleaning() # Gọi hàm làm sạch toàn bộ quy trình
        
        # Hiển thị log các hành động đã thực hiện
        cleaner.display_processing_log()

        # Kiểm tra kết quả
        if result and result.get('success'):
            print("\nQuy trình làm sạch dữ liệu đã hoàn thành")
            
            if result.get('output_file'):
                print(f"Dữ liệu đã được xuất ra file: {result['output_file']}")
                 
            validation = result.get('validation_result', {})
            if validation and validation.get('all_passed', False):
                print("Tất cả các kiểm tra xác thực đều đạt yêu cầu")
            else:
                print("Có một số vấn đề trong quá trình xác thực - xem chi tiết ở trên")
                
        else:
            print("\nCó lỗi trong quá trình làm sạch dữ liệu")
            
    except Exception as e:
        print(f"\nLỖI CHƯƠNG TRÌNH: {e}")
        import traceback
        traceback.print_exc()
        
        
# Gọi hàm main khi chạy trực tiếp file
if __name__ == "__main__":
    main()