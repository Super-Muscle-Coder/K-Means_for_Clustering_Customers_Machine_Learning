"""
================================================================================
FEATURE ENGINEERING FOR MULTI-OBJECTIVE CUSTOMER CLUSTERING
================================================================================

Module: Feature Engineering cho 3 chiến lược clustering
- Class 1: DemographicFeatureEngineering (Life Stage Segmentation)
- Class 2: ProductChannelFeatureEngineering (Shopping Behavior Segmentation)
- Class 3: RFMFeatureEngineering (Customer Value Segmentation)

Input:  Customer_Behavior_cleaned.csv
Output: 3 datasets + 3 reports (TXT format, overwrite mode)
================================================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import yeojohnson
import os
import warnings

warnings.filterwarnings('ignore')


# ================================================================================
# CLASS 1: DEMOGRAPHIC FEATURE ENGINEERING
# ================================================================================

class DemographicFeatureEngineering:
    """
    Feature Engineering cho DEMOGRAPHIC CLUSTERING (Life Stage Segmentation).
    
    MỤC TIÊU: Phân khúa khách hàng theo đặc điểm nhân khẩu học và giai đoạn cuộc sống
    
    FEATURES TẠO RA (6 features):
    1. Age                                      (18-100, derived)
    2. Income                                   (original)
    3. TotalChildren                            (0-3, derived)
    4. Income_per_Family_Member_Transformed     (Yeo-Johnson transformed)
    5. Education_ord                            (0-4, original ordinal)
    6. Marital_Encoded                          (0-2, composite)
    
    Pipeline:
    Step 1   : Tải dữ liệu
    Step 2-5 : Tạo derived features (Age, TotalChildren, Income_per_Family_Member, Marital_Encoded)
    Step 6   : Transform skewed features (Yeo-Johnson if skew > 1.0)
    Step 7   : Chọn final 6 features
    Step 8   : Validate (missing, duplicates, variance)
    Step 9   : Xuất tệp CSV + Report
    
    INPUT:
    - CSV: Customer_Behavior_cleaned.csv

    OUTPUT:
    - CSV: Customer_Behavior_Demographic.csv
    - Report: Demographic_Engineering_Report.log
    """
    
    def __init__(self, input_path, output_dir, report_dir):
        """
        Khởi tạo Demographic Feature Engineering.
        
        Args:
            input_path (str): Đường dẫn dataset cleaned
            output_dir (str): Thư mục lưu dataset output
            report_dir (str): Thư mục lưu report
        """
        self.input_path = input_path
        self.output_dir = output_dir
        self.report_dir = report_dir
        
        # Create directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(report_dir, exist_ok=True)
        
        # Output paths
        self.output_csv = os.path.join(output_dir, "Customer_Behavior_Demographic.csv")
        self.report_file = os.path.join(report_dir, "Demographic_Engineering_Report.log")
        
        # Data storage
        self.df = None
        self.df_engineered = None
        self.feature_stats = {}
        self.processing_log = []
        
        # Reference year for Age calculation
        self.REFERENCE_YEAR = 2014
    
    def log_action(self, action, details=""):
        """Ghi log hành động."""
        self.processing_log.append({
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'action': action,
            'details': details
        })
    
    def _print_header(self, title, width=100, char='='):
        """In header có format."""
        print(char * width)
        print(f"{title:^{width}}")
        print(char * width)
        print()
    
    def _print_subheader(self, title, width=100, char='-'):
        """In subheader."""
        print()
        print(char * width)
        print(title)
        print(char * width)
    
    # ============================================================================
    # STEP 1: LOAD DATA
    # ============================================================================
    
    def load_data(self):
        """Load cleaned dataset."""
        self._print_header("DEMOGRAPHIC FEATURE ENGINEERING - LOAD DATA")
        
        try:
            self.df = pd.read_csv(self.input_path)
            
            print(f"Loaded dataset successfully")
            print(f"   Shape: {self.df.shape[0]:,} rows x {self.df.shape[1]} columns")
            print(f"   Memory: {self.df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
            print()
            
            self.log_action("Load data", f"Shape: {self.df.shape}")
            
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            self.log_action("Load data FAILED", str(e))
            return False
    
    # ============================================================================
    # STEP 2: CREATE DERIVED FEATURES
    # ============================================================================
    def create_age_feature(self):
        """Tạo Age từ Year_Birth."""
        self._print_header("STEP 1: CREATE AGE FEATURE")
        
        if 'Year_Birth' not in self.df.columns:
            print("Year_Birth column not found. Skipping.")
            return
        
        # Calculate Age
        self.df['Age'] = self.REFERENCE_YEAR - self.df['Year_Birth']
        
        # Cap outliers
        MAX_AGE = 100
        outliers = (self.df['Age'] > MAX_AGE).sum()
        if outliers > 0:
            print(f"Capping {outliers} ages > {MAX_AGE} to {MAX_AGE}")
            self.df.loc[self.df['Age'] > MAX_AGE, 'Age'] = MAX_AGE
        
        # Stats
        age_stats = {
            'min': self.df['Age'].min(),
            'max': self.df['Age'].max(),
            'mean': self.df['Age'].mean(),
            'median': self.df['Age'].median(),
            'std': self.df['Age'].std(),
            'skewness': self.df['Age'].skew()
        }
        
        print(f"Created 'Age' feature")
        print(f"   Formula: Age = {self.REFERENCE_YEAR} - Year_Birth")
        print(f"   Range: [{age_stats['min']:.0f}, {age_stats['max']:.0f}]")
        print(f"   Mean: {age_stats['mean']:.1f}, Median: {age_stats['median']:.1f}")
        print(f"   Std: {age_stats['std']:.1f}, Skewness: {age_stats['skewness']:.3f}")
        print()
        
        self.feature_stats['Age'] = age_stats
        self.log_action("Create Age", f"Range: [{age_stats['min']:.0f}, {age_stats['max']:.0f}]")
    
    def create_total_children_feature(self):
        """Tạo TotalChildren từ Kidhome + Teenhome."""
        self._print_header("STEP 2: CREATE TOTALCHILDREN FEATURE")
        
        if 'Kidhome' not in self.df.columns or 'Teenhome' not in self.df.columns:
            print("Kidhome or Teenhome not found. Skipping.")
            return
        
        self.df['TotalChildren'] = self.df['Kidhome'] + self.df['Teenhome']
        
        # Distribution
        dist = self.df['TotalChildren'].value_counts().sort_index()
        
        print(f"Created 'TotalChildren' feature")
        print(f"   Formula: TotalChildren = Kidhome + Teenhome")
        print(f"\nDistribution:")
        for num, count in dist.items():
            pct = count / len(self.df) * 100
            print(f"   {num} children: {count:>5,} customers ({pct:>5.1f}%)")
        print()
        
        self.feature_stats['TotalChildren'] = {
            'min': self.df['TotalChildren'].min(),
            'max': self.df['TotalChildren'].max(),
            'mean': self.df['TotalChildren'].mean(),
            'distribution': dist.to_dict()
        }
        
        self.log_action("Create TotalChildren", f"Distribution: {dist.to_dict()}")
    
    def create_income_per_family_member(self):
        """Tạo Income_per_Family_Member."""
        self._print_header("STEP 3: CREATE INCOME_PER_FAMILY_MEMBER FEATURE")
        
        if 'Income' not in self.df.columns or 'TotalChildren' not in self.df.columns:
            print("Income or TotalChildren not found. Skipping.")
            return
        
        # Family size = 2 adults + TotalChildren
        family_size = 2 + self.df['TotalChildren']
        self.df['Income_per_Family_Member'] = self.df['Income'] / family_size
        
        stats = {
            'min': self.df['Income_per_Family_Member'].min(),
            'max': self.df['Income_per_Family_Member'].max(),
            'mean': self.df['Income_per_Family_Member'].mean(),
            'median': self.df['Income_per_Family_Member'].median(),
            'std': self.df['Income_per_Family_Member'].std(),
            'skewness': self.df['Income_per_Family_Member'].skew()
        }
        
        print(f"Created 'Income_per_Family_Member' feature")
        print(f"   Formula: Income / (2 + TotalChildren)")
        print(f"   Range: [{stats['min']:,.0f}, {stats['max']:,.0f}]")
        print(f"   Mean: {stats['mean']:,.0f}, Median: {stats['median']:,.0f}")
        print(f"   Skewness: {stats['skewness']:.3f}")
        print()
        
        self.feature_stats['Income_per_Family_Member'] = stats
        self.log_action("Create Income_per_Family_Member", 
                       f"Mean: {stats['mean']:,.0f}, Skew: {stats['skewness']:.3f}")
    
    def create_marital_encoded(self):
        """Tạo Marital_Encoded từ Marital_* dummies."""
        self._print_header("STEP 4: CREATE MARITAL_ENCODED FEATURE")
        
        # Define mapping
        marital_mapping = {
            'Marital_Single': 0,
            'Marital_Divorced': 0,
            'Marital_Widow': 0,
            'Marital_Together': 1,
            'Marital_Married': 2
        }
        
        # Check which columns exist
        existing_cols = [col for col in marital_mapping.keys() if col in self.df.columns]
        
        if not existing_cols:
            print("No Marital_* columns found. Skipping.")
            return
        
        # Encode
        self.df['Marital_Encoded'] = 0
        for col, value in marital_mapping.items():
            if col in self.df.columns:
                self.df['Marital_Encoded'] += self.df[col] * value
        
        # Distribution
        dist = self.df['Marital_Encoded'].value_counts().sort_index()
        
        print(f"Created 'Marital_Encoded' feature")
        print(f"\nMapping:")
        print(f"   0: Single/Divorced/Widow (solo)")
        print(f"   1: Together (partnership)")
        print(f"   2: Married (legal partnership)")
        print(f"\nDistribution:")
        for code, count in dist.items():
            pct = count / len(self.df) * 100
            print(f"   Code {code}: {count:>5,} customers ({pct:>5.1f}%)")
        print()
        
        self.feature_stats['Marital_Encoded'] = {
            'distribution': dist.to_dict(),
            'mapping': marital_mapping
        }
        
        self.log_action("Create Marital_Encoded", f"Distribution: {dist.to_dict()}")
    
    # ============================================================================
    # STEP 3: TRANSFORM SKEWED FEATURES
    # ============================================================================
    
    def transform_skewed_features(self):
        """Apply Yeo-Johnson transform to skewed features."""
        self._print_header("STEP 5: TRANSFORM SKEWED FEATURES")
        
        # Check skewness threshold
        SKEW_THRESHOLD = 1.0
        
        # Candidate feature for transformation
        if 'Income_per_Family_Member' not in self.df.columns:
            print("Income_per_Family_Member not found. Skipping.")
            return
        
        original_skew = self.df['Income_per_Family_Member'].skew()
        
        print(f"Checking skewness of Income_per_Family_Member:")
        print(f"   Original skewness: {original_skew:.3f}")
        
        if abs(original_skew) < SKEW_THRESHOLD:
            print(f"   Skewness < {SKEW_THRESHOLD} -> No transform needed")
            self.df['Income_per_Family_Member_Transformed'] = self.df['Income_per_Family_Member']
            return
        
        # Apply Yeo-Johnson transform
        try:
            transformed_data, lambda_param = yeojohnson(self.df['Income_per_Family_Member'])
            self.df['Income_per_Family_Member_Transformed'] = transformed_data
            
            new_skew = pd.Series(transformed_data).skew()
            
            print(f"\nApplied Yeo-Johnson transform:")
            print(f"   Lambda parameter: {lambda_param:.3f}")
            print(f"   New skewness: {new_skew:.3f}")
            print(f"   Improvement: {abs(original_skew) - abs(new_skew):.3f}")
            print()
            
            self.feature_stats['Income_per_Family_Member_Transformed'] = {
                'original_skew': original_skew,
                'new_skew': new_skew,
                'lambda': lambda_param
            }
            
            self.log_action("Transform Income_per_Family_Member", 
                           f"Skew: {original_skew:.3f} -> {new_skew:.3f}")
            
        except Exception as e:
            print(f"Transform failed: {e}")
            self.df['Income_per_Family_Member_Transformed'] = self.df['Income_per_Family_Member']
    
    # ============================================================================
    # STEP 4: SELECT FINAL FEATURES
    # ============================================================================
    
    def select_final_features(self):
        """Select 6 final features for demographic clustering."""
        self._print_header("STEP 6: SELECT FINAL FEATURES")
        
        # Define final features
        final_features = [
            'Age',
            'Income',
            'TotalChildren',
            'Income_per_Family_Member_Transformed',
            'Education_ord',
            'Marital_Encoded'
        ]
        
        # Filter existing features
        existing_features = [f for f in final_features if f in self.df.columns]
        
        # Create final dataset
        self.df_engineered = self.df[existing_features].copy()
        
        print(f"Selected {len(existing_features)} features for demographic clustering:")
        print()
        
        for i, feat in enumerate(existing_features, 1):
            dtype = str(self.df_engineered[feat].dtype)
            n_unique = self.df_engineered[feat].nunique()
            print(f"{i}. {feat:<40} ({dtype}, {n_unique} unique values)")
        
        print()
        print(f"Final dataset shape: {self.df_engineered.shape}")
        print()
        
        self.log_action("Select final features", f"{len(existing_features)} features selected")
    
    # ============================================================================
    # STEP 5: VALIDATE & EXPORT
    # ============================================================================
    
    def validate_features(self):
        """Validate engineered features."""
        self._print_header("STEP 7: VALIDATE FEATURES")
        
        print("Data Quality Checks:")
        print()
        
        # Missing values
        missing = self.df_engineered.isnull().sum()
        if missing.sum() == 0:
            print("   Missing values: None (PASS)")
        else:
            print(f"   Missing values: {missing.sum()} (WARNING)")
            for col in missing[missing > 0].index:
                print(f"      {col}: {missing[col]}")
        
        # Duplicates
        dup_count = self.df_engineered.duplicated().sum()
        print(f"   Duplicate rows: {dup_count} ({dup_count/len(self.df_engineered)*100:.2f}%)")
        
        # Variance
        low_var_features = []
        for col in self.df_engineered.columns:
            var = self.df_engineered[col].var()
            if var < 0.01:
                low_var_features.append(col)
        
        if low_var_features:
            print(f"   Low variance features: {len(low_var_features)} (WARNING)")
            for feat in low_var_features:
                print(f"      {feat}: var={self.df_engineered[feat].var():.6f}")
        else:
            print(f"   Low variance features: None (PASS)")
        
        print()
        self.log_action("Validate features", f"Checks completed")
    
    def export_dataset(self):
        """Export engineered dataset to CSV."""
        self._print_header("STEP 8: EXPORT DATASET")
        
        try:
            self.df_engineered.to_csv(self.output_csv, index=False)
            
            print(f"Exported dataset successfully:")
            print(f"   Path: {self.output_csv}")
            print(f"   Shape: {self.df_engineered.shape}")
            print(f"   Size: {os.path.getsize(self.output_csv) / 1024:.2f} KB")
            print()
            
            self.log_action("Export dataset", self.output_csv)
            return True
            
        except Exception as e:
            print(f"Export failed: {e}")
            self.log_action("Export FAILED", str(e))
            return False
    
    def export_report(self):
        """Export engineering report."""
        self._print_header("STEP 9: EXPORT REPORT")
        
        try:
            with open(self.report_file, 'w', encoding='utf-8') as f:
                # Header
                f.write("=" * 100 + "\n")
                f.write("DEMOGRAPHIC FEATURE ENGINEERING REPORT\n")
                f.write("=" * 100 + "\n\n")
                
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Input: {self.input_path}\n")
                f.write(f"Output: {self.output_csv}\n\n")
                
                # Dataset info
                f.write("DATASET INFORMATION:\n")
                f.write("-" * 100 + "\n")
                f.write(f"Original dataset: {self.df.shape[0]:,} rows x {self.df.shape[1]} columns\n")
                f.write(f"Engineered dataset: {self.df_engineered.shape[0]:,} rows x {self.df_engineered.shape[1]} columns\n\n")
                
                # Features created
                f.write("FEATURES CREATED:\n")
                f.write("-" * 100 + "\n")
                for i, col in enumerate(self.df_engineered.columns, 1):
                    f.write(f"{i}. {col}\n")
                    if col in self.feature_stats:
                        stats = self.feature_stats[col]
                        if isinstance(stats, dict):
                            for key, val in stats.items():
                                f.write(f"   {key}: {val}\n")
                    f.write("\n")
                
                # Processing log
                f.write("\nPROCESSING LOG:\n")
                f.write("-" * 100 + "\n")
                for i, log in enumerate(self.processing_log, 1):
                    f.write(f"{i}. [{log['timestamp']}] {log['action']}\n")
                    if log['details']:
                        f.write(f"   Details: {log['details']}\n")
                
                f.write("\n" + "=" * 100 + "\n")
                f.write("Report generated successfully\n")
                f.write("=" * 100 + "\n")
            
            print(f"Exported report successfully:")
            print(f"   Path: {self.report_file}")
            print(f"   Size: {os.path.getsize(self.report_file) / 1024:.2f} KB")
            print()
            
            self.log_action("Export report", self.report_file)
            return True
            
        except Exception as e:
            print(f"Report export failed: {e}")
            return False
    
    # ============================================================================
    # MAIN PIPELINE
    # ============================================================================
    
    def run_engineering(self):
        """Run complete demographic feature engineering pipeline."""
        print("\n" + "=" * 100)
        print("DEMOGRAPHIC FEATURE ENGINEERING PIPELINE".center(100))
        print("=" * 100 + "\n")
        
        start_time = datetime.now()
        
        # Step 1: Load data
        if not self.load_data():
            print("Pipeline failed at load_data()")
            return False
        
        # Step 2: Create derived features
        self.create_age_feature()
        self.create_total_children_feature()
        self.create_income_per_family_member()
        self.create_marital_encoded()
        
        # Step 3: Transform skewed features
        self.transform_skewed_features()
        
        # Step 4: Select final features
        self.select_final_features()
        
        # Step 5: Validate
        self.validate_features()
        
        # Step 6: Export
        self.export_dataset()
        self.export_report()
        
        # Summary
        elapsed = (datetime.now() - start_time).total_seconds()
        
        print("=" * 100)
        print("PIPELINE COMPLETED".center(100))
        print("=" * 100)
        print(f"Time elapsed: {elapsed:.2f} seconds")
        print(f"Output dataset: {self.output_csv}")
        print(f"Output report: {self.report_file}")
        print("=" * 100 + "\n")
        
        return True


# ================================================================================
# CLASS 2: PRODUCT + CHANNEL FEATURE ENGINEERING
# ================================================================================

class ProductChannelFeatureEngineering:
    """
    Feature Engineering cho PRODUCT + CHANNEL CLUSTERING (Shopping Behavior Segmentation).
    
    MỤC TIÊU: Phân khúa khách hàng theo hành vi mua sắm (sản phẩm + kênh)
    
    FEATURES TẠO RA (12 features):
    
    PRODUCT PREFERENCE RATIOS (6 features):
    1. Wine_Ratio              (MntWines / Total_Spent)
    2. Meat_Ratio              (MntMeatProducts / Total_Spent)
    3. Fish_Ratio              (MntFishProducts / Total_Spent)
    4. Fruit_Ratio             (MntFruits / Total_Spent)
    5. Sweet_Ratio             (MntSweetProducts / Total_Spent)
    6. Gold_Ratio              (MntGoldProds / Total_Spent)
    
    CHANNEL PREFERENCE RATIOS (4 features):
    7. Web_Ratio               (NumWebPurchases / TotalPurchases)
    8. Catalog_Ratio           (NumCatalogPurchases / TotalPurchases)
    9. Store_Ratio             (NumStorePurchases / TotalPurchases)
    10. Deal_Ratio             (NumDealsPurchases / TotalPurchases)
    
    CONTEXT FEATURES (2 features):
    11. TotalPurchases         (sum of all Num*Purchases)
    12. NumWebVisitsMonth      (original, digital engagement)
    
    Pipeline:
    Step 1   : Load data
    Step 2   : Tạo Total_Spent (sum all Mnt* features)
    Step 3   : Tạo TotalPurchases (sum all Num*Purchases)
    Step 4   : Tạo Product Ratios (6 features)
    Step 5   : Tạo Channel Ratios (4 features)
    Step 6   : Chọn ra 12 đặc trưng cuối cùng
    Step 7   : Validate (missing, duplicates, variance)
    Step 8   : Xuất tệp CSV + Report
    
    INPUT:
    - CSV: Customer_Behavior_cleaned.csv

    OUTPUT:
    - CSV: Customer_Behavior_ProductChannel.csv
    - Report: ProductChannel_Engineering_Report.log
    """
    
    def __init__(self, input_path, output_dir, report_dir):
        """
        Khởi tạo Product+Channel Feature Engineering.
        
        Args:
            input_path (str): Đường dẫn dataset cleaned
            output_dir (str): Thư mục lưu dataset output
            report_dir (str): Thư mục lưu report
        """
        self.input_path = input_path
        self.output_dir = output_dir
        self.report_dir = report_dir
        
        # Create directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(report_dir, exist_ok=True)
        
        # Output paths
        self.output_csv = os.path.join(output_dir, "Customer_Behavior_ProductChannel.csv")
        self.report_file = os.path.join(report_dir, "ProductChannel_Engineering_Report.log")
        
        # Data storage
        self.df = None
        self.df_engineered = None
        self.feature_stats = {}
        self.processing_log = []
    
    def log_action(self, action, details=""):
        """Ghi log hành động."""
        self.processing_log.append({
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'action': action,
            'details': details
        })
    
    def _print_header(self, title, width=100, char='='):
        """In header có format."""
        print(char * width)
        print(f"{title:^{width}}")
        print(char * width)
        print()
    
    def _print_subheader(self, title, width=100, char='-'):
        """In subheader."""
        print()
        print(char * width)
        print(title)
        print(char * width)
    
    # ============================================================================
    # STEP 1: LOAD DATA
    # ============================================================================
    
    def load_data(self):
        """Load cleaned dataset."""
        self._print_header("PRODUCT+CHANNEL FEATURE ENGINEERING - LOAD DATA")
        
        try:
            self.df = pd.read_csv(self.input_path)
            
            print(f"Loaded dataset successfully")
            print(f"   Shape: {self.df.shape[0]:,} rows x {self.df.shape[1]} columns")
            print(f"   Memory: {self.df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
            print()
            
            self.log_action("Load data", f"Shape: {self.df.shape}")
            
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            self.log_action("Load data FAILED", str(e))
            return False
    
    # ============================================================================
    # STEP 2: CREATE TOTAL_SPENT
    # ============================================================================
    
    def create_total_spent(self):
        """Tạo Total_Spent từ sum(Mnt* features)."""
        self._print_header("STEP 1: CREATE TOTAL_SPENT FEATURE")
        
        # Define Mnt columns
        mnt_cols = ['MntWines', 'MntFruits', 'MntMeatProducts', 
                    'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
        
        # Check existing columns
        existing_mnt_cols = [col for col in mnt_cols if col in self.df.columns]
        
        if not existing_mnt_cols:
            print("No Mnt* columns found. Skipping.")
            return
        
        # Calculate Total_Spent
        self.df['Total_Spent'] = self.df[existing_mnt_cols].sum(axis=1)
        
        # Stats
        stats = {
            'min': self.df['Total_Spent'].min(),
            'max': self.df['Total_Spent'].max(),
            'mean': self.df['Total_Spent'].mean(),
            'median': self.df['Total_Spent'].median(),
            'std': self.df['Total_Spent'].std(),
            'skewness': self.df['Total_Spent'].skew()
        }
        
        print(f"Created 'Total_Spent' feature")
        print(f"   Formula: Sum of [{', '.join(existing_mnt_cols)}]")
        print(f"   Range: [{stats['min']:,.0f}, {stats['max']:,.0f}]")
        print(f"   Mean: {stats['mean']:,.0f}, Median: {stats['median']:,.0f}")
        print(f"   Std: {stats['std']:,.0f}, Skewness: {stats['skewness']:.3f}")
        print()
        
        # Check for zeros
        zero_count = (self.df['Total_Spent'] == 0).sum()
        if zero_count > 0:
            print(f"   WARNING: {zero_count} customers with Total_Spent = 0")
            print(f"   These will be handled in ratio calculations (avoid division by zero)")
            print()
        
        self.feature_stats['Total_Spent'] = stats
        self.log_action("Create Total_Spent", f"Mean: {stats['mean']:,.0f}, Skew: {stats['skewness']:.3f}")
    
    # ============================================================================
    # STEP 3: CREATE TOTALPURCHASES
    # ============================================================================
    
    def create_total_purchases(self):
        """Tạo TotalPurchases từ sum(Num*Purchases)."""
        self._print_header("STEP 2: CREATE TOTALPURCHASES FEATURE")
        
        # Define Num*Purchases columns
        purchase_cols = ['NumWebPurchases', 'NumCatalogPurchases', 
                         'NumStorePurchases', 'NumDealsPurchases']
        
        # Check existing columns
        existing_purchase_cols = [col for col in purchase_cols if col in self.df.columns]
        
        if not existing_purchase_cols:
            print("No Num*Purchases columns found. Skipping.")
            return
        
        # Calculate TotalPurchases
        self.df['TotalPurchases'] = self.df[existing_purchase_cols].sum(axis=1)
        
        # Stats
        stats = {
            'min': self.df['TotalPurchases'].min(),
            'max': self.df['TotalPurchases'].max(),
            'mean': self.df['TotalPurchases'].mean(),
            'median': self.df['TotalPurchases'].median(),
            'std': self.df['TotalPurchases'].std(),
            'skewness': self.df['TotalPurchases'].skew()
        }
        
        print(f"Created 'TotalPurchases' feature")
        print(f"   Formula: Sum of [{', '.join(existing_purchase_cols)}]")
        print(f"   Range: [{stats['min']:.0f}, {stats['max']:.0f}]")
        print(f"   Mean: {stats['mean']:.1f}, Median: {stats['median']:.1f}")
        print(f"   Std: {stats['std']:.1f}, Skewness: {stats['skewness']:.3f}")
        print()
        
        # Check for zeros
        zero_count = (self.df['TotalPurchases'] == 0).sum()
        if zero_count > 0:
            print(f"   WARNING: {zero_count} customers with TotalPurchases = 0")
            print(f"   These will be handled in channel ratio calculations")
            print()
        
        self.feature_stats['TotalPurchases'] = stats
        self.log_action("Create TotalPurchases", f"Mean: {stats['mean']:.1f}, Skew: {stats['skewness']:.3f}")
    
    # ============================================================================
    # STEP 4: CREATE PRODUCT RATIOS
    # ============================================================================
    
    def create_product_ratios(self):
        """Tạo 6 Product Preference Ratios."""
        self._print_header("STEP 3: CREATE PRODUCT PREFERENCE RATIOS")
        
        if 'Total_Spent' not in self.df.columns:
            print("Total_Spent not found. Skipping product ratios.")
            return
        
        # Define product categories
        product_mapping = {
            'MntWines': 'Wine_Ratio',
            'MntMeatProducts': 'Meat_Ratio',
            'MntFishProducts': 'Fish_Ratio',
            'MntFruits': 'Fruit_Ratio',
            'MntSweetProducts': 'Sweet_Ratio',
            'MntGoldProds': 'Gold_Ratio'
        }
        
        print("Creating product preference ratios:")
        print()
        
        created_ratios = []
        
        for mnt_col, ratio_name in product_mapping.items():
            if mnt_col not in self.df.columns:
                print(f"   {mnt_col} not found. Skipping {ratio_name}.")
                continue
            
            # Calculate ratio (avoid division by zero)
            self.df[ratio_name] = self.df[mnt_col] / self.df['Total_Spent'].replace(0, np.nan)
            
            # Fill NaN with 0 (customers with Total_Spent = 0)
            self.df[ratio_name] = self.df[ratio_name].fillna(0)
            
            # Stats
            mean_ratio = self.df[ratio_name].mean()
            median_ratio = self.df[ratio_name].median()
            max_ratio = self.df[ratio_name].max()
            
            print(f"   {ratio_name:<15} = {mnt_col:20} / Total_Spent")
            print(f"      Mean: {mean_ratio:>6.3f}, Median: {median_ratio:>6.3f}, Max: {max_ratio:>6.3f}")
            
            self.feature_stats[ratio_name] = {
                'mean': mean_ratio,
                'median': median_ratio,
                'max': max_ratio
            }
            
            created_ratios.append(ratio_name)
        
        print()
        print(f"Created {len(created_ratios)} product ratios")
        print()
        
        self.log_action("Create Product Ratios", f"{len(created_ratios)} ratios created")
    
    # ============================================================================
    # STEP 5: CREATE CHANNEL RATIOS
    # ============================================================================
    
    def create_channel_ratios(self):
        """Tạo 4 Channel Preference Ratios."""
        self._print_header("STEP 4: CREATE CHANNEL PREFERENCE RATIOS")
        
        if 'TotalPurchases' not in self.df.columns:
            print("TotalPurchases not found. Skipping channel ratios.")
            return
        
        # Define channel mapping
        channel_mapping = {
            'NumWebPurchases': 'Web_Ratio',
            'NumCatalogPurchases': 'Catalog_Ratio',
            'NumStorePurchases': 'Store_Ratio',
            'NumDealsPurchases': 'Deal_Ratio'
        }
        
        print("Creating channel preference ratios:")
        print()
        
        created_ratios = []
        
        for num_col, ratio_name in channel_mapping.items():
            if num_col not in self.df.columns:
                print(f"   {num_col} not found. Skipping {ratio_name}.")
                continue
            
            # Calculate ratio (avoid division by zero)
            self.df[ratio_name] = self.df[num_col] / self.df['TotalPurchases'].replace(0, np.nan)
            
            # Fill NaN with 0 (customers with TotalPurchases = 0)
            self.df[ratio_name] = self.df[ratio_name].fillna(0)
            
            # Stats
            mean_ratio = self.df[ratio_name].mean()
            median_ratio = self.df[ratio_name].median()
            max_ratio = self.df[ratio_name].max()
            
            print(f"   {ratio_name:<15} = {num_col:20} / TotalPurchases")
            print(f"      Mean: {mean_ratio:>6.3f}, Median: {median_ratio:>6.3f}, Max: {max_ratio:>6.3f}")
            
            self.feature_stats[ratio_name] = {
                'mean': mean_ratio,
                'median': median_ratio,
                'max': max_ratio
            }
            
            created_ratios.append(ratio_name)
        
        print()
        print(f"Created {len(created_ratios)} channel ratios")
        print()
        
        self.log_action("Create Channel Ratios", f"{len(created_ratios)} ratios created")
    
    # ============================================================================
    # STEP 6: SELECT FINAL FEATURES
    # ============================================================================
    
    def select_final_features(self):
        """Select 12 final features for product+channel clustering."""
        self._print_header("STEP 5: SELECT FINAL FEATURES")
        
        # Define final features
        final_features = [
            # Product ratios (6 đặc trưng)
            'Wine_Ratio',
            'Meat_Ratio',
            'Fish_Ratio',
            'Fruit_Ratio',
            'Sweet_Ratio',
            'Gold_Ratio',
            # Channel ratios (4 đặc trưng)
            'Web_Ratio',
            'Catalog_Ratio',
            'Store_Ratio',
            'Deal_Ratio',
            # Context (2 đặc trưng)
            'TotalPurchases',
            'NumWebVisitsMonth'
        ]
        
        # Filter existing features
        existing_features = [f for f in final_features if f in self.df.columns]
        
        # Create final dataset
        self.df_engineered = self.df[existing_features].copy()
        
        print(f"Selected {len(existing_features)} features for product+channel clustering:")
        print()
        
        # Group by type
        product_ratios = [f for f in existing_features if 'Wine' in f or 'Meat' in f or 'Fish' in f or 'Fruit' in f or 'Sweet' in f or 'Gold' in f]
        channel_ratios = [f for f in existing_features if 'Web_Ratio' in f or 'Catalog' in f or 'Store' in f or 'Deal' in f]
        context = [f for f in existing_features if f in ['TotalPurchases', 'NumWebVisitsMonth']]
        
        print(f"PRODUCT RATIOS ({len(product_ratios)} features):")
        for i, feat in enumerate(product_ratios, 1):
            dtype = str(self.df_engineered[feat].dtype)
            n_unique = self.df_engineered[feat].nunique()
            mean_val = self.df_engineered[feat].mean()
            print(f"   {i}. {feat:<20} ({dtype}, {n_unique:>4} unique, mean={mean_val:.3f})")
        
        print()
        print(f"CHANNEL RATIOS ({len(channel_ratios)} features):")
        for i, feat in enumerate(channel_ratios, 1):
            dtype = str(self.df_engineered[feat].dtype)
            n_unique = self.df_engineered[feat].nunique()
            mean_val = self.df_engineered[feat].mean()
            print(f"   {i}. {feat:<20} ({dtype}, {n_unique:>4} unique, mean={mean_val:.3f})")
        
        print()
        print(f"CONTEXT FEATURES ({len(context)} features):")
        for i, feat in enumerate(context, 1):
            dtype = str(self.df_engineered[feat].dtype)
            n_unique = self.df_engineered[feat].nunique()
            mean_val = self.df_engineered[feat].mean()
            print(f"   {i}. {feat:<20} ({dtype}, {n_unique:>4} unique, mean={mean_val:.1f})")
        
        print()
        print(f"Final dataset shape: {self.df_engineered.shape}")
        print()
        
        self.log_action("Select final features", f"{len(existing_features)} features selected")
    
    # ============================================================================
    # STEP 7: VALIDATE & EXPORT
    # ============================================================================
    
    def validate_features(self):
        """Validate engineered features."""
        self._print_header("STEP 6: VALIDATE FEATURES")
        
        print("Data Quality Checks:")
        print()
        
        # Missing values
        missing = self.df_engineered.isnull().sum()
        if missing.sum() == 0:
            print("   Missing values: None (PASS)")
        else:
            print(f"   Missing values: {missing.sum()} (WARNING)")
            for col in missing[missing > 0].index:
                print(f"      {col}: {missing[col]}")
        
        # Duplicates
        dup_count = self.df_engineered.duplicated().sum()
        print(f"   Duplicate rows: {dup_count} ({dup_count/len(self.df_engineered)*100:.2f}%)")
        
        # Variance
        low_var_features = []
        for col in self.df_engineered.columns:
            var = self.df_engineered[col].var()
            if var < 0.001:  # Lower threshold for ratios
                low_var_features.append(col)
        
        if low_var_features:
            print(f"   Low variance features: {len(low_var_features)} (WARNING)")
            for feat in low_var_features:
                print(f"      {feat}: var={self.df_engineered[feat].var():.6f}")
        else:
            print(f"   Low variance features: None (PASS)")
        
        # Range check for ratios (should be [0, 1])
        print()
        print("   Ratio range validation:")
        ratio_cols = [col for col in self.df_engineered.columns if '_Ratio' in col]
        for col in ratio_cols:
            min_val = self.df_engineered[col].min()
            max_val = self.df_engineered[col].max()
            if min_val < 0 or max_val > 1:
                print(f"      {col}: [{min_val:.3f}, {max_val:.3f}] (WARNING: outside [0,1])")
            else:
                print(f"      {col}: [{min_val:.3f}, {max_val:.3f}] (PASS)")
        
        print()
        self.log_action("Validate features", f"Checks completed")
    
    def export_dataset(self):
        """Export engineered dataset to CSV."""
        self._print_header("STEP 7: EXPORT DATASET")
        
        try:
            self.df_engineered.to_csv(self.output_csv, index=False)
            
            print(f"Exported dataset successfully:")
            print(f"   Path: {self.output_csv}")
            print(f"   Shape: {self.df_engineered.shape}")
            print(f"   Size: {os.path.getsize(self.output_csv) / 1024:.2f} KB")
            print()
            
            self.log_action("Export dataset", self.output_csv)
            return True
            
        except Exception as e:
            print(f"Export failed: {e}")
            self.log_action("Export FAILED", str(e))
            return False
    
    def export_report(self):
        """Export engineering report."""
        self._print_header("STEP 8: EXPORT REPORT")
        
        try:
            with open(self.report_file, 'w', encoding='utf-8') as f:
                # Header
                f.write("=" * 100 + "\n")
                f.write("PRODUCT+CHANNEL FEATURE ENGINEERING REPORT\n")
                f.write("=" * 100 + "\n\n")
                
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Input: {self.input_path}\n")
                f.write(f"Output: {self.output_csv}\n\n")
                
                # Dataset info
                f.write("DATASET INFORMATION:\n")
                f.write("-" * 100 + "\n")
                f.write(f"Original dataset: {self.df.shape[0]:,} rows x {self.df.shape[1]} columns\n")
                f.write(f"Engineered dataset: {self.df_engineered.shape[0]:,} rows x {self.df_engineered.shape[1]} columns\n\n")
                
                # Features created
                f.write("FEATURES CREATED:\n")
                f.write("-" * 100 + "\n")
                for i, col in enumerate(self.df_engineered.columns, 1):
                    f.write(f"{i}. {col}\n")
                    if col in self.feature_stats:
                        stats = self.feature_stats[col]
                        if isinstance(stats, dict):
                            for key, val in stats.items():
                                f.write(f"   {key}: {val}\n")
                    f.write("\n")
                
                # Processing log
                f.write("\nPROCESSING LOG:\n")
                f.write("-" * 100 + "\n")
                for i, log in enumerate(self.processing_log, 1):
                    f.write(f"{i}. [{log['timestamp']}] {log['action']}\n")
                    if log['details']:
                        f.write(f"   Details: {log['details']}\n")
                
                f.write("\n" + "=" * 100 + "\n")
                f.write("Report generated successfully\n")
                f.write("=" * 100 + "\n")
            
            print(f"Exported report successfully:")
            print(f"   Path: {self.report_file}")
            print(f"   Size: {os.path.getsize(self.report_file) / 1024:.2f} KB")
            print()
            
            self.log_action("Export report", self.report_file)
            return True
            
        except Exception as e:
            print(f"Report export failed: {e}")
            return False
    
    # ============================================================================
    # MAIN PIPELINE
    # ============================================================================
    
    def run_engineering(self):
        """Run complete product+channel feature engineering pipeline."""
        print("\n" + "=" * 100)
        print("PRODUCT+CHANNEL FEATURE ENGINEERING PIPELINE".center(100))
        print("=" * 100 + "\n")
        
        start_time = datetime.now()
        
        # Step 1: Load data
        if not self.load_data():
            print("Pipeline failed at load_data()")
            return False
        
        # Step 2-3: Create base features
        self.create_total_spent()
        self.create_total_purchases()
        
        # Step 4-5: Create ratios
        self.create_product_ratios()
        self.create_channel_ratios()
        
        # Step 6: Select final features
        self.select_final_features()
        
        # Step 7: Validate
        self.validate_features()
        
        # Step 8: Export
        self.export_dataset()
        self.export_report()
        
        # Summary
        elapsed = (datetime.now() - start_time).total_seconds()
        
        print("=" * 100)
        print("PIPELINE COMPLETED".center(100))
        print("=" * 100)
        print(f"Time elapsed: {elapsed:.2f} seconds")
        print(f"Output dataset: {self.output_csv}")
        print(f"Output report: {self.report_file}")
        print("=" * 100 + "\n")
        
        return True


# ================================================================================
# CLASS 3: RFM FEATURE ENGINEERING
# ================================================================================

class RFMFeatureEngineering:
    """
    Feature Engineering cho RFM CLUSTERING (Customer Value Segmentation).
    
    MỤC TIÊU: Phân khúc khách hàng theo giá trị kinh tế (RFM Analysis)
    
    FEATURES TẠO RA (7 features):
    
    CORE RFM FEATURES (3 features):
    1. Recency                          (0-99 days, original)
    2. TotalPurchases                   (Frequency, derived)
    3. Total_Spent                      (Monetary, derived)
    
    DERIVED VALUE FEATURES (4 features):
    4. AvgPerPurchase                   (AOV = Total_Spent / TotalPurchases)
    5. AvgPerPurchase_Transformed       (Yeo-Johnson if skewed)
    6. Income                           (Financial capacity, original)
    7. Income_per_Family_Member_Transformed  (Financial pressure, derived from Class 1)
    
    Pipeline:
    Step 1   : Tải data
    Step 2   : Tạo TotalPurchases (sum Num*Purchases)
    Step 3   : Tạo Total_Spent (sum Mnt*)
    Step 4   : Tạo AvgPerPurchase (AOV)
    Step 5   : Transform AvgPerPurchase if skewed (Yeo-Johnson)
    Step 6   : Tạo Income_per_Family_Member (from Class 1 logic)
    Step 7   : Transform Income_per_Family_Member if skewed
    Step 8   : Chọn ra 7 đặc trưng cuối cùng (R, F, M + derived)
    Step 9   : Validate (missing, duplicates, variance)
    Step 10  : Xuất tệp CSV + Report
    
    INPUT:
    - CSV: Customer_Behavior_cleaned.csv

    OUTPUT:
    - CSV: Customer_Behavior_RFM.csv
    - Report: RFM_Engineering_Report.log
    """
    
    def __init__(self, input_path, output_dir, report_dir):
        """
        Khởi tạo RFM Feature Engineering.
        
        Args:
            input_path (str): Đường dẫn dataset cleaned
            output_dir (str): Thư mục lưu dataset output
            report_dir (str): Thư mục lưu report
        """
        self.input_path = input_path
        self.output_dir = output_dir
        self.report_dir = report_dir
        
        # Create directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(report_dir, exist_ok=True)
        
        # Output paths
        self.output_csv = os.path.join(output_dir, "Customer_Behavior_RFM.csv")
        self.report_file = os.path.join(report_dir, "RFM_Engineering_Report.log")
        
        # Data storage
        self.df = None
        self.df_engineered = None
        self.feature_stats = {}
        self.processing_log = []
    
    def log_action(self, action, details=""):
        """Ghi log hành động."""
        self.processing_log.append({
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'action': action,
            'details': details
        })
    
    def _print_header(self, title, width=100, char='='):
        """In header có format."""
        print(char * width)
        print(f"{title:^{width}}")
        print(char * width)
        print()
    
    def _print_subheader(self, title, width=100, char='-'):
        """In subheader."""
        print()
        print(char * width)
        print(title)
        print(char * width)
    
    # ============================================================================
    # STEP 1: LOAD DATA
    # ============================================================================
    
    def load_data(self):
        """Load cleaned dataset."""
        self._print_header("RFM FEATURE ENGINEERING - LOAD DATA")
        
        try:
            self.df = pd.read_csv(self.input_path)
            
            print(f"Loaded dataset successfully")
            print(f"   Shape: {self.df.shape[0]:,} rows x {self.df.shape[1]} columns")
            print(f"   Memory: {self.df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
            print()
            
            self.log_action("Load data", f"Shape: {self.df.shape}")
            
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            self.log_action("Load data FAILED", str(e))
            return False
    
    # ============================================================================
    # STEP 2: CREATE TOTALPURCHASES (FREQUENCY)
    # ============================================================================
    
    def create_total_purchases(self):
        """Tạo TotalPurchases (Frequency in RFM)."""
        self._print_header("STEP 1: CREATE TOTALPURCHASES (FREQUENCY)")
        
        # Define purchase columns
        purchase_cols = ['NumWebPurchases', 'NumCatalogPurchases', 
                         'NumStorePurchases', 'NumDealsPurchases']
        
        # Check existing columns
        existing_cols = [col for col in purchase_cols if col in self.df.columns]
        
        if not existing_cols:
            print("No Num*Purchases columns found. Skipping.")
            return
        
        # Calculate TotalPurchases
        self.df['TotalPurchases'] = self.df[existing_cols].sum(axis=1)
        
        # Stats
        stats = {
            'min': self.df['TotalPurchases'].min(),
            'max': self.df['TotalPurchases'].max(),
            'mean': self.df['TotalPurchases'].mean(),
            'median': self.df['TotalPurchases'].median(),
            'std': self.df['TotalPurchases'].std(),
            'skewness': self.df['TotalPurchases'].skew()
        }
        
        print(f"Created 'TotalPurchases' (Frequency)")
        print(f"   Formula: Sum of [{', '.join(existing_cols)}]")
        print(f"   Range: [{stats['min']:.0f}, {stats['max']:.0f}]")
        print(f"   Mean: {stats['mean']:.1f}, Median: {stats['median']:.1f}")
        print(f"   Std: {stats['std']:.1f}, Skewness: {stats['skewness']:.3f}")
        print()
        
        # Check distribution
        zero_count = (self.df['TotalPurchases'] == 0).sum()
        if zero_count > 0:
            print(f"   WARNING: {zero_count} customers with TotalPurchases = 0")
            print(f"   These customers may be inactive or new")
            print()
        
        self.feature_stats['TotalPurchases'] = stats
        self.log_action("Create TotalPurchases", f"Mean: {stats['mean']:.1f}, Skew: {stats['skewness']:.3f}")
    
    # ============================================================================
    # STEP 3: CREATE TOTAL_SPENT (MONETARY)
    # ============================================================================
    
    def create_total_spent(self):
        """Tạo Total_Spent (Monetary in RFM)."""
        self._print_header("STEP 2: CREATE TOTAL_SPENT (MONETARY)")
        
        # Define spending columns
        mnt_cols = ['MntWines', 'MntFruits', 'MntMeatProducts', 
                    'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
        
        # Check existing columns
        existing_cols = [col for col in mnt_cols if col in self.df.columns]
        
        if not existing_cols:
            print("No Mnt* columns found. Skipping.")
            return
        
        # Calculate Total_Spent
        self.df['Total_Spent'] = self.df[existing_cols].sum(axis=1)
        
        # Stats
        stats = {
            'min': self.df['Total_Spent'].min(),
            'max': self.df['Total_Spent'].max(),
            'mean': self.df['Total_Spent'].mean(),
            'median': self.df['Total_Spent'].median(),
            'std': self.df['Total_Spent'].std(),
            'skewness': self.df['Total_Spent'].skew()
        }
        
        print(f"Created 'Total_Spent' (Monetary)")
        print(f"   Formula: Sum of [{', '.join(existing_cols)}]")
        print(f"   Range: [{stats['min']:,.0f}, {stats['max']:,.0f}]")
        print(f"   Mean: {stats['mean']:,.0f}, Median: {stats['median']:,.0f}")
        print(f"   Std: {stats['std']:,.0f}, Skewness: {stats['skewness']:.3f}")
        print()
        
        # Check for zeros
        zero_count = (self.df['Total_Spent'] == 0).sum()
        if zero_count > 0:
            print(f"   WARNING: {zero_count} customers with Total_Spent = 0")
            print(f"   These customers have no purchase history")
            print()
        
        self.feature_stats['Total_Spent'] = stats
        self.log_action("Create Total_Spent", f"Mean: {stats['mean']:,.0f}, Skew: {stats['skewness']:.3f}")
    
    # ============================================================================
    # STEP 4: CREATE AVGPERPURCHASE (AVERAGE ORDER VALUE)
    # ============================================================================
    
    def create_avg_per_purchase(self):
        """Tạo AvgPerPurchase (Average Order Value)."""
        self._print_header("STEP 3: CREATE AVGPERPURCHASE (AOV)")
        
        if 'Total_Spent' not in self.df.columns or 'TotalPurchases' not in self.df.columns:
            print("Total_Spent or TotalPurchases not found. Skipping.")
            return
        
        # Calculate AOV (avoid division by zero)
        self.df['AvgPerPurchase'] = self.df['Total_Spent'] / self.df['TotalPurchases'].replace(0, np.nan)
        
        # Fill NaN with 0 (customers with TotalPurchases = 0)
        self.df['AvgPerPurchase'] = self.df['AvgPerPurchase'].fillna(0)
        
        # Stats
        stats = {
            'min': self.df['AvgPerPurchase'].min(),
            'max': self.df['AvgPerPurchase'].max(),
            'mean': self.df['AvgPerPurchase'].mean(),
            'median': self.df['AvgPerPurchase'].median(),
            'std': self.df['AvgPerPurchase'].std(),
            'skewness': self.df['AvgPerPurchase'].skew()
        }
        
        print(f"Created 'AvgPerPurchase' (Average Order Value)")
        print(f"   Formula: Total_Spent / TotalPurchases")
        print(f"   Range: [{stats['min']:,.2f}, {stats['max']:,.2f}]")
        print(f"   Mean: {stats['mean']:,.2f}, Median: {stats['median']:,.2f}")
        print(f"   Std: {stats['std']:,.2f}, Skewness: {stats['skewness']:.3f}")
        print()
        
        self.feature_stats['AvgPerPurchase'] = stats
        self.log_action("Create AvgPerPurchase", f"Mean: {stats['mean']:,.2f}, Skew: {stats['skewness']:.3f}")
    
    # ============================================================================
    # STEP 5: TRANSFORM AVGPERPURCHASE IF SKEWED
    # ============================================================================
    
    def transform_avg_per_purchase(self):
        """Apply Yeo-Johnson transform to AvgPerPurchase if skewed."""
        self._print_header("STEP 4: TRANSFORM AVGPERPURCHASE IF SKEWED")
        
        SKEW_THRESHOLD = 1.0
        
        if 'AvgPerPurchase' not in self.df.columns:
            print("AvgPerPurchase not found. Skipping.")
            return
        
        original_skew = self.df['AvgPerPurchase'].skew()
        
        print(f"Checking skewness of AvgPerPurchase:")
        print(f"   Original skewness: {original_skew:.3f}")
        
        if abs(original_skew) < SKEW_THRESHOLD:
            print(f"   Skewness < {SKEW_THRESHOLD} -> No transform needed")
            self.df['AvgPerPurchase_Transformed'] = self.df['AvgPerPurchase']
            self.log_action("Transform AvgPerPurchase", "No transform needed (skew < 1.0)")
            return
        
        # Apply Yeo-Johnson transform
        try:
            transformed_data, lambda_param = yeojohnson(self.df['AvgPerPurchase'])
            self.df['AvgPerPurchase_Transformed'] = transformed_data
            
            new_skew = pd.Series(transformed_data).skew()
            
            print(f"\nApplied Yeo-Johnson transform:")
            print(f"   Lambda parameter: {lambda_param:.3f}")
            print(f"   New skewness: {new_skew:.3f}")
            print(f"   Improvement: {abs(original_skew) - abs(new_skew):.3f}")
            print()
            
            self.feature_stats['AvgPerPurchase_Transformed'] = {
                'original_skew': original_skew,
                'new_skew': new_skew,
                'lambda': lambda_param
            }
            
            self.log_action("Transform AvgPerPurchase", 
                           f"Skew: {original_skew:.3f} -> {new_skew:.3f}")
            
        except Exception as e:
            print(f"Transform failed: {e}")
            self.df['AvgPerPurchase_Transformed'] = self.df['AvgPerPurchase']
            self.log_action("Transform AvgPerPurchase FAILED", str(e))
    
    # ============================================================================
    # STEP 6: CREATE INCOME_PER_FAMILY_MEMBER (REUSE CLASS 1 LOGIC)
    # ============================================================================
    
    def create_income_per_family_member(self):
        """Tạo Income_per_Family_Member (Financial pressure indicator)."""
        self._print_header("STEP 5: CREATE INCOME_PER_FAMILY_MEMBER")
        
        # Check required columns
        if 'Income' not in self.df.columns:
            print("Income not found. Skipping.")
            return
        
        # Create TotalChildren if not exists
        if 'TotalChildren' not in self.df.columns:
            if 'Kidhome' in self.df.columns and 'Teenhome' in self.df.columns:
                self.df['TotalChildren'] = self.df['Kidhome'] + self.df['Teenhome']
                print("Created TotalChildren (Kidhome + Teenhome)")
            else:
                print("Kidhome or Teenhome not found. Cannot create Income_per_Family_Member.")
                return
        
        # Calculate Income_per_Family_Member
        family_size = 2 + self.df['TotalChildren']
        self.df['Income_per_Family_Member'] = self.df['Income'] / family_size
        
        # Stats
        stats = {
            'min': self.df['Income_per_Family_Member'].min(),
            'max': self.df['Income_per_Family_Member'].max(),
            'mean': self.df['Income_per_Family_Member'].mean(),
            'median': self.df['Income_per_Family_Member'].median(),
            'std': self.df['Income_per_Family_Member'].std(),
            'skewness': self.df['Income_per_Family_Member'].skew()
        }
        
        print(f"Created 'Income_per_Family_Member'")
        print(f"   Formula: Income / (2 + TotalChildren)")
        print(f"   Range: [{stats['min']:,.0f}, {stats['max']:,.0f}]")
        print(f"   Mean: {stats['mean']:,.0f}, Median: {stats['median']:,.0f}")
        print(f"   Skewness: {stats['skewness']:.3f}")
        print()
        
        self.feature_stats['Income_per_Family_Member'] = stats
        self.log_action("Create Income_per_Family_Member", 
                       f"Mean: {stats['mean']:,.0f}, Skew: {stats['skewness']:.3f}")
    
    # ============================================================================
    # STEP 7: TRANSFORM INCOME_PER_FAMILY_MEMBER IF SKEWED
    # ============================================================================
    
    def transform_income_per_family_member(self):
        """Apply Yeo-Johnson transform to Income_per_Family_Member if skewed."""
        self._print_header("STEP 6: TRANSFORM INCOME_PER_FAMILY_MEMBER IF SKEWED")
        
        SKEW_THRESHOLD = 1.0
        
        if 'Income_per_Family_Member' not in self.df.columns:
            print("Income_per_Family_Member not found. Skipping.")
            return
        
        original_skew = self.df['Income_per_Family_Member'].skew()
        
        print(f"Checking skewness of Income_per_Family_Member:")
        print(f"   Original skewness: {original_skew:.3f}")
        
        if abs(original_skew) < SKEW_THRESHOLD:
            print(f"   Skewness < {SKEW_THRESHOLD} -> No transform needed")
            self.df['Income_per_Family_Member_Transformed'] = self.df['Income_per_Family_Member']
            return
        
        # Apply Yeo-Johnson transform
        try:
            transformed_data, lambda_param = yeojohnson(self.df['Income_per_Family_Member'])
            self.df['Income_per_Family_Member_Transformed'] = transformed_data
            
            new_skew = pd.Series(transformed_data).skew()
            
            print(f"\nApplied Yeo-Johnson transform:")
            print(f"   Lambda parameter: {lambda_param:.3f}")
            print(f"   New skewness: {new_skew:.3f}")
            print(f"   Improvement: {abs(original_skew) - abs(new_skew):.3f}")
            print()
            
            self.feature_stats['Income_per_Family_Member_Transformed'] = {
                'original_skew': original_skew,
                'new_skew': new_skew,
                'lambda': lambda_param
            }
            
            self.log_action("Transform Income_per_Family_Member", 
                           f"Skew: {original_skew:.3f} -> {new_skew:.3f}")
            
        except Exception as e:
            print(f"Transform failed: {e}")
            self.df['Income_per_Family_Member_Transformed'] = self.df['Income_per_Family_Member']
    
    # ============================================================================
    # STEP 8: SELECT FINAL FEATURES
    # ============================================================================
    
    def select_final_features(self):
        """Select 7 final features for RFM clustering."""
        self._print_header("STEP 7: SELECT FINAL FEATURES")
        
        # Define final features
        final_features = [
            # Core RFM (3)
            'Recency',
            'TotalPurchases',
            'Total_Spent',
            # Value features (4)
            'AvgPerPurchase_Transformed',
            'Income',
            'Income_per_Family_Member_Transformed'
        ]
        
        # Filter existing features
        existing_features = [f for f in final_features if f in self.df.columns]
        
        # Create final dataset
        self.df_engineered = self.df[existing_features].copy()
        
        print(f"Selected {len(existing_features)} features for RFM clustering:")
        print()
        
        # Group by type
        core_rfm = [f for f in existing_features if f in ['Recency', 'TotalPurchases', 'Total_Spent']]
        value_features = [f for f in existing_features if f not in core_rfm]
        
        print(f"CORE RFM FEATURES ({len(core_rfm)} features):")
        for i, feat in enumerate(core_rfm, 1):
            dtype = str(self.df_engineered[feat].dtype)
            n_unique = self.df_engineered[feat].nunique()
            mean_val = self.df_engineered[feat].mean()
            print(f"   {i}. {feat:<35} ({dtype}, {n_unique:>4} unique, mean={mean_val:,.1f})")
        
        print()
        print(f"VALUE FEATURES ({len(value_features)} features):")
        for i, feat in enumerate(value_features, 1):
            dtype = str(self.df_engineered[feat].dtype)
            n_unique = self.df_engineered[feat].nunique()
            mean_val = self.df_engineered[feat].mean()
            print(f"   {i}. {feat:<35} ({dtype}, {n_unique:>4} unique, mean={mean_val:,.1f})")
        
        print()
        print(f"Final dataset shape: {self.df_engineered.shape}")
        print()
        
        self.log_action("Select final features", f"{len(existing_features)} features selected")
    
    # ============================================================================
    # STEP 9: VALIDATE & EXPORT
    # ============================================================================
    
    def validate_features(self):
        """Validate engineered features."""
        self._print_header("STEP 8: VALIDATE FEATURES")
        
        print("Data Quality Checks:")
        print()
        
        # Missing values
        missing = self.df_engineered.isnull().sum()
        if missing.sum() == 0:
            print("   Missing values: None (PASS)")
        else:
            print(f"   Missing values: {missing.sum()} (WARNING)")
            for col in missing[missing > 0].index:
                print(f"      {col}: {missing[col]}")
        
        # Duplicates
        dup_count = self.df_engineered.duplicated().sum()
        print(f"   Duplicate rows: {dup_count} ({dup_count/len(self.df_engineered)*100:.2f}%)")
        
        # Variance
        low_var_features = []
        for col in self.df_engineered.columns:
            var = self.df_engineered[col].var()
            if var < 0.01:
                low_var_features.append(col)
        
        if low_var_features:
            print(f"   Low variance features: {len(low_var_features)} (WARNING)")
            for feat in low_var_features:
                print(f"      {feat}: var={self.df_engineered[feat].var():.6f}")
        else:
            print(f"   Low variance features: None (PASS)")
        
        print()
        self.log_action("Validate features", f"Checks completed")
    
    def export_dataset(self):
        """Export engineered dataset to CSV."""
        self._print_header("STEP 9: EXPORT DATASET")
        
        try:
            self.df_engineered.to_csv(self.output_csv, index=False)
            
            print(f"Exported dataset successfully:")
            print(f"   Path: {self.output_csv}")
            print(f"   Shape: {self.df_engineered.shape}")
            print(f"   Size: {os.path.getsize(self.output_csv) / 1024:.2f} KB")
            print()
            
            self.log_action("Export dataset", self.output_csv)
            return True
            
        except Exception as e:
            print(f"Export failed: {e}")
            self.log_action("Export FAILED", str(e))
            return False
    
    def export_report(self):
        """Export engineering report."""
        self._print_header("STEP 10: EXPORT REPORT")
        
        try:
            with open(self.report_file, 'w', encoding='utf-8') as f:
                # Header
                f.write("=" * 100 + "\n")
                f.write("RFM FEATURE ENGINEERING REPORT\n")
                f.write("=" * 100 + "\n\n")
                
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Input: {self.input_path}\n")
                f.write(f"Output: {self.output_csv}\n\n")
                
                # Dataset info
                f.write("DATASET INFORMATION:\n")
                f.write("-" * 100 + "\n")
                f.write(f"Original dataset: {self.df.shape[0]:,} rows x {self.df.shape[1]} columns\n")
                f.write(f"Engineered dataset: {self.df_engineered.shape[0]:,} rows x {self.df_engineered.shape[1]} columns\n\n")
                
                # Features created
                f.write("FEATURES CREATED:\n")
                f.write("-" * 100 + "\n")
                for i, col in enumerate(self.df_engineered.columns, 1):
                    f.write(f"{i}. {col}\n")
                    if col in self.feature_stats:
                        stats = self.feature_stats[col]
                        if isinstance(stats, dict):
                            for key, val in stats.items():
                                f.write(f"   {key}: {val}\n")
                    f.write("\n")
                
                # Processing log
                f.write("\nPROCESSING LOG:\n")
                f.write("-" * 100 + "\n")
                for i, log in enumerate(self.processing_log, 1):
                    f.write(f"{i}. [{log['timestamp']}] {log['action']}\n")
                    if log['details']:
                        f.write(f"   Details: {log['details']}\n")
                
                f.write("\n" + "=" * 100 + "\n")
                f.write("Report generated successfully\n")
                f.write("=" * 100 + "\n")
            
            print(f"Exported report successfully:")
            print(f"   Path: {self.report_file}")
            print(f"   Size: {os.path.getsize(self.report_file) / 1024:.2f} KB")
            print()
            
            self.log_action("Export report", self.report_file)
            return True
            
        except Exception as e:
            print(f"Report export failed: {e}")
            return False
    
    # ============================================================================
    # MAIN PIPELINE
    # ============================================================================
    
    def run_engineering(self):
        """Run complete RFM feature engineering pipeline."""
        print("\n" + "=" * 100)
        print("RFM FEATURE ENGINEERING PIPELINE".center(100))
        print("=" * 100 + "\n")
        
        start_time = datetime.now()
        
        # Step 1: Load data
        if not self.load_data():
            print("Pipeline failed at load_data()")
            return False
        
        # Step 2-3: Create core RFM features
        self.create_total_purchases()
        self.create_total_spent()
        
        # Step 4-5: Create and transform AOV
        self.create_avg_per_purchase()
        self.transform_avg_per_purchase()
        
        # Step 6-7: Create and transform Income_per_Family_Member
        self.create_income_per_family_member()
        self.transform_income_per_family_member()
        
        # Step 8: Select final features
        self.select_final_features()
        
        # Step 9: Validate
        self.validate_features()
        
        # Step 10: Export
        self.export_dataset()
        self.export_report()
        
        # Summary
        elapsed = (datetime.now() - start_time).total_seconds()
        
        print("=" * 100)
        print("PIPELINE COMPLETED".center(100))
        print("=" * 100)
        print(f"Time elapsed: {elapsed:.2f} seconds")
        print(f"Output dataset: {self.output_csv}")
        print(f"Output report: {self.report_file}")
        print("=" * 100 + "\n")
        
        return True

# ================================================================================
# MAIN EXECUTION
# ================================================================================

def main():
    """Main function - Run all feature engineering classes."""
    # Paths
    input_path = r"C:\Project\Machine_Learning\Machine_Learning\dataset\Customer_Behavior_cleaned.csv"
    output_dir = r"C:\Project\Machine_Learning\Machine_Learning\dataset"
    report_dir = r"C:\Project\Machine_Learning\Machine_Learning\report\Feature Extraction & Engineering_report"
    
    print("\n" + "=" * 100)
    print("MULTI-OBJECTIVE FEATURE ENGINEERING".center(100))
    print("=" * 100 + "\n")
    
    # ========================================
    # CLASS 1: DEMOGRAPHIC
    # ========================================
    print("\n" + "=" * 100)
    print("CLASS 1: DEMOGRAPHIC CLUSTERING - FEATURE ENGINEERING".center(100))
    print("=" * 100 + "\n")
    
    demographic_fe = DemographicFeatureEngineering(input_path, output_dir, report_dir)
    demo_success = demographic_fe.run_engineering()
    
    if demo_success:
        print("\nClass 1 (Demographic) completed successfully")
    else:
        print("\nClass 1 (Demographic) failed")
    
    # ========================================
    # CLASS 2: PRODUCT+CHANNEL
    # ========================================
    print("\n" + "=" * 100)
    print("CLASS 2: PRODUCT+CHANNEL CLUSTERING - FEATURE ENGINEERING".center(100))
    print("=" * 100 + "\n")
    
    productchannel_fe = ProductChannelFeatureEngineering(input_path, output_dir, report_dir)
    pc_success = productchannel_fe.run_engineering()
    
    if pc_success:
        print("\nClass 2 (Product+Channel) completed successfully")
    else:
        print("\nClass 2 (Product+Channel) failed")
    
    # ========================================
    # CLASS 3: RFM
    # ========================================
    print("\n" + "=" * 100)
    print("CLASS 3: RFM CLUSTERING - FEATURE ENGINEERING".center(100))
    print("=" * 100 + "\n")
    
    rfm_fe = RFMFeatureEngineering(input_path, output_dir, report_dir)
    rfm_success = rfm_fe.run_engineering()
    
    if rfm_success:
        print("\nClass 3 (RFM) completed successfully")
    else:
        print("\nClass 3 (RFM) failed")
    
    # ========================================
    # FINAL SUMMARY
    # ========================================
    print("\n" + "=" * 100)
    print("FINAL SUMMARY".center(100))
    print("=" * 100)
    print(f"Class 1 (Demographic)    : {'SUCCESS' if demo_success else 'FAILED'}")
    print(f"Class 2 (Product+Channel): {'SUCCESS' if pc_success else 'FAILED'}")
    print(f"Class 3 (RFM)            : {'SUCCESS' if rfm_success else 'FAILED'}")
    print()
    
    total_success = sum([demo_success, pc_success, rfm_success])
    print(f"TOTAL: {total_success}/3 classes completed successfully")
    
    if total_success == 3:
        print()
        print("ALL FEATURE ENGINEERING COMPLETED")
        print("Ready for next stage: Feature Scaling & K-Means Clustering")
    
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()