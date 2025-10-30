"""
================================================================================
FEATURE SCALING FOR MULTI-OBJECTIVE CUSTOMER CLUSTERING
================================================================================

Module: Feature Scaling cho 3 chiến lược clustering
- Class 1: DemographicScaler (Life Stage Segmentation)
- Class 2: ProductChannelScaler (Shopping Behavior Segmentation)
- Class 3: RFMScaler (Customer Value Segmentation)

Scaling Method: StandardScaler (Z-score normalization)
Additional option: RobustScaler (median / IQR) to reduce outlier impact

Input:  3 engineered datasets (CSV)
Output: 3 scaled datasets (CSV) (+ optional robust-scaled CSVs) + 3 reports (LOG format)

================================================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler, RobustScaler
import os
import warnings
import joblib

warnings.filterwarnings('ignore')


# ================================================================================
# CLASS 1: DEMOGRAPHIC SCALER
# ================================================================================

class DemographicScaler:
    """
    Feature Scaling cho Phân Cụm Nhân Khẩu Học (Demographic Clustering)

    New optional args:
      - use_robust (bool): nếu True sẽ tạo thêm dataset sử dụng RobustScaler
      - robust_columns (list[str] | None): danh sách cột áp dụng RobustScaler (None -> tất cả numeric)

    Outputs when use_robust=True:
      - Customer_Behavior_Demographic_robust_scaled.csv
      - demographic_robust_scaler.pkl
    """
    
    def __init__(self, input_path, output_dir, report_dir, use_robust=False, robust_columns=None):
        """
        Khởi tạo Demographic Scaler.
        """
        self.input_path = input_path
        self.output_dir = output_dir
        self.report_dir = report_dir
        self.use_robust = use_robust
        self.robust_columns = robust_columns
        
        # Create directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(report_dir, exist_ok=True)
        
        # Output paths
        self.output_csv = os.path.join(output_dir, "Customer_Behavior_Demographic_scaled.csv")
        self.report_file = os.path.join(report_dir, "Demographic_Scaling_Report.log")
        self.scaler_file = os.path.join(output_dir, "demographic_scaler.pkl")
        # Robust outputs
        self.robust_output_csv = os.path.join(output_dir, "Customer_Behavior_Demographic_robust_scaled.csv")
        self.robust_scaler_file = os.path.join(output_dir, "demographic_robust_scaler.pkl")
        
        # Data storage
        self.df = None
        self.df_scaled = None
        self.df_robust_scaled = None
        self.scaler = StandardScaler()
        self.robust_scaler = None
        self.pre_stats = {}
        self.post_stats = {}
        self.processing_log = []
    
    def log_action(self, action, details=""):
        """Ghi log hành động."""
        self.processing_log.append({
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'action': action,
            'details': details
        })
    
    def _print_header(self, title, width=100, char='='):
        print(char * width)
        print(f"{title:^{width}}")
        print(char * width)
        print()
    
    # STEP 1: LOAD DATA
    def load_data(self):
        self._print_header("DEMOGRAPHIC SCALING - LOAD DATA")
        try:
            self.df = pd.read_csv(self.input_path)
            print(f"Loaded dataset successfully")
            print(f"   Path: {self.input_path}")
            print(f"   Shape: {self.df.shape[0]:,} rows x {self.df.shape[1]} columns")
            print(f"   Memory: {self.df.memory_usage(deep=True).sum() / 1024:.2f} KB")
            print()
            print(f"Features loaded:")
            for i, col in enumerate(self.df.columns, 1):
                print(f"   {i}. {col:<40} ({self.df[col].dtype})")
            print()
            self.log_action("Load data", f"Shape: {self.df.shape}")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            self.log_action("Load data FAILED", str(e))
            return False
    
    # STEP 2: VALIDATE DATA
    def validate_data(self):
        self._print_header("STEP 1: VALIDATE DATA QUALITY")
        print("Pre-Scaling Data Quality Checks:")
        print()
        missing = self.df.isnull().sum()
        if missing.sum() == 0:
            print("   Missing values: None (PASS)")
        else:
            print("   Missing values detected:")
            for col in missing[missing > 0].index:
                print(f"      {col}: {missing[col]} ({missing[col]/len(self.df)*100:.2f}%)")
        inf_count = np.isinf(self.df.select_dtypes(include=[np.number])).sum().sum()
        if inf_count == 0:
            print("   Infinite values: None (PASS)")
        else:
            print(f"   Infinite values: {inf_count} (WARNING)")
        non_numeric = self.df.select_dtypes(exclude=[np.number]).columns.tolist()
        if not non_numeric:
            print("   Data types: All numeric (PASS)")
        else:
            print(f"   Non-numeric columns: {non_numeric} (WARNING)")
        print()
        self.log_action("Validate data", "Checks completed")
    
    # STEP 3: PRE STATS
    def compute_pre_stats(self):
        self._print_header("STEP 2: PRE-SCALING STATISTICS")
        print("Original feature statistics:")
        print()
        for col in self.df.columns:
            stats = {
                'mean': self.df[col].mean(),
                'std': self.df[col].std(),
                'min': self.df[col].min(),
                'max': self.df[col].max(),
                'median': self.df[col].median(),
                'skewness': self.df[col].skew()
            }
            self.pre_stats[col] = stats
            print(f"{col}:")
            print(f"   Mean: {stats['mean']:>12.3f}")
            print(f"   Std:  {stats['std']:>12.3f}")
            print(f"   Min:  {stats['min']:>12.3f}")
            print(f"   Max:  {stats['max']:>12.3f}")
            print(f"   Median: {stats['median']:>10.3f}")
            print(f"   Skewness: {stats['skewness']:>8.3f}")
            print()
        self.log_action("Compute pre-scaling stats", f"{len(self.df.columns)} features")
    
    # STEP 4: APPLY STANDARDSCALER
    def apply_scaling(self):
        self._print_header("STEP 3: APPLY STANDARDSCALER")
        print("Applying Z-score normalization...")
        print(f"   Formula: z = (x - mean) / std")
        print(f"   Target: mean=0, std=1")
        print()
        try:
            scaled_data = self.scaler.fit_transform(self.df)
            self.df_scaled = pd.DataFrame(scaled_data, columns=self.df.columns, index=self.df.index)
            print(f"Scaling completed successfully")
            print(f"   Scaled shape: {self.df_scaled.shape}")
            print()
            print("Scaler parameters (learned from data):")
            for i, col in enumerate(self.df.columns):
                print(f"   {col}:")
                print(f"      Mean (μ): {self.scaler.mean_[i]:>12.3f}")
                print(f"      Std (σ):  {self.scaler.scale_[i]:>12.3f}")
            print()
            self.log_action("Apply StandardScaler", "Success")
            return True
        except Exception as e:
            print(f"Scaling failed: {e}")
            self.log_action("Apply StandardScaler FAILED", str(e))
            return False
    
    # STEP 5: POST-SCALING STATS (keeps previous robust detection)
    def compute_post_stats(self):
        self._print_header("STEP 4: POST-SCALING STATISTICS")
        print("Scaled feature statistics:")
        print()
        extreme_values_summary = []
        for col in self.df_scaled.columns:
            stats = {
                'mean': self.df_scaled[col].mean(),
                'std': self.df_scaled[col].std(ddof=0),
                'min': self.df_scaled[col].min(),
                'max': self.df_scaled[col].max(),
                'median': self.df_scaled[col].median(),
                'skewness': self.df_scaled[col].skew()
            }
            self.post_stats[col] = stats
            print(f"{col}:")
            print(f"   Mean: {stats['mean']:>12.6f} (target: 0.000)")
            print(f"   Std:  {stats['std']:>12.6f} (target: 1.000)")
            print(f"   Min:  {stats['min']:>12.3f}")
            print(f"   Max:  {stats['max']:>12.3f}")
            print(f"   Median: {stats['median']:>10.3f}")
            print(f"   Skewness: {stats['skewness']:>8.3f}")
            if abs(stats['min']) > 3 or abs(stats['max']) > 3:
                extreme_values_summary.append((col, stats['min'], stats['max']))
                print(f"   NOTE: Contains extreme values (|z| > 3)")
            print()
        if extreme_values_summary:
            print()
            print("EXTREME VALUES SUMMARY (|z-score| > 3):")
            print("-" * 100)
            for col, min_val, max_val in extreme_values_summary:
                outlier_count = ((self.df_scaled[col] < -3) | (self.df_scaled[col] > 3)).sum()
                outlier_pct = outlier_count / len(self.df_scaled) * 100
                print(f"   {col:<40} Range: [{min_val:>7.3f}, {max_val:>7.3f}]  Outliers: {outlier_count} ({outlier_pct:.2f}%)")
            print()
        self.log_action("Compute post-scaling stats", f"{len(self.df_scaled.columns)} features")
    
    # STEP 6: VALIDATE SCALING
    def validate_scaling(self):
        self._print_header("STEP 5: VALIDATE SCALING RESULTS")
        print("Validation checks:")
        print()
        mean_check = []
        for col in self.df_scaled.columns:
            mean = self.df_scaled[col].mean()
            status = "PASS" if abs(mean) < 1e-10 else "WARNING"
            mean_check.append((col, mean, status))
        print("Mean ≈ 0 check:")
        for col, mean, status in mean_check:
            print(f"   {col:<40} mean={mean:>12.10f} ({status})")
        print()
        std_check = []
        for col in self.df_scaled.columns:
            std = self.df_scaled[col].std(ddof=0)
            status = "PASS" if abs(std - 1.0) < 1e-10 else "WARNING"
            std_check.append((col, std, status))
        print("Std ≈ 1 check (using ddof=0 to match StandardScaler):")
        for col, std, status in std_check:
            print(f"   {col:<40} std={std:>12.10f} ({status})")
        print()
        missing_after = self.df_scaled.isnull().sum().sum()
        if missing_after == 0:
            print("Missing values after scaling: None (PASS)")
        else:
            print(f"Missing values after scaling: {missing_after} (FAILED)")
        print()
        self.log_action("Validate scaling", "Checks completed")
    
    # NEW STEP: APPLY ROBUST SCALER (OPTIONAL)
    def apply_robust_scaling(self):
        """Apply RobustScaler to selected numeric columns and export alternative dataset."""
        if not self.use_robust:
            return False
        self._print_header("OPTIONAL: APPLY ROBUSTSCALER")
        # choose columns
        if self.robust_columns is None:
            cols = [c for c in self.df.columns if np.issubdtype(self.df[c].dtype, np.number)]
        else:
            cols = [c for c in self.robust_columns if c in self.df.columns and np.issubdtype(self.df[c].dtype, np.number)]
        if not cols:
            print("No numeric columns found for RobustScaler. Skipping robust scaling.")
            self.log_action("Apply RobustScaler SKIPPED", "No numeric columns")
            return False
        print(f"Applying RobustScaler to columns: {cols}")
        try:
            self.robust_scaler = RobustScaler()
            arr = self.robust_scaler.fit_transform(self.df[cols])
            df_rob = self.df.copy()
            df_rob[cols] = arr
            self.df_robust_scaled = df_rob
            # save scaler and CSV
            joblib.dump(self.robust_scaler, self.robust_scaler_file)
            df_rob.to_csv(self.robust_output_csv, index=False)
            print(f"Exported robust-scaled dataset: {self.robust_output_csv}")
            print(f"Saved robust scaler: {self.robust_scaler_file}")
            self.log_action("Apply RobustScaler", f"Columns: {cols}")
            self.log_action("Export robust dataset", self.robust_output_csv)
            self.log_action("Save robust scaler", self.robust_scaler_file)
            return True
        except Exception as e:
            print(f"Robust scaling failed: {e}")
            self.log_action("Apply RobustScaler FAILED", str(e))
            return False
    
    # STEP 7: EXPORT SCALED DATASET
    def export_dataset(self):
        self._print_header("STEP 6: EXPORT SCALED DATASET")
        try:
            self.df_scaled.to_csv(self.output_csv, index=False)
            print(f"Exported scaled dataset successfully:")
            print(f"   Path: {self.output_csv}")
            print(f"   Shape: {self.df_scaled.shape}")
            print(f"   Size: {os.path.getsize(self.output_csv) / 1024:.2f} KB")
            print()
            self.log_action("Export scaled dataset", self.output_csv)
            return True
        except Exception as e:
            print(f"Export failed: {e}")
            self.log_action("Export FAILED", str(e))
            return False
    
    # STEP 8: EXPORT REPORT
    def export_report(self):
        self._print_header("STEP 7: EXPORT REPORT")
        try:
            with open(self.report_file, 'w', encoding='utf-8') as f:
                f.write("=" * 100 + "\n")
                f.write("DEMOGRAPHIC FEATURE SCALING REPORT\n")
                f.write("=" * 100 + "\n\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Input: {self.input_path}\n")
                f.write(f"Output: {self.output_csv}\n")
                f.write(f"Scaling Method: StandardScaler (Z-score normalization)\n")
                if self.use_robust:
                    f.write(f"RobustScaler output: {self.robust_output_csv}\n")
                f.write("\n")
                f.write("DATASET INFORMATION:\n")
                f.write("-" * 100 + "\n")
                f.write(f"Shape: {self.df.shape[0]:,} rows x {self.df.shape[1]} columns\n\n")
                f.write("PRE-SCALING STATISTICS:\n")
                f.write("-" * 100 + "\n")
                for col, stats in self.pre_stats.items():
                    f.write(f"{col}:\n")
                    for key, val in stats.items():
                        f.write(f"   {key}: {val}\n")
                    f.write("\n")
                f.write("\nSCALER PARAMETERS:\n")
                f.write("-" * 100 + "\n")
                for i, col in enumerate(self.df.columns):
                    f.write(f"{col}:\n")
                    f.write(f"   mean: {self.scaler.mean_[i]}\n")
                    f.write(f"   scale: {self.scaler.scale_[i]}\n\n")
                f.write("\nPOST-SCALING STATISTICS:\n")
                f.write("-" * 100 + "\n")
                for col, stats in self.post_stats.items():
                    f.write(f"{col}:\n")
                    for key, val in stats.items():
                        f.write(f"   {key}: {val}\n")
                    f.write("\n")
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
    
    # MAIN PIPELINE
    def run_scaling(self):
        print("\n" + "=" * 100)
        print("... FEATURE SCALING PIPELINE".center(100))
        print("=" * 100 + "\n")
        start_time = datetime.now()
        if not self.load_data():
            print("Pipeline failed at load_data()")
            return False
        self.validate_data()
        self.compute_pre_stats()
        if not self.apply_scaling():
            print("Pipeline failed at apply_scaling()")
            return False
        self.compute_post_stats()
        self.validate_scaling()
        if not self.export_dataset():
            print("Pipeline failed at export_dataset()")
            return False
        # optional robust scaling
        if self.use_robust:
            self.apply_robust_scaling()
        self.save_scaler()
        self.export_report()
        elapsed = (datetime.now() - start_time).total_seconds()
        print("=" * 100)
        print("PIPELINE COMPLETED".center(100))
        print("=" * 100)
        print(f"Time elapsed: {elapsed:.2f} seconds")
        print(f"Output dataset: {self.output_csv}")
        if self.use_robust:
            print(f"Robust dataset: {self.robust_output_csv}")
        print(f"Output report: {self.report_file}")
        print(f"Saved scaler: {self.scaler_file}")
        print("=" * 100 + "\n")
        return True

    def save_scaler(self):
        try:
            joblib.dump(self.scaler, self.scaler_file)
            print(f"Saved scaler to: {self.scaler_file}")
            self.log_action("Save scaler", self.scaler_file)
            return True
        except Exception as e:
            print(f"Failed to save scaler: {e}")
            self.log_action("Save scaler FAILED", str(e))
            return False


# ================================================================================
# CLASS 2: PRODUCT+CHANNEL SCALER
# ================================================================================

class ProductChannelScaler:
    """
    ProductChannel scaler with optional RobustScaler.
    Same pattern as DemographicScaler.
    """
    def __init__(self, input_path, output_dir, report_dir, use_robust=False, robust_columns=None):
        self.input_path = input_path
        self.output_dir = output_dir
        self.report_dir = report_dir
        self.use_robust = use_robust
        self.robust_columns = robust_columns

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(report_dir, exist_ok=True)

        self.output_csv = os.path.join(output_dir, "Customer_Behavior_ProductChannel_scaled.csv")
        self.report_file = os.path.join(report_dir, "ProductChannel_Scaling_Report.log")
        self.scaler_file = os.path.join(output_dir, "productchannel_scaler.pkl")
        self.robust_output_csv = os.path.join(output_dir, "Customer_Behavior_ProductChannel_robust_scaled.csv")
        self.robust_scaler_file = os.path.join(output_dir, "productchannel_robust_scaler.pkl")

        self.df = None
        self.df_scaled = None
        self.df_robust_scaled = None
        self.scaler = StandardScaler()
        self.robust_scaler = None
        self.pre_stats = {}
        self.post_stats = {}
        self.processing_log = []

    def log_action(self, action, details=""):
        self.processing_log.append({
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'action': action,
            'details': details
        })

    def _print_header(self, title, width=100, char='='):
        print(char * width)
        print(f"{title:^{width}}")
        print(char * width)
        print()

    def load_data(self):
        self._print_header("PRODUCT+CHANNEL SCALING - LOAD DATA")
        try:
            self.df = pd.read_csv(self.input_path)
            print(f"Loaded dataset successfully")
            print(f"   Path: {self.input_path}")
            print(f"   Shape: {self.df.shape[0]:,} rows x {self.df.shape[1]} columns")
            print(f"   Memory: {self.df.memory_usage(deep=True).sum() / 1024:.2f} KB")
            print()
            print(f"Features loaded:")
            for i, col in enumerate(self.df.columns, 1):
                print(f"   {i}. {col:<40} ({self.df[col].dtype})")
            print()
            self.log_action("Load data", f"Shape: {self.df.shape}")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            self.log_action("Load data FAILED", str(e))
            return False

    def validate_data(self):
        self._print_header("STEP 1: VALIDATE DATA QUALITY")
        print("Pre-Scaling Data Quality Checks:")
        print()
        missing = self.df.isnull().sum()
        if missing.sum() == 0:
            print("   Missing values: None (PASS)")
        else:
            print("   Missing values detected:")
            for col in missing[missing > 0].index:
                print(f"      {col}: {missing[col]} ({missing[col]/len(self.df)*100:.2f}%)")
        inf_count = np.isinf(self.df.select_dtypes(include=[np.number])).sum().sum()
        if inf_count == 0:
            print("   Infinite values: None (PASS)")
        else:
            print(f"   Infinite values: {inf_count} (WARNING)")
        non_numeric = self.df.select_dtypes(exclude=[np.number]).columns.tolist()
        if not non_numeric:
            print("   Data types: All numeric (PASS)")
        else:
            print(f"   Non-numeric columns: {non_numeric} (WARNING)")
        print()
        self.log_action("Validate data", "Checks completed")

    def compute_pre_stats(self):
        self._print_header("STEP 2: PRE-SCALING STATISTICS")
        print("Original feature statistics:")
        print()
        for col in self.df.columns:
            stats = {
                'mean': self.df[col].mean(),
                'std': self.df[col].std(),
                'min': self.df[col].min(),
                'max': self.df[col].max(),
                'median': self.df[col].median(),
                'skewness': self.df[col].skew()
            }
            self.pre_stats[col] = stats
            print(f"{col}:")
            print(f"   Mean: {stats['mean']:>12.3f}")
            print(f"   Std:  {stats['std']:>12.3f}")
            print(f"   Min:  {stats['min']:>12.3f}")
            print(f"   Max:  {stats['max']:>12.3f}")
            print(f"   Median: {stats['median']:>10.3f}")
            print(f"   Skewness: {stats['skewness']:>8.3f}")
            print()
        self.log_action("Compute pre-scaling stats", f"{len(self.df.columns)} features")

    def apply_scaling(self):
        self._print_header("STEP 3: APPLY STANDARDSCALER")
        print("Applying Z-score normalization...")
        print(f"   Formula: z = (x - mean) / std")
        print(f"   Target: mean=0, std=1")
        print()
        try:
            scaled_data = self.scaler.fit_transform(self.df)
            self.df_scaled = pd.DataFrame(scaled_data, columns=self.df.columns, index=self.df.index)
            print(f"Scaling completed successfully")
            print(f"   Scaled shape: {self.df_scaled.shape}")
            print()
            print("Scaler parameters (learned from data):")
            for i, col in enumerate(self.df.columns):
                print(f"   {col}:")
                print(f"      Mean (μ): {self.scaler.mean_[i]:>12.3f}")
                print(f"      Std (σ):  {self.scaler.scale_[i]:>12.3f}")
            print()
            self.log_action("Apply StandardScaler", "Success")
            return True
        except Exception as e:
            print(f"Scaling failed: {e}")
            self.log_action("Apply StandardScaler FAILED", str(e))
            return False

    def compute_post_stats(self):
        self._print_header("STEP 4: POST-SCALING STATISTICS")
        print("Scaled feature statistics:")
        print()
        extreme_values_summary = []
        for col in self.df_scaled.columns:
            stats = {
                'mean': self.df_scaled[col].mean(),
                'std': self.df_scaled[col].std(ddof=0),
                'min': self.df_scaled[col].min(),
                'max': self.df_scaled[col].max(),
                'median': self.df_scaled[col].median(),
                'skewness': self.df_scaled[col].skew()
            }
            self.post_stats[col] = stats
            print(f"{col}:")
            print(f"   Mean: {stats['mean']:>12.6f} (target: 0.000)")
            print(f"   Std:  {stats['std']:>12.6f} (target: 1.000)")
            print(f"   Min:  {stats['min']:>12.3f}")
            print(f"   Max:  {stats['max']:>12.3f}")
            print(f"   Median: {stats['median']:>10.3f}")
            print(f"   Skewness: {stats['skewness']:>8.3f}")
            if abs(stats['min']) > 3 or abs(stats['max']) > 3:
                extreme_values_summary.append((col, stats['min'], stats['max']))
                print(f"   NOTE: Contains extreme values (|z| > 3)")
            print()
        if extreme_values_summary:
            print()
            print("EXTREME VALUES SUMMARY (|z-score| > 3):")
            print("-" * 100)
            for col, min_val, max_val in extreme_values_summary:
                outlier_count = ((self.df_scaled[col] < -3) | (self.df_scaled[col] > 3)).sum()
                outlier_pct = outlier_count / len(self.df_scaled) * 100
                print(f"   {col:<40} Range: [{min_val:>7.3f}, {max_val:>7.3f}]  Outliers: {outlier_count} ({outlier_pct:.2f}%)")
            print()
        self.log_action("Compute post-scaling stats", f"{len(self.df_scaled.columns)} features")

    def validate_scaling(self):
        self._print_header("STEP 5: VALIDATE SCALING RESULTS")
        print("Validation checks:")
        print()
        mean_check = []
        for col in self.df_scaled.columns:
            mean = self.df_scaled[col].mean()
            status = "PASS" if abs(mean) < 1e-10 else "WARNING"
            mean_check.append((col, mean, status))
        print("Mean ≈ 0 check:")
        for col, mean, status in mean_check:
            print(f"   {col:<40} mean={mean:>12.10f} ({status})")
        print()
        std_check = []
        for col in self.df_scaled.columns:
            std = self.df_scaled[col].std(ddof=0)
            status = "PASS" if abs(std - 1.0) < 1e-10 else "WARNING"
            std_check.append((col, std, status))
        print("Std ≈ 1 check (using ddof=0 to match StandardScaler):")
        for col, std, status in std_check:
            print(f"   {col:<40} std={std:>12.10f} ({status})")
        print()
        missing_after = self.df_scaled.isnull().sum().sum()
        if missing_after == 0:
            print("Missing values after scaling: None (PASS)")
        else:
            print(f"Missing values after scaling: {missing_after} (FAILED)")
        print()
        self.log_action("Validate scaling", "Checks completed")

    def apply_robust_scaling(self):
        if not self.use_robust:
            return False
        self._print_header("OPTIONAL: APPLY ROBUSTSCALER")
        if self.robust_columns is None:
            cols = [c for c in self.df.columns if np.issubdtype(self.df[c].dtype, np.number)]
        else:
            cols = [c for c in self.robust_columns if c in self.df.columns and np.issubdtype(self.df[c].dtype, np.number)]
        if not cols:
            print("No numeric columns found for RobustScaler. Skipping robust scaling.")
            self.log_action("Apply RobustScaler SKIPPED", "No numeric columns")
            return False
        print(f"Applying RobustScaler to columns: {cols}")
        try:
            self.robust_scaler = RobustScaler()
            arr = self.robust_scaler.fit_transform(self.df[cols])
            df_rob = self.df.copy()
            df_rob[cols] = arr
            self.df_robust_scaled = df_rob
            joblib.dump(self.robust_scaler, self.robust_scaler_file)
            df_rob.to_csv(self.robust_output_csv, index=False)
            print(f"Exported robust-scaled dataset: {self.robust_output_csv}")
            print(f"Saved robust scaler: {self.robust_scaler_file}")
            self.log_action("Apply RobustScaler", f"Columns: {cols}")
            self.log_action("Export robust dataset", self.robust_output_csv)
            self.log_action("Save robust scaler", self.robust_scaler_file)
            return True
        except Exception as e:
            print(f"Robust scaling failed: {e}")
            self.log_action("Apply RobustScaler FAILED", str(e))
            return False

    def export_dataset(self):
        self._print_header("STEP 6: EXPORT SCALED DATASET")
        try:
            self.df_scaled.to_csv(self.output_csv, index=False)
            print(f"Exported scaled dataset successfully:")
            print(f"   Path: {self.output_csv}")
            print(f"   Shape: {self.df_scaled.shape}")
            print(f"   Size: {os.path.getsize(self.output_csv) / 1024:.2f} KB")
            print()
            self.log_action("Export scaled dataset", self.output_csv)
            return True
        except Exception as e:
            print(f"Export failed: {e}")
            self.log_action("Export FAILED", str(e))
            return False

    def export_report(self):
        self._print_header("STEP 7: EXPORT REPORT")
        try:
            with open(self.report_file, 'w', encoding='utf-8') as f:
                f.write("=" * 100 + "\n")
                f.write("PRODUCT+CHANNEL FEATURE SCALING REPORT\n")
                f.write("=" * 100 + "\n\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Input: {self.input_path}\n")
                f.write(f"Output: {self.output_csv}\n")
                f.write(f"Scaling Method: StandardScaler (Z-score normalization)\n")
                if self.use_robust:
                    f.write(f"RobustScaler output: {self.robust_output_csv}\n")
                f.write("\n")
                f.write("DATASET INFORMATION:\n")
                f.write("-" * 100 + "\n")
                f.write(f"Shape: {self.df.shape[0]:,} rows x {self.df.shape[1]} columns\n\n")
                f.write("PRE-SCALING STATISTICS:\n")
                f.write("-" * 100 + "\n")
                for col, stats in self.pre_stats.items():
                    f.write(f"{col}:\n")
                    for key, val in stats.items():
                        f.write(f"   {key}: {val}\n")
                    f.write("\n")
                f.write("\nSCALER PARAMETERS:\n")
                f.write("-" * 100 + "\n")
                for i, col in enumerate(self.df.columns):
                    f.write(f"{col}:\n")
                    f.write(f"   mean: {self.scaler.mean_[i]}\n")
                    f.write(f"   scale: {self.scaler.scale_[i]}\n\n")
                f.write("\nPOST-SCALING STATISTICS:\n")
                f.write("-" * 100 + "\n")
                for col, stats in self.post_stats.items():
                    f.write(f"{col}:\n")
                    for key, val in stats.items():
                        f.write(f"   {key}: {val}\n")
                    f.write("\n")
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

    def run_scaling(self):
        print("\n" + "=" * 100)
        print("PRODUCT+CHANNEL FEATURE SCALING PIPELINE".center(100))
        print("=" * 100 + "\n")
        start_time = datetime.now()
        if not self.load_data():
            print("Pipeline failed at load_data()")
            return False
        self.validate_data()
        self.compute_pre_stats()
        if not self.apply_scaling():
            print("Pipeline failed at apply_scaling()")
            return False
        self.compute_post_stats()
        self.validate_scaling()
        if not self.export_dataset():
            print("Pipeline failed at export_dataset()")
            return False
        if self.use_robust:
            self.apply_robust_scaling()
        try:
            joblib.dump(self.scaler, self.scaler_file)
            self.log_action("Save scaler", self.scaler_file)
        except Exception:
            pass
        self.export_report()
        elapsed = (datetime.now() - start_time).total_seconds()
        print("=" * 100)
        print("PIPELINE COMPLETED".center(100))
        print("=" * 100)
        print(f"Time elapsed: {elapsed:.2f} seconds")
        print(f"Output dataset: {self.output_csv}")
        if self.use_robust:
            print(f"Robust dataset: {self.robust_output_csv}")
        print(f"Output report: {self.report_file}")
        print(f"Saved scaler: {self.scaler_file}")
        print("=" * 100 + "\n")
        return True


# ================================================================================
# CLASS 3: RFM SCALER
# ================================================================================

class RFMScaler:
    """
    RFM scaler with optional RobustScaler.
    """
    def __init__(self, input_path, output_dir, report_dir, use_robust=False, robust_columns=None):
        self.input_path = input_path
        self.output_dir = output_dir
        self.report_dir = report_dir
        self.use_robust = use_robust
        self.robust_columns = robust_columns

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(report_dir, exist_ok=True)

        self.output_csv = os.path.join(output_dir, "Customer_Behavior_RFM_scaled.csv")
        self.report_file = os.path.join(report_dir, "RFM_Scaling_Report.log")
        self.scaler_file = os.path.join(output_dir, "rfm_scaler.pkl")
        self.robust_output_csv = os.path.join(output_dir, "Customer_Behavior_RFM_robust_scaled.csv")
        self.robust_scaler_file = os.path.join(output_dir, "rfm_robust_scaler.pkl")

        self.df = None
        self.df_scaled = None
        self.df_robust_scaled = None
        self.scaler = StandardScaler()
        self.robust_scaler = None
        self.pre_stats = {}
        self.post_stats = {}
        self.processing_log = []

    def log_action(self, action, details=""):
        self.processing_log.append({
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'action': action,
            'details': details
        })

    def _print_header(self, title, width=100, char='='):
        print(char * width)
        print(f"{title:^{width}}")
        print(char * width)
        print()

    def load_data(self):
        self._print_header("RFM SCALING - LOAD DATA")
        try:
            self.df = pd.read_csv(self.input_path)
            print(f"Loaded dataset successfully")
            print(f"   Path: {self.input_path}")
            print(f"   Shape: {self.df.shape[0]:,} rows x {self.df.shape[1]} columns")
            print(f"   Memory: {self.df.memory_usage(deep=True).sum() / 1024:.2f} KB")
            print()
            print(f"Features loaded:")
            for i, col in enumerate(self.df.columns, 1):
                print(f"   {i}. {col:<40} ({self.df[col].dtype})")
            print()
            self.log_action("Load data", f"Shape: {self.df.shape}")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            self.log_action("Load data FAILED", str(e))
            return False

    def validate_data(self):
        self._print_header("STEP 1: VALIDATE DATA QUALITY")
        print("Pre-Scaling Data Quality Checks:")
        print()
        missing = self.df.isnull().sum()
        if missing.sum() == 0:
            print("   Missing values: None (PASS)")
        else:
            print("   Missing values detected:")
            for col in missing[missing > 0].index:
                print(f"      {col}: {missing[col]} ({missing[col]/len(self.df)*100:.2f}%)")
        inf_count = np.isinf(self.df.select_dtypes(include=[np.number])).sum().sum()
        if inf_count == 0:
            print("   Infinite values: None (PASS)")
        else:
            print(f"   Infinite values: {inf_count} (WARNING)")
        non_numeric = self.df.select_dtypes(exclude=[np.number]).columns.tolist()
        if not non_numeric:
            print("   Data types: All numeric (PASS)")
        else:
            print(f"   Non-numeric columns: {non_numeric} (WARNING)")
        print()
        self.log_action("Validate data", "Checks completed")

    def compute_pre_stats(self):
        self._print_header("STEP 2: PRE-SCALING STATISTICS")
        print("Original feature statistics:")
        print()
        for col in self.df.columns:
            stats = {
                'mean': self.df[col].mean(),
                'std': self.df[col].std(),
                'min': self.df[col].min(),
                'max': self.df[col].max(),
                'median': self.df[col].median(),
                'skewness': self.df[col].skew()
            }
            self.pre_stats[col] = stats
            print(f"{col}:")
            print(f"   Mean: {stats['mean']:>12.3f}")
            print(f"   Std:  {stats['std']:>12.3f}")
            print(f"   Min:  {stats['min']:>12.3f}")
            print(f"   Max:  {stats['max']:>12.3f}")
            print(f"   Median: {stats['median']:>10.3f}")
            print(f"   Skewness: {stats['skewness']:>8.3f}")
            print()
        self.log_action("Compute pre-scaling stats", f"{len(self.df.columns)} features")

    def apply_scaling(self):
        self._print_header("STEP 3: APPLY STANDARDSCALER")
        print("Applying Z-score normalization...")
        print(f"   Formula: z = (x - mean) / std")
        print(f"   Target: mean=0, std=1")
        print()
        try:
            scaled_data = self.scaler.fit_transform(self.df)
            self.df_scaled = pd.DataFrame(scaled_data, columns=self.df.columns, index=self.df.index)
            print(f"Scaling completed successfully")
            print(f"   Scaled shape: {self.df_scaled.shape}")
            print()
            print("Scaler parameters (learned from data):")
            for i, col in enumerate(self.df.columns):
                print(f"   {col}:")
                print(f"      Mean (μ): {self.scaler.mean_[i]:>12.3f}")
                print(f"      Std (σ):  {self.scaler.scale_[i]:>12.3f}")
            print()
            self.log_action("Apply StandardScaler", "Success")
            return True
        except Exception as e:
            print(f"Scaling failed: {e}")
            self.log_action("Apply StandardScaler FAILED", str(e))
            return False

    def compute_post_stats(self):
        self._print_header("STEP 4: POST-SCALING STATISTICS")
        print("Scaled feature statistics:")
        print()
        extreme_values_summary = []
        for col in self.df_scaled.columns:
            stats = {
                'mean': self.df_scaled[col].mean(),
                'std': self.df_scaled[col].std(ddof=0),
                'min': self.df_scaled[col].min(),
                'max': self.df_scaled[col].max(),
                'median': self.df_scaled[col].median(),
                'skewness': self.df_scaled[col].skew()
            }
            self.post_stats[col] = stats
            print(f"{col}:")
            print(f"   Mean: {stats['mean']:>12.6f} (target: 0.000)")
            print(f"   Std:  {stats['std']:>12.6f} (target: 1.000)")
            print(f"   Min:  {stats['min']:>12.3f}")
            print(f"   Max:  {stats['max']:>12.3f}")
            print(f"   Median: {stats['median']:>10.3f}")
            print(f"   Skewness: {stats['skewness']:>8.3f}")
            if abs(stats['min']) > 3 or abs(stats['max']) > 3:
                extreme_values_summary.append((col, stats['min'], stats['max']))
                print(f"   NOTE: Contains extreme values (|z| > 3)")
            print()
        if extreme_values_summary:
            print()
            print("EXTREME VALUES SUMMARY (|z-score| > 3):")
            print("-" * 100)
            for col, min_val, max_val in extreme_values_summary:
                outlier_count = ((self.df_scaled[col] < -3) | (self.df_scaled[col] > 3)).sum()
                outlier_pct = outlier_count / len(self.df_scaled) * 100
                print(f"   {col:<40} Range: [{min_val:>7.3f}, {max_val:>7.3f}]  Outliers: {outlier_count} ({outlier_pct:.2f}%)")
            print()
        self.log_action("Compute post-scaling stats", f"{len(self.df_scaled.columns)} features")

    def validate_scaling(self):
        self._print_header("STEP 5: VALIDATE SCALING RESULTS")
        print("Validation checks:")
        print()
        mean_check = []
        for col in self.df_scaled.columns:
            mean = self.df_scaled[col].mean()
            status = "PASS" if abs(mean) < 1e-10 else "WARNING"
            mean_check.append((col, mean, status))
        print("Mean ≈ 0 check:")
        for col, mean, status in mean_check:
            print(f"   {col:<40} mean={mean:>12.10f} ({status})")
        print()
        std_check = []
        for col in self.df_scaled.columns:
            std = self.df_scaled[col].std(ddof=0)
            status = "PASS" if abs(std - 1.0) < 1e-10 else "WARNING"
            std_check.append((col, std, status))
        print("Std ≈ 1 check (using ddof=0 to match StandardScaler):")
        for col, std, status in std_check:
            print(f"   {col:<40} std={std:>12.10f} ({status})")
        print()
        missing_after = self.df_scaled.isnull().sum().sum()
        if missing_after == 0:
            print("Missing values after scaling: None (PASS)")
        else:
            print(f"Missing values after scaling: {missing_after} (FAILED)")
        print()
        self.log_action("Validate scaling", "Checks completed")

    def apply_robust_scaling(self):
        if not self.use_robust:
            return False
        self._print_header("OPTIONAL: APPLY ROBUSTSCALER")
        if self.robust_columns is None:
            cols = [c for c in self.df.columns if np.issubdtype(self.df[c].dtype, np.number)]
        else:
            cols = [c for c in self.robust_columns if c in self.df.columns and np.issubdtype(self.df[c].dtype, np.number)]
        if not cols:
            print("No numeric columns found for RobustScaler. Skipping robust scaling.")
            self.log_action("Apply RobustScaler SKIPPED", "No numeric columns")
            return False
        print(f"Applying RobustScaler to columns: {cols}")
        try:
            self.robust_scaler = RobustScaler()
            arr = self.robust_scaler.fit_transform(self.df[cols])
            df_rob = self.df.copy()
            df_rob[cols] = arr
            self.df_robust_scaled = df_rob
            joblib.dump(self.robust_scaler, self.robust_scaler_file)
            df_rob.to_csv(self.robust_output_csv, index=False)
            print(f"Exported robust-scaled dataset: {self.robust_output_csv}")
            print(f"Saved robust scaler: {self.robust_scaler_file}")
            self.log_action("Apply RobustScaler", f"Columns: {cols}")
            self.log_action("Export robust dataset", self.robust_output_csv)
            self.log_action("Save robust scaler", self.robust_scaler_file)
            return True
        except Exception as e:
            print(f"Robust scaling failed: {e}")
            self.log_action("Apply RobustScaler FAILED", str(e))
            return False

    def export_dataset(self):
        self._print_header("STEP 6: EXPORT SCALED DATASET")
        try:
            self.df_scaled.to_csv(self.output_csv, index=False)
            print(f"Exported scaled dataset successfully:")
            print(f"   Path: {self.output_csv}")
            print(f"   Shape: {self.df_scaled.shape}")
            print(f"   Size: {os.path.getsize(self.output_csv) / 1024:.2f} KB")
            print()
            self.log_action("Export scaled dataset", self.output_csv)
            return True
        except Exception as e:
            print(f"Export failed: {e}")
            self.log_action("Export FAILED", str(e))
            return False

    def export_report(self):
        self._print_header("STEP 7: EXPORT REPORT")
        try:
            with open(self.report_file, 'w', encoding='utf-8') as f:
                f.write("=" * 100 + "\n")
                f.write("RFM FEATURE SCALING REPORT\n")
                f.write("=" * 100 + "\n\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Input: {self.input_path}\n")
                f.write(f"Output: {self.output_csv}\n")
                f.write(f"Scaling Method: StandardScaler (Z-score normalization)\n")
                if self.use_robust:
                    f.write(f"RobustScaler output: {self.robust_output_csv}\n")
                f.write("\n")
                f.write("DATASET INFORMATION:\n")
                f.write("-" * 100 + "\n")
                f.write(f"Shape: {self.df.shape[0]:,} rows x {self.df.shape[1]} columns\n\n")
                f.write("PRE-SCALING STATISTICS:\n")
                f.write("-" * 100 + "\n")
                for col, stats in self.pre_stats.items():
                    f.write(f"{col}:\n")
                    for key, val in stats.items():
                        f.write(f"   {key}: {val}\n")
                    f.write("\n")
                f.write("\nSCALER PARAMETERS:\n")
                f.write("-" * 100 + "\n")
                for i, col in enumerate(self.df.columns):
                    f.write(f"{col}:\n")
                    f.write(f"   mean: {self.scaler.mean_[i]}\n")
                    f.write(f"   scale: {self.scaler.scale_[i]}\n\n")
                f.write("\nPOST-SCALING STATISTICS:\n")
                f.write("-" * 100 + "\n")
                for col, stats in self.post_stats.items():
                    f.write(f"{col}:\n")
                    for key, val in stats.items():
                        f.write(f"   {key}: {val}\n")
                    f.write("\n")
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

    def run_scaling(self):
        print("\n" + "=" * 100)
        print("RFM FEATURE SCALING PIPELINE".center(100))
        print("=" * 100 + "\n")
        start_time = datetime.now()
        if not self.load_data():
            print("Pipeline failed at load_data()")
            return False
        self.validate_data()
        self.compute_pre_stats()
        if not self.apply_scaling():
            print("Pipeline failed at apply_scaling()")
            return False
        self.compute_post_stats()
        self.validate_scaling()
        if not self.export_dataset():
            print("Pipeline failed at export_dataset()")
            return False
        if self.use_robust:
            self.apply_robust_scaling()
        try:
            joblib.dump(self.scaler, self.scaler_file)
            self.log_action("Save scaler", self.scaler_file)
        except Exception:
            pass
        self.export_report()
        elapsed = (datetime.now() - start_time).total_seconds()
        print("=" * 100)
        print("PIPELINE COMPLETED".center(100))
        print("=" * 100)
        print(f"Time elapsed: {elapsed:.2f} seconds")
        print(f"Output dataset: {self.output_csv}")
        if self.use_robust:
            print(f"Robust dataset: {self.robust_output_csv}")
        print(f"Output report: {self.report_file}")
        print(f"Saved scaler: {self.scaler_file}")
        print("=" * 100 + "\n")
        return True


# ================================================================================
# MAIN EXECUTION
# ================================================================================

def main():
    """Main function - Run all scaling classes."""
    dataset_dir = r"C:\Project\Machine_Learning\Machine_Learning\dataset"
    report_dir = r"C:\Project\Machine_Learning\Machine_Learning\report\Feature Scaling & Selection_report"

    demographic_input = os.path.join(dataset_dir, "Customer_Behavior_Demographic.csv")
    productchannel_input = os.path.join(dataset_dir, "Customer_Behavior_ProductChannel.csv")
    rfm_input = os.path.join(dataset_dir, "Customer_Behavior_RFM.csv")

    print("\n" + "=" * 100)
    print("MULTI-OBJECTIVE FEATURE SCALING".center(100))
    print("=" * 100 + "\n")

    # Demographic: enable robust option if desired
    demographic_scaler = DemographicScaler(demographic_input, dataset_dir, report_dir, use_robust=True)
    demo_success = demographic_scaler.run_scaling()

    productchannel_scaler = ProductChannelScaler(productchannel_input, dataset_dir, report_dir, use_robust=True)
    pc_success = productchannel_scaler.run_scaling()

    rfm_scaler = RFMScaler(rfm_input, dataset_dir, report_dir, use_robust=True)
    rfm_success = rfm_scaler.run_scaling()

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
        print("ALL FEATURE SCALING COMPLETED")
        print("DATA PREPARATION STAGE FINISHED")
        print()
        print("Ready for next stage: K-Means Clustering")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()