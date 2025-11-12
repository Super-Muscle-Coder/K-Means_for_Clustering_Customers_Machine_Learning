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
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
warnings.filterwarnings('ignore')

# ================================================================================
# CLASS 1: DEMOGRAPHIC SCALER (UPDATED - SEPARATE STANDARD & ROBUST OUTPUTS)
# ================================================================================

class DemographicScaler:
    """
    Feature Scaling cho Phan Cum Nhan Khau Hoc (Demographic Clustering)

    INPUT (from Feature_Engineering.py):
    - 5 columns total:
      * 3 CLUSTERING: Age, Income, Dependency_Ratio (transformed)
      * 2 POST-HOC: Education_ord, Life_Stage (NOT scaled)

    SCALING STRATEGY (UPDATED):
    - Dataset 1: StandardScaler ONLY (mean=0, std=1)
    - Dataset 2: RobustScaler ONLY (median=0, IQR=1)
    - Preserve 2 post-hoc features unchanged in both datasets
    - NO sequential scaling (Standard → Robust)

    OUTPUT:
    - Customer_Behavior_Demographic_Standard_scaled.csv (StandardScaler only)
    - Customer_Behavior_Demographic_Robust_scaled.csv (RobustScaler only)
    - Demographic_Scaling_Report_Standard.log
    - Demographic_Scaling_Report_Robust.log
    - Visualization graphs with Standard/Robust suffixes
    """
    
    def __init__(self, input_path, output_dir, report_dir):
        """
        Initialize Demographic Scaler.

        Args:
            input_path: Path to engineered demographic CSV
            output_dir: Directory for scaled outputs
            report_dir: Directory for reports
        """
        self.input_path = input_path
        self.output_dir = output_dir
        self.report_dir = report_dir
        
        # Create directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(report_dir, exist_ok=True)
        
        # ============================================================================
        # STANDARD SCALER OUTPUTS
        # ============================================================================
        self.output_standard_csv = os.path.join(output_dir, "Customer_Behavior_Demographic_Standard_scaled.csv")
        self.report_standard_file = os.path.join(report_dir, "Demographic_Scaling_Report_Standard.log")
        
        # ============================================================================
        # ROBUST SCALER OUTPUTS
        # ============================================================================
        self.output_robust_csv = os.path.join(output_dir, "Customer_Behavior_Demographic_Robust_scaled.csv")
        self.report_robust_file = os.path.join(report_dir, "Demographic_Scaling_Report_Robust.log")
        
        # Data storage
        self.df = None
        self.df_standard_scaled = None
        self.df_robust_scaled = None
        
        # Scalers
        self.standard_scaler = StandardScaler()
        self.robust_scaler = RobustScaler()
        
        # Statistics (separate for Standard and Robust)
        self.pre_stats = {}
        self.post_standard_stats = {}
        self.post_robust_stats = {}
        self.processing_log = []
        
        # Feature definitions
        self.CLUSTERING_FEATURES = ['Age', 'Income', 'Dependency_Ratio']
        self.POST_HOC_FEATURES = ['Education_ord', 'Life_Stage']
        self.EXPECTED_TOTAL_COLUMNS = 5
    
    # ============================================================================
    # UTILITY METHODS
    # ============================================================================
    
    def log_action(self, action, details=""):
        """Log processing actions with timestamp."""
        self.processing_log.append({
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'action': action,
            'details': details
        })
    
    def _print_header(self, title, width=100, char='='):
        """Print formatted section header."""
        print(char * width)
        print(f"{title:^{width}}")
        print(char * width)
        print()
    
    # ============================================================================
    # STEP 1: LOAD DATA
    # ============================================================================
    
    def load_data(self):
        """Load engineered demographic dataset with strict validation."""
        self._print_header("DEMOGRAPHIC SCALING - LOAD DATA")
        
        try:
            self.df = pd.read_csv(self.input_path)
            
            print(f"Dataset loaded successfully")
            print(f"   Path         : {self.input_path}")
            print(f"   Shape        : {self.df.shape[0]:,} rows x {self.df.shape[1]} columns")
            print(f"   Memory usage : {self.df.memory_usage(deep=True).sum() / 1024:.2f} KB")
            print()
            
            # Strict structure validation
            print(f"STRUCTURE VALIDATION (Expected: 5 columns = 3 clustering + 2 post-hoc):")
            print("-" * 100)
            
            if self.df.shape[1] != self.EXPECTED_TOTAL_COLUMNS:
                print(f"   ERROR: Expected {self.EXPECTED_TOTAL_COLUMNS} columns, found {self.df.shape[1]}")
                print(f"   Actual columns: {list(self.df.columns)}")
                self.log_action("Load data FAILED", "Column count mismatch")
                return False
            
            # Check clustering features
            missing_clustering = [f for f in self.CLUSTERING_FEATURES if f not in self.df.columns]
            if missing_clustering:
                print(f"   ERROR: Missing clustering features: {missing_clustering}")
                self.log_action("Load data FAILED", f"Missing: {missing_clustering}")
                return False
            else:
                print(f"   PASS: Clustering features: {self.CLUSTERING_FEATURES}")
            
            # Check post-hoc features
            missing_posthoc = [f for f in self.POST_HOC_FEATURES if f not in self.df.columns]
            if missing_posthoc:
                print(f"   WARNING: Missing post-hoc features: {missing_posthoc}")
            else:
                print(f"   PASS: Post-hoc features: {self.POST_HOC_FEATURES}")
            
            print("-" * 100)
            print()
            
            self.log_action("Load data", f"Shape: {self.df.shape}, Validation: PASS")
            return True
            
        except FileNotFoundError:
            print(f"ERROR: File not found: {self.input_path}")
            self.log_action("Load data FAILED", "File not found")
            return False
        except Exception as e:
            print(f"ERROR: Failed to load data: {e}")
            self.log_action("Load data FAILED", str(e))
            return False
    
    # ============================================================================
    # STEP 2: VALIDATE DATA QUALITY
    # ============================================================================
    
    def validate_data(self):
        """Validate data quality for clustering features."""
        self._print_header("STEP 1: VALIDATE DATA QUALITY")
        print("Pre-Scaling Data Quality Checks (Clustering Features Only):")
        print()
        
        clustering_data = self.df[self.CLUSTERING_FEATURES]
        
        # Check 1: Missing values
        missing = clustering_data.isnull().sum()
        if missing.sum() == 0:
            print("   PASS: Missing values: None")
        else:
            print("   WARNING: Missing values detected:")
            for col in missing[missing > 0].index:
                print(f"      {col}: {missing[col]} ({missing[col]/len(self.df)*100:.2f}%)")
        
        # Check 2: Infinite values
        inf_count = np.isinf(clustering_data).sum().sum()
        if inf_count == 0:
            print("   PASS: Infinite values: None")
        else:
            print(f"   WARNING: Infinite values: {inf_count}")
        
        # Check 3: Data types
        all_numeric = all(np.issubdtype(clustering_data[col].dtype, np.number) 
                         for col in clustering_data.columns)
        if all_numeric:
            print("   PASS: Data types: All clustering features are numeric")
        else:
            non_numeric = [col for col in clustering_data.columns 
                          if not np.issubdtype(clustering_data[col].dtype, np.number)]
            print(f"   ERROR: Non-numeric clustering features: {non_numeric}")
        
        print()
        self.log_action("Validate data", "Quality checks completed")
    
    # ============================================================================
    # STEP 3: PRE-SCALING STATISTICS
    # ============================================================================
    
    def compute_pre_stats(self):
        """Compute statistics before scaling."""
        self._print_header("STEP 2: PRE-SCALING STATISTICS")
        print("Original feature statistics (Clustering Features Only):")
        print()
        
        for col in self.CLUSTERING_FEATURES:
            if col not in self.df.columns:
                print(f"   WARNING: {col} not found")
                continue
            
            stats = {
                'mean': self.df[col].mean(),
                'std': self.df[col].std(),
                'min': self.df[col].min(),
                'max': self.df[col].max(),
                'median': self.df[col].median(),
                'q1': self.df[col].quantile(0.25),
                'q3': self.df[col].quantile(0.75),
                'iqr': self.df[col].quantile(0.75) - self.df[col].quantile(0.25),
                'skewness': self.df[col].skew(),
                'variance': self.df[col].var()
            }
            self.pre_stats[col] = stats
            
            print(f"{col}:")
            print(f"   Mean      : {stats['mean']:>15,.3f}")
            print(f"   Std       : {stats['std']:>15,.3f}")
            print(f"   Variance  : {stats['variance']:>15,.3f}")
            print(f"   Min       : {stats['min']:>15,.3f}")
            print(f"   Max       : {stats['max']:>15,.3f}")
            print(f"   Median    : {stats['median']:>15,.3f}")
            print(f"   Q1        : {stats['q1']:>15,.3f}")
            print(f"   Q3        : {stats['q3']:>15,.3f}")
            print(f"   IQR       : {stats['iqr']:>15,.3f}")
            print(f"   Skewness  : {stats['skewness']:>15,.3f}")
            print()
        
        self.log_action("Compute pre-scaling stats", f"{len(self.CLUSTERING_FEATURES)} features")
    
    # ============================================================================
    # STEP 4A: APPLY STANDARD SCALER
    # ============================================================================
    
    def apply_standard_scaling(self):
        """Apply StandardScaler to 3 clustering features."""
        self._print_header("STEP 3A: APPLY STANDARDSCALER (3 CLUSTERING FEATURES)")
        print("Applying Z-score normalization to Age, Income, Dependency_Ratio...")
        print(f"   Formula: z = (x - mean) / std")
        print(f"   Target: mean=0, std=1")
        print()
        
        try:
            existing_features = [f for f in self.CLUSTERING_FEATURES if f in self.df.columns]
            
            if len(existing_features) != 3:
                print(f"ERROR: Expected 3 clustering features, found {len(existing_features)}")
                self.log_action("Apply StandardScaler FAILED", "Feature count mismatch")
                return False
            
            # Fit and transform
            scaled_arr = self.standard_scaler.fit_transform(self.df[existing_features])
            
            # Create scaled dataframe
            self.df_standard_scaled = self.df.copy()
            self.df_standard_scaled[existing_features] = scaled_arr
            
            print(f"StandardScaler applied successfully")
            print(f"   Scaled shape: {self.df_standard_scaled.shape}")
            print()
            
            print("StandardScaler parameters (learned from data):")
            print(f"{'Feature':<25} {'Mean (μ)':<20} {'Std (σ)':<20}")
            print("-" * 65)
            for i, col in enumerate(existing_features):
                print(f"{col:<25} {self.standard_scaler.mean_[i]:>19,.3f} {self.standard_scaler.scale_[i]:>19,.3f}")
            print()
            
            self.log_action("Apply StandardScaler", f"Scaled features: {existing_features}")
            return True
            
        except Exception as e:
            print(f"ERROR: StandardScaler failed: {e}")
            self.log_action("Apply StandardScaler FAILED", str(e))
            return False
    
    # ============================================================================
    # STEP 4B: APPLY ROBUST SCALER
    # ============================================================================
    
    def apply_robust_scaling(self):
        """Apply RobustScaler to 3 clustering features."""
        self._print_header("STEP 3B: APPLY ROBUSTSCALER (3 CLUSTERING FEATURES)")
        print("Applying Robust scaling to Age, Income, Dependency_Ratio...")
        print(f"   Formula: z = (x - median) / IQR")
        print(f"   Target: median=0, IQR=1")
        print()
        
        try:
            existing_features = [f for f in self.CLUSTERING_FEATURES if f in self.df.columns]
            
            if len(existing_features) != 3:
                print(f"ERROR: Expected 3 clustering features, found {len(existing_features)}")
                self.log_action("Apply RobustScaler FAILED", "Feature count mismatch")
                return False
            
            # ✅ FIT ROBUST FROM ORIGINAL DATA (NOT STANDARD-SCALED!)
            robust_arr = self.robust_scaler.fit_transform(self.df[existing_features])
            
            # Create scaled dataframe
            self.df_robust_scaled = self.df.copy()
            self.df_robust_scaled[existing_features] = robust_arr
            
            print(f"RobustScaler applied successfully")
            print(f"   Scaled shape: {self.df_robust_scaled.shape}")
            print()
            
            print("RobustScaler parameters (learned from data):")
            print(f"{'Feature':<25} {'Median (center)':<20} {'IQR (scale)':<20}")
            print("-" * 65)
            for i, col in enumerate(existing_features):
                print(f"{col:<25} {self.robust_scaler.center_[i]:>19,.3f} {self.robust_scaler.scale_[i]:>19,.3f}")
            print()
            
            self.log_action("Apply RobustScaler", f"Scaled features: {existing_features}")
            return True
            
        except Exception as e:
            print(f"ERROR: RobustScaler failed: {e}")
            self.log_action("Apply RobustScaler FAILED", str(e))
            return False
    
    # ============================================================================
    # STEP 5A: POST-SCALING STATISTICS (STANDARD)
    # ============================================================================
    
    def compute_post_standard_stats(self):
        """Compute statistics after StandardScaler."""
        self._print_header("STEP 4A: POST-SCALING STATISTICS (STANDARDSCALER)")
        print("StandardScaled feature statistics (Clustering Features Only):")
        print()
        
        extreme_values_summary = []
        
        for col in self.CLUSTERING_FEATURES:
            if col not in self.df_standard_scaled.columns:
                continue
            
            stats = {
                'mean': self.df_standard_scaled[col].mean(),
                'std': self.df_standard_scaled[col].std(ddof=0),
                'min': self.df_standard_scaled[col].min(),
                'max': self.df_standard_scaled[col].max(),
                'median': self.df_standard_scaled[col].median(),
                'skewness': self.df_standard_scaled[col].skew(),
                'variance': self.df_standard_scaled[col].var(ddof=0)
            }
            self.post_standard_stats[col] = stats
            
            print(f"{col}:")
            print(f"   Mean      : {stats['mean']:>15.10f}  (target: 0.000000)")
            print(f"   Std       : {stats['std']:>15.10f}  (target: 1.000000)")
            print(f"   Variance  : {stats['variance']:>15.10f}  (target: 1.000000)")
            print(f"   Min       : {stats['min']:>15,.3f}")
            print(f"   Max       : {stats['max']:>15,.3f}")
            print(f"   Median    : {stats['median']:>15,.3f}")
            print(f"   Skewness  : {stats['skewness']:>15,.3f}")
            
            if abs(stats['min']) > 3 or abs(stats['max']) > 3:
                extreme_values_summary.append((col, stats['min'], stats['max']))
                print(f"   NOTE: Contains extreme values (|z| > 3)")
            print()
        
        if extreme_values_summary:
            print()
            print("EXTREME VALUES SUMMARY (|z-score| > 3):")
            print("-" * 100)
            for col, min_val, max_val in extreme_values_summary:
                outlier_count = ((self.df_standard_scaled[col] < -3) | (self.df_standard_scaled[col] > 3)).sum()
                outlier_pct = outlier_count / len(self.df_standard_scaled) * 100
                print(f"   {col:<25} Range: [{min_val:>8.3f}, {max_val:>8.3f}]  "
                      f"Outliers: {outlier_count:>5} ({outlier_pct:>5.2f}%)")
            print()
        
        self.log_action("Compute post-standard-scaling stats", f"{len(self.CLUSTERING_FEATURES)} features checked")
    
    # ============================================================================
    # STEP 5B: POST-SCALING STATISTICS (ROBUST)
    # ============================================================================
    
    def compute_post_robust_stats(self):
        """Compute statistics after RobustScaler."""
        self._print_header("STEP 4B: POST-SCALING STATISTICS (ROBUSTSCALER)")
        print("RobustScaled feature statistics (Clustering Features Only):")
        print()
        
        extreme_values_summary = []
        
        for col in self.CLUSTERING_FEATURES:
            if col not in self.df_robust_scaled.columns:
                continue
            
            stats = {
                'median': self.df_robust_scaled[col].median(),
                'q1': self.df_robust_scaled[col].quantile(0.25),
                'q3': self.df_robust_scaled[col].quantile(0.75),
                'iqr': self.df_robust_scaled[col].quantile(0.75) - self.df_robust_scaled[col].quantile(0.25),
                'min': self.df_robust_scaled[col].min(),
                'max': self.df_robust_scaled[col].max(),
                'skewness': self.df_robust_scaled[col].skew(),
                'std': self.df_robust_scaled[col].std()
            }
            self.post_robust_stats[col] = stats
            
            print(f"{col}:")
            print(f"   Median    : {stats['median']:>15.10f}  (target: 0.000000)")
            print(f"   IQR       : {stats['iqr']:>15.10f}  (target: 1.000000)")
            print(f"   Q1        : {stats['q1']:>15.10f}")
            print(f"   Q3        : {stats['q3']:>15.10f}")
            print(f"   Min       : {stats['min']:>15,.3f}")
            print(f"   Max       : {stats['max']:>15,.3f}")
            print(f"   Skewness  : {stats['skewness']:>15,.3f}")
            print(f"   Std       : {stats['std']:>15,.3f}")
            
            if abs(stats['min']) > 3 or abs(stats['max']) > 3:
                extreme_values_summary.append((col, stats['min'], stats['max']))
                print(f"   NOTE: Contains extreme values (|z| > 3)")
            print()
        
        if extreme_values_summary:
            print()
            print("EXTREME VALUES SUMMARY (|z-score| > 3):")
            print("-" * 100)
            for col, min_val, max_val in extreme_values_summary:
                outlier_count = ((self.df_robust_scaled[col] < -3) | (self.df_robust_scaled[col] > 3)).sum()
                outlier_pct = outlier_count / len(self.df_robust_scaled) * 100
                print(f"   {col:<25} Range: [{min_val:>8.3f}, {max_val:>8.3f}]  "
                      f"Outliers: {outlier_count:>5} ({outlier_pct:>5.2f}%)")
            print()
        
        self.log_action("Compute post-robust-scaling stats", f"{len(self.CLUSTERING_FEATURES)} features checked")
    
    # ============================================================================
    # STEP 6A: VALIDATE STANDARD SCALING RESULTS
    # ============================================================================
    
    def validate_standard_scaling(self):
        """Validate StandardScaler quality."""
        self._print_header("STEP 5A: VALIDATE STANDARDSCALER RESULTS")
        print("Validation checks (Clustering Features Only):")
        print()
        
        # Check 1: Mean approximately 0
        print("Mean approximately 0 check:")
        all_mean_pass = True
        for col in self.CLUSTERING_FEATURES:
            if col not in self.df_standard_scaled.columns:
                continue
            mean = self.df_standard_scaled[col].mean()
            status = "PASS" if abs(mean) < 1e-10 else "WARNING"
            if abs(mean) >= 1e-10:
                all_mean_pass = False
            print(f"   {col:<25} mean = {mean:>15.12f}  ({status})")
        print()
        
        # Check 2: Std approximately 1
        print("Std approximately 1 check (using ddof=0 to match StandardScaler):")
        all_std_pass = True
        for col in self.CLUSTERING_FEATURES:
            if col not in self.df_standard_scaled.columns:
                continue
            std = self.df_standard_scaled[col].std(ddof=0)
            status = "PASS" if abs(std - 1.0) < 1e-10 else "WARNING"
            if abs(std - 1.0) >= 1e-10:
                all_std_pass = False
            print(f"   {col:<25} std  = {std:>15.12f}  ({status})")
        print()
        
        # Check 3: No missing values
        missing_after = self.df_standard_scaled.isnull().sum().sum()
        if missing_after == 0:
            print("Missing values after scaling: None (PASS)")
        else:
            print(f"Missing values after scaling: {missing_after} (FAILED)")
        print()
        
        # Check 4: Post-hoc features preserved
        print("Post-hoc features preservation check:")
        for col in self.POST_HOC_FEATURES:
            if col in self.df_standard_scaled.columns and col in self.df.columns:
                unchanged = (self.df[col] == self.df_standard_scaled[col]).all()
                status = "PRESERVED" if unchanged else "MODIFIED"
                print(f"   {col:<25} ({status})")
        print()
        
        self.log_action("Validate StandardScaler", 
                       f"Mean check: {all_mean_pass}, Std check: {all_std_pass}")
    
    # ============================================================================
    # STEP 6B: VALIDATE ROBUST SCALING RESULTS
    # ============================================================================
    
    def validate_robust_scaling(self):
        """Validate RobustScaler quality."""
        self._print_header("STEP 5B: VALIDATE ROBUSTSCALER RESULTS")
        print("Validation checks (Clustering Features Only):")
        print()
        
        # Check 1: Median approximately 0
        print("Median approximately 0 check:")
        all_median_pass = True
        for col in self.CLUSTERING_FEATURES:
            if col not in self.df_robust_scaled.columns:
                continue
            median = self.df_robust_scaled[col].median()
            status = "PASS" if abs(median) < 1e-10 else "WARNING"
            if abs(median) >= 1e-10:
                all_median_pass = False
            print(f"   {col:<25} median = {median:>15.12f}  ({status})")
        print()
        
        # Check 2: IQR approximately 1
        print("IQR approximately 1 check:")
        all_iqr_pass = True
        for col in self.CLUSTERING_FEATURES:
            if col not in self.df_robust_scaled.columns:
                continue
            q1 = self.df_robust_scaled[col].quantile(0.25)
            q3 = self.df_robust_scaled[col].quantile(0.75)
            iqr = q3 - q1
            status = "PASS" if abs(iqr - 1.0) < 1e-10 else "WARNING"
            if abs(iqr - 1.0) >= 1e-10:
                all_iqr_pass = False
            print(f"   {col:<25} IQR  = {iqr:>15.12f}  ({status})")
        print()
        
        # Check 3: No missing values
        missing_after = self.df_robust_scaled.isnull().sum().sum()
        if missing_after == 0:
            print("Missing values after scaling: None (PASS)")
        else:
            print(f"Missing values after scaling: {missing_after} (FAILED)")
        print()
        
        # Check 4: Post-hoc features preserved
        print("Post-hoc features preservation check:")
        for col in self.POST_HOC_FEATURES:
            if col in self.df_robust_scaled.columns and col in self.df.columns:
                unchanged = (self.df[col] == self.df_robust_scaled[col]).all()
                status = "PRESERVED" if unchanged else "MODIFIED"
                print(f"   {col:<25} ({status})")
        print()
        
        self.log_action("Validate RobustScaler", 
                       f"Median check: {all_median_pass}, IQR check: {all_iqr_pass}")
    
    # ============================================================================
    # STEP 7A: EXPORT STANDARD SCALED DATASET
    # ============================================================================
    
    def export_standard_dataset(self):
        """Export StandardScaler dataset."""
        self._print_header("STEP 6A: EXPORT STANDARD SCALED DATASET")
        
        try:
            self.df_standard_scaled.to_csv(self.output_standard_csv, index=False)
            
            print(f"Exported StandardScaled dataset successfully:")
            print(f"   Path:  {self.output_standard_csv}")
            print(f"   Shape: {self.df_standard_scaled.shape}")
            print(f"   Size:  {os.path.getsize(self.output_standard_csv) / 1024:.2f} KB")
            print()
            
            print(f"Scaled columns (StandardScaler):")
            for col in self.CLUSTERING_FEATURES:
                if col in self.df_standard_scaled.columns:
                    print(f"   - {col}")
            print()
            
            print(f"Preserved columns (Post-hoc - no scaling):")
            for col in self.POST_HOC_FEATURES:
                if col in self.df_standard_scaled.columns:
                    print(f"   - {col}")
            print()
            
            self.log_action("Export standard scaled dataset", self.output_standard_csv)
            return True
            
        except Exception as e:
            print(f"ERROR: Export failed: {e}")
            self.log_action("Export standard FAILED", str(e))
            return False
    
    # ============================================================================
    # STEP 7B: EXPORT ROBUST SCALED DATASET
    # ============================================================================
    
    def export_robust_dataset(self):
        """Export RobustScaler dataset."""
        self._print_header("STEP 6B: EXPORT ROBUST SCALED DATASET")
        
        try:
            self.df_robust_scaled.to_csv(self.output_robust_csv, index=False)
            
            print(f"Exported RobustScaled dataset successfully:")
            print(f"   Path:  {self.output_robust_csv}")
            print(f"   Shape: {self.df_robust_scaled.shape}")
            print(f"   Size:  {os.path.getsize(self.output_robust_csv) / 1024:.2f} KB")
            print()
            
            print(f"Scaled columns (RobustScaler):")
            for col in self.CLUSTERING_FEATURES:
                if col in self.df_robust_scaled.columns:
                    print(f"   - {col}")
            print()
            
            print(f"Preserved columns (Post-hoc - no scaling):")
            for col in self.POST_HOC_FEATURES:
                if col in self.df_robust_scaled.columns:
                    print(f"   - {col}")
            print()
            
            self.log_action("Export robust scaled dataset", self.output_robust_csv)
            return True
            
        except Exception as e:
            print(f"ERROR: Export failed: {e}")
            self.log_action("Export robust FAILED", str(e))
            return False
    
    # ============================================================================
    # STEP 8A: EXPORT STANDARD SCALING REPORT
    # ============================================================================
    
    def export_standard_report(self):
        """Export StandardScaler detailed report."""
        self._print_header("STEP 7A: EXPORT STANDARD SCALING REPORT")
        
        try:
            with open(self.report_standard_file, 'w', encoding='utf-8') as f:
                # Header
                f.write("=" * 100 + "\n")
                f.write("DEMOGRAPHIC FEATURE SCALING REPORT - STANDARDSCALER\n")
                f.write("=" * 100 + "\n\n")
                
                f.write(f"Generated       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Input           : {self.input_path}\n")
                f.write(f"Output          : {self.output_standard_csv}\n")
                f.write(f"Scaling Method  : StandardScaler (Z-score normalization)\n\n")
                
                # Strategy
                f.write("SCALING STRATEGY:\n")
                f.write("-" * 100 + "\n")
                f.write("- Scale ONLY 3 clustering features: Age, Income, Dependency_Ratio\n")
                f.write("- Post-hoc features (Education_ord, Life_Stage) are PRESERVED (not scaled)\n")
                f.write("- StandardScaler: z = (x - mean) / std → target: mean=0, std=1\n\n")
                
                # Dataset info
                f.write("DATASET INFORMATION:\n")
                f.write("-" * 100 + "\n")
                f.write(f"Shape: {self.df.shape[0]:,} rows x {self.df.shape[1]} columns\n")
                f.write(f"Clustering features: {self.CLUSTERING_FEATURES}\n")
                f.write(f"Post-hoc features: {self.POST_HOC_FEATURES}\n\n")
                
                # Pre-scaling stats
                f.write("PRE-SCALING STATISTICS (Clustering Features Only):\n")
                f.write("-" * 100 + "\n")
                for col, stats in self.pre_stats.items():
                    f.write(f"{col}:\n")
                    for key, val in stats.items():
                        f.write(f"   {key}: {val}\n")
                    f.write("\n")
                
                # StandardScaler parameters
                f.write("STANDARDSCALER PARAMETERS:\n")
                f.write("-" * 100 + "\n")
                for i, col in enumerate(self.CLUSTERING_FEATURES):
                    if col in self.df.columns:
                        f.write(f"{col}:\n")
                        f.write(f"   mean (μ): {float(self.standard_scaler.mean_[i])}\n")
                        f.write(f"   scale (σ): {float(self.standard_scaler.scale_[i])}\n\n")
                
                # Post-scaling stats
                f.write("POST-SCALING STATISTICS (StandardScaler):\n")
                f.write("-" * 100 + "\n")
                for col, stats in self.post_standard_stats.items():
                    f.write(f"{col}:\n")
                    for key, val in stats.items():
                        f.write(f"   {key}: {val}\n")
                    f.write("\n")
                
                # Processing log
                f.write("PROCESSING LOG:\n")
                f.write("-" * 100 + "\n")
                for i, log in enumerate(self.processing_log, 1):
                    f.write(f"{i:>3}. [{log['timestamp']}] {log['action']}\n")
                    if log['details']:
                        f.write(f"     Details: {log['details']}\n")
                
                f.write("\n" + "=" * 100 + "\n")
                f.write("Report generated successfully\n")
                f.write("=" * 100 + "\n")
            
            print(f"Exported StandardScaler report successfully:")
            print(f"   Path: {self.report_standard_file}")
            print(f"   Size: {os.path.getsize(self.report_standard_file) / 1024:.2f} KB")
            print()
            
            self.log_action("Export standard report", self.report_standard_file)
            return True
            
        except Exception as e:
            print(f"ERROR: Report export failed: {e}")
            self.log_action("Export standard report FAILED", str(e))
            return False
    
    # ============================================================================
    # STEP 8B: EXPORT ROBUST SCALING REPORT
    # ============================================================================
    
    def export_robust_report(self):
        """Export RobustScaler detailed report."""
        self._print_header("STEP 7B: EXPORT ROBUST SCALING REPORT")
        
        try:
            with open(self.report_robust_file, 'w', encoding='utf-8') as f:
                # Header
                f.write("=" * 100 + "\n")
                f.write("DEMOGRAPHIC FEATURE SCALING REPORT - ROBUSTSCALER\n")
                f.write("=" * 100 + "\n\n")
                
                f.write(f"Generated       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Input           : {self.input_path}\n")
                f.write(f"Output          : {self.output_robust_csv}\n")
                f.write(f"Scaling Method  : RobustScaler (Median/IQR normalization)\n\n")
                
                # Strategy
                f.write("SCALING STRATEGY:\n")
                f.write("-" * 100 + "\n")
                f.write("- Scale ONLY 3 clustering features: Age, Income, Dependency_Ratio\n")
                f.write("- Post-hoc features (Education_ord, Life_Stage) are PRESERVED (not scaled)\n")
                f.write("- RobustScaler: z = (x - median) / IQR → target: median=0, IQR=1\n")
                f.write("- Less sensitive to outliers compared to StandardScaler\n\n")
                
                # Dataset info
                f.write("DATASET INFORMATION:\n")
                f.write("-" * 100 + "\n")
                f.write(f"Shape: {self.df.shape[0]:,} rows x {self.df.shape[1]} columns\n")
                f.write(f"Clustering features: {self.CLUSTERING_FEATURES}\n")
                f.write(f"Post-hoc features: {self.POST_HOC_FEATURES}\n\n")
                
                # Pre-scaling stats
                f.write("PRE-SCALING STATISTICS (Clustering Features Only):\n")
                f.write("-" * 100 + "\n")
                for col, stats in self.pre_stats.items():
                    f.write(f"{col}:\n")
                    for key, val in stats.items():
                        f.write(f"   {key}: {val}\n")
                    f.write("\n")
                
                # RobustScaler parameters
                f.write("ROBUSTSCALER PARAMETERS:\n")
                f.write("-" * 100 + "\n")
                for i, col in enumerate(self.CLUSTERING_FEATURES):
                    if col in self.df.columns:
                        f.write(f"{col}:\n")
                        f.write(f"   center (median): {float(self.robust_scaler.center_[i])}\n")
                        f.write(f"   scale (IQR): {float(self.robust_scaler.scale_[i])}\n\n")
                
                # Post-scaling stats
                f.write("POST-SCALING STATISTICS (RobustScaler):\n")
                f.write("-" * 100 + "\n")
                for col, stats in self.post_robust_stats.items():
                    f.write(f"{col}:\n")
                    for key, val in stats.items():
                        f.write(f"   {key}: {val}\n")
                    f.write("\n")
                
                # Processing log
                f.write("PROCESSING LOG:\n")
                f.write("-" * 100 + "\n")
                for i, log in enumerate(self.processing_log, 1):
                    f.write(f"{i:>3}. [{log['timestamp']}] {log['action']}\n")
                    if log['details']:
                        f.write(f"     Details: {log['details']}\n")
                
                f.write("\n" + "=" * 100 + "\n")
                f.write("Report generated successfully\n")
                f.write("=" * 100 + "\n")
            
            print(f"Exported RobustScaler report successfully:")
            print(f"   Path: {self.report_robust_file}")
            print(f"   Size: {os.path.getsize(self.report_robust_file) / 1024:.2f} KB")
            print()
            
            self.log_action("Export robust report", self.report_robust_file)
            return True
            
        except Exception as e:
            print(f"ERROR: Report export failed: {e}")
            self.log_action("Export robust report FAILED", str(e))
            return False
    
    # ============================================================================
    # VISUALIZATION METHODS (STANDARD)
    # ============================================================================
    
    def plot_standard_before_after_histogram(self, graph_dir_standard):
        """Plot before/after histograms for StandardScaler."""
        os.makedirs(graph_dir_standard, exist_ok=True)
        
        for col in self.CLUSTERING_FEATURES:
            if col not in self.df.columns:
                continue
            
            try:
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                
                # Before scaling
                axes[0].hist(self.df[col].dropna(), bins=40, color='tab:blue', 
                           alpha=0.7, edgecolor='black')
                axes[0].axvline(self.df[col].mean(), color='red', linestyle='--', 
                              linewidth=2, label=f'Mean={self.df[col].mean():.2f}')
                axes[0].set_title(f'Before Scaling: {col}', fontsize=12, fontweight='bold')
                axes[0].set_xlabel(col, fontsize=11)
                axes[0].set_ylabel('Frequency', fontsize=11)
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
                
                # After scaling
                axes[1].hist(self.df_standard_scaled[col].dropna(), bins=40, color='tab:green', 
                           alpha=0.7, edgecolor='black')
                axes[1].axvline(self.df_standard_scaled[col].mean(), color='red', linestyle='--', 
                              linewidth=2, label=f'Mean={self.df_standard_scaled[col].mean():.6f}')
                axes[1].set_title(f'After Standard Scaling: {col}', fontsize=12, fontweight='bold')
                axes[1].set_xlabel(f'{col} (scaled)', fontsize=11)
                axes[1].set_ylabel('Frequency', fontsize=11)
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
                
                fname = os.path.join(graph_dir_standard, f"{col}_Standard_before_after_hist.png")
                fig.tight_layout()
                fig.savefig(fname, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"   Saved: {fname}")
                
            except Exception as e:
                print(f"   Failed: {col} - {e}")
                plt.close('all')
    
    def plot_standard_outlier_boxplot(self, graph_dir_standard):
        """Plot outlier boxplots for StandardScaler."""
        os.makedirs(graph_dir_standard, exist_ok=True)
        
        for col in self.CLUSTERING_FEATURES:
            if col not in self.df.columns:
                continue
            
            try:
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                
                # Before scaling
                bp1 = axes[0].boxplot(self.df[col].dropna(), vert=True, patch_artist=True)
                bp1['boxes'][0].set_facecolor('lightblue')
                axes[0].set_title(f'Before Scaling: {col}', fontsize=12, fontweight='bold')
                axes[0].set_ylabel(col, fontsize=11)
                axes[0].grid(True, alpha=0.3)
                
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers_before = ((self.df[col] < Q1 - 1.5*IQR) | (self.df[col] > Q3 + 1.5*IQR)).sum()
                axes[0].text(0.5, 0.95, f'Outliers: {outliers_before}', 
                           transform=axes[0].transAxes, ha='center', va='top', 
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                # After scaling
                bp2 = axes[1].boxplot(self.df_standard_scaled[col].dropna(), vert=True, patch_artist=True)
                bp2['boxes'][0].set_facecolor('lightgreen')
                axes[1].set_title(f'After Standard Scaling: {col}', fontsize=12, fontweight='bold')
                axes[1].set_ylabel(f'{col} (scaled)', fontsize=11)
                axes[1].grid(True, alpha=0.3)
                axes[1].axhline(y=3, color='red', linestyle='--', linewidth=1, alpha=0.7, label='z=±3')
                axes[1].axhline(y=-3, color='red', linestyle='--', linewidth=1, alpha=0.7)
                
                extreme_after = ((self.df_standard_scaled[col] < -3) | (self.df_standard_scaled[col] > 3)).sum()
                axes[1].text(0.5, 0.95, f'Extreme (|z|>3): {extreme_after}', 
                           transform=axes[1].transAxes, ha='center', va='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                axes[1].legend(loc='upper right')
                
                fname = os.path.join(graph_dir_standard, f"{col}_Standard_outlier_boxplot.png")
                fig.tight_layout()
                fig.savefig(fname, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"   Saved: {fname}")
                
            except Exception as e:
                print(f"   Failed: {col} - {e}")
                plt.close('all')
    
    def plot_standard_scaling_quality_heatmap(self, graph_dir_standard):
        """Plot StandardScaler quality heatmap."""
        os.makedirs(graph_dir_standard, exist_ok=True)
        
        try:
            data = []
            for col in self.CLUSTERING_FEATURES:
                if col in self.df_standard_scaled.columns:
                    mean_val = self.df_standard_scaled[col].mean()
                    std_val = self.df_standard_scaled[col].std(ddof=0)
                    data.append([mean_val, std_val])
            
            df_heatmap = pd.DataFrame(data, columns=['Mean', 'Std'], 
                                     index=self.CLUSTERING_FEATURES)
            
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(df_heatmap, annot=True, fmt='.6f', cmap='RdYlGn_r',
                       center=0.5, vmin=-0.1, vmax=1.1,
                       cbar_kws={'label': 'Value'}, ax=ax,
                       linewidths=0.5, linecolor='gray')
            
            ax.set_title('StandardScaler Quality Heatmap (3 Clustering Features)\n' + 
                        '(Target: Mean≈0, Std≈1)',
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Metrics', fontsize=12)
            ax.set_ylabel('Features', fontsize=12)
            
            fname = os.path.join(graph_dir_standard, "Standard_scaling_quality_heatmap.png")
            fig.tight_layout()
            fig.savefig(fname, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"   Saved: {fname}")
            
        except Exception as e:
            print(f"   Failed heatmap: {e}")
            plt.close('all')
    
    # ============================================================================
    # VISUALIZATION METHODS (ROBUST)
    # ============================================================================
    
    def plot_robust_before_after_histogram(self, graph_dir_robust):
        """Plot before/after histograms for RobustScaler."""
        os.makedirs(graph_dir_robust, exist_ok=True)
        
        for col in self.CLUSTERING_FEATURES:
            if col not in self.df.columns:
                continue
            
            try:
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                
                # Before scaling
                axes[0].hist(self.df[col].dropna(), bins=40, color='tab:blue', 
                           alpha=0.7, edgecolor='black')
                axes[0].axvline(self.df[col].median(), color='red', linestyle='--', 
                              linewidth=2, label=f'Median={self.df[col].median():.2f}')
                axes[0].set_title(f'Before Scaling: {col}', fontsize=12, fontweight='bold')
                axes[0].set_xlabel(col, fontsize=11)
                axes[0].set_ylabel('Frequency', fontsize=11)
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
                
                # After scaling
                axes[1].hist(self.df_robust_scaled[col].dropna(), bins=40, color='tab:orange', 
                           alpha=0.7, edgecolor='black')
                axes[1].axvline(self.df_robust_scaled[col].median(), color='red', linestyle='--', 
                              linewidth=2, label=f'Median={self.df_robust_scaled[col].median():.6f}')
                axes[1].set_title(f'After Robust Scaling: {col}', fontsize=12, fontweight='bold')
                axes[1].set_xlabel(f'{col} (scaled)', fontsize=11)
                axes[1].set_ylabel('Frequency', fontsize=11)
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
                
                fname = os.path.join(graph_dir_robust, f"{col}_Robust_before_after_hist.png")
                fig.tight_layout()
                fig.savefig(fname, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"   Saved: {fname}")
                
            except Exception as e:
                print(f"   Failed: {col} - {e}")
                plt.close('all')
    
    def plot_robust_outlier_boxplot(self, graph_dir_robust):
        """Plot outlier boxplots for RobustScaler."""
        os.makedirs(graph_dir_robust, exist_ok=True)
        
        for col in self.CLUSTERING_FEATURES:
            if col not in self.df.columns:
                continue
            
            try:
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                
                # Before scaling
                bp1 = axes[0].boxplot(self.df[col].dropna(), vert=True, patch_artist=True)
                bp1['boxes'][0].set_facecolor('lightblue')
                axes[0].set_title(f'Before Scaling: {col}', fontsize=12, fontweight='bold')
                axes[0].set_ylabel(col, fontsize=11)
                axes[0].grid(True, alpha=0.3)
                
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers_before = ((self.df[col] < Q1 - 1.5*IQR) | (self.df[col] > Q3 + 1.5*IQR)).sum()
                axes[0].text(0.5, 0.95, f'Outliers: {outliers_before}', 
                           transform=axes[0].transAxes, ha='center', va='top', 
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                # After scaling
                bp2 = axes[1].boxplot(self.df_robust_scaled[col].dropna(), vert=True, patch_artist=True)
                bp2['boxes'][0].set_facecolor('lightyellow')
                axes[1].set_title(f'After Robust Scaling: {col}', fontsize=12, fontweight='bold')
                axes[1].set_ylabel(f'{col} (scaled)', fontsize=11)
                axes[1].grid(True, alpha=0.3)
                axes[1].axhline(y=3, color='red', linestyle='--', linewidth=1, alpha=0.7, label='z=±3')
                axes[1].axhline(y=-3, color='red', linestyle='--', linewidth=1, alpha=0.7)
                
                extreme_after = ((self.df_robust_scaled[col] < -3) | (self.df_robust_scaled[col] > 3)).sum()
                axes[1].text(0.5, 0.95, f'Extreme (|z|>3): {extreme_after}', 
                           transform=axes[1].transAxes, ha='center', va='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                axes[1].legend(loc='upper right')
                
                fname = os.path.join(graph_dir_robust, f"{col}_Robust_outlier_boxplot.png")
                fig.tight_layout()
                fig.savefig(fname, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"   Saved: {fname}")
                
            except Exception as e:
                print(f"   Failed: {col} - {e}")
                plt.close('all')
    
    def plot_robust_scaling_quality_heatmap(self, graph_dir_robust):
        """Plot RobustScaler quality heatmap."""
        os.makedirs(graph_dir_robust, exist_ok=True)
        
        try:
            data = []
            for col in self.CLUSTERING_FEATURES:
                if col in self.df_robust_scaled.columns:
                    median_val = self.df_robust_scaled[col].median()
                    q1 = self.df_robust_scaled[col].quantile(0.25)
                    q3 = self.df_robust_scaled[col].quantile(0.75)
                    iqr = q3 - q1
                    data.append([median_val, iqr])
            
            df_heatmap = pd.DataFrame(data, columns=['Median', 'IQR'], 
                                     index=self.CLUSTERING_FEATURES)
            
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(df_heatmap, annot=True, fmt='.6f', cmap='RdYlGn_r',
                       center=0.5, vmin=-0.1, vmax=1.1,
                       cbar_kws={'label': 'Value'}, ax=ax,
                       linewidths=0.5, linecolor='gray')
            
            ax.set_title('RobustScaler Quality Heatmap (3 Clustering Features)\n' + 
                        '(Target: Median≈0, IQR≈1)',
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Metrics', fontsize=12)
            ax.set_ylabel('Features', fontsize=12)
            
            fname = os.path.join(graph_dir_robust, "Robust_scaling_quality_heatmap.png")
            fig.tight_layout()
            fig.savefig(fname, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"   Saved: {fname}")
            
        except Exception as e:
            print(f"   Failed heatmap: {e}")
            plt.close('all')
    
    def generate_all_plots(self, graph_dir_standard, graph_dir_robust):
        """Generate all visualization plots for Standard and Robust."""
        # ============================================================================
        # STANDARD SCALING PLOTS
        # ============================================================================
        self._print_header("GENERATING STANDARD SCALING VISUALIZATION PLOTS")
        
        print("Creating before/after histograms (Standard)...")
        self.plot_standard_before_after_histogram(graph_dir_standard)
        
        print("\nCreating outlier detection boxplots (Standard)...")
        self.plot_standard_outlier_boxplot(graph_dir_standard)
        
        print("\nCreating scaling quality heatmap (Standard)...")
        self.plot_standard_scaling_quality_heatmap(graph_dir_standard)
        
        print(f"\nStandard plots saved to: {graph_dir_standard}\n")
        
        # ============================================================================
        # ROBUST SCALING PLOTS
        # ============================================================================
        self._print_header("GENERATING ROBUST SCALING VISUALIZATION PLOTS")
        
        print("Creating before/after histograms (Robust)...")
        self.plot_robust_before_after_histogram(graph_dir_robust)
        
        print("\nCreating outlier detection boxplots (Robust)...")
        self.plot_robust_outlier_boxplot(graph_dir_robust)
        
        print("\nCreating scaling quality heatmap (Robust)...")
        self.plot_robust_scaling_quality_heatmap(graph_dir_robust)
        
        print(f"\nRobust plots saved to: {graph_dir_robust}\n")
        
        self.log_action("Generate visualization plots", f"Standard: {graph_dir_standard}, Robust: {graph_dir_robust}")
    
    # ============================================================================
    # MAIN PIPELINE
    # ============================================================================
    
    def run_scaling(self):
        """Run complete scaling pipeline for both Standard and Robust."""
        print("\n" + "=" * 100)
        print("DEMOGRAPHIC FEATURE SCALING PIPELINE (STANDARD & ROBUST)".center(100))
        print("=" * 100 + "\n")
        
        start_time = datetime.now()
        
        # ============================================================================
        # LOAD & VALIDATE
        # ============================================================================
        if not self.load_data():
            print("\nPIPELINE FAILED: Could not load data")
            return False
        
        self.validate_data()
        self.compute_pre_stats()
        
        # ============================================================================
        # STANDARD SCALER PIPELINE
        # ============================================================================
        if not self.apply_standard_scaling():
            print("\nPIPELINE FAILED: StandardScaler step failed")
            return False
        
        self.compute_post_standard_stats()
        self.validate_standard_scaling()
        
        if not self.export_standard_dataset():
            print("\nPIPELINE FAILED: Could not export standard dataset")
            return False
        
        self.export_standard_report()
        
        # ============================================================================
        # ROBUST SCALER PIPELINE
        # ============================================================================
        if not self.apply_robust_scaling():
            print("\nPIPELINE FAILED: RobustScaler step failed")
            return False
        
        self.compute_post_robust_stats()
        self.validate_robust_scaling()
        
        if not self.export_robust_dataset():
            print("\nPIPELINE FAILED: Could not export robust dataset")
            return False
        
        self.export_robust_report()
        
        # ============================================================================
        # VISUALIZATIONS
        # ============================================================================
        graph_dir_standard = r"C:\Project\Machine_Learning\Machine_Learning\graph\Feature Scaling & Selection_graph\Demographic\Standard"
        graph_dir_robust = r"C:\Project\Machine_Learning\Machine_Learning\graph\Feature Scaling & Selection_graph\Demographic\Robust"
        
        self.generate_all_plots(graph_dir_standard, graph_dir_robust)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        # ============================================================================
        # FINAL SUMMARY
        # ============================================================================
        print("=" * 100)
        print("PIPELINE COMPLETED SUCCESSFULLY".center(100))
        print("=" * 100)
        print(f"Time elapsed: {elapsed:.2f} seconds")
        print(f"\nOutput files:")
        print(f"  - Standard scaled dataset:     {self.output_standard_csv}")
        print(f"  - Robust scaled dataset:       {self.output_robust_csv}")
        print(f"  - Standard report:             {self.report_standard_file}")
        print(f"  - Robust report:               {self.report_robust_file}")
        print(f"  - Standard visualizations:     {graph_dir_standard}")
        print(f"  - Robust visualizations:       {graph_dir_robust}")
        print("=" * 100 + "\n")
        
        return True


# ================================================================================
# CLASS 2: PRODUCT+CHANNEL SCALER (UPDATED - SEPARATE STANDARD & ROBUST OUTPUTS)
# ================================================================================

class ProductChannelScaler:
    """
    Feature Scaling cho Product + Channel Clustering (SEPARATE STANDARD & ROBUST)
    
    INPUT (from Feature_Engineering.py):
    - 12 columns total (CORRECTED from 13):
      * 5 CLUSTERING: Product_HHI, Store_Preference, Web_Engagement, PC1_Total_TotalPurchases, (+ 1 more if available)
      * 7 REFERENCE (POST-HOC): Wine/Meat/Fish/Fruit/Sweet/Gold_Preference, Dominant_Product
    
    SCALING STRATEGY (UPDATED):
    - Dataset 1: StandardScaler ONLY (mean=0, std=1)
    - Dataset 2: RobustScaler ONLY (median=0, IQR=1)
    - Preserve reference features unchanged in both datasets
    - NO sequential scaling (Standard → Robust)
    
    OUTPUT:
    - Customer_Behavior_ProductChannel_Standard_scaled.csv (StandardScaler only)
    - Customer_Behavior_ProductChannel_Robust_scaled.csv (RobustScaler only)
    - ProductChannel_Scaling_Report_Standard.log
    - ProductChannel_Scaling_Report_Robust.log
    - Visualization graphs with Standard/Robust suffixes
    """
    
    def __init__(self, input_path, output_dir, report_dir):
        """
        Initialize ProductChannel Scaler.
        
        Args:
            input_path: Path to engineered ProductChannel CSV
            output_dir: Directory for scaled outputs
            report_dir: Directory for reports
        """
        self.input_path = input_path
        self.output_dir = output_dir
        self.report_dir = report_dir
        
        # Create directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(report_dir, exist_ok=True)
        
        # ============================================================================
        # STANDARD SCALER OUTPUTS
        # ============================================================================
        self.output_standard_csv = os.path.join(output_dir, "Customer_Behavior_ProductChannel_Standard_scaled.csv")
        self.report_standard_file = os.path.join(report_dir, "ProductChannel_Scaling_Report_Standard.log")
        
        # ============================================================================
        # ROBUST SCALER OUTPUTS
        # ============================================================================
        self.output_robust_csv = os.path.join(output_dir, "Customer_Behavior_ProductChannel_Robust_scaled.csv")
        self.report_robust_file = os.path.join(report_dir, "ProductChannel_Scaling_Report_Robust.log")
        
        # Data storage
        self.df = None
        self.df_standard_scaled = None
        self.df_robust_scaled = None
        
        # Scalers
        self.standard_scaler = StandardScaler()
        self.robust_scaler = RobustScaler()
        
        # Statistics (separate for Standard and Robust)
        self.pre_stats = {}
        self.post_standard_stats = {}
        self.post_robust_stats = {}
        self.processing_log = []
        
        # Expected structure (from Feature_Engineering.py)
        # NOTE: Dynamically detected from actual data, but these are typical clustering features
        self.CLUSTERING_FEATURES = None  # Will be auto-detected
        self.REFERENCE_FEATURES = None   # Will be auto-detected
        self.EXPECTED_TOTAL_COLUMNS = None  # Will be detected from actual data
    
    # ============================================================================
    # UTILITY METHODS
    # ============================================================================
    
    def log_action(self, action, details=""):
        """Log processing actions with timestamp."""
        self.processing_log.append({
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'action': action,
            'details': details
        })
    
    def _print_header(self, title, width=100, char='='):
        """Print formatted section header."""
        print(char * width)
        print(f"{title:^{width}}")
        print(char * width)
        print()
    
    def _auto_detect_features(self):
        """Auto-detect clustering vs reference features based on actual data."""
        print("AUTO-DETECTING FEATURE TYPES...")
        print()
        
        # Define potential clustering features (typically derived/transformed)
        potential_clustering = [
            'Product_HHI', 'Store_Preference', 'Web_Engagement',
            'PC1_Total_TotalPurchases', 'PC1_TotalPurchases_Total',
            'Total_Spent_Transformed', 'TotalPurchases'
        ]
        
        # Define potential reference/preference features
        potential_reference = [
            'Wine_Preference', 'Meat_Preference', 'Fish_Preference',
            'Fruit_Preference', 'Sweet_Preference', 'Gold_Preference',
            'Dominant_Product', 'Top_Product_Share'
        ]
        
        # Find actual clustering features in dataset
        clustering = [col for col in self.df.columns if col in potential_clustering]
        
        # All remaining columns are reference features
        reference = [col for col in self.df.columns if col not in clustering]
        
        self.CLUSTERING_FEATURES = clustering
        self.REFERENCE_FEATURES = reference
        self.EXPECTED_TOTAL_COLUMNS = len(clustering) + len(reference)
        
        print(f"Detected {len(clustering)} CLUSTERING features:")
        for i, col in enumerate(clustering, 1):
            print(f"   {i}. {col}")
        print()
        
        print(f"Detected {len(reference)} REFERENCE features:")
        for i, col in enumerate(reference, 1):
            print(f"   {i}. {col}")
        print()
    
    # ============================================================================
    # STEP 1: LOAD DATA
    # ============================================================================
    
    def load_data(self):
        """Load engineered ProductChannel dataset with flexible validation."""
        self._print_header("PRODUCT+CHANNEL SCALING - LOAD DATA")
        
        try:
            self.df = pd.read_csv(self.input_path)
            
            print(f"Dataset loaded successfully")
            print(f"   Path         : {self.input_path}")
            print(f"   Shape        : {self.df.shape[0]:,} rows x {self.df.shape[1]} columns")
            print(f"   Memory usage : {self.df.memory_usage(deep=True).sum() / 1024:.2f} KB")
            print()
            
            # Auto-detect features
            self._auto_detect_features()
            
            # Flexible validation
            print(f"STRUCTURE VALIDATION:")
            print("-" * 100)
            
            if self.df.shape[1] < 5:
                print(f"   ERROR: Expected at least 5 columns, found {self.df.shape[1]}")
                self.log_action("Load data FAILED", "Not enough columns")
                return False
            
            # Check clustering features
            if len(self.CLUSTERING_FEATURES) == 0:
                print(f"   ERROR: No recognized clustering features found")
                print(f"   Available columns: {list(self.df.columns)}")
                self.log_action("Load data FAILED", "No clustering features detected")
                return False
            else:
                print(f"   PASS: Clustering features ({len(self.CLUSTERING_FEATURES)}): {self.CLUSTERING_FEATURES}")
            
            # Check reference features
            if len(self.REFERENCE_FEATURES) == 0:
                print(f"   WARNING: No reference features detected")
            else:
                print(f"   PASS: Reference features ({len(self.REFERENCE_FEATURES)}): {self.REFERENCE_FEATURES}")
            
            print("-" * 100)
            print()
            
            self.log_action("Load data", f"Shape: {self.df.shape}, Features: {len(self.CLUSTERING_FEATURES)} clustering + {len(self.REFERENCE_FEATURES)} reference")
            return True
            
        except FileNotFoundError:
            print(f"ERROR: File not found: {self.input_path}")
            self.log_action("Load data FAILED", "File not found")
            return False
        except Exception as e:
            print(f"ERROR: Failed to load data: {e}")
            self.log_action("Load data FAILED", str(e))
            return False
    
    # ============================================================================
    # STEP 2: VALIDATE DATA QUALITY
    # ============================================================================
    
    def validate_data(self):
        """Validate data quality for clustering features."""
        self._print_header("STEP 1: VALIDATE DATA QUALITY")
        print("Pre-Scaling Data Quality Checks (Clustering Features Only):")
        print()
        
        clustering_data = self.df[self.CLUSTERING_FEATURES]
        
        # Check 1: Missing values
        missing = clustering_data.isnull().sum()
        if missing.sum() == 0:
            print("   PASS: Missing values: None")
        else:
            print("   WARNING: Missing values detected:")
            for col in missing[missing > 0].index:
                print(f"      {col}: {missing[col]} ({missing[col]/len(self.df)*100:.2f}%)")
        
        # Check 2: Infinite values
        inf_count = np.isinf(clustering_data).sum().sum()
        if inf_count == 0:
            print("   PASS: Infinite values: None")
        else:
            print(f"   WARNING: Infinite values: {inf_count}")
        
        # Check 3: Data types
        all_numeric = all(np.issubdtype(clustering_data[col].dtype, np.number) 
                         for col in clustering_data.columns)
        if all_numeric:
            print("   PASS: Data types: All clustering features are numeric")
        else:
            non_numeric = [col for col in clustering_data.columns 
                          if not np.issubdtype(clustering_data[col].dtype, np.number)]
            print(f"   ERROR: Non-numeric clustering features: {non_numeric}")
        
        print()
        self.log_action("Validate data", "Quality checks completed")
    
    # ============================================================================
    # STEP 3: PRE-SCALING STATISTICS
    # ============================================================================
    
    def compute_pre_stats(self):
        """Compute statistics before scaling."""
        self._print_header("STEP 2: PRE-SCALING STATISTICS")
        print("Original feature statistics (Clustering Features Only):")
        print()
        
        for col in self.CLUSTERING_FEATURES:
            if col not in self.df.columns:
                print(f"   WARNING: {col} not found")
                continue
            
            stats = {
                'mean': self.df[col].mean(),
                'std': self.df[col].std(),
                'min': self.df[col].min(),
                'max': self.df[col].max(),
                'median': self.df[col].median(),
                'q1': self.df[col].quantile(0.25),
                'q3': self.df[col].quantile(0.75),
                'iqr': self.df[col].quantile(0.75) - self.df[col].quantile(0.25),
                'skewness': self.df[col].skew(),
                'variance': self.df[col].var()
            }
            self.pre_stats[col] = stats
            
            print(f"{col}:")
            print(f"   Mean      : {stats['mean']:>15,.6f}")
            print(f"   Std       : {stats['std']:>15,.6f}")
            print(f"   Variance  : {stats['variance']:>15,.6f}")
            print(f"   Min       : {stats['min']:>15,.6f}")
            print(f"   Max       : {stats['max']:>15,.6f}")
            print(f"   Median    : {stats['median']:>15,.6f}")
            print(f"   Q1        : {stats['q1']:>15,.6f}")
            print(f"   Q3        : {stats['q3']:>15,.6f}")
            print(f"   IQR       : {stats['iqr']:>15,.6f}")
            print(f"   Skewness  : {stats['skewness']:>15,.6f}")
            print()
        
        self.log_action("Compute pre-scaling stats", f"{len(self.CLUSTERING_FEATURES)} features")
    
    # ============================================================================
    # STEP 4A: APPLY STANDARD SCALER
    # ============================================================================
    
    def apply_standard_scaling(self):
        """Apply StandardScaler to clustering features."""
        self._print_header("STEP 3A: APPLY STANDARDSCALER (CLUSTERING FEATURES)")
        print("Applying Z-score normalization...")
        print(f"   Formula: z = (x - mean) / std")
        print(f"   Target: mean=0, std=1")
        print()
        
        try:
            existing_features = [f for f in self.CLUSTERING_FEATURES if f in self.df.columns]
            
            if len(existing_features) == 0:
                print(f"ERROR: No clustering features found to scale")
                self.log_action("Apply StandardScaler FAILED", "No clustering features")
                return False
            
            # Fit and transform
            scaled_arr = self.standard_scaler.fit_transform(self.df[existing_features])
            
            # Create scaled dataframe
            self.df_standard_scaled = self.df.copy()
            self.df_standard_scaled[existing_features] = scaled_arr
            
            print(f"StandardScaler applied successfully")
            print(f"   Scaled {len(existing_features)} features")
            print(f"   Scaled shape: {self.df_standard_scaled.shape}")
            print()
            
            print("StandardScaler parameters (learned from data):")
            print(f"{'Feature':<35} {'Mean (μ)':<20} {'Std (σ)':<20}")
            print("-" * 75)
            for i, col in enumerate(existing_features):
                print(f"{col:<35} {self.standard_scaler.mean_[i]:>19,.6f} {self.standard_scaler.scale_[i]:>19,.6f}")
            print()
            
            self.log_action("Apply StandardScaler", f"Scaled {len(existing_features)} features")
            return True
            
        except Exception as e:
            print(f"ERROR: StandardScaler failed: {e}")
            self.log_action("Apply StandardScaler FAILED", str(e))
            return False
    
    # ============================================================================
    # STEP 4B: APPLY ROBUST SCALER
    # ============================================================================
    
    def apply_robust_scaling(self):
        """Apply RobustScaler to clustering features."""
        self._print_header("STEP 3B: APPLY ROBUSTSCALER (CLUSTERING FEATURES)")
        print("Applying Robust scaling...")
        print(f"   Formula: z = (x - median) / IQR")
        print(f"   Target: median=0, IQR=1")
        print()
        
        try:
            existing_features = [f for f in self.CLUSTERING_FEATURES if f in self.df.columns]
            
            if len(existing_features) == 0:
                print(f"ERROR: No clustering features found to scale")
                self.log_action("Apply RobustScaler FAILED", "No clustering features")
                return False
            
            # ✅ FIT ROBUST FROM ORIGINAL DATA (NOT STANDARD-SCALED!)
            robust_arr = self.robust_scaler.fit_transform(self.df[existing_features])
            
            # Create scaled dataframe
            self.df_robust_scaled = self.df.copy()
            self.df_robust_scaled[existing_features] = robust_arr
            
            print(f"RobustScaler applied successfully")
            print(f"   Scaled {len(existing_features)} features")
            print(f"   Scaled shape: {self.df_robust_scaled.shape}")
            print()
            
            print("RobustScaler parameters (learned from data):")
            print(f"{'Feature':<35} {'Median (center)':<20} {'IQR (scale)':<20}")
            print("-" * 75)
            for i, col in enumerate(existing_features):
                print(f"{col:<35} {self.robust_scaler.center_[i]:>19,.6f} {self.robust_scaler.scale_[i]:>19,.6f}")
            print()
            
            self.log_action("Apply RobustScaler", f"Scaled {len(existing_features)} features")
            return True
            
        except Exception as e:
            print(f"ERROR: RobustScaler failed: {e}")
            self.log_action("Apply RobustScaler FAILED", str(e))
            return False
    
    # ============================================================================
    # STEP 5A: POST-SCALING STATISTICS (STANDARD)
    # ============================================================================
    
    def compute_post_standard_stats(self):
        """Compute statistics after StandardScaler."""
        self._print_header("STEP 4A: POST-SCALING STATISTICS (STANDARDSCALER)")
        print("StandardScaled feature statistics (Clustering Features Only):")
        print()
        
        extreme_values_summary = []
        
        for col in self.CLUSTERING_FEATURES:
            if col not in self.df_standard_scaled.columns:
                continue
            
            stats = {
                'mean': self.df_standard_scaled[col].mean(),
                'std': self.df_standard_scaled[col].std(ddof=0),
                'min': self.df_standard_scaled[col].min(),
                'max': self.df_standard_scaled[col].max(),
                'median': self.df_standard_scaled[col].median(),
                'skewness': self.df_standard_scaled[col].skew(),
                'variance': self.df_standard_scaled[col].var(ddof=0)
            }
            self.post_standard_stats[col] = stats
            
            print(f"{col}:")
            print(f"   Mean      : {stats['mean']:>15.10f}  (target: 0.000000)")
            print(f"   Std       : {stats['std']:>15.10f}  (target: 1.000000)")
            print(f"   Variance  : {stats['variance']:>15.10f}  (target: 1.000000)")
            print(f"   Min       : {stats['min']:>15,.6f}")
            print(f"   Max       : {stats['max']:>15,.6f}")
            print(f"   Median    : {stats['median']:>15,.6f}")
            print(f"   Skewness  : {stats['skewness']:>15,.6f}")
            
            if abs(stats['min']) > 3 or abs(stats['max']) > 3:
                extreme_values_summary.append((col, stats['min'], stats['max']))
                print(f"   NOTE: Contains extreme values (|z| > 3)")
            print()
        
        if extreme_values_summary:
            print()
            print("EXTREME VALUES SUMMARY (|z-score| > 3):")
            print("-" * 100)
            for col, min_val, max_val in extreme_values_summary:
                outlier_count = ((self.df_standard_scaled[col] < -3) | (self.df_standard_scaled[col] > 3)).sum()
                outlier_pct = outlier_count / len(self.df_standard_scaled) * 100
                print(f"   {col:<35} Range: [{min_val:>8.3f}, {max_val:>8.3f}]  "
                      f"Outliers: {outlier_count:>5} ({outlier_pct:>5.2f}%)")
            print()
        
        self.log_action("Compute post-standard-scaling stats", f"{len(self.CLUSTERING_FEATURES)} features checked")
    
    # ============================================================================
    # STEP 5B: POST-SCALING STATISTICS (ROBUST)
    # ============================================================================
    
    def compute_post_robust_stats(self):
        """Compute statistics after RobustScaler."""
        self._print_header("STEP 4B: POST-SCALING STATISTICS (ROBUSTSCALER)")
        print("RobustScaled feature statistics (Clustering Features Only):")
        print()
        
        extreme_values_summary = []
        
        for col in self.CLUSTERING_FEATURES:
            if col not in self.df_robust_scaled.columns:
                continue
            
            stats = {
                'median': self.df_robust_scaled[col].median(),
                'q1': self.df_robust_scaled[col].quantile(0.25),
                'q3': self.df_robust_scaled[col].quantile(0.75),
                'iqr': self.df_robust_scaled[col].quantile(0.75) - self.df_robust_scaled[col].quantile(0.25),
                'min': self.df_robust_scaled[col].min(),
                'max': self.df_robust_scaled[col].max(),
                'skewness': self.df_robust_scaled[col].skew(),
                'std': self.df_robust_scaled[col].std()
            }
            self.post_robust_stats[col] = stats
            
            print(f"{col}:")
            print(f"   Median    : {stats['median']:>15.10f}  (target: 0.000000)")
            print(f"   IQR       : {stats['iqr']:>15.10f}  (target: 1.000000)")
            print(f"   Q1        : {stats['q1']:>15.10f}")
            print(f"   Q3        : {stats['q3']:>15.10f}")
            print(f"   Min       : {stats['min']:>15,.6f}")
            print(f"   Max       : {stats['max']:>15,.6f}")
            print(f"   Skewness  : {stats['skewness']:>15,.6f}")
            print(f"   Std       : {stats['std']:>15,.6f}")
            
            if abs(stats['min']) > 3 or abs(stats['max']) > 3:
                extreme_values_summary.append((col, stats['min'], stats['max']))
                print(f"   NOTE: Contains extreme values (|z| > 3)")
            print()
        
        if extreme_values_summary:
            print()
            print("EXTREME VALUES SUMMARY (|z-score| > 3):")
            print("-" * 100)
            for col, min_val, max_val in extreme_values_summary:
                outlier_count = ((self.df_robust_scaled[col] < -3) | (self.df_robust_scaled[col] > 3)).sum()
                outlier_pct = outlier_count / len(self.df_robust_scaled) * 100
                print(f"   {col:<35} Range: [{min_val:>8.3f}, {max_val:>8.3f}]  "
                      f"Outliers: {outlier_count:>5} ({outlier_pct:>5.2f}%)")
            print()
        
        self.log_action("Compute post-robust-scaling stats", f"{len(self.CLUSTERING_FEATURES)} features checked")
    
    # ============================================================================
    # STEP 6A: VALIDATE STANDARD SCALING RESULTS
    # ============================================================================
    
    def validate_standard_scaling(self):
        """Validate StandardScaler quality."""
        self._print_header("STEP 5A: VALIDATE STANDARDSCALER RESULTS")
        print("Validation checks (Clustering Features Only):")
        print()
        
        # Check 1: Mean approximately 0
        print("Mean approximately 0 check:")
        all_mean_pass = True
        for col in self.CLUSTERING_FEATURES:
            if col not in self.df_standard_scaled.columns:
                continue
            mean = self.df_standard_scaled[col].mean()
            status = "PASS" if abs(mean) < 1e-10 else "WARNING"
            if abs(mean) >= 1e-10:
                all_mean_pass = False
            print(f"   {col:<35} mean = {mean:>15.12f}  ({status})")
        print()
        
        # Check 2: Std approximately 1
        print("Std approximately 1 check (using ddof=0 to match StandardScaler):")
        all_std_pass = True
        for col in self.CLUSTERING_FEATURES:
            if col not in self.df_standard_scaled.columns:
                continue
            std = self.df_standard_scaled[col].std(ddof=0)
            status = "PASS" if abs(std - 1.0) < 1e-10 else "WARNING"
            if abs(std - 1.0) >= 1e-10:
                all_std_pass = False
            print(f"   {col:<35} std  = {std:>15.12f}  ({status})")
        print()
        
        # Check 3: No missing values
        missing_after = self.df_standard_scaled.isnull().sum().sum()
        if missing_after == 0:
            print("Missing values after scaling: None (PASS)")
        else:
            print(f"Missing values after scaling: {missing_after} (FAILED)")
        print()
        
        # Check 4: Reference features preserved
        if len(self.REFERENCE_FEATURES) > 0:
            print("Reference features preservation check:")
            for col in self.REFERENCE_FEATURES:
                if col in self.df_standard_scaled.columns and col in self.df.columns:
                    unchanged = (self.df[col] == self.df_standard_scaled[col]).all()
                    status = "PRESERVED" if unchanged else "MODIFIED"
                    print(f"   {col:<35} ({status})")
            print()
        
        self.log_action("Validate StandardScaler", 
                       f"Mean check: {all_mean_pass}, Std check: {all_std_pass}")
    
    # ============================================================================
    # STEP 6B: VALIDATE ROBUST SCALING RESULTS
    # ============================================================================
    
    def validate_robust_scaling(self):
        """Validate RobustScaler quality."""
        self._print_header("STEP 5B: VALIDATE ROBUSTSCALER RESULTS")
        print("Validation checks (Clustering Features Only):")
        print()
        
        # Check 1: Median approximately 0
        print("Median approximately 0 check:")
        all_median_pass = True
        for col in self.CLUSTERING_FEATURES:
            if col not in self.df_robust_scaled.columns:
                continue
            median = self.df_robust_scaled[col].median()
            status = "PASS" if abs(median) < 1e-10 else "WARNING"
            if abs(median) >= 1e-10:
                all_median_pass = False
            print(f"   {col:<35} median = {median:>15.12f}  ({status})")
        print()
        
        # Check 2: IQR approximately 1
        print("IQR approximately 1 check:")
        all_iqr_pass = True
        for col in self.CLUSTERING_FEATURES:
            if col not in self.df_robust_scaled.columns:
                continue
            q1 = self.df_robust_scaled[col].quantile(0.25)
            q3 = self.df_robust_scaled[col].quantile(0.75)
            iqr = q3 - q1
            status = "PASS" if abs(iqr - 1.0) < 1e-10 else "WARNING"
            if abs(iqr - 1.0) >= 1e-10:
                all_iqr_pass = False
            print(f"   {col:<35} IQR  = {iqr:>15.12f}  ({status})")
        print()
        
        # Check 3: No missing values
        missing_after = self.df_robust_scaled.isnull().sum().sum()
        if missing_after == 0:
            print("Missing values after scaling: None (PASS)")
        else:
            print(f"Missing values after scaling: {missing_after} (FAILED)")
        print()
        
        # Check 4: Reference features preserved
        if len(self.REFERENCE_FEATURES) > 0:
            print("Reference features preservation check:")
            for col in self.REFERENCE_FEATURES:
                if col in self.df_robust_scaled.columns and col in self.df.columns:
                    unchanged = (self.df[col] == self.df_robust_scaled[col]).all()
                    status = "PRESERVED" if unchanged else "MODIFIED"
                    print(f"   {col:<35} ({status})")
            print()
        
        self.log_action("Validate RobustScaler", 
                       f"Median check: {all_median_pass}, IQR check: {all_iqr_pass}")
    
    # ============================================================================
    # STEP 7A: EXPORT STANDARD SCALED DATASET
    # ============================================================================
    
    def export_standard_dataset(self):
        """Export StandardScaler dataset."""
        self._print_header("STEP 6A: EXPORT STANDARD SCALED DATASET")
        
        try:
            self.df_standard_scaled.to_csv(self.output_standard_csv, index=False)
            
            print(f"Exported StandardScaled dataset successfully:")
            print(f"   Path:  {self.output_standard_csv}")
            print(f"   Shape: {self.df_standard_scaled.shape}")
            print(f"   Size:  {os.path.getsize(self.output_standard_csv) / 1024:.2f} KB")
            print()
            
            print(f"Scaled columns (StandardScaler):")
            for col in self.CLUSTERING_FEATURES:
                if col in self.df_standard_scaled.columns:
                    print(f"   - {col}")
            print()
            
            if len(self.REFERENCE_FEATURES) > 0:
                print(f"Preserved columns (Reference - no scaling):")
                for col in self.REFERENCE_FEATURES:
                    if col in self.df_standard_scaled.columns:
                        print(f"   - {col}")
                print()
            
            self.log_action("Export standard scaled dataset", self.output_standard_csv)
            return True
            
        except Exception as e:
            print(f"ERROR: Export failed: {e}")
            self.log_action("Export standard FAILED", str(e))
            return False
    
    # ============================================================================
    # STEP 7B: EXPORT ROBUST SCALED DATASET
    # ============================================================================
    
    def export_robust_dataset(self):
        """Export RobustScaler dataset."""
        self._print_header("STEP 6B: EXPORT ROBUST SCALED DATASET")
        
        try:
            self.df_robust_scaled.to_csv(self.output_robust_csv, index=False)
            
            print(f"Exported RobustScaled dataset successfully:")
            print(f"   Path:  {self.output_robust_csv}")
            print(f"   Shape: {self.df_robust_scaled.shape}")
            print(f"   Size:  {os.path.getsize(self.output_robust_csv) / 1024:.2f} KB")
            print()
            
            print(f"Scaled columns (RobustScaler):")
            for col in self.CLUSTERING_FEATURES:
                if col in self.df_robust_scaled.columns:
                    print(f"   - {col}")
            print()
            
            if len(self.REFERENCE_FEATURES) > 0:
                print(f"Preserved columns (Reference - no scaling):")
                for col in self.REFERENCE_FEATURES:
                    if col in self.df_robust_scaled.columns:
                        print(f"   - {col}")
                print()
            
            self.log_action("Export robust scaled dataset", self.output_robust_csv)
            return True
            
        except Exception as e:
            print(f"ERROR: Export failed: {e}")
            self.log_action("Export robust FAILED", str(e))
            return False
    
    # ============================================================================
    # STEP 8A: EXPORT STANDARD SCALING REPORT
    # ============================================================================
    
    def export_standard_report(self):
        """Export StandardScaler detailed report."""
        self._print_header("STEP 7A: EXPORT STANDARD SCALING REPORT")
        
        try:
            with open(self.report_standard_file, 'w', encoding='utf-8') as f:
                # Header
                f.write("=" * 100 + "\n")
                f.write("PRODUCT+CHANNEL FEATURE SCALING REPORT - STANDARDSCALER\n")
                f.write("=" * 100 + "\n\n")
                
                f.write(f"Generated       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Input           : {self.input_path}\n")
                f.write(f"Output          : {self.output_standard_csv}\n")
                f.write(f"Scaling Method  : StandardScaler (Z-score normalization)\n\n")
                
                # Strategy
                f.write("SCALING STRATEGY:\n")
                f.write("-" * 100 + "\n")
                f.write(f"- Scale ONLY {len(self.CLUSTERING_FEATURES)} clustering features: {', '.join(self.CLUSTERING_FEATURES)}\n")
                f.write(f"- Preserve {len(self.REFERENCE_FEATURES)} reference features unchanged (for post-hoc interpretation)\n")
                f.write("- StandardScaler: z = (x - mean) / std → target: mean=0, std=1\n\n")
                
                # Dataset info
                f.write("DATASET INFORMATION:\n")
                f.write("-" * 100 + "\n")
                f.write(f"Shape: {self.df.shape[0]:,} rows x {self.df.shape[1]} columns\n")
                f.write(f"Clustering features ({len(self.CLUSTERING_FEATURES)}): {self.CLUSTERING_FEATURES}\n")
                f.write(f"Reference features ({len(self.REFERENCE_FEATURES)}): {self.REFERENCE_FEATURES}\n\n")
                
                # Pre-scaling stats
                f.write("PRE-SCALING STATISTICS (Clustering Features Only):\n")
                f.write("-" * 100 + "\n")
                for col, stats in self.pre_stats.items():
                    f.write(f"{col}:\n")
                    for key, val in stats.items():
                        f.write(f"   {key}: {val}\n")
                    f.write("\n")
                
                # StandardScaler parameters
                f.write("STANDARDSCALER PARAMETERS:\n")
                f.write("-" * 100 + "\n")
                for i, col in enumerate(self.CLUSTERING_FEATURES):
                    if col in self.df.columns:
                        f.write(f"{col}:\n")
                        f.write(f"   mean (μ): {float(self.standard_scaler.mean_[i])}\n")
                        f.write(f"   scale (σ): {float(self.standard_scaler.scale_[i])}\n\n")
                
                # Post-scaling stats
                f.write("POST-SCALING STATISTICS (StandardScaler):\n")
                f.write("-" * 100 + "\n")
                for col, stats in self.post_standard_stats.items():
                    f.write(f"{col}:\n")
                    for key, val in stats.items():
                        f.write(f"   {key}: {val}\n")
                    f.write("\n")
                
                # Processing log
                f.write("PROCESSING LOG:\n")
                f.write("-" * 100 + "\n")
                for i, log in enumerate(self.processing_log, 1):
                    f.write(f"{i:>3}. [{log['timestamp']}] {log['action']}\n")
                    if log['details']:
                        f.write(f"     Details: {log['details']}\n")
                
                f.write("\n" + "=" * 100 + "\n")
                f.write("Report generated successfully\n")
                f.write("=" * 100 + "\n")
            
            print(f"Exported StandardScaler report successfully:")
            print(f"   Path: {self.report_standard_file}")
            print(f"   Size: {os.path.getsize(self.report_standard_file) / 1024:.2f} KB")
            print()
            
            self.log_action("Export standard report", self.report_standard_file)
            return True
            
        except Exception as e:
            print(f"ERROR: Report export failed: {e}")
            self.log_action("Export standard report FAILED", str(e))
            return False
    
    # ============================================================================
    # STEP 8B: EXPORT ROBUST SCALING REPORT
    # ============================================================================
    
    def export_robust_report(self):
        """Export RobustScaler detailed report."""
        self._print_header("STEP 7B: EXPORT ROBUST SCALING REPORT")
        
        try:
            with open(self.report_robust_file, 'w', encoding='utf-8') as f:
                # Header
                f.write("=" * 100 + "\n")
                f.write("PRODUCT+CHANNEL FEATURE SCALING REPORT - ROBUSTSCALER\n")
                f.write("=" * 100 + "\n\n")
                
                f.write(f"Generated       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Input           : {self.input_path}\n")
                f.write(f"Output          : {self.output_robust_csv}\n")
                f.write(f"Scaling Method  : RobustScaler (Median/IQR normalization)\n\n")
                
                # Strategy
                f.write("SCALING STRATEGY:\n")
                f.write("-" * 100 + "\n")
                f.write(f"- Scale ONLY {len(self.CLUSTERING_FEATURES)} clustering features: {', '.join(self.CLUSTERING_FEATURES)}\n")
                f.write(f"- Preserve {len(self.REFERENCE_FEATURES)} reference features unchanged (for post-hoc interpretation)\n")
                f.write("- RobustScaler: z = (x - median) / IQR → target: median=0, IQR=1\n")
                f.write("- Less sensitive to outliers compared to StandardScaler\n\n")
                
                # Dataset info
                f.write("DATASET INFORMATION:\n")
                f.write("-" * 100 + "\n")
                f.write(f"Shape: {self.df.shape[0]:,} rows x {self.df.shape[1]} columns\n")
                f.write(f"Clustering features ({len(self.CLUSTERING_FEATURES)}): {self.CLUSTERING_FEATURES}\n")
                f.write(f"Reference features ({len(self.REFERENCE_FEATURES)}): {self.REFERENCE_FEATURES}\n\n")
                
                # Pre-scaling stats
                f.write("PRE-SCALING STATISTICS (Clustering Features Only):\n")
                f.write("-" * 100 + "\n")
                for col, stats in self.pre_stats.items():
                    f.write(f"{col}:\n")
                    for key, val in stats.items():
                        f.write(f"   {key}: {val}\n")
                    f.write("\n")
                
                # RobustScaler parameters
                f.write("ROBUSTSCALER PARAMETERS:\n")
                f.write("-" * 100 + "\n")
                for i, col in enumerate(self.CLUSTERING_FEATURES):
                    if col in self.df.columns:
                        f.write(f"{col}:\n")
                        f.write(f"   center (median): {float(self.robust_scaler.center_[i])}\n")
                        f.write(f"   scale (IQR): {float(self.robust_scaler.scale_[i])}\n\n")
                
                # Post-scaling stats
                f.write("POST-SCALING STATISTICS (RobustScaler):\n")
                f.write("-" * 100 + "\n")
                for col, stats in self.post_robust_stats.items():
                    f.write(f"{col}:\n")
                    for key, val in stats.items():
                        f.write(f"   {key}: {val}\n")
                    f.write("\n")
                
                # Processing log
                f.write("PROCESSING LOG:\n")
                f.write("-" * 100 + "\n")
                for i, log in enumerate(self.processing_log, 1):
                    f.write(f"{i:>3}. [{log['timestamp']}] {log['action']}\n")
                    if log['details']:
                        f.write(f"     Details: {log['details']}\n")
                
                f.write("\n" + "=" * 100 + "\n")
                f.write("Report generated successfully\n")
                f.write("=" * 100 + "\n")
            
            print(f"Exported RobustScaler report successfully:")
            print(f"   Path: {self.report_robust_file}")
            print(f"   Size: {os.path.getsize(self.report_robust_file) / 1024:.2f} KB")
            print()
            
            self.log_action("Export robust report", self.report_robust_file)
            return True
            
        except Exception as e:
            print(f"ERROR: Report export failed: {e}")
            self.log_action("Export robust report FAILED", str(e))
            return False
    
    # ============================================================================
    # VISUALIZATION METHODS (STANDARD)
    # ============================================================================
    
    def plot_standard_before_after_histogram(self, graph_dir_standard):
        """Plot before/after histograms for StandardScaler."""
        os.makedirs(graph_dir_standard, exist_ok=True)
        
        for col in self.CLUSTERING_FEATURES:
            if col not in self.df.columns:
                continue
            
            try:
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                
                # Before scaling
                axes[0].hist(self.df[col].dropna(), bins=40, color='tab:blue', 
                           alpha=0.7, edgecolor='black')
                axes[0].axvline(self.df[col].mean(), color='red', linestyle='--', 
                              linewidth=2, label=f'Mean={self.df[col].mean():.2f}')
                axes[0].set_title(f'Before Scaling: {col}', fontsize=12, fontweight='bold')
                axes[0].set_xlabel(col, fontsize=11)
                axes[0].set_ylabel('Frequency', fontsize=11)
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
                
                # After scaling
                axes[1].hist(self.df_standard_scaled[col].dropna(), bins=40, color='tab:green', 
                           alpha=0.7, edgecolor='black')
                axes[1].axvline(self.df_standard_scaled[col].mean(), color='red', linestyle='--', 
                              linewidth=2, label=f'Mean={self.df_standard_scaled[col].mean():.6f}')
                axes[1].set_title(f'After Standard Scaling: {col}', fontsize=12, fontweight='bold')
                axes[1].set_xlabel(f'{col} (scaled)', fontsize=11)
                axes[1].set_ylabel('Frequency', fontsize=11)
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
                
                fname = os.path.join(graph_dir_standard, f"{col}_Standard_before_after_hist.png")
                fig.tight_layout()
                fig.savefig(fname, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"   Saved: {fname}")
                
            except Exception as e:
                print(f"   Failed: {col} - {e}")
                plt.close('all')
    
    def plot_standard_outlier_boxplot(self, graph_dir_standard):
        """Plot outlier boxplots for StandardScaler."""
        os.makedirs(graph_dir_standard, exist_ok=True)
        
        for col in self.CLUSTERING_FEATURES:
            if col not in self.df.columns:
                continue
            
            try:
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                
                # Before scaling
                bp1 = axes[0].boxplot(self.df[col].dropna(), vert=True, patch_artist=True)
                bp1['boxes'][0].set_facecolor('lightblue')
                axes[0].set_title(f'Before Scaling: {col}', fontsize=12, fontweight='bold')
                axes[0].set_ylabel(col, fontsize=11)
                axes[0].grid(True, alpha=0.3)
                
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers_before = ((self.df[col] < Q1 - 1.5*IQR) | (self.df[col] > Q3 + 1.5*IQR)).sum()
                axes[0].text(0.5, 0.95, f'Outliers: {outliers_before}', 
                           transform=axes[0].transAxes, ha='center', va='top', 
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                # After scaling
                bp2 = axes[1].boxplot(self.df_standard_scaled[col].dropna(), vert=True, patch_artist=True)
                bp2['boxes'][0].set_facecolor('lightgreen')
                axes[1].set_title(f'After Standard Scaling: {col}', fontsize=12, fontweight='bold')
                axes[1].set_ylabel(f'{col} (scaled)', fontsize=11)
                axes[1].grid(True, alpha=0.3)
                axes[1].axhline(y=3, color='red', linestyle='--', linewidth=1, alpha=0.7, label='z=±3')
                axes[1].axhline(y=-3, color='red', linestyle='--', linewidth=1, alpha=0.7)
                
                extreme_after = ((self.df_standard_scaled[col] < -3) | (self.df_standard_scaled[col] > 3)).sum()
                axes[1].text(0.5, 0.95, f'Extreme (|z|>3): {extreme_after}', 
                           transform=axes[1].transAxes, ha='center', va='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                axes[1].legend(loc='upper right')
                
                fname = os.path.join(graph_dir_standard, f"{col}_Standard_outlier_boxplot.png")
                fig.tight_layout()
                fig.savefig(fname, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"   Saved: {fname}")
                
            except Exception as e:
                print(f"   Failed: {col} - {e}")
                plt.close('all')
    
    def plot_standard_scaling_quality_heatmap(self, graph_dir_standard):
        """Plot StandardScaler quality heatmap."""
        os.makedirs(graph_dir_standard, exist_ok=True)
        
        try:
            data = []
            for col in self.CLUSTERING_FEATURES:
                if col in self.df_standard_scaled.columns:
                    mean_val = self.df_standard_scaled[col].mean()
                    std_val = self.df_standard_scaled[col].std(ddof=0)
                    data.append([mean_val, std_val])
            
            df_heatmap = pd.DataFrame(data, columns=['Mean', 'Std'], 
                                     index=self.CLUSTERING_FEATURES)
            
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(df_heatmap, annot=True, fmt='.6f', cmap='RdYlGn_r',
                       center=0.5, vmin=-0.1, vmax=1.1,
                       cbar_kws={'label': 'Value'}, ax=ax,
                       linewidths=0.5, linecolor='gray')
            
            ax.set_title('StandardScaler Quality Heatmap\n' + 
                        f'({len(self.CLUSTERING_FEATURES)} Clustering Features)\n' +
                        '(Target: Mean≈0, Std≈1)',
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Metrics', fontsize=12)
            ax.set_ylabel('Features', fontsize=12)
            
            fname = os.path.join(graph_dir_standard, "Standard_scaling_quality_heatmap.png")
            fig.tight_layout()
            fig.savefig(fname, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"   Saved: {fname}")
            
        except Exception as e:
            print(f"   Failed heatmap: {e}")
            plt.close('all')
    
    # ============================================================================
    # VISUALIZATION METHODS (ROBUST)
    # ============================================================================
    
    def plot_robust_before_after_histogram(self, graph_dir_robust):
        """Plot before/after histograms for RobustScaler."""
        os.makedirs(graph_dir_robust, exist_ok=True)
        
        for col in self.CLUSTERING_FEATURES:
            if col not in self.df.columns:
                continue
            
            try:
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                
                # Before scaling
                axes[0].hist(self.df[col].dropna(), bins=40, color='tab:blue', 
                           alpha=0.7, edgecolor='black')
                axes[0].axvline(self.df[col].median(), color='red', linestyle='--', 
                              linewidth=2, label=f'Median={self.df[col].median():.2f}')
                axes[0].set_title(f'Before Scaling: {col}', fontsize=12, fontweight='bold')
                axes[0].set_xlabel(col, fontsize=11)
                axes[0].set_ylabel('Frequency', fontsize=11)
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
                
                # After scaling
                axes[1].hist(self.df_robust_scaled[col].dropna(), bins=40, color='tab:orange', 
                           alpha=0.7, edgecolor='black')
                axes[1].axvline(self.df_robust_scaled[col].median(), color='red', linestyle='--', 
                              linewidth=2, label=f'Median={self.df_robust_scaled[col].median():.6f}')
                axes[1].set_title(f'After Robust Scaling: {col}', fontsize=12, fontweight='bold')
                axes[1].set_xlabel(f'{col} (scaled)', fontsize=11)
                axes[1].set_ylabel('Frequency', fontsize=11)
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
                
                fname = os.path.join(graph_dir_robust, f"{col}_Robust_before_after_hist.png")
                fig.tight_layout()
                fig.savefig(fname, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"   Saved: {fname}")
                
            except Exception as e:
                print(f"   Failed: {col} - {e}")
                plt.close('all')
    
    def plot_robust_outlier_boxplot(self, graph_dir_robust):
        """Plot outlier boxplots for RobustScaler."""
        os.makedirs(graph_dir_robust, exist_ok=True)
        
        for col in self.CLUSTERING_FEATURES:
            if col not in self.df.columns:
                continue
            
            try:
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                
                # Before scaling
                bp1 = axes[0].boxplot(self.df[col].dropna(), vert=True, patch_artist=True)
                bp1['boxes'][0].set_facecolor('lightblue')
                axes[0].set_title(f'Before Scaling: {col}', fontsize=12, fontweight='bold')
                axes[0].set_ylabel(col, fontsize=11)
                axes[0].grid(True, alpha=0.3)
                
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers_before = ((self.df[col] < Q1 - 1.5*IQR) | (self.df[col] > Q3 + 1.5*IQR)).sum()
                axes[0].text(0.5, 0.95, f'Outliers: {outliers_before}', 
                           transform=axes[0].transAxes, ha='center', va='top', 
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                # After scaling
                bp2 = axes[1].boxplot(self.df_robust_scaled[col].dropna(), vert=True, patch_artist=True)
                bp2['boxes'][0].set_facecolor('lightyellow')
                axes[1].set_title(f'After Robust Scaling: {col}', fontsize=12, fontweight='bold')
                axes[1].set_ylabel(f'{col} (scaled)', fontsize=11)
                axes[1].grid(True, alpha=0.3)
                axes[1].axhline(y=3, color='red', linestyle='--', linewidth=1, alpha=0.7, label='z=±3')
                axes[1].axhline(y=-3, color='red', linestyle='--', linewidth=1, alpha=0.7)
                
                extreme_after = ((self.df_robust_scaled[col] < -3) | (self.df_robust_scaled[col] > 3)).sum()
                axes[1].text(0.5, 0.95, f'Extreme (|z|>3): {extreme_after}', 
                           transform=axes[1].transAxes, ha='center', va='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                axes[1].legend(loc='upper right')
                
                fname = os.path.join(graph_dir_robust, f"{col}_Robust_outlier_boxplot.png")
                fig.tight_layout()
                fig.savefig(fname, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"   Saved: {fname}")
                
            except Exception as e:
                print(f"   Failed: {col} - {e}")
                plt.close('all')
    
    def plot_robust_scaling_quality_heatmap(self, graph_dir_robust):
        """Plot RobustScaler quality heatmap."""
        os.makedirs(graph_dir_robust, exist_ok=True)
        
        try:
            data = []
            for col in self.CLUSTERING_FEATURES:
                if col in self.df_robust_scaled.columns:
                    median_val = self.df_robust_scaled[col].median()
                    q1 = self.df_robust_scaled[col].quantile(0.25)
                    q3 = self.df_robust_scaled[col].quantile(0.75)
                    iqr = q3 - q1
                    data.append([median_val, iqr])
            
            df_heatmap = pd.DataFrame(data, columns=['Median', 'IQR'], 
                                     index=self.CLUSTERING_FEATURES)
            
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(df_heatmap, annot=True, fmt='.6f', cmap='RdYlGn_r',
                       center=0.5, vmin=-0.1, vmax=1.1,
                       cbar_kws={'label': 'Value'}, ax=ax,
                       linewidths=0.5, linecolor='gray')
            
            ax.set_title('RobustScaler Quality Heatmap\n' + 
                        f'({len(self.CLUSTERING_FEATURES)} Clustering Features)\n' +
                        '(Target: Median≈0, IQR≈1)',
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Metrics', fontsize=12)
            ax.set_ylabel('Features', fontsize=12)
            
            fname = os.path.join(graph_dir_robust, "Robust_scaling_quality_heatmap.png")
            fig.tight_layout()
            fig.savefig(fname, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"   Saved: {fname}")
            
        except Exception as e:
            print(f"   Failed heatmap: {e}")
            plt.close('all')
    
    def generate_all_plots(self, graph_dir_standard, graph_dir_robust):
        """Generate all visualization plots for Standard and Robust."""
        # ============================================================================
        # STANDARD SCALING PLOTS
        # ============================================================================
        self._print_header("GENERATING STANDARD SCALING VISUALIZATION PLOTS")
        
        print("Creating before/after histograms (Standard)...")
        self.plot_standard_before_after_histogram(graph_dir_standard)
        
        print("\nCreating outlier detection boxplots (Standard)...")
        self.plot_standard_outlier_boxplot(graph_dir_standard)
        
        print("\nCreating scaling quality heatmap (Standard)...")
        self.plot_standard_scaling_quality_heatmap(graph_dir_standard)
        
        print(f"\nStandard plots saved to: {graph_dir_standard}\n")
        
        # ============================================================================
        # ROBUST SCALING PLOTS
        # ============================================================================
        self._print_header("GENERATING ROBUST SCALING VISUALIZATION PLOTS")
        
        print("Creating before/after histograms (Robust)...")
        self.plot_robust_before_after_histogram(graph_dir_robust)
        
        print("\nCreating outlier detection boxplots (Robust)...")
        self.plot_robust_outlier_boxplot(graph_dir_robust)
        
        print("\nCreating scaling quality heatmap (Robust)...")
        self.plot_robust_scaling_quality_heatmap(graph_dir_robust)
        
        print(f"\nRobust plots saved to: {graph_dir_robust}\n")
        
        self.log_action("Generate visualization plots", f"Standard: {graph_dir_standard}, Robust: {graph_dir_robust}")
    
    # ============================================================================
    # MAIN PIPELINE
    # ============================================================================
    
    def run_scaling(self):
        """Run complete scaling pipeline for both Standard and Robust."""
        print("\n" + "=" * 100)
        print("PRODUCT+CHANNEL FEATURE SCALING PIPELINE (STANDARD & ROBUST)".center(100))
        print("=" * 100 + "\n")
        
        start_time = datetime.now()
        
        # ============================================================================
        # LOAD & VALIDATE
        # ============================================================================
        if not self.load_data():
            print("\nPIPELINE FAILED: Could not load data")
            return False
        
        self.validate_data()
        self.compute_pre_stats()
        
        # ============================================================================
        # STANDARD SCALER PIPELINE
        # ============================================================================
        if not self.apply_standard_scaling():
            print("\nPIPELINE FAILED: StandardScaler step failed")
            return False
        
        self.compute_post_standard_stats()
        self.validate_standard_scaling()
        
        if not self.export_standard_dataset():
            print("\nPIPELINE FAILED: Could not export standard dataset")
            return False
        
        self.export_standard_report()
        
        # ============================================================================
        # ROBUST SCALER PIPELINE
        # ============================================================================
        if not self.apply_robust_scaling():
            print("\nPIPELINE FAILED: RobustScaler step failed")
            return False
        
        self.compute_post_robust_stats()
        self.validate_robust_scaling()
        
        if not self.export_robust_dataset():
            print("\nPIPELINE FAILED: Could not export robust dataset")
            return False
        
        self.export_robust_report()
        
        # ============================================================================
        # VISUALIZATIONS
        # ============================================================================
        graph_dir_standard = r"C:\Project\Machine_Learning\Machine_Learning\graph\Feature Scaling & Selection_graph\ProductChannel\Standard"
        graph_dir_robust = r"C:\Project\Machine_Learning\Machine_Learning\graph\Feature Scaling & Selection_graph\ProductChannel\Robust"
        
        self.generate_all_plots(graph_dir_standard, graph_dir_robust)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        # ============================================================================
        # FINAL SUMMARY
        # ============================================================================
        print("=" * 100)
        print("PIPELINE COMPLETED SUCCESSFULLY".center(100))
        print("=" * 100)
        print(f"Time elapsed: {elapsed:.2f} seconds")
        print(f"\nOutput files:")
        print(f"  - Standard scaled dataset:     {self.output_standard_csv}")
        print(f"  - Robust scaled dataset:       {self.output_robust_csv}")
        print(f"  - Standard report:             {self.report_standard_file}")
        print(f"  - Robust report:               {self.report_robust_file}")
        print(f"  - Standard visualizations:     {graph_dir_standard}")
        print(f"  - Robust visualizations:       {graph_dir_robust}")
        print("=" * 100 + "\n")
        
        return True



# ================================================================================
# CLASS 3: RFM SCALER (FIXED - SEQUENTIAL SCALING)
# ================================================================================

# ================================================================================
# CLASS 3: RFM SCALER (IMPROVED - SEPARATE STANDARD & ROBUST OUTPUTS)
# ================================================================================

class RFMScaler:
    """
    Feature Scaling cho RFM Clustering (IMPROVED VERSION - SEPARATE OUTPUTS)
    
    INPUT (from Feature_Engineering.py):
    - All RFM features (numeric only)
    
    SCALING STRATEGY:
    - Dataset 1: StandardScaler ONLY (mean=0, std=1)
    - Dataset 2: RobustScaler ONLY (median=0, IQR=1)
    - NO sequential scaling (Standard → Robust)
    
    OUTPUT:
    - Customer_Behavior_RFM_Standard_scaled.csv (StandardScaler only)
    - Customer_Behavior_RFM_Robust_scaled.csv (RobustScaler only)
    - RFM_Scaling_Report_Standard.log
    - RFM_Scaling_Report_Robust.log
    - Visualization graphs with Standard/Robust suffixes
    """
    
    def __init__(self, input_path, output_dir, report_dir):
        """
        Initialize RFM Scaler.
        
        Args:
            input_path: Path to engineered RFM CSV
            output_dir: Directory for scaled outputs
            report_dir: Directory for reports
        """
        self.input_path = input_path
        self.output_dir = output_dir
        self.report_dir = report_dir

        # Create directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(report_dir, exist_ok=True)

        # ============================================================================
        # STANDARD SCALER OUTPUTS
        # ============================================================================
        self.output_standard_csv = os.path.join(output_dir, "Customer_Behavior_RFM_Standard_scaled.csv")
        self.report_standard_file = os.path.join(report_dir, "RFM_Scaling_Report_Standard.log")

        # ============================================================================
        # ROBUST SCALER OUTPUTS
        # ============================================================================
        self.output_robust_csv = os.path.join(output_dir, "Customer_Behavior_RFM_Robust_scaled.csv")
        self.report_robust_file = os.path.join(report_dir, "RFM_Scaling_Report_Robust.log")

        # Data storage
        self.df = None
        self.df_standard_scaled = None
        self.df_robust_scaled = None
        
        # Scalers
        self.standard_scaler = StandardScaler()
        self.robust_scaler = RobustScaler()
        
        # Statistics
        self.pre_stats = {}
        self.post_standard_stats = {}
        self.post_robust_stats = {}
        self.processing_log = []

    def log_action(self, action, details=""):
        """Log processing actions with timestamp."""
        self.processing_log.append({
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'action': action,
            'details': details
        })

    def _print_header(self, title, width=100, char='='):
        """Print formatted section header."""
        print(char * width)
        print(f"{title:^{width}}")
        print(char * width)
        print()

    # ============================================================================
    # STEP 1: LOAD DATA
    # ============================================================================

    def load_data(self):
        """Load engineered RFM dataset."""
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

    # ============================================================================
    # STEP 2: VALIDATE DATA QUALITY
    # ============================================================================

    def validate_data(self):
        """Validate data quality."""
        self._print_header("STEP 1: VALIDATE DATA QUALITY")
        print("Pre-Scaling Data Quality Checks (All Features):")
        print()
        
        # Check 1: Missing values
        missing = self.df.isnull().sum()
        if missing.sum() == 0:
            print("   PASS: Missing values: None")
        else:
            print("   WARNING: Missing values detected:")
            for col in missing[missing > 0].index:
                print(f"      {col}: {missing[col]} ({missing[col]/len(self.df)*100:.2f}%)")
        
        # Check 2: Infinite values
        inf_count = np.isinf(self.df.select_dtypes(include=[np.number])).sum().sum()
        if inf_count == 0:
            print("   PASS: Infinite values: None")
        else:
            print(f"   WARNING: Infinite values: {inf_count}")
        
        # Check 3: Data types
        non_numeric = self.df.select_dtypes(exclude=[np.number]).columns.tolist()
        if not non_numeric:
            print("   PASS: Data types: All numeric")
        else:
            print(f"   WARNING: Non-numeric columns: {non_numeric}")
        
        print()
        self.log_action("Validate data", "Checks completed")

    # ============================================================================
    # STEP 3: PRE-SCALING STATISTICS
    # ============================================================================

    def compute_pre_stats(self):
        """Compute statistics before scaling."""
        self._print_header("STEP 2: PRE-SCALING STATISTICS")
        print("Original feature statistics (All Features):")
        print()
        
        for col in self.df.columns:
            stats = {
                'mean': self.df[col].mean(),
                'std': self.df[col].std(),
                'min': self.df[col].min(),
                'max': self.df[col].max(),
                'median': self.df[col].median(),
                'q1': self.df[col].quantile(0.25),
                'q3': self.df[col].quantile(0.75),
                'iqr': self.df[col].quantile(0.75) - self.df[col].quantile(0.25),
                'skewness': self.df[col].skew(),
                'variance': self.df[col].var()
            }
            self.pre_stats[col] = stats
            
            print(f"{col}:")
            print(f"   Mean      : {stats['mean']:>15,.3f}")
            print(f"   Std       : {stats['std']:>15,.3f}")
            print(f"   Variance  : {stats['variance']:>15,.3f}")
            print(f"   Min       : {stats['min']:>15,.3f}")
            print(f"   Max       : {stats['max']:>15,.3f}")
            print(f"   Median    : {stats['median']:>15,.3f}")
            print(f"   Q1        : {stats['q1']:>15,.3f}")
            print(f"   Q3        : {stats['q3']:>15,.3f}")
            print(f"   IQR       : {stats['iqr']:>15,.3f}")
            print(f"   Skewness  : {stats['skewness']:>15,.3f}")
            print()
        
        self.log_action("Compute pre-scaling stats", f"{len(self.df.columns)} features")

    # ============================================================================
    # STEP 4A: APPLY STANDARD SCALER
    # ============================================================================

    def apply_standard_scaling(self):
        """Apply StandardScaler to all features."""
        self._print_header("STEP 3A: APPLY STANDARDSCALER (ALL FEATURES)")
        print("Applying Z-score normalization to all features...")
        print(f"   Formula: z = (x - mean) / std")
        print(f"   Target: mean=0, std=1")
        print()
        
        try:
            scaled_data = self.standard_scaler.fit_transform(self.df)
            self.df_standard_scaled = pd.DataFrame(scaled_data, columns=self.df.columns, index=self.df.index)
            
            print(f"StandardScaler applied successfully")
            print(f"   Scaled shape: {self.df_standard_scaled.shape}")
            print()
            
            print("StandardScaler parameters (learned from data):")
            print(f"{'Feature':<40} {'Mean (μ)':<20} {'Std (σ)':<20}")
            print("-" * 80)
            for i, col in enumerate(self.df.columns):
                print(f"{col:<40} {self.standard_scaler.mean_[i]:>19,.6f} {self.standard_scaler.scale_[i]:>19,.6f}")
            print()
            
            self.log_action("Apply StandardScaler", f"{len(self.df.columns)} features scaled")
            return True
            
        except Exception as e:
            print(f"ERROR: StandardScaler failed: {e}")
            self.log_action("Apply StandardScaler FAILED", str(e))
            return False

    # ============================================================================
    # STEP 4B: APPLY ROBUST SCALER
    # ============================================================================

    def apply_robust_scaling(self):
        """Apply RobustScaler to all features."""
        self._print_header("STEP 3B: APPLY ROBUSTSCALER (ALL FEATURES)")
        print("Applying Robust scaling to all features...")
        print(f"   Formula: z = (x - median) / IQR")
        print(f"   Target: median=0, IQR=1")
        print()
        
        try:
            # ✅ FIT ROBUST FROM ORIGINAL DATA (NOT STANDARD-SCALED!)
            robust_data = self.robust_scaler.fit_transform(self.df)
            self.df_robust_scaled = pd.DataFrame(robust_data, columns=self.df.columns, index=self.df.index)
            
            print(f"RobustScaler applied successfully")
            print(f"   Scaled shape: {self.df_robust_scaled.shape}")
            print()
            
            print("RobustScaler parameters (learned from data):")
            print(f"{'Feature':<40} {'Median (center)':<20} {'IQR (scale)':<20}")
            print("-" * 80)
            for i, col in enumerate(self.df.columns):
                print(f"{col:<40} {self.robust_scaler.center_[i]:>19,.6f} {self.robust_scaler.scale_[i]:>19,.6f}")
            print()
            
            self.log_action("Apply RobustScaler", f"{len(self.df.columns)} features scaled")
            return True
            
        except Exception as e:
            print(f"ERROR: RobustScaler failed: {e}")
            self.log_action("Apply RobustScaler FAILED", str(e))
            return False

    # ============================================================================
    # STEP 5A: POST-SCALING STATISTICS (STANDARD)
    # ============================================================================

    def compute_post_standard_stats(self):
        """Compute statistics after StandardScaler."""
        self._print_header("STEP 4A: POST-SCALING STATISTICS (STANDARDSCALER)")
        print("StandardScaled feature statistics (All Features):")
        print()
        
        extreme_values_summary = []
        
        for col in self.df_standard_scaled.columns:
            stats = {
                'mean': self.df_standard_scaled[col].mean(),
                'std': self.df_standard_scaled[col].std(ddof=0),
                'min': self.df_standard_scaled[col].min(),
                'max': self.df_standard_scaled[col].max(),
                'median': self.df_standard_scaled[col].median(),
                'skewness': self.df_standard_scaled[col].skew(),
                'variance': self.df_standard_scaled[col].var(ddof=0)
            }
            self.post_standard_stats[col] = stats
            
            print(f"{col}:")
            print(f"   Mean      : {stats['mean']:>15.10f}  (target: 0.000000)")
            print(f"   Std       : {stats['std']:>15.10f}  (target: 1.000000)")
            print(f"   Variance  : {stats['variance']:>15.10f}  (target: 1.000000)")
            print(f"   Min       : {stats['min']:>15,.6f}")
            print(f"   Max       : {stats['max']:>15,.6f}")
            print(f"   Median    : {stats['median']:>15,.6f}")
            print(f"   Skewness  : {stats['skewness']:>15,.6f}")
            
            if abs(stats['min']) > 3 or abs(stats['max']) > 3:
                extreme_values_summary.append((col, stats['min'], stats['max']))
                print(f"   NOTE: Contains extreme values (|z| > 3)")
            print()
        
        if extreme_values_summary:
            print()
            print("EXTREME VALUES SUMMARY (|z-score| > 3):")
            print("-" * 100)
            for col, min_val, max_val in extreme_values_summary:
                outlier_count = ((self.df_standard_scaled[col] < -3) | (self.df_standard_scaled[col] > 3)).sum()
                outlier_pct = outlier_count / len(self.df_standard_scaled) * 100
                print(f"   {col:<40} Range: [{min_val:>7.3f}, {max_val:>7.3f}]  "
                      f"Outliers: {outlier_count} ({outlier_pct:.2f}%)")
            print()
        
        self.log_action("Compute post-standard-scaling stats", f"{len(self.df.columns)} features checked")

    # ============================================================================
    # STEP 5B: POST-SCALING STATISTICS (ROBUST)
    # ============================================================================

    def compute_post_robust_stats(self):
        """Compute statistics after RobustScaler."""
        self._print_header("STEP 4B: POST-SCALING STATISTICS (ROBUSTSCALER)")
        print("RobustScaled feature statistics (All Features):")
        print()
        
        extreme_values_summary = []
        
        for col in self.df_robust_scaled.columns:
            stats = {
                'median': self.df_robust_scaled[col].median(),
                'q1': self.df_robust_scaled[col].quantile(0.25),
                'q3': self.df_robust_scaled[col].quantile(0.75),
                'iqr': self.df_robust_scaled[col].quantile(0.75) - self.df_robust_scaled[col].quantile(0.25),
                'min': self.df_robust_scaled[col].min(),
                'max': self.df_robust_scaled[col].max(),
                'skewness': self.df_robust_scaled[col].skew(),
                'std': self.df_robust_scaled[col].std()
            }
            self.post_robust_stats[col] = stats
            
            print(f"{col}:")
            print(f"   Median    : {stats['median']:>15.10f}  (target: 0.000000)")
            print(f"   IQR       : {stats['iqr']:>15.10f}  (target: 1.000000)")
            print(f"   Q1        : {stats['q1']:>15.10f}")
            print(f"   Q3        : {stats['q3']:>15.10f}")
            print(f"   Min       : {stats['min']:>15,.6f}")
            print(f"   Max       : {stats['max']:>15,.6f}")
            print(f"   Skewness  : {stats['skewness']:>15,.6f}")
            print(f"   Std       : {stats['std']:>15,.6f}")
            
            if abs(stats['min']) > 3 or abs(stats['max']) > 3:
                extreme_values_summary.append((col, stats['min'], stats['max']))
                print(f"   NOTE: Contains extreme values (|z| > 3)")
            print()
        
        if extreme_values_summary:
            print()
            print("EXTREME VALUES SUMMARY (|z-score| > 3):")
            print("-" * 100)
            for col, min_val, max_val in extreme_values_summary:
                outlier_count = ((self.df_robust_scaled[col] < -3) | (self.df_robust_scaled[col] > 3)).sum()
                outlier_pct = outlier_count / len(self.df_robust_scaled) * 100
                print(f"   {col:<40} Range: [{min_val:>7.3f}, {max_val:>7.3f}]  "
                      f"Outliers: {outlier_count} ({outlier_pct:.2f}%)")
            print()
        
        self.log_action("Compute post-robust-scaling stats", f"{len(self.df.columns)} features checked")

    # ============================================================================
    # STEP 6A: VALIDATE STANDARD SCALING RESULTS
    # ============================================================================

    def validate_standard_scaling(self):
        """Validate StandardScaler quality."""
        self._print_header("STEP 5A: VALIDATE STANDARDSCALER RESULTS")
        print("Validation checks (All Features):")
        print()
        
        # Check 1: Mean approximately 0
        print("Mean approximately 0 check:")
        all_mean_pass = True
        for col in self.df_standard_scaled.columns:
            mean = self.df_standard_scaled[col].mean()
            status = "PASS" if abs(mean) < 1e-10 else "WARNING"
            if abs(mean) >= 1e-10:
                all_mean_pass = False
            print(f"   {col:<40} mean = {mean:>15.12f}  ({status})")
        print()
        
        # Check 2: Std approximately 1
        print("Std approximately 1 check (using ddof=0 to match StandardScaler):")
        all_std_pass = True
        for col in self.df_standard_scaled.columns:
            std = self.df_standard_scaled[col].std(ddof=0)
            status = "PASS" if abs(std - 1.0) < 1e-10 else "WARNING"
            if abs(std - 1.0) >= 1e-10:
                all_std_pass = False
            print(f"   {col:<40} std  = {std:>15.12f}  ({status})")
        print()
        
        # Check 3: No missing values
        missing_after = self.df_standard_scaled.isnull().sum().sum()
        if missing_after == 0:
            print("Missing values after scaling: None (PASS)")
        else:
            print(f"Missing values after scaling: {missing_after} (FAILED)")
        print()
        
        self.log_action("Validate StandardScaler", 
                       f"Mean check: {all_mean_pass}, Std check: {all_std_pass}")

    # ============================================================================
    # STEP 6B: VALIDATE ROBUST SCALING RESULTS
    # ============================================================================

    def validate_robust_scaling(self):
        """Validate RobustScaler quality."""
        self._print_header("STEP 5B: VALIDATE ROBUSTSCALER RESULTS")
        print("Validation checks (All Features):")
        print()
        
        # Check 1: Median approximately 0
        print("Median approximately 0 check:")
        all_median_pass = True
        for col in self.df_robust_scaled.columns:
            median = self.df_robust_scaled[col].median()
            status = "PASS" if abs(median) < 1e-10 else "WARNING"
            if abs(median) >= 1e-10:
                all_median_pass = False
            print(f"   {col:<40} median = {median:>15.12f}  ({status})")
        print()
        
        # Check 2: IQR approximately 1
        print("IQR approximately 1 check:")
        all_iqr_pass = True
        for col in self.df_robust_scaled.columns:
            q1 = self.df_robust_scaled[col].quantile(0.25)
            q3 = self.df_robust_scaled[col].quantile(0.75)
            iqr = q3 - q1
            status = "PASS" if abs(iqr - 1.0) < 1e-10 else "WARNING"
            if abs(iqr - 1.0) >= 1e-10:
                all_iqr_pass = False
            print(f"   {col:<40} IQR  = {iqr:>15.12f}  ({status})")
        print()
        
        # Check 3: No missing values
        missing_after = self.df_robust_scaled.isnull().sum().sum()
        if missing_after == 0:
            print("Missing values after scaling: None (PASS)")
        else:
            print(f"Missing values after scaling: {missing_after} (FAILED)")
        print()
        
        self.log_action("Validate RobustScaler", 
                       f"Median check: {all_median_pass}, IQR check: {all_iqr_pass}")

    # ============================================================================
    # STEP 7A: EXPORT STANDARD DATASET
    # ============================================================================

    def export_standard_dataset(self):
        """Export StandardScaler dataset."""
        self._print_header("STEP 6A: EXPORT STANDARD SCALED DATASET")
        
        try:
            self.df_standard_scaled.to_csv(self.output_standard_csv, index=False)
            
            print(f"Exported StandardScaled dataset successfully:")
            print(f"   Path: {self.output_standard_csv}")
            print(f"   Shape: {self.df_standard_scaled.shape}")
            print(f"   Size: {os.path.getsize(self.output_standard_csv) / 1024:.2f} KB")
            print()
            
            self.log_action("Export standard scaled dataset", self.output_standard_csv)
            return True
            
        except Exception as e:
            print(f"ERROR: Export failed: {e}")
            self.log_action("Export standard FAILED", str(e))
            return False

    # ============================================================================
    # STEP 7B: EXPORT ROBUST DATASET
    # ============================================================================

    def export_robust_dataset(self):
        """Export RobustScaler dataset."""
        self._print_header("STEP 6B: EXPORT ROBUST SCALED DATASET")
        
        try:
            self.df_robust_scaled.to_csv(self.output_robust_csv, index=False)
            
            print(f"Exported RobustScaled dataset successfully:")
            print(f"   Path: {self.output_robust_csv}")
            print(f"   Shape: {self.df_robust_scaled.shape}")
            print(f"   Size: {os.path.getsize(self.output_robust_csv) / 1024:.2f} KB")
            print()
            
            self.log_action("Export robust scaled dataset", self.output_robust_csv)
            return True
            
        except Exception as e:
            print(f"ERROR: Export failed: {e}")
            self.log_action("Export robust FAILED", str(e))
            return False

    # ============================================================================
    # STEP 8A: EXPORT STANDARD REPORT
    # ============================================================================

    def export_standard_report(self):
        """Export StandardScaler report."""
        self._print_header("STEP 7A: EXPORT STANDARD SCALING REPORT")
        
        try:
            with open(self.report_standard_file, 'w', encoding='utf-8') as f:
                f.write("=" * 100 + "\n")
                f.write("RFM FEATURE SCALING REPORT - STANDARDSCALER\n")
                f.write("=" * 100 + "\n\n")
                
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Input: {self.input_path}\n")
                f.write(f"Output: {self.output_standard_csv}\n")
                f.write(f"Scaling Method: StandardScaler (Z-score normalization)\n\n")
                
                f.write("SCALING STRATEGY:\n")
                f.write("-" * 100 + "\n")
                f.write("- Scale ALL RFM features\n")
                f.write("- StandardScaler: z = (x - mean) / std → target: mean=0, std=1\n\n")
                
                f.write("DATASET INFORMATION:\n")
                f.write("-" * 100 + "\n")
                f.write(f"Shape: {self.df.shape[0]:,} rows x {self.df.shape[1]} columns\n")
                f.write(f"Features: {list(self.df.columns)}\n\n")
                
                f.write("PRE-SCALING STATISTICS:\n")
                f.write("-" * 100 + "\n")
                for col, stats in self.pre_stats.items():
                    f.write(f"{col}:\n")
                    for key, val in stats.items():
                        f.write(f"   {key}: {val}\n")
                    f.write("\n")
                
                f.write("STANDARDSCALER PARAMETERS:\n")
                f.write("-" * 100 + "\n")
                for i, col in enumerate(self.df.columns):
                    f.write(f"{col}:\n")
                    f.write(f"   mean (μ): {float(self.standard_scaler.mean_[i])}\n")
                    f.write(f"   scale (σ): {float(self.standard_scaler.scale_[i])}\n\n")
                
                f.write("POST-SCALING STATISTICS (StandardScaler):\n")
                f.write("-" * 100 + "\n")
                for col, stats in self.post_standard_stats.items():
                    f.write(f"{col}:\n")
                    for key, val in stats.items():
                        f.write(f"   {key}: {val}\n")
                    f.write("\n")
                
                f.write("PROCESSING LOG:\n")
                f.write("-" * 100 + "\n")
                for i, log in enumerate(self.processing_log, 1):
                    f.write(f"{i:>3}. [{log['timestamp']}] {log['action']}\n")
                    if log['details']:
                        f.write(f"     Details: {log['details']}\n")
                
                f.write("\n" + "=" * 100 + "\n")
                f.write("Report generated successfully\n")
                f.write("=" * 100 + "\n")
            
            print(f"Exported StandardScaler report successfully:")
            print(f"   Path: {self.report_standard_file}")
            print(f"   Size: {os.path.getsize(self.report_standard_file) / 1024:.2f} KB")
            print()
            
            self.log_action("Export standard report", self.report_standard_file)
            return True
            
        except Exception as e:
            print(f"ERROR: Report export failed: {e}")
            self.log_action("Export standard report FAILED", str(e))
            return False

    # ============================================================================
    # STEP 8B: EXPORT ROBUST REPORT
    # ============================================================================

    def export_robust_report(self):
        """Export RobustScaler report."""
        self._print_header("STEP 7B: EXPORT ROBUST SCALING REPORT")
        
        try:
            with open(self.report_robust_file, 'w', encoding='utf-8') as f:
                f.write("=" * 100 + "\n")
                f.write("RFM FEATURE SCALING REPORT - ROBUSTSCALER\n")
                f.write("=" * 100 + "\n\n")
                
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Input: {self.input_path}\n")
                f.write(f"Output: {self.output_robust_csv}\n")
                f.write(f"Scaling Method: RobustScaler (Median/IQR normalization)\n\n")
                
                f.write("SCALING STRATEGY:\n")
                f.write("-" * 100 + "\n")
                f.write("- Scale ALL RFM features\n")
                f.write("- RobustScaler: z = (x - median) / IQR → target: median=0, IQR=1\n")
                f.write("- Less sensitive to outliers compared to StandardScaler\n\n")
                
                f.write("DATASET INFORMATION:\n")
                f.write("-" * 100 + "\n")
                f.write(f"Shape: {self.df.shape[0]:,} rows x {self.df.shape[1]} columns\n")
                f.write(f"Features: {list(self.df.columns)}\n\n")
                
                f.write("PRE-SCALING STATISTICS:\n")
                f.write("-" * 100 + "\n")
                for col, stats in self.pre_stats.items():
                    f.write(f"{col}:\n")
                    for key, val in stats.items():
                        f.write(f"   {key}: {val}\n")
                    f.write("\n")
                
                f.write("ROBUSTSCALER PARAMETERS:\n")
                f.write("-" * 100 + "\n")
                for i, col in enumerate(self.df.columns):
                    f.write(f"{col}:\n")
                    f.write(f"   center (median): {float(self.robust_scaler.center_[i])}\n")
                    f.write(f"   scale (IQR): {float(self.robust_scaler.scale_[i])}\n\n")
                
                f.write("POST-SCALING STATISTICS (RobustScaler):\n")
                f.write("-" * 100 + "\n")
                for col, stats in self.post_robust_stats.items():
                    f.write(f"{col}:\n")
                    for key, val in stats.items():
                        f.write(f"   {key}: {val}\n")
                    f.write("\n")
                
                f.write("PROCESSING LOG:\n")
                f.write("-" * 100 + "\n")
                for i, log in enumerate(self.processing_log, 1):
                    f.write(f"{i:>3}. [{log['timestamp']}] {log['action']}\n")
                    if log['details']:
                        f.write(f"     Details: {log['details']}\n")
                
                f.write("\n" + "=" * 100 + "\n")
                f.write("Report generated successfully\n")
                f.write("=" * 100 + "\n")
            
            print(f"Exported RobustScaler report successfully:")
            print(f"   Path: {self.report_robust_file}")
            print(f"   Size: {os.path.getsize(self.report_robust_file) / 1024:.2f} KB")
            print()
            
            self.log_action("Export robust report", self.report_robust_file)
            return True
            
        except Exception as e:
            print(f"ERROR: Report export failed: {e}")
            self.log_action("Export robust report FAILED", str(e))
            return False

    # ============================================================================
    # VISUALIZATION METHODS (STANDARD)
    # ============================================================================
    
    def plot_standard_before_after_histogram(self, graph_dir_standard):
        """Plot before/after histograms for StandardScaler."""
        os.makedirs(graph_dir_standard, exist_ok=True)
        
        for col in self.df.columns:
            try:
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                
                axes[0].hist(self.df[col].dropna(), bins=40, color='tab:blue', 
                           alpha=0.7, edgecolor='black')
                axes[0].axvline(self.df[col].mean(), color='red', linestyle='--', 
                              linewidth=2, label=f'Mean={self.df[col].mean():.2f}')
                axes[0].set_title(f'Before Scaling: {col}', fontsize=12, fontweight='bold')
                axes[0].set_xlabel(col, fontsize=11)
                axes[0].set_ylabel('Frequency', fontsize=11)
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
                
                axes[1].hist(self.df_standard_scaled[col].dropna(), bins=40, color='tab:green', 
                           alpha=0.7, edgecolor='black')
                axes[1].axvline(self.df_standard_scaled[col].mean(), color='red', linestyle='--', 
                              linewidth=2, label=f'Mean={self.df_standard_scaled[col].mean():.6f}')
                axes[1].set_title(f'After Standard Scaling: {col}', fontsize=12, fontweight='bold')
                axes[1].set_xlabel(f'{col} (scaled)', fontsize=11)
                axes[1].set_ylabel('Frequency', fontsize=11)
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
                
                fname = os.path.join(graph_dir_standard, f"{col}_Standard_before_after_hist.png")
                fig.tight_layout()
                fig.savefig(fname, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"   Saved: {fname}")
                
            except Exception as e:
                print(f"   Failed: {col} - {e}")
                plt.close('all')
    
    def plot_standard_outlier_boxplot(self, graph_dir_standard):
        """Plot outlier boxplots for StandardScaler."""
        os.makedirs(graph_dir_standard, exist_ok=True)
        
        for col in self.df.columns:
            try:
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                
                bp1 = axes[0].boxplot(self.df[col].dropna(), vert=True, patch_artist=True)
                bp1['boxes'][0].set_facecolor('lightblue')
                axes[0].set_title(f'Before Scaling: {col}', fontsize=12, fontweight='bold')
                axes[0].set_ylabel(col, fontsize=11)
                axes[0].grid(True, alpha=0.3)
                
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers_before = ((self.df[col] < Q1 - 1.5*IQR) | (self.df[col] > Q3 + 1.5*IQR)).sum()
                axes[0].text(0.5, 0.95, f'Outliers: {outliers_before}', 
                           transform=axes[0].transAxes, ha='center', va='top', 
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                bp2 = axes[1].boxplot(self.df_standard_scaled[col].dropna(), vert=True, patch_artist=True)
                bp2['boxes'][0].set_facecolor('lightgreen')
                axes[1].set_title(f'After Standard Scaling: {col}', fontsize=12, fontweight='bold')
                axes[1].set_ylabel(f'{col} (scaled)', fontsize=11)
                axes[1].grid(True, alpha=0.3)
                axes[1].axhline(y=3, color='red', linestyle='--', linewidth=1, alpha=0.7, label='z=±3')
                axes[1].axhline(y=-3, color='red', linestyle='--', linewidth=1, alpha=0.7)
                
                extreme_after = ((self.df_standard_scaled[col] < -3) | (self.df_standard_scaled[col] > 3)).sum()
                axes[1].text(0.5, 0.95, f'Extreme (|z|>3): {extreme_after}', 
                           transform=axes[1].transAxes, ha='center', va='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                axes[1].legend(loc='upper right')
                
                fname = os.path.join(graph_dir_standard, f"{col}_Standard_outlier_boxplot.png")
                fig.tight_layout()
                fig.savefig(fname, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"   Saved: {fname}")
                
            except Exception as e:
                print(f"   Failed: {col} - {e}")
                plt.close('all')
    
    def plot_standard_scaling_quality_heatmap(self, graph_dir_standard):
        """Plot StandardScaler quality heatmap."""
        os.makedirs(graph_dir_standard, exist_ok=True)
        
        try:
            data = []
            for col in self.df_standard_scaled.columns:
                mean_val = self.df_standard_scaled[col].mean()
                std_val = self.df_standard_scaled[col].std(ddof=0)
                data.append([mean_val, std_val])
            
            df_heatmap = pd.DataFrame(data, columns=['Mean', 'Std'], 
                                     index=self.df_standard_scaled.columns)
            
            fig, ax = plt.subplots(figsize=(6, max(6, len(self.df.columns) * 0.5)))
            sns.heatmap(df_heatmap, annot=True, fmt='.6f', cmap='RdYlGn_r',
                       center=0.5, vmin=-0.1, vmax=1.1,
                       cbar_kws={'label': 'Value'}, ax=ax,
                       linewidths=0.5, linecolor='gray')
            
            ax.set_title('StandardScaler Quality Heatmap\n(Target: Mean≈0, Std≈1)',
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Metrics', fontsize=12)
            ax.set_ylabel('Features', fontsize=12)
            
            fname = os.path.join(graph_dir_standard, "Standard_scaling_quality_heatmap.png")
            fig.tight_layout()
            fig.savefig(fname, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"   Saved: {fname}")
            
        except Exception as e:
            print(f"   Failed heatmap: {e}")
            plt.close('all')

    # ============================================================================
    # VISUALIZATION METHODS (ROBUST)
    # ============================================================================
    
    def plot_robust_before_after_histogram(self, graph_dir_robust):
        """Plot before/after histograms for RobustScaler."""
        os.makedirs(graph_dir_robust, exist_ok=True)
        
        for col in self.df.columns:
            try:
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                
                axes[0].hist(self.df[col].dropna(), bins=40, color='tab:blue', 
                           alpha=0.7, edgecolor='black')
                axes[0].axvline(self.df[col].median(), color='red', linestyle='--', 
                              linewidth=2, label=f'Median={self.df[col].median():.2f}')
                axes[0].set_title(f'Before Scaling: {col}', fontsize=12, fontweight='bold')
                axes[0].set_xlabel(col, fontsize=11)
                axes[0].set_ylabel('Frequency', fontsize=11)
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
                
                axes[1].hist(self.df_robust_scaled[col].dropna(), bins=40, color='tab:orange', 
                           alpha=0.7, edgecolor='black')
                axes[1].axvline(self.df_robust_scaled[col].median(), color='red', linestyle='--', 
                              linewidth=2, label=f'Median={self.df_robust_scaled[col].median():.6f}')
                axes[1].set_title(f'After Robust Scaling: {col}', fontsize=12, fontweight='bold')
                axes[1].set_xlabel(f'{col} (scaled)', fontsize=11)
                axes[1].set_ylabel('Frequency', fontsize=11)
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
                
                fname = os.path.join(graph_dir_robust, f"{col}_Robust_before_after_hist.png")
                fig.tight_layout()
                fig.savefig(fname, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"   Saved: {fname}")
                
            except Exception as e:
                print(f"   Failed: {col} - {e}")
                plt.close('all')
    
    def plot_robust_outlier_boxplot(self, graph_dir_robust):
        """Plot outlier boxplots for RobustScaler."""
        os.makedirs(graph_dir_robust, exist_ok=True)
        
        for col in self.df.columns:
            try:
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                
                bp1 = axes[0].boxplot(self.df[col].dropna(), vert=True, patch_artist=True)
                bp1['boxes'][0].set_facecolor('lightblue')
                axes[0].set_title(f'Before Scaling: {col}', fontsize=12, fontweight='bold')
                axes[0].set_ylabel(col, fontsize=11)
                axes[0].grid(True, alpha=0.3)
                
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers_before = ((self.df[col] < Q1 - 1.5*IQR) | (self.df[col] > Q3 + 1.5*IQR)).sum()
                axes[0].text(0.5, 0.95, f'Outliers: {outliers_before}', 
                           transform=axes[0].transAxes, ha='center', va='top', 
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                bp2 = axes[1].boxplot(self.df_robust_scaled[col].dropna(), vert=True, patch_artist=True)
                bp2['boxes'][0].set_facecolor('lightyellow')
                axes[1].set_title(f'After Robust Scaling: {col}', fontsize=12, fontweight='bold')
                axes[1].set_ylabel(f'{col} (scaled)', fontsize=11)
                axes[1].grid(True, alpha=0.3)
                axes[1].axhline(y=3, color='red', linestyle='--', linewidth=1, alpha=0.7, label='z=±3')
                axes[1].axhline(y=-3, color='red', linestyle='--', linewidth=1, alpha=0.7)
                
                extreme_after = ((self.df_robust_scaled[col] < -3) | (self.df_robust_scaled[col] > 3)).sum()
                axes[1].text(0.5, 0.95, f'Extreme (|z|>3): {extreme_after}', 
                           transform=axes[1].transAxes, ha='center', va='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                axes[1].legend(loc='upper right')
                
                fname = os.path.join(graph_dir_robust, f"{col}_Robust_outlier_boxplot.png")
                fig.tight_layout()
                fig.savefig(fname, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"   Saved: {fname}")
                
            except Exception as e:
                print(f"   Failed: {col} - {e}")
                plt.close('all')
    
    def plot_robust_scaling_quality_heatmap(self, graph_dir_robust):
        """Plot RobustScaler quality heatmap."""
        os.makedirs(graph_dir_robust, exist_ok=True)
        
        try:
            data = []
            for col in self.df_robust_scaled.columns:
                median_val = self.df_robust_scaled[col].median()
                iqr = self.df_robust_scaled[col].quantile(0.75) - self.df_robust_scaled[col].quantile(0.25)
                data.append([median_val, iqr])
            
            df_heatmap = pd.DataFrame(data, columns=['Median', 'IQR'], 
                                     index=self.df_robust_scaled.columns)
            
            fig, ax = plt.subplots(figsize=(6, max(6, len(self.df.columns) * 0.5)))
            sns.heatmap(df_heatmap, annot=True, fmt='.6f', cmap='RdYlGn_r',
                       center=0.5, vmin=-0.1, vmax=1.1,
                       cbar_kws={'label': 'Value'}, ax=ax,
                       linewidths=0.5, linecolor='gray')
            
            ax.set_title('RobustScaler Quality Heatmap\n(Target: Median≈0, IQR≈1)',
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Metrics', fontsize=12)
            ax.set_ylabel('Features', fontsize=12)
            
            fname = os.path.join(graph_dir_robust, "Robust_scaling_quality_heatmap.png")
            fig.tight_layout()
            fig.savefig(fname, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"   Saved: {fname}")
            
        except Exception as e:
            print(f"   Failed heatmap: {e}")
            plt.close('all')
    
    def generate_all_plots(self, graph_dir_standard, graph_dir_robust):
        """Generate all visualization plots for Standard and Robust."""
        self._print_header("GENERATING STANDARD SCALING VISUALIZATION PLOTS")
        
        print("Creating before/after histograms (Standard)...")
        self.plot_standard_before_after_histogram(graph_dir_standard)
        
        print("\nCreating outlier detection boxplots (Standard)...")
        self.plot_standard_outlier_boxplot(graph_dir_standard)
        
        print("\nCreating scaling quality heatmap (Standard)...")
        self.plot_standard_scaling_quality_heatmap(graph_dir_standard)
        
        print(f"\nStandard plots saved to: {graph_dir_standard}\n")
        
        self._print_header("GENERATING ROBUST SCALING VISUALIZATION PLOTS")
        
        print("Creating before/after histograms (Robust)...")
        self.plot_robust_before_after_histogram(graph_dir_robust)
        
        print("\nCreating outlier detection boxplots (Robust)...")
        self.plot_robust_outlier_boxplot(graph_dir_robust)
        
        print("\nCreating scaling quality heatmap (Robust)...")
        self.plot_robust_scaling_quality_heatmap(graph_dir_robust)
        
        print(f"\nRobust plots saved to: {graph_dir_robust}\n")
        
        self.log_action("Generate visualization plots", f"Standard & Robust plots generated")

    # ============================================================================
    # MAIN PIPELINE
    # ============================================================================
    
    def run_scaling(self):
        """Run complete scaling pipeline for both Standard and Robust."""
        print("\n" + "=" * 100)
        print("RFM FEATURE SCALING PIPELINE (STANDARD & ROBUST)".center(100))
        print("=" * 100 + "\n")
        
        start_time = datetime.now()
        
        if not self.load_data():
            print("\nPIPELINE FAILED: Could not load data")
            return False
        
        self.validate_data()
        self.compute_pre_stats()
        
        # STANDARD SCALER PIPELINE
        if not self.apply_standard_scaling():
            print("\nPIPELINE FAILED: StandardScaler step failed")
            return False
        
        self.compute_post_standard_stats()
        self.validate_standard_scaling()
        
        if not self.export_standard_dataset():
            print("\nPIPELINE FAILED: Could not export standard dataset")
            return False
        
        self.export_standard_report()
        
        # ROBUST SCALER PIPELINE
        if not self.apply_robust_scaling():
            print("\nPIPELINE FAILED: RobustScaler step failed")
            return False
        
        self.compute_post_robust_stats()
        self.validate_robust_scaling()
        
        if not self.export_robust_dataset():
            print("\nPIPELINE FAILED: Could not export robust dataset")
            return False
        
        self.export_robust_report()
        
        # VISUALIZATIONS
        graph_dir_standard = r"C:\Project\Machine_Learning\Machine_Learning\graph\Feature Scaling & Selection_graph\RFM\Standard"
        graph_dir_robust = r"C:\Project\Machine_Learning\Machine_Learning\graph\Feature Scaling & Selection_graph\RFM\Robust"
        
        self.generate_all_plots(graph_dir_standard, graph_dir_robust)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        print("=" * 100)
        print("PIPELINE COMPLETED SUCCESSFULLY".center(100))
        print("=" * 100)
        print(f"Time elapsed: {elapsed:.2f} seconds")
        print(f"\nOutput files:")
        print(f"  - Standard scaled dataset:     {self.output_standard_csv}")
        print(f"  - Robust scaled dataset:       {self.output_robust_csv}")
        print(f"  - Standard report:             {self.report_standard_file}")
        print(f"  - Robust report:               {self.report_robust_file}")
        print(f"  - Standard visualizations:     {graph_dir_standard}")
        print(f"  - Robust visualizations:       {graph_dir_robust}")
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

    # ============================================================================
    # CLASS 1: DEMOGRAPHIC SCALER
    # ============================================================================
    # DemographicScaler tự động tạo cả 2 datasets (Standard & Robust)
    # Không cần tham số use_robust
    demographic_scaler = DemographicScaler(
        demographic_input, 
        dataset_dir, 
        report_dir
    )
    demo_success = demographic_scaler.run_scaling()

    # ============================================================================
    # CLASS 2: PRODUCT+CHANNEL SCALER
    # ============================================================================
    productchannel_scaler = ProductChannelScaler(
        productchannel_input, 
        dataset_dir, 
        report_dir, 
    )
    pc_success = productchannel_scaler.run_scaling()

    # ============================================================================
    # CLASS 3: RFM SCALER
    # ============================================================================
    rfm_scaler = RFMScaler(
        rfm_input, 
        dataset_dir, 
        report_dir, 
    )
    rfm_success = rfm_scaler.run_scaling()

    # ============================================================================
    # FINAL SUMMARY
    # ============================================================================
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