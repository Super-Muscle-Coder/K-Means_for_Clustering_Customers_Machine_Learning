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
from scipy.stats import boxcox, yeojohnson
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.decomposition import PCA

sns.set(style="whitegrid")
warnings.filterwarnings('ignore')


# ================================================================================
# CLASS 1: DEMOGRAPHIC FEATURE ENGINEERING 
# ================================================================================

class DemographicFeatureEngineering:
    """
    Feature Engineering
    Chiến lược: 
    – Loại bỏ biến Ordinal/Nominal khỏi quá trình clustering
    - Biến dùng để clustering: Age, Income, Dependency_Ratio (3 biến liên tục thuần túy)
    - Biến hậu kiểm (post-hoc): Education_ord, Life_Stage (chỉ dùng để phân tích sau khi clustering)
    - Tự động phát hiện phương pháp biến đổi: Box-Cox hoặc Yeo-Johnson
    - Box-Cox dùng cho dữ liệu thuần dương (min > 0)
    - Yeo-Johnson dùng cho dữ liệu có giá trị 0 hoặc âm
    - Loại Education_ord khỏi clustering (chỉ giữ cho phân tích hậu kiểm)
    - Giữ Life_Stage ở dạng phân loại (không one-hot) cho hậu kiểm
    - Không thêm timestamp vào tên file (chế độ ghi đè)
    - Ghi log thống kê đầy đủ trong output
    - Bao gồm biểu đồ trực quan hóa

    Các biến đầu ra cuối cùng (5 cột):
    - Biến clustering (3):
        + Age (liên tục, biến đổi nếu phân phối lệch)
        + Income (liên tục, biến đổi nếu phân phối lệch)
        + Dependency_Ratio (liên tục, biến đổi nếu phân phối lệch)

    - Biến hậu kiểm (2):
        + Education_ord (ordinal 0–4, không dùng cho clustering)
        + Life_Stage (categorical 0–4, không dùng cho clustering)

    Pipeline:
    - Bước 1: Load dữ liệu
    - Bước 2: Tạo biến cơ sở (Age, TotalChildren, Income_per_Family_Member)
    - Bước 3: Tạo Life_Stage (biến phân loại, chỉ dùng hậu kiểm)
    - Bước 4: Tạo Dependency_Ratio (biến dùng cho clustering)
    - Bước 5: Tự động phát hiện và áp dụng biến đổi (Box-Cox hoặc Yeo-Johnson)
    - Bước 6: Chọn 5 biến cuối cùng (3 clustering + 2 hậu kiểm)
    - Bước 7: Kiểm tra/validate các biến
    - Bước 8: Xuất dataset + báo cáo đầy đủ
    - Bước 9: Tạo biểu đồ trực quan hóa
    """
    
    def __init__(self, input_path, output_dir, report_dir):
        self.input_path = input_path
        self.output_dir = output_dir
        self.report_dir = report_dir
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(report_dir, exist_ok=True)
        
        # Fixed filenames (no timestamp)
        self.output_csv = os.path.join(output_dir, "Customer_Behavior_Demographic.csv")
        self.report_file = os.path.join(report_dir, "Demographic_Engineering_Report.txt")
        
        self.df = None
        self.df_engineered = None
        self.feature_stats = {}
        self.processing_log = []
        
        self.REFERENCE_YEAR = 2014
        self.SKEW_THRESHOLD = 0.5  # Lowered from 1.0 for better symmetry
    
    def log_action(self, action, details=""):
        """Ghi log hanh dong."""
        self.processing_log.append({
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'action': action,
            'details': details
        })
    
    def _print_header(self, title, width=100, char='='):
        """In header co format."""
        print(char * width)
        print(f"{title:^{width}}")
        print(char * width)
        print()
    
    # ============================================================================
    # STEP 1: LOAD DATA
    # ============================================================================
    
    def load_data(self):
        """Load cleaned dataset."""
        self._print_header("DEMOGRAPHIC FEATURE ENGINEERING - LOAD DATA")
        
        try:
            self.df = pd.read_csv(self.input_path)
            
            print(f"Dataset loaded successfully")
            print(f"   Shape         : {self.df.shape[0]:,} rows x {self.df.shape[1]} columns")
            print(f"   Memory usage  : {self.df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
            print()
            
            self.log_action("Load data", f"Shape: {self.df.shape}")
            return True
            
        except Exception as e:
            print(f"ERROR loading data: {e}")
            self.log_action("Load data FAILED", str(e))
            return False
    
    # ============================================================================
    # STEP 2: CREATE BASE FEATURES
    # ============================================================================
    
    def create_age_feature(self):
        """Tao Age tu Year_Birth."""
        self._print_header("STEP 1: CREATE AGE FEATURE")
        
        if 'Year_Birth' not in self.df.columns:
            print("WARNING: Year_Birth column not found. Skipping.")
            return
        
        self.df['Age'] = self.REFERENCE_YEAR - self.df['Year_Birth']
        
        # Cap extreme ages
        MAX_AGE = 100
        outliers = (self.df['Age'] > MAX_AGE).sum()
        if outliers > 0:
            print(f"Capping {outliers} ages > {MAX_AGE} to {MAX_AGE}")
            self.df.loc[self.df['Age'] > MAX_AGE, 'Age'] = MAX_AGE
        
        stats = {
            'min': float(self.df['Age'].min()),
            'max': float(self.df['Age'].max()),
            'mean': float(self.df['Age'].mean()),
            'median': float(self.df['Age'].median()),
            'std': float(self.df['Age'].std()),
            'skewness': float(self.df['Age'].skew())
        }
        
        print(f"Feature created  : Age")
        print(f"Formula          : {self.REFERENCE_YEAR} - Year_Birth")
        print(f"Range            : [{stats['min']:.0f}, {stats['max']:.0f}]")
        print(f"Mean             : {stats['mean']:.2f}")
        print(f"Median           : {stats['median']:.2f}")
        print(f"Std Dev          : {stats['std']:.2f}")
        print(f"Skewness         : {stats['skewness']:.3f}")
        print()
        
        self.feature_stats['Age_original'] = stats
        self.log_action("Create Age", f"Range: [{stats['min']:.0f}, {stats['max']:.0f}], Skew: {stats['skewness']:.3f}")
    
    def create_total_children_feature(self):
        """Tao TotalChildren (intermediate)."""
        self._print_header("STEP 2: CREATE TOTALCHILDREN (INTERMEDIATE)")
        
        if 'Kidhome' not in self.df.columns or 'Teenhome' not in self.df.columns:
            print("WARNING: Kidhome or Teenhome not found. Skipping.")
            return
        
        self.df['TotalChildren'] = self.df['Kidhome'] + self.df['Teenhome']
        
        dist = self.df['TotalChildren'].value_counts().sort_index()
        
        print(f"Feature created  : TotalChildren")
        print(f"Formula          : Kidhome + Teenhome")
        print()
        print(f"Distribution:")
        for num, count in dist.items():
            pct = count / len(self.df) * 100
            bar = '█' * int(pct / 2)
            print(f"   {num} children  : {count:>5,} ({pct:>5.1f}%) {bar}")
        print()
        
        self.feature_stats['TotalChildren'] = {
            'distribution': dist.to_dict()
        }
        
        self.log_action("Create TotalChildren", f"Distribution: {dist.to_dict()}")
    
    def create_income_per_family_member(self):
        """Tao Income_per_Family_Member (intermediate)."""
        self._print_header("STEP 3: CREATE INCOME_PER_FAMILY_MEMBER (INTERMEDIATE)")
        
        if 'Income' not in self.df.columns or 'TotalChildren' not in self.df.columns:
            print("WARNING: Required columns not found. Skipping.")
            return
        
        family_size = 2 + self.df['TotalChildren']
        self.df['Income_per_Family_Member'] = self.df['Income'] / family_size
        
        stats = {
            'min': float(self.df['Income_per_Family_Member'].min()),
            'max': float(self.df['Income_per_Family_Member'].max()),
            'mean': float(self.df['Income_per_Family_Member'].mean()),
            'median': float(self.df['Income_per_Family_Member'].median()),
            'std': float(self.df['Income_per_Family_Member'].std()),
            'skewness': float(self.df['Income_per_Family_Member'].skew())
        }
        
        print(f"Feature created  : Income_per_Family_Member")
        print(f"Formula          : Income / (2 + TotalChildren)")
        print(f"Range            : [{stats['min']:,.0f}, {stats['max']:,.0f}]")
        print(f"Mean             : {stats['mean']:,.2f}")
        print(f"Median           : {stats['median']:,.2f}")
        print(f"Skewness         : {stats['skewness']:.3f}")
        print()
        
        self.feature_stats['Income_per_Family_Member'] = stats
        self.log_action("Create Income_per_Family_Member", f"Mean: {stats['mean']:,.0f}, Skew: {stats['skewness']:.3f}")
    
    def create_marital_encoded(self):
        """Tao Marital_Encoded (intermediate for Life_Stage)."""
        self._print_header("STEP 4: CREATE MARITAL_ENCODED (INTERMEDIATE)")
        
        marital_mapping = {
            'Marital_Single': 0,
            'Marital_Divorced': 0,
            'Marital_Widow': 0,
            'Marital_Together': 1,
            'Marital_Married': 2
        }
        
        existing_cols = [col for col in marital_mapping.keys() if col in self.df.columns]
        
        if not existing_cols:
            print("WARNING: No Marital_* columns found. Skipping.")
            return
        
        self.df['Marital_Encoded'] = 0
        for col, value in marital_mapping.items():
            if col in self.df.columns:
                self.df['Marital_Encoded'] += self.df[col] * value
        
        dist = self.df['Marital_Encoded'].value_counts().sort_index()
        
        print(f"Feature created  : Marital_Encoded")
        print()
        print(f"Mapping:")
        print(f"   0  : Single/Divorced/Widow")
        print(f"   1  : Together")
        print(f"   2  : Married")
        print()
        print(f"Distribution:")
        for code, count in dist.items():
            pct = count / len(self.df) * 100
            bar = '█' * int(pct / 2)
            print(f"   Code {code}  : {count:>5,} ({pct:>5.1f}%) {bar}")
        print()
        
        self.feature_stats['Marital_Encoded'] = {
            'distribution': dist.to_dict()
        }
        
        self.log_action("Create Marital_Encoded", f"Distribution: {dist.to_dict()}")
    
    # ============================================================================
    # STEP 3: CREATE LIFE_STAGE (POST-HOC FEATURE)
    # ============================================================================
    
    def create_life_stage(self):
        """Tao Life_Stage (categorical, for post-hoc analysis only)."""
        self._print_header("STEP 5: CREATE LIFE_STAGE (POST-HOC CATEGORICAL)")
        
        if 'Age' not in self.df.columns or 'TotalChildren' not in self.df.columns or 'Marital_Encoded' not in self.df.columns:
            print("WARNING: Required columns not found. Skipping.")
            return
        
        self.df['Life_Stage'] = 0
        
        # Define life stage rules
        young_single = (self.df['Age'] < 35) & (self.df['TotalChildren'] == 0) & (self.df['Marital_Encoded'] == 0)
        young_family = (self.df['Age'] < 45) & (self.df['TotalChildren'] > 0) & (self.df['Marital_Encoded'] > 0)
        mature_family = (self.df['Age'] >= 45) & (self.df['Age'] < 60) & (self.df['TotalChildren'] > 0)
        empty_nest = (self.df['Age'] >= 60)
        single_parent = (self.df['TotalChildren'] > 0) & (self.df['Marital_Encoded'] == 0)
        
        self.df.loc[young_single, 'Life_Stage'] = 0
        self.df.loc[young_family, 'Life_Stage'] = 1
        self.df.loc[mature_family, 'Life_Stage'] = 2
        self.df.loc[empty_nest, 'Life_Stage'] = 3
        self.df.loc[single_parent, 'Life_Stage'] = 4
        
        dist = self.df['Life_Stage'].value_counts().sort_index()
        
        print(f"Feature created  : Life_Stage (CATEGORICAL - NOT FOR CLUSTERING)")
        print(f"Formula          : Composite of Age x Marital x Children")
        print(f"Purpose          : POST-HOC ANALYSIS after clustering")
        print()
        print(f"Life Stage Distribution:")
        stage_names = ['Young Single', 'Young Family', 'Mature Family', 'Empty Nest', 'Single Parent']
        for stage, count in dist.items():
            pct = count / len(self.df) * 100
            bar = '█' * int(pct / 2)
            print(f"   {stage} ({stage_names[int(stage)]:15})  : {count:>5,} ({pct:>5.1f}%) {bar}")
        print()
        
        self.feature_stats['Life_Stage'] = {
            'distribution': dist.to_dict(),
            'stage_names': stage_names,
            'usage': 'POST-HOC only (excluded from clustering)'
        }
        
        self.log_action("Create Life_Stage", f"Distribution: {dist.to_dict()} (POST-HOC)")
    
    # ============================================================================
    # STEP 4: CREATE DEPENDENCY_RATIO (CLUSTERING FEATURE)
    # ============================================================================
    
    def create_dependency_ratio(self):
        """Tao Dependency_Ratio (clustering feature)."""
        self._print_header("STEP 6: CREATE DEPENDENCY_RATIO (CLUSTERING FEATURE)")
        
        if 'TotalChildren' not in self.df.columns or 'Income_per_Family_Member' not in self.df.columns:
            print("WARNING: Required columns not found. Skipping.")
            return
        
        # Formula: (TotalChildren * 10000) / Income_per_Family_Member
        self.df['Dependency_Ratio'] = (self.df['TotalChildren'] * 10000) / (self.df['Income_per_Family_Member'] + 1e-6)
        
        # Clip at 99th percentile
        upper = self.df['Dependency_Ratio'].quantile(0.99)
        outliers = (self.df['Dependency_Ratio'] > upper).sum()
        if outliers > 0:
            print(f"Clipping {outliers} extreme values at 99th percentile (>{upper:.3f})")
            self.df['Dependency_Ratio'] = self.df['Dependency_Ratio'].clip(upper=upper)
        
        stats = {
            'min': float(self.df['Dependency_Ratio'].min()),
            'max': float(self.df['Dependency_Ratio'].max()),
            'mean': float(self.df['Dependency_Ratio'].mean()),
            'median': float(self.df['Dependency_Ratio'].median()),
            'std': float(self.df['Dependency_Ratio'].std()),
            'skewness': float(self.df['Dependency_Ratio'].skew()),
            'variance': float(self.df['Dependency_Ratio'].var())
        }
        
        # Kiem tra co gia tri 0 khong
        zero_count = (self.df['Dependency_Ratio'] == 0).sum()
        zero_pct = zero_count / len(self.df) * 100
        
        print(f"Feature created  : Dependency_Ratio (CLUSTERING FEATURE)")
        print(f"Formula          : (TotalChildren x 10000) / Income_per_Family_Member")
        print(f"Clipping         : Values > 99th percentile clipped to {upper:.3f}")
        print(f"Range            : [{stats['min']:.3f}, {stats['max']:.3f}]")
        print(f"Mean             : {stats['mean']:.3f}")
        print(f"Median           : {stats['median']:.3f}")
        print(f"Std Dev          : {stats['std']:.3f}")
        print(f"Skewness         : {stats['skewness']:.3f}")
        print(f"Variance         : {stats['variance']:.3f}")
        print(f"Zero count       : {zero_count} ({zero_pct:.1f}%)")
        print()
        
        self.feature_stats['Dependency_Ratio_original'] = stats
        self.log_action("Create Dependency_Ratio", f"Mean: {stats['mean']:.3f}, Skew: {stats['skewness']:.3f}, Zero: {zero_count}")
    
    # ============================================================================
    # STEP 5: AUTO-DETECT AND APPLY TRANSFORM
    # ============================================================================
    
    def detect_and_apply_transform(self, feature_data, feature_name):
        """
        Tu dong phat hien va ap dung transform method phu hop.
        
        Args:
            feature_data: Series du lieu can transform
            feature_name: Ten cua feature
            
        Returns:
            Tuple: (transformed_data, method_name, lambda_param, transform_stats)
        """
        
        original_skew = feature_data.skew()
        
        print(f"Feature: {feature_name}")
        print(f"   Original Skewness: {original_skew:.3f}")
        print(f"   Min: {feature_data.min():.3f}, Max: {feature_data.max():.3f}")
        
        # Kiem tra xem co can transform khong
        if abs(original_skew) < self.SKEW_THRESHOLD:
            print(f"   Decision: NO TRANSFORM (|skew| < {self.SKEW_THRESHOLD})")
            print()
            return feature_data, "NO_TRANSFORM", None, {
                'original_skew': float(original_skew),
                'new_skew': float(original_skew),
                'method': 'NO_TRANSFORM'
            }
        
        # Chon transform method
        has_zero = (feature_data == 0).any()
        has_negative = (feature_data < 0).any()
        zero_count = (feature_data == 0).sum() if has_zero else 0
        zero_pct = (zero_count / len(feature_data) * 100) if has_zero else 0.0
        
        try:
            if has_negative:
                print(f"   Zero count: {zero_count}, Negative count: {(feature_data < 0).sum()}")
                print(f"   Decision: YEO-JOHNSON (data co gia tri am)")
                method = "Yeo-Johnson"
                transformed_data, lambda_param = yeojohnson(feature_data)
                
            elif has_zero:
                print(f"   Zero count: {zero_count} ({zero_pct:.1f}%)")
                print(f"   Decision: YEO-JOHNSON (data co gia tri 0)")
                method = "Yeo-Johnson"
                transformed_data, lambda_param = yeojohnson(feature_data)
                
            else:  # Thuan duong (> 0)
                print(f"   Zero count: 0")
                print(f"   Decision: BOX-COX (data thuan duong)")
                method = "Box-Cox"
                transformed_data, lambda_param = boxcox(feature_data)
            
            new_skew = pd.Series(transformed_data).skew()
            improvement = abs(original_skew) - abs(new_skew)
            
            print(f"   Method: {method}")
            print(f"   Lambda: {lambda_param:.3f}")
            print(f"   New Skewness: {new_skew:.3f}")
            print(f"   Improvement: {improvement:.3f}")
            print()
            
            transform_stats = {
                'method': method,
                'lambda': float(lambda_param),
                'original_skew': float(original_skew),
                'new_skew': float(new_skew),
                'improvement': float(improvement),
                'has_zero': bool(has_zero),
                'zero_count': int(zero_count),
                'has_negative': bool(has_negative)
            }
            
            return transformed_data, method, lambda_param, transform_stats
            
        except Exception as e:
            print(f"   ERROR: Transform failed - {e}")
            print()
            return feature_data, "FAILED", None, {
                'original_skew': float(original_skew),
                'error': str(e)
            }
    
    def transform_features(self):
        """Ap dung transform cho features bi skew."""
        self._print_header("STEP 7: AUTO-DETECT AND APPLY TRANSFORM")
        
        clustering_features = ['Age', 'Income', 'Dependency_Ratio']
        
        print(f"Checking skewness for clustering features...")
        print(f"Threshold: |skew| > {self.SKEW_THRESHOLD}")
        print()
        
        for feature in clustering_features:
            if feature not in self.df.columns:
                print(f"WARNING: {feature} not found. Skipping.")
                continue
            
            transformed_data, method, lambda_param, transform_stats = self.detect_and_apply_transform(
                self.df[feature], feature
            )
            
            if method != "NO_TRANSFORM" and method != "FAILED":
                # Cap nhat du lieu
                self.df[feature] = transformed_data
                
                # Luu thong tin transform
                self.feature_stats[f'{feature}_transform'] = transform_stats
                
                # Cap nhat final stats
                self.feature_stats[f'{feature}_final'] = {
                    'min': float(self.df[feature].min()),
                    'max': float(self.df[feature].max()),
                    'mean': float(self.df[feature].mean()),
                    'median': float(self.df[feature].median()),
                    'std': float(self.df[feature].std()),
                    'skewness': float(pd.Series(self.df[feature]).skew())
                }
                
                self.log_action(f"Transform {feature}", 
                               f"{method}: Skew {transform_stats['original_skew']:.3f} -> {transform_stats['new_skew']:.3f}")
            
            elif method == "NO_TRANSFORM":
                self.feature_stats[f'{feature}_transform'] = transform_stats
                self.log_action(f"Transform {feature}", "No transform needed")
            
            else:
                self.log_action(f"Transform {feature} FAILED", "See error above")
        
        print(f"Transformation completed")
        print()
    
    # ============================================================================
    # STEP 6: SELECT FINAL FEATURES
    # ============================================================================
    
    def select_final_features(self):
        """Chon 5 features cuoi cung (3 clustering + 2 post-hoc)."""
        self._print_header("STEP 8: SELECT FINAL FEATURES (OPTION C)")
        
        # 3 clustering features + 2 post-hoc
        final_features = [
            # CLUSTERING FEATURES (3)
            'Age',
            'Income',
            'Dependency_Ratio',
            # POST-HOC FEATURES (2)
            'Education_ord',
            'Life_Stage'
        ]
        
        existing_features = [f for f in final_features if f in self.df.columns]
        
        if len(existing_features) < len(final_features):
            missing = set(final_features) - set(existing_features)
            print(f"WARNING: Missing features: {missing}")
            print()
        
        self.df_engineered = self.df[existing_features].copy()
        
        print(f"Selected {len(existing_features)} features for demographic dataset:")
        print()
        
        # Group by usage
        clustering_features = [f for f in existing_features if f in ['Age', 'Income', 'Dependency_Ratio']]
        posthoc_features = [f for f in existing_features if f in ['Education_ord', 'Life_Stage']]
        
        print(f"CLUSTERING FEATURES ({len(clustering_features)} features - PURE CONTINUOUS):")
        print(f"{'-' * 100}")
        for i, feat in enumerate(clustering_features, 1):
            dtype = str(self.df_engineered[feat].dtype)
            n_unique = self.df_engineered[feat].nunique()
            mean_val = self.df_engineered[feat].mean()
            std_val = self.df_engineered[feat].std()
            
            # Get final stats if transformed
            if f'{feat}_final' in self.feature_stats:
                transformed = 'YES'
                method = self.feature_stats[f'{feat}_transform'].get('method', 'Unknown')
            else:
                transformed = 'NO'
                method = 'Original'
            
            print(f"   {i}. {feat:<20} | Type: {dtype:<10} | Unique: {n_unique:>4} | Mean: {mean_val:>10.3f} | Std: {std_val:>8.3f} | Transformed: {transformed} ({method})")
        
        print()
        print(f"POST-HOC FEATURES ({len(posthoc_features)} features - FOR ANALYSIS AFTER CLUSTERING):")
        print(f"{'-' * 100}")
        for i, feat in enumerate(posthoc_features, 1):
            dtype = str(self.df_engineered[feat].dtype)
            n_unique = self.df_engineered[feat].nunique()
            
            if feat == 'Education_ord':
                print(f"   {i}. {feat:<20} | Type: Ordinal     | Unique: {n_unique:>4} | Range: [0, 4] | Purpose: Post-hoc education analysis")
            elif feat == 'Life_Stage':
                print(f"   {i}. {feat:<20} | Type: Categorical | Unique: {n_unique:>4} | Range: [0, 4] | Purpose: Post-hoc life stage analysis")
        
        print()
        print(f"CORRELATION MATRIX (CLUSTERING FEATURES ONLY):")
        print(f"{'-' * 100}")
        
        if len(clustering_features) >= 2:
            corr_matrix = self.df_engineered[clustering_features].corr()
            
            # Print header
            print(f"{'':>20}", end='')
            for feat in clustering_features:
                print(f"{feat:>20}", end='')
            print()
            
            # Print correlation values
            max_corr = 0.0
            for i, feat1 in enumerate(clustering_features):
                print(f"{feat1:>20}", end='')
                for j, feat2 in enumerate(clustering_features):
                    corr_val = corr_matrix.loc[feat1, feat2]
                    if i != j and abs(corr_val) > max_corr:
                        max_corr = abs(corr_val)
                    
                    flag = ' (!)' if (i != j and abs(corr_val) > 0.7) else '    '
                    print(f"{corr_val:>17.3f}{flag}", end='')
                print()
            
            print(f"{'-' * 100}")
            print(f"Maximum correlation (off-diagonal)  : {max_corr:.3f}")
            print(f"Note: (!) indicates high correlation (|r| > 0.7)")
            print()
            
            self.feature_stats['max_correlation'] = float(max_corr)
        
        print(f"DESIGN NOTE:")
        print(f"   - Clustering uses ONLY 3 pure continuous features (no ordinal/categorical)")
        print(f"   - Education_ord and Life_Stage saved for post-hoc interpretation")
        print(f"   - Transform method: Auto-detected (Box-Cox or Yeo-Johnson based on data)")
        print(f"   - Scaling will be handled by Feature_Scaling.py")
        print()
        
        print(f"Final dataset shape: {self.df_engineered.shape}")
        print()
        
        self.log_action("Select final features", f"{len(existing_features)} features (3 clustering + 2 post-hoc)")
    
    # ============================================================================
    # STEP 7: VALIDATE FEATURES
    # ============================================================================
    
    def validate_features(self):
        """Validate engineered features."""
        self._print_header("STEP 9: VALIDATE FEATURES")
        
        print(f"DATA QUALITY CHECKS:")
        print(f"{'-' * 100}")
        
        # Missing values
        missing = self.df_engineered.isnull().sum()
        if missing.sum() == 0:
            print(f"   Missing values        : None (PASS)")
        else:
            print(f"   Missing values        : {missing.sum()} (WARNING)")
            for col in missing[missing > 0].index:
                print(f"      {col}: {missing[col]}")
        
        # Duplicates
        dup_count = self.df_engineered.duplicated().sum()
        dup_pct = dup_count / len(self.df_engineered) * 100
        print(f"   Duplicate rows        : {dup_count} ({dup_pct:.2f}%)")
        
        # Low variance
        low_var_features = []
        print()
        print(f"   Variance check (clustering features only):")
        for col in ['Age', 'Income', 'Dependency_Ratio']:
            if col in self.df_engineered.columns:
                var = self.df_engineered[col].var()
                print(f"      {col:<20} : variance = {var:.6f}")
                if var < 0.001:
                    low_var_features.append((col, var))
        
        if low_var_features:
            print()
            print(f"   Low variance features : {len(low_var_features)} (WARNING)")
            for feat, var in low_var_features:
                print(f"      {feat}: var={var:.6f}")
        else:
            print()
            print(f"   Low variance features : None (PASS)")
        
        # Infinite values
        inf_count = np.isinf(self.df_engineered.select_dtypes(include=[np.number])).sum().sum()
        if inf_count == 0:
            print(f"   Infinite values       : None (PASS)")
        else:
            print(f"   Infinite values       : {inf_count} (WARNING)")
        
        print(f"{'-' * 100}")
        print()
        
        self.log_action("Validate features", "All checks completed")
    
    # ============================================================================
    # STEP 8: EXPORT DATASET & REPORT
    # ============================================================================
    
    def export_dataset(self):
        """Export engineered dataset to CSV."""
        self._print_header("STEP 10: EXPORT DATASET")
        
        try:
            self.df_engineered.to_csv(self.output_csv, index=False)
            
            file_size = os.path.getsize(self.output_csv) / 1024
            
            print(f"Dataset exported successfully")
            print(f"   Path   : {self.output_csv}")
            print(f"   Shape  : {self.df_engineered.shape[0]:,} rows x {self.df_engineered.shape[1]} columns")
            print(f"   Size   : {file_size:.2f} KB")
            print()
            
            self.log_action("Export dataset", self.output_csv)
            return True
            
        except Exception as e:
            print(f"ERROR: Export failed - {e}")
            self.log_action("Export FAILED", str(e))
            return False
    
    def export_report(self):
        """Export detailed engineering report."""
        self._print_header("STEP 11: EXPORT REPORT")
        
        try:
            with open(self.report_file, 'w', encoding='utf-8') as f:
                # Header
                f.write("=" * 100 + "\n")
                f.write("DEMOGRAPHIC FEATURE ENGINEERING REPORT (IMPROVED - OPTION C)\n")
                f.write("=" * 100 + "\n\n")
                
                f.write(f"Generated       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Input           : {self.input_path}\n")
                f.write(f"Output          : {self.output_csv}\n")
                f.write(f"Strategy        : Option C (Exclude Ordinal/Nominal from Clustering)\n\n")
                
                # Design philosophy
                f.write("DESIGN PHILOSOPHY:\n")
                f.write("-" * 100 + "\n")
                f.write("- Use ONLY pure continuous features for clustering (Age, Income, Dependency_Ratio)\n")
                f.write("- Exclude Education_ord (ordinal) from clustering to avoid dominance issue\n")
                f.write("- Exclude Life_Stage (categorical) from clustering\n")
                f.write("- Save Education_ord and Life_Stage for POST-HOC ANALYSIS\n")
                f.write("- Auto-detect transform method: Box-Cox for positive data, Yeo-Johnson for data with zeros\n")
                f.write("- Transform only if |skewness| > 0.5 (tighter threshold for better symmetry)\n")
                f.write("- NO SCALING in this step (handled by Feature_Scaling.py)\n\n")
                
                # Transform method decision
                f.write("TRANSFORM METHOD DECISION LOGIC:\n")
                f.write("-" * 100 + "\n")
                f.write("- Box-Cox: Used when min > 0 (data purely positive)\n")
                f.write("- Yeo-Johnson: Used when min <= 0 (data contains zeros or negatives)\n")
                f.write("- Auto-detection: Applied to each feature independently\n")
                f.write("- Threshold: Transform only if |skewness| > 0.5\n\n")
                
                # Final features
                f.write(f"FINAL FEATURES ({self.df_engineered.shape[1]} columns):\n")
                f.write("-" * 100 + "\n\n")
                
                f.write("CLUSTERING FEATURES (3 - pure continuous):\n")
                for i, col in enumerate(['Age', 'Income', 'Dependency_Ratio'], 1):
                    if col in self.df_engineered.columns:
                        f.write(f"{i}. {col}\n")
                        
                        # Original stats
                        if f'{col}_original' in self.feature_stats:
                            f.write(f"   Original statistics:\n")
                            for key, val in self.feature_stats[f'{col}_original'].items():
                                f.write(f"      {key:<15} : {val}\n")
                        
                        # Transform info
                        if f'{col}_transform' in self.feature_stats:
                            f.write(f"   Transformation applied:\n")
                            for key, val in self.feature_stats[f'{col}_transform'].items():
                                f.write(f"      {key:<20} : {val}\n")
                        
                        # Final stats
                        if f'{col}_final' in self.feature_stats:
                            f.write(f"   Final statistics (after transform):\n")
                            for key, val in self.feature_stats[f'{col}_final'].items():
                                f.write(f"      {key:<15} : {val}\n")
                        
                        f.write("\n")
                
                f.write("POST-HOC FEATURES (2 - for analysis after clustering):\n")
                f.write("4. Education_ord\n")
                f.write("   Type            : Ordinal (0-4)\n")
                f.write("   Purpose         : Post-hoc education distribution analysis\n")
                f.write("   Not used in     : K-Means distance calculation\n\n")
                
                f.write("5. Life_Stage\n")
                f.write("   Type            : Categorical (0-4)\n")
                f.write("   Purpose         : Post-hoc life stage distribution analysis\n")
                f.write("   Not used in     : K-Means distance calculation\n")
                if 'Life_Stage' in self.feature_stats:
                    f.write(f"   Distribution    : {self.feature_stats['Life_Stage']['distribution']}\n")
                    f.write(f"   Stage names     : {self.feature_stats['Life_Stage']['stage_names']}\n")
                f.write("\n")
                
                # Correlation
                f.write("CORRELATION ANALYSIS (CLUSTERING FEATURES ONLY):\n")
                f.write("-" * 100 + "\n")
                if 'max_correlation' in self.feature_stats:
                    f.write(f"Maximum correlation (off-diagonal) : {self.feature_stats['max_correlation']:.3f}\n")
                f.write("\n")
                
                # Dataset statistics
                f.write("DATASET STATISTICS:\n")
                f.write("-" * 100 + "\n")
                f.write(f"Original dataset shape  : {self.df.shape[0]:,} rows x {self.df.shape[1]} columns\n")
                f.write(f"Final dataset shape     : {self.df_engineered.shape[0]:,} rows x {self.df_engineered.shape[1]} columns\n")
                f.write(f"Clustering features     : 3 (Age, Income, Dependency_Ratio)\n")
                f.write(f"Post-hoc features       : 2 (Education_ord, Life_Stage)\n\n")
                
                # Processing log
                f.write("PROCESSING LOG:\n")
                f.write("-" * 100 + "\n")
                for i, log in enumerate(self.processing_log, 1):
                    f.write(f"{i:>3}. [{log['timestamp']}] {log['action']}\n")
                    if log['details']:
                        f.write(f"     Details: {log['details']}\n")
                
                f.write("\n" + "=" * 100 + "\n")
                f.write("END OF REPORT\n")
                f.write("=" * 100 + "\n")
            
            file_size = os.path.getsize(self.report_file) / 1024
            
            print(f"Report exported successfully")
            print(f"   Path  : {self.report_file}")
            print(f"   Size  : {file_size:.2f} KB")
            print()
            
            self.log_action("Export report", self.report_file)
            return True
            
        except Exception as e:
            print(f"ERROR: Report export failed - {e}")
            self.log_action("Export report FAILED", str(e))
            return False
    
    # ============================================================================
    # STEP 9: GENERATE VISUALIZATIONS
    # ============================================================================
    
    def plot_histograms(self, graph_dir):
        """Vẽ histogram + KDE cho cac clustering features."""
        os.makedirs(graph_dir, exist_ok=True)
        
        clustering_cols = ['Age', 'Income', 'Dependency_Ratio']
        
        for col in clustering_cols:
            if col not in self.df_engineered.columns:
                continue
            
            try:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                sns.histplot(self.df_engineered[col].dropna(), bins=50, kde=True, ax=ax, color='steelblue')
                
                mean_val = self.df_engineered[col].mean()
                median_val = self.df_engineered[col].median()
                
                ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
                ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.3f}')
                
                # Determine if transformed
                is_transformed = f'{col}_transform' in self.feature_stats
                transform_method = self.feature_stats[f'{col}_transform'].get('method', 'Unknown') if is_transformed else 'Original'
                
                ax.set_title(f'Distribution: {col} ({transform_method})', fontsize=14, fontweight='bold')
                ax.set_xlabel(col, fontsize=12)
                ax.set_ylabel('Frequency', fontsize=12)
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                fname = os.path.join(graph_dir, f"{col}_histogram.png")
                fig.tight_layout()
                fig.savefig(fname, dpi=150, bbox_inches='tight')
                plt.close(fig)
                
                print(f"   Saved: {col}_histogram.png")
                
            except Exception as e:
                print(f"   ERROR: Failed to plot {col} - {e}")
                plt.close('all')
    
    def plot_boxplots(self, graph_dir):
        """Vẽ boxplot cho clustering features."""
        os.makedirs(graph_dir, exist_ok=True)
        
        clustering_cols = ['Age', 'Income', 'Dependency_Ratio']
        
        for col in clustering_cols:
            if col not in self.df_engineered.columns:
                continue
            
            try:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                sns.boxplot(y=self.df_engineered[col].dropna(), ax=ax, color='lightcoral')
                
                # Determine if transformed
                is_transformed = f'{col}_transform' in self.feature_stats
                transform_method = self.feature_stats[f'{col}_transform'].get('method', 'Unknown') if is_transformed else 'Original'
                
                ax.set_title(f'Boxplot: {col} ({transform_method})', fontsize=14, fontweight='bold')
                ax.set_ylabel(col, fontsize=12)
                ax.grid(True, alpha=0.3, axis='y')
                
                fname = os.path.join(graph_dir, f"{col}_boxplot.png")
                fig.tight_layout()
                fig.savefig(fname, dpi=150, bbox_inches='tight')
                plt.close(fig)
                
                print(f"   Saved: {col}_boxplot.png")
                
            except Exception as e:
                print(f"   ERROR: Failed to plot {col} - {e}")
                plt.close('all')
    
    def plot_correlation_heatmap(self, graph_dir):
        """Vẽ correlation heatmap (clustering features only)."""
        os.makedirs(graph_dir, exist_ok=True)
        
        clustering_cols = ['Age', 'Income', 'Dependency_Ratio']
        existing_cols = [c for c in clustering_cols if c in self.df_engineered.columns]
        
        if len(existing_cols) < 2:
            print("   Skipped: Not enough clustering features for heatmap")
            return
        
        try:
            corr = self.df_engineered[existing_cols].corr()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            sns.heatmap(corr, annot=True, fmt=".3f", cmap="RdYlGn_r", vmin=-1, vmax=1, 
                       center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
            
            ax.set_title('Correlation Matrix (Clustering Features Only)', fontsize=16, fontweight='bold', pad=20)
            
            fname = os.path.join(graph_dir, "correlation_heatmap.png")
            fig.tight_layout()
            fig.savefig(fname, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"   Saved: correlation_heatmap.png")
            
        except Exception as e:
            print(f"   ERROR: Failed to plot correlation heatmap - {e}")
            plt.close('all')
    
    def plot_before_after_transform(self, graph_dir):
        """Vẽ biểu đồ before/after cho cac features đã transform."""
        os.makedirs(graph_dir, exist_ok=True)
        
        for feature in ['Age', 'Income', 'Dependency_Ratio']:
            # Check if transform was applied
            if f'{feature}_transform' not in self.feature_stats:
                continue
            
            if self.feature_stats[f'{feature}_transform'].get('method') == 'NO_TRANSFORM':
                continue
            
            # Get original data (before transform)
            if f'{feature}_original' not in self.feature_stats:
                continue
            
            try:
                # We need to reload original data for before plot
                df_original = pd.read_csv(self.input_path)
                
                # Recreate original feature if needed
                if feature == 'Age' and 'Year_Birth' in df_original.columns:
                    original_data = self.REFERENCE_YEAR - df_original['Year_Birth']
                    original_data = original_data.clip(upper=100)
                elif feature == 'Dependency_Ratio':
                    # Recreate Dependency_Ratio
                    if 'Kidhome' in df_original.columns and 'Teenhome' in df_original.columns and 'Income' in df_original.columns:
                        total_children = df_original['Kidhome'] + df_original['Teenhome']
                        family_size = 2 + total_children
                        income_per_fm = df_original['Income'] / family_size
                        original_data = (total_children * 10000) / (income_per_fm + 1e-6)
                        original_data = original_data.clip(upper=original_data.quantile(0.99))
                    else:
                        continue
                elif feature == 'Income' and 'Income' in df_original.columns:
                    original_data = df_original['Income']
                else:
                    continue
                
                # Plot before/after
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                
                transform_info = self.feature_stats[f'{feature}_transform']
                method = transform_info.get('method', 'Unknown')
                
                # Before
                sns.histplot(original_data.dropna(), bins=40, kde=True, ax=axes[0], color='steelblue')
                axes[0].set_title(f'Before: {feature}', fontsize=12, fontweight='bold')
                axes[0].set_xlabel(feature, fontsize=11)
                axes[0].set_ylabel('Frequency', fontsize=11)
                axes[0].axvline(original_data.mean(), color='red', linestyle='--', 
                               label=f'Mean: {original_data.mean():.2f}')
                axes[0].axvline(original_data.median(), color='green', linestyle='--', 
                               label=f'Median: {original_data.median():.2f}')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
                axes[0].text(0.02, 0.98, f"Skewness: {original_data.skew():.3f}", 
                            transform=axes[0].transAxes, fontsize=10, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                # After
                transformed_data = self.df_engineered[feature]
                sns.histplot(transformed_data.dropna(), bins=40, kde=True, ax=axes[1], color='seagreen')
                axes[1].set_title(f'After: {feature} ({method})', fontsize=12, fontweight='bold')
                axes[1].set_xlabel(f'{feature} (transformed)', fontsize=11)
                axes[1].set_ylabel('Frequency', fontsize=11)
                axes[1].axvline(transformed_data.mean(), color='red', linestyle='--', 
                               label=f'Mean: {transformed_data.mean():.3f}')
                axes[1].axvline(transformed_data.median(), color='green', linestyle='--', 
                               label=f'Median: {transformed_data.median():.3f}')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
                
                axes[1].text(0.02, 0.98, 
                            f"Skewness: {transform_info['new_skew']:.3f}\nLambda: {transform_info['lambda']:.3f}\nMethod: {method}", 
                            transform=axes[1].transAxes, fontsize=10, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
                
                fname = os.path.join(graph_dir, f"{feature}_before_after.png")
                fig.tight_layout()
                fig.savefig(fname, dpi=150, bbox_inches='tight')
                plt.close(fig)
                
                print(f"   Saved: {feature}_before_after.png")
                
            except Exception as e:
                print(f"   ERROR: Failed before/after for {feature} - {e}")
                plt.close('all')
    
    def plot_posthoc_distributions(self, graph_dir):
        """Vẽ distribution cho post-hoc features (Education, Life_Stage)."""
        os.makedirs(graph_dir, exist_ok=True)
        
        # Education_ord distribution
        if 'Education_ord' in self.df_engineered.columns:
            try:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                edu_counts = self.df_engineered['Education_ord'].value_counts().sort_index()
                edu_labels = ['None/Basic', '2nd Cycle', 'Graduation', 'Master', 'PhD']
                
                ax.bar(edu_counts.index, edu_counts.values, color='coral', alpha=0.7)
                ax.set_xlabel('Education Level', fontsize=12)
                ax.set_ylabel('Count', fontsize=12)
                ax.set_title('Education Distribution (Post-Hoc Feature)', fontsize=14, fontweight='bold')
                ax.set_xticks(range(len(edu_labels)))
                ax.set_xticklabels(edu_labels, rotation=45, ha='right')
                ax.grid(True, alpha=0.3, axis='y')
                
                # Add percentage labels
                total = edu_counts.sum()
                for idx, count in edu_counts.items():
                    pct = count / total * 100
                    ax.text(idx, count, f'{count}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9)
                
                fname = os.path.join(graph_dir, "Education_distribution.png")
                fig.tight_layout()
                fig.savefig(fname, dpi=150, bbox_inches='tight')
                plt.close(fig)
                
                print(f"   Saved: Education_distribution.png")
                
            except Exception as e:
                print(f"   ERROR: Failed to plot Education - {e}")
                plt.close('all')
        
        # Life_Stage distribution
        if 'Life_Stage' in self.df_engineered.columns:
            try:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                stage_counts = self.df_engineered['Life_Stage'].value_counts().sort_index()
                stage_names = ['Young Single', 'Young Family', 'Mature Family', 'Empty Nest', 'Single Parent']
                
                ax.bar(stage_counts.index, stage_counts.values, color='steelblue', alpha=0.7)
                ax.set_xlabel('Life Stage', fontsize=12)
                ax.set_ylabel('Count', fontsize=12)
                ax.set_title('Life Stage Distribution (Post-Hoc Feature)', fontsize=14, fontweight='bold')
                ax.set_xticks(range(len(stage_names)))
                ax.set_xticklabels(stage_names, rotation=45, ha='right')
                ax.grid(True, alpha=0.3, axis='y')
                
                # Add percentage labels
                total = stage_counts.sum()
                for idx, count in stage_counts.items():
                    pct = count / total * 100
                    ax.text(idx, count, f'{count}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9)
                
                fname = os.path.join(graph_dir, "Life_Stage_distribution.png")
                fig.tight_layout()
                fig.savefig(fname, dpi=150, bbox_inches='tight')
                plt.close(fig)
                
                print(f"   Saved: Life_Stage_distribution.png")
                
            except Exception as e:
                print(f"   ERROR: Failed to plot Life_Stage - {e}")
                plt.close('all')
    
    def generate_all_plots(self, graph_dir):
        """Tao tat ca cac bieu do."""
        self._print_header("STEP 12: GENERATE VISUALIZATIONS")
        
        print(f"Creating plots in: {graph_dir}")
        print()
        
        print(f"Creating histograms (clustering features)...")
        self.plot_histograms(graph_dir)
        
        print()
        print(f"Creating boxplots (clustering features)...")
        self.plot_boxplots(graph_dir)
        
        print()
        print(f"Creating correlation heatmap...")
        self.plot_correlation_heatmap(graph_dir)
        
        print()
        print(f"Creating before/after transform plots...")
        self.plot_before_after_transform(graph_dir)
        
        print()
        print(f"Creating post-hoc feature distributions...")
        self.plot_posthoc_distributions(graph_dir)
        
        print()
        print(f"All plots saved to: {graph_dir}")
        print()
        
        self.log_action("Generate plots", f"Saved to {graph_dir}")
    
    # ============================================================================
    # MAIN PIPELINE
    # ============================================================================
    
    def run_engineering(self):
        """Chay toan bo pipeline demographic feature engineering."""
        print("\n" + "=" * 100)
        print("DEMOGRAPHIC FEATURE ENGINEERING PIPELINE (IMPROVED - OPTION C)".center(100))
        print("=" * 100 + "\n")
        
        start = datetime.now()
        
        # Load data
        if not self.load_data():
            return False
        
        # Create base features
        self.create_age_feature()
        self.create_total_children_feature()
        self.create_income_per_family_member()
        self.create_marital_encoded()
        
        # Create composite features
        self.create_life_stage()
        self.create_dependency_ratio()
        
        # Transform skewed features with auto-detection (Box-Cox or Yeo-Johnson)
        self.transform_features()
        
        # Select final features (3 clustering + 2 post-hoc)
        self.select_final_features()
        
        # Validate
        self.validate_features()
        
        # Export
        if not self.export_dataset():
            return False
        
        if not self.export_report():
            return False
        
        # Generate plots
        graph_dir = r"C:\Project\Machine_Learning\Machine_Learning\graph\Feature Extraction & Engineering_graph\Demographic"
        self.generate_all_plots(graph_dir)
        
        elapsed = (datetime.now() - start).total_seconds()
        
        print("=" * 100)
        print(f"PIPELINE COMPLETED in {elapsed:.2f}s".center(100))
        print("=" * 100 + "\n")
        
        return True


# ================================================================================
# CLASS 2: PRODUCT + CHANNEL FEATURE ENGINEERING (v5 - AUTO-DETECT TRANSFORM + AUTO PCA)
# ================================================================================

class ProductChannelFeatureEngineering:
    """
    Feature Engineering
    Triết lý thiết kế:
    - Tập trung vào “MUA BAO NHIÊU” chứ không phải “MUA CÁI GÌ”
    - Khối lượng (Total_Spent + Frequency) + Đa dạng (HHI) + Kênh (Store/Web)
    - Loại bỏ các biến phụ thuộc toán học (AOV = Total_Spent / Frequency)
    - Áp dụng PCA để loại bỏ đa cộng tuyến (correlation ≥ 0.7)

    Các biến đầu ra cuối cùng (4–5 biến clustering + 8 biến tham chiếu):
    - Biến clustering (4–5 biến):
        + Nếu tương quan ≥ 0.7: Áp dụng PCA → PC1 (tổng hợp) → còn 4 biến
        + Nếu tương quan < 0.7: Giữ nguyên 5 biến gốc

    - 5 biến mặc định:
        + Total_Spent_Transformed (biến đổi tự động – khối lượng chi tiêu)
        + TotalPurchases (tần suất – số lượng giao dịch)
        + Product_HHI (mức độ tập trung: mua nhiều một loại hay đa dạng)
        + Store_Preference (kênh: tỷ lệ mua tại cửa hàng)
        + Web_Engagement (hành vi số: lượt truy cập web)

    - Biến tham chiếu (8 biến – hậu kiểm):
        + Wine/Meat/Fish/Fruit/Sweet/Gold_Preference (mô tả cơ cấu sản phẩm)
        + Dominant_Product, Top_Product_Share (dùng để gán nhãn cluster)

    Pipeline:
    - Bước 1: Load dữ liệu
    - Bước 2: Tạo Total_Spent
    - Bước 3: Biến đổi Total_Spent (tự động Box-Cox hoặc Yeo-Johnson)
    - Bước 4: Tạo TotalPurchases
    - Bước 5: Tạo tỷ lệ ưu tiên sản phẩm (biến tham chiếu)
    - Bước 6: Tạo Product_HHI
    - Bước 7: Xác định Dominant_Product
    - Bước 8: Tạo Store_Preference
    - Bước 9: Tạo Web_Engagement
    - Bước 10: Xác định các cặp có tương quan cao & áp dụng PCA nếu cần
    - Bước 11: Chọn biến cuối cùng (4–5 clustering + 8 tham chiếu)
    - Bước 12: Kiểm tra/validate
    - Bước 13: Xuất dataset + báo cáo
    - Bước 14: Tạo biểu đồ trực quan hóa

    """
    
    def __init__(self, input_path, output_dir, report_dir):
        self.input_path = input_path
        self.output_dir = output_dir
        self.report_dir = report_dir
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(report_dir, exist_ok=True)
        
        self.output_csv = os.path.join(output_dir, "Customer_Behavior_ProductChannel.csv")
        self.report_file = os.path.join(report_dir, "ProductChannel_Engineering_Report.txt")
        
        self.df = None
        self.df_engineered = None
        self.feature_stats = {}
        self.processing_log = []
        self.pca_applied_pairs = []  # Track PCA applications
        
        self.SKEW_THRESHOLD = 0.8
        self.CORRELATION_THRESHOLD = 0.7  # Threshold for PCA
    
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
    
    # ============================================================================
    # STEP 1: LOAD DATA
    # ============================================================================
    
    def load_data(self):
        self._print_header("PRODUCT+CHANNEL FEATURE ENGINEERING (v5 - AUTO-DETECT + AUTO PCA) - LOAD DATA")
        
        try:
            self.df = pd.read_csv(self.input_path)
            print(f"Dataset loaded successfully")
            print(f"   Shape         : {self.df.shape[0]:,} rows x {self.df.shape[1]} columns")
            print(f"   Memory usage  : {self.df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
            print()
            self.log_action("Load data", f"Shape: {self.df.shape}")
            return True
        except Exception as e:
            print(f"ERROR loading data: {e}")
            self.log_action("Load data FAILED", str(e))
            return False
    
    # ============================================================================
    # STEP 2: CREATE TOTAL_SPENT
    # ============================================================================
    
    def create_total_spent(self):
        self._print_header("STEP 1: CREATE TOTAL_SPENT (PRIMARY VOLUME)")
        
        mnt_cols = ['MntWines', 'MntFruits', 'MntMeatProducts', 
                    'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
        existing = [c for c in mnt_cols if c in self.df.columns]
        
        if not existing:
            print("WARNING: No Mnt* columns. Skipping.")
            return
        
        self.df['Total_Spent'] = self.df[existing].sum(axis=1)
        
        stats = {
            'min': float(self.df['Total_Spent'].min()),
            'max': float(self.df['Total_Spent'].max()),
            'mean': float(self.df['Total_Spent'].mean()),
            'median': float(self.df['Total_Spent'].median()),
            'std': float(self.df['Total_Spent'].std()),
            'skewness': float(self.df['Total_Spent'].skew()),
            'variance': float(self.df['Total_Spent'].var())
        }
        
        zero_count = (self.df['Total_Spent'] == 0).sum()
        zero_pct = zero_count / len(self.df) * 100
        
        print(f"Feature created  : Total_Spent")
        print(f"Range            : [{stats['min']:,.0f}, {stats['max']:,.0f}]")
        print(f"Mean             : {stats['mean']:,.2f}")
        print(f"Median           : {stats['median']:,.2f}")
        print(f"Std Dev          : {stats['std']:,.2f}")
        print(f"Skewness         : {stats['skewness']:.3f}")
        print(f"Variance         : {stats['variance']:.3f}")
        print(f"Zero count       : {zero_count} ({zero_pct:.1f}%)")
        print()
        
        self.feature_stats['Total_Spent_original'] = stats
        self.log_action("Create Total_Spent", f"Mean: {stats['mean']:,.0f}, Skew: {stats['skewness']:.3f}, Zero: {zero_count}")
    
    # ============================================================================
    # STEP 3: TRANSFORM TOTAL_SPENT (AUTO-DETECT METHOD)
    # ============================================================================
    
    def transform_total_spent(self):
        self._print_header("STEP 2: TRANSFORM TOTAL_SPENT (AUTO-DETECT METHOD)")
        
        if 'Total_Spent' not in self.df.columns:
            print("WARNING: Total_Spent not found. Skipping.")
            return
        
        original_skew = self.df['Total_Spent'].skew()
        feature_data = self.df['Total_Spent']
        
        print(f"Checking skewness:")
        print(f"   Original skewness : {original_skew:.3f}")
        print(f"   Min: {feature_data.min():.3f}, Max: {feature_data.max():.3f}")
        print(f"   Threshold         : {self.SKEW_THRESHOLD}")
        print()
        
        if abs(original_skew) < self.SKEW_THRESHOLD:
            print(f"   Decision: NO TRANSFORM (|skew| < {self.SKEW_THRESHOLD})")
            self.df['Total_Spent_Transformed'] = self.df['Total_Spent']
            self.feature_stats['Total_Spent_transform'] = {
                'method': 'NO_TRANSFORM',
                'original_skew': float(original_skew)
            }
            self.log_action("Transform Total_Spent", "No transform needed")
            print()
            return
        
        has_zero = (feature_data == 0).any()
        has_negative = (feature_data < 0).any()
        zero_count = (feature_data == 0).sum() if has_zero else 0
        zero_pct = (zero_count / len(feature_data) * 100) if has_zero else 0.0
        
        try:
            if has_negative:
                print(f"   Zero count: {zero_count}, Negative count: {(feature_data < 0).sum()}")
                print(f"   Decision: YEO-JOHNSON (data co gia tri am)")
                method = "Yeo-Johnson"
                transformed_data, lambda_param = yeojohnson(feature_data)
                
            elif has_zero:
                print(f"   Zero count: {zero_count} ({zero_pct:.1f}%)")
                print(f"   Decision: YEO-JOHNSON (data co gia tri 0)")
                method = "Yeo-Johnson"
                transformed_data, lambda_param = yeojohnson(feature_data)
                
            else:
                print(f"   Zero count: 0")
                print(f"   Decision: BOX-COX (data thuan duong)")
                method = "Box-Cox"
                transformed_data, lambda_param = boxcox(feature_data)
            
            self.df['Total_Spent_Transformed'] = transformed_data
            
            new_skew = pd.Series(transformed_data).skew()
            improvement = abs(original_skew) - abs(new_skew)
            
            print(f"Applied {method}:")
            print(f"   Lambda    : {lambda_param:.3f}")
            print(f"   New skew  : {new_skew:.3f}")
            print(f"   Improve   : {improvement:.3f}")
            print()
            
            self.feature_stats['Total_Spent_transform'] = {
                'method': method,
                'lambda': float(lambda_param),
                'original_skew': float(original_skew),
                'new_skew': float(new_skew),
                'improvement': float(improvement),
                'has_zero': bool(has_zero),
                'zero_count': int(zero_count),
                'has_negative': bool(has_negative)
            }
            
            self.feature_stats['Total_Spent_Transformed'] = {
                'min': float(pd.Series(transformed_data).min()),
                'max': float(pd.Series(transformed_data).max()),
                'mean': float(pd.Series(transformed_data).mean()),
                'median': float(pd.Series(transformed_data).median()),
                'std': float(pd.Series(transformed_data).std()),
                'skewness': float(new_skew),
                'variance': float(pd.Series(transformed_data).var())
            }
            
            self.log_action("Transform Total_Spent", 
                           f"{method}: Skew {original_skew:.3f} -> {new_skew:.3f}")
            
        except Exception as e:
            print(f"   ERROR: {e}")
            self.df['Total_Spent_Transformed'] = self.df['Total_Spent']
            self.log_action("Transform FAILED", str(e))
            print()
    
    # ============================================================================
    # STEP 4: CREATE TOTALPURCHASES (FREQUENCY)
    # ============================================================================
    
    def create_total_purchases(self):
        self._print_header("STEP 3: CREATE TOTALPURCHASES (FREQUENCY)")
        
        purchase_cols = ['NumWebPurchases', 'NumCatalogPurchases', 
                         'NumStorePurchases', 'NumDealsPurchases']
        existing = [c for c in purchase_cols if c in self.df.columns]
        
        if not existing:
            print("WARNING: No Num*Purchases columns. Skipping.")
            return
        
        self.df['TotalPurchases'] = self.df[existing].sum(axis=1)
        
        stats = {
            'min': float(self.df['TotalPurchases'].min()),
            'max': float(self.df['TotalPurchases'].max()),
            'mean': float(self.df['TotalPurchases'].mean()),
            'median': float(self.df['TotalPurchases'].median()),
            'std': float(self.df['TotalPurchases'].std()),
            'skewness': float(self.df['TotalPurchases'].skew()),
            'variance': float(self.df['TotalPurchases'].var())
        }
        
        print(f"Feature created  : TotalPurchases")
        print(f"Range            : [{stats['min']:.0f}, {stats['max']:.0f}]")
        print(f"Mean             : {stats['mean']:.2f}")
        print(f"Median           : {stats['median']:.2f}")
        print(f"Std Dev          : {stats['std']:.2f}")
        print(f"Skewness         : {stats['skewness']:.3f}")
        print(f"Variance         : {stats['variance']:.3f}")
        print()
        
        self.feature_stats['TotalPurchases'] = stats
        self.log_action("Create TotalPurchases", f"Mean: {stats['mean']:.1f}, Skew: {stats['skewness']:.3f}")
    
    # ============================================================================
    # STEP 5: CREATE PRODUCT PREFERENCES (REFERENCE)
    # ============================================================================
    
    def create_product_preferences_reference(self):
        self._print_header("STEP 4: CREATE PRODUCT PREFERENCES (REFERENCE)")
        
        if 'Total_Spent' not in self.df.columns:
            print("WARNING: Total_Spent not found. Skipping.")
            return
        
        product_mapping = {
            'MntWines': 'Wine_Preference',
            'MntMeatProducts': 'Meat_Preference',
            'MntFishProducts': 'Fish_Preference',
            'MntFruits': 'Fruit_Preference',
            'MntSweetProducts': 'Sweet_Preference',
            'MntGoldProds': 'Gold_Preference'
        }
        
        print(f"Creating ratios (FOR POST-HOC ONLY):")
        print()
        
        for mnt_col, ratio_name in product_mapping.items():
            if mnt_col not in self.df.columns:
                continue
            
            self.df[ratio_name] = self.df[mnt_col] / self.df['Total_Spent'].replace(0, np.nan)
            self.df[ratio_name] = self.df[ratio_name].fillna(0)
            
            stats = {
                'min': float(self.df[ratio_name].min()),
                'max': float(self.df[ratio_name].max()),
                'mean': float(self.df[ratio_name].mean()),
                'std': float(self.df[ratio_name].std())
            }
            print(f"   {ratio_name:<20} : min={stats['min']:.3f}, max={stats['max']:.3f}, mean={stats['mean']:.3f}")
            self.feature_stats[f'{ratio_name}_reference'] = stats
        
        print()
        self.log_action("Create product preferences", "6 ratios (reference)")
    
    # ============================================================================
    # STEP 6: CREATE PRODUCT_HHI (CONCENTRATION)
    # ============================================================================
    
    def create_product_hhi(self):
        self._print_header("STEP 5: CREATE PRODUCT_HHI (CONCENTRATION)")
        
        product_cols = ['Wine_Preference', 'Meat_Preference', 'Fish_Preference',
                        'Fruit_Preference', 'Sweet_Preference', 'Gold_Preference']
        existing = [c for c in product_cols if c in self.df.columns]
        
        if len(existing) < 2:
            print("WARNING: Not enough preferences. Skipping.")
            return
        
        prod_df = self.df[existing].fillna(0).clip(lower=0)
        self.df['Product_HHI'] = (prod_df ** 2).sum(axis=1)
        
        stats = {
            'min': float(self.df['Product_HHI'].min()),
            'max': float(self.df['Product_HHI'].max()),
            'mean': float(self.df['Product_HHI'].mean()),
            'median': float(self.df['Product_HHI'].median()),
            'std': float(self.df['Product_HHI'].std()),
            'skewness': float(self.df['Product_HHI'].skew()),
            'variance': float(self.df['Product_HHI'].var())
        }
        
        print(f"Feature created  : Product_HHI")
        print(f"Range            : [{stats['min']:.3f}, {stats['max']:.3f}]")
        print(f"Mean             : {stats['mean']:.3f}")
        print(f"Median           : {stats['median']:.3f}")
        print(f"Std Dev          : {stats['std']:.3f}")
        print(f"Variance         : {stats['variance']:.3f}")
        print()
        print(f"NOTE: Product_Entropy DROPPED (corr = -0.98 with HHI)")
        print()
        
        self.feature_stats['Product_HHI'] = stats
        self.log_action("Create Product_HHI", f"Mean: {stats['mean']:.3f}, Skew: {stats['skewness']:.3f}")
    
    # ============================================================================
    # STEP 7: CREATE DOMINANT_PRODUCT (REFERENCE)
    # ============================================================================
    
    def create_dominant_product(self):
        self._print_header("STEP 6: CREATE DOMINANT_PRODUCT (REFERENCE)")
        
        product_cols = ['Wine_Preference', 'Meat_Preference', 'Fish_Preference',
                        'Fruit_Preference', 'Sweet_Preference', 'Gold_Preference']
        existing = [c for c in product_cols if c in self.df.columns]
        
        if len(existing) < 2:
            print("WARNING: Not enough preferences. Skipping.")
            return
        
        prod_df = self.df[existing].fillna(0)
        
        self.df['Dominant_Product'] = prod_df.idxmax(axis=1).str.replace('_Preference', '')
        self.df['Top_Product_Share'] = prod_df.max(axis=1)
        
        print(f"Created:")
        print(f"   Dominant_Product   : Top product name")
        print(f"   Top_Product_Share  : Share percentage")
        print()
        
        dist = self.df['Dominant_Product'].value_counts()
        print(f"Distribution:")
        for prod, count in dist.items():
            pct = count / len(self.df) * 100
            bar = '█' * int(pct / 2)
            print(f"   {prod:<10} : {count:>5,} ({pct:>5.1f}%) {bar}")
        print()
        
        self.feature_stats['Dominant_Product_dist'] = dist.to_dict()
        self.log_action("Create Dominant_Product", "For labeling")
    
    # ============================================================================
    # STEP 8: CREATE STORE_PREFERENCE (CHANNEL)
    # ============================================================================
    
    def create_store_preference(self):
        self._print_header("STEP 7: CREATE STORE_PREFERENCE (CHANNEL)")
        
        if 'TotalPurchases' not in self.df.columns or 'NumStorePurchases' not in self.df.columns:
            print("WARNING: Required columns not found. Skipping.")
            return
        
        self.df['Store_Preference'] = self.df['NumStorePurchases'] / self.df['TotalPurchases'].replace(0, np.nan)
        self.df['Store_Preference'] = self.df['Store_Preference'].fillna(0)
        
        stats = {
            'min': float(self.df['Store_Preference'].min()),
            'max': float(self.df['Store_Preference'].max()),
            'mean': float(self.df['Store_Preference'].mean()),
            'median': float(self.df['Store_Preference'].median()),
            'std': float(self.df['Store_Preference'].std()),
            'skewness': float(self.df['Store_Preference'].skew()),
            'variance': float(self.df['Store_Preference'].var())
        }
        
        print(f"Feature created  : Store_Preference")
        print(f"Range            : [{stats['min']:.3f}, {stats['max']:.3f}]")
        print(f"Mean             : {stats['mean']:.3f}")
        print(f"Median           : {stats['median']:.3f}")
        print(f"Std Dev          : {stats['std']:.3f}")
        print(f"Skewness         : {stats['skewness']:.3f}")
        print(f"Variance         : {stats['variance']:.3f}")
        print()
        
        self.feature_stats['Store_Preference'] = stats
        self.log_action("Create Store_Preference", f"Mean: {stats['mean']:.3f}, Skew: {stats['skewness']:.3f}")
    
    # ============================================================================
    # STEP 9: CREATE WEB_ENGAGEMENT (DIGITAL)
    # ============================================================================
    
    def create_web_engagement(self):
        self._print_header("STEP 8: CREATE WEB_ENGAGEMENT (DIGITAL)")
        
        if 'NumWebVisitsMonth' not in self.df.columns:
            print("WARNING: NumWebVisitsMonth not found. Skipping.")
            return
        
        self.df['Web_Engagement'] = self.df['NumWebVisitsMonth'].copy()
        
        stats = {
            'min': float(self.df['Web_Engagement'].min()),
            'max': float(self.df['Web_Engagement'].max()),
            'mean': float(self.df['Web_Engagement'].mean()),
            'median': float(self.df['Web_Engagement'].median()),
            'std': float(self.df['Web_Engagement'].std()),
            'skewness': float(self.df['Web_Engagement'].skew()),
            'variance': float(self.df['Web_Engagement'].var())
        }
        
        print(f"Feature created  : Web_Engagement")
        print(f"Range            : [{stats['min']:.0f}, {stats['max']:.0f}]")
        print(f"Mean             : {stats['mean']:.2f}")
        print(f"Median           : {stats['median']:.2f}")
        print(f"Std Dev          : {stats['std']:.2f}")
        print(f"Skewness         : {stats['skewness']:.3f}")
        print(f"Variance         : {stats['variance']:.3f}")
        print()
        
        self.feature_stats['Web_Engagement'] = stats
        self.log_action("Create Web_Engagement", f"Mean: {stats['mean']:.1f}, Skew: {stats['skewness']:.3f}")
    
    # ============================================================================
    # STEP 10: AUTO-DETECT & APPLY PCA FOR HIGH CORRELATION PAIRS
    # ============================================================================
    
    def apply_pca_for_high_correlation(self):
        """Tự động phát hiện và áp dụng PCA cho các cặp đặc trưng có tương quan cao."""
        self._print_header("STEP 9: AUTO-DETECT & APPLY PCA FOR HIGH CORRELATION PAIRS")
        
        clustering_features = [
            'Total_Spent_Transformed',
            'TotalPurchases',
            'Product_HHI',
            'Store_Preference',
            'Web_Engagement'
        ]
        
        existing_features = [f for f in clustering_features if f in self.df.columns]
        
        if len(existing_features) < 2:
            print("Not enough clustering features for correlation analysis. Skipping PCA.")
            return existing_features
        
        # Compute correlation matrix
        corr_matrix = self.df[existing_features].corr()
        
        print(f"CORRELATION MATRIX ({len(existing_features)}x{len(existing_features)}):")
        print(f"{'-' * 100}")
        print(f"{'':>30}", end='')
        for col in existing_features:
            print(f"{col[:15]:>15}", end='')
        print()
        
        for i, col1 in enumerate(existing_features):
            print(f"{i+1}. {col1[:27]:<27}", end='')
            for col2 in existing_features:
                val = corr_matrix.loc[col1, col2]
                flag = '!' if (col1 != col2 and abs(val) > self.CORRELATION_THRESHOLD) else ' '
                print(f"{val:>14.3f}{flag}", end='')
            print()
        print()
        
        # Find high correlation pairs
        high_corr_pairs = []
        for i in range(len(existing_features)):
            for j in range(i+1, len(existing_features)):
                corr_val = abs(corr_matrix.iloc[i, j])
                if corr_val >= self.CORRELATION_THRESHOLD:
                    high_corr_pairs.append({
                        'feature1': existing_features[i],
                        'feature2': existing_features[j],
                        'correlation': corr_val
                    })
        
        # Display high correlation pairs
        if high_corr_pairs:
            print(f"HIGH CORRELATION PAIRS (|r| >= {self.CORRELATION_THRESHOLD}):")
            print(f"{'-' * 100}")
            for pair in high_corr_pairs:
                print(f"   {pair['feature1']:<30} ↔ {pair['feature2']:<30} : r = {pair['correlation']:.3f}")
            print()
        else:
            print(f"No high correlation pairs found (|r| >= {self.CORRELATION_THRESHOLD})")
            print()
            self.log_action("PCA Check", "No high correlation pairs detected")
            return existing_features
        
        # Apply PCA to each high correlation pair
        remaining_features = existing_features.copy()
        
        for pair in high_corr_pairs:
            feat1 = pair['feature1']
            feat2 = pair['feature2']
            corr_val = pair['correlation']
            
            # Check if both features still exist (not already transformed)
            if feat1 not in remaining_features or feat2 not in remaining_features:
                continue
            
            print(f"Applying PCA to [{feat1}, {feat2}] (r = {corr_val:.3f})...")
            
            try:
                # Extract data for PCA
                data_for_pca = self.df[[feat1, feat2]].copy()
                
                # Standardize data for PCA
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                data_scaled = scaler.fit_transform(data_for_pca)
                
                # Apply PCA
                pca = PCA(n_components=1)
                pc1_data = pca.fit_transform(data_scaled)
                
                variance_explained = pca.explained_variance_ratio_[0]
                loadings = pca.components_[0]
                
                # Create composite feature name
                pc_name = f"PC1_{feat1.split('_')[0]}_{feat2.split('_')[0]}"
                self.df[pc_name] = pc1_data.flatten()
                
                # Remove original features from clustering set
                remaining_features.remove(feat1)
                remaining_features.remove(feat2)
                remaining_features.append(pc_name)
                
                # Log PCA info
                pca_info = {
                    'feature1': feat1,
                    'feature2': feat2,
                    'correlation': float(corr_val),
                    'pc_name': pc_name,
                    'variance_explained': float(variance_explained),
                    'loadings': {
                        feat1: float(loadings[0]),
                        feat2: float(loadings[1])
                    }
                }
                
                self.pca_applied_pairs.append(pca_info)
                self.feature_stats[f'{pc_name}_info'] = pca_info
                
                print(f"   ✓ Created {pc_name}")
                print(f"   ✓ Variance explained: {variance_explained*100:.2f}%")
                print(f"   ✓ Formula: {loadings[0]:.4f}×{feat1} + {loadings[1]:.4f}×{feat2}")
                print()
                
                self.log_action("Apply PCA", 
                               f"{pc_name}: Variance={variance_explained*100:.2f}%, "
                               f"Corr({feat1}, {feat2})={corr_val:.3f}")
                
            except Exception as e:
                print(f"   ✗ Failed: {e}")
                print()
                self.log_action("Apply PCA FAILED", f"{feat1} x {feat2}: {str(e)}")
        
        print(f"CLUSTERING FEATURES AFTER PCA:")
        print(f"{'-' * 100}")
        for i, feat in enumerate(remaining_features, 1):
            print(f"   {i}. {feat}")
        print()
        
        self.log_action("PCA Complete", f"Applied to {len(self.pca_applied_pairs)} pair(s)")
        
        return remaining_features
    
    # ============================================================================
    # STEP 11: SELECT FINAL FEATURES (4-5 clustering + 8 reference)
    # ============================================================================
    
    def select_final_features(self):
        self._print_header("STEP 10: SELECT FINAL FEATURES (4-5 CLUSTERING + 8 REFERENCE)")
        
        # Get clustering features (may include PCA components)
        clustering_features = self.apply_pca_for_high_correlation()
        
        reference_features = [
            'Wine_Preference',
            'Meat_Preference',
            'Fish_Preference',
            'Fruit_Preference',
            'Sweet_Preference',
            'Gold_Preference',
            'Dominant_Product',
            'Top_Product_Share'
        ]
        
        all_features = clustering_features + reference_features
        existing = [f for f in all_features if f in self.df.columns]
        
        if len(existing) < len(clustering_features):
            missing = set(clustering_features) - set(existing)
            print(f"WARNING: Missing clustering features: {missing}")
            print()
        
        self.df_engineered = self.df[existing].copy()
        
        existing_clustering = [f for f in clustering_features if f in existing]
        existing_reference = [f for f in reference_features if f in existing]
        
        print(f"CLUSTERING FEATURES ({len(existing_clustering)} features):")
        print(f"{'-' * 100}")
        for i, feat in enumerate(existing_clustering, 1):
            mean_val = self.df_engineered[feat].mean()
            std_val = self.df_engineered[feat].std()
            var_val = self.df_engineered[feat].var()
            
            # Check if PCA component
            if feat.startswith('PC1_'):
                source = "(PCA composite)"
            else:
                is_transformed = f'{feat}'.replace('_Transformed', '_transform') in self.feature_stats
                source = "(Transformed)" if is_transformed else "(Original)"
            
            print(f"   {i}. {feat:<30} | Mean: {mean_val:>10.3f} | Std: {std_val:>8.3f} | Var: {var_val:>10.3f} | {source}")
        
        print()
        print(f"REFERENCE FEATURES ({len(existing_reference)} - POST-HOC):")
        print(f"{'-' * 100}")
        for i, feat in enumerate(existing_reference, 1):
            if feat == 'Dominant_Product':
                print(f"   {i}. {feat:<30} | Type: Categorical")
            else:
                mean_val = self.df_engineered[feat].mean()
                print(f"   {i}. {feat:<30} | Mean: {mean_val:>10.3f}")
        
        print()
        print(f"CORRELATION MATRIX (CLUSTERING FEATURES):")
        print(f"{'-' * 100}")
        
        if len(existing_clustering) >= 2:
            corr_matrix = self.df_engineered[existing_clustering].corr()
            
            print(f"{'':>30}", end='')
            for i in range(len(existing_clustering)):
                print(f"{i+1:>8}", end='')
            print()
            
            max_corr = 0.0
            max_pair = ('', '')
            
            for i, feat1 in enumerate(existing_clustering):
                print(f"{i+1:>2}. {feat1[:27]:<27}", end='')
                for feat2 in existing_clustering:
                    corr_val = corr_matrix.loc[feat1, feat2]
                    if feat1 != feat2 and abs(corr_val) > max_corr:
                        max_corr = abs(corr_val)
                        max_pair = (feat1, feat2)
                    
                    flag = '!' if (feat1 != feat2 and abs(corr_val) > 0.7) else ''
                    print(f"{corr_val:>7.2f}{flag:1}", end='')
                print()
            
            print(f"{'-' * 100}")
            print(f"Max correlation: {max_corr:.3f} ({max_pair[0]} ↔ {max_pair[1]})")
            
            if max_corr < 0.7:
                print(f"✓ EXCELLENT: All correlations < 0.7 (good independence)")
            elif max_corr < 0.8:
                print(f"✓ GOOD: All correlations < 0.8 (acceptable)")
            else:
                print(f"⚠ WARNING: Some correlations >= 0.8 (possible multicollinearity)")
            print()
            
            self.feature_stats['max_correlation'] = float(max_corr)
            self.feature_stats['max_corr_pair'] = max_pair
        
        print(f"IMPROVEMENTS (v5 - AUTO-DETECT + AUTO PCA):")
        print(f"   ✓ Auto-detect transform (Box-Cox or Yeo-Johnson)")
        print(f"   ✓ Dropped AvgPerPurchase_Transformed (corr 0.966 with Total_Spent)")
        print(f"   ✓ Dropped Product_Entropy (corr -0.98 with HHI)")
        print(f"   ✓ {len(existing_clustering)} clustering features")
        print(f"   ✓ {len(self.pca_applied_pairs)} PCA reduction(s) applied")
        print()
        
        print(f"Final dataset: {self.df_engineered.shape}")
        print()
        
        self.log_action("Select features", f"{len(existing)} ({len(existing_clustering)} clustering + {len(existing_reference)} ref)")
    
    # ============================================================================
    # STEP 12: VALIDATE FEATURES
    # ============================================================================
    
    def validate_features(self):
        self._print_header("STEP 11: VALIDATE FEATURES")
        
        print(f"DATA QUALITY CHECKS:")
        print(f"{'-' * 100}")
        
        missing = self.df_engineered.isnull().sum()
        print(f"   Missing values   : {missing.sum()} {'(PASS)' if missing.sum() == 0 else '(WARNING)'}")
        
        dup_count = self.df_engineered.duplicated().sum()
        print(f"   Duplicate rows   : {dup_count} ({dup_count/len(self.df_engineered)*100:.2f}%)")
        
        clustering_cols = [c for c in self.df_engineered.columns if not c.startswith(('Wine', 'Meat', 'Fish', 'Fruit', 'Sweet', 'Gold', 'Dominant', 'Top'))]
        
        print()
        print(f"   Variance check (clustering features):")
        low_var = []
        for col in clustering_cols:
            if col in self.df_engineered.columns:
                var = self.df_engineered[col].var()
                status = "HIGH" if var > 1 else "LOW"
                print(f"      {col:<30} : {var:>10.3f}  ({status})")
                if var < 0.01:
                    low_var.append(col)
        
        if low_var:
            print(f"\n   WARNING: Low variance: {low_var}")
        else:
            print(f"\n   OK: Adequate variance")
        
        inf_count = np.isinf(self.df_engineered.select_dtypes(include=[np.number])).sum().sum()
        print(f"   Infinite values  : {inf_count} {'(PASS)' if inf_count == 0 else '(WARNING)'}")
        
        print(f"{'-' * 100}")
        print()
        
        self.log_action("Validate", "Completed")
    
    # ============================================================================
    # STEP 13: EXPORT DATASET & REPORT
    # ============================================================================
    
    def export_dataset(self):
        self._print_header("STEP 12: EXPORT DATASET")
        
        try:
            self.df_engineered.to_csv(self.output_csv, index=False)
            file_size = os.path.getsize(self.output_csv) / 1024
            
            print(f"Dataset exported")
            print(f"   Path   : {self.output_csv}")
            print(f"   Shape  : {self.df_engineered.shape[0]:,} rows x {self.df_engineered.shape[1]} columns")
            print(f"   Size   : {file_size:.2f} KB")
            print()
            
            self.log_action("Export dataset", self.output_csv)
            return True
        except Exception as e:
            print(f"ERROR: {e}")
            self.log_action("Export FAILED", str(e))
            return False
    
    def export_report(self):
        self._print_header("STEP 13: EXPORT REPORT")
        
        try:
            with open(self.report_file, 'w', encoding='utf-8') as f:
                f.write("=" * 100 + "\n")
                f.write("PRODUCT+CHANNEL FEATURE ENGINEERING REPORT (v5 - AUTO-DETECT TRANSFORM + AUTO PCA)\n")
                f.write("=" * 100 + "\n\n")
                
                f.write(f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Input     : {self.input_path}\n")
                f.write(f"Output    : {self.output_csv}\n")
                f.write(f"Strategy  : Volume-Based + Auto-Detect Transform + Auto-Detect PCA\n\n")
                
                f.write("IMPROVEMENTS (v5):\n")
                f.write("-" * 100 + "\n")
                f.write("OK: Auto-detect transform method (Box-Cox for positive data, Yeo-Johnson for data with zeros)\n")
                f.write("OK: Transformed Total_Spent (auto-detected method)\n")
                f.write("OK: DROPPED AvgPerPurchase_Transformed (correlation 0.966 with Total_Spent)\n")
                f.write("OK: Dropped Product_Entropy (correlation -0.98 with HHI)\n")
                f.write("OK: Auto-detect PCA for highly correlated pairs (|r| >= 0.7)\n")
                f.write(f"OK: Applied PCA to {len(self.pca_applied_pairs)} feature pair(s)\n")
                f.write("OK: Reduced clustering features from 5 to 4 (if PCA applied)\n")
                f.write("OK: Better independence between clustering features\n\n")
                
                f.write("TRANSFORM METHOD DECISION LOGIC:\n")
                f.write("-" * 100 + "\n")
                f.write("- Box-Cox: Used when min > 0 (data purely positive)\n")
                f.write("- Yeo-Johnson: Used when min <= 0 (data contains zeros or negatives)\n")
                f.write("- Auto-detection: Applied independently to each feature\n")
                f.write("- Threshold: Transform only if |skewness| > 0.8\n\n")
                
                f.write("PCA REDUCTION LOGIC:\n")
                f.write("-" * 100 + "\n")
                f.write("- Auto-detect high correlation pairs: |r| >= 0.7\n")
                f.write("- Apply PCA(n_components=1) to composite features\n")
                f.write("- Result: n_components=1 captures 90%+ variance from 2 correlated features\n")
                f.write("- Benefit: Eliminates multicollinearity, improves K-Means clustering\n\n")
                
                if self.pca_applied_pairs:
                    f.write("PCA REDUCTIONS APPLIED:\n")
                    f.write("-" * 100 + "\n")
                    for i, pca_info in enumerate(self.pca_applied_pairs, 1):
                        f.write(f"{i}. {pca_info['pc_name']}\n")
                        f.write(f"   Source features: {pca_info['feature1']} x {pca_info['feature2']}\n")
                        f.write(f"   Correlation: {pca_info['correlation']:.3f}\n")
                        f.write(f"   Variance explained: {pca_info['variance_explained']*100:.2f}%\n")
                        f.write(f"   Formula: {pca_info['loadings'][pca_info['feature1']]:.4f}×{pca_info['feature1']} + {pca_info['loadings'][pca_info['feature2']]:.4f}×{pca_info['feature2']}\n\n")
                else:
                    f.write("PCA REDUCTIONS APPLIED: None (all correlations < 0.7)\n\n")
                
                f.write(f"FINAL FEATURES ({self.df_engineered.shape[1]} columns):\n")
                f.write("-" * 100 + "\n\n")
                
                f.write("CLUSTERING FEATURES:\n")
                clustering_cols = [c for c in self.df_engineered.columns if not c.startswith(('Wine', 'Meat', 'Fish', 'Fruit', 'Sweet', 'Gold', 'Dominant', 'Top'))]
                for i, col in enumerate(clustering_cols, 1):
                    f.write(f"{i}. {col}\n")
                    if f'{col}_info' in self.feature_stats:
                        f.write(f"   [PCA Composite Feature]\n")
                    f.write("\n")
                
                f.write("REFERENCE FEATURES (post-hoc):\n")
                ref_cols = [c for c in self.df_engineered.columns if c.startswith(('Wine', 'Meat', 'Fish', 'Fruit', 'Sweet', 'Gold', 'Dominant', 'Top'))]
                for i, col in enumerate(ref_cols, 1):
                    f.write(f"{i}. {col} (not for clustering)\n")
                
                f.write("\n")
                
                f.write("CORRELATION ANALYSIS:\n")
                f.write("-" * 100 + "\n")
                if 'max_correlation' in self.feature_stats:
                    f.write(f"Max correlation : {self.feature_stats['max_correlation']:.3f}\n")
                    if 'max_corr_pair' in self.feature_stats:
                        f.write(f"Between         : {self.feature_stats['max_corr_pair'][0]} ↔ {self.feature_stats['max_corr_pair'][1]}\n")
                f.write("\n")
                
                f.write("PROCESSING LOG:\n")
                f.write("-" * 100 + "\n")
                for i, log in enumerate(self.processing_log, 1):
                    f.write(f"{i:>3}. [{log['timestamp']}] {log['action']}\n")
                    if log['details']:
                        f.write(f"     {log['details']}\n")
                
                f.write("\n" + "=" * 100 + "\n")
                f.write("END OF REPORT\n")
                f.write("=" * 100 + "\n")
            
            file_size = os.path.getsize(self.report_file) / 1024
            
            print(f"Report exported")
            print(f"   Path  : {self.report_file}")
            print(f"   Size  : {file_size:.2f} KB")
            print()
            
            self.log_action("Export report", self.report_file)
            return True
        except Exception as e:
            print(f"ERROR: {e}")
            self.log_action("Export report FAILED", str(e))
            return False
    
    # ============================================================================
    # STEP 14: VISUALIZATION
    # ============================================================================
    
    def plot_histograms(self, graph_dir):
        """Vẽ histogram + KDE cho clustering features."""
        os.makedirs(graph_dir, exist_ok=True)
        
        clustering_cols = [c for c in self.df_engineered.columns if not c.startswith(('Wine', 'Meat', 'Fish', 'Fruit', 'Sweet', 'Gold', 'Dominant', 'Top'))]
        
        for col in clustering_cols:
            if col not in self.df_engineered.columns:
                continue
            
            try:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                sns.histplot(self.df_engineered[col].dropna(), bins=50, kde=True, ax=ax, color='steelblue')
                
                mean_val = self.df_engineered[col].mean()
                median_val = self.df_engineered[col].median()
                
                ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
                ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.3f}')
                
                if col.startswith('PC1_'):
                    title_suffix = "(PCA Composite)"
                else:
                    title_suffix = "(Original)" if f'{col}_transform' not in self.feature_stats else "(Transformed)"
                
                ax.set_title(f'Distribution: {col} {title_suffix}', fontsize=14, fontweight='bold')
                ax.set_xlabel(col, fontsize=12)
                ax.set_ylabel('Frequency', fontsize=12)
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                fname = os.path.join(graph_dir, f"{col}_histogram.png")
                fig.tight_layout()
                fig.savefig(fname, dpi=150, bbox_inches='tight')
                plt.close(fig)
                
                print(f"   Saved: {col}_histogram.png")
            except Exception as e:
                print(f"   ERROR: {col} - {e}")
                plt.close('all')
    
    def plot_boxplots(self, graph_dir):
        """Vẽ boxplot cho clustering features."""
        os.makedirs(graph_dir, exist_ok=True)
        
        clustering_cols = [c for c in self.df_engineered.columns if not c.startswith(('Wine', 'Meat', 'Fish', 'Fruit', 'Sweet', 'Gold', 'Dominant', 'Top'))]
        
        for col in clustering_cols:
            if col not in self.df_engineered.columns:
                continue
            
            try:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                sns.boxplot(y=self.df_engineered[col].dropna(), ax=ax, color='lightcoral')
                
                if col.startswith('PC1_'):
                    title_suffix = "(PCA Composite)"
                else:
                    title_suffix = "(Original)" if f'{col}_transform' not in self.feature_stats else "(Transformed)"
                
                ax.set_title(f'Boxplot: {col} {title_suffix}', fontsize=14, fontweight='bold')
                ax.set_ylabel(col, fontsize=12)
                ax.grid(True, alpha=0.3, axis='y')
                
                fname = os.path.join(graph_dir, f"{col}_boxplot.png")
                fig.tight_layout()
                fig.savefig(fname, dpi=150, bbox_inches='tight')
                plt.close(fig)
                
                print(f"   Saved: {col}_boxplot.png")
            except Exception as e:
                print(f"   ERROR: {col} - {e}")
                plt.close('all')
    
    def plot_correlation_heatmap(self, graph_dir):
        """Vẽ correlation heatmap (clustering features only)."""
        os.makedirs(graph_dir, exist_ok=True)
        
        clustering_cols = [c for c in self.df_engineered.columns if not c.startswith(('Wine', 'Meat', 'Fish', 'Fruit', 'Sweet', 'Gold', 'Dominant', 'Top'))]
        existing_cols = [c for c in clustering_cols if c in self.df_engineered.columns]
        
        if len(existing_cols) < 2:
            print("   Skipped: Not enough features for heatmap")
            return
        
        try:
            corr = self.df_engineered[existing_cols].corr()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            sns.heatmap(corr, annot=True, fmt=".3f", cmap="RdYlGn_r", vmin=-1, vmax=1, 
                       center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
            
            title = f'Correlation Matrix ({len(existing_cols)} Clustering Features)'
            if self.pca_applied_pairs:
                title += f' - After PCA Reduction'
            ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
            
            fname = os.path.join(graph_dir, "correlation_heatmap.png")
            fig.tight_layout()
            fig.savefig(fname, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"   Saved: correlation_heatmap.png")
        except Exception as e:
            print(f"   ERROR: Failed heatmap - {e}")
            plt.close('all')
    
    def plot_before_after_transform(self, graph_dir):
        """Vẽ before/after cho Total_Spent transform."""
        os.makedirs(graph_dir, exist_ok=True)
        
        if 'Total_Spent_transform' not in self.feature_stats:
            print("   Skipped: No transform applied")
            return
        
        if self.feature_stats['Total_Spent_transform'].get('method') == 'NO_TRANSFORM':
            print("   Skipped: NO_TRANSFORM applied")
            return
        
        if 'Total_Spent_original' not in self.feature_stats:
            return
        
        try:
            df_original = pd.read_csv(self.input_path)
            
            mnt_cols = ['MntWines', 'MntFruits', 'MntMeatProducts', 
                        'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
            existing = [c for c in mnt_cols if c in df_original.columns]
            
            if not existing:
                return
            
            original_data = df_original[existing].sum(axis=1)
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Before
            sns.histplot(original_data.dropna(), bins=40, kde=True, ax=axes[0], color='steelblue')
            axes[0].set_title(f'Before: Total_Spent', fontsize=12, fontweight='bold')
            axes[0].set_xlabel('Total_Spent', fontsize=11)
            axes[0].set_ylabel('Frequency', fontsize=11)
            axes[0].axvline(original_data.mean(), color='red', linestyle='--', 
                           label=f'Mean: {original_data.mean():.2f}')
            axes[0].axvline(original_data.median(), color='green', linestyle='--', 
                           label=f'Median: {original_data.median():.2f}')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            axes[0].text(0.02, 0.98, f"Skewness: {original_data.skew():.3f}", 
                        transform=axes[0].transAxes, fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # After
            transformed_data = self.df_engineered['Total_Spent_Transformed']
            sns.histplot(transformed_data.dropna(), bins=40, kde=True, ax=axes[1], color='seagreen')
            
            transform_info = self.feature_stats['Total_Spent_transform']
            method = transform_info.get('method', 'Unknown')
            
            axes[1].set_title(f'After: Total_Spent ({method})', fontsize=12, fontweight='bold')
            axes[1].set_xlabel('Total_Spent (transformed)', fontsize=11)
            axes[1].set_ylabel('Frequency', fontsize=11)
            axes[1].axvline(transformed_data.mean(), color='red', linestyle='--', 
                           label=f'Mean: {transformed_data.mean():.3f}')
            axes[1].axvline(transformed_data.median(), color='green', linestyle='--', 
                           label=f'Median: {transformed_data.median():.3f}')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            axes[1].text(0.02, 0.98, 
                        f"Skewness: {transform_info['new_skew']:.3f}\nLambda: {transform_info['lambda']:.3f}\nMethod: {method}", 
                        transform=axes[1].transAxes, fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
            
            fname = os.path.join(graph_dir, "Total_Spent_before_after.png")
            fig.tight_layout()
            fig.savefig(fname, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"   Saved: Total_Spent_before_after.png")
        except Exception as e:
            print(f"   ERROR: Failed before/after - {e}")
            plt.close('all')
    
    def plot_reference_distributions(self, graph_dir):
        """Vẽ distribution cho reference features."""
        os.makedirs(graph_dir, exist_ok=True)
        
        # Dominant_Product distribution
        if 'Dominant_Product' in self.df_engineered.columns:
            try:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                prod_counts = self.df_engineered['Dominant_Product'].value_counts()
                
                ax.bar(prod_counts.index, prod_counts.values, color='steelblue', alpha=0.7)
                ax.set_xlabel('Product', fontsize=12)
                ax.set_ylabel('Count', fontsize=12)
                ax.set_title('Dominant Product Distribution (Reference Feature)', fontsize=14, fontweight='bold')
                ax.set_xticklabels(prod_counts.index, rotation=45, ha='right')
                ax.grid(True, alpha=0.3, axis='y')
                
                total = prod_counts.sum()
                for idx, (prod, count) in enumerate(prod_counts.items()):
                    pct = count / total * 100
                    ax.text(idx, count, f'{count}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9)
                
                fname = os.path.join(graph_dir, "Dominant_Product_distribution.png")
                fig.tight_layout()
                fig.savefig(fname, dpi=150, bbox_inches='tight')
                plt.close(fig)
                
                print(f"   Saved: Dominant_Product_distribution.png")
            except Exception as e:
                print(f"   ERROR: Failed Dominant_Product - {e}")
                plt.close('all')
    
    def generate_all_plots(self, graph_dir):
        """Tạo tất cả plots."""
        self._print_header("STEP 14: GENERATE VISUALIZATIONS")
        
        print(f"Creating plots in: {graph_dir}")
        print()
        
        print(f"Creating histograms...")
        self.plot_histograms(graph_dir)
        
        print()
        print(f"Creating boxplots...")
        self.plot_boxplots(graph_dir)
        
        print()
        print(f"Creating correlation heatmap...")
        self.plot_correlation_heatmap(graph_dir)
        
        print()
        print(f"Creating before/after transform...")
        self.plot_before_after_transform(graph_dir)
        
        print()
        print(f"Creating reference distributions...")
        self.plot_reference_distributions(graph_dir)
        
        print()
        print(f"All plots saved to: {graph_dir}")
        print()
        
        self.log_action("Generate plots", f"Saved to {graph_dir}")
    
    # ============================================================================
    # MAIN PIPELINE
    # ============================================================================
    
    def run_engineering(self):
        print("\n" + "=" * 100)
        print("PRODUCT+CHANNEL FEATURE ENGINEERING (v5 - AUTO-DETECT TRANSFORM + AUTO PCA)".center(100))
        print("=" * 100 + "\n")
        
        start = datetime.now()
        
        if not self.load_data():
            return False
        
        # Volume features
        self.create_total_spent()
        self.transform_total_spent()
        self.create_total_purchases()
        
        # Reference features
        self.create_product_preferences_reference()
        
        # Diversity/concentration
        self.create_product_hhi()
        self.create_dominant_product()
        
        # Channel features
        self.create_store_preference()
        self.create_web_engagement()
        
        # PCA Reduction & Finalize
        self.select_final_features()
        self.validate_features()
        
        # Export
        if not self.export_dataset():
            return False
        
        self.export_report()
        
        # VISUALIZATION
        graph_dir = r"C:\Project\Machine_Learning\Machine_Learning\graph\Feature Extraction & Engineering_graph\ProductChannel"
        self.generate_all_plots(graph_dir)
        
        elapsed = (datetime.now() - start).total_seconds()
        
        print("=" * 100)
        print(f"PIPELINE COMPLETED in {elapsed:.2f}s".center(100))
        print("=" * 100 + "\n")
        
        return True



# ================================================================================
# CLASS 3: RFM FEATURE ENGINEERING (AUTO-DETECT TRANSFORM + AUTO PCA)
# ================================================================================

class RFMFeatureEngineering:
    """
    Feature Engineering
    Mục tiêu: Phân khúc khách hàng theo giá trị kinh tế (RFM Analysis)
    Các biến đầu ra cuối cùng (4–6 biến clustering):
    - Nếu tương quan ≥ 0.7: Áp dụng PCA → PC1 (tổng hợp) → còn 5 biến
    - Nếu tương quan < 0.7: Giữ nguyên 6 biến gốc

    - 6 biến mặc định:
        + Recency (số ngày kể từ lần mua gần nhất)
        + TotalPurchases (tần suất – số lượng giao dịch)
        + Total_Spent (giá trị tiền tệ – tổng chi tiêu)
        + AvgPerPurchase_Transformed (biến đổi tự động – AOV)
        + Income (khả năng tài chính)
        + Income_per_Family_Member_Transformed (biến đổi tự động – áp lực tài chính)

    Pipeline:
    - Bước 1: Load dữ liệu
    - Bước 2: Tạo TotalPurchases (tổng số lượng giao dịch)
    - Bước 3: Tạo Total_Spent (tổng chi tiêu)
    - Bước 4: Tạo AvgPerPurchase (AOV = Total_Spent / TotalPurchases)
    - Bước 5: Biến đổi AvgPerPurchase (tự động Box-Cox hoặc Yeo-Johnson)
    - Bước 6: Tạo Income_per_Family_Member (Income / (2 + TotalChildren))
    - Bước 7: Biến đổi Income_per_Family_Member (tự động Box-Cox hoặc Yeo-Johnson)
    - Bước 8: Xác định các cặp có tương quan cao & áp dụng PCA nếu cần
    - Bước 9: Chọn biến cuối cùng (4–6 clustering)
    - Bước 10: Kiểm tra/validate
    - Bước 11: Xuất dataset + báo cáo
    - Bước 12: Tạo biểu đồ trực quan hóa
    """
    
    def __init__(self, input_path, output_dir, report_dir):
        """
        Khoi tao RFM Feature Engineering.
        
        Args:
            input_path (str): Duong dan dataset cleaned
            output_dir (str): Thu muc luu dataset output
            report_dir (str): Thu muc luu report
        """
        self.input_path = input_path
        self.output_dir = output_dir
        self.report_dir = report_dir
        
        # Create directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(report_dir, exist_ok=True)
        
        # Output paths
        self.output_csv = os.path.join(output_dir, "Customer_Behavior_RFM.csv")
        self.report_file = os.path.join(report_dir, "RFM_Engineering_Report.txt")
        
        # Data storage
        self.df = None
        self.df_engineered = None
        self.feature_stats = {}
        self.processing_log = []
        self.pca_applied_pairs = []  # Track PCA applications
        
        self.SKEW_THRESHOLD = 1.0
        self.CORRELATION_THRESHOLD = 0.7  # Threshold for PCA
    
    def log_action(self, action, details=""):
        """Ghi log hanh dong."""
        self.processing_log.append({
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'action': action,
            'details': details
        })
    
    def _print_header(self, title, width=100, char='='):
        """In header co format."""
        print(char * width)
        print(f"{title:^{width}}")
        print(char * width)
        print()
    
    # ============================================================================
    # STEP 1: LOAD DATA
    # ============================================================================
    
    def load_data(self):
        """Load cleaned dataset."""
        self._print_header("RFM FEATURE ENGINEERING (v5 - AUTO-DETECT TRANSFORM + AUTO PCA) - LOAD DATA")
        
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
        """Tao TotalPurchases (Frequency in RFM)."""
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
            'min': float(self.df['TotalPurchases'].min()),
            'max': float(self.df['TotalPurchases'].max()),
            'mean': float(self.df['TotalPurchases'].mean()),
            'median': float(self.df['TotalPurchases'].median()),
            'std': float(self.df['TotalPurchases'].std()),
            'skewness': float(self.df['TotalPurchases'].skew())
        }
        
        zero_count = (self.df['TotalPurchases'] == 0).sum()
        zero_pct = zero_count / len(self.df) * 100
        
        print(f"Created 'TotalPurchases' (Frequency)")
        print(f"   Formula: Sum of [{', '.join(existing_cols)}]")
        print(f"   Range: [{stats['min']:.0f}, {stats['max']:.0f}]")
        print(f"   Mean: {stats['mean']:.1f}, Median: {stats['median']:.1f}")
        print(f"   Std: {stats['std']:.1f}, Skewness: {stats['skewness']:.3f}")
        print(f"   Zero count: {zero_count} ({zero_pct:.1f}%)")
        print()
        
        self.feature_stats['TotalPurchases'] = stats
        self.log_action("Create TotalPurchases", f"Mean: {stats['mean']:.1f}, Skew: {stats['skewness']:.3f}")
    
    # ============================================================================
    # STEP 3: CREATE TOTAL_SPENT (MONETARY)
    # ============================================================================
    
    def create_total_spent(self):
        """Tao Total_Spent (Monetary in RFM)."""
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
            'min': float(self.df['Total_Spent'].min()),
            'max': float(self.df['Total_Spent'].max()),
            'mean': float(self.df['Total_Spent'].mean()),
            'median': float(self.df['Total_Spent'].median()),
            'std': float(self.df['Total_Spent'].std()),
            'skewness': float(self.df['Total_Spent'].skew())
        }
        
        zero_count = (self.df['Total_Spent'] == 0).sum()
        zero_pct = zero_count / len(self.df) * 100
        
        print(f"Created 'Total_Spent' (Monetary)")
        print(f"   Formula: Sum of [{', '.join(existing_cols)}]")
        print(f"   Range: [{stats['min']:,.0f}, {stats['max']:,.0f}]")
        print(f"   Mean: {stats['mean']:,.0f}, Median: {stats['median']:,.0f}")
        print(f"   Std: {stats['std']:,.0f}, Skewness: {stats['skewness']:.3f}")
        print(f"   Zero count: {zero_count} ({zero_pct:.1f}%)")
        print()
        
        self.feature_stats['Total_Spent'] = stats
        self.log_action("Create Total_Spent", f"Mean: {stats['mean']:,.0f}, Skew: {stats['skewness']:.3f}")
    
    # ============================================================================
    # STEP 4: CREATE AVGPERPURCHASE (AVERAGE ORDER VALUE)
    # ============================================================================
    
    def create_avg_per_purchase(self):
        """Tao AvgPerPurchase (Average Order Value)."""
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
            'min': float(self.df['AvgPerPurchase'].min()),
            'max': float(self.df['AvgPerPurchase'].max()),
            'mean': float(self.df['AvgPerPurchase'].mean()),
            'median': float(self.df['AvgPerPurchase'].median()),
            'std': float(self.df['AvgPerPurchase'].std()),
            'skewness': float(self.df['AvgPerPurchase'].skew())
        }
        
        zero_count = (self.df['AvgPerPurchase'] == 0).sum()
        zero_pct = zero_count / len(self.df) * 100
        
        print(f"Created 'AvgPerPurchase' (Average Order Value)")
        print(f"   Formula: Total_Spent / TotalPurchases")
        print(f"   Range: [{stats['min']:,.2f}, {stats['max']:,.2f}]")
        print(f"   Mean: {stats['mean']:,.2f}, Median: {stats['median']:,.2f}")
        print(f"   Std: {stats['std']:,.2f}, Skewness: {stats['skewness']:.3f}")
        print(f"   Zero count: {zero_count} ({zero_pct:.1f}%)")
        print()
        
        self.feature_stats['AvgPerPurchase'] = stats
        self.log_action("Create AvgPerPurchase", f"Mean: {stats['mean']:,.2f}, Skew: {stats['skewness']:.3f}")
    
    # ============================================================================
    # STEP 5: TRANSFORM AVGPERPURCHASE (AUTO-DETECT METHOD)
    # ============================================================================
    
    def transform_avg_per_purchase(self):
        """Apply auto-detect transform to AvgPerPurchase if skewed."""
        self._print_header("STEP 4: TRANSFORM AVGPERPURCHASE (AUTO-DETECT METHOD)")
        
        if 'AvgPerPurchase' not in self.df.columns:
            print("AvgPerPurchase not found. Skipping.")
            return
        
        original_skew = self.df['AvgPerPurchase'].skew()
        feature_data = self.df['AvgPerPurchase']
        
        print(f"Checking skewness of AvgPerPurchase:")
        print(f"   Original skewness: {original_skew:.3f}")
        print(f"   Min: {feature_data.min():.3f}, Max: {feature_data.max():.3f}")
        print(f"   Threshold: {self.SKEW_THRESHOLD}")
        print()
        
        # Kiem tra xem co can transform khong
        if abs(original_skew) < self.SKEW_THRESHOLD:
            print(f"   Decision: NO TRANSFORM (|skew| < {self.SKEW_THRESHOLD})")
            self.df['AvgPerPurchase_Transformed'] = self.df['AvgPerPurchase']
            self.feature_stats['AvgPerPurchase_transform'] = {
                'method': 'NO_TRANSFORM',
                'original_skew': float(original_skew)
            }
            self.log_action("Transform AvgPerPurchase", "No transform needed")
            print()
            return
        
        # Chon transform method
        has_zero = (feature_data == 0).any()
        has_negative = (feature_data < 0).any()
        zero_count = (feature_data == 0).sum() if has_zero else 0
        zero_pct = (zero_count / len(feature_data) * 100) if has_zero else 0.0
        
        try:
            if has_negative:
                print(f"   Zero count: {zero_count}, Negative count: {(feature_data < 0).sum()}")
                print(f"   Decision: YEO-JOHNSON (data co gia tri am)")
                method = "Yeo-Johnson"
                transformed_data, lambda_param = yeojohnson(feature_data)
                
            elif has_zero:
                print(f"   Zero count: {zero_count} ({zero_pct:.1f}%)")
                print(f"   Decision: YEO-JOHNSON (data co gia tri 0)")
                method = "Yeo-Johnson"
                transformed_data, lambda_param = yeojohnson(feature_data)
                
            else:  # Thuan duong (> 0)
                print(f"   Zero count: 0")
                print(f"   Decision: BOX-COX (data thuan duong)")
                method = "Box-Cox"
                transformed_data, lambda_param = boxcox(feature_data)
            
            self.df['AvgPerPurchase_Transformed'] = transformed_data
            
            new_skew = pd.Series(transformed_data).skew()
            improvement = abs(original_skew) - abs(new_skew)
            
            print(f"Applied {method}:")
            print(f"   Lambda: {lambda_param:.3f}")
            print(f"   New skewness: {new_skew:.3f}")
            print(f"   Improvement: {improvement:.3f}")
            print()
            
            self.feature_stats['AvgPerPurchase_transform'] = {
                'method': method,
                'lambda': float(lambda_param),
                'original_skew': float(original_skew),
                'new_skew': float(new_skew),
                'improvement': float(improvement),
                'has_zero': bool(has_zero),
                'zero_count': int(zero_count),
                'has_negative': bool(has_negative)
            }
            
            self.feature_stats['AvgPerPurchase_Transformed'] = {
                'min': float(pd.Series(transformed_data).min()),
                'max': float(pd.Series(transformed_data).max()),
                'mean': float(pd.Series(transformed_data).mean()),
                'median': float(pd.Series(transformed_data).median()),
                'std': float(pd.Series(transformed_data).std()),
                'skewness': float(new_skew)
            }
            
            self.log_action("Transform AvgPerPurchase", 
                           f"{method}: Skew {original_skew:.3f} -> {new_skew:.3f}")
            
        except Exception as e:
            print(f"   ERROR: {e}")
            self.df['AvgPerPurchase_Transformed'] = self.df['AvgPerPurchase']
            self.log_action("Transform FAILED", str(e))
            print()
    
    # ============================================================================
    # STEP 6: CREATE INCOME_PER_FAMILY_MEMBER
    # ============================================================================
    
    def create_income_per_family_member(self):
        """Tao Income_per_Family_Member (Financial pressure indicator)."""
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
            'min': float(self.df['Income_per_Family_Member'].min()),
            'max': float(self.df['Income_per_Family_Member'].max()),
            'mean': float(self.df['Income_per_Family_Member'].mean()),
            'median': float(self.df['Income_per_Family_Member'].median()),
            'std': float(self.df['Income_per_Family_Member'].std()),
            'skewness': float(self.df['Income_per_Family_Member'].skew())
        }
        
        print(f"Created 'Income_per_Family_Member'")
        print(f"   Formula: Income / (2 + TotalChildren)")
        print(f"   Range: [{stats['min']:,.0f}, {stats['max']:,.0f}]")
        print(f"   Mean: {stats['mean']:,.0f}, Median: {stats['median']:,.0f}")
        print(f"   Std: {stats['std']:,.0f}, Skewness: {stats['skewness']:.3f}")
        print()
        
        self.feature_stats['Income_per_Family_Member'] = stats
        self.log_action("Create Income_per_Family_Member", 
                       f"Mean: {stats['mean']:,.0f}, Skew: {stats['skewness']:.3f}")
    
    # ============================================================================
    # STEP 7: TRANSFORM INCOME_PER_FAMILY_MEMBER (AUTO-DETECT METHOD)
    # ============================================================================
    
    def transform_income_per_family_member(self):
        """Apply auto-detect transform to Income_per_Family_Member if skewed."""
        self._print_header("STEP 6: TRANSFORM INCOME_PER_FAMILY_MEMBER (AUTO-DETECT METHOD)")
        
        if 'Income_per_Family_Member' not in self.df.columns:
            print("Income_per_Family_Member not found. Skipping.")
            return
        
        original_skew = self.df['Income_per_Family_Member'].skew()
        feature_data = self.df['Income_per_Family_Member']
        
        print(f"Checking skewness of Income_per_Family_Member:")
        print(f"   Original skewness: {original_skew:.3f}")
        print(f"   Min: {feature_data.min():.3f}, Max: {feature_data.max():.3f}")
        print(f"   Threshold: {self.SKEW_THRESHOLD}")
        print()
        
        # Kiem tra xem co can transform khong
        if abs(original_skew) < self.SKEW_THRESHOLD:
            print(f"   Decision: NO TRANSFORM (|skew| < {self.SKEW_THRESHOLD})")
            self.df['Income_per_Family_Member_Transformed'] = self.df['Income_per_Family_Member']
            self.feature_stats['Income_per_Family_Member_transform'] = {
                'method': 'NO_TRANSFORM',
                'original_skew': float(original_skew)
            }
            self.log_action("Transform Income_per_Family_Member", "No transform needed")
            print()
            return
        
        # Chon transform method
        has_zero = (feature_data == 0).any()
        has_negative = (feature_data < 0).any()
        zero_count = (feature_data == 0).sum() if has_zero else 0
        zero_pct = (zero_count / len(feature_data) * 100) if has_zero else 0.0
        
        try:
            if has_negative:
                print(f"   Zero count: {zero_count}, Negative count: {(feature_data < 0).sum()}")
                print(f"   Decision: YEO-JOHNSON (data co gia tri am)")
                method = "Yeo-Johnson"
                transformed_data, lambda_param = yeojohnson(feature_data)
                
            elif has_zero:
                print(f"   Zero count: {zero_count} ({zero_pct:.1f}%)")
                print(f"   Decision: YEO-JOHNSON (data co gia tri 0)")
                method = "Yeo-Johnson"
                transformed_data, lambda_param = yeojohnson(feature_data)
                
            else:  # Thuan duong (> 0)
                print(f"   Zero count: 0")
                print(f"   Decision: BOX-COX (data thuan duong)")
                method = "Box-Cox"
                transformed_data, lambda_param = boxcox(feature_data)
            
            self.df['Income_per_Family_Member_Transformed'] = transformed_data
            
            new_skew = pd.Series(transformed_data).skew()
            improvement = abs(original_skew) - abs(new_skew)
            
            print(f"Applied {method}:")
            print(f"   Lambda: {lambda_param:.3f}")
            print(f"   New skewness: {new_skew:.3f}")
            print(f"   Improvement: {improvement:.3f}")
            print()
            
            self.feature_stats['Income_per_Family_Member_transform'] = {
                'method': method,
                'lambda': float(lambda_param),
                'original_skew': float(original_skew),
                'new_skew': float(new_skew),
                'improvement': float(improvement),
                'has_zero': bool(has_zero),
                'zero_count': int(zero_count),
                'has_negative': bool(has_negative)
            }
            
            self.feature_stats['Income_per_Family_Member_Transformed'] = {
                'min': float(pd.Series(transformed_data).min()),
                'max': float(pd.Series(transformed_data).max()),
                'mean': float(pd.Series(transformed_data).mean()),
                'median': float(pd.Series(transformed_data).median()),
                'std': float(pd.Series(transformed_data).std()),
                'skewness': float(new_skew)
            }
            
            self.log_action("Transform Income_per_Family_Member", 
                           f"{method}: Skew {original_skew:.3f} -> {new_skew:.3f}")
            
        except Exception as e:
            print(f"   ERROR: {e}")
            self.df['Income_per_Family_Member_Transformed'] = self.df['Income_per_Family_Member']
            self.log_action("Transform FAILED", str(e))
            print()
    
    # ============================================================================
    # STEP 8: AUTO-DETECT & APPLY PCA FOR HIGH CORRELATION PAIRS (MỚI)
    # ============================================================================
    
    def apply_pca_for_high_correlation(self):
        """Tự động phát hiện và áp dụng PCA cho các cặp đặc trưng có tương quan cao."""
        self._print_header("STEP 7: AUTO-DETECT & APPLY PCA FOR HIGH CORRELATION PAIRS")
        
        clustering_features = [
            'Recency',
            'TotalPurchases',
            'Total_Spent',
            'AvgPerPurchase_Transformed',
            'Income',
            'Income_per_Family_Member_Transformed'
        ]
        
        existing_features = [f for f in clustering_features if f in self.df.columns]
        
        if len(existing_features) < 2:
            print("Not enough clustering features for correlation analysis. Skipping PCA.")
            return existing_features
        
        # Compute correlation matrix
        corr_matrix = self.df[existing_features].corr()
        
        print(f"CORRELATION MATRIX ({len(existing_features)}x{len(existing_features)}):")
        print(f"{'-' * 100}")
        print(f"{'':>30}", end='')
        for col in existing_features:
            print(f"{col[:15]:>15}", end='')
        print()
        
        for i, col1 in enumerate(existing_features):
            print(f"{i+1}. {col1[:27]:<27}", end='')
            for col2 in existing_features:
                val = corr_matrix.loc[col1, col2]
                flag = '!' if (col1 != col2 and abs(val) > self.CORRELATION_THRESHOLD) else ' '
                print(f"{val:>14.3f}{flag}", end='')
            print()
        print()
        
        # Find high correlation pairs
        high_corr_pairs = []
        for i in range(len(existing_features)):
            for j in range(i+1, len(existing_features)):
                corr_val = abs(corr_matrix.iloc[i, j])
                if corr_val >= self.CORRELATION_THRESHOLD:
                    high_corr_pairs.append({
                        'feature1': existing_features[i],
                        'feature2': existing_features[j],
                        'correlation': corr_val
                    })
        
        # Display high correlation pairs
        if high_corr_pairs:
            print(f"HIGH CORRELATION PAIRS (|r| >= {self.CORRELATION_THRESHOLD}):")
            print(f"{'-' * 100}")
            for pair in high_corr_pairs:
                print(f"   {pair['feature1']:<30} ↔ {pair['feature2']:<30} : r = {pair['correlation']:.3f}")
            print()
        else:
            print(f"No high correlation pairs found (|r| >= {self.CORRELATION_THRESHOLD})")
            print()
            self.log_action("PCA Check", "No high correlation pairs detected")
            return existing_features
        
        # Apply PCA to each high correlation pair
        remaining_features = existing_features.copy()
        
        for pair in high_corr_pairs:
            feat1 = pair['feature1']
            feat2 = pair['feature2']
            corr_val = pair['correlation']
            
            # Check if both features still exist (not already transformed)
            if feat1 not in remaining_features or feat2 not in remaining_features:
                continue
            
            print(f"Applying PCA to [{feat1}, {feat2}] (r = {corr_val:.3f})...")
            
            try:
                # Extract data for PCA
                data_for_pca = self.df[[feat1, feat2]].copy()
                
                # Standardize data for PCA
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                data_scaled = scaler.fit_transform(data_for_pca)
                
                # Apply PCA
                pca = PCA(n_components=1)
                pc1_data = pca.fit_transform(data_scaled)
                
                variance_explained = pca.explained_variance_ratio_[0]
                loadings = pca.components_[0]
                
                # Create composite feature name
                pc_name = f"PC1_{feat1.split('_')[0]}_{feat2.split('_')[0]}"
                self.df[pc_name] = pc1_data.flatten()
                
                # Remove original features from clustering set
                remaining_features.remove(feat1)
                remaining_features.remove(feat2)
                remaining_features.append(pc_name)
                
                # Log PCA info
                pca_info = {
                    'feature1': feat1,
                    'feature2': feat2,
                    'correlation': float(corr_val),
                    'pc_name': pc_name,
                    'variance_explained': float(variance_explained),
                    'loadings': {
                        feat1: float(loadings[0]),
                        feat2: float(loadings[1])
                    }
                }
                
                self.pca_applied_pairs.append(pca_info)
                self.feature_stats[f'{pc_name}_info'] = pca_info
                
                print(f"   ✓ Created {pc_name}")
                print(f"   ✓ Variance explained: {variance_explained*100:.2f}%")
                print(f"   ✓ Formula: {loadings[0]:.4f}×{feat1} + {loadings[1]:.4f}×{feat2}")
                print()
                
                self.log_action("Apply PCA", 
                               f"{pc_name}: Variance={variance_explained*100:.2f}%, "
                               f"Corr({feat1}, {feat2})={corr_val:.3f}")
                
            except Exception as e:
                print(f"   ✗ Failed: {e}")
                print()
                self.log_action("Apply PCA FAILED", f"{feat1} x {feat2}: {str(e)}")
        
        print(f"CLUSTERING FEATURES AFTER PCA:")
        print(f"{'-' * 100}")
        for i, feat in enumerate(remaining_features, 1):
            print(f"   {i}. {feat}")
        print()
        
        self.log_action("PCA Complete", f"Applied to {len(self.pca_applied_pairs)} pair(s)")
        
        return remaining_features
    
    # ============================================================================
    # STEP 9: SELECT FINAL FEATURES (CẬP NHẬT)
    # ============================================================================
    
    def select_final_features(self):
        """Select final features for RFM clustering."""
        self._print_header("STEP 8: SELECT FINAL FEATURES")
        
        # Get clustering features (may include PCA components)
        clustering_features = self.apply_pca_for_high_correlation()
        
        # Create final dataset
        existing_features = [f for f in clustering_features if f in self.df.columns]
        self.df_engineered = self.df[existing_features].copy()
        
        print(f"Selected {len(existing_features)} features for RFM clustering:")
        print()
        
        print(f"CLUSTERING FEATURES ({len(existing_features)} features):")
        print(f"{'-' * 100}")
        for i, feat in enumerate(existing_features, 1):
            dtype = str(self.df_engineered[feat].dtype)
            n_unique = self.df_engineered[feat].nunique()
            mean_val = self.df_engineered[feat].mean()
            
            # Check if PCA component
            if feat.startswith('PC1_'):
                source = "(PCA composite)"
            else:
                is_transformed = f'{feat}'.replace('_Transformed', '_transform') in self.feature_stats
                source = "(Transformed)" if is_transformed else "(Original)"
            
            print(f"   {i}. {feat:<35} ({dtype:<10}) Unique: {n_unique:>4} Mean: {mean_val:>10.2f} {source}")
        
        print()
        print(f"CORRELATION MATRIX (CLUSTERING FEATURES):")
        print(f"{'-' * 100}")
        
        if len(existing_features) >= 2:
            corr_matrix = self.df_engineered[existing_features].corr()
            
            print(f"{'':>30}", end='')
            for i in range(len(existing_features)):
                print(f"{i+1:>8}", end='')
            print()
            
            max_corr = 0.0
            max_pair = ('', '')
            
            for i, feat1 in enumerate(existing_features):
                print(f"{i+1:>2}. {feat1[:27]:<27}", end='')
                for feat2 in existing_features:
                    corr_val = corr_matrix.loc[feat1, feat2]
                    if feat1 != feat2 and abs(corr_val) > max_corr:
                        max_corr = abs(corr_val)
                        max_pair = (feat1, feat2)
                    
                    flag = '!' if (feat1 != feat2 and abs(corr_val) > 0.7) else ''
                    print(f"{corr_val:>7.2f}{flag:1}", end='')
                print()
            
            print(f"{'-' * 100}")
            print(f"Max correlation: {max_corr:.3f} ({max_pair[0]} ↔ {max_pair[1]})")
            
            if max_corr < 0.7:
                print(f"✓ EXCELLENT: All correlations < 0.7 (good independence)")
            elif max_corr < 0.8:
                print(f"✓ GOOD: All correlations < 0.8 (acceptable)")
            else:
                print(f"⚠ WARNING: Some correlations >= 0.8 (possible multicollinearity)")
            print()
            
            self.feature_stats['max_correlation'] = float(max_corr)
            self.feature_stats['max_corr_pair'] = max_pair
        
        print(f"IMPROVEMENTS (v5 - AUTO-DETECT TRANSFORM + AUTO PCA):")
        print(f"   ✓ Auto-detect transform (Box-Cox or Yeo-Johnson)")
        print(f"   ✓ AvgPerPurchase with auto-detect")
        print(f"   ✓ Income_per_Family_Member with auto-detect")
        print(f"   ✓ Auto-detect PCA for highly correlated pairs")
        print(f"   ✓ {len(existing_features)} clustering features")
        print(f"   ✓ {len(self.pca_applied_pairs)} PCA reduction(s) applied")
        print()
        
        print(f"Final dataset shape: {self.df_engineered.shape}")
        print()
        
        self.log_action("Select final features", f"{len(existing_features)} features selected")
    
    # ============================================================================
    # STEP 10: VALIDATE FEATURES
    # ============================================================================
    
    def validate_features(self):
        """Validate engineered features."""
        self._print_header("STEP 9: VALIDATE FEATURES")
        
        print("DATA QUALITY CHECKS:")
        print(f"{'-' * 100}")
        
        # Missing values
        missing = self.df_engineered.isnull().sum()
        if missing.sum() == 0:
            print(f"   Missing values    : None (PASS)")
        else:
            print(f"   Missing values    : {missing.sum()} (WARNING)")
            for col in missing[missing > 0].index:
                print(f"      {col}: {missing[col]}")
        
        # Duplicates
        dup_count = self.df_engineered.duplicated().sum()
        dup_pct = dup_count / len(self.df_engineered) * 100
        print(f"   Duplicate rows    : {dup_count} ({dup_pct:.2f}%)")
        
        # Variance
        print()
        print(f"   Variance check (all features):")
        low_var = []
        for col in self.df_engineered.columns:
            var = self.df_engineered[col].var()
            print(f"      {col:<35} : {var:>10.3f}")
            if var < 0.01:
                low_var.append(col)
        
        if low_var:
            print(f"\n   WARNING: Low variance: {low_var}")
        else:
            print(f"\n   OK: Adequate variance")
        
        # Infinite values
        inf_count = np.isinf(self.df_engineered.select_dtypes(include=[np.number])).sum().sum()
        print(f"   Infinite values   : {inf_count} {'(PASS)' if inf_count == 0 else '(WARNING)'}")
        
        print(f"{'-' * 100}")
        print()
        
        self.log_action("Validate features", "All checks completed")
    
    # ============================================================================
    # STEP 11: EXPORT DATASET & REPORT
    # ============================================================================
    
    def export_dataset(self):
        """Export engineered dataset to CSV."""
        self._print_header("STEP 10: EXPORT DATASET")
        
        try:
            self.df_engineered.to_csv(self.output_csv, index=False)
            
            file_size = os.path.getsize(self.output_csv) / 1024
            
            print(f"Dataset exported successfully")
            print(f"   Path  : {self.output_csv}")
            print(f"   Shape : {self.df_engineered.shape[0]:,} rows x {self.df_engineered.shape[1]} columns")
            print(f"   Size  : {file_size:.2f} KB")
            print()
            
            self.log_action("Export dataset", self.output_csv)
            return True
            
        except Exception as e:
            print(f"ERROR: Export failed - {e}")
            self.log_action("Export FAILED", str(e))
            return False
    
    def export_report(self):
        """Export engineering report."""
        self._print_header("STEP 11: EXPORT REPORT")
        
        try:
            with open(self.report_file, 'w', encoding='utf-8') as f:
                # Header
                f.write("=" * 100 + "\n")
                f.write("RFM FEATURE ENGINEERING REPORT (v5 - AUTO-DETECT TRANSFORM + AUTO PCA)\n")
                f.write("=" * 100 + "\n\n")
                
                f.write(f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Input     : {self.input_path}\n")
                f.write(f"Output    : {self.output_csv}\n")
                f.write(f"Strategy  : RFM Analysis + Auto-Detect Transform + Auto-Detect PCA\n\n")
                
                f.write("IMPROVEMENTS (v5):\n")
                f.write("-" * 100 + "\n")
                f.write("OK: Auto-detect transform method (Box-Cox for positive data, Yeo-Johnson for data with zeros)\n")
                f.write("OK: AvgPerPurchase with auto-detected transform\n")
                f.write("OK: Income_per_Family_Member with auto-detected transform\n")
                f.write("OK: Auto-detect PCA for highly correlated pairs (|r| >= 0.7)\n")
                f.write(f"OK: Applied PCA to {len(self.pca_applied_pairs)} feature pair(s)\n")
                f.write("OK: Better independence between clustering features\n")
                f.write("OK: Full validation and visualization\n\n")
                
                f.write("TRANSFORM METHOD DECISION LOGIC:\n")
                f.write("-" * 100 + "\n")
                f.write("- Box-Cox: Used when min > 0 (data purely positive)\n")
                f.write("- Yeo-Johnson: Used when min <= 0 (data contains zeros or negatives)\n")
                f.write("- Auto-detection: Applied independently to each skewed feature\n")
                f.write("- Threshold: Transform only if |skewness| > 1.0\n\n")
                
                f.write("PCA REDUCTION LOGIC:\n")
                f.write("-" * 100 + "\n")
                f.write("- Auto-detect high correlation pairs: |r| >= 0.7\n")
                f.write("- Apply PCA(n_components=1) to composite features\n")
                f.write("- Result: n_components=1 captures 90%+ variance from 2 correlated features\n")
                f.write("- Benefit: Eliminates multicollinearity, improves K-Means clustering\n\n")
                
                if self.pca_applied_pairs:
                    f.write("PCA REDUCTIONS APPLIED:\n")
                    f.write("-" * 100 + "\n")
                    for i, pca_info in enumerate(self.pca_applied_pairs, 1):
                        f.write(f"{i}. {pca_info['pc_name']}\n")
                        f.write(f"   Source features: {pca_info['feature1']} x {pca_info['feature2']}\n")
                        f.write(f"   Correlation: {pca_info['correlation']:.3f}\n")
                        f.write(f"   Variance explained: {pca_info['variance_explained']*100:.2f}%\n")
                        f.write(f"   Formula: {pca_info['loadings'][pca_info['feature1']]:.4f}×{pca_info['feature1']} + {pca_info['loadings'][pca_info['feature2']]:.4f}×{pca_info['feature2']}\n\n")
                else:
                    f.write("PCA REDUCTIONS APPLIED: None (all correlations < 0.7)\n\n")
                
                f.write(f"FINAL FEATURES ({self.df_engineered.shape[1]} columns):\n")
                f.write("-" * 100 + "\n\n")
                
                for i, col in enumerate(self.df_engineered.columns, 1):
                    f.write(f"{i}. {col}\n")
                    if f'{col}_info' in self.feature_stats:
                        f.write(f"   [PCA Composite Feature]\n")
                    f.write("\n")
                
                f.write("CORRELATION ANALYSIS:\n")
                f.write("-" * 100 + "\n")
                if 'max_correlation' in self.feature_stats:
                    f.write(f"Max correlation : {self.feature_stats['max_correlation']:.3f}\n")
                    if 'max_corr_pair' in self.feature_stats:
                        f.write(f"Between         : {self.feature_stats['max_corr_pair'][0]} ↔ {self.feature_stats['max_corr_pair'][1]}\n")
                f.write("\n")
                
                f.write("PROCESSING LOG:\n")
                f.write("-" * 100 + "\n")
                for i, log in enumerate(self.processing_log, 1):
                    f.write(f"{i:>3}. [{log['timestamp']}] {log['action']}\n")
                    if log['details']:
                        f.write(f"     {log['details']}\n")
                
                f.write("\n" + "=" * 100 + "\n")
                f.write("END OF REPORT\n")
                f.write("=" * 100 + "\n")
            
            file_size = os.path.getsize(self.report_file) / 1024
            
            print(f"Report exported successfully")
            print(f"   Path  : {self.report_file}")
            print(f"   Size  : {file_size:.2f} KB")
            print()
            
            self.log_action("Export report", self.report_file)
            return True
            
        except Exception as e:
            print(f"ERROR: Report export failed - {e}")
            self.log_action("Export report FAILED", str(e))
            return False
    
    # ============================================================================
    # STEP 12: VISUALIZATION
    # ============================================================================
    
    def plot_histograms(self, graph_dir):
        """Vẽ histogram + KDE cho clustering features."""
        os.makedirs(graph_dir, exist_ok=True)
        
        cols = self.df_engineered.select_dtypes(include=[np.number]).columns
        
        for col in cols:
            if col not in self.df_engineered.columns:
                continue
            
            try:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                sns.histplot(self.df_engineered[col].dropna(), bins=50, kde=True, ax=ax, color='steelblue')
                
                mean_val = self.df_engineered[col].mean()
                median_val = self.df_engineered[col].median()
                
                ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
                ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.3f}')
                
                if col.startswith('PC1_'):
                    title_suffix = "(PCA Composite)"
                else:
                    title_suffix = "(Original)" if f'{col}_transform' not in self.feature_stats else "(Transformed)"
                
                ax.set_title(f'Distribution: {col} {title_suffix}', fontsize=14, fontweight='bold')
                ax.set_xlabel(col, fontsize=12)
                ax.set_ylabel('Frequency', fontsize=12)
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                fname = os.path.join(graph_dir, f"{col}_histogram.png")
                fig.tight_layout()
                fig.savefig(fname, dpi=150, bbox_inches='tight')
                plt.close(fig)
                
                print(f"   Saved: {col}_histogram.png")
                
            except Exception as e:
                print(f"   ERROR: {col} - {e}")
                plt.close('all')
    
    def plot_boxplots(self, graph_dir):
        """Vẽ boxplot cho clustering features."""
        os.makedirs(graph_dir, exist_ok=True)
        
        cols = self.df_engineered.select_dtypes(include=[np.number]).columns
        
        for col in cols:
            if col not in self.df_engineered.columns:
                continue
            
            try:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                sns.boxplot(y=self.df_engineered[col].dropna(), ax=ax, color='lightcoral')
                
                if col.startswith('PC1_'):
                    title_suffix = "(PCA Composite)"
                else:
                    title_suffix = "(Original)" if f'{col}_transform' not in self.feature_stats else "(Transformed)"
                
                ax.set_title(f'Boxplot: {col} {title_suffix}', fontsize=14, fontweight='bold')
                ax.set_ylabel(col, fontsize=12)
                ax.grid(True, alpha=0.3, axis='y')
                
                fname = os.path.join(graph_dir, f"{col}_boxplot.png")
                fig.tight_layout()
                fig.savefig(fname, dpi=150, bbox_inches='tight')
                plt.close(fig)
                
                print(f"   Saved: {col}_boxplot.png")
                
            except Exception as e:
                print(f"   ERROR: {col} - {e}")
                plt.close('all')
    
    def plot_correlation_heatmap(self, graph_dir):
        """Vẽ correlation heatmap (clustering features only)."""
        os.makedirs(graph_dir, exist_ok=True)
        
        existing_cols = self.df_engineered.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(existing_cols) < 2:
            print("   Skipped: Not enough features for heatmap")
            return
        
        try:
            corr = self.df_engineered[existing_cols].corr()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            sns.heatmap(corr, annot=True, fmt=".3f", cmap="RdYlGn_r", vmin=-1, vmax=1, 
                       center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
            
            title = f'Correlation Matrix ({len(existing_cols)} RFM Features)'
            if self.pca_applied_pairs:
                title += f' - After PCA Reduction'
            ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
            
            fname = os.path.join(graph_dir, "correlation_heatmap.png")
            fig.tight_layout()
            fig.savefig(fname, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"   Saved: correlation_heatmap.png")
            
        except Exception as e:
            print(f"   ERROR: Failed heatmap - {e}")
            plt.close('all')
    
    def plot_before_after_transform(self, graph_dir):
        """Vẽ before/after cho transformed features."""
        os.makedirs(graph_dir, exist_ok=True)
        
        features_to_plot = [
            ('AvgPerPurchase', 'AvgPerPurchase_Transformed'),
            ('Income_per_Family_Member', 'Income_per_Family_Member_Transformed')
        ]
        
        for original, transformed in features_to_plot:
            if original not in self.df.columns or transformed not in self.df.columns:
                continue
            
            if f'{original}_transform' not in self.feature_stats:
                continue
            
            if self.feature_stats[f'{original}_transform'].get('method') == 'NO_TRANSFORM':
                continue
            
            try:
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                
                # Before
                sns.histplot(self.df[original].dropna(), bins=40, kde=True, ax=axes[0], color='steelblue')
                axes[0].set_title(f'Before: {original}', fontsize=12, fontweight='bold')
                axes[0].set_xlabel(original, fontsize=11)
                axes[0].set_ylabel('Frequency', fontsize=11)
                axes[0].axvline(self.df[original].mean(), color='red', linestyle='--', 
                               label=f'Mean: {self.df[original].mean():.2f}')
                axes[0].axvline(self.df[original].median(), color='green', linestyle='--', 
                               label=f'Median: {self.df[original].median():.2f}')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
                axes[0].text(0.02, 0.98, f"Skewness: {self.df[original].skew():.3f}", 
                            transform=axes[0].transAxes, fontsize=10, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                # After
                transformed_data = self.df_engineered[transformed]
                sns.histplot(transformed_data.dropna(), bins=40, kde=True, ax=axes[1], color='seagreen')
                
                transform_info = self.feature_stats[f'{original}_transform']
                method = transform_info.get('method', 'Unknown')
                
                axes[1].set_title(f'After: {transformed} ({method})', fontsize=12, fontweight='bold')
                axes[1].set_xlabel(f'{transformed}', fontsize=11)
                axes[1].set_ylabel('Frequency', fontsize=11)
                axes[1].axvline(transformed_data.mean(), color='red', linestyle='--', 
                               label=f'Mean: {transformed_data.mean():.3f}')
                axes[1].axvline(transformed_data.median(), color='green', linestyle='--', 
                               label=f'Median: {transformed_data.median():.3f}')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
                
                axes[1].text(0.02, 0.98, 
                            f"Skewness: {transform_info['new_skew']:.3f}\nLambda: {transform_info['lambda']:.3f}\nMethod: {method}", 
                            transform=axes[1].transAxes, fontsize=10, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
                
                fname = os.path.join(graph_dir, f"{original}_before_after.png")
                fig.tight_layout()
                fig.savefig(fname, dpi=150, bbox_inches='tight')
                plt.close(fig)
                
                print(f"   Saved: {original}_before_after.png")
                
            except Exception as e:
                print(f"   ERROR: Failed before/after for {original} - {e}")
                plt.close('all')
    
    def generate_all_plots(self, graph_dir):
        """Tạo tất cả plots."""
        self._print_header("STEP 12: GENERATE VISUALIZATIONS")
        
        print(f"Creating plots in: {graph_dir}")
        print()
        
        print(f"Creating histograms...")
        self.plot_histograms(graph_dir)
        
        print()
        print(f"Creating boxplots...")
        self.plot_boxplots(graph_dir)
        
        print()
        print(f"Creating correlation heatmap...")
        self.plot_correlation_heatmap(graph_dir)
        
        print()
        print(f"Creating before/after transform plots...")
        self.plot_before_after_transform(graph_dir)
        
        print()
        print(f"All plots saved to: {graph_dir}")
        print()
        
        self.log_action("Generate plots", f"Saved to {graph_dir}")
    
    # ============================================================================
    # MAIN PIPELINE
    # ============================================================================
    
    def run_engineering(self):
        print("\n" + "=" * 100)
        print("RFM FEATURE ENGINEERING PIPELINE (v5 - AUTO-DETECT TRANSFORM + AUTO PCA)".center(100))
        print("=" * 100 + "\n")
        
        start = datetime.now()
        
        if not self.load_data():
            return False
        
        self.create_total_purchases()
        self.create_total_spent()
        self.create_avg_per_purchase()
        self.transform_avg_per_purchase()
        self.create_income_per_family_member()
        self.transform_income_per_family_member()
        self.select_final_features()
        self.validate_features()
        self.export_dataset()
        self.export_report()
        
        elapsed = (datetime.now() - start).total_seconds()
        
        # Generate plots
        graph_dir = r"C:\Project\Machine_Learning\Machine_Learning\graph\Feature Extraction & Engineering_graph\RFM"
        self.generate_all_plots(graph_dir)
        
        print("=" * 100)
        print(f"PIPELINE COMPLETED in {elapsed:.2f}s".center(100))
        print("=" * 100 + "\n")
        
        return True


# ================================================================================
# MAIN EXECUTION
# ================================================================================
def main():
    """Main function - Run all feature engineering classes."""
    input_path = r"C:\Project\Machine_Learning\Machine_Learning\dataset\Customer_Behavior_cleaned.csv"
    output_dir = r"C:\Project\Machine_Learning\Machine_Learning\dataset"
    report_dir = r"C:\Project\Machine_Learning\Machine_Learning\report\Feature Extraction & Engineering_report"
    
    print("\n" + "=" * 100)
    print("MULTI-OBJECTIVE FEATURE ENGINEERING".center(100))
    print("=" * 100 + "\n")
    
    # CLASS 1: DEMOGRAPHIC (IMPROVED)
    print("\n" + "=" * 100)
    print("CLASS 1: DEMOGRAPHIC CLUSTERING - FEATURE ENGINEERING (IMPROVED)".center(100))
    print("=" * 100 + "\n")
    
    demographic_fe = DemographicFeatureEngineering(input_path, output_dir, report_dir)
    demo_success = demographic_fe.run_engineering()
    
    # CLASS 2: PRODUCT+CHANNEL
    print("\n" + "=" * 100)
    print("CLASS 2: PRODUCT+CHANNEL CLUSTERING - FEATURE ENGINEERING".center(100))
    print("=" * 100 + "\n")
    
    productchannel_fe = ProductChannelFeatureEngineering(input_path, output_dir, report_dir)
    pc_success = productchannel_fe.run_engineering()
    
    # CLASS 3: RFM
    print("\n" + "=" * 100)
    print("CLASS 3: RFM CLUSTERING - FEATURE ENGINEERING".center(100))
    print("=" * 100 + "\n")
    
    rfm_fe = RFMFeatureEngineering(input_path, output_dir, report_dir)
    rfm_success = rfm_fe.run_engineering()
    
    # FINAL SUMMARY
    print("\n" + "=" * 100)
    print("FINAL SUMMARY".center(100))
    print("=" * 100)
    print(f"Class 1 (Demographic - Improved) : {'SUCCESS' if demo_success else 'FAILED'}")
    print(f"Class 2 (Product+Channel)        : {'SUCCESS' if pc_success else 'FAILED'}")
    print(f"Class 3 (RFM)                    : {'SUCCESS' if rfm_success else 'FAILED'}")
    print()
    
    total = sum([demo_success, pc_success, rfm_success])
    print(f"TOTAL: {total}/3 classes completed successfully")
    
    if total == 3:
        print("\nALL FEATURE ENGINEERING COMPLETED")
        print("Ready for next stage: Feature Scaling & K-Means Clustering")
    
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()
