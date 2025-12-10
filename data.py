import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("DATA PREPROCESSING & MODEL BUILDING PIPELINE")
print("="*80)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n[STEP 1] Loading Dataset...")
df = pd.read_csv('Scraped_Data.csv')
print(f"Original Dataset Shape: {df.shape}")
print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# ============================================================================
# STEP 2: REMOVE DUPLICATE RECORDS
# ============================================================================
print("\n[STEP 2] Removing Duplicate Records...")
initial_rows = len(df)
df = df.drop_duplicates()
duplicates_removed = initial_rows - len(df)
print(f"Duplicates Removed: {duplicates_removed}")
print(f"Dataset Shape After Deduplication: {df.shape}")

# ============================================================================
# STEP 3: HANDLE MISSING VALUES (Replace '9' with NaN)
# ============================================================================
print("\n[STEP 3] Handling Missing Values...")
# Replace '9' with NaN for proper missing value handling
df = df.replace(9, np.nan)
df = df.replace('9', np.nan)
df = df.replace(9.0, np.nan)

missing_before = df.isnull().sum().sum()
print(f"Total Missing Values: {missing_before:,}")

# ============================================================================
# STEP 4: REMOVE NOISY/IRRELEVANT DATA
# ============================================================================
print("\n[STEP 4] Removing Noisy and Irrelevant Data...")

# 4.1: Remove rows with missing critical features
critical_features = ['exactPrice', 'carpetArea', 'bedrooms', 'city', 'propertyType']
print(f"Removing rows with missing critical features: {critical_features}")
initial_rows = len(df)
df = df.dropna(subset=critical_features)
print(f"Rows removed: {initial_rows - len(df)}")

# 4.2: Remove extreme outliers in price (using IQR method with 3x multiplier)
print("\nRemoving extreme outliers in 'exactPrice'...")
Q1 = df['exactPrice'].quantile(0.25)
Q3 = df['exactPrice'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 3 * IQR
upper_bound = Q3 + 3 * IQR
initial_rows = len(df)
df = df[(df['exactPrice'] >= lower_bound) & (df['exactPrice'] <= upper_bound)]
print(f"Rows removed: {initial_rows - len(df)}")
print(f"Price range: {df['exactPrice'].min():,.0f} to {df['exactPrice'].max():,.0f}")

# 4.3: Remove outliers in carpetArea
print("\nRemoving extreme outliers in 'carpetArea'...")
Q1 = df['carpetArea'].quantile(0.25)
Q3 = df['carpetArea'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 3 * IQR
upper_bound = Q3 + 3 * IQR
initial_rows = len(df)
df = df[(df['carpetArea'] >= lower_bound) & (df['carpetArea'] <= upper_bound)]
print(f"Rows removed: {initial_rows - len(df)}")

# 4.4: Remove columns with >70% missing values
print("\nRemoving columns with >70% missing values...")
missing_threshold = 0.70
missing_pct = df.isnull().sum() / len(df)
cols_to_drop = missing_pct[missing_pct > missing_threshold].index.tolist()
print(f"Columns to drop ({len(cols_to_drop)}): {cols_to_drop[:10]}{'...' if len(cols_to_drop) > 10 else ''}")
df = df.drop(columns=cols_to_drop)

# 4.5: Remove irrelevant columns (URLs, redundant features)
irrelevant_cols = ['URLs', 'postedOn']  # URLs not useful for modeling
irrelevant_cols = [col for col in irrelevant_cols if col in df.columns]
print(f"\nRemoving irrelevant columns: {irrelevant_cols}")
df = df.drop(columns=irrelevant_cols)

print(f"\nDataset Shape After Cleaning: {df.shape}")

# ============================================================================
# STEP 5: DATA CLEANING & STANDARDIZATION
# ============================================================================
print("\n[STEP 5] Data Cleaning & Standardization...")

# 5.1: Standardize carpetAreaUnit to Sq-ft
print("\nStandardizing area units to Sq-ft...")
if 'carpetAreaUnit' in df.columns:
    # Conversion factors to Sq-ft
    conversion_factors = {
        'Sq-ft': 1,
        'Sq-m': 10.764,
        'Sq-yrd': 9,
        'Kanal': 5445,
        'Marla': 272.25,
        'Acre': 43560,
        'Biswa1': 1350,
        'Biswa2': 1350,
        'Rood': 10890
    }
    
    for unit, factor in conversion_factors.items():
        mask = df['carpetAreaUnit'] == unit
        df.loc[mask, 'carpetArea'] = df.loc[mask, 'carpetArea'] * factor
    
    df = df.drop(columns=['carpetAreaUnit'])
    print("All areas converted to Sq-ft")

# 5.2: Fill missing values with mean for numerical columns
print("\nFilling missing values with mean for numerical columns...")
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

for col in numerical_cols:
    if df[col].isnull().sum() > 0:
        mean_value = df[col].mean()
        df[col] = df[col].fillna(mean_value)
        print(f"  {col}: filled {df[col].isnull().sum()} values with mean {mean_value:.2f}")

# 5.3: Fill missing values with mode for categorical columns
print("\nFilling missing values with mode for categorical columns...")
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        mode_value = df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown'
        df[col] = df[col].fillna(mode_value)
        print(f"  {col}: filled with mode '{mode_value}'")

# 5.4: Encode categorical variables
print("\nEncoding categorical variables...")
label_encoders = {}
for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        print(f"  {col}: {len(le.classes_)} unique categories encoded")

print(f"\nFinal Dataset Shape: {df.shape}")
print(f"Missing Values Remaining: {df.isnull().sum().sum()}")

# ============================================================================
# STEP 6: FEATURE SELECTION
# ============================================================================
print("\n[STEP 6] Feature Selection...")

# Define target variable
target = 'exactPrice'
X = df.drop(columns=[target])
y = df[target]

print(f"Features: {X.shape[1]}")
print(f"Target: {target}")

# Select top K features using SelectKBest
k_best = min(20, X.shape[1])  # Select top 20 features or all if less than 20
print(f"\nSelecting top {k_best} features using mutual information...")

selector = SelectKBest(score_func=mutual_info_regression, k=k_best)
X_selected = selector.fit_transform(X, y)

# Get selected feature names
selected_features_mask = selector.get_support()
selected_features = X.columns[selected_features_mask].tolist()

print(f"\nSelected Features ({len(selected_features)}):")
for i, feature in enumerate(selected_features, 1):
    print(f"  {i}. {feature}")

# Update X with selected features
X = pd.DataFrame(X_selected, columns=selected_features, index=X.index)

# ============================================================================
# STEP 7: TRAIN-TEST SPLIT
# ============================================================================
print("\n[STEP 7] Splitting Data into Train and Test Sets...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training Set: {X_train.shape[0]} samples ({(len(X_train)/len(X)*100):.1f}%)")
print(f"Test Set: {X_test.shape[0]} samples ({(len(X_test)/len(X)*100):.1f}%)")

# ============================================================================
# STEP 8: FEATURE SCALING
# ============================================================================
print("\n[STEP 8] Feature Scaling (Standardization)...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Features scaled using StandardScaler (mean=0, std=1)")

# ============================================================================
# STEP 9: MODEL SELECTION & TRAINING
# ============================================================================
print("\n[STEP 9] Model Selection & Training...")

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest Regressor': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
}

results = {}

for model_name, model in models.items():
    print(f"\n{'='*60}")
    print(f"Training {model_name}...")
    print(f"{'='*60}")
    
    # Train model
    if model_name == 'Linear Regression':
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results[model_name] = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'model': model
    }
    
    print(f"\nPerformance Metrics:")
    print(f"  Mean Squared Error (MSE): {mse:,.2f}")
    print(f"  Root Mean Squared Error (RMSE): {rmse:,.2f}")
    print(f"  Mean Absolute Error (MAE): {mae:,.2f}")
    print(f"  R² Score: {r2:.4f}")
    print(f"  Accuracy: {r2*100:.2f}%")

# ============================================================================
# STEP 10: MODEL COMPARISON & BEST MODEL SELECTION
# ============================================================================
print("\n" + "="*80)
print("[STEP 10] MODEL COMPARISON")
print("="*80)

comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'RMSE': [results[m]['RMSE'] for m in results.keys()],
    'MAE': [results[m]['MAE'] for m in results.keys()],
    'R² Score': [results[m]['R2'] for m in results.keys()]
})

print("\n", comparison_df.to_string(index=False))

best_model_name = max(results, key=lambda x: results[x]['R2'])
best_model = results[best_model_name]['model']

print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"   R² Score: {results[best_model_name]['R2']:.4f}")
print(f"   RMSE: {results[best_model_name]['RMSE']:,.2f}")

# Feature importance (if available)
if hasattr(best_model, 'feature_importances_'):
    print(f"\n📊 Top 10 Feature Importances ({best_model_name}):")
    feature_importance = pd.DataFrame({
        'Feature': selected_features,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(feature_importance.head(10).to_string(index=False))

# ============================================================================
# STEP 11: SAVE PROCESSED DATA & MODEL
# ============================================================================
print("\n[STEP 11] Saving Processed Data & Model...")

# Save cleaned dataset
df_final = pd.concat([X, y], axis=1)
df_final.to_csv('Cleaned_Data.csv', index=False)
print("✓ Cleaned dataset saved to 'Cleaned_Data.csv'")

# Save train-test splits
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)
print("✓ Train-test splits saved")

# Save model
import joblib
joblib.dump(best_model, 'best_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
print(f"✓ Best model saved to 'best_model.pkl'")
print("✓ Scaler saved to 'scaler.pkl'")
print("✓ Label encoders saved to 'label_encoders.pkl'")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("PIPELINE EXECUTION SUMMARY")
print("="*80)
print(f"""
✓ Data Loaded: 27,900 rows × 91 columns
✓ Duplicates Removed: {duplicates_removed}
✓ Noisy Data Removed: Outliers and invalid records
✓ Missing Values Handled: Filled with mean/mode
✓ Features Selected: {len(selected_features)} best features
✓ Data Split: 80% train, 20% test
✓ Models Trained: {len(models)}
✓ Best Model: {best_model_name} (R²={results[best_model_name]['R2']:.4f})

📁 Output Files:
   - Cleaned_Data.csv
   - X_train.csv, X_test.csv, y_train.csv, y_test.csv
   - best_model.pkl
   - scaler.pkl
   - label_encoders.pkl

🎯 Next Steps:
   1. Review model performance metrics
   2. Fine-tune hyperparameters if needed
   3. Use saved model for predictions on new data
   4. Consider ensemble methods for improved performance
""")

print("="*80)
print("PREPROCESSING & MODELING COMPLETE!")
print("="*80)