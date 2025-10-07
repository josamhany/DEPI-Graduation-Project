# ======================================================
# 📁 FILE: 3_feature_engineering.py
# 📌 PURPOSE:
#   - Combine small and big datasets
#   - Create new features
#   - Encode categorical variables
# ======================================================

import pandas as pd
from sklearn.preprocessing import LabelEncoder

def run(big_df, small_df):
    print("🔹 Step 3: Feature engineering and merging datasets...")

    # Merge both datasets based on common columns (like N, P, K, temperature, humidity, ph)
    merge_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph']
    merged_df = pd.merge(big_df, small_df, on=merge_cols, how='inner')

    print(f"\n✅ Merged dataset shape: {merged_df.shape}")

    # Encode target variable (status / label)
    if 'label' in merged_df.columns:
        le = LabelEncoder()
        merged_df['label'] = le.fit_transform(merged_df['label'])
        print("\n✅ Encoded target variable 'label'")

    # Example: create new ratio features
    merged_df['NPK_sum'] = merged_df['N'] + merged_df['P'] + merged_df['K']
    merged_df['temp_humidity_ratio'] = merged_df['temperature'] / (merged_df['humidity'] + 1)

    print("\n✅ Feature engineering complete.")

    # Save engineered dataset
    merged_df.to_csv("data/merged_df_ready.csv", index=False)
    print("\n💾 Saved as 'data/merged_df_ready.csv'")

    return merged_df

if __name__ == "__main__":
    big_df = pd.read_csv("data/big_df_no_outliers.csv")
    small_df = pd.read_csv("data/Crop_recommendation.csv")
    run(big_df, small_df)
