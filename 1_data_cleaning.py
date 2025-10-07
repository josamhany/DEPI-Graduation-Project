# ======================================================
# 📁 FILE: 1_data_cleaning.py
# 📌 PURPOSE:
#   - Load the datasets (Crop_recommendation & TARP)
#   - Detect missing values
#   - Handle missing data using KNN Imputer
#   - Save the cleaned dataset
# ======================================================

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def run():
    print("🔹 Step 1: Loading datasets...")

    # Load the small and big datasets
    small_df = pd.read_csv('data/Crop_recommendation.csv')
    big_df = pd.read_csv('data/TARP.csv')

    print("\n✅ Data loaded successfully!")
    print("\nSmall Dataset Info:")
    print(small_df.info())
    print("\nBig Dataset Info:")
    print(big_df.info())

    # --------------------------------------------------
    # 🧩 Detect Missing Values
    # --------------------------------------------------
    missing_cols = big_df.columns[big_df.isnull().any()]
    print("\nColumns with missing values:", list(missing_cols))

    # Sampling a subset to test KNN imputer
    sample = big_df[missing_cols].dropna()
    if len(sample) > 1000:
        sample = sample.sample(1000, random_state=42)
    print("Sample shape:", sample.shape)

    # Randomly mask 10% of the sample
    masked_sample = sample.copy()
    mask = np.random.rand(*sample.shape) < 0.1
    masked_sample[mask] = np.nan

    print(f"Number of hidden (masked) values: {mask.sum()}")

    # --------------------------------------------------
    # 🔧 Test different K values for KNN
    # --------------------------------------------------
    errors = {}
    for k in [2, 3, 5, 7, 9, 11]:
        imputer = KNNImputer(n_neighbors=k)
        imputed = imputer.fit_transform(masked_sample)
        mse = mean_squared_error(sample.values[~mask], imputed[~mask])
        errors[k] = mse
        print(f"K={k} --> MSE={mse:.5f}")

    # Find the best K
    best_k = min(errors, key=errors.get)
    print(f"\n✅ Best K value for imputation: {best_k}")

    # Plot MSE vs K
    plt.figure(figsize=(8,5))
    plt.plot(list(errors.keys()), list(errors.values()), marker='o')
    plt.xlabel('K (number of neighbors)')
    plt.ylabel('Mean Squared Error')
    plt.title('Choosing the Best K for KNN Imputer')
    plt.grid(True)
    plt.show()

    # --------------------------------------------------
    # 🧠 Apply KNN Imputation to Numeric Columns
    # --------------------------------------------------
    numeric_columns = big_df.select_dtypes(include=[np.number]).columns
    imputer = KNNImputer(n_neighbors=best_k)
    imputed_array = imputer.fit_transform(big_df[numeric_columns])
    imputed_df = pd.DataFrame(imputed_array, columns=numeric_columns)

    # Combine back with non-numeric columns
    non_numeric_columns = big_df.select_dtypes(exclude=[np.number]).columns
    big_df = pd.concat([imputed_df, big_df[non_numeric_columns].reset_index(drop=True)], axis=1)

    print("\n✅ Missing values handled successfully!")
    print(big_df.info())

    # --------------------------------------------------
    # 💾 Save temporary cleaned dataset
    # --------------------------------------------------
    big_df.to_csv("data/big_df_cleaned.csv", index=False)
    print("\n💾 Cleaned dataset saved as 'data/big_df_cleaned.csv'")

    return big_df, small_df

# ------------------------------------------------------
# 🔰 Run this file independently (for testing)
# ------------------------------------------------------
if __name__ == "__main__":
    run()
