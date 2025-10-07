# ======================================================
# 📁 FILE: 2_outlier_detection.py
# 📌 PURPOSE:
#   - Detect outliers using IQR method
#   - Remove or cap outliers for numeric features
# ======================================================

import pandas as pd
import numpy as np

def run(big_df):
    print("🔹 Step 2: Detecting and handling outliers...")

    # Select numeric columns
    numeric_cols = big_df.select_dtypes(include=[np.number]).columns

    # Create a copy to avoid changing the original data
    cleaned_df = big_df.copy()

    outlier_report = {}

    for col in numeric_cols:
        Q1 = cleaned_df[col].quantile(0.25)
        Q3 = cleaned_df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        # Count true outliers
        outlier_count = ((cleaned_df[col] < lower) | (cleaned_df[col] > upper)).sum()
        outlier_report[col] = int(outlier_count)

        # Cap outliers (winsorization)
        cleaned_df[col] = np.where(cleaned_df[col] < lower, lower,
                                   np.where(cleaned_df[col] > upper, upper, cleaned_df[col]))

    print("\n✅ Outlier detection complete.")
    print("\nOutlier counts per column:")
    for col, count in outlier_report.items():
        print(f"{col} — {count} outliers handled")

    # Save the cleaned file
    cleaned_df.to_csv("data/big_df_no_outliers.csv", index=False)
    print("\n💾 File saved as 'data/big_df_no_outliers.csv'")

    return cleaned_df

if __name__ == "__main__":
    df = pd.read_csv("data/big_df_cleaned.csv")
    run(df)
