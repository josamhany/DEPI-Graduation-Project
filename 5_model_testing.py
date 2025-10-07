# ======================================================
# 📁 FILE: 5_model_testing.py
# 📌 PURPOSE:
#   - Load saved model
#   - Evaluate on test data
#   - Save predictions to CSV
# ======================================================

import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report

def run():
    print("🔹 Step 5: Testing saved model...")

    # Load model
    model = joblib.load("models/catboost_best_model.pkl")

    # Load test data (same as before)
    merged_df = pd.read_csv("data/merged_df_ready.csv")
    X = merged_df.drop(columns=["label"])
    y = merged_df["label"]

    # Predict
    preds = model.predict(X)

    acc = accuracy_score(y, preds)
    print(f"\n🎯 Model Test Accuracy: {acc:.5f}")

    print("\nClassification Report:")
    print(classification_report(y, preds))

    # Save predictions
    output = merged_df.copy()
    output["predicted_label"] = preds
    output.to_csv("data/test_predictions.csv", index=False)

    print("\n💾 Predictions saved as 'data/test_predictions.csv'")

if __name__ == "__main__":
    run()
