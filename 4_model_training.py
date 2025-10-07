# ======================================================
# 📁 FILE: 4_model_training.py
# 📌 PURPOSE:
#   - Train CatBoost Classifier using Optuna hyperparameter tuning
#   - Evaluate performance and save best model
# ======================================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from catboost import CatBoostClassifier
import optuna
import joblib

def objective(trial, X_train, y_train, X_test, y_test):
    params = {
        "iterations": trial.suggest_int("iterations", 200, 800),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "loss_function": "Logloss",
        "verbose": 0
    }

    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return accuracy_score(y_test, preds)

def run(merged_df):
    print("🔹 Step 4: Model training with CatBoost + Optuna...")

    # Split data
    X = merged_df.drop(columns=["label"])
    y = merged_df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Hyperparameter tuning
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test), n_trials=15)

    best_params = study.best_params
    print("\n✅ Best Parameters Found:", best_params)

    # Train final model
    final_model = CatBoostClassifier(**best_params, loss_function="Logloss", verbose=0)
    final_model.fit(X_train, y_train)

    preds = final_model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"\n🎯 Final Accuracy: {acc:.5f}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds))

    # Save model
    joblib.dump(final_model, "models/catboost_best_model.pkl")
    print("\n💾 Model saved to 'models/catboost_best_model.pkl'")

    return final_model

if __name__ == "__main__":
    merged_df = pd.read_csv("data/merged_df_ready.csv")
    run(merged_df)
