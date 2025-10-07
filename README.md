# 🌱 Smart Irrigation System - ML Model

This project develops a **machine learning-based Smart Irrigation System** that predicts whether irrigation should be **ON or OFF** based on environmental, soil, and nutrient data.  
The pipeline covers **data cleaning, outlier detection, feature engineering, model training, and testing** using the **CatBoost classifier optimized with Optuna**.

---

## 📚 Project Overview

Modern agriculture requires **efficient water management** to reduce waste and improve crop yield.  
This model analyzes environmental parameters like soil moisture, temperature, humidity, rainfall, and nutrient levels (N, P, K) to determine irrigation needs automatically.

---

## 🧩 Workflow Overview

| Step | File | Description |
|------|------|--------------|
| 1️⃣ | `1_data_cleaning.py` | Loads raw datasets and imputes missing values using **KNN Imputer** |
| 2️⃣ | `2_outlier_detection.py` | Detects and caps outliers using the **IQR method** |
| 3️⃣ | `3_feature_engineering.py` | Merges datasets, encodes labels, and creates new derived features the data called 'Final_irregation_optimization_data' and in code called 'merged_df'|
| 4️⃣ | `4_model_training.py` | Tunes **CatBoost** hyperparameters using **Optuna** and trains the model |
| 5️⃣ | `5_model_testing.py` | Loads the saved model, evaluates it, and saves predictions |

---

## 🧠 Dataset Description

### 1. `Crop_recommendation.csv`
Contains data about soil nutrients and environmental conditions for crop recommendations.  
**Columns:**

### 2. `TARP.csv`
Contains real-time sensor readings from the irrigation system.  
**Columns include:**
Smart_Irrigation_Model/
│
├── data/
│   ├── Crop_recommendation.csv
│   ├── TARP.csv
│
├── models/
│
├── 1_data_cleaning.py
├── 2_outlier_detection.py
├── 3_feature_engineering.py
├── 4_model_training.py
├── 5_model_testing.py
└── README.md
# Step 1: Data cleaning
python 1_data_cleaning.py

# Step 2: Outlier detection and handling
python 2_outlier_detection.py

# Step 3: Feature engineering
python 3_feature_engineering.py

# Step 4: Train and optimize CatBoost model
python 4_model_training.py

# Step 5: Evaluate and test saved model
python 5_model_testing.py
📈 Model Information

Algorithm: CatBoost Classifier

Optimization: Optuna hyperparameter tuning

Evaluation Metric: Accuracy + Classification Report

Final Model Saved: models/catboost_best_model.pkl

📊 Output Files
File	Description
data/big_df_cleaned.csv	After missing value imputation
data/big_df_no_outliers.csv	After outlier removal
data/merged_df_ready.csv	After feature engineering
models/catboost_best_model.pkl	Trained model
data/test_predictions.csv	Final model predictions
🧩 Example Prediction Columns
Feature	Example Value
soil_moisture	33.5
air_temperature_(c)	27.4
humidity	62
n	45
p	30
k	50
ph	6.4
Predicted Status	1 (Irrigation ON)

🧑‍💻 Authors

Josam Hany & Badr Elsafy 
Computer Science Student | Data Scientist & ML Engineer
📍 Alexandria, Egypt


🏁 License

This project is released under the MIT License — feel free to use and modify it with credit.
