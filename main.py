from scripts import (
    data_cleaning,
    outlier_detection,
    data_merging,
    feature_engineering,
    model_training,
    advanced_models,
    reporting_and_saving
)

if __name__ == "__main__":
    print("🚀 Starting Irrigation Optimization Pipeline...\n")

    data_cleaning.run()
    outlier_detection.run()
    data_merging.run()
    feature_engineering.run()
    model_training.run()
    advanced_models.run()
    reporting_and_saving.run()

    print("\n✅ Pipeline Completed Successfully!")
