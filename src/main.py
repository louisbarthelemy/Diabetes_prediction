from src import data_processing, analysis, modeling, ethics_audit

def main():
    print("Starting diabetes risk modeling pipeline...")

    # 1. Load & Clean Data
    print("Loading and cleaning data")
    raw_data = data_processing.load_and_clean_data(
        "data/diabetes_012_health_indicators_BRFSS2015.csv")

    # 2. Diagnostic Analysis
    print("Running diagnostic analysis")
    corrs, clusters = analysis.perform_diagnostic_analysis(raw_data)

    # 3. Prepare Data for Modeling
    print("Preparing binary classification dataset")
    X_res, y_res = data_processing.prepare_binary_classification(raw_data)

    # 4. Train Model
    print("Training risk prediction model")
    model = modeling.train_risk_model(X_res, y_res)

    # 5. Generate Predictions
    print("Generating predictions")
    preds = model.predict(X_res)

    # Add predictions to dataset
    X_res_with_preds = X_res.copy()
    X_res_with_preds["prediction"] = preds
    X_res_with_preds["true_label"] = y_res

    # 6. Ethics & Fairness Audit
    print("Running fairness audit")
    audit_results = ethics_audit.run_fairness_audit(X_res_with_preds)

    print("Pipeline executed successfully")
    print("Ethics-aware model is ready")

    return model, audit_results


if __name__ == "__main__":
    model, audit_results = main()