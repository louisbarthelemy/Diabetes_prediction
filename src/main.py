from src import data_processing, analysis, modeling, ethics_audit

# 1. Load & Clean
raw_data = data_processing.load_and_clean_data("data/diabetes_012_health_indicators_BRFSS2015.csv")

# 2. Diagnostic Analysis
corrs, clusters = analysis.perform_diagnostic_analysis(raw_data)

# 3. Model Training
X_res, y_res = data_processing.prepare_binary_classification(raw_data)
model = modeling.train_risk_model(X_res, y_res)

# 4. Ethics & Fairness Audit
# (Assuming predictions are added to a dataframe)
audit_results = ethics_audit.run_fairness_audit(X_res_with_preds)

print("Pipeline executed successfully. Ethics-aware model is ready.")