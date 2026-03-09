def simulate_bmi_shift(X_test, model, shift_value=0.1):
    """
    Simulates a distribution shift in BMI to test model robustness.
    
    CONCLUSION: Model performance (F1-score) declines when BMI distributions 
    shift, highlighting the need for continuous monitoring.
    """
    X_shifted = X_test.copy()
    X_shifted['BMI'] += shift_value
    preds = model.predict(X_shifted)
    return preds