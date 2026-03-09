from sklearn.ensemble import RandomForestClassifier
import shap

def train_risk_model(X_train, y_train):
    """
    Trains the final Random Forest model.
    
    RATIONALE: Random Forest was selected over Logistic Regression because it 
    captures non-linear relationships and interaction effects while 
    maintaining interpretability.
    """
    model = RandomForestClassifier(
        criterion="gini",
        max_depth=6,
        max_leaf_nodes=16,
        n_estimators=100,
        random_state=1
    )
    model.fit(X_train, y_train)
    return model

def explain_model(model, X_test):
    """
    Uses SHAP to provide local explainability.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    return shap_values