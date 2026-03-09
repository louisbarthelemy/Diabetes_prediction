import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import MinMaxScaler

def load_and_clean_data(filepath):
    """
    Loads the diabetes dataset and performs initial cleaning.
    
    RATIONALE: The dataset contains 253,680 observations. All variables 
    are encoded as float64, but many represent categorical data.
    """
    df = pd.read_csv(filepath)
    # Mapping logic from Descriptive notebook
    age_mapping = {1: "18-24", 2: "25-29", 3: "30-34", 4: "35-39", 5: "40-44",
                   6: "45-49", 7: "50-54", 8: "55-59", 9: "60-64", 10: "65-69",
                   11: "70-74", 12: "75-79", 13: "80+"}
    # We maintain the raw data for analysis but provide a cleaning hook
    return df

def prepare_binary_classification(df):
    """
    Prepares data for predictive modeling.
    
    CHOICE: Prediabetes cases (~1.3%) are removed to simplify to a binary 
    classification problem (0=No Diabetes, 1=Diabetes).
    """
    df_binary = df[df["Diabetes_012"].isin([0, 2])].copy()
    df_binary.loc[df_binary["Diabetes_012"] == 2, "Diabetes_012"] = 1
    
    # Balancing via Under-sampling
    X = df_binary.drop("Diabetes_012", axis=1)
    y = df_binary["Diabetes_012"]
    
    rus = RandomUnderSampler(random_state=1)
    X_resampled, y_resampled = rus.fit_resample(X, y)
    
    return X_resampled, y_resampled