import logging
import pandas as pd
import matplotlib.pyplot as plt
from aequitas.group import Group
from aequitas.bias import Bias
from aequitas.plotting import Plot


def run_fairness_audit(df_with_predictions):

    df = df_with_predictions.copy()

    # Required columns
    df["score"] = df["prediction"]
    df["label_value"] = df["true_label"]

    df["BPxChol"] = df["HighBP"] * df["HighChol"]

    bins = [0, 18.5, 25, 30, 35, 40, 100]
    labels = ["Underweight", "Healthy", "Overweight", "Obesity Class I",
          "Obesity Class II", "Severe Obesity"]
    df['BMI_categorical'] = pd.cut(df['BMI'], bins=bins, labels=labels, include_lowest=True)

    df['Age'] = pd.cut(
        df['Age'],
        bins=[0, 2, 4, 6, 8, 10, 13],
        labels=['0-2', '2-4', '4-6', '6-8', '8-10', '10-13']
    )

    df['Income'] = pd.cut(
        df['Income'],
        bins=[1, 3, 5, 6, 8],
        labels=['1-2', '3-4', '5-6', '7-8'],
        include_lowest=True
    )

    # Feature selection
    df = df[
        [
            'score',
            'label_value',
            'Age',
            'BMI_categorical',
            'BPxChol',
            'GenHlth',
            'Income'
        ]
    ]

    # Convert categorical columns to string
    non_string_cols = df.columns[(df.dtypes != object) & (df.dtypes != str)]
    df[non_string_cols] = df[non_string_cols].astype(str)

    df['score'] = df['score'].astype(float)
    df['label_value'] = df['label_value'].astype(float)

    # Run fairness analysis
    g = Group()
    xtab, _ = g.get_crosstabs(df)

    b = Bias()
    bias_df = b.get_disparity_predefined_groups(
        xtab,
        original_df=df,
        ref_groups_dict={
            'Age': '6-8',
            'BMI_categorical': 'Healthy',
            'BPxChol': '0.0',
            'GenHlth': '2.0',
            'Income': '3-4'
        },
        alpha=0.05,
        check_significance=True,
        mask_significance=True
    )

    logging.info("Fairness Audit Complete.")

    return bias_df