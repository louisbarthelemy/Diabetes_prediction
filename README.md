# Diabetes Risk Prediction with Responsible Data Analytics

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A full data science pipeline exploring diabetes risk factors using public health data.  
This project combines **machine learning, exploratory analysis, clustering, explainability, and fairness evaluation** to study how socio-economic, behavioral, and health indicators relate to diabetes.

The project follows a **responsible analytics workflow** emphasizing **bias detection, transparency, and fairness** in healthcare analytics.

---

# Project Goals

- Identify key indicators associated with diabetes risk
- Build a predictive machine learning model
- Understand feature relationships through diagnostic analytics
- Create interpretable predictions using SHAP
- Evaluate fairness and bias across demographic groups

---

# Dataset

**Source:** Behavioral Risk Factor Surveillance System (BRFSS)

- ~253,000 individuals
- 22 health, behavioral, and socio-economic features

Examples of variables:

| Category | Variables |
|---|---|
| Health | BMI, HighBP, HighChol, GenHlth |
| Lifestyle | Smoking, Physical Activity, Alcohol |
| Socio-economic | Income, Education |
| Demographics | Age, Sex |

---

# Analytics Pipeline

The project is organized into **six analysis stages**.

## 1. Bias Analysis
Explores ethical considerations and dataset limitations:

- Historical bias
- Representation bias
- Measurement bias
- Privacy considerations

---

## 2. Descriptive Analysis

Exploratory data analysis of the dataset:

- Distribution of health indicators
- Population demographics
- Diabetes prevalence
- Lifestyle behavior patterns

---

## 3. Diagnostic Analysis

This stage investigates **relationships between variables and diabetes**.

### Correlation Analysis

Correlation matrices were created for:

- **All features**
- **Socio-economic features**
- **Health indicators**
- **Behavioral variables**

Key insights:

- Strong positive correlation with diabetes:
  - General Health
  - High Blood Pressure
  - BMI
  - Difficulty Walking
  - High Cholesterol
  - Age

- Negative correlations:
  - Income
  - Education
  - Physical Activity

Behavioral variables such as fruit consumption or alcohol intake show **weak correlations with diabetes**.

---

### Clustering Analysis

Unsupervised learning was applied using **K-Means clustering**.

Two clustering experiments were conducted:

#### Clustering on All Features
- Target variable removed
- Data standardized
- K-Means with **2 clusters**

Results:
- Non-diabetics mainly grouped in one cluster
- Diabetics appear in both clusters
- The dataset does **not naturally separate diabetic individuals**

---

#### Clustering on Selected Biological Features

Selected features:

- BMI
- High Blood Pressure
- Age
- Physical Activity
- Fruit Consumption
- Income

Findings:

- Diabetics concentrate more in one cluster
- However, clusters still contain mixed classes
- Even using key health features, diabetes **does not form clearly separable clusters**

Alternative clustering with **DBSCAN** was attempted but produced **>100 clusters**, offering no meaningful segmentation.

---

## 4. Predictive Analysis

Binary classification problem:

- Predict diabetic vs non-diabetic individuals

Models tested:

- Logistic Regression
- Decision Trees
- Random Forest

Final model:

**Random Forest**

Performance:

| Metric | Score |
|---|---|
| Accuracy | ~73% |
| F1 Score | ~73% |

Key predictive features:

- General Health
- BMI
- Age
- HighBP × HighChol interaction

---

## 5. Prescriptive Analytics

Predictions were transformed into **actionable healthcare insights**.

Using **SHAP explainability**:

- Feature importance interpretation
- Local prediction explanations
- Risk-based decision support rules

Example rule:

```
IF (Diabetes Risk = High)
AND (GenHlth ≥ 4)
AND (Age > 60)
AND (HighBP & HighChol = True)

THEN
Recommend: Schedule an immediate diagnostic check-up
Priority level: High
```

---

## 6. Fairness Analysis

Fairness evaluation performed using **Aequitas**.

Metrics analyzed:

- Precision disparity
- True Positive Rate disparity

Key findings:

- Higher recall for individuals with poor health indicators
- Lower sensitivity for younger and healthier populations
- Model bias partly due to missing genetic features

---

# Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Seaborn / Matplotlib
- SHAP
- Aequitas
- Jupyter Notebook

---

## Repository Structure

```
.
├── bias_analysis.ipynb
├── descriptive_analysis.ipynb
├── diagnostic_analysis.ipynb
├── predictive_analysis.ipynb
├── prescriptive_analysis.ipynb
├── fairness_analysis.ipynb
└── README.md
```

