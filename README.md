# Diabetes-Project-ML
# Diabetes Risk Prediction Analysis

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-latest-orange.svg)
![Pandas](https://img.shields.io/badge/pandas-latest-red.svg)

##  Project Overview
This project focuses on building a binary classification system to predict the risk of diabetes based on clinical patient data. The study implements a comprehensive data science workflow—from rigorous data cleaning to hyperparameter tuning—with a core focus on overcoming severe **class imbalance** to maximize the model's clinical sensitivity (**Recall**).

## Dataset
The dataset utilized for this study is sourced from Kaggle: [Diabetes Prediction Dataset](https://www.kaggle.com/datasets/mrsimple07/diabetes-prediction). 

It contains 1,000 clinical records with the following diagnostic features:
- **Pregnancies**: Number of times pregnant.
- **Glucose**: Plasma glucose concentration.
- **BloodPressure**: Diastolic blood pressure (mm Hg).
- **SkinThickness**: Triceps skin fold thickness (mm).
- **Insulin**: 2-Hour serum insulin (mu U/ml).
- **BMI**: Body mass index (weight in kg/(height in m)²).
- **DiabetesPedigreeFunction**: Diabetes pedigree function (genetic history).
- **Age**: Age in years.
- **Diagnosis (Target)**: `0` = Non-Diabetic, `1` = Diabetic.

---

## Steps:

### 1. Data Cleaning
To ensure high data quality and avoid misleading model performance, a strict cleaning pipeline was executed:
* **Negative Values Removal**: Filtered out erroneous negative numbers caused by data entry issues.
* **Deduplication**: Identified and dropped duplicate records.
* **Zero-Value Treatment**: Invalid hidden missing values (e.g., $0$ values in biological features like `BMI`, `BloodPressure`, or `Glucose` where zero is physiologically impossible) were properly handled.

### 2. Exploratory Data Analysis (EDA)
* **Boxplot Analysis**: Utilized to visualize the distribution of clinical indicators and flag significant upper outliers in features like `BMI`, `Insulin`, and `Glucose`.
* **Correlation Matrix**: Revealed that features possess very low linear dependencies individually with `Diagnosis` (maximum correlation of ~0.06).

### 3. Model Preparation
* **Data Splitting**: Partitioned the clean dataset into standard training and testing subsets to validate generalizability.
* **Feature Scaling (Normalization)**: Applied `StandardScaler` to handle vastly different feature magnitudes (e.g., `Insulin` values reaching ~150 vs. `DiabetesPedigreeFunction` sitting at ~0.2). This step ensures distance-based and linear models treat all features with equal weight.

### 4. Model Execution & Evaluation
We trained and benchmarked **four** different classification algorithms:
1. **K-Nearest Neighbors (k-NN)**
2. **Decision Tree**
3. **Random Forest**
4. **Logistic Regression (Winner)**: Troughs out the best performance baseline.

### 5. Optimization & Hyperparameter Tuning
To combat the massive class imbalance (where healthy patients heavily outnumber diabetic patients), we performed optimization via **`GridSearchCV`**. 

By testing different values for `C`, `penalty`, and crucially introducing **`class_weight='balanced'`**, we forced the Logistic Regression model to penalize missing a diabetic case. 

**The Result:** The model successfully shifted from an ineffective baseline to capturing **1,511 True Positives**



---
