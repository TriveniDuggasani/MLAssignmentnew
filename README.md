# Fetal Health Classification - ML Assignment

## A. Problem Statement

The objective of this project is to classify fetal health status into three categories based on cardiotocography (CTG) measurements:
- **Normal (Class 0)** - Healthy fetal condition
- **Suspect (Class 1)** - Intermediate risk, requires monitoring
- **Pathological (Class 2)** - High risk, requires immediate intervention

Using various fetal heart rate measurements and clinical indicators, multiple machine learning models are trained to predict fetal health status. This classification helps healthcare practitioners quickly identify at-risk pregnancies and take appropriate medical interventions to improve outcomes.

**Target Variable:** `fetal_health` (encoded as 1, 2, 3 in original data → converted to 0, 1, 2 for modeling)

---

## B. Dataset Description

**Dataset Name:** Fetal Health Dataset  
**Location:** `data/fetal_health.csv`  
**Total Features:** 21 input features + 1 target variable

### Features:
1. **baseline_value** - Fetal heart rate baseline (bpm)
2. **accelerations** - Number of fetal heart rate accelerations per second
3. **fetal_movement** - Number of fetal movements per second
4. **uterine_contractions** - Number of uterine contractions per second
5. **light_decelerations** - Number of light decelerations per second
6. **severe_decelerations** - Number of severe decelerations per second
7. **prolongued_decelerations** - Number of prolonged decelerations per second
8. **abnormal_short_term_variability** - Percentage of abnormal short-term variability (0-100)
9. **mean_value_of_short_term_variability** - Mean value of short-term variability
10. **percentage_of_time_with_abnormal_long_term_variability** - Percentage (0-100)
11. **mean_value_of_long_term_variability** - Mean value of long-term variability
12. **histogram_width** - Width of fetal heart rate histogram
13. **histogram_min** - Minimum fetal heart rate in histogram
14. **histogram_max** - Maximum fetal heart rate in histogram
15. **histogram_number_of_peaks** - Number of peaks in histogram
16. **histogram_number_of_zeroes** - Number of zeros in histogram
17. **histogram_mode** - Mode value in histogram
18. **histogram_mean** - Mean value in histogram
19. **histogram_median** - Median value in histogram
20. **histogram_variance** - Variance in histogram
21. **fetal_health** - Target variable (1=Normal, 2=Suspect, 3=Pathological)

### Data Characteristics:
- **Data Split:** 80% training, 20% testing (stratified)
- **Random State:** 42 (for reproducibility)
- **Class Distribution:** Balanced using stratified splitting
- **Feature Types:** All continuous numerical features
- **Missing Values:** Handled using imputation (median/mean strategies)

---

## C. Models Used

Six machine learning models were trained and evaluated:

### 1. **Logistic Regression**
- **Type:** Linear Classification Model
- **Pipeline:** Imputer (median) → StandardScaler → LogisticRegression
- **Parameters:** `max_iter=2000, multi_class='multinomial'`
- **Rationale:** Baseline linear model for multiclass classification

### 2. **Decision Tree Classifier**
- **Type:** Tree-based Classification
- **Pipeline:** Imputer (median) → DecisionTreeClassifier
- **Parameters:** `random_state=42, class_weight='balanced'`
- **Rationale:** Captures non-linear patterns with interpretable decision rules

### 3. **K-Nearest Neighbors (KNN)**
- **Type:** Distance-based Classification
- **Pipeline:** Imputer (median) → StandardScaler → KNeighborsClassifier
- **Parameters:** `n_neighbors=15`
- **Rationale:** Instance-based learning for local pattern detection

### 4. **Gaussian Naive Bayes**
- **Type:** Probabilistic Classification
- **Pipeline:** Imputer (mean) → GaussianNB
- **Parameters:** Default (assumes Gaussian distribution)
- **Rationale:** Fast probabilistic baseline for continuous features

### 5. **Random Forest Classifier (Ensemble)**
- **Type:** Ensemble Tree-based Method
- **Pipeline:** Imputer (median) → RandomForestClassifier
- **Parameters:** `n_estimators=400, random_state=42, class_weight='balanced_subsample'`
- **Rationale:** Robust ensemble reducing overfitting with feature importance

### 6. **XGBoost Classifier (Ensemble)**
- **Type:** Gradient Boosting Ensemble
- **Pipeline:** Imputer (median) → XGBClassifier
- **Parameters:** `n_estimators=400, max_depth=6, learning_rate=0.1, objective='multi:softprob', eval_metric='mlogloss'`
- **Rationale:** State-of-the-art gradient boosting for optimal performance

---

## D. Model Performance Comparison

### Performance Metrics Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1-Score | MCC |
|---|---|---|---|---|---|---|
| **Logistic Regression** | 0.8375 (83.75%) | 0.9397 | 0.6885 | 0.7514 | 0.7147 | 0.6010 |
| **Decision Tree** | 0.9000 (90.00%) | 0.8346 | 0.7847 | 0.7571 | 0.7695 | 0.7192 |
| **kNN** | 0.8625 (86.25%) | 0.9428 | 0.7744 | 0.6746 | 0.7114 | 0.5836 |
| **Naive Bayes** | 0.8125 (81.25%) | 0.8853 | 0.6493 | 0.7552 | 0.6859 | 0.5863 |
| **Random Forest (Ensemble)** | **0.9125 (91.25%)** | **0.9620** | **0.8793** | 0.7584 | **0.8081** | 0.7435 |
| **XGBoost (Ensemble)** | **0.9313 (93.13%)** ⭐ | **0.9650** ⭐ | **0.8963** ⭐ | **0.8143** ⭐ | **0.8507** ⭐ | **0.8023** ⭐ |

---

## E. Model Performance Observations

### 1. **Logistic Regression**
**Performance:** ⭐⭐⭐ (Baseline)
- **Accuracy:** 83.75%
- **Strengths:**
  - Good baseline performance with reasonable accuracy
  - Excellent AUC (0.9397) demonstrates strong discrimination ability
  - Computationally efficient and interpretable
  - Stable predictions with low variance
- **Weaknesses:**
  - Linear decision boundaries limit capturing complex patterns
  - Lowest precision (0.6885) among all models
  - Moderate MCC (0.6010) indicates fair agreement
- **Use Case:** Quick baseline model for comparison

---

### 2. **Decision Tree Classifier**
**Performance:** ⭐⭐⭐⭐ (Good)
- **Accuracy:** 90.00%
- **Strengths:**
  - Strong accuracy improvement (90%) compared to logistic regression
  - Highly interpretable with visual decision rules
  - Captures non-linear relationships effectively
  - Good MCC (0.7192) for balanced performance
- **Weaknesses:**
  - Lower AUC (0.8346) suggests weaker probabilistic discrimination
  - Risk of overfitting on training data
  - May need pruning for better generalization
- **Use Case:** Interpretable model for clinical decision support

---

### 3. **K-Nearest Neighbors (KNN)**
**Performance:** ⭐⭐⭐⭐ (Good)
- **Accuracy:** 86.25%
- **Strengths:**
  - Highest AUC (0.9428) among all models - excellent discrimination
  - Simple and intuitive algorithm
  - Natural multiclass classification
  - Non-parametric approach
- **Weaknesses:**
  - Lower recall (0.6746) - misses some pathological cases
  - Computational cost increases with dataset size
  - Not scalable for very large datasets
  - Memory-intensive (stores all training samples)
  - Sensitive to feature scaling (though mitigated by StandardScaler)
- **Use Case:** Real-time classification on moderate datasets

---

### 4. **Gaussian Naive Bayes**
**Performance:** ⭐⭐ (Below Average)
- **Accuracy:** 81.25%
- **Strengths:**
  - Fastest training and prediction among all models
  - Good recall (0.7552) catches many pathological cases
  - Probabilistic framework provides uncertainty estimates
  - Handles multiclass naturally
- **Weaknesses:**
  - **Lowest accuracy (81.25%)** among all models
  - Violates independence assumption for features
  - Lower precision (0.6493) leads to false alarms
  - Weakest MCC (0.5863) indicates poor overall agreement
  - Oversimplified model for complex medical data
- **Use Case:** Quick baseline or real-time applications where speed is critical

---

### 5. **Random Forest Classifier (Ensemble)**
**Performance:** ⭐⭐⭐⭐⭐ (Excellent)
- **Accuracy:** 91.25%
- **Strengths:**
  - **Best precision among traditional models (0.8793)** - minimal false alarms
  - Excellent AUC (0.9620) - superior discrimination
  - Strong F1-score (0.8081) balancing precision and recall
  - Robust to outliers and non-linear patterns
  - Provides feature importance rankings for interpretability
  - Ensemble approach reduces overfitting risk
  - Good MCC (0.7435) indicates strong balanced performance
- **Weaknesses:**
  - Slightly lower recall (0.7584) compared to XGBoost
  - Less accurate probabilistic estimates than boosting
  - More memory-intensive than single models
- **Use Case:** **Recommended production baseline model**

---

### 6. **XGBoost Classifier (Ensemble)** 🏆
**Performance:** ⭐⭐⭐⭐⭐ (Best Overall)
- **Accuracy:** 93.13% - **Highest accuracy**
- **Strengths:**
  - **BEST OVERALL PERFORMANCE** across all metrics
  - Highest MCC (0.8023) - strongest agreement coefficient
  - Excellent AUC (0.9650) - near-perfect discrimination
  - Best precision (0.8963) and recall (0.8143) balance
  - Highest F1-score (0.8507) - best harmonic mean
  - Sequential boosting reduces bias effectively
  - Captures complex non-linear patterns
  - Handles class imbalance well
- **Weaknesses:**
  - More computationally expensive than alternatives
  - Requires careful hyperparameter tuning
  - More prone to overfitting if not properly regularized
  - Slower training/prediction than simple models
- **Use Case:** **Recommended for production deployment**

---

## F. Summary & Key Findings

### Performance Ranking:
1. 🥇 **XGBoost** - 93.13% accuracy (Best)
2. 🥈 **Random Forest** - 91.25% accuracy
3. 🥉 **Decision Tree** - 90.00% accuracy
4. **KNN** - 86.25% accuracy
5. **Logistic Regression** - 83.75% accuracy
6. **Naive Bayes** - 81.25% accuracy

### Key Insights:
✅ **Ensemble methods significantly outperform single models**
- XGBoost and Random Forest achieve 91.25%+ accuracy
- Both achieve AUC > 0.96, indicating excellent discrimination

✅ **Non-linear models capture complex patterns better**
- Tree-based models outperform linear models
- Decision boundaries matter for medical data

✅ **Balance between Precision and Recall is critical**
- XGBoost achieves the best balance (89.63% precision, 81.43% recall)
- Random Forest prioritizes precision (87.93%) over recall

⚠️ **Naive Bayes underperforms due to independence assumption**
- Feature dependence in medical data violates model assumptions
- Linear models struggle with non-linear relationships

### Recommendations:
🎯 **Deploy XGBoost as primary model** - highest accuracy and MCC
🎯 **Use Random Forest as fallback** - similar performance, simpler interpretation
🎯 **Implement ensemble voting** - combine XGBoost + Random Forest predictions
🎯 **Monitor model drift** - retrain periodically with new data
🎯 **Use SHAP values** - explain model predictions for clinical acceptance

---

## G. Project Structure

```
MLAssignmentnew/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── streamlit_app.py                   # Web UI for model testing
├── data/
│   └── fetal_health.csv              # Clinical dataset
└── model/
    ├── logistic_regression_fetal_health.py
    ├── decision_tree_fetal_health.py
    ├── knn_fetal_health.py
    ├── gaussian_nb_fetal_health.py
    ├── random_forest_fetal_health.py
    ├── xgboost_fetal_health.py
    └── README_models.txt              # Detailed model documentation
```

---

## H. How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Individual Models
```bash
# Run each model from the model directory
python model/logistic_regression_fetal_health.py
python model/decision_tree_fetal_health.py
python model/knn_fetal_health.py
python model/gaussian_nb_fetal_health.py
python model/random_forest_fetal_health.py
python model/xgboost_fetal_health.py
```

### 3. Run Web Application
```bash
# Interactive Streamlit UI for model comparison
streamlit run streamlit_app.py
```

### 4. Upload Your Dataset
- Download sample data from the sidebar
- Upload your own .csv / .xlsx file with `fetal_health` column
- Select model and view evaluation metrics

---

## I. Metrics Definitions

- **Accuracy:** Proportion of correct predictions
- **AUC (Area Under Curve):** Probability the model ranks a random positive example higher than a random negative example
- **Precision:** Of predicted positive cases, how many were actually positive (minimizes false alarms)
- **Recall:** Of all actual positive cases, how many were correctly identified (minimizes missed cases)
- **F1-Score:** Harmonic mean of precision and recall (balanced metric)
- **MCC (Matthews Correlation Coefficient):** Correlation between predicted and actual values (-1 to 1, where 1 is perfect)

---

**Generated:** February 15, 2026  
**Project:** Fetal Health Classification using Machine Learning