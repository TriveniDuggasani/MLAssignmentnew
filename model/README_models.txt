================================================================================
                    FETAL HEALTH CLASSIFICATION
                      ML MODELS DOCUMENTATION
================================================================================

================================================================================
A. PROBLEM STATEMENT
================================================================================

The objective of this project is to classify fetal health into three categories:
- Normal (Class 0)
- Suspect (Class 1)
- Pathological (Class 2)

Using cardiotocography (fetal heart rate) measurements and other clinical 
indicators, machine learning models are trained to predict fetal health status. 
This helps healthcare practitioners quickly identify at-risk pregnancies and 
take appropriate medical interventions.

The target variable is "fetal_health" (originally encoded as 1, 2, 3, 
converted to 0, 1, 2 for model compatibility).


================================================================================
B. DATASET DESCRIPTION
================================================================================

Dataset Name: Fetal Health Dataset (fetal_health.csv)
Location: data/fetal_health.csv

Features (21 total):
1. baseline_value - Fetal heart rate baseline (bpm)
2. accelerations - Number of fetal heart rate accelerations per second
3. fetal_movement - Number of fetal movements per second
4. uterine_contractions - Number of uterine contractions per second
5. light_decelerations - Number of light decelerations per second
6. severe_decelerations - Number of severe decelerations per second
7. prolongued_decelerations - Number of prolonged decelerations per second
8. abnormal_short_term_variability - Percentage (0-100)
9. mean_value_of_short_term_variability - Mean value
10. percentage_of_time_with_abnormal_long_term_variability - Percentage (0-100)
11. mean_value_of_long_term_variability - Mean value
12. histogram_width - Width of FHR histogram
13. histogram_min - Minimum FHR in histogram
14. histogram_max - Maximum FHR in histogram
15. histogram_number_of_peaks - Number of peaks in histogram
16. histogram_number_of_zeroes - Number of zeros in histogram
17. histogram_mode - Mode value in histogram
18. histogram_mean - Mean value in histogram
19. histogram_median - Median value in histogram
20. histogram_variance - Variance in histogram

Target Variable:
- fetal_health: 1 (Normal), 2 (Suspect), 3 (Pathological)

Data Split:
- Training set: 80%
- Testing set: 20%
- Random state: 42
- Stratified split: Yes (maintains class distribution)


================================================================================
C. MODELS USED
================================================================================

The following 6 machine learning models were trained and evaluated:

1. LOGISTIC REGRESSION
   - Type: Linear Classification
   - Pipeline: Imputer (median) → StandardScaler → LogisticRegression
   - Parameters: max_iter=2000
   - Use Case: Baseline model for multiclass classification

2. DECISION TREE CLASSIFIER
   - Type: Tree-based Classification
   - Pipeline: Imputer (median) → DecisionTreeClassifier
   - Parameters: random_state=42, class_weight='balanced'
   - Use Case: Interpretable, non-linear decision boundaries

3. K-NEAREST NEIGHBORS (KNN)
   - Type: Distance-based Classification
   - Pipeline: Imputer (median) → StandardScaler → KNeighborsClassifier
   - Parameters: n_neighbors=15
   - Use Case: Instance-based learning without explicit training phase

4. GAUSSIAN NAIVE BAYES
   - Type: Probabilistic Classification
   - Pipeline: Imputer (mean) → GaussianNB
   - Parameters: Default (assumes Gaussian distribution)
   - Use Case: Fast probabilistic baseline for continuous features

5. RANDOM FOREST CLASSIFIER (Ensemble)
   - Type: Ensemble Tree-based
   - Pipeline: Imputer (median) → RandomForestClassifier
   - Parameters: n_estimators=400, random_state=42, class_weight='balanced_subsample'
   - Use Case: Strong baseline ensemble with feature importance analysis

6. XGBOOST CLASSIFIER (Ensemble)
   - Type: Gradient Boosting Ensemble
   - Pipeline: Imputer (median) → XGBClassifier
   - Parameters: n_estimators=400, max_depth=6, learning_rate=0.1, 
                 objective='multi:softprob', eval_metric='mlogloss'
   - Use Case: State-of-the-art gradient boosting for best performance


================================================================================
D. MODEL PERFORMANCE & OBSERVATIONS
================================================================================

Evaluation Metrics Used:
- Accuracy: Proportion of correct predictions
- AUC (macro OvR): Area Under Curve (One-vs-Rest, macro average)
- Precision (macro): Average precision across all classes
- Recall (macro): Average recall across all classes
- F1-score (macro): Harmonic mean of precision and recall
- MCC: Matthews Correlation Coefficient (-1 to 1)

---

1. LOGISTIC REGRESSION

   Accuracy:  0.8375 (83.75%)
   AUC:       0.9397 (93.97%)
   Precision: 0.6885 (68.85%)
   Recall:    0.7514 (75.14%)
   F1-score:  0.7147 (71.47%)
   MCC:       0.6010
   
   Observations:
   • Baseline model performs reasonably well with ~84% accuracy
   • Good AUC (0.94) indicates strong discrimination ability
   • Linear decision boundaries may limit capturing complex patterns
   • Stable and computationally efficient
   • Moderate MCC suggests reasonable agreement beyond chance

---

2. DECISION TREE CLASSIFIER

   Accuracy:  0.9000 (90.00%)
   AUC:       0.8346 (83.46%)
   Precision: 0.7847 (78.47%)
   Recall:    0.7571 (75.71%)
   F1-score:  0.7695 (76.95%)
   MCC:       0.7192
   
   Observations:
   • Strong accuracy improvement (90%) over logistic regression
   • Decision tree captures non-linear patterns effectively
   • Lower AUC (0.83) suggests some class separation challenges
   • Good MCC (0.72) indicates balanced performance
   • Interpretable model - easy to visualize decision rules
   • Risk of overfitting due to tree depth (should monitor validation curves)

---

3. K-NEAREST NEIGHBORS (KNN)

   Accuracy:  0.8625 (86.25%)
   AUC:       0.9428 (94.28%)
   Precision: 0.7744 (77.44%)
   Recall:    0.6746 (67.46%)
   F1-score:  0.7114 (71.14%)
   MCC:       0.5836
   
   Observations:
   • Highest AUC (0.94+) indicates excellent probabilistic discrimination
   • Lower recall (67.46%) suggests false negatives for some classes
   • Computational cost increases with dataset size (not scalable for huge datasets)
   • Sensitive to feature scaling (mitigated by StandardScaler in pipeline)
   • Instance-based method: no explicit learning, memory-intensive
   • Performance depends on optimal k selection (n_neighbors=15 chosen)

---

4. GAUSSIAN NAIVE BAYES

   Accuracy:  0.8125 (81.25%)
   AUC:       0.8853 (88.53%)
   Precision: 0.6493 (64.93%)
   Recall:    0.7552 (75.52%)
   F1-score:  0.6859 (68.59%)
   MCC:       0.5863
   
   Observations:
   • Lowest accuracy (81.25%) among all models
   • Simplistic assumption of feature independence may not hold
   • Good recall (75.52%) but lower precision indicates class imbalance issues
   • Fast training and prediction - useful for real-time applications
   • MCC (0.59) suggests weaker overall agreement
   • Best used as quick baseline rather than production model
   • May benefit from feature engineering to reduce dependencies

---

5. RANDOM FOREST CLASSIFIER (Ensemble)

   Accuracy:  0.9125 (91.25%)
   AUC:       0.9620 (96.20%)
   Precision: 0.8793 (87.93%)
   Recall:    0.7584 (75.84%)
   F1-score:  0.8081 (80.81%)
   MCC:       0.7435
   
   Observations:
   • Excellent performance: 91.25% accuracy, highest AUC (0.96)
   • Best precision (87.93%) and F1-score (80.81%) among all models
   • Ensemble approach reduces overfitting risk vs single tree
   • Provides feature importance rankings - useful for interpretability
   • Robust to outliers and non-linear patterns
   • Good MCC (0.74) indicates strong balanced performance
   • Handles multi-class classification naturally
   • Recommended as production baseline model

---

6. XGBOOST CLASSIFIER (Ensemble)

   Accuracy:  0.9313 (93.13%)
   AUC:       0.9650 (96.50%)
   Precision: 0.8963 (89.63%)
   Recall:    0.8143 (81.43%)
   F1-score:  0.8507 (85.07%)
   MCC:       0.8023
   
   Observations:
   • BEST OVERALL PERFORMANCE: 93.13% accuracy
   • Highest MCC (0.80) - strongest agreement metric
   • Excellent AUC (0.965) with balanced precision/recall
   • Gradient boosting captures complex non-linear patterns
   • Sequential ensembling reduces bias more effectively than bagging
   • Highest F1-score (0.85) balancing precision and recall well
   • More computationally expensive than Random Forest
   • Requires careful hyperparameter tuning to avoid overfitting
   • RECOMMENDED as the best production model for this dataset

================================================================================
E. SUMMARY & RECOMMENDATIONS
================================================================================

PERFORMANCE RANKING (by Accuracy):
1. XGBoost         - 93.13% ⭐⭐⭐⭐⭐ (Best)
2. Random Forest   - 91.25% ⭐⭐⭐⭐
3. Decision Tree   - 90.00% ⭐⭐⭐
4. KNN             - 86.25% ⭐⭐
5. Logistic Regr.  - 83.75% ⭐⭐
6. Naive Bayes     - 81.25% ⭐

KEY FINDINGS:
✓ Ensemble methods (XGBoost, Random Forest) significantly outperform others
✓ XGBoost achieves best balance across all metrics (Accuracy, Precision, Recall, MCC)
✓ Linear model (Logistic Regression) provides reasonable baseline
✓ Probabilistic models (Naive Bayes) underperform due to independence assumption
✓ All models benefit from feature scaling and imputation

RECOMMENDATIONS:
→ Deploy XGBoost for production use (highest accuracy & MCC)
→ Use Random Forest as fallback (similar AUC, faster training)
→ Implement model explainability using SHAP values (especially for XGBoost)
→ Monitor model drift in production with periodic retraining
→ Consider ensemble voting: combine XGBoost + Random Forest predictions
→ Conduct cost-benefit analysis for false positives vs false negatives
→ Further hyperparameter tuning may yield marginal improvements

================================================================================
File Generated: 2026-02-15
================================================================================
