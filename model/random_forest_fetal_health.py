
#!/usr/bin/env python3
"""
Standalone script for training and evaluating Random Forest on fetal_health.csv
Prints: Accuracy, AUC (macro OvR), Precision (macro), Recall (macro), F1 (macro), MCC
Saves: classification report (txt) & confusion matrix (csv)
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support, matthews_corrcoef, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'data', 'fetal_health.csv')
    df = pd.read_csv(data_path)
    X = df.drop('fetal_health', axis=1)
    # shift labels from {1,2,3} -> {0,1,2} for compatibility
    y = df['fetal_health'].astype(int) - 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=400, random_state=42, class_weight='balanced_subsample')

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)
        auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
    else:
        auc = float('nan')
    acc = accuracy_score(y_test, y_pred)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)
    print(f"Model: Random Forest")
    print(f"Accuracy: {acc:.4f}")
    print(f"AUC (macro OvR): {auc:.4f}")
    print(f"Precision (macro): {prec_macro:.4f}")
    print(f"Recall (macro): {rec_macro:.4f}")
    print(f"F1-score (macro): {f1_macro:.4f}")
    print(f"MCC: {mcc:.4f}")
    with open('random_forest_classification_report.txt', 'w') as f:
        f.write(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    pd.DataFrame(cm).to_csv('random_forest_confusion_matrix.csv', index=False)

if __name__ == '__main__':
    main()
