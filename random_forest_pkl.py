#!/usr/bin/env python3
"""
Script to train Random Forest model and save it as a pickle file.
"""

import os
import pickle
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

def load_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, 'data', 'fetal_health.csv')
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    
    df = pd.read_csv(data_path)
    X = df.drop('fetal_health', axis=1)
    y = df['fetal_health'].astype(int) - 1
    return X, y

def main():
    try:
        print("=" * 70)
        print("RANDOM FOREST MODEL - PICKLE GENERATION")
        print("=" * 70 + "\n")
        
        print("📊 Loading dataset...")
        X, y = load_data()
        print(f"✅ Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features\n")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print("⏳ Building and training Random Forest...")
        model = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('clf', RandomForestClassifier(
                n_estimators=400,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced_subsample'
            ))
        ])
        model.fit(X_train, y_train)
        print("✅ Model training completed\n")
        
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        print(f"📊 Model Performance:")
        print(f"   Training Accuracy: {train_score:.4f}")
        print(f"   Testing Accuracy:  {test_score:.4f}\n")
        
        os.makedirs('model/pickles', exist_ok=True)
        pkl_path = 'model/pickles/random_forest.pkl'
        with open(pkl_path, 'wb') as f:
            pickle.dump(model, f)
        
        file_size = os.path.getsize(pkl_path) / (1024 * 1024)
        print(f"✅ Model saved: {pkl_path} ({file_size:.2f} MB)\n")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
