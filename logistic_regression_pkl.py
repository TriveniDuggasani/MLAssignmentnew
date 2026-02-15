#!/usr/bin/env python3
"""
Script to train Logistic Regression model and save it as a pickle file.
"""

import os
import pickle
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

def load_data():
    """Load and preprocess the fetal health dataset."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, 'data', 'fetal_health.csv')
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    
    df = pd.read_csv(data_path)
    
    # Extract features and target
    X = df.drop('fetal_health', axis=1)
    y = df['fetal_health'].astype(int) - 1  # Convert {1,2,3} -> {0,1,2}
    
    return X, y

def main():
    try:
        print("=" * 70)
        print("LOGISTIC REGRESSION MODEL - PICKLE GENERATION")
        print("=" * 70 + "\n")
        
        # Load data
        print("📊 Loading dataset...")
        X, y = load_data()
        print(f"✅ Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features\n")
        
        # Train/test split
        print("📈 Splitting data (80/20 train/test)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"✅ Training set: {X_train.shape[0]} samples")
        print(f"✅ Testing set: {X_test.shape[0]} samples\n")
        
        # Build and train model
        print("⏳ Building and training Logistic Regression...")
        model = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=2000, random_state=42))
        ])
        model.fit(X_train, y_train)
        print("✅ Model training completed\n")
        
        # Evaluate
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        print(f"📊 Model Performance:")
        print(f"   Training Accuracy: {train_score:.4f}")
        print(f"   Testing Accuracy:  {test_score:.4f}\n")
        
        # Create output directory
        os.makedirs('model/pickles', exist_ok=True)
        
        # Save model
        pkl_path = 'model/pickles/logistic_regression.pkl'
        with open(pkl_path, 'wb') as f:
            pickle.dump(model, f)
        
        file_size = os.path.getsize(pkl_path) / 1024  # Convert to KB
        print(f"✅ Model saved as pickle file")
        print(f"   📂 Location: {pkl_path}")
        print(f"   📏 File size: {file_size:.2f} KB\n")
        
        # Usage instructions
        print("=" * 70)
        print("USAGE INSTRUCTIONS")
        print("=" * 70)
        print("Load and use the saved model:")
        print("\n   import pickle")
        print("   with open('model/pickles/logistic_regression.pkl', 'rb') as f:")
        print("       model = pickle.load(f)")
        print("   predictions = model.predict(X_new)")
        print("\n" + "=" * 70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
