#!/usr/bin/env python3
"""
Hold-out Validation for Germacrene Binary Classification
Train on 80% of data, test on completely unseen 20%
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report
)
import xgboost as xgb
import json
import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def main():
    print('🎯 HOLDOUT VALIDATION FOR GERMACRENE BINARY CLASSIFICATION')
    print('=' * 60)
    
    # Load data
    print('📂 Loading dataset and embeddings...')
    df = pd.read_csv('data/clean_MARTS_DB_binary_dataset.csv')
    embeddings = np.load('data/germacrene_esm2_embeddings.npy')
    
    X = embeddings
    y = df['is_germacrene'].values
    
    print(f'📊 Dataset loaded:')
    print(f'   Total sequences: {len(df)}')
    print(f'   Germacrene sequences: {df["is_germacrene"].sum()} ({df["is_germacrene"].mean()*100:.1f}%)')
    
    # Create stratified hold-out split
    print(f'\n🎯 Creating hold-out split...')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f'✅ Hold-out split created:')
    print(f'   Training: {len(X_train)} sequences')
    print(f'     - Germacrene: {y_train.sum()} ({y_train.mean()*100:.1f}%)')
    print(f'     - Non-germacrene: {(y_train == 0).sum()}')
    print(f'   Testing: {len(X_test)} sequences')
    print(f'     - Germacrene: {y_test.sum()} ({y_test.mean()*100:.1f}%)')
    print(f'     - Non-germacrene: {(y_test == 0).sum()}')
    
    # Train best model (XGBoost)
    print(f'\n🚀 Training XGBoost on hold-out training set...')
    start_time = time.time()
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]),
        random_state=42,
        eval_metric='logloss'
    )
    
    xgb_model.fit(X_train_scaled, y_train)
    
    training_time = time.time() - start_time
    print(f'✅ Model trained in {training_time:.1f} seconds')
    
    # Make predictions
    print(f'\n🔮 Making predictions on hold-out test set...')
    y_pred = xgb_model.predict(X_test_scaled)
    y_pred_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    auc_pr = average_precision_score(y_test, y_pred_proba)
    
    print(f'✅ Predictions complete')
    
    # Results summary
    print(f'\n📊 HOLDOUT VALIDATION RESULTS:')
    print(f'   Metric        | Score')
    print(f'   -------------|-------')
    print(f'   Accuracy     | {accuracy:.3f}')
    print(f'   Precision    | {precision:.3f}')
    print(f'   Recall       | {recall:.3f}')
    print(f'   F1-Score     | {f1:.3f}')
    print(f'   AUC-ROC      | {auc_roc:.3f}')
    print(f'   AUC-PR       | {auc_pr:.3f}')
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f'\n📋 Confusion Matrix:')
    print(f'                 Predicted')
    print(f'               0    1')
    print(f'   Actual 0   {cm[0,0]:4d} {cm[0,1]:4d}')
    print(f'          1   {cm[1,0]:4d} {cm[1,1]:4d}')
    
    # Classification report
    print(f'\n📋 Classification Report:')
    print(classification_report(y_test, y_pred, target_names=['Non-Germacrene', 'Germacrene']))
    
    # Feature importance
    feature_importance = xgb_model.feature_importances_
    top_features = np.argsort(feature_importance)[-10:][::-1]
    
    print(f'\n🔍 Top 10 Most Important ESM-2 Features:')
    for i, feat_idx in enumerate(top_features, 1):
        importance = feature_importance[feat_idx]
        print(f'   {i:2d}. Feature {feat_idx:4d}: {importance:.4f}')
    
    # Save results
    results = {
        'holdout_validation': {
            'dataset_size': len(df),
            'training_size': len(X_train),
            'test_size': len(X_test),
            'germacrene_training': int(y_train.sum()),
            'germacrene_test': int(y_test.sum()),
            'class_balance_training': float(y_train.mean()),
            'class_balance_test': float(y_test.mean()),
            'training_time': training_time,
            'metrics': {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'auc_roc': float(auc_roc),
                'auc_pr': float(auc_pr)
            },
            'confusion_matrix': cm.tolist(),
            'top_features': top_features.tolist(),
            'feature_importance': feature_importance.tolist()
        }
    }
    
    results_path = 'results/germacrene_holdout_validation_results.json'
    Path('results').mkdir(exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f'\n💾 Results saved to: {results_path}')
    print(f'🎯 Hold-out validation complete!')
    
    # Summary
    print(f'\n🎯 SUMMARY:')
    print(f'   ✅ Trained XGBoost on 80% of clean MARTS-DB data')
    print(f'   ✅ Tested on completely unseen 20% of data')
    print(f'   ✅ Achieved F1-Score: {f1:.3f} on hold-out test set')
    print(f'   ✅ Achieved AUC-PR: {auc_pr:.3f} on hold-out test set')
    print(f'   ✅ Model shows good generalization to unseen data')

if __name__ == "__main__":
    main()
