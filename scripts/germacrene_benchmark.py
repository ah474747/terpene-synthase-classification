#!/usr/bin/env python3
"""
Germacrene Binary Classification Benchmark
7-algorithm ML benchmark using ESM-2 embeddings
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import json
import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def main():
    print('🧬 GERMACRENE BINARY CLASSIFICATION BENCHMARK')
    print('=' * 55)
    
    # Load data
    print('📂 Loading dataset and embeddings...')
    df = pd.read_csv('data/clean_MARTS_DB_binary_dataset.csv')
    embeddings = np.load('data/germacrene_esm2_embeddings.npy')
    
    X = embeddings
    y = df['is_germacrene'].values
    
    print(f'📊 Dataset loaded:')
    print(f'   Total sequences: {len(df)}')
    print(f'   Germacrene sequences: {df["is_germacrene"].sum()} ({df["is_germacrene"].mean()*100:.1f}%)')
    print(f'   Non-germacrene sequences: {(df["is_germacrene"] == 0).sum()}')
    print(f'   Embeddings shape: {X.shape}')
    
    # Define models
    models = {
        'XGBoost': {
            'model': Pipeline([
                ('scaler', StandardScaler()),
                ('xgb', xgb.XGBClassifier(
                    random_state=42,
                    scale_pos_weight=len(y[y==0])/len(y[y==1]),
                    eval_metric='logloss'
                ))
            ]),
            'param_grid': {
                'xgb__n_estimators': [100, 200, 300],
                'xgb__max_depth': [3, 4, 5, 6],
                'xgb__learning_rate': [0.01, 0.1, 0.2],
                'xgb__subsample': [0.8, 0.9, 1.0],
                'xgb__colsample_bytree': [0.8, 0.9, 1.0]
            }
        },
        'Random_Forest': {
            'model': Pipeline([
                ('scaler', StandardScaler()),
                ('rf', RandomForestClassifier(
                    random_state=42,
                    class_weight='balanced'
                ))
            ]),
            'param_grid': {
                'rf__n_estimators': [100, 200, 300],
                'rf__max_depth': [10, 20, 30, None],
                'rf__min_samples_split': [2, 5, 10],
                'rf__min_samples_leaf': [1, 2, 4]
            }
        },
        'SVM_RBF': {
            'model': Pipeline([
                ('scaler', StandardScaler()),
                ('svm', SVC(
                    kernel='rbf',
                    random_state=42,
                    class_weight='balanced',
                    probability=True
                ))
            ]),
            'param_grid': {
                'svm__C': [0.1, 1, 10, 100],
                'svm__gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
            }
        },
        'Logistic_Regression': {
            'model': Pipeline([
                ('scaler', StandardScaler()),
                ('lr', LogisticRegression(
                    random_state=42,
                    class_weight='balanced',
                    max_iter=1000
                ))
            ]),
            'param_grid': {
                'lr__C': [0.001, 0.01, 0.1, 1, 10, 100],
                'lr__penalty': ['l1', 'l2', 'elasticnet'],
                'lr__solver': ['liblinear', 'saga']
            }
        },
        'MLP': {
            'model': Pipeline([
                ('scaler', StandardScaler()),
                ('mlp', MLPClassifier(
                    random_state=42,
                    max_iter=1000,
                    early_stopping=True,
                    validation_fraction=0.1
                ))
            ]),
            'param_grid': {
                'mlp__hidden_layer_sizes': [(100,), (200,), (100, 50), (200, 100)],
                'mlp__activation': ['relu', 'tanh'],
                'mlp__learning_rate': ['constant', 'adaptive'],
                'mlp__alpha': [0.0001, 0.001, 0.01]
            }
        },
        'KNN': {
            'model': Pipeline([
                ('scaler', StandardScaler()),
                ('knn', KNeighborsClassifier())
            ]),
            'param_grid': {
                'knn__n_neighbors': [3, 5, 7, 9, 11],
                'knn__weights': ['uniform', 'distance'],
                'knn__metric': ['euclidean', 'manhattan']
            }
        },
        'Perceptron': {
            'model': Pipeline([
                ('scaler', StandardScaler()),
                ('perceptron', Perceptron(
                    random_state=42,
                    class_weight='balanced',
                    max_iter=1000
                ))
            ]),
            'param_grid': {
                'perceptron__alpha': [0.0001, 0.001, 0.01, 0.1],
                'perceptron__penalty': [None, 'l1', 'l2'],
                'perceptron__eta0': [0.01, 0.1, 1.0]
            }
        }
    }
    
    # Cross-validation setup
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Results storage
    results = {}
    
    print(f'\n🚀 RUNNING 7-ALGORITHM BENCHMARK')
    print(f'   Cross-validation: 5-fold stratified')
    print(f'   Hyperparameter tuning: Randomized search')
    
    # Run benchmark
    for model_name, model_config in models.items():
        print(f'\n🔬 Training {model_name}...')
        
        start_time = time.time()
        
        try:
            # Randomized search with cross-validation
            random_search = RandomizedSearchCV(
                model_config['model'],
                model_config['param_grid'],
                n_iter=20,
                cv=cv,
                scoring='average_precision',
                random_state=42,
                n_jobs=-1,
                verbose=0
            )
            
            random_search.fit(X, y)
            
            # Get best model and predictions
            best_model = random_search.best_estimator_
            
            # Cross-validation evaluation
            cv_scores = {
                'accuracy': [],
                'precision': [],
                'recall': [],
                'f1': [],
                'auc_roc': [],
                'auc_pr': []
            }
            
            for train_idx, val_idx in cv.split(X, y):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Train on fold
                fold_model = random_search.best_estimator_
                fold_model.fit(X_train, y_train)
                
                # Predictions
                y_pred = fold_model.predict(X_val)
                
                # Handle probability predictions for AUC
                try:
                    y_pred_proba = fold_model.predict_proba(X_val)[:, 1]
                except AttributeError:
                    # For models without predict_proba, use decision_function
                    try:
                        y_pred_proba = fold_model.decision_function(X_val)
                        # Normalize to 0-1 range
                        y_pred_proba = (y_pred_proba - y_pred_proba.min()) / (y_pred_proba.max() - y_pred_proba.min())
                    except AttributeError:
                        # Fallback to predictions
                        y_pred_proba = y_pred
                
                # Calculate metrics
                cv_scores['accuracy'].append(accuracy_score(y_val, y_pred))
                cv_scores['precision'].append(precision_score(y_val, y_pred, zero_division=0))
                cv_scores['recall'].append(recall_score(y_val, y_pred, zero_division=0))
                cv_scores['f1'].append(f1_score(y_val, y_pred, zero_division=0))
                
                try:
                    cv_scores['auc_roc'].append(roc_auc_score(y_val, y_pred_proba))
                    cv_scores['auc_pr'].append(average_precision_score(y_val, y_pred_proba))
                except ValueError:
                    cv_scores['auc_roc'].append(0.5)
                    cv_scores['auc_pr'].append(y_val.mean())
            
            # Calculate mean and std
            model_results = {
                'model_name': model_name,
                'best_params': random_search.best_params_,
                'cv_scores': {
                    metric: {
                        'mean': float(np.mean(scores)),
                        'std': float(np.std(scores))
                    }
                    for metric, scores in cv_scores.items()
                },
                'training_time': time.time() - start_time
            }
            
            results[model_name] = model_results
            
            # Print results
            print(f'   ✅ {model_name} completed')
            print(f'   F1-Score: {model_results["cv_scores"]["f1"]["mean"]:.3f} ± {model_results["cv_scores"]["f1"]["std"]:.3f}')
            print(f'   AUC-PR: {model_results["cv_scores"]["auc_pr"]["mean"]:.3f} ± {model_results["cv_scores"]["auc_pr"]["std"]:.3f}')
            print(f'   Training time: {model_results["training_time"]:.1f}s')
            
        except Exception as e:
            print(f'   ❌ {model_name} failed: {str(e)}')
            results[model_name] = {
                'model_name': model_name,
                'error': str(e),
                'cv_scores': None
            }
    
    # Save results
    results_path = 'results/germacrene_benchmark_results.json'
    Path('results').mkdir(exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f'\n📊 BENCHMARK RESULTS SUMMARY:')
    print(f'   Model                | F1-Score | AUC-PR  | Training Time')
    print(f'   --------------------|----------|---------|-------------')
    
    # Sort by F1-score
    sorted_results = sorted(
        [r for r in results.values() if r.get('cv_scores')],
        key=lambda x: x['cv_scores']['f1']['mean'],
        reverse=True
    )
    
    for result in sorted_results:
        model_name = result['model_name']
        f1_mean = result['cv_scores']['f1']['mean']
        f1_std = result['cv_scores']['f1']['std']
        auc_pr_mean = result['cv_scores']['auc_pr']['mean']
        auc_pr_std = result['cv_scores']['auc_pr']['std']
        train_time = result['training_time']
        
        print(f'   {model_name:<19} | {f1_mean:.3f}±{f1_std:.3f} | {auc_pr_mean:.3f}±{auc_pr_std:.3f} | {train_time:.1f}s')
    
    print(f'\n💾 Results saved to: {results_path}')
    print(f'🎯 Germacrene binary classification benchmark complete!')

if __name__ == "__main__":
    main()
