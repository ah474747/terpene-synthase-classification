#!/usr/bin/env python3
"""
Ent-Kaurene Binary Classifier Benchmark

This script implements a comprehensive benchmark of 7 machine learning algorithms
for binary classification of ent-kaurene synthases using ESM-2 embeddings.

Algorithms tested:
1. XGBoost Classifier
2. Random Forest Classifier  
3. Support Vector Machine (RBF)
4. Logistic Regression
5. Multi-Layer Perceptron
6. k-Nearest Neighbors
7. Perceptron

Author: Cursor AI
Date: October 17, 2025
"""

import pandas as pd
import numpy as np
import json
import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, 
    precision_score, recall_score, confusion_matrix,
    classification_report
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Algorithms
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

# Progress tracking
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

class EntKaureneBenchmark:
    """Comprehensive ML benchmark for ent-kaurene binary classification"""
    
    def __init__(self, data_file="data/ent_kaurene_binary_dataset.csv", 
                 embeddings_file="data/esm2_embeddings.npy"):
        """
        Initialize the benchmark
        
        Args:
            data_file: Path to the binary labeled dataset
            embeddings_file: Path to the ESM-2 embeddings
        """
        self.data_file = data_file
        self.embeddings_file = embeddings_file
        
        # Load data
        self.df = pd.read_csv(data_file)
        self.embeddings = np.load(embeddings_file)
        self.y = self.df['is_ent_kaurene'].values
        
        print(f"📊 Dataset loaded: {len(self.df)} sequences")
        print(f"📊 Embeddings loaded: {self.embeddings.shape}")
        print(f"📊 Class distribution: {np.bincount(self.y)}")
        print(f"📊 Class balance ratio: {np.bincount(self.y)[0]/np.bincount(self.y)[1]:.1f}:1")
        
        # Results storage
        self.results = {}
        self.cv_results = {}
        
        # Cross-validation setup
        self.cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
    def define_algorithms(self):
        """Define the 7 algorithms to benchmark"""
        
        algorithms = {
            'XGBoost': {
                'model': xgb.XGBClassifier(
                    random_state=42,
                    n_jobs=-1,
                    scale_pos_weight=np.bincount(self.y)[0]/np.bincount(self.y)[1]
                ),
                'param_grid': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 4, 5, 6],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0]
                }
            },
            
            'Random Forest': {
                'model': RandomForestClassifier(
                    random_state=42,
                    n_jobs=-1,
                    class_weight='balanced'
                ),
                'param_grid': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            
            'SVM (RBF)': {
                'model': Pipeline([
                    ('scaler', StandardScaler()),
                    ('svm', SVC(
                        kernel='rbf',
                        random_state=42,
                        probability=True,
                        class_weight='balanced'
                    ))
                ]),
                'param_grid': {
                    'svm__C': [0.1, 1, 10, 100],
                    'svm__gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
                }
            },
            
            'Logistic Regression': {
                'model': Pipeline([
                    ('scaler', StandardScaler()),
                    ('lr', LogisticRegression(
                        random_state=42,
                        class_weight='balanced',
                        max_iter=1000
                    ))
                ]),
                'param_grid': {
                    'lr__C': [0.01, 0.1, 1, 10, 100],
                    'lr__penalty': ['l1', 'l2'],
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
                    'mlp__alpha': [0.0001, 0.001, 0.01],
                    'mlp__learning_rate': ['constant', 'adaptive']
                }
            },
            
            'k-NN': {
                'model': Pipeline([
                    ('scaler', StandardScaler()),
                    ('knn', KNeighborsClassifier())
                ]),
                'param_grid': {
                    'knn__n_neighbors': [3, 5, 7, 9, 11],
                    'knn__weights': ['uniform', 'distance'],
                    'knn__metric': ['euclidean', 'manhattan', 'cosine']
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
        
        return algorithms
    
    def evaluate_model(self, model, X_train, X_val, y_train, y_val):
        """Evaluate a model on validation set"""
        
        # Train model
        model.fit(X_train, y_train)
        
        # Get predictions
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        y_pred = model.predict(X_val)
        
        # Calculate metrics
        auc_roc = roc_auc_score(y_val, y_pred_proba)
        auc_pr = average_precision_score(y_val, y_pred_proba)
        f1 = f1_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        
        return {
            'auc_roc': auc_roc,
            'auc_pr': auc_pr,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'y_pred_proba': y_pred_proba,
            'y_pred': y_pred,
            'y_true': y_val
        }
    
    def run_cross_validation(self, algorithm_name, model, param_grid=None, n_iter=20):
        """Run cross-validation for a single algorithm"""
        
        print(f"\\n🔄 Running {algorithm_name}...")
        
        cv_scores = []
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(tqdm(self.cv.split(self.embeddings, self.y), 
                                                         desc=f"{algorithm_name} CV")):
            
            X_train, X_val = self.embeddings[train_idx], self.embeddings[val_idx]
            y_train, y_val = self.y[train_idx], self.y[val_idx]
            
            # Hyperparameter tuning if param_grid provided
            if param_grid is not None:
                random_search = RandomizedSearchCV(
                    model, param_grid, n_iter=n_iter, cv=3, 
                    random_state=42, n_jobs=-1, scoring='roc_auc'
                )
                random_search.fit(X_train, y_train)
                best_model = random_search.best_estimator_
            else:
                best_model = model
            
            # Evaluate model
            fold_result = self.evaluate_model(best_model, X_train, X_val, y_train, y_val)
            fold_results.append(fold_result)
            cv_scores.append(fold_result['auc_pr'])  # Use AUC-PR as primary metric
        
        # Aggregate results
        mean_auc_pr = np.mean(cv_scores)
        std_auc_pr = np.std(cv_scores)
        
        # Calculate mean metrics across folds
        metrics = ['auc_roc', 'auc_pr', 'f1', 'precision', 'recall']
        mean_metrics = {}
        std_metrics = {}
        
        for metric in metrics:
            values = [fold_result[metric] for fold_result in fold_results]
            mean_metrics[metric] = np.mean(values)
            std_metrics[metric] = np.std(values)
        
        result = {
            'mean_auc_pr': mean_auc_pr,
            'std_auc_pr': std_auc_pr,
            'mean_metrics': mean_metrics,
            'std_metrics': std_metrics,
            'fold_results': fold_results
        }
        
        print(f"✅ {algorithm_name} completed - AUC-PR: {mean_auc_pr:.3f} ± {std_auc_pr:.3f}")
        
        return result
    
    def run_benchmark(self):
        """Run the complete benchmark"""
        
        print("🚀 ENT-KAURENE BINARY CLASSIFIER BENCHMARK")
        print("=" * 55)
        print(f"Algorithms: 7")
        print(f"Cross-validation: 5-fold stratified")
        print(f"Primary metric: AUC-PR")
        print(f"Dataset: {len(self.df)} sequences")
        print()
        
        algorithms = self.define_algorithms()
        
        # Run each algorithm
        for name, config in algorithms.items():
            start_time = time.time()
            
            result = self.run_cross_validation(
                name, 
                config['model'], 
                config['param_grid']
            )
            
            end_time = time.time()
            result['training_time'] = end_time - start_time
            
            self.results[name] = result
        
        print("\\n🎉 Benchmark completed!")
        
    def generate_report(self):
        """Generate comprehensive performance report"""
        
        print("\\n📊 PERFORMANCE REPORT")
        print("=" * 30)
        
        # Create results summary
        summary_data = []
        
        for name, result in self.results.items():
            summary_data.append({
                'Algorithm': name,
                'AUC-PR (mean)': f"{result['mean_auc_pr']:.3f}",
                'AUC-PR (std)': f"± {result['std_auc_pr']:.3f}",
                'F1-Score': f"{result['mean_metrics']['f1']:.3f}",
                'Precision': f"{result['mean_metrics']['precision']:.3f}",
                'Recall': f"{result['mean_metrics']['recall']:.3f}",
                'Training Time': f"{result['training_time']:.1f}s"
            })
        
        # Sort by AUC-PR
        summary_data.sort(key=lambda x: float(x['AUC-PR (mean)']), reverse=True)
        
        # Display results
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        
        # Find best algorithm
        best_algorithm = summary_data[0]['Algorithm']
        best_auc_pr = float(summary_data[0]['AUC-PR (mean)'])
        
        print(f"\\n🏆 BEST ALGORITHM: {best_algorithm}")
        print(f"   AUC-PR: {best_auc_pr:.3f}")
        
        # Save detailed results
        self.save_results()
        
        return summary_df
    
    def save_results(self):
        """Save detailed results to files"""
        
        # Save summary results
        summary_data = []
        for name, result in self.results.items():
            summary_data.append({
                'algorithm': name,
                'mean_auc_pr': result['mean_auc_pr'],
                'std_auc_pr': result['std_auc_pr'],
                'mean_auc_roc': result['mean_metrics']['auc_roc'],
                'std_auc_roc': result['std_metrics']['auc_roc'],
                'mean_f1': result['mean_metrics']['f1'],
                'std_f1': result['std_metrics']['f1'],
                'mean_precision': result['mean_metrics']['precision'],
                'std_precision': result['std_metrics']['precision'],
                'mean_recall': result['mean_metrics']['recall'],
                'std_recall': result['std_metrics']['recall'],
                'training_time': result['training_time']
            })
        
        # Save to CSV
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv('results/benchmark_summary.csv', index=False)
        
        # Save detailed results as JSON
        with open('results/benchmark_detailed.json', 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = {}
            for name, result in self.results.items():
                json_results[name] = {
                    'mean_auc_pr': float(result['mean_auc_pr']),
                    'std_auc_pr': float(result['std_auc_pr']),
                    'mean_metrics': {k: float(v) for k, v in result['mean_metrics'].items()},
                    'std_metrics': {k: float(v) for k, v in result['std_metrics'].items()},
                    'training_time': float(result['training_time'])
                }
            json.dump(json_results, f, indent=2)
        
        print(f"\\n💾 Results saved:")
        print(f"   📊 Summary: results/benchmark_summary.csv")
        print(f"   📋 Detailed: results/benchmark_detailed.json")

def main():
    """Main benchmark execution"""
    
    # Initialize benchmark
    benchmark = EntKaureneBenchmark()
    
    # Run benchmark
    benchmark.run_benchmark()
    
    # Generate report
    summary_df = benchmark.generate_report()
    
    print("\\n🎯 BENCHMARK COMPLETED SUCCESSFULLY!")
    print("   Ready for model selection and deployment!")

if __name__ == "__main__":
    main()
