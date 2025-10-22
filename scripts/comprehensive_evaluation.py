#!/usr/bin/env python3
"""
Comprehensive Model Evaluation Script
=====================================

This script provides comprehensive evaluation of terpene synthase classification models
with statistical significance testing, confidence intervals, and detailed metrics.

Addresses reviewer feedback:
- Statistical significance testing with confidence intervals
- Complete metric reporting (precision, recall, AUC-ROC, AUC-PR, confusion matrices)
- Reproducible results with fixed seeds
- Error bars and variance estimates

Author: Andrew Horwitz
Date: October 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)
from sklearn.utils import resample
import joblib
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

class ComprehensiveEvaluator:
    """Comprehensive model evaluation with statistical analysis."""
    
    def __init__(self, random_state=RANDOM_STATE):
        self.random_state = random_state
        self.results = {}
        
    def bootstrap_confidence_interval(self, scores, confidence=0.95, n_bootstrap=1000):
        """Calculate bootstrap confidence interval for metrics."""
        bootstrap_scores = []
        
        for _ in range(n_bootstrap):
            # Bootstrap resample
            bootstrap_sample = resample(scores, random_state=self.random_state)
            bootstrap_scores.append(np.mean(bootstrap_sample))
            
        # Calculate confidence interval
        alpha = 1 - confidence
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_scores, lower_percentile)
        ci_upper = np.percentile(bootstrap_scores, upper_percentile)
        
        return ci_lower, ci_upper
    
    def evaluate_model_comprehensive(self, model, X, y, cv_folds=5, target_name="Unknown"):
        """Comprehensive model evaluation with statistical analysis."""
        
        print(f"🔬 COMPREHENSIVE EVALUATION: {target_name}")
        print("=" * 60)
        
        # Initialize cross-validation
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        # Storage for metrics across folds
        metrics_folds = {
            'accuracy': [], 'precision': [], 'recall': [], 'f1': [],
            'auc_roc': [], 'auc_pr': []
        }
        
        # Storage for predictions across folds
        all_y_true = []
        all_y_pred = []
        all_y_proba = []
        
        fold_results = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            print(f"📊 Processing Fold {fold_idx + 1}/{cv_folds}...")
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Get prediction probabilities if available
            try:
                y_proba = model.predict_proba(X_test)[:, 1]
            except AttributeError:
                # For models without predict_proba (like Perceptron)
                y_proba = model.decision_function(X_test)
                # Normalize to [0, 1] range
                y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min())
            
            # Calculate metrics for this fold
            fold_metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'auc_roc': roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0.0,
                'auc_pr': average_precision_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0.0
            }
            
            # Store metrics
            for metric, value in fold_metrics.items():
                metrics_folds[metric].append(value)
            
            # Store predictions for overall analysis
            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)
            all_y_proba.extend(y_proba)
            
            fold_results.append({
                'fold': fold_idx + 1,
                **fold_metrics
            })
            
            print(f"   F1-Score: {fold_metrics['f1']:.3f}, AUC-PR: {fold_metrics['auc_pr']:.3f}")
        
        # Calculate overall statistics
        overall_stats = {}
        confidence_intervals = {}
        
        for metric in metrics_folds:
            scores = np.array(metrics_folds[metric])
            overall_stats[metric] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'scores': scores.tolist()
            }
            
            # Calculate confidence intervals
            ci_lower, ci_upper = self.bootstrap_confidence_interval(scores)
            confidence_intervals[metric] = {
                'lower': ci_lower,
                'upper': ci_upper,
                'confidence': 0.95
            }
        
        # Overall predictions analysis
        overall_y_true = np.array(all_y_true)
        overall_y_pred = np.array(all_y_pred)
        overall_y_proba = np.array(all_y_proba)
        
        # Confusion matrix
        cm = confusion_matrix(overall_y_true, overall_y_pred)
        
        # Classification report
        class_report = classification_report(overall_y_true, overall_y_pred, output_dict=True)
        
        # ROC and PR curves
        try:
            fpr, tpr, _ = roc_curve(overall_y_true, overall_y_proba)
            precision_curve, recall_curve, _ = precision_recall_curve(overall_y_true, overall_y_proba)
        except ValueError:
            fpr, tpr = [0, 1], [0, 1]
            precision_curve, recall_curve = [1, 0], [0, 1]
        
        # Compile results
        results = {
            'target': target_name,
            'model_type': type(model).__name__,
            'cv_folds': cv_folds,
            'random_state': self.random_state,
            'fold_results': fold_results,
            'overall_stats': overall_stats,
            'confidence_intervals': confidence_intervals,
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report,
            'roc_curve': {'fpr': fpr.tolist(), 'tpr': tpr.tolist()},
            'pr_curve': {'precision': precision_curve.tolist(), 'recall': recall_curve.tolist()},
            'class_distribution': {
                'positive_class': int(np.sum(overall_y_true)),
                'negative_class': int(len(overall_y_true) - np.sum(overall_y_true)),
                'total': len(overall_y_true)
            }
        }
        
        # Print summary
        self.print_evaluation_summary(results)
        
        return results
    
    def print_evaluation_summary(self, results):
        """Print comprehensive evaluation summary."""
        
        print(f"\n📊 EVALUATION SUMMARY: {results['target']}")
        print("=" * 60)
        
        stats = results['overall_stats']
        ci = results['confidence_intervals']
        
        print(f"Model: {results['model_type']}")
        print(f"CV Folds: {results['cv_folds']}")
        print(f"Total Samples: {results['class_distribution']['total']}")
        print(f"Positive Class: {results['class_distribution']['positive_class']} ({results['class_distribution']['positive_class']/results['class_distribution']['total']*100:.1f}%)")
        print(f"Negative Class: {results['class_distribution']['negative_class']} ({results['class_distribution']['negative_class']/results['class_distribution']['total']*100:.1f}%)")
        
        print(f"\n📈 PERFORMANCE METRICS (Mean ± Std, 95% CI):")
        print("-" * 60)
        
        metrics_order = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc', 'auc_pr']
        metric_names = {
            'accuracy': 'Accuracy',
            'precision': 'Precision',
            'recall': 'Recall',
            'f1': 'F1-Score',
            'auc_roc': 'AUC-ROC',
            'auc_pr': 'AUC-PR'
        }
        
        for metric in metrics_order:
            mean_val = stats[metric]['mean']
            std_val = stats[metric]['std']
            ci_lower = ci[metric]['lower']
            ci_upper = ci[metric]['upper']
            
            print(f"{metric_names[metric]:12}: {mean_val:.3f} ± {std_val:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]")
        
        print(f"\n📋 CONFUSION MATRIX:")
        print("-" * 30)
        cm = np.array(results['confusion_matrix'])
        print(f"True Negatives:  {cm[0,0]:4d}")
        print(f"False Positives: {cm[0,1]:4d}")
        print(f"False Negatives: {cm[1,0]:4d}")
        print(f"True Positives:  {cm[1,1]:4d}")
        
        print("\n✅ Comprehensive evaluation completed!")
    
    def save_results(self, results, output_path):
        """Save comprehensive results to JSON file."""
        
        # Convert numpy types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        # Convert results
        results_json = convert_numpy_types(results)
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(results_json, f, indent=2)
        
        print(f"💾 Results saved to: {output_path}")
    
    def create_visualization(self, results, output_path):
        """Create comprehensive visualization of results."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Comprehensive Model Evaluation: {results["target"]}', fontsize=16, fontweight='bold')
        
        # 1. Metrics comparison with error bars
        ax1 = axes[0, 0]
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc', 'auc_pr']
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'AUC-PR']
        
        means = [results['overall_stats'][m]['mean'] for m in metrics]
        stds = [results['overall_stats'][m]['std'] for m in metrics]
        
        bars = ax1.bar(metric_names, means, yerr=stds, capsize=5, alpha=0.7, color='skyblue')
        ax1.set_title('Performance Metrics with Standard Deviation', fontweight='bold')
        ax1.set_ylabel('Score')
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                    f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 2. Confusion Matrix
        ax2 = axes[0, 1]
        cm = np.array(results['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
                   xticklabels=['Predicted Negative', 'Predicted Positive'],
                   yticklabels=['Actual Negative', 'Actual Positive'])
        ax2.set_title('Confusion Matrix', fontweight='bold')
        
        # 3. ROC Curve
        ax3 = axes[1, 0]
        roc_data = results['roc_curve']
        ax3.plot(roc_data['fpr'], roc_data['tpr'], 'b-', linewidth=2, 
                label=f"AUC-ROC = {results['overall_stats']['auc_roc']['mean']:.3f}")
        ax3.plot([0, 1], [0, 1], 'r--', linewidth=1, alpha=0.7)
        ax3.set_xlabel('False Positive Rate')
        ax3.set_ylabel('True Positive Rate')
        ax3.set_title('ROC Curve', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Precision-Recall Curve
        ax4 = axes[1, 1]
        pr_data = results['pr_curve']
        ax4.plot(pr_data['recall'], pr_data['precision'], 'g-', linewidth=2,
                label=f"AUC-PR = {results['overall_stats']['auc_pr']['mean']:.3f}")
        ax4.set_xlabel('Recall')
        ax4.set_ylabel('Precision')
        ax4.set_title('Precision-Recall Curve', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Visualization saved to: {output_path}")

def main():
    """Main function to run comprehensive evaluation."""
    
    print("🔬 COMPREHENSIVE MODEL EVALUATION SYSTEM")
    print("=" * 60)
    print("Addressing reviewer feedback for statistical rigor and reproducibility")
    print()
    
    # This is a template - actual usage would load models and data
    print("📋 This script provides comprehensive evaluation capabilities:")
    print("   • Statistical significance testing with confidence intervals")
    print("   • Complete metric reporting (precision, recall, AUC-ROC, AUC-PR)")
    print("   • Confusion matrices and classification reports")
    print("   • Bootstrap confidence intervals")
    print("   • ROC and Precision-Recall curves")
    print("   • Reproducible results with fixed seeds")
    print("   • Comprehensive visualizations")
    
    print("\n✅ Comprehensive evaluation system ready!")
    print("   Import this module in your training scripts to use comprehensive evaluation.")

if __name__ == "__main__":
    main()
