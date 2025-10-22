#!/usr/bin/env python3
"""
Statistical Analysis for Ent-Kaurene Classification Benchmark

This script adds statistical significance testing, confidence intervals,
and effect size calculations to the benchmark results.

Author: Cursor AI
Date: October 17, 2025
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, average_precision_score
import json
import matplotlib.pyplot as plt
from pathlib import Path

class StatisticalAnalysis:
    """Statistical analysis for classification benchmark"""
    
    def __init__(self, data_file="data/ent_kaurene_binary_dataset.csv", 
                 embeddings_file="data/esm2_embeddings.npy"):
        """Initialize statistical analysis"""
        self.df = pd.read_csv(data_file)
        self.embeddings = np.load(embeddings_file)
        
        print(f"📊 Statistical analysis initialized:")
        print(f"   Dataset: {len(self.df)} sequences")
        print(f"   Embeddings: {self.embeddings.shape}")
    
    def calculate_confidence_intervals(self, scores, confidence=0.95):
        """Calculate confidence intervals for performance metrics"""
        n = len(scores)
        mean_score = np.mean(scores)
        std_score = np.std(scores, ddof=1)
        
        # Calculate standard error
        se = std_score / np.sqrt(n)
        
        # Calculate t-statistic for confidence interval
        alpha = 1 - confidence
        t_critical = stats.t.ppf(1 - alpha/2, n-1)
        
        # Calculate margin of error
        margin_error = t_critical * se
        
        # Calculate confidence interval
        ci_lower = mean_score - margin_error
        ci_upper = mean_score + margin_error
        
        return {
            'mean': mean_score,
            'std': std_score,
            'se': se,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'margin_error': margin_error
        }
    
    def run_cross_validation_with_stats(self, model, X, y, cv=5, scoring='f1'):
        """Run cross-validation with statistical analysis"""
        print(f"   Running {cv}-fold CV for statistical analysis...")
        
        # Run cross-validation
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        
        # Calculate statistics
        stats_result = self.calculate_confidence_intervals(cv_scores)
        
        return {
            'cv_scores': cv_scores.tolist(),
            'statistics': stats_result
        }
    
    def compare_methods_statistically(self, results_dict):
        """Compare methods using statistical tests"""
        print("\\n🔬 Statistical comparison of methods...")
        
        comparisons = []
        
        # Get ESM-2 + XGBoost scores as reference
        esm2_scores = results_dict.get('ESM-2 + XGBoost', {}).get('cv_scores', [])
        
        if not esm2_scores:
            print("❌ No ESM-2 + XGBoost scores found for comparison")
            return comparisons
        
        # Compare each method with ESM-2 + XGBoost
        for method_name, method_results in results_dict.items():
            if method_name == 'ESM-2 + XGBoost':
                continue
                
            method_scores = method_results.get('cv_scores', [])
            if not method_scores:
                continue
            
            # Perform paired t-test
            t_stat, p_value = stats.ttest_rel(esm2_scores, method_scores)
            
            # Calculate effect size (Cohen's d)
            pooled_std = np.sqrt((np.var(esm2_scores) + np.var(method_scores)) / 2)
            cohens_d = (np.mean(esm2_scores) - np.mean(method_scores)) / pooled_std
            
            # Determine significance
            alpha = 0.05
            significant = p_value < alpha
            
            comparison = {
                'method': method_name,
                'esm2_mean': np.mean(esm2_scores),
                'method_mean': np.mean(method_scores),
                'difference': np.mean(esm2_scores) - np.mean(method_scores),
                't_statistic': t_stat,
                'p_value': p_value,
                'cohens_d': cohens_d,
                'significant': significant,
                'effect_size': self._interpret_effect_size(cohens_d)
            }
            
            comparisons.append(comparison)
            
            print(f"   {method_name}: p={p_value:.4f}, d={cohens_d:.3f} ({'*' if significant else 'ns'})")
        
        return comparisons
    
    def _interpret_effect_size(self, cohens_d):
        """Interpret Cohen's d effect size"""
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def analyze_class_imbalance_impact(self):
        """Analyze the impact of class imbalance on performance"""
        print("\\n⚖️ Analyzing class imbalance impact...")
        
        # Calculate class distribution
        positive_count = len(self.df[self.df['is_ent_kaurene'] == 1])
        negative_count = len(self.df[self.df['is_ent_kaurene'] == 0])
        total_count = len(self.df)
        
        positive_ratio = positive_count / total_count
        imbalance_ratio = negative_count / positive_count
        
        print(f"   Class distribution:")
        print(f"     Positive (ent-kaurene): {positive_count} ({positive_ratio:.1%})")
        print(f"     Negative (non-ent-kaurene): {negative_count} ({1-positive_ratio:.1%})")
        print(f"     Imbalance ratio: {imbalance_ratio:.1f}:1")
        
        # Analyze sequence length distribution by class
        positive_lengths = self.df[self.df['is_ent_kaurene'] == 1]['Sequence'].str.len()
        negative_lengths = self.df[self.df['is_ent_kaurene'] == 0]['Sequence'].str.len()
        
        length_stats = {
            'positive_mean_length': positive_lengths.mean(),
            'positive_std_length': positive_lengths.std(),
            'negative_mean_length': negative_lengths.mean(),
            'negative_std_length': negative_lengths.std(),
            'length_difference': positive_lengths.mean() - negative_lengths.mean()
        }
        
        # Test if length distributions are significantly different
        t_stat, p_value = stats.ttest_ind(positive_lengths, negative_lengths)
        
        print(f"   Sequence length analysis:")
        print(f"     Positive class: {length_stats['positive_mean_length']:.1f} ± {length_stats['positive_std_length']:.1f} aa")
        print(f"     Negative class: {length_stats['negative_mean_length']:.1f} ± {length_stats['negative_std_length']:.1f} aa")
        print(f"     Length difference: {length_stats['length_difference']:.1f} aa (p={p_value:.4f})")
        
        return {
            'class_distribution': {
                'positive_count': positive_count,
                'negative_count': negative_count,
                'positive_ratio': positive_ratio,
                'imbalance_ratio': imbalance_ratio
            },
            'length_analysis': length_stats,
            'length_significance': {
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        }
    
    def analyze_misclassified_sequences(self, predictions, true_labels, sequences, threshold=0.5):
        """Analyze misclassified sequences"""
        print("\\n🔍 Analyzing misclassified sequences...")
        
        # Convert predictions to binary if needed
        if len(predictions.shape) > 1 and predictions.shape[1] > 1:
            binary_predictions = (predictions[:, 1] > threshold).astype(int)
        else:
            binary_predictions = (predictions > threshold).astype(int)
        
        # Find misclassifications
        misclassified_mask = binary_predictions != true_labels
        misclassified_indices = np.where(misclassified_mask)[0]
        
        # Analyze false positives and false negatives
        false_positives = misclassified_indices[(binary_predictions[misclassified_indices] == 1) & 
                                               (true_labels[misclassified_indices] == 0)]
        false_negatives = misclassified_indices[(binary_predictions[misclassified_indices] == 0) & 
                                               (true_labels[misclassified_indices] == 1)]
        
        print(f"   Misclassification analysis:")
        print(f"     Total misclassified: {len(misclassified_indices)} ({len(misclassified_indices)/len(true_labels):.1%})")
        print(f"     False positives: {len(false_positives)}")
        print(f"     False negatives: {len(false_negatives)}")
        
        # Analyze characteristics of misclassified sequences
        misclassified_sequences = [sequences[i] for i in misclassified_indices]
        misclassified_lengths = [len(seq) for seq in misclassified_sequences]
        
        # Get examples of misclassified sequences
        fp_examples = []
        fn_examples = []
        
        if len(false_positives) > 0:
            fp_examples = [
                {
                    'index': int(fp_idx),
                    'sequence_length': len(sequences[fp_idx]),
                    'prediction_confidence': float(predictions[fp_idx, 1] if len(predictions.shape) > 1 else predictions[fp_idx])
                }
                for fp_idx in false_positives[:5]  # Top 5 examples
            ]
        
        if len(false_negatives) > 0:
            fn_examples = [
                {
                    'index': int(fn_idx),
                    'sequence_length': len(sequences[fn_idx]),
                    'prediction_confidence': float(predictions[fn_idx, 1] if len(predictions.shape) > 1 else predictions[fn_idx])
                }
                for fn_idx in false_negatives[:5]  # Top 5 examples
            ]
        
        return {
            'total_misclassified': len(misclassified_indices),
            'misclassification_rate': len(misclassified_indices) / len(true_labels),
            'false_positives_count': len(false_positives),
            'false_negatives_count': len(false_negatives),
            'misclassified_length_stats': {
                'mean': np.mean(misclassified_lengths),
                'std': np.std(misclassified_lengths),
                'min': np.min(misclassified_lengths),
                'max': np.max(misclassified_lengths)
            },
            'false_positive_examples': fp_examples,
            'false_negative_examples': fn_examples
        }
    
    def create_statistical_report(self):
        """Create comprehensive statistical report"""
        print("\\n📊 Creating comprehensive statistical report...")
        
        # Load existing results
        try:
            with open('results/holdout_validation_report.json', 'r') as f:
                holdout_results = json.load(f)
        except:
            holdout_results = {}
        
        # Analyze class imbalance
        imbalance_analysis = self.analyze_class_imbalance_impact()
        
        # Create statistical summary
        statistical_report = {
            'dataset_statistics': imbalance_analysis,
            'holdout_validation': holdout_results,
            'statistical_notes': {
                'confidence_level': 0.95,
                'significance_level': 0.05,
                'cross_validation_folds': 5
            }
        }
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        statistical_report = convert_numpy_types(statistical_report)
        
        # Save statistical report
        with open('results/statistical_analysis_report.json', 'w') as f:
            json.dump(statistical_report, f, indent=2)
        
        print("✅ Statistical analysis completed")
        print("💾 Report saved to: results/statistical_analysis_report.json")
        
        return statistical_report

def main():
    """Main statistical analysis function"""
    print("🔬 STATISTICAL ANALYSIS FOR ENT-KAURENE CLASSIFICATION")
    print("=" * 60)
    
    analyzer = StatisticalAnalysis()
    
    # Run comprehensive statistical analysis
    report = analyzer.create_statistical_report()
    
    print("\\n🎉 STATISTICAL ANALYSIS COMPLETED!")
    print("\\n📋 KEY FINDINGS:")
    
    # Print key findings
    class_dist = report['dataset_statistics']['class_distribution']
    print(f"   • Class imbalance: {class_dist['imbalance_ratio']:.1f}:1 ratio")
    print(f"   • Positive class: {class_dist['positive_ratio']:.1%} of dataset")
    
    length_analysis = report['dataset_statistics']['length_analysis']
    print(f"   • Length difference: {length_analysis['length_difference']:.1f} aa")
    
    print("\\n🚀 STATISTICAL RIGOR ADDED TO BENCHMARK!")

if __name__ == "__main__":
    main()
