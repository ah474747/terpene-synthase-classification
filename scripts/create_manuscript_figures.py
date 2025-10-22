#!/usr/bin/env python3
"""
Create manuscript figures for terpene synthase classification study
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def create_figure1_algorithm_comparison():
    """Figure 1: Machine Learning Algorithm Performance Comparison"""
    
    # Load results
    with open('results/germacrene_benchmark_results.json', 'r') as f:
        germacrene_results = json.load(f)
    with open('results/pinene_benchmark_results.json', 'r') as f:
        pinene_results = json.load(f)
    with open('results/myrcene_benchmark_results.json', 'r') as f:
        myrcene_results = json.load(f)
    
    # Prepare data
    algorithms = []
    germacrene_f1 = []
    pinene_f1 = []
    myrcene_f1 = []
    
    for model_name in ['XGBoost', 'Random_Forest', 'SVM_RBF', 'Logistic_Regression', 'MLP', 'KNN', 'Perceptron']:
        algorithms.append(model_name.replace('_', ' '))
        germacrene_f1.append(germacrene_results[model_name]['cv_scores']['f1']['mean'])
        pinene_f1.append(pinene_results[model_name]['cv_scores']['f1']['mean'])
        myrcene_f1.append(myrcene_results[model_name]['cv_scores']['f1']['mean'])
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    x = np.arange(len(algorithms))
    width = 0.25
    
    bars1 = ax.bar(x - width, germacrene_f1, width, label='Germacrene (7.4%)', alpha=0.8)
    bars2 = ax.bar(x, pinene_f1, width, label='Pinene (6.5%)', alpha=0.8)
    bars3 = ax.bar(x + width, myrcene_f1, width, label='Myrcene (4.2%)', alpha=0.8)
    
    ax.set_xlabel('Machine Learning Algorithm', fontsize=12)
    ax.set_ylabel('F1-Score', fontsize=12)
    ax.set_title('Machine Learning Algorithm Performance Across Target Products', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 0.8)
    
    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Add labels only to best performing bars for clarity
    for i, (g, p, m) in enumerate(zip(germacrene_f1, pinene_f1, myrcene_f1)):
        max_val = max(g, p, m)
        if max_val == g:
            ax.text(bars1[i].get_x() + bars1[i].get_width()/2., g + 0.02,
                   f'{g:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        elif max_val == p:
            ax.text(bars2[i].get_x() + bars2[i].get_width()/2., p + 0.02,
                   f'{p:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        else:
            ax.text(bars3[i].get_x() + bars3[i].get_width()/2., m + 0.02,
                   f'{m:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/figure1_algorithm_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_figure2_traditional_vs_ml():
    """Figure 2: Traditional Methods vs ESM-2 + ML Performance (Germacrene Only)"""
    
    # Data for traditional vs ML comparison (Germacrene only)
    methods = ['ESM-2 + SVM-RBF', 'Sequence\nSimilarity', 'AA\nComposition', 'Length-\nbased', 'Motif-\nbased']
    germacrene_f1 = [0.591, 0.449, 0.347, 0.307, 0.139]
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    x = np.arange(len(methods))
    
    # Create bars with different colors
    colors = ['#32CD32', '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    bars = ax.bar(x, germacrene_f1, alpha=0.8, color=colors)
    
    ax.set_xlabel('Classification Method', fontsize=12)
    ax.set_ylabel('F1-Score', fontsize=12)
    ax.set_title('ESM-2 + ML vs Traditional Methods Performance\n(Germacrene Classification)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 0.7)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, germacrene_f1)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{value:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add improvement percentages for traditional methods
    for i in range(1, len(methods)):
        improvement = ((germacrene_f1[0] - germacrene_f1[i]) / germacrene_f1[i]) * 100
        ax.text(i, germacrene_f1[i] + 0.05,
               f'+{improvement:.0f}%', ha='center', va='bottom', fontsize=8, fontweight='bold', color='red')
    
    plt.tight_layout()
    plt.savefig('results/figure2_traditional_vs_ml.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_figure3_class_balance_impact():
    """Figure 3: Impact of Class Balance on Performance"""
    
    # Data
    products = ['Germacrene', 'Pinene', 'Myrcene']
    class_balance = [7.4, 6.5, 4.2]
    best_f1_scores = [0.591, 0.663, 0.439]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Class balance vs performance
    ax1.scatter(class_balance, best_f1_scores, s=200, alpha=0.7, c=['#2E8B57', '#4169E1', '#DC143C'])
    ax1.plot(class_balance, best_f1_scores, '--', alpha=0.5, color='gray')
    
    for i, product in enumerate(products):
        ax1.annotate(f'{product}\n{class_balance[i]}%', 
                    (class_balance[i], best_f1_scores[i]),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                    fontsize=10)
    
    ax1.set_xlabel('Class Balance (%)', fontsize=12)
    ax1.set_ylabel('Best F1-Score', fontsize=12)
    ax1.set_title('Class Balance Impact on Performance', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Dataset composition
    sizes = [93, 82, 53, 1034]  # germacrene, pinene, myrcene, other
    labels = ['Germacrene\n(93)', 'Pinene\n(82)', 'Myrcene\n(53)', 'Other\n(1034)']
    colors = ['#2E8B57', '#4169E1', '#DC143C', '#D3D3D3']
    
    wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Dataset Composition (1,262 sequences)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/figure3_class_balance_impact.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_figure4_holdout_validation():
    """Figure 4: Hold-out Validation Results"""
    
    # Load hold-out results
    with open('results/germacrene_holdout_validation_results.json', 'r') as f:
        holdout_results = json.load(f)
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'AUC-PR']
    values = [
        holdout_results['holdout_validation']['metrics']['accuracy'],
        holdout_results['holdout_validation']['metrics']['precision'],
        holdout_results['holdout_validation']['metrics']['recall'],
        holdout_results['holdout_validation']['metrics']['f1_score'],
        holdout_results['holdout_validation']['metrics']['auc_roc'],
        holdout_results['holdout_validation']['metrics']['auc_pr']
    ]
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    bars = ax.bar(metrics, values, alpha=0.7, color=['#2E8B57', '#4169E1', '#DC143C', '#FF8C00', '#8A2BE2', '#20B2AA'])
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Hold-out Validation Results (Germacrene Classification)', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
               f'{value:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('results/figure4_holdout_validation.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print('🎨 CREATING MANUSCRIPT FIGURES')
    print('=' * 35)
    
    # Ensure results directory exists
    Path('results').mkdir(exist_ok=True)
    
    print('📊 Creating Figure 1: Algorithm Performance Comparison...')
    create_figure1_algorithm_comparison()
    
    print('📊 Creating Figure 2: Traditional vs ML Methods...')
    create_figure2_traditional_vs_ml()
    
    print('📊 Creating Figure 3: Class Balance Impact...')
    create_figure3_class_balance_impact()
    
    print('📊 Creating Figure 4: Hold-out Validation...')
    create_figure4_holdout_validation()
    
    print('✅ All figures created successfully!')
    print('📁 Figures saved in results/ directory:')
    print('   - figure1_algorithm_comparison.png')
    print('   - figure2_traditional_vs_ml.png')
    print('   - figure3_class_balance_impact.png')
    print('   - figure4_holdout_validation.png')

if __name__ == "__main__":
    main()
