#!/usr/bin/env python3
"""
Traditional Methods Benchmark for Germacrene Classification
Compare ML methods against sequence-based approaches
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from Bio import pairwise2
from Bio.Seq import Seq
from collections import Counter
import json
import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def sequence_similarity_classification(X_seqs, y, cv):
    """Sequence similarity-based classification"""
    results = []
    
    for train_idx, test_idx in cv.split(X_seqs, y):
        X_train, X_test = X_seqs[train_idx], X_seqs[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        y_pred = []
        for test_seq in X_test:
            similarities = []
            for train_seq in X_train:
                # Simple sequence identity
                identity = sum(a == b for a, b in zip(test_seq, train_seq)) / len(test_seq)
                similarities.append(identity)
            
            # Predict based on most similar sequence
            most_similar_idx = np.argmax(similarities)
            y_pred.append(y_train[most_similar_idx])
        
        results.append({
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0)
        })
    
    return results

def motif_based_classification(X_seqs, y, cv):
    """Motif-based classification using common patterns"""
    
    # Define common terpene synthase motifs
    motifs = [
        'DDXXD',  # Metal binding motif
        'NSE/DTE',  # Protonation site
        'RRX8W',  # Plant terpene synthase motif
        'GXGXG',  # Glycine-rich region
    ]
    
    results = []
    
    for train_idx, test_idx in cv.split(X_seqs, y):
        X_train, X_test = X_seqs[train_idx], X_seqs[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Count motifs in training sequences
        train_motif_counts = []
        for seq in X_train:
            counts = [seq.count(motif.replace('X', '').replace('/', '')) for motif in motifs]
            train_motif_counts.append(counts)
        
        train_motif_counts = np.array(train_motif_counts)
        
        # Count motifs in test sequences
        test_motif_counts = []
        for seq in X_test:
            counts = [seq.count(motif.replace('X', '').replace('/', '')) for motif in motifs]
            test_motif_counts.append(counts)
        
        test_motif_counts = np.array(test_motif_counts)
        
        # Use motif counts as features for classification
        rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        rf.fit(train_motif_counts, y_train)
        y_pred = rf.predict(test_motif_counts)
        
        results.append({
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0)
        })
    
    return results

def length_based_classification(X_seqs, y, cv):
    """Length-based classification"""
    results = []
    
    for train_idx, test_idx in cv.split(X_seqs, y):
        X_train, X_test = X_seqs[train_idx], X_seqs[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Get sequence lengths
        train_lengths = np.array([[len(seq)] for seq in X_train])
        test_lengths = np.array([[len(seq)] for seq in X_test])
        
        # Use length as feature
        rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        rf.fit(train_lengths, y_train)
        y_pred = rf.predict(test_lengths)
        
        results.append({
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0)
        })
    
    return results

def aa_composition_classification(X_seqs, y, cv):
    """Amino acid composition-based classification"""
    results = []
    
    def get_aa_composition(seq):
        aa_counts = Counter(seq)
        total = len(seq)
        composition = [aa_counts.get(aa, 0) / total for aa in 'ACDEFGHIKLMNPQRSTVWY']
        return composition
    
    for train_idx, test_idx in cv.split(X_seqs, y):
        X_train, X_test = X_seqs[train_idx], X_seqs[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Get AA compositions
        train_compositions = np.array([get_aa_composition(seq) for seq in X_train])
        test_compositions = np.array([get_aa_composition(seq) for seq in X_test])
        
        # Use AA composition as features
        rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        rf.fit(train_compositions, y_train)
        y_pred = rf.predict(test_compositions)
        
        results.append({
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0)
        })
    
    return results

def main():
    print('🔬 TRADITIONAL METHODS BENCHMARK FOR GERMACRENE CLASSIFICATION')
    print('=' * 65)
    
    # Load data
    print('📂 Loading dataset and embeddings...')
    df = pd.read_csv('data/clean_MARTS_DB_binary_dataset.csv')
    embeddings = np.load('data/germacrene_esm2_embeddings.npy')
    
    X_embeddings = embeddings
    X_seqs = df['Aminoacid_sequence'].values
    y = df['is_germacrene'].values
    
    print(f'📊 Dataset loaded:')
    print(f'   Total sequences: {len(df)}')
    print(f'   Germacrene sequences: {df["is_germacrene"].sum()} ({df["is_germacrene"].mean()*100:.1f}%)')
    
    # Cross-validation setup
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Traditional methods
    print(f'\n🚀 RUNNING TRADITIONAL METHODS BENCHMARK')
    
    traditional_results = {}
    
    # 1. Sequence Similarity
    print(f'\n🔍 Sequence Similarity Classification...')
    start_time = time.time()
    seq_sim_results = sequence_similarity_classification(X_seqs, y, cv)
    traditional_results['Sequence_Similarity'] = {
        'accuracy': np.mean([r['accuracy'] for r in seq_sim_results]),
        'precision': np.mean([r['precision'] for r in seq_sim_results]),
        'recall': np.mean([r['recall'] for r in seq_sim_results]),
        'f1': np.mean([r['f1'] for r in seq_sim_results]),
        'training_time': time.time() - start_time
    }
    print(f'   F1-Score: {traditional_results["Sequence_Similarity"]["f1"]:.3f}')
    
    # 2. Motif-based
    print(f'\n🔍 Motif-based Classification...')
    start_time = time.time()
    motif_results = motif_based_classification(X_seqs, y, cv)
    traditional_results['Motif_Based'] = {
        'accuracy': np.mean([r['accuracy'] for r in motif_results]),
        'precision': np.mean([r['precision'] for r in motif_results]),
        'recall': np.mean([r['recall'] for r in motif_results]),
        'f1': np.mean([r['f1'] for r in motif_results]),
        'training_time': time.time() - start_time
    }
    print(f'   F1-Score: {traditional_results["Motif_Based"]["f1"]:.3f}')
    
    # 3. Length-based
    print(f'\n🔍 Length-based Classification...')
    start_time = time.time()
    length_results = length_based_classification(X_seqs, y, cv)
    traditional_results['Length_Based'] = {
        'accuracy': np.mean([r['accuracy'] for r in length_results]),
        'precision': np.mean([r['precision'] for r in length_results]),
        'recall': np.mean([r['recall'] for r in length_results]),
        'f1': np.mean([r['f1'] for r in length_results]),
        'training_time': time.time() - start_time
    }
    print(f'   F1-Score: {traditional_results["Length_Based"]["f1"]:.3f}')
    
    # 4. AA Composition
    print(f'\n🔍 Amino Acid Composition Classification...')
    start_time = time.time()
    aa_comp_results = aa_composition_classification(X_seqs, y, cv)
    traditional_results['AA_Composition'] = {
        'accuracy': np.mean([r['accuracy'] for r in aa_comp_results]),
        'precision': np.mean([r['precision'] for r in aa_comp_results]),
        'recall': np.mean([r['recall'] for r in aa_comp_results]),
        'f1': np.mean([r['f1'] for r in aa_comp_results]),
        'training_time': time.time() - start_time
    }
    print(f'   F1-Score: {traditional_results["AA_Composition"]["f1"]:.3f}')
    
    # Load ML results for comparison
    print(f'\n📊 COMPARISON WITH ML METHODS')
    with open('results/germacrene_benchmark_results.json', 'r') as f:
        ml_results = json.load(f)
    
    # Get best ML result (XGBoost)
    best_ml = None
    best_f1 = 0
    for model_name, result in ml_results.items():
        if result.get('cv_scores') and result['cv_scores']['f1']['mean'] > best_f1:
            best_f1 = result['cv_scores']['f1']['mean']
            best_ml = model_name
    
    traditional_results['ESM-2 + XGBoost'] = {
        'accuracy': ml_results[best_ml]['cv_scores']['accuracy']['mean'],
        'precision': ml_results[best_ml]['cv_scores']['precision']['mean'],
        'recall': ml_results[best_ml]['cv_scores']['recall']['mean'],
        'f1': ml_results[best_ml]['cv_scores']['f1']['mean'],
        'training_time': ml_results[best_ml]['training_time']
    }
    
    # Save results
    results_path = 'results/germacrene_traditional_benchmark_results.json'
    Path('results').mkdir(exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(traditional_results, f, indent=2)
    
    print(f'\n🏆 COMPARATIVE BENCHMARK RESULTS:')
    print(f'   Method                | F1-Score | Accuracy | Training Time')
    print(f'   --------------------|----------|----------|-------------')
    
    # Sort by F1-score
    sorted_results = sorted(
        traditional_results.items(),
        key=lambda x: x[1]['f1'],
        reverse=True
    )
    
    for method, results in sorted_results:
        f1 = results['f1']
        acc = results['accuracy']
        train_time = results['training_time']
        print(f'   {method:<19} | {f1:.3f}    | {acc:.3f}    | {train_time:.1f}s')
    
    print(f'\n💾 Results saved to: {results_path}')
    print(f'🎯 Traditional methods benchmark complete!')

if __name__ == "__main__":
    main()
