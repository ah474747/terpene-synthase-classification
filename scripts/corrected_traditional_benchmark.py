#!/usr/bin/env python3
"""
Corrected Traditional Methods Benchmark (without problematic HMM)

This script provides a clean comparison of working traditional methods
vs ESM-2 + ML, removing the failed HMM implementation.

Author: Cursor AI
Date: October 17, 2025
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from Bio import pairwise2
from Bio.Seq import Seq
import json
from pathlib import Path
import matplotlib.pyplot as plt
from collections import Counter
import re

class CorrectedTraditionalBenchmark:
    """Corrected traditional methods benchmark without problematic HMM"""
    
    def __init__(self, data_file="data/ent_kaurene_binary_dataset.csv"):
        """Initialize the corrected benchmark"""
        self.df = pd.read_csv(data_file)
        
        print(f"📊 Corrected traditional benchmark initialized:")
        print(f"   Total sequences: {len(self.df)}")
        print(f"   Ent-kaurene sequences: {len(self.df[self.df['is_ent_kaurene'] == 1])}")
        print(f"   Non-ent-kaurene sequences: {len(self.df[self.df['is_ent_kaurene'] == 0])}")
    
    def create_holdout_split(self, test_size=0.2, random_state=42):
        """Create hold-out split for fair comparison"""
        print("\\n🎯 Creating hold-out split for corrected benchmark...")
        
        X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
            self.df['Sequence'], self.df['is_ent_kaurene'], 
            np.arange(len(self.df)), test_size=test_size, 
            stratify=self.df['is_ent_kaurene'], random_state=random_state
        )
        
        train_df = self.df.iloc[train_idx].reset_index(drop=True)
        test_df = self.df.iloc[test_idx].reset_index(drop=True)
        
        print(f"✅ Hold-out split created:")
        print(f"   Training: {len(train_df)} sequences")
        print(f"   Testing: {len(test_df)} sequences")
        
        return train_df, test_df
    
    def sequence_similarity_classification(self, train_df, test_df):
        """Sequence similarity-based classification"""
        print("\\n🔗 Sequence similarity-based classification...")
        
        # Get training ent-kaurene sequences
        train_ent_kaurene = train_df[train_df['is_ent_kaurene'] == 1]['Sequence'].tolist()
        
        predictions = []
        similarity_scores = []
        
        for test_seq in test_df['Sequence']:
            # Calculate maximum similarity to any ent-kaurene sequence
            max_similarity = 0
            
            for train_seq in train_ent_kaurene:
                try:
                    # Use simple sequence alignment
                    alignment = pairwise2.align.globalxx(test_seq, train_seq)
                    if alignment:
                        score = alignment[0][2] / max(len(test_seq), len(train_seq))
                        max_similarity = max(max_similarity, score)
                except:
                    continue
            
            # Classify based on similarity threshold
            prediction = 1 if max_similarity > 0.3 else 0
            predictions.append(prediction)
            similarity_scores.append(max_similarity)
        
        # Calculate metrics
        y_true = test_df['is_ent_kaurene'].values
        f1 = f1_score(y_true, predictions)
        accuracy = accuracy_score(y_true, predictions)
        
        print(f"✅ Sequence similarity classification completed:")
        print(f"   F1-Score: {f1:.3f}")
        print(f"   Accuracy: {accuracy:.3f}")
        
        return {
            'method': 'Sequence_Similarity',
            'predictions': predictions,
            'scores': similarity_scores,
            'f1_score': f1,
            'accuracy': accuracy
        }
    
    def motif_based_classification(self, train_df, test_df):
        """Motif-based classification using conserved patterns"""
        print("\\n🧬 Motif-based classification...")
        
        # Get training ent-kaurene sequences
        train_ent_kaurene = train_df[train_df['is_ent_kaurene'] == 1]['Sequence'].tolist()
        
        # Find conserved motifs in ent-kaurene sequences
        motifs = self._find_conserved_motifs(train_ent_kaurene)
        
        predictions = []
        motif_scores = []
        
        for test_seq in test_df['Sequence']:
            # Calculate motif score
            motif_score = self._calculate_motif_score(test_seq, motifs)
            
            # Classify based on motif score
            prediction = 1 if motif_score > 0.5 else 0
            predictions.append(prediction)
            motif_scores.append(motif_score)
        
        # Calculate metrics
        y_true = test_df['is_ent_kaurene'].values
        f1 = f1_score(y_true, predictions)
        accuracy = accuracy_score(y_true, predictions)
        
        print(f"✅ Motif-based classification completed:")
        print(f"   F1-Score: {f1:.3f}")
        print(f"   Accuracy: {accuracy:.3f}")
        
        return {
            'method': 'Motif_Based',
            'predictions': predictions,
            'scores': motif_scores,
            'f1_score': f1,
            'accuracy': accuracy
        }
    
    def length_based_classification(self, train_df, test_df):
        """Length-based classification (baseline)"""
        print("\\n📏 Length-based classification...")
        
        # Calculate optimal length threshold from training data
        train_ent_kaurene_lengths = train_df[train_df['is_ent_kaurene'] == 1]['Sequence'].str.len()
        train_negative_lengths = train_df[train_df['is_ent_kaurene'] == 0]['Sequence'].str.len()
        
        # Use mean length as threshold
        threshold = (train_ent_kaurene_lengths.mean() + train_negative_lengths.mean()) / 2
        
        predictions = []
        length_scores = []
        
        for test_seq in test_df['Sequence']:
            length = len(test_seq)
            prediction = 1 if length > threshold else 0
            predictions.append(prediction)
            length_scores.append(length)
        
        # Calculate metrics
        y_true = test_df['is_ent_kaurene'].values
        f1 = f1_score(y_true, predictions)
        accuracy = accuracy_score(y_true, predictions)
        
        print(f"✅ Length-based classification completed:")
        print(f"   F1-Score: {f1:.3f}")
        print(f"   Accuracy: {accuracy:.3f}")
        
        return {
            'method': 'Length_Based',
            'predictions': predictions,
            'scores': length_scores,
            'f1_score': f1,
            'accuracy': accuracy
        }
    
    def aa_composition_classification(self, train_df, test_df):
        """Amino acid composition-based classification"""
        print("\\n🧪 Amino acid composition-based classification...")
        
        # Calculate amino acid composition for training sequences
        train_ent_kaurene = train_df[train_df['is_ent_kaurene'] == 1]['Sequence']
        train_negative = train_df[train_df['is_ent_kaurene'] == 0]['Sequence']
        
        # Get composition profiles
        ent_kaurene_profile = self._calculate_composition_profile(train_ent_kaurene)
        negative_profile = self._calculate_composition_profile(train_negative)
        
        predictions = []
        composition_scores = []
        
        for test_seq in test_df['Sequence']:
            # Calculate composition similarity
            test_profile = self._calculate_sequence_composition(test_seq)
            
            # Calculate similarity to ent-kaurene profile
            ent_kaurene_similarity = self._calculate_profile_similarity(test_profile, ent_kaurene_profile)
            negative_similarity = self._calculate_profile_similarity(test_profile, negative_profile)
            
            # Classify based on which profile is more similar
            prediction = 1 if ent_kaurene_similarity > negative_similarity else 0
            predictions.append(prediction)
            composition_scores.append(ent_kaurene_similarity - negative_similarity)
        
        # Calculate metrics
        y_true = test_df['is_ent_kaurene'].values
        f1 = f1_score(y_true, predictions)
        accuracy = accuracy_score(y_true, predictions)
        
        print(f"✅ Amino acid composition classification completed:")
        print(f"   F1-Score: {f1:.3f}")
        print(f"   Accuracy: {accuracy:.3f}")
        
        return {
            'method': 'AA_Composition',
            'predictions': predictions,
            'scores': composition_scores,
            'f1_score': f1,
            'accuracy': accuracy
        }
    
    def _find_conserved_motifs(self, sequences, min_length=5, min_frequency=0.3):
        """Find conserved motifs in sequences"""
        motifs = []
        
        if len(sequences) < 2:
            return motifs
        
        # Use the first sequence as reference
        ref_seq = sequences[0]
        
        # Look for conserved subsequences
        for length in range(min_length, min(len(ref_seq), 15)):
            for start in range(len(ref_seq) - length + 1):
                subsequence = ref_seq[start:start + length]
                
                # Count how many sequences contain this subsequence
                count = sum(1 for seq in sequences if subsequence in seq)
                frequency = count / len(sequences)
                
                if frequency >= min_frequency:
                    motifs.append(subsequence)
        
        return motifs[:10]  # Return top 10 motifs
    
    def _calculate_motif_score(self, sequence, motifs):
        """Calculate motif score for a sequence"""
        if not motifs:
            return 0
        
        matches = sum(1 for motif in motifs if motif in sequence)
        return matches / len(motifs)
    
    def _calculate_composition_profile(self, sequences):
        """Calculate average amino acid composition profile"""
        all_aa = ''.join(sequences)
        total_count = len(all_aa)
        
        profile = {}
        for aa in 'ACDEFGHIKLMNPQRSTVWY':
            profile[aa] = all_aa.count(aa) / total_count
        
        return profile
    
    def _calculate_sequence_composition(self, sequence):
        """Calculate amino acid composition for a single sequence"""
        total_count = len(sequence)
        
        profile = {}
        for aa in 'ACDEFGHIKLMNPQRSTVWY':
            profile[aa] = sequence.count(aa) / total_count
        
        return profile
    
    def _calculate_profile_similarity(self, profile1, profile2):
        """Calculate similarity between two composition profiles"""
        similarity = 0
        for aa in 'ACDEFGHIKLMNPQRSTVWY':
            similarity += min(profile1.get(aa, 0), profile2.get(aa, 0))
        
        return similarity
    
    def run_corrected_benchmark(self):
        """Run corrected traditional methods benchmark"""
        print("\\n🚀 CORRECTED TRADITIONAL METHODS BENCHMARK")
        print("=" * 50)
        
        # Create hold-out split
        train_df, test_df = self.create_holdout_split()
        
        # Run traditional methods
        results = []
        
        # Sequence similarity
        seq_sim_result = self.sequence_similarity_classification(train_df, test_df)
        results.append(seq_sim_result)
        
        # Motif-based
        motif_result = self.motif_based_classification(train_df, test_df)
        results.append(motif_result)
        
        # Length-based
        length_result = self.length_based_classification(train_df, test_df)
        results.append(length_result)
        
        # Amino acid composition
        aa_comp_result = self.aa_composition_classification(train_df, test_df)
        results.append(aa_comp_result)
        
        # Add ESM-2 + XGBoost results from previous benchmark
        esm2_result = {
            'method': 'ESM-2 + XGBoost',
            'f1_score': 0.907,
            'accuracy': 0.920
        }
        results.append(esm2_result)
        
        # Create comparison table
        comparison_data = []
        for result in results:
            comparison_data.append({
                'Method': result['method'],
                'F1-Score': result['f1_score'],
                'Accuracy': result['accuracy']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Save results
        comparison_df.to_csv('results/corrected_traditional_methods_benchmark.csv', index=False)
        
        with open('results/corrected_traditional_methods_detailed.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\\n🏆 CORRECTED BENCHMARK RESULTS:")
        print("=" * 40)
        print(comparison_df.to_string(index=False))
        
        print("\\n💾 Results saved to:")
        print("   • results/corrected_traditional_methods_benchmark.csv")
        print("   • results/corrected_traditional_methods_detailed.json")
        
        # Create visualization
        self.create_corrected_visualization(comparison_df)
        
        return results
    
    def create_corrected_visualization(self, comparison_df):
        """Create visualization for corrected benchmark"""
        print("\\n📈 Creating corrected benchmark visualization...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        methods = comparison_df['Method'].tolist()
        f1_scores = comparison_df['F1-Score'].tolist()
        accuracies = comparison_df['Accuracy'].tolist()
        
        colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#1f77b4']
        
        # F1-Score comparison
        bars1 = ax1.bar(methods, f1_scores, color=colors)
        ax1.set_title('F1-Score Comparison: Corrected Traditional vs ESM-2 + ML', fontsize=14, fontweight='bold')
        ax1.set_ylabel('F1-Score')
        ax1.set_ylim(0, 1)
        ax1.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='Good Performance Threshold')
        
        # Add value labels
        for bar, score in zip(bars1, f1_scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=10)
        
        ax1.legend()
        ax1.tick_params(axis='x', rotation=45)
        
        # Accuracy comparison
        bars2 = ax2.bar(methods, accuracies, color=colors)
        ax2.set_title('Accuracy Comparison: Corrected Traditional vs ESM-2 + ML', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Accuracy')
        ax2.set_ylim(0, 1)
        
        # Add value labels
        for bar, score in zip(bars2, accuracies):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=10)
        
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('results/figure10_corrected_traditional_vs_esm2_benchmark.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Corrected benchmark visualization saved to: results/figure10_corrected_traditional_vs_esm2_benchmark.png")

def main():
    """Main corrected benchmark function"""
    print("🔧 CORRECTED TRADITIONAL METHODS BENCHMARK")
    print("=" * 50)
    
    benchmark = CorrectedTraditionalBenchmark()
    
    # Run corrected benchmark
    results = benchmark.run_corrected_benchmark()
    
    print("\\n🎉 CORRECTED BENCHMARK COMPLETED!")
    print("\\n📋 SUMMARY:")
    print("   • Removed problematic HMM implementation")
    print("   • Focused on working traditional methods")
    print("   • Maintained fair comparison with ESM-2 + ML")
    print("\\n🚀 CLEAN TRADITIONAL METHODS COMPARISON READY!")

if __name__ == "__main__":
    main()
