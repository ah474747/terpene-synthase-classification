#!/usr/bin/env python3
"""
Comprehensive Dataset Characterization for Ent-Kaurene Classification

This script provides detailed analysis of the dataset including organism diversity,
experimental validation levels, sequence characteristics, and data quality metrics.

Author: Cursor AI
Date: October 17, 2025
"""

import pandas as pd
import numpy as np
from collections import Counter
import re
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

class DatasetCharacterization:
    """Comprehensive dataset characterization"""
    
    def __init__(self, data_file="data/ent_kaurene_binary_dataset.csv"):
        """Initialize dataset characterization"""
        self.df = pd.read_csv(data_file)
        
        print(f"📊 Dataset characterization initialized:")
        print(f"   Total sequences: {len(self.df)}")
        print(f"   Columns: {list(self.df.columns)}")
    
    def analyze_sequence_characteristics(self):
        """Analyze sequence length, composition, and diversity"""
        print("\\n🧬 Analyzing sequence characteristics...")
        
        # Sequence length analysis
        sequence_lengths = self.df['Sequence'].str.len()
        
        length_stats = {
            'mean_length': sequence_lengths.mean(),
            'median_length': sequence_lengths.median(),
            'std_length': sequence_lengths.std(),
            'min_length': sequence_lengths.min(),
            'max_length': sequence_lengths.max(),
            'q25_length': sequence_lengths.quantile(0.25),
            'q75_length': sequence_lengths.quantile(0.75)
        }
        
        # Length distribution by class
        positive_lengths = self.df[self.df['is_ent_kaurene'] == 1]['Sequence'].str.len()
        negative_lengths = self.df[self.df['is_ent_kaurene'] == 0]['Sequence'].str.len()
        
        class_length_stats = {
            'positive_mean_length': positive_lengths.mean(),
            'positive_std_length': positive_lengths.std(),
            'negative_mean_length': negative_lengths.mean(),
            'negative_std_length': negative_lengths.std()
        }
        
        # Amino acid composition analysis
        all_sequences = ''.join(self.df['Sequence'])
        aa_counts = Counter(all_sequences)
        total_aa = sum(aa_counts.values())
        
        aa_composition = {aa: count/total_aa for aa, count in aa_counts.items()}
        
        print(f"   Sequence length statistics:")
        print(f"     Mean: {length_stats['mean_length']:.1f} aa")
        print(f"     Range: {length_stats['min_length']}-{length_stats['max_length']} aa")
        print(f"     Std: {length_stats['std_length']:.1f} aa")
        
        return {
            'length_statistics': length_stats,
            'class_length_statistics': class_length_stats,
            'amino_acid_composition': aa_composition,
            'total_amino_acids': total_aa
        }
    
    def analyze_organism_diversity(self):
        """Analyze organism diversity and taxonomic distribution"""
        print("\\n🌿 Analyzing organism diversity...")
        
        # Extract organism information from Source_ID
        # Assuming Source_ID contains organism information
        organism_patterns = []
        
        for source_id in self.df['Source_ID']:
            # Try to extract organism name patterns
            if '_' in source_id:
                parts = source_id.split('_')
                if len(parts) >= 2:
                    organism_patterns.append(parts[0] + '_' + parts[1])
                else:
                    organism_patterns.append(parts[0])
            else:
                organism_patterns.append(source_id)
        
        organism_counts = Counter(organism_patterns)
        
        # Analyze organism diversity
        unique_organisms = len(organism_counts)
        most_common_organisms = organism_counts.most_common(10)
        
        print(f"   Organism diversity:")
        print(f"     Unique organism patterns: {unique_organisms}")
        print(f"     Most common organisms:")
        for organism, count in most_common_organisms[:5]:
            print(f"       {organism}: {count} sequences")
        
        return {
            'unique_organisms': unique_organisms,
            'organism_distribution': dict(organism_counts),
            'most_common_organisms': most_common_organisms
        }
    
    def analyze_product_diversity(self):
        """Analyze product diversity and distribution"""
        print("\\n🧪 Analyzing product diversity...")
        
        # Analyze all products
        all_products = []
        for products_str in self.df['Products_Concat']:
            products = [p.strip() for p in products_str.split(';')]
            all_products.extend(products)
        
        product_counts = Counter(all_products)
        
        # Analyze product diversity
        unique_products = len(product_counts)
        most_common_products = product_counts.most_common(15)
        
        # Analyze ent-kaurene specifically
        ent_kaurene_sequences = self.df[self.df['is_ent_kaurene'] == 1]
        ent_kaurene_count = len(ent_kaurene_sequences)
        
        print(f"   Product diversity:")
        print(f"     Unique products: {unique_products}")
        print(f"     Ent-kaurene sequences: {ent_kaurene_count}")
        print(f"     Most common products:")
        for product, count in most_common_products[:8]:
            print(f"       {product}: {count} sequences")
        
        return {
            'unique_products': unique_products,
            'product_distribution': dict(product_counts),
            'most_common_products': most_common_products,
            'ent_kaurene_count': ent_kaurene_count
        }
    
    def analyze_data_quality(self):
        """Analyze data quality and potential issues"""
        print("\\n🔍 Analyzing data quality...")
        
        # Check for missing values
        missing_values = self.df.isnull().sum()
        
        # Check for duplicate sequences
        duplicate_sequences = self.df['Sequence'].duplicated().sum()
        
        # Check for very short sequences (potential errors)
        very_short_sequences = (self.df['Sequence'].str.len() < 100).sum()
        
        # Check for very long sequences (potential errors)
        very_long_sequences = (self.df['Sequence'].str.len() > 1000).sum()
        
        # Check for sequences with unusual amino acids
        valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
        invalid_sequences = 0
        for seq in self.df['Sequence']:
            if not set(seq).issubset(valid_aa):
                invalid_sequences += 1
        
        print(f"   Data quality metrics:")
        print(f"     Missing values: {missing_values.sum()}")
        print(f"     Duplicate sequences: {duplicate_sequences}")
        print(f"     Very short sequences (<100 aa): {very_short_sequences}")
        print(f"     Very long sequences (>1000 aa): {very_long_sequences}")
        print(f"     Invalid amino acids: {invalid_sequences}")
        
        return {
            'missing_values': missing_values.to_dict(),
            'duplicate_sequences': duplicate_sequences,
            'very_short_sequences': very_short_sequences,
            'very_long_sequences': very_long_sequences,
            'invalid_sequences': invalid_sequences
        }
    
    def analyze_class_distribution(self):
        """Analyze class distribution and balance"""
        print("\\n⚖️ Analyzing class distribution...")
        
        # Basic class distribution
        class_counts = self.df['is_ent_kaurene'].value_counts()
        total_sequences = len(self.df)
        
        positive_count = class_counts.get(1, 0)
        negative_count = class_counts.get(0, 0)
        
        positive_ratio = positive_count / total_sequences
        negative_ratio = negative_count / total_sequences
        imbalance_ratio = negative_count / positive_count if positive_count > 0 else float('inf')
        
        # Analyze class distribution by sequence length
        length_bins = [0, 200, 400, 600, 800, 1000, float('inf')]
        length_labels = ['0-200', '200-400', '400-600', '600-800', '800-1000', '1000+']
        
        class_by_length = []
        for i in range(len(length_bins)-1):
            bin_mask = (self.df['Sequence'].str.len() >= length_bins[i]) & \
                      (self.df['Sequence'].str.len() < length_bins[i+1])
            bin_data = self.df[bin_mask]
            
            if len(bin_data) > 0:
                bin_positive = bin_data['is_ent_kaurene'].sum()
                bin_negative = len(bin_data) - bin_positive
                class_by_length.append({
                    'length_range': length_labels[i],
                    'total_sequences': len(bin_data),
                    'positive_count': bin_positive,
                    'negative_count': bin_negative,
                    'positive_ratio': bin_positive / len(bin_data)
                })
        
        print(f"   Class distribution:")
        print(f"     Positive (ent-kaurene): {positive_count} ({positive_ratio:.1%})")
        print(f"     Negative (non-ent-kaurene): {negative_count} ({negative_ratio:.1%})")
        print(f"     Imbalance ratio: {imbalance_ratio:.1f}:1")
        
        return {
            'positive_count': positive_count,
            'negative_count': negative_count,
            'positive_ratio': positive_ratio,
            'negative_ratio': negative_ratio,
            'imbalance_ratio': imbalance_ratio,
            'class_by_length': class_by_length
        }
    
    def create_characterization_report(self):
        """Create comprehensive dataset characterization report"""
        print("\\n📊 Creating comprehensive dataset characterization...")
        
        # Run all analyses
        sequence_analysis = self.analyze_sequence_characteristics()
        organism_analysis = self.analyze_organism_diversity()
        product_analysis = self.analyze_product_diversity()
        quality_analysis = self.analyze_data_quality()
        class_analysis = self.analyze_class_distribution()
        
        # Combine all analyses
        characterization_report = {
            'dataset_overview': {
                'total_sequences': len(self.df),
                'columns': list(self.df.columns),
                'data_source': 'MARTS-DB (deduplicated)'
            },
            'sequence_characteristics': sequence_analysis,
            'organism_diversity': organism_analysis,
            'product_diversity': product_analysis,
            'data_quality': quality_analysis,
            'class_distribution': class_analysis
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
        
        characterization_report = convert_numpy_types(characterization_report)
        
        # Save report
        with open('results/dataset_characterization_report.json', 'w') as f:
            json.dump(characterization_report, f, indent=2)
        
        print("✅ Dataset characterization completed")
        print("💾 Report saved to: results/dataset_characterization_report.json")
        
        return characterization_report
    
    def create_visualizations(self):
        """Create visualization plots for dataset characterization"""
        print("\\n📈 Creating dataset visualizations...")
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Dataset Characterization: Ent-Kaurene Classification', fontsize=16, fontweight='bold')
        
        # 1. Sequence length distribution
        sequence_lengths = self.df['Sequence'].str.len()
        axes[0, 0].hist(sequence_lengths, bins=30, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Sequence Length Distribution')
        axes[0, 0].set_xlabel('Sequence Length (aa)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(sequence_lengths.mean(), color='red', linestyle='--', label=f'Mean: {sequence_lengths.mean():.1f}')
        axes[0, 0].legend()
        
        # 2. Class distribution
        class_counts = self.df['is_ent_kaurene'].value_counts()
        labels = ['Non-ent-kaurene', 'Ent-kaurene']
        colors = ['lightcoral', 'lightblue']
        axes[0, 1].pie(class_counts.values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[0, 1].set_title('Class Distribution')
        
        # 3. Sequence length by class
        positive_lengths = self.df[self.df['is_ent_kaurene'] == 1]['Sequence'].str.len()
        negative_lengths = self.df[self.df['is_ent_kaurene'] == 0]['Sequence'].str.len()
        
        axes[0, 2].hist([negative_lengths, positive_lengths], bins=20, alpha=0.7, 
                       label=['Non-ent-kaurene', 'Ent-kaurene'], color=['lightcoral', 'lightblue'])
        axes[0, 2].set_title('Sequence Length by Class')
        axes[0, 2].set_xlabel('Sequence Length (aa)')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].legend()
        
        # 4. Top 10 products
        all_products = []
        for products_str in self.df['Products_Concat']:
            products = [p.strip() for p in products_str.split(';')]
            all_products.extend(products)
        
        product_counts = Counter(all_products)
        top_products = product_counts.most_common(10)
        
        products, counts = zip(*top_products)
        axes[1, 0].barh(range(len(products)), counts)
        axes[1, 0].set_yticks(range(len(products)))
        axes[1, 0].set_yticklabels(products)
        axes[1, 0].set_title('Top 10 Products')
        axes[1, 0].set_xlabel('Count')
        
        # 5. Amino acid composition
        all_sequences = ''.join(self.df['Sequence'])
        aa_counts = Counter(all_sequences)
        total_aa = sum(aa_counts.values())
        
        aa_composition = {aa: count/total_aa for aa, count in aa_counts.items()}
        aa_sorted = sorted(aa_composition.items(), key=lambda x: x[1], reverse=True)
        
        aas, compositions = zip(*aa_sorted)
        axes[1, 1].bar(aas, compositions)
        axes[1, 1].set_title('Amino Acid Composition')
        axes[1, 1].set_xlabel('Amino Acid')
        axes[1, 1].set_ylabel('Frequency')
        
        # 6. Class distribution by length bins
        length_bins = [0, 200, 400, 600, 800, 1000, float('inf')]
        length_labels = ['0-200', '200-400', '400-600', '600-800', '800-1000', '1000+']
        
        bin_positive_ratios = []
        bin_totals = []
        
        for i in range(len(length_bins)-1):
            bin_mask = (self.df['Sequence'].str.len() >= length_bins[i]) & \
                      (self.df['Sequence'].str.len() < length_bins[i+1])
            bin_data = self.df[bin_mask]
            
            if len(bin_data) > 0:
                bin_positive_ratio = bin_data['is_ent_kaurene'].mean()
                bin_positive_ratios.append(bin_positive_ratio)
                bin_totals.append(len(bin_data))
            else:
                bin_positive_ratios.append(0)
                bin_totals.append(0)
        
        axes[1, 2].bar(length_labels, bin_positive_ratios)
        axes[1, 2].set_title('Positive Class Ratio by Length')
        axes[1, 2].set_xlabel('Length Range (aa)')
        axes[1, 2].set_ylabel('Ent-kaurene Ratio')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('results/figure9_dataset_characterization.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Visualizations created and saved to: results/figure9_dataset_characterization.png")

def main():
    """Main dataset characterization function"""
    print("📊 COMPREHENSIVE DATASET CHARACTERIZATION")
    print("=" * 50)
    
    characterizer = DatasetCharacterization()
    
    # Run comprehensive characterization
    report = characterizer.create_characterization_report()
    
    # Create visualizations
    characterizer.create_visualizations()
    
    print("\\n🎉 DATASET CHARACTERIZATION COMPLETED!")
    print("\\n📋 KEY FINDINGS:")
    
    # Print key findings
    overview = report['dataset_overview']
    class_dist = report['class_distribution']
    seq_stats = report['sequence_characteristics']['length_statistics']
    
    print(f"   • Dataset size: {overview['total_sequences']} sequences")
    print(f"   • Class imbalance: {class_dist['imbalance_ratio']:.1f}:1 ratio")
    print(f"   • Sequence length: {seq_stats['mean_length']:.1f} ± {seq_stats['std_length']:.1f} aa")
    print(f"   • Positive class: {class_dist['positive_ratio']:.1%} of dataset")
    
    print("\\n🚀 COMPREHENSIVE DATASET ANALYSIS READY!")

if __name__ == "__main__":
    main()
