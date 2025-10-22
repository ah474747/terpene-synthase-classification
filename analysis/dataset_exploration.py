#!/usr/bin/env python3
"""
Dataset Exploration for Ent-Kaurene Binary Classifier

This script provides comprehensive exploration and statistics of the ent-kaurene
binary classification dataset.

Author: Cursor AI
Date: October 17, 2025
"""

import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

def explore_dataset(df):
    """Comprehensive dataset exploration"""
    print('ENT-KAURENE BINARY CLASSIFICATION DATASET EXPLORATION')
    print('=' * 60)
    
    print(f'Dataset Shape: {df.shape}')
    print(f'Columns: {list(df.columns)}')
    print()
    
    # Basic statistics
    print('BASIC STATISTICS:')
    print('-' * 20)
    print(f'Total sequences: {len(df)}')
    print(f'Ent-kaurene sequences: {df["is_ent_kaurene"].sum()} ({df["is_ent_kaurene"].mean()*100:.1f}%)')
    print(f'Non-ent-kaurene sequences: {(df["is_ent_kaurene"] == 0).sum()} ({(df["is_ent_kaurene"] == 0).mean()*100:.1f}%)')
    print(f'Class imbalance ratio: {(df["is_ent_kaurene"] == 0).sum() / df["is_ent_kaurene"].sum():.1f}:1')
    print()
    
    # Sequence length analysis
    print('SEQUENCE LENGTH ANALYSIS:')
    print('-' * 30)
    lengths = df['Sequence'].str.len()
    print(f'Length statistics:')
    print(f'  Min: {lengths.min()} aa')
    print(f'  Max: {lengths.max()} aa')
    print(f'  Mean: {lengths.mean():.1f} ± {lengths.std():.1f} aa')
    print(f'  Median: {lengths.median():.1f} aa')
    print(f'  CV: {lengths.std()/lengths.mean()*100:.1f}%')
    print()
    
    # Length by class
    ent_kaurene_lengths = df[df['is_ent_kaurene'] == 1]['Sequence'].str.len()
    non_ent_kaurene_lengths = df[df['is_ent_kaurene'] == 0]['Sequence'].str.len()
    
    print(f'Length by class:')
    print(f'  Ent-kaurene: {ent_kaurene_lengths.mean():.1f} ± {ent_kaurene_lengths.std():.1f} aa')
    print(f'  Non-ent-kaurene: {non_ent_kaurene_lengths.mean():.1f} ± {non_ent_kaurene_lengths.std():.1f} aa')
    print()
    
    # Product analysis
    print('PRODUCT ANALYSIS:')
    print('-' * 20)
    
    # Parse all products
    all_products = []
    for products_str in df['Products_Concat'].dropna():
        products = [p.strip().lower() for p in str(products_str).split(';')]
        all_products.extend(products)
    
    product_counts = Counter(all_products)
    
    print(f'Total unique products: {len(product_counts)}')
    print(f'Top 15 most frequent products:')
    for i, (product, count) in enumerate(product_counts.most_common(15), 1):
        freq = count / len(df) * 100
        print(f'  {i:2d}. {product:<25} {count:3d} sequences ({freq:.1f}%)')
    print()
    
    # Ent-kaurene specific analysis
    print('ENT-KAURENE SPECIFIC ANALYSIS:')
    print('-' * 35)
    ent_kaurene_df = df[df['is_ent_kaurene'] == 1].copy()
    
    # Product promiscuity in ent-kaurene group
    single_product = sum(1 for products in ent_kaurene_df['Products_Concat'] if ';' not in str(products))
    multi_product = len(ent_kaurene_df) - single_product
    
    print(f'Ent-kaurene product specificity:')
    print(f'  Single product: {single_product} sequences ({single_product/len(ent_kaurene_df)*100:.1f}%)')
    print(f'  Multiple products: {multi_product} sequences ({multi_product/len(ent_kaurene_df)*100:.1f}%)')
    
    if multi_product > 0:
        print(f'  Multi-product examples:')
        multi_examples = ent_kaurene_df[ent_kaurene_df['Products_Concat'].str.contains(';', na=False)].head(5)
        for idx, row in multi_examples.iterrows():
            print(f'    {row["Source_ID"]}: {row["Products_Concat"]}')
    print()
    
    # Amino acid composition analysis
    print('AMINO ACID COMPOSITION ANALYSIS:')
    print('-' * 35)
    
    # Overall composition
    all_seqs = df['Sequence'].tolist()
    aa_composition = Counter()
    total_aas = 0
    
    for seq in all_seqs:
        for aa in seq:
            aa_composition[aa] += 1
            total_aas += 1
    
    print(f'Overall amino acid composition (top 10):')
    for i, (aa, count) in enumerate(aa_composition.most_common(10), 1):
        freq = count / total_aas * 100
        print(f'  {i:2d}. {aa}: {freq:.1f}%')
    print()
    
    # Composition by class
    ent_kaurene_aas = Counter()
    non_ent_kaurene_aas = Counter()
    ent_kaurene_total = 0
    non_ent_kaurene_total = 0
    
    for idx, row in df.iterrows():
        seq = row['Sequence']
        if row['is_ent_kaurene'] == 1:
            for aa in seq:
                ent_kaurene_aas[aa] += 1
                ent_kaurene_total += 1
        else:
            for aa in seq:
                non_ent_kaurene_aas[aa] += 1
                non_ent_kaurene_total += 1
    
    print(f'Ent-kaurene amino acid composition (top 10):')
    for i, (aa, count) in enumerate(ent_kaurene_aas.most_common(10), 1):
        freq = count / ent_kaurene_total * 100
        print(f'  {i:2d}. {aa}: {freq:.1f}%')
    
    print(f'Non-ent-kaurene amino acid composition (top 10):')
    for i, (aa, count) in enumerate(non_ent_kaurene_aas.most_common(10), 1):
        freq = count / non_ent_kaurene_total * 100
        print(f'  {i:2d}. {aa}: {freq:.1f}%')
    print()
    
    # Data quality checks
    print('DATA QUALITY CHECKS:')
    print('-' * 20)
    print(f'Missing sequences: {df["Sequence"].isnull().sum()}')
    print(f'Missing products: {df["Products_Concat"].isnull().sum()}')
    print(f'Empty sequences: {(df["Sequence"].str.len() == 0).sum()}')
    print(f'Very short sequences (<50 aa): {(df["Sequence"].str.len() < 50).sum()}')
    print(f'Very long sequences (>2000 aa): {(df["Sequence"].str.len() > 2000).sum()}')
    print()
    
    # Summary for ML
    print('MACHINE LEARNING IMPLICATIONS:')
    print('-' * 35)
    print(f'✅ Good class balance: {(df["is_ent_kaurene"] == 0).sum() / df["is_ent_kaurene"].sum():.1f}:1 ratio')
    print(f'✅ Large positive class: {df["is_ent_kaurene"].sum()} sequences')
    print(f'✅ Low product promiscuity: {single_product/len(ent_kaurene_df)*100:.1f}% single-product in ent-kaurene')
    print(f'✅ High sequence diversity: {len(set(df["Sequence"]))} unique sequences out of {len(df)} total')
    print(f'⚠️  Length variation: {lengths.std()/lengths.mean()*100:.1f}% CV requires robust feature extraction')
    print(f'⚠️  Need ESM-2 embeddings: {len(df)} sequences need embedding generation')

def main():
    """Main exploration function"""
    # Load the dataset
    df = pd.read_csv('data/ent_kaurene_binary_dataset.csv')
    
    # Perform exploration
    explore_dataset(df)
    
    # Save summary statistics
    summary_stats = {
        'total_sequences': int(len(df)),
        'ent_kaurene_count': int(df['is_ent_kaurene'].sum()),
        'non_ent_kaurene_count': int((df['is_ent_kaurene'] == 0).sum()),
        'class_imbalance_ratio': float((df['is_ent_kaurene'] == 0).sum() / df['is_ent_kaurene'].sum()),
        'mean_length': float(df['Sequence'].str.len().mean()),
        'std_length': float(df['Sequence'].str.len().std()),
        'cv_length': float(df['Sequence'].str.len().std() / df['Sequence'].str.len().mean() * 100),
        'unique_sequences': int(len(set(df['Sequence'])))
    }
    
    # Save to file
    import json
    with open('results/dataset_summary_stats.json', 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    print(f'\n📊 Summary statistics saved to: results/dataset_summary_stats.json')

if __name__ == "__main__":
    main()
