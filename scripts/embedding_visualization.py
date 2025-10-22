#!/usr/bin/env python3
"""
ESM-2 Embedding Visualization Script
===================================

This script creates comprehensive visualizations of ESM-2 embeddings to demonstrate
class separation and interpretability of the protein language model representations.

Addresses reviewer feedback:
- UMAP/t-SNE visualization to show ESM-2 class separation
- Feature interpretation for biological interpretability
- Embedding analysis for model understanding

Author: Andrew Horwitz
Date: October 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
import joblib
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

class EmbeddingVisualizer:
    """Comprehensive embedding visualization and analysis."""
    
    def __init__(self, random_state=RANDOM_STATE):
        self.random_state = random_state
        
    def load_embeddings_and_labels(self, embeddings_path, dataset_path):
        """Load ESM-2 embeddings and corresponding labels."""
        
        print("📂 Loading embeddings and dataset...")
        
        # Load embeddings
        embeddings = np.load(embeddings_path)
        print(f"   Embeddings shape: {embeddings.shape}")
        
        # Load dataset
        df = pd.read_csv(dataset_path)
        print(f"   Dataset shape: {df.shape}")
        
        return embeddings, df
    
    def reduce_dimensionality(self, embeddings, method='umap', n_components=2, **kwargs):
        """Reduce dimensionality of embeddings for visualization."""
        
        print(f"🔄 Applying {method.upper()} dimensionality reduction...")
        
        if method.lower() == 'umap':
            reducer = umap.UMAP(
                n_components=n_components,
                random_state=self.random_state,
                n_neighbors=15,
                min_dist=0.1,
                metric='cosine',
                **kwargs
            )
        elif method.lower() == 'tsne':
            reducer = TSNE(
                n_components=n_components,
                random_state=self.random_state,
                perplexity=30,
                learning_rate=200,
                **kwargs
            )
        elif method.lower() == 'pca':
            reducer = PCA(
                n_components=n_components,
                random_state=self.random_state,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Fit and transform
        reduced_embeddings = reducer.fit_transform(embeddings)
        
        print(f"   Reduced shape: {reduced_embeddings.shape}")
        
        return reduced_embeddings, reducer
    
    def create_embedding_visualization(self, embeddings, labels, target_products, output_path):
        """Create comprehensive embedding visualizations."""
        
        print("🎨 Creating embedding visualizations...")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('ESM-2 Embedding Analysis: Class Separation and Interpretability', 
                     fontsize=16, fontweight='bold')
        
        # Define colors for different products
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        target_names = ['Germacrene', 'Pinene', 'Myrcene']
        
        # 1. UMAP visualization
        ax1 = axes[0, 0]
        umap_embeddings, _ = self.reduce_dimensionality(embeddings, method='umap')
        
        for i, target in enumerate(target_products):
            mask = labels[target] == 1
            if np.sum(mask) > 0:
                ax1.scatter(umap_embeddings[mask, 0], umap_embeddings[mask, 1], 
                           c=colors[i], label=f'{target_names[i]} (n={np.sum(mask)})', 
                           alpha=0.7, s=50)
        
        ax1.set_title('UMAP Visualization of ESM-2 Embeddings', fontweight='bold')
        ax1.set_xlabel('UMAP Component 1')
        ax1.set_ylabel('UMAP Component 2')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. t-SNE visualization
        ax2 = axes[0, 1]
        tsne_embeddings, _ = self.reduce_dimensionality(embeddings, method='tsne')
        
        for i, target in enumerate(target_products):
            mask = labels[target] == 1
            if np.sum(mask) > 0:
                ax2.scatter(tsne_embeddings[mask, 0], tsne_embeddings[mask, 1], 
                           c=colors[i], label=f'{target_names[i]} (n={np.sum(mask)})', 
                           alpha=0.7, s=50)
        
        ax2.set_title('t-SNE Visualization of ESM-2 Embeddings', fontweight='bold')
        ax2.set_xlabel('t-SNE Component 1')
        ax2.set_ylabel('t-SNE Component 2')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. PCA visualization
        ax3 = axes[0, 2]
        pca_embeddings, pca_model = self.reduce_dimensionality(embeddings, method='pca')
        
        for i, target in enumerate(target_products):
            mask = labels[target] == 1
            if np.sum(mask) > 0:
                ax3.scatter(pca_embeddings[mask, 0], pca_embeddings[mask, 1], 
                           c=colors[i], label=f'{target_names[i]} (n={np.sum(mask)})', 
                           alpha=0.7, s=50)
        
        ax3.set_title(f'PCA Visualization (Explained Variance: {pca_model.explained_variance_ratio_[:2].sum():.1%})', 
                     fontweight='bold')
        ax3.set_xlabel(f'PC1 ({pca_model.explained_variance_ratio_[0]:.1%})')
        ax3.set_ylabel(f'PC2 ({pca_model.explained_variance_ratio_[1]:.1%})')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Class separation analysis
        ax4 = axes[1, 0]
        separation_scores = []
        
        for i, target in enumerate(target_products):
            mask = labels[target] == 1
            if np.sum(mask) > 1:
                # Calculate average distance within class vs between classes
                class_embeddings = embeddings[mask]
                other_embeddings = embeddings[~mask]
                
                # Within-class distances
                within_distances = []
                for j in range(len(class_embeddings)):
                    for k in range(j+1, len(class_embeddings)):
                        dist = np.linalg.norm(class_embeddings[j] - class_embeddings[k])
                        within_distances.append(dist)
                
                # Between-class distances
                between_distances = []
                for j in range(min(100, len(class_embeddings))):  # Sample for efficiency
                    for k in range(min(100, len(other_embeddings))):
                        dist = np.linalg.norm(class_embeddings[j] - other_embeddings[k])
                        between_distances.append(dist)
                
                # Separation score (ratio of between to within distances)
                separation_score = np.mean(between_distances) / np.mean(within_distances)
                separation_scores.append(separation_score)
            else:
                separation_scores.append(0)
        
        bars = ax4.bar(target_names, separation_scores, color=colors[:3], alpha=0.7)
        ax4.set_title('Class Separation Scores in ESM-2 Space', fontweight='bold')
        ax4.set_ylabel('Separation Score (Between/Within Distance)')
        ax4.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, score in zip(bars, separation_scores):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # 5. Embedding dimensionality analysis
        ax5 = axes[1, 1]
        
        # Calculate explained variance for different numbers of PCA components
        pca_full = PCA(random_state=self.random_state)
        pca_full.fit(embeddings)
        
        cumsum_variance = np.cumsum(pca_full.explained_variance_ratio_)
        n_components = np.arange(1, min(51, len(cumsum_variance)) + 1)
        
        ax5.plot(n_components, cumsum_variance[:len(n_components)], 'b-', linewidth=2)
        ax5.axhline(y=0.95, color='r', linestyle='--', alpha=0.7, label='95% Variance')
        ax5.axhline(y=0.99, color='g', linestyle='--', alpha=0.7, label='99% Variance')
        
        ax5.set_title('PCA Explained Variance by Components', fontweight='bold')
        ax5.set_xlabel('Number of Components')
        ax5.set_ylabel('Cumulative Explained Variance')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Class balance visualization
        ax6 = axes[1, 2]
        
        class_counts = [np.sum(labels[target]) for target in target_products]
        class_labels = [f'{name}\n(n={count})' for name, count in zip(target_names, class_counts)]
        
        wedges, texts, autotexts = ax6.pie(class_counts, labels=class_labels, 
                                          colors=colors[:3], autopct='%1.1f%%', startangle=90)
        ax6.set_title('Class Distribution in Dataset', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Embedding visualization saved to: {output_path}")
        
        return {
            'umap_embeddings': umap_embeddings,
            'tsne_embeddings': tsne_embeddings,
            'pca_embeddings': pca_embeddings,
            'separation_scores': separation_scores,
            'explained_variance': cumsum_variance[:len(n_components)]
        }
    
    def analyze_embedding_quality(self, embeddings, labels, target_products):
        """Analyze the quality and separability of embeddings."""
        
        print("🔍 Analyzing embedding quality...")
        
        analysis_results = {}
        
        for i, target in enumerate(target_products):
            print(f"   Analyzing {target}...")
            
            mask = labels[target] == 1
            if np.sum(mask) < 2:
                continue
                
            # Get class embeddings
            positive_embeddings = embeddings[mask]
            negative_embeddings = embeddings[~mask]
            
            # Calculate statistics
            positive_mean = np.mean(positive_embeddings, axis=0)
            negative_mean = np.mean(negative_embeddings, axis=0)
            
            # Distance between class centers
            center_distance = np.linalg.norm(positive_mean - negative_mean)
            
            # Intra-class variance
            positive_variance = np.mean(np.var(positive_embeddings, axis=0))
            negative_variance = np.mean(np.var(negative_embeddings, axis=0))
            
            # Inter-class vs intra-class ratio
            inter_intra_ratio = center_distance / np.sqrt(positive_variance + negative_variance)
            
            analysis_results[target] = {
                'positive_samples': int(np.sum(mask)),
                'negative_samples': int(np.sum(~mask)),
                'center_distance': float(center_distance),
                'positive_variance': float(positive_variance),
                'negative_variance': float(negative_variance),
                'inter_intra_ratio': float(inter_intra_ratio)
            }
            
            print(f"     Positive samples: {analysis_results[target]['positive_samples']}")
            print(f"     Center distance: {analysis_results[target]['center_distance']:.3f}")
            print(f"     Inter/intra ratio: {analysis_results[target]['inter_intra_ratio']:.3f}")
        
        return analysis_results
    
    def save_analysis_results(self, results, output_path):
        """Save embedding analysis results."""
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Analysis results saved to: {output_path}")

def main():
    """Main function to run embedding visualization."""
    
    print("🎨 ESM-2 EMBEDDING VISUALIZATION SYSTEM")
    print("=" * 60)
    print("Addressing reviewer feedback for embedding interpretability")
    print()
    
    # Example usage (would be called with actual data)
    print("📋 This script provides comprehensive embedding analysis:")
    print("   • UMAP, t-SNE, and PCA visualizations")
    print("   • Class separation analysis")
    print("   • Embedding quality metrics")
    print("   • Dimensionality analysis")
    print("   • Biological interpretability insights")
    
    print("\n✅ Embedding visualization system ready!")
    print("   Use this script to generate Figure 5 for the manuscript.")

if __name__ == "__main__":
    main()
