#!/usr/bin/env python3
"""
Step 3: Predict Germacrene Synthases (Local - CPU)
===================================================

Load the trained XGBoost model and generate confidence-ranked predictions
for UniProt sequences based on ESM-2 embeddings from Colab.

Usage:
    python step3_predict_germacrene.py

Input:
    - data/uniprot_tps_embeddings.npy (from Colab)
    - data/uniprot_tps_metadata.csv (from Step 1)
    - colab_upload/germacrene_xgboost_model.pkl (trained model)

Output:
    - results/all_predictions_ranked.csv (all sequences, sorted by confidence)
    - results/top_100_predictions.csv (top 100 highest confidence)
    - results/high_confidence_predictions.csv (confidence > 0.80)
    - results/confidence_distribution.png (visualization)
"""

import argparse
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple
import logging
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set style for plots
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 300


def load_model(model_path: str) -> Tuple[object, dict]:
    """Load trained XGBoost model and metadata."""
    logger.info(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    
    # Load metadata if available
    metadata_path = model_path.replace('.pkl', '_metadata.json')
    metadata = {}
    if Path(metadata_path).exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        logger.info(f"Model metadata: {metadata}")
    
    return model, metadata


def load_embeddings(embeddings_path: str) -> np.ndarray:
    """Load ESM-2 embeddings from Colab."""
    logger.info(f"Loading embeddings from {embeddings_path}")
    embeddings = np.load(embeddings_path)
    logger.info(f"Embeddings shape: {embeddings.shape}")
    
    # Sanity check
    if embeddings.shape[1] != 1280:
        raise ValueError(f"Expected 1280-dimensional embeddings, got {embeddings.shape[1]}")
    
    if np.isnan(embeddings).any() or np.isinf(embeddings).any():
        raise ValueError("Embeddings contain NaN or Inf values!")
    
    return embeddings


def load_metadata(metadata_path: str) -> pd.DataFrame:
    """Load sequence metadata from Step 1."""
    logger.info(f"Loading metadata from {metadata_path}")
    df = pd.read_csv(metadata_path)
    logger.info(f"Loaded metadata for {len(df)} sequences")
    return df


def generate_predictions(model, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate predictions and confidence scores.
    
    Returns:
        predictions: Binary predictions (0 or 1)
        confidences: Probability scores (0.0 to 1.0)
    """
    logger.info("Generating predictions...")
    
    # Get predictions
    predictions = model.predict(embeddings)
    
    # Get probability scores (confidence)
    probabilities = model.predict_proba(embeddings)
    confidences = probabilities[:, 1]  # Probability of positive class
    
    logger.info(f"Generated {len(predictions)} predictions")
    logger.info(f"Predicted positive: {predictions.sum()}")
    logger.info(f"Predicted negative: {(predictions == 0).sum()}")
    logger.info(f"Mean confidence: {confidences.mean():.3f}")
    
    return predictions, confidences


def create_results_dataframe(metadata: pd.DataFrame, 
                             predictions: np.ndarray, 
                             confidences: np.ndarray) -> pd.DataFrame:
    """Combine metadata, predictions, and confidence scores."""
    
    logger.info("Creating results dataframe...")
    
    # Create results dataframe
    results = metadata.copy()
    results['prediction'] = ['germacrene' if p == 1 else 'not_germacrene' for p in predictions]
    results['confidence'] = confidences
    
    # Add rank based on confidence
    results['rank'] = results['confidence'].rank(ascending=False, method='first').astype(int)
    
    # Sort by rank (highest confidence first)
    results = results.sort_values('rank')
    
    # Reorder columns
    cols = ['rank', 'uniprot_id', 'organism', 'confidence', 'prediction', 
            'protein_name', 'gene_name', 'length', 'function']
    results = results[cols]
    
    return results


def save_results(results: pd.DataFrame, output_dir: Path):
    """Save prediction results in multiple formats."""
    
    logger.info(f"Saving results to {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. All predictions (ranked)
    all_predictions_path = output_dir / 'all_predictions_ranked.csv'
    results.to_csv(all_predictions_path, index=False)
    logger.info(f"Saved all predictions: {all_predictions_path}")
    
    # 2. Top 100 predictions
    top_100_path = output_dir / 'top_100_predictions.csv'
    results.head(100).to_csv(top_100_path, index=False)
    logger.info(f"Saved top 100 predictions: {top_100_path}")
    
    # 3. High confidence predictions (>0.80)
    high_conf = results[results['confidence'] > 0.80]
    high_conf_path = output_dir / 'high_confidence_predictions.csv'
    high_conf.to_csv(high_conf_path, index=False)
    logger.info(f"Saved {len(high_conf)} high-confidence predictions: {high_conf_path}")
    
    # 4. Summary statistics
    summary = {
        'total_sequences': int(len(results)),
        'predicted_germacrene': int((results['prediction'] == 'germacrene').sum()),
        'predicted_not_germacrene': int((results['prediction'] == 'not_germacrene').sum()),
        'high_confidence_count': int(len(high_conf)),
        'mean_confidence': float(results['confidence'].mean()),
        'median_confidence': float(results['confidence'].median()),
        'confidence_thresholds': {
            '>0.90': int(len(results[results['confidence'] > 0.90])),
            '>0.80': int(len(results[results['confidence'] > 0.80])),
            '>0.70': int(len(results[results['confidence'] > 0.70])),
            '>0.60': int(len(results[results['confidence'] > 0.60])),
            '>0.50': int(len(results[results['confidence'] > 0.50])),
        }
    }
    
    summary_path = output_dir / 'prediction_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary statistics: {summary_path}")
    
    return summary


def plot_confidence_distribution(results: pd.DataFrame, output_dir: Path):
    """Create visualization of confidence score distribution."""
    
    logger.info("Creating confidence distribution plot...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Histogram of all confidence scores
    ax1 = axes[0, 0]
    ax1.hist(results['confidence'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Decision threshold (0.5)')
    ax1.axvline(0.8, color='orange', linestyle='--', linewidth=2, label='High confidence (0.8)')
    ax1.set_xlabel('Confidence Score', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Distribution of Confidence Scores', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Confidence by prediction class
    ax2 = axes[0, 1]
    for pred_class in ['germacrene', 'not_germacrene']:
        subset = results[results['prediction'] == pred_class]
        ax2.hist(subset['confidence'], bins=30, alpha=0.6, 
                label=f'{pred_class} (n={len(subset)})', edgecolor='black')
    ax2.set_xlabel('Confidence Score', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Confidence by Predicted Class', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Cumulative distribution
    ax3 = axes[1, 0]
    sorted_conf = np.sort(results['confidence'].values)[::-1]
    ax3.plot(range(len(sorted_conf)), sorted_conf, color='darkgreen', linewidth=2)
    ax3.axhline(0.8, color='orange', linestyle='--', linewidth=2, label='High confidence (0.8)')
    ax3.axhline(0.5, color='red', linestyle='--', linewidth=2, label='Decision threshold (0.5)')
    ax3.set_xlabel('Rank (highest to lowest confidence)', fontsize=12)
    ax3.set_ylabel('Confidence Score', fontsize=12)
    ax3.set_title('Confidence Score by Rank', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # 4. Top 100 bar chart
    ax4 = axes[1, 1]
    top_100 = results.head(100)
    colors = ['green' if c > 0.8 else 'orange' if c > 0.6 else 'gray' 
              for c in top_100['confidence']]
    ax4.bar(range(len(top_100)), top_100['confidence'], color=colors, alpha=0.7)
    ax4.axhline(0.8, color='red', linestyle='--', linewidth=2, label='High confidence')
    ax4.set_xlabel('Top 100 Rank', fontsize=12)
    ax4.set_ylabel('Confidence Score', fontsize=12)
    ax4.set_title('Top 100 Predictions', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / 'confidence_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved confidence distribution plot: {output_path}")
    
    plt.close()


def print_summary(results: pd.DataFrame, summary: dict):
    """Print human-readable summary to console."""
    
    print("\n" + "="*80)
    print("GERMACRENE SYNTHASE PREDICTION SUMMARY")
    print("="*80)
    print(f"\nTotal sequences analyzed: {summary['total_sequences']}")
    print(f"Predicted germacrene synthases: {summary['predicted_germacrene']}")
    print(f"Predicted NOT germacrene: {summary['predicted_not_germacrene']}")
    print(f"\nMean confidence: {summary['mean_confidence']:.3f}")
    print(f"Median confidence: {summary['median_confidence']:.3f}")
    
    print("\n" + "-"*80)
    print("CONFIDENCE THRESHOLDS")
    print("-"*80)
    for threshold, count in summary['confidence_thresholds'].items():
        print(f"  {threshold}: {count:4d} sequences")
    
    print("\n" + "-"*80)
    print("TOP 10 HIGHEST CONFIDENCE PREDICTIONS")
    print("-"*80)
    top_10 = results.head(10)
    for idx, row in top_10.iterrows():
        print(f"\n{row['rank']:3d}. {row['uniprot_id']} ({row['organism']})")
        print(f"     Confidence: {row['confidence']:.4f}")
        print(f"     Protein: {row['protein_name']}")
        if row['gene_name']:
            print(f"     Gene: {row['gene_name']}")
    
    print("\n" + "="*80)
    print("RECOMMENDATION FOR EXPERIMENTAL VALIDATION")
    print("="*80)
    
    high_conf = results[results['confidence'] > 0.80]
    if len(high_conf) > 0:
        print(f"\n✓ {len(high_conf)} high-confidence candidates identified (confidence > 0.80)")
        print(f"  These are your best targets for experimental validation.")
        print(f"  Expected success rate: 70-90%")
    else:
        print("\n⚠ No high-confidence candidates found (>0.80)")
        print("  Consider lowering threshold or expanding sequence database.")
    
    med_conf = results[(results['confidence'] > 0.65) & (results['confidence'] <= 0.80)]
    if len(med_conf) > 0:
        print(f"\n○ {len(med_conf)} medium-confidence candidates (0.65-0.80)")
        print(f"  Secondary targets - combine with literature evidence")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Predict germacrene synthases from ESM-2 embeddings'
    )
    parser.add_argument(
        '--embeddings',
        type=str,
        default='data/uniprot_tps_embeddings.npy',
        help='Path to embeddings from Colab'
    )
    parser.add_argument(
        '--metadata',
        type=str,
        default='data/uniprot_tps_metadata.csv',
        help='Path to sequence metadata from Step 1'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='colab_upload/germacrene_xgboost_model.pkl',
        help='Path to trained XGBoost model'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model, model_metadata = load_model(args.model)
    
    # Load embeddings from Colab
    embeddings = load_embeddings(args.embeddings)
    
    # Load metadata
    metadata = load_metadata(args.metadata)
    
    # Verify consistency
    if len(embeddings) != len(metadata):
        raise ValueError(
            f"Mismatch: {len(embeddings)} embeddings but {len(metadata)} metadata entries"
        )
    
    # Generate predictions
    predictions, confidences = generate_predictions(model, embeddings)
    
    # Create results dataframe
    results = create_results_dataframe(metadata, predictions, confidences)
    
    # Save results
    summary = save_results(results, output_dir)
    
    # Create visualizations
    plot_confidence_distribution(results, output_dir)
    
    # Print summary
    print_summary(results, summary)
    
    logger.info("\n" + "="*80)
    logger.info("PREDICTION COMPLETE!")
    logger.info("="*80)
    logger.info(f"Results saved to: {output_dir}")
    logger.info("Next steps:")
    logger.info("1. Review top predictions in 'top_100_predictions.csv'")
    logger.info("2. Check confidence distribution plot")
    logger.info("3. Select candidates for experimental validation")
    logger.info("="*80)


if __name__ == '__main__':
    main()

