# Machine Learning Classification of Terpene Synthases using ESM-2 Protein Language Model Embeddings

A comprehensive benchmark study comparing machine learning approaches using ESM-2 embeddings against traditional sequence-based methods for binary classification of terpene synthases from the MARTS-DB dataset.

## Overview

This repository contains a complete pipeline for binary classification of terpene synthases using state-of-the-art protein language model embeddings. We benchmark ESM-2 embeddings combined with machine learning algorithms against traditional bioinformatics methods across three different terpene products: germacrene, pinene, and myrcene.

## Key Results

- **Germacrene Classification**: SVM-RBF achieves F1-score = 0.591, AUC-PR = 0.645
- **Pinene Classification**: KNN achieves F1-score = 0.663, AUC-PR = 0.711  
- **Myrcene Classification**: XGBoost achieves F1-score = 0.439, AUC-PR = 0.356

ESM-2 + ML approaches consistently outperform traditional methods (24-77% improvement in F1-score).

## Dataset

**Clean MARTS-DB Dataset**: 1,262 deduplicated terpene synthase sequences with verified experimental validation
- Germacrene: 93 sequences (7.4% class balance)
- Pinene: 82 sequences (6.5% class balance)  
- Myrcene: 53 sequences (4.2% class balance)

All sequences are from the MARTS-DB (Manual Annotation of the Reaction and Substrate specificity of Terpene Synthases Database) with complete experimental validation and proper data provenance.

## Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Run Complete Pipeline

```bash
python scripts/run_complete_pipeline.py
```

This will:
1. Generate ESM-2 embeddings for all sequences
2. Run 7-algorithm ML benchmarks for all three products
3. Compare against traditional methods
4. Perform hold-out validation
5. Generate comprehensive results

### Individual Components

**Generate ESM-2 Embeddings:**
```bash
python scripts/generate_germacrene_embeddings.py
```

**Run ML Benchmarks:**
```bash
python scripts/germacrene_benchmark.py
python scripts/pinene_benchmark.py  
python scripts/myrcene_benchmark.py
```

**Traditional Methods Comparison:**
```bash
python scripts/germacrene_traditional_benchmark.py
```

**Hold-out Validation:**
```bash
python scripts/germacrene_holdout_validation.py
```

## Repository Structure

```
├── data/
│   ├── clean_MARTS_DB_binary_dataset.csv      # Main dataset with binary labels
│   ├── clean_MARTS_DB_deduplicated.csv        # Deduplicated sequences
│   ├── original_MARTS_DB_reactions.csv        # Original MARTS-DB data
│   └── germacrene_esm2_embeddings.npy         # ESM-2 embeddings (1,262 × 1,280)
├── results/
│   ├── germacrene_benchmark_results.json      # ML benchmark results
│   ├── pinene_benchmark_results.json          # ML benchmark results
│   ├── myrcene_benchmark_results.json         # ML benchmark results
│   ├── germacrene_traditional_benchmark_results.json  # Traditional methods
│   └── germacrene_holdout_validation_results.json     # Hold-out validation
├── scripts/
│   ├── generate_germacrene_embeddings.py      # ESM-2 embedding generation
│   ├── germacrene_benchmark.py                # ML benchmark for germacrene
│   ├── pinene_benchmark.py                    # ML benchmark for pinene
│   ├── myrcene_benchmark.py                   # ML benchmark for myrcene
│   ├── germacrene_traditional_benchmark.py    # Traditional methods comparison
│   └── germacrene_holdout_validation.py       # Hold-out validation
├── MANUSCRIPT_DRAFT.md                        # Complete manuscript
├── README.md                                  # This file
└── requirements.txt                           # Python dependencies
```

## Methods

### Data Processing
- **Deduplication**: Sequences deduplicated by amino acid sequence with product consolidation
- **Product Simplification**: Stereoisomers consolidated (e.g., "(-)-germacrene D" → "germacrene")
- **Quality Control**: All sequences verified from MARTS-DB with experimental validation

### ESM-2 Embeddings
- **Model**: facebook/esm2_t33_650M_UR50D
- **Processing**: Batch size 8, max length 1,024 amino acids
- **Output**: 1,280-dimensional average-pooled embeddings

### Machine Learning Pipeline
- **Algorithms**: XGBoost, Random Forest, SVM-RBF, Logistic Regression, MLP, KNN, Perceptron
- **Preprocessing**: StandardScaler with class imbalance handling
- **Validation**: 5-fold stratified cross-validation
- **Hyperparameter Tuning**: Randomized search (20 iterations)
- **Metrics**: F1-score, AUC-PR, AUC-ROC, Accuracy, Precision, Recall

### Traditional Methods
- **Sequence Similarity**: Pairwise sequence identity
- **Motif-based**: Conserved terpene synthase motifs (DDXXD, NSE/DTE, RRX8W, GXGXG)
- **Length-based**: Sequence length as primary feature
- **Amino Acid Composition**: 20-dimensional AA frequency vectors

## Results Summary

| Product | Best Model | F1-Score | AUC-PR | Class Balance |
|---------|------------|----------|--------|---------------|
| Germacrene | SVM-RBF | 0.591 | 0.645 | 7.4% |
| Pinene | KNN | 0.663 | 0.711 | 6.5% |
| Myrcene | XGBoost | 0.439 | 0.356 | 4.2% |

**Key Findings:**
- ESM-2 embeddings significantly outperform traditional methods (24-77% improvement)
- Class balance strongly impacts performance
- Different algorithms excel for different products
- Robust generalization confirmed by hold-out validation

## Citation

If you use this work, please cite:

```bibtex
@article{horwitz2024terpene,
  title={Machine Learning Classification of Terpene Synthases using ESM-2 Protein Language Model Embeddings: A Multi-Product Benchmark Study},
  author={Horwitz, Andrew},
  journal={[Journal]},
  year={2024},
  url={https://github.com/ah474747/ent-kaurene-classification}
}
```

## License

MIT License - see LICENSE file for details.

## Data Availability

- **MARTS-DB**: [https://tpsdb.uochb.cas.cz/](https://tpsdb.uochb.cas.cz/)
- **ESM-2 Model**: [https://huggingface.co/facebook/esm2_t33_650M_UR50D](https://huggingface.co/facebook/esm2_t33_650M_UR50D)

## Acknowledgments

- MARTS-DB database curators for providing the gold-standard dataset
- Meta AI for the ESM-2 protein language model
- The scientific community for open-source bioinformatics tools

## Contact

For questions or collaboration, please contact: [your-email@domain.com]

---

**Note**: This repository contains only clean, verified data from MARTS-DB with complete experimental validation. All previous analyses using compromised datasets have been removed to maintain scientific rigor and reproducibility.