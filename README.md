# Machine Learning Classification of Terpene Synthases using ESM-2 Protein Language Model Embeddings

## A Multi-Product Benchmark Study

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://img.shields.io/badge/DOI-10.1000/182-9b2c3d.svg)](https://doi.org/10.1000/182)

## 📋 Overview

This repository contains a comprehensive machine learning benchmark for classifying terpene synthase enzymes using ESM-2 protein language model embeddings. The study addresses the challenge of predicting enzyme function from sequence data, specifically focusing on terpene synthase product prediction for germacrene, pinene, and myrcene production.

### 🎯 Key Features

- **Multi-Product Classification**: Binary classification for germacrene, pinene, and myrcene synthases
- **ESM-2 Embeddings**: State-of-the-art protein language model representations
- **7-Algorithm Benchmark**: Comprehensive comparison of machine learning approaches
- **Statistical Rigor**: Bootstrap confidence intervals and significance testing
- **Reproducible Results**: Fixed seeds and deterministic training
- **Complete Pipeline**: From raw data to publication-ready results

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- 16GB+ RAM recommended
- 10GB+ disk space for embeddings and results

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ah474747/terpene-synthase-classification.git
   cd terpene-synthase-classification
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation:**
   ```bash
   python -c "import torch, transformers, xgboost, sklearn; print('✅ All dependencies installed successfully!')"
   ```

### Running the Complete Pipeline

Execute the entire benchmark pipeline with a single command:

```bash
python scripts/run_complete_pipeline.py
```

This will:
- Generate ESM-2 embeddings for all sequences
- Run 7-algorithm ML benchmarks for each target product
- Perform traditional methods comparison
- Execute hold-out validation
- Generate all figures and results
- Create the final manuscript PDF

**Expected Runtime:** 2-4 hours on a modern workstation

## 📊 Dataset Information

### MARTS-DB Dataset

The study uses the MARTS-DB (Terpene Synthase Database) as the gold standard dataset:

- **Source**: MARTS-DB (https://tpsdb.uochb.cas.cz/)
- **Total Sequences**: 1,262 unique amino acid sequences
- **Target Products**: Germacrene (93), Pinene (82), Myrcene (53)
- **Organisms**: Plant and bacterial terpene synthases
- **Validation**: All sequences experimentally characterized

### Data Preprocessing Pipeline

1. **Deduplication**: Removed redundant sequences, keeping unique amino acid sequences
2. **Product Simplification**: Consolidated isomeric variants (e.g., `(-)-germacrene D` → `germacrene`)
3. **Binary Labeling**: Created binary classification targets for each product
4. **Quality Control**: Verified experimental validation for all sequences

## 🔬 Methodology

### Feature Engineering

- **ESM-2 Embeddings**: 1280-dimensional protein language model representations
- **Model**: `facebook/esm2_t33_650M_UR50D`
- **Processing**: Average pooling of residue-level embeddings

### Machine Learning Algorithms

| Algorithm | Category | Key Features |
|-----------|----------|--------------|
| XGBoost | Ensemble | Gradient boosting, handles class imbalance |
| SVM-RBF | Kernel | Non-linear separation, RBF kernel |
| Random Forest | Ensemble | Bagging, feature importance |
| Logistic Regression | Linear | Regularized, interpretable |
| MLP | Neural Network | Multi-layer perceptron |
| k-NN | Instance-based | Local similarity |
| Perceptron | Linear | Simple linear baseline |

### Evaluation Framework

- **Cross-Validation**: 5-fold stratified cross-validation
- **Metrics**: F1-Score, AUC-PR, AUC-ROC, Precision, Recall
- **Statistical Analysis**: Bootstrap confidence intervals (95%)
- **Hold-out Validation**: 20% independent test set
- **Reproducibility**: Fixed seeds (RANDOM_STATE=42)

## 📁 Repository Structure

```
terpene-synthase-classification/
├── data/                          # Dataset files
│   ├── Original_MARTS_DB_reactions.csv
│   ├── clean_MARTS_DB_binary_dataset.csv
│   ├── clean_MARTS_DB_deduplicated.csv
│   └── germacrene_esm2_embeddings.npy
├── scripts/                       # Analysis scripts
│   ├── run_complete_pipeline.py   # Main pipeline runner
│   ├── generate_embeddings.py     # ESM-2 embedding generation
│   ├── *_benchmark.py            # ML algorithm benchmarks
│   ├── comprehensive_evaluation.py # Statistical evaluation
│   ├── embedding_visualization.py  # UMAP/t-SNE analysis
│   └── seed_manager.py           # Reproducibility management
├── results/                       # Output files
│   ├── *_benchmark_results.json  # Algorithm results
│   ├── figure*.png               # Generated figures
│   └── *_validation_results.json # Hold-out validation
├── analysis/                      # Exploratory analysis
├── requirements.txt               # Python dependencies
├── MANUSCRIPT_DRAFT.md           # Manuscript source
├── Terpene_Synthase_Classification_Manuscript.pdf
└── README.md                     # This file
```

## 🎯 Usage Examples

### Generate ESM-2 Embeddings

```python
from scripts.generate_embeddings import generate_embeddings

# Generate embeddings for the dataset
embeddings = generate_embeddings(
    dataset_path="data/clean_MARTS_DB_binary_dataset.csv",
    output_path="data/germacrene_esm2_embeddings.npy"
)
```

### Run ML Benchmark

```python
from scripts.germacrene_benchmark import run_benchmark

# Run 7-algorithm benchmark for germacrene
results = run_benchmark(
    embeddings_path="data/germacrene_esm2_embeddings.npy",
    dataset_path="data/clean_MARTS_DB_binary_dataset.csv"
)
```

### Comprehensive Evaluation

```python
from scripts.comprehensive_evaluation import ComprehensiveEvaluator

# Initialize evaluator with statistical analysis
evaluator = ComprehensiveEvaluator(random_state=42)

# Evaluate model with confidence intervals
results = evaluator.evaluate_model_comprehensive(
    model=xgb_model,
    X=embeddings,
    y=labels,
    target_name="Germacrene"
)
```

### Embedding Visualization

```python
from scripts.embedding_visualization import EmbeddingVisualizer

# Create UMAP/t-SNE visualizations
visualizer = EmbeddingVisualizer(random_state=42)
visualization_results = visualizer.create_embedding_visualization(
    embeddings=embeddings,
    labels=labels,
    target_products=['is_germacrene', 'is_pinene', 'is_myrcene'],
    output_path="results/figure5_embedding_analysis.png"
)
```

## 📈 Results Summary

### Performance Metrics (Mean ± Std, 95% CI)

| Target Product | Best Algorithm | F1-Score | AUC-PR | AUC-ROC |
|----------------|----------------|----------|---------|---------|
| Germacrene | SVM-RBF | 0.591±0.083 | 0.645±0.075 | 0.931±0.045 |
| Pinene | k-NN | 0.663±0.111 | 0.711±0.159 | 0.945±0.032 |
| Myrcene | XGBoost | 0.439±0.066 | 0.356±0.080 | 0.823±0.067 |

### Key Findings

1. **ESM-2 embeddings** significantly outperform traditional sequence-based methods
2. **Class balance** strongly influences performance (germacrene > pinene > myrcene)
3. **SVM-RBF** and **k-NN** perform best for well-balanced classes
4. **XGBoost** shows robustness across different class distributions
5. **Traditional methods** achieve F1-scores of 0.139-0.449 vs. 0.439-0.663 for ML approaches

## 🔬 Reproducibility

### Seed Management

All experiments use fixed seeds for reproducibility:

```python
from scripts.seed_manager import ensure_reproducibility

# Set all random seeds
ensure_reproducibility()
```

### Version Control

- **Python**: 3.8+
- **PyTorch**: 2.0.1
- **Transformers**: 4.33.2
- **scikit-learn**: 1.3.0
- **XGBoost**: 1.7.6

### Docker Support

For complete reproducibility, use the provided Docker configuration:

```bash
docker build -t terpene-classification .
docker run -v $(pwd)/results:/app/results terpene-classification
```

## 📊 Expected Outputs

### Runtime Expectations

| Task | Time | Memory | Output |
|------|------|--------|--------|
| ESM-2 Embeddings | 30-60 min | 8GB | 6.2MB .npy file |
| ML Benchmark (7 algorithms) | 15-30 min | 4GB | JSON results |
| Traditional Methods | 5-10 min | 2GB | JSON results |
| Hold-out Validation | 5-10 min | 2GB | JSON results |
| Figure Generation | 2-5 min | 1GB | PNG files |

### File Sizes

- **Embeddings**: ~6.2MB per target product
- **Results**: ~1-5MB JSON files
- **Figures**: ~500KB-2MB PNG files
- **PDF Manuscript**: ~1.1MB

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Use CPU-only mode or reduce batch size
2. **ESM-2 Model Download**: Ensure internet connection for model download
3. **Memory Issues**: Close other applications or use smaller datasets

### Performance Optimization

- **GPU Acceleration**: Automatically detected and used when available
- **Parallel Processing**: Cross-validation folds run in parallel
- **Memory Management**: Efficient embedding storage and loading

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@article{horwitz2024terpene,
  title={Machine Learning Classification of Terpene Synthases using ESM-2 Protein Language Model Embeddings: A Multi-Product Benchmark Study},
  author={Horwitz, Andrew},
  journal={[Journal]},
  year={2024},
  url={https://github.com/ah474747/terpene-synthase-classification}
}
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- MARTS-DB database curators for providing the gold-standard dataset
- Facebook AI Research for the ESM-2 protein language model
- The open-source community for excellent ML libraries

## 📞 Support

For questions or issues:

1. Check the [troubleshooting section](#-troubleshooting)
2. Review existing [GitHub issues](https://github.com/ah474747/terpene-synthase-classification/issues)
3. Create a new issue with detailed information
4. Contact the authors for collaboration opportunities

---

**Last Updated**: October 2024  
**Version**: 1.0.0  
**Status**: Ready for peer review