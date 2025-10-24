# Changelog

All notable changes to this project will be documented in this file.

## [v0.4.0] - 2025-10-24

### Major Addition: Novel Germacrene Synthase Discovery from UniProt

**Discovered 38 high-confidence novel germacrene synthase candidates from 5,000 UniProt sequences**

#### New Features
- **Real-World Application Workflow**: Complete 3-step pipeline for discovering novel enzymes from UniProt
  - Step 1: Download sequences with MARTS-DB exclusion (`step1_download_uniprot_tps.py`)
  - Step 2: Generate ESM-2 embeddings on Google Colab (`step2_generate_embeddings_colab.ipynb`)
  - Step 3: Generate confidence-ranked predictions locally (`step3_predict_germacrene.py`)
- **Novel Discoveries**: 38 high-confidence germacrene synthases (>0.80 confidence)
  - Top hit: *Trichormus variabilis* cyanobacterium (98.7% confidence)
  - 8/10 top predictions from *Streptomyces* species
  - Demonstrated cross-taxonomic generalization (plant training → microbial prediction)
- **Data Leakage Prevention**: Rigorous exclusion of all 1,262 MARTS-DB training sequences
- **Google Colab Integration**: GPU-accelerated workflow for computationally intensive tasks

#### Manuscript Updates
- **Authorship**: Added Cursor AI (first author), Google Gemini (second author), Andrew Horwitz (corresponding author)
- **New Objectives**: Added AI-assisted methodology evaluation as third objective
- **Novel Discovery Section**: Comprehensive results with biological plausibility assessment (Table 3)
- **AI-Assisted Methodology Discussion**: Analysis of AI democratization of ML for biologists
- **Peer Review Section**: Documentation of ChatGPT review and implemented revisions
- **Enhanced Methods**: Added computational infrastructure details (Google Colab, GPU specs)

#### Documentation
- **PREDICTION_WORKFLOW_README.md**: Comprehensive 25-page guide to discovery workflow
- **QUICK_START.md**: Condensed quick reference for 3-step workflow
- **DATASET_NOTES.md**: Documentation of UniProt download strategy and quality verification

#### Technical Improvements
- Fixed critical data leakage bug in initial UniProt download
- Implemented family-based UniProt query (`family:"terpene synthase family"`)
- Added automatic exclusion filtering during sequence download
- Created confidence-ranked output files with multiple thresholds
- Added comprehensive visualization of confidence distribution

#### Results Summary
- **Total analyzed**: 5,000 novel terpene synthase sequences
- **Predicted germacrene synthases**: 171 (3.4%)
- **High confidence (>0.80)**: 38 sequences
- **Ultra-high confidence (>0.90)**: 6 sequences
- **Mean confidence**: 0.072 (appropriate selectivity)

#### Biological Insights
- Model successfully generalized from plant-dominated training set to microbial enzymes
- Many *Streptomyces* predictions have literature-confirmed germacrene/germacradienol associations
- Identified potential novel cyanobacterial germacrene synthase
- Demonstrated functional convergence across distant phylogenetic groups

---

## [v0.3.1] - 2025-10-22

### Reviewer Feedback Implementation

#### Major Revisions
- Added comprehensive requirements.txt with exact package versions
- Implemented bootstrap confidence intervals (95% CI) for all metrics
- Enhanced statistical reporting with mean ± std across CV folds
- Added UMAP/t-SNE visualizations of embedding space
- Created complete reproducibility pipeline scripts
- Corrected traditional methods reporting (germacrene-only benchmark)

#### Minor Revisions
- Standardized figure labels and error bars
- Enhanced README with preprocessing documentation
- Added runtime expectations for all steps
- Improved manuscript organization

---

## [v0.3.0] - 2025-10-20

### Multi-Product Benchmark Study

#### Features
- Benchmarked 7 ML algorithms on 3 terpene products
- Comprehensive comparison with traditional bioinformatics methods
- Hold-out validation for unbiased performance assessment
- Publication-quality figures and visualizations

#### Results
- Germacrene: F1 = 0.591 (SVM-RBF), AUC-PR = 0.680 (XGBoost)
- Pinene: F1 = 0.663 (KNN), AUC-PR = 0.711 (KNN)
- Myrcene: F1 = 0.439 (XGBoost), AUC-PR = 0.537 (XGBoost)
- Traditional methods significantly underperform (F1 = 0.347 for best)

---

## [v0.2.0] - 2025-10-15

### ESM-2 Embedding Integration

#### Features
- Integrated ESM-2 protein language model embeddings
- Implemented 7-algorithm ML benchmark
- Added class imbalance handling strategies
- Created comprehensive evaluation metrics

---

## [v0.1.0] - 2025-10-10

### Initial Release

#### Features
- MARTS-DB dataset integration and cleaning
- Data deduplication and product simplification
- Binary classification framework
- Basic ML pipeline with XGBoost and Random Forest

