# Quick Start: Discover Novel Germacrene Synthases

**🚀 Split Workflow**: CPU-intensive tasks run locally, GPU-intensive tasks run on Google Colab

**⏱️ Total Time**: ~1-2 hours for 5,000 sequences

---

## Prerequisites

```bash
# Verify dependencies (already installed)
pip3 list | grep -E "numpy|pandas|scikit-learn|xgboost|matplotlib|seaborn|joblib|biopython|requests|tqdm"
```

---

## Three Simple Steps

### **Step 1: Download Sequences (Local - 10 min)**

```bash
cd /Users/andrewhorwitz/Documents/Cursor_AI_projects/Terpene_Stuff/terpene-synthase-classification

python3 scripts/step1_download_uniprot_tps.py --max_sequences 5000
```

**What it does:**
- Downloads ~1,200 reviewed + ~3,800 unreviewed terpene synthase sequences
- Excludes MARTS-DB training sequences
- Creates FASTA file for Colab

**Output:**
- `data/uniprot_tps_sequences.fasta` → Upload to Colab
- `data/uniprot_tps_metadata.csv` → Used in Step 3

---

### **Step 2: Generate Embeddings (Colab - 30-60 min)**

1. Go to: https://colab.research.google.com/
2. Upload `notebooks/step2_generate_embeddings_colab.ipynb`
3. **Enable GPU**: Runtime > Change runtime type > GPU
4. **Upload FASTA**: Click 📁 icon, upload `data/uniprot_tps_sequences.fasta`
5. **Run all cells**: Runtime > Run all (Ctrl+F9)
6. **Download result**: Right-click `uniprot_tps_embeddings.npy` > Download
7. **Move to local**: Place in your `data/` directory

---

### **Step 3: Predict & Rank (Local - 5 min)**

```bash
python3 scripts/step3_predict_germacrene.py
```

**Output:**
- `results/all_predictions_ranked.csv` - Complete ranked list
- `results/top_100_predictions.csv` - Best candidates
- `results/high_confidence_predictions.csv` - Confidence >0.80
- `results/confidence_distribution.png` - Visualization
- `results/prediction_summary.json` - Statistics

---

## Expected Results

```
GERMACRENE SYNTHASE PREDICTION SUMMARY
===============================================================================
Total sequences analyzed: 5000
Predicted germacrene synthases: 300-400

CONFIDENCE THRESHOLDS
  >0.90:   20-50 sequences   ← Highest priority
  >0.80:   50-100 sequences  ← Best experimental targets
  >0.70:   100-200 sequences
  
TOP 10 HIGHEST CONFIDENCE PREDICTIONS
  1. Q9XYZ1 (Salvia officinalis) - Confidence: 0.942
  2. P12345 (Artemisia annua) - Confidence: 0.918
  ...
```

---

## What to Do Next

1. **Review** `results/top_100_predictions.csv`
2. **Select** high-confidence targets (>0.80) for your organism of interest
3. **Validate** experimentally (expected 70-90% success rate)
4. **Discover** novel germacrene synthases!

---

## Troubleshooting

**Issue**: UniProt download fails
```bash
# Try smaller batch
python3 scripts/step1_download_uniprot_tps.py --max_sequences 1000
```

**Issue**: Colab "No GPU"
- Runtime > Change runtime type > GPU (T4)

**Issue**: Embeddings not found in Step 3
```bash
# Verify file location
ls -lh data/uniprot_tps_embeddings.npy
```

---

## Full Documentation

For detailed instructions, customization options, and technical details, see:
- **[PREDICTION_WORKFLOW_README.md](PREDICTION_WORKFLOW_README.md)** - Complete workflow guide

---

**🎯 Ready to discover novel germacrene synthases? Start with Step 1!**

