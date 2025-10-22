# Installation Guide

This guide provides detailed installation instructions for the Ent-Kaurene Synthase Classification pipeline.

## 🖥️ System Requirements

### Minimum Requirements
- **Operating System**: Linux, macOS, or Windows
- **Python**: 3.8 or higher
- **RAM**: 8 GB (32 GB recommended for optimal performance)
- **Storage**: 2 GB free space
- **CPU**: Multi-core processor (embedding generation is CPU-intensive)

### Recommended Requirements
- **RAM**: 32 GB or more
- **CPU**: 8+ cores for faster embedding generation
- **GPU**: Optional, for faster ESM-2 inference (CUDA-compatible)

## 🐍 Python Installation

### Option 1: Using Anaconda (Recommended)

1. Download and install [Anaconda](https://www.anaconda.com/products/distribution)
2. Create a new environment:
   ```bash
   conda create -n ent-kaurene python=3.9
   conda activate ent-kaurene
   ```

### Option 2: Using pip

1. Ensure you have Python 3.8+ installed
2. Create a virtual environment:
   ```bash
   python -m venv ent-kaurene-env
   source ent-kaurene-env/bin/activate  # On Windows: ent-kaurene-env\Scripts\activate
   ```

## 📦 Package Installation

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/your-username/ent-kaurene-classification.git
cd ent-kaurene-classification

# Install dependencies
pip install -r requirements.txt
```

### GPU Support (Optional)

If you have a CUDA-compatible GPU and want faster ESM-2 inference:

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

## 🔧 Verification

### Test Installation

Run the following to verify everything is installed correctly:

```bash
python -c "import torch, transformers, xgboost, sklearn, pandas, numpy; print('All packages imported successfully!')"
```

### Test Pipeline

Run a quick test to ensure the pipeline works:

```bash
# Test dataset loading
python -c "
import pandas as pd
df = pd.read_csv('data/ent_kaurene_binary_dataset.csv')
print(f'Dataset loaded: {len(df)} sequences')
print('Installation successful!')
"
```

## 🚀 Quick Start

Once installed, you can run the complete pipeline:

```bash
# 1. Generate ESM-2 embeddings (requires ~36 minutes on CPU)
python scripts/generate_embeddings.py

# 2. Run ML benchmark
python scripts/ent_kaurene_benchmark.py

# 3. Run hold-out validation
python scripts/holdout_validation.py
```

## 🐳 Docker Installation (Alternative)

If you prefer using Docker:

```bash
# Build the Docker image
docker build -t ent-kaurene-classification .

# Run the pipeline
docker run -v $(pwd)/data:/app/data -v $(pwd)/results:/app/results ent-kaurene-classification
```

## 🔍 Troubleshooting

### Common Issues

#### 1. Memory Issues
If you encounter memory errors during embedding generation:
- Reduce batch size in `generate_embeddings.py`
- Use a machine with more RAM
- Consider using GPU acceleration

#### 2. Package Conflicts
If you have package conflicts:
```bash
# Create a fresh environment
conda create -n ent-kaurene-fresh python=3.9
conda activate ent-kaurene-fresh
pip install -r requirements.txt
```

#### 3. CUDA Issues
If CUDA installation fails:
- Check your GPU compatibility
- Install CPU-only version: `pip install torch torchvision torchaudio`
- Verify CUDA version compatibility

#### 4. Slow Performance
For faster performance:
- Use GPU acceleration
- Increase batch size (if memory allows)
- Use a machine with more CPU cores

### Getting Help

If you encounter issues:
1. Check the [troubleshooting section](#troubleshooting)
2. Search existing [GitHub issues](https://github.com/your-username/ent-kaurene-classification/issues)
3. Create a new issue with:
   - Your system information
   - Error messages
   - Steps to reproduce

## 📊 Performance Benchmarks

### Expected Runtime (CPU)
- ESM-2 embedding generation: ~36 minutes (1,788 sequences)
- ML benchmark: ~10 minutes
- Hold-out validation: ~2 minutes
- Traditional methods: ~5 minutes

### Expected Runtime (GPU)
- ESM-2 embedding generation: ~10 minutes (with CUDA)
- Other steps remain the same

## 🔄 Updates

To update the repository:

```bash
git pull origin main
pip install -r requirements.txt --upgrade
```

## 📝 Notes

- The pipeline is designed to work on CPU by default
- GPU acceleration is optional but recommended for faster embedding generation
- All results are reproducible with the same random seeds
- The dataset is included in the repository for convenience

## 🆘 Support

For installation support:
- Check the troubleshooting section above
- Open an issue on GitHub
- Contact the maintainers
