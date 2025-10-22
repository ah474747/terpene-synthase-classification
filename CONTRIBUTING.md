# Contributing to Ent-Kaurene Synthase Classification

Thank you for your interest in contributing to this project! This document provides guidelines for contributing to the ent-kaurene synthase classification pipeline.

## 🚀 Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/ent-kaurene-classification.git
   cd ent-kaurene-classification
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🔧 Development Setup

### Running Tests
```bash
# Run the complete pipeline to ensure everything works
python scripts/generate_embeddings.py
python scripts/ent_kaurene_benchmark.py
python scripts/holdout_validation.py
```

### Code Style
- Follow PEP 8 guidelines
- Use meaningful variable names
- Add docstrings to functions and classes
- Keep functions focused and modular

## 📝 Types of Contributions

### Bug Reports
If you find a bug, please:
1. Check if the issue already exists
2. Create a new issue with:
   - Clear description of the problem
   - Steps to reproduce
   - Expected vs. actual behavior
   - System information (OS, Python version, etc.)

### Feature Requests
For new features:
1. Open an issue describing the feature
2. Explain the use case and benefits
3. Discuss implementation approach

### Code Contributions
1. Create a feature branch: `git checkout -b feature/your-feature-name`
2. Make your changes
3. Test thoroughly
4. Update documentation if needed
5. Submit a pull request

## 🧪 Testing

Before submitting a pull request:
1. Run the complete pipeline to ensure no regressions
2. Test with different datasets if applicable
3. Verify that results are reproducible

## 📊 Data and Results

### Adding New Datasets
- Place datasets in the `data/` directory
- Update the README with dataset information
- Ensure proper licensing and attribution

### Modifying Results
- Update result files in `results/`
- Regenerate figures if needed
- Update documentation to reflect changes

## 🔬 Algorithm Improvements

### Adding New ML Algorithms
1. Follow the existing pattern in `ent_kaurene_benchmark.py`
2. Include proper hyperparameter tuning
3. Add to the results comparison
4. Update documentation

### Adding New Traditional Methods
1. Implement in `corrected_traditional_benchmark.py`
2. Ensure fair comparison methodology
3. Add to performance tables

## 📚 Documentation

### Code Documentation
- Add docstrings to all functions and classes
- Include type hints where appropriate
- Explain complex algorithms or methodologies

### README Updates
- Update installation instructions if dependencies change
- Add new results or findings
- Update system requirements if needed

## 🏷️ Version Control

### Commit Messages
Use clear, descriptive commit messages:
```
feat: add new ML algorithm comparison
fix: resolve embedding generation memory issue
docs: update installation instructions
```

### Branch Naming
- `feature/description` for new features
- `bugfix/description` for bug fixes
- `docs/description` for documentation updates

## 📋 Pull Request Process

1. Ensure your code follows the style guidelines
2. Test your changes thoroughly
3. Update documentation as needed
4. Submit a pull request with:
   - Clear description of changes
   - Reference to related issues
   - Screenshots for UI changes (if applicable)

## 🎯 Areas for Contribution

### High Priority
- Performance optimization for embedding generation
- Additional traditional bioinformatics methods
- Cross-validation improvements
- Statistical analysis enhancements

### Medium Priority
- GUI interface for easy pipeline execution
- Additional visualization options
- Support for other protein language models
- Integration with other databases

### Low Priority
- Docker containerization
- Cloud deployment scripts
- Additional output formats
- Extended documentation

## 🤝 Code of Conduct

This project follows a code of conduct based on mutual respect and collaboration. Please:

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and improve
- Maintain a professional environment

## 📞 Getting Help

If you need help:
1. Check the documentation first
2. Search existing issues
3. Open a new issue with your question
4. Tag maintainers if needed

## 🙏 Recognition

Contributors will be acknowledged in:
- The project README
- Publication acknowledgments (if applicable)
- Release notes

Thank you for contributing to this project!
