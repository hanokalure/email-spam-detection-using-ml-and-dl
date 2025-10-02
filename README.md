# 🚀 Email Spam Detection System

A comprehensive spam detection system with multiple machine learning models including Enhanced Transformers, SVM, and CatBoost classifiers.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Models](https://img.shields.io/badge/models-3-green.svg)](#models)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## 📋 Table of Contents

- [Features](#-features)
- [Quick Start](#-quick-start) 
- [Installation](#-installation)
- [Models](#-models)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Training](#-training)
- [Documentation](#-documentation)
- [Contributing](#-contributing)

## ✨ Features

- **Multiple ML Models**: Enhanced Transformer (99.48% spam recall), SVM, and CatBoost
- **Auto-Download**: Scripts to download pre-trained models from Google Drive
- **Interactive CLI**: Easy-to-use command-line interface for spam detection
- **Performance Reports**: Generate detailed PDF performance analysis
- **Clean Architecture**: Well-organized codebase with separate training/prediction modules
- **Cross-Platform**: Works on Windows, macOS, and Linux

## ⚡ Quick Start

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd spam-detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download models (choose one option)
# Option A - Python script
python scripts/download_models.py

# Option B - PowerShell (Windows)
.\scripts\download_models.ps1

# 4. Run spam detection
python predictors/predict_main.py
```

## 🔧 Installation

### Prerequisites
- Python 3.8 or higher
- 4GB+ RAM (for transformer models)
- GPU recommended (but not required)

### Install Dependencies

```bash
pip install torch torchvision torchaudio
pip install scikit-learn catboost joblib
pip install gdown pandas numpy
pip install reportlab toml  # For PDF reports and config
```

Or use requirements.txt:
```bash
pip install -r requirements.txt
```

## 🤖 Models

| Model | Accuracy | Size | Speed | Best For |
|-------|----------|------|-------|----------|
| **Enhanced Transformer** | 99.48% spam recall | ~172MB | 10-20ms | Modern spam patterns |
| **SVM** | ~95-98% | ~15MB | 1-5ms | Fast classification |
| **CatBoost** | ~96-97% | ~10MB | 5-15ms | Balanced performance |

### Model Download

Models are hosted on Google Drive due to their size (10MB-200MB each).

**Automatic Download:**
```bash
# Download all models
python scripts/download_models.py

# Download specific models
python scripts/download_models.py svm catboost

# List available models
python scripts/download_models.py --list
```

**Manual Download:**
1. [SVM Model (15MB)](https://drive.google.com/file/d/1Vxjz4QV3FESvm7gMeKNqklA5uARPntLv/view?usp=sharing)
2. [Enhanced Transformer (172MB)](https://drive.google.com/file/d/1kGD6Tg5JLIko0XhYPk2-WtAswj1S2Pgs/view?usp=sharing)  
3. [CatBoost Model (10MB)](https://drive.google.com/file/d/1ofS_IU9QiypgkvFqNGLjUvSdUfEi9hjO/view?usp=sharing)

Place downloaded files in the `models/` directory.

## 📖 Usage

### Interactive Mode
```bash
python predictors/predict_main.py
```
This launches an interactive CLI where you can:
- Select from available models
- Enter email text for classification
- Switch between models
- View model information

### Direct Prediction
```bash
# Using specific predictors
python predictors/predict_enhanced_transformer.py "Your email text here"
python predictors/predict_svm.py "Your email text here"
python predictors/predict_catboost.py "Your email text here"
```

### Generate Performance Report
```bash
python scripts/generate_direct_pdf_report.py
```
This creates a detailed PDF report comparing all models across different spam/ham categories.

## 📁 Project Structure

```
spam-detection/
├── 📁 config/           # Configuration files
│   └── models.toml      # Model URLs and metadata
├── 📁 data/             # Datasets (excluded from git)
├── 📁 docs/             # Documentation
│   ├── EMAIL_SPAM_GUIDE.md
│   ├── MODEL_COMPARISON_GUIDE.md
│   └── RUN_MODEL_GUIDE.md
├── 📁 models/           # Model files (excluded from git)
│   └── README.md
├── 📁 predictors/       # Prediction scripts
│   ├── predict_main.py  # Interactive CLI
│   ├── predict_enhanced_transformer.py
│   ├── predict_svm.py
│   └── predict_catboost.py
├── 📁 scripts/          # Utility scripts
│   ├── download_models.py
│   ├── download_models.ps1
│   └── generate_direct_pdf_report.py
├── 📁 src/              # Core modules
│   ├── enhanced_transformer_classifier.py
│   ├── svm_classifier.py
│   └── train_*.py files
├── 📁 training/         # Training scripts
│   ├── train_enhanced_transformer.py
│   ├── train_svm.py
│   └── train_catboost.py
└── 📄 README.md         # This file
```

## 🎯 Training

To train your own models:

### Enhanced Transformer
```bash
python training/train_enhanced_transformer.py
```

### SVM Model  
```bash
python training/train_svm.py
```

### CatBoost Model
```bash
python training/train_catboost.py
```

**Training Requirements:**
- Large dataset (10K+ emails recommended)
- 8GB+ RAM for transformer training
- GPU strongly recommended for transformers
- Training time: 30min-2hours depending on model and hardware

## 📚 Documentation

Detailed guides available in the `docs/` folder:

- **[Email Spam Guide](docs/EMAIL_SPAM_GUIDE.md)** - Understanding spam detection
- **[Model Comparison Guide](docs/MODEL_COMPARISON_GUIDE.md)** - Choosing the right model
- **[Run Model Guide](docs/RUN_MODEL_GUIDE.md)** - Step-by-step usage instructions

## 🔍 Model Performance

### Enhanced Transformer
- **Spam Recall**: 99.48% (catches almost all spam)
- **Overall Accuracy**: 96.65%
- **Perfect Categories**: Financial Scams, Phishing
- **Best For**: Modern spam patterns, complex text analysis

### SVM (Support Vector Machine)
- **Overall Accuracy**: ~95-98%
- **Speed**: Fastest (1-5ms)
- **Perfect Categories**: Traditional spam patterns
- **Best For**: High-speed processing, resource-constrained environments

### CatBoost
- **Overall Accuracy**: ~96-97%
- **Perfect Categories**: Balanced across spam/ham types
- **Best For**: General-purpose classification, feature engineering

## 🛠️ Advanced Usage

### Custom Thresholds
```python
from predictors.predict_svm import SVMPredictor

predictor = SVMPredictor('models/svm_full.pkl')
result = predictor.predict("email text", threshold=0.7)
```

### Batch Processing
```python
texts = ["email1", "email2", "email3"]
results = [predictor.predict(text) for text in texts]
```

### Model Comparison
```bash
python scripts/generate_direct_pdf_report.py
# Generates comprehensive performance comparison PDF
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📋 Requirements

```
torch>=2.0.0
scikit-learn>=1.3.0
catboost>=1.2.0
joblib>=1.3.0
pandas>=2.0.0
numpy>=1.21.0
gdown>=4.7.1
reportlab>=4.0.0
toml>=0.10.2
```

## 🐛 Troubleshooting

### Common Issues

**Models not found:**
```bash
python scripts/download_models.py
```

**Import errors:**
- Make sure you're in the project root directory
- Check Python path: `python -c "import sys; print(sys.path)"`

**GPU not detected:**
- Install CUDA-compatible PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cu118`

**Memory issues:**
- Use smaller batch sizes
- Try CPU-only mode for transformers
- Consider using SVM for low-memory environments

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with PyTorch, scikit-learn, and CatBoost
- Trained on comprehensive email datasets
- Performance optimized for real-world usage

---

**📧 Questions?** Open an issue or check the [documentation](docs/) folder for detailed guides.