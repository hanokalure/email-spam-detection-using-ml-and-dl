# Models Directory

This directory contains the trained machine learning models for email spam detection.

## Model Files (Not included in repository)

Due to file size limitations, the trained models are not included in the GitHub repository. You can:

### Option 1: Train Your Own Models
```bash
# Train the primary email spam model
python src/simple_svm_classifier.py --data path/to/email/dataset.csv --model models/svm_best.pkl

# Train the mixed model (if you have both email and SMS data)
python src/simple_svm_classifier.py --data path/to/mixed/dataset.csv --model models/svm.pkl
```

### Option 2: Download Pre-trained Models
If available, download the pre-trained models and place them in this directory:
- `svm_best.pkl` - Primary email spam model (99.66% accuracy)
- `svm.pkl` - Mixed model for broader compatibility (98.90% accuracy)

## Model Information

| Model | Accuracy | Training Data | Size | Purpose |
|-------|----------|---------------|------|---------|
| `svm_best.pkl` | 99.66% | SpamAssassin Email Corpus (5,796 emails) | ~420KB | Primary email spam detection |
| `svm.pkl` | 98.90% | Mixed dataset (11,368 messages) | ~422KB | Fallback/universal model |

## Usage

Once you have the models in place, you can use them with the prediction scripts:

```bash
# Use the default model (svm_best.pkl)
python predict.py "your email content here"

# Specify a different model
python predict.py "your email content here" --model models/svm.pkl
```