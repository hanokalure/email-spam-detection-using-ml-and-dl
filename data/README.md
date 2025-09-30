# Data Directory

This directory contains datasets for training email spam detection models.

## Dataset Files (Not included in repository)

Due to file size and licensing considerations, datasets are not included in the GitHub repository.

## Recommended Datasets

### For Email Spam Detection
1. **SpamAssassin Public Corpus**
   - Download from: https://spamassassin.apache.org/old/publiccorpus/
   - Contains thousands of spam and ham email messages
   - Good for training email-specific models

2. **Enron Email Dataset**
   - Available through UCI ML Repository
   - Large collection of legitimate business emails
   - Useful for ham (non-spam) examples

### Dataset Format

Your CSV files should have the following format:
```
text,label
"Email message content here","ham"
"Spam message content here","spam"
```

Or with numeric labels:
```
text,target
"Email message content here",0
"Spam message content here",1
```

## Training Your Own Models

Once you have your dataset(s) in this directory:

```bash
# Train on your email dataset
python src/simple_svm_classifier.py --data data/your_dataset.csv --model models/svm_best.pkl
```

## File Structure Expected

```
data/
├── README.md (this file)
├── your_email_dataset.csv
├── spamassassin_corpus.csv
└── other_datasets.csv
```

## Data Preprocessing

The classifier handles various input formats automatically:
- Supports both string labels ("spam"/"ham") and numeric labels (1/0)
- Automatically detects column formats
- Includes comprehensive text preprocessing
- Handles missing data and encoding issues