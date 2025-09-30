# 🚀 SMS Spam Detection - Quick Start Guide

## **BEST LATEST MODEL (96.04% Accuracy)**

### **1. Environment Setup**
```powershell
# Activate virtual environment
sms_env\Scripts\Activate.ps1
```

### **2. Train Latest Model (if needed)**
```powershell
# This creates the best latest model with all improvements
python src\simple_svm_classifier.py --data data\large_spamassassin_corpus.csv --model models\svm_best_latest.pkl
```

### **3. Use the Model**

**Single Message:**
```powershell
python predict.py "Your message here"
```

**Interactive Mode:**
```powershell
python predict.py --interactive
```

**Specify Model Explicitly:**
```powershell
python predict.py --model models\svm_best_latest.pkl "Your message"
```

### **4. Test Examples**

**Spam Detection:**
```powershell
python predict.py "FREE! Win $1000 cash prize now!"
python predict.py "URGENT! Your account will be suspended!"
python predict.py "Congratulations! You've won a free iPhone!"
```

**Legitimate Messages:**
```powershell
python predict.py "From: team@company.com - Meeting moved to 3 PM"
python predict.py "Date: Today Subject: Project update and next steps"
```

## **📊 Model Performance**
- **Accuracy:** 96.04%
- **Spam Precision:** 91.5%
- **Spam Recall:** 96.2%
- **F1-Score:** 93.8%

## **🔧 Model Features**
- ✅ Character n-grams (3-5 chars) - Better for short texts
- ✅ Preserves digits and punctuation ($, £, !, ?)
- ✅ Class balancing for better spam detection
- ✅ Optimized preprocessing for spam indicators

## **📁 File Structure**
```
C:\006\
├── models\
│   └── svm_best_latest.pkl     # ← BEST MODEL
├── src\
│   └── simple_svm_classifier.py
├── predict.py                   # ← MAIN INTERFACE
└── data\
    └── large_spamassassin_corpus.csv
```

## **🎯 Usage Tips**
- The model works best on email-style messages (training data)
- SMS-style messages may have lower confidence due to domain mismatch
- High confidence (>1.0) = Very confident prediction
- Low confidence (<0.5) = Less confident, manual review recommended