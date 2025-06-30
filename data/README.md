<p align="center">
  <a href="README.md"><img src="https://img.shields.io/badge/lang-English-blue.svg"></a>
  <a href="README.zh-TW.md"><img src="https://img.shields.io/badge/lang-繁體中文-green.svg"></a>
</p>

# Data Directory Description

This directory contains all project-related data, including raw inputs, preprocessing results, extracted feature vectors, and intermediate files prepared for model training and evaluation.

---

## Directory Structure
```bash
data/
├── db/          # Reference databases (.json, .pkl, .faiss)
├── features/    # Extracted feature vectors (.npy)
├── processed/   # Cleaned and split datasets (train/val/test)
└── raw/         # Original CSV/JSON input data
```

---

## 📌 Data Types

### Raw Data
- **Source**: collected manually or via crawlers from Taiwanese news outlets and fact-checking organizations  
- **Format**: CSV / JSON (uncleaned)  
- **Usage**: input for preprocessing  

### Processed Data
- Cleaned and segmented results, with possible train/val/test splits  
- Ready for direct use in model training  

### Feature Vectors
- Embeddings from models (e.g., MacBERT, BGEM3, Text2Vec)  
- Format: `.npy` files, including both input features and labels  
  - Examples: `bert_X_train.npy`, `bgem3_title_X_val.npy`, `bert_y_test.npy`  

### Databases
- Reference collections of news and fact-checking entries  
- Formats: `.json` (raw reference), `.pkl` (indexed DB)  
- Two main sets:  
  - **all** → database built from all reference news  
  - **mgp** → database built from MyGoPen (MGP) fact-checking data  

---

## Data Processing Pipeline

```bash
Raw Data → Preprocessing → Feature Extraction → Model Input
```

Each step produces reusable files to avoid redundant computation.

⚠️ Due to size and copyright restrictions, the dataset itself is **not included** in this repository.  
Users may collect comparable data from the listed news sources and fact-checking organizations, or contact us for further information regarding data access.
