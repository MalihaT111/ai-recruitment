# 📁 AI-Assisted Resume-Job Matching: Project Structure

Complete guide to understanding the file organization and purpose of each component.

---

## 🌳 Directory Tree Overview

```
ai-recruitment/
├── 📂 data/                    # All datasets and generated data
├── 📂 models/                  # Trained ML models
├── 📂 notebooks/               # Jupyter notebooks & scripts (main work)
├── 📂 precomputed/             # Production-ready precomputed data
├── 📂 utils/                   # Utility functions
├── 📂 oldnotebooks/            # Archive of experimental notebooks
├── 📄 requirements.txt         # Python dependencies
├── 📄 README.md               # Project overview
└── 📄 .gitignore              # Git ignore rules
```

---

## 📂 **1. `/data/` - All Datasets**

### **Purpose:** Store raw, cleaned, and processed datasets

```
data/
├── original/                   # Raw and cleaned source data
│   ├── resumes.csv            # Original 2,484 resumes (Kaggle)
│   ├── resumes_cleaned.csv    # Cleaned resumes (HTML removed, preprocessed)
│   ├── jobs.csv               # Original 5,448 job postings (Kaggle)
│   └── jobs_cleaned.csv       # Cleaned jobs (combined fields, preprocessed)
│
├── embeddings/                 # Precomputed embeddings (multiple models)
│   ├── resume_emb_e5_large.npy          # e5-large-v2 (1024-dim, BEST)
│   ├── job_emb_e5_large.npy             # e5-large-v2 job embeddings
│   ├── resume_emb_all_mpnet.npy         # all-mpnet-base-v2 (768-dim)
│   ├── job_emb_all_mpnet.npy            # all-mpnet-base-v2 jobs
│   ├── resume_emb_multi_qa_mpnet.npy    # multi-qa-mpnet (768-dim)
│   ├── job_emb_multi_qa_mpnet.npy       # multi-qa-mpnet jobs
│   ├── resume_emb_all-MiniLM.npy        # all-MiniLM-L6-v2 (384-dim, FAST)
│   └── job_emb_all-MiniLM.npy           # all-MiniLM-L6-v2 jobs
│
├── train_pairs_rich.csv        # Training pairs (115,775 pairs)
│   # Columns: job_id, resume_id, cosine_sim, keyword_overlap,
│   #          skill_score, education_score, experience_score,
│   #          domain_match, hybrid_score, label
│
└── test_pairs_rich.csv         # Test pairs (20,425 pairs)
    # Same structure as train_pairs_rich.csv
```

**Key Files:**
- **`resumes_cleaned.csv`**: 2,484 resumes with `Resume_clean` column
- **`jobs_cleaned.csv`**: 5,448 jobs with `job_text_clean` column
- **`*_emb_e5_large.npy`**: Production embeddings (best quality)
- **`train_pairs_rich.csv`**: 85% of jobs × 25 resumes each
- **`test_pairs_rich.csv`**: 15% of jobs × 25 resumes each

---

## 📂 **2. `/models/` - Trained ML Models**

### **Purpose:** Store trained model files for production use

```
models/
├── model_random_forest.pkl          # Random Forest (from gpt.ipynb)
├── model_logreg.pkl                 # Logistic Regression (from gpt.ipynb)
├── model_gradient_boosting.pkl      # Gradient Boosting (from comparison)
├── model_svm_(rbf).pkl              # SVM with RBF kernel
├── model_neural_network.pkl         # Multi-layer Perceptron
├── model_rf_tuned.pkl               # Tuned Random Forest (optional)
└── model_gb_tuned.pkl               # Tuned Gradient Boosting (optional)
```

**How to Load:**
```python
import joblib
model = joblib.load("models/model_random_forest.pkl")
```

**Model Sizes:**
- Logistic Regression: ~10 KB (smallest)
- Random Forest: ~50-100 MB
- Gradient Boosting: ~20-50 MB
- Neural Network: ~5-10 MB

---

## 📂 **3. `/notebooks/` - Main Development Work**

### **Purpose:** All analysis, training, and evaluation code

```
notebooks/
├── 📓 CORE PIPELINE (Run in order)
│   ├── 1preprocessing.ipynb              # Step 1: Clean raw data
│   ├── 2feature_extraction.ipynb         # Step 2: Generate embeddings
│   ├── 3cosine_sim.ipynb                 # Step 3: Compute similarity
│   ├── 4similarity_diagnostics.ipynb     # Step 4: Analyze distributions
│   ├── 5evaluation.ipynb                 # Step 5: Unsupervised eval
│   └── gpt.ipynb                         # Step 6: Train ML models
│
├── 📊 MODEL COMPARISON (New!)
│   ├── model_comparison.py               # Compare 5 ML models
│   ├── hyperparameter_tuning.py          # Tune best models
│   └── 6model_comparison.ipynb           # Interactive comparison
│
├── 🚀 PRODUCTION CODE
│   ├── eval.py                           # Real-time ranking inference
│   ├── precompute_embeddings.py          # Precompute resume embeddings
│   └── feature_extractors.py            # Feature extraction utilities
│
├── 📚 DOCUMENTATION
│   ├── 6_model_comparison.md             # Model comparison guide
│   └── MODEL_COMPARISON_GUIDE.md         # Detailed guide
│
└── 🎨 ASSETS
    └── Poppins-Medium.ttf                # Font for visualizations
```

### **Detailed File Descriptions:**

#### **Core Pipeline Notebooks:**

**`1preprocessing.ipynb`** - Data Cleaning
- Removes HTML tags from resumes
- Handles missing values
- Removes duplicates
- Combines job description fields
- Applies text preprocessing (lowercase, stopwords, lemmatization)
- **Output:** `resumes_cleaned.csv`, `jobs_cleaned.csv`

**`2feature_extraction.ipynb`** - Embedding Generation
- Loads cleaned data
- Uses SentenceTransformer to generate embeddings
- Tests multiple models (e5-large, mpnet, MiniLM)
- Normalizes embeddings (L2 norm)
- **Output:** `*_emb_*.npy` files in `data/embeddings/`

**`3cosine_sim.ipynb`** - Similarity Analysis
- Computes cosine similarity matrix (5448 × 2484)
- Shows top-K matches for sample jobs
- Implements cross-encoder reranking
- Demonstrates pure semantic matching
- **Output:** Similarity visualizations

**`4similarity_diagnostics.ipynb`** - Distribution Analysis
- Analyzes similarity score distributions
- Computes timing benchmarks
- Generates histograms and statistics
- **Output:** Diagnostic plots

**`5evaluation.ipynb`** - Unsupervised Evaluation
- K-Means clustering on combined embeddings
- Computes clustering metrics (Silhouette, Calinski-Harabasz, Davies-Bouldin)
- Analyzes resume-job mixing in clusters
- PCA visualization (2D projection)
- **Output:** Clustering quality metrics

**`gpt.ipynb`** - ML Model Training (MAIN TRAINING NOTEBOOK)
- Splits jobs 85/15 (train/test)
- Generates training pairs with pseudo-labels
- Extracts 5 features (keyword_overlap, skill_score, etc.)
- Trains Random Forest + Logistic Regression
- Evaluates with ranking metrics (P@10, R@10, MAP@10, NDCG@10)
- **Output:** `train_pairs_rich.csv`, `test_pairs_rich.csv`, trained models

---

#### **Model Comparison Scripts:**

**`model_comparison.py`** - Compare Multiple Models
```python
# Compares 5 models:
# 1. Logistic Regression
# 2. Random Forest
# 3. Gradient Boosting
# 4. SVM (RBF)
# 5. Neural Network (MLP)

# Evaluates on:
# - Classification: ROC-AUC, Avg Precision
# - Ranking: P@10, R@10, MAP@10, NDCG@10

# Outputs:
# - CSV results table
# - 3 visualization plots
# - All trained models
```

**`hyperparameter_tuning.py`** - Optimize Best Models
```python
# Uses GridSearchCV to tune:
# - Random Forest (n_estimators, max_depth, etc.)
# - Gradient Boosting (learning_rate, n_estimators, etc.)

# Outputs:
# - model_rf_tuned.pkl
# - model_gb_tuned.pkl
# - hyperparameter_tuning_results.csv
```

---

#### **Production Code:**

**`eval.py`** - Real-Time Ranking
```python
# Production inference script
# Usage:
#   from eval import rank_resumes
#   results = rank_resumes(job_text, top_k=10, alpha=0.7)

# Features:
# - Loads precomputed resume embeddings
# - Embeds new job posting
# - Computes cosine similarity
# - Extracts ML features
# - Combines signals (α×ML + (1-α)×cosine)
# - Returns top-K ranked resumes
```

**`precompute_embeddings.py`** - Precompute Resume Embeddings
```python
# For production deployment
# Reads .docx resumes from /Resumes/ folder
# Cleans and embeds all resumes
# Saves to /precomputed/ for fast inference

# Outputs:
# - resume_vectors.npy (embeddings)
# - resume_texts.csv (cleaned text)
# - resume_index.json (filename mapping)
```

**`feature_extractors.py`** - Feature Utilities
```python
# Shared utility functions:
# - clean_text_for_domain()
# - detect_domain()
# - skill_match()
# - education_score()
# - seniority_score()
# - keyword_overlap()

# Used by: gpt.ipynb, eval.py, model_comparison.py
```

---

## 📂 **4. `/precomputed/` - Production Data**

### **Purpose:** Fast inference with precomputed embeddings

```
precomputed/
├── resume_vectors.npy          # Precomputed embeddings (N × 1024)
├── resume_texts.csv            # Cleaned resume texts
└── resume_index.json           # Filename → index mapping
```

**How it works:**
1. Run `precompute_embeddings.py` once
2. Loads all resumes from `/Resumes/*.docx`
3. Embeds them with e5-large-v2
4. Saves to `/precomputed/`
5. `eval.py` loads these for fast inference

**Benefit:** Only need to embed job posting at runtime (not all resumes)

---

## 📂 **5. `/utils/` - Utility Functions**

### **Purpose:** Shared helper functions

```
utils/
└── utils.py                    # Text preprocessing utilities
    # Functions:
    # - preprocess_text()      # Lowercase, remove stopwords, lemmatize
    # - clean_html()           # Remove HTML tags
    # - tokenize()             # Word tokenization
```

**Used by:** `1preprocessing.ipynb`

---

## 📂 **6. `/oldnotebooks/` - Archive**

### **Purpose:** Experimental notebooks (not part of main pipeline)

```
oldnotebooks/
├── 3edanormal.ipynb                    # Early EDA
├── 4edaembedding.ipynb                 # Embedding experiments
├── 5clustering.ipynb                   # Clustering attempts
├── 6clustering.ipynb                   # More clustering
├── 7clustering.ipynb                   # Even more clustering
├── 8trainingset.ipynb                  # Training set experiments
├── 9trainingset.ipynb                  # More training experiments
├── AI_Assisted_Recruitment.ipynb       # Original prototype
└── mali_AI_Assisted_Recruitment.ipynb  # Another prototype
```

**Note:** These are kept for reference but not used in the main pipeline.

---

## 📄 **Root Files**

### **`requirements.txt`** - Python Dependencies
```txt
numpy
pandas
scikit-learn
sentence-transformers
torch
transformers
nltk
matplotlib
seaborn
jupyter
joblib
python-docx
tqdm
```

### **`README.md`** - Project Overview
- High-level description
- Installation instructions
- Quick start guide
- Results summary

### **`.gitignore`** - Git Ignore Rules
```
__pycache__/
*.pyc
.DS_Store
*.pkl
*.npy
data/original/*.csv
precomputed/
```

---

## 🔄 **Data Flow Diagram**

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA PREPARATION                          │
└─────────────────────────────────────────────────────────────┘
                            ↓
    resumes.csv, jobs.csv (Kaggle)
                            ↓
    [1preprocessing.ipynb] → resumes_cleaned.csv, jobs_cleaned.csv
                            ↓
    [2feature_extraction.ipynb] → resume_emb_*.npy, job_emb_*.npy
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                         │
└─────────────────────────────────────────────────────────────┘
                            ↓
    [gpt.ipynb]
    ├─ Split jobs 85/15
    ├─ Generate pairs with labels
    ├─ Extract features
    ├─ Train RF + LR
    └─ Evaluate
                            ↓
    train_pairs_rich.csv, test_pairs_rich.csv
    model_random_forest.pkl, model_logreg.pkl
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    MODEL COMPARISON                          │
└─────────────────────────────────────────────────────────────┘
                            ↓
    [model_comparison.py]
    ├─ Train 5 models
    ├─ Evaluate all
    └─ Generate plots
                            ↓
    model_*.pkl (5 models)
    model_comparison_results.csv
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    PRODUCTION DEPLOYMENT                     │
└─────────────────────────────────────────────────────────────┘
                            ↓
    [precompute_embeddings.py] → /precomputed/
                            ↓
    [eval.py] → Real-time ranking
```

---

## 🎯 **Quick Navigation Guide**

### **I want to...**

| Goal | File to Use |
|------|-------------|
| Clean raw data | `notebooks/1preprocessing.ipynb` |
| Generate embeddings | `notebooks/2feature_extraction.ipynb` |
| Train ML models | `notebooks/gpt.ipynb` |
| Compare models | `notebooks/model_comparison.py` |
| Rank resumes in production | `notebooks/eval.py` |
| Precompute embeddings | `notebooks/precompute_embeddings.py` |
| Understand features | `notebooks/feature_extractors.py` |
| See results | `data/model_comparison_results.csv` |
| Load trained model | `models/model_random_forest.pkl` |

---

## 📊 **File Size Reference**

```
Total Project Size: ~2-3 GB

Breakdown:
├── data/embeddings/        ~1.5 GB  (8 embedding files)
├── data/original/          ~50 MB   (CSV files)
├── data/train_pairs.csv    ~30 MB
├── data/test_pairs.csv     ~5 MB
├── models/                 ~200 MB  (all models)
├── precomputed/            ~500 MB  (production embeddings)
└── notebooks/              ~100 MB  (notebooks + outputs)
```

---

## 🚀 **Recommended Workflow**

### **For New Users:**
1. Read `README.md`
2. Install dependencies: `pip install -r requirements.txt`
3. Run notebooks 1-5 in order
4. Run `gpt.ipynb` to train models
5. Run `model_comparison.py` to compare models

### **For Production Deployment:**
1. Run `precompute_embeddings.py` on your resume database
2. Use `eval.py` for real-time ranking
3. Load best model from `models/`

### **For Experimentation:**
1. Modify `feature_extractors.py` to add features
2. Retrain with `gpt.ipynb`
3. Compare with `model_comparison.py`

---

**Questions?** Check the README or individual file docstrings!
