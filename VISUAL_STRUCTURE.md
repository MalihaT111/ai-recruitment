# 🎨 Visual Project Structure

Quick visual reference for understanding the project organization.

---

## 📊 **High-Level Architecture**

```
┌─────────────────────────────────────────────────────────────────┐
│                         YOUR PROJECT                             │
│                    AI Resume-Job Matching                        │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   DATA       │    │  NOTEBOOKS   │    │   MODELS     │
│  (Storage)   │    │  (Analysis)  │    │  (Trained)   │
└──────────────┘    └──────────────┘    └──────────────┘
        │                     │                     │
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ PRECOMPUTED  │    │    UTILS     │    │ PRODUCTION   │
│ (Fast Load)  │    │  (Helpers)   │    │   (eval.py)  │
└──────────────┘    └──────────────┘    └──────────────┘
```

---

## 🗂️ **Folder Hierarchy with Purpose**

```
📦 ai-recruitment/
│
├── 📂 data/                          ← ALL YOUR DATA LIVES HERE
│   ├── 📂 original/                  ← Raw & cleaned CSVs
│   │   ├── 📄 resumes.csv           (2,484 resumes - raw)
│   │   ├── 📄 resumes_cleaned.csv   (2,484 resumes - clean) ✅
│   │   ├── 📄 jobs.csv              (5,448 jobs - raw)
│   │   └── 📄 jobs_cleaned.csv      (5,448 jobs - clean) ✅
│   │
│   ├── 📂 embeddings/                ← Precomputed vectors
│   │   ├── 🔢 resume_emb_e5_large.npy      (BEST - 1024-dim) ⭐
│   │   ├── 🔢 job_emb_e5_large.npy         (BEST - 1024-dim) ⭐
│   │   ├── 🔢 resume_emb_all_mpnet.npy     (768-dim)
│   │   ├── 🔢 job_emb_all_mpnet.npy        (768-dim)
│   │   ├── 🔢 resume_emb_multi_qa_mpnet.npy (768-dim)
│   │   ├── 🔢 job_emb_multi_qa_mpnet.npy    (768-dim)
│   │   ├── 🔢 resume_emb_all-MiniLM.npy    (384-dim, FAST) 🚀
│   │   └── 🔢 job_emb_all-MiniLM.npy       (384-dim, FAST) 🚀
│   │
│   ├── 📄 train_pairs_rich.csv       ← Training data (115K pairs)
│   └── 📄 test_pairs_rich.csv        ← Test data (20K pairs)
│
├── 📂 models/                         ← TRAINED ML MODELS
│   ├── 🤖 model_random_forest.pkl    (Main model) ⭐
│   ├── 🤖 model_logreg.pkl           (Baseline)
│   ├── 🤖 model_gradient_boosting.pkl (Best accuracy)
│   ├── 🤖 model_svm_(rbf).pkl        (Experimental)
│   └── 🤖 model_neural_network.pkl   (Deep learning)
│
├── 📂 notebooks/                      ← YOUR MAIN WORKSPACE
│   │
│   ├── 📓 PIPELINE (Run in order 1→6)
│   │   ├── 1️⃣ 1preprocessing.ipynb          (Clean data)
│   │   ├── 2️⃣ 2feature_extraction.ipynb     (Generate embeddings)
│   │   ├── 3️⃣ 3cosine_sim.ipynb             (Compute similarity)
│   │   ├── 4️⃣ 4similarity_diagnostics.ipynb (Analyze distributions)
│   │   ├── 5️⃣ 5evaluation.ipynb             (Clustering eval)
│   │   └── 6️⃣ gpt.ipynb                     (Train ML models) ⭐
│   │
│   ├── 🔬 MODEL COMPARISON
│   │   ├── 📊 model_comparison.py           (Compare 5 models) ⭐
│   │   ├── 🎯 hyperparameter_tuning.py      (Optimize models)
│   │   └── 📓 6model_comparison.ipynb       (Interactive)
│   │
│   ├── 🚀 PRODUCTION
│   │   ├── ⚡ eval.py                       (Real-time ranking) ⭐
│   │   ├── 💾 precompute_embeddings.py      (Precompute resumes)
│   │   └── 🛠️ feature_extractors.py         (Feature utilities)
│   │
│   └── 🎨 Poppins-Medium.ttf                (Font for plots)
│
├── 📂 precomputed/                    ← PRODUCTION-READY DATA
│   ├── 🔢 resume_vectors.npy         (Fast loading)
│   ├── 📄 resume_texts.csv           (Cleaned texts)
│   └── 📋 resume_index.json          (Filename mapping)
│
├── 📂 utils/                          ← HELPER FUNCTIONS
│   └── 🛠️ utils.py                   (Text preprocessing)
│
├── 📂 oldnotebooks/                   ← ARCHIVE (experiments)
│   └── 📓 [9 old notebooks]          (Not used in main pipeline)
│
├── 📄 requirements.txt                ← Python dependencies
├── 📄 README.md                       ← Project overview
├── 📄 PROJECT_STRUCTURE.md            ← This guide! 📖
└── 📄 .gitignore                      ← Git rules
```

---

## 🔄 **Data Flow: From Raw Data to Production**

```
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 1: DATA PREPARATION                                       │
└─────────────────────────────────────────────────────────────────┘

    📥 Kaggle Dataset
         │
         ├─ resumes.csv (2,484 resumes)
         └─ jobs.csv (5,448 jobs)
         │
         ▼
    [1preprocessing.ipynb]
    • Remove HTML
    • Handle missing values
    • Remove duplicates
    • Text preprocessing
         │
         ▼
    📤 data/original/
         ├─ resumes_cleaned.csv ✅
         └─ jobs_cleaned.csv ✅

┌─────────────────────────────────────────────────────────────────┐
│ STAGE 2: EMBEDDING GENERATION                                   │
└─────────────────────────────────────────────────────────────────┘

    📥 resumes_cleaned.csv + jobs_cleaned.csv
         │
         ▼
    [2feature_extraction.ipynb]
    • Load SentenceTransformer
    • Encode all texts
    • Test 4 different models
    • Normalize embeddings
         │
         ▼
    📤 data/embeddings/
         ├─ resume_emb_e5_large.npy ⭐ (BEST)
         ├─ job_emb_e5_large.npy ⭐
         └─ [6 other embedding files]

┌─────────────────────────────────────────────────────────────────┐
│ STAGE 3: TRAINING DATA GENERATION                               │
└─────────────────────────────────────────────────────────────────┘

    📥 Embeddings + Cleaned CSVs
         │
         ▼
    [gpt.ipynb]
    • Split jobs 85/15
    • Generate pairs (job × resume)
    • Use cosine sim for labels
    • Extract 5 features
    • Create train/test sets
         │
         ▼
    📤 data/
         ├─ train_pairs_rich.csv (115,775 pairs)
         └─ test_pairs_rich.csv (20,425 pairs)

┌─────────────────────────────────────────────────────────────────┐
│ STAGE 4: MODEL TRAINING                                         │
└─────────────────────────────────────────────────────────────────┘

    📥 train_pairs_rich.csv
         │
         ▼
    [gpt.ipynb] OR [model_comparison.py]
    • Train Random Forest
    • Train Logistic Regression
    • Train Gradient Boosting
    • Train SVM
    • Train Neural Network
    • Evaluate all models
         │
         ▼
    📤 models/
         ├─ model_random_forest.pkl ⭐
         ├─ model_logreg.pkl
         ├─ model_gradient_boosting.pkl
         ├─ model_svm_(rbf).pkl
         └─ model_neural_network.pkl

┌─────────────────────────────────────────────────────────────────┐
│ STAGE 5: PRODUCTION DEPLOYMENT                                  │
└─────────────────────────────────────────────────────────────────┘

    📥 Real resumes (*.docx files)
         │
         ▼
    [precompute_embeddings.py]
    • Load all resumes
    • Clean text
    • Embed with e5-large-v2
    • Save for fast loading
         │
         ▼
    📤 precomputed/
         ├─ resume_vectors.npy
         ├─ resume_texts.csv
         └─ resume_index.json
         │
         ▼
    [eval.py] ← PRODUCTION INFERENCE
    • Load precomputed embeddings
    • Embed new job posting
    • Compute cosine similarity
    • Extract ML features
    • Combine signals
    • Return top-K resumes
         │
         ▼
    📤 Top-10 Ranked Resumes 🎯
```

---

## 🎯 **File Importance Matrix**

```
┌─────────────────────────────────────────────────────────────┐
│                    IMPORTANCE LEVELS                         │
├─────────────────────────────────────────────────────────────┤
│ ⭐⭐⭐ CRITICAL - Must have for system to work              │
│ ⭐⭐  IMPORTANT - Needed for full functionality             │
│ ⭐   OPTIONAL - Nice to have, not essential                │
└─────────────────────────────────────────────────────────────┘

DATA FILES:
⭐⭐⭐ resumes_cleaned.csv          (Core dataset)
⭐⭐⭐ jobs_cleaned.csv             (Core dataset)
⭐⭐⭐ resume_emb_e5_large.npy      (Best embeddings)
⭐⭐⭐ job_emb_e5_large.npy         (Best embeddings)
⭐⭐  train_pairs_rich.csv         (Training data)
⭐⭐  test_pairs_rich.csv          (Evaluation data)
⭐   Other embedding files         (Alternative models)

NOTEBOOKS:
⭐⭐⭐ gpt.ipynb                    (Main training)
⭐⭐⭐ eval.py                      (Production inference)
⭐⭐  model_comparison.py          (Model selection)
⭐⭐  1preprocessing.ipynb         (Data cleaning)
⭐⭐  2feature_extraction.ipynb    (Embedding generation)
⭐⭐  feature_extractors.py        (Utilities)
⭐   3cosine_sim.ipynb            (Analysis)
⭐   4similarity_diagnostics.ipynb (Analysis)
⭐   5evaluation.ipynb            (Analysis)
⭐   precompute_embeddings.py     (Production prep)
⭐   hyperparameter_tuning.py     (Optimization)

MODELS:
⭐⭐⭐ model_random_forest.pkl      (Main production model)
⭐⭐  model_gradient_boosting.pkl  (Best accuracy)
⭐   model_logreg.pkl             (Fast baseline)
⭐   Other models                  (Experimental)

UTILITIES:
⭐⭐  feature_extractors.py        (Feature functions)
⭐   utils.py                      (Text preprocessing)
```

---

## 🚦 **Quick Start Paths**

### **Path 1: I want to understand the system**
```
1. Read README.md
2. Open gpt.ipynb
3. Look at eval.py
4. Check model_comparison.py
```

### **Path 2: I want to train models**
```
1. Run 1preprocessing.ipynb
2. Run 2feature_extraction.ipynb
3. Run gpt.ipynb
4. Run model_comparison.py
```

### **Path 3: I want to deploy to production**
```
1. Collect real resumes (*.docx)
2. Run precompute_embeddings.py
3. Use eval.py for inference
4. Load best model from models/
```

### **Path 4: I want to improve the system**
```
1. Modify feature_extractors.py (add features)
2. Retrain with gpt.ipynb
3. Compare with model_comparison.py
4. Tune with hyperparameter_tuning.py
```

---

## 📏 **Size Reference**

```
SMALL FILES (<1 MB):
├── All .py scripts
├── All .md documentation
└── requirements.txt

MEDIUM FILES (1-100 MB):
├── resumes_cleaned.csv (~50 MB)
├── jobs_cleaned.csv (~10 MB)
├── train_pairs_rich.csv (~30 MB)
├── test_pairs_rich.csv (~5 MB)
└── Most .pkl models (~10-50 MB each)

LARGE FILES (>100 MB):
├── All .npy embedding files (~100-200 MB each)
├── model_random_forest.pkl (~100 MB)
└── precomputed/resume_vectors.npy (~500 MB)

TOTAL PROJECT: ~2-3 GB
```

---

## 🎓 **For Presentations/Interviews**

**Show this structure when explaining your project:**

```
"My project has 3 main components:

1️⃣ DATA PIPELINE (notebooks 1-2)
   → Clean data and generate embeddings

2️⃣ TRAINING PIPELINE (gpt.ipynb + model_comparison.py)
   → Train and compare ML models

3️⃣ PRODUCTION SYSTEM (eval.py + precomputed/)
   → Real-time ranking with precomputed embeddings

The key innovation is the hybrid approach:
combining semantic similarity (embeddings) with
rule-based features (skills, education) for
better ranking quality."
```

---

**Questions?** See `PROJECT_STRUCTURE.md` for detailed explanations!
