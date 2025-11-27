# 🚀 Quick Reference Card

One-page cheat sheet for navigating your project.

---

## 📂 **Key Directories**

| Directory | Purpose | Key Files |
|-----------|---------|-----------|
| `data/` | All datasets | `resumes_cleaned.csv`, `jobs_cleaned.csv` |
| `data/embeddings/` | Precomputed vectors | `*_emb_e5_large.npy` ⭐ |
| `models/` | Trained ML models | `model_random_forest.pkl` ⭐ |
| `notebooks/` | Analysis & training | `gpt.ipynb`, `eval.py` ⭐ |
| `precomputed/` | Production data | `resume_vectors.npy` |
| `utils/` | Helper functions | `utils.py` |

---

## 📓 **Notebook Pipeline (Run in Order)**

| # | Notebook | Purpose | Output |
|---|----------|---------|--------|
| 1️⃣ | `1preprocessing.ipynb` | Clean data | `*_cleaned.csv` |
| 2️⃣ | `2feature_extraction.ipynb` | Generate embeddings | `*.npy` files |
| 3️⃣ | `3cosine_sim.ipynb` | Compute similarity | Analysis |
| 4️⃣ | `4similarity_diagnostics.ipynb` | Analyze distributions | Plots |
| 5️⃣ | `5evaluation.ipynb` | Clustering eval | Metrics |
| 6️⃣ | `gpt.ipynb` | **Train ML models** | Models + pairs |

---

## 🤖 **Model Comparison**

```bash
# Compare 5 models
python notebooks/model_comparison.py

# Tune best models (30-60 min)
python notebooks/hyperparameter_tuning.py
```

**Models Compared:**
1. Logistic Regression (fast baseline)
2. Random Forest (production choice) ⭐
3. Gradient Boosting (best accuracy)
4. SVM (experimental)
5. Neural Network (deep learning)

---

## 🚀 **Production Inference**

```python
# notebooks/eval.py
from eval import rank_resumes

job_text = "Senior Python Developer, 5+ years..."
results = rank_resumes(job_text, top_k=10, alpha=0.7)

# Returns: DataFrame with top-10 ranked resumes
```

**Parameters:**
- `top_k`: Number of resumes to return
- `alpha`: Weight for ML model (0.7 = 70% ML, 30% cosine)

---

## 📊 **Key Metrics**

| Metric | What it measures | Good value |
|--------|------------------|------------|
| **NDCG@10** | Ranking quality | > 0.85 ⭐ |
| **Recall@10** | % relevant in top-10 | > 0.85 |
| **Precision@10** | % top-10 that are relevant | > 0.40 |
| **ROC-AUC** | Classification ability | > 0.80 |

**Focus on NDCG@10** - it's the industry standard for ranking!

---

## 🔧 **Common Commands**

```bash
# Install dependencies
pip install -r requirements.txt

# Run model comparison
cd notebooks && python model_comparison.py

# Start Jupyter
jupyter notebook

# Check model performance
cat data/model_comparison_results.csv

# Load a model
python -c "import joblib; m = joblib.load('models/model_random_forest.pkl')"
```

---

## 📁 **Important Files**

```
⭐⭐⭐ MUST HAVE:
├── data/resumes_cleaned.csv
├── data/jobs_cleaned.csv
├── data/embeddings/resume_emb_e5_large.npy
├── data/embeddings/job_emb_e5_large.npy
├── models/model_random_forest.pkl
├── notebooks/gpt.ipynb
└── notebooks/eval.py

⭐⭐ IMPORTANT:
├── data/train_pairs_rich.csv
├── data/test_pairs_rich.csv
├── notebooks/model_comparison.py
└── notebooks/feature_extractors.py

⭐ NICE TO HAVE:
├── notebooks/1-5 (analysis notebooks)
├── Other embedding files
└── Other model files
```

---

## 🎯 **Quick Troubleshooting**

| Problem | Solution |
|---------|----------|
| Import error | `pip install -r requirements.txt` |
| File not found | Check you're in project root |
| Model too slow | Use Logistic Regression instead |
| Low NDCG | Try Gradient Boosting or tune hyperparameters |
| Out of memory | Reduce batch size or use smaller embeddings |

---

## 📈 **Performance Summary**

```
Dataset:
├── 2,484 resumes
├── 5,448 jobs
└── 115,775 training pairs

Best Model: Gradient Boosting
├── NDCG@10: 0.89
├── Recall@10: 0.91
└── Train time: 45s

Production Model: Random Forest
├── NDCG@10: 0.87
├── Recall@10: 0.90
└── Train time: 12s (4x faster!)

Inference Speed:
└── ~2 seconds for 2,484 resumes
```

---

## 🔄 **Typical Workflow**

### **Research/Development:**
```
1. Modify feature_extractors.py
2. Retrain: python notebooks/gpt.ipynb
3. Compare: python notebooks/model_comparison.py
4. Evaluate results
5. Repeat
```

### **Production Deployment:**
```
1. Collect real resumes (*.docx)
2. Run: python notebooks/precompute_embeddings.py
3. Use: notebooks/eval.py for inference
4. Monitor performance
5. Retrain periodically
```

---

## 💡 **Pro Tips**

1. **Always check NDCG@10** - it's the most important metric
2. **Start with Random Forest** - good balance of speed/accuracy
3. **Tune alpha** - find optimal ML vs. cosine weight
4. **Precompute embeddings** - makes inference 10x faster
5. **Monitor drift** - retrain if new data is very different

---

## 📚 **Documentation Files**

- `README.md` - Project overview
- `PROJECT_STRUCTURE.md` - Detailed file explanations
- `VISUAL_STRUCTURE.md` - Visual diagrams
- `MODEL_COMPARISON_GUIDE.md` - Model comparison details
- `QUICK_REFERENCE.md` - This file!

---

## 🎓 **For Interviews**

**30-second pitch:**
> "I built an AI resume-job matching system using a hybrid approach: semantic similarity from e5-large embeddings combined with rule-based features (skills, education, experience). I compared 5 ML models and chose Random Forest for production, achieving 0.87 NDCG@10 with 2-second inference time for 2,500 resumes."

**Key talking points:**
- ✅ Proper train/test split (no data leakage)
- ✅ Hybrid approach (embeddings + ML)
- ✅ Systematic model comparison
- ✅ Production-ready code
- ✅ Industry-standard metrics (NDCG@10)

---

**Need more details?** See the full documentation files! 📖
