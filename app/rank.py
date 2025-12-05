import sys
import os
sys.path.append('../notebooks')
sys.path.append('../utils')

import numpy as np
import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer, util

from feature_extractors import (
    clean_text_for_domain,
    detect_domain,
    skill_match,
    education_score,
    seniority_score,
    keyword_overlap,
)

# Load data and models
RESUME_CSV_PATH = "../data/original/resumes_cleaned.csv"
RESUME_EMB_PATH = "../data/embeddings/resume_emb_e5_large.npy"
MODEL_PATH = "../models/model_random_forest.pkl"  # Changed to Random Forest

print("Loading resumes and embeddings...")
resume_df = pd.read_csv(RESUME_CSV_PATH)
resume_texts = resume_df["Resume_clean"].fillna("").astype(str).tolist()  # For matching
resume_originals = resume_df["Resume_str"].fillna("").astype(str).tolist()  # For display
resume_emb = np.load(RESUME_EMB_PATH)

print("Loading model...")
model = joblib.load(MODEL_PATH)

print("Loading embedder...")
embedder = SentenceTransformer("intfloat/e5-large-v2")

MODEL_FEATURES = [
    "keyword_overlap",
    "skill_score",
    "experience_score",
    "education_score",
    "domain_match",
]

def rank_resumes_for_job(job_text: str, top_k: int = 10, alpha: float = 0.8) -> pd.DataFrame:
    job_clean = clean_text_for_domain(job_text)
    job_vec = embedder.encode(job_clean, convert_to_numpy=True)

    cos_sims = util.cos_sim(job_vec, resume_emb)[0].cpu().numpy()

    rows = []

    for r_idx, resume_text_raw in enumerate(resume_texts):
        resume_clean = clean_text_for_domain(resume_text_raw)

        kw = keyword_overlap(job_clean, resume_clean)
        skl = skill_match(job_clean, resume_clean)
        exp = seniority_score(resume_clean)
        edu = education_score(resume_clean)

        job_dom, _ = detect_domain(job_clean)
        res_dom, _ = detect_domain(resume_clean)
        dom_match = 1 if job_dom == res_dom else 0

        feat_df = pd.DataFrame([{
            "keyword_overlap": kw,
            "skill_score": skl,
            "experience_score": exp,
            "education_score": edu,
            "domain_match": dom_match,
        }])[MODEL_FEATURES]

        ml_prob = float(model.predict_proba(feat_df)[0, 1])
        cos_sim = float(cos_sims[r_idx])
        final_score = alpha * ml_prob + (1 - alpha) * cos_sim

        rows.append({
            "resume_idx": int(r_idx),
            "final_score": final_score,
            "ml_prob": ml_prob,
            "cos_sim": cos_sim,
            "resume_text": resume_originals[r_idx],  # Original resume text
        })

    out = pd.DataFrame(rows).sort_values("final_score", ascending=False).head(top_k)
    return out[["resume_idx", "final_score", "ml_prob", "cos_sim", "resume_text"]]