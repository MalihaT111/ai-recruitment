"""
rank.py - Resume Ranking API Route

PURPOSE:
    Ranks resumes against a job description using a hybrid ML + semantic similarity approach.
    Combines a trained Gradient Boosting classifier with sentence-transformer embeddings
    to produce a final relevance score for each resume.

INPUTS (via POST /rank):
    - job_description (str): Raw text of the job posting
    - top_k (int, default=10): Number of top resumes to return
    - alpha (float, default=0.7): Weight for ML model vs semantic similarity
      final_score = alpha * ml_probability + (1 - alpha) * cosine_similarity

OUTPUTS:
    JSON array of ranked resumes, each containing:
    - resume_idx (int): Index of the resume in the dataset
    - final_score (float): Combined relevance score (0-1)
    - ml_prob (float): ML model probability that resume matches job
    - cos_sim (float): Semantic similarity between job and resume embeddings
    - resume_text (str): First 400 characters of the resume

HOW TO USE:
    POST http://localhost:8000/rank
    Content-Type: application/json
    
    {
      "job_description": "We need a Python developer with 3+ years of experience...",
      "top_k": 5,
      "alpha": 0.7
    }

DEPENDENCIES:
    - models/model_gradient_boosting.pkl: Trained GB classifier
    - data/original/resumes_cleaned.csv: Resume corpus
    - data/embeddings/resume_emb_e5_large.npy: Precomputed resume embeddings
    - notebooks/feature_extractors.py: Feature engineering functions

DEVICE:
    Automatically uses GPU (CUDA) if available, falls back to CPU.
"""

import numpy as np
import pandas as pd
import joblib
from fastapi import APIRouter
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import sys
from pathlib import Path

# Add notebooks to path for feature_extractors import
notebooks_path = Path(__file__).parent.parent / "notebooks"
sys.path.insert(0, str(notebooks_path))

from feature_extractors import (
    clean_text_for_domain,
    detect_domain,
    skill_match,
    education_score,
    seniority_score,
    keyword_overlap,
)

# -------------------------------------------------
# 1. LOAD ARTIFACTS
# -------------------------------------------------
BASE_DIR = Path(__file__).parent.parent
RESUME_CSV_PATH = BASE_DIR / "data" / "original" / "resumes_cleaned.csv"
RESUME_EMB_PATH = BASE_DIR / "data" / "embeddings" / "resume_emb_e5_large.npy"
MODEL_PATH = BASE_DIR / "models" / "model_gradient_boosting.pkl"

print("[RANK] Loading resumes and embeddings...")
resume_df = pd.read_csv(RESUME_CSV_PATH)
if "Resume_clean" in resume_df.columns:
    resume_df["Resume_clean"] = resume_df["Resume_clean"].astype(str)
resume_emb = np.load(RESUME_EMB_PATH)

print("[RANK] Loading tuned Gradient Boosting model...")
model = joblib.load(MODEL_PATH)

# Detect device (GPU if available, else CPU)
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[RANK] Using device: {device}")

print("[RANK] Loading embedding model (intfloat/e5-large-v2)...")
embedder = SentenceTransformer("intfloat/e5-large-v2", device=device)

# Features the GB model was trained on (order matters)
MODEL_FEATURES = [
    "keyword_overlap",
    "skill_score",
    "experience_score",
    "education_score",
    "domain_match",
]


# -------------------------------------------------
# 2. CORE RANKING FUNCTION
# -------------------------------------------------
def rank_resumes_for_job(job_text: str, top_k: int = 10, alpha: float = 0.7) -> pd.DataFrame:
    """Rank resumes for a given job description.

    final_score = alpha * ml_prob + (1 - alpha) * cosine_similarity
    """
    job_clean = clean_text_for_domain(job_text)
    job_vec = embedder.encode(job_clean, convert_to_tensor=True, device=device)

    # cosine similarity against all precomputed resume embeddings (move to same device)
    import torch
    resume_emb_tensor = torch.from_numpy(resume_emb).to(device)
    cos_sims = util.cos_sim(job_vec, resume_emb_tensor)[0].cpu().numpy()

    rows = []

    for r_idx, resume_text_raw in enumerate(resume_df["Resume_clean"]):
        resume_clean = clean_text_for_domain(resume_text_raw)

        # feature extraction (same as training)
        kw = keyword_overlap(job_clean, resume_clean)
        skl = skill_match(job_clean, resume_clean)
        exp = seniority_score(resume_clean)
        edu = education_score(resume_clean)

        job_dom, _ = detect_domain(job_clean)
        res_dom, _ = detect_domain(resume_clean)
        dom_match = 1 if job_dom == res_dom else 0

        feat_df = pd.DataFrame(
            [
                {
                    "keyword_overlap": kw,
                    "skill_score": skl,
                    "experience_score": exp,
                    "education_score": edu,
                    "domain_match": dom_match,
                }
            ]
        )[MODEL_FEATURES]

        ml_prob = float(model.predict_proba(feat_df)[0, 1])
        cos_sim = float(cos_sims[r_idx])
        final_score = alpha * ml_prob + (1 - alpha) * cos_sim

        raw_text = resume_df.iloc[r_idx]["Resume_clean"]
        text_str = "" if pd.isna(raw_text) else str(raw_text)
        snippet = text_str[:400]

        rows.append(
            {
                "resume_idx": int(r_idx),
                "final_score": final_score,
                "ml_prob": ml_prob,
                "cos_sim": cos_sim,
                "resume_text": snippet,
            }
        )

    out = pd.DataFrame(rows).sort_values("final_score", ascending=False).head(top_k)
    return out[["resume_idx", "final_score", "ml_prob", "cos_sim", "resume_text"]]


# -------------------------------------------------
# 3. PYDANTIC MODELS
# -------------------------------------------------
class RankRequest(BaseModel):
    job_description: str
    top_k: int = 10
    alpha: float = 0.7


class RankedResume(BaseModel):
    resume_idx: int
    final_score: float
    ml_prob: float
    cos_sim: float
    resume_text: str


# -------------------------------------------------
# 4. ROUTER
# -------------------------------------------------
router = APIRouter(prefix="/rank", tags=["ranking"])


@router.post("", response_model=list[RankedResume])
def rank_endpoint(payload: RankRequest):
    """Rank resumes for the given job description and return top K."""
    df = rank_resumes_for_job(
        job_text=payload.job_description,
        top_k=payload.top_k,
        alpha=payload.alpha,
    )
    return df.to_dict(orient="records")
