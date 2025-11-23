import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util

from .feature_extractors import (
    clean_text_for_domain,
    detect_domain,
    skill_match,
    education_score,
    seniority_score,
    keyword_overlap,
)
# -------------------------------------------------
# 1. LOAD ARTIFACTS (same as api_pipeline.py)
# -------------------------------------------------
RESUME_CSV_PATH = "data/original/resumes_cleaned.csv"
RESUME_EMB_PATH = "data/embeddings/resume_emb_e5_large.npy"
MODEL_PATH = "models/model_gb_tuned.pkl"  # tuned GB model

print("[API] Loading resumes and embeddings...")
resume_df = pd.read_csv(RESUME_CSV_PATH)
# ensure text column is string to avoid slicing errors
if "Resume_clean" in resume_df.columns:
    resume_df["Resume_clean"] = resume_df["Resume_clean"].astype(str)
resume_emb = np.load(RESUME_EMB_PATH)

print("[API] Loading tuned Gradient Boosting model...")
model = joblib.load(MODEL_PATH)

print("[API] Loading embedding model (intfloat/e5-large-v2)...")
embedder = SentenceTransformer("intfloat/e5-large-v2")

# Features the GB model was trained on (order matters)
MODEL_FEATURES = [
    "keyword_overlap",
    "skill_score",
    "experience_score",
    "education_score",
    "domain_match",
]


# -------------------------------------------------
# 2. CORE RANKING FUNCTION (reused from api_pipeline.py)
# -------------------------------------------------
def rank_resumes_for_job(job_text: str, top_k: int = 10, alpha: float = 0.7) -> pd.DataFrame:
    """Rank resumes for a given job description.

    final_score = alpha * ml_prob + (1 - alpha) * cosine_similarity
    """
    job_clean = clean_text_for_domain(job_text)
    job_vec = embedder.encode(job_clean, convert_to_numpy=True)

    # cosine similarity against all precomputed resume embeddings
    cos_sims = util.cos_sim(job_vec, resume_emb)[0].cpu().numpy()

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
# 3. FASTAPI SETUP
# -------------------------------------------------
app = FastAPI(title="AI Recruitment Ranking API")


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


@app.post("/rank", response_model=list[RankedResume])
def rank_endpoint(payload: RankRequest):
    """Rank resumes for the given job description and return top K."""
    df = rank_resumes_for_job(
        job_text=payload.job_description,
        top_k=payload.top_k,
        alpha=payload.alpha,
    )
    return df.to_dict(orient="records")


@app.get("/")
def root():
    return {"message": "AI Recruitment Ranking API is up"}
