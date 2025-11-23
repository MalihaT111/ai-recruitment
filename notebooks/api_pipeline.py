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

# -------------------------------------------------
# 1. LOAD ARTIFACTS
# -------------------------------------------------
RESUME_CSV_PATH = "data/original/resumes_cleaned.csv"
RESUME_EMB_PATH = "data/embeddings/resume_emb_e5_large.npy"
MODEL_PATH      = "models/model_gb_tuned.pkl"   # tuned GB model

print("Loading resumes and embeddings...")
resume_df  = pd.read_csv(RESUME_CSV_PATH)
resume_emb = np.load(RESUME_EMB_PATH)

print("Loading tuned Gradient Boosting model...")
model = joblib.load(MODEL_PATH)

print("Loading embedding model (intfloat/e5-large-v2)...")
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
# 2. SCORING + RANKING
# -------------------------------------------------
def score_single_resume(job_clean: str, resume_clean: str) -> tuple[float, float, float]:
    """
    Returns (final_score, ml_prob, cos_sim) for one resume vs job.
    """
    # semantic similarity (cosine)
    job_vec = embedder.encode(job_clean, convert_to_numpy=True)
    res_vec = embedder.encode(resume_clean, convert_to_numpy=True)
    cos_sim = float(util.cos_sim(job_vec, res_vec)[0][0])

    # feature-based scores
    kw  = keyword_overlap(job_clean, resume_clean)
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

    # You can tune alpha later; 0.7 gives more weight to ML model
    alpha = 0.7
    final_score = alpha * ml_prob + (1 - alpha) * cos_sim

    return final_score, ml_prob, cos_sim


def rank_resumes_for_job(job_text: str, top_k: int = 10, alpha: float = 0.7) -> pd.DataFrame:
    """
    Rank all resumes for a given job description using:
      final_score = alpha * ml_prob + (1 - alpha) * cosine_similarity
    Uses precomputed resume embeddings for cosine similarity for speed.
    """
    # clean + embed job once
    job_clean = clean_text_for_domain(job_text)
    job_vec   = embedder.encode(job_clean, convert_to_numpy=True)

    # cosine sim against all precomputed resume embeddings
    cos_sims = util.cos_sim(job_vec, resume_emb)[0].cpu().numpy()

    rows = []

    for r_idx, resume_text_raw in enumerate(resume_df["Resume_clean"]):
        resume_clean = clean_text_for_domain(resume_text_raw)

        # features (same as training)
        kw  = keyword_overlap(job_clean, resume_clean)
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

        # safe text snippet (handle NaNs / non-strings)
        raw_text = resume_df.iloc[r_idx]["Resume_clean"]
        text_str = "" if pd.isna(raw_text) else str(raw_text)
        snippet = text_str[:400]

        rows.append({
            "resume_idx": int(r_idx),
            "final_score": final_score,
            "ml_prob": ml_prob,
            "cos_sim": cos_sim,
            "resume_text": snippet,
        })

    out = pd.DataFrame(rows).sort_values("final_score", ascending=False).head(top_k)
    return out[["resume_idx", "final_score", "ml_prob", "cos_sim", "resume_text"]]


# -------------------------------------------------
# 3. DEMO: HARDCODED JOB DESCRIPTION
# -------------------------------------------------
if __name__ == "__main__":
    job_description = """
    We need a backend engineer with strong Python, SQL, cloud (AWS or Azure),
    and REST API development experience. Exposure to microservices and Docker
    is preferred.
    """

    top_resumes = rank_resumes_for_job(job_description, top_k=5, alpha=0.7)
    print(top_resumes.to_string(index=False))
