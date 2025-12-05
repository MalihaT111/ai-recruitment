import numpy as np
import pandas as pd
import joblib
import time
from sentence_transformers import SentenceTransformer, util

# ----------------------------
# 1. IMPORT FEATURE UTILITIES
# ----------------------------
from feature_extractors import (
    clean_text_for_domain,
    detect_domain,
    skill_match,
    education_score,
    seniority_score,
    keyword_overlap,
)

# ----------------------------
# 2. LOAD MODEL + EMBEDDER
# ----------------------------
model = joblib.load("../models/model_random_forest.pkl")
embedder = SentenceTransformer("intfloat/e5-large-v2")

MODEL_FEATURES = [
    "keyword_overlap",
    "skill_score",
    "experience_score",
    "education_score",
    "domain_match",
]

# ----------------------------
# 3. LOAD PRECOMPUTED RESUME EMBEDDINGS
# ----------------------------
# resume_vectors = np.load("../precomputed/resume_vectors.npy")
# resume_texts = pd.read_csv("../precomputed/resume_texts.csv")["resume_text"].fillna("").astype(str).tolist()

resume_vectors = np.load("../data/embeddings/resume_emb_e5_large.npy")
resume_texts = pd.read_csv("../data/original/resumes_cleaned.csv")["Resume_clean"] \
                    .fillna("").astype(str).tolist()



# ----------------------------
# 4. MAIN RANKING FUNCTION (FAST)
# ----------------------------
def rank_resumes(job_text, top_k=5, alpha=0.5):
    """
    job_text : string (job posting)
    top_k    : how many resumes to return
    alpha    : weight for ML model vs cosine similarity
    """

    print("=== Ranking Started (FAST MODE) ===")
    t0 = time.time()

    # Clean job text
    job_clean = clean_text_for_domain(job_text)

    # Embed job text
    t_embed_job_0 = time.time()
    job_vec = embedder.encode(job_clean, convert_to_numpy=True)
    t_embed_job_1 = time.time()

    # Compute cosine similarity to ALL resumes at once
    cos_sims = util.cos_sim(job_vec, resume_vectors)[0].cpu().numpy()

    scored = []

    for i, resume_clean in enumerate(resume_texts):

        # ML FEATURES
        kw   = keyword_overlap(job_clean, resume_clean)
        skl  = skill_match(job_clean, resume_clean)
        exp  = seniority_score(resume_clean)
        edu  = education_score(resume_clean)

        job_dom, _ = detect_domain(job_clean)
        res_dom, _ = detect_domain(resume_clean)
        dom_match = int(job_dom == res_dom)

        X = pd.DataFrame([{
            "keyword_overlap": kw,
            "skill_score": skl,
            "experience_score": exp,
            "education_score": edu,
            "domain_match": dom_match,
        }])

        ml_prob = float(model.predict_proba(X)[0, 1])
        cos_sim = float(cos_sims[i])

        final = alpha * ml_prob + (1 - alpha) * cos_sim

        scored.append({
            "resume_idx": i,
            "final_score": final,
            "ml_prob": ml_prob,
            "cos_sim": cos_sim,
            "resume_text": resume_clean,
        })

    df = pd.DataFrame(scored)
    df = df.sort_values("final_score", ascending=False).head(top_k)

    # TIMING
    t1 = time.time()
    print("\n=== TIMING REPORT ===")
    print(f"Total time: {t1 - t0:.4f} sec")
    print(f"Job embedding time: {t_embed_job_1 - t_embed_job_0:.4f} sec")
    print("=====================\n")

    return df


# ----------------------------
# 5. EXAMPLE RUN
# ----------------------------
job_posting = """
Software Engineer, New Grad (2025 Start)

Location: New York, NY (Hybrid)
Team: Engineering
Employment Type: Full-time

About the Role

We’re looking for a Software Engineer to join our growing engineering organization. In this role, you will build systems that power customer-facing products, optimize performance at scale, and contribute to high-impact features across our platform. You’ll work closely with senior engineers, product managers, and designers to ship production-ready code and influence the future of our technical stack.

What You’ll Do

Design, build, and maintain backend services, APIs, and distributed systems.

Collaborate across engineering, product, and design to deliver new features end-to-end.

Own components from proposal to deployment: writing specs, implementing solutions, testing, and monitoring.

Improve system performance, reliability, scalability, and observability.

Participate in code reviews, architecture discussions, and team-wide technical planning.

Build internal tools that enhance developer productivity and automate workflows.

What We’re Looking For

BS or MS in Computer Science, Engineering, or related field (completed by June 2025).

Strong foundation in data structures, algorithms, and systems programming.

Experience with at least one of: Python, Java, TypeScript/Node.js, Go, or C++.

Understanding of distributed systems, APIs, databases, or cloud-native development.

Ability to break complex problems into clean, maintainable solutions.

Strong communication and willingness to work collaboratively on cross-functional teams.

Nice to Have

Experience with cloud platforms (AWS, GCP, Azure) or containerization (Docker, Kubernetes).

Familiarity with React, Next.js, or modern frontend frameworks.

Hands-on experience with ML pipelines, data engineering, or large-scale system design.

Internship or project experience in a production-like environment.

Why Join Us

Work on meaningful, high-impact systems used by thousands of customers.

Mentorship from senior engineers and clear growth pathways.

Competitive salary, equity, benefits, and wellness support.

A team that values craftsmanship, curiosity, and continuous learning.

Inclusive and human-centered engineering culture.

How to Apply

Submit your résumé, GitHub/portfolio link, and a brief note about a technical project you’re proud of.
"""

top10 = rank_resumes(job_posting, top_k=1, alpha=0.9)
print(top10.to_string(index=False))
