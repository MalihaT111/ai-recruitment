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
            "resume_preview": resume_clean[:300],
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
📌 Job Posting: Senior Java Full-Stack Engineer (Remote / NYC Preferred)

Senior Java Full-Stack Engineer — SaaS Platform (Remote, US)

We are seeking a Senior Java Full-Stack Engineer to join our engineering team responsible for building scalable, cloud-native SaaS applications. The ideal candidate has strong backend experience in Java, Spring Boot, Microservices, and modern frontend frameworks such as React or Angular.

Responsibilities

Design, develop, and maintain scalable backend services using Java / Spring Boot

Build and consume RESTful APIs and microservices

Implement front-end features using React, Angular, or similar

Optimize application performance, reliability, and security

Collaborate with product managers, QA, and DevOps to deliver high-quality features

Contribute to architecture decisions and technical design discussions

Participate in code reviews, testing, and CI/CD automation

Required Qualifications

5+ years professional experience in software engineering

Strong proficiency in Java, Spring Boot, JPA/Hibernate, and REST APIs

Hands-on experience with JavaScript, TypeScript, React, or Angular

Experience with AWS, Docker, or Kubernetes

Knowledge of relational databases (MySQL/PostgreSQL)

Understanding of microservices, distributed systems, and event-driven architecture

Nice to Have

Experience with Kafka, RabbitMQ, or other message queues

CI/CD using GitHub Actions, Jenkins, or GitLab

Experience working with offshore teams

Familiarity with Agile methodologies
"""

top10 = rank_resumes(job_posting, top_k=10, alpha=0.7)
print(top10.to_string(index=False))
