import sys
import os
sys.path.append('../notebooks')
sys.path.append('../utils')

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util

from feature_extractors import (
    clean_text_for_domain,
    detect_domain,
    skill_match,
    education_score,
    seniority_score,
    keyword_overlap,
)
from rank import rank_resumes_for_job

app = FastAPI(title="AI Recruitment API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    df = rank_resumes_for_job(
        job_text=payload.job_description,
        top_k=payload.top_k,
        alpha=payload.alpha,
    )
    return df.to_dict(orient="records")

@app.get("/")
def root():
    return {"message": "AI Recruitment API is running"}