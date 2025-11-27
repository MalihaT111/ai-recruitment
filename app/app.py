from fastapi import FastAPI
from app.rank import router as rank_router

# -------------------------------------------------
# FASTAPI APP INITIALIZATION
# -------------------------------------------------
app = FastAPI(
    title="AI Recruitment Ranking API",
    description="ML-powered resume ranking for job descriptions",
    version="1.0.0"
)

# Register routes
app.include_router(rank_router)


@app.get("/")
def root():
    """Health check endpoint"""
    return {"message": "AI Recruitment Ranking API is up"}
