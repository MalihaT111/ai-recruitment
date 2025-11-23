import os
import json
import numpy as np
import pandas as pd
import docx
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# -------------------------------------------
# 1. FEATURE CLEANER
# -------------------------------------------
from feature_extractors import clean_text_for_domain


# -------------------------------------------
# 2. READ A SINGLE DOCX FILE
# -------------------------------------------
def load_docx(path):
    """
    Reads a .docx file and returns plain text.
    """
    doc = docx.Document(path)
    text = "\n".join([p.text for p in doc.paragraphs]).strip()
    return text


# -------------------------------------------
# 3. READ ALL RESUMES FROM FOLDER
# -------------------------------------------
def load_all_resumes(folder):
    """
    Loads and cleans all .docx resumes in a folder.
    
    Returns:
        texts      -> list of cleaned resume text
        filenames  -> list of filenames
    """
    files = sorted([f for f in os.listdir(folder) if f.lower().endswith(".docx")])
    
    texts = []
    filenames = []

    print(f"Found {len(files)} resumes in {folder}\n")

    for f in tqdm(files, desc="Reading resumes"):
        full_path = os.path.join(folder, f)
        raw_text = load_docx(full_path)
        clean = clean_text_for_domain(raw_text)

        texts.append(clean)
        filenames.append(f)

    return texts, filenames


# -------------------------------------------
# 4. MAIN PIPELINE
# -------------------------------------------
def run(folder="../Resumes/", outdir="../precomputed/"):
    os.makedirs(outdir, exist_ok=True)

    print("STEP 1: Loading and cleaning resumes...")
    texts, filenames = load_all_resumes(folder)

    print("\nSTEP 2: Embedding resumes...")
    embedder = SentenceTransformer("intfloat/e5-large-v2")

    vectors = embedder.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    # Save embeddings
    np.save(f"{outdir}/resume_vectors.npy", vectors)

    # Save cleaned text
    pd.DataFrame({"resume_text": texts}).to_csv(
        f"{outdir}/resume_texts.csv", index=False
    )

    # Save filename→ID mapping
    with open(f"{outdir}/resume_index.json", "w") as f:
        json.dump({i: filenames[i] for i in range(len(filenames))}, f, indent=2)

    print("\n=======================================")
    print(" DONE — Precomputed resume embeddings ")
    print(" Saved to:", outdir)
    print("=======================================\n")


# -------------------------------------------
# 5. RUN
# -------------------------------------------
if __name__ == "__main__":
    run()
