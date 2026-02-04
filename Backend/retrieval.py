import json
import faiss
import re
from pathlib import Path
from sentence_transformers import SentenceTransformer

# -----------------------------
# Load Dataset
# -----------------------------
DATA_PATH = Path("data/dataset.json")

with open(DATA_PATH, "r", encoding="utf-8") as f:
    DATA = json.load(f)

# -----------------------------
# Embedding Model (Multilingual)
# -----------------------------
embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# -----------------------------
# Normalize Text
# -----------------------------
def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())

# -----------------------------
# Exact Match Tables
# -----------------------------
EXACT_SI = {normalize(d["question_si"]): d for d in DATA}
EXACT_TA = {normalize(d["question_ta"]): d for d in DATA}

# -----------------------------
# Dual Semantic Indexes
# -----------------------------
DOCS_SI = [d["question_si"] for d in DATA]
DOCS_TA = [d["question_ta"] for d in DATA]

print(" Encoding Sinhala questions...")
EMB_SI = embedder.encode(DOCS_SI, normalize_embeddings=True)

print(" Encoding Tamil questions...")
EMB_TA = embedder.encode(DOCS_TA, normalize_embeddings=True)

index_si = faiss.IndexFlatIP(EMB_SI.shape[1])
index_ta = faiss.IndexFlatIP(EMB_TA.shape[1])

index_si.add(EMB_SI)
index_ta.add(EMB_TA)

print(" FAISS Semantic Index Ready")

# -----------------------------
# Semantic Search
# -----------------------------
def search(query: str, lang="si", k=5):

    q_emb = embedder.encode([query], normalize_embeddings=True)

    if lang == "si":
        scores, idxs = index_si.search(q_emb, k)
    else:
        scores, idxs = index_ta.search(q_emb, k)

    results = []
    for score, idx in zip(scores[0], idxs[0]):
        results.append({
            "score": float(score),
            "item": DATA[int(idx)]
        })

    return results
