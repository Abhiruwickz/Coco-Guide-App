# import json
# import faiss
# import re
# from pathlib import Path
# from sentence_transformers import SentenceTransformer

# # =====================================================
# # Load Dataset
# # =====================================================
# DATA_PATH = Path("data/dataset.json")

# with open(DATA_PATH, "r", encoding="utf-8") as f:
#     DATA = json.load(f)

# # =====================================================
# # Embedding Model
# # =====================================================
# embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# # =====================================================
# # Normalize Function
# # =====================================================
# def normalize(text: str) -> str:
#     text = text.lower()
#     text = re.sub(r"[^\w\s]", "", text)
#     return " ".join(text.split())

# # =====================================================
# # Exact Match Lookup Tables
# # =====================================================
# EXACT_SI = {normalize(d["question_si"]): d for d in DATA}
# EXACT_TA = {normalize(d["question_ta"]): d for d in DATA}

# # =====================================================
# # Build Dual FAISS Index
# # =====================================================

# DOCS_SI = [d["question_si"] for d in DATA]
# DOCS_TA = [d["question_ta"] for d in DATA]

# print("🔧 Encoding Sinhala questions...")
# EMB_SI = embedder.encode(DOCS_SI, normalize_embeddings=True)

# print("🔧 Encoding Tamil questions...")
# EMB_TA = embedder.encode(DOCS_TA, normalize_embeddings=True)

# # Create indexes
# index_si = faiss.IndexFlatIP(EMB_SI.shape[1])
# index_ta = faiss.IndexFlatIP(EMB_TA.shape[1])

# index_si.add(EMB_SI)
# index_ta.add(EMB_TA)

# print("✅ Dual FAISS Index Ready!")


# # =====================================================
# # Semantic Search Function
# # =====================================================
# def search(query: str, lang: str, k: int = 5, threshold: float = 0.65):
#     """
#     Semantic retrieval using correct language index.
#     Returns list of {"score": float, "item": row}
#     """

#     q_emb = embedder.encode([query], normalize_embeddings=True)

#     # Choose index
#     if lang == "si":
#         scores, idxs = index_si.search(q_emb, k)
#     else:
#         scores, idxs = index_ta.search(q_emb, k)

#     results = []

#     for score, idx in zip(scores[0], idxs[0]):
#         score = float(score)

#         if score < threshold:
#             continue

#         results.append({
#             "score": score,
#             "item": DATA[int(idx)]
#         })

#     return results

import json
import faiss
import re
from pathlib import Path
from sentence_transformers import SentenceTransformer

# =====================================================
# Load dataset
# =====================================================
DATA_PATH = Path("data/dataset.json")

with open(DATA_PATH, "r", encoding="utf-8") as f:
    DATA = json.load(f)

# =====================================================
# Embedding model
# =====================================================
embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# =====================================================
# Normalize
# =====================================================
def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())

# =====================================================
# Exact match lookup
# =====================================================
EXACT_SI = {normalize(d["question_si"]): d for d in DATA}
EXACT_TA = {normalize(d["question_ta"]): d for d in DATA}

# =====================================================
# Dual FAISS indexes (Sinhala only / Tamil only)
# =====================================================
DOCS_SI = [d["question_si"] for d in DATA]
DOCS_TA = [d["question_ta"] for d in DATA]

print("🔧 Encoding Sinhala questions...")
EMB_SI = embedder.encode(DOCS_SI, normalize_embeddings=True)

print("🔧 Encoding Tamil questions...")
EMB_TA = embedder.encode(DOCS_TA, normalize_embeddings=True)

index_si = faiss.IndexFlatIP(EMB_SI.shape[1])
index_ta = faiss.IndexFlatIP(EMB_TA.shape[1])

index_si.add(EMB_SI)
index_ta.add(EMB_TA)

print("✅ Dual FAISS indexes ready")

# =====================================================
# Semantic search (returns top scores for gating)
# =====================================================
def search(query: str, lang: str, k: int = 5):
    """
    Return list: [{"score": float, "item": DATA_ROW}, ...]
    NOTE: No threshold here -> main.py decides semantic vs fallback.
    """
    q_emb = embedder.encode([query], normalize_embeddings=True)

    if lang == "si":
        scores, idxs = index_si.search(q_emb, k)
    else:
        scores, idxs = index_ta.search(q_emb, k)

    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx < 0:
            continue
        results.append({
            "score": float(score),
            "item": DATA[int(idx)]
        })

    return results
