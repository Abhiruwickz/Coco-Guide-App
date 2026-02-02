# from fastapi.middleware.cors import CORSMiddleware
# from fastapi import FastAPI
# from pydantic import BaseModel

# from retrieval import search, EXACT_SI, EXACT_TA, normalize
# from intents import detect_smalltalk, smalltalk_reply

# # =====================================================
# # FastAPI App
# # =====================================================
# app = FastAPI()

# # Enable CORS for Expo Go
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # =====================================================
# # Request Schema
# # =====================================================
# class ChatRequest(BaseModel):
#     message: str
#     language: str  # "si" or "ta"

# # =====================================================
# # Fallback Replies
# # =====================================================
# FALLBACK_SI = (
#     "කණගාටුයි, මෙම ප්‍රශ්නයට ප්‍රමාණවත් තොරතුරු නොමැත. "
#     "කරුණාකර පොල් වගාව හා සම්බන්ධ තොරතුරු පමණක් විමසන්න.."
# )

# FALLBACK_TA = (
#     "மன்னிக்கவும், தேவையான தகவல் இல்லை. "
#     "தயவுசெய்து தென்னை பயிர்ச்செய்கை தொடர்பான தகவல்களை மாத்திரம் வினவவும்.."
# )

# # =====================================================
# # Chat Endpoint
# # =====================================================
# @app.post("/chat")
# def chat(req: ChatRequest):

#     user_q = normalize(req.message)

#     # -----------------------------
#     # 1) Smalltalk
#     # -----------------------------
#     kind = detect_smalltalk(req.message, req.language)
#     if kind:
#         return {
#             "reply": smalltalk_reply(kind, req.language),
#             "match_type": "smalltalk"
#         }

#     best = None
#     source = None

#     # -----------------------------
#     # 2) Exact Match
#     # -----------------------------
#     if req.language == "si" and user_q in EXACT_SI:
#         best = EXACT_SI[user_q]
#         source = "exact-si"

#     elif req.language == "ta" and user_q in EXACT_TA:
#         best = EXACT_TA[user_q]
#         source = "exact-ta"

#     else:
#         # -----------------------------
#         # 3) Semantic Search (Balanced Threshold)
#         # -----------------------------

#         # Sinhala slightly higher than Tamil
#         threshold = 0.68 if req.language == "si" else 0.65

#         hits = search(
#             req.message,
#             lang=req.language,
#             k=5,
#             threshold=threshold
#         )

#         if not hits:
#             return {
#                 "reply": FALLBACK_TA if req.language == "ta" else FALLBACK_SI,
#                 "match_type": "fallback"
#             }

#         # Confidence margin check (lighter)
#         top1 = hits[0]["score"]
#         top2 = hits[1]["score"] if len(hits) > 1 else 0.0

#         if (top1 - top2) < 0.03:
#             return {
#                 "reply": FALLBACK_TA if req.language == "ta" else FALLBACK_SI,
#                 "match_type": "fallback"
#             }

#         best = hits[0]["item"]
#         source = "semantic"

#     # -----------------------------
#     # 4) Return Dataset Answer
#     # -----------------------------
#     final_reply = best["answer_ta"] if req.language == "ta" else best["answer_si"]

#     return {
#         "reply": final_reply,
#         "match_type": source,
#         "matched_question_si": best.get("question_si", ""),
#         "matched_question_ta": best.get("question_ta", ""),
#         "category": best.get("category", "")
#     }

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from pydantic import BaseModel

from retrieval import search, EXACT_SI, EXACT_TA, normalize
from intents import detect_smalltalk, smalltalk_reply
from finetuned_llm import generate_grounded_answer

app = FastAPI()

# Enable CORS for Expo Go
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    language: str  # "si" or "ta"

# Fallback
FALLBACK_SI = "කණගාටුයි, මෙම ප්‍රශ්නයට ප්‍රමාණවත් තොරතුරු නොමැත. කරුණාකර කෘෂිකර්ම නිලධාරියෙකු අමතන්න."
FALLBACK_TA = "மன்னிக்கவும், தேவையான தகவல் இல்லை. தயவுசெய்து வேளாண்மை அதிகாரியை தொடர்பு கொள்ளுங்கள்."

# Clarification message (middle-confidence range)
CLARIFY_SI = "ඔබ අසන ප්‍රශ්නය තව ටිකක් පැහැදිලි කරලා කියන්න පුළුවන්ද? (උදා: රෝගය/පලිබෝධය/පොහොර/ජලය ගැනද?)"
CLARIFY_TA = "உங்கள் கேள்வியை இன்னும் சற்று தெளிவாக சொல்ல முடியுமா? (உதா: நோய்/பூச்சி/உரம்/நீர் பற்றியதா?)"


@app.post("/chat")
def chat(req: ChatRequest):

    user_q = normalize(req.message)

    # 1) smalltalk
    kind = detect_smalltalk(req.message, req.language)
    if kind:
        return {"reply": smalltalk_reply(kind, req.language), "match_type": "smalltalk"}

    # 2) exact match (always best)
    if req.language == "si" and user_q in EXACT_SI:
        best = EXACT_SI[user_q]
        context_answer = best["answer_si"]
        return {
            "reply": context_answer,
            "match_type": "exact-si",
            "matched_question_si": best.get("question_si", ""),
            "matched_question_ta": best.get("question_ta", ""),
            "category": best.get("category", "")
        }

    if req.language == "ta" and user_q in EXACT_TA:
        best = EXACT_TA[user_q]
        context_answer = best["answer_ta"]
        return {
            "reply": context_answer,
            "match_type": "exact-ta",
            "matched_question_si": best.get("question_si", ""),
            "matched_question_ta": best.get("question_ta", ""),
            "category": best.get("category", "")
        }

    # 3) semantic candidates (no threshold in retrieval, threshold here)
    hits = search(req.message, lang=req.language, k=5)

    if not hits:
        return {"reply": FALLBACK_TA if req.language == "ta" else FALLBACK_SI, "match_type": "fallback"}

    top1 = hits[0]["score"]
    top2 = hits[1]["score"] if len(hits) > 1 else 0.0
    margin = top1 - top2

    # Language-specific confidence ranges (tuned for Sinhala/Tamil)
    HIGH = 0.70 if req.language == "si" else 0.68
    MID = 0.60 if req.language == "si" else 0.58
    MARGIN = 0.03

    # A) high confidence -> semantic
    if top1 >= HIGH and margin >= MARGIN:
        best = hits[0]["item"]
        context_answer = best["answer_ta"] if req.language == "ta" else best["answer_si"]

        final = generate_grounded_answer(req.message, context_answer, req.language)

        return {
            "reply": final,
            "match_type": "semantic",
            "matched_question_si": best.get("question_si", ""),
            "matched_question_ta": best.get("question_ta", ""),
            "category": best.get("category", ""),
            "score": round(top1, 3)
        }

    # B) mid confidence -> clarification question (better than wrong answer)
    if top1 >= MID:
        return {
            "reply": CLARIFY_TA if req.language == "ta" else CLARIFY_SI,
            "match_type": "clarification",
            "score": round(top1, 3)
        }

    # C) low confidence -> fallback
    return {
        "reply": FALLBACK_TA if req.language == "ta" else FALLBACK_SI,
        "match_type": "fallback",
        "score": round(top1, 3)
    }
