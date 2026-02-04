from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import FastAPI
from pydantic import BaseModel

from retrieval import search, EXACT_SI, EXACT_TA, normalize
from intents import detect_smalltalk, smalltalk_reply

from finetuned_llm import generate_grounded_answer

app = FastAPI()

# -----------------------------
# Enable  Access
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Toggle Fine-Tuned Model
# -----------------------------
USE_FINE_TUNED_MODEL = False   

# -----------------------------
# Request Schema
# -----------------------------
class ChatRequest(BaseModel):
    message: str
    language: str  # "si" or "ta"

# -----------------------------
# Messages
# -----------------------------
FALLBACK_SI = "කණගාටුයි, මට සහාය විය හැක්කේ පොල් වගාවට අදාළ කරුණු සඳහා පමණි. කරුණාකර ඔබේ ප්‍රශ්නය නැවත විමසන්න."
FALLBACK_TA = "மன்னிக்கவும், அந்தத் தகவல் தற்போது எங்களிடம் இல்லை. தயவுசெய்து மேலதிக ஆலோசனைகளுக்கு தென்னை பயிர்ச்செய்கை அதிகாரியைத் தொடர்பு கொள்ளவும்."

CLARIFY_SI = "කරුණාකර ඔබගේ ප්‍රශ්නය තව විස්තර කරන්න."
CLARIFY_TA = "தயவுசெய்து உங்கள் கேள்வியை மேலும் விளக்கவும்."


# =====================================================
# Chat Endpoint
# =====================================================
@app.post("/chat")
def chat(req: ChatRequest):

    print(" Question:", req.message)

    user_q = normalize(req.message)

    # -----------------------------
    # Smalltalk
    # -----------------------------
    kind = detect_smalltalk(req.message, req.language)
    if kind:
        return JSONResponse(content={
            "reply": smalltalk_reply(kind, req.language),
            "match_type": "smalltalk"
        })

    # -----------------------------
    # Exact Match
    # -----------------------------
    if req.language == "si" and user_q in EXACT_SI:
        best = EXACT_SI[user_q]
        source = "exact"

    elif req.language == "ta" and user_q in EXACT_TA:
        best = EXACT_TA[user_q]
        source = "exact"

    else:
        # -----------------------------
        # Semantic Search
        # -----------------------------
        hits = search(req.message, lang=req.language)

        top = hits[0]["score"]

        if top < 0.60:
            return JSONResponse(content={
                "reply": FALLBACK_TA if req.language == "ta" else FALLBACK_SI,
                "match_type": "fallback"
            })

        if top < 0.72:
            return JSONResponse(content={
                "reply": CLARIFY_TA if req.language == "ta" else CLARIFY_SI,
                "match_type": "clarification"
            })

        best = hits[0]["item"]
        source = "semantic"

    # -----------------------------
    # Answer Selection
    # -----------------------------
    context_answer = best["answer_ta"] if req.language == "ta" else best["answer_si"]

    # -----------------------------
    # Fine-Tuned Model Layer (Optional)
    # -----------------------------
    if USE_FINE_TUNED_MODEL:
        final_reply = generate_grounded_answer(req.message, context_answer, req.language)
    else:
        final_reply = context_answer

    return JSONResponse(content={
        "reply": final_reply,
        "match_type": source,
        "category": best.get("category", "")
    })
