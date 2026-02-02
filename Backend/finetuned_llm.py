# import os
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
# from peft import PeftModel

# # -----------------------------
# # Model Paths
# # -----------------------------
# BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
# LORA_PATH = "./qwen-coconut-grounded-lora"

# print("🔧 Loading tokenizer...")
# tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
# tokenizer.pad_token = tokenizer.eos_token

# # -----------------------------
# # 4-bit Quantization Config
# # -----------------------------
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.float16,
#     bnb_4bit_use_double_quant=True
# )

# print("🔧 Loading base model in 4-bit...")
# base_model = AutoModelForCausalLM.from_pretrained(
#     BASE_MODEL,
#     quantization_config=bnb_config,
#     device_map="auto"
# )

# print("🔧 Loading LoRA adapter...")
# model = PeftModel.from_pretrained(base_model, LORA_PATH)

# model.eval()
# print("✅ Fine-tuned Qwen model loaded successfully!")


# # =====================================================
# # Grounded Answer Generator (Question + Context)
# # =====================================================
# def generate_grounded_answer(user_question: str, context: str, lang: str) -> str:

#     lang_name = "Sinhala" if lang == "si" else "Tamil"

#     prompt = f"""
# You are CocoGuide, a coconut farming assistant.

# STRICT RULES:
# 1. Use ONLY the information in the given Context.
# 2. Do NOT add extra fertilizer amounts, pesticide names, or new advice.
# 3. If context is not enough, ask the user to contact an agriculture officer.
# 4. Reply ONLY in {lang_name}.

# Context:
# {context}

# User Question:
# {user_question}

# Answer:
# """.strip()

#     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

#     output = model.generate(
#         **inputs,
#         max_new_tokens=150,
#         temperature=0.2,
#          do_sample=False,
#     repetition_penalty=1.15,
#     no_repeat_ngram_size=3
#     )

#     decoded = tokenizer.decode(output[0], skip_special_tokens=True)

#     # Return only generated part after "Answer:"
#     return decoded.split("Answer:")[-1].strip()
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os

# =====================================================
# CONFIG
# =====================================================
BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"   # change only if you trained on another base
ADAPTER_PATH = "./qwen-coconut-grounded-lora"  # your downloaded adapter folder

# =====================================================
# Load Model + Tokenizer (GPU if available)
# =====================================================
print("🔧 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

print("🔧 Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    trust_remote_code=True
)

print("🔧 Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

# Disable tokenizer parallel warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# =====================================================
# Grounded generator
# =====================================================
def generate_grounded_answer(user_question: str, context_answer: str, lang: str) -> str:
    """
    Generates grounded answer using fine-tuned Qwen adapter.
    Uses strict prompt and anti-repeat decoding.
    """

    if lang == "si":
        lang_name = "Sinhala"
        fallback = "කණගාටුයි, මෙම ප්‍රශ්නයට ප්‍රමාණවත් තොරතුරු නොමැත. කරුණාකර කෘෂිකර්ම නිලධාරියෙකු අමතන්න."
    else:
        lang_name = "Tamil"
        fallback = "மன்னிக்கவும், தேவையான தகவல் இல்லை. தயவுசெய்து வேளாண்மை அதிகாரியை தொடர்பு கொள்ளுங்கள்."

    prompt = f"""
You are CocoGuide, a bilingual coconut cultivation advisory assistant for Sri Lankan farmers.

STRICT RULES:
1) Use ONLY the information in Context.
2) Do NOT add new pesticide names, fertilizer amounts, or extra advice.
3) If Context does not contain enough information, reply exactly with the fallback message.
4) Respond ONLY in {lang_name}. No English.
5) Keep it short and friendly (2–5 sentences).

Fallback message:
{fallback}

Context:
{context_answer}

User Question:
{user_question}

Answer:
""".strip()

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=180,
            do_sample=False,                 # deterministic
            repetition_penalty=1.15,         # stop loops
            no_repeat_ngram_size=3,          # prevent repeating phrases
            eos_token_id=tokenizer.eos_token_id
        )

    text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Extract only answer section (after "Answer:")
    if "Answer:" in text:
        answer = text.split("Answer:")[-1].strip()
    else:
        answer = text.strip()

    # Safety: if output too short or broken, fallback to context
    if len(answer) < 10:
        return context_answer

    return answer
