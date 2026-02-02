import json
import random
from pathlib import Path

# ==============================
# Paths
# ==============================
DATA_PATH = Path("data/dataset.json")

TRAIN_OUT = Path("train.jsonl")
VAL_OUT = Path("val.jsonl")

random.seed(42)

# ==============================
# System Prompt (Grounded + Safe)
# ==============================
SYSTEM_PROMPT = """You are CocoGuide, a bilingual coconut cultivation advisory assistant for farmers in Sri Lanka.

RULES:
1. Answer ONLY using the given Context.
2. Do NOT add new pesticide names, fertilizer amounts, or extra advice.
3. If the Context does not contain enough information, reply with:

   - Sinhala: "කණගාටුයි, මෙම ප්‍රශ්නයට ප්‍රමාණවත් තොරතුරු නොමැත. කරුණාකර කෘෂිකර්ම නිලධාරියෙකු අමතන්න."
   - Tamil: "மன்னிக்கவும், தேவையான தகவல் இல்லை. தயவுசெய்து வேளாண்மை அதிகாரியை தொடர்பு கொள்ளுங்கள்."

4. Respond ONLY in the user’s language.
"""

# ==============================
# Build Training Sample
# ==============================
def make_sample(question, answer):
    """
    Creates one grounded training prompt sample
    """
    return {
        "text": f"""### System:
{SYSTEM_PROMPT}

### Context:
{answer}

### Question:
{question}

### Answer:
"""
    }


# ==============================
# Main Function
# ==============================
def main():

    # Load dataset
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []

    # ------------------------------
    # Convert Sinhala + Tamil rows
    # ------------------------------
    for row in data:

        # Sinhala Q&A sample
        samples.append(
            make_sample(
                row["question_si"],
                row["answer_si"]
            )
        )

        # Tamil Q&A sample
        samples.append(
            make_sample(
                row["question_ta"],
                row["answer_ta"]
            )
        )

    print("✅ Total bilingual samples created:", len(samples))

    # ------------------------------
    # Shuffle samples
    # ------------------------------
    random.shuffle(samples)

    # ------------------------------
    # Split 80% Train / 20% Validation
    # ------------------------------
    split_index = int(len(samples) * 0.8)

    train_samples = samples[:split_index]
    val_samples = samples[split_index:]

    print("✅ Train samples:", len(train_samples))
    print("✅ Validation samples:", len(val_samples))

    # ------------------------------
    # Save JSONL Files
    # ------------------------------
    with open(TRAIN_OUT, "w", encoding="utf-8") as f:
        for s in train_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    with open(VAL_OUT, "w", encoding="utf-8") as f:
        for s in val_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print("\n🎉 Done!")
    print("Saved training file:", TRAIN_OUT)
    print("Saved validation file:", VAL_OUT)


# ==============================
# Run Script
# ==============================
if __name__ == "__main__":
    main()
