# import pandas as pd
# import requests

# df = pd.read_csv("evaluation.csv")

# correct = 0

# print("✅ Running evaluation...")

# for _, row in df.iterrows():

#     res = requests.post(
#         "http://localhost:8000/chat",
#         json={
#             "message": row["input_question"],
#             "language": row["language"]
#         }
#     ).json()

#     expected = row["expected_behavior"].lower()
#     predicted = res["match_type"].lower()

#     # ✅ Accept fallback if semantic was expected (safe behavior)
#     if expected == "semantic" and predicted in ["semantic", "fallback"]:
#         correct += 1

#     elif predicted.startswith(expected):
#         correct += 1

# print("\n===================================")
# print("✅ Correct:", correct, "/", len(df))
# print("🎯 Accuracy:", round(correct / len(df) * 100, 2), "%")
# print("===================================")

import pandas as pd
import requests

API_URL = "http://localhost:8000/chat"
EVAL_FILE = "evaluation.csv"

df = pd.read_csv(EVAL_FILE)

print("\n✅ Running Thesis Evaluation on", len(df), "questions...\n")

# strict: expected must match predicted exactly
strict_correct = 0

# safe: semantic can be semantic OR clarification OR fallback (safe)
safe_correct = 0

expected_counts = {"exact": 0, "semantic": 0, "fallback": 0}
predicted_counts = {"exact": 0, "semantic": 0, "fallback": 0, "clarification": 0}

wrong_rows = []

for _, row in df.iterrows():

    question = row["input_question"]
    lang = row["language"]
    expected = row["expected_behavior"].lower()

    expected_counts[expected] += 1

    res = requests.post(API_URL, json={"message": question, "language": lang}).json()
    predicted = res["match_type"].lower()

    # normalize predicted categories
    if predicted.startswith("exact"):
        predicted_type = "exact"
    elif predicted.startswith("semantic"):
        predicted_type = "semantic"
    elif predicted.startswith("clarification"):
        predicted_type = "clarification"
    else:
        predicted_type = "fallback"

    predicted_counts[predicted_type] += 1

    # -------------------------
    # STRICT
    # -------------------------
    if expected == predicted_type:
        strict_correct += 1
    else:
        wrong_rows.append({
            "question": question,
            "language": lang,
            "expected": expected,
            "predicted": predicted_type,
            "bot_reply": res.get("reply", "")
        })

    # -------------------------
    # SAFE
    # semantic can be semantic OR clarification OR fallback
    # -------------------------
    if expected == "semantic" and predicted_type in ["semantic", "clarification", "fallback"]:
        safe_correct += 1
    elif expected == predicted_type:
        safe_correct += 1

total = len(df)

print("===================================")
print("📌 OVERALL RESULTS")
print("===================================")
print(f"Strict Accuracy: {strict_correct}/{total} = {strict_correct/total*100:.2f}%")
print(f"Safe Accuracy:   {safe_correct}/{total} = {safe_correct/total*100:.2f}%")

print("\n===================================")
print("📌 EXPECTED BEHAVIOR COUNTS")
print("===================================")
print(expected_counts)

print("\n===================================")
print("📌 PREDICTED MATCH TYPE COUNTS")
print("===================================")
print(predicted_counts)

wrong_file = "wrong_predictions_thesis.csv"
pd.DataFrame(wrong_rows).to_csv(wrong_file, index=False, encoding="utf-8-sig")

print("\n===================================")
print("📌 WRONG PREDICTIONS SAVED")
print("===================================")
print("Saved:", wrong_file)
print("Total Wrong:", len(wrong_rows))

print("\n✅ Thesis Evaluation Completed!\n")
