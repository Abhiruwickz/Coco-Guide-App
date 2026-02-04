import pandas as pd
import requests

API_URL = "http://127.0.0.1:8000/chat"
EVAL_FILE = "evaluation.csv"

df = pd.read_csv(EVAL_FILE)

correct = 0

print("\n Running Prototype Evaluation on", len(df), "questions...\n")

for _, row in df.iterrows():

    question = row["input_question"]
    lang = row["language"]
    expected = str(row["expected_behavior"]).strip().lower()

    res = requests.post(API_URL, json={
        "message": question,
        "language": lang
    }).json()

    predicted = res["match_type"].strip().lower()

    # Normalize predicted types
    if predicted.startswith("exact"):
        predicted = "exact"
    elif predicted.startswith("semantic"):
        predicted = "semantic"
    elif predicted.startswith("fallback"):
        predicted = "fallback"
    elif predicted.startswith("clarification"):
        predicted = "semantic"   

    # Count correct
    if expected == predicted:
        correct += 1

accuracy = correct / len(df)

print("===================================")
print("Correct:", correct, "/", len(df))
print("Accuracy:", round(accuracy * 100, 2), "%")
print("===================================")
