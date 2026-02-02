import pandas as pd
import json
from pathlib import Path

EXCEL_PATH = Path("./Coconut_Advisory_Dataset.xlsx")
OUT_JSON = Path("data/dataset.json")

def main():
    df = pd.read_excel(EXCEL_PATH)
    df.columns = [c.strip() for c in df.columns]

    rows = []
    for _, r in df.iterrows():
        rows.append({
            "id": int(r["id"]),
            "question_si": str(r["question_si"]).strip(),
            "question_ta": str(r["question_ta"]).strip(),
            "answer_si": str(r["answer_si"]).strip(),
            "answer_ta": str(r["answer_ta"]).strip(),
            "category": str(r["category"]).strip(),
            "keywords": str(r.get("Keywords", "")).strip(),
        })

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print("✅ Saved dataset rows:", len(rows))
    if rows:
        print("🔍 First row:", rows[0])

if __name__ == "__main__":
    main()
