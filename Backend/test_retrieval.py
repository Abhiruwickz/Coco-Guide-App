from retrieval import search

res = search("පොල් කොළ කහ වෙන හේතුව මොකක්ද?", k=3)
for r in res:
    print(r["score"], r["item"]["question_si"])
