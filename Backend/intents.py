import re

SI_GREET = ["ආයුබෝවන්", "හෙලෝ","හලෝ", "හායි", "සුභ උදෑසනක්", "සුභ සන්ධ්‍යාවක්"]
TA_GREET = ["வணக்கம்", "ஹலோ", "ஹாய்", "காலை வணக்கம்", "மாலை வணக்கம்"]

SI_THANKS = ["ස්තුතියි", "බොහොම ස්තුතියි", "තෑන්ක්ස්"]
TA_THANKS = ["நன்றி", "மிக்க நன்றி", "தாங்க்ஸ்"]

def detect_smalltalk(text: str, lang: str):
    t = text.strip().lower()
    if lang == "si":
        if any(w.lower() in t for w in SI_GREET):
            return "greet"
        if any(w.lower() in t for w in SI_THANKS):
            return "thanks"
    if lang == "ta":
        if any(w.lower() in t for w in TA_GREET):
            return "greet"
        if any(w.lower() in t for w in TA_THANKS):
            return "thanks"
    if re.fullmatch(r"(hi|hello|hey|thanks|thank you)", t):
        return "greet" if "h" in t else "thanks"
    return None

def smalltalk_reply(kind: str, lang: str):
    if lang == "si":
        if kind == "greet":
            return "ආයුබෝවන්! ඔබට පොල් වගාව ගැන මොනවාද අහන්න තියෙන්නේ?"
        if kind == "thanks":
            return "ස්තුතියි! තවත් ප්‍රශ්නයක් තිබ්බොත් අහන්න."
    if lang == "ta":
        if kind == "greet":
            return "வணக்கம்! தேங்காய் சாகுபடி பற்றி என்ன கேட்க விரும்புகிறீர்கள்?"
        if kind == "thanks":
            return "நன்றி! இன்னும் கேள்விகள் இருந்தால் கேளுங்கள்."
    return "Hi!"
