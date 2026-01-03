import re
import nltk
from nltk.corpus import stopwords

STOPWORDS = set (stopwords.words("english"))

def clean_text(text:str) -> str:
    text = text.lower()
    text = re.sub(r"[a-zA-Z\s]","",text)
    tokens = [ w for w in text.split() if w not in STOPWORDS]
    return " ".join(tokens)
