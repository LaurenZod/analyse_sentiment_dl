# scripts/utils_text.py
import re

# Regex réutilisables
URL_RE = re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE)
USER_RE = re.compile(r'@\w+')
HASHTAG_RE = re.compile(r'#(\w+)')
MULTISPACE_RE = re.compile(r'\s+')

# ---  A. Normalisation "légère" (pour DL et BERT) ---
def normalize_light(text: str) -> str:
    # Remplace URL/USER par des tokens neutres (utile pour BERT/CNN/LSTM)
    t = URL_RE.sub(" <URL> ", str(text))
    t = USER_RE.sub(" <USER> ", t)
    # garde le mot du hashtag (#happy -> happy)
    t = HASHTAG_RE.sub(r"\1", t)
    t = MULTISPACE_RE.sub(" ", t).strip()
    return t

# ---  B. Transform pour BoW/TF-IDF (classique) ---
def transform_bow(text: str) -> str:
    # TF-IDF : minuscule + suppression brute des URL/USER + hashtag->mot
    t = str(text).lower()
    t = URL_RE.sub(" ", t)
    t = USER_RE.sub(" ", t)
    t = HASHTAG_RE.sub(r"\1", t)
    # enlève ponctuation isolée mais garde les apostrophes/négations
    t = re.sub(r"[^a-z0-9' ]+", " ", t)
    t = MULTISPACE_RE.sub(" ", t).strip()
    return t

