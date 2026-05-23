"""
rebuild_model.py
----------------
Place this file in:  Fake-Content-Detection-Social-Media-main/model/
Run with:            python3 rebuild_model.py

Requires Fake.csv and True.csv in ../Datasets/
Install deps first:  pip3 install pandas scikit-learn xgboost nltk
"""

import os, re, pickle
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import WordPunctTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

nltk.download("stopwords")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, "..", "Datasets")

# ── 1. Load & label raw data ─────────────────────────────────────────────────
print("Loading datasets...")
fake = pd.read_csv(os.path.join(DATASETS_DIR, "Fake.csv"))
real = pd.read_csv(os.path.join(DATASETS_DIR, "True.csv"))

fake["class"] = 0
real["class"] = 1

df = pd.concat([fake, real], ignore_index=True)

# Build combined text column (title + body if available)
if "title" in df.columns and "text" in df.columns:
    df["title_text"] = df["title"].fillna("") + " " + df["text"].fillna("")
elif "title" in df.columns:
    df["title_text"] = df["title"].fillna("")
else:
    df["title_text"] = df.iloc[:, 0].fillna("")

df = df.dropna(subset=["title_text"])
print(f"Dataset size: {df.shape[0]} rows")

# ── 2. Text preprocessing ────────────────────────────────────────────────────
ps = PorterStemmer()

def text_processing(text):
    text = str(text)
    token = WordPunctTokenizer()
    stop_words = set(stopwords.words("english"))
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = re.sub(r"[0-9]", " ", text)
    text = [ps.stem(w) for w in token.tokenize(text.lower()) if w not in stop_words]
    return " ".join(text)

print("Processing text (this may take a few minutes)...")
processed_text = [text_processing(t) for t in df["title_text"]]

# ── 3. Vectorise ─────────────────────────────────────────────────────────────
print("Fitting TF-IDF vectorizer...")
tfidf = TfidfVectorizer(
    strip_accents=None, lowercase=False, preprocessor=None,
    tokenizer=None, use_idf=True, norm="l2",
    smooth_idf=True, ngram_range=(1, 3)
)
X = tfidf.fit_transform(processed_text)
y = df["class"]

# ── 4. Train / test split ────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── 5. Train XGBoost model ───────────────────────────────────────────────────
print("Training XGBoost model (this may take several minutes)...")
xgb = XGBClassifier(eval_metric="logloss")
xgb.fit(X_train, y_train)

preds = xgb.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")

# ── 6. Save model & vectorizer ───────────────────────────────────────────────
pickle.dump(tfidf, open(os.path.join(BASE_DIR, "vectorizer.pkl"), "wb"))
pickle.dump(xgb,   open(os.path.join(BASE_DIR, "model.pkl"),      "wb"))
print("✅  Saved vectorizer.pkl and model.pkl — you can now run python3 app.py")
