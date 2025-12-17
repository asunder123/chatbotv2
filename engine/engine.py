# engine.py
# Stable document QA engine
# - Sentence-level semantic retrieval
# - Query paraphrasing
# - EXPLANATORY answers
# - FORENSIC evidence (clearly distinct)

import re
from typing import List, Dict
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# Light structural cleanup
# ============================================================

def extract_core_text(text: str) -> str:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    cleaned = []
    for l in lines:
        if re.fullmatch(r"page\s*\d+", l.lower()):
            continue
        cleaned.append(l)
    return " ".join(cleaned)


# ============================================================
# Sentence handling
# ============================================================

def split_sentences(text: str) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [
        s.strip()
        for s in sentences
        if len(s.strip()) > 12 and re.search(r"[A-Za-z]{3,}", s)
    ]


# ============================================================
# Query paraphrasing (maneuverability)
# ============================================================

PARAPHRASE_MAP = {
    "why": ["what caused", "reason for", "root cause of"],
    "cause": ["reason", "trigger", "source"],
    "fail": ["failure", "breakdown", "issue"],
    "issue": ["problem", "incident", "failure"],
    "fix": ["resolve", "mitigate", "correct"],
    "impact": ["effect", "consequence", "affected"],
    "when": ["time", "date", "duration"],
}

def generate_query_paraphrases(query: str) -> List[str]:
    query = query.lower()
    variants = {query}
    for key, repls in PARAPHRASE_MAP.items():
        if key in query:
            for r in repls:
                variants.add(query.replace(key, r))
    return list(variants)


# ============================================================
# Answer paraphrasing (EXPLANATORY)
# ============================================================

ACRONYM_MAP = {
    "svc": "service",
    "cfg": "configuration",
    "db": "database",
    "auth": "authentication",
    "lat": "latency",
}

def paraphrase_answer(sentence: str) -> str:
    """
    Turns evidence into a human explanation.
    """
    s = sentence.strip()

    rewrites = [
        (r"\bwas caused by\b", "happened because of"),
        (r"\bled to\b", "which resulted in"),
        (r"\bresulted in\b", "which caused"),
        (r"\bdue to\b", "because of"),
        (r"\bwas observed\b", "was noticed"),
        (r"\bexperienced\b", "encountered"),
    ]

    for pat, repl in rewrites:
        s = re.sub(pat, repl, s, flags=re.IGNORECASE)

    words = s.split()
    expanded = []
    for w in words:
        key = w.lower().strip(".,()")
        expanded.append(ACRONYM_MAP.get(key, w))

    explanation = " ".join(expanded)

    # Make it explicitly explanatory
    explanation = explanation[0].upper() + explanation[1:]
    return explanation


# ============================================================
# Evidence formatting (FORENSIC)
# ============================================================

def format_evidence(sentence: str) -> str:
    """
    Makes evidence look like a quoted source, not an answer.
    """
    return f"“{sentence.strip()}”"


# ============================================================
# Core Index
# ============================================================

class CaseIndex:
    def __init__(self):
        self.sentences: List[str] = []
        self.vectorizer = None
        self.matrix = None

    def build_from_text(self, raw_text: str):
        core = extract_core_text(raw_text)
        self.sentences = split_sentences(core)

        if not self.sentences:
            self.matrix = None
            return

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2)
        )
        self.matrix = self.vectorizer.fit_transform(self.sentences)

    def query(self, query: str, top_k: int = 2) -> List[Dict]:
        if not self.sentences or self.matrix is None:
            return []

        queries = generate_query_paraphrases(query)
        best_scores = np.zeros(len(self.sentences))

        for q in queries:
            try:
                q_vec = self.vectorizer.transform([q])
            except Exception:
                continue
            scores = cosine_similarity(q_vec, self.matrix)[0]
            best_scores = np.maximum(best_scores, scores)

        ranked = np.argsort(best_scores)[::-1]

        results = []
        for idx in ranked[:top_k]:
            if best_scores[idx] <= 0:
                continue

            raw = self.sentences[idx]

            results.append({
                "answer": paraphrase_answer(raw),     # human explanation
                "evidence": format_evidence(raw),     # quoted source
                "score": float(best_scores[idx])
            })

        return results
