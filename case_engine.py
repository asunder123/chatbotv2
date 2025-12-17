# case_engine.py
# Deterministic case QA engine
# Strong paraphraseability + AGILE narrowing
# No optional dependencies, no module exceptions

import re
import json
import zipfile
from typing import List, Dict
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# Intent & Paraphrase Maps
# ============================================================

INTENT_KEYWORDS = {
    "cause": ["cause", "reason", "due to", "result of", "led to"],
    "fix": ["fix", "fixed", "resolved", "mitigated", "rollback", "addressed"],
    "impact": ["impact", "effect", "affected", "resulted in"],
    "when": ["when", "time", "duration", "occurred"],
}

GENERIC_PATTERNS = [
    r"engineers are investigating",
    r"details are unclear",
    r"multiple components",
    r"various factors",
    r"still under review",
    r"no definitive conclusion"
]


# ============================================================
# Document Extraction (Safe)
# ============================================================

def extract_text_from_file(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    raw = uploaded_file.read()

    if name.endswith((".txt", ".md", ".csv")):
        return raw.decode("utf-8", errors="ignore")

    if name.endswith(".json"):
        try:
            obj = json.loads(raw.decode("utf-8", errors="ignore"))
            return json.dumps(obj, indent=2)
        except Exception:
            return raw.decode("utf-8", errors="ignore")

    if name.endswith(".html"):
        return re.sub(r"<[^>]+>", " ", raw.decode("utf-8", errors="ignore"))

    if name.endswith(".docx"):
        try:
            with zipfile.ZipFile(uploaded_file) as z:
                xml = z.read("word/document.xml").decode("utf-8", errors="ignore")
                return re.sub(r"<[^>]+>", " ", xml)
        except Exception:
            return ""

    if name.endswith(".pdf"):
        text = raw.decode("latin1", errors="ignore")
        return re.sub(r"[^A-Za-z0-9.,;:?!()\n ]+", " ", text)

    return raw.decode("utf-8", errors="ignore")


# ============================================================
# Text Utilities
# ============================================================

def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\n", " ")).strip()


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + chunk_size]))
        i += max(1, chunk_size - overlap)
    return chunks


def split_sentences(text: str) -> List[str]:
    s = re.split(r'(?<=[.!?])\s+', text)
    return [x.strip() for x in s if len(x.strip()) > 20]


# ============================================================
# Intent Detection
# ============================================================

def detect_intent(query: str) -> str:
    q = query.lower()
    for intent, keys in INTENT_KEYWORDS.items():
        if any(k in q for k in keys):
            return intent
    return "general"


# ============================================================
# Core Index
# ============================================================

class CaseIndex:
    def __init__(self):
        self.vectorizer = None
        self.matrix = None
        self.chunks: List[str] = []

    def build(self, chunks: List[str]) -> None:
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english"
        )
        self.matrix = self.vectorizer.fit_transform(chunks)

    # --------------------------------------------------------

    def _rank_sentences(
        self,
        sentences: List[str],
        query: str,
        intent: str,
        top_n: int
    ) -> List[Dict]:

        vec = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english"
        )
        sent_matrix = vec.fit_transform(sentences)
        q_vec = vec.transform([query])

        scores = cosine_similarity(q_vec, sent_matrix)[0]
        ranked = np.argsort(scores)[::-1]

        results = []
        for idx in ranked:
            score = float(scores[idx])
            if score <= 0:
                continue

            sentence = sentences[idx]
            s_lower = sentence.lower()

            # Intent boost
            if intent in INTENT_KEYWORDS:
                if any(k in s_lower for k in INTENT_KEYWORDS[intent]):
                    score *= 1.4
                else:
                    score *= 0.7

            # Generic penalty
            if any(re.search(p, s_lower) for p in GENERIC_PATTERNS):
                score *= 0.3

            results.append({
                "sentence": sentence,
                "sentence_score": score
            })

            if len(results) >= top_n:
                break

        return results

    # --------------------------------------------------------

    def query(
        self,
        query_text: str,
        top_k_chunks: int = 3,
        sentences_per_chunk: int = 2,
        min_chunk_score: float = 0.08
    ) -> List[Dict]:

        intent = detect_intent(query_text)

        q_vec = self.vectorizer.transform([query_text])
        chunk_scores = cosine_similarity(q_vec, self.matrix)[0]
        ranked_chunks = np.argsort(chunk_scores)[::-1]

        answers = []

        for idx in ranked_chunks[:top_k_chunks]:
            chunk_score = float(chunk_scores[idx])
            if chunk_score < min_chunk_score:
                continue

            chunk = self.chunks[idx]
            sentences = split_sentences(chunk)

            ranked_sentences = self._rank_sentences(
                sentences,
                query_text,
                intent,
                sentences_per_chunk
            )

            for s in ranked_sentences:
                answers.append({
                    "answer": s["sentence"],
                    "sentence_score": s["sentence_score"],
                    "chunk_score": chunk_score,
                    "source_chunk": chunk
                })

        if not answers:
            return []

        # ====================================================
        # AGILE NARROWING (CRITICAL)
        # ====================================================

        # 1️⃣ Sort by sentence dominance
        answers.sort(key=lambda x: x["sentence_score"], reverse=True)

        # 2️⃣ If top answer is clearly dominant → return only it
        if len(answers) > 1:
            if answers[0]["sentence_score"] > 1.5 * answers[1]["sentence_score"]:
                return [answers[0]]

        # 3️⃣ Otherwise keep only strong contenders
        max_score = answers[0]["sentence_score"]
        answers = [
            a for a in answers
            if a["sentence_score"] >= 0.7 * max_score
        ]

        # 4️⃣ Final cap (avoid verbosity)
        return answers[:2]


# ============================================================
# Builder
# ============================================================

def build_case_index_from_text(
    text: str,
    chunk_size: int,
    overlap: int
) -> CaseIndex:
    cleaned = clean_text(text)
    chunks = chunk_text(cleaned, chunk_size, overlap)
    index = CaseIndex()
    index.build(chunks)
    return index
