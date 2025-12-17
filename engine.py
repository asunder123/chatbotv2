# engine.py
# Similarity-driven document QA engine
# - Character n-gram TF-IDF
# - Strict similarity dependency
# - Safe sentence stitching (context expansion)
# - Evidence preserved verbatim

import re
import numpy as np
from typing import List, Dict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# Text preprocessing
# ============================================================

def extract_core_text(text: str) -> str:
    """
    Very light cleanup: keep everything meaningful.
    """
    return " ".join(line.strip() for line in text.splitlines() if line.strip())


def split_sentences(text: str) -> List[str]:
    """
    Conservative sentence splitter.
    Keeps short but meaningful fragments.
    """
    parts = re.split(r'(?<=[.!?])\s+|\n+', text)
    return [p.strip() for p in parts if re.search(r"[A-Za-z]", p)]


# ============================================================
# Answer & evidence formatting
# ============================================================

def paraphrase_answer(sentence: str) -> str:
    """
    Light smoothing for readability.
    Does NOT add new information.
    """
    s = sentence.strip()

    rewrites = [
        (r"\bwas caused by\b", "happened because of"),
        (r"\bdue to\b", "because of"),
        (r"\bled to\b", "which resulted in"),
    ]

    for pat, repl in rewrites:
        s = re.sub(pat, repl, s, flags=re.IGNORECASE)

    return s[0].upper() + s[1:] if s else s


def format_evidence(sentences: List[str]) -> str:
    """
    Evidence is verbatim and clearly quoted.
    """
    return " ".join(f"“{s.strip()}”" for s in sentences)


# ============================================================
# Sentence stitching logic (SAFE)
# ============================================================

def stitch_sentences(
    sentences: List[str],
    scores: np.ndarray,
    anchor_idx: int,
    alpha: float = 0.5
) -> List[str]:
    """
    Expand context around the anchor sentence.

    Rules:
    - Anchor sentence always included
    - Only immediate neighbors (i-1, i+1)
    - Neighbor must have relative similarity >= alpha * anchor_score
    """
    stitched = []
    anchor_score = scores[anchor_idx]

    # Previous sentence
    if anchor_idx - 1 >= 0:
        if scores[anchor_idx - 1] >= alpha * anchor_score:
            stitched.append(sentences[anchor_idx - 1])

    # Anchor sentence
    stitched.append(sentences[anchor_idx])

    # Next sentence
    if anchor_idx + 1 < len(sentences):
        if scores[anchor_idx + 1] >= alpha * anchor_score:
            stitched.append(sentences[anchor_idx + 1])

    return stitched


# ============================================================
# Core Index
# ============================================================

class CaseIndex:
    """
    Document index using strict similarity-based retrieval.
    """

    def __init__(self):
        self.sentences: List[str] = []
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix = None

    # --------------------------------------------------------
    # Build index
    # --------------------------------------------------------

    def build_from_text(self, raw_text: str):
        core = extract_core_text(raw_text)
        self.sentences = split_sentences(core)

        if not self.sentences:
            self.matrix = None
            return

        # Character n-gram TF-IDF for robust lexical similarity
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1
        )

        self.matrix = self.vectorizer.fit_transform(self.sentences)

    # --------------------------------------------------------
    # Query
    # --------------------------------------------------------

    def query(self, query: str, alpha: float = 0.5) -> List[Dict]:
        """
        Execute a similarity-based query.

        Returns:
        - answer: paraphrased anchor sentence
        - evidence: stitched verbatim sentences
        - score: similarity score of anchor
        """

        if not self.sentences or self.matrix is None:
            return []

        q = query.strip().lower()
        if not q:
            return []

        # Vectorize query
        q_vec = self.vectorizer.transform([q])
        scores = cosine_similarity(q_vec, self.matrix)[0]

        # Anchor sentence
        anchor_idx = int(np.argmax(scores))
        anchor_score = float(scores[anchor_idx])

        # STRICT similarity requirement
        if anchor_score <= 0:
            return []

        # Stitch neighboring context
        stitched = stitch_sentences(
            self.sentences,
            scores,
            anchor_idx,
            alpha=alpha
        )

        return [{
            "answer": paraphrase_answer(self.sentences[anchor_idx]),
            "evidence": format_evidence(stitched),
            "score": anchor_score,
            "stitched_count": len(stitched)
        }]
