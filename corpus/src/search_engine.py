"""
search_engine.py

Core retrieval logic for the policy search tool.

Approach: chunk each document into sections, vectorize every chunk with
TF-IDF, and retrieve by cosine similarity between the query vector and chunk
vectors. This is intentionally simple -- no vector database, no neural
embeddings -- because the corpus is small (a handful of documents) and the
point of this project is to show a clean, fully-understood retrieval
pipeline end to end rather than to lean on a black-box pretrained model.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

CACHE_FILE = "tfidf_cache.json"

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# A short stopword list -- common English function words that carry little
# topical signal and would otherwise dominate term-frequency counts.
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are",
    "was", "were", "be", "been", "being", "this", "that", "these", "those",
    "it", "its", "as", "by", "with", "at", "from", "into", "over", "under",
    "than", "then", "so", "such", "but", "not", "no", "nor", "if", "while",
    "their", "they", "which", "who", "whom", "what", "when", "where", "how",
    "also", "can", "could", "would", "should", "will", "shall", "may",
    "might", "must", "has", "have", "had", "does", "do", "did", "about",
    "between", "through", "across", "within", "per", "each", "any", "all",
    "both", "more", "most", "some", "other", "out", "up", "down",
}


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1]


@dataclass
class Chunk:
    doc_id: str
    doc_title: str
    section: str
    text: str
    chunk_index: int


def load_corpus(corpus_dir: Path) -> list[Chunk]:
    """Read every .txt file in corpus_dir and split it into section-level chunks.

    Each document is expected to use a simple convention: a title on the
    first line, then sections separated by a line of dashes underneath a
    section heading. This keeps chunking predictable without needing a
    general-purpose document parser.
    """
    chunks: list[Chunk] = []

    for path in sorted(corpus_dir.glob("*.txt")):
        raw = path.read_text(encoding="utf-8")
        lines = raw.strip().split("\n")
        doc_title = lines[0].strip()

        # Split on blank-line-separated blocks; each block after the title
        # is treated as one section (heading line + body).
        body = "\n".join(lines[1:])
        blocks = re.split(r"\n\s*\n", body.strip())

        chunk_index = 0
        for block in blocks:
            block_lines = [l for l in block.split("\n") if not re.match(r"^-+$", l.strip())]
            block_text = "\n".join(block_lines).strip()
            if not block_text:
                continue

            # First non-empty line of the block is treated as the section name;
            # strip it from the displayed body so it isn't shown twice.
            section_lines = block_text.split("\n")
            section = section_lines[0].strip()
            text = "\n".join(section_lines[1:]).strip()

            if len(text) < 20:
                continue

            chunks.append(
                Chunk(
                    doc_id=path.stem,
                    doc_title=doc_title,
                    section=section,
                    text=text,
                    chunk_index=chunk_index,
                )
            )
            chunk_index += 1

    return chunks


class PolicySearchEngine:
    """TF-IDF retrieval over the local corpus.

    Why TF-IDF instead of neural embeddings: this corpus is six short
    documents. A pretrained sentence embedding model would work, but it
    would also hide the actual retrieval mechanics behind a model download
    and a black box. TF-IDF is fully transparent, has zero external
    dependencies beyond numpy, and -- for a corpus this size and this
    keyword-dense (legal/policy text leans heavily on specific named terms
    like "PDPL" or "NSDAI") -- retrieves about as well as a small embedding
    model would, while being something I can fully explain term by term.
    """

    def __init__(self, corpus_dir: str | Path, cache_dir: str | Path | None = None):
        self.corpus_dir = Path(corpus_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else self.corpus_dir
        self.chunks: list[Chunk] = []
        self.vocab: dict[str, int] = {}          # term -> column index
        self.idf: np.ndarray | None = None        # inverse document frequency per term
        self.tfidf_matrix: np.ndarray | None = None  # (n_chunks, n_terms), L2-normalized rows

    # ---- index construction -------------------------------------------------

    def build_index(self, force_rebuild: bool = False) -> None:
        cache_path = self.cache_dir / CACHE_FILE

        if not force_rebuild and cache_path.exists():
            self._load_cache(cache_path)
            return

        self.chunks = load_corpus(self.corpus_dir)
        if not self.chunks:
            raise ValueError(f"No documents found in {self.corpus_dir}")

        token_lists = [tokenize(c.text) for c in self.chunks]

        # Build vocabulary
        vocab: dict[str, int] = {}
        for tokens in token_lists:
            for t in set(tokens):
                if t not in vocab:
                    vocab[t] = len(vocab)
        self.vocab = vocab

        n_docs = len(self.chunks)
        n_terms = len(vocab)

        # Term frequency matrix
        tf = np.zeros((n_docs, n_terms), dtype=np.float64)
        for row, tokens in enumerate(token_lists):
            counts = Counter(tokens)
            total = sum(counts.values()) or 1
            for term, count in counts.items():
                tf[row, vocab[term]] = count / total  # normalized term frequency

        # Inverse document frequency: log(N / (1 + doc_freq)), smoothed
        doc_freq = (tf > 0).sum(axis=0)
        idf = np.log((1 + n_docs) / (1 + doc_freq)) + 1
        self.idf = idf

        tfidf = tf * idf
        # L2-normalize each row so cosine similarity is a plain dot product
        norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self.tfidf_matrix = tfidf / norms

        self._save_cache(cache_path)

    def _vectorize_query(self, query: str) -> np.ndarray:
        tokens = tokenize(query)
        vec = np.zeros(len(self.vocab), dtype=np.float64)
        counts = Counter(tokens)
        total = sum(counts.values()) or 1
        for term, count in counts.items():
            if term in self.vocab:
                vec[self.vocab[term]] = (count / total) * self.idf[self.vocab[term]]
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    # ---- persistence ----------------------------------------------------------

    def _save_cache(self, cache_path: Path) -> None:
        data = {
            "chunks": [asdict(c) for c in self.chunks],
            "vocab": self.vocab,
            "idf": self.idf.tolist(),
            "tfidf_matrix": self.tfidf_matrix.tolist(),
        }
        cache_path.write_text(json.dumps(data))

    def _load_cache(self, cache_path: Path) -> None:
        data = json.loads(cache_path.read_text())
        self.chunks = [Chunk(**c) for c in data["chunks"]]
        self.vocab = data["vocab"]
        self.idf = np.array(data["idf"])
        self.tfidf_matrix = np.array(data["tfidf_matrix"])

    # ---- search ----------------------------------------------------------------

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if self.tfidf_matrix is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        query_vec = self._vectorize_query(query)
        scores = self.tfidf_matrix @ query_vec  # cosine similarity (rows are L2-normalized)

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            results.append(
                {
                    "score": float(scores[idx]),
                    "doc_title": chunk.doc_title,
                    "doc_id": chunk.doc_id,
                    "section": chunk.section,
                    "text": chunk.text,
                }
            )
        return results
