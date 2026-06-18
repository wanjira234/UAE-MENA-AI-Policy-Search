"""
test_search_engine.py

Lightweight tests for the retrieval pipeline. Run with:
    python -m pytest tests/ -v
or, without pytest installed:
    python tests/test_search_engine.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from search_engine import PolicySearchEngine, tokenize, load_corpus  # noqa: E402

CORPUS_DIR = Path(__file__).parent.parent / "corpus"


def test_tokenize_strips_stopwords_and_lowercases():
    tokens = tokenize("The UAE's National Strategy for AI was Announced in 2017.")
    assert "the" not in tokens
    assert "for" not in tokens
    assert "uae" in tokens
    assert "strategy" in tokens
    assert "2017" in tokens
    # confirm lowercasing happened
    assert "National".lower() == "national"
    assert "National" not in tokens


def test_load_corpus_finds_all_documents():
    chunks = load_corpus(CORPUS_DIR)
    doc_ids = {c.doc_id for c in chunks}
    assert len(doc_ids) == 6, f"expected 6 source documents, found {len(doc_ids)}: {doc_ids}"
    assert len(chunks) > 6, "expected multiple chunks per document"


def test_each_chunk_has_nonempty_text_and_section():
    chunks = load_corpus(CORPUS_DIR)
    for c in chunks:
        assert c.text.strip(), f"empty chunk text in {c.doc_id}"
        assert c.section.strip(), f"empty section label in {c.doc_id}"


def test_search_returns_requested_number_of_results():
    engine = PolicySearchEngine(corpus_dir=CORPUS_DIR)
    engine.build_index(force_rebuild=True)
    results = engine.search("data protection", top_k=3)
    assert len(results) == 3


def test_search_ranks_relevant_document_first():
    """A query naming the PDPL by name should surface the PDPL document first."""
    engine = PolicySearchEngine(corpus_dir=CORPUS_DIR)
    engine.build_index(force_rebuild=True)
    results = engine.search("Federal Decree-Law 45 2021 personal data protection", top_k=1)
    assert "PDPL" in results[0]["doc_title"] or "Personal Data Protection" in results[0]["doc_title"]


def test_search_scores_are_sorted_descending():
    engine = PolicySearchEngine(corpus_dir=CORPUS_DIR)
    engine.build_index(force_rebuild=True)
    results = engine.search("artificial intelligence strategy", top_k=5)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_unknown_query_terms_do_not_crash():
    """Words that don't appear anywhere in the corpus should not break search,
    they should just contribute zero signal."""
    engine = PolicySearchEngine(corpus_dir=CORPUS_DIR)
    engine.build_index(force_rebuild=True)
    results = engine.search("zzqxw nonsense gibberish term", top_k=3)
    assert len(results) == 3  # still returns results, just low-confidence ones


if __name__ == "__main__":
    # Allow running without pytest installed
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
