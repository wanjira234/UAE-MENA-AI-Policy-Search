# UAE / MENA AI Policy Search

A small, fully-transparent semantic search tool over a corpus of UAE and
Saudi Arabia AI policy documents. Built as a fast, scoped project to
practice the retrieval fundamentals that sit underneath the larger
retrieval-augmented-generation system in my main portfolio, without the
overhead of a deployed app or a managed vector database.

## What it does

Given a natural-language query, the tool returns the most relevant
passages from a corpus of six original summary documents covering:

- The UAE National Strategy for Artificial Intelligence 2031
- The UAE Personal Data Protection Law (Federal Decree-Law No. 45 of 2021)
- UAE AI governance institutions and the National AI Security Policy
- Saudi Arabia's National Strategy for Data and AI (NSDAI) and SDAIA
- A comparative look at UAE vs. Saudi AI policy approaches
- AI deployment specifically in Gulf healthcare systems

Each result shows the source document, the matched section, the passage
text, and a similarity score, ranked highest first.

```
$ python src/cli.py "what does the UAE data protection law say about automated decisions"

1. UAE Personal Data Protection Law (Federal Decree-Law No. 45 of 2021)  ·  Rights for individuals
   Data subjects gain a defined set of rights under the law, including the
   right to be informed about how their data is processed and a right
   relevant specifically to automated decision-making...
   score 0.307  ██████
```

## Why TF-IDF instead of neural embeddings

This is a deliberate choice, not a shortcut. The corpus is six documents —
small enough that a sentence-transformer model would technically work, but
would also turn the core mechanic into a black box behind a 400MB model
download. TF-IDF (term frequency, scaled by how rare a term is across the
corpus) is something I can fully derive and explain term by term, has zero
dependencies beyond NumPy, builds an index in under a second, and — because
policy and legal text leans heavily on specific named terms ("PDPL",
"NSDAI", "Federal Decree-Law No. 45") — retrieves about as well here as a
small embedding model would for a corpus this size.

The trade-off: TF-IDF matches on shared vocabulary, not meaning, so a query
using entirely different words than the source text (a true paraphrase)
will score lower than it would with neural embeddings. For a corpus this
size and this keyword-dense, that trade-off is worth it. For the larger RAG
project in my main portfolio, where the corpus is bigger and paraphrase
robustness matters more, that one uses custom transformer embeddings with
MMR/RRF reranking instead — this project is intentionally the simpler
sibling, not a replacement for that approach.

## How it works

1. **Chunking** (`load_corpus`): each `.txt` file in `corpus/` is split into
   sections using a blank-line convention — first line is the document
   title, then each section is a heading followed by body text.
2. **Vectorizing** (`build_index`): every chunk is tokenized (lowercased,
   stopwords removed), converted to a normalized term-frequency vector,
   then scaled by inverse document frequency (rarer terms get more weight).
   Each resulting vector is L2-normalized.
3. **Querying** (`search`): the query is tokenized and vectorized the same
   way, then compared against every chunk vector via cosine similarity —
   which, since everything is L2-normalized, is just a dot product.
4. **Caching**: the index is saved to `tfidf_cache.json` so repeat runs
   don't rebuild from scratch. Pass `--rebuild` to force a fresh build
   after editing the corpus.

## Setup

```bash
pip install -r requirements.txt
```

No model download, no API key, no internet connection needed after install.

## Usage

```bash
# Single query
python src/cli.py "how does Saudi Arabia regulate high-risk AI systems"

# Control how many results come back
python src/cli.py "AI talent development" --top-k 3

# Interactive mode — just run with no query
python src/cli.py

# Force the index to rebuild (e.g. after editing files in corpus/)
python src/cli.py --rebuild
```

## Running the tests

```bash
python -m pytest tests/ -v
# or, if pytest isn't installed:
python tests/test_search_engine.py
```

Tests cover tokenization, corpus loading, result ranking order, and that a
query naming a specific law correctly surfaces that law's document first.

## A note on sourcing

The corpus documents are original summaries I wrote based on public
reporting and official government sources (UAE's u.ae portal, the OECD.AI
policy observatory, SDAIA's public materials, and several legal/compliance
analyses), not reproductions of any single source text. This was a
deliberate choice so the corpus itself is freely shareable in this repo.

## Project structure

```
policy-search/
├── corpus/                       # six .txt policy summary documents
├── src/
│   ├── search_engine.py          # chunking, TF-IDF, cosine similarity search
│   └── cli.py                    # command-line interface
├── tests/
│   └── test_search_engine.py
├── requirements.txt
└── README.md
```

## Possible extensions

- Swap in a sentence-transformer embedding backend for paraphrase-robust
  matching, and compare retrieval quality against this TF-IDF baseline
  head to head.
- Add BM25 as a middle ground between TF-IDF and neural embeddings.
- Expand the corpus to more GCC countries (Qatar, Bahrain) for broader
  comparative queries.
