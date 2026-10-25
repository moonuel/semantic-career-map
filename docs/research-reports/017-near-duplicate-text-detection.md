# Near-Duplicate Text Detection — Containment (Trimmed Supersets) and Near-Exact Reposts

> Date: 2026-10-16
> Context: Semantic Career Mapping Platform — research into detecting significant text duplication between job postings, in two specific shapes: (A) one posting is a superset (or close to it) of another because it was trimmed or trimmed-from, and (B) two postings are almost identical except for a few tokens (a new posting date, a changed deadline, a couple of swapped words). Feeds the ingestion dedup layer (`backend/ingestion/`, report 013) and the pre-LLM-cleaning gate (experiment 003 cache in `data/.llm_clean_cache_*`).

---

## Table of Contents

1. [Problem Statement: Two Duplication Modes (and Why They Need Different Tools)](#1-problem-statement-two-duplication-modes-and-why-they-need-different-tools)
2. [The Core Similarity Measures](#2-the-core-similarity-measures)
3. [Containment Detection (Mode A)](#3-containment-detection-mode-a)
4. [Near-Exact Detection (Mode B)](#4-near-exact-detection-mode-b)
5. [Tool and Library Survey](#5-tool-and-library-survey)
6. [Scale Math: Pairwise vs LSH](#6-scale-math-pairwise-vs-lsh)
7. [Normalization and the Volatile-Field Mask](#7-normalization-and-the-volatile-field-mask)
8. [Recommended Pipeline for This Project](#8-recommended-pipeline-for-this-project)
9. [Evaluation Notes](#9-evaluation-notes)
10. [Sources](#10-sources)

---

## 1. Problem Statement: Two Duplication Modes (and Why They Need Different Tools)

Jobposting duplication has a specific texture. Reposts are rarely byte-identical, and they are rarely paraphrases either. Two dominant shapes:

| Mode | Shape | Example behavior |
|---|---|---|
| **A — Containment / trim** | `B ⊆ A` (or nearly): one text is all of the other plus extra, or a cut of it | "Full JD with an intro and benefits section" vs. the same JD copied into a short teaser listing that keeps only the requirements bullet list; a posting trimmed to respect a crawl length |
| **B — Near-exact copy with edits** | Almost identical, a few tokens differ | Same JD, new posting date / deadline / requisition id |

Two separate diagnosis toolkits apply, and combined pipelines are standard practice:

- **Mode A is the containment / trim failure mode.** Jaccard similarity fails badly on it: if `B ⊂ A`, then Jaccard = `|A∩B| / |A∪B| ≈ |B|/|A|`, which collapses as B shrinks (§2.3). The correct measure is *containment* / *overlap*, and the scalable tools are substring detectors (suffix arrays) or MinHash **LSH Ensemble**.
- **Mode B defeats exact-hash dedup** (the edit) **and exact substring matching** (a changed word breaks every shingle spanning it). The correct tool is an *edit-similarity or near-duplicate* measure: RapidFuzz ratios, SimHash, MinHash Jaccard on shingles with a high threshold, or char n-gram cosine.

---

## 2. The Core Similarity Measures

### 2.1 Edit-similarity family (RapidFuzz / difflib / python-Levenshtein)

The classical fuzzy-matching stack is Levenshtein-family. In Python, **RapidFuzz** (C++ core, MIT, wheels) has superseded `python-Levenshtein` (now a wrapper over RapidFuzz anyway) and `fuzzywuzzy` (unmaintained; also RapidFuzz-backed):

- `fuzz.ratio(a, b)` — normalized indel-edit similarity (Levenshtein family), order-sensitive. Catches mode B (ratio ≈ 0.98 after adding a date at the head); drops toward ~0.4 on a hard trim.
- `fuzz.partial_ratio(a, b)` — best-scoring **substring** alignment of the shorter against the longer. This is exactly the "is one a (fuzzy) containment of the other" query for mode A: `partial_ratio(trimmed_teaser, full_jd) ≈ 1.0`.
- `fuzz.token_set_ratio / token_sort_ratio` — order-insensitive / multiset variants; robust to reordering but *loose*, so high false-positive risk on genuinely different postings that share boilerplate vocabulary.
- `rapidfuzz.process.cdist(list_a, list_b, workers=-1, score_cutoff=t)` — vectorized many-vs-many scoring, SSE-threaded; the workhorse for candidate verification (see §6).
- `difflib.SequenceMatcher.get_matching_blocks()` (stdlib) — slow, but produces the actual aligned blocks, which makes it the right tool for the *explain* step: "these two postings differ by 4 words: `2025-06-01` → `2025-07-01`".

Careful with `difflib` on long strings: `autojunk=True` (the default) discards "popular" characters on inputs > 200 chars and silently degrades quality — set it to `False`.

### 2.2 Shingle-set measures (Jaccard / containment / overlap coefficient)

Pick a unit — **word 5-grams** (the LLM-literature default since Lee et al. 2021), **character n-grams** (more robust to OCR/HTML noise; word shingles die on a re-tokenized HTML body), or sentences — turn text into a set, then:

- **Jaccard** `|A∩B|/|A∪B|` — symmetric, scale-friendly (MinHash-able), but *bad for mode A* whenever the sets differ in size by more than ~2×.
- **Containment** `C = |A∩B| / min(|A|,|B|)` — the symmetric form of the overlap coefficient; `C ≈ 1.0` whenever the smaller set is almost a subset of the larger. This is the quantity LSH Ensemble (§3.2) is built to bound, defined there as `|S∩Q| / |S|` (asymmetric, per-side).
- **Tversky index** `|A∩B| / (α|A\B| + β|B\A| + |A∩B|)` — the generalization containing Jaccard (α=β=1) and Dice (α=β=0.5); lets you penalize only one direction, e.g., "new posting may add boilerplate to itself but must not omit anything from the old one."

### 2.3 Worked example — why Jaccard fails on trims

- `A` = full JD → 1,000 word-5-gram shingles.
- `B` = teaser trimmed from `A`, keeping the requirements list intact → 100 shingles. Most of `B`'s shingles still appear inside `A`; say ~90 survive normalization and edge effects, plus ~10 span the cut boundaries and exist only in `B`. Then `Jaccard(A,B) = 90 / (1000 + 10) ≈ 0.09` — far below any 0.8 threshold — while `containment = 90 / 100 = 0.90` — far above it. The *same pair* is "unrelated" to Jaccard and "near-identical" to containment.

The practical takeaway: **a pair that shingle-Jaccard ranks as "unrelated" can still be an exact superset-subset pair.** Any dedup design that only thresholds Jaccard will leak trimmed reposts.

### 2.4 Dependency placement in the stack

Semantic embeddings (which this project already has — `all-MiniLM-L6-v2`) are the *wrong* first-line tool for this problem: paraphrases at cosine ≥ 0.9 exist that are genuinely different postings (two studios' identical seniority language), while verbatim reposts often embed in the 0.8s. Use embeddings as a *third* optional lens (dedup of near-paraphrase reporting), not as the near-exact detector. Their cost (a forward pass each) is also orders-of-magnitude above hash operations.

---

## 3. Containment Detection (Mode A)

### 3.1 Exact-substring dedup via suffix arrays (the LLM-training-literature tool)

Lee et al. (2021), *"Deduplicating Training Data Makes Language Models Better"*, define the strongest ground result for containment: build a **suffix array** over the token/byte stream of the whole (commons-normalized) corpus, find exact repeats ≥ some length (they use 50-token windows), and drop the shorter of any containment pair. This is the textbook answer to "is one text a superset of another, possibly at scale" — but it is exact, so the "new date" tokens break it into two long runs instead of one; it is purely mode A.

Python implementations:

- **`text-dedup`** (ChenghaoMou/text-dedup, MIT) — the one-stop dedup toolkit for LLM-style corpora: exact-hash dedup, MinHash (BigQuery-style), SimHash with hamming-distance banding, embedding dedup, and the exact-substring suffix-array pass described above. Runs from the CLI or as a library. If you adopt one tool for a heavyweight corpus pass, it is this one.
- **`pydivsufsort`** / **`pysais`** — fast suffix-array backends if you roll your own longest-common-substring pass.
- Below a few thousand short documents you likely never need this machinery: `partial_ratio` plus a length ratio (§3.3) covers mode A interactively.

The cross-cutting caveat: suffix-array containment finds *any one long identical run*. A repost that reorders two paragraphs breaks every inside-block 5-gram boundary but the *majority* of the text is still identical — a per-block or shingle-containment measure (below) is what catches reordering; the array approach deliberately doesn't.

### 3.2 MinHash LSH Ensemble (containment at scale, datasketch)

**datasketch** (ekzhu/datasketch, pure-Python + numpy, Apache-2) implements **MinHash LSH Ensemble** (Zhu et al., PVLDB 2016), which indexes shingle sets to answer *"find sets S whose containment of query Q is ≥ t"* in sublinear time:

```python
from datasketch import MinHash, MinHashLSHEnsemble, MinHash

def shingles(text, k=5):
    words = text.split()
    return {" ".join(words[i:i+k]) for i in range(len(words)-k+1)}

lsh = MinHashLSHEnsemble(threshold=0.8, num_perm=128, m=16)
lsh.index([m1, m2, ...], [doc_id1, doc_id2, ...])
for s, did in lsh.query(query_set):
    ...
```

- `MinHash(num_perm=128)` gives a few-percent error at 128 permutations on Jaccard (sufficient above thresholds ≥ 0.8).
- The same `MinHash` objects give Jaccard estimates directly (`m1.jaccard(m2)`), so a single signature serves both mode B (high-threshold Jaccard) and mode A (via LSH Ensemble) — a real implementation convenience.
- Weighted variants (`WeightedMinHash`) and `MinHashLSH` (Jaccard-only, but supports *streaming insertion* — good for incremental ingestion, §6) round out the package.

### 3.3 When do you need a *fuzzy* containment measure at all?

Exact-substring and LSH Ensemble both work on the *normalized* stream. If normalization follows §7 (case/punctuation/whitespace folding plus volatile-field masking), the residual sources of "almost-contained but not exactly" are human cutting, OCR/HTML slop, and boilerplate differences — which shingle-containment (MinHash) catches statistically. **A decision guide:**

| Signal wanted | Tool |
|---|---|
| Exact substring presence, ground truth | suffix array (pydivsufsort / text-dedup) |
| Probabilistic containment of shingle sets, scale | datasketch MinHashLSHEnsemble |
| Two-text interactive check, n ≤ ~10⁴ | `fuzz.partial_ratio` |
| Explain *what* changed | `difflib` / `rapidfuzz` alignment blocks |

---

## 4. Near-Exact Detection (Mode B)

### 4.1 RapidFuzz ratios with a cutoff (the default small-scale answer)

Simplest correct answer for pairs: compute `fuzz.ratio` (after normalization) and flag pairs ≥ ~0.92; add `partial_ratio` ≥ 0.98 as the containment guard (§3.3). The `token_set_ratio` variant fights reordering, but it is weakly discriminative on job boilerplate (every posting shares "with benefits", "medical, dental, vision") — prefer ratio/partial_ratio on normalized text and reserve token-set scores for the human-review queue as a *finding* rather than a decision rule.

### 4.2 SimHash (web-crawling classic)

SimHash (Manku, Jain, Das Sarma, WWW 2007 — the Google web-dedup paper) compresses a document into a 64-bit fingerprint so that **near-duplicates land within Hamming distance 3** (after char-fingerprinting). It is the *fastest* in-memory discriminator at multi-million corpus scale and is what a crawler under you gives you for free; at our data rate it is analysis overkill relative to simply keeping MinHash signatures (128 × 32-bit words) or not needing an approximation at all. Python: `simhash` (PyPI, leonsim/simhash) or `simhashlib` / `text-dedup`'s implementation. Use it if/when the corpus passes ~10⁶ docs of *near-exact* mode B.

### 4.3 MinHash Jaccard with a high threshold

The same shingle set machinery from §3.2 answers mode B if you set the LSH Jaccard threshold high (0.85–0.92 with word 5-grams). Two postings differing by "only an updated date and a couple of words" end up at Jaccard ≈ 0.8–0.95 on word 5-grams — the *exact* band the LSH threshold selects. This is the reference implementation pattern: **one MinHash side index, two queries** (LSH Ensemble for containment, MinHashLSH for Jaccard).

### 4.4 Char n-gram TF-IDF cosine as a vectorized first pass (scikit-learn)

`TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5))` + `linear_kernel` (or `cosine_similarity` on L2-normalized rows) is a dense, vectorized near-duplicate funnel that requires **zero new dependencies** (sklearn is already in the project):

- order-preserving but edit-tolerant; near-exact reposts land ≥ 0.9 cosine comfortably.
- Char n-grams survive the one-word-change mode better than word 5-grams (a changed token destroys 5 word-shingles but only ~k char-shingles at its span).

This is the cheapest *candidate generation* layer (see §6), followed by a RapidFuzz verification pass on the surviving pairs.

---

## 5. Tool and Library Survey

| Tool | What it detects / does | Mode | Type | Notes |
|---|---|---|---|---|
| **RapidFuzz** (`rapidfuzz`) | Levenshtein/Jaro ratios, `partial_ratio`, `process.cdist`, alignment | A (via partial) + B | Pairwise, C++ | MIT, wheels; supersedes `python-Levenshtein`, `fuzzywuzzy` |
| **difflib** (stdlib) | `SequenceMatcher` ratios and matching blocks | both (verification/explain) | Pairwise, pure Python | Slow; `autojunk=False` for long text |
| **datasketch** | MinHash/LSH Jaccard, **MinHashLSHEnsemble** containment, HyperLogLog | A + B | Probabilistic, scale | numpy; streaming insert in `MinHashLSH` |
| **text-dedup** | Exact, MinHash, SimHash, sentence-embedding and suffix-array substring dedup | A + B | Pipeline toolkit, CLI or lib | MIT; LLM-corpus proven; best one-stop option |
| **simhash** (`simhash`) | 64-bit SimHash fingerprints, Hamming distance | B | Probabilistic, web-scale classic | Only worth it above ~10⁶ docs |
| **pydivsufsort** / **pysais** | Fast suffix arrays | A exact | Building block | For DIY exact-substring passes |
| **scikit-learn** | char/word n-gram TF-IDF + cosine cdist | A + B first pass | Vectorized | Already a project dependency |
| **TLSH / Nilsimsa** | locality-sensitive hashes (byte/local level) | B | Esoteric | Byte-order anchored; ≥ 256-byte minimum (TLSH) disqualifies short postings; niche |
| **sentence-transformers** (already in stack) | paraphrase-level similarity | *near*-paraphrase only | Adjacent lens | Wrong first line for byte-dup detection (§2.4) |

Dependency note: RapidFuzz and datasketch are the two plausible *additions*. Per repository rules ("do not install packages — the user manages dependencies manually") nothing is installed as part of this research; §8 gives a zero-new-dependency first pass (sklearn + stdlib) with these two as opt-in upgrades.

---

## 6. Scale Math: Pairwise vs LSH

All-vs-all pairwise scoring is `O(n²)`, but the constant depends heavily on the scorer. Raw `process.cdist` on edit distances is only comfortable to a few thousand documents (~10⁹ char-ops at 30k×30k). A *sparse* TF-IDF cosine matrix (`linear_kernel` on `TfidfVectorizer` output in batches) is far cheaper per pair and stays practical to roughly `n = 2–5 × 10⁴`. Beyond that — or for *incremental* ingestion, where each new batch only compares against the existing corpus (already `O(n)` per batch) — switch to locality-sensitive bucketing:

1. **MinHashLSH (Jaccard ≥ 0.85)** → catches mode B reposts, supports streaming inserts (`lsh.insert(m2, id2)`).
2. **MinHashLSHEnsemble (containment ≥ 0.8)** → catches mode A trims, batch-rebuilt per ingestion run; keep it in the same script loop (rebuild is cheap at corpus sizes ≤ 10⁵).
3. **Optional third lens (semantic):** embedding cosine ≥ ~0.95 *within the same employer* as a hand-review queue for paraphrase-level dupes — examined by a human, not automated (§2.4).

A concrete sizing note for the corpus being processed here: current scale is tens of postings (dataset "27" per the status docs), scraped per-ATS boards. At every realistic scraping rate for the MVP (≤ ~5k postings/week), the *pairwise char n-gram + RapidFuzz verify* pass on the day's delta against the whole corpus stays comfortably cheap; the LSH machinery becomes mandatory only if the corpus ever approaches 10⁵+ *and* the pipeline must run inside a batch window.

---

## 7. Normalization and the Volatile-Field Mask

Mode B's classic false-negative is a repost where the only change is an embedded volatile token — a posting date, deadline, requisition id, timestamp. If you hash raw text, this diff defeats an exact hash **and** after normalization still destroys a handful of char-shingles, dragging cosine down marginally. Solve it before comparing, not after:

1. **Unicode fold + case fold + whitespace collapse** (NFKC, lowercase, ` \s+ → " "`, strip trailing punctuation).
2. **Strip HTML/Markdown residue** — the corpus already has a dedicated cleaning step (LLM extraction, experiment 003) and its cache lives beside this data (`data/.llm_clean_cache_007/`); dedup on *cleaned* text, but hash-match the *raw* text first so a verbatim repost is caught (cheaply) before it ever reaches an LLM call. This is also where the second-order win lives: dedup *before* cleaning saves LLM spend, since a repost by definition shares a cleaned text.
3. **Mask volatile tokens** in the string fed to shingling/hashing:
   - ISO dates and common `YYYY-MM-DD`, `Mon DD` patterns → `<DATE>`
   - bare ints ≥ 3 digits near keywords like deadline/close/closes/apply-by → `<NUM>`
   - UUIDv7/requisition ids and bare long integer ids → `<ID>`
   - do **not** mask URLs wholesale (differing `?ref=` querystrings matter as *evidence*, not signal of a different posting — mask the querystring only).
4. Keep both views: **masked-and-normalized** for shingling/dedup, and the **raw** strings retained in the dedup *record* so the reviewer can see exactly what differs (difflib/RapidFuzz alignment to render "differs by: `2025-06-01` → `2025-07-01`, ` requisition #481` → `#489`").

Also apply the *field-level* exact key before any fuzzy work: post id from source, `(employer, title-normalized, location-normalized)` — report 013 documents the per-ATS id fields (Greenhouse job-post `id` vs `internal_job_id`; Lever UUID; Ashby UUID; `isListed` caveats). Exact keys catch the bulk of true reposts for free and shrink the fuzzy workload.

---

## 8. Recommended Pipeline for This Project

Layered, cheapest first, each layer shrinking the next:

1. **Layer 0 — exact keys:** source post id, content hash (raw), then `(employer, normalized title, normalized location)` — dedup before anything else (ties to report 013's per-source field table).
2. **Layer 1 — exact content dedup:** SHA-256 of the *normalized* text; kills dateless reposts and trailing-whitespace variants.
3. **Layer 2 — vectorized fuzzy pass (no new deps):** char `(3,5)`-gram TF-IDF (`TfidfVectorizer`, `analyzer="char_wb"`) over the day's delta + the existing corpus's cached matrix → `linear_kernel` top-K per new posting (K≈4, cosine ≥ 0.80 as the candidate gate); verify each candidate with `fuzz.ratio ≥ 0.90`, `partial_ratio ≥ 0.97`, and content-length ratio ≥ 0.9 (checked asymmetrically: shorter/longer).
4. **Layer 3 — containment check on flagged pairs:** compute both `containment = |S_new ∩ S_old| / |S_old|` and `|S_new \ S_old|` shingle-set deficits on word 5-grams; declare mode A when containment ≥ 0.8 and the *newer* text is the shorter one (trim) — else mode B; emit *decision*: keep the **most complete** variant (largest shingle set / longest normal form) as canonical, record the other as a duplicate with a `dup_of` provenance pointer, and keep the difflib "what differs" payload for audit.
5. **Layer 4 (opt-in later):** swap layer 2's linear scan for `datasketch` MinHashLSH + LSHEnsemble when n grows past ~10⁵, or adopt `text-dedup` for a full-corpus historical pass (its suffix-array mode is the strongest exact-superset detector available).

Placement in the lakehouse layout (report 009): raw copies land in bronze untouched; layers 0–4 run bronze→silver; silver holding the canonical text plus `dup_of`, `dup_mode` (A/B), and the diff payload lets audit-grade reports reach the LLM-cleaning layer, which only ever sees canonical postings — which is the point (the cache files in `data/.llm_clean_cache_*` key on content hash; fewer deduped inputs = fewer paid cleaning calls AND the same canonical text keeps hits across reposts).

Pseudocode shape (no code shipped in this research — implementation belongs to a later change proposal):

```
canon = exact_key_and_hash_dedupe(new_batch)          # layers 0–1
pairs  = tfidf_candidates(new_batch, corpus_matrix)   # layer 2 (cosine ≥ 0.80, top-K)
for pair in pairs:                                    # layer 3 verify
    if fuzzy_verify(pair, ratio≥0.90 or partial≥0.97): classify_and_record(pair)
corpus_matrix.update(new_batch)                       # incremental
```

---

## 9. Evaluation Notes

- Hand-label a small pairwise sample (the hand-maintained `data/golden_set.json` is the right home for a curated dup/non-dup pair-list; keep fixtures synthetic and < 10KB per the repo's test rules).
- Measure **pair-level precision and recall** per mode separately — a global F1 hides the fact that mode A recall and mode B recall degrade for *different* reasons (label each flagged pair A/B/other at capture time, §8 step 3).
- Sensitivity sweeps worth documenting when implemented: containment threshold 0.7/0.8/0.9 × cosine gate 0.75/0.80/0.85; report the threshold pair that keeps precision at 1.0 (false dedup of two genuinely different postings is the money-loss error, not the missed date-update repost).
- Synthetic unit fixtures to demanding values: (a) trimmed teaser ⊂ full JD (mode A, expected flag), (b) full JD copy with only a date and a requisition id changed (mode B), (c) two unrelated postings from the *same* employer sharing boilerplate (the precision trap that token-set ratios would flag — expected *non*-flag).

---

## 10. Sources

- Lee, Ippolito, Nystrom, Zhang — *Deduplicating Training Data Makes Language Models Better* (arXiv:2107.06499, 2021): suffix-array exact-substring dedup, 50-token windows, normalized token streams.
- Manku, Jain, Das Sarma — *Detecting Near-Duplicates for Web Crawling* (WWW 2007): SimHash fingerprinting methods, Hamming-distance bands, k=3 tolerance at web scale.
- Zhu et al. — *LSH Ensemble* (PVLDB 2016): containment similarity search via cardinality-partitioned MinHash indexes, implemented in `datasketch` (github.com/ekzhu/datasketch).
- `text-dedup` — github.com/ChenghaoMou/text-dedup (MIT): MinHash/SimHash/exact/suffix-array/substring/sentence-embedding dedup toolkit used across LLM corpus work.
- RapidFuzz docs — maxbachmann.github.io/RapidFuzz: `fuzz.*` ratios, `partial_ratio`, `process.cdist`, alignment, and the `python-Levenshtein` migration note.
- Python stdlib docs — `difflib.SequenceMatcher` (`autojunk` behavior).
- Project-local references: `research-reports/013-job-board-api-scrapers.md` (per-source dedup-unit ids and change semantics), `research-reports/009-data-lake-design-best-practices.md` (bronze/silver/gold placement), `experiments/003-llm-extraction.md` and the `data/.llm_clean_cache_007/` cleaning cache (dedup-before-cleaning rationale).
