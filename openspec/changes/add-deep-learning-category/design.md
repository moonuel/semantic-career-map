## Context

The 12-category ML/AI responsibility taxonomy (Experiment 006) classifies job postings by function. `classical-ml` covers traditional approaches — XGBoost, scikit-learn, random forests, linear models. It does NOT cover deep learning approaches — CNNs, RNNs, transformers, GANs, VAEs, or general PyTorch/TensorFlow/JAX model training. The existing specialized categories (`computer-vision`, `reinforcement-learning`, `llm-fine-tuning`) cover specific sub-fields of deep learning but leave a gap for the general neural network engineer who designs architectures, writes training loops, and tunes hyperparameters for tasks that aren't vision, RL, or LLM-specific.

10 of the 27 postings contain explicit deep learning signals (PyTorch, TensorFlow, JAX, "deep learning", "neural network"). The Scribd golden subset posting is the canonical example — it lists "deep learning" as a required field and "custom Neural Networks in PyTorch" as a core activity, yet has no taxonomy category that cleanly fits.

This change adds the 13th category and re-runs the Experiment 006 pipeline.

## Goals / Non-Goals

**Goals:**
- Add a 13th taxonomy category: `deep-learning — Neural Network Engineering`, defined as designing and training neural network architectures (CNNs, RNNs, transformers, GANs, VAEs) using frameworks like PyTorch, TensorFlow, or JAX
- Add `"deep-learning": "Deep Learning Engineer"` to the controlled title vocabulary
- Update golden subset labels, particularly Scribd (`[llm-fine-tuning, llm-information-retrieval]` → `[deep-learning, llm-information-retrieval]`)
- Re-run label generation on all 27 postings (cache auto-invalidates via prompt hash)
- Re-generate UMAP visualization and co-occurrence analysis

**Non-Goals:**
- Re-defining or merging other taxonomy categories
- Changing the LLM model, temperature, or embedding pipeline
- Expanding the golden set beyond 5 postings
- Adding interactive UMAP toggles (deferred to follow-up experiment)

## Decisions

### D1: Category placement — 13th peer category (not a sub-category of any existing)

**Decision:** Add `deep-learning` as a standalone 13th category, numbered after `ai-safety-governance` (category 12).

**Rationale:** It is a first-class function, not a specialization of any existing category. `classical-ml` and `deep-learning` are siblings — both are "predictive modeling," one via traditional approaches, one via neural networks. Making it a sub-category of `classical-ml` would conflate two very different skill sets. Making it a parent of `computer-vision`, `reinforcement-learning`, and `llm-fine-tuning` would create an unnecessary hierarchy for a flat taxonomy.

**Alternatives considered:**
- Merge into `classical-ml` as "ML modeling" — rejected because the distinction between tree-based and neural-network-based approaches is meaningful for clustering and role classification
- Create as parent of `computer-vision`/`reinforcement-learning`/`llm-fine-tuning` — rejected because the taxonomy is intentionally flat and adding hierarchy adds complexity without clear benefit

### D2: Category definition scope — general NN engineering, not modality-specific

**Decision:** Define `deep-learning` as general neural network design and training, independent of modality. Cover model architecture design, training loop engineering, hyperparameter tuning, GPU optimization, transfer learning.

**Rationale:** This fills the gap for roles that build deep models for tasks like content classification, recommendation, ranking, or generation that aren't specifically LLM, vision, or RL. Keeping it modality-agnostic prevents the category from becoming another specialization that doesn't fit general DL roles.

**Boundaries:**
- `deep-learning` vs `llm-fine-tuning`: LLM fine-tuning implies adapting a pre-existing LLM (LoRA, DPO). `deep-learning` covers building/training architectures that are NOT pre-existing LLMs.
- `deep-learning` vs `computer-vision`: CV is a specialization of DL, applying DL techniques specifically to image/video tasks. `deep-learning` is the underlying capability; `computer-vision` is the application domain. A posting that trains PyTorch models for text classification gets `deep-learning`, not `computer-vision`.
- `deep-learning` vs `classical-ml`: Neural networks vs. trees/linear/statistical approaches. A posting that uses both (like Scribd) can receive both tags.

### D3: Golden subset update — Scribd only

**Decision:** Update only the Scribd golden label from `[llm-fine-tuning, llm-information-retrieval]` to `[deep-learning, llm-information-retrieval]`.

**Rationale:** The Scribd posting explicitly lists "deep learning" as a required field and "custom Neural Networks in PyTorch" as a core activity. LLM fine-tuning is implicit but not explicit in the posting text. The other 4 golden postings do not have deep learning signals strong enough to justify changing their labels. Bmo mentions "deep learning models" but the posting's primary signal is classical ML + analytics.

### D4: Prompt hash invalidation — automatic via definition change

**Decision:** No manual cache deletion required. Adding a new category to `TAXONOMY_DEFINITIONS` changes the prompt text, which changes the SHA-256 prompt hash stored in the cache metadata, which triggers automatic cache invalidation in `generate_labels.py`.

**Rationale:** This is the existing invalidation mechanism from Experiment 006 (prompt hash mismatch detection). Consistent with Exp 003/005 caching patterns. No manual intervention needed.

## Risks / Trade-offs

- **LLM miscategorization risk:** Adding a 13th category increases the label space. The LLM may struggle to distinguish `deep-learning` from `classical-ml` for postings that mention both (like Scribd). Mitigation: the golden recall metric provides a calibration check; the UMAP plot provides qualitative feedback.
- **Co-occurrence inflation:** `deep-learning` will likely co-occur with `mlops-production` (as most categories do) and potential with `classical-ml` for hybrid postings. Mitigation: co-occurrence analysis will surface this pattern; it's not a problem, just a data characteristic to document.
- **Cache invalidation cost:** All 27 LLM calls will be re-executed (~$0.008). Mitigation: cost is negligible.
