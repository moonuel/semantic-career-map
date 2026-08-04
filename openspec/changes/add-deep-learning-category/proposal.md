## Why

The 12-category ML/AI responsibility taxonomy from Experiment 006 has a gap: `classical-ml` covers traditional approaches (XGBoost, scikit-learn, trees) but there is no category for designing and training neural network architectures (CNNs, RNNs, transformers, GANs, VAEs) using frameworks like PyTorch, TensorFlow, or JAX. The existing specialized categories (computer-vision, reinforcement-learning, llm-fine-tuning) cover subsets of deep learning but leave a gap for general NN engineering — the role of someone who trains deep models for classification, prediction, or generation that isn't specifically LLM fine-tuning, vision, or RL. 10 of the 27 postings in the dataset contain explicit deep learning signals that currently have no proper taxonomic home.

## What Changes

- Add a 13th taxonomy category: `deep-learning — Neural Network Engineering` to the LLM labeling prompt in `generate_labels.py`
- Add `"deep-learning": "Deep Learning Engineer"` to the controlled title vocabulary
- Re-run taxonomy labeling on all 27 postings with the updated 13-category taxonomy
- Update the 5-posting golden subset labels where applicable (particularly Scribd, which has strong deep learning signal)
- Invalidate the label cache so prompt hash mismatch triggers full re-labeling
- Re-generate the UMAP plot and co-occurrence analysis with the updated labels
- Update the Experiment 006 report to reflect the 13-category taxonomy and new results

## Capabilities

### New Capabilities

<!-- No new capabilities — this modifies the existing taxonomy-label-generation taxonomy definitions -->

### Modified Capabilities

- `taxonomy-label-generation`: Add `deep-learning` as a 13th taxonomy category, update controlled title vocabulary, update golden subset labels, re-run labeling and visualization

## Impact

- Affected code: `scripts/006_taxonomy_labels/generate_labels.py` (taxonomy definitions, title vocabulary, golden labels), `scripts/006_taxonomy_labels/compare_labels.py` (no code change needed — reads labels from JSON)
- Affected data: `data/006_taxonomy_labels.json` (will be overwritten with new labels), `data/.006_taxonomy_cache.json` (will be invalidated and rebuilt), `data/plots/006_taxonomy_labels_umap.png` (will be regenerated)
- Affected docs: `docs/experiments/006-taxonomy-role-labeling.md` (will be updated to reflect 13-category taxonomy)
- No changes to backend modules, pipeline, or other experiments
