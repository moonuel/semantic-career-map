"""Tag-set-colored UMAP visualization of taxonomy-labeled job postings.

Phase 2.1b — Embeds llm_clean_text with MiniLM-L6-v2, assigns one color per
unique function_tags combination, generates a UMAP scatter plot for qualitative
inspection of taxonomy-aligned clustering, and prints tag co-occurrence counts.

Usage:
    python scripts/006_taxonomy_labels/compare_labels.py
"""

from __future__ import annotations

import json
from collections import Counter
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
JOBS_PATH = PROJECT_ROOT / "data" / "jobs.json"
LABELS_PATH = PROJECT_ROOT / "data" / "006_taxonomy_labels.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "plots"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RANDOM_SEED = 42

TAG_SET_COLORS: list[str] = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#aec7e8",
    "#ffbb78",
    "#98df8a",
    "#ff9896",
    "#c5b0d5",
    "#c49c94",
    "#f7b6d2",
    "#c7c7c7",
    "#dbdb8d",
    "#9edae5",
]


def embed_texts(texts: list[str], model: SentenceTransformer) -> np.ndarray:
    return model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)


def print_cooccurrence(tag_lists: list[list[str]]) -> None:
    tag_pairs: Counter[tuple[str, str]] = Counter()
    tag_triplets: Counter[tuple[str, ...]] = Counter()

    for tags in tag_lists:
        for pair in combinations(sorted(tags), 2):
            tag_pairs[pair] += 1
        if len(tags) >= 3:
            for triplet in combinations(sorted(tags), 3):
                tag_triplets[triplet] += 1

    if tag_pairs:
        print(f"\n{'=' * 60}")
        print("TAG PAIR CO-OCCURRENCE")
        print(f"{'=' * 60}")
        for (a, b), cnt in tag_pairs.most_common():
            if cnt >= 2:
                print(f"  {a:30s} + {b:30s} = {cnt}")

    if tag_triplets:
        print(f"\n{'=' * 60}")
        print("TAG TRIPLET CO-OCCURRENCE")
        print(f"{'=' * 60}")
        for triplet, cnt in tag_triplets.most_common():
            if cnt >= 2:
                print(f"  {' + '.join(triplet):70s} = {cnt}")


def plot_tag_set_umap(
    embeddings: np.ndarray,
    posting_ids: list[str],
    tag_set_labels: list[str],
    tag_set_counts: Counter[str],
    output_path: Path,
) -> None:
    import umap

    n_neighbors = min(5, embeddings.shape[0] - 1)
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=0.15,
        metric="cosine",
        random_state=RANDOM_SEED,
        n_jobs=1,
    )
    coords = reducer.fit_transform(embeddings)

    sorted_sets = [label for label, _ in tag_set_counts.most_common()]
    color_map: dict[str, str] = {
        label: TAG_SET_COLORS[i % len(TAG_SET_COLORS)]
        for i, label in enumerate(sorted_sets)
    }

    fig, ax = plt.subplots(figsize=(11, 8))

    for tag_set in sorted_sets:
        mask = np.array([ts == tag_set for ts in tag_set_labels])
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=color_map[tag_set],
            label=tag_set,
            s=70,
            alpha=0.85,
            edgecolors="white",
            linewidth=0.5,
        )

    for i, label in enumerate(posting_ids):
        ax.annotate(
            label[:10],
            (coords[i, 0], coords[i, 1]),
            fontsize=5,
            alpha=0.7,
            textcoords="offset points",
            xytext=(3, 3),
        )

    ax.set_title(
        f"UMAP — Tag-Set-Colored Postings ({len(sorted_sets)} unique tag sets)",
        fontsize=13,
    )
    ax.grid(True, alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        fontsize=7,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.97])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"UMAP saved -> {output_path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(JOBS_PATH) as f:
        jobs = json.load(f)
    with open(LABELS_PATH) as f:
        tax_labels = json.load(f)

    tax_by_id = {r["posting_id"]: r for r in tax_labels}
    posting_ids = [j["id"] for j in jobs]
    tag_lists = [tax_by_id[pid]["function_tags"] for pid in posting_ids]
    tag_set_labels = [json.dumps(sorted(tags)) for tags in tag_lists]
    tag_set_counts = Counter(tag_set_labels)
    llm_clean_texts = [j["llm_clean_text"] for j in jobs]

    model = SentenceTransformer(MODEL_NAME, device="cpu")

    print(f"Embedding {len(llm_clean_texts)} postings with {MODEL_NAME}...")
    embeddings = embed_texts(llm_clean_texts, model)

    print_cooccurrence(tag_lists)

    print(f"\n{'=' * 60}")
    print("TAG SET DISTRIBUTION")
    print(f"{'=' * 60}")
    for label, count in tag_set_counts.most_common():
        print(f"  {count:2d} x {json.loads(label)}")

    plot_tag_set_umap(
        embeddings,
        posting_ids,
        tag_set_labels,
        tag_set_counts,
        OUTPUT_DIR / "006_taxonomy_labels_umap.png",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
