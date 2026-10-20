"""Compare embedding variants for Exp 008 cleaning outputs.

Phase 1.4c — Experiment 008, Step 5. Ports Exp 003's compare_variants.py
methodology verbatim (MiniLM-L6-v2 on CPU, normalized float32 embeddings,
hand-rolled self-retrieval with partial credit, O(n^2) pairwise separation
gap, UMAP grid plot with CATEGORY_COLORS) so 008 numbers stay directly
comparable with 003's.

Variants: raw, section-cleaned, golden, C2 (003 prompt baseline), C1, and
the five model arms A1–A5 — 10 panels. Missing texts (md-only postings
lack section-cleaned text; golden and arm outputs may lack postings) fall
back to raw text so every panel plots the same 39 points; separation gap
is computed on the labeled 27-posting subset only in every panel.

Limitation recorded in output: 12 of 39 postings are md-only and
role-unlabeled — they participate in self-retrieval and pairwise
statistics, are excluded from separation gap, and n=27 with role-heavy
classes keeps the Fisher formalization weak; all metrics are comparative.

Usage:
    python scripts/008_boilerplate_only_cleaning/compare_variants.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_DATA_DIR = Path(__file__).resolve().parent / "data"
CORPUS_PATH = SCRIPT_DATA_DIR / "008_corpus.json"
GOLDEN_PATH = PROJECT_ROOT / "data" / "golden_cleaned.json"  # shared golden set
OUTPUTS_PATH = SCRIPT_DATA_DIR / "008_clean_outputs.json"
OUTPUT_DIR = SCRIPT_DATA_DIR / "plots"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RANDOM_SEED = 42

CATEGORY_COLORS: dict[str, str] = {
    "Data Scientist": "#1f77b4",
    "ML Engineer": "#ff7f0e",
    "AI Engineer": "#2ca02c",
    "Applied/Research": "#d62728",
    "Data Engineer": "#9467bd",
    "Other": "#8c564b",
    "Unlabeled": "#7f7f7f",
}


class EmbeddingResult(NamedTuple):
    name: str
    embeddings: np.ndarray
    labels: list[str]
    roles: list[str]


def embed_texts(texts: list[str], model: SentenceTransformer) -> np.ndarray:
    return model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)


def self_retrieval(embeddings: np.ndarray, labels: list[str]) -> float:
    sim = embeddings @ embeddings.T
    hits = 0
    total = len(labels)
    for i in range(total):
        rankings = np.argsort(-sim[i])
        top_idx = rankings[0]
        if top_idx == i:
            hits += 1
        else:
            second = rankings[1] if len(rankings) > 1 else None
            if second == i:
                hits += 0.5
    return hits / total * 100


def separation_gap_labeled(
    embeddings: np.ndarray, roles: list[str]
) -> tuple[float, int]:
    """Pairwise mean same-role − mean diff-role cosine over labeled postings only."""
    idx = [i for i, r in enumerate(roles) if r]
    sim = embeddings @ embeddings.T
    same: list[float] = []
    diff: list[float] = []
    for a, i in enumerate(idx):
        for j in idx[a + 1 :]:
            if roles[i] == roles[j]:
                same.append(sim[i, j])
            else:
                diff.append(sim[i, j])
    if not same or not diff:
        return 0.0, len(idx)
    return float(np.mean(same) - np.mean(diff)), len(idx)


def plot_umap_grid(results: list[EmbeddingResult], output_path: Path) -> None:
    import umap

    n = len(results)
    n_rows, n_cols = 2, (n + 1) // 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 4.2 * n_rows))
    axes_flat = axes.flatten()

    for ax, result in zip(axes_flat, results):
        n_neighbors = min(5, result.embeddings.shape[0] - 1)
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=0.15,
            metric="cosine",
            random_state=RANDOM_SEED,
            n_jobs=1,
        )
        coords = reducer.fit_transform(result.embeddings)

        for role in sorted(set(result.roles)):
            mask = np.array([r == role for r in result.roles])
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                c=CATEGORY_COLORS.get(role, "#7f7f7f"),
                label=role,
                s=45,
                alpha=0.85,
                edgecolors="white",
                linewidth=0.5,
            )
        for i, label in enumerate(result.labels):
            ax.annotate(
                label[:10],
                (coords[i, 0], coords[i, 1]),
                fontsize=4,
                alpha=0.7,
                textcoords="offset points",
                xytext=(3, 3),
            )

        ax.set_title(result.name, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    handles, labels_legend = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels_legend,
        fontsize=6,
        loc="lower center",
        ncol=min(len(labels_legend), 7),
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle("UMAP — Exp 008 Cleaning Variant Comparison", fontsize=12)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"UMAP grid saved → {output_path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    corpus = json.loads(CORPUS_PATH.read_text())
    postings = corpus["postings"]
    golden = {g["posting_id"]: g["text"] for g in json.loads(GOLDEN_PATH.read_text())}
    outputs = json.loads(OUTPUTS_PATH.read_text())

    ids = [p["posting_id"] for p in postings]
    labels = [
        (p["company"].split(" (")[0].split(" ")[0][:8] if p["company"] else pid[:8])
        for p, pid in zip(postings, ids)
    ]
    roles = [p["role_category"] or "" for p in postings]

    raw = [p["raw_full_text"] for p in postings]
    section = [p.get("clean_text") or raw[i] for i, p in enumerate(postings)]
    variants: list[tuple[str, list[str]]] = [
        ("Raw", raw),
        ("Section-Cleaned", section),
        ("Golden", [golden.get(pid, raw[i]) for i, pid in enumerate(ids)]),
        (
            "C2 gpt-5.4-nano old-prompt",
            [
                outputs["arms"]["C2"]["outputs"].get(pid, {}).get("text") or raw[i]
                for i, pid in enumerate(ids)
            ],
        ),
        (
            "C1 gpt-5.4-nano new-prompt",
            [
                outputs["arms"]["C1"]["outputs"].get(pid, {}).get("text") or raw[i]
                for i, pid in enumerate(ids)
            ],
        ),
    ]
    model_arms = ["A1", "A2", "A3", "A4", "A5"]
    for arm_id in model_arms:
        arm = outputs["arms"][arm_id]
        variants.append(
            (
                f"{arm_id} {arm['model'].split('/')[-1]}",
                [
                    arm["outputs"].get(pid, {}).get("text") or raw[i]
                    for i, pid in enumerate(ids)
                ],
            )
        )

    model = SentenceTransformer(MODEL_NAME, device="cpu")
    results: list[EmbeddingResult] = []
    summary = {}
    for name, texts in variants:
        embeddings = embed_texts(texts, model)
        sr = self_retrieval(embeddings, labels)
        gap, n_labeled = separation_gap_labeled(embeddings, roles)
        sim_matrix = embeddings @ embeddings.T
        off_diag = sim_matrix[~np.eye(sim_matrix.shape[0], dtype=bool)]
        print(f"\n{'=' * 64}")
        print(f"Variant: {name} (n embeddings = {embeddings.shape[0]})")
        print(f"  Self-retrieval @ rank-0 (+0.5 partial): {sr:.1f}%")
        print(f"  Separation gap (labeled {n_labeled} only): {gap:+.4f}")
        print(f"  Mean pairwise cosine: {off_diag.mean():.4f}")
        print(f"  Max  pairwise cosine: {off_diag.max():.4f}")
        results.append(
            EmbeddingResult(
                name=name, embeddings=embeddings, labels=labels, roles=roles
            )
        )
        summary[name] = {
            "self_retrieval_pct": sr,
            "separation_gap_labeled27": gap,
            "mean_pairwise_cosine": float(off_diag.mean()),
            "max_pairwise_cosine": float(off_diag.max()),
        }

    print(f"\n{'=' * 64}")
    print("SUMMARY (comparative; 12/39 postings role-unlabeled; n=27 separation)")
    print(f"{'-' * 64}")
    for name, stats in summary.items():
        print(
            f"{name:<30} SR {stats['self_retrieval_pct']:>5.1f}%  "
            f"Gap {stats['separation_gap_labeled27']:>+.4f}  "
            f"Mean {stats['mean_pairwise_cosine']:.4f}  Max {stats['max_pairwise_cosine']:.4f}"
        )

    plot_umap_grid(results, OUTPUT_DIR / "008-cleaning_comparison.png")

    results_path = SCRIPT_DATA_DIR / "008_embedding_comparison.json"
    with open(results_path, "w") as f:
        json.dump(
            {
                "experiment": "008_boilerplate_only_cleaning",
                "embedding_model": MODEL_NAME,
                "normalized": True,
                "seed": RANDOM_SEED,
                "n_postings": len(ids),
                "n_labeled_separation": int(sum(1 for r in roles if r)),
                "summary": summary,
                "artifacts": [str(OUTPUT_DIR / "008-cleaning_comparison.png")],
                "limitations": [
                    (
                        "12 of 39 postings md-only and role-unlabeled: excluded from "
                        "separation gap, included in self-retrieval and pairwise stats."
                    ),
                    (
                        "Missing variant texts fall back to raw_full_text so panels "
                        "stay aligned at n=39."
                    ),
                    (
                        "n=27 with role-heavy classes keeps formal Fisher testing "
                        "weak; metrics are comparative, not inferential."
                    ),
                ],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Summary → {results_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
