"""Compare embedding variants: raw_full_text vs clean_text vs llm_clean_text.

Phase 1.4b integration — embeds each text variant with MiniLM, computes
self-retrieval accuracy, NN audit, and separation metrics, then saves
a UMAP comparison plot showing all three variants side by side.

Usage:
    python scripts/003_llm_extraction/compare_variants.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
JOBS_PATH = PROJECT_ROOT / "data" / "jobs.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "plots"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RANDOM_SEED = 42

CATEGORY_COLORS: dict[str, str] = {
    "Data Scientist": "#1f77b4",
    "ML Engineer": "#ff7f0e",
    "AI Engineer": "#2ca02c",
    "Applied/Research": "#d62728",
    "Data Engineer": "#9467bd",
    "Other": "#8c564b",
}


class EmbeddingResult(NamedTuple):
    name: str
    embeddings: np.ndarray
    labels: list[str]
    roles: list[str]


def load_jobs() -> list[dict]:
    with open(JOBS_PATH) as f:
        return json.load(f)


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


def separation_gap(embeddings: np.ndarray, roles: list[str]) -> float:
    role_set = sorted(set(roles))
    if len(role_set) < 2:
        return 0.0
    sim = embeddings @ embeddings.T
    same_role_sims = []
    diff_role_sims = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            if roles[i] == roles[j]:
                same_role_sims.append(sim[i, j])
            else:
                diff_role_sims.append(sim[i, j])
    if not same_role_sims or not diff_role_sims:
        return 0.0
    return np.mean(same_role_sims) - np.mean(diff_role_sims)


def plot_umap_comparison(results: list[EmbeddingResult], output_path: Path) -> None:
    import umap

    n_variants = len(results)
    fig, axes = plt.subplots(1, n_variants, figsize=(6 * n_variants, 6))

    for ax, result in zip(axes, results):
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
                s=60,
                alpha=0.85,
                edgecolors="white",
                linewidth=0.5,
            )
        for i, label in enumerate(result.labels):
            ax.annotate(
                label[:10],
                (coords[i, 0], coords[i, 1]),
                fontsize=5,
                alpha=0.7,
                textcoords="offset points",
                xytext=(3, 3),
            )

        ax.set_title(result.name)
        ax.grid(True, alpha=0.3)

    handles, labels_legend = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels_legend,
        fontsize=7,
        loc="lower center",
        ncol=len(labels_legend),
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle("UMAP — Embedding Variant Comparison", fontsize=14)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"UMAP comparison saved → {output_path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    jobs = load_jobs()
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    variants = [
        ("raw_full_text", "Raw"),
        ("clean_text", "Section-Cleaned"),
        ("llm_clean_text", "LLM-Cleaned"),
    ]

    results: list[EmbeddingResult] = []
    labels = [f"{j['company'].split(' (')[0].split(' ')[0][:8]}" for j in jobs]
    roles = [j["role_category"] for j in jobs]

    for field, display_name in variants:
        texts = [j.get(field, j.get("raw_full_text", "")) for j in jobs]
        available = sum(1 for t in texts if t.strip())
        print(f"\n{'=' * 60}")
        print(
            f"Variant: {display_name} ({field}) — {available}/{len(texts)} texts available"
        )

        embeddings = embed_texts(texts, model)

        sr = self_retrieval(embeddings, labels)
        print(f"  Self-retrieval @ rank-0: {sr:.1f}%")

        gap = separation_gap(embeddings, roles)
        print(
            f"  Separation gap (mean same-role − mean diff-role cosine similarity): {gap:.4f}"
        )

        sim_matrix = embeddings @ embeddings.T
        np.fill_diagonal(sim_matrix, 0)
        print(f"  Mean pairwise cosine sim: {sim_matrix.mean():.4f}")
        print(f"  Max pairwise cosine sim:  {sim_matrix.max():.4f}")

        results.append(
            EmbeddingResult(
                name=display_name,
                embeddings=embeddings,
                labels=labels,
                roles=roles,
            )
        )

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(
        f"{'Variant':<20} {'Self-Retrieval':<16} {'Separation':<12} {'Mean CosSim':<12} {'Max CosSim'}"
    )
    print(f"{'-' * 75}")
    for result in results:
        sr = self_retrieval(result.embeddings, result.labels)
        gap = separation_gap(result.embeddings, result.roles)
        sim = result.embeddings @ result.embeddings.T
        np.fill_diagonal(sim, 0)
        print(
            f"{result.name:<20} {sr:>5.1f}%         {gap:>+.4f}        {sim.mean():>.4f}        {sim.max():>.4f}"
        )

    plot_umap_comparison(results, OUTPUT_DIR / "003-llm_cleaning_comparison.png")
    print("\nDone.")


if __name__ == "__main__":
    main()
