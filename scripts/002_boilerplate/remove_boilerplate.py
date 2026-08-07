#!/usr/bin/env python3
"""Experiment 1.4: Boilerplate removal.

Hypothesis: Removing company descriptions, "About Us", EEO statements,
salary disclosures, and recruiter notes from embedding input increases
role-level separation in the embedding space.

Uses the same SentenceTransformer and L2 normalization as baseline.
Generates side-by-side UMAP, PCA, and t-SNE plots (raw vs cleaned) and
prints a before/after metric comparison table.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer

JOBS_PATH = Path("data/jobs.json")
PLOTS_DIR = Path("data/plots")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RANDOM_SEED = 42

KEEP_SECTIONS = ["about_role", "responsibilities", "qualifications", "nice_to_have"]

CATEGORY_COLORS: dict[str, str] = {
    "Data Scientist": "#1f77b4",
    "ML Engineer": "#ff7f0e",
    "AI Engineer": "#2ca02c",
    "Applied/Research": "#d62728",
    "Data Engineer": "#9467bd",
    "Other": "#8c564b",
}


def build_clean_text(sections: dict[str, str]) -> str:
    """Concatenate only role-relevant sections."""
    parts: list[str] = []
    for key in KEEP_SECTIONS:
        val = sections.get(key, "").strip()
        if val:
            parts.append(val)
    return "\n\n".join(parts)


def embed_texts(texts: list[str], model: SentenceTransformer) -> np.ndarray:
    """Embed and L2-normalize a list of texts."""
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    norms = np.linalg.norm(emb, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), (
        f"L2 norm check failed: min={norms.min():.6f} max={norms.max():.6f}"
    )
    return emb.astype(np.float32)


def _scatter_role_category(ax, coords, jobs, roles_sorted):
    for role in roles_sorted:
        mask = np.array([j["role_category"] == role for j in jobs])
        color = CATEGORY_COLORS.get(role, "#7f7f7f")
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=color,
            label=role,
            s=100,
            alpha=0.85,
            edgecolors="white",
            linewidth=0.5,
        )


def _annotate_labels(ax, coords, jobs):
    for i, job in enumerate(jobs):
        label = f"{job['company'].split(' (')[0].split(' ')[0][:8]}\n{job['title_raw'][:20]}"
        ax.annotate(
            label,
            (coords[i, 0], coords[i, 1]),
            fontsize=6,
            alpha=0.8,
            textcoords="offset points",
            xytext=(5, 4),
        )


def plot_umap_comparison(
    raw_emb: np.ndarray,
    clean_emb: np.ndarray,
    jobs: list[dict],
    output_path: Path,
) -> None:
    import umap

    reducer = umap.UMAP(
        n_neighbors=5,
        min_dist=0.15,
        metric="cosine",
        random_state=RANDOM_SEED,
        n_jobs=1,
    )
    raw_coords = reducer.fit_transform(raw_emb)
    clean_coords = reducer.fit_transform(clean_emb)

    roles_sorted = sorted({j["role_category"] for j in jobs})
    fig, axes = plt.subplots(1, 2, figsize=(22, 9))

    for ax, coords, title in [
        (axes[0], raw_coords, "Raw Full Text — UMAP"),
        (axes[1], clean_coords, "Boilerplate Removed — UMAP"),
    ]:
        _scatter_role_category(ax, coords, jobs, roles_sorted)
        _annotate_labels(ax, coords, jobs)
        ax.set_title(title)
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"UMAP comparison saved → {output_path}")


def plot_pca_scatter_comparison(
    raw_emb: np.ndarray,
    clean_emb: np.ndarray,
    jobs: list[dict],
    output_path: Path,
) -> None:
    from sklearn.decomposition import PCA

    raw_pca = PCA(n_components=min(len(jobs), 10), random_state=RANDOM_SEED)
    raw_coords = raw_pca.fit_transform(raw_emb)
    raw_var = raw_pca.explained_variance_ratio_

    clean_pca = PCA(n_components=min(len(jobs), 10), random_state=RANDOM_SEED)
    clean_coords = clean_pca.fit_transform(clean_emb)
    clean_var = clean_pca.explained_variance_ratio_

    roles_sorted = sorted({j["role_category"] for j in jobs})
    fig, axes = plt.subplots(1, 2, figsize=(22, 9))

    for ax, coords, var, title in [
        (
            axes[0],
            raw_coords,
            raw_var,
            f"Raw Full Text — PCA (PC1: {raw_var[0]:.1%}, PC2: {raw_var[1]:.1%})",
        ),
        (
            axes[1],
            clean_coords,
            clean_var,
            f"Boilerplate Removed — PCA (PC1: {clean_var[0]:.1%}, PC2: {clean_var[1]:.1%})",
        ),
    ]:
        _scatter_role_category(ax, coords, jobs, roles_sorted)
        _annotate_labels(ax, coords, jobs)
        ax.set_title(title)
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"PCA scatter comparison saved → {output_path}")


def plot_pca_scree_comparison(
    raw_emb: np.ndarray,
    clean_emb: np.ndarray,
    jobs: list[dict],
    output_path: Path,
) -> None:
    from sklearn.decomposition import PCA

    raw_pca = PCA(n_components=min(len(jobs), 10), random_state=RANDOM_SEED)
    raw_pca.fit(raw_emb)

    clean_pca = PCA(n_components=min(len(jobs), 10), random_state=RANDOM_SEED)
    clean_pca.fit(clean_emb)

    components = range(1, len(raw_pca.explained_variance_ratio_) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    for ax, pca, title in [
        (axes[0], raw_pca, "Scree Plot — Raw Full Text"),
        (axes[1], clean_pca, "Scree Plot — Boilerplate Removed"),
    ]:
        var = pca.explained_variance_ratio_
        ax.bar(components, var, color="#1f77b4", alpha=0.7)
        ax.plot(
            components,
            np.cumsum(var),
            "o-",
            color="#d62728",
            linewidth=2,
            label="Cumulative",
        )
        ax.set_title(title)
        ax.set_xlabel("Principal Component")
        ax.set_ylabel("Explained Variance Ratio")
        ax.set_xticks(components)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"PCA scree comparison saved → {output_path}")


def plot_tsne_comparison(
    raw_emb: np.ndarray,
    clean_emb: np.ndarray,
    jobs: list[dict],
    output_path: Path,
) -> None:
    from sklearn.manifold import TSNE

    perplexity = min(5, len(jobs) - 1)
    raw_coords = TSNE(
        n_components=2,
        metric="cosine",
        perplexity=perplexity,
        random_state=RANDOM_SEED,
        n_jobs=1,
    ).fit_transform(raw_emb)
    clean_coords = TSNE(
        n_components=2,
        metric="cosine",
        perplexity=perplexity,
        random_state=RANDOM_SEED,
        n_jobs=1,
    ).fit_transform(clean_emb)

    roles_sorted = sorted({j["role_category"] for j in jobs})
    fig, axes = plt.subplots(1, 2, figsize=(22, 9))

    for ax, coords, title in [
        (axes[0], raw_coords, f"Raw Full Text — t-SNE (perplexity={perplexity})"),
        (
            axes[1],
            clean_coords,
            f"Boilerplate Removed — t-SNE (perplexity={perplexity})",
        ),
    ]:
        _scatter_role_category(ax, coords, jobs, roles_sorted)
        _annotate_labels(ax, coords, jobs)
        ax.set_title(title)
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"t-SNE comparison saved → {output_path}")


def compute_metrics(embeddings: np.ndarray, jobs: list[dict], label: str) -> dict:
    """Compute summary metrics for a set of embeddings."""
    n = len(jobs)
    sim_matrix = embeddings @ embeddings.T
    np.fill_diagonal(sim_matrix, 0)

    self_positions = []
    for i in range(n):
        ranked = np.argsort(-sim_matrix[i])
        pos = int(np.where(ranked == i)[0][0])
        self_positions.append(pos)

    same_role_sims = []
    diff_role_sims = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = float(sim_matrix[i, j])
            if jobs[i]["role_category"] == jobs[j]["role_category"]:
                same_role_sims.append(sim)
            else:
                diff_role_sims.append(sim)

    return {
        "label": label,
        "self_retrieval_rank_0": sum(1 for p in self_positions if p == 0),
        "self_retrieval_total": n,
        "self_retrieval_worst_rank": max(self_positions),
        "mean_pairwise_cos": float(sim_matrix.mean()),
        "max_pairwise_cos": float(sim_matrix.max()),
        "min_pairwise_cos": float(sim_matrix.min()),
        "same_role_mean_sim": float(np.mean(same_role_sims)),
        "cross_role_mean_sim": float(np.mean(diff_role_sims)),
        "separation_gap": float(np.mean(same_role_sims) - np.mean(diff_role_sims)),
    }


def print_comparison(raw: dict, clean: dict) -> None:
    """Print a side-by-side metric comparison table."""
    float_rows = [
        ("Mean pairwise cos", "mean_pairwise_cos"),
        ("Max pairwise cos", "max_pairwise_cos"),
        ("Min pairwise cos", "min_pairwise_cos"),
        ("Same-role mean sim", "same_role_mean_sim"),
        ("Cross-role mean sim", "cross_role_mean_sim"),
        ("Separation gap (title)", "separation_gap"),
    ]

    print(f"\n{'Metric':<35s} {'Raw':>10s} {'Cleaned':>10s} {'Δ':>10s}")
    print("-" * 65)
    for label, key in float_rows:
        rv = raw[key]
        cv = clean[key]
        delta = cv - rv
        print(f"{label:<35s} {rv:>10.4f} {cv:>10.4f} {delta:>+10.4f}")
    print(
        f"{'Self-retrieval @0':<35s} {raw['self_retrieval_rank_0']}/{raw['self_retrieval_total']:>6} {clean['self_retrieval_rank_0']}/{clean['self_retrieval_total']:>6}"
    )


def print_section_sizes(jobs: list[dict]) -> None:
    """Print raw vs clean text size comparison."""
    print("\n── Section Size Impact ──")
    print(f"{'Posting':<50s} {'Raw':>6s} {'Clean':>6s} {'%Kept':>6s} {'Excluded':>12s}")
    print("-" * 90)
    for job in jobs:
        raw_len = len(job["raw_full_text"])
        clean_len = len(job["clean_text"])
        excluded = [k for k in job["sections"] if k not in KEEP_SECTIONS]
        print(
            f"{job['id']:<50s} {raw_len:>6d} {clean_len:>6d} "
            f"{clean_len / max(raw_len, 1):>5.0%}  {','.join(excluded) if excluded else '(none)':>12s}"
        )


def main() -> None:
    """Run boilerplate removal experiment: clean, embed, compare, visualize."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(JOBS_PATH, encoding="utf-8") as f:
        jobs = json.load(f)
    print(f"Loaded {len(jobs)} postings from {JOBS_PATH}")

    raw_texts = [j["raw_full_text"] for j in jobs]
    for job in jobs:
        job["clean_text"] = build_clean_text(job["sections"])

    with open(JOBS_PATH, "w", encoding="utf-8") as f:
        json.dump(jobs, f, indent=2, ensure_ascii=False)
    print(f"Updated {JOBS_PATH} with clean_text field")

    print_section_sizes(jobs)

    print("\n── Embedding ──")
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    print(f"Model: {MODEL_NAME}")

    raw_emb = embed_texts(raw_texts, model)
    clean_emb = embed_texts([j["clean_text"] for j in jobs], model)

    raw_metrics = compute_metrics(raw_emb, jobs, "Raw full text")
    clean_metrics = compute_metrics(clean_emb, jobs, "Boilerplate removed")

    print_comparison(raw_metrics, clean_metrics)

    print("\n── Visualizing ──")
    plot_umap_comparison(
        raw_emb, clean_emb, jobs, PLOTS_DIR / "exp_boilerplate_umap.png"
    )
    plot_pca_scatter_comparison(
        raw_emb, clean_emb, jobs, PLOTS_DIR / "exp_boilerplate_pca.png"
    )
    plot_pca_scree_comparison(
        raw_emb, clean_emb, jobs, PLOTS_DIR / "exp_boilerplate_scree.png"
    )
    plot_tsne_comparison(
        raw_emb, clean_emb, jobs, PLOTS_DIR / "exp_boilerplate_tsne.png"
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
