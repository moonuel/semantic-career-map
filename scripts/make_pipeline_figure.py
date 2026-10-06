"""Generate the conceptual pipeline figure for the architecture docs.

Produces a 2x2 figure explaining the semantic pipeline to a non-technical
audience:

    text -> 3D point cloud -> coloured clusters -> 2D projection (colours kept)

The top row shows a sample of job posting text and the same text embedded into
a 3D UMAP space as a single colourless cloud of points. The bottom row shows
that cloud coloured by semantic cluster, then projected into 2D by a second
UMAP fit — the cluster colours are carried over unchanged, which is the key
takeaway: dimensionality reduction shrinks the cloud to two dimensions while
preserving which points belong together.

This is a documentation artifact, not an experiment. It lives in an unnumbered
directory under ``scripts/`` (like ``migrate_selected_postings/``) and embeds a
small synthetic corpus of role descriptions so the clusters look clean and
intuitive. The output path is configurable via CLI.

Usage:
    python scripts/make_pipeline_figure.py [--output docs/architecture/pipeline.png]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import umap
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "docs" / "architecture" / "pipeline.png"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RANDOM_SEED = 42
N_CLUSTERS = 4

ROLE_GROUPS: dict[str, list[str]] = {
    "Data Engineering": [
        "Design and maintain batch pipelines that move data between systems.",
        "Write the ETL jobs that clean and load warehouse tables nightly.",
        "Build the data platform that streams events into the lakehouse.",
        "Tune SQL queries and manage the schemas behind the reporting cube.",
        "Operate the orchestration that schedules and retries data jobs.",
    ],
    "Machine Learning": [
        "Train and evaluate predictive models on historical customer data.",
        "Run experiments to tune hyperparameters and compare model baselines.",
        "Build the feature store that feeds models in production.",
        "Measure offline metrics and monitor drift once models are live.",
        "Prototype a recommendation model and ship it behind an API.",
    ],
    "Data Analysis": [
        "Explore sales data to find trends and explain them to stakeholders.",
        "Build the dashboards that track weekly business metrics.",
        "Run ad-hoc SQL analyses to answer questions from product teams.",
        "Prepare clean datasets and summary tables for the analytics team.",
        "Communicate findings with charts and written recommendations.",
    ],
    "Software Engineering": [
        "Write and review the backend services behind the web product.",
        "Build the REST APIs that mobile and web clients call.",
        "Refactor the service layer and add integration tests.",
        "Maintain the deployment scripts and improve build times.",
        "Implement the feature that syncs user data across devices.",
    ],
}


def build_corpus() -> tuple[list[str], np.ndarray]:
    """Return (texts, integer labels) where labels map texts to role groups."""
    texts: list[str] = []
    labels: list[int] = []
    for index, group in enumerate(ROLE_GROUPS.values()):
        texts.extend(group)
        labels.extend([index] * len(group))
    return texts, np.asarray(labels, dtype=int)


def _style_3d_axes(ax, title: str) -> None:
    """Apply the common look to a 3D axes panel."""
    ax.set_title(title)
    ax.set_xlabel("meaning 1")
    ax.set_ylabel("meaning 2")
    ax.set_zlabel("meaning 3")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(elev=22, azim=-62)


def _style_2d_axes(ax, title: str) -> None:
    """Apply the common look to a 2D axes panel."""
    ax.set_title(title)
    ax.set_xlabel("meaning (1D)")
    ax.set_ylabel("meaning (2D)")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.3)


def _render_text_panel(ax, texts: list[str]) -> None:
    """Draw a job posting text sample on a plain axes panel."""
    ax.axis("off")
    ax.set_title("1. Ingest job postings")
    # sample = "\n\n".join(texts[:2])
    sample = 'job_posting = "In this role, you will develop, deploy, \nand maintain ML models end-to-end"...\n\nembed(job_posting) # convert into high-dimensional vector'
    ax.text(
        0.5,
        0.5,
        sample,
        ha="center",
        va="center",
        fontsize=11,
        style="italic",
        wrap=True,
        transform=ax.transAxes,
    )


def render_figure(
    texts: list[str],
    points3d: np.ndarray,
    points2d: np.ndarray,
    labels: np.ndarray,
    output: Path,
) -> None:
    """Draw the four panels of a 2x2 grid and write the combined figure.

    Top row: job posting text -> unlabelled 3D point cloud.
    Bottom row: the same cloud coloured by cluster -> 2D projection with
    colours kept.
    """
    colors = plt.get_cmap("tab10")
    color_map = {i: colors(i % 10) for i in range(N_CLUSTERS)}

    fig = plt.figure(figsize=(13, 9))
    # fig.suptitle(
    #     "Semantic Career Map: from text to a 2D map of roles",
    #     fontsize=14,
    #     fontweight="bold",
    #     y=0.97,
    # )

    ax_text = fig.add_subplot(2, 2, 1)
    _render_text_panel(ax_text, texts)

    ax3d_plain = fig.add_subplot(2, 2, 2, projection="3d")
    ax3d_plain.scatter(
        points3d[:, 0],
        points3d[:, 1],
        points3d[:, 2],
        c="0.35",
        s=30,
        alpha=0.85,
    )
    _style_3d_axes(ax3d_plain, "2. Text projected into high-dimensional space")

    ax3d = fig.add_subplot(2, 2, 3, projection="3d")
    for i in range(N_CLUSTERS):
        mask = labels == i
        ax3d.scatter(
            points3d[mask, 0],
            points3d[mask, 1],
            points3d[mask, 2],
            c=[color_map[i]],
            s=30,
            alpha=0.85,
            label=list(ROLE_GROUPS)[i],
        )
    _style_3d_axes(ax3d, "3. Postings are clustered by semantic similarity")

    ax2d = fig.add_subplot(2, 2, 4)
    for i in range(N_CLUSTERS):
        mask = labels == i
        ax2d.scatter(
            points2d[mask, 0],
            points2d[mask, 1],
            c=[color_map[i]],
            s=30,
            alpha=0.85,
            label=list(ROLE_GROUPS)[i],
        )
    _style_2d_axes(ax2d, "4. Low-dimensional projections provide interpretable visuals")

    handles, labels_out = ax2d.get_legend_handles_labels()
    fig.legend(handles, labels_out, loc="lower center", ncol=len(ROLE_GROUPS), frameon=False, fontsize=10)

    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"pipeline figure saved -> {output}")


def _apply_science_style() -> None:
    """Apply a hand-rolled equivalent of the SciencePlots ``science`` preset.

    Replicates the preset's matplotlib rcParams directly so no LaTeX or extra
    dependency is needed. The LaTeX-dependent ``text.usetex`` key is omitted
    (no TeX distribution is installed).
    """
    plt.rcParams.update(
        {
            "axes.prop_cycle": plt.cycler(
                color=["#0C5DA5", "#00B945", "#FF9500", "#FF2C00", "#845B97", "#474747", "#9e9e9e"]
            ),
            "xtick.direction": "in",
            "xtick.major.size": 3,
            "xtick.major.width": 0.5,
            "xtick.minor.size": 1.5,
            "xtick.minor.width": 0.5,
            "xtick.minor.visible": True,
            "xtick.top": True,
            "ytick.direction": "in",
            "ytick.major.size": 3,
            "ytick.major.width": 0.5,
            "ytick.minor.size": 1.5,
            "ytick.minor.width": 0.5,
            "ytick.minor.visible": True,
            "ytick.right": True,
            "axes.linewidth": 0.5,
            "grid.linewidth": 0.5,
            "lines.linewidth": 1.0,
            "legend.frameon": False,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "font.family": "serif",
            "mathtext.fontset": "dejavuserif",
        }
    )


def main() -> None:
    _apply_science_style()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="where to write the PNG (default: docs/architecture/pipeline.png)",
    )
    args = parser.parse_args()

    texts, labels = build_corpus()
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)

    reducer3d = umap.UMAP(n_components=3, random_state=RANDOM_SEED, n_neighbors=8, min_dist=0.1)
    points3d = reducer3d.fit_transform(embeddings)

    kmeans = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=RANDOM_SEED)
    kmeans.fit(points3d)
    labels = kmeans.labels_

    reducer2d = umap.UMAP(n_components=2, random_state=RANDOM_SEED, n_neighbors=8, min_dist=0.1)
    points2d = reducer2d.fit_transform(embeddings)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    render_figure(texts, points3d, points2d, labels, args.output)


if __name__ == "__main__":
    main()
