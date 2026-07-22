#!/usr/bin/env python3
"""Bootstrap pipeline: parse 27 job postings, embed with MiniLM, visualize.

Phase 1 bootstrap (Steps 1.0–1.2 combined). Reads markdown files from
data/selected-job-postings/, extracts metadata, embeds raw full text with
all-MiniLM-L6-v2, saves structured JSON and L2-normalized embeddings, and
generates PCA + UMAP plots to inspect the embedding space qualitatively.

This script is deliberately monolithic for fast iteration. It will be
refactored into per-step scripts (parse_postings.py, embed_baseline.py,
visualize.py) once the experiment loop begins in Step 1.3.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer

POSTINGS_DIR = Path("data/selected-job-postings")
OUTPUT_JOBS = Path("data/jobs.json")
OUTPUT_EMBEDDINGS = Path("data/raw_embeddings.npy")
OUTPUT_PLOTS_DIR = Path("data/plots")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RANDOM_SEED = 42

ROLE_CATEGORIES: dict[str, str] = {
    "affirm-ai-solutions-engineer": "AI Engineer",
    "affirm-ml-engineer-2": "ML Engineer",
    "bmo-associate-data-scientist": "Data Scientist",
    "cerebras-ml-performance-benchmarking-engineer": "ML Engineer",
    "clario-ai-engineer": "AI Engineer",
    "clio-ml-engineer": "ML Engineer",
    "coca-cola-data-scientist": "Data Scientist",
    "dayforce-product-ai-intern": "Other",
    "deloitte-ai-ml-models-consultant-analyst": "Other",
    "ebay-applied-researcher-1": "Applied/Research",
    "fgf-brands-ai-engineer-new-grad": "AI Engineer",
    "hellofresh-ml-engineer-operations-technology": "ML Engineer",
    "huawei-ai-ml-researcher": "Applied/Research",
    "industrial-and-commercial-bank-of-china-data-science-analyst": "Other",
    "intact-data-scientist-2": "Data Scientist",
    "kinaxis-data-analytics-coop-intern": "Other",
    "mastercard-data-scientist-2": "Data Scientist",
    "motion-recruitment-ml-engineer": "ML Engineer",
    "ontario-teachers-pension-plan-data-scientist": "Data Scientist",
    "pinterest-ml-engineer-2": "ML Engineer",
    "rbc-data-scientist-ai-model-risk": "Data Scientist",
    "scotiabank-ai-ml-data-scientist": "Data Scientist",
    "scribd-data-scientist-2": "Data Scientist",
    "tactable-data-engineer": "Data Engineer",
    "td-applied-ml-scientist-1": "Data Scientist",
    "thumbtack-data-scientist-2-monetization-pricing": "Data Scientist",
    "venterra-realty-data-decision-scientist": "Other",
}

CATEGORY_COLORS: dict[str, str] = {
    "Data Scientist": "#1f77b4",
    "ML Engineer": "#ff7f0e",
    "AI Engineer": "#2ca02c",
    "Applied/Research": "#d62728",
    "Data Engineer": "#9467bd",
    "Other": "#8c564b",
}

SECTION_PATTERNS: list[tuple[str, str]] = [
    (r"(?i)^about\s+(the\s+)?(role|job|this\s+opportunity|the\s+data\s+science\s+team)\b", "about_role"),
    (r"(?i)^(summary|overview|role\s+overview|the\s+role|role\s+details|the\s+opportunity):?\s*$", "about_role"),
    (r"(?i)^(job\s+description|additional\s+job\s+description)\b", "about_role"),
    (r"(?i)^what\s+you('ll|\u2019ll|\u2019ll| will)\s+(do|accomplish|be\s+doing|get\s+to\s+do|do\s+here)", "responsibilities"),
    (r"(?i)^what\s+you('ll|\u2019ll|\u2019ll)\s+do\b", "responsibilities"),
    (r"(?i)^(key\s+responsibilities|primary\s+responsibilities)", "responsibilities"),
    (r"(?i)^(a\s+day\s+in\s+the\s+life|what\s+your\s+typical\s+day)", "responsibilities"),
    (r"(?i)^(job\s+description\s+and\s+responsibilities|scope\s+of\s+the\s+team)", "responsibilities"),
    (r"(?i)^(what\s+you('ll|\u2019ll|\u2019ll| will)\s+bring|what\s+you\s+bring)", "qualifications"),
    (r"(?i)^(what\s+we('re|\u2019re|\u2019re| are)\s+looking\s+for|what\s+we\s+look\s+for)", "qualifications"),
    (r"(?i)^(qualifications|requirements|skills\s*(&|and)\s*(qualifications|experience|we\s+value))", "qualifications"),
    (r"(?i)^(job\s+requirements|must-?have\s+experience|required\s+(skills|experience))", "qualifications"),
    (r"(?i)^(you\s+are\s+someone\s+with|what\s+you\s+need|what\s+you\s+may\s+have)", "qualifications"),
    (r"(?i)^(in\s+order\s+to\s+be\s+successful|do\s+you\s+have\s+the\s+skills|essential\s+skills)", "qualifications"),
    (r"(?i)^(nice\s+to\s+have|preferred\s+skills|assets\s*\(|bonus\s+points|desired\s+skills)", "nice_to_have"),
    (r"(?i)^what\s+would\s+really\s+make\s+you\s+stand\s+out", "nice_to_have"),
    (r"(?i)^what\s+(we\s+offer|you('ll|\u2019ll|\u2019ll| will)\s+love|you\s+will\s+find)", "what_we_offer"),
    (r"(?i)^(what('s|\u2019s|\u2019s| is)\s+in\s+it\s+for\s+you|why\s+you('ll|\u2019ll|\u2019ll)\s+love)", "what_we_offer"),
    (r"(?i)^(total\s+rewards|compensation|why\s+join|benefits)$", "what_we_offer"),
    (r"(?i)^about\s+(the\s+)?(team|us|pinterest|kinaxis|the\s+company)", "about_team"),
    (r"(?i)^(our\s+(team|purpose|mission)|what\s+your\s+team\s+does)", "about_team"),
    (r"(?i)^(who\s+you('ll|\u2019ll|\u2019ll| will)\s+work\s+with|about\s+the\s+ideal\s+candidate)", "about_team"),
]

EXPLICIT_TITLE_PATTERNS = [
    r"(?i)^job\s+title:\s*(.+)",
    r"(?i)^title\s+and\s+summary\s*\n+(.+)",
    r"(?i)^#+\s+(.+)",
    r"(?i)^(analyst/consultant,?\s*.+?models)",
    r"(?i)^(data\s+scientist\s+ii|machine\s+learning\s+engineer)\s*$",
]

COMPANY_KNOWN: dict[str, str] = {
    "affirm": "Affirm",
    "bmo": "BMO Capital Markets",
    "cerebras": "Cerebras Systems",
    "clario": "Clario (Thermo Fisher Scientific)",
    "clio": "Clio",
    "coca-cola": "Coca-Cola Canada Bottling",
    "dayforce": "Dayforce",
    "deloitte": "Deloitte",
    "ebay": "eBay",
    "fgf-brands": "FGF Brands",
    "hellofresh": "HelloFresh",
    "huawei": "Huawei Canada",
    "industrial-and-commercial-bank-of-china": "ICBC",
    "intact": "Intact",
    "kinaxis": "Kinaxis",
    "mastercard": "Mastercard",
    "motion-recruitment": "Motion Recruitment",
    "ontario-teachers-pension-plan": "Ontario Teachers' Pension Plan",
    "pinterest": "Pinterest",
    "rbc": "RBC",
    "scotiabank": "Scotiabank",
    "scribd": "Scribd",
    "tactable": "Tactable",
    "td": "TD Bank",
    "thumbtack": "Thumbtack",
    "venterra-realty": "Venterra Realty",
}


def _extract_title(text: str, filepath: Path) -> str:
    """Extract a human-readable job title from posting text or filename."""
    for pattern in EXPLICIT_TITLE_PATTERNS:
        m = re.search(pattern, text, re.MULTILINE)
        if m:
            candidate = m.group(1).strip()
            if len(candidate) >= 8:
                return candidate
    stem = filepath.stem
    parts = stem.split("-")
    keywords = [
        "data", "scientist", "analyst", "engineer", "researcher",
        "intern", "consultant", "ml", "ai", "applied", "machine",
        "learning", "solutions", "product", "benchmarking", "decision",
        "performance",
    ]
    title_parts = []
    for part in parts:
        if part.replace("1", "").replace("2", "").isdigit():
            continue
        if part.lower() in keywords or part.lower() in {"new", "grad", "coop", "co-op"}:
            title_parts.append(part.upper() if part.lower() in {"ml", "ai"} else part.title())
        elif title_parts:
            title_parts.append(part.title())
    if not title_parts:
        title_parts = [p.title() for p in parts[1:]]
    title = " ".join(title_parts)
    title = title.replace("Ml ", "ML ").replace(" Ai ", " AI ").replace("Coop", "Co-op")
    return title


def _extract_company(text: str, posting_id: str) -> str:
    """Extract company name from text or filename prefix."""
    first_line = text.split("\n")[0].strip() if text else ""
    for key, name in COMPANY_KNOWN.items():
        if posting_id.startswith(key):
            return name
    if first_line and len(first_line) < 60:
        return first_line
    return posting_id.split("-")[0].title()


def _parse_sections(text: str) -> dict[str, str]:
    """Detect known section headers and split text into labeled sections."""
    lines = text.split("\n")
    sections: dict[str, list[str]] = {}
    current_section = "_preamble"
    sections[current_section] = []

    for line in lines:
        stripped = line.strip()
        matched = None
        for pattern, label in SECTION_PATTERNS:
            if re.search(pattern, stripped):
                matched = label
                break
        if matched:
            current_section = matched
            sections.setdefault(current_section, [])
            sections[current_section].append(stripped)
        else:
            sections.setdefault(current_section, [])
            sections[current_section].append(stripped)

    return {k: "\n".join(v).strip() for k, v in sections.items() if v}


def parse_posting(filepath: Path) -> dict:
    """Parse a single markdown posting into structured metadata.

    Args:
        filepath: Path to a .md file in selected-job-postings/.

    Returns:
        Dict with id, title, company, role_category, sections, raw_full_text.
    """
    text = filepath.read_text(encoding="utf-8").strip()
    posting_id = filepath.stem
    title = _extract_title(text, filepath)
    company = _extract_company(text, posting_id)
    role_category = ROLE_CATEGORIES.get(posting_id, "Other")
    sections = _parse_sections(text)

    return {
        "id": posting_id,
        "title_raw": title,
        "company": company,
        "source_file": filepath.name,
        "role_category": role_category,
        "sections": sections,
        "raw_full_text": text,
    }


def parse_all_postings(postings_dir: Path) -> list[dict]:
    """Parse all .md files in the postings directory."""
    files = sorted(postings_dir.glob("*.md"))
    jobs = []
    for fp in files:
        jobs.append(parse_posting(fp))
    print(f"Parsed {len(jobs)} postings")
    return jobs


def embed_postings(jobs: list[dict], model: SentenceTransformer) -> np.ndarray:
    """Embed the raw_full_text of each posting with L2 normalization.

    Args:
        jobs: List of parsed job dicts.
        model: A SentenceTransformer instance.

    Returns:
        Float32 array of shape (n_jobs, 384) with unit L2 norm per row.
    """
    texts = [job["raw_full_text"] for job in jobs]
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), f"L2 norm check failed: min={norms.min():.6f} max={norms.max():.6f}"
    print(f"Embedded {len(jobs)} postings → shape {embeddings.shape}, L2 norms ∈ [{norms.min():.6f}, {norms.max():.6f}]")
    return embeddings.astype(np.float32)


def _role_color(role: str) -> str:
    return CATEGORY_COLORS.get(role, "#7f7f7f")


def plot_pca(embeddings: np.ndarray, jobs: list[dict], output_dir: Path) -> None:
    """Generate a 2D PCA scatter plot and a scree plot.

    Args:
        embeddings: (n, 384) float32 array.
        jobs: List of job dicts matching the row order.
        output_dir: Directory to write .png files.
    """
    from sklearn.decomposition import PCA

    pca = PCA(n_components=min(embeddings.shape[0], 10), random_state=RANDOM_SEED)
    coords = pca.fit_transform(embeddings)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    ax = axes[0]
    for role in sorted(set(j["role_category"] for j in jobs)):
        mask = np.array([j["role_category"] == role for j in jobs])
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=_role_color(role), label=role, s=80, alpha=0.85, edgecolors="white", linewidth=0.5,
        )
    for i, job in enumerate(jobs):
        label = job["company"].split(" (")[0].split(" ")[0][:8]
        ax.annotate(label, (coords[i, 0], coords[i, 1]), fontsize=7, alpha=0.7,
                    textcoords="offset points", xytext=(4, 3))

    var = pca.explained_variance_ratio_
    ax.set_title(f"PCA — 2 Components (PC1: {var[0]:.1%}, PC2: {var[1]:.1%})")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    components = range(1, len(var) + 1)
    ax2.bar(components, var, color="#1f77b4", alpha=0.7)
    ax2.plot(components, np.cumsum(var), "o-", color="#d62728", linewidth=2, label="Cumulative")
    ax2.set_title("Explained Variance per Component")
    ax2.set_xlabel("Principal Component")
    ax2.set_ylabel("Explained Variance Ratio")
    ax2.set_xticks(components)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "raw_pca.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"PCA plot saved → {output_dir / 'raw_pca.png'}")


def plot_umap(embeddings: np.ndarray, jobs: list[dict], output_dir: Path) -> None:
    """Generate a 2D UMAP scatter plot using cosine distance.

    Args:
        embeddings: (n, 384) float32 array.
        jobs: List of job dicts matching the row order.
        output_dir: Directory to write .png files.
    """
    import umap

    reducer = umap.UMAP(
        n_neighbors=5, min_dist=0.15, metric="cosine",
        random_state=RANDOM_SEED, n_jobs=1,
    )
    coords = reducer.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(12, 9))
    for role in sorted(set(j["role_category"] for j in jobs)):
        mask = np.array([j["role_category"] == role for j in jobs])
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=_role_color(role), label=role, s=100, alpha=0.85, edgecolors="white", linewidth=0.5,
        )
    for i, job in enumerate(jobs):
        label = f"{job['company'].split(' (')[0].split(' ')[0][:8]}\n{job['title_raw'][:20]}"
        ax.annotate(label, (coords[i, 0], coords[i, 1]), fontsize=6, alpha=0.8,
                    textcoords="offset points", xytext=(5, 4))

    ax.set_title("UMAP — Embedding Space (cosine metric, n_neighbors=5)")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(output_dir / "raw_umap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"UMAP plot saved → {output_dir / 'raw_umap.png'}")


def print_summary(jobs: list[dict], embeddings: np.ndarray) -> None:
    """Log a summary of the parsed data and embedding space."""
    from collections import Counter

    role_counts = Counter(j["role_category"] for j in jobs)
    print("\n── Role distribution ──")
    for role, count in role_counts.most_common():
        print(f"  {role}: {count}")

    print("\n── Embedding stats ──")
    norms = np.linalg.norm(embeddings, axis=1)
    sim_matrix = embeddings @ embeddings.T
    np.fill_diagonal(sim_matrix, 0)
    print(f"  Shape:       {embeddings.shape}")
    print(f"  L2 norms:    [{norms.min():.6f}, {norms.max():.6f}]")
    print(f"  Mean pairwise cosine sim: {sim_matrix.mean():.4f}")
    print(f"  Max pairwise cosine sim:  {sim_matrix.max():.4f}")
    print(f"  Min pairwise cosine sim:  {sim_matrix.min():.4f}")

    print("\n── Section coverage ──")
    section_keys = {"about_role", "responsibilities", "qualifications", "nice_to_have", "what_we_offer", "about_team"}
    for key in sorted(section_keys):
        count = sum(1 for j in jobs if j["sections"].get(key, "").strip())
        print(f"  {key}: {count}/{len(jobs)}")

    print("\n── Postings ──")
    for job in jobs:
        title = job["title_raw"][:50]
        print(f"  {job['id'][:45]:45s} | {job['role_category'][:16]:16s} | {title}")


def main() -> None:
    """Parse, embed, visualize all 27 postings."""
    OUTPUT_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("── Step 1: Parse markdown ──")
    jobs = parse_all_postings(POSTINGS_DIR)
    with open(OUTPUT_JOBS, "w", encoding="utf-8") as f:
        json.dump(jobs, f, indent=2, ensure_ascii=False)
    print(f"Saved → {OUTPUT_JOBS}")

    print("\n── Step 2: Embed ──")
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    embeddings = embed_postings(jobs, model)
    np.save(OUTPUT_EMBEDDINGS, embeddings)
    print(f"Saved → {OUTPUT_EMBEDDINGS}")

    print("\n── Step 3: Visualize ──")
    plot_pca(embeddings, jobs, OUTPUT_PLOTS_DIR)
    plot_umap(embeddings, jobs, OUTPUT_PLOTS_DIR)

    print_summary(jobs, embeddings)
    print("\nDone.")


if __name__ == "__main__":
    main()
