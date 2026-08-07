## Context

The project has 6 experiments (001–006) implemented across `scripts/`, 2 planned backend modules (`embeddings.py`, `preprocessing.py`), and an evaluation framework with proxy metrics. Only the taxonomy-labeling capability is specified in OpenSpec. A documentation site (`docs/`) carries narrative descriptions of architecture, pipeline stages, and experiment results — but these are prose, not structured requirements. The goal is to complete the mirror: every stable system behavior documented in `openspec/specs/`, with `docs/` serving as the human-facing narrative layer.

## Goals / Non-Goals

**Goals:**
- Create 5 new capability specs that capture stable, observable system behavior from existing code and documentation
- Each spec SHALL contain requirements with WHEN/THEN scenarios
- Zero code or data changes — purely documenting existing behavior
- Specs serve as the internal living specification; `docs/` becomes the human-readable gallery

**Non-Goals:**
- Do not modify experiment scripts, backend modules, or data files
- Do not change the mkdocs site or CROSS-REFERENCE-INDEX.md (that's a follow-up change)
- Do not specify planned/future capabilities (FastAPI, Docker, IR metrics, augmentation) — only what exists and runs today
- Do not duplicate content from `docs/architecture.md` — specs describe *what*, not *why*

## Decisions

**Decision 1: One change covers all 5 specs**
Rather than 5 separate changes (one per capability), this single change creates all 5 spec deltas. Rationale: the capabilities are interdependent (evaluation references embedding; semantic-partitioning references LLM extraction), and reviewing them together ensures consistency. The task list breaks them down independently so each can be implemented and verified in isolation.

**Decision 2: Specs derived from existing code behavior, not documentation prose**
Each spec's requirements are validated against the actual script implementations (e.g., `bootstrap.py` for data-ingestion, `extract_clean_text.py` for llm-text-extraction, `compare_variants.py` for evaluation). The `docs/` pages are secondary sources — used for context, not as spec templates.

**Decision 3: Research artifacts get specs with explicit status annotations**
`semantic-partitioning` is a research artifact (not in production pipeline), but it produces stable output format and metrics. Its spec includes a note that it is not yet integrated into the production pipeline, making its status explicit without making it invisible.

**Decision 4: Metrics are specified as computation contracts, not as target values**
The evaluation framework spec defines *what* self-retrieval and separation gap compute (the algorithm), not *what values* are acceptable. This keeps specs stable even as metric targets evolve — the contract is the computation, not the threshold.

Alternative considered: Deferring the evaluation spec until IR metrics (Precision@K, MRR, NDCG via pytrec_eval) are finalized. Rejected because the proxy metrics are stable, implemented, and producing numbers — deferring would leave a gap when they're the only metrics currently in use.

## Risks / Trade-offs

- **Spec drift**: If the experiment scripts change behavior without corresponding spec updates, the mirror becomes stale. Mitigation: future changes that modify pipeline behavior create spec deltas as part of the change (the normal OpenSpec workflow).
- **Specification of research artifacts**: Semantic partitioning and evaluation are still evolving. Specifying them now risks capturing behavior that changes. Mitigation: research-artifact specs are minimal — only the stable output contract and core algorithm, not implementation details. Status annotated as research when applicable.
- **Over-specification**: Specs that describe too much implementation detail become brittle. Mitigation: the config.yaml spec rules enforce scenarios describing observable behavior, not implementation.
