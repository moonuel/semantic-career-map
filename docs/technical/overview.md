# Technical Details

Implementation notes for the production pipeline and the experiment code behind it. The architecture pages describe **what the system is**; this section records **how the stages work** and which invariants they enforce.

## Current Stack

| Layer | Implementation |
|---|---|
| Ingestion | Python scrapers + ATS registry; Lambda container for ATS boards, Fargate task for the dataroles curator |
| Raw storage | S3 canonical JSON, content-addressed by SHA-256 of the canonical payload |
| Query | AWS Glue Data Catalog + Athena; 256 MiB per-query scan cap |
| Cleaning | Gemini 2.5 Flash Lite with the exp-008 boilerplate-only prompt; buffered Parquet parts and checkpoint-first cache |
| Evaluation | Exp-003 metrics over a 39-posting golden set; manual promotion gate |
| Embedding | Amazon Titan Text Embeddings V2 via Bedrock, 1,024-d, L2-normalized; MiniLM 384-d rollback |
| Projection | UMAP cosine (`n_neighbors=15`, `min_dist=0.1`, seed 42); optional PCA pre-reduction |
| Visualization | Deck.GL / DataMapPlot 2D map, 3D coordinates, lazy detail buckets |
| Deployment | ECR, Lambda, ECS Fargate, EventBridge Scheduler, SNS, SQS, SSM; imperative boto3 provisioners |
| CI | GitHub Actions: `ruff` + `pytest` on Python 3.14; Zensical docs build + Pages deploy |

## Invariants

- **Raw identity is content identity.** `source_hash` is the canonical payload hash and the join handle across raw, cleaned, and embedding rows. No posting ids are minted.
- **Configurations are hashed, not named.** `configuration_hash` captures model, prompt, and processing parameters; it is a row column, never a path segment.
- **A stable embedding prefix has one owner.** Version claims prevent a second configuration from joining `version=v1`.
- **Vectors are unit length.** Both encoders enforce the 1e-5 normalization postcondition and fail the batch rather than write a bad row.
- **Durable writes precede checkpoints.** Cleaning and embedding parts commit before their flush checkpoint; reconciliation back-fills missing checkpoints and removes duplicate rows.
- **Selection is an anti-join.** Scheduled cleaning and embedding process only missing rows; re-running a stage is a no-op.

## Page Guide

- [Embedding Pipeline](embedding-pipeline.md) — Titan and MiniLM encoders, normalization, versioning, and planned feature engineering.
- [Data Processing](data-processing.md) — the experiment corpus, cleaning prompts, semantic partitioning, and quality metrics.
- [Evaluation](evaluation.md) — proxy metrics, golden set, IR metrics plan, and limitations.
- [Technology Choices](tech-choices.md) — model/storage/retrieval alternatives and the optimization ladder.
- [Favourite Resources](resources.md) — the sources applied in the design.

## Running the Pipeline

```bash
# API/CLI help
python -m backend.pipeline.cleaning --help
python -m backend.pipeline.eval --help
python scripts/embed.py --help

# Local map (full corpus or bounded smoke)
python -m scripts.map
python -m scripts.map --limit 1000

# Query-layer provisioning and gate
python -m scripts.provision_query_layer
python scripts/provision_query_layer/gate.py --embeddings
```

Runtime/redeploy commands and AWS specifics live in the [deploy README](https://github.com/moonuel/semantic-career-map/blob/main/deploy/README.md).