"""Function taxonomy for Experiment 012 weak labels.

Copied from ``scripts/006_taxonomy_labels/generate_labels.py`` so the numbered
Experiment 006 artifact stays immutable. ``other`` is added for postings that
do not fit the ML/AI taxonomy.
"""

from __future__ import annotations

TAXONOMY_DEFINITIONS = """## 1. agentic-ai — Autonomous Agent Systems
Builds systems that reason, plan, and act autonomously. Core activities:
Multi-agent orchestration (LangGraph, CrewAI, AutoGen), tool use and function \
calling (MCP protocol), stateful workflow design, guardrail implementation, \
agent evaluation, memory and state management.

## 2. llm-fine-tuning — Model Adaptation
Changes how a model behaves by adjusting its weights. Core activities:
Parameter-efficient fine-tuning (LoRA, QLoRA), supervised fine-tuning (SFT) \
on curated datasets, preference optimization (DPO, KTO, ORPO), model \
distillation into smaller models (SLMs), dataset curation.

## 3. llm-information-retrieval — Search & Retrieval with LLMs
Designs systems that find the right content and ground LLM outputs in it. \
Core activities: RAG pipeline design (chunking, embedding, indexing, \
re-ranking), hybrid search (BM25 + dense), vector database optimization, \
retrieval evaluation (MRR, NDCG, recall@k), query understanding, \
context window budgeting.

## 4. classical-ml — Traditional Predictive Modeling
Trains models on structured/tabular data to predict or classify. Core \
activities: Feature engineering on tabular data, XGBoost/LightGBM/random \
forests/linear regression, hyperparameter tuning, cross-validation, \
handling class imbalance and missing data, statistical modeling \
(time-series, survival analysis).

## 5. mlops-production — Production ML Operations
Puts models into production and keeps them running reliably. Core activities: \
CI/CD pipelines for ML, model serving (Docker, Kubernetes, Triton), \
monitoring (data drift, concept drift, latency SLAs), retraining automation, \
model versioning and registry (MLflow), deployment strategies \
(canary, blue-green, A/B testing).

## 6. data-engineering — Data Infrastructure
Builds the pipes that feed everything else. Core activities: ETL/ELT pipeline \
design, data warehousing/lakehouse (Snowflake, Databricks, BigQuery), \
orchestration (Airflow, dbt, Dagster), batch/streaming processing \
(Spark, Kafka), data quality frameworks, SQL optimization.

## 7. analytics-storytelling — Insights & Communication
Turns data into decisions humans can act on. Core activities: Exploratory \
Data Analysis, A/B test design and analysis, dashboard development \
(Tableau, Power BI), stakeholder presentations, metric definition and \
KPI tracking, decision-support analysis.

## 8. computer-vision — Visual Understanding Systems
Builds systems that interpret images and video. Core activities: Image \
classification, object detection (YOLO), OCR and document understanding, \
visual-language models (CLIP), camera systems, video analysis, \
medical imaging/satellite imagery.

## 9. ml-platform — Shared ML Infrastructure
Builds the tools and platforms that other ML teams use. Core activities: \
Feature store design, model registry infrastructure, experiment tracking \
platforms, training orchestration (Kubeflow), shared inference platforms, \
developer experience tooling.

## 10. reinforcement-learning — Sequential Decision Making
Trains agents to learn optimal behavior through trial, error, and reward. \
Core activities: RL algorithm implementation (PPO, SAC, DQN), environment \
simulation (Gymnasium, MuJoCo), reward function engineering, RLHF \
infrastructure, distributed RL training.

## 11. research — Novel Investigation
Explores the unknown. May never ship to production. Core activities: \
Reading and reproducing academic papers, novel algorithm design, \
prototyping cutting-edge methods, publishing at conferences, frontier \
model exploration.

## 12. ai-safety-governance — Responsible AI
Ensures AI systems are safe, fair, and auditable. Core activities: Bias \
detection and fairness auditing, explainability (SHAP, LIME), model cards \
and system cards, adversarial robustness testing, AI compliance \
(EU AI Act), privacy-preserving ML.

## 13. deep-learning — Neural Network Engineering
Designs and trains neural network architectures. Core activities: Model \
architecture design (MLPs, CNNs, RNNs, transformers, GANs, VAEs), training \
loop engineering (PyTorch, TensorFlow, JAX), hyperparameter tuning and \
experiment tracking, GPU optimization and distributed training, transfer \
learning and pre-training from foundation models.

## 14. other — Outside the ML/AI Function Taxonomy
Use only when none of the categories above fits the primary function of the \
posting. Examples: generic IT support, sales, finance, HR, or a role whose \
actual duties are not described."""

VALID_FUNCTIONS = (
    "agentic-ai",
    "llm-fine-tuning",
    "llm-information-retrieval",
    "classical-ml",
    "mlops-production",
    "data-engineering",
    "analytics-storytelling",
    "computer-vision",
    "ml-platform",
    "reinforcement-learning",
    "research",
    "ai-safety-governance",
    "deep-learning",
    "other",
)

SYSTEM_PROMPT = f"""You are a job posting classifier. Analyze the provided job \
posting and classify it according to the 14-category function taxonomy below.

{TAXONOMY_DEFINITIONS}

Assign primary_function: exactly one category name from the 14 categories \
above. Choose the category that best represents the PRIMARY function of the \
role, not every activity mentioned.

Assign function_tags: a list of 0-3 category names selected STRICTLY from the \
14 categories above. Include all that clearly match. If primary_function is \
"other", function_tags must be empty.

Assign label_confidence: one of "high", "medium", or "low". Use "low" when the \
posting is too short, ambiguous, non-English, or does not describe the work.

Assign rationale: a 1-2 sentence explanation referencing specific duties from \
the posting.

Output a JSON object with exactly four fields:
- primary_function: string
- function_tags: list of strings
- label_confidence: string
- rationale: string

Output ONLY the JSON object. No markdown, no explanation, no code fences."""

USER_MESSAGE_TEMPLATE = "Title: {title}\n\nCleaned posting:\n\n{text}"
