# ML/AI Responsibility Taxonomy — Research Report

**Date:** 2026-07-22
**Purpose:** Inform function-based role labeling for the Semantic Career Map clustering
evaluation pipeline (Phase 1, Step 1.2c)

---

## 1. Executive Summary

Job titles in the ML/AI field — Data Scientist, ML Engineer, AI Engineer — are a noisy
signal for what someone actually does. A "Data Scientist" at Coca-Cola building demand
forecasting models and a "Data Scientist" at Scribd building NLP/GenAI systems share a
title but almost nothing else in their daily work. For clustering evaluation, we need
labels that reflect *function* — what the job actually does — not what it's called.

This report synthesizes 2025–2026 job market research, career comparison guides, real
job postings, and engineering practice literature to produce a 12-category function
taxonomy suitable for labeling 27 hand-collected job postings.

**Key finding:** The industry has converged on a three-way conceptual split (Data
Scientists answer questions, ML Engineers build systems, AI Engineers ship products),
but real-world postings cross these boundaries aggressively. A finer-grained,
domain-specific taxonomy is needed for honest clustering evaluation.

---

## 2. Industry Context: The Title Problem

### 2.1 The Convergent Three-Way Split

Multiple 2026 career guides and market analyses have converged on the same framing:

> **Data Scientists answer questions.** They analyze data, run experiments, and
> communicate insights.
>
> **ML Engineers build systems.** They train custom models, deploy to production,
> and monitor pipelines.
>
> **AI Engineers ship products.** They wire LLMs into features, build agents,
> and design prompts.

Sources:
- [Let's Data Science: Data Scientist vs ML Engineer vs AI Engineer in
  2026](https://letsdatascience.com/blog/data-scientist-vs-ml-engineer-vs-ai-engineer-in-2026)
- [AI Agents Kit: AI Engineer vs ML Engineer vs Data Scientist Which
  Career](https://aiagentskit.com/blog/ai-engineer-vs-ml-engineer-vs-data-scientist)
- [Nucamp: AI Engineer vs ML Engineer vs Data Scientist in 2026: What's the
  Difference](https://www.nucamp.co/blog/ai-engineer-vs-ml-engineer-vs-data-scientist-in-2026-what-s-the-difference)
- [JoinAI: AI Engineer vs ML Engineer vs Data Scientist — What's the
  Difference?](https://www.joinai.com/blog/ai-engineer-vs-ml-engineer-vs-data-scientist)

### 2.2 The Growth of AI Engineering

AI Engineering was LinkedIn's fastest-growing job role in the US for 2026, with
AI/ML engineering role postings growing **163% from 2024 to 2025** ([Axial Search
analysis](https://www.acceler8talent.com/resources/blog/the-most-in-demand-machine-learning-roles-in-2026--managing-the-ai-talent-frontier)).
"AI Engineer" barely existed as a distinct title before 2023. It now represents a
genuinely different craft: API composition, prompt design, RAG pipelines, agent
orchestration — not model training.

### 2.3 AI Is Absorbed Into Traditional Titles

Despite the rise of AI-specific roles, the majority of AI-related work still carries
traditional titles. [Grow Data Skills' analysis of 700+ job postings in
2026](https://learn.growdataskills.com/blog/AI_and_the_Data_Scientist_Job_Market_in_2026_-_What_700%2B_Job_Postings_Reveal_About_Skills%2C_Salaries%2C_and_Seniority)
found that roughly one-third of AI-related postings carry titles like "Data Scientist"
or "Senior Data Scientist," with AI expectations integrated into the standard
definition rather than creating a separate category.

This validates our approach: **title-based categories are insufficient for clustering
evaluation.**

### 2.4 Domain Experts vs Versatile Professionals

[365 Data Science's 2026 ML Engineer
research](https://365datascience.com/career-advice/career-guides/machine-learning-engineer-job-outlook-2025)
introduced a Job Posting Specialization Score (JPSS): 58% of ML engineer postings seek
Domain Experts (1–4 skill categories), while 42% seek Versatile Professionals (5–8
skill categories). Larger companies tend toward specialists; smaller companies toward
generalists. This means some postings will genuinely span 2–3 function categories.

---

## 3. Function Category Taxonomy

Each category defines a distinct craft — a set of tools, deliverables, and problems
that differ meaningfully from the others. A posting may carry 1–2 function tags.

### 3.1 `agentic-ai` — Autonomous Agent Systems

**Builds systems that reason, plan, and act autonomously.**

Core activities:
- Multi-agent orchestration (LangGraph, CrewAI, AutoGen, Microsoft Agent Framework)
- Tool use and function calling (MCP protocol, A2A inter-agent communication)
- Stateful workflow design: routing, parallelization, branching, error recovery
- Guardrail implementation: circuit breakers, action limits, policy enforcement
- Agent evaluation: task completion rate, hallucination rate, safety audits
- Memory and state management across multi-turn interactions

Deliverable: A running agent system that makes decisions and takes actions. Not a
model. Not a retrieval pipeline. A decision-making loop.

Distinction from RAG: RAG retrieves documents; agents *decide* what to do (including
whether to retrieve). An agentic system may *use* RAG as a tool, but the craft is
orchestration and decision logic.

Sources:
- [Domino.ai: RAG vs Agentic AI: Why Agents Are More Than
  Workflow](https://domino.ai/blog/rag-vs-agentic-ai)
- [AgentMarketCap: LangGraph vs CrewAI vs AutoGen: The 2026 Enterprise Framework
  Split](https://agentmarketcap.ai/blog/2026/04/07/langgraph-crewai-autogen-enterprise-framework-adoption-split)
- [Accenture: Lead Agentic AI Engineer job
  posting](https://www.accenture.com/us-en/careers/jobdetails?id=R00335391_en)
- [Benchling: Agentic AI Engineer job
  posting](https://agentic-engineering-jobs.com/jobs/benchling-agentic-ai-engineer-c0wwUC)

### 3.2 `llm-fine-tuning` — Model Adaptation

**Changes how a model behaves by adjusting its weights.**

Core activities:
- Parameter-efficient fine-tuning: LoRA, QLoRA
- Supervised Fine-Tuning (SFT) on curated instruction/example datasets
- Preference optimization: DPO, KTO, ORPO (modern alternatives to RLHF)
- Model distillation into smaller, task-specific models (SLMs)
- Dataset curation and quality control for fine-tuning corpora

Deliverable: A trained/adapted model artifact. Not a system around the model — the
model itself has changed.

Distinction from agentic-ai: Agent engineering uses models as-is via APIs.
Fine-tuning modifies the model's behavior/format/voice at the weight level. "Fine-tuning
teaches behavior, not facts" — if you want the model to know your pricing page,
retrieve it; if you want the model to sound like your brand, that's where fine-tuning
earns its keep.

Sources:
- [Metacto: RAG vs. Fine-Tuning vs. Other LLM Techniques: A 2026 Decision
  Guide](https://www.metacto.com/blogs/rag-vs-fine-tuning-vs-other-llm-techniques-choosing-the-right-approach)
- [Toloka AI: The distinction between RAG and
  fine-tuning](https://toloka.ai/blog/the-distinction-between-rag-and-fine-tuning)

### 3.3 `llm-information-retrieval` — Search & Retrieval with LLMs

**Designs systems that find the right content and ground LLM outputs in it.**

Core activities:
- RAG pipeline design: chunking strategies → embedding model selection → indexing →
  retrieval → re-ranking
- Hybrid search architecture: BM25 (lexical) + dense embeddings (semantic)
- Vector database selection and optimization (Pinecone, pgvector, Qdrant, Weaviate)
- Retrieval evaluation: MRR, NDCG, recall@k, precision@k
- Query understanding and reformulation
- Context window budgeting: what to include, how to compact, how to assemble
- GraphRAG for structured knowledge retrieval

Deliverable: A retrieval system that consistently returns relevant, faithful context
for LLM generation. The craft is search quality, not decision-making.

Distinction from agentic-ai: RAG engineers optimize *what documents the system sees*.
Agentic engineers optimize *what the system decides to do*. A RAG pipeline can be
building material for an agent, but the skill set (chunking, embedding, retrieval
evaluation) is distinct from orchestration and decision logic.

Sources:
- [n1n.ai: RAG Is Not Machine Learning: Moving Beyond the ML Toolkit for Enterprise
  Document Intelligence](https://explore.n1n.ai/blog/rag-is-not-machine-learning-enterprise-document-intelligence-2026-06-02)
- [EPAM: Lead AI Engineer, Agentic and RAG Systems job
  posting](https://careers.epam.com/en/vacancy/lead-ai-engineer-agentic-and-rag-systems-blt0gwqdcd06g43wvh3_en)

### 3.4 `classical-ml` — Traditional Predictive Modeling

**Trains models on structured/tabular data to predict or classify.**

Core activities:
- Feature engineering on tabular data: transformations, encodings, interactions
- Model training: XGBoost, LightGBM, random forests, linear/logistic regression, SVMs
- Hyperparameter tuning: grid search, Bayesian optimization
- Cross-validation, model selection, evaluation metrics (AUC, F1, RMSE)
- Handling class imbalance, missing data, outliers in structured datasets
- Statistical modeling: time-series, survival analysis, causal inference (classical)

Deliverable: A trained predictive model. The "scikit-learn flowchart" kind of work.

Distinction from deep learning: Classical ML works with structured rows-and-columns
data, often with interpretable models and feature-level reasoning. The skills
(statistics, feature engineering on tables, experiment design with structured data)
are distinct from neural network design.

### 3.5 `mlops-production` — Production ML Operations

**Puts models into production and keeps them running reliably.**

Core activities:
- CI/CD pipelines for ML: automated training, evaluation gates, staged rollouts
- Model serving infrastructure: Docker, Kubernetes, Triton, TorchServe, vLLM
- Monitoring: data drift, concept drift, prediction drift, latency SLAs, throughput
- Retraining automation: scheduled and trigger-based retraining pipelines
- Model versioning and registry: MLflow, Weights & Biases, DVC
- Deployment strategies: canary releases, blue-green, shadow deployments, A/B testing
  infrastructure
- Incident response for ML systems, on-call rotation

Deliverable: Reliable production ML systems. Not a trained model — the system that
makes the model *usable at scale with guarantees.*

Distinction from ML platform engineering: MLOps operates *specific models in
production*. ML Platform builds *shared infrastructure* that MLOps teams use (see §3.9).

Sources:
- [TensorBlue: MLOps Best Practices 2025: CI/CD & Model
  Monitoring](https://tensorblue.com/blog/mlops-best-practices-cicd-model-monitoring-production-deployment-2025)
- [rework: MLOps Engineer Job Description
  Template](https://resources.rework.com/libraries/job-description-templates/mlops-engineer)
- [Whileresume: AI ML Engineer Job Description: Roles, Skills &
  Responsibilities](https://whileresume.com/article/ai-ml-engineer-job-description)

### 3.6 `data-engineering` — Data Infrastructure

**Builds the pipes that feed everything else.**

Core activities:
- ETL/ELT pipeline design and maintenance
- Data warehousing and lakehouse architecture (Snowflake, Databricks, BigQuery)
- Orchestration: Apache Airflow, dbt, Dagster, Prefect
- Batch and streaming processing: Spark, Kafka, Flink, Kinesis
- Data quality frameworks: schema enforcement, drift detection, data contracts
- SQL optimization, data modeling (star schema, dimensional modeling)
- Data governance, cataloging, lineage tracking

Deliverable: Reliable, clean data at scale. No models. No dashboards. Foundation layer
that everything else depends on.

### 3.7 `analytics-storytelling` — Insights & Communication

**Turns data into decisions humans can act on.**

Core activities:
- Exploratory Data Analysis (EDA): descriptive statistics, pattern identification,
  hypothesis generation
- A/B test design, execution, and analysis
- Dashboard development: Tableau, Power BI, Looker, Metabase
- Stakeholder presentations: translating complex findings into clear narratives
- Metric definition and KPI tracking
- Decision-support analysis: "What should we do about this?"
- Causal inference for business decisions (diff-in-diff, instrumental variables when
  experimentation isn't possible)

Deliverable: A recommendation, slide deck, or dashboard. Not a model. Not a pipeline.
The output is *human understanding* that drives a business decision.

Distinction from classical ML: EDA is investigative (describing what happened), not
predictive (guessing what will happen). The analytics engineer's deliverable is a
narrative; the ML engineer's deliverable is a model artifact.

### 3.8 `computer-vision` — Visual Understanding Systems

**Δ New from research.** Builds systems that interpret images and video.

Core activities:
- Image classification, object detection (YOLO, Faster R-CNN), semantic segmentation
- Optical Character Recognition (OCR) and document understanding
- Visual-language models (VLMs): CLIP, GLM-OCR, DeepSeek-OCR, dots.ocr
- Camera systems: calibration, multi-view geometry, depth estimation
- Video analysis: action recognition, object tracking, temporal modeling
- On-device optimization for embedded vision systems
- Medical imaging, satellite imagery, autonomous driving perception
- Image augmentation, synthetic data generation for training

Deliverable: A visual understanding system. Completely different data modality
(pixels) from text/tabular ML.

Distinction: CV engineering uses different frameworks (OpenCV, specific CNN/ViT
architectures), different data formats (images/video), and different evaluation
paradigms (IoU, mAP, precision/recall curves). A CV engineer should not cluster with
a classical ML engineer doing XGBoost.

Sources:
- [InterviewGuy: Computer Vision Engineer Job Description
  2026](https://interviewguy.com/computer-vision-engineer-job-description/)
- [InterviewKickstart: Computer Vision Engineer Job Description: Skills & Pay
  2026](https://interviewkickstart.com/job-description/computer-vision-engineer)
- [Slava Dubrov: The Definitive Guide to OCR in
  2026](https://slavadubrov.github.io/blog/2026/03/04/ocr-guide)

### 3.9 `ml-platform` — Shared ML Infrastructure

**Δ New from research.** Builds the tools and platforms that other ML teams use.

Core activities:
- Feature store design and operation: Feast, Tecton, or custom — ensuring point-in-time
  correctness, online/offline consistency
- Model registry infrastructure: version tracking, stage promotion (staging →
  production → archived), alias management
- Experiment tracking platforms: MLflow, Weights & Biases, Neptune.ai setup and
  maintenance
- Training orchestration: Kubeflow, Metaflow, distributed GPU cluster scheduling
- Shared inference platforms: multi-tenant model serving, GPU autoscaling, cost
  allocation
- Developer experience: turning notebook experiments into reusable components,
  documentation, templates

Deliverable: A platform that makes other ML teams more productive. Not any specific
model — the factory that builds the models.

Distinction from MLOps:
- MLOps: "I deploy our fraud detection model. I monitor its drift. I'm on call if it
  breaks."
- ML Platform: "I built the feature store and model registry that 15 teams use to
  deploy their models. If the registry goes down, all 15 teams are blocked."

Sources:
- [JobDescription.org: ML Platform Engineer Job Description, Salary & Career
  Outlook](https://jobdescription.org/jobs/artificial-intelligence/ml-platform-engineer)
- [Apple: ML Platform Engineer job
  posting](https://jobs.apple.com/en-us/details/200653150-3956/ml-platform-engineer)
- [rework MLOps guide](https://resources.rework.com/libraries/job-description-templates/mlops-engineer)
  (distinguishes MLOps from ML Platform)
- [ML Academy: Feature Store, Experiment Tracking & Model
  Registry](https://www.mlacademy.ai/articles/free-mlops-course-feature-store-model-registry-and-experiment-tracking)

### 3.10 `reinforcement-learning` — Sequential Decision Making

**Δ New from research.** Trains agents to learn optimal behavior through trial,
error, and reward.

Core activities:
- RL algorithm implementation: PPO, SAC, TD3, DQN, actor-critic variants
- Environment simulation: Gymnasium, MuJoCo, custom simulators
- Reward function engineering: shaping strategies, multi-objective rewards, sparse
  reward handling
- Offline RL and imitation learning for domains where exploration is unsafe
- Distributed RL training: experience collection, replay buffers, GPU-accelerated
  training
- RLHF infrastructure: PPO-based policy optimization for LLM alignment, GRPO, reward
  model training

Deliverable: A trained policy — a model that learns from consequences, not labeled data.

Distinction: RL operates in a fundamentally different paradigm from
supervised/unsupervised ML. No labeled dataset — just reward signals from an
environment. The engineer designs reward functions and training environments, not
features and labels. This is a genuinely distinct craft even within ML engineering.

Real-world applications in 2026: Tesla Optimus robot locomotion/manipulation, Tesla
Full Self-Driving planning, Anthropic Claude RLHF training infrastructure, DoorDash
delivery optimization, CoreWeave RL training pipelines.

Sources:
- [Tesla: Reinforcement Learning Engineer, Policy,
  Optimus](https://www.tesla.com/careers/search/job/reinforcement-learning-engineer-policy-optimus-222416)
- [Tesla: Reinforcement Learning Engineer,
  Self-Driving](https://www.tesla.com/careers/search/job/reinforcement-learning-engineer-self-driving-221945)
- [Anthropic: ML Systems Engineer, RL
  Engineering](https://jobs.menlovc.com/companies/anthropic/jobs/69674504-machine-learning-systems-engineer-rl-engineering)
- [RLHF Engineering Implementation
  Guide](https://ajing.github.io/posts/2025-12-31-rlhf-engineering-implementation)
  (PPO, GRPO, DPO from an engineering perspective)

### 3.11 `research` — Novel Investigation

**Explores the unknown. May never ship to production.**

Core activities:
- Reading, reproducing, and extending academic papers
- Novel algorithm and architecture design
- Prototyping cutting-edge methods without production constraints
- Publishing at conferences and journals
- Theoretical investigation and formal analysis
- Frontier model exploration
- Performance optimization research (attention variants, quantization strategies,
  novel training techniques)

Deliverable: Knowledge, prototypes, papers. High uncertainty. Low probability of
direct production impact.

Distinction from applied roles: The researcher's output is *understanding*, not
products. They work on problems that may not have known solutions and may take months
or years to bear fruit. This is not the same as an ML engineer doing experimentation
in service of a product goal.

### 3.12 `ai-safety-governance` — Responsible AI

**Δ New from research.** Ensures AI systems are safe, fair, and auditable.

Core activities:
- Bias detection and fairness auditing across model outputs
- Explainability: SHAP, LIME, integrated gradients, attention visualization
- Model cards and system cards: documentation of intended use, limitations, and
  evaluation results
- Adversarial robustness testing and red-teaming
- AI compliance: regulatory alignment (EU AI Act, US executive orders)
- Privacy-preserving ML: differential privacy, federated learning, data minimization
- Ethical review of AI systems before deployment

Deliverable: Guardrails and assurance documentation. The craft is evaluation
methodology and governance, not model building.

Distinction from agentic guardrails: Agentic guardrails are *runtime* (action limits,
circuit breakers). AI safety is *evaluation and methodology* — testing for bias before
deployment, not preventing bad actions during operation.

This category is growing rapidly (per LinkedIn and WEF job market reports) but remains
rare as a primary role outside frontier labs (Anthropic, OpenAI, Google DeepMind) and
heavily regulated industries (finance, healthcare). In most organizations, these
responsibilities are absorbed into ML engineering or research roles.

Sources:
- [365 Data Science: AI Engineer Job Outlook 2026
  research](https://365datascience.com/career-advice/career-guides/machine-learning-engineer-job-outlook-2025)
  — notes AI trainers, ethicists, and explainability experts as emerging roles
- [NU.edu: 59 AI Job Statistics: Future of U.S.
  Jobs](https://www.nu.edu/blog/ai-job-statistics) — notes AI safety/ethics as growing
  category

---

## 4. Taxonomy Summary Table

| # | Category | Domain | Ships | Distinct from |
|---|---|---|---|---|
| 1 | `agentic-ai` | LLM/AI | Decision-making systems | RAG, fine-tuning |
| 2 | `llm-fine-tuning` | LLM/AI | Adapted model weights | Agentic systems |
| 3 | `llm-information-retrieval` | LLM/AI | Retrieval quality | Agentic decisions |
| 4 | `classical-ml` | Classical ML | Predictive models (tabular) | Deep learning, analytics |
| 5 | `mlops-production` | ML Ops | Reliable production ML | ML Platform (builds for others) |
| 6 | `data-engineering` | Data | Clean data at scale | All model/insight work |
| 7 | `analytics-storytelling` | Analytics | Insights & recommendations | Model training |
| 8 | `computer-vision` | CV | Visual understanding | NLP, tabular ML |
| 9 | `ml-platform` | ML Infra | Shared tools for ML teams | MLOps (per-model ops) |
| 10 | `reinforcement-learning` | RL | Policies that learn from rewards | Supervised/unsupervised ML |
| 11 | `research` | Research | Knowledge & prototypes | Applied product work |
| 12 | `ai-safety-governance` | Safety | Guardrails & assurance | Runtime guardrails |

---

## 5. Practical Notes for Labeling 27 Postings

### 5.1 Expected Frequency

| Category | Expected count | Confidence |
|---|---|---|
| `classical-ml` | 5–7 | High — banks, insurance, Thumbtack monetization |
| `analytics-storytelling` | 4–5 | High — Coca-Cola, Thumbtack, some "Other" |
| `mlops-production` | 3–4 | Medium — Affirm, JPMorgan, Stripe |
| `agentic-ai` | 2–3 | Medium — Clio, possibly Scribd, JPMorgan AI |
| `data-engineering` | 1–2 | Medium — Tactable DE, pipeline-heavy roles |
| `llm-information-retrieval` | 1–2 | Medium — RAG/search-specific roles |
| `llm-fine-tuning` | 1–2 | Low — roles explicitly mentioning tuning/distillation |
| `computer-vision` | 0–1 | Low — unlikely in our white-collar posting set |
| `research` | 0–1 | Low — eBay Applied Researcher may qualify |
| `ml-platform` | 0–1 | Low — needs explicit "platform"/"infrastructure" language |
| `reinforcement-learning` | 0–1 | Very low — almost certainly absent |
| `ai-safety-governance` | 0–1 | Very low — rare as primary role |

### 5.2 Multi-Label Rule

A single posting may carry 1–2 function tags. Guidelines:

- **1 tag:** The posting is clearly dominated by one function (e.g., pure analytics,
  pure DE, pure CV).
- **2 tags:** The posting blends two functions meaningfully (e.g., "train model AND
  deploy to production" → `classical-ml` + `mlops-production`; "build RAG pipeline AND
  orchestrate agents" → `llm-information-retrieval` + `agentic-ai`).

### 5.3 Clustering Evaluation Strategy

With 27 postings and 12 possible labels (many sparse), the evaluation approach:

1. **Separation gap:** Compute for each label that has ≥2 postings. The `function`
   separation gap is the primary metric — same-function postings should be closer
   together in embedding space than cross-function postings.

2. **HDBSCAN cross-tabulation:** Compare unsupervised cluster assignments against
   function labels. A cluster containing only `classical-ml` postings is a strong
   signal. A cluster mixing `classical-ml` and `computer-vision` is a failure.

3. **Nearest-neighbor audit:** For each posting, check whether its k-nearest neighbors
   share at least one function tag. This is more robust than title-based audit since
   "Data Scientist" neighbors could legitimately span multiple functions.

---

## 6. References

1. [Let's Data Science: Data Scientist vs ML Engineer vs AI Engineer in
   2026](https://letsdatascience.com/blog/data-scientist-vs-ml-engineer-vs-ai-engineer-in-2026)
2. [AI Agents Kit: AI Engineer vs ML Engineer vs Data Scientist Which
   Career](https://aiagentskit.com/blog/ai-engineer-vs-ml-engineer-vs-data-scientist)
3. [Nucamp: AI Engineer vs ML Engineer vs Data Scientist in
   2026](https://www.nucamp.co/blog/ai-engineer-vs-ml-engineer-vs-data-scientist-in-2026-what-s-the-difference)
4. [JoinAI: AI Engineer vs ML Engineer vs Data
   Scientist](https://www.joinai.com/blog/ai-engineer-vs-ml-engineer-vs-data-scientist)
5. [Acceler8 Talent: The Most In-Demand ML Roles in
   2026](https://www.acceler8talent.com/resources/blog/the-most-in-demand-machine-learning-roles-in-2026--managing-the-ai-talent-frontier)
6. [Grow Data Skills: AI and the Data Scientist Job Market in 2026 — 700+ Job
   Postings](https://learn.growdataskills.com/blog/AI_and_the_Data_Scientist_Job_Market_in_2026_-_What_700%2B_Job_Postings_Reveal_About_Skills%2C_Salaries%2C_and_Seniority)
7. [365 Data Science: ML Engineer Job Outlook
   2026](https://365datascience.com/career-advice/career-guides/machine-learning-engineer-job-outlook-2025)
8. [Domino.ai: RAG vs Agentic AI: Why Agents Are More Than
   Workflow](https://domino.ai/blog/rag-vs-agentic-ai)
9. [AgentMarketCap: LangGraph vs CrewAI vs AutoGen: The 2026 Enterprise Framework
   Split](https://agentmarketcap.ai/blog/2026/04/07/langgraph-crewai-autogen-enterprise-framework-adoption-split)
10. [Metacto: RAG vs. Fine-Tuning vs. Other LLM Techniques: A 2026 Decision
    Guide](https://www.metacto.com/blogs/rag-vs-fine-tuning-vs-other-llm-techniques-choosing-the-right-approach)
11. [Toloka AI: The distinction between RAG and
    fine-tuning](https://toloka.ai/blog/the-distinction-between-rag-and-fine-tuning)
12. [n1n.ai: RAG Is Not Machine Learning: Enterprise Document
    Intelligence](https://explore.n1n.ai/blog/rag-is-not-machine-learning-enterprise-document-intelligence-2026-06-02)
13. [TensorBlue: MLOps Best Practices 2025: CI/CD & Model
    Monitoring](https://tensorblue.com/blog/mlops-best-practices-cicd-model-monitoring-production-deployment-2025)
14. [rework: MLOps Engineer Job Description
    Template](https://resources.rework.com/libraries/job-description-templates/mlops-engineer)
15. [Whileresume: AI ML Engineer Job
    Description](https://whileresume.com/article/ai-ml-engineer-job-description)
16. [InterviewGuy: Computer Vision Engineer Job Description
    2026](https://interviewguy.com/computer-vision-engineer-job-description/)
17. [InterviewKickstart: Computer Vision Engineer Job Description: Skills & Pay
    2026](https://interviewkickstart.com/job-description/computer-vision-engineer)
18. [Slava Dubrov: The Definitive Guide to OCR in
    2026](https://slavadubrov.github.io/blog/2026/03/04/ocr-guide)
19. [JobDescription.org: ML Platform Engineer Job
    Description](https://jobdescription.org/jobs/artificial-intelligence/ml-platform-engineer)
20. [Apple: ML Platform Engineer job
    posting](https://jobs.apple.com/en-us/details/200653150-3956/ml-platform-engineer)
21. [ML Academy: Feature Store, Experiment Tracking & Model
    Registry](https://www.mlacademy.ai/articles/free-mlops-course-feature-store-model-registry-and-experiment-tracking)
22. [Tesla: Reinforcement Learning Engineer, Policy,
    Optimus](https://www.tesla.com/careers/search/job/reinforcement-learning-engineer-policy-optimus-222416)
23. [Tesla: Reinforcement Learning Engineer,
    Self-Driving](https://www.tesla.com/careers/search/job/reinforcement-learning-engineer-self-driving-221945)
24. [Anthropic: ML Systems Engineer, RL
    Engineering](https://jobs.menlovc.com/companies/anthropic/jobs/69674504-machine-learning-systems-engineer-rl-engineering)
25. [RLHF Engineering Implementation
    Guide](https://ajing.github.io/posts/2025-12-31-rlhf-engineering-implementation)
26. [NU.edu: 59 AI Job Statistics: Future of U.S.
    Jobs](https://www.nu.edu/blog/ai-job-statistics)
27. [Accenture: Lead Agentic AI Engineer job
    posting](https://www.accenture.com/us-en/careers/jobdetails?id=R00335391_en)
28. [Benchling: Agentic AI Engineer job
    posting](https://agentic-engineering-jobs.com/jobs/benchling-agentic-ai-engineer-c0wwUC)
29. [EPAM: Lead AI Engineer, Agentic and RAG
    Systems](https://careers.epam.com/en/vacancy/lead-ai-engineer-agentic-and-rag-systems-blt0gwqdcd06g43wvh3_en)
30. [Nike: Senior Data Scientist, ITC job
    posting](https://careers.nike.com/en/senior-data-scientist-itc/job/R-87980)
