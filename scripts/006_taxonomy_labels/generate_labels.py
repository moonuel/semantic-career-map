"""LLM-driven taxonomy labeling of job postings via Kilo Gateway.

Phase 2.1 — Assigns 12-category ML/AI function taxonomy tags and generated
functional titles to 27 job postings based on their llm_clean_text.

Evaluates LLM labeling accuracy against a manually labeled 5-posting
golden subset.

Usage:
    python scripts/006_taxonomy_labels/generate_labels.py
"""

import hashlib
import json
import os
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
JOBS_PATH = PROJECT_ROOT / "data" / "jobs.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "006_taxonomy_labels.json"
CACHE_PATH = PROJECT_ROOT / "data" / ".006_taxonomy_cache.json"

API_KEY = os.getenv("KILO_API_KEY")
BASE_URL = "https://api.kilo.ai/api/gateway/chat/completions"
MODEL = "openai/gpt-5.4-nano"
TEMPERATURE = 0.0
MAX_TOKENS = 512
RATE_LIMIT_SECONDS = 1.0

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
learning and pre-training from foundation models."""

TITLE_VOCABULARY = {
    "agentic-ai": "Agentic AI Engineer",
    "llm-fine-tuning": "LLM Fine-Tuning Engineer",
    "llm-information-retrieval": "Information Retrieval Engineer",
    "classical-ml": "Classical ML Engineer",
    "mlops-production": "ML Operations Engineer",
    "data-engineering": "Data Engineer",
    "analytics-storytelling": "Analytics Engineer",
    "computer-vision": "Computer Vision Engineer",
    "ml-platform": "ML Platform Engineer",
    "reinforcement-learning": "Reinforcement Learning Engineer",
    "research": "AI Research Scientist",
    "ai-safety-governance": "AI Safety Engineer",
    "deep-learning": "Deep Learning Engineer",
}

GOLDEN_LABELS: dict[str, list[str]] = {
    "bmo-associate-data-scientist": [
        "classical-ml",
        "analytics-storytelling",
    ],
    "affirm-ml-engineer-2": [
        "classical-ml",
        "mlops-production",
    ],
    "hellofresh-ml-engineer-operations-technology": [
        "agentic-ai",
        "mlops-production",
        "ml-platform",
    ],
    "mastercard-data-scientist-2": [
        "classical-ml",
        "mlops-production",
    ],
    "scribd-data-scientist-2": [
        "classical-ml",
        "deep-learning",
        "llm-information-retrieval",
    ],
}

SYSTEM_PROMPT = f"""You are a job posting classifier. Analyze the provided job \
posting text and classify it according to the 13-category ML/AI function \
taxonomy below.

{TAXONOMY_DEFINITIONS}

Assign function_tags: a list of 1-3 category names selected STRICTLY from the \
13 categories above. Valid values are: agentic-ai, llm-fine-tuning, \
llm-information-retrieval, classical-ml, mlops-production, data-engineering, \
analytics-storytelling, computer-vision, ml-platform, reinforcement-learning, \
research, ai-safety-governance, deep-learning. Do NOT invent new category \
names. Include all that match — do not force a single label if multiple \
apply. If the posting clearly blends two functions (e.g., trains models AND \
deploys to production), include both.

Assign generated_title: a functional title from this controlled vocabulary: \
{json.dumps(TITLE_VOCABULARY, indent=2)}

Choose the title whose taxonomy category best represents the PRIMARY function \
of this role. Use EXACTLY the title string from the vocabulary — do not \
modify or invent new titles.

Provide rationale: a 1-2 sentence explanation of why these tags and title \
were chosen, referencing specific duties from the posting.

Output a JSON object with exactly three fields:
- function_tags: list of strings (1-3 taxonomy category names from the 13 listed)
- generated_title: string (exactly one title from the vocabulary above)
- rationale: string (1-2 sentence explanation)

Output ONLY the JSON object. No markdown, no explanation, no code fences."""

USER_MESSAGE_TEMPLATE = "Classify this job posting:\n\n---\n{text}"

MODEL_PRICING = {
    "google/gemini-2.5-flash": (0.15, 0.60),
    "openai/gpt-5.4-nano": (0.15, 0.60),
    "deepseek/deepseek-v4-flash": (0.20, 0.80),
}


def build_prompt_hash() -> str:
    return hashlib.sha256(
        (SYSTEM_PROMPT + USER_MESSAGE_TEMPLATE).encode("utf-8")
    ).hexdigest()


class TaxonomyLabeler:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = MODEL,
        base_url: str = BASE_URL,
        temperature: float = TEMPERATURE,
        cache_path: Path = CACHE_PATH,
    ) -> None:
        self.api_key = api_key or API_KEY
        if not self.api_key:
            raise RuntimeError(
                "KILO_API_KEY is not set. Add it to .env or pass api_key= explicitly."
            )
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.cache_path = cache_path
        self.prompt_hash = build_prompt_hash()
        self.cache: dict = self._load_cache()

    def _load_cache(self) -> dict:
        if self.cache_path.exists():
            with open(self.cache_path) as f:
                data = json.load(f)
        else:
            data = {"meta": {}, "entries": {}}

        cached_hash = data.get("meta", {}).get("prompt_hash", "")
        if cached_hash != self.prompt_hash:
            print("Prompt hash mismatch detected. Invalidating cache.")
            data = {"meta": {}, "entries": {}}

        data["meta"] = {
            "prompt_hash": self.prompt_hash,
            "model": self.model,
        }
        return data

    def _save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w") as f:
            json.dump(self.cache, f, indent=2, ensure_ascii=False)

    def label(self, posting_id: str, text: str) -> dict:
        input_hash = hashlib.sha256((posting_id + text).encode("utf-8")).hexdigest()

        entry = self.cache.get("entries", {}).get(input_hash)
        if entry:
            print(f"  [cache hit] {posting_id}")
            return entry["output"]

        print(f"  [api call] {posting_id}")
        user_message = USER_MESSAGE_TEMPLATE.format(text=text)
        response = httpx.post(
            self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": MAX_TOKENS,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
            },
            timeout=httpx.Timeout(120.0),
        )
        response.raise_for_status()
        data = response.json()

        content = data["choices"][0]["message"]["content"]
        if not content.strip():
            raise RuntimeError(f"Empty response for {posting_id}")

        parsed = json.loads(content)

        usage = data.get("usage", {})
        self.cache.setdefault("entries", {})[input_hash] = {
            "posting_id": posting_id,
            "output": parsed,
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
            },
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        self._save_cache()
        return parsed

    def label_all(self, postings: list[dict]) -> list[dict]:
        results: list[dict] = []
        valid_tags = set(TITLE_VOCABULARY.keys())
        valid_titles = set(TITLE_VOCABULARY.values())
        total = len(postings)

        for i, posting in enumerate(postings, start=1):
            pid = posting["id"]
            text = posting.get("llm_clean_text", "")
            print(f"[{i}/{total}] {pid}")

            try:
                parsed = self.label(pid, text)
            except Exception as e:
                print(f"  ERROR: {e}")
                parsed = {
                    "function_tags": [],
                    "generated_title": "",
                    "rationale": f"Error: {e}",
                }

            tags = parsed.get("function_tags", [])
            if isinstance(tags, str):
                tags = [tags]
            tags = [
                t.strip()
                for t in tags
                if isinstance(t, str) and t.strip() in valid_tags
            ]

            title = parsed.get("generated_title", "").strip()
            if title not in valid_titles:
                print(
                    f"  WARNING: title '{title}' not in controlled vocabulary, storing as-is"
                )

            rationale = parsed.get("rationale", "").strip()

            result = {
                "posting_id": pid,
                "original_title": posting.get("title_raw", ""),
                "original_category": posting.get("role_category", ""),
                "function_tags": tags,
                "generated_title": title,
                "rationale": rationale,
            }
            results.append(result)

            if i < total:
                time.sleep(RATE_LIMIT_SECONDS)

        return results


def validate_golden(
    results: list[dict],
    golden_labels: dict[str, list[str]],
) -> None:
    results_by_id = {r["posting_id"]: r for r in results}
    matches = 0
    total = 0

    print("\n" + "=" * 60)
    print("GOLDEN SUBSET VALIDATION")
    print("=" * 60)

    for pid, golden_tags in golden_labels.items():
        total += 1
        result = results_by_id.get(pid)
        if not result:
            print(f"  {pid}: not found in results")
            continue

        llm_tags = result["function_tags"]
        hit = any(gt in llm_tags for gt in golden_tags)
        if hit:
            matches += 1
            status = "MATCH"
        else:
            status = "MISMATCH"

        print(f"  {pid}")
        print(f"    Golden:  {golden_tags}")
        print(f"    LLM:     {llm_tags}")
        print(f"    Title:   {result['generated_title']}")
        print(f"    Status:  {status}")
        if not hit:
            print(f"    Rationale: {result['rationale']}")

    recall = matches / total * 100 if total > 0 else 0.0
    print(f"\n  Recall: {matches}/{total} ({recall:.1f}%)")


def estimate_cost(cache: dict, model: str) -> float:
    price_in, price_out = MODEL_PRICING.get(model, (0.15, 0.60))
    total = 0.0
    for entry in cache.get("entries", {}).values():
        u = entry.get("usage", {})
        total += (
            u.get("prompt_tokens", 0) * price_in / 1_000_000
            + u.get("completion_tokens", 0) * price_out / 1_000_000
        )
    return total


def main() -> None:
    with open(JOBS_PATH) as f:
        jobs = json.load(f)

    print(f"Loaded {len(jobs)} postings from {JOBS_PATH}")

    labeler = TaxonomyLabeler()
    results = labeler.label_all(jobs)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nWrote labeled results to {OUTPUT_PATH}")

    validate_golden(results, GOLDEN_LABELS)

    cost = estimate_cost(labeler.cache, labeler.model)
    print(f"\nEstimated cost: ${cost:.4f}")

    total_tokens = 0
    for entry in labeler.cache.get("entries", {}).values():
        u = entry.get("usage", {})
        total_tokens += u.get("prompt_tokens", 0) + u.get("completion_tokens", 0)
    print(f"Total tokens: {total_tokens}")

    print("\nLabel distribution:")
    titles = [r["generated_title"] for r in results]
    from collections import Counter

    for title, count in Counter(titles).most_common():
        print(f"  {title}: {count}")


if __name__ == "__main__":
    main()
