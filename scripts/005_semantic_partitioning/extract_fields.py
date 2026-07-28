"""LLM-based semantic partitioning of job postings via Kilo Gateway.

Phase 2.0 — Feeds undifferentiated golden text to an LLM across two passes
to extract job-context ("what is this job?") and role-context ("what purpose
does this role serve?") fields independently.

Usage:
    python scripts/005_semantic_partitioning/extract_fields.py
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
GOLDEN_PATH = PROJECT_ROOT / "data" / "golden_cleaned.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "005-semantic_partitions.json"
CACHE_PATH = PROJECT_ROOT / "data" / ".005-semantic_partition_cache.json"

API_KEY = os.getenv("KILO_API_KEY")
BASE_URL = "https://api.kilo.ai/api/gateway/chat/completions"
MODEL = "openai/gpt-5.4-nano"
TEMPERATURE = 0.0
MAX_TOKENS = 2048
RATE_LIMIT_SECONDS = 1.0

JOB_CONTEXT_PROMPT = """You are a job posting analyzer. Extract only the parts \
that describe what the job is: duties, responsibilities, required skills, \
qualifications, and experience. Remove everything else.

Categories to strip completely:
- Company description, "About Us", mission statements, values
- Team descriptions: what the team does, team structure, team culture, \
  how the team fits into the organization
- Statements about the role's purpose within the organization, scope of \
  impact, or how the role supports the team's mission
- Salary ranges, pay grades, equity, compensation details
- Benefits: health/dental/vision, vacation/PTO, parental leave, wellness
- EEO/diversity statements, "equal opportunity employer" boilerplate
- Recruiter notes, application instructions
- Office locations, hybrid/remote policy boilerplate
- Perks, "why you'll love working here", employee testimonials

Preserve verbatim (do not summarize, paraphrase, or invent):
- All job duties, responsibilities, and day-to-day tasks
- All technical skills, tools, frameworks, languages, platforms
- All required qualifications and experience levels
- All preferred/nice-to-have qualifications
- All education and certification requirements
- Original wording of preserved sections

Output only the extracted text. No headers, prefixes, explanations, or \
formatting."""

ROLE_CONTEXT_PROMPT = """You are a job posting analyzer. Extract only the parts \
that describe the purpose the role serves: what team the role is on, what that \
team does and why it exists, how the role fits into the broader organization, \
and the impact or scope of the position.

Extract verbatim (do not summarize or paraphrase):
- The team name and a description of what the team does and why
- How the role fits into the team and organization
- The mission or purpose of the team within the company
- Any description of the team's structure, stakeholders, or cross-functional \
  relationships
- The scope or impact of the role within the organization

Strip completely:
- Specific job duties, responsibilities, and day-to-day tasks
- Required qualifications, experience levels, education requirements
- Technical skills, tools, frameworks, languages, platforms
- Preferred or nice-to-have qualifications
- Salary, benefits, company culture, EEO statements, or any boilerplate

Output only the extracted organizational purpose text. No headers, prefixes, \
explanations, or formatting."""

USER_MESSAGE_TEMPLATE = "Extract the relevant information from this job posting:\n\n---\n{text}"

MODEL_PRICING = {
    "google/gemini-2.5-flash": (0.15, 0.60),
    "openai/gpt-5.4-nano": (0.15, 0.60),
    "deepseek/deepseek-v4-flash": (0.20, 0.80),
}


def build_pass_hash(prompt: str) -> str:
    return hashlib.sha256(
        (prompt + USER_MESSAGE_TEMPLATE).encode("utf-8")
    ).hexdigest()


class PartitionExtractor:
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
        self.prompts = {
            "job-context": JOB_CONTEXT_PROMPT,
            "role-context": ROLE_CONTEXT_PROMPT,
        }
        self.prompt_hashes = {
            field: build_pass_hash(prompt)
            for field, prompt in self.prompts.items()
        }
        self.cache: dict = self._load_cache()

    def _load_cache(self) -> dict:
        if self.cache_path.exists():
            with open(self.cache_path, "r") as f:
                data = json.load(f)
        else:
            data = {"meta": {}, "entries": {}}

        cached_hashes = data.get("meta", {}).get("prompt_hashes", {})
        if cached_hashes != self.prompt_hashes:
            print(
                "Prompt hash mismatch detected. Invalidating cache."
            )
            data = {"meta": {}, "entries": {}}

        data["meta"] = {
            "prompt_hashes": self.prompt_hashes,
            "model": self.model,
        }
        return data

    def _save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w") as f:
            json.dump(self.cache, f, indent=2, ensure_ascii=False)

    def extract(self, posting_id: str, text: str, field: str) -> str:
        prompt = self.prompts[field]
        input_hash = hashlib.sha256(
            (posting_id + text + field).encode("utf-8")
        ).hexdigest()

        entry = self.cache.get("entries", {}).get(input_hash)
        if entry:
            print(f"  [cache hit] {posting_id} ({field})")
            return entry["output"]

        print(f"  [api call] {posting_id} ({field})")
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
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_message},
                ],
            },
            timeout=httpx.Timeout(120.0),
        )
        response.raise_for_status()
        data = response.json()

        content = data["choices"][0]["message"]["content"]
        if not content.strip():
            raise RuntimeError(f"Empty response for {posting_id} ({field})")

        usage = data.get("usage", {})
        self.cache.setdefault("entries", {})[input_hash] = {
            "posting_id": posting_id,
            "field": field,
            "output": content,
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
            },
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        self._save_cache()
        return content

    def extract_all(self, postings: list[dict]) -> list[dict]:
        results: list[dict] = []
        total = len(postings)
        for i, posting in enumerate(postings, start=1):
            pid = posting["posting_id"]
            text = posting["text"]
            print(f"[{i}/{total}] {pid}")

            result = {
                "posting_id": pid,
                "source": posting.get("source", ""),
            }

            for field in ("job-context", "role-context"):
                try:
                    extracted = self.extract(pid, text, field)
                    result[f"llm_{field.replace('-', '_')}"] = extracted
                except Exception as e:
                    print(f"  ERROR ({field}): {e}")
                    result[f"llm_{field.replace('-', '_')}"] = ""

            result["golden_job_context"] = posting["job-context"]
            result["golden_role_context"] = posting["role-context"]
            result["undifferentiated_text"] = text
            results.append(result)

            if i < total:
                time.sleep(RATE_LIMIT_SECONDS)

        return results


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
    with open(GOLDEN_PATH, "r") as f:
        postings = json.load(f)

    print(f"Loaded {len(postings)} postings from {GOLDEN_PATH}")

    extractor = PartitionExtractor()
    results = extractor.extract_all(postings)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Wrote partitioned results to {OUTPUT_PATH}")

    cost = estimate_cost(extractor.cache, extractor.model)
    print(f"Estimated cost: ${cost:.4f}")

    total_tokens = 0
    for entry in extractor.cache.get("entries", {}).values():
        u = entry.get("usage", {})
        total_tokens += u.get("prompt_tokens", 0) + u.get("completion_tokens", 0)
    print(f"Total tokens: {total_tokens}")


if __name__ == "__main__":
    main()
