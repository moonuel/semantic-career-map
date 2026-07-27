"""LLM-based boilerplate removal via Kilo Gateway.

Phase 1.4b — Feeds raw_full_text to an LLM to extract only job-signal content
(responsibilities, qualifications, skills), stripping company descriptions, benefits,
EEO statements, and recruiter boilerplate.

Usage:
    python scripts/003_llm_extraction/extract_clean_text.py
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
CACHE_PATH = PROJECT_ROOT / "data" / ".llm_clean_cache_v2.json"

API_KEY = os.getenv("KILO_API_KEY")
BASE_URL = "https://api.kilo.ai/api/gateway/chat/completions"
# MODEL = "google/gemini-2.5-flash"
# MODEL = "openai/gpt-5.4-nano"
MODEL = "openai/gpt-5.4-nano"
TEMPERATURE = 0.0
MAX_TOKENS = 2048
RATE_LIMIT_SECONDS = 1.0

SYSTEM_PROMPT = """You are a job posting cleaner. Extract only the parts of a job posting that \
describe the job itself: responsibilities, required qualifications, and \
preferred/nice-to-have skills. Remove everything else.

Categories to strip completely:
- Company description, "About Us", mission statements, values
- Team descriptions: what the team does, team structure, team culture, team function, team domain, team mission within the organization
- Role-purpose narratives: "About the Role" descriptions that frame the role in terms of team context rather than specific duties
- If a sentence describes both the team's domain/function AND specific duties, remove the entire sentence — keep responsibilities pure
- Salary ranges, pay grades, equity, compensation details
- Benefits: health/dental/vision, vacation/PTO, parental leave, wellness
- EEO/diversity statements, "equal opportunity employer" boilerplate
- Recruiter notes, application instructions
- Office locations, hybrid/remote policy boilerplate
- Perks, "why you'll love working here", employee testimonials

Preserve verbatim (do not summarize, paraphrase, or invent):
- All job duties, responsibilities, and day-to-day tasks (not domain-framing, but the actual work)
- All technical skills, tools, frameworks, languages, platforms
- All required qualifications and experience levels
- All preferred/nice-to-have qualifications
- All education and certification requirements
- Original wording of preserved sections

Output only the cleaned text. No headers, prefixes, explanations, or formatting."""

USER_MESSAGE_TEMPLATE = "Clean this job posting:\n\n---\n{raw_full_text}"


def build_prompt_hash() -> str:
    return hashlib.sha256(
        (SYSTEM_PROMPT + USER_MESSAGE_TEMPLATE).encode("utf-8")
    ).hexdigest()


class LLMCleaner:
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
            with open(self.cache_path, "r") as f:
                data = json.load(f)
        else:
            data = {"meta": {}, "entries": {}}

        cached_hash = data.get("meta", {}).get("prompt_hash")
        if cached_hash and cached_hash != self.prompt_hash:
            print(
                f"Prompt hash mismatch (cached={cached_hash[:8]}... "
                f"current={self.prompt_hash[:8]}...). Invalidating cache."
            )
            data = {"meta": {}, "entries": {}}

        data["meta"] = {"prompt_hash": self.prompt_hash, "model": self.model}
        return data

    def _save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w") as f:
            json.dump(self.cache, f, indent=2, ensure_ascii=False)

    def clean(self, posting_id: str, raw_text: str) -> str:
        input_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()

        entry = self.cache.get("entries", {}).get(posting_id)
        if entry and entry.get("input_hash") == input_hash:
            print(f"  [cache hit] {posting_id}")
            return entry["output"]

        print(f"  [api call] {posting_id}")
        user_message = USER_MESSAGE_TEMPLATE.format(raw_full_text=raw_text)
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

        usage = data.get("usage", {})
        self.cache.setdefault("entries", {})[posting_id] = {
            "input_hash": input_hash,
            "output": content,
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
            },
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        self._save_cache()
        return content

    def clean_batch(self, jobs: list[dict]) -> None:
        total = len(jobs)
        for i, job in enumerate(jobs, start=1):
            posting_id = job["id"]
            raw_text = job["raw_full_text"]
            print(f"[{i}/{total}] {posting_id}")
            try:
                cleaned = self.clean(posting_id, raw_text)
                job["llm_clean_text"] = cleaned
            except Exception as e:
                print(f"  ERROR: {e}")
            if i < total:
                time.sleep(RATE_LIMIT_SECONDS)


MODEL_PRICING = {
    "google/gemini-2.5-flash": (0.15, 0.60),
    "openai/gpt-5.4-nano": (0.15, 0.60),
    "deepseek/deepseek-v4-flash": (0.20, 0.80),
}


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
    with open(JOBS_PATH, "r") as f:
        jobs = json.load(f)

    print(f"Loaded {len(jobs)} postings from {JOBS_PATH}")
    cleaner = LLMCleaner()
    cleaner.clean_batch(jobs)

    with open(JOBS_PATH, "w") as f:
        json.dump(jobs, f, indent=2, ensure_ascii=False)
    print(f"Wrote llm_clean_text fields to {JOBS_PATH}")

    cost = estimate_cost(cleaner.cache, cleaner.model)
    print(f"Estimated cost: ${cost:.4f}")

    total_tokens = 0
    for entry in cleaner.cache.get("entries", {}).values():
        u = entry.get("usage", {})
        total_tokens += u.get("prompt_tokens", 0) + u.get("completion_tokens", 0)
    print(f"Total tokens: {total_tokens}")


if __name__ == "__main__":
    main()
