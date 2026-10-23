"""CLI entry point: ``python -m backend.pipeline.cleaning`` (task 3.7).

Fails fast per posting: a first worker failure or a SIGINT ends the
run — committed flushes stay durable, the run record is written
(``failed`` / ``interrupted``) and the CLI reports the merged record
before exiting. Loads the project-root ``.env`` (003 convention) and
builds the real store + gateway client; see
:func:`backend.pipeline.cleaning.runner.run_cleaning` for the injectable
variant tests use.
"""

from __future__ import annotations

import argparse
import json
import subprocess

from dotenv import load_dotenv

from backend.pipeline.cleaning import llm_client, runner
from backend.pipeline.cleaning.llm_client import GatewayClient


def _git_sha() -> str:
    """Best-effort pipeline version: the short HEAD sha (``dev`` fallback)."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
    except OSError, subprocess.SubprocessError:
        return "dev"


def _parse_bytes(text: str) -> int:
    """Parse a byte budget accepting ``128``, ``64KiB``, ``2MiB`` forms."""
    text = text.strip().lower()
    units = {
        "kib": 1024,
        "mib": 1024 * 1024,
        "b": 1,
        "": 1,
    }
    for suffix, factor in units.items():
        if text.endswith(suffix) and suffix:
            return int(text[: -len(suffix)].strip()) * factor
    if text.endswith("b"):
        return int(text[:-1].strip())
    return int(text)


def main() -> None:
    """CLI flow: manifest → runner over the real store and client."""
    load_dotenv()  # KILO_API_KEY + STORE_S3_* live in .env (003 convention)
    parser = argparse.ArgumentParser(
        prog="python -m backend.pipeline.cleaning",
        description="LLM boilerplate-only cleaning over a pinned manifest "
        "(fail-fast per posting; flush-watermark durability).",
    )
    parser.add_argument(
        "--sample-manifest",
        required=True,
        help="Pinned manifest path (either producer's format).",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Gateway model id (priced via the frozen 007 probe table).",
    )
    parser.add_argument("--workers", type=int, default=8, help="Bounded pool size.")
    parser.add_argument(
        "--flush-rows", type=int, default=128, help="Row-trigger threshold."
    )
    parser.add_argument(
        "--flush-bytes",
        type=str,
        default="2MiB",
        help="Bytes-trigger threshold ('2MiB' shape).",
    )
    parser.add_argument(
        "--flush-max-seconds",
        type=float,
        default=60.0,
        help="Dwell-trigger threshold in seconds.",
    )
    parser.add_argument(
        "--version-root",
        default="v1",
        help="Dataset version token (append-only; never overwritten).",
    )
    args = parser.parse_args()

    from backend.pipeline.cleaning.prompts import (
        BOILERPLATE_ONLY_SYSTEM_PROMPT,
        USER_MESSAGE_TEMPLATE,
    )
    from backend.storage.store import ObjectStore

    manifest = runner.load_manifest(args.sample_manifest)
    record = runner.run_cleaning(
        ObjectStore.from_env(),
        manifest,
        input_dataset=args.sample_manifest,
        model=args.model,
        system_prompt=BOILERPLATE_ONLY_SYSTEM_PROMPT,
        user_template=USER_MESSAGE_TEMPLATE,
        client=GatewayClient(),
        pipeline_version=_git_sha(),
        workers=args.workers,
        version=args.version_root,
        flush_rows=args.flush_rows,
        flush_bytes=_parse_bytes(args.flush_bytes),
        flush_seconds=args.flush_max_seconds,
        pacing_budget=llm_client.RATE_LIMIT_SECONDS,
    )
    print(json.dumps(record, indent=2, ensure_ascii=False))
    if record["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
