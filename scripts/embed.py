"""Embedding-tier CLI: resolve the run configuration, then run the driver.

The entry point mirrors the cleaning tier's driver convention: local runs
load ``.env`` via :func:`dotenv`, the shared Athena runner owns region rules
and workgroups, and the summary JSON is the observable outcome (the
Layer-4 run record will persist it). The run is delta-only: the
``select_unembedded`` anti-join selects postings with no embedding row for
the run's ``configuration_hash``, so re-runs land the healthy no-op
(``embedded: 0``) and config changes re-embed under a new version prefix.

Config identity — two entry paths, never a mixed one:

- **Pointer path** (default: no config-shaping flag and no
  ``--config-file``/``--sql-file``): the run configuration is resolved from
  the production pointer (the resolution chain: pointer → claim →
  ``configs/<hash>.json``) and printed as JSON before the
  run — the config a scheduled run used is always observable, and the run's
  digest equals the claim owner by construction, so the anti-join is bound
  to the right generation.
- **Build path** (any config-shaping flag or ``--config-file``/``--sql-file``
  set — the loud escape hatch, printed): flags build the config exactly as
  before; ``--revision auto`` resolves the current latest revision (printed
  once — provenance stays the pinned string in the config, and the encoder
  asserts it at load). This path may target ``x-`` ephemeral tokens for
  throwaway runs.

``--write-config-to`` dumps the exact resolved config JSON for reproducible
re-runs (``--config-file``), since the hash is part of the identity and
flags must not drift between runs. The driver (:func:`run_embedding`) is
config-agnostic — both paths feed it the same object.

Usage::

    python scripts/embed.py                        # scheduled path: pointer-resolved config
    python scripts/embed.py --config-file embedding-config.json   # explicit re-run
    python scripts/embed.py ... --write-config-to embedding-config.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from dotenv import load_dotenv

from backend.pipeline.athena import aws_region
from backend.pipeline.embedding import selection
from backend.pipeline.embedding.config import (
    DEFAULT_INPUT_ARTIFACT,
    EmbeddingConfig,
    default_model_segment,
)
from backend.pipeline.embedding.driver import (
    DATABASE_DEFAULT,
    WORKGROUP_DEFAULT,
    run_embedding,
)
from backend.pipeline.embedding.encoder import resolve_pinned_revision

# The build path's own defaults (the pointer path bypasses creation
# entirely). Flags default to None; a None field on the build path resolves
# here — explicit flags never silently change meaning.
_BUILD_DEFAULTS: dict[str, object] = {
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "variant": "minilm",
    "text_variant": "variant_text",
    "batch_size": 16,
    "max_length": 256,
    "input_artifact": DEFAULT_INPUT_ARTIFACT,
    "input_version": "v1",
    "version": "v1",
}

# The flags whose presence selects the build path (the loud escape hatch).
_POINTER_FLAGS = (
    "model",
    "revision",
    "model_segment",
    "variant",
    "text_variant",
    "fallback_text_variant",
    "batch_size",
    "max_length",
    "input_artifact",
    "input_version",
    "version",
    "sql_file",
    "config_file",
)


def pointer_mode(args: argparse.Namespace) -> bool:
    """True when no config-shaping flag or file names an explicit generation.

    Args:
        args: The parsed CLI namespace.

    Returns:
        ``True`` to resolve through the production pointer (scheduled
        path); ``False`` to build from flags (escape hatch).
    """
    return all(getattr(args, name, None) is None for name in _POINTER_FLAGS)


def config_from_store(store) -> EmbeddingConfig:
    """Resolve the scheduled run's config through the production pointer.

    Args:
        store: The object store.

    Returns:
        The first pointer-named embeddings generation's loaded config (its
        digest equals the claim owner by construction).

    Raises:
        RuntimeError: naming the fix when pointer/claim/config is absent
            or the chain disagrees (never a silent fallback).
    """
    from backend.pipeline.production import resolve_production

    production = resolve_production(store)
    return production.embeddings[0].config


def build_config(args: argparse.Namespace) -> EmbeddingConfig:
    """Resolve CLI flags (or a config file) into the run configuration.

    Unset flags fall back to the build path's own defaults
    (:data:`_BUILD_DEFAULTS`) so explicit flags keep today's behavior.

    Args:
        args: The parsed CLI namespace (``None``-able option fields).

    Returns:
        The embedding configuration.

    Raises:
        OSError: a missing config file or unreadable SQL file.
        ValueError: a revision could not be resolved from ``auto``.
    """
    if args.config_file:
        with open(args.config_file, encoding="utf-8") as handle:
            return EmbeddingConfig(**json.load(handle))

    def value(name: str) -> object:
        """The explicit flag value, or the build-path default."""
        return (
            getattr(args, name)
            if getattr(args, name) is not None
            else _BUILD_DEFAULTS[name]
        )

    model_name = value("model")
    revision = (
        value("revision")
        if args.revision not in (None, "auto")
        else resolve_pinned_revision(model_name)
    )
    if args.sql_file:
        with open(args.sql_file, encoding="utf-8") as handle:
            sql_text = handle.read().strip()
    else:
        sql_text = selection.select_unembedded_sql()
    return EmbeddingConfig(
        model_name=model_name,
        model_revision=revision,
        model_segment=args.model_segment or default_model_segment(model_name),
        variant=value("variant"),
        text_variant=value("text_variant"),
        fallback_text_variant=args.fallback_text_variant,
        batch_size=value("batch_size"),
        max_length=value("max_length"),
        normalize=True,
        input_artifact=value("input_artifact"),
        input_version=value("input_version"),
        version=value("version"),
        sql_text=sql_text,
    )


def main() -> None:
    """CLI flow: resolve config (pointer or build) → one run → summary JSON."""
    load_dotenv()  # local convention; containers rely on task-def env
    parser = argparse.ArgumentParser(
        prog="python scripts/embed.py",
        description="Embedding tier: production-pointer-resolved config "
        "(or, with any config flag set, the explicit build escape hatch) → "
        "anti-join delta selection → CPU encode → lake parts + dataset "
        "manifest, with configuration-hash provenance.",
    )
    parser.add_argument("--config-file", default=None)
    parser.add_argument(
        "--model", default=None, help="build path: full model id (sets the build path)"
    )
    parser.add_argument(
        "--revision",
        default=None,
        help='build path: pinned model revision; "auto" resolves the current latest (printed)',
    )
    parser.add_argument("--model-segment", default=None)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--text-variant", default=None)
    parser.add_argument("--fallback-text-variant", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--input-artifact", default=None)
    parser.add_argument("--input-version", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--sql-file", default=None)
    parser.add_argument("--database", default=DATABASE_DEFAULT)
    parser.add_argument("--workgroup", default=WORKGROUP_DEFAULT)
    parser.add_argument(
        "--write-config-to",
        default=None,
        help="dump the resolved config JSON for deterministic re-runs",
    )
    args = parser.parse_args()

    if pointer_mode(args):
        from backend.storage import keys
        from backend.storage.store import ObjectStore

        store = ObjectStore.from_env()
        from backend.pipeline.production import resolve_production

        production = resolve_production(store)
        config = production.embeddings[0].config
        entry = production.embeddings[0]
        print(
            json.dumps(
                {
                    "pointer": keys.state_pointer(),
                    "cleaning_stack": production.cleaning.stack,
                    "embeddings_stack": entry.stack,
                    "configuration_hash": entry.configuration_hash,
                    "config_path": keys.config(entry.configuration_hash),
                    "source": "production pointer",
                },
                indent=2,
            )
        )
    else:
        overridden = {
            name for name in _POINTER_FLAGS if getattr(args, name, None) is not None
        }
        print(
            "pointer bypassed: explicit flags ("
            + ", ".join(sorted(overridden))
            + ") — build path"
        )
        config = build_config(args)
    if args.write_config_to:
        with open(args.write_config_to, "w", encoding="utf-8") as handle:
            json.dump(asdict(config), handle, indent=2, sort_keys=True)

    import boto3

    from backend.storage.store import ObjectStore

    athena = boto3.Session().client("athena", region_name=aws_region())
    summary = run_embedding(
        ObjectStore.from_env(),
        athena,
        _encoder_for(config),
        config,
        database=args.database,
        workgroup=args.workgroup,
    )
    print(summary_json(summary))


def _encoder_for(config: EmbeddingConfig):
    from backend.pipeline.embedding.encoder import SentenceEncoder

    return SentenceEncoder(config)


def summary_json(summary) -> str:
    return json.dumps(asdict(summary), indent=2)


if __name__ == "__main__":
    main()
