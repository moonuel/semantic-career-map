"""Cleaning stage: LLM boilerplate-only cleaning over raw lake payloads.

Extraction target for the experiment logic that proved itself in
``scripts/008_boilerplate_only_cleaning`` — the Kilo Gateway client
(HTTP retry + empty-response semantics) and the production
boilerplate-only system prompt. The experiment scripts stay untouched as
immutable artifacts; each extracted piece asserts fidelity back to its
origin (task 3.1 verification).

Runner and row-schema tasks land here next: normalization (3.2),
configuration identity (3.3), the flush-watermark writer (3.4), cache
resolution (3.5), run manifests (3.6), and the CLI (3.7).
"""
