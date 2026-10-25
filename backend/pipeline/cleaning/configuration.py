"""Cleaning-configuration identity (task 3.3, the authoritative hash).

``configuration_hash`` is the artifact-row key that pairs with
``source_hash``: it determines which configuration cleaned a posting,
what the ``select_uncleaned`` anti-join considers "already cleaned", and
what a cache hit means (an artifact row exists for the pair). Defined as
SHA-256 over ``model + \\n + system prompt + \\n + user template`` — the
same digest selection.py previously carried as a local stand-in; 3.3
makes the cleaning module authoritative and selection delegates.

Sensitivity is maximum (any character change in any field changes the
digest): the accepted churn is observable in run manifests rather than
silently papered over (design risk table — revisit with canonicalization
only if churn bites).
"""

from __future__ import annotations

import hashlib


def configuration_hash(model: str, system_prompt: str, user_template: str) -> str:
    """Digest one cleaning configuration (model + prompt pieces).

    Args:
        model: Gateway model identifier.
        system_prompt: The cleaning system prompt.
        user_template: The user message template (its ``{raw_full_text}``
            placeholder is part of the identity).

    Returns:
        64-char lowercase hex SHA-256 over the newline-joined fields.
    """
    return hashlib.sha256(
        f"{model}\n{system_prompt}\n{user_template}".encode()
    ).hexdigest()
