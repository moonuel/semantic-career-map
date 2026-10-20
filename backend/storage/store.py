"""S3-compatible object store client for the semantic-career-map objects.

The single S3-compatible store (real S3 or the local Garage container) holds
every branch of ``docs/architecture/data.md`` — the legacy ``live``/``configs``
branches and the lakehouse ``lake`` branch. This module extracts the inline
round-trip client from ``scripts/migrate_selected_postings`` now that a second
consumer (``scripts/seed_lake_raw``) exists.

Keys are always validated through :func:`backend.storage.keys.validate_key`
before use, so no code path can mint a key outside the layout.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from backend.storage import keys

# S3 error codes meaning "the object is not there": real S3 uses ``NoSuchKey``
# on GET, while HEAD carries no body and surfaces as the bare status "404"
# (Garage does the same). Everything else — auth, bucket, throttling — must
# propagate, not masquerade as a missing object.
_NOT_FOUND_CODES = frozenset({"NoSuchKey", "NotFound", "404"})


def _is_not_found(exc: ClientError) -> bool:
    if exc.response.get("Error", {}).get("Code") in _NOT_FOUND_CODES:
        return True
    return exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode") == 404


class ObjectStore:
    """Thin boto3 S3-compatible client, validated through ``keys``.

    Points at real S3 (no endpoint override; credentials resolved by
    boto3's default chain) or at a local Garage store (endpoint override;
    explicit keys) — the two are behaviourally identical from the caller's
    side. Keys are always validated before use.
    """

    def __init__(
        self,
        *,
        bucket: str,
        endpoint_url: str | None,
        region: str,
        access_key: str | None = None,
        secret_key: str | None = None,
    ) -> None:
        self._bucket = bucket
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint_url or None,
            region_name=region,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            # Endpoint-style servers (Garage) require path-style addressing;
            # real S3 keeps boto3's auto resolution.
            config=Config(s3={"addressing_style": "path"}) if endpoint_url else None,
        )

    @classmethod
    def from_env(cls) -> ObjectStore:
        """Construct from the ``STORE_S3_*`` variables loaded from ``.env``.

        ``STORE_S3_ENDPOINT`` selects the store. Set, it targets a local
        S3-compatible server (Garage) and the explicit
        ``STORE_S3_ACCESS_KEY_ID`` / ``STORE_S3_SECRET_ACCESS_KEY`` pair is
        required — Garage has no credential chain of its own. Empty or
        absent, it targets real S3: credentials must come from boto3's
        default chain (CLI profile locally, execution role in AWS), and
        static keys in the environment are refused, not used.

        Raises:
            RuntimeError: if required variables are missing or the
                endpoint/key pairing is invalid.
        """
        bucket = os.environ.get("STORE_S3_BUCKET")
        endpoint_url = os.environ.get("STORE_S3_ENDPOINT")
        region = os.environ.get("STORE_S3_REGION")
        access_key = os.environ.get("STORE_S3_ACCESS_KEY_ID")
        secret_key = os.environ.get("STORE_S3_SECRET_ACCESS_KEY")

        missing = [
            name
            for name, value in (
                ("STORE_S3_BUCKET", bucket),
                ("STORE_S3_REGION", region),
            )
            if not value
        ]
        if endpoint_url:
            if not access_key or not secret_key:
                missing += ["STORE_S3_ACCESS_KEY_ID", "STORE_S3_SECRET_ACCESS_KEY"]
        elif access_key or secret_key:
            raise RuntimeError(
                "STORE_S3_ENDPOINT is empty (real S3): static keys are not "
                "accepted; remove STORE_S3_ACCESS_KEY_ID and "
                "STORE_S3_SECRET_ACCESS_KEY and let the default credential "
                "chain resolve (CLI profile locally, IAM role in AWS)"
            )
        if missing:
            raise RuntimeError(f"missing environment variables: {', '.join(missing)}")
        return cls(
            bucket=bucket,
            endpoint_url=endpoint_url,
            region=region,
            access_key=access_key,
            secret_key=secret_key,
        )

    def put(self, key: str, data: bytes) -> None:
        """Write ``data`` at ``key`` (overwrite if present)."""
        keys.validate_key(key)
        self._client.put_object(Bucket=self._bucket, Key=key, Body=data)

    def get(self, key: str) -> bytes:
        """Read the bytes stored at ``key``.

        Raises:
            FileNotFoundError: if the object is 404-class absent.
        """
        keys.validate_key(key)
        try:
            return self._client.get_object(Bucket=self._bucket, Key=key)["Body"].read()
        except ClientError as exc:
            if _is_not_found(exc):
                raise FileNotFoundError(key) from exc
            raise

    def exists(self, key: str) -> bool:
        """Return whether ``key`` exists.

        Only 404-class ``ClientError``s count as absent; auth, bucket, and
        throttling failures propagate rather than answering ``False``.
        """
        keys.validate_key(key)
        try:
            self._client.head_object(Bucket=self._bucket, Key=key)
            return True
        except ClientError as exc:
            if _is_not_found(exc):
                return False
            raise

    def list_keys(self, prefix: str) -> Iterator[str]:
        """Yield every key under ``prefix`` (paginated)."""
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                yield obj["Key"]
