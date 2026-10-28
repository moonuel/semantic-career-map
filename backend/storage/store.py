"""S3-compatible object store client for the semantic-career-map objects.

The store is AWS S3. The client still accepts an endpoint override so any
S3-compatible server can stand in (tests, CI, an offline mirror), but the
project no longer ships or supports a local object store: the retired Garage
container was removed in favour of the dev S3 bucket.

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
# (some S3-compatible servers do the same). Everything else — auth, bucket,
# throttling — must propagate, not masquerade as a missing object.
_NOT_FOUND_CODES = frozenset({"NoSuchKey", "NotFound", "404"})


def _is_not_found(exc: ClientError) -> bool:
    if exc.response.get("Error", {}).get("Code") in _NOT_FOUND_CODES:
        return True
    return exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode") == 404


# S3 answers a failed ``If-None-Match: *`` with 412 ``PreconditionFailed``. A
# concurrent conditional write can instead surface as 409
# ``ConditionalRequestConflict`` on some S3-compatible servers; both mean "not
# written by this call", never "the store is broken".
_PRECONDITION_FAILED_CODES = frozenset(
    {"PreconditionFailed", "ConditionalRequestConflict"}
)


def _is_precondition_failed(exc: ClientError) -> bool:
    if exc.response.get("Error", {}).get("Code") in _PRECONDITION_FAILED_CODES:
        return True
    return exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode") == 412


class ObjectStore:
    """Thin boto3 S3 client, validated through ``keys``.

    Points at AWS S3 by default (no endpoint override; credentials resolved
    by boto3's default chain) or at any S3-compatible endpoint when one is
    supplied explicitly. The two are behaviourally identical from the
    caller's side. Keys are always validated before use.
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
            # Endpoint-style servers need path-style addressing; AWS keeps
            # boto3's auto resolution.
            config=Config(s3={"addressing_style": "path"}) if endpoint_url else None,
        )

    @classmethod
    def from_env(cls) -> ObjectStore:
        """Construct from the ``STORE_S3_*`` variables loaded from ``.env``.

        ``STORE_S3_ENDPOINT`` selects the target. Set, it names an
        S3-compatible server and the explicit ``STORE_S3_ACCESS_KEY_ID`` /
        ``STORE_S3_SECRET_ACCESS_KEY`` pair is required — such servers have
        no credential chain of their own. Empty or absent, it targets AWS S3:
        credentials must come from boto3's default chain (CLI profile
        locally, execution role in AWS), and static keys in the environment
        are refused, not used.

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

    def put_if_absent(self, key: str, data: bytes) -> bool:
        """Write ``data`` at ``key`` only when no object is already there.

        The conditional write is S3's ``If-None-Match: *``: the object store
        itself arbitrates the race, so two concurrent writers cannot both
        believe they created the object. Used by the version-claim registry
        to bind one ``configuration_hash`` to one prefix.

        Args:
            key: The validated object key.
            data: The bytes to write.

        Returns:
            ``True`` when this call created the object, ``False`` when it
            already existed.

        Raises:
            ClientError: for any failure other than a failed precondition.
        """
        keys.validate_key(key)
        try:
            self._client.put_object(
                Bucket=self._bucket, Key=key, Body=data, IfNoneMatch="*"
            )
            return True
        except ClientError as exc:
            if _is_precondition_failed(exc):
                return False
            raise

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

    def delete(self, key: str) -> None:
        """Remove the object at ``key`` (idempotent, S3 semantics).

        S3's ``delete_object`` succeeds even when the object is absent, so a
        double-delete race stays clean. Callers should delete only objects
        they can prove are remediable state (a healing run's emptied
        duplicate parts) — the lake is otherwise append-only.

        Args:
            key: The validated object key.
        """
        keys.validate_key(key)
        self._client.delete_object(Bucket=self._bucket, Key=key)

    def list_keys(self, prefix: str) -> Iterator[str]:
        """Yield every key under ``prefix`` (paginated)."""
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                yield obj["Key"]
