-- select_uncleaned: the production selection for a cleaning configuration (design D12).
--
-- Anti-join semantics: raw keys with no artifact row for the configuration,
-- minus postings recorded as failed in prior run manifests (the failure
-- ledger is applied by the manifest adapter, not here — run records are
-- JSON documents, not catalog tables).
--
-- Parameterization (Athena named parameters):
--   :config — the run's SHA-256 configuration_hash (hex string).
--
-- Each source is a pinned per-shape table at its own prefix (no partition
-- politics — provision_query_layer), so the source value is stamped as a
-- literal per branch. `source_hash` is derived from the hidden "$path"
-- column (the content address; no JSON SerDe can read object keys);
-- `artifacts_cleaned` registers `lake/artifacts/type=cleaned_text/` with
-- `source_hash` + `configuration_hash` rows (tasks 3.4–3.6 schema).
--
-- The cost-bearing scan is the raw side; the artifacts side is tiny and
-- pruned by configuration_hash.
--
-- Deployment bounds (applied by the adapter, backend/pipeline/selection.py
-- :func:`select_uncleaned_sql`): the scheduled daily tier wraps this
-- anti-join with a manifests_ingestion recency window and a deterministic
-- row cap, so a configuration change can never silently re-enroll the
-- whole historical lake through the scheduled cloud tier. Full-lake
-- backfills run as explicit ``--backfill`` invocations (bounds disabled).

SELECT 'ashby' AS source,
       regexp_extract(r."$path", 'sha256=([0-9a-f]{64})\.json', 1) AS source_hash,
       r.source_id
FROM raw_ashby r
LEFT JOIN artifacts_cleaned a
  ON a.source_hash = regexp_extract(r."$path", 'sha256=([0-9a-f]{64})\.json', 1)
 AND a.configuration_hash = :config
WHERE a.source_hash IS NULL

UNION ALL

SELECT 'dataroles' AS source,
       regexp_extract(r."$path", 'sha256=([0-9a-f]{64})\.json', 1) AS source_hash,
       r.source_id
FROM raw_dataroles r
LEFT JOIN artifacts_cleaned a
  ON a.source_hash = regexp_extract(r."$path", 'sha256=([0-9a-f]{64})\.json', 1)
 AND a.configuration_hash = :config
WHERE a.source_hash IS NULL

UNION ALL

SELECT 'greenhouse' AS source,
       regexp_extract(r."$path", 'sha256=([0-9a-f]{64})\.json', 1) AS source_hash,
       r.source_id
FROM raw_greenhouse r
LEFT JOIN artifacts_cleaned a
  ON a.source_hash = regexp_extract(r."$path", 'sha256=([0-9a-f]{64})\.json', 1)
 AND a.configuration_hash = :config
WHERE a.source_hash IS NULL

UNION ALL

SELECT 'lever' AS source,
       regexp_extract(r."$path", 'sha256=([0-9a-f]{64})\.json', 1) AS source_hash,
       r.source_id
FROM raw_lever r
LEFT JOIN artifacts_cleaned a
  ON a.source_hash = regexp_extract(r."$path", 'sha256=([0-9a-f]{64})\.json', 1)
 AND a.configuration_hash = :config
WHERE a.source_hash IS NULL

UNION ALL

SELECT 'hiringcafe' AS source,
       regexp_extract(r."$path", 'sha256=([0-9a-f]{64})\.json', 1) AS source_hash,
       r.source_id
FROM raw_hiringcafe r
LEFT JOIN artifacts_cleaned a
  ON a.source_hash = regexp_extract(r."$path", 'sha256=([0-9a-f]{64})\.json', 1)
 AND a.configuration_hash = :config
WHERE a.source_hash IS NULL

UNION ALL

SELECT 'linkedin' AS source,
       regexp_extract(r."$path", 'sha256=([0-9a-f]{64})\.json', 1) AS source_hash,
       r.source_id
FROM raw_linkedin r
LEFT JOIN artifacts_cleaned a
  ON a.source_hash = regexp_extract(r."$path", 'sha256=([0-9a-f]{64})\.json', 1)
 AND a.configuration_hash = :config
WHERE a.source_hash IS NULL
