-- select_unembedded: the embedding tier's delta selection (the cache).
--
-- Anti-join semantics, the mirror of select_uncleaned.sql: cleaned artifact
-- rows (the input side, artifacts_cleaned pinned to its registered version
-- location) with NO embedding row for the run's configuration within the
-- run's model/variant/version partitions. The continuously-ingesting lake
-- embeds only deltas; a zero-row render is the healthy no-op.
--
-- Parameterization (Athena placeholders, bound by
-- backend.pipeline.embedding.selection.bind_sql_params):
--   :config   — the run's SHA-256 embedding configuration_hash
--
-- Partition scoping is the registration's job, not the query's: the
-- `postings_embeddings` table pins its location to one model/variant/version
-- generation (the artifacts table's discipline), so the query only filters
-- payload columns. The join key `(source_hash, configuration_hash)` is the
-- single cache identity every embedding row carries in its data.
--
-- The cost-bearing side is the small embeddings Parquet partition; the
-- artifacts side is also Parquet-compact. The raw JSON lake is untouched.

SELECT a.source_hash,
       a.cleaned_text AS variant_text
FROM artifacts_cleaned a
LEFT JOIN postings_embeddings e
  ON e.source_hash = a.source_hash
 AND e.configuration_hash = :config
WHERE e.source_hash IS NULL
