---
id: "005"
title: "Object Storage Naming Conventions"
date: "2026-09-19"
summary: "Survey of bucket and object naming rules across AWS S3, Google Cloud Storage, Azure Blob Storage, and S3-compatible providers, plus best practices for consistency, discoverability, scalability, security, and lifecycle management."
---
# Object Storage Naming Conventions — Research Report

**Date:** 2026-09-19
**Purpose:** Establish a naming standard for bucket and object names in an S3-compatible object
store, informed by the rules and best practices of the major cloud providers.

---

## 1. Executive Summary

Object storage is a flat key-value store: a container (bucket/container) holds objects, and each
object is identified by a name (key). There is no real directory hierarchy — folder-like structure
is simulated with `/` in names, and every provider recommends treating the name as a first-class
design artifact because names drive performance, security, lifecycle rules, and cost reporting.

Three facts recur across every major provider:

1. **Bucket/container names are global and immutable.** Once created they cannot be renamed, and
   the name is publicly visible and probeable. Naming mistakes are expensive to correct.
2. **Object names are effectively free-form** (any UTF-8, up to ~1,024 bytes) but a constrained
   subset of characters is *recommended* for tooling, URL, and cross-platform compatibility.
3. **Sequential names are an anti-pattern at scale.** Date-stamped or incrementing keys concentrate
   traffic on one hot partition; providers recommend hash or entropy prefixes to distribute load.

**Key recommendation:** use a short, all-lowercase, hyphen-delimited convention for buckets
(`company-layer-region-account-env`), and a leftmost-general-to-rightmost-specific prefix layout
for object keys (`source/entity/year=YYYY/month=MM/day=DD/<name>`), adding an entropy prefix only
for workloads that need sustained high request rates on a single prefix.

---

## 2. Common Patterns & Standards Across Major Providers

### 2.1 AWS S3

**General purpose bucket naming rules** ([AWS docs](https://docs.aws.amazon.com/AmazonS3/latest/userguide/bucketnamingrules.html)):

- 3–63 characters.
- Lowercase letters, numbers, periods (`.`), and hyphens (`-`) only.
- Must begin and end with a letter or number.
- No two adjacent periods; must not be formatted as an IP address (`192.168.5.4`).
- Reserved prefixes: `xn--`, `sthree-`, `amzn-s3-demo-`. Reserved suffixes: `-s3alias`,
  `--ol-s3`, `.mrap`, `--x-s3`, `--table-s3`.
- Buckets with Transfer Acceleration must not contain periods.
- Names live in a **global namespace** (unique across all accounts within a partition). AWS now
  also offers an **account regional namespace** (`<prefix>-<accountid>-<region>-an`) where the
  name is guaranteed to stay under your control.
- Best practices: prefer the account regional namespace; append GUIDs for unpredictability; avoid
  periods (breaks virtual-host-style addressing over HTTPS); choose relevant names; don't delete
  buckets just to reuse names.

**Object keys** ([AWS docs](https://docs.aws.amazon.com/AmazonS3/latest/userguide/object-keys.html)):

- Up to 1,024 bytes of UTF-8; case-sensitive; keys are sorted lexicographically by UTF-8 bytes.
- Flat namespace; hierarchy is simulated with the `/` delimiter and prefixes.
- Safe characters: `a–z A–Z 0–9 ! - _ . * ' ( )`.
- Avoid: `& $ @ = ; : + , ?` (require URL-encoding), and especially `\ { } ^ } % \` " ] > [ ~ < # |`
  plus non-printable characters. Avoid `./` and `../` path segments and period-only segments.
- Keys must not end with a trailing dot or slash for best tool compatibility.

### 2.2 Google Cloud Storage

**Buckets** ([GCP docs](https://docs.cloud.google.com/storage/docs/buckets)):

- 3–63 characters (222 if dots are used, but each dot-separated component ≤ 63).
- Lowercase letters, numbers, dashes (`-`), underscores (`_`), and dots (`.`). No spaces.
- Must start and end with a number or letter; cannot be an IP address.
- Cannot start with `goog`; cannot contain `google` or close misspellings (e.g. `g00gle`).
- Names are **globally unique and publicly visible**; anyone can probe for a name's existence.
- Best practices: don't put user IDs, emails, project names/numbers, or PII in names; use names
  that are hard to guess (e.g. append a random suffix); if you delete a bucket, anyone can take
  the name — prefer emptying and retaining it.

**Objects** ([GCP docs](https://docs.cloud.google.com/storage/docs/objects)):

- Any sequence of valid Unicode characters; no carriage return or line feed.
- Cannot start with `.well-known/acme-challenge/`; cannot be `.` or `..`.
- Avoid: XML-1.0 control characters, `#` (clashes with gcloud version syntax), `[ ] * ?`
  (interpreted as wildcards; also invalid in Windows filenames), `: " < > |` (invalid in Windows
  filenames), and `./` / `../` path segments.
- Object names are more visible than object data (they appear in URLs and listings) — don't embed
  sensitive or PII information in them.

### 2.3 Azure Blob Storage

Azure uses three levels: **storage account → container → blob**.

**Storage accounts** ([Azure docs](https://learn.microsoft.com/en-us/azure/azure-resource-manager/management/resource-name-rules)):

- 3–24 characters; lowercase letters and numbers only; globally unique.

**Containers**:

- 3–63 characters; valid DNS name; lowercase letters, numbers, and hyphens only.
- Must start and end with a letter or number; **no consecutive hyphens**.
- Containers cannot be nested.

**Blobs**:

- 1–1,024 characters; case-sensitive; any URL-compatible characters, properly percent-encoded.
- ≤ 254 path segments in flat namespaces; ≤ 63 path segments (including account and container)
  with hierarchical namespace enabled.
- Avoid names ending in `.`, `/`, or `\`; avoid control characters and non-ASCII/private-use
  Unicode code points.

**Partitioning**: the blob partition key is `account + container + blob name`. Azure explicitly
warns against sequential names like `log20160101` and recommends a hash prefix or a seconds-value
placed early in the name to spread load.

### 2.4 S3-Compatible Providers

Providers that implement the S3 API generally inherit S3's bucket rules, sometimes with
provider-specific tweaks:

- **Backblaze B2** ([docs](https://www.backblaze.com/docs/cloud-storage-buckets)): native API
  allows 6–63 chars of letters/digits/hyphens/periods and forbids the `b2-` prefix; the
  **S3-Compatible API enforces AWS S3 rules** (3–63, lowercase/digits/dots/hyphens). Backblaze
  recommends lowercase letters + digits + hyphens only, ≥ 6 chars, no dots.
- **Cloudflare R2** ([docs](https://developers.cloudflare.com/r2/buckets/create-buckets/)):
  3–63 characters; lowercase `a–z`, `0–9`, and hyphens only; cannot begin or end with a hyphen.
- **MinIO** ([docs](https://minio.community/community/minio-object-store/administration/object-management.html)):
  S3-compatible; bucket names become subdomains when virtual-host lookup is enabled (`MINIO_DOMAIN`),
  so DNS-compatible names matter. Object names may not contain `\` or `:`; `/` creates folder
  structure. Lifecycle rules are S3-compatible.
- **Oracle OCI / Wasabi / IBM COS / others**: bucket naming follows the S3 rule set (3–63,
  lowercase, digits, hyphens; unique within a namespace), reinforcing the S3 convention as the
  de-facto industry standard.

### 2.5 Provider Comparison Table

| Aspect | AWS S3 | Google Cloud Storage | Azure Blob Storage | S3-compatible (R2/B2/MinIO) |
|---|---|---|---|---|
| Container scope of uniqueness | Global within partition (or per-account regional) | Global | Account (storage account); account name global | Global for B2/R2; per-deployment for MinIO |
| Container length | 3–63 | 3–63 (222 w/ dots) | 3–63 | 3–63 |
| Allowed chars | lowercase, digits, `.`, `-` | lowercase, digits, `-`, `_`, `.` | lowercase, digits, `-` (no consecutive) | lowercase, digits, `-` (R2); plus `.`/case (B2 native) |
| Reserved names | `xn--`, `sthree-`, `amzn-s3-demo-`, `-s3alias`, `.mrap`, etc. | `goog` prefix, contains `google` | — (DNS rules apply) | `b2-` (B2), S3 reserved set |
| Object max length | 1,024 bytes | effectively unlimited (practically long) | 1,024 chars | ~1,000–1,024 bytes |
| Object char set | any UTF-8 (subset recommended) | any Unicode (subset recommended) | any URL chars, percent-encoded | any UTF-8; `\` and `:` banned (MinIO) |
| Case sensitivity | Case-sensitive | Case-sensitive | Case-sensitive | Case-sensitive |
| Separator | `/` (prefix + delimiter) | `/` | `/` (virtual dirs) | `/` |
| Performance driver | Prefix partitioning | Key-range index / auto-scaling | Partition key = acct + container + blob | S3 semantics |

**Takeaway:** the intersection of all rule sets is a safe universal convention — **lowercase
letters, digits, and hyphens; 3–63 characters; start and end with alphanumeric** — which is valid
everywhere including DNS and HTTPS virtual-host addressing.

---

## 3. Best Practices for Consistency and Discoverability

### 3.1 Adopt a Written Naming Standard

A naming standard is governance, not style. It should be published, versioned, and enforced by
tooling (CI validators, Infrastructure-as-Code templates, SCP/IAM conditions). AWS's prescriptive
guidance for data lakes ([Defining Amazon S3 bucket and path names](https://docs.aws.amazon.com/prescriptive-guidance/latest/defining-bucket-names-data-lakes/welcome.html))
is the canonical template:

```
<companyname>-<layer>-<awsregion>-<awsaccount|uniqid>-<env>
s3://anycompany-raw-useast1-12345-dev/socialmedia/us/tb_products/year=2021/month=03/day=01/products_20210301.csv
```

The pattern decomposes into stable dimensions that each solve a specific problem:

| Component | Purpose | Example |
|---|---|---|
| `companyname` | Ownership / collision avoidance | `anycompany` |
| `layer` | Lifecycle & access tier | `landingzone`, `raw`, `stage`, `analytics` |
| `region` | Geo-locality & cost reporting | `useast1`, `euwest2` |
| `account/uniqid` | Cost allocation & uniqueness | `12345` |
| `env` | Environment isolation | `dev`, `test`, `prod` |

Separate buckets per data layer are recommended because **versioning, encryption, access, and
archiving requirements vary by layer and are configured at the bucket level, not the prefix level**.

### 3.2 Object Key Layout: General-to-Specific, Left to Right

Order key elements from the most general to the most specific so that prefix-based listing,
IAM policies, and lifecycle rules select meaningful slices:

```
source / business_unit / table / year=YYYY / month=MM / day=DD / <file>.<ext>
```

- Use **key-value partition pairs** (`year=2026`, `month=09`) rather than bare values (`2026/09`)
  so query engines (Athena, BigQuery, Spark `MSCK REPAIR TABLE`) can discover partitions
  automatically.
- Keep timestamps in `YYYY-MM-DD` / `YYYY-MM-DDTHH:MM:SS` (ISO-8601) form so lexicographic
  ordering equals chronological ordering.
- Use lowercase everywhere, including in keys — sorting is case-sensitive and mixed case splits
  related keys.
- Reserve the rightmost segment for the file name; keep it stable and descriptive of content.

### 3.3 Consistency Rules of Thumb

- One delimiter (`/`) everywhere; never mix `\`.
- One convention per account/tenant, not per team.
- Avoid redundancy: don't repeat the bucket's layer/env in every object key.
- Choose names that survive tooling: no spaces, no non-ASCII in keys that must pass through
  Windows tooling, no trailing dots/slashes.
- Document the standard and update it when a new source or data domain is added.

### 3.4 Discoverability

- **Prefixes are the index**: structure keys around how data will be queried, not how the source
  produced it.
- **Delimited listing**: use the `/` delimiter in `ListObjectsV2`/`list` calls so tools render
  folders; avoid deep hierarchies that slow recursive listing.
- **Metadata + tags**: put machine-readable facts in object metadata (`x-amz-meta-*`,
  `x-goog-meta-*`) and use cost-allocation tags on buckets instead of encoding every attribute in
  the name.
- **Catalog integration**: name keys so external catalogs (Glue/Hive-style partitions) and
  schema-discovery tools can ingest them without renaming.
- **Lexicographic awareness**: because keys sort by bytes, uppercase letters sort before
  lowercase, and non-ASCII sorts after ASCII — a single casing policy keeps listings predictable.

---

## 4. Recommended Naming Rules for Buckets and Objects

### 4.1 Bucket Rules (portable across providers)

| Rule | Why |
|---|---|
| 3–63 characters, lowercase letters, digits, hyphens only | Valid everywhere, DNS-safe, HTTPS virtual-host safe |
| Start and end with a letter or number | Required by S3/GCP/Azure/Backblaze |
| No dots | Avoids TLS/virtual-host issues; only use for static web hosting |
| No underscores | Invalid in R2, discouraged by B2, breaks DNS compatibility |
| No IP-address-shaped names, no consecutive hyphens, no `xn--`/`goog`-type reserved tokens | Explicitly forbidden |
| Globally unique assumption | Handles eventual global namespaces; handle 409/BucketAlreadyExists in code |
| No PII, no account IDs, no project names, no secrets in the name | Names are public and probeable |
| Prefer unpredictable names (GUID/random suffix) for shared-namespace buckets | Prevents name squatting and enumeration |
| Retain empty buckets instead of deleting them | Prevents name reuse/squatting after deletion |

### 4.2 Object Key Rules

| Rule | Why |
|---|---|
| UTF-8, ≤ 1,024 bytes; lowercase | Max length caps at S3; lowercase keeps sorting sane |
| Use `/` as the only hierarchy delimiter | Simulates folders across all providers |
| Safe characters only: `a–z 0–9 - _ . /` | Compatible with every tool, URL, and OS |
| No `./`, `../`, `.`/`..` segments, trailing dots or slashes | Prevents path-normalization bugs and Windows issues |
| No `#`, `%`, `?`, `*`, `[ ]`, `:`, `<`, `>`, `|`, `"`, control chars | Reserved/wildcard/URL-encoding conflicts |
| Key-value date partitions (`year=/month=/day=`) | Catalog- and lifecycle-friendly |
| ISO-8601 timestamps | Lexicographic order = time order |
| No secrets, tokens, or PII in keys | Keys appear in URLs, logs, and listings |

### 4.3 A Minimal Convention You Can Adopt Today

```
Buckets:  <company>-<layer>[-<env>]
Objects:  <domain>/<entity>/<year>=<YYYY>/<month>=<MM>/<day>=<DD>/<file>.<ext>
```

Concrete example:

```
company-raw-prod
  socialmedia/us/tb_products/year=2026/month=09/day=17/products_20260917.csv
```

For high-throughput append workloads, insert an entropy prefix between the domain and entity:

```
company-raw-prod
  socialmedia/a3f/us/tb_products/year=2026/month=09/day=17/products_20260917.csv
```

---

## 5. Scalability Considerations

### 5.1 Sequential Keys Create Hot Partitions

All three hyperscale providers partition data by name range and load-balance across servers. A
monotonically increasing prefix (timestamp, auto-increment ID) funnels all new writes onto one
partition:

- **AWS S3**: historically ~3,500 PUT / 5,500 GET per second per partitioned prefix; S3
  auto-partitions in response to sustained load but may return transient HTTP 503 (`Slow Down`)
  while scaling ([docs](https://docs.aws.amazon.com/AmazonS3/latest/userguide/optimizing-performance.html)).
- **Google Cloud**: sequential names slow the key-range index auto-scaling and can cause
  temporarily elevated latency/error rates ([request rate docs](https://docs.cloud.google.com/storage/docs/request-rate)).
- **Azure Blob**: append-only/prepend-only partition keys concentrate traffic on one partition
  server and can exceed scalability targets ([partition docs](https://learn.microsoft.com/en-us/azure/storage/blobs/storage-performance-blob-partitions)).

### 5.2 Mitigations

| Technique | How it works | When to use |
|---|---|---|
| **Entropy/hash prefix** | Prepend a short hash (e.g. 1–4 hex chars of MD5) or a randomized bucket-id before the logical key | Sustained high request rate on one logical prefix |
| **Reverse the identifier** | Reverse timestamp digits (`ssyyyymmdd`) or domain components so leading chars fan out across partitions | Append-only logs, event ingestion |
| **Write to many prefixes in parallel** | Spread bulk uploads/deletes across prefixes rather than one sequential run | Bulk ETL, data migration |
| **Ramp gradually** | Ramp request rate slowly when opening a new prefix; let auto-scaling catch up | Cold-start on a new partition |

AWS notes the randomness need not be the first character: entropy *after* a common prefix is
effective for that prefix, but write-rate benefits reset each hour and across prefixes. For
listing-heavy analytics workloads, keep the hash *shallow* (1–2 chars) so parallel prefix-list
calls can still enumerate data.

### 5.3 Newer Architectures Remove the Entropy Need

- **S3 directory buckets / S3 Express One Zone** maintain a true hierarchical namespace and
  auto-distribute load — AWS explicitly recommends **against** entropy prefixes there because they
  create sparse, slow directories. Prefer dense directories (thousands of entries) and delimited
  list operations.
- **GCP hierarchical namespace** buckets provide file-system-like folders that improve
  performance for large-scale, file-oriented workloads.
- **Azure ADLS Gen2** hierarchical namespace changes the path-segment limit (63 instead of 254)
  but gives atomic directory operations.

### 5.4 Listing Performance

- Use delimited list calls (`delimiter=/`) so only the requested directory level is traversed.
- Avoid very deep key hierarchies in flat-namespace buckets: each `ListObjectsV2` page crosses
  lexicographic ranges.
- For large datasets, prefer many files under a shallow, well-partitioned tree over a deeply
  nested one.

---

## 6. Security Considerations

### 6.1 Names Are Public and Probeable

Bucket names are globally visible and enumerable by existence probes; object names appear in
URLs, listings, and logs. Providers are explicit about the consequences:

- **GCP**: "Bucket names are publicly visible. Don't use user IDs, email addresses, project
  names, project numbers, or any PII in bucket names because anyone can probe for the existence
  of a bucket."
- **AWS**: "Don't include sensitive information in the bucket name. The bucket name is visible in
  the URLs that point to the objects in the bucket."
- **GCP (objects)**: "Object names are more broadly visible than object data… avoid using
  sensitive information as part of bucket or object names." Prefer meaningless codenames
  (`somemeaninglesscodename-prod`) and move sensitive metadata into headers (`x-goog-meta-*`)
  rather than encoding it in names.

### 6.2 Predictability Enables Enumeration and Hijacking

- Predictable or sequential names let attackers enumerate objects and infer scale/business
  activity.
- In a shared global namespace, **deleting a bucket surrenders the name**; another account can
  recreate it and potentially receive traffic intended for yours. Mitigations: keep buckets
  (empty them instead), use unpredictable/GUID names, and use AWS's account regional namespace
  (`...-an`) where names can never be re-created by others.
- Verify ownership with bucket-owner conditions when names could collide across accounts.

### 6.3 Naming Rules That Reduce Risk

| Rule | Security effect |
|---|---|
| No PII, IDs, emails, project numbers, or secret values in names | Prevents information leakage through URLs/logs/listings |
| Unpredictable (random-suffix) bucket names | Prevents existence probing and name squatting |
| Entropy prefix for multi-tenant object keys | Obscures sequential user IDs / counts |
| No "sensitive"/"secret"/"internal" markers that advertise value | Reduces targeting of high-value data |
| Keep ACLs disabled; use policies + bucket owner enforced | Prevents per-object ACL confusion and leakage |
| Enforce the naming standard via SCP/IAM conditions (e.g. `s3:x-amz-bucket-namespace`) | Prevents misconfigurations at creation time |

### 6.4 Data Protection Complements Naming

Naming cannot protect content — pair the convention with default encryption (SSE-S3/SSE-KMS,
GCP CMEK, Azure SSE), blocking public access by default, versioning, Object Lock/WORM for
immutable retention, and sensitivity scanning (e.g. Macie) for PII inside objects.

---

## 7. Lifecycle Management Considerations

### 7.1 Lifecycle Rules Are Prefix-Based

S3 lifecycle policies, GCP lifecycle conditions, and Azure lifecycle management rules act on
**prefixes** (and object metadata like size/age). Therefore the object key layout *is* the
lifecycle map:

- Date partitions (`year=2026/month=09/day=17/`) let one rule transition or expire an entire
  day/month/quarter with a single prefix filter.
- Layered buckets (`raw`, `stage`, `analytics`) allow *bucket-level* policies: enable versioning
  and tight retention on `raw`, longer archiving on `analytics`.
- Keep the archive/retention dimension in the *prefix*, not in per-object metadata, so rules
  remain trivially declarative.

### 7.2 Versioning Interacts with Names

- Versioning is a bucket-level setting — the name can't scope it. Enabling versioning on a bucket
  affects every prefix, which is why AWS recommends per-layer buckets rather than one mega-bucket.
- With versioning, deletes are soft until a version-removal rule expires them; retention costs
  grow with the number of versions kept, so tune the `NoncurrentVersionExpiration`/transition
  rules to the layer.

### 7.3 Naming-Friendly Lifecycle Checklist

| Item | Recommendation |
|---|---|
| Abort incomplete multipart uploads | Set a global rule (e.g. > 7 days) on every write-heavy bucket |
| Tier transitions | Put `Glacier`/`Coldline`/`Cool`-eligible data under its own prefix so one rule targets it |
| Expiration | Use date-partition prefixes so retention windows align with the key structure |
| Immutable data | Object Lock / WORM is bucket- or object-level — keep regulated data in dedicated buckets |
| Replication | Keep replicated data in a separate `-replica`/region-suffixed bucket, not a prefix, to avoid recursive replication loops |

### 7.4 Renaming Is Not Free

Buckets cannot be renamed; objects can only be renamed by copy+delete. This is the strongest
argument for getting the standard right early: a poorly named schema requires a full data
migration, while a good one lets lifecycle rules and reports be derived from the name structure
for the system's lifetime.

---

## 8. Recommended Convention Summary

```
Bucket template:   <org>-<layer>[-<region>][-<env>]
Layer values:      landingzone | raw | stage | analytics | archive | backup
Object template:   <source>/<business-unit>/<entity>/<year>=<YYYY>/<month>=<MM>/<day>=<DD>/<file>.<ext>
Entropy (optional): <hash2-4>/<source>/...
```

Rules checklist before creating any bucket or writing any key:

- [ ] 3–63 chars, lowercase, `a–z 0–9 -` only; starts/ends alphanumeric
- [ ] No dots, underscores, IP shapes, consecutive hyphens, reserved prefixes/suffixes
- [ ] No PII, account IDs, secrets, or high-value markers
- [ ] Layer + env encoded at bucket level; domain/entity/date at key level
- [ ] Key-value date partitions; ISO-8601; leftmost = most general
- [ ] `/` only delimiter; safe character set only
- [ ] Entropy prefix added for known high-throughput append workloads only
- [ ] Versioning/lifecycle/encryption decided per layer, then written into the standard

---

## 9. References

1. [AWS: General purpose bucket naming rules](https://docs.aws.amazon.com/AmazonS3/latest/userguide/bucketnamingrules.html)
2. [AWS: Namespaces for general purpose buckets](https://docs.aws.amazon.com/AmazonS3/latest/userguide/gpbucketnamespaces.html)
3. [AWS: Naming Amazon S3 objects (object keys)](https://docs.aws.amazon.com/AmazonS3/latest/userguide/object-keys.html)
4. [AWS: Optimizing performance with S3](https://docs.aws.amazon.com/AmazonS3/latest/userguide/optimizing-performance.html)
5. [AWS Prescriptive Guidance: Defining Amazon S3 bucket and path names for data lake layers](https://docs.aws.amazon.com/prescriptive-guidance/latest/defining-bucket-names-data-lakes/welcome.html)
6. [AWS: Common general purpose bucket patterns](https://docs.aws.amazon.com/AmazonS3/latest/userguide/common-bucket-patterns.html)
7. [AWS: Security best practices for S3](https://docs.aws.amazon.com/AmazonS3/latest/userguide/security-best-practices.html)
8. [AWS: S3 object key naming patterns (re:Post)](https://repost.aws/knowledge-center/s3-object-key-naming-pattern)
9. [AWS Blog: Optimizing S3 for high concurrency in distributed workloads](https://aws.amazon.com/blogs/big-data/optimizing-amazon-s3-for-high-concurrency-in-distributed-workloads/)
10. [AWS News Blog: S3 Performance Tips & Tricks](https://aws.amazon.com/blogs/aws/amazon-s3-performance-tips-tricks-seattle-hiring-event/)
11. [GCP: About Cloud Storage buckets](https://docs.cloud.google.com/storage/docs/buckets)
12. [GCP: About Cloud Storage objects](https://docs.cloud.google.com/storage/docs/objects)
13. [GCP: Best practices for Cloud Storage](https://docs.cloud.google.com/storage/docs/best-practices)
14. [GCP: Request rate and access distribution guidelines](https://docs.cloud.google.com/storage/docs/request-rate)
15. [Azure: Naming and referencing containers, blobs, and metadata](https://learn.microsoft.com/en-us/rest/api/storageservices/naming-and-referencing-containers--blobs--and-metadata)
16. [Azure: Resource naming rules (storage)](https://learn.microsoft.com/en-us/azure/azure-resource-manager/management/resource-name-rules)
17. [Azure: Optimize blob partitions](https://learn.microsoft.com/en-us/azure/storage/blobs/storage-performance-blob-partitions)
18. [Backblaze: Cloud Storage buckets (naming rules)](https://www.backblaze.com/docs/cloud-storage-buckets)
19. [Cloudflare: Create R2 buckets](https://developers.cloudflare.com/r2/buckets/create-buckets/)
20. [MinIO: Object management](https://minio.community/community/minio-object-store/administration/object-management.html)
