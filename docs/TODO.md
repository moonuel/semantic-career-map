# Backlog

Items I'm still working on.

## Pipeline & Architecture

- "Formalization of the pipeline in multiple stages in backend modules may be helpful in the short term." — [2026-07-23](dev-log/2026.md#july-23-2026)
- "An architecture document to accommodate modularity and future expansion would be good." — [2026-07-23](dev-log/2026.md#july-23-2026)
- ~~"Boilerplate removal should probably be its own pipeline step." — [2026-07-27](dev-log/2026.md#july-27-2026)~~ **landed as the production cleaning stage** — [2026-10-23](dev-log/2026.md#october-23-2026)
- "It would fall under pre-processing - the boilerplate-cleaned text minus the raw text should always be empty (no hallucination)" — [2026-07-27](dev-log/2026.md#july-27-2026)
- "wire in the remaining evaluation jobs" — [2026-10-25](dev-log/2026.md#october-25-2026)

## Deployment & Ops

- "consider automatic redeploy on git tag push, or keeping it manual" — [2026-10-25](dev-log/2026.md#october-25-2026)
- "push the embedding vector job to production and schedule it in a batch job as well" — [2026-10-25](dev-log/2026.md#october-25-2026)

## Data & Versioning

- "The better first step is to formalize and capture this existing structure I think. SQL database perhaps, for the raw and important intermediate representations (like cleaned data and engineered features)." — [2026-08-25](dev-log/2026.md#august-25-2026)
- "Planning for the future, ML systems require both data and experiment versioning. From Huyen (DSML), you want to capture the entire computational experiment that produced the data." — [2026-08-25](dev-log/2026.md#august-25-2026)
- ~~"Data versioning will become crucial. Use S3 object store" — [2026-09-16](dev-log/2026.md#september-16-2026)~~ Done
- ~~"One blocker is the limited data. Will plan to generate synthetic data, in lieu of more expensive and true data engineering techniques." — [2026-09-16](dev-log/2026.md#september-16-2026)~~ Used real data scrapers
- Scheduled data scraping
- ~~MLflow-style run lineage: dataset versions + run manifests~~ — **landed for the cleaning+eval branch** [2026-10-23](dev-log/2026.md#october-23-2026): `manifests/runs|datasets` live with `pipeline_version` (git sha) + `configuration_hash`, per-run flush checkpoints, per-dataset manifests for artifacts/golden/results. Still open from the original thread: no **environment snapshot** (runtime-dependency digest — two runs on the same commit aren't provably the same code), and the hard-kill window the 5.1 lineage verify caught live (a SIGKILLed run leaves part + checkpoint but no run record — reconstrucible, not complete). The sha pins the checked-out commit, not the working tree — `git describe --dirty` is the cheap fix if it ever matters.
- Next cleaning variant: the **organizational-context field** — split company-context lines off the cleaned text as a second artifact column so over-deletion has a measured home instead of a 5% flag — and a fuller golden set beyond the 39 references. — [2026-10-23](dev-log/2026.md#october-23-2026)
- Clean-tier promotion gate — the eval-before-promote rule is runbook prose only; no enforced check ties a `:clean` push to an eval run or pinned-tag promotion. Filed as moonuel/semantic-career-map#15, 2026-10-25; sibling lineage gap (moving-tag runs record tag names, ATS Lambda records `dev`) filed as #16; the active-pipeline-version pointer (switch scheduled tiers between experimental variant/version branches) filed as #17.
- The four residual AWS-push risks each filed: egress/security reconsideration path (#18), host-scoped flock → co-run decision (#19), LLM fallback posture ratification (#20), backlog daily metric without orchestrator (#21). — 2026-10-26

## Taxonomy & Clustering Validation

- "One of the most immediate core ideas to validate is whether the role taxonomy is effective for clustering similar postings." — [2026-07-23](dev-log/2026.md#july-23-2026)
- "Likely I will need more postings (or synthetic data) to benchmark it properly." — [2026-07-23](dev-log/2026.md#july-23-2026)
- "I also want to validate whether the role taxonomy provides a better basis for judging clustering" — [2026-07-25](dev-log/2026.md#july-25-2026) *(Exp 006)*
- "Something like generating job labels from the taxonomy" — [2026-07-25](dev-log/2026.md#july-25-2026)
- "To do this at scale is still something I can't imagine doing without some LLMs and evals." — [2026-07-25](dev-log/2026.md#july-25-2026)
- "I might stick to small samples for now and solve the scale problem later." — [2026-07-25](dev-log/2026.md#july-25-2026)
- AnyJev or similar for multi-label classification? The DSML scalable architecture is to have N classifiers for N labels 

## Embedding & Clustering Behavior

- "My working assumption is that the embedding model clusters based on semantic similarity, not necessarily similarity of text" — [2026-07-25](dev-log/2026.md#july-25-2026)
- "But this assumption needs to be tested. I guess this experiment addresses that." — [2026-07-25](dev-log/2026.md#july-25-2026) *(tested in Exp 005)*
- "One goal is to compare clustering results between
    - Cleaned postings, with/without the contextual information about the role within the organization (stored as a separate field)
    - Engineered features, maybe based on the role taxonomies, maybe something like generated description strings based on the cleaned data" — [2026-07-25](dev-log/2026.md#july-25-2026) *(Exp 005)*
- "build embedding vector key builder and pipeline code" — [2026-10-25](dev-log/2026.md#october-25-2026)

## Features & Demo

- "Ohhh you know what would be fire. Have the demo use some variations of my resume, embed them all, and see how different variations map to different job groups." — [2026-07-26](dev-log/2026.md#july-26-2026)
- "I've done so much more than what can fit on a single resume sheet, this would be such a flex to also demonstrate my breadth" — [2026-07-26](dev-log/2026.md#july-26-2026)
- "map out potential live interactive visualization tools" — [2026-10-25](dev-log/2026.md#october-25-2026)

## Ideas & Future Directions

- "An interesting future direction could be some kind of counterfactual analysis
    - Suppose a user (me) embeds their resume and clicks on a job posting.
    - Calculate the set difference of words and run the skill extraction pipeline.
    - Add those skills back to the resume to see if it improves match scores.
    - This can be used to identify critical missing skills that are hindering job matching.
    - Something like a personalized recommendation system for skill development and career path optimization." — [2026-08-03](dev-log/2026.md#august-03-2026)
- "A k-NN clustering metric can be defined by taking the k-nearest neighbours to any given job posting \(P\) and counting whether it shares a tag with \(P\)." — [2026-08-03](dev-log/2026.md#august-03-2026)
- "This requires the tag generation using the [taxonomy](research-reports/003-ml-ai-responsibility-taxonomy.md) to assign consistent labels to postings." — [2026-08-03](dev-log/2026.md#august-03-2026)
- My dream is to have a more transparent "AI search" feature. Something like, skill extraction and description of responsibilities from the postings. Then people describe their experiences or ideal job, a similar skill extraction and capability description is generated (editable by humans) and then rankings are generated from that. The purpose would be job exploration, not connection to live postings. (2026-10-13)

## Reading

- "Some reading of standard feature engineering will probably be good." — [2026-07-23](dev-log/2026.md#july-23-2026)
- "Then some reading of standard feature engineering will probably be good." — [2026-07-25](dev-log/2026.md#july-25-2026)
- "Read about Siamese network / adversarial / dual-decoder models for better semantic partitioning" — [2026-07-29](dev-log/2026.md#july-29-2026)
