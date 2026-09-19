# Backlog

Items I'm still working on.

## Pipeline & Architecture

- "Formalization of the pipeline in multiple stages in backend modules may be helpful in the short term." — [2026-07-23](dev-log/2026.md#july-23-2026)
- "An architecture document to accommodate modularity and future expansion would be good." — [2026-07-23](dev-log/2026.md#july-23-2026)
- "Boilerplate removal should probably be its own pipeline step." — [2026-07-27](dev-log/2026.md#july-27-2026) *(realized in Exp 002/003)*
- "It would fall under pre-processing - the boilerplate-cleaned text minus the raw text should always be empty (no hallucination)" — [2026-07-27](dev-log/2026.md#july-27-2026)

## Data & Versioning

- "The better first step is to formalize and capture this existing structure I think. SQL database perhaps, for the raw and important intermediate representations (like cleaned data and engineered features)." — [2026-08-25](dev-log/2026.md#august-25-2026)
- "Planning for the future, ML systems require both data and experiment versioning. From Huyen (DSML), you want to capture the entire computational experiment that produced the data." — [2026-08-25](dev-log/2026.md#august-25-2026)
- "Data versioning will become crucial. Use S3 object store" — [2026-09-16](dev-log/2026.md#september-16-2026)
- "One blocker is the limited data. Will plan to generate synthetic data, in lieu of more expensive and true data engineering techniques." — [2026-09-16](dev-log/2026.md#september-16-2026)

## Taxonomy & Clustering Validation

- "One of the most immediate core ideas to validate is whether the role taxonomy is effective for clustering similar postings." — [2026-07-23](dev-log/2026.md#july-23-2026)
- "Likely I will need more postings (or synthetic data) to benchmark it properly." — [2026-07-23](dev-log/2026.md#july-23-2026)
- "I also want to validate whether the role taxonomy provides a better basis for judging clustering" — [2026-07-25](dev-log/2026.md#july-25-2026) *(Exp 006)*
- "Something like generating job labels from the taxonomy" — [2026-07-25](dev-log/2026.md#july-25-2026)
- "To do this at scale is still something I can't imagine doing without some LLMs and evals." — [2026-07-25](dev-log/2026.md#july-25-2026)
- "I might stick to small samples for now and solve the scale problem later." — [2026-07-25](dev-log/2026.md#july-25-2026)

## Embedding & Clustering Behavior

- "My working assumption is that the embedding model clusters based on semantic similarity, not necessarily similarity of text" — [2026-07-25](dev-log/2026.md#july-25-2026)
- "But this assumption needs to be tested. I guess this experiment addresses that." — [2026-07-25](dev-log/2026.md#july-25-2026) *(tested in Exp 005)*
- "One goal is to compare clustering results between
    - Cleaned postings, with/without the contextual information about the role within the organization (stored as a separate field)
    - Engineered features, maybe based on the role taxonomies, maybe something like generated description strings based on the cleaned data" — [2026-07-25](dev-log/2026.md#july-25-2026) *(Exp 005)*

## Features & Demo

- "Ohhh you know what would be fire. Have the demo use some variations of my resume, embed them all, and see how different variations map to different job groups." — [2026-07-26](dev-log/2026.md#july-26-2026)
- "I've done so much more than what can fit on a single resume sheet, this would be such a flex to also demonstrate my breadth" — [2026-07-26](dev-log/2026.md#july-26-2026)

## Ideas & Future Directions

- "An interesting future direction could be some kind of counterfactual analysis
    - Suppose a user (me) embeds their resume and clicks on a job posting.
    - Calculate the set difference of words and run the skill extraction pipeline.
    - Add those skills back to the resume to see if it improves match scores.
    - This can be used to identify critical missing skills that are hindering job matching.
    - Something like a personalized recommendation system for skill development and career path optimization." — [2026-08-03](dev-log/2026.md#august-03-2026)
- "A k-NN clustering metric can be defined by taking the k-nearest neighbours to any given job posting \(P\) and counting whether it shares a tag with \(P\)." — [2026-08-03](dev-log/2026.md#august-03-2026)
- "This requires the tag generation using the [taxonomy](research-reports/003-ml-ai-responsibility-taxonomy.md) to assign consistent labels to postings." — [2026-08-03](dev-log/2026.md#august-03-2026)

## Reading

- "Some reading of standard feature engineering will probably be good." — [2026-07-23](dev-log/2026.md#july-23-2026)
- "Then some reading of standard feature engineering will probably be good." — [2026-07-25](dev-log/2026.md#july-25-2026)
- "Read about Siamese network / adversarial / dual-decoder models for better semantic partitioning" — [2026-07-29](dev-log/2026.md#july-29-2026)
