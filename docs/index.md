---
title: Introduction
---

<div style="text-align: center; padding: 3em 0 0.5em;" markdown="1">

# AI/ML Career Map

A semantic search tool for mapping your interests and experience to actual job families in AI and machine learning, based on skills and not title.

</div>

!!! info "Currently working on:"

    - RAG context engineering tools for better semantic clustering
    - Benchmarking of clustering methods and metrics 
    - Evaluation and data monitoring layers
    - Personality quiz for the user, supporting interactive retrieval based on interests

<!-- --- -->

<div class="grid cards" markdown>

-   **Problem**

    ---

    Traditional job searches struggle when *responsibilities don't match the title*. 
    
    A "Machine Learning Engineer" won't match an "Applied Scientist" role, even when the work is identical.

    **How do you find the jobs that are right for you?**

-   **Solution**

    ---

    Dense text embeddings provide *semantic clustering of job postings* based on responsibilities.

    Clustering and similarity rankings *map your interests to job families* based on skills and experience.

    Dimension-reduction provides *intuitive 2D visualizations* of high-dimensional spaces.

<!-- -   :material-pipe:{ .lg .middle } **System** -->

<!-- --- -->

<!-- Embedding pipeline → vector index → similarity search → API service. Precomputed embeddings with L2 normalization enable dot-product similarity at query time — no GPU required. -->

<!-- -   :material-chart-bar:{ .lg .middle } **Evaluation** -->

<!-- --- -->

<!-- Quantitative retrieval benchmarks: self-retrieval (100%), separation gap (+0.0493, LLM-cleaned). pytrec_eval integration planned for Precision@5, MRR, NDCG. -->

</div>

<div style="text-align: center; padding: 0.5em 0 0.5em;" markdown="1">

[Launch Demo](demo.md){ .md-button .md-button--primary }
<!-- [View Source Code](https://github.com/moonuel/semantic-career-map){ .md-button .md-button--warn } -->
</div>

<div style="text-align: center; padding: 0 0 2em;" markdown="1">

![Python](https://img.shields.io/badge/Python-3.14+-blue)
![Amazon Titan V2](https://img.shields.io/badge/Embeddings-Amazon%20Titan%20V2-green)
![Amazon Bedrock](https://img.shields.io/badge/Inference-Amazon%20Bedrock-orange)
![AWS](https://img.shields.io/badge/Pipeline-S3%20%2B%20Glue%20%2B%20Athena-232F3E?logo=amazonaws&logoColor=white)
![UMAP-learn](https://img.shields.io/badge/UMAP--learn-green)
![Docker](https://img.shields.io/badge/Pipeline-Docker-blue?logo=docker&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-planned-lightgrey)
![CI](https://img.shields.io/badge/CI-GitHub%20Actions-black?logo=githubactions&logoColor=white)

</div>
