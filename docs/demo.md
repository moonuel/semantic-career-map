# Demo

Interactive map of the current job-posting corpus. Each point is a posting placed by UMAP over its MiniLM embedding, so nearby postings describe similar work.

<iframe
  src="map/index.html"
  title="Semantic job map"
  style="width: 100%; height: 80vh; min-height: 640px; border: 0; border-radius: 8px; background: #fff;"
  loading="lazy">
</iframe>

<p style="text-align: right;">
  <a href="map/index.html" target="_blank" rel="noopener">Open the map full screen</a>
</p>

!!! info "What this demo is"
    Hover a point for the job title and employer. Click a point for the posting details. The map is generated offline by `scripts/map.py`; the resume-to-role search interface is still planned for Phases 4 and 7 of the [project scope](project.md).

## How to read the map

- **Distance** indicates semantic similarity: postings close together use similar language and responsibilities.
- **Hover** a point to see its title and employer.
- **Click** a point to open the posting details panel.
- The current artifact covers 9,354 postings from the MiniLM generation. The map loads its Deck.GL runtime from a CDN, so an internet connection is required.

<!-- ## Example query

**Query:** "I'm interested in data science and have focused a lot of data engineering, feature engineering, and unsupervised/weakly-supervised learning tasks where there is no ground truth usable to train models. I'm not really that interested in Agentic AI, but am willing to use it. I like high-dimensional linear algebra, optimizing algorithm performance using multithreading and low-level programming concepts, and hard engineering problems. I lack experience with model deployment, PyTorch/deep learning, and model maintenance, but can easily obtain it due to my background in Mathematics. What kind of jobs would be a good fit for me?"

**Planned behavior:** Map the query to job families in the interactive UMAP plot, rank the best-match roles, and list the required skills. Roles with different titles but overlapping skill requirements should both rank highly. -->
