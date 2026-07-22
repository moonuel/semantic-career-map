# Experiment Notes

## July 22, 2026: Baseline embedding pipeline

Results for baseline implementation of the shared semantic space using SentenceTransformers and FAISS vector database. 

AI was used for rapid prototyping. 

### Experimental design
- 27 job postings were selected from LinkedIn (and 1 from Indeed) and their content pasted into markdown files in `data/selected-job-postings`.
- A baseline embedding and visualization was implemented in `parse_and_embed_quickstart.py`, which extracts key metadata from each posting with the following schema:
```
    "id": posting_id,
    "title_raw": title,
    "company": company,
    "source_file": filepath.name,
    "role_category": role_category,
    "sections": sections,
    "raw_full_text": text,
```
- The raw full text was embedded using `all-MiniLM-L6-v2` into a 384-dimensional dense embedding space, and normalized to the unit sphere.
- PCA, UMAP, and t-SNE visualizations were used, coloured by role category and annotated by company and job title.

### Results

#### PCA
![](../data/plots/raw_pca_baseline.png)

- PCA shows a 2D subspace of the high-dimensional embedding space spanning the first two orthogonal axes holding the greatest variance. 
    - All points are projected down to this subspace. 
- There appears to be some weak clustering of the ML Engineer postings (orange).
- Data scientist roles (blue) are spread all over the place, varying most among the first principal component. 
    - Embedding dimensions are meaningless, so there are no useful factors to extract here.
- "Other" (brown) roles show signs of weak clusetering around the origin, but they are not well-defined enough for meaningful analysis.
- AI Engineer (green) roles also show signs of clustering around (PC1, PC2) = (0.15, -0.1)

- Data Scientist roles are known (anecdotally, from LinkedIn) to have a lot of overlap with ML Engineer roles, which may explain their large variance across the PCA embedding. 
- **This directly motivates the need for more detailed taxonomies for each posting and derive job categories separately from the job posting.**

#### UMAP
![](../data/plots/raw_umap_baseline.png)
- UMAP (uniform manifold approximation and projection) finds a high-dimensional Riemannian manifold "close" to the data and projects it down to 2 dimensions.
    - By using Riemannian manifolds, the hope is to preserve local neighborhood structure.
- The clustering results are similar. 
- AI Engineer roles (green) appear to be clustered. These roles involve agentic work, which might carry significantly different semantic meanings from other roles. 
- Research roles (red) are very far apart. 
    - The eBay Applied Researcher 1 listing describes search ranking as a primary responsibility.
    - Huawei's AI/ML Researcher mentions multimodal model development and integration. 
    - It might be typical for research roles to be very different in scope. 
- Interestingly, Data Scientist roles (blue) appear to have more meaningful clustering structure. 
    - They vary most along the second UMAP axis (locally orthogonal?). 
    - Again, embedding space dimensions are meaningless so we cannot infer semantic relationships from them.
- The rest (ML Engineer, Other) show weak clustering. 
- More sample points and a more refined taxonomy may improve interpretation results.

#### t-SNE
![](../data/plots/raw_tsne_baseline.png)
- t-SNE (t-distributed stochastic neighbour embedding) calculates the pair-wise Euclidean distance between every point in the embedding space and uses it to construct a Gaussian distribution for each point's local neighborhood, capturing local structure.
    - It then uses the probability distribution to map the high-dimensional points down to a lower one, while choosing lower-dimensional one through minimizing the Kullback-Leibler divergence.
    - The resulting visualization preserves local structure but distorts global relationships and complicates interpretation.
- This visualization appears weakest of the three.
- Data Scientist roles (blue) surprisingly exhibit the greatest clustering, being arranged roughly in the perimeter of a circle centered around (0,-40). 
- Actually, all other roles appear scattered in rough clusters with around the same diameter. 
    - The only exception are the Applied/Research roles, which again show large separation.
    - Interestingly, the closest neighbour to the eBay search ranking role is the Pinterest ML Engineer role, which also involves search ranking. 
    - Not sure if the HelloFresh ML Engineer role (Agentic AI) bears much semantic similarity to the Huawei researcher (Multi-modal AI) role though. 


### Discussion
- One major limitation of this experiment is that the embedded text is not engineered at all. 
    - There are several passages of text not related to the work responsibilities, for example the "About Us" or "Company Culture" sections that discuss company values, benefits, and general information unrelated to the specific job duties being advertised.
    - The presence of these non-essential sections may dilute the semantic signal of the actual job responsibilities, potentially causing the embeddings to cluster based on company culture rather than technical role requirements.
    - This suggests that a preprocessing step to extract only the "Responsibilities" and "Requirements" sections would likely improve the discriminative power of the embeddings.
- Another limitation is the small sample size of 27 postings, which may not capture the full diversity of AI/ML roles across different industries and company sizes, limiting the generalizability of the findings.
    - We would require either **many more postings scraped from the internet, or simulated job postings generated from the real ones**, or something like that. 
- A smaller limitation is the use of a single embedding model, but feature engineering is likely to have a larger impact than trying different models at this time. 
- One final limitation is the spread of actual responsibilities across similar or (seemingly) unrelated job titles. 
    - This motivates the generation of a more granular taxonomy that can better capture the semantic relationships. 

- **3-dimensional visualizations** may also be interesting at some point, but likely unnecessary at the present state of the work, especially since embedding dimensions are not semantically meaningful.


### Questions
- Conceptual underpinnings of UMAP and t-SNE are modest and not complete. Not quite necessary at this time but a notable gap for future development.


### Follow-ups 
- Taxonomy of AI/ML roles and responsiblities produced at `docs/research-reports/ml-ai-responsibility-taxonomy.md`
    - AI-generated research report enumerating different responsibilities for AI/ML jobs and collecting them under different role labels independently of job postings. 
- **Feature engineering** from the job postings for improved embedding quality
    - Perhaps keyword extraction, per-section analysis, removal of non-essential content (e.g., company culture sections), normalization of text, LLM-generated summaries
        - **LLM evals** may one day become necessary...


## July 22, 2026: Baseline embedding + boilerplate removal~~

This experiment tests the effect of removing non-essential content (company culture, about us, etc.) from job postings before embedding.

**Hypothesis:** Embedding of only the body text should improve cluster separation and reduce noise from unrelated company information.

### Experimental design
- The `jobs.json` contains parsed job postings with fields per-section (title, responsibilities, requirements, company culture, etc.)
- Only the `responsibilities` and `requirements` sections are concatenated for embedding, excluding `company_culture`, `about_us`, and other non-essential sections.
    - A `clean_text` field was added to the JSON structure containing the concatenated body text.
    - Only the following fields were kept: 
    `KEEP_SECTIONS = ["about_role", "responsibilities", "qualifications", "nice_to_have"]`
- The same embedding pipeline is used as in the baseline experiment (all-MiniLM-L6-v2, L2-normalized), but applied to the cleaned text.
- UMAP, ~~PCA and t-SNE visualizations~~ are generated for direct comparison against the baseline embeddings. 
- `diff_vscode.py` is used to generate raw and clean_text files diff-able with VSCode

### Results
- Results are improved but unfortunately not satisfactory. 
    - Remaining text still includes non-job related content, such as benefits or company culture information.
    - The boilerplate removal regex patterns fail to capture all variations of these sections across different posting formats.
- Visualizations do show improvement in cluster separation compared to the baseline however, with less overlap remaining between role categories.

#### PCA

![](../data/plots/exp_boilerplate_pca.png)
- Interestingly, the PCA plot seems to put the Data Scientist roles and all the other roles on two orthogonal axes. Very interesting! 
    - I wonder what this means. So far we've worked under the assumption that there is no meaning to the dimensions in the embedding space. 
    - However, PCA does reveal that the first two principal components capture the most variance in the data, and the orthogonal arrangement suggests that different role categories may be distinguished by distinct semantic features captured along these axes.

#### UMAP

![](../data/plots/exp_boilerplate_umap.png)
- UMAP seems to also capture more meaningful separation between groups. 
- Data Scientist roles (blue) occupy a diagonal band across the plot. Not sure if this is better or worse. 
    - Clustering performance seems worse but the separation between distinct role categories appears improved.

#### t-SNE

![](../data/plots/exp_boilerplate_tsne.png)
- t-SNE performance remains the poorest of the three. No discernible structure gained, although the embedding appears to span a larger area than in the baseline.

### **Update**
- Early results suggest that removing non-essential content improves cluster separation and reduces noise from unrelated company information.
- But the boilerplate removal step is brittle, depending on complicated regex and pre-determined section extraction that may not scale to real-world job posting variance. 
    - At the present moment, it doesn't even clean up all the unwanted text.
- Switching early to an **LLM extraction-based workflow** to automatically identify and extract relevant sections (responsibilities, requirements) from unstructured job postings.