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


## July 22, 2026: Baseline embedding + boilerplate removal

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

### Discussion
- Early results suggest that removing non-essential content improves cluster separation and reduces noise from unrelated company information.
- But the boilerplate removal step is brittle, depending on complicated regex and pre-determined section extraction that may not scale to real-world job posting variance. 
    - At the present moment, it doesn't even clean up all the unwanted text.
- Switching early to an **LLM extraction-based workflow** to automatically identify and extract relevant sections (responsibilities, requirements) from unstructured job postings.

## July 23, 2026: LLM extraction follow-up

This experiment tests the use of LLMs to automatically extract relevant job posting sections (responsibilities, requirements) from unstructured text, replacing brittle regex-based boilerplate removal with AI-driven content parsing.

### Experimental design
- Three scripts were generated to automate:
    - Extraction of relevant job posting sections (responsibilities and requirements) from unstructured text using LLMs
    - Judge the quality of extraction against manually-tuned "golden sets" of text
    - Test self-retrieval accuracy when embedded

#### `extract_clean_text.py`

LLM extraction pipeline for boilerplate removal

- The complete raw text of the job descriptions were passed into various lightweight LLMs, along with the system prompt, for text extraction.
- Uses the following system prompt:
```python
SYSTEM_PROMPT = """You are a job posting cleaner. Extract only the parts of a job posting that \
describe the job itself: responsibilities, required qualifications, and \
preferred/nice-to-have skills. Remove everything else.

Categories to strip completely:
- Company description, "About Us", mission statements, values
- Team descriptions (what the team does, team structure, team culture)
- Salary ranges, pay grades, equity, compensation details
- Benefits: health/dental/vision, vacation/PTO, parental leave, wellness
- EEO/diversity statements, "equal opportunity employer" boilerplate
- Recruiter notes, application instructions
- Office locations, hybrid/remote policy boilerplate
- Perks, "why you'll love working here", employee testimonials

Preserve verbatim (do not summarize, paraphrase, or invent):
- All job duties, responsibilities, and day-to-day tasks
- All technical skills, tools, frameworks, languages, platforms
- All required qualifications and experience levels
- All preferred/nice-to-have qualifications
- All education and certification requirements
- Original wording of preserved sections

Output only the cleaned text. No headers, prefixes, explanations, or formatting."""
```
- The cleaned text was stored in `data/jobs.json` as a new field `llm_clean_text`
- Models tested: 
    - `gemini-2.5-flash`
    - `gpt-5.4-nano`
    - `deepseek-v4-flash`

#### `eval_llm_cleaning.py`

Evaluation script comparing LLM output against "golden sets" — hand-cleaned reference texts for accuracy assessment.

- Several metrics were used to gauge the quality of the extraction:
- **Jaccard similarity** measures the overlap between LLM-extracted text and golden reference text
    - Calculated as the ratio of set cardinalities; the intersection of words in both texts divided by the union of words in both texts
- **Boilerplate detection** checks for residual non-essential content (company culture, benefits, etc.) in the cleaned output
    - Manually selected markers detected through regex pattern matching
    - The following markers were used:
```
r"salary|pay\s*(grade|range|type)|base\s+pay|compensation|equity\s+grade"
r"benefits|health\s+insurance|dental|vision|vacation|PTO|parental\s+leave"
r"equal\s+opportunity\s+employer|EEO|\bdiversity\s+statement\b"
r"why\s+you('|’)\s*ll\s+love|testimonials|perks"
r"note\s+to\s+recruiters|unsolicited\s+resumes|recruiting\s+agency"
r"about\s+us\b.*mission|values|purpose"
r"\$[\d,]+.*(?:CAD|USD)|[\d,]+.*(?:CAD|USD).*\bsalary\b"
```
- **Hallucination detection** identifies hallucinated content or missing responsibilities that should be present based on the golden set
    - Calculated as the set difference between the llm-cleaned text minus the raw text. 
    - No new words should be introduced.
- **Over-deletion rate** measures excessive removal of relevant content, indicating aggressive cleaning that may omit critical job details
    - The set difference of the manually tuned "golden words" minus the llm-cleaned text is used to count how many words are missing
    - Calculated as the ratio of set cardinalities: the above set difference divided by the golden words.

#### `comp_embedding_variants.py`

Script for comparing different embedding variants (raw text, section-cleaned text, LLM-cleaned text) for cluster separation and self-retrieval accuracy.

- The script iterates over all three of: raw text, regex-cleaned text, and llm-cleaned text to compare their clustering performance and separation over visualizations. 
- Two metrics are used: self-retrieval, and separation gap.
    - Self-retrieval is a simplified variant of **1-NN leave-one-out accuracy**.
    - Separation gap is a modified version of **Fisher's linear discriminant**. 
    - *Metrics are subject to formalization and refinement. References from literature are ideal.*  
- **Self-retrieval** is a degeneracy check. It verifies that each variant's embedding is mapped to a different vector, indicating that the embedding space does not collapse over the utilized dimensions. 
    - Let `A` be the `n-by-d` matrix of normalized embeddings, where `n` is the number of postings and `d` is the embedding dimensionality. 
    - Then, `AA^T` performs the dot product of every embedded vector with every other embedded vector.
    - Since the vectors are normalized, this is equivalent to computing the cosine similarity between every pair of embeddings.
    - Since the angle between every vector `v` and itself is 0, we expect that the diagonal of `AA^T` contains values of 1 (maximum cosine similarity).
    - The degenerate case to catch is if any two vectors are mapped to the same thing.
        - This could happen with duplicate inputs or if the embedding model was faulty and collapsed the space.
    - Thus by checking
    - The self-retrieval score is calculated by checking if the highest similarity for each posting is with itself (diagonal element) or with another posting (off-diagonal element), with partial credit given for second-highest matches.
- **Separation gap** is a performance check of how well an embedding variant clusters jobs by role. 
    - For a given job role, let `WC` (for, *within-class*) be the set of embedded vectors for the same role, and `BC` (for, *between-class*) be the set of all other embedded vectors. 
    - Between every pair of vectors in `WC` and `BC`, compute the cosine similarity (`sim = AA^T`, from self-retrieval). 
    - Let the mean cosine similarity in the *within-class* group be `x_w`, and the mean cosine similarity in the *between-class* group be `x_b`
    - Separation gap is simply the difference `x_w - x_b`

##### Implementation notes
`self_retrieval()` is implemented as follows: 
```
Compute the self-similarity matrix `sim = AA^T`. This matrix is symmetric (PSD).

For every row, sort in descending value. 

If the first instance of the number `1` in the row is in the diagonal position, then increment `hits` by 1. 

Elif the second instance of the number `1` in the row is in the diagonal position, then increment `hits` by 0.5. 

Else, we have a degenerate case where more than two vectors are mapped to the same thing. Don't increment `hits.`

At the end, compute the ratio `hits`/`total rows`.
```
The real implementation leverages the stable sorting of `argsort` and assumption of unity cosine similarity along the diagonal.


### Results 
- The LLM approach was overall successful.
- Each of the three scripts was run in sequence, with `MiniLM-L6-v2` used as the embedding model, and `gpt-5.4-nano` used as the LLM extraction model when multiple models aren't specified.

#### Per-model text cleaning performance (Golden set vs. raw text, gpt-5.4-nano)
- The per-model performance over 5 postings is summarized below:

| Posting | Jaccard vs Golden | Boilerplate | Hallucinations | Over-deletion |
|---|---|---|---|---|
| BMO Data Scientist | 0.801 | CLEAN | CLEAN | 19.9% |
| Affirm ML Engineer 2 | 1.000 | CLEAN | CLEAN | 0.0% |
| HelloFresh ML Engineer | 0.775 | CLEAN | CLEAN | 22.5% |
| Mastercard Data Scientist 2 | 0.610 | CLEAN | CLEAN | 29.9% |
| Scribd Data Scientist 2 | 0.614 | CLEAN | CLEAN | 38.6% |

- No boilerplate or hallucinations were detected in any golden posting, suggesting that a low temperature and simple model is enough for this scale of text extraction. 
    - Larger experiment sizes may be ideal for validating these results.
- gpt-5.4-nano achieves 100% reproduction of the **Affirm ML Engineer 2** golden set, but might strip too aggressively on other postings.
    - **System prompt tuning might be helpful for improving evals.** 

##### Overdeletion: Mastercard Data Scientist 2
- For a concrete investigation of overdeletion, the **Mastercard Data Scientist 2** posting was selected for a comparison between the golden set and llm-cleaned text.
- The set intersection of the golden set and llm-cleaned text shows that the LLM successfully preserved critical responsibilities while removing non-essential boilerplate content:

```
'Ability to identify appropriate analytical techniques and validate solutions through structured evaluation'
'Analyze large-scale transaction, merchant, and related entity data to identify patterns, trends, and anomalies'
"Bachelor's degree in Data Science, Statistics, Mathematics, Computer Science, or another quantitative discipline such as Engineering, Economics, or Physics"
'Contribute to feature engineering, entity resolution, and analytical workflows that improve merchant-level intelligence'
'Critical thinking and a drive to produce high quality work, ensuring that all solutions meet rigorous standards'
'Experience in applying data science and machine learning to solve real business problems'
'Experience with Python and SQL; familiarity with tools such as Pandas for data manipulation and analysis'
'Experience with writing clean, modular, and well-documented code following Data Science best practices. Ability to collaborate effectively through code contributions, peer reviews, and shared development workflows to ensure robust, maintainable, and efficient solutions'
'Exposure to large datasets and interest in scalable data processing; familiarity with Spark is a plus'
'Good communication skills, enabling effective collaboration with team members and stakeholders'
'Help build and maintain scalable data and model workflows using Databricks and Spark-based environments'
'Identify appropriate techniques for different analytical problems and help validate solutions through structured testing, benchmarking, and performance evaluation'
'Openness to learn and apply new technologies, staying current with industry trends and advancements'
'Prototype machine learning and analytical solutions under guidance from senior team members'
'Self-driven with a collaborative mindset and enthusiasm for learning in a fast-paced, innovative environment'
'Solid foundation in statistics, analytics, and core machine learning concepts'
'Support model monitoring, benchmarking, and iterative improvement of data science solutions'
'Support the research and development of a merchant registry and profiling capability to strengthen merchant risk assessment during onboarding and ongoing monitoring'
'Understanding of Agile methodologies, with the ability to contribute to iterative delivery'
'Work closely with partners across Data Science, Product, and Engineering in an Agile environment to support iterative delivery and continuous improvement'
'You will contribute to the design and development of data science capabilities that improve merchant risk assessment across onboarding and monitoring workflows.'
  ```

- The set difference `golden - llm` shows what text the LLM removed that we decided to keep.
- These passages reflect the responsibilities inherited by applicant from their team's role within the organization (but also a few headers):
```
'All About You'
'In addition to building models, the team is responsible for the research and development of scalable end-to-end data science capabilities covering the full lifecycle of model creation—from data extraction and feature engineering to validation, deployment, and monitoring. These capabilities must be designed to scale and to be repeatable, resilient, and industrialized so they can support long-term product growth and evolving business needs.'
'Key Responsibilities'
'Role'
"The Security Solutions Data Science team is responsible for developing Artificial Intelligence (AI) and Machine Learning (ML) models that power Mastercard's Identity and risk solutions across authentication and authorization use cases. These models are production-ready and designed to support key products and capabilities that help make digital transactions safer, smarter, and more trusted."
'You will join a dynamic and innovative team working at scale with modern big data platforms and technologies. In this role, you will focus on solving merchant risk during onboarding and ongoing monitoring through the research and development of a merchant registry and profiling capability within the Identity Data Science portfolio. This includes helping build the foundational data assets, profiling logic, analytical workflows, and machine learning approaches needed to better understand merchant behavior, relationships, and risk signals over time.'
```

##### Overdeletion: Scribd Data Scientist 2
 
- The set intersection between `golden_set` and `llm_clean_text` is below.
- It suggests that key experience and responsibilities markers are also kept by the LLM extraction.

```
3+ years of post qualification experience developing machine learning models, working with systems at scale and deploying to production environments.
Align with stakeholders through written and verbal communications methods on the approaches and results of projects, while writing detailed, accurate and concise project documentation
Bachelors or Masters in relevant quantitative discipline including but not limited to Statistics, Computer Science, Data Science, Artificial Intelligence or another field with a strong quantitative focus.
Collaborate with other Data Scientists, Machine Learning Engineers and ML Data Engineers on cross-functional projects
Focus on a variety of content classification use cases, leveraging everything from traditional NLP to sophisticated LLMs and generative models
Hands-on experience building ML pipelines and working with distributed data processing frameworks like Apache Spark, Databricks, or similar.
Intermediate level in at least three of these fields: classification algorithms, natural language processing, search, information retrieval, named entity recognition, deep learning, generative models.
Intermediate level or greater experience with SQL or PySpark.
Investigate methods of solving our most challenging problems at Scribd, at scale
Leverage any algorithm at your disposal: from classical Scikit-learn and NumPy models to custom Neural Networks in PyTorch to third party LLM APIs
Process massive amounts of data with Python, SQL and Spark
Proficiency in Python.
Requirements
Responsibilities
```

- The set difference `golden_set - llm_clean_text` is below. 
- It seems the LLM removed key context about the role of the team within the organization.
- Not entirely sure that it's wrong to be honest. 
- Starting to think that storing organizational role context alongside the job reqiurements may be useful.

```
Our areas of impact include content enrichment, representation learning, recommendations, search, translation and many others, applied to diverse media across text, image, and audio. We operate at a scale of hundreds of millions of documents, millions of users and billions of user interactions.
Role Overview
The Applied Research team is a group of data scientists and content specialists who are experts in leveraging machine learning, natural language processing and generative AI models to develop solutions which deliver value to our users and business. We act as a key driver for innovation, whether it's in product surface experimentation, metadata generation or model development. Along with Product and Engineering partners, we design solutions and collaborate in cross-functional squads to maximize business impact.
We are seeking a Data Scientist II with experience developing and deploying machine learning models. You will help design and implement high impact AI and ML systems. We work in cross-functional teams collaborating with Machine Learning Engineers, Data Engineers and Product. We are seeking a curious and collaborative individual with an eye for simplicity, end-end visibility and impact and that is excited about building models using massive amounts of data, using language models and deploying models.
```

##### Overdeletion: HelloFresh ML Engineer

- The set intersection `golden_set AND llm_clean_text` is below. 
- The content reflects the roles and responsibilities as desired, suggesting that the LLM kept the correct markers. 

```
A passion for building developer tools and platforms that improve productivity and enable other teams to scale ML adoption.
Ability to communicate complex technical findings to both technical and non-technical stakeholders.
All other duties, as assigned
Collaborate with data scientists, product managers, and culinary stakeholders to translate business needs into reliable AI & ML products.
Contribute to our MLOps platform — a set of tools and services built on Databricks, MLFlow, and Prefect that help data scientists develop, deploy, and maintain ML applications.
Design and build scalable, production-grade ML systems capable of handling large datasets, complex model pipelines, and real-time API integrations.
Design and implement production-grade AI Agent evaluations to measure quality, reliability, and performance of generative outputs.
Develop and maintain an agentic GenAI system that integrates with HelloFresh's culinary content pipeline (API development, LLM orchestration, image generation).
Experience with modern MLOps tooling such as MLFlow, feature engineering, Kubernetes, and workflow orchestration (e.g., Prefect). Experience with PySpark and Databricks is a plus.
Familiarity or interest in generative AI (LLMs, image generation, prompt engineering, agentic architectures).
Hands-on experience building and maintaining production ML systems, including model serving, monitoring, and CI/CD pipelines.
Improve operational excellence through automation, monitoring, drift detection, and data quality practices.
Stay up-to-date on advancements in generative AI, agentic workflows, and MLOps tooling, and propose improvements to the team's development lifecycle.
Strong software development skills in Python and SQL, including DevOps practices, test-driven development (TDD), spec-driven development, and fluency working with AI coding agents.
```

- The set difference `golden - llm` is below.
- Again, the LLM removed text that describes the role of the position within the context of the organization. 
- Definitely will be storing this information since it seems to be semantically meaningful. 

```
We are hiring a Machine Learning Engineer in our ML Solutions squad to help build innovative generative AI products that provide AI-powered assistants to HelloFresh's culinary Chefs helping them editorialize content, validate recipe data, detect issues, suggest corrections, and ideate. As a Machine Learning Engineer, you will primarily contribute to our agentic generative AI system that automates recipe content production, while also supporting the MLOps platform that enables data scientists across the company to develop, deploy, and monitor ML models at scale.
What This Role Will Be Responsible For
What we're looking for
```

##### No deletion: Affirm ML Engineer 2 

- As a contrasting case, let's investigate what the LLM did to the 100% match case. 
- The set intersection `golden AND llm` is below.
- Same as before, the requirements and responsibilities are kept, but the LLM included team-related context this time: "On the Servicing ML team,..."
- Some system prompt tuning and storage of the team-related content is definitely warranted. 

```
Experience building and evaluating models for tabular classification problems (preferably gradient-boosted decision trees like LightGBM/XGBoost/CatBoost).
Experience building applications with LLM APIs (e.g., OpenAI, Anthropic), including structured extraction, prompt engineering, and orchestration frameworks like LangChain or LangGraph.
Experience with ML lifecycle tooling for training orchestration, experimentation, and model monitoring (e.g., Kubeflow, Airflow, MLflow, or equivalent internal platforms).
Familiarity with document and unstructured data processing (PDF/image extraction, text parsing, or similar).
On the Servicing ML team, you will build and improve machine learning and AI systems that automate customer operations such as disputes, returns, fraud, and chargebacks to make the best decisions for Affirm and our customers. You will work closely with experienced ML engineers, platform partners, and cross-functional stakeholders to take models from idea to prototype to production, and to keep them healthy with strong measurement and monitoring.
Proficient in using AI-powered developer tools (e.g., Claude Code, Cursor, or similar) to accelerate iteration, debugging, and code quality as part of day-to-day development workflows.
Strong Python skills and experience writing production-quality code
What We Look For
What You'll Do
You are comfortable navigating a large code base, debugging others' code, and providing feedback to other engineers through code reviews.
You have a total of 2+ years of experience as a machine learning engineer
You have mastered taking a simple problem or business scenario into a solution that interacts with multiple software components, and executing on it by writing clear, easily understood, well tested and extensible code.
You have strong verbal and written communication skills that support effective collaboration with our global engineering team.
You will build and maintain evidence extraction pipelines that process unstructured data using LLM-powered workflows to produce structured, actionable outputs.
You will build models that automate refunds, getting money back to our customers faster.
You will collaborate across Engineering, Servicing Operations, Product, and ML Platform to define requirements, evaluate tradeoffs, and communicate results clearly to both technical and non-technical audiences.
You will develop AI systems that automate dispute and chargeback handling using structured evidence and business logic, creating a better experience for our customers.
You will prototype new modeling ideas, run offline experiments, and drive the best-performing approaches into production with appropriate risk controls.
Your experience demonstrates that you take ownership of your growth, proactively seeking feedback from your team, your manager, and your stakeholders.
```


#### Embedding comparison (raw vs section-cleaned vs LLM-cleaned, MiniLM-L6-v2 and gpt-5.4-nano)

- Regardless, the LLM-cleaned text achieved the highest separation gap, and a significant improvement over the raw embedded text.
- This suggests an improvement of semantic signaling by removing boilerplate and non-essential content.

| Variant | Self-Retrieval | Separation Gap | Mean CosSim |
|---|---|---|---|
| Raw (`raw_full_text`) | 100.0% | +0.0060 | 0.4045 |
| Section-Cleaned (`clean_text`) | 100.0% | +0.0203 | 0.4298 |
| **LLM-Cleaned (`llm_clean_text`)** | **100.0%** | **+0.0493** | 0.5280 |

- None of the text variants were collapsed in the embedding space, reflected by the 100% self-retrieval score. 
- The separation gap was greatest for the LLM-cleaned variant, reflecting the improved semantic separation between role categories.

### Discussion

- Overall, the quality of text extraction was quite good. 
- In no test cases (5 examples only, to be fair) did the LLM remove information about the job requirements and responsibilities. 
- In most cases it seemed to remove contextual information about the role of the position witihn the organization. 
    - This might be a meaningful marker to keep, alongside the specific role responsibilities. 
- **Tuning of the system prompt** and regex boilerplate may help to **extract text more accurately** and with finer granularity.

