# Experiment 003 — LLM Extraction Follow-up

**Date:** July 23, 2026

This experiment tests the use of LLMs to automatically extract relevant job posting sections (responsibilities, requirements) from unstructured text, replacing brittle regex-based boilerplate removal with AI-driven content parsing.

## Experimental Design

Three scripts were generated to automate:

- Extraction of relevant job posting sections (responsibilities and requirements) from unstructured text using LLMs
- Judge the quality of extraction against manually-tuned "golden sets" of text
- Test self-retrieval accuracy when embedded

### `scripts/003_llm_extraction/extract_clean_text.py`

LLM extraction pipeline for boilerplate removal.

- The complete raw text of the job descriptions were passed into various lightweight LLMs, along with the system prompt, for text extraction.
- Uses the following system prompt:

```
You are a job posting cleaner. Extract only the parts of a job posting that
describe the job itself: responsibilities, required qualifications, and
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

Output only the cleaned text. No headers, prefixes, explanations, or formatting.
```

- The cleaned text was stored in `data/jobs.json` as a new field `llm_clean_text`
- Models tested:
    - `gemini-2.5-flash`
    - `gpt-5.4-nano`
    - `deepseek-v4-flash`

### `scripts/003_llm_extraction/eval_cleaning.py`

Evaluation script comparing LLM output against "golden sets" — hand-cleaned reference texts for accuracy assessment.

- Several metrics were used to gauge the quality of the extraction:
- **Jaccard similarity** measures the overlap between LLM-extracted text and golden reference text
    - Calculated as the ratio of set cardinalities; the intersection of words in both texts divided by the union of words in both texts
- **Boilerplate detection** checks for residual non-essential content (company culture, benefits, etc.) in the cleaned output
    - Manually selected markers detected through regex pattern matching
    - The following markers were used:
    ```
    salary, pay grade/range/type, compensation, equity grade
    benefits, health insurance, dental, vacation, PTO, parental leave
    equal opportunity employer, EEO, diversity statement
    why you'll love, testimonials, perks
    note to recruiters, unsolicited resumes, recruiting agency
    about us, mission, values, purpose
    salary figures (e.g., $XXX,XXX CAD/USD)
    ```
- **Hallucination detection** identifies hallucinated content or missing responsibilities that should be present based on the golden set
    - Calculated as the set difference between the llm-cleaned text minus the raw text.
    - No new words should be introduced.
- **Over-deletion rate** measures excessive removal of relevant content, indicating aggressive cleaning that may omit critical job details
    - The set difference of the manually tuned "golden words" minus the llm-cleaned text is used to count how many words are missing
    - Calculated as the ratio of set cardinalities: the above set difference divided by the golden words.

### `scripts/003_llm_extraction/compare_variants.py`

Script for comparing different embedding variants (raw text, section-cleaned text, LLM-cleaned text) for cluster separation and self-retrieval accuracy.

- The script iterates over all three of: raw text, regex-cleaned text, and llm-cleaned text to compare their clustering performance and separation over visualizations.
- Two metrics are used: self-retrieval, and separation gap.
    - Self-retrieval is a simplified variant of **1-NN leave-one-out accuracy**.
    - Separation gap is a modified version of **Fisher's linear discriminant**.
    - *Metrics are subject to formalization and refinement. References from literature are ideal.*

#### Self-Retrieval

Self-retrieval is a degeneracy check. It verifies that each variant's embedding is mapped to a different vector, indicating that the embedding space does not collapse over the utilized dimensions.

- Let `A` be the `n × d` matrix of normalized embeddings, where `n` is the number of postings and `d` is the embedding dimensionality.
- Then, `AA^T` performs the dot product of every embedded vector with every other embedded vector.
- Since the vectors are normalized, this is equivalent to computing the cosine similarity between every pair of embeddings.
- Since the angle between every vector `v` and itself is 0, we expect that the diagonal of `AA^T` contains values of 1 (maximum cosine similarity).
- The degenerate case to catch is if any two vectors are mapped to the same thing.
    - This could happen with duplicate inputs or if the embedding model was faulty and collapsed the space.
- The self-retrieval score is calculated by checking if the highest similarity for each posting is with itself (diagonal element) or with another posting (off-diagonal element), with partial credit given for second-highest matches.

#### Separation Gap

Separation gap is a performance check of how well an embedding variant clusters jobs by role.

- For a given job role, let `WC` (for, *within-class*) be the set of embedded vectors for the same role, and `BC` (for, *between-class*) be the set of all other embedded vectors.
- Between every pair of vectors in `WC` and `BC`, compute the cosine similarity (`sim = AA^T`, from self-retrieval).
- Let the mean cosine similarity in the *within-class* group be `x_w`, and the mean cosine similarity in the *between-class* group be `x_b`
- Separation gap is simply the difference `x_w − x_b`

##### Implementation Notes

`self_retrieval()` is implemented as follows:

```
Compute the self-similarity matrix sim = AA^T. This matrix is symmetric (PSD).

For every row, sort in descending value.

If the first instance of the number 1 in the row is in the diagonal position,
then increment hits by 1.

Elif the second instance of the number 1 in the row is in the diagonal position,
then increment hits by 0.5.

Else, we have a degenerate case where more than two vectors are mapped to
the same thing. Don't increment hits.

At the end, compute the ratio hits / total rows.
```

The real implementation leverages the stable sorting of `argsort` and assumption of unity cosine similarity along the diagonal.

## Results

The LLM approach was overall successful.

Each of the three scripts was run in sequence, with `MiniLM-L6-v2` used as the embedding model, and `gpt-5.4-nano` used as the LLM extraction model when multiple models aren't specified.

### Per-Model Text Cleaning Performance (Golden set vs. raw text, gpt-5.4-nano)

The per-model performance over 5 postings is summarized below:

| Posting | Jaccard vs Golden | Boilerplate | Hallucinations | Over-deletion | Golden Words | LLM Words |
|---|---|---|---|---|---|---|---|
| BMO Data Scientist | 0.801 | CLEAN | CLEAN | 19.9% | 333 | 248 |
| Affirm ML Engineer 2 | 0.872 | CLEAN | CLEAN | 12.8% | 386 | 311 |
| HelloFresh ML Engineer | 0.775 | CLEAN | CLEAN | 22.5% | 338 | 245 |
| Mastercard Data Scientist 2 | 0.592 | CLEAN | CLEAN | 30.7% | 534 | 400 |
| Scribd Data Scientist 2 | 0.606 | CLEAN | CLEAN | 39.4% | 403 | 205 |

- No boilerplate or hallucinations were detected in any golden posting, suggesting that a low temperature and simple model is enough for this scale of text extraction.
    - Larger experiment sizes may be ideal for validating these results.
- gpt-5.4-nano strips aggressively across all postings, with over-deletion ranging from 12.8% (Affirm) to 39.4% (Scribd).
    - **System prompt tuning may be needed to reduce over-deletion while preserving boilerplate removal.**

### Overdeletion: Mastercard Data Scientist 2

For a concrete investigation of overdeletion, the **Mastercard Data Scientist 2** posting was selected for a comparison between the golden set and llm-cleaned text.

The set intersection of the golden set and llm-cleaned text shows that the LLM successfully preserved critical responsibilities while removing non-essential boilerplate content:

- "Ability to identify appropriate analytical techniques and validate solutions through structured evaluation"
- "Analyze large-scale transaction, merchant, and related entity data to identify patterns, trends, and anomalies"
- "Bachelor's degree in Data Science, Statistics, Mathematics, Computer Science, or another quantitative discipline..."
- "Contribute to feature engineering, entity resolution, and analytical workflows that improve merchant-level intelligence"
- "Critical thinking and a drive to produce high quality work, ensuring that all solutions meet rigorous standards"
- "Experience in applying data science and machine learning to solve real business problems"
- "Experience with Python and SQL; familiarity with tools such as Pandas for data manipulation and analysis"
- "Experience with writing clean, modular, and well-documented code following Data Science best practices..."
- "Understanding of Agile methodologies, with the ability to contribute to iterative delivery"
- And more (see full experiment notes)

The set difference `golden − llm` shows what text the LLM removed that we decided to keep. These passages reflect the responsibilities inherited by applicant from their team's role within the organization (but also a few headers):

- "All About You"
- "The Security Solutions Data Science team is responsible for developing Artificial Intelligence (AI) and Machine Learning (ML) models that power Mastercard's Identity and risk solutions..."
- "You will join a dynamic and innovative team working at scale with modern big data platforms and technologies..."
- "Key Responsibilities"
- "Role"

### Overdeletion: Scribd Data Scientist 2

The set intersection between `golden_set` and `llm_clean_text` suggests that key experience and responsibilities markers are also kept by the LLM extraction:

- "3+ years of post qualification experience developing machine learning models, working with systems at scale and deploying to production environments."
- "Align with stakeholders through written and verbal communications methods on the approaches and results of projects..."
- "Bachelors or Masters in relevant quantitative discipline including but not limited to Statistics, Computer Science, Data Science, Artificial Intelligence..."
- "Collaborate with other Data Scientists, Machine Learning Engineers and ML Data Engineers on cross-functional projects"
- "Focus on a variety of content classification use cases, leveraging everything from traditional NLP to sophisticated LLMs and generative models"
- "Hands-on experience building ML pipelines and working with distributed data processing frameworks like Apache Spark, Databricks, or similar."
- "Intermediate level in at least three of these fields: classification algorithms, natural language processing, search, information retrieval, named entity recognition, deep learning, generative models."
- And more

The set difference `golden_set − llm_clean_text` reveals the LLM removed key context about the role of the team within the organization. Not entirely sure that it's wrong to be honest. Starting to think that storing organizational role context alongside the job requirements may be useful.

### Overdeletion: HelloFresh ML Engineer

The set intersection reflects the roles and responsibilities as desired, suggesting that the LLM kept the correct markers:

- "A passion for building developer tools and platforms that improve productivity and enable other teams to scale ML adoption."
- "Ability to communicate complex technical findings to both technical and non-technical stakeholders."
- "Collaborate with data scientists, product managers, and culinary stakeholders to translate business needs into reliable AI & ML products."
- "Contribute to our MLOps platform — a set of tools and services built on Databricks, MLFlow, and Prefect..."
- "Design and build scalable, production-grade ML systems capable of handling large datasets, complex model pipelines, and real-time API integrations."
- "Design and implement production-grade AI Agent evaluations to measure quality, reliability, and performance of generative outputs."
- "Develop and maintain an agentic GenAI system that integrates with HelloFresh's culinary content pipeline..."
- And more

The set difference `golden − llm` again shows the LLM removed text that describes the role of the position within the context of the organization. Definitely will be storing this information since it seems to be semantically meaningful.

### Affirm ML Engineer 2 — Unexpected Over-Deletion

This posting was initially thought to be a near-perfect match, but the actual results show 12.8% over-deletion (Jaccard 0.872) — the LLM output is 311 words compared to a 386-word golden set.

The set intersection shows the same pattern — the requirements and responsibilities are kept, but meaningfully more content was removed than previously reported. This case now aligns with the other postings in showing that the LLM strips too aggressively.

### Embedding Comparison (raw vs section-cleaned vs LLM-cleaned, MiniLM-L6-v2 and gpt-5.4-nano)

Regardless, the LLM-cleaned text achieved the highest separation gap, and a significant improvement over the raw embedded text. This suggests an improvement of semantic signaling by removing boilerplate and non-essential content.

| Variant | Self-Retrieval | Separation Gap | Mean CosSim | Max CosSim |
|---|---|---|---|---|
| Raw (`raw_full_text`) | 100.0% | +0.0060 | 0.4045 | 0.7443 |
| Section-Cleaned (`clean_text`) | 100.0% | +0.0203 | 0.4298 | 0.7652 |
| **LLM-Cleaned (`llm_clean_text`)** | **100.0%** | **+0.0485** | 0.5284 | 0.7903 |

- None of the text variants were collapsed in the embedding space, reflected by the 100% self-retrieval score.
- The separation gap was greatest for the LLM-cleaned variant, reflecting the improved semantic separation between role categories.
- Max CosSim is lowest for raw text (0.7443) and highest for LLM-cleaned (0.7903), consistent with reduced variance after boilerplate removal.

![UMAP comparison](../assets/images/llm_cleaning_comparison.png)

## Discussion

- Overall, the quality of text extraction was quite good.
- In no test cases (5 examples only, to be fair) did the LLM remove information about the job requirements and responsibilities.
- In most cases it seemed to remove contextual information about the role of the position within the organization.
    - **This might be a meaningful marker to keep, alongside the specific role responsibilities.**
- **Tuning of the system prompt** and regex boilerplate may help to **extract text more accurately** and with finer granularity.
