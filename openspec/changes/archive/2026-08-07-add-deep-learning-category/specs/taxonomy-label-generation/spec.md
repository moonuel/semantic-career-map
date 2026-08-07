## MODIFIED Requirements

### Requirement: LLM assigns taxonomy function tags and generates functional titles for job postings
The system SHALL accept a job posting's `llm_clean_text` field and the 13-category ML/AI responsibility taxonomy definitions with a controlled title vocabulary, and produce a structured JSON output containing a list of applicable function tags, a generated functional title from the controlled vocabulary, and a rationale for the assignment.

#### Scenario: Input from llm_clean_text
- **WHEN** processing any job posting
- **THEN** the system uses the posting's `llm_clean_text` field as the uniform input for taxonomy classification

#### Scenario: Deep learning category available
- **WHEN** a posting involves designing or training neural network architectures (CNNs, RNNs, transformers, GANs, VAEs) using PyTorch, TensorFlow, or JAX, not specifically limited to LLM fine-tuning, computer vision, or reinforcement learning
- **THEN** the system SHALL be able to assign the `deep-learning` taxonomy category

#### Scenario: Structured output format
- **WHEN** the LLM processes a posting
- **THEN** the output SHALL be a JSON object with fields `function_tags` (a list of 1-N taxonomy category names), `generated_title` (a functional title from the controlled vocabulary), and `rationale` (a 1-2 sentence explanation)

#### Scenario: Controlled title vocabulary
- **WHEN** the LLM generates a functional title
- **THEN** the title SHALL be selected from a predefined mapping of taxonomy categories to title descriptors (e.g., `classical-ml` → "Classical ML Engineer", `deep-learning` → "Deep Learning Engineer"), ensuring the same function type always maps to the same title

#### Scenario: Multi-label assignment
- **WHEN** a posting's responsibilities span multiple taxonomy categories
- **THEN** the LLM SHALL include all applicable categories in `function_tags` without artificial primary/secondary distinction

#### Scenario: Cached results skip API calls
- **WHEN** a posting has already been labeled and cached
- **THEN** the system SHALL return the cached result without making a new API call

#### Scenario: Prompt hash mismatch invalidates cache
- **WHEN** the taxonomy definitions or prompt text change between runs
- **THEN** the system SHALL detect the hash mismatch and invalidate the cache, re-labeling all postings

## ADDED Requirements

### Requirement: Golden subset reflects deep learning category
The system SHALL update the 5-posting golden subset labels to include `deep-learning` where the posting's llm_clean_text contains explicit deep learning signals (PyTorch, TensorFlow, JAX, "deep learning", "neural network") at a level that makes it a primary function, not just a passing mention.

#### Scenario: Scribd golden label updated
- **WHEN** the golden subset is loaded for validation
- **THEN** the Scribd posting (scribd-data-scientist-2) SHALL have golden labels `["deep-learning", "llm-information-retrieval"]`, replacing the previous `["llm-fine-tuning", "llm-information-retrieval"]`
