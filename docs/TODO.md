# Current TODOs

## July 23:
- I'm happy with the approximate pipeline right now but I have several experiments that may start to drift. 
- Formalization of the pipeline in multiple stages in backend modules may be helpful in the short term.
- An architecture document to accommodate modularity and future expansion would be good. 
---
- One of the most immediate core ideas to validate is whether the role taxonomy is effective for clustering similar postings.
- Likely I will need more postings (or synthetic data) to benchmark it properly. 
---
- Then some reading of standard feature engineering will probably be good. 
---
## July 25:
- I think the next major experiment will tie together a few open threads.

- One goal is to compare clustering results between
    - Cleaned postings, with/without the contextual information about the role within the organization (stored as a separate field)
    - Engineered features, maybe based on the role taxonomies, maybe something like generated description strings based on the cleaned data
- My working assumption is that the embedding model clusters based on semantic similarity, not necessarily similarity of text 
- But this assumption needs to be tested. I guess this experiment addresses that. 
---
- I also want to validate whether the role taxonomy provides a better basis for judging clustering
- Something like generating job labels from the taxonomy
- Then re-evaluating the clustering results against these generated labels
- To do this at scale is still something I can't imagine doing without some LLMs and evals. 
- I might stick to small samples for now and solve the scale problem later. 
- Then some reading of standard feature engineering will probably be good.
---
## July 26:
- Ohhh you know what would be fire. Have the demo use some variations of my resume, embed them all, and see how different variations map to different job groups.
- I've done so much more than what can fit on a single resume sheet, this would be such a flex to also demonstrate my breadth 
---
## July 27
- Boilerplate removal should probably be its own pipeline step.
- It would fall under pre-processing - the boilerplate-cleaned text minus the raw text should always be empty (no hallucination)