# Current TODOs

## July 23:
- I'm happy with the approximate pipeline right now but I have several experiments that may start to drift. 
- Formalization of the pipeline in multiple stages in backend modules may be helpful in the short term.
- An architecture document to accommodate modularity and future expansion would be good. 
---
- Further experiments will quickly clog up the `scripts/` directory. 
- A useful pattern may be to mirror the experiements log and prepend triple digits to each script to organize it. 
    - Perhaps a [0-9]^3{|A-Z} (i.e. 000 or 000A, in pseudo-regex set notation) would be good. 
        - To be clear, the alphabetical characters may be useful if there are multiple experiments with the same number. 
    - Git supports a "get it right the first time" philosophy so some early planning is probably prudent :(
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