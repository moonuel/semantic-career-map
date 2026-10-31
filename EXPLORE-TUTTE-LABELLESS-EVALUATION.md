# Explore Toponymy (Tutte Institute) for Label-less Cluster Evaluation

Internal note. Not part of the published docs.

## Idea

Explore [Toponymy](https://github.com/TutteInstitute/toponymy) (Tutte Institute) as a tool
to evaluate clusters subjectively, without requiring domain-specific labels.

## Rationale

The custom label taxonomy currently in the repo is not yet mature, so it should **not** be
used as the basis for cluster evaluation. A strict label-less approach would support better
prompt engineering and unsupervised clustering performance. Toponymy is a candidate for
producing interpretable, label-free cluster descriptions that can be judged subjectively
(e.g. by a human or an LLM) rather than against a fixed taxonomy.

## Open questions

- How does Toponymy's label-free topic naming compare against the current clustering output?
- Can its descriptions be used as prompts or evaluation rubrics without leaking a taxonomy?
- What does the runtime/inference cost look like on CPU-only deployment?

## Next steps

- Prototype against an existing embedding set in a numbered script under `scripts/` (do not
  modify existing experiment scripts).
- Write up findings in `docs/experiments/` once results exist.
