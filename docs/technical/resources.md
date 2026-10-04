# Favourite Resources

Just my favourite resources used in this project.

## Engineering & System Design

| Resource | Author | Why it matters |
|---|---|---|
| [Object Storage System Design](https://www.systemdesignhandbook.com/guides/object-storage-system-design/) | Fahim Ul Haq | Grounds the object-store layout: buckets, keys, metadata, and durability trade-offs used in [Data Layout & Object Storage](../architecture/data-layout.md). |
| [Designing Machine Learning Systems](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) | Chip Huyen | End-to-end ML system design: data engineering, feature stores, deployment, monitoring. |
| [AI Engineering](https://www.oreilly.com/library/view/ai-engineering/9781098166298/) | Chip Huyen | Evaluation, RAG, agents, and the AI engineering stack. |

## LLM Evaluation

Synthesized in [Research Report 004 — LLM Evaluations](../research-reports/004-llm-evals-report.md). The three primary sources:

| Resource | Author | Why it matters |
|---|---|---|
| [Eval-Driven Development: Lessons from Evaluating GenAI at Scale](https://medium.com/airbnb-engineering/eval-driven-development-lessons-from-evaluating-genai-at-scale-e817e5ae5788) | Rohit Girme (Airbnb) | Layered evaluation stack; scalable judge calibration. |
| [From Weeks to a Day: How We Made LLM Evaluation Fast Enough to Iterate On](https://medium.com/airbnb-engineering/from-weeks-to-a-day-how-we-made-llm-evaluation-fast-enough-to-iterate-on-14e2d35198b4) | Baharak Saberidokht (Airbnb) | Deterministic evaluation via caching, micro-adapters, end-to-end validation at the seams. |
| [LLM Evals: Everything You Need to Know](https://hamel.dev/blog/posts/evals-faq/) | Hamel Husain & Shreya Shankar | Error analysis (open → axial coding), binary pass/fail, synthetic data generation, custom annotation tooling. |

## Key Takeaways Applied Here

- Manual error analysis before automated evaluators.
- My data engineering design instincts are good! 
- Scalable design for multi-label classification.