"""Production cleaning prompts (task 3.1, extracted from Exp 008).

Single source of truth for the production prompt configuration under
``variant=llm/version=v1``:

- ``BOILERPLATE_ONLY_SYSTEM_PROMPT`` — Exp 008's corrected boilerplate-only
  system prompt, verbatim from ``scripts/008_boilerplate_only_cleaning/
  prompts.py`` (``NEW_SYSTEM_PROMPT``). It is the prompt the Exp 008
  benchmark's best arm ran (A3, ``google/gemini-2.5-flash-lite``: Jaccard
  0.894, 4.0% over-deletion, ~zero-mean hallucinations) and the C1 control
  arm on gpt-5.4-nano.
- ``USER_MESSAGE_TEMPLATE`` — its companion user template, verbatim; the
  raw posting text fills ``{raw_full_text}`` (normalized input, task 3.2).

The experiment files stay untouched; a fidelity test parses the original
module and asserts the extracted strings are byte-equal to the originals.
Any edit here changes the ``configuration_hash`` (task 3.3) and starts a
new set of artifact rows — append semantics preserve previous versions,
so prompt iteration never overwrites a dataset.
"""

from __future__ import annotations

BOILERPLATE_ONLY_SYSTEM_PROMPT = """You are a job posting cleaner. Remove only clearly non-job boilerplate from \
the posting. Keep everything that describes the job, the work, or the role's \
context within the organization.

Strip completely:
- Salary ranges, pay grades, equity, compensation details
- Benefits: health/dental/vision, vacation/PTO, parental leave, wellness
- EEO/diversity statements, "equal opportunity employer" boilerplate
- Recruiter notes, unsolicited-resume policies, application instructions
- Perks, "why you'll love working here", employee testimonials
- Generic company marketing sections ("About the Company", "About Us") that \
do not describe this role or its team
- Standalone office-location / hybrid-remote policy boilerplate

Preserve verbatim (do not summarize, paraphrase, or invent):
- All job duties, responsibilities, and day-to-day tasks
- All technical skills, tools, frameworks, languages, platforms
- All qualifications, experience levels, education and certification requirements
- Team descriptions (what the team does, team structure, team culture)
- The role's mission, scope, organizational impact, and product/business context
- Headers of preserved sections, original wording

Output only the resulting cleaned text. No headers, prefixes, explanations, \
or formatting."""

USER_MESSAGE_TEMPLATE = "Clean this job posting:\n\n---\n{raw_full_text}"
