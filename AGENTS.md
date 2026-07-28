# AGENTS.md — Semantic Career Mapping Platform

> AGENTS.md is for AI coding agents. README.md is for humans. This file contains operational instructions to produce correct, convention-aligned code in this repository. Keep under 100 lines.

---

## Project at a Glance

- **Stack:** Python 3.12+, Sentence Transformers, FastAPI, Docker
- **Package manager:** uv + pip (pyproject.toml + requirements.txt)
- **No virtual environment set up yet** — create one with `uv venv` before installing
- **No tests exist yet** — test framework is pytest, install with `pip install pytest`

## Development Commands

```bash
# Create venv (do this first if .venv doesn't exist)
python -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Format code
ruff format .

# Lint
ruff check .

# Type check (when mypy is configured)
mypy backend/

# Run tests
pytest tests/ -v

# Run a single test file
pytest tests/test_preprocessing.py -v
```

Use file-scoped variant when changing a single file:
- `ruff check path/to/file.py`
- `pytest tests/test_preprocessing.py::test_function_name -v`

## Code Conventions

- Python 3.11+ with full type hints on all public functions
- Docstrings are Google-style: `"""Brief summary.\n\nArgs:\n    name: description.\n\nReturns:\n    description.\n"""`
- Imports order: stdlib → third-party → local, each group separated by a blank line
- File names: `snake_case.py`
- Class names: `PascalCase`
- Function names: `snake_case`
- Constants: `UPPER_SNAKE_CASE`

## Architecture Constraints

- Backend code lives in `backend/`
- Data pipeline scripts live in `scripts/` (not `backend/`)
- Tests live in `tests/` mirroring the source structure
- Data files (JSON, npy, CSV) live in `data/` — never commit large binary files
- Documentation lives in `docs/`
- Frontend lives in `frontend/`

## Tests

- Write tests for all new preprocessing, embedding, and retrieval code
- Test data lives in `tests/fixtures/` — keep fixtures small (< 10KB)
- Do NOT use real job posting data in tests — create minimal synthetic fixtures
- Golden set for evaluation lives in `data/golden_set.json` — hand-maintained, do not auto-generate

## Living Documentation

- **`docs/implementation-plan.md`** is the canonical phase plan — reference it before starting any implementation work
- **`README.md`** contains a "Current Status" section tracking phase progress — update it after completing each project phase
- **`docs/mvp-project-idea.md`** defines the project finish line — scope is locked to deliverables listed there
- **`docs/embedding-optimization-research.md`** contains the research background for preprocessing decisions

## Experiment Scripts — Immutable Artifacts

- `scripts/` contains numbered experiment directories (e.g., `scripts/003_llm_extraction/`). These are static artifacts paired with their corresponding experiment reports in `docs/experiments/`.
- **Do NOT modify existing experiment scripts unless explicitly instructed.** Experiment scripts are immutable records of what was run — changing them and reusing them violates scientific integrity.
- When functionality from an experiment script needs to be reused, extract and modularize it into `backend/` as a reusable module. The original experiment script stays untouched.

## When Adding a New Script

- Add an entry to `requirements.txt` if it imports a new third-party package
- Include a module docstring explaining what the script does and which phase/step it belongs to
- If the script has side effects (writes files), make output paths configurable via CLI argument, not hardcoded

## Do NOT

- Do not install any packages — this includes `pip install`, `uv pip install`, `uv add`, or any other package manager. The user manages all dependencies manually.
- Do not add GPU dependencies (cupy, cudf) unless explicitly requested — this project targets CPU-first deployment
- Do not add new frameworks (Django, Flask, Streamlit) unless explicitly discussed — stick to FastAPI
- Do not modify `docs/initial-project-idea.md` — it's a historical reference document
- Do not auto-generate README content — write it by hand
- Do not commit API keys, secrets, or `.env` files
