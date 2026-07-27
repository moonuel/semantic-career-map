# OpenSpec Workflow Guide

> OpenSpec is an AI-native, spec-driven development workflow integrated into this project via Kilo. Think before you build, document what you learn, close cleanly when done.

---

## The Lifecycle

```
Explore  ──→  Propose  ──→  Apply  ──→  (Sync)  ──→  Archive
(free)      (artifacts)    (code)     (specs)     (cleanup)
```

---

## Commands

| Command | What It Does | When to Use |
|---|---|---|
| `/opsx:explore` | Think freely, investigate, sketch diagrams. No code, no artifacts. | Starting a new idea, unsure of direction, need to research. |
| `/opsx:propose <name>` | Creates `proposal.md`, `design.md`, `tasks.md` for a change. | You know what you want to build and are ready to plan. |
| `/opsx:apply <name>` | Implements the change task by task, checking them off. | After proposal is ready. Add the change name to run it. |
| `/opsx:update <name>` | Revises existing planning artifacts (proposal, design, tasks). | During implementation, if design changes or new requirements surface. |
| `/opsx:sync <name>` | Merges delta specs from a change into main specs. | Before archiving, to persist learned requirements. |
| `/opsx:archive <name>` | Moves the change to `archive/YYYY-MM-DD-<name>/`. | When implementation is complete. |

---

## The Artifacts (spec-driven schema)

Each change produces three files:

| Artifact | File | Purpose |
|---|---|---|
| **proposal** | `proposal.md` | What & why — scope, goals, non-goals, success criteria |
| **design** | `design.md` | How — architecture, trade-offs, key decisions |
| **tasks** | `tasks.md` | Implementation steps — checklist of `- [ ]` items |

The agent creates these in dependency order: proposal → design → tasks. Each artifact is generated from a template and constrained by project context and rules defined in `openspec/config.yaml`.

---

## How to Use It

### Starting fresh work

```
1. /opsx:explore                        # think it through
2. /opsx:propose my-change-name         # creates artifacts
3. /opsx:apply my-change-name           # implements task by task
4. /opsx:archive my-change-name         # clean up when done
```

### Picking up existing work

```
/opsx:apply my-change-name              # continues where you left off
```

### Changing direction mid-implementation

```
/opsx:update my-change-name             # revise artifacts
         ↓
/opsx:apply my-change-name              # continue with updated plan
```

---

## Location of OpenSpec Files

```
openspec/
  config.yaml              # Schema, project context, per-artifact rules
  changes/                 # Active changes (created by openspec new change)
  archive/                 # Archived changes (created by /opsx:archive)
```

The CLI resolves the nearest `openspec/` root by walking up from the current directory. All commands operate on the project root's `openspec/` directory.

---

## Tips

- **Keep change names short and kebab-case**: `add-auth`, `fix-embedding-bug`, `skill-extraction`
- **One change, one concern**: Don't bundle unrelated work into a single change — it makes review and archiving harder.
- **Archive when done**: An archived change is a clean record of what was built and why. Leaving active changes around adds noise to `openspec list`.
- **Sync before archiving**: If the change surfaced new requirements that should be tracked permanently, run `/opsx:sync` first. This merges delta specs into the main specs.
- **Update artifacts mid-flight**: If implementation reveals a design flaw, use `/opsx:update` to revise the plan — don't just code around it silently.
- **Explore is free**: Use `/opsx:explore` for research, problem investigation, or comparing approaches. No pressure to formalize.