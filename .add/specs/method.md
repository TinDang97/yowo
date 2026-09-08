---
type: Spec
title: Method
lens: method
project: yowo
description: how work proceeds, and what a gate costs
tags: [add, tdd, review]
sources: [CLAUDE.md, .add/index.md]
generated: { by: add/3.5.0, at: 2026-09-08 }
---
## Now

Work proceeds through ADD's three beats — Direction → Build → Verify — on one atomic task node,
red/green TDD inside Build, trusted on a recorded receipt rather than on a diff that reads well.
The repo's own CLAUDE.md already mandated red/green TDD and parallel specialist review before
commit; ADD is the rail that makes both recorded instead of remembered.

**The gate's price on this project**
- `add run <slug> -- uv run pytest <narrowest scope that reports every bound check> -q --junitxml=…`
  is the receipt. The full suite rides CI.
- The three residue lenses at Verify are not optional here: **security** (weight deserialization,
  credential handling) is a HARD-STOP; **concurrency** (threaded readers, batch scheduler, shared
  caches) and **architecture** (the backend Protocol, the engine hierarchy) are where this codebase
  actually breaks.

**Existing review roster** (`.claude/agents/`, invoked in parallel before commit per CLAUDE.md):
`backend-compliance-reviewer` · `inference-perf-auditor` · `test-coverage-guardian` ·
`api-contract-guardian` · `arch-correctness-reviewer`. These are the project's own personas by
another name; an ADD beat may delegate to them.

## Decisions that bind

- **The closed floor applies literally here.** Anything touching weight download/deserialization,
  the backend Protocol, the public `__init__` surface, or the engine hierarchy is a Task — never
  Quick, whatever its size.
- **`sensitive_paths:` in `.add/index.md` carries the floor**, so the engine computes the authority
  rather than the author remembering it.
- **Squash-merge PRs** (`gh pr merge --squash`) — the commit that lands is the receipt, and
  semantic-release reads its type to move the version.
- **Commit format is fixed** by CLAUDE.md: `<type>(<scope>): <summary>` + body + footer, written to
  a file and committed with `git commit -F`.
- **A green unit suite is not proof of backend parity.** CI runs one OS, one Python, one backend;
  any claim beyond that needs its own evidence or it is not claimed.

## Deltas
- 2026-09-08 · authored at bundle init; reconciles ADD's loop with the review roster and commit
  discipline CLAUDE.md already established.
