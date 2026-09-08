---
type: Spec
title: Method
lens: method
project: yowo
description: how work proceeds, and what a gate costs
tags: [add, tdd, review]
sources: [CLAUDE.md, .add/index.md]
generated: { by: add/3.5.0, at: 2026-09-08 }
delta_seq: 3
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
- [ADD · M3 · open · 2026-09-08] A check that asserts a workflow's shape cannot see whether the workflow WORKS. `publish` was gated on a step output that nothing wrote — every check green, PyPI publish silently never running. Static config assertions need a companion claim about what they cannot prove. (evidence: /tasks/pypi-trusted-publish.md)
- [ADD · M2 · open · 2026-09-08] hatchling's sdist `include` ADDS to a whole-repo sweep; only `only-include` restricts it. A probe before freeze caught this — the contract would have frozen the wrong key and shipped green checks over a still-broken artifact. (evidence: /tasks/sdist-manifest.md)
- [ADD · M1 · open · 2026-09-08] A roadmap drafted as milestones + tasks with dependencies written in prose is a partition, not a DAG. 'milestone:' is the ONE edge key that accepts a bare slug (add.py:452); every other key — depends_on: included — needs a full '/tasks/<slug>.md' cid or the value stays silently unresolved and yields no edge. Symptom: graph.json shows only milestone edges, and 'add status' walks slug order, so the resume point lands in a late milestone on a task whose real prerequisites do not exist yet. Draw depends_on edges when the tasks are created, then confirm with 'add wave <milestone>' that the levels match the intended order. (evidence: 25e907e)
  discipline CLAUDE.md already established.
