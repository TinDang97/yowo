---
type: Spec
title: Method
lens: method
project: yowo
description: how work proceeds, and what a gate costs
tags: [add, tdd, review]
sources: [CLAUDE.md, .add/index.md]
generated: { by: add/3.5.0, at: 2026-09-08 }
delta_seq: 7
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
- [ADD · M11 · open · 2026-09-10] M7 again, one step earlier: a FROZEN node that is not COMMITTED is invisible to a worktree too. `task-aware-weight-resolution` was authored, interviewed, advised and frozen, and none of it was in a commit — the builder's worktree, cut from `main`, saw nothing, read the node out of the shared checkout instead, and said so in its report. It was right to. Freezing records a decision; committing is what publishes it. The check before spawning is `git status` on `.add/tasks/`, not `add show`. Same for a script the node's CHECKS name: `scripts/measure_weight_digests.py` was committed on a feature branch, so the builder could not see that either, and a check citing it would have failed for a reason having nothing to do with the code. (evidence: /tasks/task-aware-weight-resolution.md gate reason)
- [ADD · M10 · open · 2026-09-10] A bare check name that more than one test module defines resolves to AMBIGUOUS and proves nothing. `resolve_check` (add.py:4671) matches a bare citation against the TAIL of every reported id; two or more hits prove nothing, deliberately, because a name that means two tests cannot entitle a claim about one. The trap is that it depends on the RECEIPT COMMAND, not on the citation: `test_load_verified_state_dict_signature_is_unchanged` is defined in three files here, and `export-digest-threading` gated green on the bare name only because its receipt ran one file. Add a second file to the command and the same citation silently stops binding. `sidecar-not-self-attesting`'s gate was REFUSED for exactly this - M5 had no passing check while the test was passing - and the refusal was right. Fix is the qualified grammar, `tests.unit.<module>::<name>`, which is add's own ID shape; before citing a name bare, grep the test tree for other definitions of it. (evidence: /tasks/sidecar-not-self-attesting.d/runs/1.md refused vs runs/2.md)
- [ADD · M9 · open · 2026-09-10] A check written as `assert "<token>" in inspect.getsource(...)` proves the token is spelled somewhere, not that the behaviour happens. Four of the five checks in `test_weight_integrity.py` are this shape and every one survives deleting the thing it names: `test_cache_hit_is_verified_every_load` is `"sha256" in src or "verify" in src` and stays green with `verify_digest` deleted, because `meta.sha256 is None` still spells sha256; `test_unverifiable_cached_weight_is_refetched_with_a_warning` is `"warn" in src.lower()` over the WHOLE module and stays green with every warning call deleted, because `import warnings` spells warn; `test_download_verifies_before_moving_into_cache` stays green with the verification moved AFTER `os.replace`, which is the precise defect it names. The node's own replan record shows a builder already tried to strengthen one of these by stripping comments before matching — and it is still vacuous, because stripping comments does not strip DOCSTRINGS, and the docstring explaining why `weights_only=True` cannot be used contains the string it searches for. A source-substring assertion is admissible only ALONGSIDE a behavioural one, never instead of it; the test that a check is real is whether it fails when you delete the code it names, and that mutation takes a minute to run. (evidence: tests/unit/test_weight_integrity.py, /tasks/weight-integrity.md replan record)
- [ADD · M8 · open · 2026-09-10] A CHECKS line whose `covers:` list wraps to the next line registers NOTHING. `COVERS_IN_CHECK` (add.py:4100) matches per line and needs `- <id> · covers: <rules> ·` with the closing separator on the SAME line; a wrap after `covers: M3,` yields no citation at all, so the check is invisible to `bind` and its id never enters the receipt's `passed:` list. It fails silently and it fails UPWARD: the gate still passed here because a sibling parametrised case cited the same two rules, so nothing read as unbound - the line simply claimed a binding it did not make. Reflow so the trailing `·` closes on the id's own line, and check the receipt's `passed:` count against the runner's own total: 13 tests but 12 recorded ids is the tell. (evidence: /tasks/export-digest-threading.d/runs/3.md vs runs/5.md)
- [ADD · M7 · open · 2026-09-10] A subagent worktree branches from main, not from the orchestrator's checked-out branch, so a frozen node committed only on a feature branch is INVISIBLE inside it. Three builds in a row read a stale scaffold and worked from their spawn prompt instead. Carrying the frozen decisions inline in the prompt is the fix and it works; asserting 'this worktree's copy IS current' is the error, and it was still wrong the second time after being corrected once. Either merge the direction to main before spawning, or tell the builder plainly that its bundle copy is stale and the prompt is authoritative. (evidence: /tasks/export-digest-threading.d)
- [ADD · M6 · open · 2026-09-09] When an ASSUMPTION's parenthetical names concrete entries and a MUST says 'observed only', the MUST wins and the delta is reported at the gate. narrow-loader-allowlist's A3 and A2 named torch.nn.parameter.Parameter and nn.Flatten; neither is ever constructed - parameters travel as torch._utils._rebuild_parameter, and both our Classify and upstream's call .flatten(1). Authoring from the measurement rather than from the assumption's wording kept two unobserved entries out of a trust boundary. (evidence: /tasks/narrow-loader-allowlist.d/runs/4.md)
- [ADD · M5 · open · 2026-09-09] Bind every Must and Reject to a covers: key as you write CHECKS, not at gate time. Two rules went unbound here and the gate caught it after the freeze, forcing a refreeze+rebrief cycle for what was pure bookkeeping. (evidence: /tasks/rtsp-credential-redaction.md)
- [ADD · M4 · open · 2026-09-09] Building several frozen tasks in one working tree defeats the gate's scope check: each node's gate sees the others' edits as undeclared sensitive changes. Committing them together makes the refusal vanish without fixing anything — the honest replay is one task alone in the tree, gated, then committed. (evidence: /tasks/version-single-source.md)
- [ADD · M3 · open · 2026-09-08] A check that asserts a workflow's shape cannot see whether the workflow WORKS. `publish` was gated on a step output that nothing wrote — every check green, PyPI publish silently never running. Static config assertions need a companion claim about what they cannot prove. (evidence: /tasks/pypi-trusted-publish.md)
- [ADD · M2 · open · 2026-09-08] hatchling's sdist `include` ADDS to a whole-repo sweep; only `only-include` restricts it. A probe before freeze caught this — the contract would have frozen the wrong key and shipped green checks over a still-broken artifact. (evidence: /tasks/sdist-manifest.md)
- [ADD · M1 · open · 2026-09-08] A roadmap drafted as milestones + tasks with dependencies written in prose is a partition, not a DAG. 'milestone:' is the ONE edge key that accepts a bare slug (add.py:452); every other key — depends_on: included — needs a full '/tasks/<slug>.md' cid or the value stays silently unresolved and yields no edge. Symptom: graph.json shows only milestone edges, and 'add status' walks slug order, so the resume point lands in a late milestone on a task whose real prerequisites do not exist yet. Draw depends_on edges when the tasks are created, then confirm with 'add wave <milestone>' that the levels match the intended order. (evidence: 25e907e)
  discipline CLAUDE.md already established.
