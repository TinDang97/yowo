---
type: Task
title: The state_dict sidecar cannot attest to itself with a public value
status: done
depth: quick
sensitivity: security
milestone: m1-trust-the-ship
scope:
  - src/yowo/arch/
  - src/yowo/models/
  - tests/unit/
gives:
  - S1 the trust rule that decides whether a state_dict sidecar may be believed
generated: { by: add/3.5.0, at: 2026-09-10 }
verified:
  - { by: "Tin Dang", at: 2026-09-10, act: interview, authority: human, interview: "sha256:123ec1bbc42b451a", receipt: /tasks/sidecar-not-self-attesting.d/interviews/1.md, answers: "A1=confirm|A2=confirm|A3=confirm|A4=confirm|A5=confirm|A6=confirm|R:SELFKEYED=confirm|R:SILENTSERVE=confirm|R:REHASH=confirm" }
  - { by: "Tin Dang", at: 2026-09-10, act: freeze, authority: human, direction: "sha256:8576e08f4de7108b", binding: "sha256:ddf7e60f801608c3" }
  - { by: "cli", at: 2026-09-10, act: brief, authority: process, brief: "sha256:b172e589b5f81964" }
  - { by: "Tin Dang", at: 2026-09-10, act: refreeze, authority: human, direction: "sha256:8576e08f4de7108b", binding: "sha256:ddf7e60f801608c3" }
  - { by: "cli", at: 2026-09-10, act: brief, authority: process, brief: "sha256:bc391cf69dce4c2a" }
  - { by: "process:run", at: 2026-09-10, act: run, authority: process, outcome: PASS, receipt: /tasks/sidecar-not-self-attesting.d/runs/1.md }
  - { by: "Tin Dang", at: 2026-09-10, act: refreeze, authority: human, direction: "sha256:d957fa80b9b17f99", binding: "sha256:ddf7e60f801608c3" }
  - { by: "cli", at: 2026-09-10, act: brief, authority: process, brief: "sha256:340d0dc6f20f0d65" }
  - { by: "process:run", at: 2026-09-10, act: run, authority: process, outcome: PASS, receipt: /tasks/sidecar-not-self-attesting.d/runs/2.md }
  - { by: "Tin Dang", at: 2026-09-10, act: gate, authority: human, outcome: PASS, receipt: /tasks/sidecar-not-self-attesting.d/runs/2.md, brief: "sha256:340d0dc6f20f0d65", reason: "Ten checks green on a bound receipt, every rule proven. Verified independently, not on the builder's word: my own reproduction - a sidecar carrying the published registry pin beside a garbage raw file - now raises WeightIntegrityError instead of serving the planted tensor, and with the raw file deleted it raises ModelNotFoundError naming the checkpoint. I counted hashlib.sha256 constructions through load_verified_state_dict myself, at the primitive rather than at _raw_digest so a repair that re-hashed via file_digest could not hide, and got exactly one on the pinned fast path and one on the unpinned fast path: M3 and R:REHASH hold by measurement, not by assertion. The design separates the two questions properly - the pin authenticates the FILE against a measurement taken this call, the file's own digest authenticates the SIDECAR - so neither borrows the other's authority and R:SELFKEYED is closed rather than relocated. SCOPE WIDENED to src/yowo/models/ and refrozen before this gate rather than gated over a silent overrun. verify_digest gained an optional measured= parameter because M3 and R:REHASH cannot both hold without it; the alternatives were a second WeightIntegrityError construction site, which would break test_no_second_integrity_message_was_authored, and an 'if raw != pin' guard, which would re-read the file and open a real TOCTOU. That parameter is a genuine footgun and is documented as one: a way to avoid re-reading a file, never a way to supply the answer. CITATION QUALIFIED before this gate: the first attempt was REFUSED because M5 had no passing check, and the cause was that three test files now define test_load_verified_state_dict_signature_is_unchanged, so a bare citation resolved to ambiguous and proved nothing. The refusal was correct and the tool caught what I would not have. RECORDED EXCEPTION to red-first: four of the ten checks are preservation checks - E2, E3, E5 and the signature check - green before the fix. A check asserting behaviour must SURVIVE cannot be red without testing something it does not claim; the builder said so rather than manufacturing failures. The six that bind the defect were each red for the right reason, the first by returning the planted tensors. Sibling repair: verified-digest-threading's M5 mandated this defect in words and its check asserted it with assert_not_called; both amended and that node re-gated first." }
advised_by: artifact-integrity-steward
---
## CARD
goal: A `.state_dict.pt` sidecar is trusted only after the raw checkpoint it claims to describe has been hashed and matched, so a forged sidecar cannot serve tensors under a valid-looking pin.
why: `load_verified_state_dict` keys the sidecar on `sidecar_key`, which is the registry pin whenever one is supplied — and the registry pin is PUBLIC, printed in `src/yowo/models/_registry.py`. The fast path returns before `verify_digest(checkpoint_path, pin)` runs, so anyone who can write `~/.cache/yowo/weights/` can drop a sidecar carrying `raw_sha256: <the published pin>` and arbitrary tensors, and it is served with no error and no read of the raw file. Reproduced 2026-09-10 against a garbage raw checkpoint: the poisoned tensor came back, the raw file was never hashed. Code execution stays blocked because the sidecar loads under `weights_only=True`; weight integrity does not. Silently serving a backdoored detector is the harm `weight-integrity` exists to prevent, and this path routes around its control. The irony that makes it easy to miss: the UNPINNED branch already hashes the raw file, because `sidecar_key` falls back to `_raw_digest(checkpoint_path)`. Only the pinned branch — the one that is supposed to be stronger — skips it.
beat: done · next: add status

## RULES
<must>
- M1 No sidecar is trusted before the raw checkpoint file has been hashed on this call and that hash has been compared to an authenticating value.
- M2 The value a sidecar is matched against is derived from the raw file's bytes, never from a value the caller supplied. A public constant may confirm a digest; it may never stand in for one.
- M3 The raw file is hashed at most once per call. The fix closes a hole; it does not double the cost of the path it fixes.
- M4 A sidecar that fails to match is discarded and the checkpoint reconverted, not raised on. A stale sidecar from an older pin is the ordinary case, not an attack.
- M5 `load_verified_state_dict`'s signature is unchanged. This node hardens a body; three callers and the export path depend on the shape.
</must>
<reject>
- R:SELFKEYED A sidecar authenticated by a value that is public, caller-supplied, or stored in the sidecar itself. -> "SELFKEYED"
- R:SILENTSERVE A mismatch that returns tensors anyway, on any branch. -> "SILENTSERVE"
- R:REHASH A repaired path that hashes the raw checkpoint more than once for one load. -> "REHASH"
</reject>

## ASSUMPTIONS
- A1 [who] covers: S1 · the request does not say whose write access is in the threat model; taking any principal that can write `~/.cache/yowo/weights/` — a second local account, a shared CI cache, a synced directory, a restore from backup — and NOT a remote attacker, since the sidecar is never downloaded -> if wrong and only root is in scope, the fix is unnecessary work that costs one hash per load.
- A2 [which] covers: S1 · the request does not say which sidecars are in scope; taking every sidecar `load_verified_state_dict` reads, pinned and unpinned alike, since a single trusted branch is the whole hole -> if wrong, an unpinned path pays for a guarantee nobody asked it for.
- A3 [when] covers: S1 · the request does not say where the boundary between "verify" and "trust" falls; taking the raw hash MUST be computed before the sidecar's contents are read back, not merely before they are returned -> if wrong and the check lands after the read, a malformed sidecar is deserialized on attacker terms before anything rejects it.
- A4 [absent] covers: S1 · the request does not say what a missing raw checkpoint means when a sidecar exists; taking it as an error — a sidecar alone is not a loadable model, and letting it serve is exactly R:SELFKEYED with the file deleted -> if wrong, a user who deletes the `.pt` to save disk and keeps the sidecar loses a workflow that works today. · probe: deleting the raw file and loading must raise, naming the missing checkpoint.
- A5 [order] covers: S1 · the request does not say what orders the comparison when both a pin and a raw digest exist; taking raw-digest-first — hash the file, then compare that hash to the sidecar's claim AND to the pin, so one hash serves both questions and the pin never substitutes for the measurement -> if wrong and the pin is consulted first, M3 and M1 cannot both hold.
- A6 [experience] covers: S1 · the request does not say who receives a rejection; taking the operator running inference, who needs to know the cache is untrustworthy rather than that a hash differed — the message must say which file, that it was discarded, and that it is being reconverted -> if wrong, a routine stale-sidecar reconversion reads as a break-in.

## PLAN
SCOPE WIDENED 2026-09-10 to `src/yowo/models/` — M3 and R:REHASH cannot both hold without it.
After measuring the file to authenticate its sidecar, calling `verify_digest(path, pin)` would hash
5-110 MB a second time. Constructing `WeightIntegrityError` inside `arch/_weights.py` instead would
break `test_no_second_integrity_message_was_authored`, a frozen invariant of another node; guarding the
call with `if raw != pin` would re-read the file and open a real TOCTOU where a file changed between
the two reads passes. An optional `measured=` parameter on `verify_digest` keeps one message, one raise
site and one read.

contract: `load_verified_state_dict(checkpoint_path, raw_digest=None) -> dict[str, Tensor]`, unchanged. Internally: compute `raw = _raw_digest(checkpoint_path)` once, unconditionally. Trust a sidecar only when `blob["raw_sha256"] == raw`. Separately, when `raw_digest` is not None, compare `raw` to it and refuse on mismatch. The pin authenticates the FILE; the file's own digest authenticates the SIDECAR. Neither borrows the other's authority.

## EDGES
- E1 A sidecar carrying the public registry pin as `raw_sha256`, beside a raw file whose bytes hash to something else — the reproduction. Must refuse, must not return tensors.
- E2 A genuine sidecar beside a genuine file, pinned — must still take the fast path and must not unpickle the raw checkpoint.
- E3 A stale sidecar from a previous pin — discarded and reconverted, no error surfaced to the caller.
- E4 A sidecar with no raw file present — refused, naming the missing checkpoint (A4).
- E5 The unpinned path (`raw_digest=None`) — behaviour unchanged, and still exactly one hash.

## CHECKS
all in `tests/unit/test_sidecar_authentication.py`.
- test_a_sidecar_keyed_on_the_public_pin_is_refused · covers: M1, M2, A1, E1, R:SELFKEYED ·
  the reproduction: a locally planted sidecar carrying the published registry pin, beside a raw
  file that hashes to something else, must not yield tensors.
- test_no_branch_returns_tensors_on_a_mismatch · covers: R:SILENTSERVE ·
  every branch that can reach a sidecar refuses on mismatch, not just the pinned one.
- test_the_raw_file_is_hashed_exactly_once · covers: M3, A5, R:REHASH ·
  one measurement answers both questions; the fix does not double the cost of the path it fixes.
- test_the_hash_precedes_reading_the_sidecar_contents · covers: A3 ·
  the raw digest is computed before the sidecar is deserialized, not merely before it is returned.
- test_a_genuine_sidecar_still_takes_the_fast_path · covers: M4, E2 ·
  a real sidecar beside a real file is still believed, and the raw checkpoint is not unpickled.
- test_a_stale_sidecar_is_discarded_and_reconverted · covers: M4, E3 ·
  a sidecar from a previous pin is the ordinary case: discarded, reconverted, no error raised.
- test_a_discarded_sidecar_reports_why · covers: A6 ·
  the message names the file and says it is being reconverted, so a routine event does not read
  as a break-in.
- test_a_sidecar_without_its_raw_checkpoint_is_refused · covers: A4, E4 ·
  a sidecar alone is unauthenticatable; the refusal names the missing checkpoint.
- test_the_unpinned_path_is_unchanged · covers: A2, E5 ·
  `raw_digest=None` behaves as before and still hashes exactly once.
- tests.unit.test_sidecar_authentication::test_load_verified_state_dict_signature_is_unchanged · covers: M5 ·
  this node hardens a body; the three inference callers and the export path stay valid. Qualified with
  its module, not cited bare: three test files now define this name, and a bare citation matched
  against the tail of every reported id resolves to AMBIGUOUS the moment a receipt runs two of them —
  proving nothing, silently. The qualified form is add's own ID grammar.
red-first: every check MUST fail first.

## EVIDENCE
receipt: <runs/<n>.md>
gate: <PASS | RISK-ACCEPTED | HARD-STOP>

## LESSONS
- <lesson> -> add learn <lens>
