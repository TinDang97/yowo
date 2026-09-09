---
type: Task
title: Pinned SHA-256 verified on download and on cache hit; weights_only=True
status: direction
depth: deep
sensitivity: security
milestone: m1-trust-the-ship
scope:
  - src/yowo/models/
  - src/yowo/arch/_weights.py
gives:
  - S1 the registry entry for each model — `default_weights_url` and its pinned `sha256`
  - S2 `resolve_weights()` — verification on first download AND on every cache hit
  - S3 the verified `state_dict` sidecar, and the load path that reads it with `weights_only=True`
depends_on:
  - /tasks/checkpoint-loader.md
generated: { by: add/3.5.0, at: 2026-09-08 }
verified: []
advised_by: security-reviewer
---
## CARD
goal: No weight is deserialized before its pinned SHA-256 is verified — on first download and on every cache hit — and the routine load path never unpickles at all.
why: `models/_weights.py` reads `content-length` only to size a progress bar, so a truncated or substituted download is `os.replace`d into cache as success, and `if dest.exists()` then trusts it forever. `ModelMeta` carries a URL and no digest. Meanwhile `models/README.md:67` already TELLS the reader that "SHA-256 digest checked against ... sidecar `.sha256` file" and that the atomic write verifies the hash — neither exists, so the documentation currently asserts a control an auditor would rely on.

DECIDED BY THE HUMAN, 2026-09-08 — cache migration is **warn and re-fetch**: a cached weight with no verified digest is re-downloaded once, verified, and replaced, with a warning saying why. Rejected: hard-fail (breaks working and air-gapped installs on upgrade) and grandfathering (leaves exactly the hole this task exists to close).

DECIDED BY THE HUMAN, 2026-09-09 (delegated: "pick and proceed on the recommendations"), three calls:
  1. `weights_only=True` — m1 box 5 requires it, and `arch/_weights.py:86` records that it CANNOT load an
     ultralytics checkpoint: it refuses the `nn.Module` the weights live inside. Resolution: verify the
     SHA-256, unpickle ONCE through `checkpoint-loader`'s restricted allowlist against bytes already
     proven to match the pin, re-serialize as a plain tensor `state_dict` beside the cache entry, and
     load THAT with `weights_only=True` forever after. The box is satisfied literally, and the dangerous
     unpickle happens once against verified bytes instead of on every load.
     Rejected: amending the box to bless the allowlist (leaves a pickle executing on every load);
     amending plus a separate conversion task (same, deferred).
  2. Digests — fetch all 10 and pin real values, rather than pinning only what happened to be cached.
  3. README — left as written; this task makes it true, and a check asserts the claim matches the code.

FOUND DURING GROUNDING, 2026-09-09 — a live bug this task must fix first, because a digest cannot be
pinned for a file that cannot be fetched: `_registry.py:121` builds EVERY detection model's URL from
`_ASSETS_BASE` (v8.3.0), so all five YOLO26 detection weights 404. Verified:
`v8.3.0/yolo26n.pt -> 404`, `v8.4.0/yolo26n.pt -> 200`. `_ASSETS_V84` already exists and
`_make_cls_meta:151` picks it correctly; only the detection path is wrong. Half the models yowo
advertises are undownloadable today.
beat: scaffold · next: author weight-integrity's RULES, ASSUMPTIONS and CHECKS, then add freeze weight-integrity

## RULES
<must>
- M1 Every registered model carries a pinned SHA-256, and every registered URL resolves — a model that
  cannot be fetched cannot be verified.
- M2 A downloaded weight is verified against its pin BEFORE it is moved into the cache. A mismatch
  leaves no file behind.
- M3 A cache HIT is verified too, on every load, not only on first download.
- M4 The routine load path does not unpickle: after one verified conversion, weights load via
  `torch.load(..., weights_only=True)` from a plain `state_dict`.
- M5 A cached weight predating verification is re-fetched once, verified and replaced, with a warning
  saying why — never hard-failed, never grandfathered.
- M6 `models/README.md`'s description of verification matches the code.
</must>
<reject>
- R:UNVERIFIED No bytes from the network or from cache may be deserialized before their digest matches
  the pin -> "UNVERIFIED"
- R:SILENT A verification failure may never be downgraded to a warning or a fallback to the unverified
  file -> "SILENT"
</reject>

## ASSUMPTIONS
- A1 [who] covers: S1 · the request does not say who may pin a digest; taking: only a maintainer editing
  the registry, and a custom model registered by a user may carry `sha256=None` meaning "unpinned"
  -> forcing a pin on user-registered models breaks every private-bucket deployment.
- A2 [which] covers: S1 · the request does not say which models must be pinned; taking: all 10 built-ins,
  digests captured from the canonical release assets · found: all 10 fetched and hashed; yolo11n's pin
  matches both the local cache entry and the stray repo-root copy byte for byte (evidence:
  `shasum -a 256 ~/.cache/yowo/weights/yolo11/n/yolo11n.pt` == pinned value)
  -> a wrong pin bricks a model more thoroughly than no pin.
- A3 [when] covers: S1 · the request does not say what happens when upstream republishes a tag with new
  bytes; taking: verification FAILS loudly and a maintainer re-pins deliberately
  -> auto-accepting new bytes under an old pin makes the pin decorative.
- A4 [absent] covers: S1 · the request does not say what an unpinned model does; taking: it downloads
  and loads with a WARNING naming the model, not a refusal — built-ins are all pinned, so this only
  affects user-registered models (A1) -> refusing would break custom models on upgrade.
- A5 [order] n/a · the registry is a mapping; no ordering is constrained.
- A6 [experience] covers: S1 · the request does not say who reads a pin; taking: a maintainer bumping a
  model version, so each pin sits beside its URL in the registry rather than in a separate manifest
  -> a digest file that drifts from the URL list is worse than no digest.
- A7 [who] n/a · `resolve_weights` has no authorization surface.
- A8 [which] covers: S2 · the request does not say which failures delete the file; taking: BOTH a digest
  mismatch and a truncated download leave nothing behind — verification happens on the `.tmp` file
  before `os.replace` -> a half-file in cache is trusted forever by `if dest.exists()`, which is the
  present bug.
- A9 [when] covers: S2 · the request does not say how often a cache hit is verified; taking: EVERY load,
  not once per process — hashing 5-110 MB costs tens of milliseconds against a model load already in
  the hundreds · probe: the added cost is small relative to load time
  -> verifying once per process leaves a window where a file swapped mid-run is trusted.
- A10 [absent] covers: S2 · the request does not say what happens with no network and a bad cache entry;
  taking: raise, naming the file and the expected digest, rather than silently using it
  -> R:SILENT exists precisely because this is the tempting place to fall back.
- A11 [order] covers: S2 · the request does not say the order of verify vs convert; taking: verify the
  raw `.pt` FIRST, then convert — conversion unpickles, and unpickling unverified bytes is the whole
  risk -> converting first would execute the payload before checking it.
- A12 [experience] covers: S2 · the request does not say what a mismatch says; taking: the model name,
  the expected and actual digests, and the file path — enough to decide "corrupt download" vs "upstream
  changed" without re-deriving it -> "checksum mismatch" alone sends the reader hunting.
- A13 [who] n/a · the sidecar is a cache file; no authorization surface.
- A14 [which] covers: S3 · the request does not say what the sidecar contains; taking: only tensors —
  the mapped `state_dict` — so nothing in it can execute on load
  -> carrying any non-tensor object forward re-opens the unpickle hole in a new file.
- A15 [when] covers: S3 · the request does not say when conversion happens; taking: lazily, on first
  load after verification, not at download time -> converting at download time doubles the wait for a
  user who may never load that model.
- A16 [absent] covers: S3 · the request does not say what happens if the sidecar is missing or stale;
  taking: fall back to verify-and-reconvert from the raw `.pt`, never to loading the sidecar unchecked
  -> a deleted sidecar becomes an unverified load path.
- A17 [order] covers: S3 · the request does not say what invalidates a sidecar; taking: it is keyed to
  the raw file's digest, so a re-pinned or re-fetched weight regenerates it automatically
  -> a stale sidecar from a previous version silently serves old weights.
- A18 [experience] covers: S3 · the request does not say who notices conversion; taking: a one-time log
  line naming the model, since the first load is measurably slower than later ones
  -> an unexplained one-off delay reads as a hang.

## PLAN
contract:
  - S1 `ModelMeta.sha256: str | None`; `_make_meta` picks `_ASSETS_V84` for YOLO26 as `_make_cls_meta`
    already does; all 10 built-ins carry real pins.
  - S2 `resolve_weights` verifies the `.tmp` before `os.replace`, and verifies on every cache hit;
    warn-and-re-fetch for an unverifiable cached file (M5).
  - S3 `<name>.state_dict.pt` beside the cache entry, keyed to the raw digest, loaded with
    `weights_only=True`; produced by one restricted-unpickler pass over verified bytes.
strategy: fix the URL first (nothing else is testable without it), then verification, then conversion.
regression floor: `test_registry.py`, `test_classify_weights.py`, `test_checkpoint_loader.py` and
  `test_arch_model.py` stay green.

## EDGES
- E1 A digest mismatch must leave NO file in the cache — not the bad one, not a partial.
- E2 A weight cached before this task existed must not hard-fail (M5), and must not be trusted (R:UNVERIFIED).
- E3 A user-registered model with `sha256=None` loads with a warning, not a refusal (A1, A4).

## CHECKS
- test_every_builtin_carries_a_pinned_digest · covers: M1, A2 · all 10, no None.
- test_yolo26_urls_use_the_release_that_has_them · covers: M1 · asserts URL CONSTRUCTION, not live
  resolution — a unit test must not put the network in the suite. The 404 was verified by hand during
  grounding (v8.3.0 -> 404, v8.4.0 -> 200) and that evidence is in the CARD.
- test_download_verifies_before_moving_into_cache · covers: M2, R:UNVERIFIED, A8, E1 · mismatch leaves nothing.
- test_cache_hit_is_verified_every_load · covers: M3, A9 · a swapped cache file is caught.
- test_routine_load_uses_weights_only · covers: M4, A14 · the steady-state path does not unpickle.
- test_conversion_happens_after_verification · covers: A11 · unverified bytes are never unpickled.
- test_unverifiable_cached_weight_is_refetched_with_a_warning · covers: M5, E2 · warn and re-fetch.
- test_mismatch_never_falls_back_to_the_bad_file · covers: R:SILENT, A10 · raises, offline included.
- test_unpinned_custom_model_warns_but_loads · covers: A1, A4, E3 · custom models keep working.
- test_stale_sidecar_is_regenerated_not_trusted · covers: A16, A17 · keyed to the raw digest.
- test_readme_matches_the_implementation · covers: M6 · the doc's claim is asserted against the code.
red-first: every check MUST fail first.

## EVIDENCE
receipt: <runs/<n>.md>
gate: <PASS | RISK-ACCEPTED | HARD-STOP>

## LESSONS
- <lesson> -> add learn <lens>
