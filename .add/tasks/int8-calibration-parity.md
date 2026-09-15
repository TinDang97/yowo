---
type: Task
title: Calibrate on the inference preprocessing path; measure the delta
status: done
depth: deep
sensitivity: architecture
milestone: m4-honest-deployment
scope:
  - src/yowo/export/
  - tests/unit/test_int8_export.py
  - tests/unit/test_int8_decode_tail.py
  - tests/unit/test_int8_parity_gate.py
  - tests/integration/test_int8_parity.py
  - .github/workflows/
gives:
  - S1 decode_tail(model) -> set[str] — the post-head decode nodes, computed from the graph
  - S2 measure_int8_parity(fp32, int8, images, ...) -> ParityReport — post-NMS recall vs the FP32 source
  - S3 ParityReport — the frozen, JSON-native record of one parity measurement
  - S4 quantize_onnx_static(...) -> ParityReport — the gated write
  - S5 calibration_batches(paths, batch_size, input_size) — batches from the inference preprocessing path
  - S6 ExportMetadata.extra["int8_parity"] — the parity section of the export sidecar
depends_on:
  - /tasks/export-roundtrip-parity.md
generated: { by: add/3.5.0, at: 2026-09-08 }
verified:
  - { by: "process:run", at: 2026-09-15, act: run, authority: process, outcome: PASS, receipt: /tasks/int8-calibration-parity.d/runs/1.md }
  - { by: "process:run", at: 2026-09-15, act: run, authority: process, outcome: PASS, receipt: /tasks/int8-calibration-parity.d/runs/2.md }
  - { by: "process:run", at: 2026-09-15, act: run, authority: process, outcome: PASS, receipt: /tasks/int8-calibration-parity.d/runs/3.md }
  - { by: "process:run", at: 2026-09-15, act: run, authority: process, outcome: PASS, receipt: /tasks/int8-calibration-parity.d/runs/4.md }
  - { by: "Tin Dang", at: 2026-09-15, act: interview, authority: human, interview: "sha256:1bfaee6ece662a8c", receipt: /tasks/int8-calibration-parity.d/interviews/1.md, answers: "A2=confirm|A3=confirm|A4=confirm|A6=confirm|A8=confirm|A9=confirm|A10=confirm|A11=confirm|A12=confirm|A14=confirm|A16=confirm|A20=confirm|A21=confirm|A22=confirm|A24=confirm|A26=confirm|A28=confirm|A30=confirm|A31=confirm|A34=confirm|A36=confirm|R:SILENTZERO=confirm|R:PARTIAL=confirm|R:HARDCODEDTAIL=confirm|R:VACUOUS=confirm|R:UNMEASURABLE=confirm" }
  - { by: "Tin Dang", at: 2026-09-15, act: freeze, authority: human, direction: "sha256:471b1427462b29f8", binding: "sha256:255682ab551700d2" }
  - { by: "cli", at: 2026-09-15, act: brief, authority: process, brief: "sha256:77b19d0b33ed4056" }
  - { by: "process:run", at: 2026-09-15, act: run, authority: process, outcome: PASS, receipt: /tasks/int8-calibration-parity.d/runs/5.md }
  - { by: "Tin Dang", at: 2026-09-15, act: refreeze, authority: human, direction: "sha256:471b1427462b29f8", binding: "sha256:255682ab551700d2" }
  - { by: "cli", at: 2026-09-15, act: brief, authority: process, brief: "sha256:98b8221b25f68437" }
  - { by: "process:run", at: 2026-09-15, act: run, authority: process, outcome: PASS, receipt: /tasks/int8-calibration-parity.d/runs/6.md }
  - { by: "Tin Dang", at: 2026-09-15, act: refreeze, authority: human, direction: "sha256:d0ee7982cb584b09", binding: "sha256:255682ab551700d2" }
  - { by: "cli", at: 2026-09-15, act: brief, authority: process, brief: "sha256:875d8951cdc61cee" }
  - { by: "process:run", at: 2026-09-15, act: run, authority: process, outcome: PASS, receipt: /tasks/int8-calibration-parity.d/runs/7.md }
  - { by: "Tin Dang", at: 2026-09-15, act: gate, authority: human, outcome: PASS, receipt: /tasks/int8-calibration-parity.d/runs/7.md, brief: "sha256:875d8951cdc61cee" }
  - { by: loop, at: 2026-09-15, act: reopen, to: verify, reason: "Rebased onto main after #51 merged; .github/workflows/ci.yml is in scope and changed in the rebase conflict resolution, so the receipt's freshness claim no longer held." }
  - { by: "process:run", at: 2026-09-15, act: run, authority: process, outcome: PASS, receipt: /tasks/int8-calibration-parity.d/runs/8.md }
  - { by: "Tin Dang", at: 2026-09-15, act: gate, authority: human, outcome: PASS, receipt: /tasks/int8-calibration-parity.d/runs/8.md, brief: "sha256:b1c9c53136856856" }
advised_by: artifact-integrity-steward
---
## CARD
goal: An ONNX INT8 artifact either detects what its FP32 source detects, or it is never written.
why: The shipping INT8 export produces a well-formed (1,84,8400) tensor of zeros — an artifact that lies, which is worse than one that crashes, because the crash is caught and the lie is deployed.
beat: done · next: add status

sensitivity: stamped `architecture` 2026-09-15, on the executing agent's own recommendation and against the orchestrator's initial lean toward `security`. The agent's argument was accepted: this node publishes contract surfaces and changes `export_model`'s behaviour, but touches no credential or authorization boundary, and diluting the HARD-STOP lens costs the signal it exists to carry. The human floor was already unavoidable regardless — `src/yowo/export/` is listed in `.add/index.md`'s `sensitive_paths:`. One genuinely security-shaped finding did surface and was fixed in the build: the sidecar was about to ship absolute build-machine paths (`/Users/<username>/...`) to every artifact recipient, and `ParityReport.images` now stores basenames.

DECIDED 2026-09-15 by the orchestrator, with the human away and both calls flagged for reversal:
  (i) THE FLOOR GATES RECALL ONLY, for now. The repaired artifact emits 6 boxes where FP32 emits 5, at `total_recall: 1.0` — nothing refuses a false-positive flood. `int8_unmatched` RECORDS it; no rule gates it. Shipping the recall gate is the whole win (0 detections -> 5), and a precision floor picked from one image on one model would repeat the exact trap the recall floor already sits in. RESIDUAL, named not buried: the 6th box is class 7 at 0.2579, sitting just above the 0.25 default threshold, and nobody has yet established whether it is a duplicate, an NMS artifact, or a genuine weak find FP32 missed. A `parity_precision_floor` is the follow-up.
  (ii) TWO COMBINATIONS ARE REFUSED BY NAME: INT8 + `kv_cache` (a multi-input graph the single-input quantizer cannot address) and INT8 for `classify`/`obb` (the detection decoder cannot decode either output). This is a user-visible change to `export_model`, and it is the node's own thesis applied consistently: neither path could produce a working artifact, and an artifact that loads and lies is worse than an export that refuses. The alternative — warn and silently emit FP32 — would reinstate the m4 defect wearing a different hat, since the sidecar would then have to either lie about the precision or contradict the request.

## RULES
<must>
- M1 `decode_tail(model)` returns the node names reachable backward from every graph output without
  crossing a `Conv`, computed from the graph on every call — no literal node list, no node-index
  heuristic, no per-family branch, no depth parameter.
- M2 `quantize_onnx_static` passes that computed set to `quantize_static` as `nodes_to_exclude`.
- M3 `calibration_batches` produces its tensors from the same preprocessing the engine runs at
  inference — it calls `yowo.io._decode.preprocess`, so calibration and inference share one
  letterbox path by construction rather than by two implementations agreeing.
- M4 Before an INT8 artifact is published, `quantize_onnx_static` measures the artifact's post-NMS
  recall against its FP32 source over a declared image set, through the project's real
  `preprocess` -> `postprocess` path, one image at a time, on a pinned CPU provider recorded on
  the report, with the FP32 session closed before the INT8 session opens.
- M5 The floor, its margin, its evaluation threshold, its match IoU, its image set and its sample
  cap are DECLARED keyword parameters of `quantize_onnx_static` with documented defaults — never
  constants in the body. `parity_floor=0.0` disables ENFORCEMENT; it never disables MEASUREMENT.
- M6 Total recall over ALL FP32 detections — not only the gated ones — appears both in the
  `ExportError` message and in `ExportMetadata.extra["int8_parity"]`, and every `ParityReport`
  field is JSON-native so `ExportMetadata.save()` cannot fail after the artifact is promoted.
- M7 A quantization refuses BY NAME any graph whose parity this gate cannot honestly measure: a
  multi-input graph (a KV-cache export) and any task the detection decoder does not decode.
- M8 The report states the provenance of its own measurement — whether the parity set was held out
  of calibration, and where it came from — so the artifact carries the limit of its own claim.
- M9 The report counts the INT8 boxes no FP32 detection claimed. Recall alone cannot see a flood of
  false positives: an artifact whose scales saturate the classification sigmoid emits spurious
  boxes everywhere, every FP32 detection finds a partner in the flood, and recall reads 1.0. That
  count is REPORTED, never gated — this node's claim is collapse, not precision.
</must>
<reject>
- R:SILENTZERO Never publish a quantized artifact whose gated recall is below the declared floor -> "SILENTZERO"
- R:PARTIAL Never leave a rejected, partial or temporary quantization artifact on disk -> "PARTIAL"
- R:HARDCODEDTAIL Never determine the excluded set from a literal list, a family branch, or a node index -> "HARDCODEDTAIL"
- R:VACUOUS Never report a pass from a measurement in which the FP32 source produced no gated detection -> "VACUOUS"
- R:UNMEASURABLE Never quantize a graph whose parity the gate cannot measure, and never skip the measurement -> "UNMEASURABLE"
</reject>

## ASSUMPTIONS
- A1 [who] covers: S1 · n/a · a pure function over an in-memory `ModelProto`; no caller identity, no user data and no authorization boundary exists on it.
- A2 [which] covers: S1 · the request does not say whether a node reachable from an output along BOTH a Conv-crossing and a Conv-free path is in; taking: the walk is a union, so Conv-free reachability on ANY path puts it in -> if wrong a shared decode node stays quantized and the output still zeroes. · probe: in a diamond graph whose other arm crosses a Conv, the shared node is still in the tail.
- A3 [when] covers: S1 · the request does not say whether the terminating `Conv` is itself inside the boundary; taking: the Conv is OUT and stays quantizable, because that is where the entire size win lives -> if wrong we lose most of the compression for no accuracy gain (the depth-1 variant was measured upstream: same detections, +85 KB).
- A4 [absent] covers: S1 · the request does not say what a tail node with an empty `name` means; taking: it is a node `nodes_to_exclude` cannot address, so the tail is unenforceable and we raise rather than return a set that silently under-excludes -> if wrong we refuse an export that would have been fine. · probe: a tail node with an empty name raises instead of being dropped from the set.
- A5 [order] covers: S1 · n/a · the return is a `set[str]`; walk order cannot change it, and the caller sorts before handing it to ORT so the artifact is reproducible.
- A6 [experience] covers: S1 · the request does not say who reads this set; taking: the export author asking why a node quantized, so the set is node NAMES — matchable against `nodes_to_exclude` and greppable in Netron — not indices -> if wrong the reader must re-derive index-to-name by hand.
- A7 [who] covers: S2 · n/a · an in-process measurement over two local files; no identity and no user data cross it.
- A8 [which] covers: S2 · the request does not say WHICH images the delta is measured over; taking: the calibration images, capped at `parity_sample`, with `parity_images` available to pass a held-out set — and the report STAMPS `held_out: false` when the default is used, so the artifact states the limit of its own claim rather than an ASSUMPTIONS line nobody re-reads -> if wrong the artifact is judged on data it was calibrated on and nothing on the artifact says so. A held-out labelled set is COCO work and belongs to /tasks/accuracy-dataset.md, not here.
- A9 [when] covers: S2 · the request does not say whether the gating band boundary is inclusive; taking: `confidence >= parity_margin * parity_threshold` is GATED, inclusive -> if wrong a detection sitting exactly at 0.50 is merely reported when it should have been gated.
- A10 [absent] covers: S2 · the request does not say what recall means when the FP32 source itself detects nothing; taking: zero gated FP32 detections is a measurement that proves nothing, so it RAISES rather than reporting 1.0 -> if wrong a blank or unlucky parity set waves every artifact through. · probe: a parity set on which FP32 produces no gated detection raises instead of passing.
- A11 [order] covers: S2 · the request does not say what matches an INT8 box to an FP32 box when several overlap; taking: FP32 detections sorted by descending confidence, each taking the highest-IoU unmatched INT8 box of the same `class_id` at or above `parity_iou` -> if wrong the greedy order is unstated and two runs give two answers, or one INT8 box satisfies two FP32 detections and inflates recall. · probe: two overlapping same-class FP32 detections are not both matched by a single INT8 box.
- A12 [experience] covers: S2 · the request does not say who reads the delta; taking: the person deciding whether to deploy, so the report carries both recalls, the counts behind them AND the parameters that produced them, because a bare ratio cannot be re-derived -> if wrong a reader sees "0.8" with no way to learn what was lost.
- A13 [who] covers: S3 · n/a · a frozen in-process dataclass; it has no access boundary of its own.
- A14 [which] covers: S3 · the request does not say whether the record names the detections it lost; taking: it carries the missed FP32 detections' class ids, confidences and source image, not only counts -> if wrong "4/5" cannot tell you the loss was the marginal 0.3977 one rather than a confident one.
- A15 [when] covers: S3 · n/a · it records one measurement at one instant and carries no validity window; stamping the artifact it describes is the exporter's job, through the existing sidecar.
- A16 [absent] covers: S3 · the request does not say what a report means when the floor was not enforced; taking: the report is produced either way and carries `enforced` plus the floor it was judged against, so even a waived export keeps its numbers -> if wrong a waiver ships with no record of what it waived.
- A17 [order] covers: S3 · n/a · a record, not a sequence; its detection lists inherit `postprocess`'s confidence-descending order and nothing reads them positionally.
- A18 [experience] covers: S3 · n/a · S6 is the surface a human reads; S3 is only the in-process shape S6 serialises, and its audience question is answered there.
- A19 [who] covers: S4 · n/a · a local file-to-file function; the only actor is the export process itself.
- A20 [which] covers: S4 · the request does not say which calls the gate applies to; taking: every INT8 ONNX quantization this function performs, with no opt-out — a graph it cannot measure is REFUSED rather than quantized unmeasured, and `parity_floor=0.0` lowers the bar without ever skipping the measurement -> if wrong, someone who genuinely needs a below-floor artifact for study must lower the floor and read the warning, which is the right amount of friction.
- A21 [when] covers: S4 · the request does not say when the artifact becomes visible at `output_path`; taking: only after the floor is cleared — quantize to a `.partial` sibling in the SAME directory (so `os.replace` is same-filesystem and atomic), promote on pass -> if wrong a reader can observe a half-written or rejected artifact. · probe: a below-floor quantization leaves nothing at `output_path` and nothing beside it.
- A22 [absent] covers: S4 · the request does not say what happens to a PRE-EXISTING file at `output_path` when the new quantization fails; taking: it is left untouched and the `ExportError` propagates, so no stale artifact is rewritten and no sidecar claims it is fresh -> if wrong a stale INT8 model quietly survives a failed re-export.
- A23 [order] covers: S4 · n/a · one call, one artifact; the `.partial` name is derived from `output_path`, so two concurrent quantizations onto the same target were already a caller error and this node does not invent a lock for one.
- A24 [experience] covers: S4 · the request does not say what a rejected export tells its caller; taking: an `ExportError` naming total recall, gated recall, the floor and the margin -> if wrong the caller reads "quantization failed" and must open the source to learn what was lost.
- A25 [who] covers: S5 · n/a · it reads local image files the caller handed it; no identity boundary.
- A26 [which] covers: S5 · the request does not say whether letterboxing changes WHICH images are usable; taking: the same set — unreadable images are still skipped with a warning and an all-unreadable chunk still yields nothing, even though `preprocess` raises `ValueError` on an empty frame list -> if wrong an export that used to run now raises on one corrupt file.
- A27 [when] covers: S5 · n/a · the batching boundary is untouched (`batch_size` chunks, a smaller tail batch); this node moves the pixel transform, not the chunking.
- A28 [absent] covers: S5 · the request does not say what fills the letterbox padding; taking: whatever `preprocess` fills it with, because the whole point is that there is no second answer to that question -> if wrong calibration sees a padding value inference never produces, which is the defect this node exists to close.
- A29 [order] covers: S5 · n/a · order is the caller's already-sorted image list and is unchanged.
- A30 [experience] covers: S5 · the request does not say who should notice this change; taking: nobody — signature, shapes and dtype are identical, so a caller sees only a different pixel distribution -> if wrong a caller relying on stretch geometry (there is none in-repo) breaks silently. · probe: the emitted batch equals `preprocess(...).data` element-for-element on a non-square image.
- A31 [who] covers: S6 · the request does not say who reads the sidecar; taking: anyone who receives the artifact, including someone who never ran the export, so the section must be self-contained — floor, margin, threshold, provider, held-out status and counts, not a bare verdict -> if wrong the sidecar is only interpretable beside the source tree that produced it.
- A32 [which] covers: S6 · n/a · the section describes exactly the one artifact the sidecar sits beside; there is no selection to make.
- A33 [when] covers: S6 · n/a · written once, in the same instant as the rest of the sidecar, through `ExportMetadata.save()`'s existing atomic write.
- A34 [absent] covers: S6 · the request does not say what the ABSENCE of the section means; taking: the export was not INT8 — every INT8 export that publishes an artifact has been measured and therefore has a report -> if wrong a reader reads "not measured" where it means "not quantized".
- A35 [order] covers: S6 · n/a · a JSON object; key order is `asdict` insertion order and nothing reads it positionally.
- A36 [experience] covers: S6 · the request does not say what the reader needs in order to ACT; taking: total recall as a number, because "0.8" is the thing that tells a deployer a detection was lost -> if wrong a verdict-only record hides the loss the milestone box exists to surface.

## PLAN
contract:

```python
def decode_tail(model: ModelProto) -> set[str]: ...          # S1

@dataclass(frozen=True)
class MissedDetection:                                       # every field JSON-native
    class_id: int
    confidence: float
    image: str

@dataclass(frozen=True)
class ParityReport:                                          # S3 — every field JSON-native
    passed: bool
    enforced: bool               # False when parity_floor == 0.0
    output_path: str
    total_recall: float          # matched / all FP32 detections
    gated_recall: float          # matched / FP32 detections at or above the margin
    fp32_detections: int
    int8_detections: int
    gated_detections: int
    gated_matched: int
    missed: tuple[MissedDetection, ...]
    floor: float
    margin: float
    confidence_threshold: float
    iou_threshold: float
    held_out: bool               # is the parity set disjoint from the calibration set?
    parity_set_source: str       # "calibration" | "declared"
    provider: str                # the pinned ORT provider that produced the number
    images: tuple[str, ...]

def measure_int8_parity(                                     # S2
    fp32_path: Path, int8_path: Path, image_paths: Sequence[Path], *,
    model_spec: ModelSpec, input_size: int = 640,
    floor: float = 1.0, margin: float = 2.0,
    confidence_threshold: float = 0.25, iou_threshold: float = 0.5,
    held_out: bool = False, parity_set_source: str = "calibration",
) -> ParityReport: ...

def quantize_onnx_static(                                    # S4
    onnx_path: Path, output_path: Path, calibration_data: str, *,
    model_spec: ModelSpec,                # required: parity cannot be measured without it
    input_size: int = 640, batch_size: int = 8,
    parity_floor: float = 1.0, parity_margin: float = 2.0,
    parity_threshold: float = 0.25, parity_iou: float = 0.5,
    parity_images: Sequence[Path] | None = None, parity_sample: int = 8,
) -> ParityReport: ...
```

`quantize_onnx_static` returns the report ALONE, not `(path, report)`. The old `-> Path` returned
the argument the caller passed in — zero information — and `_exporter.py:181-187` already ignores
it. A tuple invites `path, _ = quantize_onnx_static(...)`, an underscore that discards the accuracy
evidence; making the evidence the return value leaves nowhere to hide it. The published path is
`report.output_path`, which after the promote is the only truthful record of what landed.

strategy:
1. `decode_tail` — backward walk from every `graph.output`, terminated by `Conv`, guarded by a
   `seen` set so a malformed cyclic graph still halts. Raise `ExportError` on an unnamed tail node,
   naming its `op_type` and output tensors so it is locatable.
2. `quantize_onnx_static` refuses first (multi-input graph · non-detection task · `parity_sample`
   of 0 · an empty resolved parity set), then quantizes to `output_path.name + ".partial"` in the
   SAME directory, measures, then either `os.replace`s into place or removes the partial and
   raises. A `try/finally` globs the partial prefix — ORT can leave more than one intermediate —
   so a mid-write death leaves nothing behind.
3. The measurement runs ONE image at a time on `CPUExecutionProvider`, FP32 session first and
   closed before the INT8 session opens.
4. `calibration_batches` builds `Frame` objects and calls `yowo.io._decode.preprocess`, returning
   `.data`; the empty-chunk guard stays because `preprocess` raises on an empty frame list.
5. `_exporter.py` records `asdict(report)` at `ExportMetadata.extra["int8_parity"]` and takes the
   published path from `report.output_path`.

scope: `src/yowo/export/`, the four test files named in the frontmatter, and `.github/workflows/`.
This node does NOT touch `src/yowo/backends/`, `src/yowo/config.py` or `src/yowo/engine.py`.
AMENDED 2026-09-15: `.github/workflows/` was reserved to the orchestrator while three agents ran in
parallel, so this node never declared it — and then the orchestrator appended the node's own CI step
to that file, which the gate correctly refused as an undeclared sensitive edit. Serializing the
writes was right; leaving the path out of `scope:` was not. The step belongs to this node, so the
path does too.

regression floor: the four quality-gate commands clean —
`uv run ruff check src/ tests/` · `uv run ruff format --check src/ tests/` ·
`uv run pyright src/yowo/` · `uv run pytest tests/unit/ -q`.

what this gate PROVES, and what it does not. Quantizing the decode subgraph destroys the graph
independently of image content — measured on yolo11n, every one of the 80 class logits is
exactly 0.0 while the 4 box-geometry rows still decode — so this instrument detects
COLLAPSE with full sensitivity even measured on the calibration set. It is a collapse detector with
a recall-shaped implementation. It is NOT an accuracy certification: by default the parity set is
the calibration set, and the report stamps `held_out: false` to say so on the artifact itself. A
mAP delta against ground truth is /tasks/accuracy-dataset.md's claim, and `parity_images` is the
seam it plugs into.

scope amendment (a) — `depends_on: /tasks/accuracy-dataset.md` is CUT.
The milestone clause is "measured delta against its FP32 SOURCE". Comparing an INT8 artifact to the
FP32 graph it was quantized from needs two artifacts and some images; it needs no labels, so it
needs no dataset. A **mAP** delta against ground truth does need COCO val2017 + annotations, and
that measurement belongs to /tasks/accuracy-dataset.md — a different claim on a different node, and
this node must not block on it.

scope amendment (b) — the recorded cause on the m4 box is WRONG and is superseded.
The box attributes the zero-detection artifact to the end2end decode/topk tail
(`ReduceMax -> ArgMax -> TopK -> GatherElements -> ...`). `end2end` is YOLO26-only
(`src/yowo/arch/_config.py:69,74`). Measured here 2026-09-15: a production-path `yolo11n` FP32 ONNX
export has 325 nodes, 87 `Conv`, and ZERO `TopK` / `ArgMax` / `GatherElements` / `ReduceMax` — and it
fails IDENTICALLY, 0 detections with every class logit exactly 0.0. The end2end tail is a YOLO26-specific
SUBSET of the real culprit, which is the POST-HEAD DECODE SUBGRAPH: DFL softmax, box decode, final
sigmoid. The rule this node ships is stated over that subgraph and needs no family branch: on
`yolo11n` the walk yields 22 nodes; on `yolo26n` the same walk yields 28, sweeping up `TopK` /
`ArgMax` / `GatherElements` / `ReduceMax` without naming any of them.

scope amendment (c) — two combinations are REFUSED, and the support question is escalated.
Measured 2026-09-15: a KV-cache detect export is a MULTI-INPUT graph
(`images`, `use_cache`, `past_k_0`, `past_v_0`) with `present_k_0` / `present_v_0` outputs, and
`_exporter.py:166-186` places the KV branch and the INT8 branch in sequence with no mutual
exclusion. Walking back from the cache outputs drags Attention internals into the tail (26 nodes
instead of 22) and the gate's single-input feed cannot run the graph at all. A classify export's
tail is 5 nodes and INCLUDES the final `Gemm` — the classifier head's heaviest layer — and the
detection decoder cannot decode its output at all; an OBB export's tail is 44 nodes (Cos/Sin/Slice
angle decode) and is likewise not decodable by the detection path. This node REFUSES both by name
rather than quantizing something it cannot measure. Whether KV+INT8 and classify/OBB INT8 should be
SUPPORTED — each needing its own gate — is a scope decision for the human, not for this node.

## EDGES
- E1 A family whose graph has no topk ops (YOLO11) and one that has them (YOLO26) go through the
  same rule, with no branch and no per-family constant.
- E2 `quantize_static` raises mid-run: nothing is left at `output_path` and no partial survives.
- E3 The parity measurement itself fails (unloadable artifact, ORT session error): treated as a
  floor failure — the partial is removed and `ExportError` is raised. It is never a pass.
- E4 The FP32 source produces no gated detection on the parity set: the measurement proves nothing
  and must raise, never report recall 1.0 on a 0/0 ratio.
- E5 The published artifact must be smaller than its FP32 source — but ONLY ever asserted together
  with the detection check, never alone: the BROKEN artifact is 2,917,035 bytes against the fixed
  2,984,512, so a size assertion on its own actively prefers the bug.
- E6 A multi-input (KV-cache) graph is refused by name — never quantized with a tail that walked
  back through the cache outputs, and never fed a single input it cannot satisfy.
- E7 `dynamic_batch=False` fixes the graph's batch dimension at 1, so the gate feeds one image at a
  time unconditionally rather than inheriting the calibration batch size.
- E8 The report survives `ExportMetadata.save()` and `.load()` as JSON — a `Path` field would raise
  `TypeError` inside `json.dumps(asdict(...))` AFTER the artifact was already promoted.
- E9 `parity_sample=0`, or a `parity_images` that resolves to nothing, is the skip-measurement door
  wearing another hat and is refused.
- E10 An INT8 artifact that emits MORE boxes than its FP32 source still passes a recall floor. The
  measured run does exactly this (6 INT8 boxes against FP32's 5), so the count of unclaimed boxes
  is recorded rather than left for a reader to infer from a recall of 1.0.

## CHECKS
- test_the_report_counts_the_int8_boxes_nobody_claimed · covers: E10 · an artifact emitting three boxes against one FP32 detection reads gated_recall 1.0 and int8_unmatched 2 — recall alone calls a hallucinating model perfect, so the unclaimed count is on the report rather than left to be inferred
- test_decode_tail_follows_a_perturbed_graph · covers: M1,R:HARDCODEDTAIL · inserting one op between the head Conv and the output grows the tail by exactly that node, so the set is computed and not recited
- test_decode_tail_follows_a_shortened_graph · covers: M1,R:HARDCODEDTAIL · removing one decode op shrinks the tail by exactly that node
- test_decode_tail_stops_at_conv · covers: M1,A3 · the terminating Conv and everything above it stay out of the set
- test_decode_tail_crosses_a_diamond_once · covers: A2,A5 · a node reachable along a Conv-crossing arm and a Conv-free arm is in the tail, and appears once
- test_decode_tail_covers_every_graph_output · covers: M1 · a second graph output is walked too
- test_decode_tail_rejects_an_unnamed_tail_node · covers: A4,R:HARDCODEDTAIL · an unaddressable node raises instead of being silently dropped from the exclusion set
- test_an_unnamed_node_outside_the_tail_is_tolerated · covers: A4 · the refusal is about what must be excluded, not about graph hygiene
- test_decode_tail_names_not_indices · covers: A6 · every element is a `graph.node[i].name`, never an index
- test_decode_tail_halts_on_a_cycle · covers: M1 · a malformed cyclic graph terminates
- test_decode_tail_is_empty_when_the_output_is_a_conv · covers: A3 · a Conv-terminated output has no tail at all
- test_quantize_excludes_the_computed_tail · covers: M2 · `nodes_to_exclude` handed to `quantize_static` is exactly `sorted(decode_tail(model))`
- test_calibration_batch_equals_the_inference_preprocess · covers: M3,A30 · on a NON-SQUARE image the emitted batch equals `preprocess(...).data` element-for-element
- test_calibration_batch_letterboxes_rather_than_stretches · covers: M3,A28 · a non-square image gains 114/255 padding and keeps its aspect ratio
- test_calibration_batches_still_skip_unreadable_images · covers: A26 · the skip-and-warn behaviour survives the preprocessing switch
- test_a_recovered_strong_detection_passes · covers: M4 · same class, overlapping box, drifted confidence is still a match
- test_the_marginal_band_is_reported_not_gated · covers: M4,A12 · a detection inside the margin band is recorded as a loss but does not decide the gate
- test_a_lost_strong_detection_fails · covers: M4,R:SILENTZERO · a detection above the margin that INT8 lost fails the floor
- test_the_gated_band_boundary_is_inclusive · covers: A9 · a detection exactly at margin*threshold is gated
- test_a_detection_just_below_the_boundary_is_not_gated · covers: A9 · the other side of the same boundary
- test_one_int8_box_cannot_match_two_fp32_detections · covers: A11 · greedy matching consumes each INT8 box once
- test_matching_is_deterministic_by_descending_confidence · covers: A11 · the greedy order is stated, so two runs give one answer
- test_a_different_class_is_not_a_match · covers: M4 · right place, wrong label is not parity
- test_a_non_overlapping_box_is_not_a_match · covers: M4 · right label, wrong place is not parity
- test_no_gated_fp32_detection_raises · covers: E4,R:VACUOUS,A10 · a 0/0 recall is refused, not reported as 1.0
- test_report_names_the_detections_it_lost · covers: A14 · the report carries each missed FP32 detection's class id, confidence and image
- test_a_waived_export_still_carries_its_numbers · covers: A16,M5 · with the floor at 0.0 the report still records the floor, still measures, and marks itself unenforced
- test_floor_parameters_are_declared_with_documented_defaults · covers: M5 · floor, margin, threshold, IoU, image set and sample cap are documented keyword parameters, not body constants
- test_the_parity_set_is_capped_by_parity_sample · covers: M5 · the cap is declared and honoured
- test_an_explicit_parity_set_overrides_the_calibration_images · covers: A8 · the seam /tasks/accuracy-dataset.md plugs a held-out set into
- test_the_report_counts_the_int8_boxes_nobody_claimed · covers: M9 · a recall of 1.0 alongside two unclaimed INT8 boxes is recorded as exactly that
- test_no_false_positives_means_nothing_unmatched · covers: M9 · the other side of the same count
- test_decode_tail_keeps_both_nodes_that_share_a_name · covers: M1 · ONNX does not enforce unique node names; the walk is keyed on tensors so a duplicate name cannot hide a node from the exclusion set
- test_the_gate_runs_one_image_per_session_call · covers: M4,E7 · asserted on BEHAVIOUR — batch of 1 per run, the pinned provider, and the FP32 session released before the INT8 one opens
- test_the_calibration_batch_is_clamped_to_the_graphs_pinned_dim · covers: E7 · a graph pinning its batch dim is read rather than requiring the caller to know
- test_external_data_beside_the_partial_is_refused · covers: R:PARTIAL,R:UNMEASURABLE · a graph without its initializers is never promoted
- test_the_report_stamps_whether_the_parity_set_was_held_out · covers: M8 · the report carries held_out, its source and the provider that produced the number
- test_the_default_parity_set_is_not_held_out · covers: M8,A8 · the default judges the artifact on its own calibration data and stamps `held_out: false` saying so
- test_below_floor_is_not_written_and_raises · covers: M4,R:SILENTZERO,A21 · a quantization under the floor raises `ExportError` and leaves nothing at `output_path`
- test_below_floor_leaves_no_partial · covers: R:PARTIAL,A21 · no partial sibling survives a rejected quantization
- test_above_floor_is_written_and_returns_its_report · covers: M4,A21 · a passing quantization is promoted and returns the evidence
- test_quantize_crash_leaves_no_artifact · covers: E2,R:PARTIAL · an exception inside `quantize_static` leaves neither artifact nor partial
- test_parity_measurement_failure_is_a_floor_failure · covers: E3 · an unloadable quantized artifact raises rather than passing
- test_a_preexisting_artifact_survives_a_failed_requantization · covers: A22 · a failed re-export neither destroys nor refreshes what is already there
- test_error_message_carries_total_recall · covers: M6,A24 · the `ExportError` text names total recall, gated recall, the floor and the margin
- test_the_report_survives_the_sidecar_round_trip · covers: M6,E8 · `ExportMetadata.save()`/`.load()` round-trips the report, so no `Path` field can raise after promotion
- test_a_kv_cache_graph_is_refused_by_name · covers: M7,R:UNMEASURABLE,E6 · a multi-input graph raises naming its extra inputs, and is not quantized
- test_a_non_detection_task_is_refused_by_name · covers: M7,R:UNMEASURABLE · classify and obb raise rather than being measured with the detection decoder
- test_an_empty_parity_set_is_refused · covers: E9,R:UNMEASURABLE · `parity_sample=0` and an empty `parity_images` are the skip door and are refused
- test_real_int8_export_detects_what_fp32_detects · covers: M4,E5 · REAL export + REAL quantization of `yolo11n`; the published artifact clears the floor AND is smaller than its FP32 source
- test_excluding_the_decode_tail_is_what_makes_int8_detect · covers: M1,M2,R:SILENTZERO · the same real graph quantized WITHOUT the exclusion still yields zero detections, so the exclusion is what is doing the work
- test_the_same_rule_covers_both_families · covers: E1 · REAL exports of `yolo11n` and `yolo26n`; the yolo26 tail contains the topk OP TYPES, the yolo11 tail contains none because the graph has none, and neither call names a family. Asserted on op types, never on node COUNTS — a count is a property of torch.onnx fusion at opset 17, not of this code
- test_sidecar_records_the_parity_report · covers: M6,A31,A34 · a real INT8 `export_model` writes `extra["int8_parity"]` carrying total recall and the parameters that produced it
red-first: every check MUST fail first.

## EVIDENCE
receipt: <runs/<n>.md>
gate: <PASS | RISK-ACCEPTED | HARD-STOP>

## LESSONS
- A diagnosis that names the loudest ops in the failing region is not a cause; the YOLO11 graph has
  none of the ops the m4 box blamed and fails identically -> add learn method
- A calibration preprocessing that differs from the inference preprocessing is a silent accuracy
  loss with a docstring claiming they match -> add learn persona:artifact-integrity-steward
- A size assertion on a quantized artifact can prefer the bug: the collapsed INT8 model is SMALLER
  than the correct one, because a destroyed graph compresses beautifully -> add learn persona:artifact-integrity-steward
- A recall floor cannot see a false-positive flood; measuring only the side that would confirm the
  fix is how a passing verdict outruns its evidence -> add learn persona:artifact-integrity-steward
- "Every logit is exactly 0.0" was prose in four files and true in none: the 4 box-geometry rows
  survive and only the 80 class rows die. An unasserted claim drifts -> add learn method
- A gate measured on the data it was calibrated on must stamp that limit on the artifact, not in a
  node nobody re-reads -> add learn persona:artifact-integrity-steward
