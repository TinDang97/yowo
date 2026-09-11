---
type: Task
title: A check that scans prose stops reporting findings that are not there
status: done
depth: quick
milestone: m1-trust-the-ship
scope:
  - tests/unit/test_licensing_provenance.py
  - SECURITY.md
gives:
  - S1 the two SECURITY.md prose scans in `test_licensing_provenance` — an LTS promise, an email intake channel
generated: { by: add/3.5.0, at: 2026-09-11 }
verified:
  - { by: "Tin Dang", at: 2026-09-11, act: freeze, authority: human, direction: "sha256:0ec9320aa4d23ccc", binding: "sha256:e3b0c44298fc1c14" }
  - { by: "cli", at: 2026-09-11, act: brief, authority: process, brief: "sha256:939befea5f71c98b" }
  - { by: "Tin Dang", at: 2026-09-11, act: refreeze, authority: human, direction: "sha256:fc4f8424985de68b", binding: "sha256:e3b0c44298fc1c14" }
  - { by: "cli", at: 2026-09-11, act: brief, authority: process, brief: "sha256:c1a564720104ec63" }
  - { by: "process:run", at: 2026-09-11, act: run, authority: process, outcome: PASS, receipt: /tasks/prose-checks-need-word-boundaries.d/runs/1.md }
  - { by: "Tin Dang", at: 2026-09-11, act: gate, authority: process, outcome: PASS, receipt: /tasks/prose-checks-need-word-boundaries.d/runs/1.md, brief: "sha256:eddee5eede7b9d61", reason: "8 bound check items green on receipt 1, zero skipped. Suite 2433 passed / 11 skipped, ruff clean, ruff-format clean, pyright 0 errors. THE TWO SCANS WERE WRONG, NOT STRICT. test_security_md_names_supported_versions asserted 'lts' not in text.lower(), which matches inside 'results', 'faults' and 'consults'. test_security_md_names_a_private_reporting_channel scanned for an email pattern over the raw document, and matched 'hunter2@10.0.0.5' in an RTSP userinfo example, because a credential on a dotted host is indistinguishable from an address to that regex. Both now read the document's PROSE -- fenced blocks and inline code removed -- and the LTS term is matched as a word. Neither assertion was loosened: what changed is what they read and how they match. REFUTED IN BOTH DIRECTIONS, which is the part that matters for a check being repaired rather than switched off. Four mutations on the helpers: dropping the word boundary reddens 1 (the original bug), scanning the raw document instead of the prose reddens 2 (the other original bug), a reducer returning '' reddens 3, and an LTS scan that never fires reddens 1. Three of the new checks exist only to catch the false-NEGATIVE direction -- a real LTS promise, a real address in prose, and a reducer that strips everything -- because a scan that can no longer find anything passes just as quietly as one that finds too much. THE END-TO-END PROOF, and the reason scope was widened. While box-6-scope-amendment was building, these two checks turned red on correct content and I got green by editing MY content: an undotted host in the userinfo example, and 'inference output' where 'inference results' was meant. That is precisely the failure Q5 names -- a check that fires on correct content teaches the author to edit the content. Leaving those two phrasings in place while claiming the checks were fixed would have been that same failure persisting in a quieter form, so the scope was widened by REFREEZE, not by silent edit, and both phrasings are restored. Measured: with the natural wording back and both scans reverted to their original form, 5 checks fail, including test_security_md_names_supported_versions itself. With the fix in place, 17 pass. The bug is reproduced against the exact content that provoked it, and closed against it. SCOPE. The two owning checks keep their names, their docstrings' covers: keys and their assertion text, so licensing-provenance's citations still bind. No other check in that file was touched." }
---
## CARD
goal: The two SECURITY.md prose scans report a finding when SECURITY.md actually contains one, and not when it merely contains a word that embeds the term they look for.
why: Both fired on correct content while `box-6-scope-amendment` was building, and both are wrong rather than strict. `test_security_md_names_supported_versions` asserts `"lts" not in text.lower()`, which matches inside `results`, `faults`, `consults` — any SECURITY.md using one of those words fails a check about long-term-support promises. `test_security_md_names_a_private_reporting_channel` scans for `[\w.+-]+@[\w-]+\.[\w.-]+`, which matched `hunter2@10.0.0.5` in an RTSP userinfo example, because a credential on a dotted host is indistinguishable from an email address to that regex. I dodged both by editing my own examples — an undotted host, and "inference output" for "inference results" — which is exactly the failure Q5 names: a check that fires on correct content teaches the author to edit the content until the check is happy. The checks stay as strict as they were; they stop being wrong.

## PLAN
contract: the two scans become named helpers in the same file — one that reduces a Markdown document to its PROSE, dropping fenced and inline code, and one predicate per scan reading that prose. The two existing checks call them and keep their current assertions. New checks drive the helpers with adversarial fixtures: a document containing `results` must not read as an LTS promise, a document containing a credential example in a fenced block must not read as an email intake, and — the half that matters — a real LTS promise and a real email in prose must STILL be caught, so the fix cannot be a silent weakening.
strategy: extract, then prove both directions before touching the assertions. A false-negative check is worse than the false positive it replaced.
SCOPE WIDENED after the first freeze, by refreeze: SECURITY.md's two contorted phrasings — an undotted host, and "inference output" where "inference results" was meant — were written to dodge these two checks. Restoring them is the only proof the fix actually works, and leaving them is the Q5 failure persisting quietly.

## CHECKS
- tests.unit.test_licensing_provenance::test_the_lts_scan_reads_a_word_not_a_substring · covers: G1 · `results` is not a long-term-support promise; this is the exact false positive.
- tests.unit.test_licensing_provenance::test_the_lts_scan_still_catches_a_real_promise · covers: G1 · the fix must not turn the check off — an actual LTS line still fails.
- tests.unit.test_licensing_provenance::test_the_email_scan_ignores_a_credential_example_in_code · covers: G1 · `rtsp://camop:hunter2@10.0.0.5/s` in a fenced block is a credential example, not a reporting address.
- tests.unit.test_licensing_provenance::test_the_email_scan_still_catches_an_address_in_prose · covers: G1 · the half that makes the fix safe rather than convenient.
- tests.unit.test_licensing_provenance::test_the_prose_reducer_leaves_prose · covers: G1 · stripping code must not strip everything — a reducer returning "" makes every scan vacuously clean.
- tests.unit.test_licensing_provenance::test_security_md_names_supported_versions · covers: G1 · the real document still passes, unchanged in intent.
- tests.unit.test_licensing_provenance::test_security_md_names_a_private_reporting_channel · covers: G1 · the real document still passes, unchanged in intent.
- tests.unit.test_box_6_scope::test_the_box_claims_nothing_a_probe_refutes · covers: G1 · SECURITY.md's restored userinfo example is executed, so the revert cannot reintroduce a false transformation.
red-first: every check MUST fail first.

## EVIDENCE
receipt: <runs/<n>.md>
gate: <PASS | RISK-ACCEPTED | HARD-STOP>

## LESSONS
- <lesson> -> add learn <lens>
