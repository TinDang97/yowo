# Security Policy

yowo is a personal, open-source project. Stated plainly, before any process detail:
suspected vulnerabilities are reported privately through this repository's GitHub
Security Advisories, only the current minor release line (2.5.x) receives fixes, and
the checkpoint trust boundary described below is exactly what m1 shipped and can be
pointed at in code — it is not a complete threat model.

## Supported versions

| Version | Supported |
|---------|-----------|
| 2.5.x   | Yes       |
| < 2.5   | No        |

There is no long-term support line for this project. A report against an older
version is still received and triaged; the fix, if any, lands on the current line,
and users on an older line are expected to upgrade to receive it.

## Reporting a vulnerability

Report suspected vulnerabilities privately through
[GitHub Security Advisories](https://github.com/TinDang97/yowo/security/advisories/new)
for this repository. Please do not open a public issue for a suspected
vulnerability. There is no email intake for security reports: a personal-project
inbox is a channel that silently rots, and a rotted channel is worse than none
because it looks live. GitHub Security Advisories gives both of us a tracked,
private thread instead.

This is a part-time-maintained personal project, not a vendor with an SLA. Expect a
response, not a guaranteed turnaround.

## The checkpoint trust boundary

Model checkpoints (`.pt` files) are untrusted input: a naive load of one can execute
arbitrary code during deserialization. yowo narrows that surface in three ways that
are shipped today, not planned:

- **A restricted, allowlisted class loader.** `src/yowo/arch/_weights.py` overrides
  the deserializer's class-resolution hook and only admits an explicit allowlist of
  `(module, name)` pairs, plus tensor storage classes and stubbed
  `ultralytics.*` / `models.*` prefixes needed to read an Ultralytics checkpoint's
  class references without executing them. Anything outside that allowlist raises
  `ModelLoadError` instead of loading.
- **Digest pinning, re-verified immediately before conversion.**
  `load_verified_state_dict` in the same file performs the one unsafe read a
  checkpoint ever gets: the result is converted to a plain tensor mapping and cached
  in a sidecar, and every later load reads that sidecar through a safe path that
  executes nothing. Where the model registry pins a digest, the raw file is compared
  against that pin again immediately before conversion — not only earlier, when it
  was first resolved.
- **Credential redaction on emit.** `redact_url` in `src/yowo/io/_redact.py` strips
  userinfo from any URL yowo prints or raises in an error message, so a credentialed
  RTSP or download URL never reaches a log line or a CLI table verbatim.

This is deliberately not an enumerated threat model: two related hardening items are
still open work in this project (narrowing the checkpoint's allowed classes further,
and extending digest verification to the export path), and a threat model committed
to today would already be behind them. What is above names code that exists — verify
it against `src/yowo/arch/_weights.py`, `src/yowo/models/_weights.py`, and
`src/yowo/io/_redact.py` directly rather than trusting this prose.
