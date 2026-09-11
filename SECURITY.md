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
- **Credential redaction at the boundary.** `redact_url` in `src/yowo/io/_redact.py`
  strips the userinfo, every query value and the whole fragment from a source URL
  where the identifier is first accepted — not at each place it is emitted — so a
  credentialed RTSP or download URL never reaches a log line, an exception message,
  a result payload or a cache key verbatim. What it does not cover, and why, is
  below under *Credentials in a source URL*.

This is deliberately not an enumerated threat model: two related hardening items are
still open work in this project (narrowing the checkpoint's allowed classes further,
and extending digest verification to the export path), and a threat model committed
to today would already be behind them. What is above names code that exists — verify
it against `src/yowo/arch/_weights.py`, `src/yowo/models/_weights.py`, and
`src/yowo/io/_redact.py` directly rather than trusting this prose.

## Credentials in a source URL

A camera URL often carries its own secret. yowo redacts that secret where the
identifier is first accepted, so it does not reach a log line, an exception message,
a result payload, or a feature-cache key. **Three of the four places a URL can carry
one are covered. The path is not.**

```
userinfo   rtsp://camop:hunter2@camera-01:554/stream       -> rtsp://camera-01:554/stream
query      rtsp://10.0.0.5:554/live/stream?token=S3CR3T-signed -> rtsp://10.0.0.5:554/live/stream?token=#q58cf83e5
fragment   rtsp://10.0.0.5:554/stream#S3CR3T-signed        -> rtsp://10.0.0.5:554/stream#qf17831c8
path       rtsp://10.0.0.5:554/live/S3CR3T-signed/stream   -> unchanged
```

Query keys and their order survive so a log stays readable, and the `#q…` suffix is a
one-way digest of the original query and fragment. It is an **identity** control, not a
security one: it exists so that `?channel=1` and `?channel=2` remain two distinct stream
ids rather than collapsing onto one.

### Why the path cannot be redacted

The path **is** the camera's identity. `rtsp://10.0.0.5:554/live/front-door` and
`rtsp://10.0.0.5:554/live/car-park` are two cameras, and the only thing telling them
apart is the path. Empty it and they become one stream id — and `DetectionRouter.register`
overwrites a duplicate key without complaint, so the front door's detections would be
delivered to the car park's callback. Redacting the path would trade a credential that
appears in logs for silent, wrong inference output on every multi-camera deployment.

So yowo does not redact the path, deliberately. That is not a trade it makes on your
behalf, which means this one is yours to handle.

### What to do instead

**Move the secret into the query.** Most cameras and CDN-fronted streams accept it there,
and yowo redacts every query value whatever the key is called:

```
rtsp://10.0.0.5:554/live/S3CR3T-signed/stream        # leaks — the secret is in the path
rtsp://10.0.0.5:554/live/stream?token=S3CR3T-signed  # redacted
```

**If you cannot** — some cameras hard-code the token into the path and offer no query
form — then treat the stream id itself as secret-bearing, because it is. Specifically:

- every **log** line naming that stream carries the token, at whatever level;
- the `RuntimeError` raised when all streams fail carries it;
- `Frame.source_id` carries it into any result payload you serialise;
- the **cache** keys derived from it hold it in memory for the process lifetime, which
  survives log scrubbing entirely.

Scrubbing logs is not sufficient on its own. Restrict who can read them, and treat a
process dump or a serialised result the same way you would treat the password itself.

### Scope

This section is about **URL** sources. A local file path, a directory, a webcam index and
a plain name carry nothing yowo could redact and are returned byte-identical — if you see
one rewritten, that is a bug, not a redaction.
