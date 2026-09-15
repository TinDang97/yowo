# Evaluation datasets

yowo publishes accuracy numbers. This page says what they are measured against,
where that data comes from, what its licence permits, and how to get it.

Nothing on this page is committed to the repository. The archives are ~1.07 GB
and are **not redistributable as project assets** — see [Licence](#licence).
They are fetched once, digest-verified, and cached.

---

## COCO val2017

The detection accuracy tier measures mAP against COCO val2017, scored through
`pycocotools` by `yowo.benchmark.evaluate_coco_map`.

| | |
|---|---|
| Source | [cocodataset.org](https://cocodataset.org/) (COCO Consortium) |
| Images | 5 000 validation images |
| Evaluated | a pinned subset of **500**, see [Which 500](#which-500) |
| Total download | ~1.07 GB |
| Cached under | `~/.cache/yowo/datasets/coco-val2017/` (override with `YOWO_CACHE_DIR`) |

### The archives, and their pins

Both are verified against a SHA-256 recorded in
`tests/support/datasets.py` before anything is unpacked, and on every cache hit
thereafter — not just at download time, because the cache-hit path is the one
that runs every subsequent time.

| Archive | Bytes | SHA-256 |
|---|---:|---|
| `val2017.zip` | 815 585 330 | `4f7e2ccb2866ec5041993c9cf2a952bbed69647b115d0f74da7ce8f4bef82f05` |
| `annotations_trainval2017.zip` | 252 907 541 | `113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268` |

Source URLs:

```
https://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip
https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip
```

> **Do not "tidy" these to `https://images.cocodataset.org/...`.**
> That hostname is an S3 bucket fronted by a certificate issued for
> `s3.amazonaws.com`, so the virtual-host HTTPS URL **fails certificate
> verification** — which is why most recipes on the internet quietly fall back
> to plain `http://`. The path-style URLs above reach byte-identical objects
> (verified by matching S3 ETags) over TLS that actually validates.

Only `annotations/instances_val2017.json` is taken from the annotations archive.
COCO publishes no val-only annotations object — `annotations_val2017.zip` and a
bare `instances_val2017.json` both return 404 — so 241 MB is fetched to use
20 MB of it.

### How the pins were obtained

They were computed from a real download on 2026-09-15, then corroborated three
independent ways before being written down:

1. **Size** matches the `Content-Length` the bucket reports for each object.
2. **Every member's CRC** verifies (`unzip -t` over both archives, clean).
3. **Both reproduce S3's own ETag.** The annotations archive is a single-part
   upload, so its ETag is a plain MD5 and matches directly
   (`f4bbac642086de4f52a3fdda2de5fa2c`). The images archive is a 98-part
   multipart upload; recomputing the multipart ETag over 8 MiB parts reproduces
   `d366be60d3dc737327160d62453e3973-98` exactly.

COCO publishes no checksums of its own, so the ETag agreement is the strongest
independent confirmation available.

### Re-pinning

A digest mismatch has two very different causes and the error message separates
them:

- **Upstream republished the object.** That is a re-pin decision for a human:
  update the constant in `tests/support/datasets.py` in a commit that says so,
  and add a line to this page. It is not a security incident.
- **The local file was corrupted or substituted.** That is.

Pins are never refreshed automatically. An auto-refresh on mismatch would make
every integrity check tautological.

---

## Which 500

`evaluate_coco_map(..., subset=N)` selects `sorted(getImgIds())[:N]` over
**ground truth** — never over predictions, because scoping a metric to the
model's own output deletes every missed image from the denominator and turns
degraded recall into a higher score.

The result of that selection at `N=500` is committed, so you can check which
images a published number covers without running anything:

**`tests/fixtures/coco_val2017_subset500.tsv`** — 500 rows of
`image_id<TAB>file_name`, ascending by id, pinned at SHA-256
`872ac411cf65f03234a725df9711e9ced36d1c8156161ab0b7ae63bd23336aad`.

The boundary is inclusive-exclusive at 500: first id `139`, 500th id `56545`,
and `57027` is the first id *outside* the subset.

Regenerate it only deliberately — changing it changes what every published mAP
means, and the integration check `test_pinned_subset_matches_the_real_annotations`
binds it to the real annotations file.

---

## Getting the data

### Let the suite fetch it

Run the accuracy tier with network access. The fetch is bounded (30 s connect,
60 s between chunks, a 1800 s per-attempt wall-clock deadline, 3 attempts with
2 s and 4 s backoff), atomic, and digest-verified before anything is unpacked.

On a slow link, raise the deadline rather than giving up:

```bash
YOWO_DATASET_FETCH_DEADLINE_S=5400 uv run pytest tests/integration/test_coco_dataset.py
```

### Or fetch by hand

```bash
mkdir -p ~/.cache/yowo/datasets/coco-val2017 && cd $_

curl -fL --retry 5 -o val2017.zip \
  https://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip
curl -fL --retry 5 -o annotations_trainval2017.zip \
  https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip

# Check them BEFORE unpacking.
shasum -a 256 -c <<'EOF'
4f7e2ccb2866ec5041993c9cf2a952bbed69647b115d0f74da7ce8f4bef82f05  val2017.zip
113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268  annotations_trainval2017.zip
EOF

unzip -q val2017.zip -d root
unzip -q annotations_trainval2017.zip 'annotations/instances_val2017.json' -d root
```

The suite rebuilds its completion marker on the next run, so the hand-built tree
is picked up without refetching.

> `unzip` honours symlinks and absolute paths inside an archive. Verify the
> digests first, as above — that is the step that makes the unpack safe, which is
> also why the code path verifies before it extracts and refuses any member that
> would write outside its destination.

### Layout

```
~/.cache/yowo/datasets/coco-val2017/root/
    val2017/
        000000000139.jpg
        ...                       (5 000 images)
    annotations/
        instances_val2017.json
    .yowo-coco-val2017-complete.json
```

The marker file is written **last**, and records the archive digests, the subset
pin and size, the annotations digest and the image count. A tree without it — or
with one that names different pins — is treated as absent and rebuilt, so an
interrupted extraction can never be mistaken for a finished one. The archives
themselves are deleted once unpacked; their digests survive in the marker as
provenance.

---

## What you will see without it

Nothing silently passes.

- **Locally**, the accuracy tier **skips**, with a message naming both archives,
  their URLs, their pinned digests and a copy-pasteable `curl` for each. You do
  not need 1.07 GB of disk to contribute.
- **Under CI** (`CI=true`), the same condition is a **failure**. A CI run that
  could not obtain the dataset evaluated nothing, and reporting that as green is
  indistinguishable from a passing test — which is precisely how an untested tier
  goes unnoticed for months.

---

## Licence

**Read this before redistributing anything derived from COCO.**

The COCO dataset splits into two differently-licensed halves, and the difference
matters:

- **Annotations** — licensed by the COCO Consortium under
  [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/).
  Sharing, adaptation and commercial use are permitted **with attribution**.
- **Images** — **not owned by the COCO Consortium.** They are Flickr photographs,
  and their use must abide by the
  [Flickr Terms of Use](https://www.flickr.com/creativecommons/). Individual
  photos carry their own, varying licences, and the Flickr terms restrict
  commercial use of Flickr materials.

COCO's own Terms of Use place responsibility on the user: *the users of the
images accept full responsibility for the use of the dataset, including but not
limited to the use of any copies of copyrighted images that they may create from
the dataset.*

### What that means for this repository

- **The archives are never committed and never redistributed as project assets.**
  yowo is Apache-2.0; bundling Flickr-licensed imagery into an Apache-2.0
  distribution would misstate the licence of bytes we do not own. They are
  fetched to a user-local cache at test time instead.
- **`tests/fixtures/coco_val2017_subset500.tsv` is derived from the CC BY 4.0
  annotations** (image ids and file names). Attribution, as CC BY 4.0 requires:
  *Microsoft COCO: Common Objects in Context*, Lin et al., 2014 —
  <https://cocodataset.org/>, annotations licensed CC BY 4.0. It carries no
  image data.
- **The distribution ships none of it.** The packaging `only-include` list
  excludes `tests/`, so no COCO-derived byte reaches the published sdist or
  wheel, and `NOTICE` correctly does not claim otherwise.

If you intend to use yowo's COCO-measured numbers, or COCO itself, in a
commercial product, read both licences. The annotations and the images do not
grant the same rights.

---

## See also

- [`CONTRIBUTING.md`](../CONTRIBUTING.md) — running the test tiers
- `tests/support/datasets.py` — the pins and the fetch path
- `tests/integration/test_coco_dataset.py` — what binds the manifest to real COCO
