# Qwen sampling correction and frozen candidate — 2026-10-17

Candidate archive SHA-256:
`75f3a81f032e3a0f89102fb9a50e0b202657f69eab71569585d7ffe4ce232bb2`.
Runtime source commit: `bcb6c045`. This is a qualification candidate, not a release.

The [previous archive](../qwen-package-candidate-20261017/README.md) remains failed
on Metal sampling. Atomic scratch access repairs the observed heap corruption;
the independent reference and acceptance thresholds are unchanged. M3 and AMD
pass 540 immediate/recorded F32/F16 sampling cases and the browser seeded batch
audit, including split/retry equivalence, cancellation, consumer callback failure,
and settled transient resources. These are operator/lifecycle checks, not blanket
model correctness.

Both Qwen 0.8B and 2B pass prepared reset and cancellation checks on both hosts.
Browser compatibility-export journeys on both GPUs pass public logits, cancellation,
reuse, device destruction after completed work and immediately after an observed
submission, rejected stale-session execution, explicit unload, and device recreation
with identical text. Deliberate destruction does not prove recovery from every
spontaneous driver failure.

The exact archive passes installed imports, exports/types, dependency/closure,
synthetic consumer, and package checks. Its first installed attempt failed on a
stale generated runtime-closure receipt after registry synchronization; refreshing
that receipt and verifying the **same archive** passes. Initial receipt retained.
Size: 1,959 files; 3,483,987 packed bytes; 14,168,991 unpacked bytes. All original
limits remain unchanged.

Matched comparisons are next, using [the retained driver](run-matched.py): raw
64-token and 1,024-token prompts, 128 greedy output tokens, both engine orders,
three measured runs, two warmups for warm runs and none for cold runs. Cold means
engine profile/cache cold with provisioned local weights; OS/driver caches are
not purged. The comparison now permits explicit cold engine order as well as warm
order. Existing fairness and token gates remain unchanged. Memory sources remain
separate where the two engines do not expose comparable GPU residency.

CI is still open. Run `39069960131` at `4a38847d` failed
`tests/training/native-lora-round-gpu.test.js` because the hosted runner had no
usable WebGPU adapter. Run `39070869994` was still running at inspection. This
cycle does not expand into LoRA work or label those failures as passed.

Component: kernel execution, tests, repository comparison tooling. Intent preserved;
no public authority or application boundary changed. Acceptance evidence is scoped
above; full CI, comparisons, delivery attribution, and representation reachability
remain incomplete.
