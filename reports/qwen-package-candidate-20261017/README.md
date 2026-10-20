# Qwen package candidate — 2026-10-17

Package size is repaired without raising limits. This is packaging evidence;
physical qualification and release acceptance remain open.

The [classification](classification.json) accounts for all twenty additions:
fifteen required public runtime/assets, one repository qualification profile,
and four experimental training API files retained because the existing public
training export reaches them. No added file is a generated duplicate. Training
implementation and public exports are unchanged.

Publication excludes 65 repository-only configuration files, preserving their
source and historical evidence. General defaults, Qwen Metal/Vulkan comparison
profiles, and their ancestors remain installed. Package exclusions are owned by
`package.json`; source closure generates `src/.npmignore` and rejects excluded
imported assets or missing profile ancestors. No runtime execution settings changed.

| Measure | Candidate | Unchanged limit |
| --- | ---: | ---: |
| Files | 1,959 | 2,004 |
| Packed bytes | 3,483,809 | 3,497,999 |
| Unpacked bytes | 14,168,056 | 14,172,897 |

Focused checks pass: package boundaries/size, package closure and its negative
controls, export parity, source dependencies, and runtime closure. Exact archive
and installed-consumer evidence will be appended before any benchmark starts.
Earlier failures remain in the [prior checkpoint](../qwen-qualification-next-20261017/README.md).
No CI, GPU, numerical, device-recovery, or performance success is inferred here.

Next: freeze the archive, pass installed-consumer smoke, execute relevant pending
Qwen GPU/browser/lifecycle checks, then run matched physical comparisons on both
GPUs and in both engine orders. Delivery attribution follows qualification;
per-tensor reachability follows delivery attribution. No other model, training,
prefix-cache, or WAN work is part of this sequence.

Component: `doppler.repository-tooling`, `doppler.docs`, package publication.
Intent preserved. Publication boundary narrowed to omit repository campaigns;
public API and runtime authority are unchanged. Acceptance is scoped to the
checks above; full release gates remain open.

## Frozen archive and physical findings

Archive `37bf7a3fccda69c8aae4871f0f66bccc3aadeeb15c1b36b82935e58114ca6ee6`
passed installed-consumer smoke. Every published file was checked against the
qualification runtime on both hosts. It is retained here, but **failed physical
Metal sampling qualification** and is not promoted.

Both Qwen sizes passed prepared reset/cancellation and the repaired regression
test on both GPUs. The regression previously printed success while leaving its
Node provider alive; both original 240-second timeouts remain retained. Initial
AMD missing-config failures are distinct from the successful configured reruns.

Browser checks on both GPUs and both sizes exercised the compatibility export,
public logits, cancellation/reuse, explicit device destruction after submitted
work completed, stale-session rejection, unload, and device recreation with
identical answer text. These are scoped lifecycle checks, not spontaneous or
mid-submission device-loss qualification. The obsolete pending logits assertion
looked for method bodies in facades; behavioral and installed-type tests now
cover the actual model handle. The original failure remains retained.

The skipped seeded GPU audit uncovered wrong Metal sampled tokens: expected
`[0,3,1,1,1,0,0]`, observed `[0,3,0,0,0,0,0]`. Independent dispatches reproduce it;
AMD passes the batch audit. Phase readbacks localize corruption to final heap
sorting/sampling. Atomic scratch access preserves heap mutations and passes
540 M3 F32/F16 immediate/recorded sampling cases. Source shader identities and
conversion recipes are regenerated; existing model artifacts are not rewritten.
A new immutable archive and additional physical qualification are required.

[Machine status](status.json) preserves unresolved gates. No matched performance
campaign started after this numerical failure. Raw attempts, corrected attempts,
phase probes, and rejected shader experiments are retained separately.
