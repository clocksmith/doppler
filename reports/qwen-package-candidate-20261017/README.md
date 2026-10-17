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
