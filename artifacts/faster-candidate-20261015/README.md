# Faster candidate: runtime package and signed Capsule (2026-10-15)

Component: signed model delivery (local qualification scope) and runtime packaging. Intent preserved.
Boundary effects: new test-signed Capsules and release events in the test authority's history (sequences
7, 8 and 9). No public pin, catalog or production signature changed. The accepted public package (Doppler
0.6.27) is untouched.

The candidate is the combined paired-kernel configuration of the kernel work:

- tiled attention;
- sliced RMSNorm;
- the sliced gated-delta state and parallel Q/K programs;
- compensated matmul.

All five are selected through the execution graph, with `prefillTokenChunkSize` 16. The Reploid evidence
is [`reploid/artifacts/faster-candidate-20261015`](https://github.com/clocksmith/reploid/tree/main/artifacts/faster-candidate-20261015).

## Packages

Every archive below was made by `git archive <commit> | tar -x`, then `npm pack --ignore-scripts`. The
same method reproduces `2df37d88...` from `a5f3ca4c`.

| Archive | Commit | Files | Used by |
|---|---|---|---|
| `21d054eeb70212a7434c9eccfe467fe0accbb8685e9358f2b72341500978cd5f` ([unrolled-1c9ec3db/](unrolled-1c9ec3db/)) | `1c9ec3db` | 1,998 | generations 1 and 2 (unrolled state program); retained |
| **`3b132ee7ff984c80ef48487d3010028ade46b64d39323304d674feebfde3bf78`** ([doppler-gpu-0.6.28.tgz](doppler-gpu-0.6.28.tgz), [candidate.json](candidate.json)) | `77c5d9b2` | 1,999 | **generation 3, the frozen candidate** |

Doppler `77c5d9b2` adds `gated_delta_state_paired_sliced_rolled.wgsl` as declared variant
`paired_state_sliced_rolled`. It is the saved non-unrolled prototype; nothing else in the runtime changes.
Reploid `variant-choice/` chose it from full-model measurements: equal prefill and decode, lower cold
preparation, identical correctness.

## Capsules

[build.js](build.js) follows `../chunked-prefill-capsule-20261014/build.js`. It reads the reports from
Reploid `placement-c/<attempt>`, the placement being mac-host browser stage 0, amd-host native stage 1,
and a second amd-host native stage 2 with the output head. It signs with the test authority and chains
the release history. Before rigging, it now also checks that every pipeline the captured stages prepared
is inside the signed source closure.

| Generation | Folder | Manifest | Semantic root | Release | Outcome |
|---|---|---|---|---|---|
| 1 | [capsule-1/](capsule-1/) | `64d74719...` | `e84a2024...` | seq 7 `2486b1e7...` (after renewal 6 `b19052d5...`) | **Refused at installed opening:** the closure lacked `rmsnorm_paired.wgsl`, which the runtime runs for the Q/K head norms outside the graph's steps (the `norm` step now names the sliced program). Stage 1 failed pipeline preparation before readiness. Hosts had already admitted seq 7 |
| 2 | [capsule-2/](capsule-2/) | `c6220c6a...` (adds mechanism kernel `paired_qk_head_norm` = `rmsnorm_paired.wgsl`) | `5a750ed8...` | seq 8 `0b40731c...` | Opened and served on the installed route (final-10 pass 2), unrolled state program; superseded by the variant choice |
| **3** | top level | **`548e6559...`** (generation 2 with `paired_state_sliced_rolled`) | `b1527013...` (envelope `759f5ab1...`, 67 artifacts) | seq 9 `dd837996...` (expires 2026-10-16T04:35:38Z) | **Frozen candidate**: bound to placement C attempt `c8`, delivered as Reploid final-10 pass 3 |

Release verification is in each folder's `build-binding.json`. In generation 3's binding, the label
`candidate 21d054ee` names the wrong runtime: verification ran from the generation-3 fixture runtime
(`exact-runtime-verification.json` `runtime`, archive `3b132ee7`). The label is now derived from the
archive hash. Every chain was verified with `verifyCapsule`
from the candidate runtime from each held checkpoint (5, 6 and 7, 8 where applicable, and 0), and from the
`2df37d88` runtime for the held checkpoints. Test authority `reploid-paired-qualification-20261012` only.
Production signing remains the public-promotion gate.

## Not established here

Physical numerical results, installed conversations and performance are Reploid evidence. This folder
records the bytes and signatures they bind.
