# doppler-gpu 0.7.1 handoff (for D and B)

Owner: E's lane, Doppler runtime correctness. Written 2026-11-11.

## Package identity

| Field | Value |
| --- | --- |
| Version | `0.7.1` |
| Source commit | `ca051fdaa9e47267fb57c8ff2c5a7c0b45e5ea47` on `main` |
| Tarball shasum | `7aa6fad5e23c1e21b55dd3bc6d89a24e5fd426e5` |
| Integrity | `sha512-nsg0sPBzUaG1tSI3+GoPtV6PIf4iV9gFtHSI77H1WPMPyNWbJ3OyQde5URb8NlRP0asR7B39y/y1nAk2fqi28Q==` |
| Size | 3,694,298 packed, 15,759,031 unpacked, 1,933 files |
| Delta vs published 0.7.0 | [package-audit.json](package-audit.json) |

Before relying on the npm copy, confirm it was published. Then check
`npm view doppler-gpu@0.7.1 dist.integrity` against the integrity above.
`0.7.0` bytes on npm are unchanged.

## What changed

1. **Gate sigmoid (`c1e4e8c0`).** `sigmoid_gated_add_f32.wgsl` and the
   `moe_shared_slot_*` kernels now match `silu.wgsl`, with no input clamp.
   Affects the Qwen shared-expert path and the Qwen 3.6 35B-A3B q4_0
   profiles. Evidence: `tests/gpu/sigmoid-gated-add-gpu.test.js` (maxAbs 0),
   `tests/kernels/silu-unclamped-physical.test.js`, and the gate-isolation
   check in `tests/gpu/native-cooperative-shared-slot-gpu.test.js` (all four
   shared-slot variants, max relative error 8.6e-8, allowance 1.9e-6), all on
   Apple M3.
2. **Budgeted memory (`6151f28b`).** Under `configureDeviceMemoryBudget`, a new
   generation's first prefill retires idle pooled buffers after confirmed GPU
   completion. Evidence: `tests/memory/retire-idle-under-device-budget.test.js`
   and the measurements below.
3. **Selective activity capture (`e56cf5d6`).** Opt-in
   `model.inspect.generate(prompt, { capture })`.

## For D: the fleet's long-prompt memory failure on bee

Reproduced on the M3 with bee's exact model bytes (manifest
`sha256:a0dc69e9…`, identical shards), KV 2048, and bee's budget of
6,442,450,944 bytes. Six back-to-back prompts of
1,666 / 1,200 / 1,666 / 900 / 1,666 / 2,000 tokens:

| Runtime | Result | Peak | Steady between requests |
| --- | --- | ---: | ---: |
| 0.7.0, manifest default (async, 2 chunks in flight) | request 3 rejected | – | – |
| 0.7.0 + `prefillChunkSubmitMode: "sync"` | all pass | 5.59 GB | 2.50 → 3.16 GB, creeping |
| 0.7.1, manifest default | 2,000-token request rejected | – | – |
| **0.7.1 + `prefillChunkSubmitMode: "sync"`** | **all pass** | **4.40 GB** | **≤ 2.6 GB** |

`prefillMaxInFlightChunks: 1` measured the same as sync. Prefill time does
not change: 2,000 tokens take about 20.5 s with sync and 20.9 s with async.

Fleet actions (D's decision and deployment):

- **Runtime:** move bee to `doppler-gpu@0.7.1`.
- **Config:** add `"prefillChunkSubmitMode": "sync"` under
  `loadOptions.runtimeConfig.inference.session` for `qwen-3-5-2b-q4k-ehaf16`.
- **Admission (separate issue):** Valera's full-envelope prompt (about
  1,636–1,645 tokens before the chat template) plus `max_output_tokens: 512`
  exceeds `maxSeqLen` 2048. Cap output to the remaining context, or reject
  before dispatch. Per-candidate prompts (about 1,070 tokens, outputs 38–76)
  fit.

Not measured: bee itself (AMD integrated GPU on Windows), cancellation
behavior, and load time on bee. Load time on the M3 was 14.6 s.

## For B: Observatory preview

0.7.1 is a candidate runtime for the preview; nothing is promoted
automatically. Qualify the preview against 0.7.1 separately. Selective
capture ships in 0.7.1.

## Limits

- **Flash-Next (Qwen 3.8) is not qualified.** The 2,187-token `inventory`
  prompt fails the unchanged full-logit gate: 30,664 violations against the
  F32 CPU reference and 30,729 against an F64-widened reference. The two CPU
  references agree within the gate. The remaining owner is the GPU's
  accumulated rounding in Layers 0–24, which flips a near-tied Layer-24
  expert. No fix is claimed. See
  [../flash-next-reference-controls-20261110/](../flash-next-reference-controls-20261110/README.md).
- **Single host.** The physical checks ran on Apple M3 (Metal) only. AMD,
  Windows and browser surfaces were not re-qualified for 0.7.1.
- **Pre-existing failures.** Six tests fail identically on `dc81876c` (one
  assertion, five time out at 400 s), and 19 digest drifts sit in tracked
  `models/local` manifests. Both predate this work.

## Replay

```sh
git checkout ca051fda
node tests/gpu/sigmoid-gated-add-gpu.test.js
node tests/kernels/silu-unclamped-physical.test.js
node tests/gpu/native-cooperative-shared-slot-gpu.test.js
node tests/memory/retire-idle-under-device-budget.test.js
npm run public:boundaries:check && npm run package:smoke
```

The memory measurement uses a whole-model `createPipeline` with
`configureDeviceMemoryBudget({ maxBytes: 6442450944 })`, the session config
above, and `generateTokens` with `maxTokens: 1` over the listed prompt
lengths, reading `inspectDeviceMemory()` after each request.
