# Flash-Next: separate F32 KV candidate and routing trace

Full-model qualification remains open. No acceptance tolerance, stored weight,
artifact manifest, default recipe or catalogue status changes. The original
F16 KV reference and its failing verdict remain intact.

## Usable local short chat

The separate loopback operator at `http://127.0.0.1:8989/v1` runs the named
F32 KV profile through the public compatibility API. It is experimental,
allows one active request and caps output at 128 tokens. It is not a Reploid
participant or quickstart catalogue promotion. Its health response states the
open long-context gate explicitly.

[Actual HTTP checks](local-http-probe.json) pass identity, chat (`Paris`),
streaming (`42`), follow-up context (`France`), concurrent-request refusal,
active-client cancellation, reuse after cancellation, and output-cap errors.
[Three separate chat smokes](f32kv-chat-20261107.json) return `Paris`, `42`
and `Bonjour`, using the Qwen chat template with thinking disabled. These tiny
checks establish the named short interactions, not general answer quality.

The operator is an application composition of the existing `createServeHandler`
and public model handle. Run `node reports/flash-next-qualification-20261107/serve-local.js
<operator-config.json>` with the retained explicit configuration. It owns model
shutdown and request cancellation; no existing participant is restarted.

## Runtime change

An explicit F32 KV indexed-attention binding complements the existing F16
binding. Both consume the same selected token rows, F32 queries and F32 output.
The new KV-only execution transform preserves embedding storage, Q4K experts
and projection arithmetic. Its classified policy admits only this exact model
with an explicitly requested F32 cache on an F16-capable device. Negative
policy tests prevent broadening that exception. No implicit precision fallback
is introduced.

`profiles/flash-next-routed-f32kv-qualification` inherits bounded routed loading
and changes only KV storage. The diagnostic results below additionally use the
previously identified HC/router subgroup graph, retained in each run JSON.
They must not be attributed to the named profile alone.

## Independent reference and unchanged gate

The unchanged pinned producer reads the same immutable artifact with independent
GGUF Q4K decoding and executes upstream F32 operators without its F16 storage
hook. It covers both the 15-token short input and the 2,304-token long input.
Its short logits reproduce the earlier independent F32 reference bit for bit
(1,489,920 values). This reference is a separate precision control, not a
replacement for the F16 reference.

- Artifact manifest: `4902a0b730a9310903b00bc0f8f587528fa1af9f37a722d2f9287bafceceec09`.
- Inputs: `06a819d529593b58abd6cb27566bb835a9b91aac13a77c0c7479b7e4e7b0c378`.
- New F32 reference: `ce4cb138b1919474d85507e06faa9ca167d87fd143bb85a7411f5443cea9a897`.
- Producer: `f6bc0f15a6545ab1d205cfc3ec412442415220e541190b63a9c712fa238ba5b0`.
- Original F16 reference: `5c22cd2b9e47a92c2988b0e461cad88e444d51919f5f0c7ab9c8cea20e377fd3`.
- Policy remains `doppler.boundary-tolerance/quantized-v1`, absolute 0.05
  plus relative 0.05; policy SHA-256
  `f1c4629d593ec2517e3628ac68131dc7568c6aba5e0f03db627d26228c720413`.

Every comparison checks all 248,320 last-position logits, tensor hashes,
shape, finiteness, actual cache dtype and input/weight identities. Top-token,
top-20 and KL diagnostics cannot substitute for that gate. The comparator
explicitly reports prefill-only scope and never declares full-model qualification.
Six adversarial fixtures pass under normal Python and `python -O`, including
a matching top token with an incorrect tail, truncation, mismatched identities,
counterfactual routing and configured/observed dtype disagreement.

| Separate F32 KV candidate | Input | Maximum absolute error | Outside unchanged policy |
| --- | --- | ---: | ---: |
| Subgroup HC and router | Short | 0.00002718 | 0 |
| Subgroup HC and router | Long | 0.109021 | 13 |
| Named F32 profile, original manifest HC/router program | Short | 0.00004911 | 0 |
| Named F32 profile, original manifest HC/router program | Long | 0.776560 | 72,330 |
| Same HC, ordinary router reduction | Long | 0.749230 | 72,104 |
| Same HC, compensated router reduction | Long | 0.109003 | 13 |
| Same graph, four-accumulator Q/K dot experiment | Long | 0.713091 | 50,893 |
| Same graph, materialized normalized softmax experiment | Long | 0.713103 | 50,895 |

The reduction candidates do not resolve the failure and are not promoted.
All first tokens and all top-20 candidates match in these F32 controls; this is
not a whole-answer quality assessment. GPU observations, run configurations,
comparison receipts and rejected-candidate generators are retained here.

## First changed expert boundary

The independent CPU observation records actual expert IDs and periodic CPU
checkpoints, without reading GPU values. All 48 layers replay the independent
reference's retained samples and complete router metadata exactly. The GPU
observer reads actual top-k buffers and never replaces them.

For the best subgroup candidate, expert sets agree at every one of the 2,304
positions through layer 24. Layer 25 first differs at positions 1,717 and
2,271. At 1,717, experts 493 and 330 straddle a source margin of
`1.9073486328125e-6`. Exact-dot controls show the incoming input drift already
reverses that ordering; changing only router accumulation cannot recover it.
Across all 48 layers, 745 of 110,592 token/layer expert sets differ. These
counts are diagnostics, not an alternate acceptance rule.

The layer-25 source trace resumes from the independent layer-23 checkpoint,
executes layers 24 and 25, and reproduces the retained router input bit for bit.
Identical-input controls separate local arithmetic from propagated inputs:

| Layer-25 operation, three retained positions | Total max difference | Local max on identical GPU inputs | Propagated max |
| --- | ---: | ---: | ---: |
| Attention HC norm | 4.86e-5 | 1.91e-6 | 4.91e-5 |
| Attention HC down projection | 1.45e-4 | 1.72e-5 | 1.44e-4 |
| MLP HC norm | 4.01e-5 | 9.54e-7 | 3.98e-5 |
| MLP HC down projection | 9.92e-5 | 1.43e-5 | 9.54e-5 |

These controls do not demonstrate a new layer-25 defect or establish that the
remaining failure is unavoidable. They narrow the next investigation to
propagated activation drift before the first consequential expert change.

## Physical and development checks

- Indexed attention passes 12 operator cases on physical Apple M3/Metal and
  AMD Radeon 8060S: F16/F32 storage, immediate/recorded dispatch, dense prefix,
  dense limit and sparse selection. Existing F64-derived bounds are unchanged.
- The F32 subgroup candidate passes six physical short-input lifecycle cases:
  queued abort, active-decode rollback, reset/reuse, paused-consumer abort,
  paused-consumer unload and reload/reuse. No GPU or cleanup errors.
  The named profile also independently passes these six checks.
- Nine focused JS tests pass (KV dispatch/view, sequence ownership, policy and
  profile isolation). Typing reports 188 strict roots and no debt growth;
  unused-local, architecture, kernel/digest, browser import, dependency,
  capability, configuration and package-closure checks pass.
- Some runs overlap CPU reference work. Their elapsed times are not controlled
  performance evidence. The earlier routed-loading comparison remains the
  scoped performance receipt; none of these numerical experiments promotes a
  new performance claim.

The existing 27B provider remains active with its original PID 3399279. The TTM
setting remains `16099953`; no existing participant service or GPU limit changes
were made. The separate local operator and its explicit configuration are bound
by [their identity receipt](local-operator-identity.json).

## Retained data and reproduction

Large, hash-bound tensors remain under
`/run/media/x/models/qwen38-flash-next-de4b8e4d/qualify` and
`.../reference/f32kv-routing-trace-20261107` on the qualification volume.
The matching reference is
`.../reference/f32kv-20261107/out/learned-forward-reference-float32-artifact.json`;
[reference arguments](reference-run.json) bind its exact producer and inputs.
The original source checkpoint is
`de4b8e4d43b917e7706784d8bb445c9af86a3540`.

`f32kv-measured-source-20261107.tar.gz` retains the measured baseline source;
SHA-256 `1a1a47b31f5708c0d5d8ba70c4bab12c9ca80f69649075ad31508e9e915042f6`.
Isolated rejected shaders are reproducible with their retained generators and
source hashes, without changing the baseline snapshot.

An initial run used the older short-only F32 reference and could not compare
long input; its incomplete observation/run remain retained. The builder now
requires both reference sequences before starting inference. Snapshot setup
failures remain on the volume; they provide no model evidence. The shader
experiment's first guard used a normalized kernel-ref digest as a raw-file
digest; it failed before dispatch and was corrected to the verified raw SHA.

Component: `doppler.runtime-source.gpu.kernels`, `doppler.runtime-source.config`,
`doppler.runtime-source.inference`, `doppler.runtime-source.rules`,
`doppler.tests`, `doppler.docs`.
Intent: preserved. Boundary effects: explicit F32 indexed-cache admission and
its KV-only compiler transform; no implicit dtype fallback or model promotion.
The loopback operator is a separate application-owned serving process.
