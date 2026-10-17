# Target-verified prompt lookup, 2026-10-16

Implemented prompt/history lookup drafts with a batched target forward pass and
greedy acceptance. No smaller draft model is loaded. Rejected recurrent state is
restored on the GPU and only accepted inputs are replayed. Whole-model contiguous
KV execution is required. This does not enable distributed speculation or the
separate draft-model decoder.

## Observations

[AMD 27B Q4K](amd-27b.json) and [Mac 0.8B plus NER LoRA](mac-08b-lora.json)
retain all generated token IDs and the explicit policy. Three prompts per host
use ordinary → lookup → ordinary execution. All three token arrays match in each
case. Cancellation after four delivered tokens and subsequent ordinary reuse
pass. The Mac receipt binds changed execution-file hashes; AMD binds the tested
source snapshot. Subsequent source changes add stricter validation/type guards,
without changing the verified decode algorithm.

Final bounded AMD policy: three draft tokens, match lengths 3–4, stop drafting
after one rejected block, resume ordinary batching. Complete generation times:

| Prompt | Ordinary before | Lookup | Ordinary after |
| --- | ---: | ---: | ---: |
| Repeated sentence | 5.766 s | 6.983 s | 5.566 s |
| Arithmetic | 4.981 s | 2.063 s | 4.786 s |
| JSON objects | 6.229 s | 7.283 s | 6.246 s |

The arithmetic run proposed **zero** tokens: its difference is the serial versus
batched decode scheduling, not a speculative acceptance speedup. Earlier
[unbounded drafting](amd-initial.json) improved the repeated-text case to 4.074 s
versus 5.468–5.800 s, but worsened JSON to 9.836 s versus 5.946–6.191 s. These are
shared-host diagnostic observations, not an isolated benchmark or a general
speed claim. The bounded policy is still slower on two tested 27B tasks.
Speculation therefore remains explicit opt-in, not a public contributor default.

## Reproduction and checks

Set `MODEL_DIR` to the exact local manifest/shards, then run
`node tests/integration/speculative-generation-physical.test.js`. For the LoRA
case also set `LORA_CONFIG` to the verified adapter descriptor and `LORA_WEIGHTS`
to its pinned SafeTensors bytes. GPU/provider and artifact identity must match
before interpreting performance. No remote addresses or model bytes are stored
in this report.

Focused tests cover proposal bounds, immutable request evidence and explicit
null disablement, wrong policy, target verification/replay failure, checkpoint
cleanup, rejected-prefix recovery, cancellation and stop delivery. Fifteen tests
pass; strict source types and source style/unused checks pass. The full source
architecture gate still fails on an unchanged pre-existing 971-line
`src/gpu/kernels/attention/executors.js` (reviewed ceiling 960); this change adds no
architecture exemption.

Component: Doppler text generation and recurrent state ownership. Intent:
preserved. Boundary effects: explicit per-request speculation policy becomes
part of generation evidence; target verification owns tentative state. This is
source implementation and scoped native evidence, not a released Reploid path,
Capsule qualification, browser GPU qualification, or smaller-model drafting.
