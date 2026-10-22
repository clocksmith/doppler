# Choice scoring on the rebuilt 0.6.32 candidate (2026-10-22)

Supersedes the [2026-10-21 candidate](../choice-scoring-successor-20261021/README.md). That
candidate's successor manifest, archive and Capsule pinned `matmul_f16w_f32a_tiled` at
`sha256:bed71e39…`, the 07d0f6ba layout that
[corrupts output on Metal](../f16w-tiled-shared-layout-20261021/README.md#correction-metal-corruption-and-7425fd58-2026-10-22).
This candidate pins the fixed `sha256:5dada682…` (7425fd58). The task, cases, labels, `maxSeqLen`
512, the 0.05 answer-logit tolerance and 12 required correct choices are unchanged.

## Model

`tools/refresh-converted-manifest.js` reapplied the current conversion configuration to a copy of
the 2026-10-21 successor (`.cache/qwen-reranker-q4k-successor-20261022`, shards hard-linked; weight
and tokenizer bytes unchanged). Against the previous successor manifest, only the tiled kernel
digest, `conversionConfigDigest` and the refresh timestamp changed. The new manifest is
[successor-manifest.json](successor-manifest.json), `6beadee3…`.

## Qualification

| Step | Result |
|---|---|
| CPU reference, `tests/integration/choice-scoring-reference.py` (Transformers Qwen3ForCausalLM eager F32, GGUF Q4_K dequantization) | 12/12 ([reference.json](reference.json)) |
| `npm pack` at `509eed3f` → `doppler-gpu-0.6.32.tgz` `3f87f551…`, installed into a fresh consumer with `--omit=optional` | 1 package, no runtime dependencies |
| Physical Node, `webgpu@0.4.0`, RADV, `tests/integration/choice-scoring-physical.js` | 12/12, max answer-logit error 0.0127 ([report](physical-node.json), [qualification](physical-node.json.qualification.json)) |
| Physical Chrome 146.0.7680.177, Reploid `tests/fixtures/doppler-choice-browser-check.js` at `9518ce6b` | 12/12, max error 0.0127 ([report](physical-browser.json), [qualification](physical-browser.json.qualification.json)) |
| Rig, `tests/integration/rig-choice-scoring-capsule.js` | Capsule `…-capsule-v2-92b23b6a7b37aa70`, test-signed ([checkpoint](checkpoint.json)) |

[contract.json](contract.json) differs from `tests/fixtures/choice-scoring-relevance-qwen.json` only
in `manifestSha256`. The answer logits match the previous candidate's maximum error, as expected
from a byte-identical kernel on this device. Reploid's GPU participants did no GPU work during the
runs ([start](other-gpu-clients-start.txt), [end](other-gpu-clients-end.txt)).

Not established: release signing (fresh local test key and test release metadata), npm
publication, Metal or any other device, and an unrelated adopter. The archive and Capsule bytes are
local (`dist/qual-0632-20261022/`, not committed).

Component: `doppler.tests` (fixtures exercised unchanged), `doppler` (evidence).
Intent: preserved.
Acceptance evidence: this directory.
Boundary effects: none.
