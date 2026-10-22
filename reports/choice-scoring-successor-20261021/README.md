# Choice scoring on the 0.6.32 candidate archive (2026-10-21)

Re-runs the existing installed choice-scoring qualification
([prior record](../choice-scoring/README.md), archive 0.6.4) for the
[reranker successor](../qwen-reranker-successor-20261021/README.md) on a 0.6.32 candidate archive.
The task, cases, labels, `maxSeqLen` 512, the 0.05 answer-logit tolerance and 12 required correct
choices are unchanged. [contract.json](contract.json) differs from
`tests/fixtures/choice-scoring-relevance-qwen.json` only in `manifestSha256` (recorded in `derivedFrom`).

| Step | Result |
|---|---|
| CPU reference, `tests/integration/choice-scoring-reference.py` (Transformers Qwen3ForCausalLM eager F32, GGUF Q4_K dequantization) | 12/12 ([reference.json](reference.json)) |
| `npm pack` at `c704cc58` → `doppler-gpu-0.6.32.tgz` `2cc4d07c…`, installed into a fresh consumer | 1 package, no runtime dependencies |
| Physical Node, `webgpu@0.4.0`, RADV, `tests/integration/choice-scoring-physical.js` | 12/12, max answer-logit error 0.0127 ([report](physical-node.json), [qualification](physical-node.json.qualification.json)) |
| Physical Chrome 146, Reploid `tests/fixtures/doppler-choice-browser-check.js` | 12/12, max error 0.0127 ([report](physical-browser.json), [qualification](physical-browser.json.qualification.json)) |
| Rig, `tests/integration/rig-choice-scoring-capsule.js` | Capsule `…-capsule-v2-77d2938ba0606000`, test-signed ([checkpoint](checkpoint.json)) |

The predecessor's qualification measured 0.0161 on the same cases. Scores remain uncalibrated.

Not established: release signing (the Capsule uses a fresh local test key and test release
metadata), npm publication, other devices, and an unrelated adopter. The archive and Capsule bytes
are local (`dist/qual-0632-20261021/`, not committed).

Component: `doppler.tests` (fixtures exercised unchanged), `doppler` (evidence).
Intent: preserved.
Acceptance evidence: this directory.
Boundary effects: none.
