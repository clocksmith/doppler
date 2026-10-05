# Installed choice scoring acceptance

[checkpoint.json](checkpoint.json) binds the tested archive and expanded installed
tree. [independent-reference.json](independent-reference.json),
[physical-node.json](physical-node.json) and
[physical-browser.json](physical-browser.json) retain all twelve reviewed results.
The task is query/document relevance, with a purpose-trained Qwen reranker.
Both physical surfaces select every reviewed answer correctly. Maximum absolute
answer-logit error is `0.01611161231994629`, below the frozen `0.05` limit.
Scores are uncalibrated. This is one AMD RDNA3 machine, not cross-device acceptance.

The [rejected screenings](negative-screenings.json) retain Gemma 270M, Gemma 1B
meaning checks and Gemma 1B routing results. None earned its frozen quality gate.
These findings do not establish universal model quality or performance rankings.

## Reproduction

Run from Doppler with the pinned original model directory available. The source
manifest digest is in [the source contract](../../tests/fixtures/choice-scoring-relevance-qwen.json).
CPU prerequisites and exact observed versions are recorded in the reference.
The physical Node fixture uses `webgpu@0.4.0`; Chromium qualification uses the
installed Chrome channel and a real WebGPU adapter. Run GPU processes sequentially.

```sh
node tests/integration/prepare-choice-scoring-model.js \
  models/local/qwen-3-reranker-0-6b-q4k-ehf16-af32 \
  tests/fixtures/choice-scoring-relevance-qwen.json \
  src/config/conversion/qwen3/qwen-3-reranker-0-6b-q4k-ehf16-af32.json \
  dist/decision-replay/model
python3 tests/integration/choice-scoring-reference.py \
  --model dist/decision-replay/model \
  --contract dist/decision-replay/model/choice-contract.json \
  --out dist/decision-replay/reference.json
```

The preparation fixture refuses to overwrite an experiment and links unchanged
weight/tokenizer bytes. It updates explicit shader identities and includes the
conversion contract's loading mechanisms. Original manifests remain unchanged.
An old manifest omitted loading shaders; signed opening rejected it correctly.

Pack and install one archive into a fresh consumer, then run:

```sh
node tests/integration/choice-scoring-physical.js \
  dist/decision-replay/consumer/node_modules/doppler-gpu \
  dist/decision-replay/model dist/decision-replay/model/choice-contract.json \
  dist/decision-replay/reference.json dist/decision-replay/node.json \
  dist/decision-replay/archive/doppler-gpu-0.6.4.tgz
```

From Reploid, `tests/fixtures/doppler-choice-browser-check.js` accepts installed
Doppler, model, contract, CPU reference, archive and output paths in that order.
It qualifies the browser independently. Both fixtures emit `.qualification.json`
only after numerical, task-quality and lifecycle gates pass.

Use existing Rig through the test fixture:

```sh
node tests/integration/rig-choice-scoring-capsule.js \
  dist/decision-replay/model dist/decision-replay/node.json.qualification.json \
  ../reploid/artifacts/doppler-choice/replay-browser.json.qualification.json \
  dist/decision-replay/capsule
```

It generates fresh local test keys and binds the actual workload and independent
reference. Test release metadata does not qualify upstream source/licensing or
authorize a public release. Keep private keys outside Git. Signed execution and
real peer delivery use [Reploid's existing fixtures](https://github.com/clocksmith/reploid/blob/main/artifacts/doppler-choice/README.md).

## Gates and boundaries

Full `npm run check:green` passed. The additional close-during-scoring regression,
final dependency inventory, public export parity and installed bytes check also
passed. The new operation requires explicit signed qualification; generation
evidence cannot grant it. ModelIR v2 scoring is not implemented.

No production recurrent arithmetic, memory ceiling, frozen partition reference,
Reploid chat pin or Doe policy changed. Scoring tolerances do not replace the
partition generation gate of `0.001` in both physical directions.

Component: `doppler.runtime-source.config`, `doppler.runtime-source.client`,
`doppler.runtime-source.inference`, `doppler.runtime-source.converter`,
`doppler.runtime-source.tooling`, `doppler.tests`, `doppler.docs`.
Intent: preserved.
Acceptance evidence: linked observations, checkpoint and executable fixtures.
Boundary effects: public `scoreChoices` contract and operation qualification;
Reploid consumes results without implementing another scorer.
