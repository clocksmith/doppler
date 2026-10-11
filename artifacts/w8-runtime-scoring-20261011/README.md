# Retained scoring regression under W8-capable runtime

The existing installed-package Node/Vulkan scoring harness passed against exact
diagnostic archive `79b7c68ceb23d3e156e5e6287d978bfbe218940b832c2460b80efe2472e9b44b`.
The model, contract, independent reference, and shader pins were unchanged.
This qualifies the retained Qwen3 reranker relevance workload on this runtime,
not Gemma scoring, W8 model quality, browser scoring, or a released package.

All 12 cases returned the expected token identities and selected answers. Every
returned choice logit passed the existing 0.05 absolute-error contract; maximum
error was 0.012686491012573242. Pre-submission cancellation, cancellation after
real GPU submission, settlement, subsequent scoring with all choices compared,
model unload, and owned-device destruction passed.

Command, from the Doppler checkout:

```sh
node tests/integration/choice-scoring-physical.js \
  /var/tmp/doppler-w8-installed-settled-20261011/consumer/node_modules/doppler-gpu \
  /home/x/deco/doppler/dist/decision-acceptance/model-qualified \
  /home/x/deco/doppler/dist/decision-acceptance/model-qualified/choice-contract.json \
  /home/x/deco/doppler/dist/decision-acceptance/qwen-qualified-reference.json \
  /home/x/deco/doppler/reports/local/w8-runtime-scoring-20261011/ordinary.json \
  /var/tmp/doppler-w8-installed-settled-20261011/doppler-gpu-0.6.27.tgz
```

`ordinary.json.gz` preserves the complete results, all returned choice scores,
input tokens, artifact identities, and lifecycle checks. `ordinary.log.gz`
preserves the command output. `summary.json` is a scoped index, not a replacement
for the independent reference or a new acceptance policy.

Component: `doppler.repository-tooling`. Intent: preserved. Acceptance evidence:
the command and retained receipt above. Boundary effects: none.
