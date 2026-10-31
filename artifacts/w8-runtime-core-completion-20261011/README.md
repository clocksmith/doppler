# Core operation regression on diagnostic archive 7c240dad

Both existing operation suites passed on archive
`7c240dad4d1ae98e204e9727e687b9bc4379abd758b16af0fd857a99d7adac8d`
from `/var/tmp/doppler-w8-installed-completion-20261011`.
These runs preserve the qualified models, weights, tokens, references, and
tolerances. Earlier archive receipts remain retained separately. This run includes the
first-token completion classification repair from `e571e220`.

| Operation | Physical surface | Result |
| --- | --- | --- |
| Qwen3 relevance scoring | Node/Vulkan, AMD RDNA3 | All 24 choice logits and 12 selections pass; maximum absolute error 0.012686491012573242 under the existing 0.05 contract. |
| Signed Qwen3 embedding | Browser/WebGPU, AMD RDNA3 | All 18 texts and 1,024 values per text pass; maximum absolute error 0.00007295608520507812 under the existing 0.02 contract. |

Scoring includes exact token identities, pre-submission cancellation, actual
submission cancellation, settlement, all-choice comparison after reuse, and
model/device cleanup. Embedding includes repeat and subdivision checks,
original order reconstruction, actual-submission cancellation, settled cleanup,
and an exactly matching reversed full batch afterward. The embedding cancellation
observed 12 submissions: active allocations remained 16,777,216 bytes across two
buffers, retained model allocations remained 1,191,632,896 bytes, and deferred
allocations were zero at completion. Its session cleanup returned no errors.

The signed embedding opening used real current eligibility time
`2026-10-11T19:31:54.471Z`, sequence 2, with the unchanged signed event expiring
`2026-10-12T12:19:10.042Z`. No backdated eligibility, new signer, or replacement
qualification was introduced.

Commands, from the Doppler checkout:

```sh
node tests/integration/choice-scoring-physical.js \
  /var/tmp/doppler-w8-installed-completion-20261011/consumer/node_modules/doppler-gpu \
  $HOME/deco/doppler/dist/decision-acceptance/model-qualified \
  $HOME/deco/doppler/dist/decision-acceptance/model-qualified/choice-contract.json \
  $HOME/deco/doppler/dist/decision-acceptance/qwen-qualified-reference.json \
  $HOME/deco/doppler/reports/local/w8-runtime-scoring-completion-20261011/ordinary.json \
  /var/tmp/doppler-w8-installed-completion-20261011/doppler-gpu-0.6.27.tgz
node tools/qualify-embedding-browser.js /tmp/doppler-w8-embedding-completion.json
```

The exact embedding config is retained in `embedding-config.json.gz`; a future
run must use its actual eligibility time and still-valid signed release evidence.
Compressed reports retain all returned values, identities, failures where
applicable, and lifecycle results. `summary.json` binds their complete hashes.

These results establish retained operation compatibility for this exact runtime
archive. They do not qualify Gemma embeddings, W8 model quality, a performance
advantage, external adoption, or the standard release. CPU checks were active;
this evidence makes no timing claim. Each GPU process/context was closed before
the next owner received the GPU.

Component: `doppler.repository-tooling`. Intent: preserved. Acceptance evidence:
the existing commands and retained reports. Boundary effects: none.
