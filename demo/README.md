# Doppler Demo Contract

`demo/index.html` and `demo/demo.js` are the hosted product surface. There is
no second demo implementation. `npm run demo:reachability:check` fails when an
unreferenced JavaScript implementation appears under `demo/` or live demo code
imports private `src/` paths.

## Public boundaries

The model picker, verified OPFS cache, model lifetime, and generation path use
the root `dr` API from `doppler-gpu`. Runtime-profile controls use
`doppler-gpu/tooling/runtime`. Evidence views use
`doppler-gpu/tooling/evidence` and the public `model.inspect` handle. Live demo
code may not import the compatibility `doppler-gpu/tooling` barrel.

## Observation tiers

Observation behavior is owned by
`src/config/inspection/observation-policies.json`.

- `demo/always-on` records artifact identity, token IDs, and existing wall
  timing. It does not alter execution or enable GPU timestamp queries.
- `demo/guided-quality` captures selected-token probabilities and word quality.
  It changes execution and its timing is not representative.
- `demo/deep-xray` enables diagnostic execution and GPU timestamps. It changes
  execution and permits diagnostic claims only.

Every inspection receipt contains a canonical full fingerprint plus separate
quality and performance fingerprints. Quality comparison rejects tokenizer
changes. Performance comparison rejects policies that modify execution and
requires matching artifact, tokenizer, prompt tokens, sampling, execution plan,
browser, and adapter identity.

Word quality uses
`doppler.word-segmentation/unicode-whitespace-v1` and
`doppler.perplexity/summed-word-surprisal-v1`. The view displays summed word
surprisal, rolling perplexity with an explicit word or token window, and
cumulative sequence perplexity.

## PWA and evidence

`npm run demo:shell:generate` resolves the deployed module graph and writes
`generated-shell-manifest.js`. Its digest creates the service-worker cache
namespace. `generated-shell-budget.json` records file, module, and byte budgets.
The service worker owns the application shell; verified models remain in OPFS.
Precision replay loads its digest-bound evidence only after the user opens it.

- `npm run test:demo:contract` exercises the real page with an explicitly
  mocked adapter and emits `doppler.demo-contract-receipt/v1`.
- `npm run test:demo:hardware` exercises real WebGPU, closes every page,
  disables networking, restores the same OPFS artifact, regenerates, and emits
  `doppler.demo-hardware-receipt/v1`.

The hardware lane also checks shell upgrade cleanup and a noncritical partial
cache miss. The goal matrix requires both receipts. Kernel verification remains
a lower-level prerequisite, not proof that the demo works.
