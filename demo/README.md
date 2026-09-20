# Doppler Demo Contract

`demo/index.html` and `demo/demo.js` are the hosted product surface. There is
no second demo implementation. `npm run demo:reachability:check` fails when an
unreferenced JavaScript implementation appears under `demo/` or live demo code
imports private `src/` paths.

The composer streams text through inspection token events. Incoming IDs are
buffered and decoded together at most once per animation frame, preserving split
Unicode characters and tokenizer spacing. The existing answer text node receives
only the changed suffix; token updates never rebuild the conversation or reload
the page. Auto-scroll follows the answer only while the reader stays at the bottom.
Stop flushes pending text and retains the partial answer without publishing a
completed receipt. Completion reconciles the text with the receipt, then displays
word quality, X-Ray evidence, and final timing. Image attachment and live token-rate
controls are not offered.
The current UI starts with Tokens enabled, X-Ray and Perplexity disabled (saved
X-Ray/Perplexity preferences take precedence), and a 256-token output limit.
Tokens selects guided-quality observation, not performance-representative timing.
Disable Tokens, X-Ray, and Perplexity to use `demo/always-on`.
Completed assistant messages render inert Markdown;
word-quality annotations stay attached to their source text. Clear conversation
resets model state before removing the conversation and its evidence, and keeps
the loaded model available. A reset failure leaves the conversation intact.
Run options expose sampling and observation choices; profile-owned policy is
shown under read-only details. Diagnostic timing notices remain visible whenever
Tokens, X-Ray, or word quality is selected. Loading, generation, cancellation, receipt
import/export, and confirmed cache removal keep their controls synchronized.

## Public boundaries

The model picker, verified OPFS cache, model lifetime, and generation path use
the root `dr` API from `doppler-gpu`. Runtime-profile controls use
`doppler-gpu/tooling/runtime`. Evidence views use
`doppler-gpu/tooling/evidence` and the public `model.inspect` handle. Live demo
code may not import the compatibility `doppler-gpu/tooling` barrel.

`model.inspect.generate(prompt, { onEvent, generation, policyId })` emits ordered
`{ type: 'token', tokenId, index, token? }` events during generation, including
the first and stop tokens. Probability-capturing policies attach the matching
inspection token record so interfaces can render confidence as the answer arrives.
Consumers decode IDs using `model.advanced.decodeTokenIds(ids)`;
an ID is not necessarily a complete character. Events do not change decode
batching or request extra GPU readbacks. Callbacks are synchronous and should
buffer work. The final `{ type: 'inspection-complete', receipt }` event and returned
receipt retain the existing contract. Aborted or failed runs do not emit completion;
callback errors reject the run and follow normal generation cleanup.

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
