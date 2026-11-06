# Doppler Demo Contract

`demo/index.html` and `demo/demo.js` are the hosted product surface. There is
no second demo implementation. `npm run demo:reachability:check` fails when an
unreferenced JavaScript implementation appears under `demo/` or live demo code
imports private `src/` paths.

First-time visitors see an interactive, explicitly labeled example of word
choices without downloading weights or presenting its timing as a device result.
Word choices, local writing, and recorded precision comparison are direct actions.
Model setup opens on demand; sampling, run receipts, and precision score tables
remain available under their respective details. The example does not become
conversation history or replace a returning conversation. Live generation and
recorded examples share the existing inspection view and execution owners.

The composer streams text through inspection token events. Incoming IDs are
buffered and decoded together at most once per animation frame, preserving split
Unicode characters and tokenizer spacing. The existing answer text node receives
only the changed suffix; token updates never rebuild the conversation or reload
the page. Auto-scroll follows the answer only while the reader stays at the bottom.
Stop flushes pending text and retains the partial answer without publishing a
completed receipt. Completion reconciles the text with the receipt, then displays
token probabilities, X-Ray evidence, and final timing. Image attachment and live token-rate
controls are not offered.
Every live answer records each generated token's exact ID and a bounded set of
alternatives; the default output limit is 256 tokens. Tokens whose selected
probability is below the threshold (50% by default) are underlined as branch
points, colored red (0%) to blue (100%); "all tokens" underlines every token.
Clicking one lists its alternatives. The latest answer shows as this underlined
token text instead of formatted Markdown; earlier turns render Markdown.
Branch, beside Send and off by default, only enables continuing: picking an
alternative continues the answer from that exact token boundary and keeps the
original, with a switcher at each fork. Turning Branch on makes the answer
already shown branchable. Branches can branch again. A branch
replays the run's exact `inputTokenIds`, the answer's token IDs before the fork,
and the chosen ID through `model.inspect.generate(prompt, { inputTokenIds })`;
generation starts from cleared attention and recurrent state, so displayed text
is never re-tokenized. Replay prefills the prefix in one pass, so its numerics
can differ slightly from the original token-by-token decode. Fragments of one
character display together but fork at their first token. Recording changes
execution (probability capture disables command batching), so every live run
uses `demo/guided-quality` (`demo/deep-xray` with X-Ray) and its timings are
diagnostic; the demo does not run the `demo/always-on` tier. Probabilities are model
likelihoods, not factual accuracy. Recorded and illustrative examples cannot
branch, and a tree is discarded when the conversation or model changes. X-Ray is a separate
GPU view with captured step durations and per-kernel totals. Missing measurements
remain unavailable. Its accumulated-duration waterfall is not an absolute dispatch
timeline or a GPU-utilization measurement. Saved X-Ray preferences remain
explicit choices.
Completed assistant messages render inert Markdown;
word-quality annotations stay attached to their source text. Clear conversation
resets model state before removing the conversation and its evidence, and keeps
the loaded model available. A reset failure leaves the conversation intact.
Run options expose sampling and observation choices; profile-owned policy is
shown under read-only details. The diagnostic timing notice stays visible because
every live answer records alternatives; X-Ray replaces it with its own. Loading, generation, cancellation, receipt
import/export, and confirmed cache removal keep their controls synchronized.

## Public boundaries

The model picker, verified OPFS cache, model lifetime, and generation path use
the root `dr` API from `doppler-gpu`. Runtime-profile controls use
`doppler-gpu/tooling/runtime`. Evidence views use
`doppler-gpu/tooling/evidence` and the public `model.inspect` handle. Live demo
code may not import the compatibility `doppler-gpu/tooling` barrel.

The demo passes `{ messages }`, including retained prior turns, to inspection.
Branch replay uses the exact templated prefix returned in `inputTokenIds`.

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

## Public hosting

The canonical demo is https://d4da.com/doppler/. D4DA owns its world-site
Firebase configuration and deployment. The source stays in this repository;
`../d4da/scripts/sync-world-doppler.mjs` copies tracked demo/runtime/catalog
assets, rewrites their URLs to `/doppler/`, and regenerates the shell inventory.
Model weights and local artifacts are excluded. This repository's hosting
commands delegate to D4DA and cannot overwrite CanvasContext's Fawn deployment.

```sh
npm run demo:hosting:build
npm run demo:hosting:preview
npm run demo:hosting:deploy
```

Verify preview startup and interactions before production. D4DA's
`/doppler/release.json` identifies the source commits. CanvasContext hosts
Fawn through D4DA's `firebase.fawn.json`; old Doppler demo, runtime, and catalog
URLs redirect to their matching `/doppler/` paths. Legacy Mandate routes retain
their m3t4.ai redirects. CanvasContext retires Doppler's old worker and shell
caches without deleting model bytes or other user storage. Local storage and
model caches remain on their original origin.

The separately inspectable precomputed run was captured with `model.inspect.generate` in Chrome WebGPU. `data/sample-inspection.js` retains the real token probabilities, artifact fingerprint, timing, capture date, and raw receipt digest. The illustrative opening performs no inference or downloads and carries no measurements from the recorded run. Prompt suggestions are editable; choosing a model is explicit before a new visitor runs their own prompt.
