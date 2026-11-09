# Doppler Demo Contract

`demo/index.html` and `demo/demo.js` are the hosted product surface. There is
no second demo implementation. `npm run demo:reachability:check` fails when an
unreferenced JavaScript implementation appears under `demo/` or live demo code
imports private `src/` paths.

The page has three sections with consistent spacing and disclosures. Section names are accessible landmarks, not visible headings. Model labels stay exactly as
published in the catalog, configuration included:

1. **Dialogue**: the model selector with a compact Load button, one animated loading/generation indicator, the conversation with clickable words and alternatives, the message box
   with Send and Stop, and answer and engine settings under **Options**.
2. **Model activity**: what to record during the next answer (activity grids,
   GPU timing, or both), selectable grids with one player and timeline, recorded timing and device details for the latest answer, Save run, Save grid and one Open file
   control, and measurements and color settings under **Details**.
3. **Precision effects**: **Your prompt** (a live 32-bit versus 16-bit run) and
   **Recorded examples** (saved Gemma 3 270M calculations), each with side-by-side
   results first, one sentence saying what changed, and the rest under Details.

Buttons fit their labels; selects and text inputs fill the available width.
Before anything is recorded, a short instruction replaces the empty grid.
Model activity uses one visible view selector. Preview shortcuts, recording files
and longer explanations live under Details, along with timing, coverage, capture method and measurement overhead. The color key and token position remain beside the visual. Saved examples retain a visible source label; live runs do not repeat model identity. Follow live appears only after pausing live
follow, and Follow latest layer appears only after pinning a diagram layer.
Routine phase names remain accessible without visible repeated status text. Errors and stopped runs stay readable; reduced motion makes the indicator static. Secondary actions use quiet text controls. Loaded models show status rather than
a redundant disabled Load button.
An empty composer receives the example prompt on both fresh and cached-model
visits, without replacing entered text. Starting a recorded run leaves static
snapshots for live Activation changes, comparing the current and previous token.
Model paths remains selectable and includes a layer-by-dimension activation
heatmap from the captured layer outputs, with a fixed signed color scale. Rows
fill at their capture boundaries; this is not a simultaneous hardware snapshot.
Wide screens place activity beside the conversation. Narrow screens bring the
activity into view on Send and keep Stop reachable while generation runs.

Fresh and returning visits automatically replay the first exchange: the real Qwen
0.8B recording of “Explain why the sky is blue in one sentence.” Its answer and activity use
one player; words appear with their corresponding completed layer stack and scrubbing
rewinds both. A small Recorded replay label identifies it; model, publication date and timing
semantics live under Details. It requires no model download or inference. Typing,
changing models, clearing or sending cancels a pending opening or automatic playback.
Once the first exchange opens, its submitted prompt is cleared from the composer.
Typed or launched drafts cancel automatic opening and are preserved. The sample never
creates live model state or enters its conversation history. Send produces the new
answer and its activity together. A single
“Try another example” button deals the other prompts from
`examples.json` in shuffled order, without repeats until the bank is used up.
There are no category buttons. Live generation and
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
alternatives; the default output limit is 256 tokens. Every token is underlined,
colored red (0%) through purple to blue (100%) by its selected probability.
Clicking any word lists its alternatives. The latest answer shows as this
underlined token text instead of formatted Markdown; earlier turns render
Markdown. Picking an alternative continues a live answer from that exact token
boundary and keeps the original, with a switcher at each fork. Branching is always
available without a toggle. Branches can branch again. A branch
replays the run's exact `inputTokenIds`, the answer's token IDs before the fork,
and the chosen ID through `model.inspect.generate(prompt, { inputTokenIds })`;
generation starts from cleared attention and recurrent state, so displayed text
is never re-tokenized. Replay prefills the prefix in one pass, so its numerics
can differ slightly from the original token-by-token decode. Fragments of one
character display together but fork at their first token. Recording changes
execution (probability capture disables command batching), so every live run
uses `demo/model-activity` by default (`demo/model-activity-timing` with GPU timing), or `demo/guided-quality` / `demo/deep-xray` after activity is turned off and its timings are
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
shown under read-only details. The recording notice in Model activity stays
visible because every live answer records alternatives; activity grids and GPU
timing replace it with the selected policy’s notice. Loading, generation, cancellation, receipt
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
- `demo/model-activity` captures bounded numerical activity and weight summaries.
  `demo/model-activity-timing` additionally records GPU timestamps. Both change
  execution and prohibit performance claims.
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

The first exchange was captured with `model.inspect.generate` in Chrome WebGPU.
`data/sample-inspection.js` retains its real probabilities, fingerprint, timing,
publication date and raw receipt digest. It matches `assets/activity-example.json.gz`
by fingerprint and exact input/output token IDs. Unknown imported recordings
remain inspectable, but do not borrow token text or probabilities from this sample.
Prompt suggestions are editable.

## Activation recording and precision views

The visible model selector sits above the conversation. The **layer grid** records
bounded asynchronous snapshots of real layer outputs during prefill and decode.
Positions stay stable across tokens, with a fixed scale and one scalar per opaque
24-bit heatmap pixel. The system dark theme uses a dark neutral for signed zero;
the grid, diagram and legend change together without rescaling data or using alpha.
Missing captures are marked, never filled with zeros.
Live follow updates the selected grid as capture events arrive during generation;
scrubbing pauses follow, and **Live** resumes it. The composite combines sequential
layer captures and does not imply simultaneous state. Recording limits stop
observation without stopping generation. Exported recordings contain numerical
samples, layout/scale, policy and model/build identity; imported playback needs
no model. Activity and GPU timing can be recorded together. **Open file** accepts
a run file or a standalone grid recording and dispatches on its schema.

**Your prompt** runs the message box's prompt, or the last message sent when the
box is empty (earlier turns are not included), against the default and
declared F16-requested paths, with identical weights, exact prompt tokens and
greedy settings. Actual resolved precision and execution identities are retained;
selective F16 paths may keep F32 residual activations. Unsupported candidates fail
visibly. The temporary second model needs additional memory and is always unloaded.

**Recorded examples** are separate, saved Gemma 270M final LM-head
arithmetic replays. They are not current-conversation measurements or complete
F32-versus-F16 model runs. Neither comparison is a throughput or quality benchmark.


## Model paths, snapshots and playback

**Model paths** is the default activity view. Its layer strip has stable positions;
the selected layer exposes the recorded attention/memory kind and the feed-forward
gate/up/product branches. Nodes show RMS from captured values, with explicitly
weighted state summaries. Missing stages stay empty. The last captured stage gets
a bold outline; the overview follows its layer during live capture. Selecting a
layer pins it until **Model paths** or **Follow latest layer** restores follow.
Connections show model structure, not per-edge influence or literal moving data.
Normalization/residual transforms and unrecorded outputs are explicitly omitted.
This view requires recorded layer types; it does not invent an architecture for
recordings without that metadata.

Views are separated in the selector and primary labels:

- **Token playback**: layer outputs, FFN stages, differences, KV, current-query
  attention, memory gates, logits/probabilities and available expert routing.
- **Captured updates**: model paths advance the captured operators; recurrent
  and convolution state advance their own chunk/decode snapshots; residency
  advances observed dispatch events. Values can remain unchanged between updates.
- **Snapshots**: fixed weights, weight quantization error and whole-prompt
  attention. These have no Play, Live or scrub controls. Whole-prompt attention
  shows the recorded prompt as a whole; it can fill in during live prefill.

## Model activity recording and playback

**Model activity** follows the displayed run. No example downloads automatically,
including for visitors with the old automatic-example preference. Details can
open a real Qwen 0.8B capture (about 23 MB compressed) together with its recorded
“Say hello in one word.” prompt and “Hello!” answer. The compressed bytes are
SHA-256 checked before import, outside the application-shell cache. An example
fetch cannot replace a newer live run, cleared conversation or imported recording.
Starting any run, including with recording off, clears the previous activity.
Model changes and branch switching also clear or replace the activity with the
matching capture. New recording files retain prompt and answer text; legacy files
without text explicitly say it was not saved. All views share one selector and player.

The loading animation has a fixed viewport anchor. Status, two-line detail and
elapsed time occupy reserved slots; full loader details remain in the text/title,
and failures appear below without moving the animation.

**Record my answers** defaults on, with a persistent opt-out. Recording scans
supported loaded weights and adds bounded GPU readbacks; its timing is diagnostic,
not representative of normal inference. Live follow defaults on for each run.

The Grid selector covers layer outputs, three FFN stages, differences, weights,
weight×input products, new and accumulated KV, recurrent state, convolution
history, attention rows and prompt relationships, memory gates, logits,
probabilities, quantization error, residency/access, and expert routing. A view
without its required captures stays visibly unavailable. A fused-away stage is
missing at that position, rather than reconstructed without notice.

All views share position, layer and timeline. Switching grids preserves the
position. **Token positions** plays completed observations for each input position;
prefill positions were computed together, not sequentially. **Operator captures**
advances 12 captured boundaries/second at 1×. **Recorded time** follows capture
timestamps, including observation overhead. Static weights do not animate.
Recurrent and convolution snapshots are after chunks; they cannot reveal
uncaptured states between positions inside a chunk.

Each pixel represents one scalar using opaque 24-bit RGB color. Signed values
use blue–neutral–red; magnitudes use dark blue–teal–yellow. Probabilities and gates
use 0–1. Other scales freeze at the initial 99th-percentile magnitude per view,
with their limits visible; changing tokens never rescales them. Missing data uses
a checker pattern, invalid numbers magenta, and padding a separate dark color.
The visible key names the quantity and bounds. Captions distinguish captured
values, reconstructed values and block summaries. Hover reads values; Details
contains full extents and capture identity. The color mapping clips extremes but
never changes saved numerical values. Legacy RGB-packed recordings still import,
with their numerical data preserved and their exported display metadata updated.

Known-size missing weight tensors retain their complete block extent. Tensors
without recorded shapes are counted and excluded, rather than given an invented
size. An explicitly selected tensor never silently substitutes another tensor.
Block boundaries do not cross tensors, and the final block can be shorter.

Weights use RMS blocks of 16,384 values. The first supported FFN down projection
retains four exact rows and squared-weight column sums grouped over 128 output
rows; these reconstruct mathematical connection products and RMS summaries
from actual post-cast inputs. They do not claim kernel-rounded arithmetic or
causal importance. Exact zoom outside the saved window needs the matching
weight bytes. Effective adapter products are unavailable when adapter matrices
were not captured. The initial residency inventory is tensor-level; observed
FFN down-dispatch bindings update it. Uploads, other operator accesses and hardware
cache hits are not measured by this view.

Attention is reconstructed from captured Q and actual stored K using the
recorded mask, scale and softcap. It uses CPU double arithmetic, not kernel
bit-exact replay. New KV writes are append-only; missing earlier keys make an
attention row unavailable. Logit softmax is full-vocabulary, before sampling
filters. Quantization error requires a retained reference: this example compares
four Q4K head rows against their F16 tied embedding rows from the same pinned
artifact, not against an original F32 checkpoint.

The default policy retains up to 32 MiB each for layer and activity payloads,
plus 16 MiB of weight summaries/windows (accounted as JavaScript numbers; engine
and metadata overhead is additional). FFN capture keeps the last four positions
of a prompt chunk. State raw windows keep 16,384 values, with 256-value RMS blocks
across the bounded state read. Reaching a limit leaves explicit missing captures
and does not substitute zeros or stop generation. Large views page at 786,432
scalars. Saved v2 files pack F32 values and F64 summaries into base64 little-endian
payloads; older v1 files remain readable. Imports validate dimensions and byte
budgets before projection, and a rejected import preserves the accepted recording.

Physical capture, cancellation/reuse and offline UI evidence:
[2026-11-08 activity report](../reports/model-activity-20261108/README.md).
This qualifies the demonstrated 0.8B recording on the reported browser/device;
it does not qualify 27B capture, MoE capture, other cache layouts or arbitrary
long recordings.
