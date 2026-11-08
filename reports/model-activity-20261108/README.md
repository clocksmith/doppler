# Model activity capture and synchronized playback — 2026-11-08

Component: `doppler.runtime-source` and `doppler.demo`. Intent preserved.
Boundary effects: public inspection receipts gain optional bounded activity data;
D4DA hosts the synchronized player, numerical example and updated runtime.

## Physical evidence

One Qwen 3.5 0.8B Q4K artifact, manifest
`c5fcf38e945f1c048faca9ccda8991f1494563317f46a6dd79e0714ad001c569`,
executed through the public browser model handle on Apple Metal (physical adapter,
Chrome 145). The input was `Say hello in one word.`, temperature 0, top-k 1,
top-p 1, maximum 3 output tokens. The reply was `Hello!`.

Both controlled pairs passed exact generated-token and recorded-token-score
comparison, cancellation during capture and subsequent reuse:

- `demo/guided-quality` versus `demo/model-activity`: [capture receipt](capture-check.json).
- `demo/deep-xray` versus `demo/model-activity-timing`: [timing capture receipt](timing-capture-check.json).

Each final activity run retained 576 operator captures, plus 72 layer-output
captures, with zero failed/dropped captures. It observed 188 supported weight
matrices; the 345-tensor inventory explicitly retains unsupported descriptors.
The activity payload accounts for 33,460,800 bytes of JavaScript numbers; layer
payload, weight summaries and metadata are additional. These are diagnostic
observations, not throughput benchmarks or new model-quality qualification.

Reproduce locally with the verified artifact staged at
`.cache/qwen08-sha256-repair/`:

```sh
node tools/experiments/activity-grids-browser.js --physical
node tools/experiments/activity-grids-browser.js --physical --timing
node tools/experiments/package-activity-example.js .cache/activation-grid/activity-receipt.json
DOPPLER_ACTIVITY_SAMPLE=1 npm run test:demo:contract
```

The actual numerical example is [activity-example.json.gz](../../demo/assets/activity-example.json.gz),
with SHA-256, byte lengths and per-view coverage in
[activity-example.json](../../demo/assets/activity-example.json). It is about
20 MB compressed and 27 MB packed JSON. It contains numerical data, identity,
input/output token IDs and fixed weight summaries/windows. No model is needed
for saved playback. Packaging verifies a lossless export/import round trip.

## Actual example coverage

These dimensions are the displayed RGB overview/window of the **0.8B recording**,
not theoretical full-detail 27B dimensions. Each RGB pixel carries three mapped
scalars; the saved numerical values retain their recording precision.

| View | Displayed pixels | Representation | Recorded quantity / unavailable reason |
|---|---:|---|---|
| Layer outputs | 91 × 91 | raw | layer-output values |
| FFN features · activated product | 170 × 170 | raw | activated gate × up-projection values |
| FFN features · gate projection | 170 × 170 | raw | ffn-gate values |
| FFN features · up projection | 170 × 170 | raw | ffn-up values |
| Activation changes | 91 × 91 | reconstructed | signed activation difference |
| Weights | 144 × 144 | aggregated | weight RMS per block |
| Weight × input | 98 × 98 | aggregated | RMS connection product per output-row group |
| New attention memory | 46 × 46 | raw | stored keys and values |
| Accumulated attention memory | 192 × 192 | raw | stored keys and values |
| Recurrent memory | 79 × 79 | aggregated | recurrent-memory · block RMS |
| Convolution history | 24 × 24 | aggregated | convolution-history · block RMS |
| Attention · current query | 17 × 17 | reconstructed | attention probabilities |
| Attention · prompt relationships | 72 × 72 | reconstructed | attention probabilities |
| Memory retention & update gates | 14 × 14 | reconstructed | retention factor, update strength per head |
| Next-token scores | 288 × 288 | raw | next-token logits |
| Next-token probabilities | 288 × 288 | reconstructed | full-vocabulary softmax probability |
| Weight quantization error | 37 × 37 | reconstructed | stored head minus tied embedding reference |
| Weight residency & access | 11 × 11 | raw | resident tensor and observed projection binding |
| Expert routing | Unavailable | raw | No expert-routing boundary was captured for this execution. |

The synchronized player has token-position, operator-capture and recorded-time
modes. Prompt positions computed together are not presented as chronological
execution steps. State views identify their chunk boundary. Fixed weights stay
static; contributions animate using retained projection inputs. Switching modes
preserves selection. Raw, summary and reconstructed views are labeled and missing
captures stay explicit. Full-detail extent and summary/window extent are separate.

## Scoped limits

- FFN gate/up values exist when those stages are materialized; fused decode
  intermediates remain missing. FFN recording keeps four positions per chunk.
- State summaries cover bounded reads; raw windows keep 16,384 values. They do
  not reconstruct intermediate states within a prefill chunk.
- Weight products cover one observed FFN down projection. Four exact rows plus
  column-energy groups support exact mathematical windows and aggregated RMS;
  other exact connections require matching weight bytes. Adapter-effective
  products are unavailable when adapter matrices were not captured.
- Attention is a CPU-double reconstruction from recorded Q, stored K, mask,
  scale and softcap. It is not kernel-bit-exact or causal attribution.
- Quantization error is four Q4K head rows minus the pinned F16 tied-embedding
  reference, not an original-F32 or entire-model error map.
- Residency is a loaded-tensor snapshot plus observed FFN down-dispatch bindings;
  hardware cache hits, upload traffic and other operator accesses are unobserved.
- Dense Qwen has no experts. MoE capture hooks exist, but no physical MoE example
  was qualified here; its option is unavailable in this example.
- 27B, other backends/cache layouts and long captures were not physically tested.
  Limits explicitly preserve gaps instead of presenting omitted values as zero.

## Browser and development checks

[Playback receipt](playback-check.json) renders all 18 enabled views using the
physical example, checks desktop/390/320 layouts, shared selection, bounded raw
windows, playback, export and import with networking blocked. The ordinary demo
contract uses a mocked model; it separately verifies controls, branching,
cancellation, profile settings, contrast and layout. Neither is a new physical
inference test. Screenshots are retained alongside this report.

Focused tests cover independent product/energy calculations, causal attention,
missing keys, append-only KV, bounded state, gates, exact-prefix comparisons,
packed interchange and malformed input. Inspection contracts and activation
representation checks pass. Strict source types, source style/architecture,
public demo reachability, generated shell, package source closure, installed
runtime closure and packed-package smoke pass. No unrelated providers were
started or stopped.

## Retained failures and corrections

The first combined capture failed because layer snapshots rejected an active
command recorder: [original failure](recorder-failure.log). Capture now submits
the producing command prefix before enqueuing its staging copy, retaining normal
recorder cleanup ownership. The matched physical lifecycle check passed afterward.

A comparison between timing-enabled capture and the non-timing baseline had
identical tokens but different scores: [unmatched-policy receipt](unmatched-policy-check.json).
Those execution modes are not interchangeable controls. The two **matched**
pairs above pass; no cross-policy bit-exact claim is made.

Browser checks found and corrected stale range bounds clamping a newly imported
recording to an old token position, grid changes resetting the timeline, duplicate
notices and stretched example buttons. Source checks also caught a private path
in a demo type annotation; it now uses its local declaration boundary.
