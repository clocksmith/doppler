# Opaque heatmaps and live activity — 2026-11-08

Component: doppler.demo. The owner requested an opaque 24-bit heatmap, a numerical key,
simpler defaults, and live updates during generation. Capture/runtime contracts are unchanged.

## Repair

Previously RGB encoded three unrelated scalar values; alpha was already 255. Each pixel
now colors one scalar. Signed values use blue–neutral–red; nonnegative summaries use a
sequential palette. Probability/gate scales are 0–1; other view scales freeze at their
initial 99th-percentile magnitude. The key states bounds and quantity. Missing data is
checker-patterned, invalid numbers magenta, and padding separate. Colors clip; recording
values are unchanged. PNG previews are three-channel, 24-bit images without alpha.

Known-size missing weight tensors retain their full block extent, unknown shapes are
counted and excluded, and selected tensors/layers no longer silently substitute another
captured tensor. Partial recurrent summaries preserve unobserved blocks. Exact products
respect the recorded input-column offset. RMS shape/nonnegative constraints are validated.
The recorded 0.8B weight overview has 188/345 tensors captured; 144 shapes are unknown.
It is not a complete picture of all model parameters. Weight RMS is not importance.

Activity recording defaults on with a persistent opt-out. Fresh capture events drive
live follow while answers stream; scrubbing pauses follow and Live resumes it. The saved
20 MB example opens as its section enters view, with a separate persisted opt-out. It
never executes a model and cannot replace a newer live run/import. Six primary view
switches remain visible; advanced capture/identity controls stay under Details.

## Verification

- activation-grid.test.js: scalar mapping, opacity, missing/invalid/zero distinction,
  legacy/new display import, exact numerical preservation and capture bounds.
- activity-grids.test.js: independently calculated products, RMS, causal attention,
  KV history, gaps, selected tensor/layer, unknown coverage and recording validation.
- TypeScript source strict check passed; demo reachability and generated-shell check passed.
- Browser contract passed with mocked model execution: default recording and persisted
  opt-out; a contribution map changes before completion; scrub/Live/cancel; 320/390/1280 layouts.
- The retained physical Qwen 0.8B example rendered all 18 supported views, each with
  fully opaque pixels; export/import works offline. Expert routing remains unavailable.
  This is recorded-data UI verification, not a new model inference qualification.
- Preview generator check passed against the digest-bound sample; no numerical bytes
  were changed. Existing legacy recordings remain importable.

Attention is a CPU-double reconstruction from captured Q/K, not a bit-exact kernel result.
Weight/input overviews are RMS products, not causal attribution. Recurrent prefill state
is recorded after chunks, not every intermediate prompt position. Raw windows and capture
budgets remain bounded. No new all-model, 27B, MoE or long-context qualification is claimed.

The deployed physical smoke at source 0c0ff3ac is retained in physical-0c0ff3ac.json: Qwen 0.8B returned “hello!”, 72/72 layer frames and 576 activity frames with no gaps or browser errors. The visible heatmap changed 40 times while generation was active. This verifies that short deployed stream, not arbitrary models or long captures.
