# Model paths and honest playback — 2026-11-08

Component: doppler.demo. Owner chose spatial model paths: layers and branches with
live capture activity, and asked to separate views that do not actually animate.

Model paths is the default activity view. It uses the recording's layerTypes,
captured operator roles, token positions and timestamps. A fixed layer strip
shows output RMS. The selected layer exposes attention/memory probes and the
feed-forward gate/up/product branches. The latest captured stage has a bold
outline. Selecting a layer pins it; the Model paths control restores following.
The last layer includes captured next-token scores. Missing architecture metadata
disables this view; missing operator outputs remain empty. Unrecorded transforms
are explicitly omitted. Lines express computational structure, not per-edge
importance or literal data motion. Summary RMS weights each block by its count.

Playback groups:

- Token playback: activations, FFN stages, differences, KV, query attention,
  gates, logits/probabilities and available expert routing.
- Captured updates: model paths, recurrent/convolution snapshots and observed
  weight-dispatch events. The latter views advance their own capture boundaries.
- Snapshots: weights, weight quantization error and whole-prompt attention.
  Their Live/Play/scrub and playback-detail controls are hidden.

Focused evidence: independent RMS/causal reference fixtures and no-future-capture
checks; TypeScript source strict; demo reachability and generated shell; browser
contract with mocked generation and retained physical example; 18 pixel views
plus the path view, 1200/390/320 layouts, static control suppression, changing
path playback, layer selection, offline export/import. See retained receipts.

One new presentation module owns the graph. The module ceiling increases by one,
with an explicit removal condition in the owning shell generator. The byte ceiling
stays 10 MB; lossless PNG compression reduces preview bytes without changing pixels.

These checks verify data projection and UI. They do not establish new numerical,
quality, 27B/MoE or long-context qualification. Runtime and serving services are unchanged.
