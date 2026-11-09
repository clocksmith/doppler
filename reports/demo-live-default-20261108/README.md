# Default prompt and live activation changes — 2026-11-08

Demo da5611f2; hosted runtime ca685808, 0.6.33. Cached-model boot now fills an
empty composer. Activation changes is the default; static snapshots yield to it
on Send. Wide screens show conversation and activity side by side. Model paths
also contains a layer-by-dimension captured activation map.

Strict source types, activity unit checks, reachability, shell checks and browser
contracts passed. The browser contract includes cached-model boot, snapshot-to-live
transition and a changing default heatmap before generation completes. Its model
is mocked; retained-recording playback is separate from physical execution.

Physical Qwen 0.8B on Metal, using the untouched default sky prompt and 16-token
output cap: 384/384 layer frames, no layer gaps, 71 visible color changes,
74 observations fully inside the viewport, zero browser errors. This validates
the layer-output / activation-change stream, not answer quality or long runs.

The first 16-token attempt failed the older full-detail zero-gap assertion:
2,200 of 2,864 detailed operator captures exceeded the existing recording budget.
The failure is retained (local checkout path redacted). The scoped follow-up
accepts only explicitly budget-dropped detailed captures while retaining zero-gap
requirements for layer outputs. It confirmed exactly that cause. Full-detail
3-token smoke remains a separate strict check; it was not rerun here. No recording
budget, runtime, numerical or performance gate was changed.

Component: doppler.demo. Intent: default presentation changed at owner request.
Boundary effects: D4DA hosted demo. GPU provider services untouched.
