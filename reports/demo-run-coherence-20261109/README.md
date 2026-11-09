# Demo run identity and loading layout — 2026-11-09

The reported first-load state paired a Gemma selection and the editable sky
prompt with an automatically opened Qwen activity capture whose output was
`Hello!`. The activity sample loader was independent of the conversation owner.

The demo now starts with the prompt only. Opening the activity example explicitly
replaces the displayed exchange with `Say hello in one word.` / `Hello!`, from
`tools/experiments/activity-grids-browser.js`. Its existing compressed capture is
unchanged and SHA-256 checked; `demo/assets/activity-example.json` now preserves
that prompt. New activity exports retain prompt and output text. Legacy captures
without text disclose that absence rather than borrowing the current draft.

Starting any generation clears previous activity, including when recording is
off. Clear/model changes and other answer imports also clear it. Switching a
completed branch restores its own trace when available. Pending example fetches
are aborted when the displayed run changes.

The loading animation is anchored independently of status height. Status, detail
and elapsed time have reserved slots. Full errors remain visible below. Duplicate
boot CSS was removed; the initial HTML owns the loading layout.

## Checks

- `DOPPLER_ACTIVITY_SAMPLE=1 npm run test:demo:contract`: passed. Mocked generation
  and live event delivery, plus playback of the existing physical Qwen capture;
  **no new model execution or GPU qualification**.
- Fresh and cached-model visits with the old auto-example preference enabled:
  editable default prompt, no answer, no sample fetch, no grid.
- Explicit sample and offline export/import: matching prompt, answer and activity.
- Recording-off generation and opening an answer without a capture: no stale grid.
- Delayed example fetch superseded by a new recording-off run: cancelled, no overwrite.
- Loading anchors unchanged for short/long/empty details at 320, 390 and 1280px.
- Strict source types, demo reachability and generated shell checks: passed.

Component: `doppler.demo` (tests and packaging tooling follow its contract).
Intent: preserved — recorded samples keep text and measurements together.
Boundary effects: demo presentation and optional recording text metadata;
D4DA hosts the refreshed demo. Runtime/model/precision gates are unchanged.
