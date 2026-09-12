# Demo UI review — 2026-09-12

Component: `doppler.demo`; supporting `doppler.tests` and `doppler.repository-tooling`.

Intent: preserved.

Boundary effects: D4DA's generated `/doppler/` demo and wrapper. Its runtime,
catalog, benchmarks, Doe surface, and deployment configuration are unchanged.
No package was published and no site was deployed.

The default page has a smaller header, less repeated copy, compact empty state,
and read-only profile details behind a disclosure. Checked controls inherit
their inverted foreground; disabled controls remain readable. Model-card hover
changes both colors immediately, focus rings remain visible, and mobile controls
have larger targets. Diagnostic timing notices remain visible outside Advanced.

Profile changes reload the active model and restore the previous configuration
on failure. Stop reports cancellation and rejects late results. Clear chat,
confirmed removal, and new generation clear obsolete export state. Receipt
settings stay bound to the completed run. Model controls are locked during load
and generation; model cards and precision sorting expose selected state. Precision
replay can retry a failed load and cannot render an older prompt's late response.

The Image and Live tok/s controls were disconnected from the completed-text
inspection API. They and their unused UI handlers were removed. Completed run
timings, word quality, X-Ray, receipt export/import and precision replay remain.

Acceptance evidence:

- `npm run test:demo:contract`: passed on the canonical page and the locally
  served D4DA wrapper. Actual HTML/CSS/JS, an explicitly mocked model adapter:
  all six profiles, sampling validation, both diagnostic toggles, keyboard input,
  cancellation, import/export, clear, failed loads, confirmed removal, the install
  prompt handoff, every curated precision prompt, failed evidence retry, late
  response rejection, and control contrast at 360/390/768/1440px. Includes an OS
  dark preference with the demo's explicitly light color scheme.
- `node tools/run-node-tests.js tests/demo tests/integration/demo-surface-contract.test.js`:
  seven files passed.
- `demo:reachability:check`, `demo:shell:check`, `typecheck:source`, and
  `source:style:check`: passed.
- D4DA `check:world:doppler`, its shell generator `--check` against the retained
  hosted runtime, and `type-check`: passed. Two missing local dependencies were
  unpacked from D4DA's exact committed vendor archives for the type check.
- Desktop and mobile screenshots use the actual catalog and hosted runtime,
  without a model adapter mock. No page errors or horizontal overflow observed.

The separate `npm run test:demo:hardware` attempt could not download the model:
the model host returned HTTP 429 for all six shards. This is not a passing
real-model inference or offline-restoration qualification. The passing browser
control checks must not be presented as hardware or model-quality evidence.

Screenshots: [desktop](desktop.png), [mobile](mobile.png),
[selected mobile controls](mobile-controls.png). Raw check logs and source file
digests are retained alongside this record.

To refresh this UI independently in D4DA:

```sh
npm run sync:world:doppler -- --demo-only
npm run check:world:doppler
node ../doppler/tools/generate-demo-shell-manifest.js \
  --root sites/d4da-com/doppler --url-prefix /doppler \
  --cache-prefix doppler-world-shell- --check
```

The scoped refresh regenerates the manifest from the files actually hosted.
The normal world deployment still runs the full synchronization, so its unrelated
runtime and benchmark changes require their own release review.
