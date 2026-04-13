# gemma4 e2b f16 replay evidence on amd radv

This directory promotes a curated subset of the local report run at
`reports/f16-precision-collapse/20260410T000000Z-gemma4-e2b/` into a tracked
surface.

- model: `gemma-4-e2b-it-q4k-ehf16-af32`
- host: AMD RDNA-3 `radeon-8060s-graphics-radv-strix-halo-`
- driver: `radv: Mesa 26.0.3-1ubuntu1`
- aggregate: `256` prompts, `61` f16-vs-f32 top-1 flips, `46` persistent
  flips, `15` healed flips

Tracked contents:

- `summary.json`: full prompt-by-prompt replay summary from the local AMD run
- `summary.md`: human-readable replay summary
- `slices/`: selected local slices that are especially interesting because they
  either flipped on this AMD/RADV host while the Apple/M3 replay stayed stable,
  or they show a materially different flip boundary on this host

Selected slices:

- `hotfix-approve-deny-choice.json`
- `pool-safe-unsafe-choice.json`
- `punc-slash-choice.json`
- `password-public-private-choice.json`
- `json-open.json`
- `code-import-what.json`
- `punc-tilde-approx.json`
- `val-normal-defined.json`
