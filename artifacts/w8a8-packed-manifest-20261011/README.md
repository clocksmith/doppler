# Packed-dot diagnostic manifest

This derivative selects the existing packed integer-dot W8A8 shader using the
same row-packed weight bytes and activation/rescale contract as the portable
manifest. It is prepared for installed-model comparison, not model acceptance,
conversion postflight, or release promotion.

- Original manifest: `6bd380101b72d79aa05c23d51783a9faab9505808d63c4bd86520e95d255c03c`.
- Derived manifest: `67eee682dea361346ca1ae16026e53f4162b4e43f45dc9c53e4b30919b79065e`.
- Local derivative: `/var/tmp/doppler-gemma270-w8a8-packed-diagnostic-20261011`.

Only the W8A8 kernel filename and canonical content digest change in execution.
The manifest-variant ID is recomputed through `buildArtifactIdentity`.
`conversionConfigPath` and `conversionConfigDigest` are removed from the derived
identity: the original converter recipe did not emit this packed shader choice.
Their original values remain in `derivation.json`. Weight-capsule identity,
tensor integrity, all seven shards, and tokenizer/config bytes remain identical.
The inherited `metadata.source` and `metadata.convertedAt` describe the original
weight conversion, not a new conversion of this derivative. The opaque original
weight-capsule ID remains unchanged and does not declare the storage dtype.

The complete semantic diff has exactly five paths: the two kernel fields,
the two removed conversion-provenance fields, and the new manifest-variant ID.
Canonical manifest validation passed without errors or warnings. All nine
non-manifest files were independently hashed after copying. The source manifest
was verified unchanged afterward.

`manifest.json.gz` preserves the exact derivative. `derive.js.gz` is the executed
derivation script; decompress it and run from the Doppler checkout, with an unused
destination directory:

```sh
node /tmp/doppler-derive-packed-manifest.js \
  /var/tmp/doppler-gemma270-w8a8-final-20261011 \
  /var/tmp/doppler-gemma270-w8a8-packed-diagnostic-20261011 \
  artifacts/w8a8-packed-manifest-20261011/derivation.json
```

Component: `doppler.repository-tooling`. Intent: preserved. Acceptance evidence:
`derivation.json`, manifest validation, exact file hashes and the bounded semantic
diff. Boundary effects: no runtime or model arithmetic changes.
