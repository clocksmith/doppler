# Model Atlas generator

`../build-model-atlas.js` relocates the original Atlas renderer and its retained
2026-11-07 analysis inputs from temporary storage. Requires Node 22 and Python 3
(standard libraries only). It reads `models/catalog.json` for current lifecycle
labels. Shape counts, tensor inventories and pipeline prose are retained
inspection records, not fresh execution or model qualification. Input digests
are recorded in `snapshot.json`; update them deliberately after reviewing an
input change. Local model files and network access are unnecessary.

Generate the public page from the Doppler repository:

```
node tools/build-model-atlas.js --output ../d4da/sites/d4da-com/doppler/atlas/index.html
node tools/build-model-atlas.js --output ../d4da/sites/d4da-com/doppler/atlas/index.html --check
```

Generation preserves existing model IDs and adds each full ID as an anchor for
Bench Explorer links. Precision comes from retained manifest fields, never IDs.
No analytics is emitted. D4DA owns deployment; the generator only writes the
explicit output file. It fails on missing/corrupt inputs, unresolved slots,
private hostnames or a stale output in check mode.
