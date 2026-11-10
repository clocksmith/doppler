# Gemma 270M: 128 MiB embedding bindings

Development check on Mac Chrome / Apple Metal, not physical-phone qualification.
The pinned public `gemma-3-270m-it-q4k-ehf16-af32` artifact was loaded through the
demo and generated a reply to “Answer with one word: what is the capital of France?”

The baseline used the adapter limits. The second page requested an actual GPUDevice
with a 128 MiB storage binding limit, 256 MiB buffer limit, and eight storage buffers
per shader stage. Its cached startup loaded the 320 MiB F16 embedding as three
sections. Both complete generation paths produced token IDs
`818,5279,529,7001,563,9079,236761,107,106`.

`summary.json` retains the model manifest identity and limits. The full inspection
receipt and browser logs/screenshots remain local (ignored); this is one prompt,
not a model-quality or performance qualification. The diagnostic demo captures
activations, so its timings are not representative inference benchmarks.

Implementation: a split4 F16-to-F32 gather uses six storage buffers, preserving
Gemma's F32 activations and original weights. The loader admits this split for the
ordinary gather path. Existing split tied-output-head execution completed too.
Cached-model failures now dismiss startup and leave the picker and error visible.
Fresh startup keeps the default model instead of selecting a prerecorded sample's
model. Unsupported geometries or unavailable features still fail explicitly.

Checks: six focused loader/gather/default/boot test files; physical immediate and
recorded gather parity at section boundaries and invalid indices; demo contract
smoke; kernel registry/uniform/digest/rule generation; source style and reachability.
Reproduce the composed GPU check with `node tools/gemma3-mobile-binding-check.js`.
The tool also accepts `DOPPLER_DEMO_URL`, `DOPPLER_DEMO_PREFIX`, and
`DOPPLER_MOBILE_REPORT` for the hosted route.

Three PWA PNGs were losslessly recompressed; pixel equality was checked. This keeps
the generated offline shell below its existing 10 MB limit, including hosted URL
prefixes. No budget, manifest, model dtype, or qualification threshold changed.
