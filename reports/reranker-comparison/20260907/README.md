# Browser reranker comparison

[Acceptance](acceptance.json) retains six interleaved fresh-browser runs on AMD
Radeon 8060S. Both engines passed the same three frozen source references,
including all input tokens, raw logits, probabilities, and exact ranking, before
timing and on every measured query. Each opening used one warmup and three timed
iterations of all three queries. OS caches were uncontrolled; models came from
local HTTP. These are scoped observations, not overall superiority claims.

| Median of three openings | Doppler | Transformers.js |
| --- | ---: | ---: |
| Browser launch to model ready | 29.424 s | 6.452 s |
| Browser launch to first result | 30.077 s | 6.762 s |
| Model loading | 28.758 s | 6.113 s |
| Warm documents/second | 4.811 | 23.983 |
| Sampled peak renderer RSS | 1,609,859,072 bytes | 6,438,203,392 bytes |

These product paths use different weight encodings and execution precision.
Doppler uses the installed `61a23b9c…` archive and the original signed Q4K/f32
application Capsule. Transformers.js 4.2.0 uses a source-derived fp16 ONNX graph
with last-position full-vocabulary logits. Both use the original true-logit
reranking score, not a replacement yes/no softmax. No generation comparison is
implied. Runtime version `0.6.0` alone does not identify the Doppler archive.

Lifecycle was observed separately, once per engine. The abort was requested after
actual GPU submission. Doppler rejected with `AbortError` after 199.8 ms;
Transformers.js continued and returned the full query after 118.6 ms. The latter
is an ignored abort, not cancellation latency. Both then passed every source
reference, rejected execution on a destroyed device, and recovered with valid
outputs in a fresh browser. Recovery included context closure, browser launch,
model opening and all references: 31.556 s for Doppler and 6.948 s for
Transformers.js. The shared recovery budget was 600 s. There is no repeated
lifecycle timing or full application-behavior equivalence claim.

The evidence archive preserves the failed q4 and q8 numerical controls, fp32
startup failure, initial third-party module MIME failure, interrupted download,
and the first cancellation probe that failed to intercept an instance queue
wrapper. The fp32 browser error does not prove its cause was memory exhaustion.
No reference tolerance was relaxed. Only the accepted fp16 path entered timing.

Reproduction uses the existing release restorer and qualifier tools:

1. Clone this repository and run `npm ci` and `npx playwright install chromium`.
   Use the [pinned release reconstruction](../../../docs/capsule-release-reproduction.md)
   to obtain its exact runtime, signed application Capsule, and pinned source
   files in a new directory, called `RESTORED` below.
2. Extract `reproduction.tar.gz` into a new directory, called `REPLAY` below.
   The tested Python environment was 3.14.4 with PyTorch 2.11.0+cu130 executing on
   CPU; exact package metadata requirements are in `export-requirements.txt`.
   Install those dependencies in an isolated Python environment. Run
   `python REPLAY/export-f16.py RESTORED/sources/reranker NEW_VENDOR_DIRECTORY`.
   The portable export was executed again and reproduced all six files exactly,
   including ONNX SHA-256
   `e145ff161cf3bab40c07afae6b9846a11b87204eec51f3700d7f96337e047e4c`.
3. From the repository, run
   `node REPLAY/configure-replay.js "$PWD" RESTORED NEW_VENDOR_DIRECTORY NEW_RUN_DIRECTORY`.
   This generates relocated inputs while preserving the frozen corpus and
   policy. It pins the actual replay source hashes; replay observations do not
   inherit historical acceptance.
4. Run the two controls, sequentially, before any timing:
   `node tools/qualify-transformersjs-reranker.js NEW_RUN_DIRECTORY/quality-transformersjs.json`
   and `node tools/qualify-installed-reranker-browser.js NEW_RUN_DIRECTORY/quality-doppler.json`.
5. Run `node REPLAY/replay-paired.js "$PWD" NEW_RUN_DIRECTORY`, then separately
   `node REPLAY/replay-lifecycle.js "$PWD" NEW_RUN_DIRECTORY`.

Replace uppercase path labels with absolute paths and use `TMPDIR=/var/tmp`.
Every run requires a new directory. Preserve application approval, retained-use
decisions, and release checkpoints. The scripts do not sign or promote releases.
The source model is `Qwen/Qwen3-Reranker-0.6B` at
`e61197ed45024b0ed8a2d74b80b4d909f1255473`; its Apache-2.0 source card and metadata
are retained by the original reconstruction. Weights are reconstructed from that
public source, not committed here. An export with different bytes must be
qualified as a new artifact.
