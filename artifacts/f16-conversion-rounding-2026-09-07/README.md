# F32 to F16 conversion rounding repair

Component: `doppler.runtime-source.converter`. Intent: preserved.
Boundary effects: newly converted tensor bytes and Q4_K scale encodings may
change. Previously converted and signed artifacts are not rewritten. No model
publication, catalog promotion, or runtime fallback is authorized by this record.

## Observed defect

An F16 precision control for Reploid's document-answer investigation sampled 16
evenly spaced values in each of 320 converted text tensors. All source/converted
shapes and the 24-layer pattern matched. Of 5,120 samples, 54 values across 14
linear-attention normalization tensors differed from source values rounded to
nearest F16, ties to even. These tensors originate as F32; the source also
contains BF16 tensors. The full sample record preserves every value, not only
failures.

The completed integrity check independently rehashed the 4,548,221,488-byte
source checkpoint and all 57 original F16 shards. The checkpoint matches its
retained SHA-256 and every shard matches its manifest BLAKE3 commitment. The
original manifest SHA-256 is
`9051eeff39378ac6bf84560ca676fc4ab24e50534fd450d99155aae20fc37734`.
`integrity.tar.gz` retains the verifier, completed report, and command output.

For example, source `0.955810546875` rounds to `0.9560546875` (`0x3ba6`), but the
converter emitted `0.95556640625` (`0x3ba5`). The old `float32ToFloat16()` discards
the lower mantissa bits. A focused regression using this exact source value
fails on the original function with `15269 !== 15270`.

The source checkpoint is `Qwen/Qwen3.5-2B` revision
`15852e8c16360a2fea060d615a32b45270f8a8fc`. Its acquisition record, original F16
manifest, conversion configuration, and numeric samples are retained in the
archive. The original conversion uses the installed Reploid Doppler 0.6.0
package. The repair modifies the owning Doppler source, not that installed
package and not the existing artifacts.

## Repair and checks

`src/converter/quantizer.js` now rounds nearest-even for normal and subnormal
halves, including exponent carry and overflow. Its declaration documents the
encoding rule. Tests cover signed zero, infinity, NaN, the exact failing source
value, and both signs immediately below, at, and above every adjacent finite-half
midpoint. A converter integration fixture verifies actual F16 shard words from
F32 source values rather than deriving expected words with the same encoder.

- Original regression: fails on the recorded value.
- `node tests/converter/quantizer.test.js`: passes.
- `node tests/converter/core-bf16-f16-conversion.test.js`: passes.
- `node tools/run-node-tests.js tests/converter`: 44 files pass.
- `npm run source:style:check`: passes.
- `npm run source:architecture:check`: passes.
- `npm run catscan:check`: passes, 30 components.
- `npm run typecheck:source`: passes.
- `npm run artifact:contract:check`: passes, seven of 31 catalog models exposed.
- `git diff --check`: passes.

The broader unit run completed: 781 test files passed. Installed package smoke,
model-release checks, and Program Bundle checks also passed. Installed execution
smokes use synthetic programs and are not physical qualification.

The fresh conversion completed with 57 shards and 320 text tensors. All 5,120
samples now equal the source rounded to F16, with no failing tensor samples.
Source checkpoint SHA-256 and all 57 shard BLAKE3 commitments passed independent
rehashing. The new manifest SHA-256 is
`b7b2f051c82aba757101eaa301fa6eccae91d3d56be9a52f1543e079544e131e`.
The 59-file executable inventory contains 3,772,899,642 bytes. This remains an
unsigned local precision control, not a qualified Capsule.

`verified-conversion.tar.gz` retains the completed conversion report, manifest,
config, source snapshots, lineage, samples, integrity verification, artifact
inventory, and completed check logs. It omits model weights. The output root is
`/tmp/reploid-qwen35-answer-screen.06HATl/f16-rne-model`.

The complete green workflow first failed the package size budget. The unchanged
1,780-file closure contains six changed shipped files adding 2,597 source bytes.
`package-audit.json` records the source revisions and per-file accounting. The
measured package is 2,107,701 packed bytes and 10,869,933 unpacked bytes. The
budget adjustment preserves closure, forbidden-file, file-count, and size gates;
the public boundary check then passed. A subsequent workflow failure exposed a
stale routing inventory after the Gemma qualification manifest was committed.
Its regenerated inventory records 30 manifests and preserves 32 surfaced
integrity findings. Neither regeneration nor this repair resolves those findings.
The subsequent complete workflow result must be checked separately.

The first physical attempt stopped before generation because the temporary
browser context offered 1.75 GB of storage against a 3.51 GB model-cache
requirement. It generated zero cases and completed cleanup without errors. This
is an infrastructure failure, not evidence about model output quality. A fresh
persistent-profile retry retains the same model bytes, prompts, generation
settings, and semantic acceptance requirements.

## Claim boundary

This establishes a conversion-code defect and its focused repair. It does not
establish that truncation caused the Q4K browser's invented carton weight, that
the new F16 model answers correctly, or that the assistant is qualified. Task
quality still requires physical execution, unchanged acceptance, an untouched
holdout, and independent semantic review. The old F16 artifact is retained as an
adverse observation and is not admitted as a source-rounded precision control.

The archive omits weights. It contains source snapshots, tests, configuration,
sampled observations, and completed check logs. Unrelated dirty model-session
and structured-generation work in Doppler is preserved and not part of this
repair.

*Last updated: September 2026*
