# Opt-in Qwen 3.8 27B Orin profile

`profiles/qwen3-8-27b-orin-prefill` is a calibration-only, experimental runtime
profile for `qwen3-8-27b-text-q4k-ehaf16`. It is not a catalog default or public
provider admission. Select it explicitly through `request.runtimeProfile` in a
calibration command; other intents are deliberately excluded.

The profile keeps the original artifact identity and compressed Q4K weights.
On devices without shader-f16, it reads packed F16 embedding words through a
split-storage GPU gather rather than expanding the entire embedding to F32.
It selects the existing F32 WideTile shader for seven prefill projections.
Decode selection, session policies and nonprojection operations are unchanged
relative to the packed-embedding baseline. Other models keep their existing
capability fallback.

## Evidence and limits

A bounded Jetson AGX Orin Node WebGPU experiment on November 18, 2026 used the
same artifact, private package and prompts for a sequential baseline/candidate
comparison. GPU-timestamped 22-token prefill fell from 24.390 s to 2.453 s.
An unprofiled counting request fell from 77.033 s to 48.205 s; its 70-token
decode stayed approximately unchanged at 44.622 s versus 44.926 s.
Both Hello requests and the complete 1–20 counting output were exact. Seven
WideTile numerical parity shapes passed, including a 5120-wide reduction.

These are short-prompt measurements from one machine, not a broad quality or
throughput claim. Load times varied substantially and must not be attributed
to this prefill-only change. Long prompts, concurrent requests, intermediate
width coverage and additional hardware remain qualification work. The existing
35B public provider was restored and separately verified after the experiment;
this profile was not deployed as a public 27B provider.

The source-bound compiler fixture records the tested manifest identity
`sha256:e29626a8d3c41c0af8c0dd945d028464dcebfef2413ba85e4f5e8695d5910d06`.
It contains the execution/architecture/quantization subset, not a loadable model
or a new artifact identity. Compiler tests prove the explicit selection survives
capability remapping and leave the stock fallback unchanged.

## Local checks

```sh
node tests/config/qwen3-8-27b-orin-prefill.test.js
node tests/gpu/gather-packed-f16.test.js
node tests/loader/packed-f16-split-embedding.test.js
```

The physical packed-gather test additionally requires a real GPU. A CPU unit
pass is not a replacement for physical model qualification.

## Separate decode candidate

`profiles/qwen3-8-27b-orin-decode` extends this profile, explicitly selecting the
existing eight-lane subgroup fused Q4K gate/up kernel during decode only. It
requires subgroups and remains calibration-only; it is not a serving default.

A same-build, sequential Orin comparison using the same Hello/count prompts
passed exact outputs. Counting decode fell from 44.887 s to 40.746 s; complete
request time fell from 48.165 s to 44.026 s. A fully covered 806-pass diagnostic
decode step measured gate/up at 99.122 ms versus 160.540 ms, and total GPU time
at 413.091 ms versus 472.092 ms. Fifteen numerical cases, including hidden width
5120, preceded the run. A synthetic repeated-row full-shape probe supported the
choice but is not a substitute for the model measurement.

This is one short-prompt comparison, not broad quality, concurrency, long-context
or cross-device qualification. The original prefill-only profile and global
fallback remain unchanged. The live 35B provider was restored separately; public
27B admission remains closed.

## Separate projection candidate

`profiles/qwen3-8-27b-orin-projections` extends the decode candidate with the
existing subgroup8 Q4K kernel for decode q/k/v/o/down projections only. Runtime
arrays replace rather than concatenate, so this profile explicitly repeats the
inherited embedding/prefill patches; the regression protects their retention.
Fused gate/up and the LM head remain unchanged. It requires subgroups, remains
calibration-only, and does not change public 27B admission or serving defaults.

A fresh sequential baseline/candidate run on Orin passed identical Hello/count
outputs (70 generated tokens for counting). Count decode fell from 40.545 s to
36.784 s; complete request time fell from 43.831 s to 40.068 s. Model opening
was 40.001 s versus 40.345 s, not a loading improvement. The profiled decode
step fell from 413.133 ms to 364.759 ms GPU time; fused gate/up remained about
99 ms. The projection changes account for the measured reduction.

Before that run, 38 numerical cases passed against CPU and baseline results,
including partial blocks and no-shader-f16 hardware. Six actual projection
shapes with synthetic repeated rows passed full-output CPU parity below
0.000067 absolute error. Eight interleaved measurements per arm (after two
warmups) reduced median submission/readback wall time by 6.5–19.8%. These
synthetic vectors are not captured model activations. Neither that probe nor
the short sequential model run establishes broad quality, long-context,
concurrency, cross-device or statistically robust superiority.

The public 35B provider is restored and checked independently after testing.
Run the configuration regression with:

```sh
node tests/config/qwen3-8-27b-orin-projections.test.js
```
