# Native MoE private-array removal, Jupiter

The retained implementation removes dynamic private-array indexing from native
Q4_0/Q4_1/Q8_0 block unpacking, vector products, and routed accumulators. Named
words/vectors preserve original packed bytes, per-pair addition order, reduction,
route slots, and F32 storage/arithmetic. No requantization or new runtime policy.
The shader digest and experimental profile pins change together. Compiler
register allocation was not inspected; the measured result establishes the
benefit, not a particular machine-code explanation.

## Controlled screen

Both sessions use the convolution repair from `1c9c08387`, 256-token chunks,
unchanged model/settings, and frozen sources. One warmup plus three measured
alternating pairs per short/long prompt; four quality tasks also repeat four
times per engine. No outliers removed. Candidate loads second.

| Mean, milliseconds | Original MoE | New MoE | Difference |
| --- | ---: | ---: | ---: |
| Short prefill | 137.20 | 133.00 | -4.20 |
| Short first token | 139.90 | 139.60 | -0.30 |
| Short 128-token request | 1824.80 | 1879.43 | +54.63 |
| Long prefill | 4181.73 | 3835.73 | -346.00 |
| Long first token | 4187.23 | 3842.43 | -344.80 |
| Long 128-token request | 5997.70 | 5656.57 | -341.13 |

Long prefill decreases 8.3% in this screen. All 48 outputs are identical between
implementations, all four quality tasks pass, and no GPU errors are recorded.
The short-request regression remains visible. Earlier own-engine screens show a
resident-session order effect on decode; do not attribute it to prefill math.
This screen supports the local implementation change, not a Pooled win, full
answer qualification, or production latency-tail claim.

## Numerical checks

Five physical browser GPU fixtures compare the original shader and CPU reference
with the new implementation: Q4_0 split GU, Q4_1 down, Q8_0 split GU, and odd
block-count Q4_0/Q8_0 tails. All output bits match the old shader; CPU-reference
tolerances remain unchanged. Partial groups, bank boundaries and native half-word alignment are exercised
by these fixtures.

Full-model one-token probes cover saved token prefixes of lengths
33, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025, 2047, 2048 and 2049.
Both numerical lanes use 2048-token chunks, the repaired convolution, identical
F32 weights/KV and a materialized-logit path. **All 3,228,160 compared logits match
exactly**, with identical greedy tokens. This isolates the MoE rewrite against
the old implementation; it does not establish an independent full-model oracle.

The separate chunk-policy check passes the unchanged absolute 0.001 gate at the
same 13 prefixes (worst 0.0001767874, identical tokens). Numeric probes are
untimed diagnostics. Whole-token replay is explicitly disabled because full
logit observers cannot run in replay mode. The first attempt correctly rejected
that incompatible configuration before completing a probe; its failed receipt
is retained. Primary throughput runs keep replay enabled.

Larger chunks remain an explicit experimental overlay. The checked-in candidate
keeps its 256-token policy: numeric probes include latency stalls, and two
throughput prompts do not justify a universal chunk-policy change. Earlier
[rejected shader experiments and convolution failure](../qwen36-35b-prefill-20261101/README.md)
remain retained. Their smaller changes are superseded by this complete rewrite.

## Reproduction and scope

[Measurements](measurements.json) retain every screen sample and both numerical
gates. [Raw-file bindings](retained-files.json) cover all attempts, readiness,
configuration, executed harnesses, failed observer setup, and numerical probes.
[Source bindings](source-files.json) cover the frozen archive and candidate patch.
External paths are relative to `/run/media/x/models/`.

Frozen candidate archive:
`deco-evidence/doppler-35b-startup-20261101/native-moe-registers/source.tar.gz`.
Baseline archive is reproducible from commit `1c9c08387`; source roots are recorded
in execution receipts. Extract on a filesystem supporting the tracked
`src/inference/config -> ../config` symlink. Runs hold the shared GPU timing lock.

Jupiter only: AMD Strix Halo, physical WebGPU, Linux, HeadlessChrome 146.
Model/artifact is the unchanged Qwen3.6-35B-A3B Q4_0 source RDRR used by the
[startup report](../qwen36-35b-startup-20261101/README.md). Native decode, speculation,
quality-admission thresholds and model release metadata are unchanged.
Browser results do not qualify Node or other GPUs. The original Pooled
common-weight/independent numerical gate and best speculative configuration
remain separate, unfinished requirements.

Passed: five physical component fixtures, 48 continuation attempts, 13 full-model
vocabulary probes, `kernels:check`, `onboarding:check`, `package:smoke`, and
`git diff --check`. No broad release or device matrix was run.

Component: `doppler.runtime-source.gpu.kernels`, `doppler.runtime-source.config`,
`doppler.tests`, `doppler` (evidence).
Intent: preserved.
Acceptance evidence: focused commands and bound receipts above.
Boundary effects: none.
