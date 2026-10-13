# Immediate paired-kernel completion repair

Component: `doppler.runtime-source.gpu.kernels`, with configuration and test owners.
Intent preserved. Boundary effects: registry policy and executor return boundary;
no shader arithmetic, model weights, reference or acceptance tolerance changes.

The retained [long-prefill scheduling control](../ordered-partitions-20261013/long-prefill-boundary/README.md#per-kernel-completion-control)
passes the original failing 184-token output stage when each immediate kernel is
settled. The production change resolves `immediateCompletion` through the
immutable registry and declares `settled` for 27 paired variants. Legacy
omission resolves to `submitted`; invalid values reject. Recorded submission
ownership is unchanged. Completion rejection and cancellation propagate after
uniform settlement. This policy is part of registry/execution identity.

Passed: focused completion-policy, unified cleanup and registry-isolation tests;
uniform generation, registry reachability and shader digest checks; source type,
architecture and config single-source checks. No shader generation was needed:
all generated checks remain current. This is source evidence, not installed
candidate or application qualification.

The broader GPU suite passes 24/26 files. Both failed sampling tests reproduce
with the original executor at `d777f2a5`: token selection at temperature 0.001
expects 4 but returns 0; distribution parity for vocabulary 4/top-P 0.75/seed 7
expects 1 but returns 0. Exact outputs and the baseline loader are retained.
These failures are not erased or counted as passes; this repair does not enable
those unsupported sampling configurations. Baseline tests use the local Apple
GPU and do not qualify the Intel model path.

## Installed candidate

Archive `2df37d88f592b523bb90c4c79840354fd301af9803a7d9009e1c8939561c71f8`
is retained with its [identity](candidate.json) and [isolated installation receipt](installed-candidate.json).
Only the six policy/executor members listed in the identity differ from `9264d265…`;
WGSL, model, references and tolerances are unchanged. The archive is not promoted.

The [original 184-token output-stage prefix](installed-prefix-comparison.json)
passes on Intel/native without scheduling or tensor debugger modification:
248,320 logits, maximum absolute error `9.465929977636733e-7` at `0.001`, selected
token 16 as expected. All 1,984 installed files were verified. Attempt, resident,
context and provider close pass; peak tracked GPU allocation is 330,004,304 bytes.
The retained supervisor receipt records normal provider exit; no new Intel fence
timeout appeared. Power was unavailable; loopback model supply does not measure
peer transfer. This is one prefix, not the full 55-prefix or application gate.

## Projection diagnostic remains failed

The [retained outlier](projection-outlier-result.json) has exact operand vectors.
[Exact rational comparison](projection-outlier-analysis.json) disproves the
hypothesis that sequential binary64 reference accumulation explains the failure:
the GPU relative error is `2.265145748959937e-12`, exceeding the original synthetic
`2e-12` limit. The reference error is `8.461521518870827e-14`. The [CPU reduction simulation](projection-outlier-simulation.json) reproduces
the physical output pair bit for bit using rounded F32 operations and exact
residuals. Accumulation/reduction rounding is sufficient to explain this retained
outlier; it does not require a GPU-specific arithmetic defect. This is one dot
product, not general arithmetic qualification, and its original threshold still
fails. The runtime arithmetic is unchanged. No threshold was weakened and no failed control is counted as passed.

Next acceptance: complete the required 55-prefix physical campaign for this exact
archive, resolve the projection diagnostic, then installed signed conversation,
storage and concurrency/lifecycle gates. The accepted public package is unchanged.
