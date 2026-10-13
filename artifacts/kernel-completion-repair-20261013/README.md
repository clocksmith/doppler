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

Next acceptance: freeze and install the new archive, rerun the original prefix
without debugger treatment, then the complete required prefix and installed
conversation/lifecycle campaigns. The accepted public package remains unchanged.
