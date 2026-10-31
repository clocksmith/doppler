# Isolated recurrent compilation investigation

Component: `doppler.runtime-source.gpu.kernels`. Intent: preserved.
Boundary effects: diagnostic native-provider process only; public participants
and accepted packages unchanged.

This directory continues the retained `../paired-gate-phases/intel-host` failure.
That failure exhausted host RAM; it did not establish a GPU allocation peak or
normal GPU settlement. Do not rerun its unchanged full workload.

The isolated probe admits no weights, tensors or GPU dispatch. It verifies the
original shader and native-provider hashes and cgroup limits before opening a
device: 6 GiB memory.high, 8 GiB memory.max, zero swap, group OOM kill, and a
300-second systemd runtime bound. These are diagnostic host-memory limits, not
the unchanged 1,420,000,000-byte executor GPU allocation ceiling.

## Preserved failures

1. `MemoryOOMGroup=yes` was rejected before launch by systemd 255. The service
   preflight instead used `OOMPolicy=kill` and verified `memory.oom.group=1`.
2. The user service inherited stale supplementary groups from the existing user
   manager. Vulkan could not read the render device; the probe rejected the
   software adapter and closed its native host. See `service-attempt/`.
3. The corrected scope inherited SSH's render group without restarting the user
   manager or changing device permissions. `probe.js` also sets and reads back
   the scope's delegated `memory.oom.group`. It opened the Intel Gen12LP adapter,
   validated the unchanged shader, then died with SIGSEGV inside
   `libvulkan_intel.so` during asynchronous pipeline initialization. See
   `isolated-events.jsonl`, `crash-stack.txt` and `crash-receipt.txt`.

The provider was PID 917114; scope invocation was
`4825273dea124235b1917fc7a511b554`. The launch exited 139. The scope's own
`Result=success` means the scope ended, not that compilation passed. No pipeline
completion or host-close event exists. Normal GPU settlement is unconfirmed.
Apport records process VmHWM 524,232 KiB. This distinct compiler crash does not
reproduce or explain the original 26,728,260 KiB anonymous-RSS OOM by itself.

`reservation.json` preserves the original service admission and probe hash;
`service-attempt/probe.js` preserves those exact probe bytes. The corrected scope
probe is separate from that immutable launch identity. Public native PID 297260
and browser controller PID 299769 remained present after the isolated crash.

## Operator repair and physical result

Source `1d1145ad` separates recurrent computation into Q/K normalization,
state update, output normalization and gating. It also removes duplicated
sigmoid division callsites without changing branch operands. These nine total
pipelines compile within the bounded native operator run, but `operator/`
retains its numerical failure. Its old harness exited zero despite `passed:
false`; the receipt is authoritative. Follow-up controls return failure status.

`gate-readback/` locates the first invalid values at log-decay preparation.
`log-boundary/` captures valid exponential outputs and logarithm inputs followed
by invalid logarithm outputs. `log-series/` and `log-alone/` then evaluate the
unchanged logarithm on those exact inputs successfully, with maximum absolute
error 1.17e-15. This establishes a compilation-composition dependency on this
provider, not the driver's internal cause.

Source `f2b5a2ef` retains four gate phases while moving the softplus exponential
into the scale phase's second workgroup row. Both rows share one exponential
callsite. An additional eight bytes per token/head retain the exact logarithm
input; the full allocation is admitted before dispatch and remains owned until
settlement. The next phase performs logarithm and the original scale operation.
Neither elementary arithmetic nor the reference changes.

[`gate-separated/result.json`](gate-separated/result.json) passes both original
packed/unpacked fixtures, all convolution, gate, recurrent-state and output
comparisons at the unchanged 2e-11 operator tolerance. All nine retained shader
hashes match the source repair. Maximum error is 3.11e-15; tracked GPU allocation
peaks at 634,976 bytes, including diagnostic snapshots. The cgroup peak is
2,348,400,640 bytes of host memory with no limit/OOM events. Submitted work,
host close, device destruction and provider exit are observed. Public native
297260 and browser controller 299769 remained present. No weights were loaded.
Existing provider caches were retained, so these timings do not qualify cold
startup or performance. Earlier failed runs remain separate evidence.

[`operator-repair.json`](operator-repair.json) binds the source, fixtures,
receipt hashes, checks and remaining gates. Four focused CPU suites, kernel
generation/registry/digests, source types, style and architecture checks pass.
Component: `doppler.runtime-source.gpu.kernels`; intent preserved. Boundary
effects: gate scratch allocation/bindings and pinned shader identities.

Exact installed-runtime qualification, all-55 full-model numerical placements,
signed N-stage ordinary chat, concurrent lifecycle and release remain open.
