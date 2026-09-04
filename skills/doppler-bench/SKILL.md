---
name: doppler-bench
description: Capture a reproducible Doppler or approved vendor benchmark receipt when a model, workload, runtime lane, and comparison question are explicitly identified.
---

# Doppler Benchmark Capture

## Prerequisites

- Run from the Doppler repository root.
- Record model/artifact identity, workload, runtime surface, cache mode, profile,
  warmup/run counts, sampling policy, hardware, and the comparison question.
- Read `docs/agents/benchmark-protocol.md` and
  `docs/style/benchmark-style-guide.md` for publication-grade evidence.

## Procedure

For a Doppler-only receipt:

```bash
npm run cli -- profiles --json
npm run bench -- --config '{"request":{"modelId":"MODEL_ID","cacheMode":"warm"},"run":{"surface":"browser","bench":{"save":true}}}' --runtime-profile profiles/throughput --json
```

For a paired engine comparison:

```bash
node tools/compare-engines.js --mode compute --warmup 1 --runs 3 --decode-profile parity --save --json
```

For vendor coverage or challenger work, use the target declared in
`benchmarks/vendors/registry.json` and the fairness requirements in
`benchmarks/vendors/local-gpu-challenger-matrix.json`. Preserve identical sampling,
seed, workload, budget, cache policy, and timing scope across compared rows.

Save raw JSON before generating matrices or summaries. Treat load, TTFT, prefill,
decode, correctness, and resource telemetry as separate measurements.

## Validation

The saved receipt must identify the exact artifacts, runtime surfaces, hardware,
fallback state, work counts, samples, correctness result, and output path. A paired
claim is valid only when its receipt's applicable fairness and claim gates pass.

## Stop Conditions

Stop comparison when identities, work, sampling, cache state, correctness policy, or
timing scope differ. Stop before remote, paid, or hardware-intensive runs without the
required authorization. Do not turn diagnostic or cross-platform rows into claims.

## Outputs

- Raw benchmark JSON receipt.
- Optional generated matrix or scoped summary derived from that receipt.

## Side Effects

Runs model workloads and writes benchmark artifacts. Registry, support-matrix, claim,
documentation, publication, and deployment changes require a separate request.

## Related Skills

- `doppler-perf` for diagnosing a measured bottleneck.
- `doppler-debug` for correctness failures.
- `doppler-convert` for artifact-format failures.
