# Config Source Of Truth

## Goal

Keep conversion, runtime, harness, and kernel configuration from defining the
same behavior in multiple places.

## Ownership Map

| Concern | Canonical Owner |
| --- | --- |
| model dimensions and layer count | conversion config stamped into manifest |
| layer pattern and attention layout | conversion config stamped into manifest |
| activation function and FFN shape | conversion config stamped into manifest |
| quantized storage layout | conversion config and converted shards |
| weight/shard identity | manifest plus shard digests |
| WGSL file, entry point, and digest | conversion `inference.execution` and manifest execution graph |
| runtime session dtype and decode-loop policy | manifest session baseline plus `runtime.inference.session` |
| runtime capability adaptation | execution graph transforms gated by `kernelPathPolicy` |
| kernel variant reachability | `src/config/kernels/registry.json` and rule maps |
| runtime profile metadata | `src/config/runtime/**` profile wrapper fields |
| benchmark fairness knobs | benchmark shared config |
| engine-specific benchmark knobs | benchmark engine overlay |
| command workload and intent | command request and command API normalizers |

## Rules

- Conversion config is the source for model-owned inference behavior. Runtime
  config may not rediscover or rewrite it.
- Runtime profiles tune execution policy only. They must not carry string
  `runtime.inference.kernelPath` registry IDs.
- `runtime.inference.kernelPath` is either `null` or an inline object generated
  from execution-v1. String registry IDs are legacy and must fail fast.
- `src/config/kernels/registry.json` is the operation-kernel registry used by
  kernel wrappers and reachability tooling. It is not a model kernel-path
  registry.
- Non-profile policy assets under `src/config/runtime/**` must declare their own
  schema and be explicitly allowlisted by checks.
- Mirrors are allowed only when generated or check-synced from the canonical
  owner.

## Program Bundle Export Implication

The Doppler Program Bundle references the manifest execution graph, WGSL
digests, runtime/capture profile, and artifact identities from these owners.
It must not invent a Doe-specific duplicate list of model behavior, kernels, or
weight layout. See `docs/integration/program-bundle.md`.
