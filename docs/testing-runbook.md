# Testing Runbook

Canonical guide for running Doppler tests.

## Quick reference

| Action | How | When |
| --- | --- | --- |
| Kernel correctness | `tests/harness.html` mode `kernels` | after kernel changes |
| Inference smoke | `tests/harness.html` mode `inference` | after pipeline changes |
| Training harness | `tests/harness.html` mode `training` | training work |
| Coverage gate | `npm run test:coverage` | before merge |

## Harness modes

- `kernels`
- `inference`
- `bench`
- `training`
- `energy`

## Example runtime config (kernels)

```json
{
  "shared": {
    "tooling": { "intent": "verify" },
    "harness": {
      "mode": "kernels",
      "autorun": true
    }
  }
}
```

## Example runtime config (inference)

```json
{
  "shared": {
    "tooling": { "intent": "verify" },
    "harness": {
      "mode": "inference",
      "autorun": true,
      "skipLoad": false,
      "modelId": "gemma3-1b-q4"
    }
  },
  "inference": {
    "prompt": "Hello from Doppler."
  }
}
```

## CI notes

- Browser automation is still local/manual in this repo.
- Node coverage policy is defined in `tools/policies/test-coverage-policy.json`.

## Related

- Test harness details: [../tests/README.md](../tests/README.md)
- Kernel coverage details: [../tests/kernels/README.md](../tests/kernels/README.md)
- Kernel benchmark baselines: [../tests/kernels/benchmarks.md](../tests/kernels/benchmarks.md)
- Kernel test design guidance: [kernel-testing-design.md](kernel-testing-design.md)
- Kernel override policy (canonical): [operations.md#kernel-overrides--compatibility](operations.md#kernel-overrides--compatibility)
