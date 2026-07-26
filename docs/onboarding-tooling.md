# Doppler Onboarding Tooling

This page is for onboarding automation (`inspect` + `check` + `scaffold`).

For first-run convert/verify/bench workflow, use [getting-started.md](getting-started.md).

## Why this exists

New model and config onboarding should be:
- deterministic
- scriptable
- fail-fast on cross-file drift

Implemented by `tools/onboarding-tooling.js`.

## source inspect mode

Before conversion, inspect the source checkpoint:

```bash
doppler onboard inspect --source <checkpoint-dir> --out <artifact-dir>
```

This produces a provenance-bearing `doppler.source-intake/v1` report, a
conservative conversion-config skeleton, and a focused contract-test plan.
Unknown, ambiguous, unsupported, and family-inferred values remain unresolved.
See [evidence-loop.md](evidence-loop.md).

## check mode

```bash
node tools/onboarding-tooling.js check [--root <repo-root>] [--strict] [--json]
```

Validates:
- checked-in config asset shape and references
- runtime profile shape and extends-chain integrity
- conversion config references and output fields
- execution graph/kernel digest integrity and kernel existence
- compare harness coverage and metric contract mapping

Exit behavior:
- errors => exit 1
- `--strict` treats warnings as failures

## scaffold mode

```bash
node tools/onboarding-tooling.js scaffold --kind <conversion|kernel|behavior> --id <id> [flags]
```

Shared flags:
- `--id <id>` (required)
- `--output <path>`
- `--force`

Kinds:
- `conversion`: create conversion config stub
- `kernel`: scaffold execution graph transform/config work
- `behavior`: create runtime profile stub

## Canonical operational sequence

1. Run `doppler onboard inspect` against source material.
2. Resolve the intake report's ambiguous and unsupported facts.
3. Run `check`.
4. Scaffold missing assets.
5. Run `check` again in `--strict` mode.
6. Execute workflow from [getting-started.md](getting-started.md).

## Related

- Conversion config details: [../src/config/conversion/GUIDE.md](../src/config/conversion/GUIDE.md)
- Support matrix generation: [model-support-matrix.md](model-support-matrix.md)
- Hosted publish and registry checks: [registry-workflow.md](registry-workflow.md)
