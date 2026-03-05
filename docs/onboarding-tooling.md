# Doppler Onboarding Tooling

This page is for onboarding automation (`check` + `scaffold`).

For first-run convert/verify/bench workflow, use [getting-started.md](getting-started.md).

## Why this exists

New model and config onboarding should be:
- deterministic
- scriptable
- fail-fast on cross-file drift

Implemented by `tools/onboarding-tooling.js`.

## check mode

```bash
node tools/onboarding-tooling.js check [--root <repo-root>] [--strict] [--json]
```

Validates:
- model preset shape and references
- runtime preset shape and extends-chain integrity
- conversion config references and output fields
- kernel path registry integrity and kernel existence
- compare harness coverage and metric contract mapping

Exit behavior:
- errors => exit 1
- `--strict` treats warnings as failures

## scaffold mode

```bash
node tools/onboarding-tooling.js scaffold --kind <model|conversion|kernel|behavior> --id <id> [flags]
```

Shared flags:
- `--id <id>` (required)
- `--output <path>`
- `--force`

Kinds:
- `model`: create model preset stub
- `conversion`: create conversion config stub
- `kernel`: create kernel-path preset stub
- `behavior`: create runtime behavior preset stub

## Canonical operational sequence

1. Run `check`.
2. Scaffold missing assets.
3. Run `check` again in `--strict` mode.
4. Execute workflow from [getting-started.md](getting-started.md).

## Related

- Conversion config details: [../tools/configs/conversion/README.md](../tools/configs/conversion/README.md)
- Support matrix generation: [model-support-matrix.md](model-support-matrix.md)
