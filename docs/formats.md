# DOPPLER Formats

This page is the format-spec index.

## Canonical specs

- Runtime model artifacts (RDRR): [rdrr-format.md](rdrr-format.md)
- Adapter manifests (LoRA): [lora-format.md](lora-format.md)

## Conversion and runtime ownership

Use [conversion-runtime-contract.md](conversion-runtime-contract.md) for
conversion-static vs runtime-overridable fields.

## CLI usage

Use [getting-started.md](getting-started.md) for first-run conversion + verify.
Use [cli-quickstart.md](cli-quickstart.md) for command-shape examples.

## Implementation references

- `src/formats/rdrr/*`
- `src/adapters/*`
- `src/converter/*`

## Kernel override note

For override and compatibility policy, use the canonical section in
[operations.md#kernel-overrides--compatibility](operations.md#kernel-overrides--compatibility).
