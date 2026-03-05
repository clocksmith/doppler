# DOPPLER Formats

This page is now an index for format specs.

## Canonical specs

- Runtime model artifacts (RDRR): [rdrr-format.md](rdrr-format.md)
- Adapter manifests (LoRA): [lora-format.md](lora-format.md)

## Conversion and runtime ownership

Use [conversion-runtime-contract.md](conversion-runtime-contract.md) for conversion-static vs runtime-overridable fields.

## CLI usage

### Convert a source model

```bash
node tools/doppler-cli.js convert --config '{
  "request": {
    "inputDir": "/path/to/source",
    "convertPayload": { "converterConfig": { ... } }
  }
}'
```

### Serve and verify

See canonical workflow in [getting-started.md](getting-started.md).

## Implementation references

- `src/formats/rdrr/*`
- `src/adapters/*`
- `src/converter/*`

## Kernel override note

For override and compatibility policy, use the canonical section in [operations.md#kernel-overrides--compatibility](operations.md#kernel-overrides--compatibility).
