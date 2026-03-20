# Model Family Configs Are Removed

## Goal

Do not add a model-family registry. Doppler no longer has registry-style family indirection or loader-driven family detection.

## When To Use This Guide

- You were looking for the old family-registry flow.
- You need to know what replaces it.

## Blast Radius

- None. This is a migration note.

## Required Touch Points

- `src/config/conversion/<family>/<model>.json`
- `docs/conversion-runtime-contract.md`
- Any tests or docs that still mention family-registry-driven behavior

## Recommended Order

1. Author the model family behavior directly in the conversion config.
2. Stamp all manifest-owned inference fields explicitly from that config.
3. Use runtime profiles only for runtime-owned overlays after conversion.
4. Remove any lingering loader or family-registry references from tests and docs in the same change.

## Verification

- `npm run onboarding:check:strict`
- Run one real convert/debug pass with the new conversion config

## Common Misses

- Reintroducing loader-driven family detection.
- Leaving required inference fields implicit in conversion output.
- Treating runtime profiles as a substitute for conversion-owned manifest fields.

## Related Guides

- [02-assign-chat-template.md](02-assign-chat-template.md)
- [04-conversion-config.md](04-conversion-config.md)
- [06-kernel-path-config.md](06-kernel-path-config.md)
- [07-manifest-runtime-field.md](07-manifest-runtime-field.md)

## Canonical References

- `src/config/conversion/`
- [../style/general-style-guide.md](../style/general-style-guide.md)
- [../config.md](../config.md)
