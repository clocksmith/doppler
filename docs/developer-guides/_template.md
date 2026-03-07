# Developer Guide Template

Use this template for files in `docs/developer-guides/`.

## Goal

One sentence describing the extension point.

## When To Use This Guide

- Scope boundaries
- What this guide does not cover

## Blast Radius

- JSON only / schema + runtime / WGSL + runtime / cross-surface / full vertical slice

## Required Touch Points

- Config/schema files
- Runtime/pipeline files
- Tooling/docs files
- Tests

## Recommended Order

1. Touch the source-of-truth files first.
2. Wire runtime or API consumers next.
3. Add or update tests before verification.
4. Sync docs or generated artifacts last if required.

## Verification

- Required commands
- Browser vs Node checks when relevant
- Human-review steps when relevant

## Common Misses

- Known failure patterns
- Generated artifacts or docs that must be synced
- Surface-parity or browser-only gotchas

## Related Guides

- Atomic guides this depends on
- Composite guides that include this work

## Canonical References

- Style guides
- Runtime/config contracts
- Source files that act as the registry or source of truth
