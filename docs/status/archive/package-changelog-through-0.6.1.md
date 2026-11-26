# Archived doppler-gpu unreleased notes: Capsule naming (0.6.0)

> Archived history of the former package doppler-gpu; version numbers here are doppler-gpu versions, not dopplerx.

Despite its file name, this snapshot holds no release entries for 0.4.16 to
0.6.1. It keeps only the `doppler-gpu` [Unreleased] notes that described the
0.6.0 Capsule rename. The complete history through 0.4.15 is in
[package-changelog-through-0.4.15.md](package-changelog-through-0.4.15.md); both
are kept outside the npm tarball so historical release prose does not consume
runtime-package budget.

## [Unreleased]

### Breaking: Capsule naming (0.6.0)

- Replace the Pack product name with Capsule throughout the public API,
  declarations, module paths, configuration, CLI, Forge, and runtime.
  Use `openCapsule()`, `doppler-gpu/capsule`, and `Capsule*` types.
  There are no former-name aliases or old-schema readers in the Capsule runtime.
- Rename identity/receipt fields and signed namespaces. Rebuild and sign new
  Capsules; editing an old signed document is not a valid migration.
- Keep model bytes, tokenizer vocabularies, and retained observations unchanged.
  Historical records are not fresh Capsule qualification.
- See [docs/capsule-naming-migration.md](../../capsule-naming-migration.md) for
  consumer and artifact migration.

The Added and Changed notes of this unreleased block (Doe provider-v1
acquisition, Program Bundles, Node WebGPU delegation) are identical to the
[Unreleased] block of the
[changelog through 0.4.15](package-changelog-through-0.4.15.md#unreleased).

## Earlier releases

The [changelog through 0.4.15](package-changelog-through-0.4.15.md) owns the
0.4.15 release entries and earlier history.
