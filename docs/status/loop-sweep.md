# Cleanup review

The [April 2026 sweep log](archive/loop-sweep-2026-04.md) is retained unchanged
as historical evidence. Its test counts, dirty-file observations, skip lists,
and claims of unused code describe those runs, not the current checkout.

## Current review (2026-10-15)

- Removed 283 TypeScript-confirmed unused import bindings from the CLI,
  artifact identity, text generator, and generator text helpers. Function
  bodies and exports are unchanged; declared device initialization is retained.
- `npm run source:unused:check` checks the four implementations listed in
  `tsconfig.source-unused.json`. Expand that list as modules are cleaned.
- The kernel-path builder has no repository importer or package export, and
  the declared computed-dependency policy names no loader for it. However,
  `tools/policies/source-architecture-policy.json` explicitly retains
  `tooling/kernel-path-builder/index.js` as an operator-loaded standalone
  entrypoint. It remains intact pending review of that retained capability.
  The sibling `custom-runtime-facts.js` has an active test consumer; the demo
  also retains generated builder JSON. A directory-wide deletion is unsafe.
- The Gemma 4 E2B performance note now links to its exact post-fix receipt
  in Git history rather than an absent working-tree file.

## Remaining historical leads

These are audit leads, not reproduced failures or authorization to delete.
The archive retains the original per-file findings and caveats.

- Recheck generated WGSL patch freshness before removing entrypoints such as
  `precompute_freqs`; repair source and generated variants together.
- Audit repeated dequantization helpers and Q4 constants with variant/rule
  reachability, not filename searches or historical "unused" labels alone.
- Recheck browser dependency closure where the old log reported Node imports.
  Current check output must establish whether a violation remains.
- Review exported symbols together with declaration consumers, package APIs,
  computed loading, and operator entrypoints. `typeof` references can require
  a value export even with no JavaScript caller.
- Audit the reported `matchesExactObject` behavior differences before merging
  duplicate rule helpers; check callers before consolidating CLI/client helpers.
- Check the dependency and peer-dependency graph before removing
  `onnxruntime-web` or other apparently unused packages.

Past test-failure totals remain in the archive. Reproduce a named failure on
current source before treating it as an open defect.
