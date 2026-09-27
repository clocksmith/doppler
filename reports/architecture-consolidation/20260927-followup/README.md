# Scoped architecture follow-up

Component: `doppler.runtime-source.inference.pipelines.text`, `doppler.demo`, `doppler.tests`, `doppler.docs`.
Intent: preserved.
Acceptance evidence: `acceptance.json`, executed logs, `package-receipt.json`, and `source-files.json` in this directory.
Boundary effects: internal text consumers import existing KV-cache owners directly; compatibility exports remain available. Demo replay visibility and browser acceptance fixtures change. No execution policy or component authority changes.

## Requirement disposition

The requested architecture implementation is already present in `344c1e51`. This audit starts at `473bb358c3592e5a82ea48485b0a5621a8340491`, which also preserves the subsequent immutable matmul specialization repair. It does not repeat the refactor.

| Requirement | Current implementation and evidence |
| --- | --- |
| Scoped rules and kernels | Constructed registries clone and freeze resolved data; validators live in companion maps. Compatibility registrations replace future defaults. Registry-instance tests cover caller mutation, separate selections, validators, nested immutability and cache identities. |
| Dependency wiring | The composition root, model host and pipeline factory capture registries during construction. TargetPlan identity binds custom execution extensions. Pipeline tests exercise serialized overlapping requests, stream return, cancellation, independent observers and closing one instance while the other continues. |
| Shared kernel execution | GeLU/ReLU retain semantic wrappers and share dispatch plus owned-output rollback. Physical tests exercise immediate/recorded equivalence, borrowed outputs, failed binding, repeated abort and cancellation during compilation. The independent GeLU fixture remains green. |
| Rig and generated contracts | Existing graph/transformation ownership remains unchanged. Registry, rule bundle, shader generation, uniform-layout and digest checks pass. No shader math or graph formats change. |
| Explicit diagnostics | Installed public imports create no debug global. Explicit installation/restoration remains tested. The browser demo explicitly installs the console API and now completes its browser control contract. |
| Contribution and enforcement | Existing kernel and configuration guides state adapter reuse, immutable registries, separate validators, generated layouts and serialized legacy execution. Source gates and installed declarations pass. |

## Corrections in this follow-up

The browser fixture referenced removed disclosure controls, old button semantics and an empty-conversation sample after generation had replaced it. It now exercises the Advanced panel, current checkboxes, declared initial observation policy, and sample after clearing the conversation. X-Ray acceptance checks the actual retained receipt instead of an obsolete section count. Cancellation, profile ownership, failure handling, contrast and responsive-layout assertions remain exercised.

Reaching precision replay exposed a real visibility defect: collapsing its panel also hid its parent, including the only button that could reopen it. Collapse now hides the panel alone. Browser assertions cover the initially visible opener and closing/reopening after a failed evidence fetch.

The current browser closure exceeded its existing module budget after the preceding partition work. Text initialization and attention now import their existing KV-cache owners directly, removing an unnecessary compatibility hop from this closure. Compatibility APIs and cache classes are preserved. Generated shell content, worker cache identity and dependency views are synchronized; budgets are unchanged. The shell remains at its module limit.

## Evidence boundaries

All final commands and exit codes are retained in `acceptance.json`. Browser execution uses the existing mocked model contract; the installed package checks use synthetic executors. Physical activation tests use local Node WebGPU and are separate from browser/model qualification. Source hashes identify what was inspected and tested; the containing commit identifies this completed diff.

The earlier failed browser result is preserved in the original report. This directory retains the reproduced failure and intervening fixture failures rather than rewriting history. An initial focused command used an incorrect test filename; its corrected run is retained separately. The initial dependency-view failure followed fixture line changes; it was fixed by regeneration.

Legacy numerical execution remains serialized. Validator functions are trusted code with immutable implementation IDs; JavaScript cannot freeze their captured external state. This is a contract for pure validators, not a sandbox. No full unit-suite rerun, release qualification, performance comparison, independent adoption, whole-model qualification or new BERT/MiniLM feature work is claimed in this follow-up. Prior full-suite evidence remains scoped to its recorded source.
