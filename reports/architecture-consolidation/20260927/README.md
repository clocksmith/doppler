# Scoped execution dependencies

Component: `doppler.runtime-source` (rules, configuration, model host, numerical pipeline adapters, GPU kernels, diagnostics), `doppler.repository-tooling`, `doppler.tests`, `doppler.docs`.
Intent: changed — extensions now bind to immutable instances and accepted execution identities; debug global installation requires application opt-in. Rig → Capsule → Run authority is preserved.
Acceptance evidence: retained source-gate, installed-package, unit, focused-contract and physical-activation logs in this directory; see `acceptance.json` for exact commands, source identities and limitations.
Boundary effects: existing model-host/program-factory ports carry registries and observers; the serialized numerical adapter activates legacy dependencies. No plugin manager, second graph format, runtime shader assembly, or production configuration switches were introduced.

## Changes

Rule registration replaces the compatibility default for future construction. Existing instances retain deeply frozen rule data. Kernel construction freezes complete resolved metadata and retains separately identified validator functions. Validators are trusted JavaScript: their captured mutable state cannot be isolated or authenticated by freezing a function. Extensions must use pure implementations and immutable implementation IDs.

Construction reads compatibility defaults independently of another instance's active execution lease. Programs retain their registry pair, custom pairs must match the accepted TargetPlan identity, and existing signed canonical plans remain unchanged. Registry identities distinguish kernel-path and compiled-pipeline caches. Legacy GPU execution remains serialized, including streamed operations and shutdown; this work does not establish concurrent GPU safety.

Reusable diagnostics no longer install a global. Applications explicitly call `installDebugGlobal()`, and separate pipeline observers receive scoped log/trace events. The demo opts in. Existing host storage remains separately injected.

GeLU and ReLU share output rollback through the existing executor. Their semantic shape calculations and thin run/record adapters remain in JavaScript. Generated uniform writers retain layout authority; extended layouts must be generated before packaging. Failed recorded outputs stay retained until recorder cleanup; borrowed outputs remain caller-owned. Abort checks cover preparation and asynchronous compilation.

Advanced tooling exports expose the constructors and explicit installation through existing adapters. Internal host imports now use their implementation owners directly; public compatibility facades remain available. Generated API, package closure, runtime closure and demo shell inventories were synchronized without raising architecture debt or demo budgets.

## Evidence and limits

The installed package test imports all public exports and browser-conditioned tooling, checks absence of an automatic debug global, exercises separate registry instances and immutable metadata, and verifies explicit installation/restoration. Its other program/host tests use synthetic executors and do not qualify model inference.

Physical AMD/Vulkan tests exercise GeLU/ReLU immediate-versus-recorded results, an independent GeLU numerical fixture and ReLU reference, borrowed/owned output failure cleanup, recorded retention and repeated abort, and cancellation before allocation and during compilation. These are operator correctness tests, not performance measurements or browser/model qualification.

The browser contract reaches its loaded-model/generation journey and verifies the explicit debug API, then fails at an existing control selector absent from the unchanged demo HTML. `demo-control-drift.json` pins the baseline inspection; `demo-contract.log` retains the executed failure. Browser acceptance remains open at `tests/demo/browser-controls.js`; this batch does not claim `check:green` or release qualification.

Initial failures are retained separately: charter word limits, export-boundary wiring, stale generated closures/declarations, and an incorrect physical-test filename were corrected. Final passing checks and the open browser-control result remain distinct.

No WGSL arithmetic, model lowering, accepted search archive, shader packaging format, model qualification, browser GPU concurrency, performance comparison, or broad architectural reorganization was changed or claimed. Learned input embeddings, complete BERT lowering and MiniLM numerical parity remain the next feature work after this architectural batch.
