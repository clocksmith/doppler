# Cache identity and dispatch cancellation

This repair starts from the reviewed `a5d7a7d2` candidate. Layout caches use normalized descriptors and object identity, shader caches bind actual source bytes, and concurrent pipeline compilation shares a retryable pending task. Device invalidation and cache resets cannot publish stale pipelines.

Direct dispatch checks cancellation and device availability after asynchronous preparation and immediately before queue submission. Errors expose whether submission occurred; completed GPU work is not described as preempted. Injected program phases retain their existing execution owner. The executor reuses the existing device-loss observer.

The new regressions fail against the original modules and pass after repair. Final focused tests and physical direct dispatch/readback are retained. The 804-file unit attempt had nine environment failures caused by `/tmp` quota and one inventory checker failure; focused reruns resolve those failures using `/dev/shm` and a real dependency directory. This is combined evidence, not a single clean full-suite run. The retained failed package attempts exposed the temporary-directory quota and an overbroad device check on injected program phases; the final repair checks device replacement on direct GPU dispatch while preserving the delegated program owner.

Standalone and installed Reploid acceptance use the same Doppler archive. Their signed fixtures inject logits; they do not requalify physical models. Exact package identities, toolchains, source hashes and consumer results are retained. The 0.6.2 archive here is distinct from the previous accepted candidate and was neither published nor deployed.

Component: `doppler.runtime-source.gpu.kernels`, `doppler.runtime-source.client`.
Intent: preserved.
Acceptance evidence: `acceptance.json`, retained test logs and installed-consumer receipts.
Boundary effects: direct dispatch outcomes and repository cache inventory; Reploid source unchanged.
