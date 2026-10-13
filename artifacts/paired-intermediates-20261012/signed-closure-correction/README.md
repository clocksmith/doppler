# Signed paired shader closure correction

Component: paired execution declaration and Run/Rig admission. Intent: preserve the approved model equations and storage policy. Boundary effect: reject an incomplete paired shader closure before execution or signing.

The first physical signed generation rejected `gather_f16_vec4.wgsl` before producing a token. Reploid commit `07b158de` retains the actual native failure, settled attempts and read-only observation. The existing program-bundle collector correctly included only declared semantic and mechanism kernels; it was not responsible for guessing wrapper dispatches.

`observed-pipeline-inputs.json` retains the pipeline observations and raw-capture hashes from the earlier two-direction 0fad full-55 run. `audit.js` maps their exact pipeline labels and entries through the registry: 15 distinct source/entry pairs were observed, and the prior declaration omitted `gather_f16_vec4.wgsl#gather_vec4` and `logit_finalize_f32.wgsl#main`. This comparison is retrospective source coverage, not qualification of the new graph.

The new `execution.json` adds those two helpers and the existing `gather_split8_f16_vec4_f32_out.wgsl` fallback. The scalar gather remains a semantic declaration. Scalar and split fallbacks are deliberately admitted source paths; these observations do not establish that they executed. Existing semantic steps, kernel declarations, shader bytes, weights, F16 cache storage and approved F64 comparison remain unchanged.

The paired admission guard and Rig specialization now share `assertPairedExecutionMechanismClosure`. Selected gather dependencies are declared in registry metadata; exact file, entry, specialization constants and digest syntax are required. Program-bundle construction independently verifies each declared digest against actual source bytes. Generic scalar admission is unchanged. `admission-before.log` retains the failing regression before the guard repair.

CPU validation passed six focused files: paired execution plan, forge stages, generic mechanism closure, paired source-scope bundle, program-bundle exporter and paired model binding. New controls remove each helper, change its entry or its digest, and verify undeclared source remains rejected. Source types, style, architecture, generated registry and dependency checks pass.

`verify-frozen-runtime.js` verifies archive SHA256 `0fadac5113d2cc154f65d9d40de50722a16454995e927c7d22bedc5f1a4925cd`, builds the new closure from that exact installed package, compares every declared shader to canonical bytes and compiles a paired plan using its actual APIs. The CPU plan uses the previously observed load policy and a synthetic generation limit of eight solely for plan admission; it generates nothing. `frozen-runtime-verification.json` records the 27 declarations. No runtime repack is required to execute these already-present shaders.

## Qualification and signing handoff

Corrected graph SHA256: `43a7ce21f1305c8901581bcae5eb15e0ebfcd2d1c4de760624459b65ab4700e6`.

1. Preserve the previous model and input trees. Copy the corrected graph into a new manifest and create a new paired manifest/graph binding. Retain identical weights, tokenizer, source facts, generation policy, approved reference, `0.001` tolerance and 1.42-billion-byte ceiling.
2. Copy the existing `/var/tmp/reploid-readiness-numerical-20261012` fixture to a new immutable directory. Regenerate delivery/config hashes and pre-output producer expectations against the new manifest/graph. Do not run its old launcher against the new graph with historical bindings.
3. After an explicit Mac/AMD reservation, use the same `tests/fixtures/native-partition-process.js` plus `tests/fixtures/browser-partition-memory-completion.js` reference phase: `REPLOID_MEMORY_PHASE=reference`, unchanged reference SHA256 `ad26dfbc38c5b890f48da2f1e4313dac9295e39e12f688411a10bf614d22fdd6`, no prompt/step selector, and `REPLOID_REFERENCE_REVERSE=0` then `1`. The new launcher must name only new model, delivery, output and binding paths. Retain actual initial execution identities and confirmed cleanup.
4. Run the existing `artifacts/signed-resident-producer-20261012/produce-physical-reports.js` with new pre-output expectations/configs and actual new captures. Do not relabel historical reports. This covers Node AMD and Mac browser in both partition roles, not browser AMD or Intel portability.
5. Rebuild ProgramBundle, program artifact closure, graph-bound initial identity, TargetPlans, Capsule signature/release and application catalog using the genuine new reports. Piece bytes may remain identical, but all graph-dependent identities must be recomputed. Then repeat actual signed opening, acquisition and conversation through the existing application. Preserve the current isolated consent, identity, cache and persistent 4-GiB sharing ledger unchanged.

No new physical run, signature or release claim is made by this CPU correction. Long concurrent 1,588-token acceptance and source-answer quality remain separate unresolved gates.
