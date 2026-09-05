# Source-qualified reranker evaluation

Component: `doppler.runtime-source.client`. This is a bounded internal evaluation
workflow, not an external adopter or a claim of general model equivalence.

## Reproduce the reference and candidate

1. Acquire the files listed by `tools/policies/qwen-reranker-source-reference.json`
   from its exact Hugging Face commit, retaining the snapshot metadata and model
   license. The capture tool checks each file's recorded source revision and
   records byte hashes. Install the declared reference dependencies separately.
2. Run `python tools/capture-reranker-source-reference.py --policy <policy.json> --out <new-reference.json>`.
   This executes the pinned source using CPU float32, retaining inputs, actual
   token IDs, source logits, scores, probabilities, dependency versions, and the
   predeclared acceptance policy. It does not use Doppler outputs as its oracle.
3. Convert the source with `tools/convert-safetensors-node.js` and an explicit
   conversion recipe. Keep candidate output directories separate. Conversion
   owns shader, layout, numerical, and scoring choices; the application cannot
   override them after signing.
4. Run `node tools/qualify-reranker-electron.js <qualification-config.json>`.
   Required fields are `mode: "model"`, `policyPath`, `referencePath`, `modelDir`,
   `packageRoot`, and a new `outputDir`. The policy pins Electron, launch
   arguments, physical adapter requirements, and runtime configuration. The
   supplied Electron/Playwright installation and desktop display are probe
   dependencies, not files in the Doppler runtime package.

The qualification report compares all documents in input order and recomputes
numerical and exact-rank checks. Neither changing a `passed` bit nor borrowing
generation evidence makes a rejected candidate eligible. Three documents are
a bounded parity test, not held-out search quality or an incumbent comparison.

## Build and execute a signed evaluation Pack

`node tools/build-reranker-evaluation-pack.js <build-config.json>` takes
`qualificationPath`, `conversionConfigPath`, `licensePath`, `applicationPath`,
`outputDir`, `authorityId`, and an explicit `revocation` policy with
`offlineExpirySeconds` and `failClosedAfterExpiry: true`.

The tool creates a closed Program Bundle, manifest-derived ModelIR v1,
rerank-qualified TargetPlan with observed initial execution identity, signed
Pack, application contract, explicit public-key trust configuration, and build
receipt. It verifies current WGSL hashes and refuses stale pins. It does not
claim source-fact ModelIR v2 construction or arbitrary semantic lowering.

Private evaluation keys are in `custody/` with restricted filesystem permissions,
outside `distribution/`. Serve only `distribution/`; never serve the build root.
They are locally generated evaluation authorities, not trusted package keys or
external operational authority. Keep or replace the authority deliberately.
`distribution/MODEL_LICENSE.txt` retains the model license as a sidecar whose
bytes match the signed `release.source.license.textDigest`; it is not counted
as an executable artifact in the Pack inventory.

Build an installable runtime with
`node tools/check-packed-package.js --retain <new-package-bundle>`.
Use its `consumer/node_modules/doppler-gpu` directory as `packageRoot` for a new
qualification config with `mode: "pack"`. Supply `packPath`, `application`,
`packageBundlePath` identifying that retained bundle,
`authorizedPack: { packId, semanticRoot }`, and `openOptions` containing explicit
`trustedSigners` and `acceptedTargetPlanDigests`. These values come from the
local application's reviewed build contract, not an arbitrary downloaded Pack.

Pack mode invokes the actual Electron renderer adapter and integrated host,
checks the frozen source outputs again, and records adapter-owned cleanup.
It blocks the original model route and external network origins. Its release
resolver is a pinned evaluation fixture, not production IPC authorization.
Application-controlled updates and revocation still require the application
coordinator. Offline operation cannot discover unseen revocation events.

For fault probes, use a new output directory and an explicit `fault` object.
`artifact-corruption` and `artifact-interruption` require the `artifactId` of
a weight shard declared in this Pack. Corruption changes only the served copy;
retained source bytes remain intact. `device-loss` destroys the evaluator's
WebGPU device after Pack opening. Fault runs remain failed qualification
reports, even when that rejection is the expected result. A separate unmodified
run must demonstrate recovery. These are fresh-process recovery probes, not
resumable range-transfer or production fleet reliability claims.

## Evidence boundaries

Keep raw rejected reports, failed builds, exact package bytes, source provenance,
conversion recipes, references, Packs, and qualifications together. Do not
rewrite a report after changing its manifest, scoring policy, or acceptance
threshold. A changed candidate needs new qualification.

- Unit/fixture tests establish contracts with synthetic execution.
- Source comparison establishes only the declared model, inputs, and tolerances.
- Pack execution establishes the observed physical host and signed closure.
- Installation, cold/warm application behavior, recovery, and incumbent
  performance require their own measurements.
- Voluntary independent use and retention through another release remain
  external evidence; free use counts fully. None is inferred from these tools.

## Retained engineering checkpoint

The [2026-09-05 evidence index](../../reports/pack-runtime/reranker-electron-20260905.json)
binds the clean installed package, physical Electron observation, source
comparison, rejected candidates, fault observations, and local handoff archive
by byte hashes. It is an immutable experiment snapshot, not the current support
registry. The archive and raw observations are retained locally; cloning this
repository does not download them. The index states that availability explicitly.

The [maintenance follow-up](../../reports/pack-runtime/reranker-maintenance-20260905.json)
retains the audit repairs, expanded repository checks, and a newly installed
runtime rerunning the same F16 Pack on the physical Electron host. Its separate
local supplement contains package bytes and raw observations; reuse the preceding
archive for model artifacts and source references. Neither snapshot replaces the
other or establishes external adoption. The follow-up index binds both archives
and records the unchanged rejected Q4K candidate.

The passing candidate is F16 with the declared `true_logit` scoring contract.
The original Q4K comparison remains rejected; its missing immutable source pin
also prevents attributing the difference solely to quantization. No acceptance
tolerance was relaxed and no model catalog promotion was performed.

To evaluate the retained archive without rebuilding Doppler, extract it into a
separate directory, verify its hash and per-file manifest, and install its
`doppler-gpu-0.5.1.tgz` with npm into the package bundle's `consumer/` directory.
Retain `receipt.json` and `source-state.json` alongside the tarball. In a new
Pack qualification config, point `modelDir` at the retained Pack's
`distribution/artifacts/model/` directory, use the installed package as
`packageRoot`, and select a new output directory. Review the supplied public
signer and TargetPlan pins before using them; the archive grants no trust.
The evaluator still requires the declared Electron/Playwright installation and
a physical desktop. CPU reference reproduction additionally requires the
recorded Python, Torch, and Transformers versions.
