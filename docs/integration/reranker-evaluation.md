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

## Locate a numerical divergence

In model mode, the same qualifier accepts an optional `diagnosticCapture`:

```json
{
  "diagnosticCapture": {
    "documentIndex": 0,
    "captureConfig": {
      "defaultLevel": "none",
      "targetLevel": "full",
      "targetOpIds": ["layer.0.attn.post_input_norm", "layer.0.attn.q_proj"]
    }
  }
}
```

The document comes from the frozen reference. The probe resets generation state,
uses the public selected-token prefill method, and requires its tokens and logits
to match ordinary reranking exactly. Captures and their separate elapsed time are
retained under `raw.diagnostic`; they are not a performance sample or Pack proof.
Failed diagnostic comparisons retain their raw observations. Capture configuration
is rejected in Pack mode rather than bypassing the signed execution interface.

`tools/q4k-projection-oracle.js --help` describes comparison of these receipts
against independent scalar Q4K and F16 projection references. Reranker receipts
must match ordinary execution and the supplied manifest bytes. A candidate can
still fail source acceptance while its projection matches the quantized-weight
oracle; those are different checks. Preserve both outcomes and change numerical
implementations only through a new Forge candidate.

## Build and execute a signed evaluation Pack

`node tools/build-reranker-evaluation-pack.js <build-config.json>` takes
`qualificationPath`, `conversionConfigPath`, `licensePath`, `applicationPath`,
`outputDir`, `authorityId`, and an explicit `revocation` policy with
`offlineExpirySeconds` and `failClosedAfterExpiry: true`.

The tool creates a closed Program Bundle, rerank-qualified TargetPlan with
observed initial execution identity, signed Pack, application contract, explicit
public-key trust configuration, and build receipt. It verifies current WGSL
hashes and refuses stale pins. Before signing, every model shader requested by
the qualification must occur in the declared bundle. Only the kernel registry's
`runtime_probe` variants are classified separately as device probes. Missing
request evidence or an undeclared observed shader rejects the build; the tool
never adds kernels to a sealed Pack. This covers the observed execution path,
not untested shapes or devices. The build receipt retains this inventory.
Without `modelIRReceiptPath`, it uses
manifest-derived ModelIR v1. Supplying that field packages the source-fact
ModelIR v2 receipt and qualifies only its lowered rerank entry point against
the operation's source comparison. Generation parity cannot qualify reranking.

For the Qwen source-fact path, use
`reports/model-ir-v2/qwen3-reranker.spec.json` with
`node tools/forge-source-truth-model-ir-v2.js --spec <spec.json> --out <new-receipt.json>`.
The recipe pins original config bytes, SafeTensors header bytes, reviewed
reference-implementation semantics, and the independent scoring reference.
Acquire the named source snapshot before running it. JSON and SafeTensors-header
source descriptors require explicit byte hashes; header reads are bounded and
exclude weight payloads. Legacy string sources retain canonical-JSON hashing.
The tool refuses to overwrite an existing receipt.

This is a bounded, authored Qwen topology mapping with mechanically checked
source facts, not automatic translation of arbitrary model code or proof of
universal semantic equivalence. Generation remains unqualified. Tensor bytes
and tokenizer identity are additionally bound by conversion and the Pack's
artifact closure.

The maintained candidate recipe is
`src/config/conversion/qwen3/qwen-3-reranker-0-6b-q4k-ehf16-af32.json`.
`reports/model-ir-v2/qwen3-reranker-q4k-local-grid.conversion.json` is a
historical recipe retained with its original observations, not the current
closed-runtime recipe.
Before constructing a new candidate, use
`node tools/sync-conversion-kernel-digests.js --check --file <candidate-recipe.json>`.
If source kernels intentionally changed, synchronize that new recipe by omitting
`--check`, then reconvert and requalify it. Do not synchronize a retained Pack
or historical manifest in place. Quantizer changes likewise create new bytes;
they cannot improve an already pinned artifact retroactively.

Execution-step kernels alone are insufficient. The recipe's explicit
`execution.mechanismKernels` also binds weight dequantization, RoPE preparation,
vectorized gathering/residuals, Q/K normalization and rotation, KV writes, and
selected-logit readback. Forge includes those source bytes in the Program Bundle
and initial execution identity. Runtime rejects every model shader outside that
verified scope, including during weight loading. Adding a missing mechanism
therefore requires a new manifest, initial-identity observation, and signed Pack;
the runtime must not fetch an undeclared shader from the package as a fallback.
`tests/tooling/reranker-kernel-closure.test.js` exercises the maintained recipe's
real WGSL source closure and retains the missing-dequantization rejection.

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
It blocks the original model route and external network origins. Without
`releaseCoordinator`, its release resolver is a pinned evaluation fixture,
not IPC authorization. Offline operation cannot discover unseen revocation events.

To exercise the installed application coordinator through actual Electron IPC,
add `releaseCoordinator` with an absolute private `statePath`, explicit
`trustedSigners`, ISO `now`, main-process initialization `actions`, and
`allowedRendererActions` (only `status` and `resolve-current` are permitted in
this probe). Initialization uses the existing IPC request shapes, including
signed activation decisions and signed revocation snapshots. Activation requires
a digest of a retained application authorization record. An internal evaluation
record must identify itself as such; it cannot assert customer or fleet adoption.

The qualifier checks the installed `main.js`, `preload.js`, and
`release-storage.js` against the retained package-consumer inventory. It compiles
that preload to sandbox-compatible CommonJS, retaining the generated bytes,
source digest, and TypeScript version. The only substituted import is the
channel constant read from the installed public `doppler-gpu/electron` export.
The main fixture authorizes the actual window, top-level frame, local origin,
and read-only action. It rejects a renderer rollback attempt before execution.
Keep Pack reference paths origin-relative (for example `/pack/pack.json`) so
durable release state survives a fresh probe server port.

Retained `coordinator.json` contains before/after state, initialization outcomes,
and actual IPC authorization observations. The qualification report embeds it.
Restart without initialization actions to exercise retained decisions; signed
revocation updates, removed signer trust, stale histories, and expiry must still
block opening. Use new output directories while retaining the same state file.
This is a reference main-process integration on the declared physical host,
not production key custody, updater deployment, external adoption, or proof that
arbitrary application IPC policies are secure.

For Pack v3, retain the complete signed `releaseEvents`, independently selected
`releaseTrustedSigners`, and `releasePolicy: { now, minimumSequence }` in
`openOptions`. Set `releaseCheckpointPath` to an absolute file in a private
application-owned directory outside the served roots. The evaluator loads the
hash-checked `release-storage.js` example from the retained package consumer,
verifies history against that file, and exposes only its checkpoint callback to
the controlled renderer. The runtime must persist the verified checkpoint before
creating the model. A caller-supplied checkpoint in JSON is rejected. Reuse the
same file across fresh process runs to test stale history rejection; never erase
it to make a replay pass. Before/after records, persistence failures, and the
explicit evaluation clock are retained even when qualification fails.

This evaluates durable local state on the declared filesystem, not trustworthy
wall-clock provision, rollback-resistant storage against filesystem snapshots,
production IPC authorization, or external adoption. Migrating unchanged model
artifacts from v2 to v3 and advancing eligibility events are lifecycle tests,
not a second model release or a performance improvement. Expiry and revocation
remain fail-closed under the selected release policy.
Rejected history is retained in the experiment report, not automatically advanced
into the successful-eligibility checkpoint. The application's revocation state
must prevent subsequent omission of known blocked events. The optional
`releaseCoordinator` lane exercises the example coordinator's cumulative signed
revocation snapshots separately from the successful-eligibility checkpoint.
The latter callback is still supplied through the controlled probe bridge;
that callback is not evidence of a deployed production IPC persistence service.

Every session invocation requires qualification for that operation on the
selected host surface. Merely implementing a method in the program adapter
does not authorize it. Legacy generation records continue to qualify generation
only; they cannot authorize reranking or sequence encoding.
Electron propagates both opening and per-request abort signals to inference.
Cancellation is observed before dispatch boundaries and between documents;
it cannot preempt work already submitted to the GPU.

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

The [quantization follow-up](../../reports/pack-runtime/reranker-quantization-20260905.json)
binds the selected-token capture repair, independent projection comparisons,
converter regressions and fixes, fresh source-pinned Q4K candidates, and another
physical run of the F16 Pack. Rejected candidates and failed diagnostic setups
remain retained; a converter improvement does not by itself establish model quality.

In these checkpoints, the passing candidate is F16 with the declared `true_logit` scoring contract.
The original historical Q4K comparison remains rejected; its missing immutable source pin
also prevents attributing the difference solely to quantization. No acceptance
tolerance was relaxed and no model catalog promotion was performed.

The subsequent source-fact Q4K work uses a distinct candidate and the unchanged
frozen reference. Its local-grid quantizer refines six-bit subblock scale/minimum
codes against the actually stored half-precision multipliers before choosing
four-bit value codes. This changes conversion bytes, not runtime policy. The
checked-in source-fact recipe above records 310 tensor headers and a reviewed
Qwen topology; only reranking is lowered and qualified by its Pack build.
See the separate `reranker-source-pack-20260905.json` checkpoint under
`reports/pack-runtime/` for the exact package, physical observations, and failed
builds. These later results do not alter any preceding report or promote the
model catalog. Source acceptance remains bounded to one query and three documents.

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
