# Plan-bound GPU token selection

A TargetPlan v2 can explicitly adopt `doppler.capsule-token-selection/v1`.
The existing `openCapsule()` session and `executeOperation()` generation path
then keep complete logits on the GPU, apply history penalties and suppression,
select a token, and copy four bytes into a mapped staging buffer. Sampling and
that copy share one command submission. The final result still contains full
text, token IDs, resolved settings, stopping reason, and verified identity.

## Declare and qualify the recipe

Supply this field in the Forge JSON configuration. The module IDs refer to the
model's declared execution closure; they are not independent shader downloads.

```json
{
  "tokenSelection": {
    "schema": "doppler.capsule-token-selection/v1",
    "generationContract": "doppler.generation-contract/v1",
    "logitsDtype": "f32",
    "kernelModules": ["sample", "rep_penalty", "logit_suppress"]
  }
}
```

The [recipe contract](../../src/config/capsule-token-selection.json) identifies
required variants through the existing kernel registry. Declare their exact
shader bytes and digests in the manifest execution graph, using its existing
`mechanismKernels` list for auxiliary kernels. The observed initial execution
identity must include the same complete closure. Forge preserves that identity
check; adding the field cannot waive missing modules or qualification evidence.
This recipe requires f32 output logits and program-backed prefill/decode phases.
Other precision lanes require their own explicitly supported recipe.

Build a new Capsule and retain the archive, manifest, shader closure, observed
identity, source comparison, and installed-consumer results. Its TargetPlan hash
and Capsule semantic root change. Applications explicitly accept the new plan.
Never edit a signed Capsule, repin a historical manifest, or infer adoption from
the npm version. Transport v1/v2 remains a separate choice.

## Sampling and lifecycle

The request uses the existing generation contract: temperature zero is greedy;
positive temperature uses the supplied seed, top-k and top-p, after repetition
and presence penalties, followed by suppression. Padding and non-finite logits
cannot win selection. Ties retain token-index ordering. Empty candidate sets
fail explicitly. The implementation does not cap top-k or silently select CPU
sampling after a GPU failure.

Cancellation is checked after asynchronous preparation, immediately before
submission, and after readback. Cancellation before sampler submission prevents
that submission. Cancellation after submission suppresses the result; it does
not interrupt GPU work already submitted. Staging buffers, sampler output, and
operation-owned logits are released on failure and completion. Session-owned
weights and attention state retain their existing owner.

The low-level `advanced.prefillWithLogits()` and `advanced.decodeStepLogits()`
interfaces remain available for reference and diagnostic score access. Plans
without this declaration preserve their existing score-returning behavior.
There is one inference implementation underneath both choices.

## Acceptance boundaries

- Contract and installed-package fixtures test selection routing, v1/v2 final
  reconstruction, completion identity, cancellation, and Reploid consumption.
  Injected selected tokens are not physical execution evidence.
- Physical operator tests compare GPU selection against the existing CPU
  reference, including finite extremes, ties, suppression, penalties, seed and
  filtering settings, and vocabulary sizes through 151,936.
- Physical model receipts establish exact output equivalence for their recorded
  model, request, runtime archive, shader bytes, and device only.
- Readback bytes count API copy sizes, not measured bus throughput. Fewer copied
  bytes do not by themselves establish lower token latency. Compare prefill,
  decode, submissions, waits, CPU cost, and application latency separately.

Use `tools/probe-installed-token-selection.js` for unsigned preparation against
frozen references, `tools/build-token-selection-evaluation-capsule.js` to package
that local evidence with the existing Forge, and the installed Capsule profiling
and capability tools for public API acceptance. These tools neither publish nor
deploy a release.
