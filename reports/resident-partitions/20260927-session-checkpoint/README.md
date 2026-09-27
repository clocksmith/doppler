# Resident session implementation checkpoint

Runtime source: `ff8925ea330ca11ea48609564d6d3fea86ebb6ef`. Reploid:
`f5763690bef36177f4ca8746f6e64f9714c437c2`. This checkpoint preserves work
at the user's handoff request; it is not distributed release qualification.

## Implemented and executed

Resident partitions reuse the existing loader, transformer kernels, scalar
sampling and stopping logic. Each attempt owns its KV cache and continuation;
loaded weights remain resident. Source includes a public factory and composition
root wiring, but this wiring has not passed end-to-end acceptance.

[checkpoint.json](checkpoint.json) records source hashes, commands and scoped
limitations. [physical.json](physical.json) contains the actual model, provider,
plan, generation settings, comparisons and owned allocation observations.
The physical Node diagnostic interleaves two conversations and compares every
logit, selected token, decoded text and length stopping against unsplit Doppler.
This same-runtime reference does not establish independent model correctness.
Cancellation coverage is before submission only.

Focused Capsule regressions, strict source types, source style, architecture,
and CATSCAN inventory checks passed. The first architecture check identified
missing ownership for the new partitions facade; assigning it to the existing
client owner fixes the inventory without adding a facade exception or relaxing
dependency policy. Retained logs describe the final checks. Full `check:green`
and installed-package smoke were not rerun for the resident changes.

## Concrete next actions

1. Track tokenization in attempt cancellation and settlement: it currently uses
   the serialized pipeline lease but is absent from `attempt.pending`. Reject
   retired attempts and test closing while tokenization is in flight.
2. Make post-close `closeAttempt` idempotent without allocating new attempt
   tombstones. Test repeated closure, delayed work, failure and device loss.
3. Snapshot direct `openCapsule` resident options before asynchronous loading.
   The public factory already snapshots its allocation; direct opening needs
   equivalent protection and a mutation regression.
4. Resolve signed TargetPlan partition authorization and root dependency
   boundaries. Whole-model qualification does not qualify a split. The current
   pure partition validation import needs reconciliation with the root charter's
   independence requirement; do not weaken the charter to excuse drift.
5. Connect Reploid to effective generation settings and prompt token context.
   Group B needs context for repetition/presence penalties. Explicitly authorize
   that disclosure; an activation-only grant must not silently disclose tokens.
   Reploid already forwards `maxTokens` and requires final decoder output.
6. Build one installed package and canonical dependency record, then exercise
   verified Capsule acquisition and real inference through Reploid's existing
   coordinator in two browser tabs/windows on this machine. Compare unsplit and
   split results, concurrent conversations, cancellation, participant loss,
   recovery, communication and resource costs. Do not claim separate-device
   capacity pooling from this same-machine test.

The accepted search release is unchanged. No new package was published, no
deployment was performed, and no performance advantage is claimed.

Component: `doppler.runtime-source.client`,
`doppler.runtime-source.inference.pipelines.text`, `doppler.repository-tooling`,
`doppler.docs`.

Intent: preserved; acceptance remains unfinished.

Acceptance evidence: [checkpoint.json](checkpoint.json) and adjacent logs.

Boundary effects: public partitions factory, Capsule composition and numerical
attempt ownership; Reploid transport contract integration remains pending.
