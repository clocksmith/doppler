# Doppler executable-model adoption plan

Status: code-aligned portfolio alternative  
Strategy unit: Doppler alone  
Repository owner: Doppler

## Outcome

Doppler becomes the preferred system for converting attributable source models
into portable, specialized JavaScript and WebGPU programs. Forge, ModelIR,
TargetPlans, immutable Capsules, the constrained Runtime, application integration,
and lifecycle support form one product.

The decisive strategy proof is voluntary adoption by an unrelated application.
The application must choose Doppler's executable-model path, preserve its own
acceptance semantics, ship or activate the result under its own authority, and
retain Doppler for another consequential release or model change.

The stronger commercial proof remains the paid production-release and
subsequent-upgrade episode defined in [Doppler goals](goals.md). Payment and
release delegation strengthen adoption evidence. They do not redefine Doppler
as only a release-governance service.

## Independence boundary

This strategy does not require Poolday, Reploid, peer delivery, or Doe.

- Ordinary authorized origins and caches are sufficient for artifact delivery.
- Browser WebGPU or another qualified provider is sufficient for execution.
- Doe is eligible only as a separately qualified provider.
- External applications validate Doppler. They do not transfer adoption from a
  sibling project.
- Repository fixtures, internal integrations, and model-count growth do not
  satisfy the decisive proof.

## Starting point

The repository already implements the source-truth to Capsule path, signed Capsule
validation, TargetPlan selection, Capsule-bound execution, application adapters,
qualification contracts, and release machinery. Current evidence establishes
mechanics. It does not establish voluntary external adoption, production use,
payment, or repeat use.

## Execution gates

| Gate | Required work | Exit evidence | Does not prove |
| --- | --- | --- | --- |
| D0: adopter contract | Name one unrelated JavaScript application, owner, pinned revision, workload, source model, incumbent, acceptance suite, supported hosts, and activation authority. | Customer-authorized, versioned integration contract with no synthetic ownership fields. | Runtime correctness or demand. |
| D1: source-to-Capsule closure | Acquire attributable source facts, forge ModelIR, produce at least one qualified TargetPlan, and seal every required artifact into one signed Capsule. | Capsule validation, source lineage, packaged qualification evidence, and retained rejected candidates. | Application acceptance. |
| D2: unchanged acceptance | Run the application's existing acceptance semantics against its incumbent and Doppler. Change only the declared provider or integration seam required to invoke Doppler. | Comparable application-level results, explicit exclusions, and no hidden fallback. | A material advantage or adoption. |
| D3: material advantage | Predeclare and measure one adopter-valued advantage such as supported capability, installation simplicity, end-to-end latency, memory, offline custody, diagnostic depth, or correction speed. | Evidence showing the advantage without weakening correctness, reliability, or workload scope. | Production use. |
| D4: voluntary adoption | The application owner chooses Doppler, distributes or activates the integration, and retains rollback authority. | Owner-attributed adoption record plus the exact shipped application, Capsule, execution, and acceptance identities. | Recurring demand. |
| D5: repeated dependence | Requalify a later model, Capsule, application, browser, driver, or device change through Doppler. | A second owner-authorized release or upgrade using the same product contract. | Broad market adoption. |

Gates remain ordered. A later gate cannot repair missing evidence from an
earlier gate.

## Product workstreams

### Developer entry

- Freeze the smallest application API that the selected adopter can maintain.
- Keep Capsule and execution identity inspectable beneath any convenience facade.
- Measure installation-to-first-accepted-output, integration changes, error
  recovery, and upgrade work.
- Do not add a simplified API whose defaults bypass manifest, TargetPlan, or
  provider policy.

### Forge and Capsule closure

- Use one real source revision and one application workload.
- Preserve unsupported operations and failed candidates as typed evidence.
- Keep model semantics in ModelIR and hardware specialization in TargetPlans.
- Bind source, tokenizer, kernels, execution graph, qualification, lifecycle,
  and previous-release identity into the Capsule.

### Runtime ownership

- Compare Doppler with the strongest eligible incumbent for the exact workload.
- Select only a prequalified TargetPlan and expose the resolved provider and
  execution identity.
- Credit Doppler-owned execution only when it supplies a predeclared adopter
  advantage. Otherwise interoperate with the stronger qualified provider.

### Application lifecycle

- Let the application own activation and rollback.
- Exercise cancellation, failure preservation, revocation, and restoration of
  the previous Capsule where the adopter's contract requires them.
- Repeat qualification after a consequential identity change.

## Measures

- External applications choosing and retaining Doppler.
- Accepted outputs under application-owned tests.
- Integration surface and installation failure rate.
- End-to-end load, first response, steady execution, and peak memory under the
  adopter's declared workload.
- Unsupported, rejected, recovered, and revoked releases.
- Repeat releases or model upgrades.
- Paid production releases and recurring release operations.

Catalog size, internal demo count, Capsule count, and isolated kernel benchmarks
remain supporting measures.

## Stop conditions

Stop or reframe this strategy when any of the following persists after testing
with credible adopters:

- Applications will run evaluations but will not adopt the execution path.
- Integration requires application-specific forks that cannot become a stable
  product surface.
- Doppler provides no material application advantage over qualified incumbents.
- Capsule qualification cannot remain reproducible across the declared support
  matrix.
- Repeat releases behave like unrelated consulting engagements rather than a
  reusable compiler and runtime product.

## Repository work queue

The next destination is a reproducible release, automatic source discovery, and
one independently reproducible optimization, in that order. Engineering proceeds
without requiring an external adopter; external adoption retains its separate
D0–D5 proof requirements.

The [local-inference baseline](capsule-physical-baseline.md) closes the implemented
Node/browser application and repeatable preparation milestone. Its linked
receipts retain the exact qualification scope. Two source revisions differing
only in README content prove orchestration repeatability, not architecture
coverage. The integrated archive's subsequent qualification is retained in the
[committed release receipt](../reports/capsule-baseline/20260907-release/acceptance.json).

1. **Reproducible release — completed for the scoped Node/Chromium baseline.**
   The linked receipt binds clean-checkout acceptance, successful remote CI,
   public-source reconstruction, and physical execution to exact identities.
   The repeatable release requirements remain: review and commit implementation, rebuild from
   an isolated checkout using documented dependencies and retained artifacts,
   and run installed Node/browser qualification against the preserved references.
   Bind acceptance to the exact commit, runtime archive, model/source identities,
   environment, and commands. Confirm remote checks and provide a retrieval path
   for everything legally distributable that does not depend on the working
   directory. Preserve private signing custody, historical observations,
   application approvals/denials, and revoked checkpoints.
2. **General model onboarding.** Extend the existing coordinator to detect an
   upstream revision automatically and produce its support assessment. Exercise
   a different architecture. Distinguish reusable configuration changes from
   missing computation; emit precise implementation tasks and executable
   reference tests. Discovery and successful preparation do not approve release
   promotion.
3. **Reusable optimization.** Connect candidate generation to physical hardware
   experiments under unchanged correctness tests. Freeze tuning and held-out
   inputs separately, demonstrate an improvement on the held-out inputs, and
   reproduce it in a second application. Preserve failed candidates and all
   costs. Outside contribution strengthens evidence but does not gate machinery.
   Keep limited artifact retention explicit: its measured memory benefit costs
   opening time, and unlimited retention remains the default.
4. **Verified inference coverage.** Qualify generation, additional physical
   devices, and Bun separately. Compare strong alternatives under equivalent
   quality requirements and visible startup, memory, throughput, cancellation,
   and recovery contracts. Record blocked or failed lanes. One AMD configuration
   does not establish general portability or superiority.
5. **Architecture through extensions.** Fix concrete coupling and defects exposed
   by these extensions, remove duplicated execution paths, enforce module
   ownership, and keep publication tooling outside runtime dependencies. Preserve
   compiler/runtime ownership and application activation authority.

For voluntary adoption, select an authorized unrelated application through the
existing design-partner process, freeze D0, retain comparable incumbent and
Doppler results before tuning, and retain owner-attributed D1–D5 evidence.
Internal application work does not satisfy that external authority requirement.

Current status remains machine-owned by the goal matrix, support registries,
runtime-ownership decisions, and retained receipts. This plan owns sequence and
proof meaning, not mutable completion state.
