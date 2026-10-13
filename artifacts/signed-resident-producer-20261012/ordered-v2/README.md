# Ordered physical producer preparation

This extends the existing offline producer with explicit
`doppler.physical-resident-producer-input/v2`; v1 remains the retained two-stage
path. `producer-before.js` preserves the exact pre-extension producer bytes.
No N-stage physical capture or qualification report was produced here. All
new tests are synthetic in-memory rejection/acceptance controls.

The workload freeze is `artifacts/ordered-partitions-20261013/three-stage-qualification-workload.json`.
It fixes Qwen's original 24 layers at cuts `[8,16]`, the same model/graph/weights,
55-prefix independent F64 reference, 0.001 tolerance, generation, and 1.42-billion
byte ceiling per executor. The three-logical-stage request is for three physical
hosts; actual provider preflight remains pending, including Intel portability.
The file intentionally says `launchReady:false`. The old 0fad runtime cannot
execute the new ordered API and cannot be reused as the N-stage package.

The producer requires exact new archive/delivery/executable/provider identities
frozen before outputs. It imports ordered plan/contract validation and generation
hashing from that extracted archive, never live source. The original raw graph
and normalized execution-identity hash domains remain distinct.

## Required capture extension in existing Reploid fixture

The existing `browser-partition-memory-completion.js` reference loop hardcodes
A/B and `executeGroup0/1`. An opt-in ordered-reference branch must instead use
explicit executor endpoints and actual `executeStage`, preserving the normal
legacy phases. `prepare-partition-memory.js` needs an explicit v3 plan option
and stage participant identity; its existing device, opening, provider and queue
observers remain the owners. The old legacy plan ID continues identifying the
reference corpus; it cannot claim the new split has the old split geometry.

The v2 producer input adds:

- `partitionPlan`, `orderedContract` (the frozen contract-v2), `logicalStageCount`,
  `physicalHostCount`, `maxGpuBufferBytes`, and `deadlines` exactly
  `{stageMs:300000,cleanupMs:10000,observationMs:2000}`.
- Ordered `placements`: `partitionIndex`, `executorId`, actual host `platform`,
  `participantId`, `transportId`, `residentId`, `executionId`. The execution ID
  is the canonical digest of the actual prepared initial execution identity.
- `providers`, `surfaces`, `executables`, `browserVersions`/`nativeHosts`, keyed
  by executor ID rather than host name. Executables have exact SHA256 and size.
- Explicit `requesterId` and `placementGeneration` for the frozen ordinary route.

Each capture run retains its actual ordered attempt identity. Every numerical
prefix retains ordered `stageExecutions`, one per stage, containing:
`partitionIndex`, `executorId`, `method:'executeStage'`, `completed`, `elapsedMs`,
returned `identity`, `step`, `tokenPosition`, actual `inputTokenCount` and
`inputTokenIds`, `generationDigest`, `initialExecutionIdentityHash`,
`activationInput`, `activationOutput`, and `logitsHash`.

The entry activation input and final activation output are explicitly null.
Other activations contain exact `shape,dtype,precision,components,elementStrideBytes,
step,seqOffset,data,byteLength,storageHash`; `data` is canonical raw-byte base64.
Every output must equal the next stage's input, with its bytes, shape and digest
independently checked. Only the final stage binds the full scalar-F32 logits;
all 55 are independently compared with the retained full F64 distributions.
Nonfinal logits hashes are explicitly null.

Descriptors retain actual prepared v2 resident metadata, original opening
observation, `executorId`, observed executable identity, and actual
`executorIdentity:{pid,contextId}`. All stage contexts must be distinct.
`residentSettlement` records name stage/executor/initial identity, confirmed
queue barriers, closed resident, closed attempt count, elapsed cleanup time,
and existing complete memory/error snapshot. Separate ordered
`executorSettlements` confirm the matching actual context closed. Multiple
native participants additionally require independently matched process/provider
releases in `nativeHostSettlements`; no copied release from another PID qualifies.

These are direct installed-runtime captures. They do not establish authenticated
network forwarding, signed application opening, or consensus. The producer emits
report-v2 only after complete per-stage records and the independent final-vector
comparison pass. The CLI then runs the exact archive's existing report and
complete-set validators before writing new files exclusively.

Validation: `node --test artifacts/signed-resident-producer-20261012/produce-physical-reports.test.js`
passed 86 CPU controls, including legacy cases, 2/3/4/7 logical stages, 3 stages
on 2 hosts, two independently settled native providers, and negative identity,
activation, deadline, descriptor and cleanup mutations. No GPU launched.

Ordered settlement requires zero retained-model, active and deferred owner counts and bytes. Allocated reusable buffers and their weight-category attribution remain reported; reusable storage is not an unsettled owner. The retained v1 report check is unchanged.
