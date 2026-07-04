# Distributed Inference Roadmap

Guide for landing distributed inference as an additive capability on canonical
RDRR artifacts.

This guide covers the lower distributed-execution layer: one canonical artifact
running across peers. Doppler's cross-family MoE support surface is drafted
separately in [`MoM Layer Draft`](../distribution/mom-layer-draft.md).

## Principles

- one canonical weight artifact
- no topology-specific repacked weight sets
- offline plan synthesis only
- exact parity lane separated from approximation/speed lanes
- dual-root identity:
  - `artifactIdentity`
  - `transcriptRoot`

## Ordering Constraint

Distributed inference should prove whole-model replicated execution before
remote model partitioning. The first claimable P2P lane is several peers running
the same canonical artifact and producing deterministic receipts for the same
request. Split-layer execution and remote expert hosting are additive lanes
after that receipt format is stable and reproducible.

MoE storage can progress earlier as local expert paging, but it is still a
storage/runtime capability until runtime receipts exist:

- shared backbone, router, and tokenizer come from the canonical artifact
- expert shards are content-addressed and bound to the artifact identity
- active expert tiles are cache-managed by config-owned policy
- router path receipts record selected expert IDs and shard hashes
- P2P expert hosting is a separate distributed-execution extension

## Phase 0 completion

1. canonical hash utility
2. block Merkle builder + verifier
3. converter / refresh-integrity integration
4. physical storage descriptor contract
5. slice compiler
6. partial-fetch verifier
7. integrity + slice regression tests

## Distributed-plan schema + validator

8. `distributed.json` schema + compatibility block
9. `src/formats/rdrr/distributed/*`
10. typed distributed-plan errors
11. parser regression coverage

## Placement synthesizer

12. `src/tooling/placement-synthesizer/`
13. tile-factor search under memory/PE constraints
14. compatibility hash emission
15. `doppler synth-placement`
16. golden-plan tests

## Collective transport

17. transport contract doc
18. reference collectives over WebRTC
19. session lifecycle + backpressure
20. induced loss/reorder/abort tests

## ExecutionGraphV2

21. distributed fields:
   - `distributedOp`
   - `partitionSpec`
   - `exactness`
   - `group`
   - `samplerPlacement`
   - `tokenizerPlacement`
22. parity-lane validator

## Transcript root

23. transcript-root contract doc
24. transcript schema
25. replay verifier + tests

## Peer identity

26. peer identity contract doc
27. session-scoped ed25519 minting
28. signed activation envelopes
29. signer/session expiry tests

## Control plane

30. control-plane extensions doc
31. assignment/session protocol
32. introspection + plan selection
33. stage ingress/egress routing

## Runtime phases

34. distributed runners
35. byte-range fetch integration
36. `text.js` capability-gated hook
37. fail-closed capability behavior

## Parity harness

38. solo vs distributed logits test
39. per-collective parity probes
40. separate parity/perf lanes

## Later phases

41. multi-plan selection
42. peer-introspection protocol
43. topology fingerprinting
44. MoE expert placement for distributed execution
45. all-to-all dispatch execution
46. expert heat profiling
47. exact-routing parity gate
48. activation provenance envelopes
49. stage replication runtime
50. KV migration
51. chaos harness
52. activation compression
53. compute/collective overlap
54. micro-batching
55. warmup weight streaming
56. WAN-simulation perf harness

## Cross-cutting work

57. prior-art survey
58. feasibility/performance modeling tool
59. this developer guide
60. `doppler infer --distributed --plan`
61. promotion-playbook distributed parity gate
62. `doppler-distributed` agent skill

## Promotion boundaries

- Phase 1: correctness-shippable
- Phase 5: latency/perf-shippable

Do not collapse those gates.
