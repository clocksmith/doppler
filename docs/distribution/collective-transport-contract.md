# Collective Transport Contract (`distribution.collectives`) v1

Contract for distributed collectives layered above shard delivery.

## Purpose

This contract governs cross-peer collective payload exchange for:
- all-reduce
- all-gather
- reduce-scatter
- all-to-all
- stage-to-stage tensor transfer

It is distinct from shard delivery:
- shard delivery fetches canonical artifact bytes
- collective transport moves request-local execution state

## Contract version

- file/runtime contract version: `1`
- unsupported versions fail closed

## Exactness classes

Each collective step must declare an exactness class.

### `bit-exact-solo`

Bitwise identical to solo execution.

Reserved for:
- conformance tests
- narrow debug lanes

### `algorithm-exact`

Bitwise identical when the declared reduction schedule is followed exactly.

Required step metadata:
- `reductionOrder`
- `treeShape` or equivalent topology
- `accumDtype`

Phase 1 default:
- `algorithm-exact`

### `tolerance-bounded`

Approximation lane only.

Required metadata:
- metric
- epsilon

Parity rule:
- parity lanes reject `tolerance-bounded`

## Step declaration

Example:

```json
{
  "step": "layer.7.attn.out.allreduce",
  "distributedOp": "all_reduce",
  "exactness": {
    "class": "algorithm-exact",
    "reductionOrder": "declared-tree",
    "treeShape": "binomial",
    "accumDtype": "f32"
  }
}
```

## Framing

Every chunk carries:

```json
{
  "groupId": "attn-layer-7",
  "opId": "allreduce-42",
  "chunkIndex": 0,
  "chunkCount": 8,
  "payloadHash": "sha256:..."
}
```

Rules:
- ordering is strict per `(groupId, opId)`
- out-of-order chunks may be buffered only within the declared receive window
- payload hash mismatch fails the collective attempt

## Chunk sizing

Default:
- `16 KiB`

Why:
- conservative default for WebRTC SCTP effective MTU and buffering behavior

Plans may override chunk size explicitly per collective.

## Backpressure

Collective channels are credit-based.

Required controls:
- `maxInflightChunks`
- `maxBufferedBytes`

Overflow behavior:
- fail closed with `DOPPLER_COLLECTIVE_BUFFER_EXHAUSTED`

## Retransmit and timeout

Transport loss:
- handled by the underlying reliable transport where available

Collective-level failure:
- stalled op times out at the collective layer
- stalled op may be replayed only under the declared recovery policy

## Cancellation

Abort is group-wide:
- one collective abort propagates to every participant
- wired through `AbortController`

No peer may continue emitting chunks for an aborted `(groupId, opId)`.

## Memory limits

Implementations must use bounded buffers:
- bounded send queue
- bounded receive ring
- bounded reorder window

Unbounded buffering is not allowed.

## Error surface

Minimum canonical errors:
- `DOPPLER_DISTRIBUTED_PLAN_STALE`
- `DOPPLER_COLLECTIVE_BUFFER_EXHAUSTED`
- `DOPPLER_COLLECTIVE_TIMEOUT`
- `DOPPLER_COLLECTIVE_ABORTED`
- `DOPPLER_COLLECTIVE_PAYLOAD_INVALID`
- `DOPPLER_COLLECTIVE_INTEGRITY_MISMATCH`
- `DOPPLER_DISTRIBUTED_STAGE_UNRECOVERABLE`

## Recovery

Recovery depends on declared redundancy:
- replicated stage/collective path: retry or redirect under plan policy
- non-replicated path: fail closed

This contract does not imply survivability on its own.

## Related contracts

- [rdrr-p2p-plan.md](rdrr-p2p-plan.md)
- [p2p-transport-contract.md](p2p-transport-contract.md)
- [../rdrr-format.md](../rdrr-format.md)
