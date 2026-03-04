# P2P Transport Contract (`p2p.transport`) v1

## Purpose
Doppler shard delivery is transport-agnostic. `p2p.transport` provides shard bytes while Doppler enforces source policy, integrity, fallback behavior, and persistence policy.

## Contract Version
- Runtime field: `loading.distribution.p2p.contractVersion`
- Current supported value: `1`
- Invalid versions fail fast with `DOPPLER_DISTRIBUTION_P2P_CONTRACT_UNSUPPORTED`.

## Request Payload (runtime -> transport)
The transport callback receives:

```js
{
  shardIndex: number,
  shardInfo: object,
  signal?: AbortSignal,
  source: 'p2p',
  timeoutMs: number,
  contractVersion: 1,
  attempt: number,
  maxRetries: number,
  expectedHash?: string | null,
  expectedSize?: number | null,
  expectedManifestVersionSet?: string | null,
}
```

Semantics:
- `attempt` starts at `0` and increments per retry.
- `maxRetries` is the retry budget for P2P only.
- `signal` cancellation must stop transport work promptly.

## Response Payload (transport -> runtime)
Accepted shapes:

1. Raw bytes:
```js
ArrayBuffer | Uint8Array
```

2. Envelope:
```js
{
  data?: ArrayBuffer | Uint8Array,
  buffer?: ArrayBuffer | Uint8Array,
  manifestVersionSet?: string | null,
  manifestHash?: string | null,
  miss?: boolean,
  notFound?: boolean,
  error?: unknown,
}
```

Notes:
- `data`/`buffer` are equivalent.
- `miss`/`notFound` indicate provider miss (`unavailable`) and trigger fallback policy.
- `manifestVersionSet` is used by anti-rollback checks when enabled.

## Error Taxonomy
Canonical P2P transport error codes:
- `DOPPLER_DISTRIBUTION_P2P_TRANSPORT_UNCONFIGURED`
- `DOPPLER_DISTRIBUTION_P2P_TRANSPORT_UNAVAILABLE`
- `DOPPLER_DISTRIBUTION_P2P_TRANSPORT_TIMEOUT`
- `DOPPLER_DISTRIBUTION_P2P_TRANSPORT_ABORTED`
- `DOPPLER_DISTRIBUTION_P2P_TRANSPORT_INTEGRITY_MISMATCH`
- `DOPPLER_DISTRIBUTION_P2P_TRANSPORT_POLICY_DENIED`
- `DOPPLER_DISTRIBUTION_P2P_TRANSPORT_INTERNAL`
- `DOPPLER_DISTRIBUTION_P2P_TRANSPORT_PAYLOAD_INVALID`

Retry behavior:
- Retryable by default: `timeout`, `internal`
- Non-retryable by default: `unavailable`, `integrity_mismatch`, `policy_denied`, `payload_invalid`, `aborted`, `unconfigured`
- Transport-specific aliases (for example `timeout`, `miss`, `policy_denied`) are normalized to canonical codes.

## Abort + Timeout Semantics
- Doppler enforces timeout using `timeoutMs`.
- Abort signals become `DOPPLER_DISTRIBUTION_P2P_TRANSPORT_ABORTED` and terminate P2P attempts immediately.
- Timeout errors map to retryable timeout errors and consume retry budget.

## Integrity + Rollback Expectations
- Doppler computes shard hash and checks expected hash/size.
- When anti-rollback requires `manifestVersionSet`, mismatched or missing version-set fails that source attempt.
- Source transition behavior remains controlled by configured source matrix (`onMiss`/`onFailure`).

## Configuration Knobs (schema)
- `loading.distribution.p2p.enabled`
- `loading.distribution.p2p.timeoutMs`
- `loading.distribution.p2p.maxRetries`
- `loading.distribution.p2p.retryDelayMs`
- `loading.distribution.p2p.contractVersion`
- `loading.distribution.p2p.transport`
