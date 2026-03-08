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
  resumeOffset?: number,
  expectedHash?: string | null,
  expectedSize?: number | null,
  expectedManifestVersionSet?: string | null,
}
```

Semantics:
- `attempt` starts at `0` and increments per retry.
- `maxRetries` is the retry budget for P2P only.
- `signal` cancellation must stop transport work promptly.
- `resumeOffset` is provided when Doppler has persisted a shard prefix and is requesting remaining bytes.

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
  rangeStart?: number | null,
  totalSize?: number | null,
  miss?: boolean,
  notFound?: boolean,
  error?: unknown,
}
```

Notes:
- `data`/`buffer` are equivalent.
- `rangeStart` should match the requested `resumeOffset` when partial resume is used.
- `totalSize` should be set when available and must align with expected shard size if provided.
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
- Doppler validates P2P resume alignment (`resumeOffset` vs payload `rangeStart`) and fails that source attempt on mismatch.
- When anti-rollback requires `manifestVersionSet`, mismatched or missing version-set fails that source attempt.
- Source transition behavior remains controlled by configured source matrix (`onMiss`/`onFailure`).

## Configuration Knobs (schema)
- `loading.distribution.p2p.enabled`
- `loading.distribution.p2p.timeoutMs`
- `loading.distribution.p2p.maxRetries`
- `loading.distribution.p2p.retryDelayMs`
- `loading.distribution.p2p.contractVersion`
- `loading.distribution.p2p.transport`
- `loading.distribution.p2p.controlPlane.enabled`
- `loading.distribution.p2p.controlPlane.contractVersion`
- `loading.distribution.p2p.controlPlane.tokenRefreshSkewMs`
- `loading.distribution.p2p.controlPlane.tokenProvider`
- `loading.distribution.p2p.controlPlane.policyEvaluator`
- `loading.distribution.p2p.webrtc.enabled`
- `loading.distribution.p2p.webrtc.peerId`
- `loading.distribution.p2p.webrtc.requestTimeoutMs`
- `loading.distribution.p2p.webrtc.maxPayloadBytes`
- `loading.distribution.p2p.webrtc.selectPeer`
- `loading.distribution.p2p.webrtc.getDataChannel`
- `loading.distribution.p2p.security.requireSessionToken`
- `loading.distribution.p2p.security.sessionToken`
- `loading.distribution.p2p.security.tokenExpiresAtMs`
- `loading.distribution.p2p.abuse.rateLimitPerMinute`
- `loading.distribution.p2p.abuse.maxConsecutiveFailures`
- `loading.distribution.p2p.abuse.quarantineMs`

Validation rules:
- Invalid `loading.distribution.sourceOrder` entries fail before delivery planning; Doppler does not silently drop unknown sources or rewrite an empty list.
- Numeric P2P policy knobs (`timeoutMs`, `maxRetries`, `retryDelayMs`, control-plane `tokenRefreshSkewMs`, abuse thresholds) must be valid integers in-range for their field or the run fails before transport execution.

## Delivery Diagnostics
- `downloadShard()` always returns `deliveryMetrics` summarizing source attempts/retries, failure codes, source RTT aggregates, and storage write latency.
- `downloadShard(..., { onDeliveryMetrics })` provides a production hook for exporting metrics snapshots to dashboards/SLO backends.
- Decision traces are controlled by:
  - `loading.distribution.sourceDecision.trace.enabled`
  - `loading.distribution.sourceDecision.trace.includeSkippedSources`
  - `loading.distribution.sourceDecision.trace.samplingRate` (0..1, deterministic when `sourceDecision.deterministic=true`)
- Higher-level observability helpers fail closed on invalid explicit SLO targets:
  - `minAvailability`, `minP2PHitRate`, and `maxHttpFallbackRate` must be finite numbers in `[0, 1]`
  - `maxP95LatencyMs` must be a finite non-negative number
  - invalid target payloads fail before summary or alert generation

## Browser WebRTC Slice (feature-flagged)
- `createBrowserWebRTCDataPlaneTransport()` implements an optional browser-only data-plane transport adapter.
- The adapter is fail-closed: enabling WebRTC without runtime capabilities (`RTCPeerConnection`) or channel wiring throws `unconfigured`.
- Wire through config with `loading.distribution.p2p.webrtc.enabled=true` and `getDataChannel` callback when no explicit `p2p.transport` callback is provided.

## Control-Plane Boundary
- `loading.distribution.p2p.controlPlane.tokenProvider` can issue/refresh session tokens (missing, expired, proactive refresh).
- `loading.distribution.p2p.controlPlane.policyEvaluator` can allow/deny each shard attempt and attach session updates.
- Policy denials are surfaced as canonical `policy_denied` transport errors and respect source-matrix transitions (`onFailure` by default).
