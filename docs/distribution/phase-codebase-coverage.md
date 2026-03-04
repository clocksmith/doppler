# Phase 1-4 Codebase Coverage (In-Repo Scope)

This document tracks what is implemented directly in the Doppler repository for the Phase 1-4 roadmap.

## Phase 1
- Transport contract versioning and typed error normalization.
- Deterministic source routing (`cache -> p2p -> http`) with optional decision trace.
- In-flight shard request dedupe.
- Persistent store-backed resume parity for both HTTP and P2P paths (including p2p `resumeOffset` threading and range-start alignment checks).
- Anti-rollback guards for expected hash, size, and manifest version-set.
- Downloader integration with manifest version-set propagation.

## Phase 2 (repo-scoped)
- Runtime contracts for source attribution telemetry in downloader state/progress.
- Per-shard delivery metrics emitted by shard delivery (`deliveryMetrics`): attempt counts, retries, failure-code tallies, source RTT summaries, and storage write latency.
- Metrics export hook (`onDeliveryMetrics`) and observability transforms for dashboard snapshots, alert extraction, and SLO gate evaluation (`src/distribution/p2p-observability.js`).
- Deterministic decision-trace sampling (`sourceDecision.trace.samplingRate`) to control trace volume while preserving reproducibility.
- Runtime P2P abuse controls exist in delivery policy (`security` session-token gating, per-transport rate limiting, and temporary quarantine after repeated failures).
- Control-vs-transport separation represented in distribution modules.
- P2P transport interface remains pluggable and transport-agnostic.
- Feature-flagged browser WebRTC data-plane slice (`src/distribution/p2p-webrtc-browser.js`) with fail-closed capability checks.
- Minimal control-plane contract wiring for token issuance/refresh and policy gating (`src/distribution/p2p-control-plane.js`) integrated into shard delivery.
- Staged resilience drill runner for origin outage / peer collapse / anti-rollback incident simulations (`tools/p2p-resilience-drill.mjs`).
- Operations export CLI for dashboard/alerts/SLO artifacts (`tools/p2p-delivery-observability.mjs`).

## Phase 3 (repo-scoped)
- Signed/provenance hooks and revocation policy represented in shared ecosystem config.
- Federation and settlement planning captured in execution plan docs.

## Phase 4 (repo-scoped)
- Added `runtime.shared.ecosystem` policy contract covering:
  - multi-tenant trust/onboarding
  - publish workflow gates/rollback
  - discovery/ranking policy
  - multi-artifact registry policy
  - hosted access/quotas/billing
  - compliance/audit/takedown/export-control policy
  - notarization/provenance/revocation policy
  - developer ecosystem tooling policy
  - network effects and incentives policy
  - ecosystem abuse policy
  - quality evidence policy
  - reliability/failover/SLA policy

## Remaining out-of-repo requirements
The following require external services or operations beyond this repository:
- live signaling/discovery/control-plane services
- managed API gateway and billing backend
- compliance/legal operations workflow execution
- global relay mesh deployment and SLA operations
- marketplace/community product surfaces
