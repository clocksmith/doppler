# Distributed Transcript Root

`transcriptRoot` is the per-request identity for distributed execution.

It is separate from canonical `artifactIdentity`.

## Purpose

`artifactIdentity` reproduces the model.

`transcriptRoot` reproduces one request execution:
- token schedule
- RNG placement/state
- peer assignments
- per-stage output hashes
- activation integrity roots
- KV integrity roots

Replay claim:
- `artifactIdentity + transcriptRoot + input tokens -> exact distributed replay`

## Required fields

Minimum contract surface:
- token schedule
- RNG state placement
- per-stage output hashes
- peer assignments
- activation Merkle roots
- KV Merkle roots

Coordinator authors this object.

Replay verification consumes it.

