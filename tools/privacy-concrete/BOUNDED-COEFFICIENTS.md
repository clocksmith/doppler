# Bounded public coefficient storage (research)

`flash-text-all-linear-ntt12-stream-check` is a distinct research protocol. It
combines the existing twelve-bit NTT arithmetic and checked integrity with
`doppler.client-compiled-public-tile-stream/v1`. Production admission and security
qualification remain closed. This is storage plumbing for encrypted linear
operations with client nonlinearities; it does not qualify complete Flash output,
malicious-adversary security or useful outsourced integrity.

## Ownership and admission

The caller supplies the authenticated complete client graph and its source/row
store leases. No source downloads, new hosts or allocations are implicit.

```python
from flash_prepared_graph import PreparedFlashGraph, EPHEMERAL_STORAGE
from flash_text_stream_plan import write_plan, PROFILE, CONFIG

prepared = PreparedFlashGraph(
    graph, executor, cache_directory, quantization_bits=12,
    storage=EPHEMERAL_STORAGE, output_chunk_rows=2048, cancelled=cancelled,
)
plan, binding = write_plan(cache_directory, prepared, cancelled=cancelled)
```

Preparation owns one temporary coefficient directory at a time. It quantizes
public source rows, verifies the complete integer matrix and derives its tile
digests from that same byte stream before evicting the file. Only public
quantizer metadata and immutable byte-derived proof objects survive. Publication
of `plan.json` is atomic and create-only; a matching existing manifest is reused,
and a different file or symlink is preserved and rejected.

`load_client_plan(directory, binding)` requires the actual complete compiler
binding; a manifest, advertised digest or `validated: true` dictionary cannot
replace it. The binding includes every projection, encoding, tile boundary and
the final immutable plan. `load_provider_plan` validates public manifest structure
only and is deliberately separate. It never authorizes client disclosure.

During execution, a projection is rematerialized in an owned temporary scope. Its
coefficient hash, encoding, bias, scale, zero point and sum must exactly match
preparation. Source leases remain mandatory. The attempt owns one projection
copy; cancellation and failure clear private bindings and remove only its owned
temporary files. Existing public source/table caches are preserved. The actual
serialized encryption parameters must match the admitted profile before graph
execution.

The checked bridge receives the real matrix used by the client. After verifying
it against the compiled tile digest, it carries that public tile beside the
opaque ciphertext in the same versioned `evaluate` frame. This adds no control
frames, retries, recipients or changes to the fixed tile sequence. The evaluator
checks the tile's digest, size and signed range before entering the ciphertext
parser and retains no coefficient disk cache. Its temporary public buffers are
bounded by one tile. No plaintext private activations or quantization metadata are
added to the wire format.

An explicitly pinned remote route can be prepared with
`SSHRoute.for_compiled_stream(directory, binding, endpoint, allowed_operators)`.
It checks the requester allowlist and compiled client binding before binding the
endpoint, host key, model, plan and installed backend. Its separate
`doppler.encrypted-ssh-stream-check-route/v1` schema cannot be substituted into a
legacy replay or checked route. Provisioning and transfer are separate actions;
the remote manifest and native implementation must already be installed.

Use `CheckedBridge(..., backend_profile=PROFILE, prepared_public_tiles=binding,
parameter_profile=CONFIG['parameterProfile'])` with the correctly configured
native executor/parameters. A generation runner must retain all candidate output,
finish the fixed route, close the bridge and call `require_integrity()` before
releasing output. See [INTEGRITY.md](INTEGRITY.md) for the unreviewed integrity
mechanism and failure/leakage limitations. The continuation owner's fixed
generation integration has not yet been provided to this worker; this module
does not silently invent or replace that runner.

## Space and transfer cost

For the installed text geometry, a complete twelve-bit endpoint cache is
250,210,795,520 bytes. This contract instead needs at most one client coefficient
matrix (largest: 1,271,398,400 bytes), public metadata/proofs, a bounded manifest,
and the existing authenticated raw checkpoint/local tables. The provider needs
no coefficient files and at most 8,388,608 bytes of public coefficients per tile,
in addition to its native key/ciphertext/kernel memory. These are file/geometry
bounds, not measured full-model RSS guarantees. The manifest limit is 64 MiB;
the existing 300,000,000-byte frame payload limit is unchanged.

The tradeoff is substantial: every step sends the public coefficient set again,
**250,210,795,520 bytes per step, 1,000,843,182,080 bytes for four steps**, plus
framing, evaluation key, ciphertexts and responses. The client also rematerializes
each projection every step. This contract removes the full-cache disk requirement;
it does not establish practical Flash latency or bandwidth. A separately versioned
evaluator that reconstructs from an already authenticated local checkpoint could
avoid this transfer; it is not implemented by this profile.

`public_coefficient_bytes_sent` counts public coefficient bytes in completely sent
frames; `sent`/`received` retain total protocol accounting. Prepared graph metrics
record initial preparation and cumulative rematerialization time. A failed partial
frame is not reported as a completely sent tile. Runners should report execution
rematerialization time by step separately from clear-reference work.

## Evidence and remaining gates

Focused tests cover actual BF16/quantizer preparation, eviction and rematerialized
metadata equality, atomic manifest preservation, complete byte-proof admission,
allowlists and legacy schema rejection, full four-step tiny mixed-attention graph
execution over real framed pipes, public corruption before ciphertext parsing,
source/parameter changes before disclosure, cancellation/cleanup, and deferred
rejection of corrupt results. Native encryption in the framed fixture is
explicitly simulated. A separate full-route metadata fixture checks all 74,501
projection definitions and 150,868 tiles without materializing weights.

`test_stream_native.py` is a separate opt-in four-step native GLWE feedback probe
over a small synthetic integer matrix and real framed pipes. It requires the
installed exact NTT build and pinned dependencies; stock or missing native
implementations reject. It uses client-held keys and actual encryption,
evaluation and decryption, with no endpoint coefficient files. Its optional
bounded report records key/wire/RSS/timing/error and integrity metrics. The
`encrypted-ntt.yml` manual workflow's `stream_storage` input selects this probe
without downloading Flash weights or running the unrelated arithmetic benchmark.
The probe is not a language model, full Flash run or separate-machine evidence.

The retained [native probe](evidence/storage/native-tile-stream-probe.json) and
[source/CI context](evidence/storage/native-tile-stream-context.json) come from
revision `e939163a28264630cff2b4e57875d28c11de4556`, successful run 44009726904.
Four native encrypted feedback steps on a synthetic 32-by-16 public integer matrix
took 13.13 seconds including setup, with a 268,435,536-byte evaluation key,
268,491,944 bytes sent, 52,093 received and 2,510,716,928 bytes process peak RSS.
Public coefficient traffic was 4,096 bytes; neither endpoint wrote coefficient
files. Integer errors ranged from 19 to 24 units per step and all four argmax
feedback choices matched the clear integer control. Four secret linear checks
passed with one sampled exact replay. These are one synthetic native run's
observations, not full-model precision, security or throughput guarantees.

Actual checkpoint hashes, floating/quantized full Flash outputs, native full-model
execution, separate-machine qualification, measured resource limits, the exact
generation integration and independent cryptographic/leakage review remain open.
The prior small-Qwen receipt is unchanged. Neither this profile nor its storage
proof is a production cryptographic capability advertisement.

## Already-present checkpoint reconstruction

`flash-text-all-linear-ntt12-rebuild-check` is a separate explicit profile for an
evaluator that already holds the pinned public checkpoint. It sends **zero public
coefficient bytes per step**. It never switches to streaming, downloads missing
source, or accepts a serialized "verified" flag when the checkpoint is absent.
The client still compiles its own complete byte binding and supplies the actual
matrix to the existing checked bridge; client-side work and integrity limitations
are unchanged.

The provider first verifies the installed linear source subset: 100 whole shards,
869 tensors and 260,799,695,848 whole-shard bytes, including complete shard hashes
and pinned auxiliary/index files. The client still needs its complete model and
local tables. The provider source subset is a linear-operator role, not evidence
of complete Flash output. Source identity, public tensor/expert/row bindings,
precision, storage contract and budget are bound to the immutable plan.

The evaluator quantizes one projection in bounded output-channel chunks, atomically
publishes it in an owned temporary directory and compares the whole coefficient
hash against client preparation. Every tile is independently hash/range checked
before ciphertext parsing. The active projection is reused across its tiles (and
successive steps when the projection is unchanged), then evicted on projection
change, finish or failure. Its cache budget is 2 GiB; the largest installed matrix
is 1,271,398,400 bytes. Scratch space must be outside the immutable raw source tree.
Cancellation, closed/changed source, insufficient space or hash mismatch terminally
reject and remove only owned cache files. Shared raw weights remain untouched.

Use `flash_text_rebuild_plan.write_plan` and pass its actual binding to
`CheckedBridge(..., backend_profile=REBUILD_PROFILE, prepared_public_tiles=binding,
provider_source_directory=existing_source_directory, ...)` for a local evaluator.
For an explicit remote evaluator use `SSHRoute.for_compiled_checkpoint`; its
endpoint additionally binds `remotePublicSource`, and its separate
`doppler.encrypted-ssh-rebuild-check-route/v1` schema cannot expand an existing
stream/replay route. The provider CLI requires `--public-source-dir` only for this
profile. No source transfer or provisioning is implicit.

Whole-source appraisal happens before provider readiness and native key parsing.
The plan explicitly allows 7,200 seconds for public-source bootstrap; per-operation
deadlines remain unchanged. On Jupiter, the pinned revision view and coordinated
exclusive CPU/IO window are mandatory operational prerequisites. Starting this
mode during another owner's timing run is not authorized by installing the code.
This worker performs no physical checkpoint read as part of these changes.

This trades network transfer for source residency and local hashing/quantization/
disk work. A complete sequential 250 GB working set will not obtain useful
cross-step LRU hits from a 2 GiB cache; a fixed 20 GiB cache alone could save at
most 20 GiB on each later pass. Reconstruction avoids transfer because the public
source is already present, while repeated full-model projection preparation still
costs CPU and IO. It is unavailable on hosts without that source and establishes
no full-model latency, accuracy or cryptographic-security guarantee.

The retained [native same-coefficient comparison](evidence/storage/native-rebuild-context.json)
records successful run 44016012728 at `d1846f11c1082ef18796885381419484eab08350`.
Across four native feedback steps, coefficient traffic fell from 4,096 bytes to
zero for the same quantized 32-by-16 matrix. Total sent bytes were 268,491,944
versus 268,487,818; a 268,435,536-byte evaluation key dominates this small fixture.
The provider verified 1,206 raw-source bytes, materialized a 1,024-byte coefficient
file once, reused it across four steps and removed it at completion. Both modes'
four argmax choices matched their clear controls. This is synthetic native
storage/traffic evidence, not a Flash run. Sequential tests have warm-import/key
variation and cumulative RSS high-water marks, so their timings/RSS do not
establish a comparative performance or memory result.
