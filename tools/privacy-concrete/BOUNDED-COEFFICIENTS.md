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

Actual checkpoint hashes, floating/quantized full Flash outputs, native full-model
execution, separate-machine qualification, measured resource limits, the exact
generation integration and independent cryptographic/leakage review remain open.
The prior small-Qwen receipt is unchanged. Neither this profile nor its storage
proof is a production cryptographic capability advertisement.
