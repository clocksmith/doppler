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
digests from that same byte stream before evicting the file. Only tensor-free
public projection identities and immutable byte-derived proof objects survive.
An identity binds the exact coefficient definition, encoding and finite CPU
scale/zero/sum/bias metadata, including each tensor's shape, dtype and bytes.
Every freshly quantized lease must match before use. This removes retained
per-projection Torch arrays from the evictable path; ordinary retained storage
still owns its prepared projections. Identities are derived locally from actual
preparation and cannot replace byte proofs or authorize imported plans. Publication
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

## Fixed generation over an admitted graph

`flash_generation.generate(prepared, prompt_tokens, bridge, secret, parameters,
context)` composes the supplied authenticated graph and admitted bridge into the
plan's fixed autoregressive route. It consumes the bridge on every exit and owns
fresh, separate encrypted, floating-reference and quantized-reference request
states. It releases candidate tokens only after every encrypted step, provider
finish, transport closure, integrity outcome and teacher-forced quantized greedy
comparison pass. Floating comparisons are reported observations. Both references
are full client computation; this is a research correctness oracle.
Generation requires a fresh route cursor and consumes it on every exit; an
already partly evaluated route cannot restart with fresh model state.

The graph has no prompt-padding mask contract. The prompt must contain exactly
`contextBucket` rows, and the entire route must fit the graph's context budget.
The original Flash profiles bind one prefill row; the multi-token public capital
prompt used by `validate_flash_checkpoint.py` cannot be passed to those profiles.
The function rejects that mismatch before any ciphertext
evaluation, rather than truncating the prompt or changing the admitted route.
EOS does not shorten execution. No source acquisition, route selection, plaintext
feedback, output callbacks or production admission are added.

Tiny mixed-attention graph tests run the actual quantizer, client state owners,
checked bridge and provider framing with explicitly simulated encryption. They
cover complete four-step generation, integrity and numerical rejection after
closure, cancellation, wrong request binding and prefill mismatch. This component
does not locate the previously mentioned generation pin, provide actual Flash
reference outputs or establish full Flash encrypted execution. Those physical
and integration gates remain open with the checkpoint/generation owners.

`flash-text-all-linear-ntt12-rebuild-prefill5-check`, implemented by
`flash_text_rebuild5_plan`, is a separate installed research profile that binds
exactly five prefill rows, followed by three single-row autoregressive steps. Its
policy, client program, manifest, plan identity and backend protocol are distinct
from the one-row reconstruction profile. The public capital fixture's tokenizer
produces five rows; no truncation or padding is performed. Preparation and the
SSH route must explicitly select this profile (`backend_profile` on
`SSHRoute.for_compiled_checkpoint`), with the actual byte-derived compiler proof.
Old manifests/routes cannot authorize it, and it cannot authorize old manifests.
The source, precision, cache budget, integrity mechanism and production closure
are unchanged. Five-row tiny graph/framed execution is separate from actual
Flash output qualification. A native storage fixture additionally checks five
encrypted input rows followed by single-row feedback, without Flash weights.

## Bounded clear reference before physical qualification

`flash_quantized_reference.BoundedQuantizedReference(graph, scratch_directory)`
is an explicit trusted-client callback for source-bound clear validation. Supply
it as `verified_client_projection` to the complete graph in
`execution='verified-projections'` mode. It reuses the encrypted route's global
input quantization, per-channel quantizer and exact tiled integer reference.
Unlike the older validator callback, it reads source weights in output-channel
chunks and evicts each temporary coefficient projection before the next one.
The scratch directory must already exist outside immutable source directories.
The two-GiB projection budget and available disk are checked before source reads;
the lease is reserved before quantization. Cancellation, insufficient space or
changed coefficient/quantizer metadata terminally reject and clean owned files.
Integer-reference cancellation is checked between bounded product tiles, so a
large output projection does not finish all remaining tiles after cancellation.
The caller retains graph/source ownership and must close request states.

`inventory(graph.order)` returns the complete ordered coefficient identities and
the same canonical inventory digest as whole-projection quantization. It cannot
publish an incomplete inventory. Only public identities survive each operation;
no activation/key/response caches are retained by the callback. Reusing this
callback still recomputes the entire clear model and reads/quantizes source on
each step. It does not remove the client correctness-oracle cost.

Installed geometry gives a largest FP32 source chunk of 50,331,648 bytes, versus
2,542,796,800 bytes for the largest complete FP32 projection, and a largest
temporary coefficient file of 1,271,398,400 bytes. Quantizer intermediates, mapped
file pages, model/client state, imports and native allocations add to resident
memory; these calculated bounds are not an eight-GiB RSS qualification. Tiny
five-row prefill plus three decode steps match the whole quantizer's complete
output and ordered inventory exactly. Actual Flash memory and output remain to
be measured. This callback has not replaced code in the paused physical run.

## Existing-checkpoint local runner

`flash_generate_run.py` connects complete authenticated decoder/output/token/PLE
loading, bounded coefficient preparation, explicit reconstruction profile,
client-held key creation, local checked evaluator and fixed generation. It reads
only existing checkpoint and inspected operator files; it downloads, installs
and provisions nothing. Native source/dependency checks and exact prompt/profile
matching precede model preparation and key creation. Scratch must be outside
both the pinned source view and resolved shard locations. The plan stays owned
until the provider has stopped; provider shutdown precedes source/scratch cleanup.

Run only after agreeing an uncontested CPU/IO window and applying the external
memory/disk limits. Before source access, the runner requires a dedicated cgroup-v2
leaf containing only itself, kernel memory maximum at most eight GiB, zero swap,
group OOM kill and CPU quota at most eight cores. The child inherits that group.
Limits are read from kernel files; an advertised or caller-supplied enforcement
flag is not accepted. Missing isolation/controllers reject without a fallback.
The client and local provider each use four CPU threads. An
existing shared lock and pause path are mandatory. While paused, the runner
neither initializes native code nor reads checkpoint headers. A new pause request
after admission cancels and closes the attempt at the next cancellation boundary;
it never removes the pause, resumes a partial route or retries with a weaker
profile. Large token/PLE table admission now checks cancellation during each
whole-file hash and cleans previously opened maps on failure.

```sh
systemd-run --user --wait --pipe --collect --unit=a2-flash-research \
  -p Type=exec -p MemoryMax=8G -p MemorySwapMax=0 \
  -p CPUQuota=800% -p OOMPolicy=kill \
  env OMP_NUM_THREADS=4 RAYON_NUM_THREADS=4 \
  /existing/research-venv/bin/python \
  /existing/pinned-code/tools/privacy-concrete/flash_generate_run.py \
  --source /existing/pinned-flash-812d96e1 \
  --reference /existing/inspected-reference/modeling_qwen4_exp.py \
  --scratch /existing/isolated-research-scratch \
  --backend-profile flash-text-all-linear-ntt12-rebuild-prefill5-check \
  --tenant-id public-validation --privacy-id local-research \
  --lock /tmp/flash-attribution-exclusive.lock \
  --pause-file /tmp/a2-flash-validate.pause --resource-wait-seconds 1 \
  --public-fixture flash-public-capital-text
```

This is a prepared execution command, not a recorded successful physical run.
The transient job properties follow the official
[resource-control](https://www.freedesktop.org/software/systemd/man/latest/systemd.resource-control.html)
and [OOM policy](https://www.freedesktop.org/software/systemd/man/latest/systemd.service.html#OOMPolicy=)
contracts. They apply only to the new isolated job; no existing provider unit or
running reference process is reconfigured. Forced kernel termination cannot run
Python cleanup; any remaining owned public scratch directory must be reclaimed
after its job has stopped. Private outputs are released only after validation.
The complete route contains 74,501 projections and 603,472 encrypted tiles across
four steps, plus integrity sampling and full client floating/quantized references.
No practical full-model latency or eight-GiB RSS claim follows from the cache
bound or runnable code. The runner has been lifecycle-tested with mocked compute;
the tiny complete graph/framing and native arithmetic evidence are separate.
The later preparation receipt below verifies complete public materialization,
and the checkpoint owner's four-token clear reference is now available. Actual
encrypted Flash output and combined-model memory remain unverified. The measured
synthetic kernel-work projection below exposes a substantial compute obstacle;
any physical attempt still requires a coordinated resource window. This runner
is a source-controlled entry point, separate from older owner-specific runs.

The retained [five-row native probe](evidence/storage/native-rebuild-prefill5-probe.json)
and [source context](evidence/storage/native-rebuild-prefill5-context.json) record
run 44022072271 at `b86904cdbd6a1c891a9a44b363adafc0a0bcacf1`. Five input rows
followed by three single-row feedback steps took 29.86 seconds on one synthetic
32-by-16 matrix. Every input row's argmax matched its clear control. Coefficient
traffic was zero, with one 1,024-byte provider materialization, 268,537,440 bytes
sent and 101,999 received. This is neither actual Flash nor full-sized-tile,
separate-machine, production or reviewed-security qualification.

The runner also selects `BoundedFloatingReference` for its post-close floating
quality observation. It slices source output channels before FP32 conversion,
retains every input channel in each product and brackets chunks with cancellation.
The reported oracle is `bounded-output-columns-fp32/v1`; BLAS column blocking can
change rounding, so this does not silently replace the prior whole-projection
floating receipt. Quantized greedy equality remains the acceptance gate. Tiny
complete-model tests compare both floating paths within the existing reference
tolerance, and source-backed tests prove actual BF16 tensor reads use row slices.
No actual full-model RSS result is inferred from those tests. The older physical
validator was observed at a 9,071,364-KiB RSS high-water mark; it was not modified
or restarted by this worker, and its current resource budget remains with its owner.

The native workflow's separate `prefill5_full_tile` input (requires
`stream_storage`) selects one fresh-process synthetic 2048-by-2048 matrix probe,
with five encrypted prefill rows and three single-row feedback steps. It records
actual key/wire/RSS/time and separates evaluator round trips from client sampled
replay/check costs. The ordinary small probes are not run in that mode. This
checks the installed maximum tile shape before full-model deployment; it still
cannot qualify complete Flash output, two-process/host peak memory or security.

The retained [maximum-tile probe](evidence/storage/native-rebuild-full-tile-probe.json)
and [context](evidence/storage/native-rebuild-full-tile-context.json) record
successful run 44032233471 at `03cbb3fded5d69c9e6290e92cb97024e95edc9d3`.
Four native steps took 28.17 seconds including 7.22 seconds of key generation;
evaluator round trips totaled 9.55 seconds, client sampled replay 6.49 seconds and
client checks 0.052 seconds. Process peak RSS was 3,111,198,720 bytes. One
8,388,608-byte coefficient file was materialized and removed; public coefficient
traffic was zero, total sent 268,537,440 bytes and received 199,535 bytes. Every
input row's argmax matched its clear integer control; maximum raw integer errors
were 498, 402, 431 and 440. These are noisy decoded integers, not exact plaintext
products. Native Rayon used four threads; the public Torch fixture used two,
as recorded in the artifact. This is a fresh-process operator observation, with
no actual model weights, independent-host memory claim or full-Flash latency
extrapolation. It does not open production admission or qualify the integrity
mechanism against malicious providers.

`flash-checkpoint-reference.json` pins the public clear reference receipt at
`f55c321830d364730bb539ec6becb349cec66efc`. The runner compares all 74,501
ordered prepared coefficient identities against that whole-projection
quantizer's digest before generating keys or starting an evaluator. Missing,
reordered, changed or differently quantized projections reject the attempt.
This checks bounded preparation against a measured clear reference; it does not
turn the clear receipt into an encrypted execution or an independent model
implementation. The reference and admitted execution graph IDs are reported
separately: a context bound belongs to the actual plan, and a different context
does not change the public coefficients. The pin compares coefficient identities,
without asserting equality between graphs with different context bounds.
Post-close numerical generation checks remain required.

Append `--prepare-only` to the same resource-gated runner to measure complete
source admission, graph construction, bounded quantization and plan publication
without generating a key or starting an evaluator. Its report distinguishes
planned tiles from zero executed tiles and asserts neither model execution nor
cryptographic qualification. Temporary projections and the plan remain owned by
the attempt and are removed before exit, including on failure or cancellation.
This is a memory/correctness gate before the longer encrypted generation run.

The [first physical bounded preparation](evidence/storage/flash-bounded-preparation-cancelled.json)
at `eeb688038` completed source admission and entered real coefficient
preparation. A shared-resource pause cancelled it after 952.30 seconds; it
released its lock and removed its temporary projections. The enforced cgroup
memory limit was 8 GiB with zero swap, including page cache. The last observed
process RSS high-water mark was 1,213,161,472 bytes, not a final peak or a complete
model-memory result. The complete coefficient reference did not pass, and no
keys, evaluator or encrypted tiles were created. The existing pause was preserved.

Single-output-tile public proofs now consume the same contiguous coefficient
stream in blocks of at most 1 MiB; strided output tiles retain row-order hashing.
Whole-file and every tile digest still derive from exactly the same owned bytes,
including with fragmented reads. Cancellation brackets each bounded read, and
corruption, truncation or trailing bytes still reject the proof. For a
2,049-by-640 int16 fixture with 2,048-row tiles this reduces nonempty read requests
from 2,049 to four, without changing identities. This is a read-count observation,
not a measured full-model speedup; the cancelled physical attempt used the earlier
implementation and was not patched or restarted.

For an observable long route, `--status-file <scratch>/progress.jsonl` opts into
a new exclusive mode-0600 client-local log. The file must be directly inside the
existing scratch directory, outside immutable source locations; existing files
and symlinks reject before execution. It constructs only plan identity and
sparse received/planned tile counts from the admitted bridge's event sequence.
Observer dictionaries, prompts, outputs, keys, ciphertexts, exceptions and
integrity/check outcomes never enter these records. Reaching the total tile count
means transport progress, not acceptance. The CLI writes `validated` only after
the complete run returns successfully through closure, numerical validation and
cleanup; preparation-only completion is separately labeled `prepared` with zero
received tiles. Every record keeps production/security qualification false.
The optional log is caller-owned and remains after exit; temporary projections
are still removed. It is not provider evidence, attestation, a reviewed leakage
guarantee or a replacement for the final execution receipt.

The client ephemeral cache owner enforces one active projection materialization
per prepared graph, including reentrant and concurrent callers. A contended
caller rejects before source quantization or temporary allocation. The slot
remains held through owned-file cleanup and is released on allocation failure,
exception or cancellation, so later attempts can reuse the public preparation.
Retained public-weight readers keep their existing behavior. This resource bound
does not merge request state or change privacy/integrity admission.

The [complete physical preparation](evidence/storage/flash-bounded-complete-preparation.json)
and [operator context](evidence/storage/flash-bounded-complete-preparation-context.json)
at `131c7dd25` passed under the enforced 8 GiB/no-swap/eight-core cap. All 74,501
ordered coefficient identities, including the output head prepared in bounded
columns, exactly matched the observed whole-projection reference. The complete
plan schedules 603,472 encrypted tiles over five-row prefill and three decode
steps. Source admission hashed 456,738,193,872 bytes including repeated shards;
public quantization/proof preparation took 2,224.47 seconds, with 2,837.11 seconds
reported through plan completion. Process lifetime peak RSS was 4,261,105,664
bytes. The unit journal reports 2,842.83 seconds wall time and an 8 GiB cgroup
peak including page cache. All temporary projections and the plan were removed
on exit, and the shared resource window was released.

This qualifies the complete bounded public preparation path for that pinned
attempt. It created no keys or evaluator, executed no encrypted tiles or model
generation, and does not qualify combined client/provider memory, latency,
malicious-provider integrity or production privacy. Later optional progress
logging and concurrent cache-lease enforcement were tested separately and were
not deployed over the running pin. The resource owner holds the following window;
encrypted generation requires a subsequent coordinated attempt.

The encrypted runner now records client-local resource phases after public
preparation, key creation, provider readiness and generation validation. A
background observer samples only the initially exclusive, kernel-enforced
cgroup, including its evaluator child. It reports a sampled sum of process RSS
separately from the kernel cgroup peak, which includes file cache. RSS sums count
shared mappings twice and sampling may miss short peaks; missing processes and
observer failures remain explicit, rather than becoming zero-memory claims.
These observations contain no process IDs, private tensors, outputs, exceptions
or integrity outcomes, and never confer hardware or security qualification. The
observer stops after the provider and before owned model sources are released.

`flash_compute_budget.py` inventories the installed four-step schedule: 15 unique
tile shapes, 150,868 tiles per step and 603,472 over the route. The native
workflow's explicit `process_resources` option runs those shapes in separate
client/evaluator processes, using real native encryption and synthetic public
matrices, with five-row prefill followed by three independent feedback steps per
shape. A physically touched 1,161,002,304-byte client allocation represents the
logical size of all Flash scale/zero/sum quantizer arrays before tensor-free
identity retention. It remains an explicit additional pressure scenario, rather
than a requirement of the current evictable representation. It does not instantiate
the actual graph, request state, proof catalogue or full-model mappings. Both
processes inherit one fresh enforced 8 GiB/no-swap/eight-core cgroup; existing
storage probes are excluded from this mode.

The report separates actual evaluator-call times, key/wire costs, client replay
and checks, sampled RSS sums and kernel cgroup memory. It weights the synthetic
evaluator calls by the installed geometry counts only as a kernel-work planning
estimate. Actual trained data, client computation, model/cache I/O, transport and
hardware differences are excluded, so this is neither measured full-model
latency nor complete Flash combined-memory qualification. Only the fixed
test-only fixture executable substitutes synthetic source geometry; normal
backend loaders remain unchanged and reject that scope. No secret key or
plaintext feedback is passed to the evaluator process. Production and security
qualification stay false, including when all synthetic controls match.

The tensor-free identity regression checks use the real installed quantizer and
complete tiny graph. Weak references show that preparation retains no scale,
zero or sum tensors after each ephemeral projection. Value, shape, dtype, bias,
device and nonfinite metadata changes reject before the lease escapes. The same
canonical metadata digest protects the bounded quantized oracle. Existing
four-step storage, source identity, corruption, cancellation and cleanup checks
continue to pass. This establishes reference release, not measured full-model
RSS savings; a new physical run is still needed for the latter.

The exact native run at `1a23a6195` completed successfully in CI
[44058653705](https://github.com/clocksmith/doppler/actions/runs/44058653705).
Its unchanged public artifact and provenance are retained as
`evidence/storage/native-separate-process-probe.json` and
`native-separate-process-context.json`. All 60 native encrypted calls and greedy
integer controls passed in 182.63 seconds. The two processes' sampled RSS sum
peaked at 5,172,228,096 bytes (shared mappings count twice); kernel cgroup memory
peaked at 4,558,774,272 bytes. The evaluation key was 268,435,536 bytes, total
framed traffic was 269,952,836 bytes sent and 2,061,602 received, and no public
coefficient bytes were transmitted. These numbers include the explicit legacy
metadata pressure allocation and do not qualify combined full-model memory.

Weighting the measured synthetic evaluator calls by the current 603,472-tile
four-step schedule gives 1,134,819.90 seconds, or about 13.13 days of evaluator
kernel work on this runner. This is a planning projection with all the exclusions
above, not measured Flash generation latency, a hardware-independent bound or a
performance claim. It exposes a practical compute obstacle before committing a
shared machine to a full route. The installed all-expert schedule, its private
routing and validation requirements remain intact; this result does not justify
skipping experts, disclosing routing or weakening integrity. A practical full
Flash path still needs substantial compute reduction, actual checkpoint output
and combined-model resource validation, in addition to security qualification.

The 13.13 days covers the **whole pinned four-output request**, not one token.
There are 150,868 tiles in each step. Five-row prefill produces the first output;
three further one-row decode steps produce the remaining outputs. The weighted
prefill evaluator work is 709,158.07 seconds (196.99 hours); the three decode
steps are 142,028.44, 141,707.49 and 141,925.90 seconds (about 39.4 hours each).
The separate derived record `native-separate-process-cost-scope.json` binds the
unchanged input artifact digest and exposes per-step and projection-family sums.
All-expert operations account for 147,456 of the 150,868 tiles per step and
1,108,097.73 of the projected 1,134,819.90 seconds. This cost cannot be removed by
silently skipping unselected experts or revealing the private routing decision.

This projection reuses one seeded synthetic matrix and one timing observation
for each of 15 shapes and four steps. It assumes sequential calls with no inferred
tile-level parallel speedup or repeated-trial confidence interval. Trained
coefficients, private activations, client nonlinearities/checks/replay, keys,
admission, cache materialization and network time are not included. In particular,
it is neither a measured per-token latency nor an average request latency.

The concrete cost-reduction candidate first targets serial CRT packing work in
the exact native arithmetic, preserving the existing tile route and checking
complete serialized-byte equality against both the previous serial algorithm
and original arithmetic. Packing-group parallelism uses the existing bounded
thread pool and public geometry; it does not skip experts or expose routing.
The candidate has a distinct native source/build version and must be measured
before projecting any reduction. Enlarging input tiles is not a free alternative:
the current 12-bit/35-bit signed-message bound already constrains the tile to
2,048 inputs. Batching projections or changing packing boundaries would require
a separately bound plan, coefficient mapping, integrity/noise checks and full
numerical revalidation; none is silently applied to the pinned request.

Append `--readiness-only` to the resource-gated runner above for the next bounded
physical memory gate. This mutually exclusive alternative to `--prepare-only`
authenticates the complete model, compiles and verifies every coefficient against
the pinned clear inventory, creates the client key and completes the local
provider's source-verified readiness handshake. It then verifies that the route
cursor is untouched, closes the owned provider and observes combined resources.
It sends the public evaluation key but no inference ciphertext, executes zero
model tiles and returns no generated output. The receipt and optional progress
label say `provider-ready`, never `validated`; production admission stays closed.
Cancellation, rejected identities and cleanup failures return no successful
receipt. This is an implemented lifecycle-tested gate awaiting its coordinated
physical run, not evidence that actual combined Flash memory already fits.
