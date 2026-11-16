# Flash encrypted execution and client operator integration

Production cryptographic admission remains closed. The paths below are source-pinned
research implementations with scoped functional evidence; none establishes full
Flash generation, independent malicious-security review or trusted hardware.

## Current execution evidence

| Path | Actual checkpoint weights | Verified outcome |
| --- | --- | --- |
| Fixed public expert 0 | First-layer gate/up/down | Four encrypted tiles and quantized-reference check passed |
| First-layer DeltaNet | Five projections, convolution, norm and recurrence | Four cached component steps; 104 encrypted tiles passed |
| MLP hyperconnection | Three projections and grouped norm | Four identity-block feedback updates; 60 encrypted tiles passed under a separately versioned precision profile |
| Complete first-layer MoE | Shared/router and all 512 experts | 1,029 encrypted modules / 2,058 tiles implemented; stock reached six-hour limit; distinct native NTT attempt running |
| Complete first-layer MLP branch | Hyperconnection plus all-expert MoE and residual | 1,032 modules / 2,073 planned tiles composed; actual FP32 sparse-source comparison is separate; full ciphertext composition not yet run |
| Full attention / QSA | Five actual layer-3 projections and client normalization | Four cached public-input steps; 96 encrypted tiles passed under a distinct NTT profile; no token generation |
| Ngram / PLE | Small local-table FP32 source fixtures | EOS-aware cached updates, local lookup and history isolation checks passed; actual Flash table unsupported |

The [dense Qwen results](FULL-QWEN.md) remain separate: stock exhausted its
six-hour budget, and both int8 NTT variants failed the third-step numerical gate
after 1,161 verified tiles. The distinct twelve-bit profile completed all 24
Qwen2.5-0.5B decoder layers plus head over four autoregressive steps and 1,548
encrypted tiles, matching greedy reference tokens on one public prompt. This is
full small-Qwen hybrid functional evidence, with client nonlinear operations and
full replay/reference validation; it does not establish full Flash output or
cryptographic qualification. Component results cannot substitute for either.

## Actual measured components

[Recorded runs and artifact digests](evidence/flash/operators-context.json) bind
these measurements to their exact revisions and public fixtures.

| Measurement | [DeltaNet four-step result](evidence/flash/delta-four-step.json) | [Hyperconnection four-step result](evidence/flash/hyper-four-step.json) |
| --- | --- | --- |
| Encrypted tiles | 104 | 60 |
| Process elapsed | 1,304.25 s | 783.96 s |
| Client encrypted replay | 649.45 s | 387.07 s |
| Evaluator round trip | 648.00 s | 388.91 s |
| Fresh evaluation key | 134,217,808 bytes | 268,435,536 bytes |
| Application bytes sent | 135,317,134 | 269,208,200 |
| Application bytes received | 1,276,835 | 1,054,392 |
| Largest encrypted-versus-quantized output error | 0.000328 | 0.007858 |
| Largest quantized-versus-floating output error | 0.001394 | 0.593400 |

Elapsed excludes dependency installation and the earlier public source download.
These runs used process pipes on a trusted test-client host, not a separately
operated network service. They did not retain RSS measurements. The hyperconnection
fixture uses identity-block feedback, so its four updates are not Flash token
generation. Its substantial quantization drift remains a quality limitation;
passing the encrypted-versus-quantized bound does not establish floating-point or
BF16 model parity. No full-model throughput is derived from these numbers.

[Expert-0 evidence](evidence/flash/expert0-sine-2560.json) records 42.04 s execution,
including 21.44 s replay, and maximum encrypted-versus-quantized error 0.0003564.
The [cache-integrity regression](evidence/flash/expert0-cache-binding.json) passed
a second four-tile run in 41.62 s, including 20.95 s replay, with error 0.000676.
Its [measurement context](evidence/flash/cache-binding-context.json) states memory
and transport limits. Expert 0 is fixed/public; this result tests no private expert
selection.

The [default-precision negative probe](evidence/flash/hyper-default-precision-failure.json)
failed the unchanged numerical bound, with a same-input down-projection error of
0.599. The explicit `glwe-precision-48-level4/v1` profile raises input/output storage
precision and changes packing decomposition while retaining encryption noise and
computation width. Hyperconnection/MLP protocol versions change and the larger
key requires a bound 300 MB frame allowance; legacy paths retain 150 MB and their
original parameters. Native tests cover exact public products and byte replay;
they do not prove exactness for every circuit or an unchanged security level.
See the [security boundary](SECURITY-BOUNDARY.md).

## Source and weight ownership

All actual components use official `Qwen/Qwen3.8-Flash-Next` revision
`812d96e10a0dd82b69d749e68693698cf6c36daa`. Installed scopes bind config/index
hashes, complete LFS SHA-256 values and exact BF16 tensor geometry:

- `moe`: two complete first-layer shards, 5,187,961,632 weight bytes.
- `mlp-hyper`: the complete third shard and four hyperconnection tensors.
- `mlp-block`: the union of MoE and hyperconnection tensors.
- `linear-attention`: the complete first shard, 1,040,155,944 bytes, with nine tensors.
- `full-attention`: the complete layer-3 shard, 1,935,433,136 bytes, with nine tensors.

`flash_weights.py` admits whole checksum-verified objects. Range/header metadata
used during development is never admitted as weight material. Existing corrupt
caches reject. Tensor reads check file identity before and after the copy; experts
are sliced locally rather than loading the whole bank. Client replay binds public
tile digests from owned verified snapshots, so a mutable mmap cannot redefine its
circuit. Only the small digest table is retained per request.

## Complete route and request state

Research profiles are installed code, not provider-advertised flags. Model,
source, program, dependency, parameter, session, key, sequence and tile identities
are bound before encrypted inputs are sent. Protocols for expert 0, DeltaNet,
hyperconnection, all-expert MoE, composed MLP and QSA are distinct. Unknown profiles
fail before key export. No retry or plaintext fallback substitutes another path.
The existing SSH manifest rejects incompatible underlying protocols.

The complete MoE route visits and replay-verifies every expert before private
selection affects the client result. Combined gate/up lowering preserves a fixed
route. Shared/router composition, normalization and selection stay on the client.
The sparse official-source implementation is used only as a trusted-client clear
reference with verified local expert slices, never as a selected-only provider path.

`flash_request_state.py` owns tenant/privacy/model/key/request/mode binding and
terminal cancellation. DeltaNet, QSA and PLE own their private state stores;
shared public weights hold no request cache. Immutable projection metadata and
owned input/output copies cross the trusted-client adapter boundary. A missing
adapter in `verified-projections` mode cannot fall back to clear execution. The
callback is trusted local code, not an encryption or remote-integrity appraisal.

## Inspected client graphs

`flash_reference.py` loads only enumerated declarations from the hash-pinned
official Transformers source. It executes no downloaded imports, model loader,
fused-kernel hub or network dispatch. Client graphs cover grouped hyperconnection
normalization/residuals, shared/routed MoE, gated-delta recurrence/convolution,
eager full-attention/QSA, and EOS-aware ngram/PLE gating and convolution.

The pinned Torch 2.3 lane declares two compatibility transforms: one QSA scatter
index is promoted to int64, and three `nn.Buffer` assignments become equivalent
registered buffers. The loader requires exactly the inspected sites and each
affected program binds its transform identifiers. Integer/hash values, weights
and projection order are unchanged. Comparisons use pinned source with these
declared transforms, not an unmodified upstream runtime claim.

QSA currently admits only bounded FP32 eager execution with an explicit rotary
width. Boolean, noncausal and fully masked query rows fail before projection.
PLE admits an explicitly budgeted materialized embedding or a mapped row store
that verifies all complete local shards before accepting private token indices.
Neither path has a remote row-fetch/cache-miss fallback. The checkpoint
shard-to-logical-row mapping is now source-pinned and fixture-tested below, but
the actual huge table has not been provisioned or executed.

## Running and remaining qualification

Workflows isolate CPU/model activity from local/fleet performance work. Large
encrypted runs require an explicit marker or manual dispatch; ordinary pushes
skip them. Only source-constructed public-fixture results are uploaded. Keys,
raw vectors, source weights, private caches and raw diagnostics are excluded.

`flash_mlp.py` requires an explicit `public-reference` or `encrypted` mode and a
pinned `--reference-source`. Clear mode cannot report encrypted completion. Its
actual FP32 sparse-source gate runs before key generation. Full ciphertext MLP
composition has not been launched; the stock all-expert job exhausted its budget
and its distinct native NTT candidate remains in progress.

Remaining gates include the changed native all-expert run; complete Flash weight
provisioning and actual mapped-table qualification; actual-weight all-layer
assembly and encrypted autoregression; complete-model quantization, quality and
cost evidence;
authenticated separate-machine execution; and independent cryptographic,
malformed-input, side-channel and active-adversary review. Full client encrypted
replay still gives no client compute saving. No attestation or production security
claim follows from these functional results.

The actual first-layer composed MLP also passed a separate client FP32 comparison
against the pinned official sparse reference: maximum error
`7.450580596923828e-09` on the fixed public sine input, visiting all 512 experts.
The quantized-versus-floating error was `0.013162888586521149` and elapsed time
84,928 ms. The [retained evidence](evidence/flash/mlp-source-reference.json)
records zero encrypted tiles: this validates the composed client graph, not
ciphertext execution, model tokens or a full Flash backend.

## Local n-gram storage mapping

The [installed storage inventory](flash-ngram-storage.json) now binds all 128
numeric checkpoint partitions to 320,001,536 by 160 BF16 rows. Its 33 complete
shards total 104,298,732,704 bytes; the logical table is 102,400,491,520 bytes.
Only bounded safetensors headers and public LFS hashes were acquired. Table
weights have not been provisioned or executed.

The mapping follows the exact pinned Transformers
[conversion registry](https://github.com/huggingface/transformers/blob/cf13eb97cddb3655cd83bada097a7ecfcf108ace/src/transformers/conversion_mapping.py)
and [loader](https://github.com/huggingface/transformers/blob/cf13eb97cddb3655cd83bada097a7ecfcf108ace/src/transformers/core_model_loading.py):
numeric shard suffix order, concatenation along rows. Header geometry, offsets,
whole-file sizes/hashes, config/index identity and source hashes are recorded.
Source-backed tests compare the actual configuration's meta table shape and
numeric order, and compare local row reads with the official concatenation on a
small public fixture. Complete real-table loading and model parity remain gates.

`local_ngram_rows.py` provides bounded local BF16 row reads without constructing
one contiguous 102 GB tensor. It verifies every complete shard before accepting
indices, maps only local files, checks source mutation before/after reads, owns
its handles and rejects missing material, cancellation or budget failure without
network fallback. It caches neither selected indices nor private outputs. Local
filesystem access patterns remain visible to the trusted client host; this is not
PIR or protection from that host. The PLE client adapter can now bind a verified local row store through
`ClientPLE.from_local_rows`, loading only the small explicitly supplied public
parameters and no contiguous full-table tensor. Request-owned leases bind lookup
to the active tenant/privacy/model/key/request context; cancelling one lease
cannot close another request's shared public mappings. Source comparisons use
complete tiny BF16 files and remain separate from actual Flash table qualification.
Actual PLE projections/weights,
all-layer composition and full Flash autoregression remain unavailable.

## Actual cached QSA candidate

`flash_qsa_model.py` binds the first full-attention layer (layer 3) to the complete
five-projection QSA route. Its whole BF16 shard is pinned by SHA-256 and size
(1,935,433,136 bytes). The client uses the inspected default text rotary methods,
QSA indexing, attention/normalization/output gate, separate per-mode KV/indexer
state and full encrypted replay. Its installed NTT profile has a distinct
protocol, native source/binary identities and no substitution into stock profiles.
The fixed public-input fixture is a four-row prefill followed by three cached
single-row updates. It is not model token generation or full Flash inference.

The isolated `flash-qsa.yml` workflow builds the retained native lock, verifies
native ciphertext equivalence and admission before provisioning the complete
public shard, then runs that component. Trigger only manually or with
`[flash-qsa-e2e]`; original Qwen/MoE runs remain untouched. The [actual result](evidence/flash/qsa-four-step.json) passed at revision
`e4d79a3cf70841eb128f1e929225fccc1c743600`: 96 encrypted tiles, 170.58 s process
elapsed, 80.37 s full client replay and 80.72 s evaluator round trip. Prefill took
91.12 s; the three single-row updates took 23.54, 23.64 and 23.63 s. The evaluation
key was 134,217,808 bytes; application frames sent/received 135,958,568/1,985,042
bytes. Largest encrypted-versus-quantized output error was 0.001262, and largest
quantized-versus-floating error was 0.014168. Client lifetime peak RSS was
1,695,096,832 bytes, including preparation; no provider peak is claimed. Setup
installation/download is excluded. This is one public component fixture on process
pipes, not full-model throughput, separate-machine security or Flash tokens.
Production admission remains closed.

`python tools/privacy-concrete/inventory_flash_ngram.py --check` reproduces the
installed storage mapping from pinned config/index, public revision-specific LFS
metadata and bounded headers. `--output <path>` writes a reviewable candidate;
neither command downloads table weights or enables a backend. The generator also
verifies the pinned conversion sources and resolves the absent seed from the
exact configuration declaration rather than introducing a runtime default.

## Wired PLE candidate and remaining data gate

`flash_ple_model.py` now composes the actual pinned PLE parameter scope, complete
local table store, EOS-aware token/conv cache, both encrypted projections and full
ciphertext replay under `flash-first-ple-ntt`. Its immutable plan binds the table
inventory identity as well as source, program, parameters, native implementation
and projection schedule. All table files must already exist, have the complete
expected sizes and verify their whole hashes before any private key is created.
There is no download, row-fetch fallback or partial-table admission in this driver.
The CLI requires an explicit `--local-storage-limit-bytes` budget.

The public parameter scope contains six BF16 tensors and three int64 hashing
buffers from shards 5 and 37. Integer buffers remain owned int64 values; converting
multipliers through floating point would change the model's hash function. The
local source tests cover the full PLE client graph across EOS and cached updates
using complete small BF16 tables, including request cancellation and shared-table
ownership. These are implementation checks, not the actual 51B table or Flash
model execution.

Actual PLE execution/quantization costs are still unavailable: no complete 104.30
GB table file set has been provisioned on an authorized execution host. The wired
path remains unqualified and outside production admission until real material,
model-wide correctness, security and deployment gates pass.

## Complete client reference assembly

`flash_model_graph.py` now assembles trusted materialized source components from
embedding through every decoder, final stream mixer and vocabulary head. It keeps
a fixed all-expert projection route, client text rotary/attention, PLE and
hyperconnections, and owns no private cache in shared weights.
`flash_model_state.py` routes each request's cache calls to the existing
DeltaNet/QSA/PLE owners. A failure clears the entire request, and all intermediate
projection geometry, dtype and finiteness checks occur before a trusted client
projection callback receives it.

The new source comparisons use three small public decoder layers mixing both
attention families and PLE, four autoregressive token steps, complete-prefix
sparse-source comparisons, concurrent cache isolation and terminal cancellation.
They execute the inspected decoder forward with an explicitly declared eval-only
`nn.Module` base in place of the upstream gradient-checkpointing wrapper. This
assembly is an unpadded text client reference, not actual 48-layer Flash weights,
full-model encrypted execution, an authenticated network backend or production
qualification. Actual full Flash artifacts and model-wide quantization evidence
remain required.

The complete client reference also accepts a local mapped PLE table inside a
mixed-layer autoregressive request. Its model identity includes the local table
inventory, not an empty placeholder tensor. The source comparison checks four
cached autoregressive steps against the sparse full-prefix decoder and verifies
that request closure releases its lease while preserving the shared public table.
This uses complete small public BF16 files; it does not remove the actual Flash
checkpoint/table provisioning gate.

## Public evidence after incomplete all-expert attempts

Future dense-MoE workflow attempts construct `flash-moe-public-evidence.json` from
the fixed public fixture and its client-local progress. The collector derives the
2,058-tile route from the installed lowering and accepts only contiguous sequence
numbers with the expected public projection names. It exports observed count and
last public projection, closed failure codes and narrowly typed success metrics;
raw diagnostics, private tensor fields and exception text are excluded. Missing
progress or conflicting/incomplete success claims cannot become a completed result.
These records distinguish observed verified tiles from total computation performed.

This change does not restart or alter a running older workflow. An older attempt
that uploads only its final result may still leave no tile-progress artifact if
killed before returning. Complete MoE evidence remains one model component and
cannot establish complete Flash execution, model quality or cryptographic security.

## Complete text-weight source scope

`flash-text-weights.json` now inventories every actual text-generation parameter
from the pinned checkpoint index: **1,294 tensors across all 48 decoder layers**,
all 96 packed expert banks, input embeddings, the final hyperconnection mixer,
vocabulary head and the complete PLE table/hash buffers. Vision and MTP tensors
are excluded from this text tensor map. Because their objects share the selected
shards, acquisition still requires **all 131 whole files: 360,000,192,888 bytes**.
The selected text tensor payloads total 353,887,799,320 bytes. These counts describe
source storage, not client peak memory or encrypted wire traffic.

The bounded generator `inventory_flash_text.py` verifies the pinned configuration
and index, obtains complete object hashes/sizes and fetches exact safetensors header
ranges. It downloaded no weight payloads. The PLE generator shares that acquisition
owner and reproduces its existing inventory exactly. A range header is discovery
metadata; it never substitutes for the whole-file SHA-256 required by `FlashSource`.

The explicit `flash_weights.py --scope text --destination /authorized/source`
provisioning path performs disk admission and complete-object verification before
materialization. Packed expert banks require an explicit expert slice at every
layer, including layer 47; their selected BF16 data becomes an owned FP32 tensor.
Legacy component source scopes retain their original contents. The text scope is
a source/materialization capability, not an installed encrypted full-Flash backend.
No complete weight set has been provisioned or executed by this inventory step.
Actual complete-model streaming/assembly, quantization and autoregressive encrypted
execution remain unverified; production cryptographic admission stays closed.

## Pinned client tokenizer path

`prepare_flash_tokenizer.py` acquires only the exact public tokenizer/configuration
files (12.8 MB tokenizer JSON plus its configuration), without model weights.
`flash_tokenizer.encode` and `decode` require local verified files, explicit token
limits and the installed fixed fast tokenizer implementation. Each call owns a
fresh tokenizer so private BPE/text caches are not shared across requests. Source
identity spans construction and tokenization/decoding; mutation, cancellation,
unsupported implementations and out-of-vocabulary IDs fail before returning data.
The helper runs neither downloaded Python nor chat-template code and performs no
network fetch during private text processing. Reference release is not zeroization.

The isolated adapter workflow qualifies the actual public vocabulary and records
one fixed public roundtrip with source/dependency identities and elapsed time.
Tokenization is client text I/O, not Flash model execution or encrypted inference.
The complete Flash model/weight/route and security acceptance gates remain closed.

The pinned tokenizers 0.19.1 [BPE implementation](https://github.com/huggingface/tokenizers/blob/v0.19.1/tokenizers/src/models/bpe/model.rs)
owns its [cache map](https://github.com/huggingface/tokenizers/blob/v0.19.1/tokenizers/src/utils/cache.rs)
inside each model instance. Fresh, locally loaded instances provide separate
cache ownership for these helper calls; this does not establish memory zeroization
or isolation from the trusted client host.

The [actual public tokenizer observation](evidence/flash/public-text-io.json)
roundtripped “The capital of France is” as IDs `[760, 6511, 314, 9338, 369]` under
Transformers 4.44.2/tokenizers 0.19.1. Encode plus decode, each with a fresh tokenizer
and source verification, took 1,490 ms in one hosted CPU observation. This is text
I/O rather than model-generated output. The [context](evidence/flash/public-text-context.json)
binds the exact revision/run and artifact hash.

The complete text-source scope now includes these pinned tokenizer files and
binds their source identity to the actual embedding/head vocabulary. Provisioning
accounts separately for auxiliary metadata/tokenizer bytes and whole weight-shard
bytes. This closes the source association; it does not complete encrypted model
assembly or validate a full Flash forward pass.

## Mapped client input embedding integration

The input embedding now uses the same complete-file BF16 row-storage owner as PLE,
with separate installed layout contracts. The existing PLE interface is preserved.
`LocalTokenEmbedding` reads owned rows through a request/context-bound lease;
failure closes only that request's owned lease and leaves shared public mappings
available to other requests. The complete client graph includes the mapped input
source in its model identity. It also binds every supplied JSON configuration
field and rejects changes before input embedding or projection callbacks.

The actual input tensor is `[248320, 2560]` in checkpoint object
`model-00130-of-00131.safetensors`; the whole object is 2,974,788,136 bytes.
`flash-embedding.yml` qualifies the actual tokenizer-to-input-activation path on an
isolated hosted CPU, comparing mapped results with independent safetensors rows.
It downloads that entire public object before accepting any indices and uploads
only the fixed public fixture's shape/digest/cost evidence. No full FP32 embedding
table is constructed by the adapter. Client-host/filesystem access patterns remain
visible; this is neither PIR nor encrypted lookup.

The separate source-reference tests exercise four autoregressive updates through
the complete mixed-layer small graph with mapped input embeddings and isolated
request leases. Actual full Flash decoder/model execution, cryptographic routing
and security qualification are separate uncompleted gates. An input activation or
reference-fixture pass must not be reported as a generated model token.

The [actual input-embedding observation](evidence/flash/input-embedding.json)
produced a `[1, 5, 2560]` FP32 tensor exactly equal to independent safetensors row
reads for the pinned public prompt. Whole-file verification plus client text I/O
and lookup took **2,617 ms**, excluding acquisition; client lifetime peak RSS was
**465,014,784 bytes**. The adapter read 25,600 BF16 row bytes from the already
verified 2,974,788,136-byte public object. These are one-observation prefix costs,
not complete-model inference or latency distributions.

All **38** source-reference checks passed on the same revision, including four
autoregressive updates through the complete small mixed-layer graph with mapped
input rows, configuration-change rejection before lookup/disclosure, and owned
lease cleanup across requests. The [context](evidence/flash/input-embedding-context.json)
separates actual checkpoint rows from small-graph reference evidence and records
the exact run/artifact identities. Neither result is an encrypted Flash forward
pass or generated model token.

## Completed stock MoE budget and explicit native candidate

The [original stock all-expert attempt](evidence/flash/dense-moe-stock-six-hour-budget.json)
reached GitHub's six-hour limit and was cancelled. Its whole-object provisioning
and lowering tests passed, but its older workflow uploaded an empty final-report
file without progress. Verified tile count is unavailable, not zero; there is no
completed encrypted MoE result from that attempt. The exact artifact was inspected
and retained before scheduling a changed implementation.

`flash-dense-moe-ntt` is a distinct installed research protocol for the same complete
1,029-linear/2,058-tile, 512-expert route. It changes the actual arithmetic
implementation to the already measured exact-NTT variant, retaining stock crypto
parameters, int8 quantization, complete all-expert access, client-local selection,
full deterministic replay and the original numerical acceptance limits. It cannot
silently replace the stock profile. Missing native identity rejects before source
preparation or key creation; safe evidence requires its actual native source and
binary identities and cannot be relabeled as a stock run.

The new isolated workflow repeats native ciphertext equivalence before the real
all-expert attempt and retains bounded progress on failure. It has a separate
non-cancelling group and a two-hour budget; unmodified stock work is not retried.
Native-kernel fixture speedups do not establish complete MoE latency or correctness.
This remains one actual model component, not full Flash generation, authenticated
separate-machine deployment or independently qualified cryptographic privacy.
