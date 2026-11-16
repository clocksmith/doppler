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
| Complete first-layer MoE | Shared/router and all 512 experts | 1,029 encrypted modules / 2,058 tiles implemented; stock reached six-hour limit; complete native NTT public fixture passed (2,058 tiles) |
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
and its distinct native NTT candidate completed the separate component fixture below.

Remaining gates include full composed MLP encrypted execution; complete Flash weight
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

## Completed native all-expert result

The [retained complete NTT MoE result](evidence/flash/dense-moe-ntt-complete.json)
passed on one public sine vector: all **512 experts**, shared expert and router,
**1,029 linear modules / 2,058 verified encrypted tiles**, followed by client-local
top-10 selection. Elapsed time was **2,357.63 s** (39.29 minutes), including
**2,227.73 s** encrypted execution. Client encrypted replay took **1,080.96 s** and
evaluator round trips **1,097.63 s**. Application frames sent/received were
**155,964,370 / 23,350,368 bytes**, including a **134,217,808-byte** evaluation key.
Component peak RSS was not retained; native microfixture RSS is not attributed to it.

Maximum encrypted-versus-quantized error was **0.0005921** and maximum
quantized-versus-floating error **0.0055964**, under the unchanged component bounds.
The [context](evidence/flash/dense-moe-ntt-context.json) binds revision, run, artifact,
native source/binary and scope. This is one actual first-layer MoE component on
process pipes; it is not a full Flash decoder, generated token, network deployment,
model-wide quality result or cryptographic qualification. Its historical driver
performed reference acceptance before `finish`; current drivers use the closed-route
validation boundary below. The result does not establish that newer ordering's
full-size performance.

## Reference checks after transport closure

All actual component drivers now share `complete_verified_route`: execute the
fixed admitted route, finish and close it, then apply the same numerical acceptance
checks. Cached outputs and hyperconnection projection diagnostics remain client-local.
A numerical mismatch cannot choose an earlier provider stopping point or select
`finish` versus close, and no partial result is released. Integrity, cancellation
and runtime failures remain immediate; timing and other failure leakage still
require review. New reports mark `after-route-close/v1`. See the
[security boundary](SECURITY-BOUNDARY.md#observable-reference-validation-feedback).


## Complete checkpoint configuration and geometry assembly

The raw checkpoint calls its indexed-attention layers `full_attention`. The pinned
configuration class normalizes those names to `indexed_attention` and supplies
missing declared defaults, including the n-gram seed. Passing the raw JSON directly
as a namespace therefore cannot correctly assemble the complete client graph.
`flash_text_config.py` now executes only that hash-pinned class's literal defaults
and its two inspected local normalization/architecture methods, with an explicit
eager reference implementation. Downloaded imports, decorators and the upstream
configuration base implementation are excluded; this is a bounded client-reference
contract, not a general Transformers configuration loader. Raw input is copied.

`flash_model_assembly.py` builds every actual decoder, embedding, mixer and head on
Torch's meta device and checks every destination against the complete text-weight
inventory. It also checks the logical PLE table against the source-backed split-row
mapping. This is a pre-materialization geometry check; no checkpoint weights are
loaded, forward operations executed or model tokens generated. Its reports state
`geometryOnly: true`, `weightsMaterialized: false` and `forwardExecuted: false`.
The isolated source-reference workflow checks it alongside the existing small-graph
autoregression/ownership regressions and retains bounded public geometry evidence.
Actual out-of-core weight materialization and full Flash autoregression remain gates.


The [actual four-step closed-route QSA run](evidence/flash/qsa-four-step-closed-route.json)
passed **96 encrypted tiles** with the new validation boundary. Elapsed time was
**191.59 s**, client replay **90.69 s** and evaluator round trips **91.58 s**;
maximum encrypted-versus-quantized error was **0.0008691**. Client lifetime peak RSS
was **1,684,160,512 bytes**. The [context](evidence/flash/qsa-closed-route-context.json)
binds revision, artifact and scope. This is one genuine component integration of
the phase-ordering fix, with unchanged numerical limits. The driver control tests
also exercise reference rejection at multiple Qwen steps after all four encrypted
steps complete; those use real Torch with mocked model/transport and are not
additional full-Qwen encrypted executions or constant-time/security qualification.


The [retained complete geometry result](evidence/flash/complete-client-geometry.json)
matched all **1,294 checkpoint tensor entries** against **1,167 assembled state
entries**, with 128 PLE partitions represented by one logical table. This covers
**48 decoder geometries: 36 linear-attention, 12 indexed-attention and 96 packed
expert banks**. All **42 source-reference checks** passed, including existing
small-fixture autoregression and ownership tests. The [context](evidence/flash/complete-client-geometry-context.json)
records the exact source revision/run/artifact. No actual checkpoint weight payloads,
forward operations, encrypted full-model steps or generated tokens were involved.
The resolver now requires an explicit complete layer schedule; it does not infer
an omitted schedule through the excluded upstream configuration base.


## Owned source staging and concrete storage options

`flash_stage_source.py` now stages one explicit decoder or output-head part using
the existing whole-object acquisition, index, checksum and tensor verifier.
Every one of the 48 decoder parameter sets is covered; input embeddings and PLE
rows retain their separate complete local row-store owners. The staged contract
binds the full checkpoint/table identity and lists the separately owned PLE table
tensors. It cannot masquerade as a complete executable model or extend a legacy
component scope. Unsupported parts reject before filesystem/network work.
Temporary stages close terminally and remove only their own files; returned tensor
copies remain owned after source cleanup.

`flash_storage_plan.py --quantization-bits 8` (or `12`) computes whole-object disk
budgets from the pinned inventory without downloads. PLE plus input row stores
require **107,273,520,840 bytes**. With those persistent stores, peak staged raw
source is **119,340,160,567 bytes**. Conservatively including one stage's complete
coefficient cache gives **121,934,139,671 bytes** at eight bits or
**124,528,118,775 bytes** at twelve bits. Complete resident source plus all stage
cache upper bounds is **485,121,142,223 / 610,229,096,783 bytes** respectively.
These estimates exclude filesystem reserves, dependencies/build products, runtime
memory and provider storage; no deduplication saving is assumed.

The source-stage owner is implemented and fixture-tested. Per-stage coefficient
cache eviction/admission in the encrypted client/evaluator is **unfinished**;
the smaller disk figures do not describe the current full-model backend. Actual
Flash forward/quality/security qualification is still unavailable. No large model
files were acquired during this resource investigation.


## Public coefficient binding after cache eviction

`public_weights.py` now derives complete-file and strided-tile SHA-256 values from
the **same single owned byte stream**, reading at most one output-tile row of
coefficients at a time. The existing bridge uses this bounded binder in place of
a model-sized `read_bytes` snapshot. This removes that particular allocation;
it is not a measured reduction in whole-model peak RSS, and model preparation
still owns larger quantization tensors.

The versioned `PreparedPublicTiles` interface accepts only a trusted client-side
public projection source. It checks every route projection's identity/geometry
before opening any source and seals an immutable complete-plan binding only after
all whole-file digests match. A source context can remove each owned coefficient
file before opening the next; completed tile bindings remain valid after eviction.
A different model/parameter/route plan cannot reuse them. No serialized provider
catalog or advertisement is accepted as proof, and this interface is not a newly
admitted cryptographic backend.

Focused tests cover partial input/output tiles, fragmented int8/int16 reads,
mutation during a read, same-inode mutation after binding, file eviction, immutable
outputs and rejection before source I/O. Full Flash encrypted cache loading and
client/evaluator stage transitions remain separate unfinished integration work.


## Bounded evaluator coefficient handles

The existing all-linear evaluator now owns one active coefficient file and one
owned tile buffer, replacing a mapping for every projection. It validates each
requested tile against a snapshot of the installed plan before opening a file,
reads only its strided rows, and closes the previous file before advancing.
Cancellation or read/evaluation failure closes the active handle terminally.
Signed int8 and little-endian int12 storage retain the same native uint64 matrix
representation, including partial tiles and repeated autoregressive passes.

This is descriptor/mapping eviction, **not disk cache eviction** or a measured
full-Flash memory result. Whole-file cache admission and client encrypted replay
remain required. Missing files reject; no remote acquisition or fallback is added.
Full Flash still needs assembled weight materialization, per-stage client/provider
disk-cache transitions and an actual all-layer autoregressive run.


The complete-route cursor also indexes projection geometry once, rather than
scanning every tile for every projection. Duplicate tile identities and disjoint
repeated projection names reject before an attempt. Per-frame model/plan identity,
transport timeout and per-projection tile/arithmetic geometry are cached as
immutable facts; public plan definitions still return fresh owned copies. This
removes model-sized parsing/hashing from each encrypted frame without changing
frame contents, numerical bounds, recipient binding or failure handling. Focused
mutation tests and real Torch/NumPy tiled dispatch with explicitly simulated
crypto cover these ownership changes; no full Flash speedup is measured.


## Client mapping scopes and cache replacement

`FullLinear` now retains coefficient path/geometry instead of an open mapping for
every prepared projection. Both its exact integer reference and encrypted tiled
execution borrow one read-only matrix for the projection's scope and close it
explicitly on success, cancellation or arithmetic failure. The clear-only wide
control, source-precision diagnostic and twelve-bit plan-domain check use that
same owner. Ordinary file changes during a scope reject after cleanup; complete
cache identity admission and the bridge's before-disclosure tile check remain
required. A missing cache file rejects and cannot select floating fallback.

This permits an external public-cache owner to replace or remove a coefficient
file between uses without leaving the client attached to an old mapping. It does
not yet regenerate an evicted file, admit a partial Flash cache, release all
original floating checkpoint weights or implement full Flash generation. Tests
cover mapping lifetimes and actual int8/int12 Torch quantization; encrypted failure
and tiled-dispatch controls simulate crypto explicitly. No model-wide memory
saving or fresh encrypted full-model execution is inferred from these checks.


Coefficient preparation now shares one atomic cache writer across `FullLinear`
and the clear-only wide control. Complete immutable public bytes determine the
content address. A fresh entry is written under an owned temporary name and
published with create-only linking; existing entries must match the complete hash
and geometry. Concurrent identical preparations reuse one file, while corruption,
symlinks, short writes and conflicting concurrent entries reject without an
overwrite. Cleanup removes only the writer's own temporary file. An explicitly
removed public coefficient can be regenerated by preparation; inference cache
misses still reject and no automatic remote acquisition is added.

Tests exposed and fixed a real publication race: removing a temporary hard link
changes inode ctime even when all coefficient bytes remain identical. Coefficient
scopes bind device/inode/size/mtime and complete content hashes; harmless link
metadata changes are accepted. Source-checkpoint verification retains its separate
signature policy. Neither file metadata nor this cache writer replaces full plan
admission or before-disclosure tile verification. This is not a malicious-kernel
or power-loss durability guarantee, nor full Flash disk-stage orchestration.


## Source-bound client MoE loading

`SourceDenseClientMoE` now supplies the complete fixed all-expert operator from an
already verified decoder-stage source, without materializing every expert bank
in its constructor. The shared dense operator preserves the same operation order;
each source-bound linear reads only its public matrix (or one packed expert),
executes locally and releases that floating material. The client graph binds
immutable tensor/slice/source identities for these meta parameters and checks
all contributing source leases before embedding or any projection callback.
A closed lease, changed source or changed configuration fails the attempt.

Five focused tests use real small BF16 SafeTensors files and actual Torch math,
including four cached steps through the complete inspected mixed-attention/PLE
reference graph in local and trusted-adapter modes. They exercise fixed expert
visitation, released weight ownership and revocation before disclosure. These are
small synthetic weights, not actual full Flash generation or encrypted execution.
No large source objects are downloaded by this change.

This removes eager expert-bank retention from that graph composition. It does
not yet provide an out-of-core loader for every decoder parameter, disk-stage
eviction, a complete encrypted Flash plan or measured full-model memory/latency.
Packed gate/up material is reread for the two client projections; no cache saving
is assumed. Production and cryptographic qualification remain unavailable.

## Reusing prepared expert projections

The streamed MoE adapter retains client-owned prepared quantization metadata and
coefficient identities after its preparation pass. Each template releases its
FP32 source module. Clear and encrypted evaluations create fresh attempt bindings
from that template rather than rereading and requantizing every expert. Only
per-output-channel scale, zero-point, coefficient sum and optional bias remain in
RAM; coefficient matrices are still borrowed from bounded read-only mappings.
Returned copies own their metadata tensors and cannot retain another attempt's
key, bridge or mode. Floating-reference evaluation explicitly reloads source
weights; asking a released prepared projection for floating execution rejects.

The complete ordered all-expert route, cached coefficient identity comparison,
whole-file reauthentication of borrowed prepared coefficients before use,
before-disclosure tile verification, full encrypted replay and production gate
are unchanged. A missing coefficient file rejects instead of falling back or
regenerating weights during inference. This removes repeated source preparation
from the existing streamed MoE adapter, not a full-model disk-eviction controller.
Metadata still scales with the number of prepared output channels. No native
crypto speedup, full Flash generation or end-to-end memory qualification follows
from the small Torch adapter checks.


## Prepared MoE cancellation ownership

The source constructor and streaming MoE client accept an optional cancellation
signal. It reaches existing shard verification, preparation boundaries and each
projection, including combined gate/up results. A cancelled preparation publishes
no partial prepared metadata and leaves existing public coefficient files alone.
Encrypted evaluation requires the same explicit signal as its bridge; bridge
activity checks reject a previously failed attempt even after an event is cleared.
Attempt copies release bridge/key/parameter bindings on cancellation.

The component runner checks cancellation before key creation and after native
work, and passes its signal into the transport. Native quantization, encryption
and evaluation already in progress remain cooperative; immediate interruption is
not promised. Focused simulated-crypto control tests exercise these boundaries,
separately from the real quantizer adapter checks. This adds no model, endpoint
or cryptographic qualification.
