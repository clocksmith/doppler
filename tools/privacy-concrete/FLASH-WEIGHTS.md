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
| Complete first-layer MoE | Shared/router and all 512 experts | 1,029 encrypted modules / 2,058 tiles implemented; original encrypted run still pending |
| Complete first-layer MLP branch | Hyperconnection plus all-expert MoE and residual | 1,032 modules / 2,073 planned tiles composed; actual FP32 sparse-source comparison is separate; full ciphertext composition not yet run |
| Full attention / QSA | Small public FP32 source fixtures | Prefill, cached decode, fixed projection route and KV/indexer isolation checks passed; actual weights and encrypted execution unqualified |
| Ngram / PLE | Small local-table FP32 source fixtures | EOS-aware cached updates, local lookup and history isolation checks passed; actual Flash table unsupported |

The original four-step [dense Qwen candidate](FULL-QWEN.md) remains a separate
running experiment. Component results cannot substitute for its full-model output.

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
hyperconnection, all-expert MoE and composed MLP are distinct. Unknown profiles
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
PLE currently admits only a complete locally materialized embedding under an
explicit pre-allocation budget. It has no remote row-fetch/cache-miss fallback.
The actual huge table and checkpoint shard-to-logical-row mapping remain
unqualified; a mapped-table implementation must verify complete local shards and
row mapping before accepting private tokens.

## Running and remaining qualification

Workflows isolate CPU/model activity from local/fleet performance work. Large
encrypted runs require an explicit marker or manual dispatch; ordinary pushes
skip them. Only source-constructed public-fixture results are uploaded. Keys,
raw vectors, source weights, private caches and raw diagnostics are excluded.

`flash_mlp.py` requires an explicit `public-reference` or `encrypted` mode and a
pinned `--reference-source`. Clear mode cannot report encrypted completion. Its
actual FP32 sparse-source gate runs before key generation. Full ciphertext MLP
composition has not been launched while the original all-expert job is pending.

Remaining gates include the two original complete runs; whole-model weights and
ngram mapping; all-layer/cache/tokenizer/rotary and autoregressive integration;
actual-weight full attention/PLE qualification; model-quality and cost evidence;
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
