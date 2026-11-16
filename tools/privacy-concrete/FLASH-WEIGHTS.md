# Actual Flash weight integration

`flash-weights.json` pins the official Qwen3.8-Flash-Next revision, config/index digests and complete LFS SHA-256 values for two first-layer shards: 5,187,961,632 weight bytes. The provisioner verifies entire objects and admits no partial/range-only object. A corrupted existing cache is rejected rather than overwritten. Tensor use rechecks admitted file identity and exact BF16 geometry; experts are streamed individually into FP32 instead of materializing the entire expert bank.

`flash_expert.py` composes the complete fixed public expert 0 SwiGLU component from actual weights. Its combined gate/up projection is 2,560 → 1,280; its down projection is 640 → 2,560. It reuses `FullLinear` quantization, admitted tiling, framed evaluator and exact pre-decryption replay. The admitted component program has two linear modules, three logical projections and four encrypted products, with client SiLU and encrypted feedback into the down projection. No plaintext activation is sent to the evaluator.

This is a component integration, **not full Flash execution or private expert selection**. Expert 0 is fixed/public. The all-expert client program has separate source-reference evidence and the research integration described below; it has not completed encrypted qualification or been admitted for production. The component entry point accepts only a fixed public sine-vector fixture; there is no private-input or prompt option. Production cryptographic admission remains closed.

The installed research profile `flash-expert0` has its own protocol/model identity and exact source/parameter/dependency admission. Unknown profile advertisements fail before key export. It cannot masquerade as the full dense-Qwen protocol, and the current SSH manifest contract rejects this different underlying protocol rather than silently substituting it.

`.github/workflows/flash-weights.yml` runs the hash-verified acquisition and one bounded encrypted component execution on an isolated GitHub CPU runner. Trigger it manually or with `[flash-expert-e2e]` in a scoped commit; other pushes skip the large download. Only a source-constructed public-fixture result is uploaded. Source shards, quantized caches, raw diagnostics and keys are excluded.

A successful result must verify all four encrypted responses before decryption and satisfy the declared quantized-reference error bounds. It records actual output-error, wire/key sizes, execution/replay time and process-lifetime memory high-water. This does not establish full-model tokens, useful service throughput, BF16 parity, hardware appraisal or independently reviewed malicious security. No success is claimed until the actual workflow result exists.

The [retained result](evidence/flash/expert0-sine-2560.json) and [measurement context](evidence/flash/context.json) record a successful actual-weight run on [revision d83f3697](https://github.com/clocksmith/doppler/actions/runs/43803495772). All four encrypted products passed exact replay and the output passed its quantized-reference check. Encrypted-versus-quantized maximum error was 0.0003564; quantized-versus-floating error was 0.003536. Execution took 42.04 seconds, including 21.44 seconds of client replay; process elapsed was 48.75 seconds, excluding dependency installation and the earlier 5.19 GB public source acquisition. The fresh evaluation key was 134,217,808 bytes; the component sent 134,260,838 bytes and received 45,796 bytes. Client lifetime peak RSS was about 1.08 GB. These are scoped functional measurements, not full-model throughput or production security qualification.


## Complete first-layer MoE integration

`flash_moe.py --execution encrypted` streams every first-layer expert and the
shared/router weights through one fixed source-bound route. It lowers 1,541
logical projections to 1,029 encrypted linear modules and 2,058 width-safe tiles,
combining each expert's gate/up projection. The client performs the SiLU,
softmax/top-k, shared sigmoid and weighted combination. Every expert is computed
and replay-verified before selection affects the client result. Only one public
expert's floating-point material is resident at a time; public int8 files remain
reusable across attempts. Secret state and the pending up-vector are local to
one attempt. No private prompt/input CLI is exposed.

The separately named `flash-dense-moe` research protocol cannot substitute for
fixed expert 0, full Qwen generation or production crypto admission. The hosted
`Actual Flash dense MoE encrypted integration` workflow tests the actual
Torch/quantizer lowering, then attempts this complete fixed public fixture with
six-hour cancellation. Its implementation and a running job do not establish a
completed encrypted MoE execution. `--execution public-reference` is explicitly
a clear computation and reports `encryptedExecutionCompleted: false`.

This remains one first-layer MoE block. Hyperconnections, attention, recurrent
state, ngram embedding, other layers and autoregressive Flash generation are not
implemented by this component. Timing/padding, remote deployment, parameter and
implementation review and malicious-security qualification remain open. Full
client encrypted replay is still required.


The [cache-integrity regression result](evidence/flash/expert0-cache-binding.json)
and [exact revision context](evidence/flash/cache-binding-context.json) record a
second successful actual-weight encrypted run after replacing mutable read-only
weight mappings with immutable public tile digests. All four encrypted products
passed replay; maximum encrypted-versus-quantized error was 0.000676, with
41.62 seconds of encrypted execution including 20.95 seconds of client replay.
The full model is not copied per request: the client retains a digest table,
derived from one verified public projection snapshot at a time.


## Hyperconnection and complete MLP residual composition

`flash_hyper_ops.py` implements the source's grouped `(1 + weight)` RMSNorm,
low-rank gated input mixing, injection weights and residual combination. Its
trusted client projection callback composes directly with the complete all-expert
MoE graph. Source-reference checks cover four successive full hyperconnection /
MoE / residual updates, including exact operation order and fail-before-MoE
behavior after invalid input or a rejected projection.

The separately installed `flash-mlp-hyper` research protocol admits the actual
first-layer MLP hyperconnection weights from the whole pinned third shard.
`flash_hyper.py` attempts four encrypted gated-residual updates with a public
identity block: 3 linear modules and 15 tiles per update, 60 total. The client
retains normalization, sigmoid, SiLU and residual operations; the evaluator only
receives encrypted linear inputs. The identity block is an explicitly named test
fixture, not Flash attention, MoE or autoregressive token generation. No result
or security qualification is claimed before its workflow completes. Its source
scope does not change the existing MoE/fixed-expert identities.


`flash_mlp.py` composes the actual MLP hyperconnection and complete streamed
all-expert block under one route, then applies the source residual update. Its
`flash-mlp-block` research profile binds both program identities, all eleven
source tensors, 1,032 linear modules and 2,073 planned encrypted tiles. It uses
one client key/session for the full route and has no selected-only expert calls.
Its CLI requires an explicit `public-reference` or `encrypted` mode; the clear
mode cannot report encrypted success. The composed actual-weight encrypted run
has not been launched while the existing full MoE run is pending, to avoid
repeating that expensive execution. Small actual Torch/quantizer adapter tests
check full operator order and rejection before any projection after a changed
hyperconnection plan. No full MLP ciphertext completion or Flash generation is
claimed from those tests.


## Client linear-attention prefill and decode

`flash_delta.py` loads only inspected, hash-pinned official pure-Torch DeltaNet,
normalization, convolution and gated-delta declarations. The source loader removes
only enumerated acceleration/deprecation decorators and executes no downloaded
imports, model loader or network dispatch. It retains the eager non-export
triangular-solver path. The same loader owns existing MoE reference extraction.

The client graph has explicit projection callbacks and one request-owned cache
for convolution/recurrent state, bound to tenant, privacy, model, key, request and
execution mode. Shared model weights hold no private request cache. The callback
is trusted local code; its name does not appraise encryption or remote integrity.
A missing callback in `verified-projections` mode fails without clear fallback.
Context changes, incorrect projection order and cancellation are terminal; later
projections cannot continue after the cancellation boundary.

Source-reference tests compare a three-token prefill plus three single-token
cached decode steps with full-prefix official evaluation, check two concurrent
request states, and exercise context changes/cancellation. These are small public
FP32 operator fixtures, not actual Flash attention weights or encrypted DeltaNet
execution. No DeltaNet backend is registered for production or encrypted research
admission by this client graph. Full attention/QSA indexing, ngram embedding and
whole-model autoregressive generation remain unfinished.


The first four-step hyperconnection fixture failed its unchanged numerical
reference bound in run 43809990622; it is not qualified. A targeted public-fixture
probe now distinguishes same-input encrypted projection error from closed-loop
state/mixed/injection trajectory drift, stops at the first failed comparison and
keeps the nonzero exit status. It publishes only bounded aggregate measurements
from this fixed public fixture. No private-input diagnostics channel or relaxed
acceptance threshold is introduced.


`flash_delta_model.py` integrates the actual first-layer attention weights from
the complete pinned first shard (1,040,155,944 bytes). The separately named
`flash-first-delta` research route has five encrypted linear modules and 26
tiles per cached step, 104 over four fixed public layer inputs. Actual source
configuration selects sigmoid output gating; both SiLU and sigmoid are explicit
inspected reference functions. Depthwise convolution, gated-delta recurrence,
normalization and activations remain on the trusted client. No plaintext layer
input or cache state is passed to the evaluator. The new bounded workflow
attempts this component; a running job is not a successful result. This is not
full Flash token generation or complete-model encrypted inference.


## Client full attention and QSA indexing

`flash_attention.py` adds the inspected eager full-attention/QSA graph with
client-owned KV and compressed-indexer buffers. It shares the request transaction
and immutable projection boundary with DeltaNet. Query/key/value/indexer/output
projections follow one fixed order; sparse token selection and mask processing
stay on the client. Only explicit FP32 eager execution, declared rotary width
and a bounded context are admitted. Boolean, noncausal and fully masked query
rows are currently unsupported and fail before projection dispatch.

Source checks compare prefill plus three cached decode steps with full-prefix
attention, keep the projection route fixed across input/selection changes, and
verify independent KV/indexer buffers. This graph has no actual Flash weight or
encrypted-execution qualification and is not registered as an encrypted backend.
Ngram/PLE and complete-model generation remain unfinished.
