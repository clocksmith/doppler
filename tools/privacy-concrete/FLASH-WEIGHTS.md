# Actual Flash weight integration

`flash-weights.json` pins the official Qwen3.8-Flash-Next revision, config/index digests and complete LFS SHA-256 values for two first-layer shards: 5,187,961,632 weight bytes. The provisioner verifies entire objects and admits no partial/range-only object. A corrupted existing cache is rejected rather than overwritten. Tensor use rechecks admitted file identity and exact BF16 geometry; experts are streamed individually into FP32 instead of materializing the entire expert bank.

`flash_expert.py` composes the complete fixed public expert 0 SwiGLU component from actual weights. Its combined gate/up projection is 2,560 → 1,280; its down projection is 640 → 2,560. It reuses `FullLinear` quantization, admitted tiling, framed evaluator and exact pre-decryption replay. The admitted component program has two linear modules, three logical projections and four encrypted products, with client SiLU and encrypted feedback into the down projection. No plaintext activation is sent to the evaluator.

This is a component integration, **not full Flash execution or private expert selection**. Expert 0 is fixed/public. The all-expert client program has separate source-reference evidence but has not been admitted as an encrypted Flash backend. The component entry point accepts only a fixed public sine-vector fixture; there is no private-input or prompt option. Production cryptographic admission remains closed.

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
