# Native Q5_K preparation and 27B tuning checkpoint

Component: `doppler.runtime-source.formats`, `doppler.runtime-source.gpu.kernels`, `doppler.runtime-source.loader`, `doppler.repository-tooling`.
Intent: preserved. Boundary effects: native Q5_K projection loading, packed embedding byte admission, offline candidate construction.

The Qwen3.8-27B source GGUF contains Q5_K linear-output projections. This change adds native F32-output GEMV and tiled-prefill execution without rewriting existing Q4-family shader identities. It also corrects packed embedding preflight: the source table occupies 715,161,600 bytes, rather than its hypothetical dense F32 extent. Q5_K encoding, CPU fallback and Q5_K embedding gather are not added.

Acceptance evidence: [source-byte preservation](model-byte-preservation.json), [AMD operator cases](operator-amd.json), [M3 Metal operator cases](operator-mars.json), [independent real-weight fixture](operator-fixture.json), [focused development checks](development-checks.json). All 851 base tensors and 15,780,284,416 bytes are preserved. The real projection fixture differs from GGUF dequantization plus NumPy F64 multiplication by at most 3.725290298461914e-8 on both GPUs. These are operator receipts, not full-model numerical qualification.

[Four toy controls](quality-controls.json) return identical tokens through Doppler, Pooled and pinned llama.cpp. All three answer `22` where the code task requires `26`. This is not an accepted quality suite. llama.cpp uses a separately retained, byte-checked base-only container because that version treats the source MTP block as a base layer.

The [exploratory screens](screening.json) use Jupiter Chrome, 28 or 2,287 input tokens and 128 output tokens. Registered paired decode kernels reduce the short measured request from 33.862 to 17.933 seconds; adding prepared whole-token execution reduces it to 14.977 seconds. A separate prefill-only candidate reduces long prefill from 72.388 to 40.563 seconds. Every candidate retains its baseline 128-token output. These levers have not been jointly qualified. Screens have only one measured observation per condition and differ in idle engine residency; they do not establish statistical performance gains.

Pooled with MTP depth three remains faster in the native baseline: 7.075 seconds short and 28.678 seconds long. The long native engine outputs first differ at output token 123. Different weight materializations and KV precision remain explicit. Full logit/reference qualification, matched-quality comparisons, repeated paired timings, attributable memory and full-model Mars qualification remain open. No universal advantage, lower total memory, energy saving or model release is claimed.

The source architecture check retains the same two failures already observed on the parent: `prefill-runtime.js` (898 lines, ceiling 896) and `artifact-storage-context.js` (946, ceiling 897). This change does not raise their budgets.

[Source identities](source-files.json) and [retained evidence identities](retained-files.json) bind this checkpoint. The initial baseline config inherited an older commit label; its frozen source identity is `source-q5k-capability-02`, recorded in the retained inventory. Runtime source is the parent plus this checkpoint's native-format and loader changes. Experimental screens do not activate a production profile or promote the artifact.

Current operator receipts compile the committed shader bytes. Removing one trailing empty line changed content digests without changing instructions. The [separate metadata derivative](manifest-refresh.json) was rebuilt through the existing manifest refresh owner, with full shard checks and no data duplication; original timing inputs remain immutable.

The extended [CPU control](cpu-control-observations.json) differs at three teacher-forced positions. Its [projection arithmetic quantizes activations](cpu-control-precision.json), so it is an independent control, not an eligible F32-activation full-logit reference. This precision boundary must be resolved before any numerical equivalence claim.
