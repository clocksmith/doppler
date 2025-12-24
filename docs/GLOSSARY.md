# DOPPLER Glossary

Short definitions for terms used across DOPPLER docs and code.

---

## WebGPU Terms

| Term | Definition |
|------|------------|
| WebGPU | Browser API for GPU compute and graphics. DOPPLER uses it for LLM inference. |
| Adapter | A handle representing a physical GPU and its capabilities. Created via `navigator.gpu.requestAdapter()`. |
| Device | A logical GPU device created from an adapter. Used to allocate buffers and submit work. |
| Feature | Optional capability that must be requested when creating the device, for example `shader-f16` or `subgroups`. |
| Limits | Per-device constraints like maximum storage buffer size and shared memory per workgroup. |
| Queue | Submission interface on a device. Used for `queue.submit()` and `queue.writeBuffer()`. |
| Buffer | A region of GPU memory. Common uses include weights, activations, KV cache, and scratch space. |
| Storage buffer | A read-write buffer bound to compute shaders. Used for large tensors. |
| Uniform buffer | A small read-only buffer for parameters (sizes, scalars). Used for per-dispatch constants. |
| Bind group | A set of resources (buffers) bound as shader inputs. Similar to an argument bundle. |
| Pipeline | A compiled compute shader plus binding layout. DOPPLER uses multiple pipelines per op (matmul, attention, dequant). |
| Command encoder | Records GPU commands. Produces a command buffer. |
| Command buffer | A recorded batch of GPU work submitted to the queue. |
| Compute pass | A section of commands where compute pipelines are dispatched. |
| Dispatch | Launch a compute shader over a grid of workgroups using `dispatchWorkgroups()`. |
| Invocation | One running instance of a shader function. Similar to one thread. |
| Workgroup | A group of invocations that can share workgroup memory and synchronize with barriers. |
| Workgroup memory | Fast scratch memory shared within a workgroup. Limited by `maxComputeWorkgroupStorageSize`. |
| Subgroup | A hardware-defined subset of a workgroup that executes in lockstep (warp or wavefront). Enables fast cross-lane ops like broadcast and reductions. |
| `shader-f16` | WebGPU feature that enables 16-bit float types and operations in shaders. Used for faster math and lower memory bandwidth. |
| `subgroups` | WebGPU feature that enables subgroup operations in WGSL. DOPPLER uses this for faster dequantization kernels. |
| WGSL | WebGPU Shading Language. DOPPLER kernels are written in WGSL. |

---

## Inference Terms

| Term | Definition |
|------|------------|
| Token | An integer ID representing a piece of text. Produced by the tokenizer. |
| Tokenizer | Converts text to token IDs and token IDs back to text. |
| Embedding | Maps token IDs to vectors (hidden states) at the model input. |
| Hidden state | The per-token vector processed through transformer layers. |
| Logits | Raw next-token scores before normalization. A vector of length `vocabSize`. |
| Sampling | Converts logits into a chosen next token, for example greedy argmax or randomized sampling. |
| Temperature | Scales logits before sampling. Lower values are more deterministic. |
| Top-k | Sampling strategy that restricts choices to the k highest-probability tokens. |
| Top-p | Sampling strategy that restricts choices to the smallest set of tokens with cumulative probability p. |
| Prefill | Forward pass over the full prompt to populate KV cache. Often compute-bound and scales with prompt length. |
| Decode | Autoregressive generation loop. Processes one new token per step using KV cache. Often memory-bound. |
| Attention | Uses queries, keys, values (Q/K/V) to mix information across positions. |
| Head | One attention subspace. Total heads is `numHeads`. |
| `headDim` | Dimensions per head. Hidden size is typically `numHeads * headDim`. |
| GQA | Grouped Query Attention. Uses fewer K/V heads (`numKVHeads`) than query heads (`numHeads`). |
| KV cache | Stored keys and values from prior tokens, per layer. Avoids recomputing past context during decode. |
| RoPE | Rotary position embeddings applied to Q and K. |
| RMSNorm | Root mean square normalization used by many LLMs. |
| Quantization | Store weights in fewer bits to reduce memory and bandwidth. Dequantization happens during compute. |
| Q4_K_M | 4-bit quantization format similar to llama.cpp. Used by many DOPPLER dense models. |

---

## DOPPLER Data and Modules

| Term | Definition |
|------|------------|
| RDRR | Recursive DOPPLER Runtime Registry. Packaging format for models delivered as a manifest plus binary shards. |
| Manifest | JSON file describing model architecture, tensors, quantization, and shard registry. |
| Shard | A chunk of model weights stored as a separate binary file for streaming and caching. |
| OPFS | Origin Private File System. Browser storage used to cache shards and tokenizer files. |
| Native Bridge | Optional Node-based helper that can read local model files and bypass browser storage limits. |
| Kernel | A WGSL compute shader implementing an operation, for example matmul or attention. |
| Kernel variant | Multiple implementations of the same op selected based on device features and limits, for example f16 vs f32 matmul. |
| Buffer pool | Reuse strategy to avoid frequent WebGPU buffer allocation and destruction. |

---

*Last updated: December 2025*

<!-- DOPPLER_KERNEL_OVERRIDES -->
## Kernel Overrides & Compatibility
See `docs/KERNEL_COMPATIBILITY.md` for runtime kernel modes (4-bit/9-bit), CLI flags (`--force-fused-q4k`, `--kernel-hints`), and the OPFS purge helper.

