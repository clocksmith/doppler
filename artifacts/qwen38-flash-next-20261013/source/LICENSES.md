# Licenses for these source files

The files in this directory are unmodified copies from two upstreams. Their
sha256 values are in [../source-binding.json](../source-binding.json). The
repository's MIT license does not apply to them.

| Files | Upstream | License |
|---|---|---|
| `config.json`, `generation_config.json`, `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, `model.safetensors.index.json`, `LICENSE` | `Qwen/Qwen3.8-Flash-Next@de4b8e4d43b917e7706784d8bb445c9af86a3540` | Qwen Community License 1.0, [LICENSE](LICENSE) (sha256 `a0dc422560841fd68e06…`) |
| `configuration_qwen4_exp.py`, `modeling_qwen4_exp.py`, `modeling_qwen3_5.py`, `modeling_rope_utils.py`, `integrations_moe.py`, `integrations_hub_kernels.py`, `integrations_accelerate.py`, `activations.py`, `cache_utils.py` | `huggingface/transformers@e0eb2810ac1b2caf6d082f247d8d37162ecf67a0` | Apache License 2.0, [LICENSE-transformers](LICENSE-transformers): that revision's `LICENSE`, byte-identical (sha256 `77fd4710def9ec3c0f62…`). Each file keeps its own copyright header. |

`configuration_qwen4_exp.py` comes from Transformers, not from the Qwen
checkpoint repository, so it is under Apache-2.0.

The Qwen Community License requires its notice in all copies or substantial
portions of the Software (clause 1). Copies of these files, and artifacts
converted from the checkpoint, keep `LICENSE` with them.
