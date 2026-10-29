"""One canonical CPU model loader and physical KV cache boundary for references."""
from binding import bind_names


def load_native_model(inputs, policy):
    import torch
    from safetensors import safe_open
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    from transformers.models.qwen3_5 import modeling_qwen3_5 as hf

    if hf.FusedRMSNormGated is not None or hf.is_fast_path_available:
        raise ValueError("native CPU reference refuses optional CUDA kernels")
    config = Qwen3_5TextConfig(**inputs["config"]["text_config"])
    config._attn_implementation = "eager"
    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(config)
    mappings = bind_names(inputs["manifest"]["tensors"],
                          {k: list(t.shape) for k, t in model.state_dict().items()}, policy)
    with safe_open(str(inputs["weights"]), framework="pt", device="cpu") as source:
        state = {m["target"]: source.get_tensor(m["source"]) for m in mappings}
    model.load_state_dict(state, strict=True, assign=True)
    del state
    model.model.rotary_emb = hf.Qwen3_5TextRotaryEmbedding(config, device="cpu")
    model.eval()
    if model.lm_head.weight is model.model.embed_tokens.weight or model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr():
        raise ValueError("canonical LM head was aliased to input embeddings")
    if any(t.device.type != "cpu" or t.dtype != torch.float32 for t in model.parameters()):
        raise ValueError("native reference parameters must all be CPU F32")
    if any(t.device.type != "cpu" for t in model.buffers()):
        raise ValueError("native reference retained a non-CPU buffer")
    return model, config, mappings


def physical_cache(config, dtype):
    import torch
    from transformers import DynamicCache

    class PhysicalKVCache(DynamicCache):
        def __init__(self):
            super().__init__(config=config)
            self.events = []

        def update(self, key_states, value_states, layer_idx, *extra, **kwargs):
            if key_states.dtype != torch.float32 or value_states.dtype != torch.float32:
                raise ValueError("eager attention K/V producer must be F32 before declared cache rounding")
            keys, values = super().update(key_states.to(dtype), value_states.to(dtype), layer_idx, *extra, **kwargs)
            layer = self.layers[layer_idx]
            if any(t.dtype != dtype for t in (keys, values, layer.keys, layer.values)):
                raise ValueError("actual cache storage differs from declared physical dtype")
            widened = keys.float(), values.float()
            if not (torch.equal(widened[0].to(dtype), keys) and torch.equal(widened[1].to(dtype), values)):
                raise ValueError("widened attention inputs are not exact cache values")
            self.events.append({"layer": layer_idx, "newLength": key_states.shape[2],
                "storedShape": list(keys.shape), "producerDtype": str(key_states.dtype),
                "physicalKeyDtype": str(layer.keys.dtype), "physicalValueDtype": str(layer.values.dtype),
                "attentionKeyDtype": str(widened[0].dtype), "attentionValueDtype": str(widened[1].dtype),
                "newKeyRoundingMax": float((key_states - key_states.to(dtype).float()).abs().max()),
                "newValueRoundingMax": float((value_states - value_states.to(dtype).float()).abs().max())})
            return widened

    return PhysicalKVCache()


def inspect_cache(cache, config):
    import torch

    full = [i for i, kind in enumerate(config.layer_types) if kind == "full_attention"]
    if [event["layer"] for event in cache.events[-len(full):]] != full:
        raise ValueError("native reference did not observe every full-attention cache update")
    linear = []
    for index, kind in enumerate(config.layer_types):
        if kind == "linear_attention":
            layer = cache.layers[index]
            if layer.conv_states.dtype != torch.float32 or layer.recurrent_states.dtype != torch.float32:
                raise ValueError("native linear state is not F32")
            linear.append({"layer": index, "convDtype": str(layer.conv_states.dtype),
                           "recurrentDtype": str(layer.recurrent_states.dtype)})
    return linear
