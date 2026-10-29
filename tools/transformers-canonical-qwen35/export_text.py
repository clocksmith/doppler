"""Full canonical text decoder lowering; tensors/state are explicit graph inputs."""
from __future__ import annotations

import torch

from export_layer import LinearAttentionExport


def cache_names(config):
    names = []
    for index, kind in enumerate(config.layer_types):
        roles = ("conv", "recurrent") if kind == "linear_attention" else ("key", "value")
        for role in roles:
            if kind == "linear_attention":
                names.append((f"past_{role}.{index}", f"present_{role}.{index}"))
            else:
                names.append((f"past_key_values.{index}.{role}", f"present.{index}.{role}"))
    return names


def empty_cache(config, past_length=0):
    result = []
    for kind in config.layer_types:
        if kind == "linear_attention":
            width = (2 * config.linear_num_key_heads * config.linear_key_head_dim +
                     config.linear_num_value_heads * config.linear_value_head_dim)
            result.extend((torch.zeros(1, width, config.linear_conv_kernel_dim),
                           torch.zeros(1, config.linear_num_value_heads, config.linear_key_head_dim,
                                       config.linear_value_head_dim)))
        else:
            shape = (1, config.num_key_value_heads, past_length, config.head_dim)
            result.extend((torch.zeros(shape, dtype=torch.float16), torch.zeros(shape, dtype=torch.float16)))
    return tuple(result)


def cache_tensors(cache, config):
    result = []
    for index, kind in enumerate(config.layer_types):
        layer = cache.layers[index]
        result.extend((layer.conv_states, layer.recurrent_states) if kind == "linear_attention"
                      else (layer.keys, layer.values))
    return tuple(result)


class TextDecoderExport(torch.nn.Module):
    def __init__(self, native, policy, mode):
        super().__init__()
        self.native = native
        self.config = native.config
        self.mode = mode
        self.linear = torch.nn.ModuleDict({str(index): LinearAttentionExport(layer.linear_attn,
                policy["prefillBound"], mode) for index, layer in enumerate(native.model.layers)
                if self.config.layer_types[index] == "linear_attention"})
        rope = self.config.rope_parameters
        if rope["rope_type"] != "default" or not rope["mrope_interleaved"]:
            raise ValueError("exporter only implements the explicitly configured default interleaved MRoPE")
        self.rotary_dimension = int(self.config.head_dim * rope["partial_rotary_factor"])
        frequencies = torch.arange(self.rotary_dimension // 2)
        self.register_buffer("height_mask", (frequencies % 3 == 1) & (frequencies < rope["mrope_section"][1] * 3))
        self.register_buffer("width_mask", (frequencies % 3 == 2) & (frequencies < rope["mrope_section"][2] * 3))

    def positions(self, position_ids):
        inverse = self.native.model.rotary_emb.inv_freq[None, :, None]
        freqs = [(inverse @ position_ids[axis, :, None, :].float()).transpose(1, 2) for axis in range(3)]
        combined = torch.where(self.height_mask, freqs[1], freqs[0])
        combined = torch.where(self.width_mask, freqs[2], combined)
        embedding = torch.cat((combined, combined), dim=-1)
        return embedding.cos().unsqueeze(1), embedding.sin().unsqueeze(1)

    def rotate(self, value, cos, sin):
        rotated = value[..., :self.rotary_dimension]
        half = self.rotary_dimension // 2
        rotated_half = torch.cat((-rotated[..., half:], rotated[..., :half]), dim=-1)
        return torch.cat((rotated * cos + rotated_half * sin, value[..., self.rotary_dimension:]), dim=-1)

    def repeat_kv(self, value):
        batch, heads, length, width = value.shape
        repetitions = self.config.num_attention_heads // self.config.num_key_value_heads
        return value[:, :, None].expand(batch, heads, repetitions, length, width).reshape(
            batch, heads * repetitions, length, width)

    def full_attention(self, layer, hidden, cos, sin, attention_mask, past_key, past_value):
        config = self.config
        batch, length, _ = hidden.shape
        projected = layer.q_proj(hidden).reshape(batch, length, config.num_attention_heads, 2 * config.head_dim)
        query = projected[..., :config.head_dim]
        gate = projected[..., config.head_dim:].reshape(batch, length, config.num_attention_heads * config.head_dim)
        query = layer.q_norm(query).transpose(1, 2)
        key = layer.k_norm(layer.k_proj(hidden).reshape(batch, length, config.num_key_value_heads,
                                                       config.head_dim)).transpose(1, 2)
        value = layer.v_proj(hidden).reshape(batch, length, config.num_key_value_heads, config.head_dim).transpose(1, 2)
        query, key = self.rotate(query, cos, sin), self.rotate(key, cos, sin)
        key, value = key.to(torch.float16), value.to(torch.float16)
        if self.mode == "decode":
            key, value = torch.cat((past_key, key), dim=2), torch.cat((past_value, value), dim=2)
            past_length = past_key.shape[2]
        else:
            past_length = 0
        present_key, present_value = key, value
        key, value = self.repeat_kv(key.float()), self.repeat_kv(value.float())
        scores = (query @ key.transpose(-1, -2)) * (config.head_dim ** -0.5)
        columns = torch.arange(key.shape[2], device=hidden.device)
        rows = torch.arange(length, device=hidden.device) + past_length
        causal = columns[None, :] <= rows[:, None]
        minimum = torch.finfo(torch.float32).min
        scores = torch.where(causal[None, None], scores, minimum)
        scores = torch.where(attention_mask[:, None, None, :] > 0, scores, minimum)
        probabilities = scores.softmax(dim=-1)
        output = (probabilities @ value).transpose(1, 2).reshape(
            batch, length, config.num_attention_heads * config.head_dim)
        return layer.o_proj(output * gate.sigmoid()), present_key, present_value

    def forward(self, inputs_embeds, position_ids, attention_mask, cache):
        hidden = inputs_embeds
        cos, sin = self.positions(position_ids)
        present = []
        for index, layer in enumerate(self.native.model.layers):
            normalized = layer.input_layernorm(hidden)
            if self.config.layer_types[index] == "linear_attention":
                output, first, second = self.linear[str(index)](normalized, cache[2 * index], cache[2 * index + 1])
            else:
                output, first, second = self.full_attention(layer.self_attn, normalized, cos, sin, attention_mask,
                                                           cache[2 * index], cache[2 * index + 1])
            hidden = hidden + output
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
            present.extend((first, second))
        hidden = self.native.model.norm(hidden)
        logits = self.native.lm_head(hidden[:, -1:])
        return (logits, *present)


class EmbeddingExport(torch.nn.Module):
    def __init__(self, native):
        super().__init__()
        self.embedding = native.model.embed_tokens

    def forward(self, input_ids):
        return self.embedding(input_ids)
