"""Exporter-only Qwen3.5 linear-attention lowering with explicit tensor state.

Native HF modules own weights and projections. This adapter replaces mutation,
sequence-valued split, triangular scatter, and shape-dependent Python loops.
The fixed prefill bound is a declared graph constraint, not token-at-a-time prefill.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


class LinearAttentionExport(torch.nn.Module):
    def __init__(self, native, prefill_bound: int, mode: str):
        super().__init__()
        if prefill_bound < 64 or prefill_bound % 64:
            raise ValueError("prefill bound must be a positive multiple of native chunk size 64")
        if mode not in ("prefill", "decode"):
            raise ValueError("export mode must explicitly be prefill or decode")
        self.native = native
        self.bound = prefill_bound
        self.mode = mode
        self.register_buffer("lower", torch.ones(64, 64).tril())
        self.register_buffer("strict_lower", torch.ones(64, 64, dtype=torch.bool).tril(-1))
        self.register_buffer("identity", torch.eye(64))
        self.register_buffer("zero_key", torch.zeros(1, native.num_v_heads, prefill_bound, native.head_k_dim))
        self.register_buffer("zero_value", torch.zeros(1, native.num_v_heads, prefill_bound, native.head_v_dim))
        self.register_buffer("zero_scalar", torch.zeros(1, native.num_v_heads, prefill_bound))
        self.register_buffer("zero_conv", torch.zeros(1, native.conv_dim, native.conv_kernel_size))
        self.register_buffer("zero_recurrent", torch.zeros(1, native.num_v_heads, native.head_k_dim, native.head_v_dim))

    @staticmethod
    def l2norm(value):
        return value * torch.rsqrt((value * value).sum(-1, keepdim=True) + 1e-6)

    @staticmethod
    def softplus(value):
        # Preserve the unchanged native CPU reference. ONNX lowering is separate.
        return F.softplus(value)

    def fixed_pad(self, value, zeros):
        return torch.cat((value, zeros), dim=2)[:, :, :self.bound]

    def chunk(self, query, key, value, g, beta, state):
        """Native chunk64 recurrence with a constant bound and scatter-free solve."""
        sequence_length = query.shape[1]
        query = self.l2norm(query).transpose(1, 2)
        key = self.l2norm(key).transpose(1, 2)
        value = value.transpose(1, 2)
        beta, g = beta.transpose(1, 2), g.transpose(1, 2)
        query, key = [self.fixed_pad(x, self.zero_key) for x in (query, key)]
        value = self.fixed_pad(value, self.zero_value)
        beta, g = [self.fixed_pad(x, self.zero_scalar) for x in (beta, g)]
        query = query * (self.native.head_k_dim ** -0.5)
        key_beta = key * beta.unsqueeze(-1)
        value_beta = value * beta.unsqueeze(-1)
        chunks = self.bound // 64
        batch, heads = query.shape[:2]
        query, key, key_beta = [x.reshape(batch, heads, chunks, 64, self.native.head_k_dim)
                               for x in (query, key, key_beta)]
        value_beta = value_beta.reshape(batch, heads, chunks, 64, self.native.head_v_dim)
        decay = g.reshape(batch, heads, chunks, 64).cumsum(-1)
        difference = decay.unsqueeze(-1) - decay.unsqueeze(-2)
        decay_mask = torch.where(self.lower.bool(), difference, 0.0).exp() * self.lower
        original = -(key_beta @ key.transpose(-1, -2)) * decay_mask
        original = torch.where(self.strict_lower, original, 0.0)
        rows = [original[..., :1, :]]
        for i in range(1, 64):
            row = original[..., i, :i]
            sub = torch.cat(rows, dim=-2)[..., :i]
            updated = row + (row.unsqueeze(-1) * sub).sum(-2)
            rows.append(torch.cat((updated, torch.zeros_like(original[..., i, i:])), -1).unsqueeze(-2))
        solved = torch.cat(rows, dim=-2) + self.identity
        solved_value = solved @ value_beta
        key_cumulative = solved @ (key_beta * decay.exp().unsqueeze(-1))
        outputs = []
        for chunk_index in range(chunks):
            q, k = query[:, :, chunk_index], key[:, :, chunk_index]
            d = decay[:, :, chunk_index]
            attention = (q @ k.transpose(-1, -2)) * decay_mask[:, :, chunk_index]
            new_value = solved_value[:, :, chunk_index] - key_cumulative[:, :, chunk_index] @ state
            outputs.append((q * d.exp().unsqueeze(-1)) @ state + attention @ new_value)
            state = (state * d[..., -1, None, None].exp() +
                     (k * (d[..., -1, None] - d).exp().unsqueeze(-1)).transpose(-1, -2) @ new_value)
        output = torch.cat(outputs, dim=2)[:, :, :sequence_length].transpose(1, 2)
        return output, state

    def recurrent(self, query, key, value, g, beta, state):
        query = self.l2norm(query)[:, 0] * (self.native.head_k_dim ** -0.5)
        key, value = self.l2norm(key)[:, 0], value[:, 0]
        state = state * g[:, 0].exp().unsqueeze(-1).unsqueeze(-1)
        memory = (state * key.unsqueeze(-1)).sum(-2)
        delta = (value - memory) * beta[:, 0].unsqueeze(-1)
        state = state + key.unsqueeze(-1) * delta.unsqueeze(-2)
        return (state * query.unsqueeze(-1)).sum(-2).unsqueeze(1), state

    def forward(self, hidden, past_conv, past_recurrent):
        layer = self.native
        batch, length, _ = hidden.shape
        mixed = layer.in_proj_qkv(hidden).transpose(1, 2)
        z = layer.in_proj_z(hidden).reshape(batch, length, layer.num_v_heads, layer.head_v_dim)
        beta = layer.in_proj_b(hidden).sigmoid()
        a = layer.in_proj_a(hidden)
        g = -layer.A_log.float().exp() * self.softplus(a.float() + layer.dt_bias)
        if self.mode == "decode":
            combined = torch.cat((past_conv, mixed), dim=-1)
            present_conv = combined[:, :, -layer.conv_kernel_size:]
            convolved = F.conv1d(combined, layer.conv1d.weight, groups=layer.conv_dim)[:, :, -1:]
        else:
            present_conv = torch.cat((self.zero_conv, mixed), dim=-1)[:, :, -layer.conv_kernel_size:]
            convolved = F.conv1d(mixed, layer.conv1d.weight, padding=layer.conv_kernel_size - 1,
                                 groups=layer.conv_dim)[:, :, :length]
        mixed = F.silu(convolved).transpose(1, 2)
        q = mixed[:, :, :layer.key_dim].reshape(batch, length, layer.num_k_heads, layer.head_k_dim)
        k = mixed[:, :, layer.key_dim:2 * layer.key_dim].reshape(batch, length, layer.num_k_heads, layer.head_k_dim)
        v = mixed[:, :, 2 * layer.key_dim:].reshape(batch, length, layer.num_v_heads, layer.head_v_dim)
        repetitions = layer.num_v_heads // layer.num_k_heads
        if repetitions > 1:
            q, k = [x.repeat_interleave(repetitions, dim=2) for x in (q, k)]
        if self.mode == "decode":
            output, present_recurrent = self.recurrent(q, k, v, g, beta, past_recurrent)
        else:
            output, present_recurrent = self.chunk(q, k, v, g, beta, self.zero_recurrent)
        output = layer.norm(output.reshape(-1, layer.head_v_dim), z.reshape(-1, layer.head_v_dim))
        output = output.reshape(batch, length, layer.value_dim)
        return layer.out_proj(output), present_conv, present_recurrent
