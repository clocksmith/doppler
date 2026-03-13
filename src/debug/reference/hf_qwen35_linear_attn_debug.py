#!/usr/bin/env python3
"""
Dump intermediate values from Qwen3.5 linear attention (GatedDeltaNet) for comparison with Doppler.

Usage:
    HF_HOME=/media/x/models/huggingface_cache python3 src/debug/reference/hf_qwen35_linear_attn_debug.py
"""

import os
import torch
import numpy as np

os.environ.setdefault("HF_HOME", "/media/x/models/huggingface_cache")

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3.5-0.8B"
PROMPT = "Hello"


def stats(name, tensor):
    t = tensor.float().detach().flatten()
    print(f"  {name}: shape={list(tensor.shape)}, "
          f"min={t.min().item():.6f}, max={t.max().item():.6f}, "
          f"mean={t.mean().item():.6f}, absMax={t.abs().max().item():.6f}")
    first8 = t[:8].tolist()
    print(f"    first8: {[f'{v:.6f}' for v in first8]}")


def main():
    print(f"Loading {MODEL_ID}...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model.eval()

    inputs = tokenizer(PROMPT, return_tensors="pt")
    input_ids = inputs["input_ids"]
    print(f"Prompt: '{PROMPT}', Token IDs: {input_ids[0].tolist()}")
    num_tokens = input_ids.shape[1]

    # Dump key weight values for layer 0
    layer0 = model.model.layers[0]
    attn = layer0.linear_attn

    print(f"\n=== Layer 0 weights ===")
    if hasattr(attn, 'A_log'):
        a_log = attn.A_log.detach().float()
        a_neg_exp = -torch.exp(a_log)
        stats("A_log", a_log)
        stats("a_neg_exp", a_neg_exp)
    if hasattr(attn, 'dt_bias'):
        stats("dt_bias", attn.dt_bias.detach().float())
    stats("conv1d.weight", attn.conv1d.weight.detach().float())
    stats("norm.weight", attn.norm.weight.detach().float())

    # Hook into the linear_attn module to capture its input and output
    captured = {}

    def hook_linear_attn_input(module, args, kwargs):
        if len(args) > 0:
            captured['linear_attn_input'] = args[0].detach().clone()
        return None

    def hook_linear_attn_output(module, args, kwargs, output):
        if isinstance(output, tuple):
            captured['linear_attn_output'] = output[0].detach().clone()
        else:
            captured['linear_attn_output'] = output.detach().clone()
        return None

    # Hook into individual projection layers
    def make_hook(name):
        def hook(module, input, output):
            captured[name] = output.detach().clone()
        return hook

    hooks = []
    hooks.append(attn.register_forward_pre_hook(hook_linear_attn_input, with_kwargs=True))
    hooks.append(attn.register_forward_hook(hook_linear_attn_output, with_kwargs=True))
    hooks.append(attn.in_proj_qkv.register_forward_hook(make_hook('qkv_proj')))
    hooks.append(attn.in_proj_z.register_forward_hook(make_hook('z_proj')))
    hooks.append(attn.in_proj_a.register_forward_hook(make_hook('a_proj')))
    hooks.append(attn.in_proj_b.register_forward_hook(make_hook('b_proj')))
    hooks.append(attn.out_proj.register_forward_hook(make_hook('out_proj')))
    hooks.append(attn.conv1d.register_forward_hook(make_hook('conv1d_raw')))
    hooks.append(attn.norm.register_forward_hook(make_hook('gated_norm')))

    # Also hook input_layernorm
    hooks.append(layer0.input_layernorm.register_forward_hook(make_hook('input_layernorm')))

    print(f"\n=== Running forward pass ===")
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)

    # Remove hooks
    for h in hooks:
        h.remove()

    print(f"\n=== Captured intermediates ===")
    for name in ['input_layernorm', 'qkv_proj', 'z_proj', 'a_proj', 'b_proj',
                  'conv1d_raw', 'gated_norm', 'linear_attn_input', 'linear_attn_output', 'out_proj']:
        if name in captured:
            stats(name, captured[name])
        else:
            print(f"  {name}: NOT CAPTURED")

    # Hidden states per layer
    print(f"\n=== Hidden states per layer (last token) ===")
    for i in range(min(6, len(outputs.hidden_states) - 1)):
        hs = outputs.hidden_states[i + 1]
        t = hs[0, -1]  # last token
        vals = t[:8].tolist()
        max_abs = t.abs().max().item()
        mean_abs = t.abs().mean().item()
        layer_type = type(model.model.layers[i]).__name__
        attn_type = "linear" if hasattr(model.model.layers[i], 'linear_attn') else "full"
        print(f"  Layer {i} ({attn_type}): first8={[f'{v:.4f}' for v in vals]}, "
              f"maxAbs={max_abs:.4f}, meanAbs={mean_abs:.4f}")

    # Logits
    logits = outputs.logits[0, -1]
    top5 = torch.topk(logits, 5)
    print(f"\nTop-5 logits: {[(tokenizer.decode([idx.item()]), f'{val.item():.2f}') for val, idx in zip(top5.values, top5.indices)]}")

    # Also trace through the linear attention manually to compare with Doppler's kernel
    print(f"\n=== Manual linear attention trace (layer 0) ===")
    with torch.no_grad():
        embed = model.model.embed_tokens(input_ids)
        normed = layer0.input_layernorm(embed)
        stats("normed_input", normed)

        qkv = attn.in_proj_qkv(normed)
        stats("qkv", qkv)

        # The HF Qwen3.5 GatedDeltaNet does conv1d on the QKV, then applies SiLU
        # The conv1d expects [batch, channels, seq_len] format
        qkv_t = qkv.transpose(1, 2)  # [1, 6144, 1]

        # Use the conv1d module directly (it has padding configured)
        conv_raw = attn.conv1d(qkv_t)
        stats("conv_raw (from module)", conv_raw.transpose(1, 2))

        # Truncate to seq_len (causal conv padding)
        conv_causal = conv_raw[..., :num_tokens]
        stats("conv_causal (truncated)", conv_causal.transpose(1, 2))

        # Apply SiLU
        conv_silu = torch.nn.functional.silu(conv_causal)
        stats("conv_silu", conv_silu.transpose(1, 2))

        # Split Q, K, V
        conv_out = conv_silu.transpose(1, 2)  # [1, seq_len, 6144]
        num_k_heads = 16
        head_k_dim = 128
        head_v_dim = 128
        num_v_heads = 16
        q_size = num_k_heads * head_k_dim  # 2048
        k_size = q_size
        v_size = num_v_heads * head_v_dim  # 2048

        q = conv_out[..., :q_size]
        k = conv_out[..., q_size:q_size + k_size]
        v = conv_out[..., q_size + k_size:]
        stats("Q (raw)", q)
        stats("K (raw)", k)
        stats("V (raw)", v)

        # Reshape for per-head processing
        # Q and K: [batch, seq, num_k_heads, head_k_dim]
        q_heads = q.view(1, num_tokens, num_k_heads, head_k_dim)
        k_heads = k.view(1, num_tokens, num_k_heads, head_k_dim)
        v_heads = v.view(1, num_tokens, num_v_heads, head_v_dim)

        # L2 normalize Q and K
        eps = 1e-6
        q_norm = torch.nn.functional.normalize(q_heads, p=2, dim=-1, eps=eps)
        k_norm = torch.nn.functional.normalize(k_heads, p=2, dim=-1, eps=eps)

        # Scale Q by 1/sqrt(head_k_dim)
        head_scale = 1.0 / (head_k_dim ** 0.5)
        q_scaled = q_norm * head_scale

        stats("Q_normed_scaled (per-head)", q_scaled.reshape(1, num_tokens, -1))
        stats("K_normed (per-head)", k_norm.reshape(1, num_tokens, -1))

        # Projections for gating
        z = attn.in_proj_z(normed)
        a_out = attn.in_proj_a(normed)
        b_out = attn.in_proj_b(normed)
        stats("z", z)
        stats("a", a_out)
        stats("b", b_out)

        # Compute gating values
        a_log = attn.A_log.detach().float()
        a_neg_exp = -torch.exp(a_log)
        dt_bias = attn.dt_bias.detach().float()

        softplus_input = a_out.squeeze(0).squeeze(0) + dt_bias
        softplus_val = torch.nn.functional.softplus(softplus_input)
        g = a_neg_exp * softplus_val
        g_exp = torch.exp(g)
        beta = torch.sigmoid(b_out.squeeze(0).squeeze(0))

        stats("softplus(a + dt_bias)", softplus_val.unsqueeze(0).unsqueeze(0))
        stats("g (decay)", g.unsqueeze(0).unsqueeze(0))
        stats("g_exp (decay factor)", g_exp.unsqueeze(0).unsqueeze(0))
        stats("beta (sigmoid(b))", beta.unsqueeze(0).unsqueeze(0))

        # Recurrent state update (for first token, state is all zeros)
        # state[head, kd, vd] = state * g_exp + k[kd] * delta[vd]
        # where delta[vd] = (v[vd] - state^T @ k * beta
        # For zero state: delta[vd] = v[vd] * beta, state = k ⊗ delta
        state = torch.zeros(num_v_heads, head_k_dim, head_v_dim)

        # Apply decay (no-op for zero state)
        for head in range(num_v_heads):
            state[head] *= g_exp[head].item()

            k_head = k_norm[0, 0, head % num_k_heads]  # broadcast q_rep
            v_head = v_heads[0, 0, head]

            # kv_mem = state @ k
            kv_mem = state[head].t() @ k_head  # [head_v_dim]

            # delta = (v - kv_mem) * beta
            delta = (v_head - kv_mem) * beta[head].item()

            # state += outer(k, delta)
            state[head] += torch.outer(k_head, delta)

        # Output: out = state^T @ q
        output_per_head = torch.zeros(1, num_tokens, num_v_heads, head_v_dim)
        for head in range(num_v_heads):
            q_head = q_scaled[0, 0, head % num_k_heads]
            out_head = state[head].t() @ q_head  # [head_v_dim]
            output_per_head[0, 0, head] = out_head

        raw_out = output_per_head.reshape(1, num_tokens, num_v_heads * head_v_dim)
        stats("Recurrent output (raw)", raw_out)

        # RMS norm per head + SiLU gate
        z_reshaped = z.view(1, num_tokens, num_v_heads, head_v_dim)
        norm_weight = attn.norm.weight.detach().float()  # [head_v_dim] (shared mode)
        rms_eps = 1e-6

        for head in range(num_v_heads):
            head_out = output_per_head[0, 0, head]  # [head_v_dim]
            mean_sq = (head_out ** 2).mean()
            inv_rms = 1.0 / torch.sqrt(mean_sq + rms_eps)
            z_gate = torch.nn.functional.silu(z_reshaped[0, 0, head])
            output_per_head[0, 0, head] = head_out * inv_rms * norm_weight * z_gate

        gated_out = output_per_head.reshape(1, num_tokens, num_v_heads * head_v_dim)
        stats("After RMSNorm + SiLU gate", gated_out)

        # Output projection
        o_result = torch.nn.functional.linear(gated_out, attn.out_proj.weight)
        stats("After out_proj", o_result)

        # Compare with captured output
        if 'linear_attn_output' in captured:
            diff = (o_result - captured['linear_attn_output']).abs()
            print(f"\n  Diff vs captured output: maxDiff={diff.max().item():.6f}")


if __name__ == "__main__":
    main()
