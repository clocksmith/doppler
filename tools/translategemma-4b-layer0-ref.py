"""
TranslateGemma 4B — Layer-0 activation reference dump.

Captures every intermediate activation at layer 0 for the prompt
"Translate English to French: Hello world." using HF Transformers
with in-memory key remapping. All values are last-token [:8].

No files on disk are modified.
"""

import json
import torch
from safetensors import safe_open
from transformers import AutoTokenizer, AutoConfig, Gemma3ForCausalLM

SNAP = (
    "/media/x/models/huggingface_cache/hub/models--google--translategemma-4b-it"
    "/snapshots/10042cb0e6e7fdce748996a71dc3dc432a4e0c89"
)
SHARDS = [
    f"{SNAP}/model-00001-of-00002.safetensors",
    f"{SNAP}/model-00002-of-00002.safetensors",
]
# Use the SAME chat-templated prompt that Doppler's harness produces
PROMPT = (
    "<start_of_turn>user\n"
    "Instructions: translate the following to French.\n"
    "Produce only the French translation, without any additional explanations or "
    "commentary. Please translate the following English text into French:\n\n\n"
    "Hello world.<end_of_turn>\n"
    "<start_of_turn>model\n"
)
EXPECTED_IDS = None  # Will print actual IDs for Doppler comparison

# ── Load model ────────────────────────────────────────────────────────────────
config = AutoConfig.from_pretrained(SNAP, local_files_only=True)
model = Gemma3ForCausalLM(config)

PREFIX = "language_model."
remapped = {}
for shard_path in SHARDS:
    with safe_open(shard_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            if key.startswith(PREFIX):
                remapped[key[len(PREFIX):]] = f.get_tensor(key)

missing, unexpected = model.load_state_dict(remapped, strict=False)
assert len(unexpected) == 0, f"Unexpected keys: {unexpected}"
assert missing == ["lm_head.weight"], f"Unexpected missing: {missing}"
model = model.to(dtype=torch.bfloat16).eval()
print(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e9:.2f}B params")

# ── Tokenize ──────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(SNAP, local_files_only=True)
inputs = tokenizer(PROMPT, return_tensors="pt")
ids = inputs.input_ids[0].tolist()
if EXPECTED_IDS is not None:
    assert ids == EXPECTED_IDS, f"Token ID mismatch: {ids} != {EXPECTED_IDS}"
    print(f"Token IDs verified: {ids}")
else:
    print(f"Token IDs ({len(ids)}): {ids}")
    print(f"EXPECTED_IDS = {ids}  # paste into Doppler comparison")

input_ids = inputs.input_ids
last = input_ids.shape[1] - 1  # last token index

# ── Hook layer 0 to capture intermediates ─────────────────────────────────────
layer0 = model.model.layers[0]
captures = {}

def fmt(t, label):
    """Format last-token [:8] as list of floats."""
    vals = t[0, last, :8].float().tolist()
    print(f"  {label}: {[round(v, 6) for v in vals]}")
    captures[label] = vals

# Hook: input to layer (post-embedding, pre-norm)
original_forward = layer0.forward

def hooked_forward(hidden_states, **kwargs):
    fmt(hidden_states, "layer0_input")

    # Input norm
    normed = layer0.input_layernorm(hidden_states)
    fmt(normed, "post_input_norm")

    # Q/K/V projections
    q = layer0.self_attn.q_proj(normed)
    k = layer0.self_attn.k_proj(normed)
    v = layer0.self_attn.v_proj(normed)
    fmt(q, "q_proj")
    fmt(k, "k_proj")
    fmt(v, "v_proj")

    # Q/K norms
    q_normed = layer0.self_attn.q_norm(q.view(-1, 8, 256)).view(q.shape)
    k_normed = layer0.self_attn.k_norm(k.view(-1, 4, 256)).view(k.shape)
    fmt(q_normed, "q_after_qknorm")
    fmt(k_normed, "k_after_qknorm")

    # Run full layer normally to get correct outputs
    result = original_forward(hidden_states, **kwargs)
    hidden_out = result[0] if isinstance(result, tuple) else result

    fmt(hidden_out, "layer0_output")
    return result

layer0.forward = hooked_forward

# ── Run forward pass ──────────────────────────────────────────────────────────
print("\nRunning forward pass...")
with torch.no_grad():
    out = model(input_ids=input_ids, output_hidden_states=True)

# Embedding output
embed_out = out.hidden_states[0]
fmt(embed_out, "embed_out")

# Logits
logits = out.logits
logits_last = logits[0, last, :5].float().tolist()
argmax = logits[0, last].argmax().item()
print(f"\n  logits[last, :5]: {[round(v, 4) for v in logits_last]}")
print(f"  logits argmax: {argmax}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n=== REFERENCE VALUES (last token, dims 0:8) ===")
for label, vals in captures.items():
    print(f"{label:20s}: {[round(v, 6) for v in vals]}")
