"""
TranslateGemma 4B — HuggingFace reference run with in-memory key remapping.

Loads the multimodal checkpoint read-only, remaps language_model.* keys to
model.* in memory, instantiates Gemma3ForCausalLM with the on-disk gemma3_text
config, and runs one deterministic translation.

No files on disk are modified.

Usage:
  python3 tools/translategemma-4b-hf-reference.py
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

# ── 1. Load config (gemma3_text on disk) ──────────────────────────────────────
print("Loading config...")
config = AutoConfig.from_pretrained(SNAP, local_files_only=True)
print(f"  model_type={config.model_type}  hidden_size={config.hidden_size}  "
      f"layers={config.num_hidden_layers}  heads={config.num_attention_heads}  "
      f"kv_heads={config.num_key_value_heads}")

# ── 2. Instantiate empty model ────────────────────────────────────────────────
print("Creating empty Gemma3ForCausalLM on CPU...")
model = Gemma3ForCausalLM(config)

# ── 3. Load + remap weights in memory ─────────────────────────────────────────
print("Loading SafeTensors with in-memory key remap...")
PREFIX = "language_model."
remapped = {}
skipped_prefixes = set()
for shard_path in SHARDS:
    with safe_open(shard_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            if key.startswith(PREFIX):
                new_key = key[len(PREFIX):]
                remapped[new_key] = f.get_tensor(key)
            else:
                prefix = key.split(".")[0]
                skipped_prefixes.add(prefix)

print(f"  Remapped {len(remapped)} text-model tensors")
print(f"  Skipped prefixes: {sorted(skipped_prefixes)}")

# Load into model
missing, unexpected = model.load_state_dict(remapped, strict=False)
print(f"  Missing keys: {len(missing)}")
print(f"  Unexpected keys: {len(unexpected)}")
if missing:
    print(f"    Missing: {missing[:5]}{'...' if len(missing) > 5 else ''}")
if unexpected:
    print(f"    Unexpected: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

model = model.to(dtype=torch.bfloat16).eval()
param_count = sum(p.numel() for p in model.parameters()) / 1e9
print(f"  Model ready: {param_count:.2f}B params, dtype=bfloat16")

# ── 4. Tokenize ──────────────────────────────────────────────────────────────
print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(SNAP, local_files_only=True)

prompt = "Translate English to French: Hello world."
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs.input_ids
print(f"Prompt: {repr(prompt)}")
print(f"Token count: {input_ids.shape[1]}")
print(f"Token IDs: {input_ids[0].tolist()}")

# ── 5. Generate ──────────────────────────────────────────────────────────────
print("\nGenerating (CPU, bfloat16, greedy)...")
with torch.no_grad():
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=32,
        do_sample=False,
    )

generated_ids = outputs[0][input_ids.shape[1]:]
text = tokenizer.decode(generated_ids, skip_special_tokens=True)
print(f"Output text: {repr(text)}")
print(f"Output IDs: {generated_ids.tolist()}")

# ── 6. Capture layer-0 hidden state + final logits for reference ─────────────
print("\nCapturing layer-0 hidden state and logits...")
with torch.no_grad():
    out = model(input_ids=input_ids, output_hidden_states=True)

logits = out.logits  # [1, seq_len, vocab_size]
hidden_0 = out.hidden_states[1]  # [1, seq_len, hidden_size] — after layer 0

last_pos = input_ids.shape[1] - 1
print(f"Logits shape: {logits.shape}")
print(f"Logits[last_pos, :5]: {logits[0, last_pos, :5].float().tolist()}")
print(f"Logits argmax: {logits[0, last_pos].argmax().item()}")
print(f"Hidden L0 shape: {hidden_0.shape}")
print(f"Hidden L0[last_pos, :8]: {hidden_0[0, last_pos, :8].float().tolist()}")

# Also capture embed output (hidden_states[0]) for comparison
embed_out = out.hidden_states[0]  # [1, seq_len, hidden_size] — embedding output
print(f"Embed out[last_pos, :8]: {embed_out[0, last_pos, :8].float().tolist()}")
print(f"Embed out[0, :8]: {embed_out[0, 0, :8].float().tolist()}")
