import torch
from safetensors.torch import load_file
import os

repo_dir = "/Users/xyz/.cache/huggingface/hub/models--google--functiongemma-270m-it/snapshots/ead2a1f9df8d6431408ccff6c9e5e60028addde0"
for f in os.listdir(repo_dir):
    if f.endswith(".safetensors"):
        tensors = load_file(os.path.join(repo_dir, f))
        for k, v in tensors.items():
            if "post_feedforward_layernorm.weight" in k:
                print(f"{k}: max={v.max().item():.2f}, min={v.min().item():.2f}, mean={v.mean().item():.2f}")
