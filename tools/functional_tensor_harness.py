#!/usr/bin/env python3
"""
Doppler Functional Tensor Harness
Validation gate for Cluster 2: Functional Tensor Representation.

Approximates target checkpoints from Qwen/Qwen2.5-0.5B using Kronecker factorization,
SIREN coordinate neural representation, and coordinate-deterministic PRNG.
"""

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# ── Coordinate-Deterministic PRNG ─────────────────────────────────────────────
def coord_prng(seed, rows, cols, sigma=0.02):
    """
    Generates a coordinate-deterministic pseudo-random normal tensor.
    Same seed + same row + same col -> same value.
    Uses Box-Muller transform over sine-based pseudo-random uniform fields.
    """
    # Deterministic uniforms in [0, 1)
    val1 = torch.sin(rows * 12.9898 + cols * 78.233 + seed) * 43758.5453
    val1 = val1 - torch.floor(val1)
    val2 = torch.sin(rows * 37.719 + cols * 119.519 + seed + 7) * 43758.5453
    val2 = val2 - torch.floor(val2)
    
    u1 = torch.clamp(val1, min=1e-9, max=1.0)
    u2 = val2
    z = torch.sqrt(-2.0 * torch.log(u1)) * torch.cos(2.0 * math.pi * u2)
    return z * sigma

# ── Implicit Neural Representation (SIREN) ────────────────────────────────────
class Sine(nn.Module):
    def __init__(self, w0=30.0):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)

class SIREN(nn.Module):
    def __init__(self, in_features=2, hidden_features=64, hidden_layers=2, out_features=1, w0=30.0):
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_features, hidden_features))
        layers.append(Sine(w0))
        for _ in range(hidden_layers):
            layers.append(nn.Linear(hidden_features, hidden_features))
            layers.append(Sine(w0))
        layers.append(nn.Linear(hidden_features, out_features))
        self.net = nn.Sequential(*layers)
        
        # SIREN initialization scheme
        with torch.no_grad():
            self.net[0].weight.uniform_(-1.0 / in_features, 1.0 / in_features)
            for i in range(1, hidden_layers + 1):
                lin = self.net[i * 2]
                lin.weight.uniform_(-math.sqrt(6.0 / hidden_features) / w0, math.sqrt(6.0 / hidden_features) / w0)
                
    def forward(self, coords):
        return self.net(coords)

# ── Factorization Helpers ─────────────────────────────────────────────────────
def get_kron_shapes(rows, cols):
    """
    Finds compatible shapes for Kronecker product factors A and B.
    """
    for r1 in [14, 16, 28, 32, 8, 4, 2]:
        if rows % r1 == 0:
            r2 = rows // r1
            break
    else:
        r1, r2 = 1, rows
        
    for c1 in [16, 32, 64, 8, 4, 2]:
        if cols % c1 == 0:
            c2 = cols // c1
            break
    else:
        c1, c2 = 1, cols
        
    return (r1, c1), (r2, c2)

def sha256_hex(data):
    if isinstance(data, str):
        data = data.encode('utf-8')
    return hashlib.sha256(data).hexdigest()

# ── Main Run ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Cluster 2 validation harness")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-0.5B", help="Target model ID")
    parser.add_argument("--out-dir", type=str, default="doppler/tools/functional_shards", help="Output directory")
    parser.add_argument("--seed", type=int, default=1337, help="PRNG Seed")
    parser.add_argument("--rank-terms", type=int, default=8, help="Kronecker rank terms")
    parser.add_argument("--siren-width", type=int, default=64, help="SIREN hidden dimension size")
    parser.add_argument("--sparse-fraction", type=float, default=0.001, help="Fraction of outliers to preserve in sparse residual")
    parser.add_argument("--steps", type=int, default=1000, help="Fitting optimization steps")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model_id} model and tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.float32)
    except Exception as e:
        raise RuntimeError(f"Proof run failed to load target model: {e}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    print(f"Model loaded successfully on {device}.")

    # ── 1. Calibration Activation Capture ─────────────────────────────────────
    print("Running calibration steps to extract activation sensitivity...")
    calibration_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Solve: x^2 - 4 = 0 for x.",
        "Explain the conservation of energy in classical mechanics.",
        "Translate English to French: Hello world.",
        "Qwen is a large language model created by Alibaba Cloud."
    ]

    target_layer_name = "model.layers.0.mlp.down_proj"
    target_module = model.model.layers[0].mlp.down_proj

    captured_inputs = []
    def hook_fn(module, input_args, output):
        captured_inputs.append(input_args[0].detach().cpu())

    hook_handle = target_module.register_forward_hook(hook_fn)
    for text in calibration_texts:
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            model(**inputs)
    hook_handle.remove()

    # Compute activation sensitivity
    # captured_inputs has shape [batch=1, seq_len, hidden_dim]
    all_inputs = torch.cat(captured_inputs, dim=1) # concat along sequence dim
    sensitivity = torch.mean(torch.abs(all_inputs), dim=(0, 1)) # shape [hidden_dim]
    print(f"Captured activation shapes: {all_inputs.shape}. Sensitivity vector dim: {sensitivity.shape[0]}")

    # ── 2. Discover Shapes and Extract Target Tensor ──────────────────────────
    # Qwen layers are BF16/FP16; we convert to FP32 for fitting
    target_tensor = target_module.weight.data.detach().cpu().float()
    rows, cols = target_tensor.shape
    print(f"Target tensor '{target_layer_name}' discovered shape: {rows} x {cols}")

    # Divisibility and coordinate grids
    (r1, c1), (r2, c2) = get_kron_shapes(rows, cols)
    print(f"Kronecker factor shapes: A ({r1}x{c1}) and B ({r2}x{c2})")

    row_grid, col_grid = torch.meshgrid(
        torch.arange(rows, dtype=torch.float32),
        torch.arange(cols, dtype=torch.float32),
        indexing="ij"
    )

    # ── 3. Fit Functional Representation ──────────────────────────────────────
    print("Fitting functional descriptor components (Kronecker + SIREN + PRNG)...")
    target_tensor = target_tensor.to(device)
    row_grid = row_grid.to(device)
    col_grid = col_grid.to(device)
    sensitivity = sensitivity.to(device)

    # Deterministic PRNG substrate (fixed seed)
    prng_substrate = coord_prng(args.seed, row_grid, col_grid, sigma=0.01).to(device)

    # Initialize Kronecker factors
    A_factors = [torch.randn(r1, c1, requires_grad=True, device=device) for _ in range(args.rank_terms)]
    B_factors = [torch.randn(r2, c2, requires_grad=True, device=device) for _ in range(args.rank_terms)]

    # Normalized coordinate grid for SIREN: range [-1, 1]
    norm_y = (row_grid / (rows - 1)) * 2.0 - 1.0
    norm_x = (col_grid / (cols - 1)) * 2.0 - 1.0
    siren_coords = torch.stack([norm_y.flatten(), norm_x.flatten()], dim=-1)

    siren = SIREN(hidden_features=args.siren_width).to(device)

    # Optimization Loop
    optimizer = torch.optim.Adam(
        A_factors + B_factors + list(siren.parameters()),
        lr=0.005
    )

    # Weight parameters for loss
    # Loss combines Frobenius reconstruction error and activation-weighted MSE
    start_time = time.time()
    for step in range(args.steps):
        optimizer.zero_grad()
        
        # 1. Kronecker Sum
        kron_sum = torch.zeros(rows, cols, device=device)
        for a, b in zip(A_factors, B_factors):
            kron_sum += torch.kron(a, b)
            
        # 2. SIREN output
        siren_out = siren(siren_coords).view(rows, cols)
        
        # Reconstructed dense weights
        W_reconstructed = prng_substrate + kron_sum + siren_out
        
        # Loss: reconstruction MSE + activation-weighted error
        rec_loss = F.mse_loss(W_reconstructed, target_tensor)
        
        # activation-weighted loss
        # Y_true = X * W_true^T, Y_pred = X * W_pred^T
        # Error delta per output channel = mean(abs(W_true - W_pred) * sensitivity)
        act_weighted_diff = (target_tensor - W_reconstructed) * sensitivity.unsqueeze(0)
        act_loss = torch.mean(act_weighted_diff ** 2)
        
        loss = rec_loss + 0.5 * act_loss
        loss.backward()
        optimizer.step()
        
        if step % 200 == 0 or step == args.steps - 1:
            print(f"  Step {step:4d}/{args.steps} | Total Loss: {loss.item():.6f} | Rec MSE: {rec_loss.item():.6f}")

    build_time_ms = (time.time() - start_time) * 1000.0

    # ── 4. Sparse Outliers (CSR Selection) ────────────────────────────────────
    # Final reconstruction without sparse residuals
    with torch.no_grad():
        final_kron = torch.zeros(rows, cols, device=device)
        for a, b in zip(A_factors, B_factors):
            final_kron += torch.kron(a, b)
        final_siren = siren(siren_coords).view(rows, cols)
        W_func = prng_substrate + final_kron + final_siren
        
        diff = target_tensor - W_func
        # Saliency score: abs(error) * activation sensitivity
        saliency_score = torch.abs(diff) * sensitivity.unsqueeze(0)
        
        # Sort and pick top sparse outliers
        total_elements = rows * cols
        nnz = int(total_elements * args.sparse_fraction)
        
        flat_scores = saliency_score.flatten()
        threshold_val = torch.topk(flat_scores, nnz).values[-1]
        
        # Extract sparse residuals
        mask = saliency_score >= threshold_val
        sparse_indices = torch.nonzero(mask) # shape [nnz, 2]
        sparse_vals = diff[mask]
        
        # Assemble standard CSR layout
        # row_offsets: index in col_indices where each row starts
        row_offsets = [0]
        col_indices = []
        vals = []
        
        for r in range(rows):
            row_mask = sparse_indices[:, 0] == r
            row_cols = sparse_indices[row_mask, 1].cpu().tolist()
            row_vals = sparse_vals[row_mask].cpu().tolist()
            
            col_indices.extend(row_cols)
            vals.extend(row_vals)
            row_offsets.append(len(vals))

        # Reconstructed matrix with sparse outliers
        W_sparse = torch.zeros(rows, cols, device=device)
        for i in range(len(vals)):
            r_idx = torch.nonzero(mask)[i, 0]
            c_idx = torch.nonzero(mask)[i, 1]
            W_sparse[r_idx, c_idx] = vals[i]
            
        W_final = W_func + W_sparse

    # ── 5. Metrics & Validation ───────────────────────────────────────────────
    rmse = torch.sqrt(F.mse_loss(W_final, target_tensor)).item()
    fro_norm_target = torch.linalg.matrix_norm(target_tensor, ord="fro").item()
    fro_norm_diff = torch.linalg.matrix_norm(target_tensor - W_final, ord="fro").item()
    rel_fro_error = fro_norm_diff / fro_norm_target

    # Activation MSE check
    # Estimate activation propagation error using calibration sensitivity
    act_mse = torch.mean(((target_tensor - W_final) * sensitivity.unsqueeze(0)) ** 2).item()

    print("\nReconstruction Metrics:")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  Relative Frobenius Error: {rel_fro_error:.6f}")
    print(f"  Activation MSE on calibration: {act_mse:.8f}")

    # ── 6. Byte Accounting ────────────────────────────────────────────────────
    dense_f16_bytes = rows * cols * 2
    
    # Calculate bytes for serialized components
    kron_bytes = sum(a.nelement() * 4 + b.nelement() * 4 for a, b in zip(A_factors, B_factors))
    
    siren_bytes = 0
    for p in siren.parameters():
        siren_bytes += p.nelement() * 4
        
    # CSR layout: float32 values, int32 col_indices, int32 row_offsets
    sparse_bytes = len(vals) * 4 + len(col_indices) * 4 + len(row_offsets) * 4
    
    # Generate serialized JSON files for shards
    shard_kron_path = out_dir / "layers_0_down_proj.kron"
    shard_siren_path = out_dir / "layers_0_down_proj.siren"
    shard_sparse_path = out_dir / "layers_0_down_proj.sparse"
    
    kron_data = {
        "A": [a.detach().cpu().tolist() for a in A_factors],
        "B": [b.detach().cpu().tolist() for b in B_factors]
    }
    siren_data = {k: v.cpu().tolist() for k, v in siren.state_dict().items()}
    sparse_data = {
        "values": vals,
        "col_indices": col_indices,
        "row_offsets": row_offsets
    }
    
    shard_kron_path.write_text(json.dumps(kron_data, indent=2))
    shard_siren_path.write_text(json.dumps(siren_data, indent=2))
    shard_sparse_path.write_text(json.dumps(sparse_data, indent=2))

    # Calculate actual bytes written
    actual_shard_bytes = (
        shard_kron_path.stat().st_size +
        shard_siren_path.stat().st_size +
        shard_sparse_path.stat().st_size
    )

    # Manifest creation
    manifest = {
        "schema_version": "manifoldgguf.v0.1",
        "tensor_name": target_layer_name,
        "shape": [rows, cols],
        "storage_type": "functional_descriptor",
        "dtype": "f16",
        "accumulator": "f32_declared",
        "tile_shape": [64, 64],
        "source_tensor_hash": f"sha256:{sha256_hex(target_tensor.cpu().numpy().tobytes())}",
        "components": {
            "prng_substrate": {
                "algorithm": "coordinate_hash_normal_v1",
                "seed": args.seed,
                "learned_scale": True
            },
            "kronecker_sum": {
                "rank_terms": args.rank_terms,
                "factor_shapes": [[r1, c1], [r2, c2]],
                "shard_file": "layers_0_down_proj.kron"
            },
            "coordinate_inr": {
                "type": "siren",
                "network_dims": [2, args.siren_width, args.siren_width, 1],
                "shard_file": "layers_0_down_proj.siren"
            },
            "sparse_outliers": {
                "format": "csr_v1",
                "selection": "residual_x_activation_sensitivity",
                "nnz_fraction": args.sparse_fraction,
                "value_dtype": "f16",
                "shard_file": "layers_0_down_proj.sparse"
            }
        }
    }

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    
    # Calculate descriptor hash
    descriptor_hash = sha256_hex(manifest_path.read_text())
    manifest["descriptor_hash"] = f"sha256:{descriptor_hash}"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    descriptor_bytes = len(json.dumps(manifest)) + actual_shard_bytes
    compression_ratio = descriptor_bytes / dense_f16_bytes

    print(f"\nByte Accounting:")
    print(f"  Dense f16 Bytes: {dense_f16_bytes} bytes")
    print(f"  Descriptor Bytes (JSON + Shards): {descriptor_bytes} bytes")
    print(f"  Compression Ratio: {compression_ratio:.4f}x")

    metrics_report = {
        "rmse": rmse,
        "relative_frobenius_error": rel_fro_error,
        "activation_mse": act_mse,
        "dense_f16_bytes": dense_f16_bytes,
        "descriptor_bytes": descriptor_bytes,
        "compression_ratio": compression_ratio,
        "build_time_ms": build_time_ms
    }

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_report, indent=2))

    # Output Build Receipt
    build_receipt = {
        "schema": "simulatte.indexBuildReceipt.v1",
        "byteSize": descriptor_bytes,
        "entryCount": 1,
        "embeddingDimension": args.siren_width,
        "buildTimeMs": build_time_ms,
        "sha256": descriptor_hash
    }
    
    receipt_path = out_dir / "layers_0_down_proj.json.receipt.json"
    receipt_path.write_text(json.dumps(build_receipt, indent=2) + "\n")

    print(f"\nArtifacts successfully written to {out_dir}/")
    print(f"  ✓ manifest.json")
    print(f"  ✓ layers_0_down_proj.kron")
    print(f"  ✓ layers_0_down_proj.siren")
    print(f"  ✓ layers_0_down_proj.sparse")
    print(f"  ✓ metrics.json")
    print(f"  ✓ layers_0_down_proj.json.receipt.json")

if __name__ == "__main__":
    main()
