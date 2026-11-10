#!/usr/bin/env python3
"""CPU-only causal-prefix verification and full-prefix (M=113) same-input operator
controls across Layers 0..24 for Qwen3.8-Flash-Next on AMD RDNA-3.

Usage:
  python3 -P reports/flash-next-new-prompts-20261110/audit-same-input-operators.py \
    --producer-dir PRODUCER_DIR \
    --source-dir SOURCE_DIR \
    --artifact-dir ARTIFACT_DIR \
    --causal-cache CAUSAL_CACHE \
    --new-prompts-cache NEW_PROMPTS_CACHE \
    --full-gpu-dir FULL_GPU_DIR \
    --out-causal causal-prefix-control.json \
    --out-controls same-input-operator-controls.json

No tensors feed back into a model, no runtime behavior changes, and no
acceptance threshold changes.
"""
import argparse
import ast
import hashlib
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from functools import wraps
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gguf import GGMLQuantizationType, quants


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def diff_metrics(a: np.ndarray, b: np.ndarray) -> dict:
    a32 = np.asarray(a, dtype=np.float32)
    b32 = np.asarray(b, dtype=np.float32)
    assert a32.shape == b32.shape, (a32.shape, b32.shape)
    assert np.isfinite(a32).all() and np.isfinite(b32).all(), "Non-finite tensor values"
    d = a32.astype(np.float64) - b32.astype(np.float64)
    abs_d = np.abs(d)
    max_idx = int(np.argmax(abs_d))
    denom = float(np.linalg.norm(b32.astype(np.float64).ravel()))
    num = float(np.linalg.norm(d.ravel()))
    bit_diffs = int(np.sum(a32.view(np.uint32) != b32.view(np.uint32)))
    return {
        "maxAbs": float(abs_d.ravel()[max_idx]),
        "meanAbs": float(np.mean(abs_d)),
        "relL2": float(num / denom) if denom > 0.0 else (0.0 if num == 0.0 else float("inf")),
        "maxFlatIndex": max_idx,
        "bitDifferingValues": bit_diffs,
        "bitExact": bit_diffs == 0,
    }


def pair_metrics(a_full: np.ndarray, b_full: np.ndarray, pos: int = 112) -> dict:
    return {
        "all113": diff_metrics(a_full, b_full),
        "pos112": diff_metrics(a_full[pos], b_full[pos]),
    }


def exact_fma_f32(a: np.float32, b: np.float32, c: np.float32) -> np.float32:
    """Exact IEEE-754 binary32 single-rounding fused multiply-add via Fraction."""
    exact = Fraction(float(a)) * Fraction(float(b)) + Fraction(float(c))
    cand = np.float32(float(exact))
    if Fraction(float(cand)) == exact:
        return cand
    ulp_up = np.nextafter(cand, np.float32(np.inf), dtype=np.float32)
    ulp_dn = np.nextafter(cand, np.float32(-np.inf), dtype=np.float32)
    return min(
        (ulp_dn, cand, ulp_up),
        key=lambda x: (abs(Fraction(float(x)) - exact), int(np.float32(x).view(np.uint32)) & 1),
    )


def simulate_subgroup_dot_f32(x_row: np.ndarray, w_row: np.ndarray, wave_size: int) -> dict:
    """Exact reproduction of matmul_f32_row_subgroup.wgsl (transpose_b = 1, K = 2560, alpha = 1.0)."""
    k_dim = x_row.shape[0]
    assert k_dim == 2560 and w_row.shape == (2560,)
    lanes = np.zeros((wave_size, 4), dtype=np.float32)
    for rank in range(wave_size):
        s = [np.float32(0.0)] * 4
        for k in range(rank * 4, k_dim, wave_size * 4):
            for c in range(4):
                s[c] = exact_fma_f32(np.float32(x_row[k + c]), np.float32(w_row[k + c]), s[c])
        lanes[rank] = s
    cur = lanes.copy()
    step = 1
    while step < wave_size:
        nxt = np.empty((cur.shape[0] // 2, 4), dtype=np.float32)
        for i in range(nxt.shape[0]):
            nxt[i] = cur[2 * i] + cur[2 * i + 1]
        cur = nxt
        step *= 2
    total = cur[0]
    xyzw = np.float32(np.float32(np.float32(total[0] + total[1]) + total[2]) + total[3])
    seq_total = np.zeros(4, dtype=np.float32)
    for rank in range(wave_size):
        seq_total = np.float32(seq_total + lanes[rank])
    seq_out = np.float32(np.float32(np.float32(seq_total[0] + seq_total[1]) + seq_total[2]) + seq_total[3])
    return {
        "waveSize": wave_size,
        "treeSubgroupAdd": float(xyzw),
        "sequentialSubgroupAdd": float(seq_out),
        "laneVectorTotal": [float(v) for v in total],
    }


def build_upstream_namespace(producer_dir: Path, source_dir: Path, artifact_dir: Path):
    binding = json.loads((producer_dir / "source-binding.json").read_text())
    for entry in binding["files"] + binding["referenceImplementation"]["files"]:
        actual = sha256_file(producer_dir / "source" / entry["path"])
        assert actual == entry["sha256"], f"source mismatch: {entry['path']}"
    headers = json.loads((producer_dir / "checkpoint-headers.json").read_text())
    acquired = {
        f["file"]: f
        for f in json.loads((producer_dir / "acquisition-receipt.json").read_text())["files"]
    }
    config = json.loads((producer_dir / "source/config.json").read_text())["text_config"]
    config_class = next(
        n
        for n in ast.parse((producer_dir / "source/configuration_qwen4_exp.py").read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "Qwen4ExpTextConfig"
    )
    defaults = {}
    for n in config_class.body:
        if isinstance(n, ast.AnnAssign) and n.value is not None:
            try:
                defaults[n.target.id] = ast.literal_eval(n.value)
            except ValueError:
                pass
    for k, v in defaults.items():
        config.setdefault(k, v)
    cfg = SimpleNamespace(**config)
    cfg.ple_embed_dim = cfg.ple_embed_dim or cfg.hidden_size
    cfg._attn_implementation = "eager"
    cfg._experts_implementation = "eager"

    def source_nodes(path, names, decorators=None, method_decorators=None):
        found = {}
        for node in ast.parse((producer_dir / "source" / path).read_text()).body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names:
                assert [ast.unparse(d) for d in node.decorator_list] == (decorators or {}).get(node.name, []), node.name
                node.decorator_list = []
                for method, expected in (method_decorators or {}).get(node.name, {}).items():
                    member = next(m for m in node.body if isinstance(m, ast.FunctionDef) and m.name == method)
                    assert [ast.unparse(d) for d in member.decorator_list] == expected, method
                    member.decorator_list = []
                found[node.name] = node
            elif isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id in names for t in node.targets
            ):
                found[node.targets[0].id] = node
        assert set(found) == set(names), set(names) - set(found)
        return [found[name] for name in names]

    def run_ast(nodes, ns, label):
        future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[])), label, "exec"), ns)

    class EagerOnlyRegistry:
        @staticmethod
        def get_interface(name, default):
            if name != "eager":
                raise ValueError(f"unsupported implementation {name!r}")
            return default

    row_width = cfg.ple_embed_dim // ((cfg.ngram_size - 1) * cfg.heads_per_ngram)
    table = headers["ngramTable"]
    shard_rows = table["shardRows"][0]
    row_bytes = table["rowWidth"] * 2
    table_shards = {s["shard"]: s for s in table["shards"]}
    handles = {}

    def handle(name):
        if name not in handles:
            path = source_dir / name
            assert path.stat().st_size == acquired[name]["bytes"], name
            handles[name] = open(path, "rb")
        return handles[name]

    class TableRows(nn.Module):
        def __init__(self, count, width):
            super().__init__()
            assert count == table["rows"] and width == row_width
            self.weight = nn.Parameter(torch.empty(0), requires_grad=False)

        def forward(self, indices):
            flat = indices.reshape(-1).tolist()
            out = np.empty((len(flat), row_width), dtype=np.uint16)
            for i, row in enumerate(flat):
                shard = table_shards[row // shard_rows]
                h = handle(shard["file"])
                h.seek(shard["fileByteBegin"] + (row % shard_rows) * row_bytes)
                out[i] = np.frombuffer(h.read(row_bytes), dtype=np.uint16)
            values = torch.from_numpy((out.astype(np.uint32) << 16).view(np.float32))
            return values.reshape(*indices.shape, row_width).to(torch.float32)

    class NNProxy:
        def __getattr__(self, name):
            return TableRows if name == "Embedding" else getattr(nn, name)

    ns = {
        "torch": torch,
        "nn": NNProxy(),
        "F": F,
        "math": math,
        "Tensor": torch.Tensor,
        "Cache": object,
        "TransformersKwargs": dict,
        "Unpack": type("Unpack", (), {"__class_getitem__": classmethod(lambda cls, item: item)}),
        "GradientCheckpointingLayer": nn.Module,
        "is_torchdynamo_exporting": lambda: False,
        "is_torchdynamo_compiling": lambda: False,
        "ALL_ATTENTION_FUNCTIONS": EagerOnlyRegistry,
        "logger": SimpleNamespace(warning_once=lambda *a, **k: None),
        "Callable": object,
        "wraps": wraps,
    }
    run_ast(
        source_nodes("activations.py", ["SiLUActivation"], {"SiLUActivation": ["use_kernel_forward_from_hub('SiLU')"]}),
        ns,
        "<act>",
    )
    ns["ACT2FN"] = {"silu": ns["SiLUActivation"](), "sigmoid": nn.Sigmoid()}
    run_ast(
        [
            n
            for n in ast.parse((producer_dir / "source/modeling_rope_utils.py").read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == "dynamic_rope_update"
        ],
        ns,
        "<rope-utils>",
    )
    names = [
        "Qwen4ExpTextRotaryEmbedding",
        "Qwen4ExpTextRMSNorm",
        "Qwen4ExpTextRMSNormGated",
        "apply_mask_to_padding_states",
        "causal_conv1d_update",
        "causal_conv1d_fn",
        "l2norm",
        "torch_chunk_gated_delta_rule",
        "torch_recurrent_gated_delta_rule",
        "Qwen4ExpTextGatedDeltaNet",
        "rotate_half",
        "apply_rotary_pos_emb",
        "Qwen4ExpTextQSAIndexer",
        "repeat_kv",
        "eager_attention_forward",
        "Qwen4ExpTextAttention",
        "Qwen4ExpTextMLP",
        "Qwen4ExpTextExperts",
        "Qwen4ExpTextTopKRouter",
        "Qwen4ExpTextSparseMoeBlock",
        "Qwen4ExpTextGatedResidual",
        "_MASK64",
        "_SPLITMIX_GAMMA",
        "_SPLITMIX_M1",
        "_SPLITMIX_M2",
        "_PRIME_1",
        "_splitmix64",
        "_build_layer_multipliers",
        "_is_prime",
        "_find_nth_prime_after",
        "Qwen4ExpTextNGramEmbedding",
        "Qwen4ExpTextPLELayer",
        "Qwen4ExpTextDecoderLayer",
    ]
    decorators = {
        "Qwen4ExpTextRMSNormGated": ["use_kernel_forward_from_hub('RMSNormGated')"],
        "causal_conv1d_update": ["use_kernel_func_from_hub_with_fallback('causal_conv1d_update', 'causal_conv1d')"],
        "causal_conv1d_fn": ["use_kernel_func_from_hub_with_fallback('causal_conv1d_fn', 'causal_conv1d')"],
        "torch_chunk_gated_delta_rule": ["use_kernel_func_from_hub_with_fallback('chunk_gated_delta_rule', 'fla')"],
        "torch_recurrent_gated_delta_rule": [
            "use_kernel_func_from_hub_with_fallback('fused_recurrent_gated_delta_rule', 'fla')"
        ],
        "Qwen4ExpTextGatedDeltaNet": [
            "use_kernelized_func([torch_recurrent_gated_delta_rule, torch_chunk_gated_delta_rule, causal_conv1d_fn, causal_conv1d_update])"
        ],
        "Qwen4ExpTextExperts": ["use_experts_implementation"],
    }
    run_ast(
        source_nodes(
            "modeling_qwen4_exp.py",
            names,
            decorators,
            {"Qwen4ExpTextGatedDeltaNet": {"forward": ["force_accelerate_hooks('conv1d')"]}},
        ),
        ns,
        "<pinned-qwen4-exp-text>",
    )

    locations = {}
    for fname, info in headers["files"].items():
        for tensor, t in info["tensors"].items():
            locations[tensor] = (fname, info["dataStart"], t)

    manifest_bytes = (artifact_dir / "manifest.json").read_bytes()
    artifact_manifest = json.loads(manifest_bytes)
    assert artifact_manifest["artifactIdentity"]["sourceRevision"] == binding["revision"]

    def read_source_tensor(name):
        fname, data_start, t = locations[name]
        begin, end = t["dataOffsets"]
        h = handle(fname)
        h.seek(data_start + begin)
        raw = h.read(end - begin)
        if t["dtype"] == "BF16":
            bits = torch.frombuffer(bytearray(raw), dtype=torch.int16).view(torch.bfloat16)
            return bits.reshape(t["shape"]).to(torch.float32)
        assert t["dtype"] == "I64", t["dtype"]
        return torch.frombuffer(bytearray(raw), dtype=torch.int64).reshape(t["shape"]).clone()

    def read_raw_artifact_bytes(name):
        t = artifact_manifest["tensors"][name]
        assert t.get("layout", "row") == "row", name
        spans = t.get("spans") or [{"shardIndex": t["shard"], "offset": t["offset"], "size": t["size"]}]
        raw = bytearray(t["size"])
        cursor = 0
        for span in spans:
            shard = artifact_manifest["shards"][span["shardIndex"]]
            with open(artifact_dir / shard["filename"], "rb") as f:
                f.seek(span["offset"])
                chunk = f.read(span["size"])
            assert len(chunk) == span["size"], name
            raw[cursor : cursor + len(chunk)] = chunk
            cursor += len(chunk)
        assert cursor == t["size"], name
        return raw, t

    pool = ThreadPoolExecutor(max_workers=8)

    def dequantize_q4k_parallel(encoded_2d: np.ndarray, width: int) -> np.ndarray:
        rows = encoded_2d.shape[0]
        if rows <= 512:
            return quants.dequantize(encoded_2d, GGMLQuantizationType.Q4_K)[:, :width].copy()
        chunks = np.array_split(encoded_2d, 8, axis=0)
        decoded = list(pool.map(lambda c: quants.dequantize(c, GGMLQuantizationType.Q4_K)[:, :width], chunks))
        return np.concatenate(decoded, axis=0)

    def read_tensor(name, expert_subset=None):
        if locations[name][2]["dtype"] == "I64":
            return read_source_tensor(name)
        raw, t = read_raw_artifact_bytes(name)
        if t["dtype"] == "F16":
            values = np.frombuffer(raw, dtype="<f2").astype(np.float32)
            return torch.from_numpy(values.reshape(t["shape"]).copy())
        assert t["dtype"] in ("Q4_K", "Q4_K_M"), (name, t["dtype"])
        width = t["shape"][-1]
        padded = ((width + 255) // 256) * 256
        row_bytes_q4 = (padded // 256) * 144
        rows = math.prod(t["shape"][:-1])
        assert len(raw) == rows * row_bytes_q4, name
        if expert_subset is not None and len(t["shape"]) == 3:
            num_e, r_dim, w_dim = t["shape"]
            out = torch.zeros((num_e, r_dim, w_dim), dtype=torch.float32)
            encoded_3d = np.frombuffer(raw, dtype=np.uint8).reshape(num_e, r_dim, row_bytes_q4)
            sub_ids = sorted(set(int(e) for e in expert_subset))
            if sub_ids:
                sub_enc = encoded_3d[sub_ids].reshape(len(sub_ids) * r_dim, row_bytes_q4)
                sub_dec = dequantize_q4k_parallel(sub_enc, w_dim).reshape(len(sub_ids), r_dim, w_dim)
                out[sub_ids] = torch.from_numpy(sub_dec)
            return out
        encoded = np.frombuffer(raw, dtype=np.uint8).reshape(rows, row_bytes_q4)
        values = dequantize_q4k_parallel(encoded, width)
        return torch.from_numpy(values.reshape(t["shape"]))

    return ns, cfg, manifest_bytes, read_tensor, handles, pool


def build_causal_prefix_summary(causal_cache: Path, new_prompts_cache: Path, script_sha: str) -> dict:
    ref_prefix_path = causal_cache / "reference/learned-forward-reference-float32-artifact.json"
    ref_full_path = new_prompts_cache / "reference/learned-forward-reference-float32-artifact.json"
    gpu_comp_path = causal_cache / "gpu-prefix-comparison.json"
    inputs_prefix_path = causal_cache / "inputs.json"
    inputs_full_path = new_prompts_cache / "frozen/inputs.json"
    cpu_receipt_path = new_prompts_cache / "source-trace/receipt.json"

    ref_prefix = json.loads(ref_prefix_path.read_text())
    ref_full = json.loads(ref_full_path.read_text())
    gpu_comp = json.loads(gpu_comp_path.read_text())
    inputs_prefix = json.loads(inputs_prefix_path.read_text())
    inputs_full = json.loads(inputs_full_path.read_text())
    cpu_receipt = json.loads(cpu_receipt_path.read_text())

    toks_prefix = inputs_prefix["sequences"][0]["tokens"]
    toks_full = [s for s in inputs_full["sequences"] if s["id"] == "inventory"][0]["tokens"]
    assert len(toks_prefix) == 113 and len(toks_full) == 2187
    assert toks_prefix == toks_full[:113], "prefix113 must equal exact first 113 tokens of inventory"

    seq_p = ref_prefix["sequences"]["inventory_prefix113"]
    seq_f = ref_full["sequences"]["inventory"]

    pos_idx_p = seq_p["router"]["0"]["positions"].index(112)
    router_diffs_48 = []
    layer24_cpu_prefix = None
    layer24_cpu_full = None
    for l in range(48):
        top_p = seq_p["router"][str(l)]["topExperts"][pos_idx_p]
        margin_p = float(seq_p["router"][str(l)]["marginKthToNext"][112])
        r_name = f"router-{l}.npz"
        r_path = new_prompts_cache / "source-trace" / r_name
        assert sha256_file(r_path) == cpu_receipt["files"][r_name], f"router hash mismatch: {r_name}"
        r_npz = np.load(r_path, allow_pickle=False)
        top_f = r_npz["experts"][112].astype(int).tolist()
        logits_f = r_npz["logits"][112].astype(np.float32)
        top11_f = np.sort(logits_f)[::-1][:11]
        margin_f = float(top11_f[9] - top11_f[10])
        if l == 24:
            layer24_cpu_prefix = {"topExperts": top_p, "margin10To11": margin_p}
            layer24_cpu_full = {"topExperts": top_f, "margin10To11": margin_f}
        if top_p != top_f:
            router_diffs_48.append({"layer": l, "prefix113": top_p, "full2187": top_f})

    p_positions = seq_p["samplePositions"]
    f_positions = seq_f["samplePositions"]
    shared_pos = [p for p in p_positions if p in f_positions]
    max_sgemm_sample_abs = 0.0
    max_sgemm_layer = None
    for l in range(48):
        sp = np.array([seq_p["layers"][str(l)]["samples"][p_positions.index(p)] for p in shared_pos], dtype=np.float32)
        sf = np.array([seq_f["layers"][str(l)]["samples"][f_positions.index(p)] for p in shared_pos], dtype=np.float32)
        m = float(np.max(np.abs(sp - sf)))
        if m > max_sgemm_sample_abs:
            max_sgemm_sample_abs = m
            max_sgemm_layer = l

    return {
        "schema": "doppler.flash-next-causal-prefix-control/v1",
        "scope": "113-token causal prefix (M=113) vs full 2,187-token inventory prompt (M=2187) at token 112 on GPU (Layers 0..24) and independent CPU F32 source (Layers 0..47); diagnostic only, no model qualification or acceptance change",
        "modelQualified": false_val(),
        "acceptanceChanged": false_val(),
        "runtimeChanged": false_val(),
        "auditScriptSha256": script_sha,
        "prefixSequenceId": "inventory_prefix113",
        "fullSequenceId": "inventory",
        "prefixTokenCount": 113,
        "fullTokenCount": 2187,
        "targetTokenIndex": 112,
        "manifestSha256": gpu_comp["manifestSha256"],
        "prefixInputsSha256": sha256_file(inputs_prefix_path),
        "fullInputsSha256": sha256_file(inputs_full_path),
        "prefixExactMatchToFullFirst113": True,
        "gpuControl": {
            "gpuComparisonSha256": sha256_file(gpu_comp_path),
            "prefixObservationSha256": gpu_comp["prefixObservationSha256"],
            "fullObservationSha256": gpu_comp["fullObservationSha256"],
            "layersChecked": [0, 24],
            "layerCount": len(gpu_comp["rows"]),
            "valuesPerLayer": 2560,
            "checkedFloat32Values": sum(r["values"] for r in gpu_comp["rows"]),
            "bitExact": all(r["bitExact"] for r in gpu_comp["rows"]),
            "differingValues": sum(r["differingValues"] for r in gpu_comp["rows"]),
            "firstDifferentLayer": gpu_comp["firstDifferentLayer"],
            "rows": gpu_comp["rows"],
        },
        "cpuSourceControl": {
            "prefixReferenceSha256": sha256_file(ref_prefix_path),
            "fullReferenceSha256": sha256_file(ref_full_path),
            "sourceTraceReceiptSha256": sha256_file(cpu_receipt_path),
            "layersChecked": [0, 47],
            "layerCount": 48,
            "position112OrderedTop10Differences": router_diffs_48,
            "position112All48LayersMatchOrderedTop10": len(router_diffs_48) == 0,
            "layer24Prefix113": layer24_cpu_prefix,
            "layer24Full2187": layer24_cpu_full,
            "sharedSamplePositions": shared_pos,
            "cpuBatchGeometrySgemmMaxSampleAbsDiff": max_sgemm_sample_abs,
            "cpuBatchGeometrySgemmMaxSampleLayer": max_sgemm_layer,
        },
    }


def false_val() -> bool:
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--producer-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--causal-cache", type=Path, required=True)
    parser.add_argument("--new-prompts-cache", type=Path, required=True)
    parser.add_argument("--full-gpu-dir", type=Path, required=True)
    parser.add_argument("--out-causal", type=Path, required=True)
    parser.add_argument("--out-controls", type=Path, required=True)
    args = parser.parse_args()

    assert not args.out_causal.exists(), f"Refusing to overwrite {args.out_causal}"
    assert not args.out_controls.exists(), f"Refusing to overwrite {args.out_controls}"

    t0 = time.time()
    torch.set_num_threads(16)
    assert torch.get_float32_matmul_precision() == "highest"
    script_sha = sha256_file(Path(__file__))

    causal_summary = build_causal_prefix_summary(args.causal_cache, args.new_prompts_cache, script_sha)
    args.out_causal.write_text(json.dumps(causal_summary, indent=2) + "\n")
    print("Wrote causal-prefix-control.json", flush=True)

    ns, cfg, manifest_bytes, read_tensor, handles, pool = build_upstream_namespace(
        args.producer_dir, args.source_dir, args.artifact_dir
    )

    obs_path = args.full_gpu_dir / "observation.json"
    obs = json.loads(obs_path.read_text())
    assert obs["diagnosticCompleted"] is True
    assert obs["gpuErrors"] == []
    assert obs["cleanupFailures"] == []
    assert obs["manifestSha256"] == hashlib.sha256(manifest_bytes).hexdigest()

    gpu_tensors = {}
    for row in obs["tensors"]:
        fpath = Path(row["file"])
        assert sha256_file(fpath) == row["sha256"], f"Tensor SHA-256 mismatch: {fpath.name}"
        gpu_tensors[(row["layer"], row["role"], row["stage"])] = row

    for key in ("indices", "weights"):
        rinfo = obs["moeRouting"][key]
        assert sha256_file(Path(rinfo["file"])) == rinfo["sha256"], f"Routing SHA-256 mismatch: {key}"

    def load_gpu(layer: int, role: str, stage: str) -> np.ndarray:
        row = gpu_tensors[(layer, role, stage)]
        return np.fromfile(row["file"], dtype=np.float32).reshape(row["shape"])

    prev_obs_path = args.causal_cache / "gpu/observation.json"
    prev_obs = json.loads(prev_obs_path.read_text())
    named_vs_prepared_bit_diffs = 0
    for l in range(25):
        prev_row = [
            r for r in prev_obs["tensors"] if r["layer"] == l and r["role"] == "mlp_hc" and r["stage"] == "branchInput"
        ][0]
        assert sha256_file(Path(prev_row["file"])) == prev_row["sha256"]
        prev_idx = prev_row["positions"].index(112)
        prev_arr = np.fromfile(prev_row["file"], dtype=np.uint32).reshape(prev_row["shape"])[prev_idx]
        cur_arr = load_gpu(l, "mlp_hc", "branchInput")[112].view(np.uint32)
        named_vs_prepared_bit_diffs += int(np.sum(prev_arr != cur_arr))
    assert named_vs_prepared_bit_diffs == 0, f"unexpected bit diff: {named_vs_prepared_bit_diffs}"

    inputs_path = args.causal_cache / "inputs.json"
    assert sha256_file(inputs_path) == obs["inputsSha256"]
    tokens = json.loads(inputs_path.read_text())["sequences"][0]["tokens"]
    seq_len = len(tokens)
    assert seq_len == 113
    ids = torch.tensor([tokens], dtype=torch.long)

    embed = read_tensor("model.language_model.embed_tokens.weight")
    rotary = ns["Qwen4ExpTextRotaryEmbedding"](cfg)
    hidden0 = embed[ids[0]].unsqueeze(0).to(torch.float32)
    position_ids = torch.arange(seq_len).view(1, 1, -1).expand(4, 1, -1)[1:]
    pos_emb = rotary(hidden0, position_ids)
    causal_mask = torch.full((seq_len, seq_len), torch.finfo(torch.float32).min, dtype=torch.float32).triu(1)[None, None]
    x_cpu = hidden0.repeat(1, 1, cfg.hc_count)
    del embed

    w_router_l24 = read_tensor("model.language_model.layers.24.mlp.gate.weight").numpy().astype(np.float64)
    w_diff_60_225 = w_router_l24[60] - w_router_l24[225]

    def eval_hc(hc_mod, u_in_np: np.ndarray, branch_out_np: np.ndarray):
        u = torch.from_numpy(u_in_np).unsqueeze(0)
        with torch.no_grad():
            u_norm = hc_mod.hc_norm(u)
            mixed, _, inj_w = hc_mod(u)
            b_out = torch.from_numpy(branch_out_np).unsqueeze(0)
            injected = u + (b_out.unsqueeze(-2) * inj_w.unsqueeze(-1)).flatten(-2)
        return u_norm[0].numpy(), mixed[0].numpy(), injected[0].numpy()

    def eval_gdn_recurrent(gdn_mod, u_in_np: np.ndarray) -> np.ndarray:
        orig_chunk = ns["torch_chunk_gated_delta_rule"]
        ns["torch_chunk_gated_delta_rule"] = ns["torch_recurrent_gated_delta_rule"]
        try:
            with torch.no_grad():
                out = gdn_mod(torch.from_numpy(u_in_np).unsqueeze(0), cache_params=None, attention_mask=None)
            return out[0].numpy()
        finally:
            ns["torch_chunk_gated_delta_rule"] = orig_chunk

    layer_reports = []
    routing_flips_all_tokens = []
    slice_loader_verified_layer0 = False

    with torch.no_grad():
        for l in range(25):
            lt0 = time.time()
            gate_w = read_tensor(f"model.language_model.layers.{l}.mlp.gate.weight")
            u_mlp_gpu_np = load_gpu(l, "mlp_hc", "branchInput")

            layer = ns["Qwen4ExpTextDecoderLayer"](cfg, l).eval()
            prefix = f"model.language_model.layers.{l}."
            state = {}
            for key, value in layer.state_dict().items():
                if "ngram_embedding.weight" in key or key.startswith("mlp.experts."):
                    continue
                tensor = read_tensor(prefix + key)
                assert tuple(tensor.shape) == tuple(value.shape), key
                if value.dtype == torch.int64:
                    assert torch.equal(tensor, value), key
                state[key] = tensor
            layer.load_state_dict(state, strict=False, assign=True)
            layer.to(torch.float32)

            rec = {"layer": l, "layerType": layer.layer_type}

            gpu_attn_in = load_gpu(l, "attn_hc", "input")
            if l == 0:
                cpu_entry = x_cpu[0].numpy()
                rec["entry"] = {
                    "op": "embed_tokens_repeat4",
                    "totalVsCpu": pair_metrics(gpu_attn_in, cpu_entry),
                    "localSameInput": pair_metrics(gpu_attn_in, cpu_entry),
                }
            elif layer.ple is not None:
                prev_gpu_out = load_gpu(l - 1, "mlp_hc", "injected")
                cpu_ple_on_cpu = (x_cpu + layer.ple(x_cpu, ids, None, conv_mask=None))[0].numpy()
                u_prev_gpu_t = torch.from_numpy(prev_gpu_out).unsqueeze(0)
                cpu_ple_on_gpu = (u_prev_gpu_t + layer.ple(u_prev_gpu_t, ids, None, conv_mask=None))[0].numpy()
                x_cpu = torch.from_numpy(cpu_ple_on_cpu).unsqueeze(0)
                rec["entry"] = {
                    "op": "ple_layer",
                    "localSameInput": pair_metrics(gpu_attn_in, cpu_ple_on_gpu),
                    "propagatedInput": pair_metrics(cpu_ple_on_gpu, cpu_ple_on_cpu),
                    "totalVsCpu": pair_metrics(gpu_attn_in, cpu_ple_on_cpu),
                }
            else:
                prev_gpu_out = load_gpu(l - 1, "mlp_hc", "injected")
                rec["entry"] = {
                    "op": "identity_from_prev_injected",
                    "localSameInput": pair_metrics(gpu_attn_in, prev_gpu_out),
                    "totalVsCpu": pair_metrics(gpu_attn_in, x_cpu[0].numpy()),
                }
                assert rec["entry"]["localSameInput"]["all113"]["bitExact"], f"L{l} entry not bit-exact!"

            gpu_attn_norm = load_gpu(l, "attn_hc", "norm")
            gpu_attn_bin = load_gpu(l, "attn_hc", "branchInput")
            gpu_attn_bout = load_gpu(l, "attn_hc", "branchOutput")
            gpu_attn_inj = load_gpu(l, "attn_hc", "injected")

            cpu_attn_norm_t = layer.attn_hyper_connection.hc_norm(x_cpu)
            cpu_attn_bin_t, cpu_hyper_in, cpu_inj_w = layer.attn_hyper_connection(x_cpu)
            if layer.layer_type == "linear_attention":
                cpu_attn_bout_t = layer.linear_attn(cpu_attn_bin_t, cache_params=None, attention_mask=None)
            else:
                cpu_attn_bout_t, _ = layer.self_attn(
                    cpu_attn_bin_t, pos_emb, attention_mask=causal_mask, past_key_values=None
                )
            cpu_attn_inj_t = cpu_hyper_in + (cpu_attn_bout_t.unsqueeze(-2) * cpu_inj_w.unsqueeze(-1)).flatten(-2)

            same_attn_norm, same_attn_bin, same_attn_inj = eval_hc(
                layer.attn_hyper_connection, gpu_attn_in, gpu_attn_bout
            )
            rec["attnHc"] = {
                "norm": {
                    "localSameInput": pair_metrics(gpu_attn_norm, same_attn_norm),
                    "propagatedInput": pair_metrics(same_attn_norm, cpu_attn_norm_t[0].numpy()),
                    "totalVsCpu": pair_metrics(gpu_attn_norm, cpu_attn_norm_t[0].numpy()),
                },
                "branchInput": {
                    "localSameInput": pair_metrics(gpu_attn_bin, same_attn_bin),
                    "propagatedInput": pair_metrics(same_attn_bin, cpu_attn_bin_t[0].numpy()),
                    "totalVsCpu": pair_metrics(gpu_attn_bin, cpu_attn_bin_t[0].numpy()),
                },
                "injected": {
                    "localSameInput": pair_metrics(gpu_attn_inj, same_attn_inj),
                    "totalVsCpu": pair_metrics(gpu_attn_inj, cpu_attn_inj_t[0].numpy()),
                },
            }

            if layer.layer_type == "linear_attention":
                same_gdn_chunk = layer.linear_attn(
                    torch.from_numpy(gpu_attn_bin).unsqueeze(0), cache_params=None, attention_mask=None
                )[0].numpy()
                same_gdn_recurrent = eval_gdn_recurrent(layer.linear_attn, gpu_attn_bin)
                rec["sequenceMixer"] = {
                    "op": "gated_delta_net",
                    "localSameInputVsChunk": pair_metrics(gpu_attn_bout, same_gdn_chunk),
                    "localSameInputVsRecurrent": pair_metrics(gpu_attn_bout, same_gdn_recurrent),
                    "cpuChunkVsCpuRecurrentOnSameGpuInput": pair_metrics(same_gdn_chunk, same_gdn_recurrent),
                    "propagatedInputChunk": pair_metrics(same_gdn_chunk, cpu_attn_bout_t[0].numpy()),
                    "totalVsCpuChunk": pair_metrics(gpu_attn_bout, cpu_attn_bout_t[0].numpy()),
                }
            else:
                same_attn_out_t, _ = layer.self_attn(
                    torch.from_numpy(gpu_attn_bin).unsqueeze(0),
                    pos_emb,
                    attention_mask=causal_mask,
                    past_key_values=None,
                )
                same_attn_out = same_attn_out_t[0].numpy()
                rec["sequenceMixer"] = {
                    "op": "full_indexed_attention",
                    "qsaRegimeAt113Tokens": "dense_causal_prefix_under_2048_token_budget",
                    "localSameInput": pair_metrics(gpu_attn_bout, same_attn_out),
                    "propagatedInput": pair_metrics(same_attn_out, cpu_attn_bout_t[0].numpy()),
                    "totalVsCpu": pair_metrics(gpu_attn_bout, cpu_attn_bout_t[0].numpy()),
                }

            gpu_mlp_in = load_gpu(l, "mlp_hc", "input")
            assert np.array_equal(gpu_mlp_in.view(np.uint32), gpu_attn_inj.view(np.uint32))
            gpu_mlp_norm = load_gpu(l, "mlp_hc", "norm")
            gpu_mlp_bin = u_mlp_gpu_np
            gpu_mlp_bout = load_gpu(l, "mlp_hc", "branchOutput")
            gpu_mlp_inj = load_gpu(l, "mlp_hc", "injected")

            cpu_mlp_norm_t = layer.mlp_hyper_connection.hc_norm(cpu_attn_inj_t)
            cpu_mlp_bin_t, cpu_mlp_hyper_in, cpu_mlp_inj_w = layer.mlp_hyper_connection(cpu_attn_inj_t)

            same_mlp_norm, same_mlp_bin, same_mlp_inj = eval_hc(
                layer.mlp_hyper_connection, gpu_mlp_in, gpu_mlp_bout
            )
            rec["mlpHc"] = {
                "norm": {
                    "localSameInput": pair_metrics(gpu_mlp_norm, same_mlp_norm),
                    "propagatedInput": pair_metrics(same_mlp_norm, cpu_mlp_norm_t[0].numpy()),
                    "totalVsCpu": pair_metrics(gpu_mlp_norm, cpu_mlp_norm_t[0].numpy()),
                },
                "branchInput": {
                    "localSameInput": pair_metrics(gpu_mlp_bin, same_mlp_bin),
                    "propagatedInput": pair_metrics(same_mlp_bin, cpu_mlp_bin_t[0].numpy()),
                    "totalVsCpu": pair_metrics(gpu_mlp_bin, cpu_mlp_bin_t[0].numpy()),
                },
                "injected": {
                    "localSameInput": pair_metrics(gpu_mlp_inj, same_mlp_inj),
                },
            }

            delta_router_in_pos112 = gpu_mlp_bin[112].astype(np.float64) - cpu_mlp_bin_t[0, 112].numpy().astype(np.float64)
            rec["pos112RouterInputProjectionOnL24Margin60Minus225"] = float(np.dot(delta_router_in_pos112, w_diff_60_225))

            cpu_router_logits, _, cpu_router_idx = layer.mlp.gate(cpu_mlp_bin_t[0])
            same_router_logits, _, same_router_idx = layer.mlp.gate(torch.from_numpy(gpu_mlp_bin))

            cpu_idx_np = cpu_router_idx.numpy()
            same_idx_np = same_router_idx.numpy()
            layer_token_flips = []
            for pos in range(seq_len):
                s_cpu = set(cpu_idx_np[pos].tolist())
                s_same = set(same_idx_np[pos].tolist())
                order_match = cpu_idx_np[pos].tolist() == same_idx_np[pos].tolist()
                if s_cpu != s_same or not order_match:
                    top11_cpu = torch.topk(cpu_router_logits[pos], 11).values
                    margin_cpu = float(top11_cpu[9] - top11_cpu[10])
                    flip_entry = {
                        "layer": l,
                        "position": pos,
                        "expertSetChanged": s_cpu != s_same,
                        "rankOrderChanged": not order_match,
                        "cpuTop10": cpu_idx_np[pos].tolist(),
                        "cpuRouterOnGpuInputTop10": same_idx_np[pos].tolist(),
                        "cpuMargin10To11": margin_cpu,
                    }
                    layer_token_flips.append(flip_entry)
                    routing_flips_all_tokens.append(flip_entry)

            active_experts = sorted(set(cpu_idx_np.flatten().tolist()) | set(same_idx_np.flatten().tolist()))
            if l == 24:
                gpu_l24_idx = np.fromfile(obs["moeRouting"]["indices"]["file"], dtype=np.uint32).reshape(seq_len, 10)
                active_experts = sorted(set(active_experts) | set(gpu_l24_idx.flatten().tolist()))

            exp_gate_up = read_tensor(f"{prefix}mlp.experts.gate_up_proj", expert_subset=active_experts)
            exp_down = read_tensor(f"{prefix}mlp.experts.down_proj", expert_subset=active_experts)
            layer.mlp.experts.load_state_dict({"gate_up_proj": exp_gate_up, "down_proj": exp_down}, assign=True)

            cpu_moe_out_t = layer.mlp(cpu_mlp_bin_t)
            same_moe_out_t = layer.mlp(torch.from_numpy(gpu_mlp_bin).unsqueeze(0))

            if l == 0:
                full_gate_up = read_tensor(f"{prefix}mlp.experts.gate_up_proj", expert_subset=None)
                full_down = read_tensor(f"{prefix}mlp.experts.down_proj", expert_subset=None)
                layer.mlp.experts.load_state_dict({"gate_up_proj": full_gate_up, "down_proj": full_down}, assign=True)
                full_same_moe = layer.mlp(torch.from_numpy(gpu_mlp_bin).unsqueeze(0))
                assert torch.equal(same_moe_out_t, full_same_moe), "Active-expert slice load differed from full load!"
                slice_loader_verified_layer0 = True
                del full_gate_up, full_down

            x_cpu = cpu_mlp_hyper_in + (cpu_moe_out_t.unsqueeze(-2) * cpu_mlp_inj_w.unsqueeze(-1)).flatten(-2)
            rec["mlpHc"]["injected"]["totalVsCpu"] = pair_metrics(gpu_mlp_inj, x_cpu[0].numpy())

            rec["moe"] = {
                "activeExpertsLoaded": len(active_experts),
                "routingFlipsAcrossAll113Tokens": layer_token_flips,
                "pos112SetMatchesCpu": set(cpu_idx_np[112].tolist()) == set(same_idx_np[112].tolist()),
                "pos112OrderMatchesCpu": cpu_idx_np[112].tolist() == same_idx_np[112].tolist(),
                "localSameInput": pair_metrics(gpu_mlp_bout, same_moe_out_t[0].numpy()),
                "propagatedInput": pair_metrics(same_moe_out_t[0].numpy(), cpu_moe_out_t[0].numpy()),
                "totalVsCpu": pair_metrics(gpu_mlp_bout, cpu_moe_out_t[0].numpy()),
            }

            if l == 24:
                x_cpu_112 = cpu_mlp_bin_t[0, 112].numpy()
                x_gpu_112 = gpu_mlp_bin[112]
                w_l24_f32 = gate_w.numpy()
                gpu_l24_idx_112 = np.fromfile(obs["moeRouting"]["indices"]["file"], dtype=np.uint32).reshape(seq_len, 10)[112].tolist()
                gpu_l24_w_112 = np.fromfile(obs["moeRouting"]["weights"]["file"], dtype=np.float32).reshape(seq_len, 10)[112].tolist()

                def exact_dot_rational(x_vec: np.ndarray, w_vec: np.ndarray) -> Fraction:
                    total = Fraction(0)
                    for a, b in zip(x_vec.tolist(), w_vec.tolist()):
                        total += Fraction(a) * Fraction(b)
                    return total

                s_exact_cpu_60 = exact_dot_rational(x_cpu_112, w_l24_f32[60])
                s_exact_cpu_225 = exact_dot_rational(x_cpu_112, w_l24_f32[225])
                s_exact_gpu_60 = exact_dot_rational(x_gpu_112, w_l24_f32[60])
                s_exact_gpu_225 = exact_dot_rational(x_gpu_112, w_l24_f32[225])

                margin_exact_cpu = s_exact_cpu_60 - s_exact_cpu_225
                margin_exact_gpu = s_exact_gpu_60 - s_exact_gpu_225
                upstream_exact_shift = margin_exact_gpu - margin_exact_cpu

                pt_cpu_60 = float(cpu_router_logits[112, 60].item())
                pt_cpu_225 = float(cpu_router_logits[112, 225].item())
                pt_gpu_60 = float(same_router_logits[112, 60].item())
                pt_gpu_225 = float(same_router_logits[112, 225].item())

                wgsl_w64_on_cpu_60 = simulate_subgroup_dot_f32(x_cpu_112, w_l24_f32[60], 64)
                wgsl_w64_on_cpu_225 = simulate_subgroup_dot_f32(x_cpu_112, w_l24_f32[225], 64)
                wgsl_w64_on_gpu_60 = simulate_subgroup_dot_f32(x_gpu_112, w_l24_f32[60], 64)
                wgsl_w64_on_gpu_225 = simulate_subgroup_dot_f32(x_gpu_112, w_l24_f32[225], 64)

                wgsl_w32_on_cpu_60 = simulate_subgroup_dot_f32(x_cpu_112, w_l24_f32[60], 32)
                wgsl_w32_on_cpu_225 = simulate_subgroup_dot_f32(x_cpu_112, w_l24_f32[225], 32)
                wgsl_w32_on_gpu_60 = simulate_subgroup_dot_f32(x_gpu_112, w_l24_f32[60], 32)
                wgsl_w32_on_gpu_225 = simulate_subgroup_dot_f32(x_gpu_112, w_l24_f32[225], 32)

                u112_t = torch.from_numpy(x_gpu_112).unsqueeze(0)
                sel_60 = torch.tensor([[60]], dtype=torch.long)
                sel_225 = torch.tensor([[225]], dtype=torch.long)
                one_w = torch.tensor([[gpu_l24_w_112[9]]], dtype=torch.float32)
                out_e60 = layer.mlp.experts(u112_t, sel_60, one_w)[0].numpy()
                out_e225 = layer.mlp.experts(u112_t, sel_225, one_w)[0].numpy()

                rec["layer24Pos112RouterAttribution"] = {
                    "cpuSelectedTop10": cpu_idx_np[112].tolist(),
                    "cpuRouterOnGpuInputTop10": same_idx_np[112].tolist(),
                    "gpuActualSelectedTop10": gpu_l24_idx_112,
                    "gpuActualSelectedWeights": gpu_l24_w_112,
                    "exactRationalAsFloat64": {
                        "cpuInput": {
                            "expert60": float(s_exact_cpu_60),
                            "expert225": float(s_exact_cpu_225),
                            "margin60Minus225": float(margin_exact_cpu),
                            "winner": 60 if margin_exact_cpu > 0 else 225,
                        },
                        "gpuInput": {
                            "expert60": float(s_exact_gpu_60),
                            "expert225": float(s_exact_gpu_225),
                            "margin60Minus225": float(margin_exact_gpu),
                            "winner": 60 if margin_exact_gpu > 0 else 225,
                        },
                        "upstreamInputShiftContribution": float(upstream_exact_shift),
                    },
                    "pytorchFLinearF32": {
                        "cpuInput": {
                            "expert60": pt_cpu_60,
                            "expert225": pt_cpu_225,
                            "margin60Minus225": pt_cpu_60 - pt_cpu_225,
                            "roundingVsExact": (pt_cpu_60 - pt_cpu_225) - float(margin_exact_cpu),
                            "winner": 60 if pt_cpu_60 > pt_cpu_225 else 225,
                        },
                        "gpuInput": {
                            "expert60": pt_gpu_60,
                            "expert225": pt_gpu_225,
                            "margin60Minus225": pt_gpu_60 - pt_gpu_225,
                            "roundingVsExact": (pt_gpu_60 - pt_gpu_225) - float(margin_exact_gpu),
                            "winner": 60 if pt_gpu_60 > pt_gpu_225 else 225,
                        },
                    },
                    "wgslSubgroupMatmulSimulationF32": {
                        "wave64OnCpuInput": {
                            "expert60": wgsl_w64_on_cpu_60["treeSubgroupAdd"],
                            "expert225": wgsl_w64_on_cpu_225["treeSubgroupAdd"],
                            "margin60Minus225": wgsl_w64_on_cpu_60["treeSubgroupAdd"] - wgsl_w64_on_cpu_225["treeSubgroupAdd"],
                            "roundingVsExact": (wgsl_w64_on_cpu_60["treeSubgroupAdd"] - wgsl_w64_on_cpu_225["treeSubgroupAdd"]) - float(margin_exact_cpu),
                            "winner": 60 if wgsl_w64_on_cpu_60["treeSubgroupAdd"] > wgsl_w64_on_cpu_225["treeSubgroupAdd"] else 225,
                        },
                        "wave64OnGpuInput": {
                            "expert60": wgsl_w64_on_gpu_60["treeSubgroupAdd"],
                            "expert225": wgsl_w64_on_gpu_225["treeSubgroupAdd"],
                            "margin60Minus225": wgsl_w64_on_gpu_60["treeSubgroupAdd"] - wgsl_w64_on_gpu_225["treeSubgroupAdd"],
                            "roundingVsExact": (wgsl_w64_on_gpu_60["treeSubgroupAdd"] - wgsl_w64_on_gpu_225["treeSubgroupAdd"]) - float(margin_exact_gpu),
                            "winner": 60 if wgsl_w64_on_gpu_60["treeSubgroupAdd"] > wgsl_w64_on_gpu_225["treeSubgroupAdd"] else 225,
                        },
                        "wave32OnCpuInput": {
                            "expert60": wgsl_w32_on_cpu_60["treeSubgroupAdd"],
                            "expert225": wgsl_w32_on_cpu_225["treeSubgroupAdd"],
                            "margin60Minus225": wgsl_w32_on_cpu_60["treeSubgroupAdd"] - wgsl_w32_on_cpu_225["treeSubgroupAdd"],
                            "roundingVsExact": (wgsl_w32_on_cpu_60["treeSubgroupAdd"] - wgsl_w32_on_cpu_225["treeSubgroupAdd"]) - float(margin_exact_cpu),
                            "winner": 60 if wgsl_w32_on_cpu_60["treeSubgroupAdd"] > wgsl_w32_on_cpu_225["treeSubgroupAdd"] else 225,
                        },
                        "wave32OnGpuInput": {
                            "expert60": wgsl_w32_on_gpu_60["treeSubgroupAdd"],
                            "expert225": wgsl_w32_on_gpu_225["treeSubgroupAdd"],
                            "margin60Minus225": wgsl_w32_on_gpu_60["treeSubgroupAdd"] - wgsl_w32_on_gpu_225["treeSubgroupAdd"],
                            "roundingVsExact": (wgsl_w32_on_gpu_60["treeSubgroupAdd"] - wgsl_w32_on_gpu_225["treeSubgroupAdd"]) - float(margin_exact_gpu),
                            "winner": 60 if wgsl_w32_on_gpu_60["treeSubgroupAdd"] > wgsl_w32_on_gpu_225["treeSubgroupAdd"] else 225,
                        },
                    },
                    "rank10Expert225Vs60BranchShock": diff_metrics(out_e225, out_e60),
                }

            elapsed = round(time.time() - lt0, 2)
            layer_reports.append(rec)
            print(json.dumps({"layer": l, "sec": elapsed, "tokenFlips113": len(layer_token_flips)}), flush=True)
            del layer

    pool.shutdown(wait=True)
    for h in handles.values():
        h.close()

    max_hc_norm_local_abs = max(
        max(r["attnHc"]["norm"]["localSameInput"]["all113"]["maxAbs"], r["mlpHc"]["norm"]["localSameInput"]["all113"]["maxAbs"])
        for r in layer_reports
    )
    max_hc_mix_local_abs = max(
        max(
            r["attnHc"]["branchInput"]["localSameInput"]["all113"]["maxAbs"],
            r["mlpHc"]["branchInput"]["localSameInput"]["all113"]["maxAbs"],
        )
        for r in layer_reports
    )
    max_hc_inj_local_abs = max(
        max(
            r["attnHc"]["injected"]["localSameInput"]["all113"]["maxAbs"],
            r["mlpHc"]["injected"]["localSameInput"]["all113"]["maxAbs"],
        )
        for r in layer_reports
    )
    gdn_layers = [r for r in layer_reports if r["layerType"] == "linear_attention"]
    attn_layers = [r for r in layer_reports if r["layerType"] == "full_attention"]
    max_gdn_local_vs_recurrent_rel = max(
        r["sequenceMixer"]["localSameInputVsRecurrent"]["pos112"]["relL2"] for r in gdn_layers
    )
    max_gdn_local_vs_chunk_rel = max(
        r["sequenceMixer"]["localSameInputVsChunk"]["pos112"]["relL2"] for r in gdn_layers
    )
    max_attn_local_rel = max(r["sequenceMixer"]["localSameInput"]["pos112"]["relL2"] for r in attn_layers)
    max_moe_local_before_l24_rel = max(
        r["moe"]["localSameInput"]["pos112"]["relL2"] for r in layer_reports if r["layer"] < 24
    )

    out_report = {
        "schema": "doppler.flash-next-same-input-operator-controls/v1",
        "scope": "Full-prefix (M=113) CPU same-input operator controls across Layers 0..24 (embedding, PLE, 50 hyper-connections, 19 GDN layers chunk+recurrent, 6 full-attention layers, 25 MoE+shared_expert blocks, and exact rational/WGSL Layer-24 router margin decomposition); diagnostic only, inventory full-logit failure remains unresolved under unchanged quantized-v1",
        "modelQualified": False,
        "acceptanceChanged": False,
        "runtimeChanged": False,
        "sequenceId": "inventory_prefix113",
        "tokenCount": 113,
        "targetTokenIndex": 112,
        "layersAudited": [0, 24],
        "layerCount": 25,
        "runtimeProfile": "profiles/flash-next-routed-f32kv-qualification",
        "namedProfileMatchesPreparedConfigPrefix113BitForBit": named_vs_prepared_bit_diffs == 0,
        "activeExpertSliceLoaderVerifiedBitExactOnLayer0": slice_loader_verified_layer0,
        "artifactManifestSha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "inputsSha256": sha256_file(inputs_path),
        "fullPrefixGpuObservationSha256": sha256_file(obs_path),
        "producerScriptSha256": sha256_file(args.producer_dir / "prepare-learned-forward-reference.py"),
        "auditScriptSha256": script_sha,
        "torchVersion": torch.__version__,
        "torchThreads": torch.get_num_threads(),
        "summary": {
            "embedLayer0BitExact": layer_reports[0]["entry"]["localSameInput"]["all113"]["bitExact"],
            "pleLayer1LocalSameInputPos112": layer_reports[1]["entry"]["localSameInput"]["pos112"],
            "maxHyperConnectionNormLocalMaxAbsAll113": max_hc_norm_local_abs,
            "maxHyperConnectionMixLocalMaxAbsAll113": max_hc_mix_local_abs,
            "maxHyperConnectionInjectLocalMaxAbsAll113": max_hc_inj_local_abs,
            "maxGdnLocalSameInputRelL2VsRecurrentPos112": max_gdn_local_vs_recurrent_rel,
            "maxGdnLocalSameInputRelL2VsChunkPos112": max_gdn_local_vs_chunk_rel,
            "maxFullAttentionLocalSameInputRelL2Pos112": max_attn_local_rel,
            "layer23FullAttentionDecompositionPos112": {
                "localSameInput": layer_reports[23]["sequenceMixer"]["localSameInput"]["pos112"],
                "propagatedInput": layer_reports[23]["sequenceMixer"]["propagatedInput"]["pos112"],
                "totalVsCpu": layer_reports[23]["sequenceMixer"]["totalVsCpu"]["pos112"],
            },
            "maxMoeLocalSameInputRelL2BeforeLayer24Pos112": max_moe_local_before_l24_rel,
            "routingFlipsAcrossAll113TokensLayers0To24": routing_flips_all_tokens,
            "layer24Pos112RouterAttribution": layer_reports[24]["layer24Pos112RouterAttribution"],
        },
        "layers": layer_reports,
    }
    args.out_controls.write_text(json.dumps(out_report, indent=2) + "\n")
    print("Wrote same-input-operator-controls.json", flush=True)


if __name__ == "__main__":
    main()
