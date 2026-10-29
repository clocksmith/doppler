"""Four-prompt greedy CPU teacher with explicit JSON-declared physical KV dtype."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil
import time
import traceback

from binding import digest, read_inputs, verify_bytes
from native_model import load_native_model, physical_cache, inspect_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", required=True, type=Path)
    parser.add_argument("--prompts-contract", required=True, type=Path)
    parser.add_argument("--engine-contract", required=True, type=Path)
    parser.add_argument("--runtime-manifest", required=True, type=Path)
    parser.add_argument("--prompt-identity", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--teacher-policy", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    policy_path = Path(__file__).with_name("contract.json")
    policy = json.loads(policy_path.read_text())
    lane = json.loads(args.teacher_policy.read_text()) if args.teacher_policy else policy["teacher"]
    inputs = read_inputs(args.canonical, policy)
    verify_bytes(inputs, policy["hashChunkBytes"])
    prompts_contract = json.loads(args.prompts_contract.read_text())
    engine_contract = json.loads(args.engine_contract.read_text())
    dtype_names = {"f16": "float16", "f32": "float32"}
    if (engine_contract["shared"]["kvDtype"] not in dtype_names or
            dtype_names[engine_contract["shared"]["kvDtype"]] != lane["kvPhysicalDtype"] or
            lane["attentionMathDtype"] != "float32"):
        raise ValueError("teacher physical-cache dtype must exactly match the explicit parent lane JSON")
    runtime = json.loads(args.runtime_manifest.read_text())
    prompt_identity = json.loads(args.prompt_identity.read_text())
    if runtime["eos_token_id"] != lane["eosTokenIds"]:
        raise ValueError("parent RDRR stop-rule IDs differ from frozen teacher policy")
    if any(source["shared"]["maxNewTokens"] != lane["maxNewTokens"]
           for source in (prompts_contract, engine_contract)):
        raise ValueError("parent and teacher generation budgets differ")
    versions = {name: importlib.metadata.version(name) for name in policy["frameworks"]}
    if versions != policy["frameworks"] or importlib.metadata.version("tokenizers") != lane["tokenizersVersion"]:
        raise ValueError("teacher framework identity differs")
    import torch
    from tokenizers import Tokenizer
    from transformers.models.qwen3_5 import modeling_qwen3_5 as hf
    import transformers.cache_utils as cache_module

    tokenizer_path = args.canonical / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    prompt_format = engine_contract["promptFormat"]
    prompts = []
    for prompt in prompts_contract["shared"]["prompts"]:
        formatted = prompt_format["before"] + prompt["text"] + prompt_format["after"]
        ids = tokenizer.encode(formatted, add_special_tokens=False).ids
        if len(ids) + lane["maxNewTokens"] > engine_contract["shared"]["maxSeqLen"]:
            raise ValueError("prompt exceeds declared context budget: " + prompt["id"])
        if prompt["id"] == "code" and ids != prompt_identity["promptIds"]:
            raise ValueError("raw canonical tokenizer differs from verified shared code prompt IDs")
        prompts.append({**prompt, "promptIds": ids})
    if len(prompts) != 4 or [p["id"] for p in prompts] != ["code", "explanation", "arithmetic", "long-context"]:
        raise ValueError("four frozen parent prompts differ")
    torch.set_num_threads(policy["threads"])
    torch.set_num_interop_threads(policy["threads"])
    model, config, mappings = load_native_model(inputs, policy)
    source_files = {"teacher": Path(__file__), "modelBoundary": Path(__file__).with_name("native_model.py"),
                    "binding": Path(__file__).with_name("binding.py"), "policy": policy_path,
                    "nativeHF": Path(hf.__file__), "nativeCache": Path(cache_module.__file__)}
    if args.teacher_policy:
        source_files["teacherLanePolicy"] = args.teacher_policy
    physical_dtype = {"float16": torch.float16, "float32": torch.float32}[lane["kvPhysicalDtype"]]
    receipt = {"schema": "doppler.native-hf-canonical-teacher.v1", "completed": False,
               "scope": "Independent CPU native-HF numerical teacher; no GPU or performance claim",
               "inputs": inputs["inputs"], "byteVerification": "passed", "parameterBindings": mappings,
               "frameworks": {**versions, "tokenizers": lane["tokenizersVersion"]},
               "sources": {key: {"path": str(value), "sha256": digest(value)} for key, value in source_files.items()},
               "contracts": {key: {"path": str(value), "sha256": digest(value)} for key, value in {
                    "prompts": args.prompts_contract, "engine": args.engine_contract,
                    "eosRuntimeManifest": args.runtime_manifest, "verifiedCodePrompt": args.prompt_identity}.items()},
               "tokenizer": {"path": str(tokenizer_path), "sha256": digest(tokenizer_path)},
               "contract": {**lane, "threads": policy["threads"], "interOpThreads": policy["threads"],
                            "recurrentState": "float32", "convState": "float32", "head": "separate canonical matrix",
                            "sampling": "raw-logit argmax, no repetition penalties or speculation",
                            "snapshotIndexing": "step0 is full-prompt prefill logits predicting generated token0"},
               "prompts": []}
    snapshot_dir = args.out / "sources"
    snapshot_dir.mkdir()
    for name, source in source_files.items():
        shutil.copyfile(source, snapshot_dir / (name + source.suffix))
    start = time.perf_counter()
    try:
        with torch.inference_mode():
            for prompt in prompts:
                entry = {**prompt, "generatedTokenIds": [], "snapshots": [], "cacheChecks": [],
                         "endedByEOS": False, "completed": False}
                receipt["prompts"].append(entry)
                cache = physical_cache(config, physical_dtype)
                next_input = torch.tensor([prompt["promptIds"]], dtype=torch.int64)
                for step in range(lane["maxNewTokens"]):
                    before_events = len(cache.events)
                    result = model(input_ids=next_input, past_key_values=cache, use_cache=True, logits_to_keep=1)
                    linear = inspect_cache(cache, config)
                    events = cache.events[before_events:]
                    if len(events) != 6 or any(e["storedShape"][2] != len(prompt["promptIds"]) + step for e in events):
                        raise ValueError("teacher physical KV length/update count differs at step " + str(step))
                    entry["cacheChecks"].append({"step": step, "fullAttention": events, "linearStates": linear})
                    logits = result.logits[0, -1].float()
                    if logits.shape != (config.vocab_size,) or not torch.isfinite(logits).all():
                        raise ValueError("teacher logits shape/finiteness failure")
                    token = int(logits.argmax())
                    if step in lane["snapshotSteps"]:
                        filename = f"{prompt['id']}-logits-{step}.f32"
                        (args.out / filename).write_bytes(logits.numpy().astype("<f4", copy=False).tobytes())
                        top_values, top_ids = logits.topk(5)
                        entry["snapshots"].append({"step": step, "file": filename, "sha256": digest(args.out / filename),
                            "dtype": "float32", "shape": [config.vocab_size], "byteOrder": "little-endian",
                            "argmax": token, "top5TokenIds": top_ids.tolist(), "top5Logits": top_values.tolist()})
                        print(json.dumps({"prompt": prompt["id"], "step": step, "token": token,
                                          "elapsedSeconds": time.perf_counter() - start}), flush=True)
                    entry["generatedTokenIds"].append(token)
                    if token in lane["eosTokenIds"]:
                        entry["endedByEOS"] = True
                        break
                    next_input = torch.tensor([[token]], dtype=torch.int64)
                entry["generatedTokens"] = len(entry["generatedTokenIds"])
                entry["text"] = tokenizer.decode(entry["generatedTokenIds"], skip_special_tokens=False)
                entry["missingSnapshotSteps"] = [s for s in lane["snapshotSteps"]
                                                  if s not in [v["step"] for v in entry["snapshots"]]]
                entry["completedBudget"] = entry["generatedTokens"] == lane["maxNewTokens"]
                entry["completed"] = True
                (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
                print(json.dumps({"prompt": prompt["id"], "generatedTokens": entry["generatedTokens"],
                                  "endedByEOS": entry["endedByEOS"], "text": entry["text"]}), flush=True)
        receipt["completed"] = True
    except Exception as error:
        receipt["error"] = {"exception": type(error).__name__, "message": str(error)}
        (args.out / "failure.log").write_text(traceback.format_exc())
    receipt["elapsedSeconds"] = time.perf_counter() - start
    (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    if not receipt["completed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
