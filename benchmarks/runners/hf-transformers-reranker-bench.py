#!/usr/bin/env python3
import argparse
import json
import math
import os
import platform
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEMANTIC_FIXTURE = REPO_ROOT / "src" / "inference" / "fixtures" / "rerank-semantic-fixtures.json"
DEFAULT_QUERY = "What API exposes GPU hardware for browser compute workloads?"
DEFAULT_DOCUMENTS = [
    "WebGPU provides low-level access to graphics processors for compute workloads.",
    "WebSocket provides full-duplex communication over TCP.",
    "CSS grid arranges page layout rows and columns.",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark a Hugging Face Transformers causal-LM reranker with Qwen yes/no-logit scoring."
    )
    parser.add_argument("--model", required=True, help="HF model ID or local model path.")
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--document", action="append", default=None)
    parser.add_argument("--documents-json", default=None)
    parser.add_argument("--rerank-config", required=True, help="Rerank scoring config JSON.")
    parser.add_argument("--semantic-fixture", default=str(DEFAULT_SEMANTIC_FIXTURE))
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def fail(message):
    print(f"[hf-reranker-bench] {message}", file=sys.stderr)
    raise SystemExit(1)


def load_json_text(raw, label):
    try:
        return json.loads(raw)
    except Exception as exc:
        fail(f"{label} must be valid JSON: {exc}")


def normalize_documents(args):
    if args.documents_json is not None:
        value = load_json_text(args.documents_json, "--documents-json")
        if not isinstance(value, list):
            fail("--documents-json must be a JSON array")
        documents = [str(item).strip() for item in value if str(item).strip()]
    elif args.document:
        documents = [str(item).strip() for item in args.document if str(item).strip()]
    else:
        documents = list(DEFAULT_DOCUMENTS)
    if not documents:
        fail("at least one rerank document is required")
    return documents


def require_text(config, key, preserve=False):
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        fail(f"rerank config requires non-empty {key}")
    return value if preserve else value.strip()


def require_token_id(config, key):
    value = config.get(key)
    if not isinstance(value, int) or value < 0:
        fail(f"rerank config requires non-negative integer {key}")
    return value


def normalize_scoring_config(raw):
    config = load_json_text(raw, "--rerank-config")
    if not isinstance(config, dict):
        fail("--rerank-config must be a JSON object")
    score_policy = config.get("score", "true_logit")
    if score_policy not in ("true_logit", "logit_diff"):
        fail('rerank config score must be "true_logit" or "logit_diff"')
    probability_policy = config.get("probability", "sigmoid")
    if probability_policy != "sigmoid":
        fail('rerank config probability must be "sigmoid"')
    return {
        "format": require_text(config, "format"),
        "instruction": require_text(config, "instruction", preserve=True),
        "inputTemplate": require_text(config, "inputTemplate", preserve=True),
        "prefix": require_text(config, "prefix", preserve=True),
        "suffix": require_text(config, "suffix", preserve=True),
        "trueToken": require_text(config, "trueToken"),
        "trueTokenId": require_token_id(config, "trueTokenId"),
        "falseToken": require_text(config, "falseToken"),
        "falseTokenId": require_token_id(config, "falseTokenId"),
        "score": score_policy,
        "probability": probability_policy,
    }


def format_prompt(query, document, config):
    body = (
        config["inputTemplate"]
        .replace("{instruction}", config["instruction"])
        .replace("{query}", query)
        .replace("{document}", document)
    )
    return f"{config['prefix']}{body}{config['suffix']}"


def sigmoid(value):
    if value >= 0:
        z = math.exp(-value)
        return 1 / (1 + z)
    z = math.exp(value)
    return z / (1 + z)


def percentile(values, pct):
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def round_float(value, digits=6):
    if value is None or not math.isfinite(value):
        return None
    return round(float(value), digits)


def resolve_device(requested, torch):
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        fail("--device cuda requested but torch.cuda.is_available() is false")
    return requested


def resolve_dtype(requested, device, torch):
    if requested == "auto":
        return torch.float32
    if requested == "float32":
        return torch.float32
    if requested == "float16":
        if device == "cpu":
            fail("--dtype float16 is not supported for this CPU baseline")
        return torch.float16
    if requested == "bfloat16":
        return torch.bfloat16
    fail(f"unsupported dtype {requested}")


def synchronize(device, torch):
    if device == "cuda":
        torch.cuda.synchronize()


def chunks(values, size):
    for index in range(0, len(values), size):
        yield index, values[index:index + size]


def rank_scores(scores):
    ranked = sorted(scores, key=lambda item: (-item["score"], item["index"]))
    out = []
    for rank, item in enumerate(ranked, start=1):
        out.append({
            "rank": rank,
            "index": item["index"],
            "document": item["document"],
            "score": round_float(item["score"]),
            "probability": round_float(item["probability"]),
            "trueLogit": round_float(item["trueLogit"]),
            "falseLogit": round_float(item["falseLogit"]),
            "tokenCount": item["tokenCount"],
        })
    return out


def score_documents(model, tokenizer, device, torch, query, documents, config, batch_size):
    scores = []
    true_id = config["trueTokenId"]
    false_id = config["falseTokenId"]
    for offset, batch_docs in chunks(documents, batch_size):
        prompts = [format_prompt(query, doc, config) for doc in batch_docs]
        encoded = tokenizer(prompts, padding=True, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            output = model(**encoded)
        logits = output.logits
        attention_mask = encoded.get("attention_mask")
        if attention_mask is None:
            last_indices = torch.full((len(batch_docs),), logits.shape[1] - 1, device=device, dtype=torch.long)
        else:
            last_indices = attention_mask.sum(dim=1).to(torch.long) - 1
        batch_indices = torch.arange(len(batch_docs), device=device)
        next_logits = logits[batch_indices, last_indices, :]
        true_logits = next_logits[:, true_id].detach().float().cpu().tolist()
        false_logits = next_logits[:, false_id].detach().float().cpu().tolist()
        token_counts = encoded["input_ids"].ne(tokenizer.pad_token_id).sum(dim=1).detach().cpu().tolist()
        for local_index, document in enumerate(batch_docs):
            true_logit = float(true_logits[local_index])
            false_logit = float(false_logits[local_index])
            score = true_logit if config["score"] == "true_logit" else true_logit - false_logit
            scores.append({
                "index": offset + local_index,
                "document": document,
                "score": score,
                "probability": sigmoid(score),
                "trueLogit": true_logit,
                "falseLogit": false_logit,
                "tokenCount": int(token_counts[local_index]),
            })
    ranking = rank_scores(scores)
    top = ranking[0] if ranking else None
    return {
        "scores": scores,
        "ranking": ranking,
        "topDocument": {
            "index": top["index"],
            "document": top["document"],
            "score": top["score"],
            "probability": top["probability"],
        } if top else None,
    }


def run_once(model, tokenizer, device, torch, query, documents, config, batch_size):
    synchronize(device, torch)
    start = time.perf_counter()
    result = score_documents(model, tokenizer, device, torch, query, documents, config, batch_size)
    synchronize(device, torch)
    elapsed_ms = (time.perf_counter() - start) * 1000
    non_finite = sum(1 for item in result["scores"] if not math.isfinite(item["score"]))
    top = result["topDocument"] or {}
    return {
        "rerankMs": elapsed_ms,
        "documentCount": len(documents),
        "topDocumentIndex": top.get("index"),
        "topDocumentScore": top.get("score"),
        "topDocumentProbability": top.get("probability"),
        "nonFiniteScores": non_finite,
        "ranking": result["ranking"],
    }


def load_semantic_fixture(path):
    payload = json.loads(Path(path).read_text())
    defaults = payload.get("defaults", {})
    cases = defaults.get("cases", [])
    if not isinstance(cases, list) or not cases:
        fail(f"{path} must contain defaults.cases")
    return {
        "cases": cases,
        "minPairAcc": float(defaults.get("minPairAcc", 0.8)),
        "minScoreMargin": float(defaults.get("minScoreMargin", 0)),
    }


def run_semantic(model, tokenizer, device, torch, fixture, config, batch_size):
    start = time.perf_counter()
    pairs = []
    passed = 0
    for case in fixture["cases"]:
        query = str(case["query"])
        docs = [str(case["positive"]), str(case["negative"])]
        result = score_documents(model, tokenizer, device, torch, query, docs, config, batch_size)
        scores = {item["index"]: item for item in result["scores"]}
        positive = scores[0]
        negative = scores[1]
        margin = positive["score"] - negative["score"]
        pair_passed = math.isfinite(margin) and margin > fixture["minScoreMargin"]
        if pair_passed:
            passed += 1
        pairs.append({
            "id": case.get("id"),
            "query": query,
            "positive": docs[0],
            "negative": docs[1],
            "passed": pair_passed,
            "positiveScore": round_float(positive["score"]),
            "negativeScore": round_float(negative["score"]),
            "positiveProbability": round_float(positive["probability"]),
            "negativeProbability": round_float(negative["probability"]),
            "margin": round_float(margin),
        })
    pair_acc = passed / len(pairs)
    return {
        "passed": pair_acc >= fixture["minPairAcc"],
        "pairAcc": pair_acc,
        "pairPassed": passed,
        "pairTotal": len(pairs),
        "minPairAcc": fixture["minPairAcc"],
        "minScoreMargin": fixture["minScoreMargin"],
        "failedCaseIds": [pair["id"] for pair in pairs if not pair["passed"]],
        "durationMs": (time.perf_counter() - start) * 1000,
        "details": {"pairs": pairs},
    }


def build_environment(model_id, device, dtype_name, torch, transformers):
    cuda = {}
    if device == "cuda":
        cuda = {
            "deviceName": torch.cuda.get_device_name(0),
            "deviceCount": torch.cuda.device_count(),
            "cudaVersion": torch.version.cuda,
        }
    return {
        "host": {
            "platform": sys.platform,
            "machine": platform.machine(),
            "pythonVersion": platform.python_version(),
            "processor": platform.processor(),
        },
        "runtime": {
            "library": "transformers",
            "version": getattr(transformers, "__version__", None),
            "torchVersion": getattr(torch, "__version__", None),
            "surface": "python",
            "device": device,
            "dtype": dtype_name,
            "modelId": model_id,
        },
        "cuda": cuda,
    }


def main():
    args = parse_args()
    if args.warmup < 0:
        fail("--warmup must be non-negative")
    if args.runs <= 0:
        fail("--runs must be positive")
    if args.batch_size <= 0:
        fail("--batch-size must be positive")
    query = str(args.query).strip()
    if not query:
        fail("--query must be non-empty")
    documents = normalize_documents(args)
    scoring_config = normalize_scoring_config(args.rerank_config)

    try:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
        transformers.logging.set_verbosity_error()
        transformers.utils.logging.disable_progress_bar()
    except Exception as exc:
        fail(f"failed to import torch/transformers: {exc}")

    device = resolve_device(args.device, torch)
    dtype = resolve_dtype(args.dtype, device, torch)
    dtype_name = str(dtype).replace("torch.", "")
    tokenizer_kwargs = {
        "local_files_only": not args.allow_download,
        "trust_remote_code": args.trust_remote_code,
    }
    load_kwargs = dict(tokenizer_kwargs)
    if dtype is not None:
        load_kwargs["torch_dtype"] = dtype

    load_start = time.perf_counter()
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, **tokenizer_kwargs)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
        model.to(device)
        model.eval()
        synchronize(device, torch)
    except Exception as exc:
        fail(f"failed to load {args.model}: {exc}")
    model_load_ms = (time.perf_counter() - load_start) * 1000

    for _ in range(args.warmup):
        run_once(model, tokenizer, device, torch, query, documents, scoring_config, args.batch_size)

    runs = []
    for _ in range(args.runs):
        runs.append(run_once(model, tokenizer, device, torch, query, documents, scoring_config, args.batch_size))

    rerank_ms = [run["rerankMs"] for run in runs if math.isfinite(run["rerankMs"])]
    valid_runs = len(rerank_ms)
    invalid_runs = len(runs) - valid_runs
    first_output = score_documents(model, tokenizer, device, torch, query, documents, scoring_config, args.batch_size)
    semantic = run_semantic(
        model,
        tokenizer,
        device,
        torch,
        load_semantic_fixture(args.semantic_fixture),
        scoring_config,
        args.batch_size,
    )
    avg_ms = statistics.fmean(rerank_ms) if rerank_ms else None
    document_count = len(documents)
    avg_reranks_per_sec = (document_count * 1000 / avg_ms) if avg_ms and avg_ms > 0 else None
    top = first_output["topDocument"] or {}

    result = {
        "schemaVersion": 1,
        "kind": "hf-transformers-reranker-bench",
        "timestamp": args.timestamp,
        "modelId": args.model,
        "task": "rerank",
        "query": query,
        "documents": documents,
        "warmupRuns": args.warmup,
        "timedRuns": args.runs,
        "batchSize": args.batch_size,
        "modelLoadMs": model_load_ms,
        "timing": {
            "modelLoadMs": round_float(model_load_ms),
            "rerankMs": round_float(statistics.median(rerank_ms)) if rerank_ms else None,
            "totalRunMs": round_float(statistics.median(rerank_ms)) if rerank_ms else None,
            "reranksPerSec": round_float(avg_reranks_per_sec),
            "cacheMode": "warm",
            "loadMode": "local-files" if not args.allow_download else "hf-download-allowed",
        },
        "metrics": {
            "warmupRuns": args.warmup,
            "timedRuns": args.runs,
            "validRuns": valid_runs,
            "invalidRuns": invalid_runs,
            "nonFiniteScores": sum(run["nonFiniteScores"] for run in runs),
            "firstRerankMs": round_float(rerank_ms[0]) if rerank_ms else None,
            "minRerankMs": round_float(min(rerank_ms)) if rerank_ms else None,
            "medianRerankMs": round_float(statistics.median(rerank_ms)) if rerank_ms else None,
            "p95RerankMs": round_float(percentile(rerank_ms, 0.95)),
            "p99RerankMs": round_float(percentile(rerank_ms, 0.99)),
            "maxRerankMs": round_float(max(rerank_ms)) if rerank_ms else None,
            "avgRerankMs": round_float(avg_ms),
            "avgReranksPerSec": round_float(avg_reranks_per_sec),
            "documentCount": document_count,
            "topDocumentIndex": top.get("index"),
            "topDocumentScore": top.get("score"),
            "topDocumentProbability": top.get("probability"),
            "semanticPassed": semantic["passed"],
            "semanticDurationMs": round_float(semantic["durationMs"]),
            "semanticPairAcc": round_float(semantic["pairAcc"]),
            "semanticPairPassed": semantic["pairPassed"],
            "semanticPairTotal": semantic["pairTotal"],
            "semanticMinPairAcc": semantic["minPairAcc"],
            "semanticMinScoreMargin": semantic["minScoreMargin"],
            "semanticFailedCases": semantic["failedCaseIds"],
            "semanticDetails": semantic["details"],
            "modelLoadMs": round_float(model_load_ms),
        },
        "output": {
            "mode": "rerank",
            "query": query,
            "documentCount": document_count,
            "topDocument": top,
            "ranking": first_output["ranking"],
            "semantic": {
                "passed": semantic["passed"],
                "pairAcc": round_float(semantic["pairAcc"]),
                "failedCaseIds": semantic["failedCaseIds"],
                "details": semantic["details"],
            },
        },
        "runs": runs,
        "environment": build_environment(args.model, device, dtype_name, torch, transformers),
        "determinism": {
            "seed": None,
            "rerank": {
                "documentCount": document_count,
                "scoringFormat": scoring_config["format"],
                "trueTokenId": scoring_config["trueTokenId"],
                "falseTokenId": scoring_config["falseTokenId"],
            },
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
