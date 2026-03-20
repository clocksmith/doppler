#!/usr/bin/env python3
"""GPT-OSS MoE benchmark ratchet scaffold.

Runs Doppler bench via the current CLI contract and emits a JSON summary.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

def parse_json_payload(output: str) -> dict | None:
    text = (output or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        help="Model ID (e.g. gpt-oss-20b-f16-xmxfp4)",
    )
    parser.add_argument("--runtime-profile", default="profiles/throughput")
    parser.add_argument("--prompt", default="short")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--output", default="benchmarks/kernels/gptoss_moe_latest.json")
    args = parser.parse_args()

    cmd = [
        "npm",
        "run",
        "--silent",
        "bench",
        "--",
        "--runtime-profile",
        args.runtime_profile,
        "--model-id",
        args.model,
        "--prompt",
        args.prompt,
        "--max-tokens",
        str(args.max_tokens),
        "--runs",
        "1",
        "--warmup",
        "1",
        "--json",
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    parsed = parse_json_payload(proc.stdout or "")
    bench_result = parsed.get("result") if isinstance(parsed, dict) else None
    bench_metrics = bench_result.get("metrics") if isinstance(bench_result, dict) else {}

    metrics = {
        "first_token_ms": bench_metrics.get("firstTokenMs"),
        "decode_tokens_per_s": bench_metrics.get("decodeTokensPerSec"),
        "prefill_tokens_per_s": bench_metrics.get("prefillTokensPerSec"),
        "model_load_ms": bench_metrics.get("modelLoadMs"),
    }

    payload = {
        "schemaVersion": 1,
        "model": args.model,
        "runtimeProfile": args.runtime_profile,
        "prompt": args.prompt,
        "maxTokens": args.max_tokens,
        "command": cmd,
        "returnCode": proc.returncode,
        "metrics": metrics,
        "parseError": None if parsed is not None else "failed to parse bench JSON output",
        "benchResult": bench_result if isinstance(bench_result, dict) else None,
        "stdoutTail": (proc.stdout or "")[-1200:],
        "stderrTail": (proc.stderr or "")[-1200:],
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(payload, indent=2))
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
