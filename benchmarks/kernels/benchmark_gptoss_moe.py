#!/usr/bin/env python3
"""
GPT-OSS MoE benchmark ratchet scaffold.

Runs Doppler bench command and emits a JSON summary that can be tracked in CI.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

METRIC_PATTERNS = {
    "first_token_ms": re.compile(r"First token\s*:\s*([0-9.]+)ms", re.IGNORECASE),
    "tokens_per_s": re.compile(r"Generated\s*:\s*[0-9]+\s*tokens\s*\(([0-9.]+) tok/s\)", re.IGNORECASE),
}


def parse_metrics(output: str) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {"first_token_ms": None, "tokens_per_s": None}
    for key, pattern in METRIC_PATTERNS.items():
        match = pattern.search(output)
        if match:
            metrics[key] = float(match.group(1))
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model ID (e.g. gpt-oss-20b)")
    parser.add_argument("--runtime-preset", default="bench")
    parser.add_argument("--prompt", default="short")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--output", default="benchmarks/kernels/gptoss_moe_latest.json")
    args = parser.parse_args()

    cmd = [
        "npm",
        "run",
        "bench",
        "--",
        "--config",
        args.runtime_preset,
        "-m",
        args.model,
        "--prompt",
        args.prompt,
        "--max-tokens",
        str(args.max_tokens),
        "--runs",
        "1",
        "--warmup",
        "1",
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    metrics = parse_metrics(output)

    payload = {
        "model": args.model,
        "runtimePreset": args.runtime_preset,
        "prompt": args.prompt,
        "maxTokens": args.max_tokens,
        "returnCode": proc.returncode,
        "metrics": metrics,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(payload, indent=2))
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
