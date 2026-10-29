"""Retain reference identity, physical cache proof and comparisons without logits."""
import argparse
import json
from pathlib import Path

from binding import digest


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--receipt", required=True, type=Path)
parser.add_argument("--out", required=True, type=Path)
args = parser.parse_args()
source = json.loads(args.receipt.read_text())
if source.get("schema") != "doppler.native-hf-canonical-reference.v1" or source.get("completed") is not True:
    raise ValueError("native reference receipt is incomplete or unsupported")
if args.out.exists():
    raise ValueError("refusing to overwrite retained reference summary")
summary = {key: source[key] for key in ("schema", "scope", "inputs", "byteVerification", "frameworks",
                                      "sourceSha256", "engineReceipts", "contract", "promptIds", "comparisons", "completed")}
summary["schema"] = "doppler.native-hf-canonical-reference-summary.v1"
summary["fullReceipt"] = {"path": str(args.receipt), "sha256": digest(args.receipt),
                          "parameterCount": len(source["parameters"])}
summary["runs"] = [{key: run[key] for key in ("name", "cacheEvents", "linearStates")} for run in source["runs"]]
args.out.write_text(json.dumps(summary, indent=2) + "\n")
