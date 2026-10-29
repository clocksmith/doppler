"""Retain sparse logits identity, tokens and complete physical-cache audit totals."""
import argparse
import json
from pathlib import Path

from binding import digest


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--receipt", required=True, type=Path)
parser.add_argument("--out", required=True, type=Path)
args = parser.parse_args()
source = json.loads(args.receipt.read_text())
if source.get("schema") != "doppler.native-hf-canonical-teacher.v1" or source.get("completed") is not True:
    raise ValueError("native teacher receipt is incomplete or unsupported")
if args.out.exists():
    raise ValueError("refusing to overwrite teacher summary")
summary = {key: source[key] for key in ("scope", "inputs", "byteVerification", "frameworks", "sources", "contracts",
                                      "tokenizer", "contract", "completed", "elapsedSeconds")}
summary["schema"] = "doppler.native-hf-canonical-teacher-summary.v1"
summary["fullReceipt"] = {"path": str(args.receipt), "sha256": digest(args.receipt),
                          "parameterCount": len(source["parameterBindings"])}
summary["prompts"] = []
for prompt in source["prompts"]:
    record = {key: value for key, value in prompt.items() if key != "cacheChecks"}
    full_events = [event for step in prompt["cacheChecks"] for event in step["fullAttention"]]
    linear_states = [state for step in prompt["cacheChecks"] for state in step["linearStates"]]
    expected_dtype = "torch." + source["contract"]["kvPhysicalDtype"]
    if any(e["physicalKeyDtype"] != expected_dtype or e["physicalValueDtype"] != expected_dtype
           or e["attentionKeyDtype"] != "torch.float32" or e["attentionValueDtype"] != "torch.float32" for e in full_events):
        raise ValueError("teacher summary observed an incorrect actual KV or attention dtype")
    if any(s["convDtype"] != "torch.float32" or s["recurrentDtype"] != "torch.float32" for s in linear_states):
        raise ValueError("teacher summary observed incorrect linear state dtype")
    record["cacheAudit"] = {"steps": len(prompt["cacheChecks"]), "fullAttentionUpdates": len(full_events),
                            "linearStateChecks": len(linear_states), "physicalKVDtype": source["contract"]["kvPhysicalDtype"],
                            "allPhysicalKVMatchesContract": True,
                            "allAttentionInputsF32": True, "allLinearStatesF32": True}
    summary["prompts"].append(record)
args.out.write_text(json.dumps(summary, indent=2) + "\n")
