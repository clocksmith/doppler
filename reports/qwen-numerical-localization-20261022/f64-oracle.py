#!/usr/bin/env python3
"""Offline F64 next-token oracle over deployed Qwen3.5 Q4_K weights for one prefix.

Reuses tools/qwen35-f64-reference.py equations and loader. Shards are verified
against an independently retained SHA-256 inventory before dequantization.
CPU diagnostic only; nothing here enters Doppler Run.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

TOOL = Path(__file__).resolve().parents[2] / 'tools' / 'qwen35-f64-reference.py'
spec = importlib.util.spec_from_file_location('qwen35_f64_reference', TOOL)
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--source-config', required=True)
    parser.add_argument('--inventory', required=True, help='JSON with files[].file/bytes/sha256')
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--continuation', type=int, nargs='*', default=[], help='Token IDs appended after the prompt')
    parser.add_argument('--watch', type=int, nargs='*', default=[], help='Token IDs whose logits are always reported')
    parser.add_argument('--top', type=int, default=10)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    root = Path(args.model)
    manifest = json.loads((root / 'manifest.json').read_text())
    inventory = json.loads(Path(args.inventory).read_text())
    trusted = {row['file']: row for row in inventory['files']}
    # One piece per shard: load_weights then checks every shard against the trusted SHA-256.
    pieces = {'files': [{'path': shard['filename'], 'size': shard['size'],
                         'pieces': [{'offset': 0, 'size': shard['size'],
                                     'identity': 'sha256:' + trusted[shard['filename']]['sha256']}]}
                        for shard in manifest['shards']]}
    weights, identities = reference.load_weights(root, manifest, pieces)
    config = json.loads(Path(args.source_config).read_text())['text_config']

    tokenizer = Tokenizer.from_file(str(root / manifest['tokenizer']['file']))
    prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=False).ids
    token_ids = prompt_ids + list(args.continuation)
    logits, _ = reference.evaluate(token_ids, weights, config, np.array([len(token_ids) - 1]))
    row = np.asarray(logits[-1], dtype=np.float64)
    order = np.argsort(-row)
    top = [{'tokenId': int(i), 'logit': float(row[i]), 'text': tokenizer.decode([int(i)])} for i in order[:args.top]]
    report = {
        'schema': 'doppler.qwen35-f64-next-token-oracle/v1',
        'toolSha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'referenceToolSha256': hashlib.sha256(TOOL.read_bytes()).hexdigest(),
        'manifestSha256': hashlib.sha256((root / 'manifest.json').read_bytes()).hexdigest(),
        'shardIdentities': identities,
        'prompt': args.prompt,
        'promptTokenIds': prompt_ids,
        'tokenIds': token_ids,
        'top': top,
        'winner': top[0]['tokenId'],
        'marginTop1Top2': top[0]['logit'] - top[1]['logit'],
        'watch': {str(t): float(row[t]) for t in args.watch},
        'logitsF64Sha256': hashlib.sha256(row.astype('<f8').tobytes()).hexdigest(),
    }
    Path(args.out).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'shardIdentities'}, indent=2))


if __name__ == '__main__':
    main()
