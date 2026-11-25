#!/usr/bin/env python3
"""Summarize a run-sessions.sh output directory: tokens, reference agreement, time to first token, per-token decode,
GPU memory and bytes. Usage: python3 webgpu/summarize-sessions.py <dir> [<dir> ...]"""
import glob
import json
import os
import statistics
import sys


def load(path):
    with open(path) as f:
        return json.load(f)


for d in sys.argv[1:]:
    gpu = load(os.path.join(d, 'gpu-evaluator.json')) if os.path.exists(os.path.join(d, 'gpu-evaluator.json')) else None
    print(f'== {os.path.basename(d.rstrip("/"))}' + (f' ({gpu["adapter"]["vendor"]} {gpu["adapter"]["architecture"]})' if gpu else ''))
    for i, req in enumerate(sorted(glob.glob(os.path.join(d, 'requester-*.json')), key=lambda p: int(p.split('-')[-1][:-5])), 1):
        r = load(req)
        cmp_path = os.path.join(d, f'compare-{i}.json')
        c = load(cmp_path) if os.path.exists(cmp_path) else {}
        steps = r['steps']
        dec = [s['roundTripSeconds'] for s in steps[1:]]
        gsteps = gpu['sessions'][i - 1]['steps'] if gpu and len(gpu['sessions']) >= i else []
        gdec = [s['computeSeconds'] for s in gsteps[1:]]
        print(f"  prompt {r['prompt']!r} -> {r['generatedText']!r} ({len(steps)} tokens) {r['outcome']} {r.get('error') or ''}")
        print(f"    integrity: {r['integrity']}; decryptions {r['decryptCalls']}; pinned reference "
              f"{c.get('matchesApproximateReference')}, float64 {c.get('matchesExactReference')}; "
              f"max logit error vs pinned {c.get('maxLogitErrorVsApproximate', float('nan')):.3f}")
        print(f"    keys: uploaded {r['evaluationKeyBytesPerEvaluator']} bytes, reused without upload {r['evaluationKeysReusedNoUpload']}, "
              f"keygen {r['keygenSeconds']:.2f} s, key upload {r['keyUploadSeconds']:.1f} s")
        print(f"    time to first token {r['timeToFirstTokenSeconds']:.2f} s (prefill round trip {steps[0]['roundTripSeconds']:.2f} s)")
        if dec:
            print(f"    decode round trip per token: median {statistics.median(dec):.2f} s, range {min(dec):.2f}-{max(dec):.2f} s")
        if gsteps:
            print(f"    GPU evaluator: prefill {gsteps[0]['computeSeconds']:.2f} s; decode median {statistics.median(gdec):.2f} s "
                  f"(range {min(gdec):.2f}-{max(gdec):.2f}); peak GPU buffers {gpu['sessions'][i - 1]['peakBufferBytes'] / 1e9:.2f} GB; "
                  f"keys from {gpu['sessions'][i - 1].get('keysFrom')}")
        up = [s['uploadBytesPerEvaluator'] for s in steps]
        down = [s['downloadBytesPerEvaluator'] for s in steps]
        print(f"    upload per decode token {up[-1]} bytes (prefill {up[0]}), download per token {down[-1]} bytes; "
              f"result digests equal across evaluators at every step: {r['integrity'].startswith('dual')}")
