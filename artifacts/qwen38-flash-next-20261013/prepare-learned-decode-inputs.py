"""Freeze source-owned continuation prefixes before testing cached GPU decode.

Usage: prepare-learned-decode-inputs.py ORIGIN_PROTOCOL BASE_INPUTS F32_REFERENCE
       SOURCE_DECODE_REFERENCE SOURCE_DECODE_INPUTS ARTIFACT NEW_OUTPUT_DIRECTORY
"""
from pathlib import Path
import datetime
import hashlib
import json
import sys

from tokenizers import Tokenizer

if len(sys.argv) != 8:
    raise SystemExit(__doc__)
protocol_path, inputs_path, reference_path, old_reference_path, old_inputs_path, artifact, output = map(Path, sys.argv[1:])
sha = lambda raw: hashlib.sha256(raw).hexdigest()
def require(condition, message):
    if not condition:
        raise ValueError(message)
raws = [p.read_bytes() for p in [protocol_path, inputs_path, reference_path, old_reference_path, old_inputs_path]]
protocol, inputs, reference, old_reference, old_inputs = map(json.loads, raws)
bindings = protocol['bindings']
require(sha(raws[1]) == bindings['inputsSha256'] == reference['inputsSha256'], 'Original input identity mismatch')
require(sha(raws[2]) == bindings['referenceSha256'], 'F32 reference identity mismatch')
require(sha(raws[4]) == old_reference['inputsSha256'], 'Source continuation input identity mismatch')
manifest_raw = (artifact / 'manifest.json').read_bytes()
require(sha(manifest_raw) == bindings['manifestSha256'] == reference['artifactManifestSha256']
        == old_reference['artifactManifestSha256'], 'Artifact identity mismatch')
require(reference.get('kvStorageDtype', 'f32') == 'f32' and old_reference['kvStorageDtype'] == 'f16', 'Unexpected reference lanes')
require(reference['checkpoint'] == old_reference['checkpoint'] and reference['producerSha256'] == old_reference['producerSha256'],
        'Independent source producer mismatch')
tokenizer_raw = (artifact / 'tokenizer.json').read_bytes()
tokenizer = Tokenizer.from_str(tokenizer_raw.decode())
rows, selections = [], []
for name in ['short', 'long']:
    base = next(r for r in inputs['sequences'] if r['id'] == name)
    first = int(reference['sequences'][name]['argmax'][-1])
    old_next = next(r for r in old_inputs['sequences'] if r['id'] == name + '-next')
    require(old_next['tokens'] == base['tokens'] + [first], 'Source continuation does not extend the exact original prefix')
    next_token = int(old_reference['sequences'][name + '-next']['argmax'][-1])
    continuation = [first, next_token]
    for step in [1, 2]:
        tokens = base['tokens'] + continuation[:step]
        old_second = next((r for r in old_inputs['sequences'] if r['id'] == name + '-next2'), None)
        if step == 2 and old_second:
            require(old_second['tokens'] == tokens, 'Retained second source prefix differs')
        rows.append(dict(id=name + '-next' + ('2' if step == 2 else ''),
                         text=tokenizer.decode(tokens, skip_special_tokens=False), tokens=tokens,
                         originalSequence=name, continuationTokenIds=continuation[:step]))
    selections.append(dict(id=name, originalTokens=len(base['tokens']), continuationTokenIds=continuation,
                           firstTokenSource='Frozen independent F32 reference',
                           secondTokenSource='Retained source F16-cache reference, to be checked against fresh F32 source before GPU use'))
output.mkdir()
payload = dict(schema=inputs['schema'], tokenizers=inputs['tokenizers'], sequences=rows)
(output / 'inputs.json').write_text(json.dumps(payload, indent=2) + '\n')
receipt = dict(schema='doppler.flash-next-decode-inputs/v1', createdAtUtc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               scope='Fixed source-owned prefixes for an independent F32 full-prefix reference; no Doppler output used',
               producerSha256=sha(Path(__file__).read_bytes()), originalInputsSha256=sha(raws[1]),
               originalF32ReferenceSha256=sha(raws[2]), sourceDecodeReferenceSha256=sha(raws[3]),
               sourceDecodeInputsSha256=sha(raws[4]), manifestSha256=sha(manifest_raw),
               tokenizerSha256=sha(tokenizer_raw), inputsSha256=sha((output / 'inputs.json').read_bytes()),
               inputs=[dict(id=r['id'], tokens=len(r['tokens'])) for r in rows], selections=selections,
               sourceStatePrecision='float32', sourceKvPrecision='float32', gpuRunsStarted=0,
               acceptanceChanged=False, normalModelQualified=False,
               mandatoryBeforeGpu='Verify fresh F32 next-token predictions equal these fixed continuation tokens; otherwise stop and prepare source-greedy prefixes')
(output / 'inputs-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(dict(inputsSha256=receipt['inputsSha256'], inputs=receipt['inputs'])))
