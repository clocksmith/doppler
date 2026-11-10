"""Verify the independent F32 greedy chain, then prepare a bounded GPU decode run.

Usage: prepare-learned-decode-run.py INPUT_RECEIPT DECODE_REFERENCE ORIGINAL_REFERENCE
       ORIGINAL_INPUTS PARENT_GPU_RUN TOLERANCE_POLICY NEW_OUTPUT_DIRECTORY
"""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np

if len(sys.argv) != 8:
    raise SystemExit(__doc__)
input_receipt_path, decode_path, original_path, inputs_path, parent_path, policy_path, output = map(Path, sys.argv[1:])
sha = lambda raw: hashlib.sha256(raw).hexdigest()
def require(condition, message):
    if not condition:
        raise ValueError(message)
input_receipt, decode, original, inputs, parent, policies = [json.loads(p.read_bytes()) for p in [
    input_receipt_path, decode_path, original_path, inputs_path, parent_path, policy_path]]
fixed_path = input_receipt_path.parent / 'inputs.json'
fixed = json.loads(fixed_path.read_bytes())
require(sha(fixed_path.read_bytes()) == input_receipt['inputsSha256'] == decode['inputsSha256'], 'Decode input identity mismatch')
require(sha(inputs_path.read_bytes()) == input_receipt['originalInputsSha256'] == original['inputsSha256'], 'Original input identity mismatch')
require(sha(original_path.read_bytes()) == input_receipt['originalF32ReferenceSha256'], 'Original reference mismatch')
require(decode['artifactManifestSha256'] == original['artifactManifestSha256'] == parent['expectedManifestSha256']
        == input_receipt['manifestSha256'], 'Artifact identity mismatch')
require(decode['producerSha256'] == original['producerSha256'] and decode['checkpoint'] == original['checkpoint'], 'Source identity mismatch')
require(decode['dtype'] == original['dtype'] == 'float32' and decode['surface'] == 'CPU', 'Source must use CPU F32')
require(decode.get('kvStorageDtype', 'f32') == original.get('kvStorageDtype', 'f32') == 'f32', 'F16 reference cannot qualify this lane')
policy_id = 'doppler.boundary-tolerance/quantized-v1'
policy = policies['tolerancePolicies'][policy_id]
source_payloads = {}
def checked_logits(reference, root, name):
    row = reference['sequences'][name]
    path = root / row['logitsFile']
    raw = path.read_bytes()
    require(sha(raw) == row['logitsSha256'], 'Source logits hash mismatch: ' + name)
    values = np.load(path, allow_pickle=False)
    require(values.shape == (len(row['samplePositions']), 248320) and values.dtype == np.float32
            and np.isfinite(values).all(), 'Invalid source logits: ' + name)
    last = row['samplePositions'].index(row['tokens'] - 1)
    require(int(np.argmax(values[last])) == row['argmax'][-1], 'Source argmax metadata mismatch: ' + name)
    source_payloads[name] = dict(shape=list(values.shape), sha256=sha(raw))
    return row
chains = []
for name in ['short', 'long']:
    base = next(s for s in inputs['sequences'] if s['id'] == name)
    a = checked_logits(original, original_path.parent, name)
    b = checked_logits(decode, decode_path.parent, name + '-next')
    c = checked_logits(decode, decode_path.parent, name + '-next2')
    first = next(s for s in fixed['sequences'] if s['id'] == name + '-next')
    second = next(s for s in fixed['sequences'] if s['id'] == name + '-next2')
    require(first['tokens'] == base['tokens'] + [a['argmax'][-1]], 'First prefix is not the F32 source trajectory')
    require(second['tokens'] == first['tokens'] + [b['argmax'][-1]], 'Second prefix is not the F32 source trajectory')
    require([a['tokens'], b['tokens'], c['tokens']] == [len(base['tokens']) + i for i in range(3)], 'Source sequence lengths differ')
    chains.append(dict(id=name, promptTokens=len(base['tokens']),
                       expectedTokenIds=[a['argmax'][-1], b['argmax'][-1], c['argmax'][-1]],
                       referenceCases=[name, name + '-next', name + '-next2']))
output.mkdir()
parent_raw = parent_path.read_bytes()
parent['sequences'] = [dict(id=name, maxTokens=3) for name in ['short', 'long']]
parent['captureRouterLayers'] = []
parent['captureQsaLayers'] = []
parent['captureLayers'] = False
parent['captureLogitRows'] = True
parent['lifecycle'] = False
parent['inputsPath'] = str(inputs_path.resolve())
parent['referencePath'] = str(original_path.resolve())
parent['outputPath'] = str((output / 'observation.json').resolve())
parent['parentRunSha256'] = sha(parent_raw)
parent['decodeQualification'] = dict(sourceDecodeReferenceSha256=sha(decode_path.read_bytes()),
                                    sourceInputsSha256=decode['inputsSha256'],
                                    sourceMethod='Independent cache-free full-prefix recomputation, not source cache execution',
                                    sourceGreedyChainVerifiedBeforeGpu=True)
(output / 'run.json').write_text(json.dumps(parent, indent=2) + '\n')
protocol = dict(schema='doppler.flash-next-learned-decode-protocol/v1',
                scope='Original prefill and two cached decode steps per prompt, against independent F32 full-prefix source forwards',
                producerSha256=sha(Path(__file__).read_bytes()), inputReceiptSha256=sha(input_receipt_path.read_bytes()),
                inputsSha256=input_receipt['originalInputsSha256'], decodeInputsSha256=decode['inputsSha256'],
                originalReferenceSha256=sha(original_path.read_bytes()), decodeReferenceSha256=sha(decode_path.read_bytes()),
                sourceProducerSha256=decode['producerSha256'], manifestSha256=decode['artifactManifestSha256'],
                parentGpuRunSha256=sha(parent_raw), runSha256=sha((output / 'run.json').read_bytes()),
                policyId=policy_id, policySha256=sha(policy_path.read_bytes()), tolerancePolicy=policy,
                chains=chains, sourceLogitPayloads=source_payloads,
                maximumGenerationRowsPerPrompt=3, decodeStepsPerPrompt=2,
                sourceGreedyChainVerifiedBeforeGpu=True, acceptanceChanged=False, normalModelQualified=False)
(output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
print(json.dumps(dict(runSha256=protocol['runSha256'], chains=chains)))
