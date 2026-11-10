"""Bind a new-prompt source reference to the unchanged frozen GPU candidate.

Usage: prepare-run.py PROTOCOL INPUT_RECEIPT INPUTS REFERENCE PARENT_RUN NEW_OUTPUT_DIRECTORY
"""
from pathlib import Path
import hashlib
import json
import sys

paths = list(map(Path, sys.argv[1:]))
protocol_path, receipt_path, inputs_path, reference_path, parent_path, output = paths
sha = lambda raw: hashlib.sha256(raw).hexdigest()
def require(condition, message):
    if not condition:
        raise ValueError(message)
raws = [p.read_bytes() for p in paths[:-1]]
protocol, receipt, inputs, reference, parent = map(json.loads, raws)
require(sha(raws[1]) == protocol['inputReceiptSha256'], 'Input receipt mismatch')
require(sha(raws[2]) == protocol['inputsSha256'] == receipt['inputsSha256'] == reference['inputsSha256'], 'Input identity mismatch')
require(reference['producerSha256'] == protocol['sourceProducerSha256'], 'Source producer mismatch')
require(reference['artifactManifestSha256'] == parent['expectedManifestSha256'] == protocol['manifestSha256'], 'Artifact identity mismatch')
require(reference['dtype'] == 'float32' and reference['surface'] == 'CPU'
        and reference.get('kvStorageDtype', 'f32') == protocol['kvDtype'] == 'f32', 'Source precision mismatch')
require(parent['runtimeConfig']['inference']['session']['kvcache']['kvDtype'] == 'f32', 'Parent runtime precision mismatch')
require(list(reference['sequences']) == [r['id'] for r in protocol['cases']]
        == [r['id'] for r in inputs['sequences']], 'Prompt coverage mismatch')
for case in protocol['cases']:
    require(reference['sequences'][case['id']]['tokens'] == case['tokens'] and case['outputRows'] == 1, 'Prompt geometry mismatch')
parent['sequences'] = [dict(id=r['id'], maxTokens=r['outputRows']) for r in protocol['cases']]
parent['captureRouterLayers'] = []
parent['captureQsaLayers'] = []
parent['captureLayers'] = False
parent['captureLogitRows'] = True
parent['lifecycle'] = False
parent['inputsPath'] = str(inputs_path.resolve())
parent['referencePath'] = str(reference_path.resolve())
parent['outputPath'] = str((output / 'observation.json').resolve())
parent['parentRunSha256'] = sha(raws[4])
output.mkdir()
(output / 'run.json').write_text(json.dumps(parent, indent=2) + '\n')
preparation = dict(schema='doppler.flash-next-new-prompt-run/v1', producerSha256=sha(Path(__file__).read_bytes()),
                   protocolSha256=sha(raws[0]), inputsSha256=sha(raws[2]), referenceSha256=sha(raws[3]),
                   parentRunSha256=sha(raws[4]), runSha256=sha((output / 'run.json').read_bytes()),
                   numericRuntimeConfigUnchanged=True, referenceTensorInjection=False,
                   acceptanceChanged=False, normalModelQualified=False)
(output / 'preparation.json').write_text(json.dumps(preparation, indent=2) + '\n')
print(json.dumps(preparation))
