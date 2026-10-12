"""Diagnostic only: restore source tying after loading verified deployed tensors."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--policy', required=True)
parser.add_argument('--out', required=True)
args = parser.parse_args()
policy = json.loads(Path(args.policy).read_text())
assert policy['schema'] == 'doppler.tied-head-sensitivity/v1'
reference_path = Path('tools/qwen35-deployed-reference.py')
spec = importlib.util.spec_from_file_location('deployed_reference', reference_path)
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
replay = reference.replay_generation


def tied_replay(model, config, manifest, options, report):
    original = Path(policy['originalWeights'])
    digest = hashlib.sha256()
    with original.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    assert digest.hexdigest() == policy['originalWeightsSha256']
    with safe_open(original, framework='pt', device='cpu') as handle:
        original_embedding = handle.get_tensor('model.language_model.embed_tokens.weight')
    embedding = model.model.embed_tokens.weight
    head = model.lm_head.weight
    assert head is not embedding
    assert original_embedding.shape == embedding.shape == head.shape
    squared_error, maximum_error = 0.0, 0.0
    embedding_digest = hashlib.sha256()
    embedding_mismatches = 0
    embedding_maximum_error = 0.0
    for start in range(0, len(embedding), 4096):
        end = start + 4096
        expected = original_embedding[start:end].float()
        embedding_digest.update(embedding[start:end].detach().half().numpy().tobytes())
        embedding_mismatches += int((embedding[start:end] != expected).sum())
        embedding_maximum_error = max(embedding_maximum_error, float((embedding[start:end] - expected).abs().max()))
        error = head[start:end] - expected
        maximum_error = max(maximum_error, float(error.abs().max()))
        squared_error += float(error.double().square().sum())
    del original_embedding
    assert embedding_digest.hexdigest() == policy['deployedEmbeddingF16Sha256']
    model.lm_head.weight = embedding
    assert model.lm_head.weight is model.model.embed_tokens.weight
    report['intervention'] = {
        'policySha256': hashlib.sha256(Path(args.policy).read_bytes()).hexdigest(),
        'toolSha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'kind': 'reuse verified deployed F16 embedding as tied output head',
        'deployedEmbeddingUnchangedSha256': embedding_digest.hexdigest(),
        'deployedEmbeddingSourceMismatchCount': embedding_mismatches,
        'deployedEmbeddingSourceMaximumError': embedding_maximum_error,
        'displacedQ4HeadMaximumError': maximum_error,
        'displacedQ4HeadRootMeanSquaredError': (squared_error / embedding.numel()) ** 0.5,
        'activeRuntimeOrModelChanged': False,
    }
    return replay(model, config, manifest, options, report)


reference.replay_generation = tied_replay
options = dict(policy['referenceArguments'])
options['out'] = args.out
reference.run(SimpleNamespace(**options))
