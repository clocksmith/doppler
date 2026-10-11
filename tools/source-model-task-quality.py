#!/usr/bin/env python3
"""Screen one frozen source-model task suite before deployment preparation."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from lib.source_task_quality import evaluate_tasks


def digest(path):
    result = hashlib.sha256()
    with open(path, 'rb') as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


def run(args):
    torch.set_num_threads(args.threads)
    source = Path(args.source)
    suite = json.loads(Path(args.suite).read_text())
    receipt = json.loads((source / 'download-receipt.json').read_text())
    assert receipt['source'] == suite['modelId'] and receipt['revision'] == suite['sourceRevision']
    for entry in receipt['files']:
        path = source / entry['path']
        assert path.stat().st_size == entry['bytes'] and digest(path) == entry['sha256']
        assert f"/resolve/{receipt['revision']}/" in entry['url']
        if path.suffix == '.safetensors':
            assert entry['expectedLfsSha256'] == entry['sha256']
    config = AutoConfig.from_pretrained(source, local_files_only=True)
    assert config.max_position_embeddings >= suite['generation']['maxSeqLen']
    model = AutoModelForCausalLM.from_pretrained(source, config=config, local_files_only=True,
                                               dtype=torch.float32, attn_implementation='eager')
    model.eval()
    tokenizer = Tokenizer.from_file(str(source / 'tokenizer.json'))
    source_tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    report = {'schema': 'doppler.original-source-task-quality-result/v1', 'qualified': False,
              'source': receipt, 'scope': suite['scope'], 'suiteSha256': digest(args.suite),
              'producerSha256': digest(__file__),
              'evaluatorSha256': digest(Path(__file__).parent / 'lib/source_task_quality.py'),
              'packages': {name: importlib.metadata.version(name)
                           for name in ['torch', 'transformers', 'tokenizers']},
              'arithmetic': 'Original pretrained weights represented exactly in F32; eager source equations; stored F16 K/V',
              'settings': suite['generation'], 'eosTokenId': config.eos_token_id,
              'padTokenId': config.pad_token_id, 'results': []}
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report) + '\n')
    evaluate_tasks(model, config, tokenizer, source_tokenizer, suite, report, output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ['source', 'suite', 'out']:
        parser.add_argument('--' + flag, required=True)
    parser.add_argument('--threads', type=int, required=True)
    run(parser.parse_args())
