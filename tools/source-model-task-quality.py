#!/usr/bin/env python3
"""Screen a frozen source-equation suite using original or exact deployed weights."""
import argparse
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from lib.source_task_quality import evaluate_tasks, probe_retained_prefix, write_quality_report


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
    original_head = model.get_output_embeddings().weight.detach().clone() if args.head_probe_contract else None
    tokenizer = Tokenizer.from_file(str(source / 'tokenizer.json'))
    deployed = None
    if args.deployed is not None:
        from lib.deployed_source_weights import load_deployed_weights
        assert args.probe_report is None, 'Original retained-prefix replay must retain original weights.'
        assert 'deployed' in suite, 'Freeze the deployed manifest identity in the suite before execution.'
        deployed = load_deployed_weights(model, args.deployed, suite['deployed'], receipt)
        deployed_tokenizer = Path(args.deployed) / 'tokenizer.json'
        assert json.loads(deployed_tokenizer.read_text()) == json.loads((source / 'tokenizer.json').read_text())
        tokenizer = Tokenizer.from_file(str(deployed_tokenizer))
        deployed['tokenizerSha256'] = digest(deployed_tokenizer)
        deployed['loaderSha256'] = digest(Path(__file__).parent / 'lib/deployed_source_weights.py')
    else:
        assert 'deployed' not in suite, 'A deployed suite requires --deployed.'
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
    if deployed is not None:
        report.update({'schema': 'doppler.deployed-source-task-quality-result/v1', 'deployed': deployed,
                       'arithmetic': 'Independently decoded exact deployed weights in F32; eager source equations; stored F16 K/V'})
        report['packages']['gguf'] = importlib.metadata.version('gguf')
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_quality_report(output, report)
    if args.head_probe_contract is not None:
        from lib.deployed_head_attribution import probe_deployed_head
        assert args.deployed is not None and args.probe_report is None
        task = json.loads(Path(args.head_probe_contract).read_text())
        baseline_path = Path(task['retainedReport'])
        assert digest(baseline_path) == task['retainedReportFileSha256']
        data = baseline_path.read_bytes()
        baseline = json.loads(gzip.decompress(data) if data[:2] == b'\x1f\x8b' else data)
        assert baseline['suiteSha256'] == digest(args.suite)
        report['attributionContractSha256'] = digest(args.head_probe_contract)
        probe_deployed_head(model, original_head, config, tokenizer, baseline, task, report, output)
    elif args.probe_report is not None:
        assert args.probe_case is not None and args.probe_step is not None
        data = Path(args.probe_report).read_bytes()
        baseline = json.loads(gzip.decompress(data) if data[:2] == b'\x1f\x8b' else data)
        report['retainedReportSha256'] = digest(args.probe_report)
        probe_retained_prefix(model, config, tokenizer, baseline, args.probe_case,
                              args.probe_step, report, output)
    else:
        assert args.probe_case is None and args.probe_step is None
        evaluate_tasks(model, config, tokenizer, source_tokenizer, suite, report, output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ['source', 'suite', 'out']:
        parser.add_argument('--' + flag, required=True)
    parser.add_argument('--threads', type=int, required=True)
    parser.add_argument('--probe-report')
    parser.add_argument('--probe-case')
    parser.add_argument('--probe-step', type=int)
    parser.add_argument('--deployed', help='Exact RDRR artifact bound by the frozen suite, for offline source equations.')
    parser.add_argument('--head-probe-contract', help='Frozen same-hidden-state original/deployed head attribution; never a quality pass.')
    run(parser.parse_args())
