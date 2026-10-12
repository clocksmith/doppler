"""Paired input-layout control using unchanged verified deployed tensors."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--policy', required=True)
parser.add_argument('--suite', required=True)
parser.add_argument('--out', required=True)
args = parser.parse_args()
policy = json.loads(Path(args.policy).read_text())
suite = json.loads(Path(args.suite).read_text())
assert policy['schema'] == 'doppler.deployed-layout-control/v1'
assert hashlib.sha256(Path(args.suite).read_bytes()).hexdigest() in policy['allowedSuiteHashes']
sys.path.insert(0, str(Path('tools').resolve()))
reference_path = Path('tools/qwen35-deployed-reference.py')
spec = importlib.util.spec_from_file_location('deployed_reference', reference_path)
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


def evaluate_loaded_model(model, config, manifest, options, report):
    from transformers import AutoTokenizer
    from tokenizers import Tokenizer
    from lib.source_task_quality import evaluate_tasks
    assert suite['modelIdentity'] == report['modelIdentity']
    assert suite['generation']['suppressTokenIds'] == []
    assert model.lm_head.weight is not model.model.embed_tokens.weight
    root = Path(policy['sourceTokenizer'])
    assert hashlib.sha256((root / 'tokenizer.json').read_bytes()).hexdigest() == policy['sourceTokenizerSha256']
    assert hashlib.sha256((root / 'chat_template.jinja').read_bytes()).hexdigest() == policy['sourceTemplateSha256']
    tokenizer = Tokenizer.from_file(str(Path(options.model) / 'tokenizer.json'))
    source_tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
    report.update({'schema': 'doppler.deployed-layout-control/v1', 'qualified': False,
                   'layoutPolicy': suite['layoutPolicy'], 'settings': suite['generation'],
                   'suiteSha256': hashlib.sha256(Path(args.suite).read_bytes()).hexdigest(),
                   'layoutToolSha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   'evaluatorSha256': hashlib.sha256(Path('tools/lib/source_task_quality.py').read_bytes()).hexdigest(),
                   'scope': 'Only input layout differs; exact verified deployed tensor values and independently quantized head unchanged.',
                   'results': []})
    evaluate_tasks(model, config, tokenizer, source_tokenizer, suite, report, Path(args.out))
    return report


reference.replay_generation = evaluate_loaded_model
options = dict(policy['referenceArguments'])
options['out'] = args.out
reference.run(SimpleNamespace(**options))
