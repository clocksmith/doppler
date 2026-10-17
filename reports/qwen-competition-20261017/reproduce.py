"""Reconstruct the measured source and run one pinned, provisioned comparison."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--size', choices=['08', '2b'], required=True)
parser.add_argument('--model-root', type=Path, required=True)
parser.add_argument('--tjs-root', type=Path, required=True)
parser.add_argument('--browser', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--cold', action='store_true')
parser.add_argument('--prepare-only', action='store_true')
args = parser.parse_args()
report = Path(__file__).resolve().parent
repo = report.parents[1]
output = args.output.resolve()
output.mkdir(parents=True, exist_ok=False)
source = output / 'source'
source.mkdir()
source_map = json.load(gzip.open(report / 'source-files.json.gz', 'rt'))
with tempfile.TemporaryFile() as archive:
    subprocess.run(['git', 'archive', source_map['baseCommit'], 'src', 'tools', 'tests',
                    'benchmarks', 'models', 'package.json', 'package-lock.json'],
                   cwd=repo, stdout=archive, check=True)
    archive.seek(0)
    with tarfile.open(fileobj=archive) as bundle:
        bundle.extractall(source, filter='data')
with tarfile.open(report / 'source-overrides.tar.gz') as bundle:
    bundle.extractall(source, filter='data')
for row in source_map['files']:
    actual = hashlib.sha256((source / row['path']).read_bytes()).hexdigest()
    if actual != row['sha256']:
        raise ValueError(f"Source mismatch: {row['path']}")
# The recorded runner uses these installed dependencies. Never silently install
# or upgrade them as part of a measurement.
deps = repo / 'node_modules'
expected = {
    '@huggingface/transformers': '4.2.0',
    '@huggingface/transformers/node_modules/onnxruntime-web': '1.26.0-dev.20260416-b7804b056c',
}
for package, version in expected.items():
    if json.loads((deps / package / 'package.json').read_text())['version'] != version:
        raise ValueError(f'Provision the pinned dependency: {package}@{version}')
(source / 'node_modules').symlink_to(deps, target_is_directory=True)
pins = json.loads((report / 'doppler-model-pins.json').read_text())[args.size]
model = args.model_root.resolve() / pins['modelId']
if hashlib.sha256((model / 'manifest.json').read_bytes()).hexdigest() != pins['manifestSha256']:
    raise ValueError('Doppler manifest differs from the receipt')
for shard in pins['shards']:
    if (model / shard['filename']).stat().st_size != shard['size']:
        raise ValueError(f"Truncated shard: {shard['filename']}")
# Runtime retains its own signed shard validation; source tokenizers use the
# provisioned artifacts rather than Git LFS pointer text from git archive.
for entry in model.iterdir():
    if entry.is_file() and entry.suffix != '.bin':
        target = source / 'models/local' / pins['modelId'] / entry.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            target.unlink()
        target.symlink_to(entry)
if args.size == '2b':
    fixture = source / 'benchmarks/vendors/fixtures/qwen-3-5-2b-reference/tokenizer.json'
    fixture.unlink(missing_ok=True)
    fixture.symlink_to(model / 'tokenizer.json')
for pin in json.loads((report / 'comparator-pins.json').read_text()):
    if pin['key'] != 'tjs-' + args.size:
        continue
    for item in pin['files']:
        file = args.tjs_root.resolve() / pin['repo'] / item['path']
        if file.stat().st_size != item['bytes']:
            raise ValueError(f'Comparator size mismatch: {file}')
        if item['sha256']:
            digest = hashlib.sha256()
            with file.open('rb') as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    digest.update(block)
            if digest.hexdigest() != item['sha256']:
                raise ValueError(f'Comparator hash mismatch: {file}')
argv = ['node', str(source / 'tools/compare-engines.js'), '--model-id', pins['modelId'],
        '--model-url', model.as_uri(), '--tjs-version', '4', '--tjs-dtype', 'q4f16',
        '--tjs-local-model-path', str(args.tjs_root.resolve()), '--prefill-tokens', '64',
        '--max-tokens', '128', '--temperature', '0', '--top-k', '1', '--top-p', '1',
        '--seed', '0', '--warmup', '0' if args.cold else '1', '--runs', '1' if args.cold else '3',
        '--mode', 'cold' if args.cold else 'compute', '--decode-profile', 'parity',
        '--doppler-surface', 'browser', '--load-mode', 'http', '--browser-executable',
        str(args.browser.resolve()), '--resource-telemetry', 'on', '--resource-telemetry-samples',
        '--save', '--save-dir', str(output / 'receipts'), '--skip-matrix-update',
        '--timeout-ms', '240000', '--json']
(output / 'command.json').write_text(json.dumps(argv, indent=2) + '\n')
if args.prepare_only:
    print('Source and provisioned comparator checks passed; command saved; no GPU run.')
else:
    with (output / 'run.log').open('w') as log:
        subprocess.run(argv, cwd=repo, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=900)
