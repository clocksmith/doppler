#!/usr/bin/env python3
"""Recreate one frozen installed-candidate lane; never downloads model weights."""
import argparse
import hashlib
import json
import pathlib
import shutil
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--host', choices=['metal', 'vulkan'], required=True)
parser.add_argument('--surface', choices=['node', 'browser'], required=True)
parser.add_argument('--size', choices=['08', '2b'], required=True)
parser.add_argument('--models', type=pathlib.Path, required=True)
parser.add_argument('--output', type=pathlib.Path, required=True)
parser.add_argument('--browser', type=pathlib.Path)
parser.add_argument('--prepare-only', action='store_true')
args = parser.parse_args()
report = pathlib.Path(__file__).resolve().parent
root = report.parents[1]
receipt = json.loads((report / 'package-receipt.json').read_text())
archive = report / receipt['package']['filename']
assert hashlib.sha256(archive.read_bytes()).hexdigest() == receipt['package']['sha256']
suffix = 'ranges' if args.surface == 'browser' else 'final'
name = f'{args.host}-{args.surface}-{args.size}-{suffix}'
config = json.loads((report / (name + '-config.json')).read_text())
model = args.models.resolve() / pathlib.Path(config['modelDir']).name
assert 'sha256:' + hashlib.sha256((model / 'manifest.json').read_bytes()).hexdigest() == config['manifestDigest']
reference = report / (args.size + '-reference.json')
assert 'sha256:' + hashlib.sha256(reference.read_bytes()).hexdigest() == config['reference']['digest']
output = args.output.resolve()
output.mkdir(parents=True, exist_ok=False)
bundle = output / 'installed'
consumer = bundle / 'consumer'
consumer.mkdir(parents=True)
shutil.copyfile(archive, bundle / archive.name)
# This is the retained archive's packaging receipt, not a new smoke-test claim.
shutil.copyfile(report / 'package-receipt.json', bundle / 'receipt.json')
(consumer / 'package.json').write_text('{"private":true,"type":"module"}\n')
subprocess.run(['npm', 'install', '--offline', '--ignore-scripts', '--no-audit', '--no-fund',
                str(bundle / archive.name)], cwd=consumer, check=True)
if args.surface == 'node':
    provider = root / 'node_modules/webgpu'
    if not provider.is_dir():
        raise RuntimeError('Provision the source lockfile WebGPU provider before Node qualification')
    installed_provider = consumer / 'node_modules/webgpu'
    if installed_provider.exists():
        expected = json.loads((provider / 'package.json').read_text())['version']
        actual = json.loads((installed_provider / 'package.json').read_text())['version']
        if actual != expected:
            raise RuntimeError(f'WebGPU provider version drift: {actual} != {expected}')
    else:
        installed_provider.symlink_to(provider.resolve(), target_is_directory=True)
config.update(packageBundlePath=str(bundle), modelDir=str(model), outputDir=str(output / 'result'))
config['reference']['path'] = str(reference)
if args.browser:
    config['browserExecutable'] = str(args.browser.resolve())
config_path = output / 'config.json'
config_path.write_text(json.dumps(config, indent=2) + '\n')
print('Prepared exact archive, manifest and reference:', config_path)
if not args.prepare_only:
    subprocess.run(['node', 'tools/qualify-installed-generation.js', str(config_path)],
                   cwd=root, check=True, timeout=660)
