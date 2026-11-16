"""Explicit public test-fixture preparation; inference itself never downloads model files."""
import argparse
from importlib.metadata import version
from pathlib import Path
from protocol import POLICY, digest
from profiles import load_profile


def prepare(destination, profile_name='tiny'):
    profile = load_profile(profile_name)
    if profile_name == 'tiny':
        for dependency in ('torch', 'transformers'):
            if version(dependency) != POLICY['dependencies'][dependency]:
                raise ValueError('Pinned preparation dependencies required')
    destination = Path(destination)
    if destination.exists():
        raise ValueError('Use a new fixture directory')
    if profile_name != 'tiny':
        import hashlib
        import shutil
        from huggingface_hub import snapshot_download
        specification = profile.specification()
        source = specification['source']
        downloaded = Path(snapshot_download(repo_id=source['repository'], revision=source['revision'], allow_patterns=list(source['files'])))
        for name, identity in source['files'].items():
            with (downloaded / name).open('rb') as handle:
                if hashlib.file_digest(handle, 'sha256').hexdigest() != identity:
                    raise ValueError('Prepared real-model artifact differs from its pinned identity')
        destination.mkdir()
        for name in source['files']:
            shutil.copyfile(downloaded / name, destination / name)
        return
    from transformers import AutoModelForCausalLM, AutoTokenizer
    source = POLICY['source']
    model = AutoModelForCausalLM.from_pretrained(source['repository'], revision=source['revision'], trust_remote_code=False)
    model.save_pretrained(destination, safe_serialization=True)
    if (digest((destination / 'model.safetensors').read_bytes()) != source['safetensorsSha256']
            or digest((destination / 'config.json').read_bytes()) != source['configSha256']):
        raise ValueError('Prepared fixture differs from the pinned artifact')
    tokenizer = AutoTokenizer.from_pretrained(source['repository'], revision=source['revision'], trust_remote_code=False)
    tokenizer.save_pretrained(destination)
    (destination / 'revision.txt').write_text(source['revision'] + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--destination', required=True)
    parser.add_argument('--profile', default='tiny', choices=['tiny', 'qwen05b-k-projection'])
    arguments = parser.parse_args()
    prepare(arguments.destination, arguments.profile)
