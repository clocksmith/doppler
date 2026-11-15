"""Explicit public test-fixture preparation; inference itself never downloads model files."""
import argparse
from importlib.metadata import version
from pathlib import Path
from protocol import POLICY, digest


def prepare(destination):
    for dependency in ('torch', 'transformers'):
        if version(dependency) != POLICY['dependencies'][dependency]:
            raise ValueError('Pinned preparation dependencies required')
    destination = Path(destination)
    if destination.exists():
        raise ValueError('Use a new fixture directory')
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
    prepare(parser.parse_args().destination)
