"""Verify native source/NTT pins before compilation and retain the public dependency lock."""
import argparse
import hashlib
import json
from pathlib import Path
import tomllib

ROOT=Path(__file__).resolve().parent
POLICY=json.loads((ROOT/'build-policy.json').read_text())


def check(path):
    value=tomllib.loads(Path(path).read_text())
    ntt=[package for package in value['package'] if package['name']=='tfhe-ntt' and package['version']==POLICY['nttCrate']['version']]
    if len(ntt)!=1 or ntt[0]['checksum']!=POLICY['nttCrate']['sha256']:raise ValueError('Exact NTT dependency checksum required')
    tfhe=[package for package in value['package'] if package['name']=='tfhe']
    if len(tfhe)!=1 or not tfhe[0]['source'].endswith('#1ec21a5e0b7c12165aa7e556c01e730c3117765a'):
        raise ValueError('Pinned underlying TFHE source required')
    print(json.dumps(dict(cargoLockSha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),nttCrateChecksum=ntt[0]['checksum'])))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('path');args=parser.parse_args();check(args.path)
