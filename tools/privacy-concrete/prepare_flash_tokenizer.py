"""Provision only whole pinned public tokenizer/configuration files; no model weights or source installation."""
import argparse
import json
from pathlib import Path
import shutil
from flash_tokenizer import SOURCE
from flash_weights import fetch_blob


def provision(directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    missing=sum(value['bytes'] for name,value in SOURCE['files'].items() if not (directory/name).exists())
    if shutil.disk_usage(directory).free<missing:raise ValueError('Insufficient storage for complete public tokenizer artifacts')
    origin='https://huggingface.co/'+SOURCE['repository']+'/resolve/'+SOURCE['revision']+'/'
    for name,record in SOURCE['files'].items():fetch_blob(origin+name+'?download=true',directory/name,record)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--destination',required=True);args=parser.parse_args();provision(args.destination)
    print(json.dumps(dict(schema=SOURCE['schema'],repository=SOURCE['repository'],revision=SOURCE['revision'],verified=True,modelWeightsAcquired=False,productionBackendAvailable=False)))
