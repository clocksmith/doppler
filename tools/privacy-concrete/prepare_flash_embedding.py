"""Explicit whole public input-embedding shard acquisition; no row-fetch service or private input."""
import argparse
import json
from pathlib import Path
import shutil
from local_token_rows import installed_inventory
from flash_weights import fetch_blob
from flash_source_scopes import TEXT


def provision(directory,local_storage_limit_bytes):
    spec=installed_inventory()
    if type(local_storage_limit_bytes) is not int or local_storage_limit_bytes<spec['wholeShardBytes']:raise ValueError('Explicit sufficient complete-file embedding budget required')
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    missing=sum(v['bytes'] for name,v in spec['shards'].items() if not (directory/name).exists())
    if shutil.disk_usage(directory).free<missing:raise ValueError('Insufficient storage for complete embedding object')
    base='https://huggingface.co/'+TEXT['repository']+'/resolve/'+TEXT['revision']+'/'
    for name,value in spec['shards'].items():fetch_blob(base+name+'?download=true',directory/name,value)
    return spec


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--destination',required=True);parser.add_argument('--local-storage-limit-bytes',type=int,required=True);args=parser.parse_args()
    spec=provision(args.destination,args.local_storage_limit_bytes)
    print(json.dumps(dict(schema=spec['schema'],wholeShardBytes=spec['wholeShardBytes'],verified=True,fullModelExecuted=False,productionBackendAvailable=False)))
