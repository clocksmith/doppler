"""Provision whole immutable Flash shards; partial/range data never becomes admitted weight material."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import tempfile
import urllib.request
from protocol import parse_unique_json
from flash_source_scopes import source_spec,SCOPES,EXPERT_TENSORS
from source_snapshot import file_signature

def verify_blob(path,record):
    path=Path(path)
    if not path.is_file() or path.stat().st_size!=record['bytes']:
        raise ValueError('Pinned complete public model shard required')
    with path.open('rb') as handle:actual=hashlib.file_digest(handle,'sha256').hexdigest()
    if actual!=record['sha256']:raise ValueError('Public model shard checksum mismatch')


def fetch_blob(url,path,record):
    path=Path(path)
    if path.exists():verify_blob(path,record);return
    partial=None
    try:
        request=urllib.request.Request(url,headers={'Accept-Encoding':'identity'})
        with urllib.request.urlopen(request,timeout=60) as response:
            if response.status!=200 or response.headers.get('Content-Range') is not None or not response.geturl().startswith('https://'):
                raise ValueError('Complete HTTPS model object required; range data is not admitted')
            with tempfile.NamedTemporaryFile(dir=path.parent,prefix=path.name+'.',suffix='.partial',delete=False) as handle:
                partial=Path(handle.name);size=0;hasher=hashlib.sha256()
                while True:
                    chunk=response.read(1024*1024)
                    if not chunk:break
                    size+=len(chunk)
                    if size>record['bytes']:raise ValueError('Oversized public model object')
                    hasher.update(chunk);handle.write(chunk)
            if size!=record['bytes'] or hasher.hexdigest()!=record['sha256']:
                raise ValueError('Public model shard checksum mismatch')
            if path.exists():raise ValueError('Public model cache changed during acquisition')
            partial.replace(path);partial=None
    finally:
        if partial is not None:partial.unlink(missing_ok=True)


def provision(directory,scope='moe'):
    specification=source_spec(scope)
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    artifacts=dict(specification['files'],**specification['shards'])
    missing=sum(item['bytes'] for name,item in artifacts.items() if not (directory/name).exists())
    if shutil.disk_usage(directory).free<missing+64*1024*1024:
        raise ValueError('Insufficient space for the complete pinned public shards')
    base='https://huggingface.co/'+specification['repository']+'/resolve/'+specification['revision']+'/'
    for name,record in artifacts.items():fetch_blob(base+name+'?download=true',directory/name,record)


class FlashSource:
    def __init__(self,directory,scope='moe'):
        self.spec=source_spec(scope)
        self.directory=Path(directory).resolve();self.signatures={};self.headers={}
        for name,record in self.spec['files'].items():verify_blob(self.directory/name,record)
        self.config=parse_unique_json((self.directory/'config.json').read_bytes())
        index=parse_unique_json((self.directory/'model.safetensors.index.json').read_bytes())
        if any(index['weight_map'].get(name)!=item['shard'] for name,item in self.spec['tensors'].items()):
            raise ValueError('Pinned source index does not map the declared tensors')
        for name,record in self.spec['shards'].items():
            path=self.directory/name
            before=file_signature(path);verify_blob(path,record)
            if file_signature(path)!=before:raise ValueError('Model source changed during verification')
            with path.open('rb') as handle:
                length=struct.unpack('<Q',handle.read(8))[0]
                if not 0<length<=16*1024*1024:raise ValueError('Bounded safetensors header required')
                raw=handle.read(length)
                if len(raw)!=length:raise ValueError('Complete safetensors header required')
                self.headers[name]=parse_unique_json(raw)
            self.signatures[name]=before
        for name,expected in self.spec['tensors'].items():
            actual=self.headers[expected['shard']].get(name)
            if (not isinstance(actual,dict) or actual.get('shape')!=expected['shape'] or actual.get('dtype')!=expected['dtype']):
                raise ValueError('Pinned public tensor geometry/dtype required')

    def tensor(self,name,expert=None):
        import torch
        from safetensors import safe_open
        expected=self.spec['tensors'][name];path=self.directory/expected['shard']
        if file_signature(path)!=self.signatures[expected['shard']]:raise ValueError('Verified model source changed')
        if name in EXPERT_TENSORS and expert is None:raise ValueError('Explicit streamed expert required')
        if expert is not None and name not in EXPERT_TENSORS:raise ValueError('Expert slicing is unavailable for this tensor')
        with safe_open(path,framework='pt',device='cpu') as source:
            value=source.get_slice(name)
            if value.get_shape()!=expected['shape']:raise ValueError('Pinned tensor geometry required')
            if expert is not None:
                if type(expert) is not int or not 0<=expert<expected['shape'][0]:raise ValueError('Invalid public expert component')
                result=value[expert]
            else:result=source.get_tensor(name)
            expected_dtype={'BF16':torch.bfloat16,'I64':torch.int64}.get(expected['dtype'])
            if expected_dtype is None or result.dtype!=expected_dtype:raise ValueError('Pinned BF16 or int64 tensor required')
            result=result.to(dtype=torch.float32) if expected_dtype==torch.bfloat16 else result.clone()
            if not torch.isfinite(result).all():raise ValueError('Finite pinned public tensor required')
            if file_signature(path)!=self.signatures[expected['shard']]:raise ValueError('Verified model source changed during tensor read')
            return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--destination',required=True)
    parser.add_argument('--scope',choices=SCOPES,default='moe');args=parser.parse_args()
    provision(args.destination,args.scope);specification=source_spec(args.scope)
    print(json.dumps(dict(schema=specification['schema'],scope=args.scope,
                          completeShardBytes=sum(item['bytes'] for item in specification['shards'].values()),verified=True)))
