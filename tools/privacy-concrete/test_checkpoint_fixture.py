"""Small actual BF16 files for public checkpoint reconstruction tests only."""
import hashlib
import json
from pathlib import Path


def checkpoint_fixture(directory,matrices):
    import torch
    from safetensors.torch import save_file
    root=Path(directory);shard=root/'public.safetensors'
    tensors={name+'.weight':value.detach().to(torch.bfloat16).contiguous() for name,value in matrices.items()}
    save_file(tensors,str(shard))
    (root/'config.json').write_text('{}')
    (root/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{name:shard.name for name in tensors}}))
    def record(path):return dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    spec=dict(schema='synthetic-public-linear-source/v1',files={name:record(root/name) for name in ('config.json','model.safetensors.index.json')},
        shards={shard.name:record(shard)},tensors={name:dict(shard=shard.name,shape=list(value.shape),dtype='BF16') for name,value in tensors.items()},
        wholeShardBytes=shard.stat().st_size)
    spec['wholeSourceBytes']=spec['wholeShardBytes']+sum(row['bytes'] for row in spec['files'].values())
    bindings={name:dict(tensor=name+'.weight',expert=None,rows=None,inputWidth=value.shape[1],outputWidth=value.shape[0]) for name,value in matrices.items()}
    return spec,bindings
