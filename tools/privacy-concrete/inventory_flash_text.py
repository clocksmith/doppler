"""Inventory complete immutable Flash text-generation weights without acquiring payloads.

Metadata enables later whole-file verification/materialization. It does not load a
model, execute encryption, qualify a backend or grant remote deployment permission.
"""
import argparse
import json
import math
from pathlib import Path
from flash_source_scopes import BASE
from flash_inventory_metadata import model_metadata,shard_headers
from flash_tokenizer import SOURCE as TOKENIZER
from full_plan import canonical
from protocol import digest

DTYPE_BYTES={'BF16':2,'F32':4,'I64':8}
MODEL=json.loads(Path(__file__).with_name('flash-readiness.json').read_text())['config']


def assemble(metadata,headers):
    index=metadata['model.safetensors.index.json'];config=metadata['config.json']['text_config'];count=MODEL['num_hidden_layers']
    names={name:file for name,file in index['weight_map'].items() if name.startswith('model.language_model.') or name=='lm_head.weight'}
    if (config['num_hidden_layers']!=count or config['num_experts']!=MODEL['num_experts'] or len(config['layer_types'])!=count
            or set(headers)!=set(names.values()) or 'lm_head.weight' not in names or 'model.language_model.embed_tokens.weight' not in names):
        raise ValueError('Complete installed Flash text model required')
    if TOKENIZER['repository']!=BASE['repository'] or TOKENIZER['revision']!=BASE['revision'] or TOKENIZER['files']['config.json']!=BASE['files']['config.json']:
        raise ValueError('Complete text weights and tokenizer must share the exact source configuration')
    tensors={};logical_bytes=0
    for name,file in sorted(names.items()):
        record,header=headers[file];value=header.get(name)
        if not isinstance(value,dict) or set(value)!={'dtype','shape','data_offsets'}:raise ValueError('Exact pinned tensor metadata required')
        shape=value['shape'];offsets=value['data_offsets'];dtype=value['dtype']
        if (dtype not in DTYPE_BYTES or not isinstance(shape,list) or not shape or any(type(n) is not int or n<=0 for n in shape)
                or not isinstance(offsets,list) or len(offsets)!=2 or any(type(n) is not int or n<0 for n in offsets)
                or offsets[1]-offsets[0]!=math.prod(shape)*DTYPE_BYTES[dtype] or offsets[1]>record['bytes']-record['headerBytes']-8):
            raise ValueError('Bounded public text tensor geometry required')
        tensors[name]=dict(shard=file,dtype=dtype,shape=shape,dataOffsets=offsets);logical_bytes+=offsets[1]-offsets[0]
    experts=[name for name in tensors if '.mlp.experts.' in name]
    if set(experts)!={f'model.language_model.layers.{i}.mlp.experts.{kind}' for i in range(count) for kind in ['gate_up_proj','down_proj']}:
        raise ValueError('All installed text expert banks required')
    if any(tensors[name]['shape'][0]!=TOKENIZER['vocabSize'] for name in ['lm_head.weight','model.language_model.embed_tokens.weight']):
        raise ValueError('Tokenizer vocabulary differs from actual embedding or output head')
    files=dict(BASE['files'],**TOKENIZER['files'])
    auxiliary_bytes=sum(value['bytes'] for value in files.values())
    return dict(schema='doppler.flash-complete-text-weights/v1',repository=BASE['repository'],revision=BASE['revision'],
                configSha256=BASE['configSha256'],indexSha256=BASE['indexSha256'],files=files,
                tokenizerSourceId=digest(canonical(TOKENIZER)),auxiliarySourceBytes=auxiliary_bytes,
                wholeSourceBytes=sum(v[0]['bytes'] for v in headers.values())+auxiliary_bytes,
                shards={name:headers[name][0] for name in sorted(headers)},tensors=tensors,
                tensorCount=len(tensors),logicalTensorBytes=logical_bytes,wholeShardBytes=sum(v[0]['bytes'] for v in headers.values()),
                layerCount=count,expertsPerLayer=MODEL['num_experts'],includesVision=False,includesMTP=False,
                weightsProvisioned=False,fullModelExecuted=False,productionBackendAvailable=False,securityQualified=False)


def build():
    metadata=model_metadata(BASE);names=metadata['model.safetensors.index.json']['weight_map']
    files={file for name,file in names.items() if name.startswith('model.language_model.') or name=='lm_head.weight'}
    return assemble(metadata,shard_headers(BASE,files))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output');parser.add_argument('--check',action='store_true');args=parser.parse_args()
    if bool(args.output)==args.check:parser.error('Choose --output or --check')
    result=build()
    if args.check:
        current=json.loads(Path(__file__).with_name('flash-text-weights.json').read_text())
        if current!=result:raise ValueError('Installed complete text inventory is stale; review regenerated output')
    else:Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
