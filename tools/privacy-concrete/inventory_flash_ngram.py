"""Acquire bounded public metadata to reproduce the installed Flash ngram inventory.

Never downloads table weights or accepts private token/row indices. Output is
reviewable source metadata, not permission to enable a runtime backend.
"""
import argparse
import ast
import json
from pathlib import Path
from flash_source_scopes import BASE
from flash_inventory_metadata import public_bytes,model_metadata,shard_headers

PREFIX='model.language_model.layers.1.ple.ple_embedding.ngram_embedding.'
CONFIG_KEYS=('vocab_size','ngram_size','heads_per_ngram','ngram_vocab_size_base','seed',
             'eos_token_id','make_ngram_vocab_size_divisible_by','ple_embed_dim','split_ngram_parts')


def build():
    metadata=model_metadata(BASE)
    config=metadata['config.json']['text_config'];index=metadata['model.safetensors.index.json']
    names={key:value for key,value in index['weight_map'].items() if key.startswith(PREFIX)}
    count=config['split_ngram_parts']
    if type(count) is not int or count!=128 or set(names)!={PREFIX+f'shard_{i}.weight' for i in range(count)}:
        raise ValueError('Complete installed Flash table partition set required')
    files=sorted(set(names.values()))
    headers=shard_headers(BASE,files)
    records={name:value[0] for name,value in headers.items()}
    parts=[];row=0
    for i in range(count):
        name=PREFIX+f'shard_{i}.weight';file=names[name];value=headers[file][1][name]
        rows,columns=value['shape'];begin,end=value['data_offsets']
        if (value['dtype']!='BF16' or columns!=160 or type(rows) is not int or rows<=0 or end-begin!=rows*columns*2):
            raise ValueError('Installed BF16 table geometry required')
        parts.append(dict(name=name,shard=file,firstRow=row,rows=rows,columns=columns,dtype='BF16',dataOffsets=[begin,end]));row+=rows
    installed=json.loads(Path(__file__).with_name('flash-ngram-storage.json').read_text())
    reference=installed['conversionReference'];raw_sources={}
    for name,record in reference['files'].items():raw_sources[name]=public_bytes(record['url'],1024*1024,record['sha256'])
    defaults={n.target.id:ast.literal_eval(n.value) for node in ast.parse(raw_sources['configuration_qwen4_exp.py']).body
              if isinstance(node,ast.ClassDef) and node.name=='Qwen4ExpTextConfig' for n in node.body
              if isinstance(n,ast.AnnAssign) and isinstance(n.target,ast.Name) and isinstance(n.value,ast.Constant)}
    return dict(schema='doppler.flash-local-ngram-storage-inventory/v1',repository=BASE['repository'],revision=BASE['revision'],
                sourceConfigSha256=BASE['configSha256'],sourceIndexSha256=BASE['files']['model.safetensors.index.json']['sha256'],
                logicalShape=[row,160],logicalBytes=row*160*2,shards={name:headers[name][0] for name in files},parts=parts,
                wholeShardBytes=sum(item['bytes'] for item in records.values()),weightsProvisioned=False,productionBackendAvailable=False,securityQualified=False,
                conversionReference=reference,resolvedNgramDefaults={key:defaults[key] for key in CONFIG_KEYS if key not in config},
                ngramConfig={key:config[key] if key in config else defaults[key] for key in CONFIG_KEYS})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output');parser.add_argument('--check',action='store_true');args=parser.parse_args()
    if bool(args.output)==args.check:parser.error('Choose --output or --check')
    result=build()
    if args.check:
        current=json.loads(Path(__file__).with_name('flash-ngram-storage.json').read_text())
        if current!=result:raise ValueError('Installed ngram storage metadata is stale; review regenerated output')
    else:Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
