"""Complete actual-checkpoint geometry assembly; no weights, forward pass or backend.

Builds all 48 inspected decoders on Torch's meta device and checks every actual
text tensor shape, including the logical concatenated PLE table. This validates
materialization destinations before the separate, uncompleted weight loader.
"""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from torch import nn
from flash_reference import load_reference
from flash_text_config import resolve_text_config
from full_plan import canonical
from protocol import digest

ROOT=Path(__file__).parent
TEXT=json.loads((ROOT/'flash-text-weights.json').read_text())
TABLE=json.loads((ROOT/'flash-ngram-storage.json').read_text())


def assemble_geometry(config_path,reference_path):
    raw=Path(config_path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=TEXT['configSha256']:raise ValueError('Pinned whole Flash checkpoint configuration required')
    config=resolve_text_config(json.loads(raw)['text_config'],Path(reference_path).with_name('configuration_qwen4_exp.py').read_bytes(),attention_implementation='eager')
    source=load_reference(reference_path,'decoder')
    with torch.device('meta'):
        root=nn.Module();root.model=nn.Module();root.model.language_model=nn.Module();client=root.model.language_model
        client.embed_tokens=nn.Embedding(config.vocab_size,config.hidden_size)
        client.layers=nn.ModuleList([source['Qwen4ExpTextDecoderLayer'](config,i) for i in range(config.num_hidden_layers)])
        client.hyper_connection_mixer=source['Qwen4ExpTextGatedResidual'](config,use_combine=False)
        root.lm_head=nn.Linear(config.hidden_size,config.vocab_size,bias=False)
    expected={name:row['shape'] for name,row in TEXT['tensors'].items()}
    parts=[part['name'] for part in TABLE['parts']]
    if any(name not in expected for name in parts):raise ValueError('Complete checkpoint PLE parts required')
    for name in parts:del expected[name]
    prefix=parts[0].rsplit('.',2)[0]
    expected[prefix+'.weight']=TABLE['logicalShape']
    actual={name:list(value.shape) for name,value in root.state_dict().items()}
    if actual!=expected:raise ValueError('Full checkpoint/client materialization geometry differs')
    if any(value.device.type!='meta' for value in root.state_dict().values()):raise ValueError('Geometry assembly allocated model weights')
    report=dict(schema='doppler.actual-flash-client-geometry-assembly/v1',sourceId=digest(canonical(TEXT)),
        configurationId=digest(canonical(vars(config))),sourceRevision=TEXT['revision'],layerCount=len(client.layers),
        linearAttentionLayers=config.layer_types.count('linear_attention'),indexedAttentionLayers=config.layer_types.count('indexed_attention'),
        checkpointTensors=len(TEXT['tensors']),assembledStateEntries=len(actual),allTensorGeometryMatched=True,
        packedExpertBanks=sum(name.endswith(('.experts.gate_up_proj','.experts.down_proj')) for name in actual),
        localTableShape=TABLE['logicalShape'],geometryOnly=True,weightsMaterialized=False,forwardExecuted=False,
        fullFlashModelExecuted=False,encryptedExecution=False,productionBackendAvailable=False,securityQualified=False)
    return root,config,report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--reference-source',required=True)
    args=parser.parse_args();_,_,report=assemble_geometry(args.config,args.reference_source)
    print(json.dumps(report,indent=2))
