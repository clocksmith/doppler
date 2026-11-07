"""Separate local HC arithmetic from propagated inputs at retained positions.

Usage: python3 check-hc-stages.py GPU CPU ARTIFACT SOURCE_DATA OUT
Unchanged source operators only, at original batch geometry; no GPU feedback.
"""
import ast
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'flash-next-qualification-20261106'))
from verified_tensor_io import read_tensor


def require(condition, message):
    if not condition:
        raise ValueError(message)


gpu_path, cpu_path, artifact, data, output = map(Path,sys.argv[1:])
sha = lambda raw: hashlib.sha256(raw).hexdigest()
gpu_bytes, cpu_bytes, manifest_bytes = [p.read_bytes() for p in [gpu_path,cpu_path,artifact/'manifest.json']]
gpu, cpu, manifest = map(json.loads,[gpu_bytes,cpu_bytes,manifest_bytes])
require(gpu['diagnosticCompleted'] and cpu['diagnosticCompleted'] and cpu['sourceRouterInputReplayBitExact'], 'Incomplete or unbound source trace')
require(not gpu['gpuErrors'] and not gpu['cleanupFailures'], 'Failed GPU observation')
require(gpu['manifestSha256'] == cpu['manifestSha256'] == sha(manifest_bytes), 'Different artifact')
require(gpu['inputsSha256'] == cpu['inputsSha256'], 'Different input')
layer = cpu['layer']
geometry = manifest['inference']['hyperConnections']
source_bytes = (data/'source/modeling_qwen4_exp.py').read_bytes()
binding = json.loads((data/'source-binding.json').read_bytes())
entry = next(r for r in binding['referenceImplementation']['files'] if r['path'] == 'modeling_qwen4_exp.py')
require(sha(source_bytes) == entry['sha256'], 'Source implementation drift')
node = next(n for n in ast.parse(source_bytes).body if isinstance(n, ast.ClassDef) and n.name == 'Qwen4ExpTextRMSNorm')
namespace = dict(torch=torch, nn=torch.nn)
exec(compile(ast.Module(body=[node],type_ignores=[]),'pinned-grouped-rmsnorm','exec'),namespace)
torch.set_num_threads(16)


def read(trace, role, stage, parent=None):
    r = next(r for r in trace['tensors'] if r['role'] == role and r['stage'] == stage)
    return read_tensor(r,parent),r


def weight(name):
    d = manifest['tensors'][name]
    require(d['dtype'] == 'F16', 'Expected preserved F16 weight storage')
    raw = bytearray()
    for span in d.get('spans') or [dict(shardIndex=d['shard'],offset=d['offset'],size=d['size'])]:
        with (artifact/manifest['shards'][span['shardIndex']]['filename']).open('rb') as handle:
            handle.seek(span['offset']);part=handle.read(span['size'])
            require(len(part)==span['size'],'Incomplete weight range')
            raw.extend(part)
    return torch.from_numpy(np.frombuffer(raw,'<f2').astype('f4').reshape(d['shape']))


def measure(actual, expected):
    delta = actual.astype('f8')-expected.astype('f8')
    return dict(maxAbsolute=float(np.abs(delta).max()),rms=float(np.sqrt(np.mean(delta*delta))))


rows = []
with torch.no_grad():
    for role,prefix in [('attn_hc','attn_hyper_connection'),('mlp_hc','mlp_hyper_connection')]:
        gx,record=read(gpu,role,'input')
        positions=record['positions']; batch_rows=record['originalShape'][0]
        width=geometry['streamCount']*geometry['streamWidth']
        norm=namespace['Qwen4ExpTextRMSNorm'](width,geometry['streamWidth'],geometry['normEps'])
        norm.weight.copy_(weight(f'model.language_model.layers.{layer}.{prefix}.hc_norm.weight'))
        operations=[('norm','input',norm),('down','norm',None),('up','lowRank',None),('logits','norm',None)]
        for stage,input_stage,operation in operations:
            if operation is None:
                suffix=dict(down='input_mix_weight_down',up='input_mix_weight_up',logits='block_inject_weight')[stage]
                w=weight(f'model.language_model.layers.{layer}.{prefix}.{suffix}.weight')
                operation=lambda x,w=w:F.linear(x,w)
            gi,_=read(gpu,role,input_stage); ci,_=read(cpu,role,input_stage,cpu_path.parent)
            go,_=read(gpu,role,stage); co,_=read(cpu,role,stage,cpu_path.parent)
            batch=torch.zeros((batch_rows,ci.shape[1]),dtype=torch.float32)
            batch[positions]=torch.from_numpy(ci.copy())
            replay=operation(batch)[positions].numpy().copy()
            require(np.array_equal(replay.view('u4'),co.view('u4')),f'CPU geometry does not replay: {role}/{stage}')
            batch[positions]=torch.from_numpy(gi.copy())
            same_input=operation(batch)[positions].numpy().copy()
            rows.append(dict(role=role,stage=stage,positions=positions,sourceReplayBitExact=True,
                total=measure(go,co),local=measure(go,same_input),propagated=measure(same_input,co)))
report=dict(scope='Identical-input HC controls, no tensor feedback or acceptance change',layer=layer,
            gpuSha256=sha(gpu_bytes),cpuSha256=sha(cpu_bytes),manifestSha256=sha(manifest_bytes),
            scriptSha256=sha(Path(__file__).read_bytes()),sourceSha256=sha(source_bytes),rows=rows)
output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps(rows,indent=2))
