"""Pinned source first-HC control. CPU-only fixture; never model qualification.

Replays the unchanged upstream class at the original batch geometry. Reads only
the embedding rows and first HC weights, never experts or n-gram tables.
"""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['artifact', 'capture', 'inputs', 'source', 'protocol', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--threads', type=int, required=True)
    args = parser.parse_args()
    require(1 <= args.threads <= 16, 'Explicit bounded CPU thread count required')
    protocol = json.loads(args.protocol.read_bytes())
    require(protocol['schema'] == 'doppler.hc-origin-witness/v1'
            and protocol['qualificationAuthority'] is False, 'Diagnostic protocol required')
    bindings = protocol['bindings']
    checked = {}

    def read_checked(file, key):
        raw = file.read_bytes()
        require(sha(raw) == bindings[key], 'Identity mismatch: ' + key)
        checked[key] = sha(raw)
        return raw

    manifest = json.loads(read_checked(args.artifact / 'manifest.json', 'manifestSha256'))
    capture = json.loads(read_checked(args.capture, 'captureSha256'))
    inputs = json.loads(read_checked(args.inputs, 'inputsSha256'))
    source = read_checked(args.source, 'sourceSha256')
    require(capture['diagnosticCompleted'] and not capture['gpuErrors']
            and not capture['cleanupFailures'], 'Incomplete GPU capture')
    require(capture['manifestSha256'] == checked['manifestSha256']
            and capture['inputsSha256'] == checked['inputsSha256'], 'Unbound capture')
    geometry = capture['hyperConnectionGeometry']
    count, width, rank = [geometry[k] for k in ['streamCount', 'streamWidth', 'projectionRank']]
    positions = protocol['positions']
    tokens = next(row['tokens'] for row in inputs['sequences'] if row['id'] == 'long')
    require(len(tokens) == protocol['originalRows'], 'Original source batch geometry changed')
    require(len(positions) == protocol['geometry']['rows'], 'Sample geometry changed')
    require(protocol['geometry'] == dict(rows=len(positions), streamCount=count,
            streamWidth=width, projectionRank=rank, normEps=geometry['normEps']), 'HC geometry drift')
    require(len(set(positions)) == len(positions)
            and all(isinstance(p, int) and 0 <= p < len(tokens) for p in positions), 'Sample positions')
    reads = []

    def tensor_range(name, begin, size):
        desc = manifest['tensors'][name]
        require(desc['dtype'] == 'F16' and 0 <= begin <= begin + size <= desc['size'], 'F16 tensor range')
        spans = desc.get('spans') or [dict(shardIndex=desc['shard'], offset=desc['offset'], size=desc['size'])]
        chunks, logical = [], 0
        for span in spans:
            low, high = max(begin, logical), min(begin + size, logical + span['size'])
            if high > low:
                file = args.artifact / manifest['shards'][span['shardIndex']]['filename']
                offset = span['offset'] + low - logical
                with file.open('rb') as handle:
                    handle.seek(offset)
                    raw = handle.read(high - low)
                require(len(raw) == high - low, 'Short artifact read')
                chunks.append(raw)
                reads.append(dict(shard=span['shardIndex'], offset=offset, bytes=len(raw), sha256=sha(raw)))
            logical += span['size']
        raw = b''.join(chunks)
        require(len(raw) == size, 'Incomplete tensor spans')
        return raw

    def gpu(stage):
        row = next(r for r in capture['tensors'] if r['layer'] == 0
                   and r['role'] == 'attn_hc' and r['stage'] == stage)
        raw = Path(row['file']).read_bytes()
        require(sha(raw) == row['sha256'] and len(raw) == math.prod(row['shape']) * 4, 'GPU payload identity')
        require(row['positions'] == positions, 'GPU positions differ')
        return np.frombuffer(raw, '<f4').reshape(row['shape']).copy()

    embedding = 'model.language_model.embed_tokens.weight'
    embedding_shape = manifest['tensors'][embedding]['shape']
    require(embedding_shape[1] == width and all(0 <= t < embedding_shape[0] for t in tokens), 'Embedding geometry')
    rows = {t: np.frombuffer(tensor_range(embedding, t * width * 2, width * 2), '<f2').astype('f4')
            for t in sorted(set(tokens))}
    streams = np.tile(np.stack([rows[t] for t in tokens]), (1, count))
    require(np.array_equal(streams[positions].view('u4'), gpu('input').view('u4')), 'Initial embedding mismatch')
    nodes = ast.parse(source).body
    names = ['Qwen4ExpTextRMSNorm', 'Qwen4ExpTextGatedResidual']
    classes = [next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == name) for name in names]
    require(all(not c.decorator_list for c in classes), 'Unexpected source class decorators')
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    namespace = dict(torch=torch, nn=nn, F=F)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, *classes], type_ignores=[])),
                 str(args.source), 'exec'), namespace)
    torch.set_num_threads(args.threads)
    require(torch.get_float32_matmul_precision() == 'highest', 'Reference must use full F32 operands')
    cfg = SimpleNamespace(hc_count=count, hidden_size=width, hc_lowrank=rank, rms_norm_eps=geometry['normEps'])
    module = namespace['Qwen4ExpTextGatedResidual'](cfg).float().cpu().eval()
    weights = {}
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            tensor = 'model.language_model.layers.0.attn_hyper_connection.' + name
            desc = manifest['tensors'][tensor]
            raw = tensor_range(tensor, 0, desc['size'])
            value = np.frombuffer(raw, '<f2').astype('f4').reshape(desc['shape'])
            require(tuple(parameter.shape) == value.shape, 'Source parameter geometry')
            parameter.copy_(torch.from_numpy(value))
            weights[name] = value
    selected, full = {}, {}

    def hook(stage):
        def observe(_module, _args, result):
            require(result.device.type == 'cpu', 'Reference must remain CPU-only')
            selected[stage] = result.numpy()[0, positions].copy()
            if stage in ['down', 'up']:
                full[stage] = result
        return observe

    for name, stage in [('hc_norm', 'norm'), ('input_mix_weight_down', 'down'),
                        ('input_mix_weight_up', 'up'), ('block_inject_weight', 'logits')]:
        getattr(module, name).register_forward_hook(hook(stage))

    def capture_low_rank(_module, arguments):
        selected['lowRank'] = arguments[0].numpy()[0, positions].copy()

    module.input_mix_weight_up.register_forward_pre_hook(capture_low_rank)
    with torch.no_grad():
        branch, _, gate = module(torch.from_numpy(streams).reshape(1, len(tokens), count * width))
        selected['branchInput'] = branch.numpy()[0, positions].copy()
        selected['gate'] = gate.numpy()[0, positions].copy()
        selected['mix'] = torch.sigmoid(full['up']).numpy()[0, positions].copy()
    args.output.mkdir()  # Refuse to replace any earlier success or failure.

    def save(name, value):
        array = np.asarray(value, dtype='<f4')
        require(np.isfinite(array).all(), 'Nonfinite reference payload')
        raw = array.tobytes()
        (args.output / (name + '.f32')).write_bytes(raw)
        return dict(file=name + '.f32', bytes=len(raw), sha256=sha(raw))

    save('input', gpu('input'))
    for name, key in [('weight', 'hc_norm.weight'), ('down-weight', 'input_mix_weight_down.weight'),
                      ('up-weight', 'input_mix_weight_up.weight'), ('inject-weight', 'block_inject_weight.weight')]:
        save(name, weights[key])
    for stage, value in selected.items():
        save('source-' + stage, value)
        save('gpu-' + stage, gpu(stage))
    payloads = {name: save(name, value.numpy()) for name, value in [
        ('full-down', full['down']), ('full-branchInput', branch), ('full-gate', gate)]}
    receipt = dict(scope='Unchanged source first-HC fixture only; no model qualification',
                   normalModelQualified=False, manifestSha256=checked['manifestSha256'],
                   inputsSha256=checked['inputsSha256'], gpuObservationSha256=checked['captureSha256'],
                   sourceSha256=checked['sourceSha256'], scriptSha256=sha(Path(__file__).read_bytes()),
                   sourceGeometry=[1, len(tokens), count * width], sourceClassUnchanged=True,
                   torchVersion=torch.__version__, torchThreads=args.threads, payloads=payloads,
                   artifactRangeReads=reads, newGpuInvocations=0, runtimeChanged=False, acceptanceChanged=False)
    (args.output / 'fixture.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(completed=True, payloads=payloads, newGpuInvocations=0)))


if __name__ == '__main__':
    main()
