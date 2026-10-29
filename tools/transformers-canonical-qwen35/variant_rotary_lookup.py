"""Freeze bounded integer-position rotary functions as export-time lookup data.

The pinned ORT WASM registry omits WebGPU Cos/Sin. All runtime tensor work
remains ONNX/WebGPU Gather/Where/Concat. Tables use the exact source ONNX CPU
subgraph, not a replacement frequency formula or runtime CPU calculation.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import onnx
from onnx import TensorProto as T, helper, numpy_helper
import onnxruntime as ort

from binding import digest


def prune(model, outputs):
    needed, kept = set(outputs), []
    for node in reversed(model.graph.node):
        if any(name in needed for name in node.output):
            kept.append(deepcopy(node))
            needed.update(node.input)
    return list(reversed(kept)), needed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    source = json.loads((args.source / 'receipt.json').read_text())
    if not source.get('completed') or 'hostShapeLowerings' not in source:
        raise ValueError('requires explicit host-shape source candidate')
    args.out.mkdir(parents=True, exist_ok=False)
    receipt = deepcopy(source)
    receipt['sourceHostShapeArtifact'] = {'path': str(args.source / 'receipt.json'), 'sha256': digest(args.source / 'receipt.json')}
    receipt['scope'] = 'Bounded rotary lookup GPU-admission candidate; no qualification'
    receipt['contract']['rotaryLookup'] = {'minimumPosition': 0, 'maximumPositionExclusive': source['contract']['maxSequenceLength'],
        'construction': 'Pinned source ONNX CPU rotary subgraph, export time only', 'runtime': 'ONNX WebGPU Gather/Where/Concat'}
    receipt['rotaryLookupLowerings'] = []
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    for entry in receipt['graphs']:
        phase, path = entry['mode'], Path(entry['path'])
        model = onnx.load(path, load_external_data=False)
        initializers = {tensor.name: tensor for tensor in model.graph.initializer}
        original_initializers = {tensor.name: tensor.SerializeToString() for tensor in model.graph.initializer}
        metadata = {value.name: value for value in list(model.graph.value_info) + list(model.graph.output)}
        trig = {node.op_type: node for node in model.graph.node if node.op_type in ('Cos', 'Sin')}
        if set(trig) != {'Cos', 'Sin'} or any(len(node.output) != 1 for node in trig.values()):
            raise ValueError('source rotary trig subgraph changed')
        names = [trig[op].output[0] for op in ('Cos', 'Sin')]
        nodes, needed = prune(model, names)
        small = [numpy_helper.from_array(numpy_helper.to_array(tensor, base_dir=str(path.parent)), tensor.name)
                 for tensor in model.graph.initializer if tensor.name in needed]
        position = deepcopy(next(value for value in model.graph.input if value.name == 'position_ids'))
        position.type.tensor_type.shape.dim[2].dim_param = 'lookup_sequence'
        for name in names:
            metadata[name] = deepcopy(metadata[name])
            metadata[name].type.tensor_type.shape.dim[1].dim_param = 'lookup_sequence'
        graph = helper.make_graph(nodes, 'source_rotary', [deepcopy(position)], [deepcopy(metadata[name]) for name in names], small)
        original = helper.make_model(graph, opset_imports=model.opset_import, ir_version=model.ir_version)
        onnx.checker.check_model(original)
        original_bytes = original.SerializeToString()
        native = ort.InferenceSession(original_bytes, options, providers=['CPUExecutionProvider'])
        bound = receipt['contract']['maxSequenceLength']
        ids = np.broadcast_to(np.arange(bound, dtype=np.int32), (3, 1, bound)).copy()
        tables = native.run(None, {'position_ids': ids})
        new_nodes = []
        for axis in range(3):
            axis_name = f'lookup_axis_{axis}'
            model.graph.initializer.append(numpy_helper.from_array(np.array(axis, dtype=np.int64), axis_name))
            new_nodes.append(helper.make_node('Gather', ['position_ids', axis_name],
                [f'lookup_positions_{axis}'], axis=0, name=f'lookup_positions_{axis}'))
        for kind, table in zip(('cos', 'sin'), tables):
            table_name = f'rotary_lookup_{kind}'
            model.graph.initializer.append(numpy_helper.from_array(table[0], table_name))
            for axis in range(3):
                new_nodes.append(helper.make_node('Gather', [table_name, f'lookup_positions_{axis}'],
                    [f'lookup_{kind}_{axis}'], axis=0, name=f'lookup_{kind}_{axis}'))
            for role in ('height', 'width'):
                mask_name = f'lookup_{role}_mask'
                if kind == 'cos':
                    mask = numpy_helper.to_array(initializers[role + '_mask'], base_dir=str(path.parent))
                    model.graph.initializer.append(numpy_helper.from_array(np.concatenate((mask, mask)), mask_name))
            new_nodes.extend((helper.make_node('Where', ['lookup_height_mask', f'lookup_{kind}_1', f'lookup_{kind}_0'],
                [f'lookup_{kind}_height'], name=f'lookup_{kind}_height'),
                helper.make_node('Where', ['lookup_width_mask', f'lookup_{kind}_2', f'lookup_{kind}_height'],
                [trig[kind.capitalize()].output[0]], name=f'lookup_{kind}_width')))
        old_nodes = {node.name for node in nodes}
        remaining = [node for node in model.graph.node if node.name not in old_nodes]
        del model.graph.node[:]
        model.graph.node.extend(new_nodes + remaining)
        new_rotary = helper.make_model(helper.make_graph(new_nodes, 'lookup_rotary', [deepcopy(position)],
            [deepcopy(metadata[name]) for name in names], [deepcopy(tensor) for tensor in model.graph.initializer
             if tensor.name in {name for node in new_nodes for name in node.input}]),
            opset_imports=model.opset_import, ir_version=model.ir_version)
        onnx.checker.check_model(new_rotary)
        candidate = ort.InferenceSession(new_rotary.SerializeToString(), options, providers=['CPUExecutionProvider'])
        comparisons = []
        for length in (1, 2, 28, 65, bound):
            ids = np.stack([(np.arange(length, dtype=np.int32) * multiplier + offset) % bound
                            for multiplier, offset in ((1, 0), (3, bound - 1), (7, bound // 2))])[:, None]
            expected, actual = native.run(None, {'position_ids': ids}), candidate.run(None, {'position_ids': ids})
            errors = [float(np.abs(a - b).max()) for a, b in zip(expected, actual)]
            if any(error != 0 for error in errors):
                raise ValueError('bounded rotary lookup differs from source ONNX CPU')
            comparisons.append({'length': length, 'allThreeAxesIndependent': True, 'maxErrors': errors, 'bitExact': True})
        for tensor in model.graph.initializer:
            if tensor.name in original_initializers and tensor.SerializeToString() != original_initializers[tensor.name]:
                raise ValueError('rotary lookup changed a source initializer')
        target = path.with_name('decoder_' + phase + '.' + args.out.name + '.onnx')
        if target.exists():
            raise ValueError('refusing to overwrite rotary lookup graph')
        onnx.save(model, target)
        onnx.checker.check_model(str(target))
        entry['path'], entry['sha256'] = str(target), digest(target)
        receipt['sessionFiles']['decoder_' + phase] = str(target)
        receipt['rotaryLookupLowerings'].append({'phase': phase, 'sourceSubgraphSha256': hashlib.sha256(original_bytes).hexdigest(),
            'tables': [{'name': kind, 'shape': list(table[0].shape), 'sha256': hashlib.sha256(table[0].tobytes()).hexdigest()}
                       for kind, table in zip(('cos', 'sin'), tables)], 'comparisons': comparisons,
            'canonicalInitializerProtobufsUnchanged': True})
    shutil.copytree(args.source / 'sources', args.out / 'sources')
    shutil.copyfile(Path(__file__), args.out / 'sources' / 'variant_rotary_lookup.py')
    receipt['sourceFiles']['variant_rotary_lookup.py'] = digest(Path(__file__))
    for filename in ('config.json', 'tokenizer.json', 'initializer-binding.json'):
        shutil.copyfile(args.source / filename, args.out / filename)
    receipt['initializerBinding']['path'] = str(args.out / 'initializer-binding.json')
    files = [Path(entry['path']) for entry in source['artifacts'] if entry['path'].endswith('.data')]
    files.extend(Path(value) for value in receipt['sessionFiles'].values())
    receipt['artifacts'] = [{'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest(path)} for path in sorted(files)]
    (args.out / 'lane-policy.json').write_text(json.dumps(receipt['contract'], indent=2) + '\n')
    (args.out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    main()
