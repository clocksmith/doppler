"""Observe learned initializers through GPU Identity outputs in inference graphs."""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import shutil

import onnx
from onnx import helper

from binding import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    source = json.loads((args.source / 'receipt.json').read_text())
    binding_path = Path(source['initializerBinding']['path'])
    if not source.get('completed') or digest(binding_path) != source['initializerBinding']['sha256']:
        raise ValueError('requires frozen completed source and complete initializer binding')
    binding = json.loads(binding_path.read_text())
    if not binding['pass'] or binding['parameterCount'] != 321:
        raise ValueError('all321 canonical bindings are required')
    args.out.mkdir(parents=True, exist_ok=False)
    receipt = deepcopy(source)
    receipt['sourceInferenceArtifact'] = {'path': str(args.source / 'receipt.json'), 'sha256': digest(args.source / 'receipt.json')}
    receipt['scope'] = 'Initializer observation outputs in actual inference sessions; no GPU or numerical qualification'
    receipt['readbackPorts'] = {}
    for phase, filename in source['sessionFiles'].items():
        path = Path(filename)
        model = onnx.load(path, load_external_data=False)
        initializer_bytes = [tensor.SerializeToString() for tensor in model.graph.initializer]
        inference_nodes = [node.SerializeToString() for node in model.graph.node]
        ports = []
        for parameter in binding['parameters']:
            matches = [entry for entry in parameter['bindings'] if entry['graph'] == phase]
            for entry in matches:
                output = 'observe_canonical_' + str(len(ports))
                model.graph.node.append(helper.make_node('Identity', [entry['initializer']], [output], name=output))
                model.graph.output.append(helper.make_tensor_value_info(output, onnx.TensorProto.FLOAT, entry['shape']))
                ports.append({'output': output, 'initializer': entry['initializer'], 'type': 'float32',
                    'shape': entry['shape'], 'bytes': 4 * math.prod(entry['shape']),
                    'expectedSha256': entry['sha256'], 'source': parameter['source'],
                    'canonicalBytesSha256': parameter['canonicalBytesSha256'], 'transform': entry['transform']})
        if initializer_bytes != [tensor.SerializeToString() for tensor in model.graph.initializer] or inference_nodes != [
                node.SerializeToString() for node in model.graph.node[:len(inference_nodes)]]:
            raise ValueError('observation construction changed an inference node or initializer')
        target = path.with_name(phase + '.' + args.out.name + '.onnx')
        if target.exists():
            raise ValueError('refusing to overwrite observation graph')
        onnx.save(model, target)
        onnx.checker.check_model(str(target))
        receipt['sessionFiles'][phase] = str(target)
        receipt['readbackPorts'][phase] = ports
        for graph in receipt['graphs']:
            if graph.get('sessionName', 'decoder_' + graph['mode']) == phase:
                graph['path'], graph['sha256'], graph['outputs'] = str(target), digest(target), [value.name for value in model.graph.output]
    receipt['observationConstruction'] = {'inferenceNodesAndInitializersByteUnchanged': True,
        'canonicalTensors': len({entry['source'] for ports in receipt['readbackPorts'].values() for entry in ports}),
        'sessionBindings': sum(len(ports) for ports in receipt['readbackPorts'].values()),
        'scope': 'GPU copies of declared initializers used by these same inference graphs; not private packed-kernel buffer introspection'}
    shutil.copytree(args.source / 'sources', args.out / 'sources')
    shutil.copyfile(Path(__file__), args.out / 'sources' / Path(__file__).name)
    receipt['sourceFiles'][Path(__file__).name] = digest(Path(__file__))
    for filename in ('config.json', 'tokenizer.json', 'initializer-binding.json'):
        shutil.copyfile(args.source / filename, args.out / filename)
    receipt['initializerBinding']['path'] = str(args.out / 'initializer-binding.json')
    files = [Path(entry['path']) for entry in source['artifacts'] if entry['path'].endswith('.data')]
    files.extend(Path(value) for value in receipt['sessionFiles'].values())
    receipt['artifacts'] = [{'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest(path)} for path in sorted(files)]
    (args.out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    main()
