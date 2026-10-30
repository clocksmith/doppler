"""Restore the pinned native SiLU division realization in a separate graph."""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import shutil

import onnx
from onnx import helper, TensorProto as T
from binding import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--native-sigmoid', action='store_true')
    args = parser.parse_args()
    source = json.loads((args.source / 'receipt.json').read_text())
    if not source.get('completed') or source['contract']['kvPhysicalDtype'] != 'float32':
        raise ValueError('requires completed F32-KV source')
    args.out.mkdir(parents=True, exist_ok=False)
    receipt = deepcopy(source)
    receipt['sourceSigmoidSiluArtifact'] = {'path': str(args.source / 'receipt.json'), 'sha256': digest(args.source / 'receipt.json')}
    receipt['scope'] = 'Separate native SiLU division realization; requires its own CPU/GPU numerical gates'
    receipt['contract']['siluLowering'] = {'precision': 'float32', 'formula': 'x / (1 + exp(-x))',
        'source': 'Pinned native SiLU CPU primitive comparison; replace only exact x*Sigmoid(x) graph patterns'}
    receipt['siluChanges'] = []
    if args.native_sigmoid:
        receipt['contract']['sigmoidLowering'] = {'precision': 'float32', 'formula': '1 / (1 + exp(-x))',
            'source': 'Pinned native sigmoid primitive comparison; retain all gates and precision'}
    for graph in receipt['graphs']:
        path = Path(graph['path'])
        model = onnx.load(path, load_external_data=False)
        initializers = [tensor.SerializeToString() for tensor in model.graph.initializer]
        producers = {name: node for node in model.graph.node for name in node.output}
        one = '__native_silu_one_f32'
        if any(tensor.name == one for tensor in model.graph.initializer):
            raise ValueError('native SiLU variant was already applied')
        model.graph.initializer.append(helper.make_tensor(one, T.FLOAT, [], [1]))
        nodes, changes = [], []
        for node in model.graph.node:
            matches = []
            if node.op_type == 'Mul' and len(node.input) == 2 and not node.domain:
                for axis in (0, 1):
                    sigmoid = producers.get(node.input[axis])
                    if sigmoid is not None and sigmoid.op_type == 'Sigmoid' and not sigmoid.domain and list(sigmoid.input) == [node.input[1 - axis]]:
                        matches.append(node.input[1 - axis])
            if not matches:
                nodes.append(node)
                continue
            if len(matches) != 1:
                raise ValueError('ambiguous native SiLU pattern')
            value, prefix = matches[0], node.name + '/native-silu/'
            nodes.extend((helper.make_node('Neg', [value], [prefix + 'negative'], name=prefix + 'negative'),
                helper.make_node('Exp', [prefix + 'negative'], [prefix + 'exponential'], name=prefix + 'exponential'),
                helper.make_node('Add', [prefix + 'exponential', one], [prefix + 'denominator'], name=prefix + 'denominator'),
                helper.make_node('Div', [value, prefix + 'denominator'], list(node.output), name=prefix + 'division')))
            changes.append({'node': node.name, 'input': value, 'output': node.output[0]})
        if len(changes) != 60:
            raise ValueError('expected24MLP+18conv+18gatedNorm SiLU sites, found ' + str(len(changes)))
        needed, kept = {value.name for value in model.graph.output}, []
        for node in reversed(nodes):
            if any(name in needed for name in node.output):
                kept.append(node); needed.update(node.input)
        del model.graph.node[:]
        model.graph.node.extend(reversed(kept))
        sigmoid_changes = []
        if args.native_sigmoid:
            lowered = []
            for node in model.graph.node:
                if node.op_type != 'Sigmoid' or node.domain:
                    lowered.append(node)
                    continue
                prefix = node.name + '/native-sigmoid/'
                lowered.extend((helper.make_node('Neg', list(node.input), [prefix + 'negative'], name=prefix + 'negative'),
                    helper.make_node('Exp', [prefix + 'negative'], [prefix + 'exponential'], name=prefix + 'exponential'),
                    helper.make_node('Add', [prefix + 'exponential', one], [prefix + 'denominator'], name=prefix + 'denominator'),
                    helper.make_node('Reciprocal', [prefix + 'denominator'], list(node.output), name=prefix + 'reciprocal')))
                sigmoid_changes.append(node.name)
            if len(sigmoid_changes) != 24:
                raise ValueError('expected18 beta+6 full-attention sigmoid sites')
            del model.graph.node[:]
            model.graph.node.extend(lowered)
        if initializers != [tensor.SerializeToString() for tensor in model.graph.initializer[:-1]]:
            raise ValueError('native SiLU lowering changed a source initializer')
        target = path.with_name('decoder_' + graph['mode'] + '.' + args.out.name + '.onnx')
        if target.exists():
            raise ValueError('refusing to overwrite native SiLU candidate')
        onnx.save(model, target)
        onnx.checker.check_model(str(target))
        graph.update(path=str(target), sha256=digest(target), operators=dict(sorted(Counter(node.op_type for node in model.graph.node).items())))
        receipt['sessionFiles']['decoder_' + graph['mode']] = str(target)
        receipt['siluChanges'].append({'phase': graph['mode'], 'sites': changes, 'sigmoidSites': sigmoid_changes,
            'canonicalInitializersUnchanged': True})
    shutil.copytree(args.source / 'sources', args.out / 'sources')
    shutil.copyfile(Path(__file__), args.out / 'sources' / Path(__file__).name)
    receipt['sourceFiles'][Path(__file__).name] = digest(Path(__file__))
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
