"""Test compensated F32 log1p within the explicit native Softplus lowering."""
import argparse
from collections import Counter
from copy import deepcopy
import json
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
    if not source.get('completed') or source['contract']['kvPhysicalDtype'] != 'float32':
        raise ValueError('requires completed explicit F32-KV source')
    args.out.mkdir(parents=True, exist_ok=False)
    receipt = deepcopy(source)
    receipt['sourceUncompensatedArtifact'] = {'path': str(args.source / 'receipt.json'), 'sha256': digest(args.source / 'receipt.json')}
    receipt['scope'] = 'Separate compensated-Softplus CPU hypothesis; no numerical/GPU qualification'
    receipt['contract']['softplusLowering'] = {'precision': 'float32', 'formula':
        'p=exp(-abs(x)); u=1+p; correction=log(u)*p/(u-1) when u>1 else p; max(x,0)+correction',
        'purpose': 'Recover small-value precision lost by rounded 1+p; no threshold changes'}
    receipt['softplusCompensation'] = []
    for graph in receipt['graphs']:
        path = Path(graph['path'])
        model = onnx.load(path, load_external_data=False)
        original_initializers = [tensor.SerializeToString() for tensor in model.graph.initializer]
        result, replacements, changes = [], {}, []
        for original in model.graph.node:
            node = deepcopy(original)
            for index, name in enumerate(node.input):
                node.input[index] = replacements.get(name, name)
            result.append(node)
            if node.op_type != 'Log' or not node.name.endswith('/primitive/logarithm'):
                continue
            prefix = node.name[:-len('logarithm')]
            if list(node.input) != [prefix + 'plus_one'] or list(node.output) != [prefix + 'logarithm']:
                raise ValueError('native Softplus lowering structure changed')
            operations = [('Sub', [prefix + 'plus_one', prefix + 'one'], 'rounded_delta'),
                ('Greater', [prefix + 'rounded_delta', prefix + 'zero'], 'nonzero_delta'),
                ('Where', [prefix + 'nonzero_delta', prefix + 'rounded_delta', prefix + 'one'], 'safe_delta'),
                ('Div', [prefix + 'exponential', prefix + 'safe_delta'], 'log1p_ratio'),
                ('Mul', [prefix + 'logarithm', prefix + 'log1p_ratio'], 'corrected_log1p'),
                ('Where', [prefix + 'nonzero_delta', prefix + 'corrected_log1p', prefix + 'exponential'], 'compensated_log1p')]
            for operator, inputs, suffix in operations:
                result.append(helper.make_node(operator, inputs, [prefix + suffix], name=prefix + suffix))
            replacements[prefix + 'logarithm'] = prefix + 'compensated_log1p'
            changes.append({'node': node.name, 'operators': [entry[0] for entry in operations]})
        if len(changes) != 18:
            raise ValueError('expected exactly18 linear-attention Softplus sites per decoder')
        del model.graph.node[:]
        model.graph.node.extend(result)
        if original_initializers != [tensor.SerializeToString() for tensor in model.graph.initializer]:
            raise ValueError('compensation changed an initializer')
        target = path.with_name('decoder_' + graph['mode'] + '.' + args.out.name + '.onnx')
        if target.exists():
            raise ValueError('refusing to overwrite compensated candidate')
        onnx.save(model, target)
        onnx.checker.check_model(str(target))
        graph.update(path=str(target), sha256=digest(target), operators=dict(sorted(Counter(node.op_type for node in model.graph.node).items())))
        receipt['sessionFiles']['decoder_' + graph['mode']] = str(target)
        receipt['softplusCompensation'].append({'phase': graph['mode'], 'sites': changes, 'initializersUnchanged': True})
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
