"""Combine explicit64/704 prefill graphs with shared embedding/decode sessions."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil

from binding import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('small', 'large', 'small-cpu', 'large-cpu', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    artifacts = {64: json.loads((args.small / 'receipt.json').read_text()), 704: json.loads((args.large / 'receipt.json').read_text())}
    gates = {64: json.loads((args.small_cpu / 'receipt.json').read_text()), 704: json.loads((args.large_cpu / 'receipt.json').read_text())}
    large = artifacts[704]
    for bound, artifact in artifacts.items():
        if (not artifact.get('completed') or artifact['contract']['prefillBound'] != bound or
                artifact['inputs'] != large['inputs'] or artifact['contract']['kvPhysicalDtype'] != 'float32' or
                not gates[bound].get('completed')):
            raise ValueError('bucket construction/native CPU proof is incomplete or different')
        for key in ('stateAbsoluteTolerance', 'stateRelativeTolerance', 'logitAbsoluteTolerance', 'maxSequenceLength', 'kvPhysicalDtype'):
            if artifact['contract'][key] != large['contract'][key]:
                raise ValueError('bucket precision or frozen numerical gate differs')
        if digest(Path(artifact['initializerBinding']['path'])) != artifact['initializerBinding']['sha256']:
            raise ValueError('bucket initializer proof changed')
    args.out.mkdir(parents=True, exist_ok=False)
    receipt = deepcopy(large)
    receipt['scope'] = 'Explicit64/704 prefill bucket composition; strict CPU numerical failures remain retained'
    receipt['contract']['prefillBuckets'] = [64, 704]
    receipt['contract']['bucketGeometryInputs'] = {str(bound): artifact['contract']['hostGeometryInputs']['prefill']
                                                  for bound, artifact in artifacts.items()}
    receipt['contract']['hostGeometryInputs']['prefill'] = []
    receipt['contract']['scope'] = 'Explicit64/704 prefill buckets; smallest admitted bucket, chunk64; context2048; no timing admission'
    receipt['sessionFiles'] = {key: value for key, value in large['sessionFiles'].items() if key != 'decoder_prefill'}
    receipt['graphs'] = []
    binding = deepcopy(json.loads(Path(large['initializerBinding']['path']).read_text()))
    binding['graphs'].pop('decoder_prefill')
    for parameter in binding['parameters']:
        parameter['bindings'] = [entry for entry in parameter['bindings'] if entry['graph'] != 'decoder_prefill']
    by_source = {parameter['source']: parameter for parameter in binding['parameters']}
    receipt['bucketSources'] = []
    for bound, artifact in artifacts.items():
        root = args.small if bound == 64 else args.large
        cpu_root = args.small_cpu if bound == 64 else args.large_cpu
        key = 'decoder_prefill_' + str(bound)
        receipt['sessionFiles'][key] = artifact['sessionFiles']['decoder_prefill']
        graph = deepcopy(next(graph for graph in artifact['graphs'] if graph['mode'] == 'prefill'))
        graph.update(sessionName=key, bound=bound)
        receipt['graphs'].append(graph)
        original = json.loads(Path(artifact['initializerBinding']['path']).read_text())
        binding['graphs'][key] = original['graphs']['decoder_prefill']
        for parameter in original['parameters']:
            target = by_source[parameter['source']]
            if target['canonicalBytesSha256'] != parameter['canonicalBytesSha256']:
                raise ValueError('canonical parameter bytes differ between buckets')
            for entry in parameter['bindings']:
                if entry['graph'] == 'decoder_prefill':
                    target['bindings'].append({**entry, 'graph': key})
        receipt['bucketSources'].append({'bound': bound, 'artifact': {'path': str(root / 'receipt.json'),
            'sha256': digest(root / 'receipt.json')}, 'cpuGate': {'path': str(cpu_root / 'receipt.json'),
            'sha256': digest(cpu_root / 'receipt.json'), 'pass': gates[bound]['pass'],
            'failedChecks': sum(not check['pass'] for check in gates[bound]['checks'])}})
    graph = deepcopy(next(graph for graph in large['graphs'] if graph['mode'] == 'decode'))
    graph['sessionName'] = 'decoder_decode'
    receipt['graphs'].append(graph)
    (args.out / 'initializer-binding.json').write_text(json.dumps(binding, indent=2) + '\n')
    receipt['initializerBinding'] = {'path': str(args.out / 'initializer-binding.json'),
        'sha256': digest(args.out / 'initializer-binding.json'), 'pass': True}
    shutil.copytree(args.large / 'sources', args.out / 'sources')
    shutil.copyfile(Path(__file__), args.out / 'sources' / Path(__file__).name)
    receipt['sourceFiles'][Path(__file__).name] = digest(Path(__file__))
    for filename in ('config.json', 'tokenizer.json'):
        shutil.copyfile(args.large / filename, args.out / filename)
    files = set(Path(value) for value in receipt['sessionFiles'].values())
    for artifact in artifacts.values():
        files.update(Path(entry['path']) for entry in artifact['artifacts'] if entry['path'].endswith('.data'))
    receipt['artifacts'] = [{'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest(path)} for path in sorted(files)]
    receipt['sessionResidency'] = 'Four independent sessions; embedding, two prefill buckets, decode. Initializers may be duplicated in GPU memory.'
    (args.out / 'lane-policy.json').write_text(json.dumps(receipt['contract'], indent=2) + '\n')
    (args.out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    main()
