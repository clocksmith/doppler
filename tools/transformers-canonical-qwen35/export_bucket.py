"""Export a native chunk64 prefill bucket while sharing frozen external weights."""
import argparse
from copy import deepcopy
import importlib.metadata
import json
from pathlib import Path
import shutil

from binding import digest, read_inputs, verify_bytes
from native_model import load_native_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('canonical', 'source', 'cpu-receipt', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    policy_path = Path(__file__).with_name('contract.json')
    policy = json.loads(policy_path.read_text())
    source = json.loads((args.source / 'receipt.json').read_text())
    proof = json.loads(args.cpu_receipt.read_text())
    lane = proof['contract']
    if not proof.get('completed') or not all(check['pass'] for check in proof['checks']) or lane['prefillBound'] != 64:
        raise ValueError('requires completed strict native CPU chunk64 adapter gate')
    if any(digest(Path(__file__).with_name(name)) != checksum for name, checksum in proof['sourceSha256'].items()):
        raise ValueError('CPU-proven frontend changed')
    if proof['policySha256'] != digest(policy_path) or not source.get('completed') or source['inputs'] != proof['inputs']:
        raise ValueError('frozen source or CPU gate identity differs')
    expected = policy['frameworks'] | policy['exportFrontend']['frameworks']
    if {name: importlib.metadata.version(name) for name in expected} != expected:
        raise ValueError('export environment differs from pinned dependencies')
    args.out.mkdir(parents=True, exist_ok=False)
    capture = args.out / 'capture'
    capture.mkdir()
    import torch
    import onnx
    from onnx import numpy_helper
    from export_text import TextDecoderExport, empty_cache, cache_names
    from compile_optimizer import optimize
    from onnx_lower import lower_rms_normalization
    from initializer_binding import initializer_inventory, bind_initializers
    torch.set_num_threads(policy['threads'])
    torch.set_num_interop_threads(policy['threads'])
    inputs = read_inputs(args.canonical, policy)
    verify_bytes(inputs, policy['hashChunkBytes'])
    native, config, mappings = load_native_model(inputs, policy)
    dimension = torch.export.Dim('sequence_length', min=2, max=lane['prefillBound'])
    states = empty_cache(config)
    arguments = (torch.zeros(1, 2, config.hidden_size), torch.zeros(3, 1, 2, dtype=torch.int64),
                 torch.ones(1, 2, dtype=torch.int64), states)
    with torch.no_grad():
        print('capture chunk64 prefill', flush=True)
        exported = torch.export.export(TextDecoderExport(native, lane, 'prefill').eval(), arguments,
            dynamic_shapes=({1: dimension}, {2: dimension}, {1: dimension}, tuple({} for state in states)), strict=True)
        program = torch.onnx.export(exported, dynamo=True, optimize=False,
            input_names=['inputs_embeds', 'position_ids', 'attention_mask'] + [a for a, b in cache_names(config)],
            output_names=['logits'] + [b for a, b in cache_names(config)], opset_version=policy['exportFrontend']['opset'])
        optimize(program)
        temporary = capture / 'decoder_prefill.onnx'
        program.save(temporary, external_data=True)
        del program, exported
        model = onnx.load(temporary, load_external_data=False)
        lowerings = lower_rms_normalization(model, check=False)
        onnx.save(model, temporary)
        old_binding = json.loads(Path(source['initializerBinding']['path']).read_text())
        old_index = {(entry['dtype'], tuple(entry['shape']), entry['sha256']): entry
                     for entry in old_binding['graphs']['decoder_prefill'] if entry['externalData']}
        new_index = {entry['name']: entry for entry in initializer_inventory(temporary)}
        shared = []
        for tensor in model.graph.initializer:
            entry = new_index[tensor.name]
            match = old_index.get((entry['dtype'], tuple(entry['shape']), entry['sha256']))
            if match:
                del tensor.external_data[:]
                for key, value in match['externalData'].items():
                    external = tensor.external_data.add()
                    external.key, external.value = key, value
                tensor.data_location = onnx.TensorProto.EXTERNAL
                tensor.ClearField('raw_data')
                shared.append({'initializer': tensor.name, 'sourceInitializer': match['name'], 'sha256': entry['sha256']})
            elif tensor.data_location == onnx.TensorProto.EXTERNAL:
                tensor.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(tensor, base_dir=str(capture)), tensor.name))
        source_paths = source.get('sessionFiles') or {name: str(args.source / 'onnx' / (name + '.onnx'))
            for name in ('embed_tokens', 'decoder_prefill', 'decoder_decode')}
        target = Path(source_paths['decoder_prefill']).with_name('decoder_prefill.' + args.out.name + '.onnx')
        if target.exists():
            raise ValueError('refusing to overwrite a bucket graph')
        onnx.save(model, target)
        onnx.checker.check_model(str(target))
        paths = {**source_paths, 'decoder_prefill': str(target)}
        binding = bind_initializers(inputs, policy, native, paths)
        (args.out / 'initializer-binding.json').write_text(json.dumps(binding, indent=2) + '\n')
    receipt = deepcopy(source)
    receipt['sourceFullArtifact'] = {'path': str(args.source / 'receipt.json'), 'sha256': digest(args.source / 'receipt.json')}
    receipt['scope'] = 'Explicit64-token prefill bucket construction; not ONNX numerical or GPU qualification'
    receipt['contract'] = lane
    receipt['cpuGate'] = {'path': str(args.cpu_receipt), 'sha256': digest(args.cpu_receipt)}
    receipt['sessionFiles'] = paths
    receipt['initializerBinding'] = {'path': str(args.out / 'initializer-binding.json'),
        'sha256': digest(args.out / 'initializer-binding.json'), 'pass': True}
    for graph in receipt['graphs']:
        if graph['mode'] == 'prefill':
            graph.update(path=str(target), sha256=digest(target), lowerings=lowerings,
                inputs=[value.name for value in model.graph.input], outputs=[value.name for value in model.graph.output])
            from collections import Counter
            graph['operators'] = dict(sorted(Counter(node.op_type for node in model.graph.node).items()))
    receipt['sharedBucketInitializers'] = shared
    shutil.copytree(args.source / 'sources', args.out / 'sources')
    for name in ('export_bucket.py', 'validate_text.py'):
        shutil.copyfile(Path(__file__).with_name(name), args.out / 'sources' / name)
        receipt['sourceFiles'][name] = digest(Path(__file__).with_name(name))
    for name in ('config.json', 'tokenizer.json'):
        shutil.copyfile(args.source / name, args.out / name)
    files = set(Path(value) for value in paths.values())
    for graph_path in paths.values():
        graph = onnx.load(graph_path, load_external_data=False)
        for tensor in graph.graph.initializer:
            external = {entry.key: entry.value for entry in tensor.external_data}
            if external:
                files.add(Path(graph_path).parent / external['location'])
    receipt['artifacts'] = [{'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest(path)} for path in sorted(files)]
    receipt['externalPayloadSharing'] = 'Single-linked original source payloads; bucket constants inline; same-byte learned weights reused'
    (args.out / 'lane-policy.json').write_text(json.dumps(lane, indent=2) + '\n')
    (args.out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    # These are exclusively the freshly-created disposable capture files. The
    # validated sibling graph now references only frozen source payloads.
    for path in capture.iterdir():
        path.unlink()
    capture.rmdir()
    print('chunk64 bucket exported; all321 bindings verified', flush=True)


if __name__ == '__main__':
    main()
