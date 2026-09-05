"""Pinned source extraction and PyTorch oracle. Never used by application inference."""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import urllib.request

os.environ['USE_TF'] = '0'
import numpy as np
import torch
import transformers
import chronos.chronos_bolt as bolt
from chronos import ChronosBoltPipeline
from huggingface_hub import snapshot_download
from safetensors import safe_open

REVISION = 'a0e552de83495b5c28c14c71c374f3e33280b340'

def digest(data):
    return 'sha256:' + hashlib.sha256(data).hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--history', required=True)
    args = parser.parse_args()
    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    source = Path(snapshot_download('amazon/chronos-bolt-tiny', revision=REVISION,
        allow_patterns=['config.json', 'model.safetensors', 'README.md']))
    for name in ['config.json', 'model.safetensors', 'README.md']:
        shutil.copyfile(source / name, dest / name)
    (dest / 'chronos_bolt.py').write_text(inspect.getsource(bolt))
    from transformers.models.t5 import modeling_t5
    (dest / 'modeling_t5.py').write_text(inspect.getsource(modeling_t5))
    license_url = 'https://raw.githubusercontent.com/amazon-science/chronos-forecasting/v2.3.1/LICENSE'
    (dest / 'LICENSE').write_bytes(urllib.request.urlopen(license_url, timeout=30).read())
    inventory = {}
    with (dest / 'weights.bin').open('wb') as output, safe_open(str(source / 'model.safetensors'), framework='np') as weights:
        for name in sorted(weights.keys()):
            tensor = weights.get_tensor(name)
            if tensor.dtype != np.float32:
                raise ValueError('This qualification lane preserves source F32 only: ' + name)
            raw = tensor.astype('<f4', copy=False).tobytes()
            inventory[name] = {'shape': list(tensor.shape), 'dtype': 'F32', 'offsetBytes': output.tell(),
                               'sizeBytes': len(raw), 'hash': digest(raw)}
            output.write(raw)
    config = json.loads((source / 'config.json').read_text())
    pipeline = ChronosBoltPipeline.from_pretrained(str(source), device_map='cpu', torch_dtype=torch.float32)
    model = pipeline.model.eval()
    torch.set_num_threads(1)
    history = json.loads(Path(args.history).read_text())
    values = [float(row['value']) for row in history['rows']]
    contexts = [
        ('eia-latest-revised', values[-512:]),
        ('eia-held-out-window', values[-564:-52]),
        ('short-context', values[-17:]),
        ('constant', [4.0] * 512),
        ('negative-trend', np.linspace(-30, 10, 512).tolist()),
    ]
    cases = []
    boundaries = {}
    handles = []
    for name, module in model.named_modules():
        if name in ['input_patch_embedding', 'encoder.final_layer_norm', 'decoder.final_layer_norm', 'output_patch_embedding']:
            def capture(_module, _input, output, label=name):
                boundaries[label] = output.detach().float().reshape(-1).tolist()
            handles.append(module.register_forward_hook(capture))
    with torch.inference_mode():
        for name, context in contexts:
            boundaries.clear()
            # Right-pad absent history with NaNs on the LEFT, exactly matching
            # the declared context envelope. NaNs never enter the browser API.
            padded = torch.full((1, 512), float('nan'), dtype=torch.float32)
            padded[0, -len(context):] = torch.tensor(context, dtype=torch.float32)
            output = model(context=padded).quantile_preds[0]
            selected = output[[0, 4, 8]].transpose(0, 1).contiguous().reshape(-1).tolist()
            cases.append({'id': name, 'context': context, 'horizon': 64, 'values': selected,
                          'boundaries': dict(boundaries)})
    for handle in handles:
        handle.remove()
    reference = {'schema': 'doppler.forecast-reference/v1', 'repository': 'amazon/chronos-bolt-tiny',
        'revision': REVISION, 'torch': torch.__version__, 'transformers': transformers.__version__,
        'chronos': '2.3.1', 'host': 'pytorch-cpu-f32', 'contextLength': 512,
        'quantiles': [0.1, 0.5, 0.9], 'tolerance': {'absolute': 0.002, 'relative': 0.0002}, 'cases': cases}
    (dest / 'reference.json').write_text(json.dumps(reference, separators=(',', ':'), allow_nan=False))
    files = {name: {'hash': digest((dest / name).read_bytes()), 'sizeBytes': (dest / name).stat().st_size}
             for name in ['config.json', 'model.safetensors', 'README.md', 'chronos_bolt.py', 'modeling_t5.py', 'LICENSE', 'weights.bin', 'reference.json']}
    intake = {'schema': 'doppler.chronos-source/v1', 'repository': 'amazon/chronos-bolt-tiny', 'revision': REVISION,
              'files': files, 'config': config, 'tensors': inventory, 'licenseUrl': license_url}
    (dest / 'source-intake.json').write_text(json.dumps(intake, indent=2))
    print(json.dumps({'out': str(dest), 'tensorCount': len(inventory), 'weightBytes': files['weights.bin']['sizeBytes'],
                      'referenceCases': len(cases), 'sourceRevision': REVISION}))

if __name__ == '__main__':
    main()
