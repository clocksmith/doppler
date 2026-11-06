"""CPU sensitivity control: run the pinned forward with one-row F32 router GEMMs.

Keep upstream forward logic and all other operations. Change only the batch
geometry of its router F.linear primitive. This control is not a replacement
reference or model qualification.
Usage: python3 -P prepare-router-geometry-forward.py <producer> <source> <output-dir> <artifact> <reference-data>
"""
import hashlib
import json
import sys
from pathlib import Path

producer, source, output, artifact, data = map(Path, sys.argv[1:])
assert not output.exists(), 'use a fresh output directory; retain earlier controls'
original = producer.read_bytes()
text = original.decode()
assert 'ARTIFACT_MANIFEST_BYTES =' in text and 'if KV_F16:' in text
root = 'ROOT = Path(__file__).resolve().parent'
assert text.count(root) == 1
text = text.replace(root, 'ROOT = Path(' + repr(str(data.resolve())) + ')')
anchor = "embed = read_tensor('model.language_model.embed_tokens.weight')"
assert text.count(anchor) == 1
text = text.replace(anchor, '''
router_class = namespace['Qwen4ExpTextTopKRouter']
original_router_forward = router_class.forward
def geometry_router_forward(module, hidden):
    original_linear = F.linear
    def geometry_linear(values, weight, bias=None):
        if weight is module.weight:
            return torch.cat([original_linear(row, weight, bias) for row in values.split(1)], dim=0)
        return original_linear(values, weight, bias)
    F.linear = geometry_linear
    try:
        return original_router_forward(module, hidden)
    finally:
        F.linear = original_linear
router_class.forward = geometry_router_forward
''' + anchor)
anchor = 'logits = F.linear(inputs[0].float().reshape(-1, inputs[0].shape[-1]), module.weight.float())'
assert text.count(anchor) == 1
text = text.replace(anchor, 'logits = output[0].float()')
anchor = "target.write_text(json.dumps(result, separators=(',', ':')) + '\\n')"
assert text.count(anchor) == 1
text = text.replace(anchor, '''
result['scope'] = 'CPU router-geometry sensitivity control: pinned upstream forward logic with each router F32 linear projection dispatched one row at a time; all other operations unchanged. No GPU values or model qualification.'
result['routingGeometryControl'] = {'batchRows': 1, 'normalModelQualified': False}
''' + anchor)
sys.argv = [str(producer), str(source), str(output), '--dtype', 'float32',
            '--artifact', str(artifact), '--sequences', 'long', '--kv-dtype', 'f16']
environment = {'__name__': '__main__', '__file__': str(producer)}
exec(compile(text, str(producer) + ':router-geometry-control', 'exec'), environment)
sha = lambda b: hashlib.sha256(b).hexdigest()
receipt = dict(schema='doppler.flash-next-router-geometry-forward/v1',
               scope='CPU-only sensitivity control; no normal model qualification',
               producerSha256=sha(original), instrumentedProducerSha256=sha(text.encode()),
               instrumentSha256=sha(Path(__file__).read_bytes()),
               artifactManifestSha256=sha(environment['ARTIFACT_MANIFEST_BYTES']),
               inputsSha256=sha(environment['INPUTS_PATH'].read_bytes()),
               files={p.name: sha(p.read_bytes()) for p in output.iterdir() if p.suffix in ('.json', '.npy')})
(output / 'control-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
