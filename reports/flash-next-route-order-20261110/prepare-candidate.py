"""Prepare an isolated expert-ID sum-order experiment; never edit its parent."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('parent', type=Path)
parser.add_argument('candidate', type=Path)
args = parser.parse_args()
parent, candidate = args.parent.resolve(), args.candidate.resolve()
if candidate.exists() or parent == candidate or parent in candidate.parents:
    raise ValueError('Use a fresh candidate outside the frozen parent runtime')
receipt_path = candidate.parent / 'preparation.json'
if receipt_path.exists():
    raise ValueError('Preserve the existing preparation receipt; choose a fresh directory')
candidate.mkdir(parents=True)
for name in ['src', 'tests']:
    shutil.copytree(parent / name, candidate / name, symlinks=True)
shutil.copy2(parent / 'package.json', candidate / 'package.json')
(candidate / 'node_modules').symlink_to((parent / 'node_modules').resolve(), target_is_directory=True)

changes = []
def change(name, edit):
    destination = candidate / name
    original = destination.read_bytes()
    revised = edit(original.decode()).encode()
    if original == revised:
        raise ValueError(f'No change to {name}')
    destination.write_bytes(revised)
    changes.append(dict(file=name, beforeSha256=hashlib.sha256(original).hexdigest(),
                        afterSha256=hashlib.sha256(revised).hexdigest()))

def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'Expected exactly one occurrence: {old!r}')
    return text.replace(old, new)

def shader(text):
    text = replace_once(text,
        '@group(0) @binding(3) var<storage, read> routing_weights:',
        '@group(0) @binding(2) var<storage, read> routing_indices: array<u32>;\n'
        '@group(0) @binding(3) var<storage, read> routing_weights:')
    text = replace_once(text, '    var sum = 0.0f;',
        '    var sum = 0.0f;\n    var previous_expert = 0u;\n    var previous_slot = 0u;')
    return replace_once(text, '        let route = token * u.top_k + k;\n        var product = 0.0f;\n        if (product_index < size) {',
        '''        var product = 0.0f;
        if (product_index < size) {
            // Experimental source-order control: ascending expert ID, paired
            // with its original route weight and output. Barriers stay uniform.
            var best_expert = 0xffffffffu;
            var best_slot = 0xffffffffu;
            for (var slot = 0u; slot < u.top_k; slot++) {
                let expert = routing_indices[token * u.top_k + slot];
                let eligible = k == 0u || expert > previous_expert
                    || (expert == previous_expert && slot > previous_slot);
                if (eligible && (expert < best_expert
                    || (expert == best_expert && slot < best_slot))) {
                    best_expert = expert;
                    best_slot = slot;
                }
            }
            previous_expert = best_expert;
            previous_slot = best_slot;
            let route = token * u.top_k + best_slot;''')
change('src/gpu/kernels/scatter_add_routes_f32.wgsl', shader)

def wrapper(text):
    text = replace_once(text, 'const { numTokens, hiddenSize, topK, outputBuffer } = options;',
        'const { numTokens, hiddenSize, topK, outputBuffer, indicesBuffer } = options;')
    text = replace_once(text, '  outputAlias(outputBuffer, [input.buffer, weights]);',
        "  extent(indicesBuffer, numTokens * topK * 4, 'routing indices');\n"
        '  outputAlias(outputBuffer, [input.buffer, indicesBuffer, weights]);')
    return replace_once(text, "'f32', [input.buffer, weights, output], {",
        "'f32', [input.buffer, indicesBuffer, weights, output], {")
change('src/gpu/kernels/moe-route.js', wrapper)
change('src/gpu/kernels/moe-route.d.ts', lambda text: replace_once(text,
    'export interface RouteScatterOptions {', 'export interface RouteScatterOptions {\n  indicesBuffer: GPUBuffer;'))
change('src/inference/pipelines/text/moe/qwen-route-executor.js', lambda text: replace_once(text,
    'scatter(down, weightsBuffer, { numTokens, hiddenSize, topK,',
    'scatter(down, weightsBuffer, { numTokens, hiddenSize, topK, indicesBuffer,'))
def registry(text):
    data = json.loads(text)
    bindings = data['operations']['scatter_add_routes']['variants']['f32']['bindings']
    assert [binding['index'] for binding in bindings] == [0, 1, 3, 5]
    bindings.insert(2, dict(index=2, name='routing_indices', type='read-only-storage'))
    return json.dumps(data, indent=2) + '\n'
change('src/config/kernels/registry.json', registry)

# Update the canonical content digest for the changed entry only. All other
# prepared shader identities and files must remain unchanged.
shader_bytes = (candidate / changes[0]['file']).read_text().replace('\r\n', '\n')
new_digest = hashlib.sha256((shader_bytes + '\n@@entry:scatter_add_routes').encode()).hexdigest()
def digest(text):
    import re
    pattern = r'("scatter_add_routes_f32\.wgsl#scatter_add_routes": ")[0-9a-f]{64}("[,])'
    text, count = re.subn(pattern, lambda match: match[1] + new_digest + match[2], text)
    assert count == 1
    return text
change('src/config/kernels/kernel-ref-digests.js', digest)

changed = {row['file'] for row in changes}
checked = 0
for file in sorted((parent / 'src').rglob('*')):
    if file.is_file():
        relative = file.relative_to(parent).as_posix()
        if relative not in changed:
            assert file.read_bytes() == (candidate / relative).read_bytes(), relative
            checked += 1
receipt = dict(schema='doppler.route-order-isolated-candidate/v1',
    scope='Private arithmetic experiment, not a production API or qualification change',
    intervention='Ascending expert-ID instead of route-slot F32 summation; separately rounded products retained',
    changes=changes, untouchedSourceFilesVerified=checked,
    shaderEntryDigest=new_digest)
with receipt_path.open('x') as handle:
    handle.write(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt))
