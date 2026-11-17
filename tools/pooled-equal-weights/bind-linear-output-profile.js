// Offline peer preparation: bind selected packed programs to this manifest.
// This changes an explicit diagnostic profile, never runtime dispatch or weights.
import assert from 'node:assert/strict';
import { pathToFileURL } from 'node:url';
import path from 'node:path';

export async function bindLinearOutputProfile({ profile, manifest, dopplerRoot }) {
  assert(path.isAbsolute(dopplerRoot), 'Explicit selected Doppler source root required');
  const module = relative => import(pathToFileURL(path.join(dopplerRoot, 'src', relative)));
  const [{ getLayerSteps, getKernelPathStepConfig }, { ggufBlockFormat }, aliases, { compileWeightRepresentationPlan }] = await Promise.all([
    module('config/kernel-path-loader.js'), module('formats/gguf/block-storage.js'),
    module('loader/layer-tensor-aliases.js'), module('loader/weight-representation-plan.js'),
  ]);
  const original = profile.runtime?.inference?.kernelPath;
  assert(original && Number.isSafeInteger(manifest.architecture?.numLayers), 'Explicit profile and model geometry required');
  const result = structuredClone(profile), bindings = [], groups = new Map();
  for (let layer = 0; layer < manifest.architecture.numLayers; layer++) {
    let name;
    for (const prefix of aliases.LAYER_PREFIXES(layer)) {
      name = aliases.LINEAR_ATTN_SUFFIXES.outProj.map(suffix => `${prefix}.${suffix}`).find(key => manifest.tensors[key]);
      if (name) break;
    }
    const programs = {};
    for (const phase of ['prefill', 'decode']) {
      const steps = structuredClone(getLayerSteps(original, layer, phase));
      if (name) {
        const format = ggufBlockFormat(manifest.tensors[name].dtype);
        assert(['q4_0', 'q4_1', 'q8_0'].includes(format.variant), 'Unsupported linear output encoding');
        const selected = steps.filter(step => step.op === 'linear_out_proj');
        assert.equal(selected.length, 1, 'One explicit linear output program required');
        const step = selected[0];
        assert(['matmul_gguf_pair_f32.wgsl', 'matmul_q4_0_tile64.wgsl'].includes(step.kernel)
          && step.constants, 'Selected shader does not declare the packed linear output contract');
        const before = { ...step.constants };
        Object.assign(step.constants, { BLOCK_VALUES: format.values, BLOCK_BYTES: format.bytes, QUANT_FORMAT: format.code });
        assert.equal(getKernelPathStepConfig('matmul', step, phase)?.weightDtype, format.variant,
          'Packed linear output program is absent from the selected kernel registry');
        if (JSON.stringify(before) !== JSON.stringify(step.constants)) bindings.push({ layer, phase, tensor: name,
          dtype: manifest.tensors[name].dtype, before, after: { ...step.constants } });
      }
      programs[phase] = { steps };
    }
    const key = JSON.stringify(programs);
    if (!groups.has(key)) groups.set(key, { layers: [], ...programs });
    groups.get(key).layers.push(layer);
  }
  result.runtime.inference.kernelPath.layerOverrides = [...groups.values()];
  const locations = new Map(Object.entries(manifest.tensors).map(([name, tensor]) => [name,
    { ...tensor, shardIndex: tensor.shardIndex ?? tensor.shard }]));
  const plan = compileWeightRepresentationPlan(locations, manifest, [result.runtime.inference.kernelPath],
    { useFusedQ4K: false, q4kLayout: 'row', keepF32Weights: true });
  return { profile: result, bindings, representationTensorsChecked: Object.keys(plan).length,
    scope: 'Explicit offline linear-output encoding binding; unchanged weights/shaders; no model or performance qualification' };
}
