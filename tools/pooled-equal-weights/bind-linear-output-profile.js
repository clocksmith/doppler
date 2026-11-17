// Offline peer preparation: bind selected packed programs to this manifest.
// This changes an explicit diagnostic profile, never runtime dispatch or weights.
import assert from 'node:assert/strict';
import { pathToFileURL } from 'node:url';
import path from 'node:path';

export async function bindLinearOutputProfile({ profile, manifest, dopplerRoot, requiredKernelPath = null }) {
  assert(path.isAbsolute(dopplerRoot), 'Explicit selected Doppler source root required');
  const module = relative => import(pathToFileURL(path.join(dopplerRoot, 'src', relative)));
  const [{ getLayerSteps, getKernelPathStepConfig }, { ggufBlockFormat }, aliases, { compileWeightRepresentationPlan },
    { KERNEL_REF_CONTENT_DIGESTS }] = await Promise.all([
    module('config/kernel-path-loader.js'), module('formats/gguf/block-storage.js'),
    module('loader/layer-tensor-aliases.js'), module('loader/weight-representation-plan.js'),
    module('config/kernels/kernel-ref-digests.js'),
  ]);
  const original = profile.runtime?.inference?.kernelPath;
  assert(original && Number.isSafeInteger(manifest.architecture?.numLayers), 'Explicit profile and model geometry required');
  const result = structuredClone(profile), bindings = [], declarations = [], semanticBindings = [], groups = new Map();
  function bindHyperConnectionSteps(steps, location) {
    if (manifest.inference?.hyperConnections == null) return steps;
    // HC owns its gated residual and norm. Its MoE caller supplies neither of
    // processFFNStandard's pending inputs; adding them would change model math.
    return steps.filter(step => {
      if (step.op !== 'moe_norm_router') return true;
      semanticBindings.push({ ...location, op: step.op, before: structuredClone(step), after: null,
        reason: 'Hyper-connection caller supplies its own normalized branch input' });
      return false;
    }).map(step => {
      if (step.op !== 'moe_shared_slot_down') return step;
      const config = getKernelPathStepConfig(step.op, step, location.phase);
      assert(config, 'Shared-slot down must be registered in the selected source');
      if (config.variantMetadata?.outerResidual !== true) return step;
      assert.equal(step.kernel, 'moe_shared_slot_down_pair_combine_residual_f32.wgsl',
        'Unsupported outer-residual fusion in hyper-connection profile');
      const after = { ...step, kernel: 'moe_shared_slot_down_pair_combine_f32.wgsl', entry: 'main' };
      const digest = KERNEL_REF_CONTENT_DIGESTS[`${after.kernel}#${after.entry}`];
      assert(digest, 'Non-residual shared-slot shader digest missing from selected source');
      after.digest = `sha256:${digest}`;
      assert.equal(getKernelPathStepConfig(step.op, after, location.phase)?.variantMetadata?.outerResidual, false);
      semanticBindings.push({ ...location, op: step.op, before: structuredClone(step), after: structuredClone(after),
        reason: 'Hyper-connection executor owns the outer gated residual' });
      return after;
    });
  }
  function includeMissing(selected, required, location) {
    const steps = structuredClone(selected), present = new Set(steps.map(step => step.op));
    for (const step of required) {
      if (present.has(step.op)) continue;
      steps.push(structuredClone(step)); present.add(step.op);
      declarations.push({ ...location, op: step.op, kernel: step.kernel });
    }
    return steps;
  }
  for (let layer = 0; layer < manifest.architecture.numLayers; layer++) {
    let name;
    for (const prefix of aliases.LAYER_PREFIXES(layer)) {
      name = aliases.LINEAR_ATTN_SUFFIXES.outProj.map(suffix => `${prefix}.${suffix}`).find(key => manifest.tensors[key]);
      if (name) break;
    }
    const programs = {};
    for (const phase of ['prefill', 'decode']) {
      const location = { layer, phase };
      const steps = bindHyperConnectionSteps(includeMissing(getLayerSteps(original, layer, phase),
        requiredKernelPath ? getLayerSteps(requiredKernelPath, layer, phase) : [], location), location);
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
  if (requiredKernelPath) for (const section of ['preLayer', 'postLayer']) {
    result.runtime.inference.kernelPath[section] = includeMissing(original[section] ?? [],
      requiredKernelPath[section] ?? [], { section });
  }
  const locations = new Map(Object.entries(manifest.tensors).map(([name, tensor]) => [name,
    { ...tensor, shardIndex: tensor.shardIndex ?? tensor.shard }]));
  const plan = compileWeightRepresentationPlan(locations, manifest, [result.runtime.inference.kernelPath],
    { useFusedQ4K: false, q4kLayout: 'row', keepF32Weights: true });
  return { profile: result, bindings, declarations, semanticBindings, representationTensorsChecked: Object.keys(plan).length,
    scope: 'Explicit offline encoding, operation and HC fusion binding; unchanged weights and shader sources; no model or performance qualification' };
}
