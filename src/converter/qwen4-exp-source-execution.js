import profile from '../config/conversion/qwen3/qwen-3-8-flash-next-text-q4k-ehf16-af32.json' with { type: 'json' };
import policy from '../config/conversion/qwen3/qwen4exp-gguf-source-policy.json' with { type: 'json' };
import registry from '../config/kernels/registry.json' with { type: 'json' };
import { KERNEL_REF_CONTENT_DIGESTS } from '../config/kernels/kernel-ref-digests.js';
import { ggufBlockFormat, isGgufBlockSourceDtype } from '../formats/gguf/block-storage.js';
import { selectRuleValue } from '../rules/rule-registry.js';
import { cloneJsonValue } from '../formats/clone-json.js';

/** Compile the GGUF source's physical projection types, never the HF Q4K recipe.
 * @type {typeof import('./qwen4-exp-source-execution.js').createQwen4ExpSourceExecution} */
export function createQwen4ExpSourceExecution(tensors, numLayers) {
  if (!Array.isArray(tensors) || tensors.length === 0 || !Number.isInteger(numLayers) || numLayers < 1) {
    throw new Error('Qwen4Exp source execution requires tensor storage facts and a positive layer count.');
  }
  const byName = new Map(tensors.map(t => [t.name, t]));
  if (byName.size !== tensors.length) throw new Error('Duplicate Qwen4Exp source tensor names.');
  const graph = /** @type {import('../config/schema/execution-v1.schema.js').ExecutionV1GraphSchema} */ (/** @type {unknown} */ (cloneJsonValue(profile.execution)));
  for (const spec of Object.values(graph.kernels)) {
    const digest = KERNEL_REF_CONTENT_DIGESTS[`${spec.kernel}#${spec.entry}`];
    if (!digest) throw new Error(`Missing family kernel digest for ${spec.kernel}.`);
    spec.digest = `sha256:${digest}`;
  }
  const operations = /** @type {import('../config/schema/kernel-registry.schema.js').KernelRegistrySchema} */ (/** @type {unknown} */ (registry)).operations;
  const floats = /** @type {Record<string, string>} */ (policy.floatVariants);
  const phases = /** @type {Record<string, number>} */ (policy.phases);
  /** @param {string} operation @param {string} variant */
  const install = (operation, variant) => {
    const spec = operations[operation]?.variants?.[variant];
    if (!spec) throw new Error(`Unsupported Qwen4Exp source kernel ${operation}:${variant}.`);
    const key = `source_${operation}_${variant}`;
    const digest = KERNEL_REF_CONTENT_DIGESTS[`${spec.wgsl}#${spec.entryPoint}`];
    if (!digest) throw new Error(`Missing source kernel digest for ${key}.`);
    graph.kernels[key] = { kernel: spec.wgsl, entry: spec.entryPoint,
      digest: `sha256:${digest}`, ...(spec.wgslOverrides ? { constants: cloneJsonValue(spec.wgslOverrides) } : {}) };
    return key;
  };
  /** @param {string} role @param {string} name @param {string} phase
   * @returns {import('../config/schema/execution-v1.schema.js').ExecutionV1StepTuple} */
  const projection = (role, name, phase) => {
    const tensor = byName.get(name);
    if (!tensor) throw new Error(`Missing Qwen4Exp projection ${name}.`);
    const variant = isGgufBlockSourceDtype(tensor.dtype)
      ? selectRuleValue('kernels', 'matmul', 'q40Variant', {
        M: phases[phase], weightDtype: ggufBlockFormat(tensor.dtype).variant,
      }) : floats[tensor.dtype];
    if (!variant) throw new Error(`Unsupported Qwen4Exp projection ${name} dtype ${tensor.dtype}.`);
    return [role, install('matmul', variant), name];
  };
  for (const phase of /** @type {const} */ (['decode', 'prefill'])) {
    // Retain non-projection semantics. Each physical projection is then bound
    // explicitly per layer, including mixed attention and shared-expert types.
    const unweighted = graph[phase].filter(step => Array.isArray(step) && step.length < 3);
    /** @type {Map<string, import('../config/schema/execution-v1.schema.js').ExecutionV1LayerGroupSchema>} */
    const groups = new Map();
    for (let layer = 0; layer < numLayers; layer++) {
      const steps = [
      ...Object.entries(policy.layerRoles).flatMap(([role, suffix]) => {
        const name = `blk.${layer}.${suffix}`;
        if (!byName.has(name)) return [];
        const step = projection(role, name, phase);
        step[2] = name.replace(`blk.${layer}.`, 'blk.{L}.');
        return [step];
      }),
      ];
      const key = JSON.stringify(steps);
      let group = groups.get(key);
      if (!group) { group = { layers: [], steps }; groups.set(key, group); }
      group.layers.push(layer);
    }
    graph[phase] = [...cloneJsonValue(unweighted), ...groups.values()];
  }
  graph.postLayer = Object.entries(policy.rootRoles).map(([role, name]) =>
    projection(role, name, role === 'lm_head_prefill' ? 'prefill' : 'decode'));
  graph.postLayer.push(['sample', 'sample']);
  const embedding = byName.get(policy.embedding);
  if (!embedding || !isGgufBlockSourceDtype(embedding.dtype)) {
    throw new Error('Qwen4Exp GGUF source execution requires a supported packed token embedding.');
  }
  graph.preLayer = [['embed', install('gather', ggufBlockFormat(embedding.dtype).variant), policy.embedding]];
  const pleTable = byName.get(policy.pleRowDecode.tensor);
  if (pleTable) {
    if (pleTable.dtype !== policy.pleRowDecode.dtype) throw new Error('Unsupported Qwen4Exp source PLE row storage.');
    const key = install(policy.pleRowDecode.operation, policy.pleRowDecode.variant);
    graph.kernels.ple_row_decode = graph.kernels[key];
    delete graph.kernels[key];
  }
  return graph;
}
