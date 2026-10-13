import { isKernelRegistry, getKernelConfigs, getKernelConfig } from './kernel-registry-contract.js';
import { computeCanonicalSha256 } from '../formats/canonical-hash.js';

/** @param {import('./execution-registry-contract.js').ExecutionRegistries | null | undefined} registries */
export function resolveExecutionRegistries(registries) {
  if (registries == null) return null;
  const { ruleRegistry, kernelRegistry } = registries;
  if ((!Object.isFrozen(ruleRegistry) || typeof ruleRegistry?.selectRuleValue !== 'function'
    || !/^sha256:[0-9a-f]{64}$/.test(ruleRegistry.identity)) || !isKernelRegistry(kernelRegistry)) {
    throw new Error('Execution requires constructed rule and kernel registry instances.');
  }
  return Object.freeze({ ruleRegistry, kernelRegistry,
    identity: Object.freeze({ rules: ruleRegistry.identity, kernels: kernelRegistry.identity }) });
}

/** @param {import('./target-plan.js').TargetPlan} targetPlan
 * @param {import('./execution-registry-contract.js').ResolvedExecutionRegistries | null} registries */
export function assertExecutionRegistriesAccepted(targetPlan, registries) {
  const declared = ('initialExecutionIdentity' in targetPlan ? targetPlan.initialExecutionIdentity?.runtimeEngine?.registries : null) ?? null;
  const observed = registries?.identity ?? null;
  if (computeCanonicalSha256(declared) !== computeCanonicalSha256(observed)) {
    throw new Error('Execution registry extensions must be bound to the accepted TargetPlan initial execution identity.');
  }
  for (const variants of Object.values(registries?.kernelRegistry.validators ?? {})) {
    for (const validator of Object.values(variants)) {
      if (validator.id.startsWith('legacy:')) {
        throw new Error('Capsule execution requires explicitly identified validators; compatibility registrations are not accepted.');
      }
    }
  }
}

/** Auxiliary dispatches are part of the accepted shader closure, even when a
 * projection wrapper orchestrates them rather than separate semantic steps.
 * @param {import('./schema/execution-v1.schema.js').ExecutionV1GraphSchema} graph
 * @param {import('./schema/execution-v1.schema.js').ExecutionV1ExpandedStepSchema[]} steps */
export function assertExecutionMechanismClosure(graph, steps) {
  const configs = Object.values(getKernelConfigs()).flatMap(variants => Object.values(variants));
  const declarations = (graph.mechanismKernels ?? []).map(key => graph.kernels[key]);
  const used = new Set([...steps, ...declarations].map(decl => `${decl.kernel}#${decl.entry}`));
  for (const config of configs) {
    if (!used.has(`${config.shaderFile}#${config.entryPoint}`)) continue;
    const dependencies = config.variantMetadata?.mechanismKernels;
    if (dependencies === undefined) continue;
    if (!Array.isArray(dependencies) || dependencies.length === 0) {
      throw new Error(`Kernel ${config.operation}/${config.variant} has invalid mechanismKernels metadata.`);
    }
    for (const dependency of dependencies) {
      if (!dependency || typeof dependency.operation !== 'string' || typeof dependency.variant !== 'string') {
        throw new Error(`Kernel ${config.operation}/${config.variant} has an invalid mechanism dependency.`);
      }
      const mechanism = getKernelConfig(dependency.operation, dependency.variant);
      const constantsHash = computeCanonicalSha256(mechanism.wgslOverrides ?? {});
      const declared = declarations.some(decl => decl.kernel === mechanism.shaderFile
        && decl.entry === mechanism.entryPoint
        && computeCanonicalSha256(decl.constants ?? {}) === constantsHash);
      if (!declared) {
        throw new Error(`Execution requires mechanism ${dependency.operation}/${dependency.variant} `
          + `in mechanismKernels with its declared shader, entry and specialization constants.`);
      }
    }
  }
}

/** The paired pipeline's wrapper-owned kernels are declared before either Run
 * admission or Rig signing. Gather retains scalar, vector and declared split
 * paths; this validates source reachability without selecting a runtime path.
 * @param {import('./schema/execution-v1.schema.js').ExecutionV1GraphSchema} graph */
export function assertPairedExecutionMechanismClosure(graph) {
  if (!graph || !graph.kernels) throw new Error('[PairedExecution] An explicit execution graph is required.');
  const configs = getKernelConfigs();
  const embeds = (graph.preLayer ?? []).filter(step => step[0] === 'embed');
  const embedding = embeds.length === 1 ? graph.kernels[embeds[0][1]] : null;
  const gather = Object.values(configs.gather).find(config => embedding
    && config.shaderFile === embedding.kernel && config.entryPoint === embedding.entry);
  const gatherMechanisms = gather?.variantMetadata?.pairedMechanismKernels;
  if (!Array.isArray(gatherMechanisms) || !gatherMechanisms.length) {
    throw new Error('[PairedExecution] A declared F32-output embedding gather closure is required.');
  }
  const required = [
    ['cast', 'f32_to_two_f32'], ['cast', 'two_f32_to_f16'],
    ['gated_delta_conv', 'paired'], ['gated_delta_recurrent', 'paired'],
    ['qwen_linear_attention_prepare', 'paired_beta'],
    ['qwen_linear_attention_prepare', 'paired_decay_scale'],
    ['qwen_linear_attention_prepare', 'paired_log_decay'],
    ['qwen_linear_attention_prepare', 'paired_decay'],
    ['split_qg', 'paired'], ['split_qkv', 'paired'], ['rope_precompute', 'paired'],
    ['paired_pointwise', 'sigmoid'], ['paired_pointwise', 'scale'],
    ['logit_finalize', 'f32'],
    ...gatherMechanisms.map(dependency => [dependency.operation, dependency.variant]),
  ];
  const declarations = (graph.mechanismKernels ?? []).map(key => graph.kernels[key]);
  // The embedding's semantic declaration itself is already reachable.
  if (embedding) declarations.push(embedding);
  for (const [operation, variant] of required) {
    const expected = getKernelConfig(operation, variant);
    if (!declarations.some(decl => decl?.kernel === expected.shaderFile && decl.entry === expected.entryPoint
      && typeof decl.digest === 'string' && /^sha256:[a-f0-9]{64}$/.test(decl.digest)
      && computeCanonicalSha256(decl.constants ?? {}) === computeCanonicalSha256(expected.wgslOverrides ?? {}))) {
      throw new Error(`[PairedExecution] Missing pinned mechanism ${operation}/${variant} with exact specialization constants.`);
    }
  }
}
