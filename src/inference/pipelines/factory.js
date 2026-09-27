import { enterDiagnosticObserver, resolveDiagnosticObserver } from '../../debug/log.js';
import { getRuleRegistry, enterRuleRegistry } from '../../rules/rule-registry.js';
import { getKernelRegistry, enterKernelRegistry } from '../../gpu/kernels/kernel-configs.js';
import { getStorageShaderSourceScope, runWithShaderSourceScope } from '../../gpu/kernels/shader-source-scope.js';
import { scopePipelineShaders } from './shader-scoped-pipeline.js';
import { releasePipelineContextGlobals } from './context.js';
import { snapshotRuntimeConfig } from '../../config/runtime.js';

export async function createInitializedPipeline(PipelineClass, manifest, contexts = {}) {
  const ruleRegistry = contexts.ruleRegistry ?? getRuleRegistry();
  const kernelRegistry = contexts.kernelRegistry ?? getKernelRegistry();
  const observer = resolveDiagnosticObserver(contexts.observer);
  const pipeline = new PipelineClass();
  const scope = getStorageShaderSourceScope(contexts.storage ?? contexts.storageContext);
  await runWithShaderSourceScope(scope, async () => {
    const restoreObserver = enterDiagnosticObserver(observer);
    let restoreRules;
    let restoreKernels;
    try {
      restoreRules = enterRuleRegistry(ruleRegistry);
      restoreKernels = enterKernelRegistry(kernelRegistry);
      await pipeline.initialize(contexts);
      await pipeline.loadModel(manifest);
      pipeline.runtimeConfig = snapshotRuntimeConfig(pipeline.runtimeConfig);
    } catch (error) {
      try { await pipeline.unload?.(); } catch { /* Preserve the construction failure. */ }
      throw error;
    } finally {
      releasePipelineContextGlobals(pipeline);
      restoreKernels?.();
      restoreRules?.();
      restoreObserver();
    }
  });
  return scopePipelineShaders(pipeline, scope, undefined, { ruleRegistry, kernelRegistry }, observer);
}
