import { getStorageShaderSourceScope, runWithShaderSourceScope } from '../../gpu/kernels/shader-source-scope.js';
import { scopePipelineShaders } from './shader-scoped-pipeline.js';
import { releasePipelineContextGlobals } from './context.js';
import { snapshotRuntimeConfig } from '../../config/runtime.js';

export async function createInitializedPipeline(PipelineClass, manifest, contexts = {}) {
  const pipeline = new PipelineClass();
  const scope = getStorageShaderSourceScope(contexts.storage ?? contexts.storageContext);
  await runWithShaderSourceScope(scope, async () => {
    try {
      await pipeline.initialize(contexts);
      await pipeline.loadModel(manifest);
      pipeline.runtimeConfig = snapshotRuntimeConfig(pipeline.runtimeConfig);
    } catch (error) {
      try { await pipeline.unload?.(); } catch { /* Preserve the construction failure. */ }
      throw error;
    } finally {
      releasePipelineContextGlobals(pipeline);
    }
  });
  return scopePipelineShaders(pipeline, scope);
}
