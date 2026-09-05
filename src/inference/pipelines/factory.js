import { getStorageShaderSourceScope, runWithShaderSourceScope } from '../../gpu/kernels/shader-source-scope.js';
import { scopePipelineShaders } from './shader-scoped-pipeline.js';

export async function createInitializedPipeline(PipelineClass, manifest, contexts = {}) {
  const pipeline = new PipelineClass();
  const scope = getStorageShaderSourceScope(contexts.storage ?? contexts.storageContext);
  await runWithShaderSourceScope(scope, async () => {
    try {
      await pipeline.initialize(contexts);
      await pipeline.loadModel(manifest);
    } catch (error) {
      try { await pipeline.unload?.(); } catch { /* Preserve the construction failure. */ }
      throw error;
    }
  });
  return scopePipelineShaders(pipeline, scope);
}
