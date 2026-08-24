import { clearShaderCaches, getShaderCacheStats } from './shader-cache.js';
import { clearPipelineCaches, getPipelineCacheStats } from './pipeline-cache.js';


export function clearKernelCaches() {
  clearShaderCaches();
  clearPipelineCaches();
}


export function clearPipelineCache() {
  clearKernelCaches();
}


export function getCacheStats() {
  const shaderStats = getShaderCacheStats();
  const pipelineStats = getPipelineCacheStats();
  return {
    pipelines: pipelineStats.pipelines,
    shaders: shaderStats.sources,
    shaderModules: shaderStats.modules,
    bindGroupLayouts: pipelineStats.bindGroupLayouts,
    pipelineLayouts: pipelineStats.pipelineLayouts,
  };
}
