export declare function clearKernelCaches(): void;
export declare function clearPipelineCache(): void;
export declare function getCacheStats(): {
  pipelines: number;
  shaders: number;
  shaderModules: number;
  bindGroupLayouts: number;
  pipelineLayouts: number;
};
