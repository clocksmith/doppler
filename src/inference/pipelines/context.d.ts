export type PipelineContextOptions = {
  applySharedDebug?: boolean;
  assignGpuContext?: boolean;
  assignUseGPU?: boolean;
  assignMemoryContext?: boolean;
  assignStorageContext?: boolean;
  assignBaseUrl?: boolean;
  assignProgress?: boolean;
};

export declare function applyPipelineContexts(
  target: Record<string, unknown>,
  contexts?: Record<string, unknown>,
  options?: PipelineContextOptions
): {
  runtimeConfig: Record<string, unknown>;
  sharedDebug: Record<string, unknown> | null | undefined;
};
