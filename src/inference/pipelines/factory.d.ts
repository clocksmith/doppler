type InitializablePipeline = {
  initialize(contexts?: Record<string, unknown>): Promise<void>;
  loadModel(manifest: unknown): Promise<void>;
};

export declare function createInitializedPipeline<T extends InitializablePipeline>(
  PipelineClass: new () => T,
  manifest: unknown,
  contexts?: Record<string, unknown>
): Promise<T>;
