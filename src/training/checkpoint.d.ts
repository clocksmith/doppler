export interface CheckpointStoreOptions {
  dbName?: string;
  storeName?: string;
  version?: number;
  configHash?: string;
  datasetHash?: string;
  tokenizerHash?: string;
  optimizerHash?: string;
  runtimePresetId?: string;
  kernelPathId?: string;
  environmentMetadata?: unknown;
  buildProvenance?: Record<string, unknown> | null;
  priorCheckpointHash?: string;
  expectedMetadata?: Record<string, unknown>;
  forceResume?: boolean;
  forceResumeReason?: string;
}

export declare function saveCheckpoint(
  key: string,
  data: unknown,
  options?: CheckpointStoreOptions
): Promise<void>;

export declare function loadCheckpoint(
  key: string,
  options?: CheckpointStoreOptions
): Promise<unknown | null>;
