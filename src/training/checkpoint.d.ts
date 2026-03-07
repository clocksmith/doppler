export interface CheckpointStoreOptions {
  dbName?: string;
  storeName?: string;
  version?: number;
  nodeDir?: string;
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
  forceResumeSource?: string;
  forceResumeOperator?: string | null;
}

export declare function saveCheckpoint(
  key: string,
  data: unknown,
  options?: CheckpointStoreOptions
): Promise<{
  key: string;
  path: string | null;
  metadata: Record<string, unknown>;
  data: unknown;
}>;

export declare function loadCheckpoint(
  key: string,
  options?: CheckpointStoreOptions
): Promise<unknown | null>;
