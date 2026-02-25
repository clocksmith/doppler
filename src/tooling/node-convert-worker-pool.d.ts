export interface WorkerTransformResult {
  tensorData: Uint8Array;
  outDtype: string | null;
  outLayout: string | null;
}

export declare class NodeConvertWorkerPool {
  constructor(options: { size: number });
  get size(): number;
  transformTensor(
    tensor: Record<string, unknown>,
    tensorData: Uint8Array,
    transformContext?: Record<string, unknown> | null
  ): Promise<WorkerTransformResult>;
  close(): Promise<void>;
}
