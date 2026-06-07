export interface RepPenaltyOptions {
  vocabSize: number;
  historyCount: number;
  penalty: number;
  batchCount: number;
  batchOffset: number;
  logitsDtype: 'f16' | 'f32';
}

export declare function selectRepPenaltyVariant(useF16: boolean): 'default' | 'default_f16';

export declare function recordRepPenalty(
  recorder: {
    device: GPUDevice;
    beginComputePass(label: string): GPUComputePassEncoder;
  },
  logitsBuffer: GPUBuffer,
  historyBuffer: GPUBuffer,
  batchTokensBuffer: GPUBuffer,
  options: RepPenaltyOptions
): Promise<void>;
