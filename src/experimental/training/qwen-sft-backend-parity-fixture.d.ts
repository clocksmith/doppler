export interface QwenSftBackendParityTensor {
  shape: number[];
  data: number[];
}

export interface QwenSftBackendParityAdapter {
  rank: number;
  alpha: number;
  A: QwenSftBackendParityTensor;
  B: QwenSftBackendParityTensor;
}

export interface QwenSftBackendParityFixture {
  artifactType: 'qwen_sft_backend_parity_fixture';
  schemaVersion: 1;
  precisionContract: Record<string, string | number>;
  model: Record<string, number>;
  layer: Record<string, number | boolean>;
  tokenIds: number[];
  targets: number[];
  unmaskedTargets: number[];
  frozen: Record<string, QwenSftBackendParityTensor>;
  adapters: Record<string, QwenSftBackendParityAdapter>;
  optimizer: {
    type: 'adamw';
    lr: number;
    beta1: number;
    beta2: number;
    eps: number;
    weightDecay: number;
  };
  claimBoundary: string;
}

export declare function createQwenSftBackendParityFixture(options?: {
  rank?: number;
  alpha?: number;
}): QwenSftBackendParityFixture;
