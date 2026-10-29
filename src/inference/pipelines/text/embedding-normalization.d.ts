import type { CommandRecorder } from '../../../gpu/command-recorder.js';
import type { Tensor } from '../../../gpu/tensor.js';
import type { WeightlessEmbeddingNormalization } from './embedding-contract.js';

export function finalizeEmbeddingOutput(
  tensor: Tensor,
  normalization: WeightlessEmbeddingNormalization | null,
  options: {
    recorder?: CommandRecorder | null;
    numTokens: number;
    hiddenSize: number;
    outputBuffer?: GPUBuffer | null;
    probeStage?: string;
    debugProbes?: unknown;
    operatorDiagnostics?: unknown;
  }
): Promise<Tensor>;
