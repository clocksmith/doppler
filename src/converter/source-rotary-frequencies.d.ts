import type { SourceTensorDescriptor } from './source-tensor-rules.js';

export interface SourceRotaryFrequencyPolicy {
  match: string;
  expectedMatches: number;
}

export declare function readSourceRotaryFrequencies(
  tensors: SourceTensorDescriptor[],
  policy: SourceRotaryFrequencyPolicy | null,
  readTensor: (tensor: SourceTensorDescriptor) => Promise<ArrayBuffer>
): Promise<number[] | null>;
