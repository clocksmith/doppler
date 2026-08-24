import type { Tensor } from '../../../gpu/tensor.js';

export interface VisionMergerWeights {
  mergerMlp0Weight?: Tensor;
  mergerMlp0Bias?: Tensor;
  mergerMlp2Weight?: Tensor;
  mergerMlp2Bias?: Tensor;
  'visual.merger.mlp.0.weight'?: Tensor;
  'visual.merger.mlp.0.bias'?: Tensor;
  'visual.merger.mlp.2.weight'?: Tensor;
  'visual.merger.mlp.2.bias'?: Tensor;
}

export declare function spatialMergeProject(params: {
  input: GPUBuffer;
  gridHeight: number;
  gridWidth: number;
  hiddenSize: number;
  outHiddenSize: number;
  spatialMergeSize: number;
  weights: VisionMergerWeights;
}): Promise<GPUBuffer>;
