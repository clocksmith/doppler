export interface VliwMlpModel {
  inputSize: number;
  hiddenSize: number;
  w1: Float32Array;
  w2: Float32Array;
}

export declare function createMlp(
  inputSize: number,
  hiddenSize: number,
  seed: number
): VliwMlpModel;

export declare function mlpParamCount(mlp: VliwMlpModel | null): number;

export declare function mlpToFlat(mlp: VliwMlpModel): Float32Array;

export declare function mlpFromFlat(
  flat: Float32Array,
  inputSize: number,
  hiddenSize: number
): VliwMlpModel;

export declare function mlpForward(
  mlp: VliwMlpModel,
  features: Float32Array
): number;

export declare function mlpForwardBatch(
  mlp: VliwMlpModel,
  featureRows: Float32Array,
  numRows: number
): Float32Array;

export declare function perturbMlp(
  mlp: VliwMlpModel,
  rng: () => number,
  count: number,
  scale: number
): VliwMlpModel;

