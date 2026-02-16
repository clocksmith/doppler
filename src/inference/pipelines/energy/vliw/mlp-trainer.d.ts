import type { VliwMlpModel } from './mlp.js';
import type { Tensor } from '../../../../gpu/tensor.js';

export interface VliwMlpTrainerConfig {
  lr?: number;
  beta1?: number;
  beta2?: number;
  eps?: number;
}

export interface VliwMlpAdamState {
  lr: number;
  beta1: number;
  beta2: number;
  eps: number;
  step: number;
}

export interface VliwMlpTrainer {
  inputSize: number;
  hiddenSize: number;
  opt: VliwMlpAdamState;
  moments: {
    w1: { m: Tensor; v: Tensor };
    w2: { m: Tensor; v: Tensor };
  };
}

export declare function createMlpTrainer(
  inputSize: number,
  hiddenSize: number,
  config?: VliwMlpTrainerConfig
): VliwMlpTrainer;

export declare function mlpTrainDistillStep(
  trainer: VliwMlpTrainer,
  studentMlp: VliwMlpModel,
  featureBatch: Float32Array,
  teacherMlp: VliwMlpModel
): Promise<VliwMlpModel>;

export declare function disposeMlpTrainer(trainer: VliwMlpTrainer): void;
