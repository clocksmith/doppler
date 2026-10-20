import type { NativeQwenLoRACheckpoint, NativeQwenLoRACheckpointTensor } from './native-qwen-lora.js';

export interface LoRATrainingSample {
  inputIds: readonly number[];
  targetIds: readonly number[];
  supervisedTokenCount: number;
}
export interface LoRATrainingAssignment {
  participantId: string;
  sampleHash: string;
  supervisedTokenCount: number;
}
export interface LoRATrainingRoundInput {
  jobId: string;
  roundId: string;
  baseManifestHash: string;
  trainingConfigHash: string;
  checkpoint: NativeQwenLoRACheckpoint;
  assignments: LoRATrainingAssignment[];
}
export interface LoRATrainingRound extends LoRATrainingRoundInput {
  schema: 'doppler.lora-training-round/v1';
  checkpointHash: string;
  roundHash: string;
}
export interface LoRAGradientContribution {
  schema: 'doppler.lora-gradient-contribution/v1';
  roundHash: string;
  participantId: string;
  sampleHash: string;
  supervisedTokenCount: number;
  loss: number;
  gradients: { A: NativeQwenLoRACheckpointTensor; B: NativeQwenLoRACheckpointTensor };
}
export declare function createLoRATrainingRound(input: LoRATrainingRoundInput): LoRATrainingRound;
export declare function validateLoRATrainingRound(round: LoRATrainingRound): void;
export declare function validateLoRAContributions(round: LoRATrainingRound, contributions: LoRAGradientContribution[]): LoRAGradientContribution[];

export declare function describeLoRATrainingSample(participantId: string, sample: LoRATrainingSample): LoRATrainingAssignment;
