import type { Tensor } from '../../../gpu/tensor.js';
import type { CommandRecorder } from '../../../gpu/command-recorder.js';

export interface SanaTimestepState {
  modulation: Tensor;
  embeddedTimestep: Tensor;
}

export interface SanaTransformerOptions {
  recorder?: CommandRecorder | null;
}

export declare function buildSanaTimestepConditioning(
  timestep: number,
  guidanceScale: number,
  weightsEntry: any,
  config: any,
  runtime: any,
  options?: SanaTransformerOptions
): Promise<SanaTimestepState>;

export declare function projectSanaContext(
  context: Tensor,
  attentionMask: Uint32Array | null | undefined,
  weightsEntry: any,
  config: any,
  runtime: any,
  options?: SanaTransformerOptions
): Promise<Tensor>;

export declare function runSanaTransformer(
  latents: Tensor,
  context: Tensor,
  timeState: SanaTimestepState,
  weightsEntry: any,
  modelConfig: any,
  runtime: any,
  options?: SanaTransformerOptions
): Promise<Tensor>;

