import type { CommandRecorder } from '../../../../gpu/command-recorder.js';
import type { Tensor } from '../../../../gpu/tensor.js';
import type { LogitsConfig, LogitsWeights } from '../logits/types.js';
export declare function mixHyperConnectionLogitsInput(recorder: CommandRecorder | null, hidden: GPUBuffer | Float32Array,
  numTokens: number, weights: LogitsWeights, config: LogitsConfig): Promise<Tensor>;
