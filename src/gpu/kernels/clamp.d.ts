import type { CommandRecorder } from '../command-recorder.js';
import type { Tensor } from '../tensor.js';

export interface ClampOptions {
  count?: number;
}

export declare function runClamp(
  input: Tensor,
  minValue: number,
  maxValue: number,
  options?: ClampOptions
): Promise<Tensor>;

export declare function recordClamp(
  recorder: CommandRecorder,
  input: Tensor,
  minValue: number,
  maxValue: number,
  options?: ClampOptions
): Promise<Tensor>;

