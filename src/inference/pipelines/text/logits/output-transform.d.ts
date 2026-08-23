import type { CommandRecorder } from '../../../../gpu/kernel-selector.js';
import type { Tensor } from '../../../../gpu/tensor.js';
import type { LogitsConfig } from './types.js';

export function finalizeLogitOutputTensor(
  tensor: Tensor,
  config: LogitsConfig,
  options: {
    recorder?: CommandRecorder | null;
    numTokens: number;
    vocabSize: number;
    operatorDiagnostics?: unknown;
  }
): Promise<Tensor>;
