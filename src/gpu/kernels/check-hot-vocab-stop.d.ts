import type { CommandRecorder } from '../command-recorder.js';

export interface CheckHotVocabStopParams {
  sampledTokenBuffer: GPUBuffer;
  nextInputTokenBuffer: GPUBuffer;
  hotTokenIndexMapBuffer: GPUBuffer;
  hotTokenSentinel: number;
  tokenIndex?: number;
  shouldStopBuffer?: GPUBuffer;
  eosTokenId: number;
  maxTokens: number;
  currentPos: number;
}

export declare function recordCheckHotVocabStop(
  recorder: CommandRecorder,
  params: CheckHotVocabStopParams
): GPUBuffer;
