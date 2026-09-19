import type { PipelineState } from '../state.js';
import type { GenerateOptions, LayerContext, PipelineStats } from '../types.js';
import type { CommandRecorder } from '../../../../gpu/command-recorder.js';
import type { DecodeBufferManager } from '../../../decode-buffers.js';

export { FinitenessError, sumProfileTimings, decodeStep, decodeStepLogits,
  readMappedBufferCopy, readSampledTokenFromStagingBuffer, shouldUseFusedDecodeSampling,
} from '../generator-steps.js';

export declare const UNKNOWN_TOKEN_TEXT: '<unknown>';
export declare const FINITENESS_RESET_WORDS: Uint32Array;
export declare function getTokenTextOrUnknown(
  tokenizer: { decode(ids: number[], skipSpecial?: boolean, cleanup?: boolean): string } | null,
  tokenId: number
): string;
export declare function isOwnedDecodeBuffer(
  candidate: GPUBuffer | null, decodeHiddenBuffer: GPUBuffer | null, decodeAltBuffer: GPUBuffer | null
): boolean;
export declare function releasePerLayerInputBuffer(
  buffer: GPUBuffer | null, recorder: CommandRecorder | null,
  decodeBuffers: Pick<DecodeBufferManager, 'ownsBuffer'> | null,
  pleCache?: { ownedBuffers: Set<GPUBuffer> } | null
): void;
export declare function getReusableSampleReadbackBuffer(
  state: { sampleReadbackBuffer?: GPUBuffer | null }, device: GPUDevice, size: number
): GPUBuffer;
export declare function getEffectiveActivationDtype(
  state: Pick<PipelineState, 'runtimeConfig'>,
  opts: { executionPlan?: { activationDtype: 'f16' | 'f32' } }
): 'f16' | 'f32';
export declare function schedulePlePrefetchForToken(state: PipelineState, tokenId: number): void;
export declare function shouldLogProfileStep(state: Pick<PipelineState, 'runtimeConfig'>, step: number): boolean;
export declare function recordDecodeProfileStep(
  state: { stats: PipelineStats }, entry: NonNullable<PipelineStats['decodeProfileSteps']>[number]
): void;
export declare function createDecodeRecorder(
  state: PipelineState, opts: GenerateOptions & { executionObserver?: boolean }
): CommandRecorder | undefined;
export declare function submitDecodeRecorderProfile(
  state: PipelineState, opts: GenerateOptions & { executionObserver?: boolean },
  recorder: CommandRecorder | undefined, profileLabel: string
): Promise<void>;
export declare function runDecodeLayers(
  state: PipelineState, tokenId: number, opts: GenerateOptions,
  helpers: {
    buildLayerContext(recorder: CommandRecorder | undefined, isDecode: boolean, debugLayers: unknown, plan: unknown): LayerContext;
    debugCheckBuffer?(buffer: GPUBuffer, label: string, numTokens: number, expectedDim?: number): Promise<void>;
    releaseSharedAttentionState?(state: unknown, recorder: CommandRecorder | null): void;
  }
): Promise<{
  hiddenStates: import('../../../../gpu/tensor.js').Tensor;
  decodeHiddenBuffer: GPUBuffer;
  decodeAltBuffer: GPUBuffer;
  debugCheckBuffer: ((buffer: GPUBuffer, label: string, numTokens: number, expectedDim?: number) => Promise<void>) | undefined;
  context: LayerContext;
}>;
