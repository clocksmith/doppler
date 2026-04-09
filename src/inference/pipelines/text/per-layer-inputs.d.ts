import type { Tensor } from '../../../gpu/tensor.js';
import type { LayerContext } from './types.js';
import type { ExecutionV1PerLayerInputsSessionSchema } from '../../../config/schema/execution-v1.schema.js';

export interface PleBufferCache {
  sliceBuffers: (GPUBuffer | null)[] | null;
  gatherSliceBuffers?: (GPUBuffer | null)[] | null;
}

export interface PrefetchedPleRow {
  tokenId: number;
  row: Float32Array;
}

export declare function preparePerLayerInputs(
  tokenIds: number[] | Uint32Array | GPUBuffer,
  inputEmbedsTensor: Tensor,
  context: LayerContext,
  options?: {
    numTokens?: number;
    indexOffset?: number;
    perLayerTokenIds?: number[] | Uint32Array | GPUBuffer | null;
    perLayerIndexOffset?: number;
    pleCache?: PleBufferCache | null;
    prefetchedRow?: PrefetchedPleRow | null;
  }
): Promise<(GPUBuffer | null)[] | null>;

export declare function createPleBufferCache(numLayers: number, sliceBytes: number): PleBufferCache;

export declare function destroyPleBufferCache(cache: PleBufferCache | null | undefined): void;

export declare function destroyPleRuntimeCache(perLayerInputWeights: object | null | undefined): void;

export declare function prefetchPerLayerRow(
  tokenId: number,
  embedTokensPerLayer: unknown,
  totalPerLayerHiddenSize: number,
  sessionConfig?: ExecutionV1PerLayerInputsSessionSchema | null
): Promise<PrefetchedPleRow | null> | null;

export declare function hasRangeBackedPerLayerInputEmbeddings(
  context: Pick<LayerContext, 'config' | 'weights'>
): boolean;

export declare function hasGpuSplitPerLayerInputEmbeddings(
  context: Pick<LayerContext, 'config' | 'weights'>
): boolean;

export declare function scalePerLayerProjectionNormWeights(
  weight: unknown,
  combineScale: number,
  rmsNormWeightOffset?: boolean
): Float32Array | null;

export declare function inferPleProjectionNormDtype(
  weight: unknown,
  hiddenSizePerLayerInput: number
): 'f16' | 'f32';

export declare function ensurePleScaledProjectionNormWeight(
  context: Pick<LayerContext, 'config' | 'weights' | 'weightConfig' | 'debugFlags'>,
  combineScale?: number
): Promise<Tensor | null>;

export declare function ensurePleGpuSplitTablesRuntime(
  context: Pick<LayerContext, 'config' | 'weights' | 'perLayerInputsSession' | 'debugFlags'>
): Promise<unknown[] | null>;

export declare function ensurePleGpuHotVocabularyRuntime(
  context: Pick<LayerContext, 'config' | 'weights' | 'perLayerInputsSession' | 'debugFlags'> & {
    tokenizer?: {
      getHotTokenIds?(limit: number): number[] | null;
      getSpecialTokens?(): Record<string, number | null | undefined>;
    } | null;
    seedTokenIds?: number[] | null;
  }
): Promise<object | null>;

export declare function getPleHotVocabularyRuntime(
  context: Pick<LayerContext, 'weights'>
): object | null;

export declare function createPerLayerInputTensor(
  buffer: GPUBuffer,
  numTokens: number,
  hiddenSizePerLayerInput: number,
  activationDtype: 'f16' | 'f32'
): Tensor;
