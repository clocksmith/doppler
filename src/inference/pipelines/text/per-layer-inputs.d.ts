import type { Tensor } from '../../../gpu/tensor.js';
import type { LayerContext } from './types.js';

export interface PleBufferCache {
  sliceBuffers: (GPUBuffer | null)[] | null;
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
  totalPerLayerHiddenSize: number
): Promise<PrefetchedPleRow | null> | null;

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

export declare function createPerLayerInputTensor(
  buffer: GPUBuffer,
  numTokens: number,
  hiddenSizePerLayerInput: number,
  activationDtype: 'f16' | 'f32'
): Tensor;
