import type { LayerLoaderContext } from './layer-loader.js';
import type { ResolvedHyperConnections, ResolvedIndexedAttention } from '../config/schema/hyper-connection-decoder.schema.js';
import type { ResolvedNgramEmbedding } from '../config/schema/ngram-embedding.schema.js';
import type { GatedResidualWeights } from '../inference/pipelines/text/hyper-connection/decoder-layer.js';
import type { PleLayerWeights } from '../inference/pipelines/text/ngram/ple-layer.js';
import type { Tensor } from '../gpu/tensor.js';
import type { WeightBuffer } from '../gpu/weight-buffer.js';

export interface MechanismLoaderContext {
  tensorLocations: LayerLoaderContext['tensorLocations'];
  loadTensor: LayerLoaderContext['loadTensor'];
  gpuBuffers: Set<GPUBuffer>;
  hyperConnections?: ResolvedHyperConnections | null;
  indexedAttention?: ResolvedIndexedAttention | null;
  ngramEmbedding?: ResolvedNgramEmbedding | null;
}

export interface HyperConnectionLayerWeights {
  hyperConnections: { attention: GatedResidualWeights; mlp: GatedResidualWeights } | null;
  qsaIndexer: { projection: WeightBuffer; qNorm: Tensor; kNorm: Tensor } | null;
  ple: PleLayerWeights | null;
}

export declare function loadHyperConnectionLayerWeights(ctx: MechanismLoaderContext, layerIdx: number): Promise<Partial<HyperConnectionLayerWeights>>;
export declare function loadHyperConnectionMixerWeights(ctx: MechanismLoaderContext): Promise<GatedResidualWeights | null>;
