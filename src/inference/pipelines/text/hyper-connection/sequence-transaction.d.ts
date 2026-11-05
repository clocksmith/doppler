import type { PipelineState } from '../state.js';
export interface HyperConnectionOptions {
  signal?: AbortSignal;
  useSpeculative?: boolean;
  speculation?: { enabled?: boolean; mode?: string } | null;
  _embeddingOnly?: boolean;
  _trainingCapture?: unknown;
  _diffusionGemmaDecoder?: boolean;
  prefixEmbeddingOverride?: unknown;
  embeddingInputIds?: unknown;
  embeddingInputSpan?: unknown;
  embeddingOverrides?: unknown;
  multimodalBidirectionalSpan?: unknown;
  attentionMask?: unknown;
  paddingMask?: unknown;
  _returnHidden?: boolean;
}
export type HyperConnectionPipeline = PipelineState & { modelPartition?: unknown; layerPipelinePlan?: unknown; };
export declare function assertHyperConnectionLoad(state: HyperConnectionPipeline): void;
export declare function assertHyperConnectionOperation(state: Pick<PipelineState, 'modelConfig'>, opts: HyperConnectionOptions): void;
export declare function assertClassicSequenceState(state: Pick<PipelineState, 'modelConfig'>, operation: string): void;
export declare function withHyperConnectionTransaction<T>(state: HyperConnectionPipeline, opts: HyperConnectionOptions,
  action: (opts: HyperConnectionOptions) => Promise<T>): Promise<T>;
