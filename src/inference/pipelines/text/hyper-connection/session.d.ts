import type { ParsedModelConfig } from '../config.js';
import type { LayerContext } from '../types.js';
import type { Tensor } from '../../../../gpu/tensor.js';
import type { CommandRecorder } from '../../../../gpu/command-recorder.js';
import type { DopplerLoader } from '../../../../loader/doppler-loader.js';
import type { HyperConnectionLayerWeights } from '../../../../loader/hyper-connection-loader.js';
import type { QsaIndexerCache, QsaCacheCheckpoint, QsaCallSelection } from '../indexed-attention/qsa-cache.js';
import type { NgramTokenHistory, NgramHistoryCheckpoint } from '../ngram/ngram-addresses.js';
import type { NgramTableReader } from '../ngram/ngram-table.js';
import type { PleConvState, PleLayerWeights } from '../ngram/ple-layer.js';
import type { HyperConnectionProjection } from './decoder-layer.js';

export interface PleCheckpoint { history: NgramHistoryCheckpoint; conv: GPUBuffer; }
export interface HyperConnectionCheckpoint { epoch: number; qsa: QsaCacheCheckpoint | null; layers: Map<number, PleCheckpoint>; }
export interface PreparedQsaSelection extends QsaCallSelection { layerIdx: number; numTokens: number; visibleRows: number; }
export interface HyperConnectionSessionInput {
  hyper: NonNullable<ParsedModelConfig['hyperConnections']>;
  ngram: ParsedModelConfig['ngramEmbedding'];
  qsa: QsaIndexerCache | null;
  layers: Map<number, { table: NgramTableReader; history: NgramTokenHistory; conv: PleConvState }>;
}
export declare class HyperConnectionSession {
  constructor(input: HyperConnectionSessionInput);
  readonly hyper: HyperConnectionSessionInput['hyper'];
  readonly ngram: HyperConnectionSessionInput['ngram'];
  readonly qsa: QsaIndexerCache | null;
  readonly layers: HyperConnectionSessionInput['layers'];
  runPle(layerIdx: number, streams: Tensor, weights: PleLayerWeights | null | undefined, context: LayerContext, project: HyperConnectionProjection): Promise<Tensor>;
  prepareQsa(layerIdx: number, input: Tensor, weights: NonNullable<HyperConnectionLayerWeights['qsaIndexer']>, context: LayerContext, project: HyperConnectionProjection): Promise<PreparedQsaSelection>;
  releaseQsa(selection: PreparedQsaSelection, recorder: CommandRecorder | null): void;
  reset(): void;
  assertLength(length: number): void;
  checkpoint(): Promise<HyperConnectionCheckpoint>;
  restore(snapshot: HyperConnectionCheckpoint): Promise<void>;
  releaseCheckpoint(snapshot: HyperConnectionCheckpoint): void;
  destroy(): Promise<void>;
}
export declare function createHyperConnectionSession(config: ParsedModelConfig, loader: DopplerLoader, capacity: number,
  layerRange?: readonly number[] | null): Promise<HyperConnectionSession | null>;
