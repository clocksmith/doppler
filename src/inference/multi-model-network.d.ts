/**
 * Multi-model execution network for FunctionGemma experts.
 *
 * @module inference/multi-model-network
 */

import type { GenerateOptions, InferencePipeline, KVCacheSnapshot } from './pipeline.js';
import type { LoRAAdapter } from './pipeline/lora.js';
import { ExpertRouter, type ExpertProfile } from './expert-router.js';
import type { MultiModelLoader } from '../loader/multi-model-loader.js';
import type { MultiPipelinePool } from './multi-pipeline-pool.js';
import { MultiModelRecorder } from '../gpu/multi-model-recorder.js';
import type { NetworkGenome, NetworkNodeGene, NetworkEdgeGene } from './network-evolution.js';

export interface ExpertNode extends ExpertProfile {
  adapterName?: string;
  adapter?: LoRAAdapter | null;
}

export interface CombinerConfig {
  type: 'weighted' | 'voting';
  weights?: number[];
}

export interface AbeOptions extends GenerateOptions {
  expertIds?: string[];
  voterIds?: string[];
  agreementTopK?: number;
  minAgreement?: number;
  mergeOnGpu?: boolean;
  prefix?: KVCacheSnapshot | null;
  adapterName?: string;
  adapter?: LoRAAdapter | null;
}

export type TopologyRouter = (context: {
  parent: ExpertNode;
  prompt: string;
  options: GenerateOptions;
  children: ExpertNode[];
  outputs: Map<string, string>;
}) => Promise<ExpertNode[] | ExpertNode | null> | ExpertNode[] | ExpertNode | null;

export interface ExpertTask {
  id: string;
  expertId: string;
  prompt: string;
}

export declare class MultiModelNetwork {
  private pipeline;
  private loader;
  private router;
  private experts;
  private sharedPrefix;
  private busy;
  private pipelinePool;
  private recorder;
  private combiner;

  constructor(
    pipeline: InferencePipeline,
    loader?: MultiModelLoader,
    pool?: MultiPipelinePool,
    recorder?: MultiModelRecorder
  );

  setRecorder(recorder: MultiModelRecorder | null): void;

  getRecorder(): MultiModelRecorder | null;

  setPipelinePool(pool: MultiPipelinePool | null): void;

  registerExpert(node: ExpertNode): void;

  getExpert(id: string): ExpertNode | null;

  listExperts(): ExpertNode[];

  setCombiner(config: CombinerConfig): void;

  setSharedPrefix(prompt: string, options?: GenerateOptions): Promise<KVCacheSnapshot>;

  setSharedPrefixSnapshot(snapshot: KVCacheSnapshot | null): void;

  getSharedPrefixSnapshot(): KVCacheSnapshot | null;

  private resolveAdapter(
    expert: ExpertNode,
    adapterName?: string,
    adapterOverride?: LoRAAdapter | null
  ): LoRAAdapter | null;

  executeExpert(
    expertId: string,
    prompt: string,
    options?: GenerateOptions,
    overrides?: { adapterName?: string; adapter?: LoRAAdapter | null; prefix?: KVCacheSnapshot | null; usePool?: boolean }
  ): Promise<string>;

  /**
   * Chain: Sequential pipeline where each expert runs once.
   * Output of each expert becomes input to the next.
   * @returns Array of all outputs in order
   */
  executeChain(expertIds: string[], prompt: string, options?: GenerateOptions): Promise<string[]>;

  /**
   * @deprecated Use executeChain instead
   */
  executeRing(expertIds: string[], prompt: string, options?: GenerateOptions): Promise<string[]>;

  executeBatch(tasks: ExpertTask[], options?: GenerateOptions): Promise<Record<string, string>>;

  executeParallel(tasks: ExpertTask[], options?: GenerateOptions): Promise<Record<string, string>>;

  generateWithABE(prompt: string, options?: AbeOptions): AsyncGenerator<string, void, void>;

  selectExpertsByEmbedding(embedding: number[], topK?: number): ExpertNode[];

  combineOutputs(outputs: string[], combinerOverride?: CombinerConfig): Promise<string>;

  executeGenome(
    genome: NetworkGenome,
    prompt: string,
    options?: GenerateOptions,
    router?: TopologyRouter
  ): Promise<string>;

  private executeGraph(
    genome: NetworkGenome,
    prompt: string,
    options: GenerateOptions,
    router?: TopologyRouter
  ): Promise<string[]>;

  private collectText(generator: AsyncGenerator<string>): Promise<string>;
}
