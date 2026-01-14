/**
 * FunctionGemma Primitives
 *
 * This module provides GPU-level primitives for multi-model inference.
 * Orchestration logic (UCB1 selection, evolution, arena, temporal rings)
 * lives in Reploid's FunctionGemmaOrchestrator.
 *
 * Architecture:
 * - Doppler (this module): Engine - executes inference primitives
 * - Reploid (FunctionGemmaOrchestrator): Driver - makes policy decisions
 *
 * @module inference/functiongemma
 * @see reploid/src/capabilities/intelligence/functiongemma-orchestrator.js
 */

import type { GenerateOptions, InferencePipeline, KVCacheSnapshot } from './pipeline.js';
import type { MultiModelLoader } from '../loader/multi-model-loader.js';
import type { MultiPipelinePool } from './multi-pipeline-pool.js';
import type { NetworkGenome } from './network-evolution.js';

// Re-export primitives from multi-model-network
export {
  MultiModelNetwork,
  type ExpertNode,
  type CombinerConfig,
  type ExpertTask,
  type TopologyRouter,
} from './multi-model-network.js';


// ============================================================================
// Primitive Types (for Reploid to use)
// ============================================================================

/**
 * Expert statistics for bandit-style selection.
 * Reploid maintains these; Doppler just executes.
 */
export interface ExpertStats {
  successes: number;
  attempts: number;
  totalScore: number;
  lastUsed: number;
}

/**
 * Task type definition for routing.
 * Reploid defines these; Doppler is agnostic.
 */
export interface TaskType {
  id: string;
  name: string;
  description: string;
  tags: string[];
}

/**
 * Task input for FunctionGemma execution.
 */
export interface FunctionGemmaTask {
  taskType: string;
  description: string;
  context?: string;
  maxTokens?: number;
  temperature?: number;
  convergenceThreshold?: number;
}

/**
 * Execution result from a single expert.
 */
export interface ExecutionResult {
  output: string;
  expertId: string;
  fitness: number;
  latencyMs: number;
  tokensGenerated: number;
}

/**
 * Arena competition result.
 */
export interface ArenaResult {
  winner: ExecutionResult;
  runnerUp: ExecutionResult | null;
  allResults: ExecutionResult[];
  turnsUsed: number;
}

/**
 * Evolution result.
 */
export interface EvolutionResult {
  bestGenome: NetworkGenome;
  bestFitness: number;
  generationsRun: number;
  populationStats: {
    avgFitness: number;
    maxFitness: number;
    minFitness: number;
  };
}

/**
 * Configuration for FunctionGemma primitives.
 */
export interface FunctionGemmaConfig {
  defaultTemperature?: number;
  defaultMaxTokens?: number;
}

// ============================================================================
// Deprecated Class
// ============================================================================

import { MultiModelNetwork, type ExpertNode } from './multi-model-network.js';

/**
 * @deprecated Use Reploid's FunctionGemmaOrchestrator instead.
 *
 * The FunctionGemma class has been moved to Reploid to enforce
 * the Engine (Doppler) vs Driver (Reploid) separation.
 *
 * Migration:
 * ```typescript
 * // Old (Doppler)
 * import { FunctionGemma } from 'doppler/inference/functiongemma.js';
 * const fg = new FunctionGemma(pipeline, loader, pool);
 * await fg.execute(task);
 *
 * // New (Reploid)
 * import FunctionGemmaOrchestrator from 'reploid/capabilities/intelligence/functiongemma-orchestrator.js';
 * const orchestrator = FunctionGemmaOrchestrator.factory(deps);
 * await orchestrator.initBase({ modelId, manifest });
 * await orchestrator.execute(task);
 * ```
 *
 * Doppler primitives are still available via MultiModelNetwork:
 * ```typescript
 * import { MultiModelNetwork } from 'doppler/inference/functiongemma.js';
 * const network = new MultiModelNetwork(pipeline, loader, pool);
 * await network.executeExpert(expertId, prompt, options);
 * await network.executeGenome(genome, prompt, options);
 * ```
 */
export declare class FunctionGemma {
  readonly network: MultiModelNetwork;
  readonly pipeline: InferencePipeline;
  readonly config: FunctionGemmaConfig;

  constructor(
    pipeline: InferencePipeline,
    loader?: MultiModelLoader,
    pool?: MultiPipelinePool,
    config?: FunctionGemmaConfig
  );

  /** @deprecated Use MultiModelNetwork.registerExpert directly */
  registerExpert(expert: ExpertNode): void;

  /** @deprecated Use MultiModelNetwork.getExpert directly */
  getExpert(id: string): ExpertNode | null;

  /** @deprecated Use MultiModelNetwork.listExperts directly */
  listExperts(): ExpertNode[];

  /** @deprecated Use MultiModelNetwork.executeExpert directly */
  executeExpert(expertId: string, prompt: string, options?: GenerateOptions): Promise<string>;

  /** @deprecated Use MultiModelNetwork.setSharedPrefix directly */
  setSharedPrefix(prompt: string, options?: GenerateOptions): Promise<KVCacheSnapshot>;

  getNetwork(): MultiModelNetwork;

  getPipeline(): InferencePipeline;
}
