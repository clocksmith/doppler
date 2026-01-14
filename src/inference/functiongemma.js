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

import { MultiModelNetwork } from './multi-model-network.js';
import { log } from '../debug/index.js';

// Re-export primitives from multi-model-network
export { MultiModelNetwork };

/**
 * @typedef {import('./multi-model-network.js').ExpertNode} ExpertNode
 * @typedef {import('./multi-model-network.js').CombinerConfig} CombinerConfig
 * @typedef {import('./multi-model-network.js').ExpertTask} ExpertTask
 * @typedef {import('./multi-model-network.js').TopologyRouter} TopologyRouter
 * @typedef {import('./network-evolution.js').NetworkGenome} NetworkGenome
 * @typedef {import('./network-evolution.js').NetworkNodeGene} NetworkNodeGene
 * @typedef {import('./network-evolution.js').NetworkEdgeGene} NetworkEdgeGene
 */

// ============================================================================
// Primitive Types (for Reploid to use)
// ============================================================================

/**
 * Expert statistics for bandit-style selection.
 * Reploid maintains these; Doppler just executes.
 * @typedef {Object} ExpertStats
 * @property {number} successes - Number of successful executions
 * @property {number} attempts - Total execution attempts
 * @property {number} totalScore - Cumulative fitness score
 * @property {number} lastUsed - Timestamp of last use
 */

/**
 * Task type definition for routing.
 * Reploid defines these; Doppler is agnostic.
 * @typedef {Object} TaskType
 * @property {string} id - Unique task type ID
 * @property {string} name - Human-readable name
 * @property {string} description - Task type description
 * @property {string[]} tags - Associated tags for matching
 */

/**
 * Task input for FunctionGemma execution.
 * @typedef {Object} FunctionGemmaTask
 * @property {string} taskType - Task type ID
 * @property {string} description - Task description/prompt
 * @property {string} [context] - Additional context
 * @property {number} [maxTokens] - Max tokens to generate
 * @property {number} [temperature] - Sampling temperature
 * @property {number} [convergenceThreshold] - For iterative refinement
 */

/**
 * Execution result from a single expert.
 * @typedef {Object} ExecutionResult
 * @property {string} output - Generated text
 * @property {string} expertId - ID of expert that generated this
 * @property {number} fitness - Fitness score (0-1)
 * @property {number} latencyMs - Execution time in milliseconds
 * @property {number} tokensGenerated - Number of tokens generated
 */

/**
 * Arena competition result.
 * @typedef {Object} ArenaResult
 * @property {ExecutionResult} winner - Winning result
 * @property {ExecutionResult | null} runnerUp - Second place
 * @property {ExecutionResult[]} allResults - All competition results
 * @property {number} turnsUsed - Rounds completed
 */

/**
 * Evolution result.
 * @typedef {Object} EvolutionResult
 * @property {NetworkGenome} bestGenome - Best evolved genome
 * @property {number} bestFitness - Best fitness achieved
 * @property {number} generationsRun - Generations completed
 * @property {{ avgFitness: number; maxFitness: number; minFitness: number }} populationStats - Population statistics
 */

/**
 * Configuration for FunctionGemma primitives.
 * @typedef {Object} FunctionGemmaConfig
 * @property {number} [defaultTemperature=0.7] - Default sampling temperature
 * @property {number} [defaultMaxTokens=2048] - Default max tokens
 */

// ============================================================================
// Deprecation Notice
// ============================================================================

/**
 * @deprecated Use Reploid's FunctionGemmaOrchestrator instead.
 *
 * The FunctionGemma class has been moved to Reploid to enforce
 * the Engine (Doppler) vs Driver (Reploid) separation.
 *
 * Migration:
 * ```javascript
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
 * ```javascript
 * import { MultiModelNetwork } from 'doppler/inference/functiongemma.js';
 * const network = new MultiModelNetwork(pipeline, loader, pool);
 * await network.executeExpert(expertId, prompt, options);
 * await network.executeGenome(genome, prompt, options);
 * ```
 */
export class FunctionGemma {
  /**
   * @param {import('./pipeline.js').InferencePipeline} pipeline
   * @param {import('../loader/multi-model-loader.js').MultiModelLoader} [loader]
   * @param {import('./multi-pipeline-pool.js').MultiPipelinePool} [pool]
   * @param {FunctionGemmaConfig} [config]
   */
  constructor(pipeline, loader, pool, config = {}) {
    log.warn(
      'FunctionGemma',
      '[DEPRECATED] FunctionGemma class is deprecated. ' +
      'Use Reploid\'s FunctionGemmaOrchestrator for orchestration, ' +
      'or MultiModelNetwork directly for primitives.'
    );

    /** @type {MultiModelNetwork} */
    this.network = new MultiModelNetwork(pipeline, loader, pool);

    /** @type {import('./pipeline.js').InferencePipeline} */
    this.pipeline = pipeline;

    /** @type {FunctionGemmaConfig} */
    this.config = {
      defaultTemperature: 0.7,
      defaultMaxTokens: 2048,
      ...config,
    };
  }

  /**
   * @deprecated Use MultiModelNetwork.registerExpert directly
   * @param {ExpertNode} expert
   */
  registerExpert(expert) {
    this.network.registerExpert(expert);
  }

  /**
   * @deprecated Use MultiModelNetwork.getExpert directly
   * @param {string} id
   * @returns {ExpertNode | null}
   */
  getExpert(id) {
    return this.network.getExpert(id);
  }

  /**
   * @deprecated Use MultiModelNetwork.listExperts directly
   * @returns {ExpertNode[]}
   */
  listExperts() {
    return this.network.listExperts();
  }

  /**
   * @deprecated Use MultiModelNetwork.executeExpert directly
   * @param {string} expertId
   * @param {string} prompt
   * @param {import('./pipeline.js').GenerateOptions} [options]
   * @returns {Promise<string>}
   */
  async executeExpert(expertId, prompt, options = {}) {
    return this.network.executeExpert(expertId, prompt, {
      maxTokens: this.config.defaultMaxTokens,
      temperature: this.config.defaultTemperature,
      ...options,
    });
  }

  /**
   * @deprecated Use MultiModelNetwork.setSharedPrefix directly
   * @param {string} prompt
   * @param {import('./pipeline.js').GenerateOptions} [options]
   * @returns {Promise<import('./pipeline.js').KVCacheSnapshot>}
   */
  async setSharedPrefix(prompt, options = {}) {
    return this.network.setSharedPrefix(prompt, options);
  }

  /**
   * @returns {import('./multi-model-network.js').MultiModelNetwork}
   */
  getNetwork() {
    return this.network;
  }

  /**
   * @returns {import('./pipeline.js').InferencePipeline}
   */
  getPipeline() {
    return this.pipeline;
  }
}
