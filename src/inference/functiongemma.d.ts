/**
 * FunctionGemma orchestrator for multi-expert code generation.
 *
 * Coordinates expert selection, execution, and evolution for task-specific
 * code generation using LoRA-adapted experts.
 *
 * @module inference/functiongemma
 */

import type { GenerateOptions, InferencePipeline, KVCacheSnapshot } from './pipeline.js';
import type { LoRAAdapter } from './pipeline/lora.js';
import type { MultiModelLoader } from '../loader/multi-model-loader.js';
import type { MultiPipelinePool } from './multi-pipeline-pool.js';
import { MultiModelNetwork, type ExpertNode, type CombinerConfig } from './multi-model-network.js';
import { type NetworkGenome, type EvolutionConfig } from './network-evolution.js';
import type { AdapterManager, AdapterState } from '../adapters/adapter-manager.js';
import type { AdapterRegistry } from '../adapters/adapter-registry.js';

// ============================================================================
// Types
// ============================================================================

export interface ExpertStats {
  successes: number;
  attempts: number;
  totalScore: number;
  lastUsed: number;
}

export interface TaskType {
  id: string;
  name: string;
  description: string;
  tags: string[];
}

export interface FunctionGemmaTask {
  taskType: string;
  description: string;
  context?: string;
  maxTokens?: number;
  temperature?: number;
  convergenceThreshold?: number;
}

export interface ExecutionResult {
  output: string;
  expertId: string;
  fitness: number;
  latencyMs: number;
  tokensGenerated: number;
}

export interface ArenaResult {
  winner: ExecutionResult;
  runnerUp: ExecutionResult | null;
  allResults: ExecutionResult[];
  turnsUsed: number;
}

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

export interface FunctionGemmaConfig {
  defaultTemperature?: number;
  defaultMaxTokens?: number;
  explorationWeight?: number;  // UCB1 exploration parameter
  fitnessDecay?: number;       // Decay rate for old fitness scores
  minAttemptsBeforeUCB?: number;
}

// ============================================================================
// FunctionGemma Orchestrator
// ============================================================================

export declare class FunctionGemma {
  private network;
  private pipeline;
  private loader;
  private pool;

  // Adapter management
  private adapterManager;
  private adapterRegistry;
  private expertAdapterMap;

  // Expert statistics for UCB1 selection
  private expertStats;
  private taskTypes;
  private config;

  // Stored winning genomes per task type
  private winnerGenomes;

  constructor(
    pipeline: InferencePipeline,
    loader?: MultiModelLoader,
    pool?: MultiPipelinePool,
    config?: FunctionGemmaConfig
  );

  // ============================================================================
  // Expert Management
  // ============================================================================

  registerExpert(expert: ExpertNode): void;

  getExpert(id: string): ExpertNode | null;

  listExperts(): ExpertNode[];

  registerTaskType(taskType: TaskType): void;

  getTaskType(id: string): TaskType | null;

  listTaskTypes(): TaskType[];

  // ============================================================================
  // Adapter Management
  // ============================================================================

  /**
   * Set the adapter manager for runtime adapter switching.
   */
  setAdapterManager(manager: AdapterManager): void;

  /**
   * Set the adapter registry for adapter discovery.
   */
  setAdapterRegistry(registry: AdapterRegistry): void;

  /**
   * Link an expert to a specific LoRA adapter.
   */
  linkExpertToAdapter(expertId: string, adapterId: string): void;

  /**
   * Get the adapter linked to an expert.
   */
  getExpertAdapter(expertId: string): string | null;

  /**
   * Auto-discover and register experts from available adapters.
   */
  autoRegisterAdapterExperts(): Promise<ExpertNode[]>;

  /**
   * Activate the adapter for a specific expert before execution.
   */
  private activateExpertAdapter(expertId: string): Promise<LoRAAdapter | null>;

  // ============================================================================
  // UCB1 Expert Selection
  // ============================================================================

  /**
   * Select expert using UCB1 algorithm for exploration/exploitation balance.
   */
  selectExpertUCB1(taskType: string, candidateIds?: string[]): string | null;

  /**
   * Update expert statistics after execution.
   */
  updateExpertStats(taskType: string, expertId: string, fitness: number, success: boolean): void;

  getExpertStats(taskType: string, expertId: string): ExpertStats | null;

  // ============================================================================
  // Task Classification
  // ============================================================================

  /**
   * Classify a task description to determine task type.
   * Uses keyword matching and pattern detection.
   */
  classifyTask(description: string): string;

  /**
   * Get experts best suited for a task type based on their adapter tags.
   */
  getExpertsForTaskType(taskType: string): ExpertNode[];

  /**
   * Select the best expert for a task, combining classification and UCB1.
   */
  selectBestExpert(task: FunctionGemmaTask): string | null;

  // ============================================================================
  // Task Execution
  // ============================================================================

  /**
   * Execute a task with automatic expert selection.
   * Uses task classification + UCB1 for optimal expert routing.
   */
  execute(task: FunctionGemmaTask, options?: GenerateOptions): Promise<ExecutionResult>;

  /**
   * Execute a task with a specific expert.
   */
  executeWithExpert(
    expertId: string,
    task: FunctionGemmaTask,
    options?: GenerateOptions
  ): Promise<ExecutionResult>;

  /**
   * Execute with temporal self-ring for iterative refinement.
   */
  executeTemporalRing(
    task: FunctionGemmaTask,
    config?: {
      turns?: number;
      temperatureStart?: number;
      temperatureDecay?: number;
      shortcutInterval?: number;
    }
  ): Promise<{
    result: ExecutionResult;
    history: Array<{ turn: number; output: string; role: string }>;
    converged: boolean;
  }>;

  // ============================================================================
  // Arena Competition
  // ============================================================================

  /**
   * Run arena competition between experts.
   */
  runArena(
    task: FunctionGemmaTask,
    expertIds?: string[],
    options?: { rounds?: number; eliminations?: number }
  ): Promise<ArenaResult>;

  /**
   * Run head-to-head comparison between two experts.
   */
  runHeadToHead(
    expertA: string,
    expertB: string,
    task: FunctionGemmaTask,
    runs?: number
  ): Promise<{
    winner: string;
    scoreA: number;
    scoreB: number;
    results: Array<{ expertId: string; fitness: number }>;
  }>;

  // ============================================================================
  // Network Evolution
  // ============================================================================

  /**
   * Run network evolution to find optimal topology.
   */
  runEvolution(
    task: FunctionGemmaTask,
    config?: {
      populationSize?: number;
      generations?: number;
      eliteCount?: number;
      mutationRate?: number;
    }
  ): Promise<EvolutionResult>;

  // ============================================================================
  // Genome Storage
  // ============================================================================

  storeWinnerGenome(taskType: string, genome: NetworkGenome, fitness: number): void;

  getBestGenome(taskType: string): NetworkGenome | null;

  getAllWinnerGenomes(taskType: string): Array<{ genome: NetworkGenome; fitness: number; timestamp: number }>;

  exportGenomes(): Record<string, Array<{ genome: NetworkGenome; fitness: number; timestamp: number }>>;

  importGenomes(data: Record<string, Array<{ genome: NetworkGenome; fitness: number; timestamp: number }>>): void;

  // ============================================================================
  // Helpers
  // ============================================================================

  private buildPrompt(task: FunctionGemmaTask): string;

  private calculateBaseFitness(output: string, task: FunctionGemmaTask): number;

  private shuffleArray<T>(array: T[]): T[];

  // ============================================================================
  // Pipeline Access
  // ============================================================================

  getNetwork(): MultiModelNetwork;

  getPipeline(): InferencePipeline;

  setSharedPrefix(prompt: string, options?: GenerateOptions): Promise<KVCacheSnapshot>;
}
