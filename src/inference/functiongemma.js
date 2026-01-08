/**
 * FunctionGemma orchestrator for multi-expert code generation.
 *
 * Coordinates expert selection, execution, and evolution for task-specific
 * code generation using LoRA-adapted experts.
 *
 * @module inference/functiongemma
 */

import { MultiModelNetwork } from './multi-model-network.js';
import { mutateGenome, crossoverGenome } from './network-evolution.js';

/**
 * @typedef {import('./pipeline.js').GenerateOptions} GenerateOptions
 * @typedef {import('./pipeline.js').InferencePipeline} InferencePipeline
 * @typedef {import('./pipeline.js').KVCacheSnapshot} KVCacheSnapshot
 * @typedef {import('./pipeline/lora.js').LoRAAdapter} LoRAAdapter
 * @typedef {import('../loader/multi-model-loader.js').MultiModelLoader} MultiModelLoader
 * @typedef {import('./multi-pipeline-pool.js').MultiPipelinePool} MultiPipelinePool
 * @typedef {import('./multi-model-network.js').ExpertNode} ExpertNode
 * @typedef {import('./multi-model-network.js').CombinerConfig} CombinerConfig
 * @typedef {import('./network-evolution.js').NetworkGenome} NetworkGenome
 * @typedef {import('./network-evolution.js').EvolutionConfig} EvolutionConfig
 * @typedef {import('../adapters/adapter-manager.js').AdapterManager} AdapterManager
 * @typedef {import('../adapters/adapter-manager.js').AdapterState} AdapterState
 * @typedef {import('../adapters/adapter-registry.js').AdapterRegistry} AdapterRegistry
 * @typedef {import('./functiongemma.js').ExpertStats} ExpertStats
 * @typedef {import('./functiongemma.js').TaskType} TaskType
 * @typedef {import('./functiongemma.js').FunctionGemmaTask} FunctionGemmaTask
 * @typedef {import('./functiongemma.js').ExecutionResult} ExecutionResult
 * @typedef {import('./functiongemma.js').ArenaResult} ArenaResult
 * @typedef {import('./functiongemma.js').EvolutionResult} EvolutionResult
 * @typedef {import('./functiongemma.js').FunctionGemmaConfig} FunctionGemmaConfig
 */

// ============================================================================
// FunctionGemma Orchestrator
// ============================================================================

export class FunctionGemma {
  /** @type {MultiModelNetwork} */
  network;

  /** @type {InferencePipeline} */
  pipeline;

  /** @type {MultiModelLoader | null} */
  loader;

  /** @type {MultiPipelinePool | null} */
  pool;

  // Adapter management
  /** @type {AdapterManager | null} */
  adapterManager = null;

  /** @type {AdapterRegistry | null} */
  adapterRegistry = null;

  /** @type {Map<string, string>} expertId -> adapterId */
  expertAdapterMap = new Map();

  // Expert statistics for UCB1 selection
  /** @type {Map<string, Map<string, ExpertStats>>} taskType -> expertId -> stats */
  expertStats = new Map();

  /** @type {Map<string, TaskType>} */
  taskTypes = new Map();

  /** @type {FunctionGemmaConfig} */
  config;

  // Stored winning genomes per task type
  /** @type {Map<string, Array<{ genome: NetworkGenome; fitness: number; timestamp: number }>>} */
  winnerGenomes = new Map();

  /**
   * @param {InferencePipeline} pipeline
   * @param {MultiModelLoader} [loader]
   * @param {MultiPipelinePool} [pool]
   * @param {FunctionGemmaConfig} [config={}]
   */
  constructor(pipeline, loader, pool, config = {}) {
    this.pipeline = pipeline;
    this.loader = loader || null;
    this.pool = pool || null;
    this.network = new MultiModelNetwork(pipeline, loader, pool);
    this.config = {
      defaultTemperature: 0.7,
      defaultMaxTokens: 2048,
      explorationWeight: 2,
      fitnessDecay: 0.95,
      minAttemptsBeforeUCB: 3,
      ...config,
    };
  }

  // ============================================================================
  // Expert Management
  // ============================================================================

  /**
   * @param {ExpertNode} expert
   * @returns {void}
   */
  registerExpert(expert) {
    this.network.registerExpert(expert);
  }

  /**
   * @param {string} id
   * @returns {ExpertNode | null}
   */
  getExpert(id) {
    return this.network.getExpert(id);
  }

  /**
   * @returns {ExpertNode[]}
   */
  listExperts() {
    return this.network.listExperts();
  }

  /**
   * @param {TaskType} taskType
   * @returns {void}
   */
  registerTaskType(taskType) {
    this.taskTypes.set(taskType.id, taskType);
    if (!this.expertStats.has(taskType.id)) {
      this.expertStats.set(taskType.id, new Map());
    }
  }

  /**
   * @param {string} id
   * @returns {TaskType | null}
   */
  getTaskType(id) {
    return this.taskTypes.get(id) || null;
  }

  /**
   * @returns {TaskType[]}
   */
  listTaskTypes() {
    return Array.from(this.taskTypes.values());
  }

  // ============================================================================
  // Adapter Management
  // ============================================================================

  /**
   * Set the adapter manager for runtime adapter switching.
   * @param {AdapterManager} manager
   * @returns {void}
   */
  setAdapterManager(manager) {
    this.adapterManager = manager;
  }

  /**
   * Set the adapter registry for adapter discovery.
   * @param {AdapterRegistry} registry
   * @returns {void}
   */
  setAdapterRegistry(registry) {
    this.adapterRegistry = registry;
  }

  /**
   * Link an expert to a specific LoRA adapter.
   * @param {string} expertId
   * @param {string} adapterId
   * @returns {void}
   */
  linkExpertToAdapter(expertId, adapterId) {
    this.expertAdapterMap.set(expertId, adapterId);

    // Update the expert node if it exists
    const expert = this.network.getExpert(expertId);
    if (expert) {
      expert.adapterName = adapterId;
    }
  }

  /**
   * Get the adapter linked to an expert.
   * @param {string} expertId
   * @returns {string | null}
   */
  getExpertAdapter(expertId) {
    return this.expertAdapterMap.get(expertId) || null;
  }

  /**
   * Auto-discover and register experts from available adapters.
   * @returns {Promise<ExpertNode[]>}
   */
  async autoRegisterAdapterExperts() {
    if (!this.adapterRegistry) {
      return [];
    }

    const adapters = await this.adapterRegistry.list();
    /** @type {ExpertNode[]} */
    const newExperts = [];

    for (const adapter of adapters) {
      const expertId = `adapter-${adapter.id}`;

      // Skip if already registered
      if (this.network.getExpert(expertId)) continue;

      /** @type {ExpertNode} */
      const expert = {
        id: expertId,
        metadata: { name: adapter.name || adapter.id },
        embedding: [], // Will be populated if available
        adapterName: adapter.id,
      };

      // Extract tags/embedding from adapter metadata
      const tags = adapter.metadata?.tags;
      if (tags) {
        // Simple tag-to-embedding: one-hot encode common tags
        const commonTags = ['code', 'python', 'javascript', 'typescript', 'react', 'sql', 'api', 'test', 'docs'];
        expert.embedding = commonTags.map((tag) => (tags.includes(tag) ? 1 : 0));
      }

      this.registerExpert(expert);
      this.linkExpertToAdapter(expertId, adapter.id);
      newExperts.push(expert);
    }

    return newExperts;
  }

  /**
   * Activate the adapter for a specific expert before execution.
   * @param {string} expertId
   * @returns {Promise<LoRAAdapter | null>}
   * @private
   */
  async activateExpertAdapter(expertId) {
    const adapterId = this.expertAdapterMap.get(expertId);
    if (!adapterId) return null;

    if (this.adapterManager) {
      // Use adapter manager for full lifecycle
      const state = this.adapterManager.getAdapterState(adapterId);
      if (state && !state.enabled) {
        this.adapterManager.enableAdapter(adapterId);
      }
      return state?.adapter || null;
    }

    if (this.loader) {
      // Fallback to loader
      return this.loader.getAdapter(adapterId);
    }

    return null;
  }

  // ============================================================================
  // UCB1 Expert Selection
  // ============================================================================

  /**
   * Select expert using UCB1 algorithm for exploration/exploitation balance.
   * @param {string} taskType
   * @param {string[]} [candidateIds]
   * @returns {string | null}
   */
  selectExpertUCB1(taskType, candidateIds) {
    const experts = candidateIds
      ? candidateIds.map((id) => this.network.getExpert(id)).filter((e) => e !== null)
      : this.network.listExperts();

    if (experts.length === 0) return null;

    const statsMap = this.expertStats.get(taskType) || new Map();
    let totalAttempts = 0;
    for (const expert of experts) {
      const stats = statsMap.get(expert.id);
      totalAttempts += stats?.attempts || 0;
    }

    // If not enough data, return random expert
    if (totalAttempts < (this.config.minAttemptsBeforeUCB || 3) * experts.length) {
      return experts[Math.floor(Math.random() * experts.length)].id;
    }

    let bestScore = -Infinity;
    let bestExpert = experts[0].id;

    for (const expert of experts) {
      const stats = statsMap.get(expert.id);

      // Never-tried experts get infinite UCB score
      if (!stats || stats.attempts === 0) {
        return expert.id;
      }

      // UCB1: mean + sqrt(explorationWeight * ln(total) / attempts)
      const mean = stats.totalScore / stats.attempts;
      const exploration = Math.sqrt(
        ((this.config.explorationWeight || 2) * Math.log(totalAttempts)) / stats.attempts
      );
      const ucbScore = mean + exploration;

      if (ucbScore > bestScore) {
        bestScore = ucbScore;
        bestExpert = expert.id;
      }
    }

    return bestExpert;
  }

  /**
   * Update expert statistics after execution.
   * @param {string} taskType
   * @param {string} expertId
   * @param {number} fitness
   * @param {boolean} success
   * @returns {void}
   */
  updateExpertStats(taskType, expertId, fitness, success) {
    if (!this.expertStats.has(taskType)) {
      this.expertStats.set(taskType, new Map());
    }

    const statsMap = /** @type {Map<string, ExpertStats>} */ (this.expertStats.get(taskType));
    const existing = statsMap.get(expertId) || {
      successes: 0,
      attempts: 0,
      totalScore: 0,
      lastUsed: 0,
    };

    // Apply decay to existing scores
    const decay = this.config.fitnessDecay || 0.95;
    existing.totalScore *= decay;
    existing.attempts = Math.max(1, Math.floor(existing.attempts * decay));

    // Update with new result
    existing.attempts++;
    existing.totalScore += fitness;
    if (success) existing.successes++;
    existing.lastUsed = Date.now();

    statsMap.set(expertId, existing);
  }

  /**
   * @param {string} taskType
   * @param {string} expertId
   * @returns {ExpertStats | null}
   */
  getExpertStats(taskType, expertId) {
    return this.expertStats.get(taskType)?.get(expertId) || null;
  }

  // ============================================================================
  // Task Classification
  // ============================================================================

  /**
   * Classify a task description to determine task type.
   * Uses keyword matching and pattern detection.
   * @param {string} description
   * @returns {string}
   */
  classifyTask(description) {
    const lower = description.toLowerCase();

    // Code generation patterns
    if (lower.includes('react') || lower.includes('component')) return 'react-component';
    if (lower.includes('api') || lower.includes('endpoint')) return 'api-endpoint';
    if (lower.includes('test') || lower.includes('spec')) return 'test-generation';
    if (lower.includes('sql') || lower.includes('query') || lower.includes('database')) return 'sql-query';
    if (lower.includes('refactor') || lower.includes('improve')) return 'code-refactor';
    if (lower.includes('fix') || lower.includes('bug') || lower.includes('error')) return 'bug-fix';
    if (lower.includes('document') || lower.includes('readme') || lower.includes('jsdoc')) return 'documentation';
    if (lower.includes('typescript') || lower.includes('type')) return 'typescript';
    if (lower.includes('python')) return 'python';
    if (lower.includes('javascript') || lower.includes('js')) return 'javascript';

    return 'general-code';
  }

  /**
   * Get experts best suited for a task type based on their adapter tags.
   * @param {string} taskType
   * @returns {ExpertNode[]}
   */
  getExpertsForTaskType(taskType) {
    const experts = this.network.listExperts();
    const taskKeywords = taskType.split('-');

    // Score experts by tag overlap
    const scored = experts.map((expert) => {
      let score = 0;
      const adapterId = this.expertAdapterMap.get(expert.id);
      const expertNode = /** @type {ExpertNode & { tags?: string[] }} */ (expert);

      // Check expert tags
      if (expertNode.tags) {
        for (const keyword of taskKeywords) {
          if (expertNode.tags.some((tag) => tag.toLowerCase().includes(keyword))) {
            score++;
          }
        }
      }

      // Check adapter name/id for hints
      if (adapterId) {
        for (const keyword of taskKeywords) {
          if (adapterId.toLowerCase().includes(keyword)) {
            score++;
          }
        }
      }

      // Check embedding similarity if available
      if (expert.embedding && expert.embedding.length > 0) {
        // Simple match: any non-zero embedding value adds score
        score += expert.embedding.filter((v) => v > 0).length * 0.1;
      }

      return { expert, score };
    });

    // Sort by score descending
    scored.sort((a, b) => b.score - a.score);

    return scored.map((s) => s.expert);
  }

  /**
   * Select the best expert for a task, combining classification and UCB1.
   * @param {FunctionGemmaTask} task
   * @returns {string | null}
   */
  selectBestExpert(task) {
    // Get experts suited for this task type
    const suitedExperts = this.getExpertsForTaskType(task.taskType);

    if (suitedExperts.length === 0) {
      // Fall back to UCB1 across all experts
      return this.selectExpertUCB1(task.taskType);
    }

    // Use UCB1 among suited experts
    const candidateIds = suitedExperts.slice(0, 5).map((e) => e.id); // Top 5 suited
    return this.selectExpertUCB1(task.taskType, candidateIds);
  }

  // ============================================================================
  // Task Execution
  // ============================================================================

  /**
   * Execute a task with automatic expert selection.
   * Uses task classification + UCB1 for optimal expert routing.
   * @param {FunctionGemmaTask} task
   * @param {GenerateOptions} [options={}]
   * @returns {Promise<ExecutionResult>}
   */
  async execute(task, options = {}) {
    // Auto-classify if task type not provided
    if (!task.taskType || task.taskType === 'auto') {
      task = { ...task, taskType: this.classifyTask(task.description) };
    }

    const expertId = this.selectBestExpert(task);
    if (!expertId) {
      throw new Error(`No experts available for task type: ${task.taskType}`);
    }

    return this.executeWithExpert(expertId, task, options);
  }

  /**
   * Execute a task with a specific expert.
   * @param {string} expertId
   * @param {FunctionGemmaTask} task
   * @param {GenerateOptions} [options={}]
   * @returns {Promise<ExecutionResult>}
   */
  async executeWithExpert(expertId, task, options = {}) {
    const start = Date.now();
    const prompt = this.buildPrompt(task);

    // Activate expert's adapter if linked
    const adapter = await this.activateExpertAdapter(expertId);

    /** @type {GenerateOptions} */
    const mergedOptions = {
      maxTokens: task.maxTokens || this.config.defaultMaxTokens,
      temperature: task.temperature || this.config.defaultTemperature,
      ...options,
    };

    const output = await this.network.executeExpert(expertId, prompt, mergedOptions, {
      adapter,
    });
    const latencyMs = Date.now() - start;

    // Estimate tokens (rough approximation)
    const tokensGenerated = Math.ceil(output.length / 4);

    // Calculate fitness (can be overridden by validator)
    const fitness = this.calculateBaseFitness(output, task);

    // Update stats
    this.updateExpertStats(task.taskType, expertId, fitness, fitness > 0.5);

    return {
      output,
      expertId,
      fitness,
      latencyMs,
      tokensGenerated,
    };
  }

  /**
   * Execute with temporal self-ring for iterative refinement.
   * @param {FunctionGemmaTask} task
   * @param {{ turns?: number; temperatureStart?: number; temperatureDecay?: number; shortcutInterval?: number }} [config={}]
   * @returns {Promise<{ result: ExecutionResult; history: Array<{ turn: number; output: string; role: string }>; converged: boolean }>}
   */
  async executeTemporalRing(task, config = {}) {
    const expertId = this.selectExpertUCB1(task.taskType) || this.network.listExperts()[0]?.id;
    if (!expertId) {
      throw new Error(`No experts available for task type: ${task.taskType}`);
    }

    const start = Date.now();
    const ringResult = await this.network.executeTemporalRing(
      expertId,
      {
        description: this.buildPrompt(task),
        maxTokens: task.maxTokens || this.config.defaultMaxTokens,
        convergenceThreshold: task.convergenceThreshold,
      },
      {
        turns: config.turns || 5,
        temperatureStart: config.temperatureStart || 0.8,
        temperatureDecay: config.temperatureDecay || 0.15,
        shortcutInterval: config.shortcutInterval || 2,
        enableShortcuts: true,
      }
    );

    const fitness = this.calculateBaseFitness(ringResult.finalOutput, task);
    this.updateExpertStats(task.taskType, expertId, fitness, fitness > 0.5);

    return {
      result: {
        output: ringResult.finalOutput,
        expertId,
        fitness,
        latencyMs: Date.now() - start,
        tokensGenerated: Math.ceil(ringResult.finalOutput.length / 4),
      },
      history: ringResult.history,
      converged: ringResult.converged,
    };
  }

  // ============================================================================
  // Arena Competition
  // ============================================================================

  /**
   * Run arena competition between experts.
   * @param {FunctionGemmaTask} task
   * @param {string[]} [expertIds]
   * @param {{ rounds?: number; eliminations?: number }} [options={}]
   * @returns {Promise<ArenaResult>}
   */
  async runArena(task, expertIds, options = {}) {
    const candidates = expertIds
      ? expertIds.map((id) => this.network.getExpert(id)).filter((e) => e !== null)
      : this.network.listExperts();

    if (candidates.length === 0) {
      throw new Error('No experts available for arena');
    }

    const { rounds = 1, eliminations = Math.ceil(candidates.length / 2) } = options;
    /** @type {ExecutionResult[]} */
    const results = [];

    // Execute all candidates
    const execPromises = candidates.map((expert) =>
      this.executeWithExpert(expert.id, task).catch((error) => ({
        output: '',
        expertId: expert.id,
        fitness: 0,
        latencyMs: 0,
        tokensGenerated: 0,
        error,
      }))
    );

    const execResults = await Promise.all(execPromises);
    results.push(.../** @type {ExecutionResult[]} */ (execResults.filter((r) => !('error' in r))));

    // Sort by fitness
    results.sort((a, b) => b.fitness - a.fitness);

    const winner = results[0] || {
      output: '',
      expertId: candidates[0].id,
      fitness: 0,
      latencyMs: 0,
      tokensGenerated: 0,
    };

    const runnerUp = results[1] || null;

    return {
      winner,
      runnerUp,
      allResults: results,
      turnsUsed: rounds,
    };
  }

  /**
   * Run head-to-head comparison between two experts.
   * @param {string} expertA
   * @param {string} expertB
   * @param {FunctionGemmaTask} task
   * @param {number} [runs=3]
   * @returns {Promise<{ winner: string; scoreA: number; scoreB: number; results: Array<{ expertId: string; fitness: number }> }>}
   */
  async runHeadToHead(expertA, expertB, task, runs = 3) {
    let scoreA = 0;
    let scoreB = 0;
    /** @type {Array<{ expertId: string; fitness: number }>} */
    const results = [];

    for (let i = 0; i < runs; i++) {
      const [resultA, resultB] = await Promise.all([
        this.executeWithExpert(expertA, task),
        this.executeWithExpert(expertB, task),
      ]);

      results.push({ expertId: expertA, fitness: resultA.fitness });
      results.push({ expertId: expertB, fitness: resultB.fitness });

      scoreA += resultA.fitness;
      scoreB += resultB.fitness;
    }

    return {
      winner: scoreA >= scoreB ? expertA : expertB,
      scoreA: scoreA / runs,
      scoreB: scoreB / runs,
      results,
    };
  }

  // ============================================================================
  // Network Evolution
  // ============================================================================

  /**
   * Run network evolution to find optimal topology.
   * @param {FunctionGemmaTask} task
   * @param {{ populationSize?: number; generations?: number; eliteCount?: number; mutationRate?: number }} [config={}]
   * @returns {Promise<EvolutionResult>}
   */
  async runEvolution(task, config = {}) {
    const experts = this.network.listExperts();
    if (experts.length === 0) {
      throw new Error('No experts registered for evolution');
    }

    /** @returns {NetworkGenome} */
    const randomGenome = () => {
      /** @type {Array<'chain' | 'ring' | 'tree' | 'mesh' | 'dag'>} */
      const types = ['chain', 'ring', 'tree', 'mesh', 'dag'];
      const nodeCount = Math.min(experts.length, Math.floor(Math.random() * 4) + 2);
      const selectedExperts = this.shuffleArray([...experts]).slice(0, nodeCount);

      const nodes = selectedExperts.map((expert) => ({
        id: expert.id,
        adapter: expert.adapterName,
        temperature: 0.5 + Math.random() * 0.5,
      }));

      /** @type {Array<{ from: string; to: string; weight: number }>} */
      const edges = [];
      for (let i = 0; i < nodes.length - 1; i++) {
        edges.push({
          from: nodes[i].id,
          to: nodes[i + 1].id,
          weight: 0.5 + Math.random() * 0.5,
        });
      }

      return {
        topology: {
          type: types[Math.floor(Math.random() * types.length)],
          depth: Math.floor(Math.random() * 3) + 2,
          branchingFactor: Math.floor(Math.random() * 2) + 1,
        },
        nodes,
        edges,
        combiner: { type: 'weighted' },
      };
    };

    /**
     * @param {NetworkGenome} genome
     * @returns {Promise<number>}
     */
    const evaluate = async (genome) => {
      try {
        const prompt = this.buildPrompt(task);
        const output = await this.network.executeGenome(genome, prompt, {
          maxTokens: task.maxTokens || this.config.defaultMaxTokens,
          temperature: task.temperature || this.config.defaultTemperature,
        });
        return this.calculateBaseFitness(output, task);
      } catch {
        return 0;
      }
    };

    const populationSize = config.populationSize || 10;
    const generations = config.generations || 5;

    // Track fitness across generations
    let population = Array.from({ length: populationSize }, () => randomGenome());
    /** @type {number[][]} */
    const fitnessHistory = [];

    for (let gen = 0; gen < generations; gen++) {
      const scored = await Promise.all(
        population.map(async (genome) => ({
          genome,
          score: await evaluate(genome),
        }))
      );
      scored.sort((a, b) => b.score - a.score);

      fitnessHistory.push(scored.map((s) => s.score));

      const elite = scored.slice(0, config.eliteCount || 2).map((item) => item.genome);
      /** @type {NetworkGenome[]} */
      const offspring = [];

      while (offspring.length < populationSize - elite.length) {
        const parentA = scored[Math.floor(Math.random() * scored.length)].genome;
        const parentB = scored[Math.floor(Math.random() * scored.length)].genome;
        const child = mutateGenome(crossoverGenome(parentA, parentB), config.mutationRate || 0.1);
        offspring.push(child);
      }

      population = [...elite, ...offspring];
    }

    // Final evaluation
    const finalScores = await Promise.all(
      population.map(async (genome) => ({ genome, score: await evaluate(genome) }))
    );
    finalScores.sort((a, b) => b.score - a.score);

    const bestGenome = finalScores[0].genome;
    const bestFitness = finalScores[0].score;

    // Store winning genome
    this.storeWinnerGenome(task.taskType, bestGenome, bestFitness);

    const allFitness = finalScores.map((s) => s.score);
    return {
      bestGenome,
      bestFitness,
      generationsRun: generations,
      populationStats: {
        avgFitness: allFitness.reduce((a, b) => a + b, 0) / allFitness.length,
        maxFitness: Math.max(...allFitness),
        minFitness: Math.min(...allFitness),
      },
    };
  }

  // ============================================================================
  // Genome Storage
  // ============================================================================

  /**
   * @param {string} taskType
   * @param {NetworkGenome} genome
   * @param {number} fitness
   * @returns {void}
   */
  storeWinnerGenome(taskType, genome, fitness) {
    if (!this.winnerGenomes.has(taskType)) {
      this.winnerGenomes.set(taskType, []);
    }

    const genomes = /** @type {Array<{ genome: NetworkGenome; fitness: number; timestamp: number }>} */ (this.winnerGenomes.get(taskType));
    genomes.push({ genome, fitness, timestamp: Date.now() });

    // Keep top 10 sorted by fitness
    genomes.sort((a, b) => b.fitness - a.fitness);
    if (genomes.length > 10) {
      genomes.length = 10;
    }
  }

  /**
   * @param {string} taskType
   * @returns {NetworkGenome | null}
   */
  getBestGenome(taskType) {
    const genomes = this.winnerGenomes.get(taskType);
    return genomes?.[0]?.genome || null;
  }

  /**
   * @param {string} taskType
   * @returns {Array<{ genome: NetworkGenome; fitness: number; timestamp: number }>}
   */
  getAllWinnerGenomes(taskType) {
    return this.winnerGenomes.get(taskType) || [];
  }

  /**
   * @returns {Record<string, Array<{ genome: NetworkGenome; fitness: number; timestamp: number }>>}
   */
  exportGenomes() {
    /** @type {Record<string, Array<{ genome: NetworkGenome; fitness: number; timestamp: number }>>} */
    const result = {};
    for (const [taskType, genomes] of this.winnerGenomes) {
      result[taskType] = [...genomes];
    }
    return result;
  }

  /**
   * @param {Record<string, Array<{ genome: NetworkGenome; fitness: number; timestamp: number }>>} data
   * @returns {void}
   */
  importGenomes(data) {
    for (const [taskType, genomes] of Object.entries(data)) {
      if (!this.winnerGenomes.has(taskType)) {
        this.winnerGenomes.set(taskType, []);
      }
      const existing = /** @type {Array<{ genome: NetworkGenome; fitness: number; timestamp: number }>} */ (this.winnerGenomes.get(taskType));
      existing.push(...genomes);
      existing.sort((a, b) => b.fitness - a.fitness);
      if (existing.length > 10) {
        existing.length = 10;
      }
    }
  }

  // ============================================================================
  // Helpers
  // ============================================================================

  /**
   * @param {FunctionGemmaTask} task
   * @returns {string}
   * @private
   */
  buildPrompt(task) {
    const taskType = this.taskTypes.get(task.taskType);
    let prompt = task.description;

    if (taskType) {
      prompt = `[Task Type: ${taskType.name}]\n${taskType.description}\n\n${prompt}`;
    }

    if (task.context) {
      prompt = `${task.context}\n\n${prompt}`;
    }

    return prompt;
  }

  /**
   * @param {string} output
   * @param {FunctionGemmaTask} task
   * @returns {number}
   * @private
   */
  calculateBaseFitness(output, task) {
    // Base fitness heuristics (can be overridden by custom validators)
    let fitness = 0.5; // Start at neutral

    // Length check
    if (output.length > 50) fitness += 0.1;
    if (output.length > 200) fitness += 0.1;

    // Code block detection
    if (output.includes('```')) fitness += 0.1;

    // JSON structure detection
    if (output.includes('{') && output.includes('}')) fitness += 0.05;

    // Error/warning detection (negative)
    if (output.toLowerCase().includes('error')) fitness -= 0.1;
    if (output.toLowerCase().includes('cannot')) fitness -= 0.05;
    if (output.toLowerCase().includes('unable')) fitness -= 0.05;

    // Empty output penalty
    if (output.trim().length === 0) fitness = 0;

    return Math.max(0, Math.min(1, fitness));
  }

  /**
   * @template T
   * @param {T[]} array
   * @returns {T[]}
   * @private
   */
  shuffleArray(array) {
    const result = [...array];
    for (let i = result.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [result[i], result[j]] = [result[j], result[i]];
    }
    return result;
  }

  // ============================================================================
  // Pipeline Access
  // ============================================================================

  /**
   * @returns {MultiModelNetwork}
   */
  getNetwork() {
    return this.network;
  }

  /**
   * @returns {InferencePipeline}
   */
  getPipeline() {
    return this.pipeline;
  }

  /**
   * @param {string} prompt
   * @param {GenerateOptions} [options={}]
   * @returns {Promise<KVCacheSnapshot>}
   */
  async setSharedPrefix(prompt, options = {}) {
    return this.network.setSharedPrefix(prompt, options);
  }
}
