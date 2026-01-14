/**
 * Multi-model execution network for FunctionGemma experts.
 *
 * @module inference/multi-model-network
 */

import { ExpertRouter } from './expert-router.js';
import { MultiModelRecorder } from '../gpu/multi-model-recorder.js';

/**
 * @typedef {import('./pipeline.js').GenerateOptions} GenerateOptions
 * @typedef {import('./pipeline.js').InferencePipeline} InferencePipeline
 * @typedef {import('./pipeline.js').KVCacheSnapshot} KVCacheSnapshot
 * @typedef {import('./pipeline/lora.js').LoRAAdapter} LoRAAdapter
 * @typedef {import('../loader/multi-model-loader.js').MultiModelLoader} MultiModelLoader
 * @typedef {import('./multi-pipeline-pool.js').MultiPipelinePool} MultiPipelinePool
 * @typedef {import('./network-evolution.js').NetworkGenome} NetworkGenome
 * @typedef {import('./network-evolution.js').NetworkNodeGene} NetworkNodeGene
 * @typedef {import('./network-evolution.js').NetworkEdgeGene} NetworkEdgeGene
 * @typedef {import('./expert-router.js').ExpertProfile} ExpertProfile
 * @typedef {import('./multi-model-network.js').ExpertNode} ExpertNode
 * @typedef {import('./multi-model-network.js').CombinerConfig} CombinerConfig
 * @typedef {import('./multi-model-network.js').TopologyRouter} TopologyRouter
 * @typedef {import('./multi-model-network.js').ExpertTask} ExpertTask
 */

export class MultiModelNetwork {
  /** @type {InferencePipeline} */
  pipeline;

  /** @type {MultiModelLoader | null} */
  loader;

  /** @type {ExpertRouter} */
  router;

  /** @type {Map<string, ExpertNode>} */
  experts;

  /** @type {KVCacheSnapshot | null} */
  sharedPrefix = null;

  /** @type {boolean} */
  busy = false;

  /** @type {MultiPipelinePool | null} */
  pipelinePool = null;

  /** @type {MultiModelRecorder | null} */
  recorder = null;

  /** @type {CombinerConfig} */
  combiner = { type: 'weighted' };

  /**
   * @param {InferencePipeline} pipeline
   * @param {MultiModelLoader} [loader]
   * @param {MultiPipelinePool} [pool]
   * @param {MultiModelRecorder} [recorder]
   */
  constructor(pipeline, loader, pool, recorder) {
    this.pipeline = pipeline;
    this.loader = loader || null;
    this.router = new ExpertRouter();
    this.experts = new Map();
    this.pipelinePool = pool || null;
    this.recorder = recorder || null;
  }

  /**
   * @param {MultiModelRecorder | null} recorder
   * @returns {void}
   */
  setRecorder(recorder) {
    this.recorder = recorder;
  }

  /**
   * @returns {MultiModelRecorder | null}
   */
  getRecorder() {
    return this.recorder;
  }

  /**
   * @param {MultiPipelinePool | null} pool
   * @returns {void}
   */
  setPipelinePool(pool) {
    this.pipelinePool = pool;
  }

  /**
   * @param {ExpertNode} node
   * @returns {void}
   */
  registerExpert(node) {
    this.experts.set(node.id, node);
    this.router.registerExpert(node);
  }

  /**
   * @param {string} id
   * @returns {ExpertNode | null}
   */
  getExpert(id) {
    return this.experts.get(id) || null;
  }

  /**
   * @returns {ExpertNode[]}
   */
  listExperts() {
    return Array.from(this.experts.values());
  }

  /**
   * @param {CombinerConfig} config
   * @returns {void}
   */
  setCombiner(config) {
    this.combiner = config;
  }

  /**
   * @param {string} prompt
   * @param {GenerateOptions} [options={}]
   * @returns {Promise<KVCacheSnapshot>}
   */
  async setSharedPrefix(prompt, options = {}) {
    const snapshot = this.recorder
      ? await this.recorder.computeSharedPrefix(this.pipeline, prompt, options)
      : await this.pipeline.prefillKVOnly(prompt, options);
    this.sharedPrefix = snapshot;
    this.pipelinePool?.setSharedPrefixSnapshot(snapshot);
    return snapshot;
  }

  /**
   * @param {KVCacheSnapshot | null} snapshot
   * @returns {void}
   */
  setSharedPrefixSnapshot(snapshot) {
    this.sharedPrefix = snapshot;
    if (this.recorder) {
      this.recorder.setSharedPrefix(snapshot);
    }
    this.pipelinePool?.setSharedPrefixSnapshot(snapshot);
  }

  /**
   * @returns {KVCacheSnapshot | null}
   */
  getSharedPrefixSnapshot() {
    return this.recorder?.getSharedPrefix() ?? this.sharedPrefix;
  }

  /**
   * @param {ExpertNode} expert
   * @param {string} [adapterName]
   * @param {LoRAAdapter | null} [adapterOverride]
   * @returns {LoRAAdapter | null}
   * @private
   */
  resolveAdapter(expert, adapterName, adapterOverride) {
    if (adapterOverride) return adapterOverride;
    const resolvedName = adapterName || expert.adapterName;
    if (resolvedName && this.loader) {
      return this.loader.getAdapter(resolvedName);
    }
    return expert.adapter || null;
  }

  /**
   * @param {string} expertId
   * @param {string} prompt
   * @param {GenerateOptions} [options={}]
   * @param {{ adapterName?: string; adapter?: LoRAAdapter | null; prefix?: KVCacheSnapshot | null; usePool?: boolean }} [overrides={}]
   * @returns {Promise<string>}
   */
  async executeExpert(expertId, prompt, options = {}, overrides = {}) {
    const expert = this.getExpert(expertId);
    if (!expert) {
      throw new Error(`Unknown expert: ${expertId}`);
    }

    const adapter = this.resolveAdapter(expert, overrides.adapterName, overrides.adapter);
    const prefix = overrides.prefix ?? this.getSharedPrefixSnapshot();

    if (this.pipelinePool && overrides.usePool) {
      return this.pipelinePool.execute(expertId, prompt, options, adapter, prefix);
    }

    this.pipeline.setLoRAAdapter(adapter);

    const generator = prefix
      ? this.pipeline.generateWithPrefixKV(prefix, prompt, options)
      : this.pipeline.generate(prompt, options);

    return this.collectText(generator);
  }

  /**
   * Chain: Sequential pipeline where each expert runs once.
   * Output of each expert becomes input to the next.
   * @param {string[]} expertIds
   * @param {string} prompt
   * @param {GenerateOptions} [options={}]
   * @returns {Promise<string[]>} Array of all outputs in order
   */
  async executeChain(expertIds, prompt, options = {}) {
    /** @type {string[]} */
    const outputs = [];
    let currentPrompt = prompt;

    for (const id of expertIds) {
      const output = await this.executeExpert(id, currentPrompt, options);
      outputs.push(output);
      currentPrompt = output;
    }

    return outputs;
  }

  /**
   * @deprecated Use executeChain instead
   * @param {string[]} expertIds
   * @param {string} prompt
   * @param {GenerateOptions} [options={}]
   * @returns {Promise<string[]>}
   */
  async executeRing(expertIds, prompt, options = {}) {
    return this.executeChain(expertIds, prompt, options);
  }

  /**
   * Circular Ring: Loops through all experts multiple times until convergence.
   * Each full loop checks if output has stabilized.
   * @param {string[]} expertIds
   * @param {string} prompt
   * @param {GenerateOptions} [options={}]
   * @param {{ maxIterations?: number; convergenceThreshold?: number }} [config={}]
   * @returns {Promise<{ output: string; iterations: number; converged: boolean }>} Final converged output and iteration count
   */
  async executeCircularRing(expertIds, prompt, options = {}, config = {}) {
    const { maxIterations = 3, convergenceThreshold = 0.95 } = config;
    let current = prompt;
    let prevOutput = '';
    let iterations = 0;

    for (let iter = 0; iter < maxIterations; iter++) {
      iterations++;
      for (const id of expertIds) {
        const output = await this.executeExpert(id, current, {
          ...options,
          // Decay temperature across iterations for refinement
          temperature: (options.temperature ?? 0.7) * Math.pow(0.9, iter),
        });
        current = output;
      }

      // Check convergence after each full loop
      if (current === prevOutput) {
        return { output: current, iterations, converged: true };
      }
      const similarity = this.computeOutputSimilarity(current, prevOutput);
      if (similarity >= convergenceThreshold) {
        return { output: current, iterations, converged: true };
      }
      prevOutput = current;
    }

    return { output: current, iterations, converged: false };
  }

  /**
   * @param {string} a
   * @param {string} b
   * @returns {number}
   * @private
   */
  computeOutputSimilarity(a, b) {
    // Simple Jaccard similarity on tokens
    const tokensA = new Set(a.toLowerCase().split(/\s+/));
    const tokensB = new Set(b.toLowerCase().split(/\s+/));
    const intersection = new Set([...tokensA].filter((x) => tokensB.has(x)));
    const union = new Set([...tokensA, ...tokensB]);
    return union.size > 0 ? intersection.size / union.size : 0;
  }

  /**
   * @param {ExpertTask[]} tasks
   * @param {GenerateOptions} [options={}]
   * @returns {Promise<Record<string, string>>}
   */
  async executeBatch(tasks, options = {}) {
    /** @type {Map<string, ExpertTask[]>} */
    const grouped = new Map();
    for (const task of tasks) {
      const expert = this.getExpert(task.expertId);
      const adapterKey = expert?.adapterName || '__base__';
      if (!grouped.has(adapterKey)) grouped.set(adapterKey, []);
      /** @type {ExpertTask[]} */ (grouped.get(adapterKey)).push(task);
    }

    /** @type {Record<string, string>} */
    const results = {};
    for (const group of grouped.values()) {
      for (const task of group) {
        results[task.id] = await this.executeExpert(task.expertId, task.prompt, options);
      }
    }

    return results;
  }

  /**
   * @param {ExpertTask[]} tasks
   * @param {GenerateOptions} [options={}]
   * @returns {Promise<Record<string, string>>}
   */
  async executeParallel(tasks, options = {}) {
    if (!this.pipelinePool) {
      if (this.busy) {
        throw new Error('MultiModelNetwork is busy. Parallel execution requires separate pipelines.');
      }
      this.busy = true;
      try {
        const entries = await Promise.all(
          tasks.map(async (task) => /** @type {const} */ ([task.id, await this.executeExpert(task.expertId, task.prompt, options)]))
        );
        return Object.fromEntries(entries);
      } finally {
        this.busy = false;
      }
    }

    const entries = await Promise.all(
      tasks.map(async (task) => {
        const output = await this.executeExpert(task.expertId, task.prompt, options, { usePool: true });
        return /** @type {const} */ ([task.id, output]);
      })
    );

    return Object.fromEntries(entries);
  }

  /**
   * @param {number[]} embedding
   * @param {number} [topK=1]
   * @returns {ExpertNode[]}
   */
  selectExpertsByEmbedding(embedding, topK = 1) {
    return /** @type {ExpertNode[]} */ (this.router.selectByEmbedding(embedding, topK));
  }

  /**
   * @param {string[]} outputs
   * @param {CombinerConfig} [combinerOverride]
   * @returns {Promise<string>}
   */
  async combineOutputs(outputs, combinerOverride) {
    if (outputs.length === 0) return '';

    const combiner = combinerOverride ?? this.combiner;

    if (combiner.type === 'voting') {
      /** @type {Map<string, number>} */
      const counts = new Map();
      for (const output of outputs) {
        counts.set(output, (counts.get(output) || 0) + 1);
      }
      const sorted = Array.from(counts.entries()).sort((a, b) => b[1] - a[1]);
      return sorted[0][0];
    }

    if (combiner.type === 'llm-merge') {
      const expertId = combiner.combinerExpertId || this.listExperts()[0]?.id;
      if (!expertId) return outputs[0];
      const mergePrompt = outputs.map((output, idx) => `Variant ${idx + 1}:\n${output}`).join('\n\n');
      return this.executeExpert(expertId, `Merge these variants into one cohesive solution:\n\n${mergePrompt}`);
    }

    const weights = combiner.weights || outputs.map(() => 1);
    let bestIdx = 0;
    let bestWeight = weights[0] ?? 0;
    for (let i = 1; i < outputs.length; i++) {
      const weight = weights[i] ?? 0;
      if (weight > bestWeight) {
        bestWeight = weight;
        bestIdx = i;
      }
    }
    return outputs[bestIdx];
  }

  /**
   * @param {NetworkGenome} genome
   * @param {string} prompt
   * @param {GenerateOptions} [options={}]
   * @param {TopologyRouter} [router]
   * @returns {Promise<string>}
   */
  async executeGenome(genome, prompt, options = {}, router) {
    /** @type {Map<string, NetworkNodeGene>} */
    const nodeLookup = new Map();
    for (const node of genome.nodes) {
      nodeLookup.set(node.id, node);
    }

    const resolvedExperts = genome.nodes.map((node) => {
      const expert = this.getExpert(node.id);
      if (!expert) {
        throw new Error(`Unknown expert: ${node.id}`);
      }
      return expert;
    });

    const combiner = genome.combiner
      ? { ...genome.combiner, combinerExpertId: genome.combiner.combinerExpertId }
      : undefined;

    if (genome.topology.type === 'mesh') {
      const outputs = await Promise.all(
        resolvedExperts.map((expert) => {
          const gene = nodeLookup.get(expert.id);
          const nodeOptions = { ...options };
          if (typeof gene?.temperature === 'number') {
            nodeOptions.temperature = gene.temperature;
          }
          return this.executeExpert(expert.id, prompt, nodeOptions, {
            adapterName: gene?.adapter,
            usePool: true,
          });
        })
      );
      return this.combineOutputs(outputs, combiner);
    }

    // Chain: sequential pipeline, each expert runs once
    if (genome.topology.type === 'chain') {
      const ordered = genome.nodes.map((node) => node.id);
      /** @type {string[]} */
      const outputs = [];
      let current = prompt;
      for (const id of ordered) {
        const gene = nodeLookup.get(id);
        const nodeOptions = { ...options };
        if (typeof gene?.temperature === 'number') {
          nodeOptions.temperature = gene.temperature;
        }
        const output = await this.executeExpert(id, current, nodeOptions, {
          adapterName: gene?.adapter,
        });
        outputs.push(output);
        current = output;
      }
      return combiner ? this.combineOutputs(outputs, combiner) : outputs[outputs.length - 1] ?? '';
    }

    // Ring: true circular - loops through all experts until convergence
    if (genome.topology.type === 'ring') {
      const ordered = genome.nodes.map((node) => node.id);
      const maxIterations = genome.topology.maxIterations ?? 3;
      /** @type {string[]} */
      const outputs = [];
      let current = prompt;
      let prevOutput = '';

      for (let iter = 0; iter < maxIterations; iter++) {
        for (const id of ordered) {
          const gene = nodeLookup.get(id);
          const nodeOptions = { ...options };
          if (typeof gene?.temperature === 'number') {
            // Decay temperature across iterations
            nodeOptions.temperature = gene.temperature * Math.pow(0.9, iter);
          }
          const output = await this.executeExpert(id, current, nodeOptions, {
            adapterName: gene?.adapter,
          });
          outputs.push(output);
          current = output;
        }

        // Check convergence after each full loop
        if (current === prevOutput || this.computeOutputSimilarity(current, prevOutput) > 0.95) {
          break;
        }
        prevOutput = current;
      }

      return combiner ? this.combineOutputs(outputs, combiner) : current;
    }

    const outputs = await this.executeGraph(genome, prompt, options, router);
    if (outputs.length === 1) {
      return outputs[0];
    }
    return this.combineOutputs(outputs, combiner);
  }

  /**
   * @param {NetworkGenome} genome
   * @param {string} prompt
   * @param {GenerateOptions} options
   * @param {TopologyRouter} [router]
   * @returns {Promise<string[]>}
   * @private
   */
  async executeGraph(genome, prompt, options, router) {
    /** @type {Map<string, NetworkEdgeGene[]>} */
    const outgoing = new Map();
    /** @type {Map<string, NetworkEdgeGene[]>} */
    const incoming = new Map();

    for (const edge of genome.edges) {
      if (!outgoing.has(edge.from)) outgoing.set(edge.from, []);
      if (!incoming.has(edge.to)) incoming.set(edge.to, []);
      /** @type {NetworkEdgeGene[]} */ (outgoing.get(edge.from)).push(edge);
      /** @type {NetworkEdgeGene[]} */ (incoming.get(edge.to)).push(edge);
    }

    for (const edges of outgoing.values()) {
      edges.sort((a, b) => b.weight - a.weight);
    }

    const rootId =
      genome.nodes.find((node) => !incoming.has(node.id))?.id || genome.nodes[0]?.id;
    if (!rootId) return [];

    const maxDepth = genome.topology.depth ?? genome.nodes.length;
    /** @type {Map<string, string>} */
    const outputs = new Map();
    /** @type {Set<string>} */
    const executed = new Set();
    /** @type {Map<string, string[]>} */
    let frontier = new Map();
    frontier.set(rootId, [prompt]);

    for (let depth = 0; depth < maxDepth && frontier.size > 0; depth++) {
      const entries = Array.from(frontier.entries());
      frontier = new Map();

      const levelOutputs = await Promise.all(
        entries.map(async ([nodeId, inputs]) => {
          if (executed.has(nodeId)) {
            return { nodeId, output: outputs.get(nodeId) ?? '' };
          }

          const gene = genome.nodes.find((node) => node.id === nodeId);
          const nodeOptions = { ...options };
          if (typeof gene?.temperature === 'number') {
            nodeOptions.temperature = gene.temperature;
          }
          const inputPrompt = inputs.join('\n\n');
          const output = await this.executeExpert(nodeId, inputPrompt, nodeOptions, {
            adapterName: gene?.adapter,
            usePool: Boolean(this.pipelinePool),
          });
          outputs.set(nodeId, output);
          executed.add(nodeId);
          return { nodeId, output };
        })
      );

      for (const { nodeId, output } of levelOutputs) {
        const edges = outgoing.get(nodeId) || [];
        if (edges.length === 0) continue;

        const candidateExperts = edges
          .map((edge) => this.getExpert(edge.to))
          .filter((expert) => Boolean(expert));

        let selectedExperts = /** @type {ExpertNode[]} */ (candidateExperts);
        if (router && candidateExperts.length > 0) {
          const parent = this.getExpert(nodeId);
          if (parent) {
            const routed = await router({
              parent,
              prompt: output,
              options,
              children: /** @type {ExpertNode[]} */ (candidateExperts),
              outputs,
            });
            if (Array.isArray(routed)) {
              selectedExperts = routed;
            } else if (routed) {
              selectedExperts = [routed];
            }
          }
        }

        const branchLimit = genome.topology.branchingFactor ?? selectedExperts.length;
        for (const expert of selectedExperts.slice(0, branchLimit)) {
          if (!frontier.has(expert.id)) {
            frontier.set(expert.id, []);
          }
          /** @type {string[]} */ (frontier.get(expert.id)).push(output);
        }
      }
    }

    /** @type {string[]} */
    const leaves = [];
    for (const node of genome.nodes) {
      const hasChildren = (outgoing.get(node.id) || []).length > 0;
      if (!hasChildren && outputs.has(node.id)) {
        leaves.push(/** @type {string} */ (outputs.get(node.id)));
      }
    }

    if (leaves.length > 0) return leaves;

    return Array.from(outputs.values());
  }

  /**
   * @param {AsyncGenerator<string>} generator
   * @returns {Promise<string>}
   * @private
   */
  async collectText(generator) {
    /** @type {string[]} */
    const chunks = [];
    for await (const token of generator) {
      chunks.push(token);
    }
    return chunks.join('');
  }
}
