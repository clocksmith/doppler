

import { ExpertRouter } from './expert-router.js';
import { MultiModelRecorder } from '../gpu/multi-model-recorder.js';



export class MultiModelNetwork {
  
  pipeline;

  
  loader;

  
  router;

  
  experts;

  
  sharedPrefix = null;

  
  busy = false;

  
  pipelinePool = null;

  
  recorder = null;

  
  combiner = { type: 'weighted' };

  
  constructor(pipeline, loader, pool, recorder) {
    this.pipeline = pipeline;
    this.loader = loader || null;
    this.router = new ExpertRouter();
    this.experts = new Map();
    this.pipelinePool = pool || null;
    this.recorder = recorder || null;
  }

  
  setRecorder(recorder) {
    this.recorder = recorder;
  }

  
  getRecorder() {
    return this.recorder;
  }

  
  setPipelinePool(pool) {
    this.pipelinePool = pool;
  }

  
  registerExpert(node) {
    this.experts.set(node.id, node);
    this.router.registerExpert(node);
  }

  
  getExpert(id) {
    return this.experts.get(id) || null;
  }

  
  listExperts() {
    return Array.from(this.experts.values());
  }

  
  setCombiner(config) {
    this.combiner = config;
  }

  
  async setSharedPrefix(prompt, options = {}) {
    const snapshot = this.recorder
      ? await this.recorder.computeSharedPrefix(this.pipeline, prompt, options)
      : await this.pipeline.prefillKVOnly(prompt, options);
    this.sharedPrefix = snapshot;
    this.pipelinePool?.setSharedPrefixSnapshot(snapshot);
    return snapshot;
  }

  
  setSharedPrefixSnapshot(snapshot) {
    this.sharedPrefix = snapshot;
    if (this.recorder) {
      this.recorder.setSharedPrefix(snapshot);
    }
    this.pipelinePool?.setSharedPrefixSnapshot(snapshot);
  }

  
  getSharedPrefixSnapshot() {
    return this.recorder?.getSharedPrefix() ?? this.sharedPrefix;
  }

  
  resolveAdapter(expert, adapterName, adapterOverride) {
    if (adapterOverride) return adapterOverride;
    const resolvedName = adapterName || expert.adapterName;
    if (resolvedName && this.loader) {
      return this.loader.getAdapter(resolvedName);
    }
    return expert.adapter || null;
  }

  
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

  
  async executeChain(expertIds, prompt, options = {}) {
    
    const outputs = [];
    let currentPrompt = prompt;

    for (const id of expertIds) {
      const output = await this.executeExpert(id, currentPrompt, options);
      outputs.push(output);
      currentPrompt = output;
    }

    return outputs;
  }

  
  async executeRing(expertIds, prompt, options = {}) {
    return this.executeChain(expertIds, prompt, options);
  }

  
  async executeBatch(tasks, options = {}) {
    
    const grouped = new Map();
    for (const task of tasks) {
      const expert = this.getExpert(task.expertId);
      const adapterKey = expert?.adapterName || '__base__';
      if (!grouped.has(adapterKey)) grouped.set(adapterKey, []);
       (grouped.get(adapterKey)).push(task);
    }

    
    const results = {};
    for (const group of grouped.values()) {
      for (const task of group) {
        results[task.id] = await this.executeExpert(task.expertId, task.prompt, options);
      }
    }

    return results;
  }

  
  async executeParallel(tasks, options = {}) {
    if (!this.pipelinePool) {
      if (this.busy) {
        throw new Error('MultiModelNetwork is busy. Parallel execution requires separate pipelines.');
      }
      this.busy = true;
      try {
        const entries = await Promise.all(
          tasks.map(async (task) =>  ([task.id, await this.executeExpert(task.expertId, task.prompt, options)]))
        );
        return Object.fromEntries(entries);
      } finally {
        this.busy = false;
      }
    }

    const entries = await Promise.all(
      tasks.map(async (task) => {
        const output = await this.executeExpert(task.expertId, task.prompt, options, { usePool: true });
        return  ([task.id, output]);
      })
    );

    return Object.fromEntries(entries);
  }

  
  selectExpertsByEmbedding(embedding, topK = 1) {
    return  (this.router.selectByEmbedding(embedding, topK));
  }

  
  async combineOutputs(outputs, combinerOverride) {
    if (outputs.length === 0) return '';

    const combiner = combinerOverride ?? this.combiner;

    if (combiner.type === 'voting') {
      
      const counts = new Map();
      for (const output of outputs) {
        counts.set(output, (counts.get(output) || 0) + 1);
      }
      const sorted = Array.from(counts.entries()).sort((a, b) => b[1] - a[1]);
      return sorted[0][0];
    }

    if (combiner.type === 'weighted') {
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

    throw new Error(`Unknown combiner type: ${combiner.type}`);
  }

  
  async executeGenome(genome, prompt, options = {}, router) {
    
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

    const combiner = genome.combiner ? { ...genome.combiner } : undefined;

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

    if (genome.topology.type === 'ring') {
      throw new Error('Topology type "ring" is an orchestration policy and must run in Reploid.');
    }

    const outputs = await this.executeGraph(genome, prompt, options, router);
    if (outputs.length === 1) {
      return outputs[0];
    }
    return this.combineOutputs(outputs, combiner);
  }

  
  async executeGraph(genome, prompt, options, router) {
    
    const outgoing = new Map();
    
    const incoming = new Map();

    for (const edge of genome.edges) {
      if (!outgoing.has(edge.from)) outgoing.set(edge.from, []);
      if (!incoming.has(edge.to)) incoming.set(edge.to, []);
       (outgoing.get(edge.from)).push(edge);
       (incoming.get(edge.to)).push(edge);
    }

    for (const edges of outgoing.values()) {
      edges.sort((a, b) => b.weight - a.weight);
    }

    const rootId =
      genome.nodes.find((node) => !incoming.has(node.id))?.id || genome.nodes[0]?.id;
    if (!rootId) return [];

    const maxDepth = genome.topology.depth ?? genome.nodes.length;
    
    const outputs = new Map();
    
    const executed = new Set();
    
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

        let selectedExperts =  (candidateExperts);
        if (router && candidateExperts.length > 0) {
          const parent = this.getExpert(nodeId);
          if (parent) {
            const routed = await router({
              parent,
              prompt: output,
              options,
              children:  (candidateExperts),
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
           (frontier.get(expert.id)).push(output);
        }
      }
    }

    
    const leaves = [];
    for (const node of genome.nodes) {
      const hasChildren = (outgoing.get(node.id) || []).length > 0;
      if (!hasChildren && outputs.has(node.id)) {
        leaves.push( (outputs.get(node.id)));
      }
    }

    if (leaves.length > 0) return leaves;

    return Array.from(outputs.values());
  }

  
  async collectText(generator) {
    
    const chunks = [];
    for await (const token of generator) {
      chunks.push(token);
    }
    return chunks.join('');
  }
}
