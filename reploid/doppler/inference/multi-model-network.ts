/**
 * Multi-model execution network for FunctionGemma experts.
 *
 * @module inference/multi-model-network
 */

import type { GenerateOptions } from './pipeline.js';
import type { InferencePipeline, KVCacheSnapshot } from './pipeline.js';
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
  type: 'weighted' | 'voting' | 'llm-merge';
  weights?: number[];
  combinerExpertId?: string;
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

export class MultiModelNetwork {
  private pipeline: InferencePipeline;
  private loader: MultiModelLoader | null;
  private router: ExpertRouter;
  private experts: Map<string, ExpertNode>;
  private sharedPrefix: KVCacheSnapshot | null = null;
  private busy = false;
  private pipelinePool: MultiPipelinePool | null = null;
  private recorder: MultiModelRecorder | null = null;
  private combiner: CombinerConfig = { type: 'weighted' };

  constructor(
    pipeline: InferencePipeline,
    loader?: MultiModelLoader,
    pool?: MultiPipelinePool,
    recorder?: MultiModelRecorder
  ) {
    this.pipeline = pipeline;
    this.loader = loader || null;
    this.router = new ExpertRouter();
    this.experts = new Map();
    this.pipelinePool = pool || null;
    this.recorder = recorder || null;
  }

  setRecorder(recorder: MultiModelRecorder | null): void {
    this.recorder = recorder;
  }

  getRecorder(): MultiModelRecorder | null {
    return this.recorder;
  }

  setPipelinePool(pool: MultiPipelinePool | null): void {
    this.pipelinePool = pool;
  }

  registerExpert(node: ExpertNode): void {
    this.experts.set(node.id, node);
    this.router.registerExpert(node);
  }

  getExpert(id: string): ExpertNode | null {
    return this.experts.get(id) || null;
  }

  listExperts(): ExpertNode[] {
    return Array.from(this.experts.values());
  }

  setCombiner(config: CombinerConfig): void {
    this.combiner = config;
  }

  async setSharedPrefix(prompt: string, options: GenerateOptions = {}): Promise<KVCacheSnapshot> {
    const snapshot = this.recorder
      ? await this.recorder.computeSharedPrefix(this.pipeline, prompt, options)
      : await this.pipeline.prefillKVOnly(prompt, options);
    this.sharedPrefix = snapshot;
    this.pipelinePool?.setSharedPrefixSnapshot(snapshot);
    return snapshot;
  }

  setSharedPrefixSnapshot(snapshot: KVCacheSnapshot | null): void {
    this.sharedPrefix = snapshot;
    if (this.recorder) {
      this.recorder.setSharedPrefix(snapshot);
    }
    this.pipelinePool?.setSharedPrefixSnapshot(snapshot);
  }

  getSharedPrefixSnapshot(): KVCacheSnapshot | null {
    return this.recorder?.getSharedPrefix() ?? this.sharedPrefix;
  }

  private resolveAdapter(
    expert: ExpertNode,
    adapterName?: string,
    adapterOverride?: LoRAAdapter | null
  ): LoRAAdapter | null {
    if (adapterOverride) return adapterOverride;
    const resolvedName = adapterName || expert.adapterName;
    if (resolvedName && this.loader) {
      return this.loader.getAdapter(resolvedName);
    }
    return expert.adapter || null;
  }

  async executeExpert(
    expertId: string,
    prompt: string,
    options: GenerateOptions = {},
    overrides: { adapterName?: string; adapter?: LoRAAdapter | null; prefix?: KVCacheSnapshot | null; usePool?: boolean } = {}
  ): Promise<string> {
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

  async executeRing(expertIds: string[], prompt: string, options: GenerateOptions = {}): Promise<string[]> {
    const outputs: string[] = [];
    let currentPrompt = prompt;

    for (const id of expertIds) {
      const output = await this.executeExpert(id, currentPrompt, options);
      outputs.push(output);
      currentPrompt = output;
    }

    return outputs;
  }

  /**
   * Temporal Self-Ring: Same model at N temporal states for self-reflective improvement.
   * Based on Gödel Agent, RISE, and Reflexion research.
   *
   * @param expertId - The expert to use for all temporal states (typically 'base')
   * @param task - Task with description, maxTokens, and optional convergenceThreshold
   * @param config - Configuration for turns, temperature decay, and shortcuts
   * @returns Final output, full history, and turns used
   */
  async executeTemporalRing(
    expertId: string,
    task: { description: string; maxTokens?: number; convergenceThreshold?: number },
    config: {
      turns?: number;
      temperatureStart?: number;
      temperatureDecay?: number;
      temperatureMin?: number;
      shortcutInterval?: number; // For Möbius Ring variant
      enableShortcuts?: boolean;
    } = {}
  ): Promise<{
    finalOutput: string;
    history: Array<{ turn: number; output: string; timestamp: number; role: 'seed' | 'reflect' | 'refine' }>;
    turnsUsed: number;
    converged: boolean;
  }> {
    const {
      turns = 5,
      temperatureStart = 0.8,
      temperatureDecay = 0.15,
      temperatureMin = 0.1,
      shortcutInterval = 2,
      enableShortcuts = false,
    } = config;

    const history: Array<{ turn: number; output: string; timestamp: number; role: 'seed' | 'reflect' | 'refine' }> = [];
    let currentOutput = '';
    let converged = false;

    for (let t = 0; t < turns; t++) {
      const temperature = Math.max(temperatureMin, temperatureStart - t * temperatureDecay);
      const role: 'seed' | 'reflect' | 'refine' = t === 0 ? 'seed' : t % 2 === 1 ? 'reflect' : 'refine';

      let prompt = this.buildTemporalPrompt(task.description, t, history, currentOutput, role);

      // Möbius Ring: Add shortcuts to earlier temporal states
      if (enableShortcuts && t >= shortcutInterval) {
        const shortcutIdx = t - shortcutInterval;
        const shortcutEntry = history[shortcutIdx];
        if (shortcutEntry) {
          prompt += `\n\n### Earlier Context (turn ${shortcutIdx}):\n${shortcutEntry.output}`;
        }
      }

      currentOutput = await this.executeExpert(expertId, prompt, {
        ...task,
        temperature,
      });

      history.push({
        turn: t,
        output: currentOutput,
        timestamp: Date.now(),
        role,
      });

      // Convergence detection
      if (this.detectTemporalConvergence(currentOutput, history, task.convergenceThreshold)) {
        converged = true;
        break;
      }
    }

    return {
      finalOutput: currentOutput,
      history,
      turnsUsed: history.length,
      converged,
    };
  }

  private buildTemporalPrompt(
    taskDescription: string,
    turn: number,
    history: Array<{ turn: number; output: string; role: string }>,
    lastOutput: string,
    role: 'seed' | 'reflect' | 'refine'
  ): string {
    if (role === 'seed') {
      return `Generate code for: ${taskDescription}\n\nOutput JSON: { "code": string, "reasoning": string }`;
    }

    if (role === 'reflect') {
      return `Review this code and identify issues:\n\n${lastOutput}\n\nOutput JSON: { "issues": string[], "severity": string, "suggestions": string[] }`;
    }

    // role === 'refine'
    const originalTurn = turn - 2;
    const originalOutput = history[originalTurn]?.output || lastOutput;
    return `Improve the code based on this feedback:\n\nOriginal code:\n${originalOutput}\n\nCritique:\n${lastOutput}\n\nOutput improved JSON: { "code": string, "changes": string[], "converged": boolean }`;
  }

  private detectTemporalConvergence(
    currentOutput: string,
    history: Array<{ output: string }>,
    threshold?: number
  ): boolean {
    if (history.length < 2) return false;

    // Check for explicit convergence signal
    if (currentOutput.includes('"converged": true') || currentOutput.includes('"converged":true')) {
      return true;
    }

    // Check for output stability (same as previous)
    const prevOutput = history[history.length - 1]?.output;
    if (currentOutput === prevOutput) {
      return true;
    }

    // Optional: similarity-based convergence (if threshold provided)
    if (threshold !== undefined && prevOutput) {
      const similarity = this.computeOutputSimilarity(currentOutput, prevOutput);
      if (similarity >= threshold) {
        return true;
      }
    }

    return false;
  }

  private computeOutputSimilarity(a: string, b: string): number {
    // Simple Jaccard similarity on tokens
    const tokensA = new Set(a.toLowerCase().split(/\s+/));
    const tokensB = new Set(b.toLowerCase().split(/\s+/));
    const intersection = new Set([...tokensA].filter((x) => tokensB.has(x)));
    const union = new Set([...tokensA, ...tokensB]);
    return union.size > 0 ? intersection.size / union.size : 0;
  }

  async executeBatch(tasks: ExpertTask[], options: GenerateOptions = {}): Promise<Record<string, string>> {
    const grouped = new Map<string, ExpertTask[]>();
    for (const task of tasks) {
      const expert = this.getExpert(task.expertId);
      const adapterKey = expert?.adapterName || '__base__';
      if (!grouped.has(adapterKey)) grouped.set(adapterKey, []);
      grouped.get(adapterKey)!.push(task);
    }

    const results: Record<string, string> = {};
    for (const group of grouped.values()) {
      for (const task of group) {
        results[task.id] = await this.executeExpert(task.expertId, task.prompt, options);
      }
    }

    return results;
  }

  async executeParallel(tasks: ExpertTask[], options: GenerateOptions = {}): Promise<Record<string, string>> {
    if (!this.pipelinePool) {
      if (this.busy) {
        throw new Error('MultiModelNetwork is busy. Parallel execution requires separate pipelines.');
      }
      this.busy = true;
      try {
        const entries = await Promise.all(
          tasks.map(async (task) => [task.id, await this.executeExpert(task.expertId, task.prompt, options)] as const)
        );
        return Object.fromEntries(entries);
      } finally {
        this.busy = false;
      }
    }

    const entries = await Promise.all(
      tasks.map(async (task) => {
        const output = await this.executeExpert(task.expertId, task.prompt, options, { usePool: true });
        return [task.id, output] as const;
      })
    );

    return Object.fromEntries(entries);
  }

  selectExpertsByEmbedding(embedding: number[], topK: number = 1): ExpertNode[] {
    return this.router.selectByEmbedding(embedding, topK) as ExpertNode[];
  }

  async combineOutputs(outputs: string[], combinerOverride?: CombinerConfig): Promise<string> {
    if (outputs.length === 0) return '';

    const combiner = combinerOverride ?? this.combiner;

    if (combiner.type === 'voting') {
      const counts = new Map<string, number>();
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

  async executeGenome(
    genome: NetworkGenome,
    prompt: string,
    options: GenerateOptions = {},
    router?: TopologyRouter
  ): Promise<string> {
    const nodeLookup = new Map<string, NetworkNodeGene>();
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

    if (genome.topology.type === 'ring') {
      const ordered = genome.nodes.map((node) => node.id);
      const outputs: string[] = [];
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

    const outputs = await this.executeGraph(genome, prompt, options, router);
    if (outputs.length === 1) {
      return outputs[0];
    }
    return this.combineOutputs(outputs, combiner);
  }

  private async executeGraph(
    genome: NetworkGenome,
    prompt: string,
    options: GenerateOptions,
    router?: TopologyRouter
  ): Promise<string[]> {
    const outgoing = new Map<string, NetworkEdgeGene[]>();
    const incoming = new Map<string, NetworkEdgeGene[]>();

    for (const edge of genome.edges) {
      if (!outgoing.has(edge.from)) outgoing.set(edge.from, []);
      if (!incoming.has(edge.to)) incoming.set(edge.to, []);
      outgoing.get(edge.from)!.push(edge);
      incoming.get(edge.to)!.push(edge);
    }

    for (const edges of outgoing.values()) {
      edges.sort((a, b) => b.weight - a.weight);
    }

    const rootId =
      genome.nodes.find((node) => !incoming.has(node.id))?.id || genome.nodes[0]?.id;
    if (!rootId) return [];

    const maxDepth = genome.topology.depth ?? genome.nodes.length;
    const outputs = new Map<string, string>();
    const executed = new Set<string>();
    let frontier = new Map<string, string[]>();
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
          .filter((expert): expert is ExpertNode => Boolean(expert));

        let selectedExperts = candidateExperts;
        if (router && candidateExperts.length > 0) {
          const parent = this.getExpert(nodeId);
          if (parent) {
            const routed = await router({
              parent,
              prompt: output,
              options,
              children: candidateExperts,
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
          frontier.get(expert.id)!.push(output);
        }
      }
    }

    const leaves: string[] = [];
    for (const node of genome.nodes) {
      const hasChildren = (outgoing.get(node.id) || []).length > 0;
      if (!hasChildren && outputs.has(node.id)) {
        leaves.push(outputs.get(node.id)!);
      }
    }

    if (leaves.length > 0) return leaves;

    return Array.from(outputs.values());
  }

  private async collectText(generator: AsyncGenerator<string>): Promise<string> {
    const chunks: string[] = [];
    for await (const token of generator) {
      chunks.push(token);
    }
    return chunks.join('');
  }
}
