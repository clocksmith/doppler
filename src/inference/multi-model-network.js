

import { ExpertRouter } from './expert-router.js';
import { MultiModelRecorder } from '../gpu/multi-model-recorder.js';
import { applyRepetitionPenalty, sample, getTopK } from './pipeline/sampling.js';
import { finalizeLogits, extractLastPositionLogits } from './pipeline/logits.js';
import { isStopToken } from './pipeline/init.js';
import { mergeMultipleLogits } from '../gpu/kernels/logit-merge.js';
import { releaseBuffer, readBuffer } from '../memory/buffer-pool.js';

const MIN_AGREEMENT_WEIGHT = 1e-4;

function buildAgreementWeights(logitsList, agreementTopK, decode) {
  const topKByModel = logitsList.map((logits) => getTopK(logits, agreementTopK, decode));

  const counts = new Map();
  const probSums = new Map();
  for (const top of topKByModel) {
    for (const entry of top) {
      const current = counts.get(entry.token) ?? 0;
      counts.set(entry.token, current + 1);
      probSums.set(entry.token, (probSums.get(entry.token) ?? 0) + entry.prob);
    }
  }

  let agreedToken = topKByModel[0]?.[0]?.token ?? 0;
  let bestCount = -1;
  let bestProbSum = -Infinity;
  for (const [token, count] of counts.entries()) {
    const probSum = probSums.get(token) ?? 0;
    if (count > bestCount || (count === bestCount && probSum > bestProbSum)) {
      agreedToken = token;
      bestCount = count;
      bestProbSum = probSum;
    }
  }

  const weights = topKByModel.map((top) => {
    const match = top.find((entry) => entry.token === agreedToken);
    return Math.max(match?.prob ?? 0, MIN_AGREEMENT_WEIGHT);
  });

  const total = weights.reduce((sum, weight) => sum + weight, 0);
  const normalized = total > 0
    ? weights.map((weight) => weight / total)
    : weights.map(() => 1 / weights.length);

  const agreement = bestCount > 0 ? bestCount / logitsList.length : 0;

  return { weights: normalized, agreedToken, agreement };
}

function mergeLogitsCpu(logitsList, weights) {
  const vocabSize = logitsList[0].length;
  const merged = new Float32Array(vocabSize);

  for (let i = 0; i < logitsList.length; i++) {
    const logits = logitsList[i];
    const weight = weights[i] ?? 0;
    for (let j = 0; j < vocabSize; j++) {
      merged[j] += logits[j] * weight;
    }
  }

  return merged;
}



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

  async *generateWithABE(prompt, options = {}) {
    if (!this.pipelinePool) {
      throw new Error('MultiModelNetwork requires a pipeline pool for ABE generation.');
    }

    const expertIds = options.expertIds ?? this.listExperts().map((expert) => expert.id);
    if (!expertIds.length) {
      throw new Error('No experts available for ABE generation.');
    }

    const voterIds = options.voterIds ?? expertIds;
    const voterSet = new Set(voterIds);

    const pipelines = new Map();
    for (const expertId of expertIds) {
      const expert = this.getExpert(expertId);
      if (!expert) {
        throw new Error(`Unknown expert: ${expertId}`);
      }
      const pipeline = await this.pipelinePool.getPipeline(expertId);
      pipeline.setLoRAAdapter(this.resolveAdapter(expert, options.adapterName, options.adapter));
      pipelines.set(expertId, pipeline);
    }

    const basePipeline = pipelines.get(voterIds[0]) ?? pipelines.get(expertIds[0]);
    if (!basePipeline) {
      throw new Error('Base pipeline unavailable for ABE generation.');
    }

    const runtimeDefaults = basePipeline.runtimeConfig.inference;
    const samplingDefaults = runtimeDefaults.sampling;
    const batchingDefaults = runtimeDefaults.batching;
    const generationDefaults = runtimeDefaults.generation;

    const opts = {
      maxTokens: options.maxTokens ?? batchingDefaults.maxTokens,
      temperature: options.temperature ?? samplingDefaults.temperature,
      topP: options.topP ?? samplingDefaults.topP,
      topK: options.topK ?? samplingDefaults.topK,
      repetitionPenalty: options.repetitionPenalty ?? samplingDefaults.repetitionPenalty,
      stopSequences: options.stopSequences ?? [],
      useChatTemplate: options.useChatTemplate
        ?? basePipeline.runtimeConfig.inference.chatTemplate?.enabled
        ?? basePipeline.modelConfig?.chatTemplateEnabled
        ?? false,
      debug: options.debug ?? basePipeline.debug,
      debugLayers: options.debugLayers,
      profile: options.profile ?? generationDefaults.profile,
      disableCommandBatching: options.disableCommandBatching ?? generationDefaults.disableCommandBatching,
      disableMultiTokenDecode: options.disableMultiTokenDecode ?? generationDefaults.disableMultiTokenDecode,
      batchSize: options.batchSize ?? batchingDefaults.batchSize,
      stopCheckMode: options.stopCheckMode ?? batchingDefaults.stopCheckMode,
    };

    const agreementTopK = options.agreementTopK ?? Math.min(opts.topK ?? 5, 8);
    const minAgreement = options.minAgreement ?? 0;
    const mergeOnGpu = options.mergeOnGpu ?? false;
    const padTokenId = basePipeline.tokenizer?.getSpecialTokens?.()?.pad ?? null;
    const stopTokenIds = basePipeline.modelConfig.stopTokenIds;
    const eosToken = basePipeline.tokenizer?.getSpecialTokens?.()?.eos;

    let prefix = options.prefix ?? this.getSharedPrefixSnapshot();
    let prefetchedFromBase = false;
    if (!prefix) {
      prefix = await basePipeline.prefillKVOnly(prompt, opts);
      prefetchedFromBase = true;
    }
    for (const pipeline of pipelines.values()) {
      if (prefetchedFromBase && pipeline === basePipeline) continue;
      pipeline.applyKVCacheSnapshot(prefix);
    }

    const currentIds = new Map();
    for (const expertId of expertIds) {
      currentIds.set(expertId, [...prefix.tokens]);
    }

    const generatedIds = [...prefix.tokens];
    const promptTokenCount = prefix.tokens.length;
    let tokensGenerated = 0;

    while (tokensGenerated < opts.maxTokens) {
      if (options.signal?.aborted) break;

      const voterResults = await Promise.all(
        voterIds.map(async (expertId) => {
          const pipeline = pipelines.get(expertId);
          if (!pipeline) {
            throw new Error(`Missing pipeline for expert ${expertId}`);
          }
          const ids = currentIds.get(expertId);
          if (!ids) {
            throw new Error(`Missing token history for expert ${expertId}`);
          }
          return pipeline.decodeStepLogits(ids, opts);
        })
      );

      const logitsList = voterResults.map((result) => result.logits);
      const { weights, agreement } = buildAgreementWeights(
        logitsList,
        agreementTopK,
        (tokens) => basePipeline.tokenizer?.decode?.(tokens) ?? ''
      );

      const normalizedWeights = agreement < minAgreement
        ? weights.map(() => 1 / weights.length)
        : weights;

      let mergedLogits = null;
      const rawVocabSize = voterResults[0]?.rawVocabSize ?? basePipeline.modelConfig.vocabSize;
      const canMergeOnGpu = mergeOnGpu
        && basePipeline.modelConfig.finalLogitSoftcapping == null
        && voterResults.every((result) => result.logitsBuffer && result.logitsDtype === 'f32')
        && voterResults.every((result) => result.rawVocabSize === rawVocabSize);

      if (canMergeOnGpu) {
        const buffers = voterResults.map((result) => result.logitsBuffer);
        const mergedBuffer = await mergeMultipleLogits(buffers, rawVocabSize, normalizedWeights, 1.0);
        const mergedData = await readBuffer(mergedBuffer, rawVocabSize * 4);
        releaseBuffer(mergedBuffer);
        const rawMerged = new Float32Array(mergedData);
        const finalized = await finalizeLogits(
          rawMerged,
          1,
          rawVocabSize,
          basePipeline.modelConfig.vocabSize,
          basePipeline.modelConfig,
          basePipeline.runtimeConfig.shared.debug.probes
        );
        mergedLogits = extractLastPositionLogits(
          finalized,
          1,
          basePipeline.modelConfig.vocabSize
        );
      } else {
        mergedLogits = mergeLogitsCpu(logitsList, normalizedWeights);
      }

      for (const result of voterResults) {
        if (result.logitsBuffer) {
          releaseBuffer(result.logitsBuffer);
        }
      }

      applyRepetitionPenalty(mergedLogits, generatedIds, opts.repetitionPenalty);
      const nextToken = sample(mergedLogits, {
        temperature: opts.temperature,
        topP: opts.topP,
        topK: opts.topK,
        padTokenId,
      });

      generatedIds.push(nextToken);
      tokensGenerated++;

      for (const expertId of expertIds) {
        const ids = currentIds.get(expertId);
        if (!ids) continue;
        ids.push(nextToken);
      }

      const followers = expertIds.filter((expertId) => !voterSet.has(expertId));
      if (followers.length > 0) {
        await Promise.all(
          followers.map(async (expertId) => {
            const pipeline = pipelines.get(expertId);
            if (!pipeline) return;
            await pipeline.advanceWithToken(nextToken, opts);
          })
        );
      }

      const tokenText = basePipeline.tokenizer.decode([nextToken], true, false);
      yield tokenText;
      if (options.onToken) options.onToken(nextToken, tokenText);

      if (isStopToken(nextToken, stopTokenIds, eosToken)) break;
      if (opts.stopSequences.length > 0) {
        const fullText = basePipeline.tokenizer.decode(generatedIds.slice(promptTokenCount), false);
        if (opts.stopSequences.some((seq) => fullText.endsWith(seq))) break;
      }
    }
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
