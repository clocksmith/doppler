/** Real prefill -> sample -> incremental decode session control for Pack v2. */

function requireGenerationOptions(options) {
  const requiredNumbers = ['maxTokens', 'temperature', 'topP', 'topK', 'repetitionPenalty', 'repetitionPenaltyWindow'];
  for (const field of requiredNumbers) {
    if (!Number.isFinite(options[field])) throw new Error(`Pack generation requires explicit ${field}.`);
  }
  if (!Number.isInteger(options.maxTokens) || options.maxTokens < 1) throw new Error('Pack generation maxTokens must be a positive integer.');
  if (!Number.isInteger(options.topK) || options.topK < 0) throw new Error('Pack generation topK must be a non-negative integer.');
  if (!Number.isInteger(options.repetitionPenaltyWindow) || options.repetitionPenaltyWindow < 1) {
    throw new Error('Pack generation repetitionPenaltyWindow must be a positive integer.');
  }
  if (options.temperature < 0 || options.topP <= 0 || options.topP > 1 || options.repetitionPenalty <= 0) {
    throw new Error('Pack generation sampling values are outside their valid ranges.');
  }
  if (options.temperature > 0 && !Number.isFinite(options.seed)) {
    throw new Error('Pack stochastic generation requires an explicit numeric seed.');
  }
  if (typeof options.useChatTemplate !== 'boolean') {
    throw new Error('Pack generation requires explicit useChatTemplate.');
  }
}

function applyRepetitionPenalty(logits, tokens, penalty, windowSize) {
  if (penalty === 1) return;
  for (const token of new Set(tokens.slice(-windowSize))) {
    if (token >= 0 && token < logits.length) logits[token] = logits[token] > 0 ? logits[token] / penalty : logits[token] * penalty;
  }
}

function seededRandom(seed) {
  const value = Math.sin(seed) * 10000;
  return value - Math.floor(value);
}

/** Pure host sampling over actual model logits. */
export function samplePackLogits(sourceLogits, contextTokens, options, tokenContract = {}) {
  const logits = Float32Array.from(sourceLogits || []);
  if (logits.length === 0) throw new Error('Pack execution returned empty logits.');
  applyRepetitionPenalty(logits, contextTokens, options.repetitionPenalty, options.repetitionPenaltyWindow);
  const suppressed = new Set(options.suppressTokenIds || []);
  if (Number.isInteger(tokenContract.padTokenId)) suppressed.add(tokenContract.padTokenId);
  for (const tokenId of suppressed) {
    if (Number.isInteger(tokenId) && tokenId >= 0 && tokenId < logits.length) logits[tokenId] = -Infinity;
  }
  const candidates = [];
  let nanCount = 0;
  let positiveInfinityCount = 0;
  let negativeInfinityCount = 0;
  for (let tokenId = 0; tokenId < logits.length; tokenId += 1) {
    const logit = logits[tokenId];
    if (Number.isFinite(logit)) {
      candidates.push({ tokenId, logit });
    } else if (Number.isNaN(logit)) {
      nanCount += 1;
    } else if (logit === Infinity) {
      positiveInfinityCount += 1;
    } else {
      negativeInfinityCount += 1;
    }
  }
  if (candidates.length === 0) {
    throw new Error(
      'Pack execution returned no finite sampling candidates '
      + `(logits=${logits.length}, nan=${nanCount}, +inf=${positiveInfinityCount}, -inf=${negativeInfinityCount}).`
    );
  }
  candidates.sort((left, right) => right.logit - left.logit || left.tokenId - right.tokenId);
  if (options.temperature === 0) return candidates[0].tokenId;
  const topK = options.topK > 0 ? Math.min(options.topK, candidates.length) : candidates.length;
  let filtered = candidates.slice(0, topK);
  const maxScaled = filtered[0].logit / options.temperature;
  let total = 0;
  for (const candidate of filtered) {
    candidate.probability = Math.exp(candidate.logit / options.temperature - maxScaled);
    total += candidate.probability;
  }
  for (const candidate of filtered) candidate.probability /= total;
  if (options.topP < 1) {
    let cumulative = 0;
    filtered = filtered.filter((candidate) => {
      if (cumulative >= options.topP) return false;
      cumulative += candidate.probability;
      return true;
    });
    const filteredTotal = filtered.reduce((sum, candidate) => sum + candidate.probability, 0);
    for (const candidate of filtered) candidate.probability /= filteredTotal;
  }
  const random = seededRandom(options.seed);
  let cumulative = 0;
  for (const candidate of filtered) {
    cumulative += candidate.probability;
    if (random < cumulative) return candidate.tokenId;
  }
  return filtered.at(-1).tokenId;
}

function shouldStop(tokenId, generatedTokens, options, tokenContract, program) {
  if (tokenContract.stopTokenIds?.includes(tokenId) || tokenId === tokenContract.eosTokenId) return true;
  if (!Array.isArray(options.stopSequences) || options.stopSequences.length === 0) return false;
  const text = program.decodeTokens(generatedTokens);
  return options.stopSequences.some((sequence) => text.endsWith(sequence));
}

/** @param {object} commandExecutor @param {object} resourceBinder @param {object} program */
export function createSessionController(commandExecutor, resourceBinder, program) {
  if (!commandExecutor || !resourceBinder || !program) {
    throw new Error('createSessionController requires commandExecutor, resourceBinder, and program.');
  }
  let closed = false;

  return {
    async *generateTokens(targetPlan, options = {}) {
      if (closed) throw new Error('Pack runtime session is closed.');
      requireGenerationOptions(options);
      if (options.signal?.aborted) throw new Error('Generation aborted before prefill.');
      const promptTokens = Array.isArray(options.promptTokens)
        ? [...options.promptTokens]
        : program.tokenize(options.prompt, { useChatTemplate: options.useChatTemplate });
      if (promptTokens.length === 0) throw new Error('Pack generation prompt must produce at least one token.');
      const dimensions = {
        seqLen: promptTokens.length,
        maxSeqLen: options.maxSeqLen,
        batchSize: 1,
      };
      if (!Number.isInteger(dimensions.maxSeqLen) || dimensions.maxSeqLen < promptTokens.length + options.maxTokens) {
        throw new Error('Pack generation requires maxSeqLen large enough for prompt and generated tokens.');
      }
      program.reset();
      resourceBinder.bindSlots(targetPlan.memoryLayout, dimensions);
      resourceBinder.writeSlot('input_tokens', Uint32Array.from(promptTokens));
      const contextTokens = [...promptTokens];
      const generatedTokens = [];
      const tokenContract = program.getTokenContract();
      let stepResult = null;
      try {
        const prefill = await commandExecutor.executePhase('prefill', targetPlan.phases.prefill, {
          signal: options.signal,
          modules: options.modules,
          context: { prompt: options.prompt ?? '', promptTokens, generationOptions: options },
        });
        stepResult = prefill.results.at(-1);
        for (let step = 0; step < options.maxTokens; step += 1) {
          if (options.signal?.aborted) throw new Error('Generation aborted during decode.');
          const tokenId = samplePackLogits(stepResult?.logits, contextTokens, options, tokenContract);
          program.releaseStepResult(stepResult);
          stepResult = null;
          generatedTokens.push(tokenId);
          contextTokens.push(tokenId);
          yield tokenId;
          if (shouldStop(tokenId, generatedTokens, options, tokenContract, program)) break;
          if (generatedTokens.length >= options.maxTokens) break;
          const decode = await commandExecutor.executePhase('decode', targetPlan.phases.decode, {
            signal: options.signal,
            modules: options.modules,
            context: { contextTokens, generationOptions: options },
          });
          stepResult = decode.results.at(-1);
        }
      } finally {
        program.releaseStepResult(stepResult);
        resourceBinder.releaseTransient();
      }
    },

    async close() {
      if (closed) return;
      closed = true;
      resourceBinder.releaseAll();
      await program.close();
    },
  };
}
