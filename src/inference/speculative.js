function createRng(seed) {
  let state = seed >>> 0;
  if (!state) state = 0x6d2b79f5;
  return () => {
    state |= 0;
    state = (state + 0x6d2b79f5) | 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function coerceLogitsVector(value, label) {
  if (value instanceof Float32Array) {
    if (value.length === 0) {
      throw new Error(`SpeculativeDecoder: ${label} must not be empty.`);
    }
    return value;
  }

  if (ArrayBuffer.isView(value)) {
    if (value.length === 0) {
      throw new Error(`SpeculativeDecoder: ${label} must not be empty.`);
    }
    return Float32Array.from(value);
  }

  if (Array.isArray(value)) {
    if (value.length === 0) {
      throw new Error(`SpeculativeDecoder: ${label} must not be empty.`);
    }

    if (typeof value[0] === 'number') {
      return Float32Array.from(value);
    }

    const last = value[value.length - 1];
    if (last instanceof Float32Array) {
      if (last.length === 0) {
        throw new Error(`SpeculativeDecoder: ${label} must not be empty.`);
      }
      return last;
    }
    if (ArrayBuffer.isView(last)) {
      if (last.length === 0) {
        throw new Error(`SpeculativeDecoder: ${label} must not be empty.`);
      }
      return Float32Array.from(last);
    }
    if (Array.isArray(last) && (last.length === 0 || typeof last[0] === 'number')) {
      if (last.length === 0) {
        throw new Error(`SpeculativeDecoder: ${label} must not be empty.`);
      }
      return Float32Array.from(last);
    }
  }

  throw new Error(
    `SpeculativeDecoder: ${label} must be a numeric logits vector (Float32Array or number[]).`
  );
}

function assertTemperature(temperature, label) {
  if (!Number.isFinite(temperature) || temperature <= 0) {
    throw new Error(`SpeculativeDecoder: ${label} must be a positive finite number.`);
  }
}

export class SpeculativeDecoder {
  numDraftTokens;
  maxRejectionRetries;
  enableTreeDraft;
  temperature;
  random;
  draftModel = null;
  mainModel = null;
  stats = {
    totalDrafted: 0,
    totalAccepted: 0,
    totalRejected: 0,
    averageAcceptRate: 0,
  };

  constructor(config = {}) {
    if (config.numDraftTokens == null) {
      throw new Error('SpeculativeDecoder requires numDraftTokens.');
    }
    if (config.maxRejectionRetries == null) {
      throw new Error('SpeculativeDecoder requires maxRejectionRetries.');
    }
    if (config.enableTreeDraft == null) {
      throw new Error('SpeculativeDecoder requires enableTreeDraft.');
    }
    if (config.temperature == null) {
      throw new Error('SpeculativeDecoder requires temperature.');
    }
    if (!Number.isFinite(config.randomSeed)) {
      throw new Error('SpeculativeDecoder requires randomSeed.');
    }

    assertTemperature(config.temperature, 'temperature');
    this.numDraftTokens = config.numDraftTokens;
    this.maxRejectionRetries = config.maxRejectionRetries;
    this.enableTreeDraft = config.enableTreeDraft;
    this.temperature = config.temperature;

    this.random = createRng(Math.floor(config.randomSeed));
  }

  setDraftModel(model) {
    this.draftModel = model;
  }

  setMainModel(model) {
    this.mainModel = model;
  }

  async _forwardForLogits(model, inputIds, kvCache, label) {
    const result = await model.forward(inputIds, kvCache);
    if (!result || typeof result !== 'object') {
      throw new Error(`SpeculativeDecoder: ${label} forward() must return an object.`);
    }
    return {
      logits: coerceLogitsVector(result.logits, `${label} logits`),
      kvCache: result.newKVCache ?? kvCache,
    };
  }

  async generateDraftTokens(inputIds, kvCache, numTokens = this.numDraftTokens) {
    if (!this.draftModel) {
      throw new Error('Draft model not set');
    }

    const draftTokens = [];
    const draftLogprobs = [];
    let draftKVCache = kvCache?.clone?.() ?? kvCache;
    let currentIds = [...inputIds];

    for (let i = 0; i < numTokens; i++) {
      const forwardResult = await this._forwardForLogits(
        this.draftModel,
        currentIds,
        draftKVCache,
        `draft step ${i}`
      );
      draftKVCache = forwardResult.kvCache;

      const { token, logprob } = this.sampleToken(forwardResult.logits, this.temperature);
      draftTokens.push(token);
      draftLogprobs.push(logprob);
      currentIds = [...currentIds, token];
    }

    return {
      tokens: draftTokens,
      logprobs: draftLogprobs,
    };
  }

  sampleToken(logits, temperature) {
    const logitsVector = coerceLogitsVector(logits, 'sample logits');
    assertTemperature(temperature, 'temperature');
    const vocabSize = logitsVector.length;

    const scaledLogits = new Float32Array(vocabSize);
    for (let i = 0; i < vocabSize; i++) {
      scaledLogits[i] = logitsVector[i] / temperature;
    }

    const logprobs = this.logSoftmax(scaledLogits);
    const probs = new Float32Array(vocabSize);
    for (let i = 0; i < vocabSize; i++) {
      probs[i] = Math.exp(logprobs[i]);
    }

    const r = this.random();
    let cumSum = 0;
    for (let i = 0; i < vocabSize; i++) {
      cumSum += probs[i];
      if (r < cumSum) {
        return { token: i, logprob: logprobs };
      }
    }

    return { token: vocabSize - 1, logprob: logprobs };
  }

  logSoftmax(logits) {
    const logitsVector = coerceLogitsVector(logits, 'logSoftmax logits');
    const n = logitsVector.length;
    const result = new Float32Array(n);

    let max = -Infinity;
    for (let i = 0; i < n; i++) {
      if (logitsVector[i] > max) max = logitsVector[i];
    }

    let sumExp = 0;
    for (let i = 0; i < n; i++) {
      sumExp += Math.exp(logitsVector[i] - max);
    }
    const logSumExp = max + Math.log(sumExp);

    for (let i = 0; i < n; i++) {
      result[i] = logitsVector[i] - logSumExp;
    }

    return result;
  }

  async verifyDraftTokens(inputIds, draftTokens, draftLogprobs, kvCache) {
    if (!this.mainModel) {
      throw new Error('Main model not set');
    }
    if (!Array.isArray(draftTokens) || !Array.isArray(draftLogprobs)) {
      throw new Error('SpeculativeDecoder: draft tokens and logprobs must be arrays.');
    }
    if (draftTokens.length !== draftLogprobs.length) {
      throw new Error('SpeculativeDecoder: draft token/logprob length mismatch.');
    }

    const numDraft = draftTokens.length;
    const acceptedTokens = [];
    let acceptedCount = 0;
    let verifyIds = [...inputIds];
    let verifyKVCache = kvCache?.clone?.() ?? kvCache;
    let rejectedMainLogits = null;

    for (let i = 0; i < numDraft; i++) {
      const forwardResult = await this._forwardForLogits(
        this.mainModel,
        verifyIds,
        verifyKVCache,
        `verify step ${i}`
      );
      verifyKVCache = forwardResult.kvCache;
      const mainLogits = forwardResult.logits;
      const mainLogprob = this.logSoftmax(mainLogits);
      const draftToken = draftTokens[i];
      const draftLogprobVec = coerceLogitsVector(draftLogprobs[i], `draft logprobs[${i}]`);

      if (!Number.isInteger(draftToken) || draftToken < 0 || draftToken >= mainLogprob.length) {
        throw new Error(
          `SpeculativeDecoder: draft token at index ${i} is out of vocabulary range.`
        );
      }
      if (draftLogprobVec.length !== mainLogprob.length) {
        throw new Error(
          `SpeculativeDecoder: draft logprobs[${i}] length (${draftLogprobVec.length}) ` +
          `does not match main logits length (${mainLogprob.length}).`
        );
      }

      const draftLogprob = draftLogprobVec[draftToken];
      const mainTokenLogprob = mainLogprob[draftToken];
      const acceptProb = Math.min(1, Math.exp(mainTokenLogprob - draftLogprob));

      if (this.random() < acceptProb) {
        acceptedTokens.push(draftToken);
        acceptedCount++;
        verifyIds = [...verifyIds, draftToken];
      } else {
        rejectedMainLogits = mainLogits;
        break;
      }
    }

    let sampledToken;
    if (acceptedCount === numDraft) {
      const continuation = await this._forwardForLogits(
        this.mainModel,
        verifyIds,
        verifyKVCache,
        'verify continuation'
      );
      sampledToken = this.sampleToken(continuation.logits, this.temperature).token;
    } else {
      if (!rejectedMainLogits) {
        throw new Error('SpeculativeDecoder: missing main logits for rejected token.');
      }
      const rejectedDraftLogprobs = draftLogprobs[acceptedCount];
      if (!rejectedDraftLogprobs) {
        throw new Error(
          `SpeculativeDecoder: missing draft logprobs for rejected token at index ${acceptedCount}.`
        );
      }
      sampledToken = this.sampleFromResidual(
        rejectedMainLogits,
        rejectedDraftLogprobs,
        true
      );
    }

    this.stats.totalDrafted += numDraft;
    this.stats.totalAccepted += acceptedCount;
    this.stats.totalRejected += numDraft - acceptedCount;
    this.stats.averageAcceptRate = this.stats.totalDrafted > 0
      ? this.stats.totalAccepted / this.stats.totalDrafted
      : 0;

    return {
      acceptedCount,
      acceptedTokens,
      sampledToken,
      allAccepted: acceptedCount === numDraft,
    };
  }

  sampleFromResidual(mainLogits, draftLogprobs, wasRejected) {
    const mainLogitsVec = coerceLogitsVector(mainLogits, 'residual main logits');
    if (!wasRejected) {
      return this.sampleToken(mainLogitsVec, this.temperature).token;
    }

    const draftLogprobVec = coerceLogitsVector(draftLogprobs, 'residual draft logprobs');
    if (draftLogprobVec.length !== mainLogitsVec.length) {
      throw new Error(
        `SpeculativeDecoder: residual draft/main length mismatch (${draftLogprobVec.length} vs ${mainLogitsVec.length}).`
      );
    }

    const vocabSize = mainLogitsVec.length;
    const mainProbs = new Float32Array(vocabSize);
    const draftProbs = new Float32Array(vocabSize);

    const mainLogprobsArr = this.logSoftmax(mainLogitsVec);
    for (let i = 0; i < vocabSize; i++) {
      mainProbs[i] = Math.exp(mainLogprobsArr[i]);
      draftProbs[i] = Math.exp(draftLogprobVec[i] ?? -Infinity);
    }

    const residual = new Float32Array(vocabSize);
    let residualSum = 0;
    for (let i = 0; i < vocabSize; i++) {
      residual[i] = Math.max(0, mainProbs[i] - draftProbs[i]);
      residualSum += residual[i];
    }

    if (residualSum > 0) {
      const r = this.random() * residualSum;
      let cumSum = 0;
      for (let i = 0; i < vocabSize; i++) {
        cumSum += residual[i];
        if (r < cumSum) {
          return i;
        }
      }
    }

    return this.sampleToken(mainLogitsVec, this.temperature).token;
  }

  async step(inputIds, mainKVCache, draftKVCache) {
    const { tokens: draftTokens, logprobs: draftLogprobs } =
      await this.generateDraftTokens(inputIds, draftKVCache);

    const result = await this.verifyDraftTokens(
      inputIds,
      draftTokens,
      draftLogprobs,
      mainKVCache
    );

    const newTokens = [...result.acceptedTokens, result.sampledToken];
    const draftedCount = draftTokens.length;

    return {
      newTokens,
      mainKVCache,
      acceptRate: draftedCount > 0 ? result.acceptedCount / draftedCount : 0,
    };
  }

  getStats() {
    return {
      ...this.stats,
      speedup: this.estimateSpeedup(),
    };
  }

  estimateSpeedup() {
    if (this.stats.totalDrafted === 0) return 1.0;

    const acceptRate = this.stats.averageAcceptRate;
    const k = this.numDraftTokens;
    const draftOverhead = 0.1 * k;
    const tokensPerCall = 1 + acceptRate * k;

    return tokensPerCall / (1 + draftOverhead);
  }

  resetStats() {
    this.stats = {
      totalDrafted: 0,
      totalAccepted: 0,
      totalRejected: 0,
      averageAcceptRate: 0,
    };
  }
}

export default SpeculativeDecoder;
