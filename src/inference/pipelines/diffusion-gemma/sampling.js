const UINT32_SCALE = 1 / 0x100000000;

export function createSeededRandom(seed) {
  if (!Number.isInteger(seed)) {
    throw new Error('DiffusionGemma seeded random requires an integer seed.');
  }
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state * UINT32_SCALE;
  };
}

function assertCanvas(canvas, canvasLength, label) {
  if (!Array.isArray(canvas) && !ArrayBuffer.isView(canvas)) {
    throw new Error(`${label} must be an array or typed array.`);
  }
  if (canvas.length !== canvasLength) {
    throw new Error(`${label} length ${canvas.length} does not match canvasLength ${canvasLength}.`);
  }
}

function randomToken(random, vocabSize) {
  return Math.min(vocabSize - 1, Math.floor(random() * vocabSize));
}

function normalizeTemperature(value) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) {
    throw new Error(`DiffusionGemma temperature must be positive; got ${String(value)}.`);
  }
  return value;
}

export function resolveDenoisingTemperature(config, step) {
  const maxSteps = config.maxDenoisingSteps;
  if (!Number.isInteger(step) || step < 1 || step > maxSteps) {
    throw new Error(`DiffusionGemma denoising step must be in [1, ${maxSteps}], got ${String(step)}.`);
  }
  return normalizeTemperature(config.tMin + ((config.tMax - config.tMin) * (step / maxSteps)));
}

export function initializeCanvas(config, random) {
  if (typeof random !== 'function') {
    throw new Error('DiffusionGemma canvas initialization requires a random function.');
  }
  const canvas = new Int32Array(config.canvasLength);
  for (let i = 0; i < canvas.length; i += 1) {
    canvas[i] = randomToken(random, config.vocabSize);
  }
  return canvas;
}

function getLogitsView(logits, tokenIndex, vocabSize) {
  const start = tokenIndex * vocabSize;
  const end = start + vocabSize;
  if (logits.length < end) {
    throw new Error(
      `DiffusionGemma logits length ${logits.length} is too small for token ${tokenIndex} and vocabSize ${vocabSize}.`
    );
  }
  return logits.subarray(start, end);
}

function softmaxStats(tokenLogits, processedLogits, temperature, random) {
  let max = -Infinity;
  for (let i = 0; i < tokenLogits.length; i += 1) {
    const value = tokenLogits[i] / temperature;
    processedLogits[i] = value;
    if (value > max) max = value;
  }

  let sum = 0;
  let weightedLogitSum = 0;
  let argmaxToken = 0;
  let argmaxValue = -Infinity;
  const probs = new Float32Array(tokenLogits.length);
  for (let i = 0; i < tokenLogits.length; i += 1) {
    const scaled = processedLogits[i];
    if (scaled > argmaxValue) {
      argmaxValue = scaled;
      argmaxToken = i;
    }
    const expValue = Math.exp(scaled - max);
    probs[i] = expValue;
    sum += expValue;
    weightedLogitSum += expValue * scaled;
  }
  if (!Number.isFinite(sum) || sum <= 0) {
    throw new Error('DiffusionGemma logits produced an invalid softmax denominator.');
  }

  const draw = random() * sum;
  let cumulative = 0;
  let sampledToken = tokenLogits.length - 1;
  for (let i = 0; i < probs.length; i += 1) {
    cumulative += probs[i];
    if (draw <= cumulative) {
      sampledToken = i;
      break;
    }
  }

  const entropy = Math.log(sum) + max - (weightedLogitSum / sum);
  return {
    argmaxToken,
    sampledToken,
    entropy,
  };
}

export function applyEntropyBoundStep(currentCanvas, logits, config, options = {}) {
  const random = options.random;
  if (typeof random !== 'function') {
    throw new Error('DiffusionGemma entropy-bound step requires a random function.');
  }
  assertCanvas(currentCanvas, config.canvasLength, 'currentCanvas');
  if (!ArrayBuffer.isView(logits)) {
    throw new Error('DiffusionGemma entropy-bound step requires typed-array logits.');
  }
  const temperature = normalizeTemperature(options.temperature);
  const nextCanvas = new Int32Array(config.canvasLength);
  const argmaxCanvas = new Int32Array(config.canvasLength);
  const sampledCanvas = new Int32Array(config.canvasLength);
  const entropies = new Float32Array(config.canvasLength);
  const processedLogits = new Float32Array(config.canvasLength * config.vocabSize);
  const entropyOrder = [];

  for (let i = 0; i < config.canvasLength; i += 1) {
    const stats = softmaxStats(
      getLogitsView(logits, i, config.vocabSize),
      getLogitsView(processedLogits, i, config.vocabSize),
      temperature,
      random
    );
    argmaxCanvas[i] = stats.argmaxToken;
    sampledCanvas[i] = stats.sampledToken;
    entropies[i] = stats.entropy;
    entropyOrder.push({ index: i, entropy: stats.entropy });
  }

  entropyOrder.sort((a, b) => a.entropy - b.entropy || a.index - b.index);
  const accepted = new Uint8Array(config.canvasLength);
  let cumulativeEntropy = 0;
  let acceptedCount = 0;
  for (const entry of entropyOrder) {
    if (cumulativeEntropy <= config.entropyBound) {
      accepted[entry.index] = 1;
      acceptedCount += 1;
    }
    cumulativeEntropy += entry.entropy;
  }

  for (let i = 0; i < config.canvasLength; i += 1) {
    nextCanvas[i] = accepted[i] ? sampledCanvas[i] : randomToken(random, config.vocabSize);
  }

  return {
    canvas: nextCanvas,
    argmaxCanvas,
    sampledCanvas,
    entropies,
    processedLogits,
    accepted,
    acceptedCount,
    meanEntropy: entropies.reduce((sum, value) => sum + value, 0) / entropies.length,
  };
}

export function applyEntropyBoundStatsStep(currentCanvas, stats, config, options = {}) {
  const random = options.random;
  if (typeof random !== 'function') {
    throw new Error('DiffusionGemma entropy-bound stats step requires a random function.');
  }
  assertCanvas(currentCanvas, config.canvasLength, 'currentCanvas');
  const argmaxCanvas = stats?.argmaxCanvas;
  const entropies = stats?.entropies;
  assertCanvas(argmaxCanvas, config.canvasLength, 'argmaxCanvas');
  if (!(entropies instanceof Float32Array) || entropies.length !== config.canvasLength) {
    throw new Error('DiffusionGemma entropy stats step requires entropies as Float32Array[canvasLength].');
  }

  const nextCanvas = new Int32Array(config.canvasLength);
  const accepted = new Uint8Array(config.canvasLength);
  const entropyOrder = [];
  for (let i = 0; i < config.canvasLength; i += 1) {
    const entropy = entropies[i];
    if (!Number.isFinite(entropy) || entropy < 0) {
      throw new Error(`DiffusionGemma entropy stats step received invalid entropy at ${i}.`);
    }
    entropyOrder.push({ index: i, entropy });
  }
  entropyOrder.sort((a, b) => a.entropy - b.entropy || a.index - b.index);

  let cumulativeEntropy = 0;
  let acceptedCount = 0;
  for (const entry of entropyOrder) {
    if (cumulativeEntropy <= config.entropyBound) {
      accepted[entry.index] = 1;
      acceptedCount += 1;
    }
    cumulativeEntropy += entry.entropy;
  }

  for (let i = 0; i < config.canvasLength; i += 1) {
    nextCanvas[i] = accepted[i] ? argmaxCanvas[i] : randomToken(random, config.vocabSize);
  }

  return {
    canvas: nextCanvas,
    argmaxCanvas: Int32Array.from(argmaxCanvas),
    sampledCanvas: Int32Array.from(argmaxCanvas),
    entropies,
    processedLogits: null,
    accepted,
    acceptedCount,
    meanEntropy: entropies.reduce((sum, value) => sum + value, 0) / entropies.length,
  };
}

export function updateStabilityState(previousArgmaxCanvas, argmaxCanvas, previousCounts, config) {
  assertCanvas(argmaxCanvas, config.canvasLength, 'argmaxCanvas');
  const counts = new Uint16Array(config.canvasLength);
  if (config.stabilityThreshold === 0) {
    counts.fill(0xffff);
    return {
      counts,
      stableTokens: config.canvasLength,
      allStable: true,
    };
  }
  let stableTokens = 0;
  for (let i = 0; i < config.canvasLength; i += 1) {
    const previousMatches = previousArgmaxCanvas && previousArgmaxCanvas[i] === argmaxCanvas[i];
    counts[i] = previousMatches ? ((previousCounts?.[i] ?? 0) + 1) : 0;
    if (counts[i] >= config.stabilityThreshold) {
      stableTokens += 1;
    }
  }
  return {
    counts,
    stableTokens,
    allStable: stableTokens === config.canvasLength,
  };
}

function releaseSelfConditioningState(state) {
  if (state && !ArrayBuffer.isView(state) && typeof state.release === 'function') {
    state.release();
  }
}

export async function denoiseCanvas(config, options) {
  const logitsProvider = options?.logitsProvider;
  const random = options?.random;
  if (typeof logitsProvider !== 'function') {
    throw new Error('DiffusionGemma denoiseCanvas requires a logitsProvider function.');
  }
  if (typeof random !== 'function') {
    throw new Error('DiffusionGemma denoiseCanvas requires a random function.');
  }

  let canvas = options.initialCanvas == null
    ? initializeCanvas(config, random)
    : Int32Array.from(options.initialCanvas);
  assertCanvas(canvas, config.canvasLength, 'initialCanvas');
  let previousArgmaxCanvas = null;
  let stabilityCounts = null;
  let selfConditioningLogits = options.selfConditioningLogits ?? null;
  let lastStep = null;
  let stepsRun = 0;

  for (let step = config.maxDenoisingSteps; step >= 1; step -= 1) {
    stepsRun += 1;
    const temperature = resolveDenoisingTemperature(config, step);
    const logits = await logitsProvider({
      canvas,
      step,
      temperature,
      selfConditioningLogits,
      canvasIndex: options.canvasIndex ?? 0,
      inputIds: options.inputIds ?? null,
    });
    const stepResult = applyEntropyBoundStep(canvas, logits, config, {
      temperature,
      random,
    });
    const stability = updateStabilityState(
      previousArgmaxCanvas,
      stepResult.argmaxCanvas,
      stabilityCounts,
      config
    );

    canvas = stepResult.canvas;
    previousArgmaxCanvas = stepResult.argmaxCanvas;
    stabilityCounts = stability.counts;
    selfConditioningLogits = stepResult.processedLogits;
    lastStep = {
      ...stepResult,
      step,
      temperature,
      stability,
    };

    if (stability.allStable && stepResult.meanEntropy < config.confidenceThreshold) {
      break;
    }
  }

  return {
    canvas,
    argmaxCanvas: previousArgmaxCanvas,
    selfConditioningLogits,
    lastStep,
    stepsRun,
  };
}

export async function denoiseCanvasWithStatsProvider(config, options) {
  const statsProvider = options?.statsProvider;
  const random = options?.random;
  if (typeof statsProvider !== 'function') {
    throw new Error('DiffusionGemma denoiseCanvasWithStatsProvider requires a statsProvider function.');
  }
  if (typeof random !== 'function') {
    throw new Error('DiffusionGemma denoiseCanvasWithStatsProvider requires a random function.');
  }

  let canvas = options.initialCanvas == null
    ? initializeCanvas(config, random)
    : Int32Array.from(options.initialCanvas);
  assertCanvas(canvas, config.canvasLength, 'initialCanvas');
  let previousArgmaxCanvas = null;
  let stabilityCounts = null;
  const initialSelfConditioningLogits = options.selfConditioningLogits ?? null;
  let selfConditioningLogits = initialSelfConditioningLogits;
  let lastStep = null;
  let stepsRun = 0;
  let completed = false;

  try {
    for (let step = config.maxDenoisingSteps; step >= 1; step -= 1) {
      stepsRun += 1;
      const temperature = resolveDenoisingTemperature(config, step);
      const inputSelfConditioningLogits = selfConditioningLogits;
      selfConditioningLogits = null;
      const stats = await statsProvider({
        canvas,
        step,
        temperature,
        selfConditioningLogits: inputSelfConditioningLogits,
        canvasIndex: options.canvasIndex ?? 0,
        inputIds: options.inputIds ?? null,
      });
      selfConditioningLogits = stats.selfConditioningLogits ?? null;
      const stepResult = applyEntropyBoundStatsStep(canvas, stats, config, {
        temperature,
        random,
      });
      const stability = updateStabilityState(
        previousArgmaxCanvas,
        stepResult.argmaxCanvas,
        stabilityCounts,
        config
      );

      canvas = stepResult.canvas;
      previousArgmaxCanvas = stepResult.argmaxCanvas;
      stabilityCounts = stability.counts;
      lastStep = {
        ...stepResult,
        step,
        temperature,
        stability,
      };

      if (stability.allStable && stepResult.meanEntropy < config.confidenceThreshold) {
        break;
      }
    }

    completed = true;
    return {
      canvas,
      argmaxCanvas: previousArgmaxCanvas,
      selfConditioningLogits,
      lastStep,
      stepsRun,
    };
  } finally {
    if (!completed && selfConditioningLogits !== initialSelfConditioningLogits) {
      releaseSelfConditioningState(selfConditioningLogits);
    }
  }
}
