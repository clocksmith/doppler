function mergeSection(base, override) {
  if (!override) return { ...base };
  return { ...base, ...override };
}

export function mergeQuintelConfig(base, override) {
  if (!base) return override ? { ...override } : null;
  if (!override) {
    return {
      ...base,
      rules: { ...base.rules },
      weights: { ...base.weights },
      clamp: { ...base.clamp },
    };
  }
  return {
    ...base,
    ...override,
    rules: mergeSection(base.rules, override.rules),
    weights: mergeSection(base.weights, override.weights),
    clamp: mergeSection(base.clamp, override.clamp),
  };
}

function applyPairEnergy(state, gradients, indexA, indexB, weight) {
  const diff = state[indexA] - state[indexB];
  const energy = weight * diff * diff;
  const grad = weight * 2 * diff;
  gradients[indexA] += grad;
  gradients[indexB] -= grad;
  return energy;
}

export function computeQuintelEnergy(state, size, config) {
  const gradients = new Float32Array(state.length);
  const components = {
    symmetry: null,
    count: null,
    center: null,
    binarize: null,
  };
  let countDiff = null;

  const rules = config.rules || {};
  const weights = config.weights || {};
  const symmetryWeight = Number.isFinite(weights.symmetry) ? weights.symmetry : 1.0;
  const countWeight = Number.isFinite(weights.count) ? weights.count : 1.0;
  const centerWeight = Number.isFinite(weights.center) ? weights.center : 1.0;
  const binarizeWeight = Number.isFinite(weights.binarize) ? weights.binarize : 0.0;
  let totalEnergy = 0;

  if (rules.mirrorX || rules.mirrorY || rules.diagonal) {
    components.symmetry = 0;
  }

  if (rules.mirrorX) {
    for (let i = 0; i < size; i++) {
      for (let j = 0; j < Math.floor(size / 2); j++) {
        const indexA = i * size + j;
        const indexB = i * size + (size - 1 - j);
        components.symmetry += applyPairEnergy(state, gradients, indexA, indexB, symmetryWeight);
      }
    }
  }

  if (rules.mirrorY) {
    for (let i = 0; i < Math.floor(size / 2); i++) {
      for (let j = 0; j < size; j++) {
        const indexA = i * size + j;
        const indexB = (size - 1 - i) * size + j;
        components.symmetry += applyPairEnergy(state, gradients, indexA, indexB, symmetryWeight);
      }
    }
  }

  if (rules.diagonal) {
    for (let i = 0; i < size; i++) {
      for (let j = i + 1; j < size; j++) {
        const indexA = i * size + j;
        const indexB = j * size + i;
        components.symmetry += applyPairEnergy(state, gradients, indexA, indexB, symmetryWeight);
      }
    }
  }

  if (rules.count) {
    const countTarget = Number.isFinite(config.countTarget) ? config.countTarget : size * size * 0.5;
    let sum = 0;
    for (let i = 0; i < state.length; i++) sum += state[i];
    const diff = sum - countTarget;
    countDiff = diff;
    const energy = countWeight * diff * diff;
    components.count = energy;
    const grad = countWeight * 2 * diff;
    for (let i = 0; i < gradients.length; i++) {
      gradients[i] += grad;
    }
  }

  if (rules.center) {
    const centerTarget = Number.isFinite(config.centerTarget) ? config.centerTarget : 1.0;
    const centerIndex = Math.floor(size / 2) * size + Math.floor(size / 2);
    const diff = state[centerIndex] - centerTarget;
    const energy = centerWeight * diff * diff;
    components.center = energy;
    gradients[centerIndex] += centerWeight * 2 * diff;
  }

  if (binarizeWeight > 0) {
    components.binarize = 0;
    for (let i = 0; i < state.length; i++) {
      const value = state[i];
      const term = value * (1 - value);
      components.binarize += binarizeWeight * term;
      gradients[i] += binarizeWeight * (1 - 2 * value);
    }
  }

  totalEnergy += (components.symmetry ?? 0);
  totalEnergy += (components.count ?? 0);
  totalEnergy += (components.center ?? 0);
  totalEnergy += (components.binarize ?? 0);

  return { energy: totalEnergy, gradients, components, countDiff };
}

export function runQuintelEnergyLoop(options) {
  const {
    state,
    size,
    config,
    loop,
    diagnostics,
    onProgress,
    onTrace,
    traceEvery,
  } = options;

  const maxSteps = Math.max(1, Math.floor(loop.maxSteps));
  const minSteps = Math.max(0, Math.floor(loop.minSteps));
  const stepSize = Number.isFinite(loop.stepSize) ? loop.stepSize : 0.1;
  const gradientScale = Number.isFinite(loop.gradientScale) ? loop.gradientScale : 1.0;
  const convergenceThreshold = Number.isFinite(loop.convergenceThreshold)
    ? loop.convergenceThreshold
    : null;

  const readbackEvery = Math.max(1, Math.floor(diagnostics.readbackEvery));
  const historyLimit = Math.max(1, Math.floor(diagnostics.historyLimit));
  const traceInterval = Math.max(0, Math.floor(traceEvery ?? diagnostics.traceEvery ?? 0));
  const clampMin = Number.isFinite(config.clamp?.min) ? config.clamp.min : 0;
  const clampMax = Number.isFinite(config.clamp?.max) ? config.clamp.max : 1;

  const energyHistory = [];
  const stepTimesMs = [];
  let lastEnergy = null;
  let lastComponents = null;
  const start = performance.now();

  for (let step = 0; step < maxSteps; step++) {
    const stepStart = performance.now();
    const { energy, gradients, components } = computeQuintelEnergy(state, size, config);
    lastEnergy = energy;
    lastComponents = components;

    const shouldRecord = step % readbackEvery === 0 || step === maxSteps - 1;
    if (shouldRecord) {
      energyHistory.push(energy);
      if (energyHistory.length > historyLimit) {
        energyHistory.shift();
      }
    }

    if (traceInterval > 0 && onTrace && step % traceInterval === 0) {
      onTrace(step, energy, components);
    }

    if (step >= minSteps && convergenceThreshold != null && energy <= convergenceThreshold) {
      stepTimesMs.push(performance.now() - stepStart);
      break;
    }

    const stepScale = stepSize * gradientScale;
    for (let i = 0; i < state.length; i++) {
      const next = state[i] - stepScale * gradients[i];
      state[i] = Math.min(clampMax, Math.max(clampMin, next));
    }

    stepTimesMs.push(performance.now() - stepStart);
    if (onProgress) {
      onProgress({
        stage: 'energy',
        percent: (step + 1) / maxSteps,
        message: `Step ${step + 1} / ${maxSteps}`,
      });
    }
  }

  return {
    state,
    energy: lastEnergy,
    energyHistory,
    energyComponents: lastComponents,
    stepTimesMs,
    steps: stepTimesMs.length,
    totalTimeMs: performance.now() - start,
  };
}
