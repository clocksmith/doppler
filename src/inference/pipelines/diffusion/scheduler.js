function resolveStepCount(config, stepsOverride) {
  const stepCount = Number.isFinite(stepsOverride) && stepsOverride > 0
    ? Math.floor(stepsOverride)
    : Math.floor(config.numSteps);
  return Math.max(1, stepCount);
}

function linspace(start, end, steps) {
  const out = new Float32Array(steps);
  if (steps === 1) {
    out[0] = start;
    return out;
  }
  const step = (end - start) / (steps - 1);
  for (let i = 0; i < steps; i++) {
    out[i] = start + step * i;
  }
  return out;
}

function buildLinearSigmaSchedule(steps) {
  return linspace(1.0, 0.0, steps);
}

function buildFlowMatchSchedule(config, steps) {
  const shift = Number.isFinite(config.shift) ? config.shift : 1.0;
  const t = linspace(1.0, 0.0, steps);
  const sigmas = new Float32Array(steps);
  for (let i = 0; i < steps; i++) {
    const ti = t[i];
    const denom = ti + shift * (1.0 - ti);
    sigmas[i] = denom === 0 ? 0 : ti / denom;
  }
  return sigmas;
}

export function buildScheduler(config, stepsOverride = null) {
  if (!config) {
    throw new Error('Scheduler config is required');
  }
  const steps = resolveStepCount(config, stepsOverride);
  const type = config.type || 'ddim';
  const sigmas = type === 'flowmatch_euler'
    ? buildFlowMatchSchedule(config, steps)
    : buildLinearSigmaSchedule(steps);
  const trainSteps = Number.isFinite(config.numTrainTimesteps)
    ? config.numTrainTimesteps
    : 1000;
  const timesteps = new Float32Array(steps);
  for (let i = 0; i < steps; i++) {
    timesteps[i] = sigmas[i] * trainSteps;
  }
  return {
    type,
    steps,
    sigmas,
    timesteps,
  };
}
