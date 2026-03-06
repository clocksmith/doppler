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

function buildScmTimesteps(steps, config) {
  const maxTimesteps = Number.isFinite(config.maxTimesteps) ? config.maxTimesteps : 1.5708;
  const intermediateTimesteps = Number.isFinite(config.intermediateTimesteps)
    ? config.intermediateTimesteps
    : 1.3;
  const count = Math.max(1, steps);
  if (count === 1) {
    return new Float32Array([maxTimesteps, 0.0]);
  }
  if (count === 2) {
    return new Float32Array([maxTimesteps, intermediateTimesteps, 0.0]);
  }
  return linspace(maxTimesteps, 0.0, count + 1);
}

export function stepScmScheduler(config, modelOutput, timestep, sample, stepIndex = 0, noise = null) {
  if (!config || config.type !== 'scm') {
    throw new Error('stepScmScheduler requires scheduler.type="scm".');
  }
  if (!(modelOutput instanceof Float32Array)) {
    throw new Error('stepScmScheduler requires modelOutput as Float32Array.');
  }
  if (!(sample instanceof Float32Array)) {
    throw new Error('stepScmScheduler requires sample as Float32Array.');
  }
  if (modelOutput.length !== sample.length) {
    throw new Error(
      `stepScmScheduler requires modelOutput and sample with matching sizes; got ${modelOutput.length} and ${sample.length}.`
    );
  }
  if (!(config.timesteps instanceof Float32Array) || config.timesteps.length < 2) {
    throw new Error('stepScmScheduler requires scheduler.timesteps with length >= 2.');
  }
  if (!Number.isInteger(stepIndex) || stepIndex < 0 || stepIndex + 1 >= config.timesteps.length) {
    throw new Error(
      `stepScmScheduler received invalid stepIndex=${stepIndex} for ${config.timesteps.length} timesteps.`
    );
  }

  const parameterization = config.predictionType ?? 'trigflow';
  if (parameterization !== 'trigflow') {
    throw new Error(`Unsupported SCM predictionType "${parameterization}".`);
  }

  const s = config.timesteps[stepIndex];
  const t = config.timesteps[stepIndex + 1];
  const predOriginalSample = new Float32Array(sample.length);
  const prevSample = new Float32Array(sample.length);

  const cosS = Math.cos(s);
  const sinS = Math.sin(s);
  const cosT = Math.cos(t);
  const sinT = Math.sin(t);

  for (let i = 0; i < sample.length; i++) {
    const predX0 = cosS * sample[i] - sinS * modelOutput[i];
    predOriginalSample[i] = predX0;
    prevSample[i] = predX0;
  }

  if (stepIndex + 1 < config.timesteps.length - 1) {
    if (!(noise instanceof Float32Array) || noise.length !== sample.length) {
      throw new Error(
        'stepScmScheduler requires a Float32Array noise tensor for multi-step SCM updates.'
      );
    }
    const sigmaData = Number.isFinite(config.sigmaData) ? config.sigmaData : 0.5;
    for (let i = 0; i < prevSample.length; i++) {
      prevSample[i] = cosT * predOriginalSample[i] + sinT * noise[i] * sigmaData;
    }
  }

  return {
    prevSample,
    predOriginalSample,
  };
}

export function buildScheduler(config, stepsOverride = null) {
  if (!config) {
    throw new Error('Scheduler config is required');
  }
  const steps = resolveStepCount(config, stepsOverride);
  const type = config.type;
  if (typeof type !== 'string' || !type) {
    throw new Error('Diffusion scheduler requires a scheduler type.');
  }
  const trainSteps = Number.isFinite(config.numTrainTimesteps)
    ? config.numTrainTimesteps
    : null;
  if (!Number.isFinite(trainSteps) || trainSteps <= 0) {
    throw new Error('Diffusion scheduler requires valid numTrainTimesteps.');
  }
  if (type === 'scm') {
    return {
      type,
      steps,
      sigmas: null,
      timesteps: buildScmTimesteps(steps, config),
      predictionType: config.predictionType ?? 'trigflow',
      sigmaData: Number.isFinite(config.sigmaData) ? config.sigmaData : 0.5,
    };
  }
  const sigmas = type === 'flowmatch_euler'
    ? buildFlowMatchSchedule(config, steps)
    : buildLinearSigmaSchedule(steps);
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
