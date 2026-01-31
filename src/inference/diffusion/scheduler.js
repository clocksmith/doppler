export function buildScheduler(config, stepsOverride = null) {
  if (!config) {
    throw new Error('Scheduler config is required');
  }
  const stepCount = Number.isFinite(stepsOverride) && stepsOverride > 0
    ? Math.floor(stepsOverride)
    : Math.floor(config.numSteps);
  const steps = Math.max(1, stepCount);
  const sigmas = new Float32Array(steps);
  if (steps === 1) {
    sigmas[0] = 1.0;
  } else {
    for (let i = 0; i < steps; i++) {
      sigmas[i] = 1.0 - i / (steps - 1);
    }
  }
  return {
    type: config.type,
    steps,
    sigmas,
  };
}
