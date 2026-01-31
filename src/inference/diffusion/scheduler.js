export function buildScheduler(config, stepsOverride = null) {
  const stepCount = Number.isFinite(stepsOverride) && stepsOverride > 0
    ? Math.floor(stepsOverride)
    : Math.floor(config?.numSteps ?? 1);
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
    type: config?.type ?? 'ddim',
    steps,
    sigmas,
  };
}
