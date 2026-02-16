export function runUnetStep(latents, scheduler, stepIndex, guidanceScale) {
  if (!scheduler || !Array.isArray(scheduler.sigmas) && !(scheduler.sigmas instanceof Float32Array)) {
    throw new Error('runUnetStep requires a scheduler with sigmas');
  }
  if (!Number.isFinite(guidanceScale)) {
    throw new Error('runUnetStep requires guidanceScale');
  }
  const sigma = scheduler.sigmas[stepIndex] ?? 0;
  const guidance = guidanceScale;
  const scale = Math.max(0.0, 1.0 - sigma * guidance * 0.02);
  for (let i = 0; i < latents.length; i++) {
    latents[i] *= scale;
  }
  return latents;
}
