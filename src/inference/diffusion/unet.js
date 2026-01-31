export function runUnetStep(latents, scheduler, stepIndex, guidanceScale = 1.0) {
  const sigma = scheduler?.sigmas?.[stepIndex] ?? 0;
  const guidance = Number.isFinite(guidanceScale) ? guidanceScale : 1.0;
  const scale = Math.max(0.0, 1.0 - sigma * guidance * 0.02);
  for (let i = 0; i < latents.length; i++) {
    latents[i] *= scale;
  }
  return latents;
}
