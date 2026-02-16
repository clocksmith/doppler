export function initWeights(mode: string, seed: number, scale: number, defaults: Float32Array): Float32Array;
export function perturbWeights(base: Float32Array, rng: () => number, count: number, scale: number): Float32Array;
