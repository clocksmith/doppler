/** Resolves the legacy GPU sampler's greedy policy after generation validation.
 * Zero temperature is always greedy, including an explicitly zero threshold.
 * topK=0 retains the full vocabulary; topK is never rounded here.
 */
export function isGreedySampling(temperature: number, topK: number, greedyThreshold: number): boolean;
