export function mulberry32(seed: number): () => number;
export function stableStringify(value: unknown): string;
export function cloneSpec(spec: Record<string, unknown>): Record<string, unknown>;
export function canonicalDepthForRound(round: number): number | null;
export function requiredCachedNodes(maxDepth: number): number;
