export function levenshteinDistance(a: string, b: string): number;
export function normalizeFlag(flag: string): string;
export function suggestFlag(flag: string): string | null;
export function resolveFlagAlias(flag: string): string | null;
export function suggestClosestFlags(flag: string): string[];
