import type { CanonicalGenerationPolicy } from './generation-contract.js';
export function validateCanonicalGenerationPolicy(policy: unknown, vocabSize: number): Readonly<CanonicalGenerationPolicy>;
export function admitCanonicalGenerationPolicy(policy: Omit<CanonicalGenerationPolicy, 'seedHex64'> & { seedHex64?: string },
  vocabSize: number, persist: (policy: Readonly<CanonicalGenerationPolicy>) => Promise<void>): Promise<Readonly<CanonicalGenerationPolicy>>;
export function canonicalGenerationIdentity(policy: CanonicalGenerationPolicy, executionIdentity: string, vocabSize: number): Readonly<{
  policy: Readonly<CanonicalGenerationPolicy>; executionIdentity: string; generationId: string;
}>;
