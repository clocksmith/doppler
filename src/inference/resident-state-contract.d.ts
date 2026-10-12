import type { CanonicalGenerationPolicy, CanonicalStopReason } from '../config/generation-contract.js';
export interface ResidentCursor {
  schema: 'doppler.cursor/v2'; cursorId: string; generationId: string; revision: number;
  promptTokenCount: number; materializedTokenCount: number; acceptedGeneratedTokenCount: number;
  nextTargetTokenIndex: number; pendingTokenId: number; materializedPrefixDigest: string;
  acceptedTokenDigest: string; terminal: boolean; stopReason: CanonicalStopReason | null;
}
export interface AdmissionOnlyResult {
  schema: 'doppler.admission-only/v1'; generationId: string; promptTokenCount: number;
  promptDigest: string; acceptedTokenIds: readonly []; stopReason: 'max-tokens' | 'context-limit';
}
export interface ResidentBlockDecision {
  acceptedTokenIds: readonly number[]; acceptedProposalCount: number; materializeInputCount: number;
  tokenRange: Readonly<{ start: number; end: number }>; successor: Readonly<ResidentCursor>;
}
export function canonicalStopReason(policy: CanonicalGenerationPolicy, tokenId: number, generatedCount: number,
  promptCount: number, materializedCount: number, textStop?: boolean): CanonicalStopReason | null;
export function validateResidentCursor(cursor: unknown, policy: CanonicalGenerationPolicy, vocabSize: number): Readonly<ResidentCursor>;
export function admitResidentPrompt(prompt: readonly number[], policy: CanonicalGenerationPolicy, vocabSize: number,
  generationId: string): Readonly<AdmissionOnlyResult> | Readonly<{kind: 'prefill'; tokenIds: readonly number[]; generationId: string}>;
export function bootstrapResidentCursor(prompt: readonly number[], firstToken: number, policy: CanonicalGenerationPolicy,
  vocabSize: number, generationId: string, textStop?: boolean): Readonly<ResidentCursor>;
export function admitResidentBlock(cursor: ResidentCursor, proposal: readonly number[], policy: CanonicalGenerationPolicy,
  vocabSize: number): readonly number[];
export function decideResidentBlock(cursor: ResidentCursor, materialized: readonly number[], proposal: readonly number[],
  targets: readonly number[], textStops: readonly boolean[], policy: CanonicalGenerationPolicy, vocabSize: number): Readonly<ResidentBlockDecision>;
