import type { CanonicalGenerationPolicy, CanonicalStopReason } from '../config/generation-contract.js';
import type { ResidentAttemptIdentity } from './resident-block-contract.js';
import type { ResidentCursor } from './resident-state-contract.js';
import type { AdmissionOnlyResult } from './resident-state-contract.js';
export interface AdmissionPublication {
  schema: 'doppler.admission-publication/v1'; publicationDigest: string; attemptId: string;
  coordinatorEpoch: string; attemptDigest: string; sequence: 0; expectedRevision: null;
  previousPublicationDigest: string | null; result: Readonly<AdmissionOnlyResult>;
}
export function validateAdmissionPublication(input: unknown, attempt: ResidentAttemptIdentity,
  policy: CanonicalGenerationPolicy, vocabSize: number): Readonly<AdmissionPublication>;
export interface StageCommitAcknowledgment {
  schema: 'doppler.stage-commit/v1'; stageId: string; attemptDigest: string; sequence: number;
  baseCursorId: string | null; decisionId: string; successorCursorId: string;
  participantEpoch: string; deviceEpoch: string; residentId: string;
  /** Immutable evidence in the host's existing authenticated transport/storage. */
  authenticationDigest: string;
}
export interface ExecutionPublication {
  schema: 'doppler.execution-publication/v2'; publicationDigest: string;
  attemptId: string; coordinatorEpoch: string; attemptDigest: string; sequence: number;
  expectedRevision: number | null; previousPublicationDigest: string | null; baseCursorId: string | null;
  decisionId: string; acceptedTokenIds: readonly number[]; tokenRange: { start: number; end: number };
  successor: Readonly<ResidentCursor>; delta: { text: string; outputDigest: string };
  stoppingOutcome: CanonicalStopReason | null; acknowledgments: readonly StageCommitAcknowledgment[];
}
export function residentPublicationDecisionDigest(input: Pick<ExecutionPublication, 'attemptDigest' | 'sequence' |
  'baseCursorId' | 'acceptedTokenIds' | 'tokenRange' | 'successor' | 'delta' | 'stoppingOutcome'>): string;
/** Validates logical commitments only. The conversation owner must additionally
 * authenticate every acknowledgment against its bound stage before writing. */
export function validateExecutionPublication(input: unknown, attempt: ResidentAttemptIdentity,
  policy: CanonicalGenerationPolicy, vocabSize: number, base: ResidentCursor | null): Readonly<ExecutionPublication>;
