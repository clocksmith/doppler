export interface ExecutionIdentity {
  schema: 'doppler.execution-identity/v2'; model: string; tokenizer: string; adapters: readonly string[];
  runtimeArchive: string; executionPlan: string; partitionPlan: string; numericalPolicy: string;
  stateLayout: string; activationLayout: string; decoderLayout: string;
}
export interface ResidentStageIdentity {
  stageId: string; participantId: string; participantEpoch: string; deviceEpoch: string;
  residentId: string; partitionIndex: number; executionId: string;
}
export interface ResidentAttemptIdentity {
  schema: 'doppler.resident-attempt/v2'; attemptId: string; requesterId: string;
  inputDigest: string; authorizationScope: string; routeId: string; generationId: string;
  coordinatorEpoch: string; publisherEpoch: string; routeKind: 'whole-request' | 'two-stage-layer';
  stages: readonly ResidentStageIdentity[];
}
export interface BlockActivation {
  schema: 'doppler.block-activation/v2'; attemptDigest: string; sequence: number;
  baseCursorId: string; proposalId: string; sourceStageId: string; destinationStageId: string;
  rows: { start: number; end: number }; tokenContextDigest: string;
  layout: { dtype: 'f16'|'f32'; precision: 'f16'|'f32'|'two-f32'; components: 1|2;
    elementStrideBytes: 2|4|8; hiddenSize: number; byteOrder: 'little-endian' };
  payloadBytes: number; payloadDigest: string;
}
export function validateExecutionIdentity(input: unknown): Readonly<ExecutionIdentity>;
export function validateResidentAttemptIdentity(input: unknown): Readonly<ResidentAttemptIdentity>;
export function residentAttemptDigest(input: ResidentAttemptIdentity): string;
export function validateBlockActivation(input: unknown, bytes: Uint8Array,
  limits: { maxRows: number; maxBytes: number }): Readonly<{ frame: Readonly<BlockActivation>; bytes: Uint8Array }>;
