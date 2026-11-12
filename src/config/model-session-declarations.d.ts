export const MODEL_SESSION_DECLARATIONS_SCHEMA_ID: 'doppler.model-session-declarations/v1';
export const MODEL_SESSION_DECLARATIONS_PATH: 'src/config/runtime/model-sessions.json';

export interface ModelSessionDeclaration {
  readonly contextLength: number;
  readonly maxSeqLen: number;
}

export function getModelSessionDeclaration(modelId: string): ModelSessionDeclaration | null;
export function listModelSessionDeclarations(): ReadonlyArray<string>;
export function applyModelSessionDeclaration(manifest: unknown): Record<string, unknown>;
