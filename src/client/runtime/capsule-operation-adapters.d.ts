import type { CapsuleOperationRequest } from '../../config/capsule-operation.js';
export interface CapsuleOperationAdapter {
  validate(request: Readonly<CapsuleOperationRequest>): void;
  execute(request: Readonly<CapsuleOperationRequest>, signal: AbortSignal): AsyncGenerator<{ delta: unknown; output: unknown }, unknown, void>;
}
export function createCapsuleOperationAdapters(ports: {
  program: Record<string, any>;
  generate(options: Record<string, unknown>): AsyncIterable<number>;
  rerank(request: Record<string, unknown>): Promise<unknown>;
  embed(request: Record<string, unknown>): Promise<unknown>;
  encodeSequence(sequence: string, options: Record<string, unknown>): Promise<unknown>;
}): Record<string, CapsuleOperationAdapter>;
