import type { PackOperationRequest } from '../../config/pack-operation.js';
export interface PackOperationAdapter {
  validate(request: Readonly<PackOperationRequest>): void;
  execute(request: Readonly<PackOperationRequest>, signal: AbortSignal): AsyncGenerator<{ delta: unknown; output: unknown }, unknown, void>;
}
export function createPackOperationAdapters(ports: {
  program: Record<string, any>;
  generate(options: Record<string, unknown>): AsyncIterable<number>;
  rerank(request: Record<string, unknown>): Promise<unknown>;
}): Record<string, PackOperationAdapter>;
