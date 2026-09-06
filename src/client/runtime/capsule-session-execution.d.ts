export interface CapsuleSessionExecution {
  run<T>(task: (signal: AbortSignal) => T, signal?: AbortSignal | null): T;
  stream<T>(create: (signal: AbortSignal) => AsyncIterable<T>, signal?: AbortSignal | null): AsyncGenerator<T, void, void>;
  close(dispose: () => void | Promise<void>): Promise<void>;
}
export function createCapsuleSessionExecution(): CapsuleSessionExecution;
