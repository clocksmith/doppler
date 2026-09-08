export interface CapsuleSessionExecution {
  run<T>(task: (signal: AbortSignal) => T, signal?: AbortSignal | null): T;
  stream<T, R>(create: (signal: AbortSignal) => AsyncGenerator<T, R, void>, signal?: AbortSignal | null): AsyncGenerator<T, R, void>;
  close(dispose: () => void | Promise<void>): Promise<void>;
}
export function createCapsuleSessionExecution(): CapsuleSessionExecution;
