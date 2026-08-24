export declare class AbortError extends Error {
  code: 'ABORT_ERR';
  constructor(message?: string);
}

export declare function isAbortError(error: unknown): boolean;
export declare function assertNotAborted(signal?: AbortSignal | null): void;
